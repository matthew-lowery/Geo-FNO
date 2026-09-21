"""Durable metrics and complete, resumable training checkpoints."""

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import wandb

from ram_dataset_loader import required_dataset_paths


def add_runtime_arguments(parser):
    parser.add_argument("--require-ood", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--eval-only", action="store_true")


def validate_runtime(args):
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be positive")
    if args.eval_only and args.resume is None:
        raise ValueError("--eval-only requires a complete --resume checkpoint")
    if args.require_ood:
        if getattr(args, "no_ood", False) or not getattr(args, "eval_ood", True):
            raise ValueError("--require-ood conflicts with --no-ood")
        for path in required_dataset_paths(args.dataset, args.data_root, ood=True):
            if not path.is_file():
                raise FileNotFoundError(f"Required OOD file missing before training: {path}")


def require_finite(value, label):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Nonfinite {label}; stopping without another optimizer step")


def check_gradients(*models):
    parameters = [p for model in models for p in model.parameters() if p.grad is not None]
    torch.nn.utils.clip_grad_norm_(parameters, float("inf"), error_if_nonfinite=True)


def normalizer_state(normalizer):
    return {key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in vars(normalizer).items()}


class RunArtifacts:
    def __init__(self, args, name):
        self.args = args
        self.checkpoint = Path(args.model_folder) / f"{name}.torch"
        self.metrics_path = self.checkpoint.with_suffix(".metrics.json")
        self.metrics = {}
        self.elapsed = 0.
        self.started = time.perf_counter()
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        if args.save and getattr(args, "calc_div", False):
            Path(args.div_folder).mkdir(parents=True, exist_ok=True)

    def log(self, metrics):
        self.metrics.update(metrics)
        serializable = {key: str(value) if isinstance(value, float) and not math.isfinite(value)
                        else value for key, value in self.metrics.items()}
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.metrics_path.with_suffix(".json.tmp")
        with temporary.open("w") as stream:
            json.dump({"config": vars(self.args), "metrics": serializable}, stream,
                      default=str, indent=2, allow_nan=False)
        os.replace(temporary, self.metrics_path)
        print("METRICS " + json.dumps({key: serializable[key] for key in metrics}), flush=True)
        if wandb.run is not None:
            wandb.run.summary.update(metrics)
            wandb.log(metrics)  # Final metrics never reuse an earlier epoch step.

    def save(self, epoch, model, normalizers, geometry, optimizers, schedulers, iphi=None):
        if not self.args.save:
            return
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "version": 1, "epoch": epoch, "config": vars(self.args),
            "model_state_dict": model.state_dict(),
            "normalizers": [normalizer_state(n) for n in normalizers],
            "geometry": geometry,
            "optimizers": [o.state_dict() for o in optimizers],
            "schedulers": [s.state_dict() for s in schedulers],
            "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "elapsed": self.elapsed + time.perf_counter() - self.started,
        }
        if iphi is not None:
            state["iphi_state_dict"] = iphi.state_dict()
        temporary = self.checkpoint.with_suffix(".torch.tmp")
        torch.save(state, temporary)
        os.replace(temporary, self.checkpoint)

    def restore(self, model, normalizers, optimizers, schedulers, iphi=None):
        if self.args.resume is None:
            return 0
        state = torch.load(self.args.resume, map_location="cpu", weights_only=False)
        required = {"normalizers", "geometry", "epoch", "optimizers", "schedulers"}
        if iphi is not None:
            required.add("iphi_state_dict")
        missing = required - state.keys()
        if missing:
            raise ValueError(f"Incomplete checkpoint; missing {sorted(missing)}")
        allowed = {"resume", "eval_only", "save", "wandb", "project_name", "model_folder",
                   "div_folder", "data_root", "device", "gpu", "checkpoint_every",
                   "require_ood", "eval_ood", "no_ood", "calc_div"}
        for key, value in state["config"].items():
            if key not in allowed and vars(self.args).get(key) != value:
                raise ValueError(f"Checkpoint configuration mismatch: {key}")
        model.load_state_dict(state["model_state_dict"])
        if iphi is not None:
            iphi.load_state_dict(state["iphi_state_dict"])
        for normalizer, values in zip(normalizers, state["normalizers"]):
            for key, value in values.items():
                current = getattr(normalizer, key)
                setattr(normalizer, key, value.to(current.device) if torch.is_tensor(value) else value)
        for optimizer, values in zip(optimizers, state["optimizers"]):
            optimizer.load_state_dict(values)
        for scheduler, values in zip(schedulers, state["schedulers"]):
            scheduler.load_state_dict(values)
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        if torch.cuda.is_available() and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        self.elapsed = state["elapsed"]
        self.started = time.perf_counter()
        return state["epoch"] + 1
