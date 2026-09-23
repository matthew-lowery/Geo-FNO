import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from run_artifacts import RunArtifacts, validate_runtime
from transolver.utils.normalizer import UnitTransformer


class Tracker:
    def __init__(self):
        self.step = 0
        self.summary = {}
        self.run = self

    def log(self, metrics, step=None):
        step = self.step if step is None else step
        if step < self.step:
            return
        self.summary.update(metrics)
        self.step = step + 1


class RunArtifactsTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.args = argparse.Namespace(model_folder=self.folder.name, save=True, resume=None,
                                       eval_only=False, require_ood=False, checkpoint_every=25)

    def test_final_metrics_survive_epoch_step_advance_and_have_local_copy(self):
        tracker = Tracker()
        tracker.log({"train_loss": 1.}, step=499)
        with patch("run_artifacts.wandb", tracker), contextlib.redirect_stdout(io.StringIO()) as output:
            artifacts = RunArtifacts(self.args, "run")
            artifacts.log({"ood_available": True})
            artifacts.log({"ood_loss": .123})
            artifacts.log({"test_div/max_abs_interior": 2.})
        self.assertEqual(tracker.summary["ood_loss"], .123)
        self.assertEqual(json.loads(artifacts.metrics_path.read_text())["metrics"]["ood_loss"], .123)
        self.assertIn('"ood_loss": 0.123', output.getvalue())

    def test_nonfinite_metric_is_not_silently_lost(self):
        with patch("run_artifacts.wandb", Tracker()):
            artifacts = RunArtifacts(self.args, "run")
            artifacts.log({"ood_loss": float("inf")})
        self.assertEqual(json.loads(artifacts.metrics_path.read_text())["metrics"]["ood_loss"], "inf")

    def test_finalize_metrics_relogs_complete_finite_record(self):
        tracker = Tracker()
        with patch("run_artifacts.wandb", tracker), contextlib.redirect_stdout(io.StringIO()) as output:
            artifacts = RunArtifacts(self.args, "run")
            artifacts.log({"test_loss": .1})
            artifacts.log({"total_train_time": 12.})
            artifacts.finalize_metrics(
                {"test_loss", "total_train_time"},
                {"test_loss", "total_train_time"},
            )
        self.assertEqual(tracker.summary["test_loss"], .1)
        self.assertIn("METRICS_COMPLETE", output.getvalue())

    def test_finalize_metrics_rejects_missing_or_nonfinite_values(self):
        with patch("run_artifacts.wandb", Tracker()):
            artifacts = RunArtifacts(self.args, "run")
            artifacts.log({"test_loss": float("nan")})
            with self.assertRaisesRegex(RuntimeError, "total_train_time"):
                artifacts.finalize_metrics(
                    {"test_loss", "total_train_time"},
                    {"test_loss", "total_train_time"},
                )
            artifacts.log({"total_train_time": 12.})
            with self.assertRaisesRegex(FloatingPointError, "test_loss"):
                artifacts.finalize_metrics(
                    {"test_loss", "total_train_time"},
                    {"test_loss", "total_train_time"},
                )

    def test_complete_checkpoint_roundtrip_and_rng(self):
        torch.manual_seed(4)
        model, iphi = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
        optimizer = torch.optim.Adam(list(model.parameters()) + list(iphi.parameters()))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
        normalizer = UnitTransformer(torch.randn(3, 4, 2))
        x = torch.randn(3, 2)
        model(iphi(x)).square().sum().backward()
        optimizer.step()
        scheduler.step()
        expected = model(iphi(x)).detach()
        artifacts = RunArtifacts(self.args, "run")
        artifacts.save(24, model, (normalizer,), {"points": np.zeros((4, 2))},
                       (optimizer,), (scheduler,), iphi)
        expected_rng = torch.rand(4)
        for p in list(model.parameters()) + list(iphi.parameters()):
            p.data.zero_()
        self.args.resume = artifacts.checkpoint
        self.assertEqual(artifacts.restore(model, (normalizer,), (optimizer,), (scheduler,), iphi), 25)
        torch.testing.assert_close(model(iphi(x)), expected)
        torch.testing.assert_close(torch.rand(4), expected_rng)
        self.assertEqual(scheduler.last_epoch, 1)

    def test_old_checkpoint_without_iphi_is_rejected(self):
        path = Path(self.folder.name) / "old.torch"
        model = torch.nn.Linear(2, 2)
        torch.save({"model_state_dict": model.state_dict()}, path)
        self.args.resume = path
        with self.assertRaisesRegex(ValueError, "iphi_state_dict"):
            RunArtifacts(self.args, "run").restore(model, (), (), (), model)

    def test_missing_required_ood_fails_before_training(self):
        self.args.require_ood = True
        self.args.dataset = "buoyancy_cavity_flow"
        self.args.data_root = self.folder.name
        with self.assertRaisesRegex(FileNotFoundError, "before training"):
            validate_runtime(self.args)


if __name__ == "__main__":
    unittest.main()
