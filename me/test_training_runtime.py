"""Exercise the real train/evaluate/save paths on tiny synthetic CPU datasets."""

import contextlib
import io
import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from ram_dataset_loader import OperatorDataset
from test_run_artifacts import Tracker


class TrainingRuntimeTest(unittest.TestCase):
    def run_training(self, entry, dimension, problem, coefficient_inputs=False):
        random = np.random.default_rng(8)
        points = random.uniform(.1, .9, (16, dimension))
        input_points = points[:2] if coefficient_inputs else points
        x = random.uniform(.2, 1., (2, len(input_points), 1))
        y = random.uniform(.2, 1., (2, 16, dimension))
        data = OperatorDataset(input_points, points, x, y, x.copy(), y.copy())
        ood = (input_points, points, x * 1.1, y * 1.1)
        tracker = Tracker()
        tracker.init = lambda **kwargs: tracker
        tracker.finish = lambda: None
        tracker.config = type("Config", (), {"update": lambda *args: None})()
        with tempfile.TemporaryDirectory() as folder, contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {"wandb": tracker}))
            stack.enter_context(patch("run_artifacts.wandb", tracker))
            stack.enter_context(patch("torch.Tensor.cuda", lambda tensor, *a, **kw: tensor))
            stack.enter_context(patch("torch.nn.Module.cuda", lambda model, *a, **kw: model))
            stack.enter_context(patch("ram_dataset_loader.load_dataset", return_value=data))
            stack.enter_context(patch("ram_dataset_loader.try_load_ood_dataset", return_value=ood))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            arguments = [entry, f"--dataset={problem}", "--ntrain=2", "--npoints=16",
                         "--epochs=2", "--batch-size=2", "--checkpoint-every=1",
                         "--save", "--calc-div", "--div-order=1", "--div-loss",
                         "--div-loss-weight=0.001", f"--model-folder={folder}",
                         f"--div-folder={folder}"]
            if entry == "transolver.train":
                arguments += ["--device=cpu", "--n-hidden=8", "--n-layers=1",
                              "--n-heads=2", "--slice-num=4"]
                stack.enter_context(patch("transolver.data.load_dataset", return_value=data))
                stack.enter_context(patch("transolver.data.load_ood_dataset", return_value=ood))
                from transolver import train
                stack.enter_context(patch.object(train, "wandb", tracker))
                execute = train.main
            else:
                arguments += ["--width=4", "--modes=2", "--res1d=4"]
                execute = lambda: runpy.run_path(str(Path(__file__).with_name(entry)), run_name="__main__")
            stack.enter_context(patch("sys.argv", arguments))
            execute()
            self.assertTrue(np.isfinite(tracker.summary["ood_loss"]))
            checkpoint = next(Path(folder).glob("*.torch"))
            state = torch.load(checkpoint, weights_only=False)
            self.assertEqual(state["epoch"], 1)
            if entry != "transolver.train":
                self.assertIn("iphi_state_dict", state)
            saved = json.loads(checkpoint.with_suffix(".metrics.json").read_text())["metrics"]
            self.assertEqual(saved["ood_loss"], tracker.summary["ood_loss"])
            expected = tracker.summary["ood_loss"]
            arguments += ["--eval-only", f"--resume={checkpoint}"]
            execute()
            self.assertAlmostEqual(tracker.summary["ood_loss"], expected, places=6)

    def test_geo_2d_ood_and_eval_only(self):
        self.run_training("ramansh_2d.py", 2, "flow_cylinder_laminar")

    def test_geo_coefficient_ood_and_eval_only(self):
        self.run_training("ramansh_2d_diff_grids.py", 2, "taylor_green_coeffs", True)

    def test_geo_3d_ood_and_eval_only(self):
        self.run_training("ramansh_3d.py", 3, "forced_turb")

    def test_transolver_ood_and_eval_only(self):
        self.run_training("transolver.train", 3, "forced_turb")


if __name__ == "__main__":
    unittest.main()
