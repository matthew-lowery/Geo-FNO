import collections
import shlex
import subprocess
import unittest
from pathlib import Path


class TrainingPlanTest(unittest.TestCase):
    def plan(self, *args):
        result = subprocess.run(
            ["bash", str(Path(__file__).with_name("train_div.sh")), "--dry-run", *args],
            check=True, capture_output=True, text=True,
        )
        jobs = []
        for line in result.stdout.splitlines():
            tokens = shlex.split(line)
            options = dict(token[2:].split("=", 1) for token in tokens[4:]
                           if token.startswith("--") and "=" in token)
            jobs.append((tokens[0], tokens[1], tokens[2], tokens[3], tokens[4:], options))
        return jobs

    def test_complete_sweeps_and_unique_jobs(self):
        jobs = self.plan()
        self.assertEqual(len(jobs), 390)
        self.assertEqual(collections.Counter(job[0] for job in jobs),
                         {"div": 288, "baseline": 72, "forced": 30})
        self.assertEqual(len({(job[0], job[2]) for job in jobs}), len(jobs))
        for phase, model, _, _, command, options in jobs:
            self.assertTrue(options["data-root"].endswith("ram_dataset"))
            self.assertIn(options["seed"], {"1", "2", "3"})
            self.assertIn("--calc-div", command)
            if phase == "div":
                self.assertIn(options["div-loss-weight"], {"0.001", "0.01", "0.1", "1"})
                self.assertIn("--div-loss", command)
            else:
                self.assertEqual(options["div-loss-weight"], "0")
                self.assertNotIn("--div-loss", command)
                self.assertNotIn("--no-ood", command)
            if model == "trans":
                self.assertIn("transolver.train", command)

    def test_forced_turbulence_sizes_and_points(self):
        jobs = self.plan("--phase", "forced")
        self.assertEqual(len(jobs), 30)
        for _, _, _, _, _, options in jobs:
            self.assertEqual(options["dataset"], "forced_turb")
            self.assertEqual(options["npoints"], "7000")
            self.assertIn(options["ntrain"], {"100", "500", "1000", "5000", "7000"})

    def test_reference_hyperparameters_and_hours(self):
        jobs = self.plan("--phase", "baseline")
        for _, model, _, hours, _, options in jobs:
            if model == "geo" and options["dataset"] == "forced_turb":
                self.assertEqual(hours, "hours=15")
                self.assertEqual(options["batch-size"], "10")
                self.assertEqual(options["modes"], "10")
            if model == "trans" and options["dataset"] == "backward_facing_step":
                self.assertEqual(hours, "hours=3")
                self.assertEqual(options["n-heads"], "8")
                self.assertEqual(options["slice-num"], "16")


if __name__ == "__main__":
    unittest.main()
