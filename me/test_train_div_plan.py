import collections
import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

from ram_dataset_loader import required_dataset_paths


class TrainingPlanTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name) / "ram_dataset"
        self.problems = ["flow_cylinder_laminar", "flow_cylinder_shedding", "lid_cavity_flow",
                         "backward_facing_step", "buoyancy_cavity_flow", "taylor_green",
                         "taylor_green_coeffs", "taylor_green_spacetime", "taylor_green_spacetime_coeffs",
                         "merge_vortices_easier", "species_transport", "forced_turb"]
        for problem in self.problems:
            for ood in (False, True):
                for path in required_dataset_paths(problem, self.root, ood):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()

    def plan(self, *args):
        result = subprocess.run(
            ["bash", str(Path(__file__).with_name("train_div.sh")), "--dry-run", *args],
            check=True, capture_output=True, text=True,
            env={**os.environ, "RAM_DATA_ROOT": str(self.root)},
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

    def test_remaining_cases_and_required_ood(self):
        jobs = self.plan("--remaining")
        self.assertEqual(len(jobs), 84)
        counts = collections.Counter((model, options["dataset"]) for _, model, _, _, _, options in jobs)
        self.assertEqual(counts, {("geo", "forced_turb"): 30, ("trans", "forced_turb"): 30,
                                 ("geo", "species_transport"): 3, ("trans", "species_transport"): 3,
                                 ("geo", "buoyancy_cavity_flow"): 3, ("trans", "buoyancy_cavity_flow"): 15})
        for phase, model, _, hours, command, options in jobs:
            self.assertEqual("--require-ood" in command, phase != "div")
            self.assertIn("rerun-20260914", options["model-folder"])
            self.assertLessEqual(int(hours.split("=")[1]), 48)
            if options["dataset"] == "species_transport":
                self.assertEqual(phase, "baseline")
                self.assertEqual(hours, "hours=32" if model == "geo" else "hours=12")

    def test_remaining_dataset_filter(self):
        jobs = self.plan("--remaining", "--dataset", "forced_turb", "--phase", "forced")
        self.assertEqual(len(jobs), 30)

    def test_missing_ood_skips_only_baselines(self):
        path = self.root / "buoyancy_cavity_flow/data_ood.mat"
        path.rename(path.with_suffix(".unavailable"))
        jobs = self.plan("--remaining")
        self.assertEqual(len(jobs), 78)
        buoyancy = [j for j in jobs if j[5]["dataset"] == "buoyancy_cavity_flow"]
        self.assertEqual(len(buoyancy), 12)
        self.assertTrue(all(j[0] == "div" and j[1] == "trans" for j in buoyancy))

    def test_missing_training_file_skips_all_affected_runs(self):
        for problem in self.problems:
            with self.subTest(problem=problem):
                path = required_dataset_paths(problem, self.root)[0]
                backup = path.with_suffix(".unavailable")
                path.rename(backup)
                try:
                    self.assertEqual(self.plan("--dataset", problem), [])
                finally:
                    backup.rename(path)

    def test_spacetime_coefficients_need_both_files(self):
        path = self.root / "taylor_green/data_coeffs_ood.mat"
        path.rename(path.with_suffix(".unavailable"))
        jobs = self.plan("--dataset", "taylor_green_spacetime_coeffs")
        self.assertEqual(len(jobs), 24)
        self.assertTrue(all(j[0] == "div" for j in jobs))

    def test_reference_hyperparameters_and_hours(self):
        jobs = self.plan("--phase", "baseline")
        for _, model, _, hours, _, options in jobs:
            if model == "geo" and options["dataset"] == "forced_turb":
                self.assertEqual(hours, "hours=36")
                self.assertEqual(options["batch-size"], "10")
                self.assertEqual(options["modes"], "10")
            if model == "trans" and options["dataset"] == "backward_facing_step":
                self.assertEqual(hours, "hours=3")
                self.assertEqual(options["n-heads"], "8")
                self.assertEqual(options["slice-num"], "16")


if __name__ == "__main__":
    unittest.main()
