"""Check selected MATLAB files before submitting any jobs."""

import os
import shlex
import sys
from pathlib import Path

from scipy.io import whosmat

from ram_dataset_loader import required_dataset_paths


def check_plan(lines):
    paths, jobs = set(), 0
    for line in lines:
        tokens = shlex.split(line)
        if not tokens:
            continue
        jobs += 1
        options = dict(t[2:].split("=", 1) for t in tokens if t.startswith("--") and "=" in t)
        problem, root = options["dataset"], options["data-root"]
        paths.update(required_dataset_paths(problem, root))
        if "--require-ood" in tokens:
            paths.update(required_dataset_paths(problem, root, ood=True))
        directory = Path(options["model-folder"])
        while not directory.exists():
            directory = directory.parent
        if not directory.is_dir() or not os.access(directory, os.W_OK):
            raise PermissionError(f"Result directory is not writable: {directory}")
    if not jobs:
        print("No runnable jobs selected; nothing to submit")
        return
    missing = sorted(str(p) for p in paths if not p.is_file())
    if missing:
        raise FileNotFoundError("Missing required files; no jobs submitted:\n" + "\n".join(missing))
    for path in sorted(paths):
        fields = {key: shape for key, shape, _ in whosmat(path)}
        if not fields:
            raise ValueError(f"Empty MATLAB file: {path}")
        for key in ("points", "velocity"):
            if key not in fields and "coeffs" not in path.name and "time" not in path.name:
                raise ValueError(f"Missing {key} in {path}")
    print(f"Preflight passed: {jobs} jobs, {len(paths)} MATLAB files")


if __name__ == "__main__":
    check_plan(sys.stdin)
