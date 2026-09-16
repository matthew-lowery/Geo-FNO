# Omitted Geo-FNO OOD baselines

`train_geo_ood_missing.sh` runs only Geo-FNO no-div baselines omitted from `train_remaining.sh`. With the current `ram_dataset` files, it selects 21 jobs: seven datasets, seeds 1, 2, 3, and 500 epochs each.

| Dataset | Training samples | Requested points | Hours per seed |
|---|---:|---:|---:|
| Backward-facing step | 500 | 1000 | 2 |
| Cylinder, no shedding | 100 | 1000 | 2 |
| Cylinder, shedding | 10000 | 1000 | 6 |
| Lid cavity | 10000 | 1000 | 4 |
| Taylor–Green | 5000 | 500 | 3 |
| Taylor–Green spacetime | 5000 | 500 | 4 |
| Merging vortices | 500 | 500 | 2 |

The divergence weight is zero; there is no divergence training penalty. Model hyperparameters, normalization, splits, and loss definitions are unchanged. Each job trains a fresh baseline, then evaluates test loss, interior divergence, and OOD loss. Fresh training is necessary because the original checkpoints omitted the trained coordinate map. The updated trainers persist OOD metrics to W&B, JSON, and job output, and save complete periodic checkpoints.

Forced turbulence, species transport, and buoyancy are excluded because they were covered by the previous rerun selection. Taylor–Green coefficient variants are candidates but skipped while `taylor_green/data_coeffs_ood.mat` is absent. Any missing required training or OOD file skips the affected case, with its path printed. All selections are checked before submission.

From `Geo-FNO/me`:

```bash
bash train_geo_ood_missing.sh          # preview
bash train_geo_ood_missing.sh --check  # preflight without submission
bash train_geo_ood_missing.sh --submit
```

Optional filters: `--dataset NAME`, `--seed N`, `--ntrain N`. The script rejects Transolver, div-training, and `--remaining` selections. Results default to `/projects/bfel/mlowery/operator-benchmarks/geo-ood-recovery-20260916`; job names begin with `oodfix_`. Existing results are not overwritten by the default paths. `RAM_RESULTS_ROOT`, `RAM_DATA_ROOT`, and `TRAIN_PYTHON` remain available as overrides. Repeated submissions are not automatically deduplicated against Slurm.

The local preflight passes for all 21 jobs and 14 MATLAB files. No jobs have been submitted.
