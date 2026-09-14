# Remaining runs and OOD recovery safeguards

## Requested jobs

Run from `Geo-FNO/me`. Every configuration uses seeds 1, 2, 3 and 500 epochs. The divergence coefficient is $\lambda$, multiplying the existing interior mean-square divergence penalty; its sweep is $\lambda\in\{0.001,0.01,0.1,1\}$. No-div runs have $\lambda=0$ and require OOD data.

| Dataset / phase | Geo-FNO jobs | Transolver jobs | Training samples | Requested points | Allocated hours, Geo / Trans |
|---|---:|---:|---|---:|---|
| Forced turbulence, no-div size sweep | 15 | 15 | 100, 500, 1000, 5000, 7000 | 7000 | 2–27 / 2–18 |
| Forced turbulence, div + no-div | 15 | 15 | 10000 | 7000 | 36 / 24 |
| Species transport, no-div only | 3 | 3 | 10000 | 7000 | 32 / 12 |
| Buoyancy cavity, Geo no-div; Trans div + no-div | 3 | 15 | 10000 | 5000 | 8 / 8 |
| Total | 36 | 48 | | | |

The 84-job rerun excludes species div training and Geo-FNO buoyancy div training. Div runs disable OOD; all baselines and the size sweep enable it. Architecture, learning rates, batch sizes, dataset splitting, and divergence orders remain those in `train_div.sh`.

Measured epoch times imply approximately 22.5 hours for Geo-FNO species, 7.2 hours for Transolver species, and 4.6 hours for Transolver buoyancy, before extra evaluation overhead. Forced-turbulence allocations are conservative estimates, not measured full-size runtimes for this batch. The size-sweep allocation scales with training count and includes an extra hour. All allocations are within Delta's documented [48-hour gpuA100x4 limit](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/running_jobs.html).

## Commands

```bash
bash train_remaining.sh                    # preview only
bash train_remaining.sh --check            # validate required files; submit nothing
bash train_remaining.sh --submit           # preflight first, then submit all selected jobs
```

Select a dataset with `--dataset forced_turb`, `--dataset species_transport`, or `--dataset buoyancy_cavity_flow`; combine with `--model geo|trans` and `--phase forced|div|baseline`. Optional `--seed` and `--ntrain` filters allow one-job pilots, for example:

```bash
bash train_remaining.sh --submit --dataset forced_turb --phase forced --model trans --seed 1 --ntrain 100
```

Do not submit the same pilot again with the full sweep unless intentionally repeating it. This launcher does not deduplicate against the live Slurm queue.

The default new result root is `/projects/bfel/mlowery/operator-benchmarks/rerun-20260914`, separate from the previous batch. Override with `RAM_RESULTS_ROOT`. `RAM_DATA_ROOT` selects the MATLAB dataset root; `TRAIN_PYTHON` selects the interpreter. Job names start with `rerun_`; output/error files go to `me/out` and `me/err`. Jobs request one A100, 32 GB host memory, partition `gpuA100x4`, account `bgcs-delta-gpu`.

**Required before buoyancy baseline submission:** `ram_dataset/buoyancy_cavity_flow/data_ood.mat`. It was missing in the previous runs and remains absent locally. The launcher will refuse the full selected submission if this file is missing on the submission host. Other datasets can be submitted separately; Transolver buoyancy div-only jobs do not require this OOD file.

Geo-FNO buoyancy must be retrained as a no-div baseline because the old saved checkpoint omitted its trained coordinate map. An old FNO-only checkpoint cannot reproduce the trained predictor. With a complete new checkpoint, the training CLI supports `--resume PATH --eval-only` using matching model/dataset arguments, to repeat evaluation without optimizer steps.

## Code and mathematical changes

- Final metrics no longer reuse the last training epoch as an explicit W&B step. They update the W&B summary and history, print a flushed `METRICS` line, and atomically update a `.metrics.json` file beside the checkpoint. Geo-FNO records both `ood_loss` and its legacy `ood/<dataset>` key. No loss values are inferred or filled in.
- Checkpoints are saved every 25 epochs and before final evaluation, using atomic replacement. They contain FNO/Transolver weights, Geo-FNO coordinate-map weights, normalizer statistics, physical grids, CLI configuration, optimizer/scheduler states, RNG states, and the last completed epoch. `--resume PATH` continues from the next epoch. Only load trusted checkpoints. Old incomplete checkpoints are explicitly rejected.
- The 2D coordinate map's fixed center and Fourier frequencies are now registered buffers: identical values, but correctly moved with the model and included in its state dictionary.
- Transolver's irregular-mesh attention previously divided by an unconstrained learned temperature. Let $\tau_h$ be the learned temperature for head $h$, $a_{h,q,g}$ its slice logit at point $q$ and slice $g$, and $G$ the number of slices. It now uses $\widetilde\tau_h=\min(5,\max(0.1,\tau_h))$ and slice weights

$$
w_{h,q,g}=\frac{\exp(a_{h,q,g}/\widetilde\tau_h)}{\sum_{r=1}^{G}\exp(a_{h,q,r}/\widetilde\tau_h)}.
$$

  This matches the existing structured-mesh temperature bound and prevents division by zero. It changes the forward map only when a temperature leaves the interval. It fixes a demonstrated singularity, but the old logs alone do not prove this caused every historical NaN.
- Nonfinite training losses or gradients abort before the next optimizer update. Finite gradients are not clipped: the norm threshold is infinite. No failed sample is silently removed from the objective.
- Final relative-error reductions use float64 to avoid float32 norm overflow; predictions and training still use their previous dtype. Existing metric definitions are preserved: both models' test loss and Transolver OOD loss compare field magnitudes; Geo-FNO 2D/3D OOD compares vector components, while the Geo coefficient entry compares magnitudes. These OOD definitions should not be treated as identical cross-model metrics.

For precision, let $\widehat u_i$ and $u_i$ be predicted and target arrays for evaluation sample $i$, with $M$ evaluation samples. The component-based relative error is $M^{-1}\sum_{i=1}^{M}\|\operatorname{vec}(\widehat u_i-u_i)\|_2/\|\operatorname{vec}(u_i)\|_2$, where $\operatorname{vec}$ flattens points and components. The magnitude-based metric replaces each point's component vector by its Euclidean norm before applying that formula. No denominator regularization has been added; nonfinite evaluation metrics remain explicitly represented in the JSON/logs.

## Verification and limits

32 regression tests cover monotone final logging, local metric persistence, complete checkpoint reload, RNG restoration, incomplete-checkpoint rejection, missing-OOD rejection, the exact 84-job plan, and finite attention at zero raw temperature. Tiny CPU runs exercise all three Geo-FNO entries and Transolver through training, OOD evaluation, checkpointing, and evaluation-only reload. An actual offline W&B run confirms OOD survives in binary history after epoch 499. A one-epoch Transolver CPU run using the real RAM forced-turbulence training/OOD files (two training samples, 128 points, reduced model) also completes with finite OOD loss persisted to JSON.

The local forced-turbulence and species preflights pass; buoyancy baseline preflight fails on its missing OOD file. No jobs have been submitted, and full-size A100 memory, numerical stability, remote OOD data, and end-to-end wall times remain to be verified on Delta.
