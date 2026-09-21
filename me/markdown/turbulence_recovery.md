# Turbulence recovery — September 16, 2026

From `Geo-FNO/me` on Delta:

```bash
bash train_turb_remaining.sh --dry-run
bash train_turb_remaining.sh --submit
```

The launcher reuses `train_div.sh` profiles. Both models retain 500 epochs,
7,000 points, seeds 1–3, and all existing optimizer/model settings. Main runs
use 10,000 training samples with divergence weights 0.001, 0.01, 0.1, 1, plus
a no-div baseline. The size sweep uses 100, 500, 1,000, 5,000, and 7,000 training
samples, **without divergence training**. Every no-div case requires OOD evaluation;
div-trained cases retain `--no-ood`. No objective, normalization, or data changed.

## Completion and duplicate protection

The copied `wandb_final/wandb` cache contains 23 completed turbulence cases:

| Model | Completed cases | Remaining candidates |
|---|---|---:|
| Geo-FNO | Sizes 100, 500, 1,000: all three seeds | 21 |
| Transolver | Sizes 100, 500, 1,000: all seeds; size 5,000: seed 1; all four main div weights: seed 1 | 16 |

`turb_recovery_snapshot.json` records the 23 source run IDs and their profiles,
so these skips work even without copying the W&B cache to Delta. Matching includes
model, size, seed, divergence weight, and training hyperparameters.

The launcher also checks current `wandb/`, `wandb_final/`, durable metrics in the
new results root and `rerun-20260914`, and pending/running Slurm jobs (including
old job-name prefixes). Submission refuses to proceed without a successful live
queue check. Concurrent invocations of this launcher share a lock. Do not launch
the older unfiltered scripts concurrently.

Five cached Geo-FNO attempts are partial, **not confirmed timed out**: the four
main div runs at seed 1 and the size-5,000 seed-1 baseline. The snapshot has no
termination record for them. Active jobs are skipped; inactive unfinished jobs
resume their periodic checkpoints when present. Checkpoints retain optimizer,
scheduler, normalization, IPHI, and RNG state. A final-epoch checkpoint proceeds
directly to evaluation. Without a checkpoint, an unfinished case starts afresh.

New output root: `/projects/bgcs/mlowery/operator-benchmarks/turb-recovery-20260916`.
Overrides: `RAM_RESULTS_ROOT` or `--results-root`, `RAM_DATA_ROOT`, `TRAIN_PYTHON`;
additional old result/cache locations: `--previous-results-root` / `--wandb-root`.
Only datasets present in `ram_dataset` are selected. `--model`, `--phase`,
`--seed`, and `--ntrain` optionally narrow selection.

## Runtime evidence

The A100 logs give approximately 233–236 seconds per epoch for full-size Geo-FNO
and 46.6–46.8 seconds for full-size Transolver. Geo-FNO's partial size-5,000 run
contains 322 logged epochs at roughly 115 seconds each. The four partial full-size
Geo runs contain 126–182 epochs; their original allocations were 36 hours.
Completed Transolver main div runs took approximately 6.5 hours, not 24 hours.

Let $n$ denote the training-set size, $r_m$ the conservative full-size seconds per
epoch for model $m$, and $H_m(n)$ the requested integer hours. The launcher uses
$r_{\mathrm{Geo}}=236$ and $r_{\mathrm{Trans}}=47$, linear scaling in $n$, 35% time
headroom, and an additional hour for setup and evaluation:

$$
H_m(n)=\left\lceil 1.35\frac{500r_m}{3600}\frac{n}{10000}+1\right\rceil. \tag{1}
$$

Equation (1) allocates the full training duration even when resuming.

| Training samples | Geo-FNO hours | Transolver hours |
|---:|---:|---:|
| 5,000 | 24 | 6 |
| 7,000 | 32 | 8 |
| 10,000 | 46 | 10 |

## Species transport findings

- Original Geo-FNO attempts hit the two-hour limit at about 44 epochs. At roughly
  162 seconds/epoch, 500 epochs need about 22.5 hours. The subsequent recovery
  script already includes three no-div Geo runs with 32 hours each; none appear
  in the new cache. Their live state could not be verified because Delta SSH
  authentication failed. Absence from this cache does not establish failure.
- Original Transolver attempts also timed out at three hours. The three new
  no-div runs completed in about 7.25 hours, within their 12-hour allocations.
  Seeds 1/2/3 logged ID losses 0.002961/0.002637/0.002612 and OOD losses
  1.587392/1.153025/1.315747. Their run IDs are `3lve36ss`, `1odurtu1`, `bsw06ns4`.
- **The raw `ram_dataset/species_transport/data_ood.mat` contains 500 exactly
  identical output velocity fields**, despite differing inlet functions. All
  500 arrays equal the first array elementwise; the first and last input arrays
  differ by up to 19.94292. The training file does not have this constant-output
  property: only one of its 10,250 targets equals its first target. This is a
  data-generation/export concern, not introduced by either loader. Its intent
  must be checked before interpreting these OOD scores.
- Large divergence is also present in the targets. On the first two held-out
  samples returned by the shared loader, using 7,000 points, physical-coordinate
  RBF-FD order 4, and the current 6,089-point interior mask, the exact target
  fields produce maximum absolute divergence 84,947.61 and median 1,475.13.
  Transolver predictions give maxima around 104,000 and medians around 980.
  The two-sample diagnostic is not the full test aggregate; it demonstrates
  that large values are not exclusive to predictions. It does not establish
  whether geometry, differentiation, or the source fields cause those values.

No species data or species training settings were modified by this recovery.
