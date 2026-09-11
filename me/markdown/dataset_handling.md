# Dataset handling and experiment sweeps

Updated 11 September 2026. Airfoil excluded.

## Shared data and execution

Both active model paths live in `Geo-FNO/me`: the Geo-FNO `ramansh_*.py` entries and `python -m transolver.train`. Transolver's runtime and MIT license are included under `me/transolver`; the original Transolver checkout, its data, and W&B logs are left intact.

Every experiment reads MATLAB files under `ram_dataset`. Both active models use [ram_dataset_loader.py](../ram_dataset_loader.py); [transolver/data.py](../transolver/data.py) only adapts coefficient inputs to Transolver's shared-grid representation. No active experiment reads legacy NPZ datasets.

**No boundary points are explicitly removed from model input/output functions.** `--npoints` still selects a Fekete subset; that subset may contain boundaries. Fekete indices are interpreted on the original stored grid, with one-based indices converted to zero-based before selection. Merging-vortex coordinates are reduced from three columns to their two spatial coordinates; this drops a coordinate, not points.

Taylor–Green coefficient targets now come from `taylor_green/data_coeffs_matt.mat`, which has the full 7,447-point grid. The older `data_coeffs.mat` has only 7,288 points. Their 10,200 coefficient pairs were checked and agree exactly. Spacetime targets use `data_time.mat` at times 0.7, 0.8, 0.9, 1.0.

The independent `ram_dataset/dataset.py` has not been rewritten. Its boundary heuristics are reproduced in [dataset_boundaries.py](../dataset_boundaries.py) for divergence only, without importing its function-grid trimming.

## Normalization

Let $a_{s,j,c}$ be a function value for training sample $s$, point $j$, and channel $c$. Let $S$ and $P$ denote the training sample count and point count used to fit the corresponding normalizer. Inputs and targets have separate fitted statistics; test and OOD inputs reuse training statistics.

| Model | Input functions | Output functions | Grid normalization with `--norm-grid` |
|---|---|---|---|
| Geo-FNO | Mean and sample standard deviation across training samples separately at every point/channel; denominator adds $10^{-5}$. | Same pointwise statistics fitted to targets. Targets remain physical; predictions are decoded before data/divergence loss. | Dataset-specific transforms in the table below. Without the flag, coordinates remain physical. |
| Transolver | Mean and sample standard deviation pooled over samples and points, separately per channel; denominator adds $10^{-8}$. Fit after coefficient broadcasting and zero-padding. | Separate pooled channel statistics on output targets. Training targets are encoded, then targets and predictions are decoded before loss. | One min–max transform on the union of input/output coordinates, including time. Denominator has lower bound $10^{-12}$. |

For Geo-FNO, define the pointwise mean $\mu_{j,c}$ and sample standard deviation $\sigma_{j,c}$ by

$$
\mu_{j,c}=\frac{1}{S}\sum_{s=1}^{S}a_{s,j,c},\qquad
\sigma_{j,c}^2=\frac{1}{S-1}\sum_{s=1}^{S}(a_{s,j,c}-\mu_{j,c})^2,\qquad
\widetilde a_{s,j,c}=\frac{a_{s,j,c}-\mu_{j,c}}{\sigma_{j,c}+10^{-5}}.
$$

For Transolver, define the channel mean $\mu_c$ and sample standard deviation $\sigma_c$ by

$$
\mu_c=\frac{1}{SP}\sum_{s=1}^{S}\sum_{j=1}^{P}a_{s,j,c},\qquad
\sigma_c^2=\frac{1}{SP-1}\sum_{s=1}^{S}\sum_{j=1}^{P}(a_{s,j,c}-\mu_c)^2,\qquad
\widetilde a_{s,j,c}=\frac{a_{s,j,c}-\mu_c}{\sigma_c+10^{-8}}.
$$

The reference `dataset.py` itself does not normalize coordinates or targets. Its optional `norm_bool` normalizes non-coefficient inputs pointwise using population standard deviation and forces constant entries to zero. That optional normalization is **not additionally applied** by either model loader.

## Dataset-specific grids and representations

Both implementations allow different input/output channel counts. Geo-FNO allows different input/output point sets but needs a common coordinate dimension. Transolver emits one prediction per input token: its union-grid construction retains input sites, pads missing input values with physical zeros, and selects only output indices for loss and evaluation. Padding is not omission of boundary points.

| Dataset | Geo-FNO geometry and grid normalization | Transolver geometry; all use union min–max when enabled | Boundary points in functions |
|---|---|---|---|
| Backward-Facing Step | Insert inlet profile into a zero two-component field on the output mesh; shared 2D min–max. The first two inlet entries are swapped to match the stored convention. | Same inlet embedding; input/output mesh coincides. | Retained if selected. |
| Cylinder, no shedding | Initial/final velocity on one 2D grid; shared min–max. | Same grid, restored output ordering after union sorting. | Retained if selected. |
| Cylinder, shedding | Same 2D geometry and normalization. | Same-grid initial/final velocity. | Retained if selected. |
| Lid-Driven Cavity | Same 2D grid; shared min–max. | Same input/output grid. | Retained if selected. |
| Buoyancy-Driven Cavity | Temperature input and vector velocity target on one 2D grid; shared min–max. | Scalar-to-vector channels on the same grid. | Retained if selected. |
| Taylor–Green | Initial/final velocity on one full 2D grid; shared min–max. | Same full spatial grid. | Retained, including periodic copies if selected. |
| Taylor–Green coefficients | Two scalar coefficients at artificial 2D sites `(0,1)`, `(0,0)`; velocity output on the full spatial mesh. Normalize input/output grids independently, adding $10^{-6}$ to spans. | Broadcast the coefficient pair as two constant channels across the output mesh. Each coefficient has its own channel normalization. | Full-grid targets; no spatial point removal. |
| Taylor–Green Spacetime | Embed the initial 2D mesh at `(x,y,0)`; output is `(x,y,t)`. Normalize spatial coordinates of both grids using initial-grid extrema; leave time unchanged. | Union of initial-time and output-time sites. Zero-pad inputs at target-only locations. | Retained at every output time. |
| Taylor–Green Spacetime coefficients | Two scalar coefficients at artificial 3D sites `(0,1,0)`, `(0,0,0)`; separate output spacetime grid. Normalize all output coordinates including time; artificial input grid unchanged. | Broadcast coefficients at initial-time spatial sites, then zero-pad target-time sites in the union. | Retained at every output time. |
| Merging Vortices | Scalar vorticity and two-component velocity on one 2D grid; shared min–max. | Same 2D grid after discarding the unused coordinate column. | Retained, including periodic copies if selected. |
| Species Transport | Inlet surface embedded at `z=0` in 3D; output on a different volume grid. Normalize both using output-grid extrema. | Union of inlet and volume points; inputs zero outside inlet. No loss at input-only sites. | Retained in inlet and output grids. |
| Forced Isotropic Turbulence | Forcing and velocity on one 3D grid; normalize both using output-grid extrema. | Same retained 3D grid. | Retained, including $2\pi$ periodic copies if selected. |

## Divergence-only exclusions

These masks are used by **both models**, for both the divergence training penalty and reported interior statistics. Derivative matrices are built on the full selected physical output grid. Boundary points can therefore remain stencil neighbors; their own divergence values are excluded from the penalty.

| Dataset | Excluded divergence evaluation locations |
|---|---|
| Backward-Facing Step | Reference rectangle edges $x=0,15$ and $y=-0.5,0.5$. |
| Cylinder, no shedding | Reference outer rectangle edges $x=0,20$ and $y=0,14$. |
| Cylinder, shedding | Same outer rectangle edges. |
| Lid-Driven Cavity | Unit-square edges. |
| Buoyancy-Driven Cavity | Unit-square edges. |
| Taylor–Green and coefficient variant | Square edges at $x,y=0,2\pi$. |
| Taylor–Green Spacetime and coefficient variant | Same spatial square edges, independently at every output time; no time derivative or temporal-boundary exclusion. |
| Merging Vortices | Square edges at $x,y=0,2\pi$. |
| Species Transport | Coordinate matches to `full_domain_boundary_points.mat`, with maximum absolute coordinate difference at most $10^{-12}$. |
| Forced Isotropic Turbulence | Points on any $2\pi$ face, moving the reference loader's periodic-copy exclusion into divergence only; zero-coordinate copies remain eligible. |

Rectangle comparisons use `numpy.isclose(..., atol=1e-12)` with its default relative tolerance; forced-turbulence face comparisons use default `isclose`. The cylinder/step heuristics deliberately reproduce `dataset.py`: they do not additionally identify the circular obstacle or internal step walls.

Let $B$ be the batch size, $T$ the number of output times (four for spacetime, otherwise one), $I$ the retained divergence-evaluation index set, $u_{b,t,k}$ the decoded predicted velocity component $k$ for batch sample $b$ and time $t$, and $D_k$ the RBF-FD differentiation matrix along spatial coordinate $k$. In spatial dimension $d$, the divergence penalty is

$$
L_{\mathrm{div}}=\frac{1}{T}\sum_{b=1}^{B}\sum_{t=1}^{T}
\frac{1}{|I|}\sum_{j\in I}
\left[\sum_{k=1}^{d}(D_k u_{b,t,k})_j\right]^2.
$$

The batch objective is $L_{\mathrm{data}}+\lambda L_{\mathrm{div}}$, where $L_{\mathrm{data}}$ is the sum of per-sample relative velocity-field $L^2$ errors and $\lambda$ is `--div-loss-weight`. Data loss still includes selected boundary points. A no-div run omits the penalty entirely; `--calc-div` does not enable training regularization. Only masked `*_interior` divergence statistics are logged; boundary-inclusive `*_all` metrics have been removed.

`--div-order` is polynomial degree $p$. With $Q=\binom{p+d}{d}$ polynomial terms, the implementation uses $2Q+1$ nearest neighbors; it does not wrap neighbor searches periodically.

## Consolidated experiment plan

[train_div.sh](../train_div.sh) defaults to a dry run. It uses model settings and hours from `train_div_geo.sh` and `train_div_trans.sh`, including datasets whose reference commands were commented out. Airfoil is excluded.

| Phase | Seeds | Datasets/models | Training sizes | Divergence coefficients | Jobs |
|---|---|---|---|---|---:|
| `div` | 1, 2, 3 | All 12 variants, both models | Reference count per dataset | 0.001, 0.01, 0.1, 1 | 288 |
| `baseline` | 1, 2, 3 | All 12 variants, both models | Reference count per dataset | No divergence training | 72 |
| `forced` | 1, 2, 3 | Forced turbulence, both models | 100, 500, 1000, 5000, 7000 | No divergence training | 30 |

Total: **390 jobs**. Every forced-turbulence job uses **7,000 points**, including the main reference-count runs. The size sweep uses nested prefixes of the same deterministic seed-0 data permutation; model seeds vary independently.

There is no forced-turbulence row in the Transolver reference script. Its provisional profile uses the 3D species architecture (128 hidden channels, 5 layers, 4 heads, 32 slices, batch size 20) and the 15-hour Geo-FNO forced-turbulence time limit. Geo-FNO forced turbulence retains its reference batch size 10. Where a Geo-FNO reference command omits `--div-order`, the existing default degree 2 is retained; explicitly specified degree 3 is preserved. Transolver uses degree 4.

```bash
bash train_div.sh --dry-run
bash train_div.sh --submit
bash train_div.sh --dry-run --phase forced --model trans
```

Run from `Geo-FNO/me`, or provide the script's absolute path. Override `RAM_DATA_ROOT`, `RAM_RESULTS_ROOT`, and `TRAIN_PYTHON` as needed. The default interpreter is the reference Delta environment; the dataset root uses the cluster RAM directory when present, otherwise the local sibling `ram_dataset`. Results are separated by model, phase, and coefficient; filenames include dataset, seed, sample count, and point count. All submitted jobs use partition `gpuA100x4` and account `bgcs-delta-gpu`.

## OOD and verification

Div-regularized sweeps skip OOD evaluation; no-div baselines and the forced size sweep attempt it after training. Missing optional files yield `ood_available=False`, with an explanatory message, without discarding trained checkpoints. The local buoyancy OOD file and Taylor–Green coefficient OOD file are absent; the latter blocks both coefficient variants. No synthetic OOD targets or placeholder losses are generated.

Validation covers Fekete indexing, boundary retention, reference-geometry masks, spacetime layouts, masked-divergence gradients, and all 390 dry-run commands. Small CPU training/evaluation checks exercise the relocated Transolver entry point. No Slurm jobs have been submitted.
