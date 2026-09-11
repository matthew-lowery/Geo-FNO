# W&B results and completion tracker

Each dataset has a no-div baseline (λ = 0) and divergence-penalty weights λ = 0.01, 0.1, 0.5, 1. Blank metric cells indicate missing results. `Missing` means no result is available in this table or the inspected Transolver cache; `NaN` and `no test result` mark attempted runs needing completion. Additional rows retain available divergence orders.

Values are mean ± population standard deviation across the listed seeds; a single seed has no standard deviation. For Transolver, only runs with finite test loss contribute to any reported mean; excluded attempts are counted in `Seeds / status`. `Div. order` is the RBF-FD polynomial degree used to evaluate divergence and, for div runs, its training penalty. Divergence uses the absolute interior values (`max / median`); values evaluated at different orders are different discrete diagnostics.

Time is seconds: Geo-FNO reports `total_train_time`; Transolver reports W&B `_runtime` (elapsed run time, including work outside training). `—` means an unavailable metric within an otherwise populated result; `∞` means the logged OOD loss was infinite for every contributing seed.

| Framework | Dataset | Run | λ | Div. order | Seeds / status | Time (s) | Test loss | Interior test div (max / median) | OOD loss |
|---|---|---|---:|---:|---|---:|---:|---:|---:|
| **Geo-FNO** | **2D Backward-Facing Step** | no-div | 0 | 5 | 1, 2, 3 | 195.44 ± 0.851 | 9.949e-5 ± 1.73e-5 | 1.967 ± 0.00248 / 4.494e-4 ± 1.12e-5 | 1.013 ± 0.194 |
|  |  | div | 0.01 | 3 | 1, 2, 3 | 217.8 ± 1.03 | 1.035e-4 ± 1.91e-5 | 1.980 ± 0.00212 / 4.848e-4 ± 8.37e-6 | 1.080 ± 0.0484 |
|  |  | div | 0.1 | 3 | 1, 2, 3 | 218.6 ± 3.51 | 1.043e-4 ± 9.78e-6 | 1.972 ± 0.00179 / 4.732e-4 ± 8.90e-6 | 0.850 ± 0.0859 |
|  |  | div | 0.5 |  | Missing |  |  |  |  |
|  |  | div | 1 | 3 | 1, 2, 3 | 216.1 ± 3.13 | 2.127e-3 ± 2.00e-5 | 1.286 ± 0.0503 / 6.612e-4 ± 5.63e-5 | 7.364 ± 8.62 |
|  | **2D Flow Past a Cylinder (no vortex shedding)** | no-div | 0 | 5 | 1, 2, 3 | 175.81 ± 0.585 | 1.618e-4 ± 4.77e-5 | 0.7033 ± 0.00041 / 9.457e-5 ± 2.19e-5 | 0.645 ± 0.306 |
|  |  | div | 0.01 | 3 | 1, 2, 3 | 180.9 ± 0.637 | 1.555e-4 ± 3.23e-5 | 0.7716 ± 0.00039 / 1.056e-4 ± 1.10e-5 | 0.666 ± 0.0625 |
|  |  | div | 0.1 | 3 | 1, 2, 3 | 180.8 ± 0.551 | 1.803e-4 ± 2.75e-5 | 0.7704 ± 0.00076 / 1.091e-4 ± 1.49e-6 | 1.696 ± 0.741 |
|  |  | div | 0.5 |  | Missing |  |  |  |  |
|  |  | div | 1 | 3 | 1, 2, 3 | 180.5 ± 0.325 | 1.838e-4 ± 2.36e-5 | 0.7601 ± 0.00248 / 1.094e-4 ± 1.01e-5 | 0.772 ± 0.381 |
|  | **2D Flow Past a Cylinder (vortex shedding)** | no-div | 0 | 5 | 1, 2, 3 | 11414 ± 10.3 | 4.783e-5 ± 1.19e-5 | 4.450 ± 0.00057 / 0.001450 ± 1.41e-5 | 0.626 ± 0.325 |
|  |  | div | 0.01 | 3 | 1, 2, 3 | 11880 ± 9.0 | 4.784e-5 ± 1.09e-5 | 4.930 ± 0.00621 / 1.433e-3 ± 1.15e-5 | 1.689 ± 1.90 |
|  |  | div | 0.1 | 3 | 1, 2, 3 | 11860 ± 10.7 | 2.032e-4 ± 2.18e-5 | 4.278 ± 0.0162 / 1.455e-3 ± 2.56e-5 | 29.18 ± 40.6 |
|  |  | div | 0.5 |  | Missing |  |  |  |  |
|  |  | div | 1 | 3 | 1, 2, 3 | 11880 ± 5.76 | 1.200e-2 ± 3.16e-3 | 1.138 ± 0.222 / 1.491e-3 ± 2.00e-4 | 352 ± 495 |
|  | **2D Lid-Driven Cavity Flow** | no-div | 0 | 5 | 1, 2, 3 | 6256.5 ± 7.2 | 1.509e-4 ± 2.04e-5 | 21.58 ± 0.0047 / 0.5734 ± 3.14e-4 | 0.590 ± 0.098 |
|  |  | div | 0.01 | 3 | 1, 2, 3 | 6756 ± 16.4 | 2.027e-3 ± 1.64e-5 | 9.507 ± 0.00391 / 0.3617 ± 5.24e-4 | 0.382 ± 0.0136 |
|  |  | div | 0.1 | 3 | 1, 2, 3 | 6710 ± 16.5 | 1.109e-2 ± 3.13e-4 | 2.314 ± 0.0205 / 0.04074 ± 6.56e-5 | 11.27 ± 10.6 |
|  |  | div | 0.5 |  | Missing |  |  |  |  |
|  |  | div | 1 | 3 | 1, 2, 3 | 6746 ± 18.6 | 1.565e-2 ± 1.08e-4 | 0.4184 ± 0.0278 / 0.008379 ± 2.09e-4 | 21.66 ± 26.5 |
|  | **2D Buoyancy-Driven Cavity Flow** | no-div | 0 | 5 | 1, 2, 3 | 21182 ± 19.3 | 1.421e-4 ± 6.24e-6 | 3.037 ± 0.00037 / 0.01977 ± 7.05e-5 | — |
|  |  | div | 0.01 |  | Missing |  |  |  |  |
|  |  | div | 0.1 |  | Missing |  |  |  |  |
|  |  | div | 0.5 |  | Missing |  |  |  |  |
|  |  | div | 1 |  | Missing |  |  |  |  |
|  | **2D Taylor–Green Vortices** | no-div | 0 |  | Missing |  |  |  |  |
|  |  | div | 0.01 |  | Missing |  |  |  |  |
|  |  | div | 0.1 |  | Missing |  |  |  |  |
|  |  | div | 0.5 |  | Missing |  |  |  |  |
|  |  | div | 1 |  | Missing |  |  |  |  |
|  | **2D Taylor–Green Vortices: Spacetime** | no-div | 0 |  | Missing |  |  |  |  |
|  |  | div | 0.01 |  | NaN | NaN | NaN | NaN / NaN | NaN |
|  |  | div | 0.1 |  | NaN | NaN | NaN | NaN / NaN | NaN |
|  |  | div | 0.5 |  | NaN | NaN | NaN | NaN / NaN | NaN |
|  |  | div | 1 |  | NaN | NaN | NaN | NaN / NaN | NaN |
|  | **2D Merging Vortices** | no-div | 0 | 5 | 1, 2, 3 | 303.46 ± 1.07 | 6.247e-4 ± 1.66e-4 | 1379 ± 477 / 0.03359 ± 2.24e-4 | 2413 ± 1310 |
|  |  | div | 0.01 |  | Missing |  |  |  |  |
|  |  | div | 0.1 |  | Missing |  |  |  |  |
|  |  | div | 0.5 |  | Missing |  |  |  |  |
|  |  | div | 1 |  | Missing |  |  |  |  |
|  | **3D Species Transport** | no-div | 0 |  | Missing |  |  |  |  |
|  |  | div | 0.01 |  | NaN | NaN | NaN | NaN / NaN | NaN |
|  |  | div | 0.1 |  | NaN | NaN | NaN | NaN / NaN | NaN |
|  |  | div | 0.5 |  | NaN | NaN | NaN | NaN / NaN | NaN |
|  |  | div | 1 |  | NaN | NaN | NaN | NaN / NaN | NaN |
|  | **3D Homogeneous Forced Isotropic Turbulence** | no-div | 0 | 5 | 1, 2, 3 | 44815 ± 101 | 1.824e-4 ± 8.67e-6 | 0.4156 ± 0.00171 / 0.02749 ± 2.70e-6 | 0.0483 ± 0.0158 |
|  |  | div | 0.01 |  | Missing |  |  |  |  |
|  |  | div | 0.1 |  | Missing |  |  |  |  |
|  |  | div | 0.5 |  | Missing |  |  |  |  |
|  |  | div | 1 |  | Missing |  |  |  |  |
| **Transolver** | **2D Backward-Facing Step** | no-div | 0 |  | Missing |  |  |  |  |
|  |  | div | 0.01 |  | Missing |  |  |  |  |
|  |  | div | 0.1 | 4 | 1, 2, 3 | 271.7 ± 0.943 | 0.0008263 ± 4.1e-05 | 1.907 ± 0.011 / 0.001049 ± 5.95e-05 | 31.46 ± 0.844 |
|  |  | div | 0.5 | 4 | 1, 2, 3 | 273 ± 1.63 | 0.00163 ± 1.6e-05 | 1.553 ± 0.0114 / 0.0009266 ± 5.95e-05 | 31.26 ± 0.603 |
|  |  | div | 1 | 4 | 1, 2, 3 | 273 ± 0.816 | 0.004223 ± 0.00124 | 0.9265 ± 0.0903 / 0.001793 ± 0.00135 | 31.6 ± 0.106 |
|  | **2D Flow Past a Cylinder (no vortex shedding)** | no-div | 0 |  | Missing |  |  |  |  |
|  |  | div | 0.01 |  | Missing |  |  |  |  |
|  |  | div | 0.1 | 4 | 1, 2, 3 | 70 ± 0.816 | 0.007668 ± 0.000994 | 0.7174 ± 0.0118 / 0.0004356 ± 7.21e-06 | 55.77 ± 1.75 |
|  |  | div | 0.5 | 4 | 1, 2, 3 | 69.33 ± 0.471 | 0.007033 ± 0.00229 | 0.6206 ± 0.0168 / 0.0003794 ± 4.53e-05 | 55.68 ± 1.7 |
|  |  | div | 1 | 4 | 1, 2, 3 | 69.33 ± 1.25 | 0.007005 ± 0.00251 | 0.5505 ± 0.0406 / 0.000412 ± 3.31e-05 | 54.97 ± 2.84 |
|  | **2D Flow Past a Cylinder (vortex shedding)** | no-div | 0 |  | Missing |  |  |  |  |
|  |  | div | 0.01 |  | Missing |  |  |  |  |
|  |  | div | 0.1 | 4 | 3 NaN |  |  |  |  |
|  |  | div | 0.5 | 4 | 3 NaN |  |  |  |  |
|  |  | div | 1 | 4 | 3 NaN |  |  |  |  |
|  | **2D Lid-Driven Cavity Flow** | no-div | 0 |  | Missing |  |  |  |  |
|  |  | div | 0.01 |  | Missing |  |  |  |  |
|  |  | div | 0.1 | 4 | 3 NaN |  |  |  |  |
|  |  | div | 0.5 | 4 | 3; 2 NaN | 4642 | 0.02799 | 1.233 / 0.0325 | 17.18 |
|  |  | div | 1 | 4 | 3 NaN |  |  |  |  |
|  | **2D Buoyancy-Driven Cavity Flow** | no-div | 0 |  | Missing |  |  |  |  |
|  |  | div | 0.01 |  | Missing |  |  |  |  |
|  |  | div | 0.1 | 4 | 3 no test result |  |  |  |  |
|  |  | div | 0.5 | 4 | 3 no test result |  |  |  |  |
|  |  | div | 1 | 4 | 3 no test result |  |  |  |  |
|  | **2D Taylor–Green Vortices** | no-div | 0 |  | Missing |  |  |  |  |
|  |  | div | 0.01 |  | Missing |  |  |  |  |
|  |  | div | 0.1 | 4 | 1, 3; 1 NaN | 2268 ± 19.5 | 0.9423 ± 0.023 | 14.53 ± 10.1 / 0.008951 ± 0.00893 | ∞ |
|  |  | div | 0.5 | 4 | 1, 2; 1 NaN | 2236 ± 10 | 0.7559 ± 0.234 | 11.2 ± 4.3 / 0.03278 ± 0.0152 | ∞ |
|  |  | div | 1 | 4 | 1; 2 NaN | 2249 | 0.9967 | 5.576 / 9.384e-05 | ∞ |
|  | **2D Taylor–Green Vortices: Spacetime** | no-div | 0 |  | Missing |  |  |  |  |
|  |  | div | 0.01 |  | NaN | NaN | NaN | NaN / NaN | NaN |
|  |  | div | 0.1 | 4 | NaN | NaN | NaN | NaN / NaN | NaN |
|  |  | div | 0.5 | 4 | NaN | NaN | NaN | NaN / NaN | NaN |
|  |  | div | 1 | 4 | NaN | NaN | NaN | NaN / NaN | NaN |
|  | **2D Merging Vortices** | no-div | 0 |  | Missing |  |  |  |  |
|  |  | div | 0.01 |  | Missing |  |  |  |  |
|  |  | div | 0.1 | 4 | 1, 2, 3 | 232.7 ± 1.25 | 0.0213 ± 0.00138 | 1.415 ± 0.321 / 0.03225 ± 0.000539 | 82.07 ± 4.91 |
|  |  | div | 0.5 | 4 | 1, 2, 3 | 232 ± 0.816 | 0.4822 ± 0.35 | 1.628 ± 1.27 / 0.01894 ± 0.00936 | 81.73 ± 7.01 |
|  |  | div | 1 | 4 | 1, 2, 3 | 232 ± 1.63 | 0.3834 ± 0.277 | 2.14 ± 1.18 / 0.03727 ± 0.0115 | 82.99 ± 7.43 |
|  | **3D Species Transport** | no-div | 0 |  | Missing |  |  |  |  |
|  |  | div | 0.01 |  | NaN | NaN | NaN | NaN / NaN | NaN |
|  |  | div | 0.1 | 4 | NaN | NaN | NaN | NaN / NaN | NaN |
|  |  | div | 0.5 | 4 | NaN | NaN | NaN | NaN / NaN | NaN |
|  |  | div | 1 | 4 | NaN | NaN | NaN | NaN / NaN | NaN |
|  | **3D Homogeneous Forced Isotropic Turbulence** | no-div | 0 |  | Missing |  |  |  |  |
|  |  | div | 0.01 |  | Missing |  |  |  |  |
|  |  | div | 0.1 |  | Missing |  |  |  |  |
|  |  | div | 0.5 |  | Missing |  |  |  |  |
|  |  | div | 1 |  | Missing |  |  |  |  |
|  | **2D Taylor–Green Vortices (coefficient input)** | no-div | 0 |  | Missing |  |  |  |  |
|  |  | div | 0.01 |  | Missing |  |  |  |  |
|  |  | div | 0.1 | 4 | 1; 2 NaN | 2212 | 0.9645 | 7.166 / 0.007574 | — |
|  |  | div | 0.5 | 4 | 3 NaN |  |  |  |  |
|  |  | div | 1 | 4 | 3; 2 NaN | 2239 | 0.9351 | 3.246 / 8.595e-05 | — |
|  | **2D Taylor–Green Vortices: Spacetime (coefficient input)** | no-div | 0 |  | Missing |  |  |  |  |
|  |  | div | 0.01 |  | NaN | NaN | NaN | NaN / NaN | NaN |
|  |  | div | 0.1 | 4 | NaN | NaN | NaN | NaN / NaN | NaN |
|  |  | div | 0.5 | 4 | NaN | NaN | NaN | NaN / NaN | NaN |
|  |  | div | 1 | 4 | NaN | NaN | NaN | NaN / NaN | NaN |

Sources: Geo-FNO values from the existing table and [earlier results](wandb_results_averaged.md); Transolver summaries and metadata from `Transolver/PDE-Solving-StandardBenchmark/wandb/run-20260903_*` through `run-20260905_*` (99 runs, code commit `280838d`). Taylor–Green coefficient-input variants are kept separate from the ten datasets in the checklist.
