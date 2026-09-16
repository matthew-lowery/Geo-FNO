# Final results

Updated from `../wandb_final/wandb`: runs started 11–16 September 2026. The cache contains the original 330 runs at commit `eecb239` and 43 additional runs at `52a150b`. For each model/dataset/training-size/divergence-weight/seed configuration, the latest attempt replaces the earlier attempt; repeated seeds are not counted twice. Older August runs and values from `new_res.md` are excluded.

Rows follow the current `train_div.sh`: no-div baseline and divergence-penalty weight $\lambda \in \{0.001, 0.01, 0.1, 1\}$. Each configuration targets seeds 1, 2, 3 and 500 epochs. The columns `ntrain` and `npoints` give the requested training-sample count and spatial point budget; spacetime layouts can expand this budget. `Div. order` is the logged RBF-FD polynomial degree, not an empirically measured convergence order. The new Geo-FNO order-2 runs are retained here because they belong to this batch; historical order-2 runs are excluded.

Metric entries are mean ± population standard deviation across seeds with finite final `test_loss`; NaN seeds are excluded and explicitly listed. A single finite seed has no standard deviation. For scalar seed values $z_1,\ldots,z_K$, where $K$ is the number of contributing seeds, the reported mean $\bar z$ and standard deviation $\sigma$ are given by Eq. (1):

$$
\bar z=\frac{1}{K}\sum_{j=1}^{K}z_j,\qquad
\sigma=\sqrt{\frac{1}{K}\sum_{j=1}^{K}(z_j-\bar z)^2}. \tag{1}
$$

`Time (s)` averages the logged `total_train_time` over all runs with that field, including NaN runs. It is not W&B runtime: Geo-FNO's timer includes final test evaluation (and its coefficient entry places it after OOD handling), while Transolver stops its timer before final evaluation.

Test losses, Transolver OOD losses, and Geo-FNO coefficient-entry OOD losses compare **field magnitudes**. Geo-FNO's other 2D/3D OOD losses compare **vector components**, so those OOD columns are not directly comparable across models. For evaluation sample $i$, let $\widehat u_{i,q}$ and $u_{i,q}$ be the predicted and target component vectors at output index $q$. Define magnitude vectors $\widehat a_i=(\|\widehat u_{i,q}\|_2)_q$ and $a_i=(\|u_{i,q}\|_2)_q$. With $M$ evaluation samples, the magnitude-based loss is Eq. (2):

$$
\frac{1}{M}\sum_{i=1}^{M}\frac{\|\widehat a_i-a_i\|_2}{\|a_i\|_2}. \tag{2}
$$

For Geo-FNO's component-based OOD metric, let $\widehat U_i$ and $U_i$ be the predicted and target vectors formed by concatenating every output point's components for sample $i$. Its OOD loss is Eq. (3), using the same sample count $M$:

$$
\frac{1}{M}\sum_{i=1}^{M}\frac{\|\widehat U_i-U_i\|_2}{\|U_i\|_2}. \tag{3}
$$

The `52a150b` reruns use float64 evaluation reductions and bounded Transolver slice temperature; retained original runs predate these fixes. Seed averages use Eq. (1) within each listed row.

Interior divergence is the maximum / median absolute discrete divergence over test samples and interior points (including time slices for spacetime datasets), followed by the seed aggregation above. Different orders produce different discrete diagnostics.

`NaN` means a logged numerical failure; `No summary` means an attempted run has metadata but no cached summary, not necessarily a NaN failure. Blank metric cells indicate no result; `—` indicates an absent metric in a populated result. `Disabled` means OOD evaluation was disabled for div training; `No OOD data` means the run logged `ood_available=false`. `Partial` marks a rerun with training progress but no final test summary; the epoch count is observed progress, not confirmation that the remote job is still running. `Missing` means no cached attempt for that seed. Unqualified seeds have finite test loss.

## Benchmark results

| Framework | Dataset | Run | $\lambda$ | Div. order | ntrain | npoints | Seeds / status | Time (s) | Test loss | Interior test div (max / median) | OOD loss |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| **Geo-FNO** | **2D Backward-Facing Step** | no-div | 0 | 3 | 500 | 1000 | 1, 2, 3 | 194.4 ± 1.325 | 9.949e-5 ± 1.728e-5 | 1.981 ± 0.002382 / 4.769e-4 ± 1.255e-5 | — |
|  |  | div | 0.001 | 3 |  |  | 1, 2, 3 | 216.2 ± 2.108 | 9.259e-5 ± 9.780e-6 | 1.983 ± 0.002885 / 4.536e-4 ± 7.371e-6 | Disabled |
|  |  | div | 0.01 | 3 |  |  | 1, 2, 3 | 219.4 ± 1.173 | 1.035e-4 ± 1.914e-5 | 1.98 ± 0.00212 / 4.848e-4 ± 8.368e-6 | Disabled |
|  |  | div | 0.1 | 3 |  |  | 1, 2, 3 | 217.8 ± 1.275 | 1.043e-4 ± 9.781e-6 | 1.972 ± 0.00179 / 4.732e-4 ± 8.895e-6 | Disabled |
|  |  | div | 1 | 3 |  |  | 1, 2, 3 | 217.3 ± 1.038 | 0.002127 ± 2.004e-5 | 1.286 ± 0.05033 / 6.612e-4 ± 5.633e-5 | Disabled |
|  | **2D Flow Past a Cylinder (no vortex shedding)** | no-div | 0 | 3 | 100 | 1000 | 1, 2, 3 | 175.5 ± 0.1199 | 1.618e-4 ± 4.766e-5 | 0.7716 ± 1.441e-4 / 1.117e-4 ± 2.307e-5 | — |
|  |  | div | 0.001 | 3 |  |  | 1, 2, 3 | 180.9 ± 1.096 | 1.850e-4 ± 4.173e-5 | 0.7716 ± 3.154e-4 / 1.147e-4 ± 2.920e-5 | Disabled |
|  |  | div | 0.01 | 3 |  |  | 1, 2, 3 | 179.9 ± 0.05814 | 1.555e-4 ± 3.234e-5 | 0.7716 ± 3.908e-4 / 1.056e-4 ± 1.103e-5 | Disabled |
|  |  | div | 0.1 | 3 |  |  | 1, 2, 3 | 180.4 ± 0.5481 | 1.803e-4 ± 2.747e-5 | 0.7704 ± 7.570e-4 / 1.091e-4 ± 1.488e-6 | Disabled |
|  |  | div | 1 | 3 |  |  | 1, 2, 3 | 180 ± 0.3947 | 1.838e-4 ± 2.357e-5 | 0.7601 ± 0.00248 / 1.094e-4 ± 1.013e-5 | Disabled |
|  | **2D Flow Past a Cylinder (vortex shedding)** | no-div | 0 | 3 | 10000 | 1000 | 1, 2, 3 | 11410 ± 13.81 | 4.783e-5 ± 1.187e-5 | 4.943 ± 5.985e-4 / 0.001447 ± 1.423e-5 | — |
|  |  | div | 0.001 | 3 |  |  | 1, 2, 3 | 11870 ± 8.008 | 3.737e-5 ± 4.751e-6 | 4.942 ± 3.975e-4 / 0.001432 ± 7.726e-6 | Disabled |
|  |  | div | 0.01 | 3 |  |  | 1, 2, 3 | 11870 ± 18.3 | 4.784e-5 ± 1.093e-5 | 4.93 ± 0.006215 / 0.001433 ± 1.145e-5 | Disabled |
|  |  | div | 0.1 | 3 |  |  | 1, 2, 3 | 11880 ± 11.91 | 2.032e-4 ± 2.179e-5 | 4.278 ± 0.01617 / 0.001455 ± 2.563e-5 | Disabled |
|  |  | div | 1 | 3 |  |  | 1, 2, 3 | 11880 ± 25.7 | 0.012 ± 0.003156 | 1.138 ± 0.2216 / 0.001491 ± 2.000e-4 | Disabled |
|  | **2D Lid-Driven Cavity Flow** | no-div | 0 | 3 | 10000 | 1000 | 1, 2, 3 | 6260 ± 9.108 | 1.509e-4 ± 2.041e-5 | 18.14 ± 9.691e-4 / 0.5722 ± 3.843e-4 | — |
|  |  | div | 0.001 | 3 |  |  | 1, 2, 3 | 6724 ± 19.75 | 1.795e-4 ± 2.001e-5 | 18.09 ± 0.01043 / 0.57 ± 1.670e-4 | Disabled |
|  |  | div | 0.01 | 3 |  |  | 1, 2, 3 | 6734 ± 12.01 | 0.002027 ± 1.645e-5 | 9.507 ± 0.003913 / 0.3617 ± 5.243e-4 | Disabled |
|  |  | div | 0.1 | 3 |  |  | 1, 2, 3 | 6756 ± 27.36 | 0.01109 ± 3.133e-4 | 2.314 ± 0.02049 / 0.04074 ± 6.557e-5 | Disabled |
|  |  | div | 1 | 3 |  |  | 1, 2, 3 | 6742 ± 1.876 | 0.01565 ± 1.076e-4 | 0.4184 ± 0.02775 / 0.008379 ± 2.090e-4 | Disabled |
|  | **2D Buoyancy-Driven Cavity Flow** | no-div | 0 | 3 | 10000 | 5000 | 1, 2, 3 | 21160 ± 7.866 | 1.421e-4 ± 6.241e-6 | 3.546 ± 0.002381 / 0.0198 ± 7.664e-5 | No OOD data |
|  |  | div | 0.001 | 3 |  |  | 1, 2, 3 | 21600 ± 19.28 | 1.571e-4 ± 1.324e-6 | 3.513 ± 0.007005 / 0.01985 ± 3.330e-5 | Disabled |
|  |  | div | 0.01 | 3 |  |  | 1, 2, 3 | 21630 ± 9.716 | 1.644e-4 ± 7.940e-6 | 3.454 ± 0.007804 / 0.01949 ± 7.262e-5 | Disabled |
|  |  | div | 0.1 | 3 |  |  | 1, 2, 3 | 21610 ± 33.21 | 0.001926 ± 2.630e-6 | 2.095 ± 0.003948 / 0.009561 ± 2.793e-4 | Disabled |
|  |  | div | 1 | 3 |  |  | 1, 2, 3 | 21610 ± 22.73 | 0.005727 ± 3.916e-5 | 0.5015 ± 0.009256 / 0.004568 ± 6.371e-4 | Disabled |
|  | **2D Taylor–Green Vortices** | no-div | 0 | 2 | 5000 | 500 | 1, 2, 3 | 2420 ± 5.561 | 4.181e-4 ± 2.939e-5 | 2.563 ± 0.9359 / 0.0937 ± 0.004725 | — |
|  |  | div | 0.001 | 2 |  |  | 1, 2, 3 | 2646 ± 4.866 | 5.070e-4 ± 1.607e-4 | 5.037 ± 5.223 / 0.06777 ± 0.004779 | Disabled |
|  |  | div | 0.01 | 2 |  |  | 1, 2, 3 | 2651 ± 3.771 | 5.149e-4 ± 1.738e-5 | 0.7696 ± 0.1014 / 0.04372 ± 0.001714 | Disabled |
|  |  | div | 0.1 | 2 |  |  | 1, 2, 3 | 2666 ± 13.9 | 0.03258 ± 0.027 | 2.818 ± 1.785 / 0.08301 ± 0.04781 | Disabled |
|  |  | div | 1 | 2 |  |  | 1, 2, 3 | 2660 ± 6.637 | 0.0108 ± 0.01376 | 0.4849 ± 0.3374 / 0.01343 ± 0.008997 | Disabled |
|  | **2D Taylor–Green Vortices: Coefficients** | no-div | 0 | 2 | 5000 | 500 | 1, 2, 3 | — | 6.248e-4 ± 6.887e-5 | — / — | No OOD data |
|  |  | div | 0.001 | 2 |  |  | 1, 2, 3 | 2357 ± 13.5 | 5.151e-4 ± 1.200e-4 | 2.781 ± 1.277 / 0.06078 ± 0.004351 | Disabled |
|  |  | div | 0.01 | 2 |  |  | 1, 2, 3 | 2361 ± 4.643 | 5.582e-4 ± 5.649e-5 | 1.206 ± 0.2318 / 0.03983 ± 0.00117 | Disabled |
|  |  | div | 0.1 | 2 |  |  | 1, 2, 3 | 2353 ± 3.408 | 6.989e-4 ± 1.637e-5 | 1.226 ± 0.8181 / 0.02358 ± 0.001319 | Disabled |
|  |  | div | 1 | 2 |  |  | 1, 2, 3 | 2355 ± 7.376 | 0.04657 ± 0.03172 | 0.8652 ± 0.3549 / 0.03606 ± 0.01804 | Disabled |
|  | **2D Taylor–Green Vortices: Spacetime** | no-div | 0 | 2 | 5000 | 500 | 1, 2, 3 | 5982 ± 1.359 | 0.001004 ± 2.361e-5 | 8.963 ± 1.252 / 0.1744 ± 0.01401 | — |
|  |  | div | 0.001 | 2 |  |  | 1, 2, 3 | 6204 ± 9.726 | 8.393e-4 ± 2.555e-4 | 3.848 ± 2.389 / 0.09993 ± 0.01043 | Disabled |
|  |  | div | 0.01 | 2 |  |  | 1, 2, 3 | 6204 ± 12.52 | 7.765e-4 ± 1.311e-4 | 2.961 ± 2.601 / 0.06441 ± 0.005443 | Disabled |
|  |  | div | 0.1 | 2 |  |  | 1, 2, 3 | 6207 ± 5.152 | 8.733e-4 ± 2.379e-4 | 0.6846 ± 0.3954 / 0.0268 ± 0.002858 | Disabled |
|  |  | div | 1 | 2 |  |  | 1, 2, 3 | 6205 ± 7.749 | 0.045 ± 0.03407 | 1.273 ± 0.6747 / 0.04854 ± 0.02696 | Disabled |
|  | **2D Taylor–Green Vortices: Spacetime Coefficients** | no-div | 0 | 2 | 5000 | 500 | 1, 2, 3 | 5734 ± 16.08 | 0.001164 ± 2.797e-4 | 6.612 ± 0.2516 / 0.1681 ± 0.01594 | No OOD data |
|  |  | div | 0.001 | 2 |  |  | 1, 2, 3 | 5971 ± 15.92 | 0.002301 ± 0.001484 | 16.49 ± 15.6 / 0.1366 ± 0.005178 | Disabled |
|  |  | div | 0.01 | 2 |  |  | 1, 2, 3 | 5948 ± 6.008 | 0.004013 ± 0.004528 | 4.21 ± 3.337 / 0.1232 ± 0.07742 | Disabled |
|  |  | div | 0.1 | 2 |  |  | 1, 2, 3 | 5963 ± 16.56 | 0.01523 ± 0.00892 | 4.005 ± 0.5283 / 0.07947 ± 0.02538 | Disabled |
|  |  | div | 1 | 2 |  |  | 1, 2, 3 | 5956 ± 9.733 | 0.0201 ± 0.01306 | 0.9095 ± 0.4716 / 0.02612 ± 0.01119 | Disabled |
|  | **2D Merging Vortices** | no-div | 0 | 2 | 500 | 500 | 1, 2, 3 | 303.6 ± 0.8918 | 0.001351 ± 0.001015 | 0.999 ± 0.003457 / 0.03244 ± 1.321e-4 | — |
|  |  | div | 0.001 | 2 |  |  | 1, 2, 3 | 322.7 ± 1.81 | 0.001124 ± 5.654e-5 | 0.9934 ± 0.01197 / 0.03228 ± 2.174e-4 | Disabled |
|  |  | div | 0.01 | 2 |  |  | 1, 2, 3 | 322.8 ± 0.3149 | 0.001202 ± 4.255e-4 | 0.9949 ± 0.004626 / 0.03249 ± 1.974e-4 | Disabled |
|  |  | div | 0.1 | 2 |  |  | 1, 2, 3 | 321.7 ± 0.8905 | 0.00116 ± 3.192e-4 | 0.985 ± 0.003261 / 0.03209 ± 2.993e-5 | Disabled |
|  |  | div | 1 | 2 |  |  | 1, 2, 3 | 322 ± 0.8656 | 0.003325 ± 3.488e-4 | 0.6326 ± 0.0241 / 0.02524 ± 4.076e-4 | Disabled |
|  | **3D Species Transport** | no-div | 0 | 3 | 10000 | 7000 | No summary: 1, 2, 3 |  |  |  |  |
|  |  | div | 0.001 | 3 |  |  | No summary: 1, 2, 3 |  |  |  |  |
|  |  | div | 0.01 | 3 |  |  | No summary: 1, 2, 3 |  |  |  |  |
|  |  | div | 0.1 | 3 |  |  | No summary: 1, 2, 3 |  |  |  |  |
|  |  | div | 1 | 3 |  |  | No summary: 1, 2, 3 |  |  |  |  |
|  | **3D Homogeneous Forced Isotropic Turbulence** | no-div | 0 | 3 | 10000 | 7000 | Missing: 1, 2, 3 |  |  |  |  |
|  |  | div | 0.001 | 3 |  |  | Partial: 1 (182/500 epochs logged); Missing: 2, 3 |  |  |  |  |
|  |  | div | 0.01 | 3 |  |  | Partial: 1 (137/500 epochs logged); Missing: 2, 3 |  |  |  |  |
|  |  | div | 0.1 | 3 |  |  | Partial: 1 (128/500 epochs logged); Missing: 2, 3 |  |  |  |  |
|  |  | div | 1 | 3 |  |  | Partial: 1 (126/500 epochs logged); Missing: 2, 3 |  |  |  |  |
| **Transolver** | **2D Backward-Facing Step** | no-div | 0 | 4 | 500 | 1000 | 1, 2, 3 | 247.6 ± 0.7593 | 7.330e-4 ± 2.395e-5 | 1.987 ± 0.001957 / 9.624e-4 ± 1.416e-4 | 0.314 ± 0.007866 |
|  |  | div | 0.001 | 4 |  |  | 1, 2, 3 | 268.7 ± 0.7835 | 7.247e-4 ± 3.489e-5 | 1.959 ± 0.01665 / 8.132e-4 ± 8.025e-5 | Disabled |
|  |  | div | 0.01 | 4 |  |  | 1, 2, 3 | 267.4 ± 0.7281 | 7.470e-4 ± 1.118e-5 | 1.957 ± 0.008743 / 0.001016 ± 3.989e-5 | Disabled |
|  |  | div | 0.1 | 4 |  |  | 1, 2, 3 | 266.7 ± 0.5923 | 8.028e-4 ± 8.946e-6 | 1.898 ± 0.01805 / 0.001142 ± 1.075e-4 | Disabled |
|  |  | div | 1 | 4 |  |  | 1, 2, 3 | 266.9 ± 0.223 | 0.003593 ± 3.770e-4 | 0.8741 ± 0.02595 / 0.001149 ± 4.726e-4 | Disabled |
|  | **2D Flow Past a Cylinder (no vortex shedding)** | no-div | 0 | 4 | 100 | 1000 | 1, 2, 3 | 56.2 ± 0.3071 | 0.00736 ± 8.214e-4 | 0.7528 ± 0.02085 / 4.141e-4 ± 1.945e-5 | 0.567 ± 0.007513 |
|  |  | div | 0.001 | 4 |  |  | 1, 2, 3 | 60.25 ± 0.3739 | 0.00606 ± 0.001932 | 0.7609 ± 0.02347 / 4.342e-4 ± 3.342e-5 | Disabled |
|  |  | div | 0.01 | 4 |  |  | 1, 2, 3 | 60.03 ± 0.1256 | 0.006854 ± 7.333e-4 | 0.7482 ± 0.01083 / 3.969e-4 ± 2.086e-5 | Disabled |
|  |  | div | 0.1 | 4 |  |  | 1, 2, 3 | 60.54 ± 0.5955 | 0.005738 ± 0.002288 | 0.709 ± 0.01802 / 3.882e-4 ± 1.069e-5 | Disabled |
|  |  | div | 1 | 4 |  |  | 1, 2, 3 | 60.15 ± 0.03407 | 0.00839 ± 8.850e-4 | 0.4693 ± 0.01988 / 4.347e-4 ± 1.550e-5 | Disabled |
|  | **2D Flow Past a Cylinder (vortex shedding)** | no-div | 0 | 4 | 10000 | 1000 | 1; NaN: 2, 3 | 4320 ± 1.227 | 1.422e-4 | 4.895 / 0.001432 | 0.2419 |
|  |  | div | 0.001 | 4 |  |  | 1, 2, 3 | 4685 ± 6.274 | 1.523e-4 ± 1.274e-5 | 4.889 ± 0.005282 / 0.001434 ± 9.184e-6 | Disabled |
|  |  | div | 0.01 | 4 |  |  | NaN: 1, 2, 3 | 4691 ± 15.19 | NaN | NaN / NaN | Disabled |
|  |  | div | 0.1 | 4 |  |  | NaN: 1, 2, 3 | 4702 ± 13.77 | NaN | NaN / NaN | Disabled |
|  |  | div | 1 | 4 |  |  | NaN: 1, 2, 3 | 4701 ± 18.2 | NaN | NaN / NaN | Disabled |
|  | **2D Lid-Driven Cavity Flow** | no-div | 0 | 4 | 10000 | 1000 | 2; NaN: 1, 3 | 4198 ± 27.24 | 2.010e-4 | 21.11 / 0.5714 | 0.5558 |
|  |  | div | 0.001 | 4 |  |  | NaN: 1, 2, 3 | 4570 ± 47.92 | NaN | NaN / NaN | Disabled |
|  |  | div | 0.01 | 4 |  |  | 1, 2; NaN: 3 | 4571 ± 9.348 | 0.002822 ± 2.432e-5 | 9.193 ± 0.02296 / 0.36 ± 0.002023 | Disabled |
|  |  | div | 0.1 | 4 |  |  | 2; NaN: 1, 3 | 4575 ± 24.3 | 0.01174 | 1.734 / 0.05139 | Disabled |
|  |  | div | 1 | 4 |  |  | NaN: 1, 2, 3 | 4586 ± 21.66 | NaN | NaN / NaN | Disabled |
|  | **2D Buoyancy-Driven Cavity Flow** | no-div | 0 | 4 | 10000 | 5000 | No summary: 1, 2, 3 |  |  |  |  |
|  |  | div | 0.001 | 4 |  |  | 1, 2, 3 | 17020 ± 6.049 | 1.932e-4 ± 1.208e-5 | 3.03 ± 0.005111 / 0.01946 ± 3.786e-5 | Disabled |
|  |  | div | 0.01 | 4 |  |  | 1, 2, 3 | 17010 ± 8.593 | 2.309e-4 ± 4.054e-5 | 3.009 ± 0.004214 / 0.01955 ± 3.800e-5 | Disabled |
|  |  | div | 0.1 | 4 |  |  | 1, 2, 3 | 17010 ± 8.423 | 0.001726 ± 1.203e-5 | 2.323 ± 0.004224 / 0.01874 ± 8.637e-5 | Disabled |
|  |  | div | 1 | 4 |  |  | 1, 2, 3 | 17030 ± 22.3 | 0.006201 ± 1.037e-4 | 0.5415 ± 0.05091 / 0.01292 ± 4.891e-4 | Disabled |
|  | **2D Taylor–Green Vortices** | no-div | 0 | 4 | 5000 | 500 | 1; NaN: 2, 3 | 2042 ± 7.277 | 1.336e-4 | 0.8245 / 0.01645 | ∞ |
|  |  | div | 0.001 | 4 |  |  | 3; NaN: 1, 2 | 2238 ± 9.686 | 1.352e-4 | 0.5808 / 0.01616 | Disabled |
|  |  | div | 0.01 | 4 |  |  | 1; NaN: 2, 3 | 2240 ± 16.52 | 1.072e-4 | 0.1992 / 0.01083 | Disabled |
|  |  | div | 0.1 | 4 |  |  | NaN: 1, 2, 3 | 2215 ± 4.33 | NaN | NaN / NaN | Disabled |
|  |  | div | 1 | 4 |  |  | NaN: 1, 2, 3 | 2244 ± 16.27 | NaN | NaN / NaN | Disabled |
|  | **2D Taylor–Green Vortices: Coefficients** | no-div | 0 | 4 | 5000 | 500 | 1, 2, 3 | 2054 ± 5.915 | 3.632e-4 ± 3.010e-5 | 0.7201 ± 0.03859 / 0.05381 ± 0.001924 | No OOD data |
|  |  | div | 0.001 | 4 |  |  | 2, 3; NaN: 1 | 2236 ± 14.83 | 3.111e-4 ± 2.901e-5 | 0.4555 ± 0.008472 / 0.03332 ± 4.768e-5 | Disabled |
|  |  | div | 0.01 | 4 |  |  | 1, 2, 3 | 2213 ± 13.73 | 3.428e-4 ± 3.690e-5 | 0.2408 ± 0.0368 / 0.01845 ± 5.025e-4 | Disabled |
|  |  | div | 0.1 | 4 |  |  | 1, 2, 3 | 2209 ± 8.647 | 3.034e-4 ± 2.914e-5 | 0.0853 ± 0.01039 / 0.007506 ± 1.666e-4 | Disabled |
|  |  | div | 1 | 4 |  |  | NaN: 1, 2, 3 | 2238 ± 15.74 | NaN | NaN / NaN | Disabled |
|  | **2D Taylor–Green Vortices: Spacetime** | no-div | 0 | 4 | 5000 | 500 | 1, 2, 3 | 4685 ± 13.05 | 0.003563 ± 0.004537 | 4.651 ± 4.601 / 0.2093 ± 0.2321 | ∞ |
|  |  | div | 0.001 | 4 |  |  | 3; NaN: 1, 2 | 4884 ± 18.46 | 2.647e-4 | 0.8653 / 0.02485 | Disabled |
|  |  | div | 0.01 | 4 |  |  | 1, 2, 3 | 4860 ± 7.404 | 2.559e-4 ± 3.448e-5 | 0.3821 ± 0.06716 / 0.01717 ± 0.00165 | Disabled |
|  |  | div | 0.1 | 4 |  |  | 3; NaN: 1, 2 | 4881 ± 10.61 | 1.757e-4 | 0.1275 / 0.006922 | Disabled |
|  |  | div | 1 | 4 |  |  | NaN: 1, 2, 3 | 4884 ± 6.245 | NaN | NaN / NaN | Disabled |
|  | **2D Taylor–Green Vortices: Spacetime Coefficients** | no-div | 0 | 4 | 5000 | 500 | 1; NaN: 2, 3 | 4677 ± 10.49 | 2.609e-4 | 1.061 / 0.03734 | No OOD data |
|  |  | div | 0.001 | 4 |  |  | 2, 3; NaN: 1 | 4871 ± 14.66 | 2.464e-4 ± 2.423e-5 | 0.526 ± 0.04795 / 0.02638 ± 0.002703 | Disabled |
|  |  | div | 0.01 | 4 |  |  | 1, 3; NaN: 2 | 4877 ± 7.472 | 2.228e-4 ± 6.353e-6 | 0.2494 ± 0.01599 / 0.0147 ± 4.525e-4 | Disabled |
|  |  | div | 0.1 | 4 |  |  | 2, 3; NaN: 1 | 4875 ± 7.728 | 1.878e-4 ± 5.173e-6 | 0.12 ± 0.006192 / 0.006433 ± 5.245e-6 | Disabled |
|  |  | div | 1 | 4 |  |  | NaN: 1, 2, 3 | 4888 ± 5.612 | NaN | NaN / NaN | Disabled |
|  | **2D Merging Vortices** | no-div | 0 | 4 | 500 | 500 | 1, 2, 3 | 209.3 ± 0.4533 | 0.003747 ± 3.802e-4 | 0.7377 ± 0.005383 / 0.03128 ± 0.001046 | 0.8056 ± 0.003242 |
|  |  | div | 0.001 | 4 |  |  | 1, 2, 3 | 229.6 ± 1.182 | 0.003397 ± 2.319e-4 | 0.7236 ± 0.01586 / 0.03081 ± 7.019e-4 | Disabled |
|  |  | div | 0.01 | 4 |  |  | 1, 2, 3 | 228 ± 0.9234 | 0.003497 ± 2.574e-4 | 0.7141 ± 0.005287 / 0.03122 ± 0.001046 | Disabled |
|  |  | div | 0.1 | 4 |  |  | 1, 2, 3 | 227.9 ± 1.358 | 0.003747 ± 2.271e-4 | 0.6439 ± 0.008907 / 0.03012 ± 9.602e-4 | Disabled |
|  |  | div | 1 | 4 |  |  | 1, 2, 3 | 227.1 ± 0.3592 | 0.005517 ± 3.048e-4 | 0.4035 ± 0.005305 / 0.02757 ± 4.458e-4 | Disabled |
|  | **3D Species Transport** | no-div | 0 | 4 | 10000 | 7000 | 1, 2, 3 | 26110 ± 39.13 | 0.002737 ± 1.587e-4 | 1.044e+5 ± 254.9 / 981.9 ± 1.891 | 1.352 ± 0.1792 |
|  |  | div | 0.001 | 4 |  |  | No summary: 1, 2, 3 |  |  |  |  |
|  |  | div | 0.01 | 4 |  |  | No summary: 1, 2, 3 |  |  |  |  |
|  |  | div | 0.1 | 4 |  |  | No summary: 1, 2, 3 |  |  |  |  |
|  |  | div | 1 | 4 |  |  | No summary: 1, 2, 3 |  |  |  |  |
|  | **3D Homogeneous Forced Isotropic Turbulence** | no-div | 0 | 4 | 10000 | 7000 | Missing: 1, 2, 3 |  |  |  |  |
|  |  | div | 0.001 | 4 |  |  | 1; Missing: 2, 3 | 23410 | 0.00116 | 2.184 / 0.03647 | Disabled |
|  |  | div | 0.01 | 4 |  |  | 1; Missing: 2, 3 | 23280 | 0.001125 | 2.157 / 0.03633 | Disabled |
|  |  | div | 0.1 | 4 |  |  | 1; Missing: 2, 3 | 23410 | 0.001182 | 1.862 / 0.03534 | Disabled |
|  |  | div | 1 | 4 |  |  | 1; Missing: 2, 3 | 23300 | 0.001993 | 0.5735 / 0.02617 | Disabled |

## Forced-turbulence training-size sweep

All rows are no-div, with 7,000 requested points and seeds 1, 2, 3. Both models have all three seeds for training sizes 100, 500, and 1,000. At 5,000, Transolver has seed 1 completed and Geo-FNO has seed 1 partial; the remaining seeds and all 7,000-training-sample runs are absent. The 10,000-training-sample configurations are listed in the main table.

| Framework | ntrain | npoints | Div. order | Seeds / status | Time (s) | Test loss | Interior test div (max / median) | OOD loss |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| **Geo-FNO** | 100 | 7000 | 3 | 1, 2, 3 | 1233 ± 3.357 | 0.003366 ± 4.830e-4 | 2.122 ± 0.1829 / 0.04704 ± 0.00103 | 0.07986 ± 0.009248 |
|  | 500 | 7000 | 3 | 1, 2, 3 | 5910 ± 53.29 | 6.709e-4 ± 5.359e-5 | 1.822 ± 0.007078 / 0.04356 ± 5.227e-5 | 0.03947 ± 0.004512 |
|  | 1000 | 7000 | 3 | 1, 2, 3 | 11920 ± 44.48 | 4.357e-4 ± 3.528e-5 | 1.809 ± 0.003655 / 0.04343 ± 1.162e-5 | 0.02816 ± 0.002435 |
|  | 5000 | 7000 | 3 | Partial: 1 (322/500 epochs logged); Missing: 2, 3 |  |  |  |  |
|  | 7000 | 7000 | 3 | Missing: 1, 2, 3 |  |  |  |  |
| **Transolver** | 100 | 7000 | 4 | 1, 2, 3 | 228.9 ± 0.7924 | 0.02715 ± 2.720e-4 | 3.111 ± 0.2398 / 0.07385 ± 0.001233 | 0.06246 ± 7.209e-5 |
|  | 500 | 7000 | 4 | 1, 2, 3 | 1132 ± 0.7685 | 0.01112 ± 2.876e-4 | 2.752 ± 0.1331 / 0.04643 ± 6.002e-4 | 0.04097 ± 8.927e-4 |
|  | 1000 | 7000 | 4 | 1, 2, 3 | 2259 ± 0.7044 | 0.005438 ± 1.573e-4 | 2.528 ± 0.1365 / 0.03984 ± 9.238e-5 | 0.02715 ± 6.270e-4 |
|  | 5000 | 7000 | 4 | 1; Missing: 2, 3 | 11320 | 0.001705 | 2.185 / 0.03686 | 0.01378 |
|  | 7000 | 7000 | 4 | Missing: 1, 2, 3 |  |  |  |  |

## Completion notes

- Of 43 additional runs, 38 have finite final test losses and five have partial training logs without final summaries. These include 12 completed Transolver buoyancy div runs, three completed Transolver species baselines, 19 completed size-sweep runs, and four completed Transolver forced-turbulence div runs.
- The table selects 358 distinct seed/configuration attempts from 373 cached runs: 270 finite final test losses, 53 NaN test losses, 30 old attempts without summaries, and five partial reruns. Fifteen earlier failed attempts are superseded, not averaged into the new results.
- All four Transolver buoyancy div weights now have seeds 1, 2, 3. Its no-div/OOD case was not rerun because the OOD file is absent. Geo-FNO's existing buoyancy baseline is retained with `No OOD data`.
- Transolver species no-div now has all three seeds and OOD results. No new Geo-FNO species result is present; species div training was outside the rerun scope.
- Forced-turbulence div training has only seed 1 cached for each weight: completed for Transolver, partial for Geo-FNO. The 10,000-training-sample no-div runs for both models are absent.
- The five partial Geo-FNO runs have logged 182/500, 137/500, 128/500, and 126/500 epochs for the four forced-turbulence div weights, respectively, and 322/500 epochs for its 5,000-sample no-div sweep. No interim training loss is substituted for final test or OOD loss.
- All 22 new completed no-div runs have OOD losses in their summaries: 19 forced-turbulence sweep runs and three Transolver species baselines. The earlier Geo-FNO baseline OOD losses remain unavailable; they were not restored by these reruns.
- Geo-FNO's original Taylor–Green coefficient baselines still lack summary time and divergence fields because of the old logging-step issue. The buoyancy and both Taylor–Green coefficient OOD datasets were missing in the original baseline runs.
- Missing files are treated as out-of-scope for the rerun launcher; old attempted rows remain visible for provenance. An absent run in this cache alone does not establish whether it was skipped, queued, running, or failed remotely.

## Run provenance

Links point to selected cached run directories in `wandb_final/wandb`. Read `files/config.yaml` and `files/wandb-summary.json`; where these are absent, use `files/wandb-metadata.json` arguments and `files/output.log`. Missing seed links mean no cached attempt. Earlier superseded attempts remain in the cache but are not listed or averaged. The training-size column distinguishes the forced-turbulence sweep from the main experiment.

| Framework | Dataset ID | ntrain | $\lambda$ | Seed 1 | Seed 2 | Seed 3 |
|---|---|---:|---:|---|---|---|
| Geo-FNO | `backward_facing_step` | 500 | 0 | [ycxutvm3](../wandb_final/wandb/run-20260911_233838-ycxutvm3/) | [635kjgnn](../wandb_final/wandb/run-20260912_000649-635kjgnn/) | [8b91c797](../wandb_final/wandb/run-20260912_002007-8b91c797/) |
|  |  | 500 | 0.001 | [h7oj0a4x](../wandb_final/wandb/run-20260911_145108-h7oj0a4x/) | [ir1kjgo3](../wandb_final/wandb/run-20260911_180045-ir1kjgo3/) | [33310qmo](../wandb_final/wandb/run-20260911_201618-33310qmo/) |
|  |  | 500 | 0.01 | [74jn4i2m](../wandb_final/wandb/run-20260911_150222-74jn4i2m/) | [fhrzluqe](../wandb_final/wandb/run-20260911_201623-fhrzluqe/) | [yj07iw9i](../wandb_final/wandb/run-20260911_202332-yj07iw9i/) |
|  |  | 500 | 0.1 | [f8vj489k](../wandb_final/wandb/run-20260911_152007-f8vj489k/) | [983yerdp](../wandb_final/wandb/run-20260911_201623-983yerdp/) | [5cxxovm1](../wandb_final/wandb/run-20260911_220947-5cxxovm1/) |
|  |  | 500 | 1 | [5ffp5e5r](../wandb_final/wandb/run-20260911_170403-5ffp5e5r/) | [iu8aiuei](../wandb_final/wandb/run-20260911_201621-iu8aiuei/) | [peeaz7va](../wandb_final/wandb/run-20260911_232802-peeaz7va/) |
|  | `flow_cylinder_laminar` | 100 | 0 | [2m13e23v](../wandb_final/wandb/run-20260911_233400-2m13e23v/) | [vdw7cuzo](../wandb_final/wandb/run-20260912_000649-vdw7cuzo/) | [ttfvriio](../wandb_final/wandb/run-20260912_001705-ttfvriio/) |
|  |  | 100 | 0.001 | [cj9415e3](../wandb_final/wandb/run-20260911_144936-cj9415e3/) | [2k4n5qkb](../wandb_final/wandb/run-20260911_180046-2k4n5qkb/) | [opnsfg7s](../wandb_final/wandb/run-20260911_201623-opnsfg7s/) |
|  |  | 100 | 0.01 | [rwtg4ak3](../wandb_final/wandb/run-20260911_145745-rwtg4ak3/) | [78hv2jrf](../wandb_final/wandb/run-20260911_195130-78hv2jrf/) | [0pq71utq](../wandb_final/wandb/run-20260911_201750-0pq71utq/) |
|  |  | 100 | 0.1 | [by13yr9t](../wandb_final/wandb/run-20260911_151819-by13yr9t/) | [t1gn9lvk](../wandb_final/wandb/run-20260911_201623-t1gn9lvk/) | [50latx00](../wandb_final/wandb/run-20260911_215942-50latx00/) |
|  |  | 100 | 1 | [gj77wdpy](../wandb_final/wandb/run-20260911_153526-gj77wdpy/) | [e6vdv7a2](../wandb_final/wandb/run-20260911_201621-e6vdv7a2/) | [tenuodn3](../wandb_final/wandb/run-20260911_225447-tenuodn3/) |
|  | `flow_cylinder_shedding` | 10000 | 0 | [99ns0put](../wandb_final/wandb/run-20260912_204212-99ns0put/) | [oumxu1rv](../wandb_final/wandb/run-20260912_212413-oumxu1rv/) | [zlv5gclw](../wandb_final/wandb/run-20260912_233900-zlv5gclw/) |
|  |  | 10000 | 0.001 | [njtv50p4](../wandb_final/wandb/run-20260911_144936-njtv50p4/) | [9b2gs67a](../wandb_final/wandb/run-20260911_222604-9b2gs67a/) | [vqyfx19x](../wandb_final/wandb/run-20260912_165151-vqyfx19x/) |
|  |  | 10000 | 0.01 | [qe4rqkfm](../wandb_final/wandb/run-20260911_150024-qe4rqkfm/) | [8aad13gp](../wandb_final/wandb/run-20260911_224329-8aad13gp/) | [9g7owsw9](../wandb_final/wandb/run-20260912_200011-9g7owsw9/) |
|  |  | 10000 | 0.1 | [d1z8hba9](../wandb_final/wandb/run-20260911_151900-d1z8hba9/) | [fjanuags](../wandb_final/wandb/run-20260912_134238-fjanuags/) | [xi05tesn](../wandb_final/wandb/run-20260912_201913-xi05tesn/) |
|  |  | 10000 | 1 | [fx8ywn9z](../wandb_final/wandb/run-20260911_153533-fx8ywn9z/) | [9uknzwsw](../wandb_final/wandb/run-20260912_164846-9uknzwsw/) | [jy6b4evq](../wandb_final/wandb/run-20260912_202111-jy6b4evq/) |
|  | `lid_cavity_flow` | 10000 | 0 | [cnugxlp3](../wandb_final/wandb/run-20260911_233400-cnugxlp3/) | [mfk11cbh](../wandb_final/wandb/run-20260912_000649-mfk11cbh/) | [8dnhozi9](../wandb_final/wandb/run-20260912_001834-8dnhozi9/) |
|  |  | 10000 | 0.001 | [gvswajsn](../wandb_final/wandb/run-20260911_145010-gvswajsn/) | [qy3rbqx6](../wandb_final/wandb/run-20260911_180045-qy3rbqx6/) | [x5mmzme5](../wandb_final/wandb/run-20260911_201623-x5mmzme5/) |
|  |  | 10000 | 0.01 | [yvsn8dn5](../wandb_final/wandb/run-20260911_150148-yvsn8dn5/) | [jo0atf8r](../wandb_final/wandb/run-20260911_201623-jo0atf8r/) | [sse7hmwg](../wandb_final/wandb/run-20260911_202159-sse7hmwg/) |
|  |  | 10000 | 0.1 | [w96njh22](../wandb_final/wandb/run-20260911_152001-w96njh22/) | [n3bvumaj](../wandb_final/wandb/run-20260911_201623-n3bvumaj/) | [rs81lrpi](../wandb_final/wandb/run-20260911_220947-rs81lrpi/) |
|  |  | 10000 | 1 | [t4qxma0g](../wandb_final/wandb/run-20260911_170108-t4qxma0g/) | [6jinerk7](../wandb_final/wandb/run-20260911_201621-6jinerk7/) | [qa2aiue0](../wandb_final/wandb/run-20260911_232804-qa2aiue0/) |
|  | `buoyancy_cavity_flow` | 10000 | 0 | [u20b6v2g](../wandb_final/wandb/run-20260913_015653-u20b6v2g/) | [g6e661vm](../wandb_final/wandb/run-20260913_020049-g6e661vm/) | [zc2993lc](../wandb_final/wandb/run-20260913_020449-zc2993lc/) |
|  |  | 10000 | 0.001 | [7pc8agtw](../wandb_final/wandb/run-20260911_145110-7pc8agtw/) | [gryumiyo](../wandb_final/wandb/run-20260912_152422-gryumiyo/) | [cn0fnrwc](../wandb_final/wandb/run-20260912_231058-cn0fnrwc/) |
|  |  | 10000 | 0.01 | [7wgfs8qe](../wandb_final/wandb/run-20260911_150325-7wgfs8qe/) | [8ajptlr6](../wandb_final/wandb/run-20260912_160551-8ajptlr6/) | [72kod1l1](../wandb_final/wandb/run-20260912_235355-72kod1l1/) |
|  |  | 10000 | 0.1 | [46irk5n0](../wandb_final/wandb/run-20260911_152108-46irk5n0/) | [j93405lt](../wandb_final/wandb/run-20260912_164144-j93405lt/) | [ijbxm5o4](../wandb_final/wandb/run-20260913_000258-ijbxm5o4/) |
|  |  | 10000 | 1 | [1pu0pgz5](../wandb_final/wandb/run-20260911_194543-1pu0pgz5/) | [m3sgoeq3](../wandb_final/wandb/run-20260912_214711-m3sgoeq3/) | [9ajf4q7j](../wandb_final/wandb/run-20260913_002801-9ajf4q7j/) |
|  | `taylor_green` | 5000 | 0 | [pab42ujk](../wandb_final/wandb/run-20260911_234645-pab42ujk/) | [ex3985tz](../wandb_final/wandb/run-20260912_000821-ex3985tz/) | [pj57jrk6](../wandb_final/wandb/run-20260912_002145-pj57jrk6/) |
|  |  | 5000 | 0.001 | [htalvo6t](../wandb_final/wandb/run-20260911_145140-htalvo6t/) | [cs8wu78w](../wandb_final/wandb/run-20260911_182203-cs8wu78w/) | [rjwakiom](../wandb_final/wandb/run-20260911_201618-rjwakiom/) |
|  |  | 5000 | 0.01 | [pqxktgp0](../wandb_final/wandb/run-20260911_150525-pqxktgp0/) | [pqk0bh1n](../wandb_final/wandb/run-20260911_201623-pqk0bh1n/) | [8fzbkism](../wandb_final/wandb/run-20260911_202332-8fzbkism/) |
|  |  | 5000 | 0.1 | [cerc1d4p](../wandb_final/wandb/run-20260911_152059-cerc1d4p/) | [ikn98daa](../wandb_final/wandb/run-20260911_201623-ikn98daa/) | [swo0z9zi](../wandb_final/wandb/run-20260911_221420-swo0z9zi/) |
|  |  | 5000 | 1 | [28p66xrs](../wandb_final/wandb/run-20260911_170913-28p66xrs/) | [oqunmwpw](../wandb_final/wandb/run-20260911_201623-oqunmwpw/) | [x8zxzf76](../wandb_final/wandb/run-20260911_232802-x8zxzf76/) |
|  | `taylor_green_coeffs` | 5000 | 0 | [a9ooy2wd](../wandb_final/wandb/run-20260911_234947-a9ooy2wd/) | [lo4au5wi](../wandb_final/wandb/run-20260912_000821-lo4au5wi/) | [9g4xmct5](../wandb_final/wandb/run-20260912_002554-9g4xmct5/) |
|  |  | 5000 | 0.001 | [624dqz15](../wandb_final/wandb/run-20260911_145411-624dqz15/) | [b8dw35r3](../wandb_final/wandb/run-20260911_183621-b8dw35r3/) | [3uv3pdh9](../wandb_final/wandb/run-20260911_201618-3uv3pdh9/) |
|  |  | 5000 | 0.01 | [6u5cwuv8](../wandb_final/wandb/run-20260911_150658-6u5cwuv8/) | [fxclt41s](../wandb_final/wandb/run-20260911_201623-fxclt41s/) | [49ruzhi5](../wandb_final/wandb/run-20260911_202332-49ruzhi5/) |
|  |  | 5000 | 0.1 | [4frnyp99](../wandb_final/wandb/run-20260911_152059-4frnyp99/) | [itdr9bd6](../wandb_final/wandb/run-20260911_201623-itdr9bd6/) | [sjpjs0ix](../wandb_final/wandb/run-20260911_221801-sjpjs0ix/) |
|  |  | 5000 | 1 | [0xkxhigv](../wandb_final/wandb/run-20260911_175128-0xkxhigv/) | [eye1c47x](../wandb_final/wandb/run-20260911_201623-eye1c47x/) | [1i4wpvq5](../wandb_final/wandb/run-20260911_232803-1i4wpvq5/) |
|  | `taylor_green_spacetime` | 5000 | 0 | [tkxk8455](../wandb_final/wandb/run-20260912_000135-tkxk8455/) | [2mjpezul](../wandb_final/wandb/run-20260912_001026-2mjpezul/) | [v9m0rnyn](../wandb_final/wandb/run-20260912_002554-v9m0rnyn/) |
|  |  | 5000 | 0.001 | [hcp20npc](../wandb_final/wandb/run-20260911_145549-hcp20npc/) | [wu95knfa](../wandb_final/wandb/run-20260911_190909-wu95knfa/) | [19n79ktf](../wandb_final/wandb/run-20260911_201623-19n79ktf/) |
|  |  | 5000 | 0.01 | [0amwh0rl](../wandb_final/wandb/run-20260911_150912-0amwh0rl/) | [tjnlb9ej](../wandb_final/wandb/run-20260911_201618-tjnlb9ej/) | [wa1p27rs](../wandb_final/wandb/run-20260911_204128-wa1p27rs/) |
|  |  | 5000 | 0.1 | [9dr2xxwx](../wandb_final/wandb/run-20260911_152059-9dr2xxwx/) | [vn67682j](../wandb_final/wandb/run-20260911_201623-vn67682j/) | [xuhry5ad](../wandb_final/wandb/run-20260911_221801-xuhry5ad/) |
|  |  | 5000 | 1 | [sjc14d3b](../wandb_final/wandb/run-20260911_175611-sjc14d3b/) | [0asmxckc](../wandb_final/wandb/run-20260911_201623-0asmxckc/) | [vn412rw7](../wandb_final/wandb/run-20260911_232803-vn412rw7/) |
|  | `taylor_green_spacetime_coeffs` | 5000 | 0 | [r7stmp01](../wandb_final/wandb/run-20260911_235558-r7stmp01/) | [n0yrjtzc](../wandb_final/wandb/run-20260912_001025-n0yrjtzc/) | [obknm49p](../wandb_final/wandb/run-20260912_002554-obknm49p/) |
|  |  | 5000 | 0.001 | [6ehtv0cv](../wandb_final/wandb/run-20260911_145541-6ehtv0cv/) | [8ca714uq](../wandb_final/wandb/run-20260911_192409-8ca714uq/) | [fkl16fnr](../wandb_final/wandb/run-20260911_201623-fkl16fnr/) |
|  |  | 5000 | 0.01 | [5ucid0l6](../wandb_final/wandb/run-20260911_150904-5ucid0l6/) | [h5cxl7p9](../wandb_final/wandb/run-20260911_201618-h5cxl7p9/) | [gcrrdd4y](../wandb_final/wandb/run-20260911_204432-gcrrdd4y/) |
|  |  | 5000 | 0.1 | [q106r6bj](../wandb_final/wandb/run-20260911_152204-q106r6bj/) | [5e8uurf8](../wandb_final/wandb/run-20260911_201623-5e8uurf8/) | [malpt36k](../wandb_final/wandb/run-20260911_222437-malpt36k/) |
|  |  | 5000 | 1 | [x9z79vx4](../wandb_final/wandb/run-20260911_175611-x9z79vx4/) | [pc0x5fw6](../wandb_final/wandb/run-20260911_201623-pc0x5fw6/) | [8vshmjid](../wandb_final/wandb/run-20260911_233305-8vshmjid/) |
|  | `merge_vortices_easier` | 500 | 0 | [aao4cuai](../wandb_final/wandb/run-20260912_000308-aao4cuai/) | [n6x347xd](../wandb_final/wandb/run-20260912_001158-n6x347xd/) | [y5oofven](../wandb_final/wandb/run-20260912_002656-y5oofven/) |
|  |  | 500 | 0.001 | [uea8ryex](../wandb_final/wandb/run-20260911_145620-uea8ryex/) | [g7jv0o3m](../wandb_final/wandb/run-20260911_194411-g7jv0o3m/) | [60kwn7o3](../wandb_final/wandb/run-20260911_201623-60kwn7o3/) |
|  |  | 500 | 0.01 | [b0l82o2r](../wandb_final/wandb/run-20260911_151038-b0l82o2r/) | [gviy8e31](../wandb_final/wandb/run-20260911_201618-gviy8e31/) | [sg2vjt18](../wandb_final/wandb/run-20260911_213320-sg2vjt18/) |
|  |  | 500 | 0.1 | [azwf5mek](../wandb_final/wandb/run-20260911_152509-azwf5mek/) | [soxmmkbw](../wandb_final/wandb/run-20260911_201623-soxmmkbw/) | [v1hl88bn](../wandb_final/wandb/run-20260911_222437-v1hl88bn/) |
|  |  | 500 | 1 | [12ucd5hg](../wandb_final/wandb/run-20260911_175741-12ucd5hg/) | [okpir40o](../wandb_final/wandb/run-20260911_201623-okpir40o/) | [iyqc0mv4](../wandb_final/wandb/run-20260911_233400-iyqc0mv4/) |
|  | `species_transport` | 10000 | 0 | [0qmutkrp](../wandb_final/wandb/run-20260912_000308-0qmutkrp/) | [lys1koon](../wandb_final/wandb/run-20260912_001330-lys1koon/) | [zsqmf0d4](../wandb_final/wandb/run-20260912_002656-zsqmf0d4/) |
|  |  | 10000 | 0.001 | [3wpvsn9k](../wandb_final/wandb/run-20260911_145753-3wpvsn9k/) | [3lqa62k7](../wandb_final/wandb/run-20260911_194543-3lqa62k7/) | [03xbpq7a](../wandb_final/wandb/run-20260911_201623-03xbpq7a/) |
|  |  | 10000 | 0.01 | [jz82d80c](../wandb_final/wandb/run-20260911_151636-jz82d80c/) | [qb32n87l](../wandb_final/wandb/run-20260911_201618-qb32n87l/) | [baflxyd7](../wandb_final/wandb/run-20260911_213954-baflxyd7/) |
|  |  | 10000 | 0.1 | [qsmmkez3](../wandb_final/wandb/run-20260911_153117-qsmmkez3/) | [0w4gmjdw](../wandb_final/wandb/run-20260911_201621-0w4gmjdw/) | [ozvzbuds](../wandb_final/wandb/run-20260911_223118-ozvzbuds/) |
|  |  | 10000 | 1 | [w6iec4ej](../wandb_final/wandb/run-20260911_175911-w6iec4ej/) | [ea81gu7b](../wandb_final/wandb/run-20260911_201623-ea81gu7b/) | [g6tkq5hq](../wandb_final/wandb/run-20260911_233400-g6tkq5hq/) |
|  | `forced_turb` | 10000 | 0.001 | [sy20d9a4](../wandb_final/wandb/run-20260916_001243-sy20d9a4/) | — | — |
|  |  | 10000 | 0.01 | [zu4bdbus](../wandb_final/wandb/run-20260916_030738-zu4bdbus/) | — | — |
|  |  | 10000 | 0.1 | [u3jqtq0p](../wandb_final/wandb/run-20260916_034334-u3jqtq0p/) | — | — |
|  |  | 10000 | 1 | [vkg3dplu](../wandb_final/wandb/run-20260916_035109-vkg3dplu/) | — | — |
|  |  | 5000 | 0 | [c68ptqq8](../wandb_final/wandb/run-20260916_014209-c68ptqq8/) | — | — |
|  |  | 1000 | 0 | [5igoh4h6](../wandb_final/wandb/run-20260915_021317-5igoh4h6/) | [ffsvzm7h](../wandb_final/wandb/run-20260915_022202-ffsvzm7h/) | [23ixpeqy](../wandb_final/wandb/run-20260915_024747-23ixpeqy/) |
|  |  | 500 | 0 | [q8m7d3h6](../wandb_final/wandb/run-20260914_175010-q8m7d3h6/) | [hwjpgp1i](../wandb_final/wandb/run-20260914_190220-hwjpgp1i/) | [ttjuxnb5](../wandb_final/wandb/run-20260915_021037-ttjuxnb5/) |
|  |  | 100 | 0 | [vvpmyw12](../wandb_final/wandb/run-20260914_164649-vvpmyw12/) | [7alv02kp](../wandb_final/wandb/run-20260914_183326-7alv02kp/) | [2jbruyxc](../wandb_final/wandb/run-20260915_014841-2jbruyxc/) |
| Transolver | `backward_facing_step` | 500 | 0 | [5cbzvmiy](../wandb_final/wandb/run-20260911_234445-5cbzvmiy/) | [s2f6ow9m](../wandb_final/wandb/run-20260912_201720-s2f6ow9m/) | [hs1eqr8i](../wandb_final/wandb/run-20260912_230611-hs1eqr8i/) |
|  |  | 500 | 0.001 | [bjgsdnqe](../wandb_final/wandb/run-20260911_145104-bjgsdnqe/) | [s9blvm9i](../wandb_final/wandb/run-20260911_205751-s9blvm9i/) | [q1k32a9r](../wandb_final/wandb/run-20260911_214431-q1k32a9r/) |
|  |  | 500 | 0.01 | [dq0ug68q](../wandb_final/wandb/run-20260911_150330-dq0ug68q/) | [zkyxkcds](../wandb_final/wandb/run-20260911_210809-zkyxkcds/) | [5hqycsca](../wandb_final/wandb/run-20260911_225949-5hqycsca/) |
|  |  | 500 | 0.1 | [msoreepr](../wandb_final/wandb/run-20260911_152010-msoreepr/) | [kcxk4b8x](../wandb_final/wandb/run-20260911_213025-kcxk4b8x/) | [5tg0x07a](../wandb_final/wandb/run-20260911_230255-5tg0x07a/) |
|  |  | 500 | 1 | [djnzes7c](../wandb_final/wandb/run-20260911_185045-djnzes7c/) | [ez1yej4g](../wandb_final/wandb/run-20260911_213823-ez1yej4g/) | [rj3lvs5r](../wandb_final/wandb/run-20260911_232803-rj3lvs5r/) |
|  | `flow_cylinder_laminar` | 100 | 0 | [ikuf3pcl](../wandb_final/wandb/run-20260911_234320-ikuf3pcl/) | [1nkh0rur](../wandb_final/wandb/run-20260912_155617-1nkh0rur/) | [xp97nnpl](../wandb_final/wandb/run-20260912_223402-xp97nnpl/) |
|  |  | 100 | 0.001 | [i39xsi90](../wandb_final/wandb/run-20260911_144940-i39xsi90/) | [q1qr7bv0](../wandb_final/wandb/run-20260911_194934-q1qr7bv0/) | [oouiqcpj](../wandb_final/wandb/run-20260911_214309-oouiqcpj/) |
|  |  | 100 | 0.01 | [u51axn0p](../wandb_final/wandb/run-20260911_150018-u51axn0p/) | [idt8fit6](../wandb_final/wandb/run-20260911_210431-idt8fit6/) | [93x4f2z8](../wandb_final/wandb/run-20260911_224801-93x4f2z8/) |
|  |  | 100 | 0.1 | [hj59wiwm](../wandb_final/wandb/run-20260911_151903-hj59wiwm/) | [04ujce2f](../wandb_final/wandb/run-20260911_212539-04ujce2f/) | [bo64f588](../wandb_final/wandb/run-20260911_230255-bo64f588/) |
|  |  | 100 | 1 | [gpruhtf7](../wandb_final/wandb/run-20260911_153528-gpruhtf7/) | [jt57afgb](../wandb_final/wandb/run-20260911_213320-jt57afgb/) | [si8bo278](../wandb_final/wandb/run-20260911_232518-si8bo278/) |
|  | `flow_cylinder_shedding` | 10000 | 0 | [2qkm11s9](../wandb_final/wandb/run-20260911_234446-2qkm11s9/) | [54136p96](../wandb_final/wandb/run-20260912_155815-54136p96/) | [y22ymvh5](../wandb_final/wandb/run-20260912_224101-y22ymvh5/) |
|  |  | 10000 | 0.001 | [gdank786](../wandb_final/wandb/run-20260911_144940-gdank786/) | [ot8sz7h2](../wandb_final/wandb/run-20260911_195131-ot8sz7h2/) | [8etgosm6](../wandb_final/wandb/run-20260911_214309-8etgosm6/) |
|  |  | 10000 | 0.01 | [dvm7tv86](../wandb_final/wandb/run-20260911_150056-dvm7tv86/) | [ki664fjv](../wandb_final/wandb/run-20260911_210431-ki664fjv/) | [nus5xyph](../wandb_final/wandb/run-20260911_225949-nus5xyph/) |
|  |  | 10000 | 0.1 | [e9rgd7yf](../wandb_final/wandb/run-20260911_152002-e9rgd7yf/) | [5gjyyjai](../wandb_final/wandb/run-20260911_212539-5gjyyjai/) | [0gp726sz](../wandb_final/wandb/run-20260911_230255-0gp726sz/) |
|  |  | 10000 | 1 | [ld914d5w](../wandb_final/wandb/run-20260911_164912-ld914d5w/) | [a42nbibb](../wandb_final/wandb/run-20260911_213522-a42nbibb/) | [sf959c8d](../wandb_final/wandb/run-20260911_232518-sf959c8d/) |
|  | `lid_cavity_flow` | 10000 | 0 | [do40p8xq](../wandb_final/wandb/run-20260911_234446-do40p8xq/) | [5t2hmcp8](../wandb_final/wandb/run-20260912_160500-5t2hmcp8/) | [kklefqk4](../wandb_final/wandb/run-20260912_230016-kklefqk4/) |
|  |  | 10000 | 0.001 | [z0fdlgvh](../wandb_final/wandb/run-20260911_145112-z0fdlgvh/) | [9nj6bc1k](../wandb_final/wandb/run-20260911_201954-9nj6bc1k/) | [3tb4qndw](../wandb_final/wandb/run-20260911_214309-3tb4qndw/) |
|  |  | 10000 | 0.01 | [h87a15yl](../wandb_final/wandb/run-20260911_150150-h87a15yl/) | [0zf9xuyz](../wandb_final/wandb/run-20260911_210809-0zf9xuyz/) | [i3hykrq4](../wandb_final/wandb/run-20260911_225950-i3hykrq4/) |
|  |  | 10000 | 0.1 | [ejr317wm](../wandb_final/wandb/run-20260911_152003-ejr317wm/) | [y49l16xp](../wandb_final/wandb/run-20260911_213025-y49l16xp/) | [emh50bl0](../wandb_final/wandb/run-20260911_230255-emh50bl0/) |
|  |  | 10000 | 1 | [6u0m5miy](../wandb_final/wandb/run-20260911_180756-6u0m5miy/) | [ronkkkqy](../wandb_final/wandb/run-20260911_213654-ronkkkqy/) | [zj5m7uzt](../wandb_final/wandb/run-20260911_232807-zj5m7uzt/) |
|  | `buoyancy_cavity_flow` | 10000 | 0 | [dr5cvy6p](../wandb_final/wandb/run-20260911_234647-dr5cvy6p/) | [x41prbl4](../wandb_final/wandb/run-20260912_202811-x41prbl4/) | [3sjhbspd](../wandb_final/wandb/run-20260912_234901-3sjhbspd/) |
|  |  | 10000 | 0.001 | [70g5xud9](../wandb_final/wandb/run-20260915_012616-70g5xud9/) | [w62ave0a](../wandb_final/wandb/run-20260915_024751-w62ave0a/) | [krmh3csu](../wandb_final/wandb/run-20260915_142930-krmh3csu/) |
|  |  | 10000 | 0.01 | [knvswevi](../wandb_final/wandb/run-20260915_024748-knvswevi/) | [u8z0osmf](../wandb_final/wandb/run-20260915_024750-u8z0osmf/) | [twqps9fi](../wandb_final/wandb/run-20260915_143835-twqps9fi/) |
|  |  | 10000 | 0.1 | [df5q1jzf](../wandb_final/wandb/run-20260915_024749-df5q1jzf/) | [4wgymhj6](../wandb_final/wandb/run-20260915_024845-4wgymhj6/) | [cxgbqlta](../wandb_final/wandb/run-20260915_150320-cxgbqlta/) |
|  |  | 10000 | 1 | [kbuccnot](../wandb_final/wandb/run-20260915_024750-kbuccnot/) | [bi5cdn2h](../wandb_final/wandb/run-20260915_040751-bi5cdn2h/) | [ekw4iuca](../wandb_final/wandb/run-20260915_152236-ekw4iuca/) |
|  | `taylor_green` | 5000 | 0 | [c4wbtisw](../wandb_final/wandb/run-20260911_234646-c4wbtisw/) | [cq0ud9qu](../wandb_final/wandb/run-20260912_214511-cq0ud9qu/) | [v8xp1nfb](../wandb_final/wandb/run-20260912_235356-v8xp1nfb/) |
|  |  | 5000 | 0.001 | [ougpjjp6](../wandb_final/wandb/run-20260911_145314-ougpjjp6/) | [q0k8266t](../wandb_final/wandb/run-20260911_210108-q0k8266t/) | [10bkljil](../wandb_final/wandb/run-20260911_214633-10bkljil/) |
|  |  | 5000 | 0.01 | [ltxj4xkw](../wandb_final/wandb/run-20260911_150526-ltxj4xkw/) | [0ihoxrs3](../wandb_final/wandb/run-20260911_211016-0ihoxrs3/) | [1judymnw](../wandb_final/wandb/run-20260911_230253-1judymnw/) |
|  |  | 5000 | 0.1 | [9gr3e052](../wandb_final/wandb/run-20260911_152109-9gr3e052/) | [990rpp1j](../wandb_final/wandb/run-20260911_213021-990rpp1j/) | [3ac14lkv](../wandb_final/wandb/run-20260911_230421-3ac14lkv/) |
|  |  | 5000 | 1 | [t6imxk42](../wandb_final/wandb/run-20260911_185255-t6imxk42/) | [m7py8977](../wandb_final/wandb/run-20260911_213823-m7py8977/) | [0x97sc6y](../wandb_final/wandb/run-20260911_232803-0x97sc6y/) |
|  | `taylor_green_coeffs` | 5000 | 0 | [gm655o1s](../wandb_final/wandb/run-20260911_235301-gm655o1s/) | [djye9c5u](../wandb_final/wandb/run-20260912_215522-djye9c5u/) | [bqbv4b1j](../wandb_final/wandb/run-20260912_235701-bqbv4b1j/) |
|  |  | 5000 | 0.001 | [6a6v3v13](../wandb_final/wandb/run-20260911_145543-6a6v3v13/) | [88qwt8wj](../wandb_final/wandb/run-20260911_210108-88qwt8wj/) | [kwo39eiq](../wandb_final/wandb/run-20260911_214633-kwo39eiq/) |
|  |  | 5000 | 0.01 | [d8953w0c](../wandb_final/wandb/run-20260911_150807-d8953w0c/) | [ysg3g12f](../wandb_final/wandb/run-20260911_211016-ysg3g12f/) | [1lzm9126](../wandb_final/wandb/run-20260911_230253-1lzm9126/) |
|  |  | 5000 | 0.1 | [7dwr41si](../wandb_final/wandb/run-20260911_152100-7dwr41si/) | [5ljkauji](../wandb_final/wandb/run-20260911_213022-5ljkauji/) | [4ehiitsv](../wandb_final/wandb/run-20260911_230554-4ehiitsv/) |
|  |  | 5000 | 1 | [wg5zpvn1](../wandb_final/wandb/run-20260911_191755-wg5zpvn1/) | [c857ct0n](../wandb_final/wandb/run-20260911_213957-c857ct0n/) | [u6e675x6](../wandb_final/wandb/run-20260911_232803-u6e675x6/) |
|  | `taylor_green_spacetime` | 5000 | 0 | [staf5qhm](../wandb_final/wandb/run-20260912_001027-staf5qhm/) | [qq1lpgbs](../wandb_final/wandb/run-20260912_220304-qq1lpgbs/) | [sihyu01k](../wandb_final/wandb/run-20260912_235703-sihyu01k/) |
|  |  | 5000 | 0.001 | [dkg38fuh](../wandb_final/wandb/run-20260911_145543-dkg38fuh/) | [fvjsrgqj](../wandb_final/wandb/run-20260911_210308-fvjsrgqj/) | [8jiw3pw8](../wandb_final/wandb/run-20260911_214633-8jiw3pw8/) |
|  |  | 5000 | 0.01 | [zs3c7igz](../wandb_final/wandb/run-20260911_150916-zs3c7igz/) | [tht0cgh6](../wandb_final/wandb/run-20260911_211016-tht0cgh6/) | [vbjcjz4s](../wandb_final/wandb/run-20260911_230253-vbjcjz4s/) |
|  |  | 5000 | 0.1 | [i7q7f69y](../wandb_final/wandb/run-20260911_152146-i7q7f69y/) | [r34jboxe](../wandb_final/wandb/run-20260911_213151-r34jboxe/) | [alzppz3f](../wandb_final/wandb/run-20260911_232043-alzppz3f/) |
|  |  | 5000 | 1 | [uuwsh5cc](../wandb_final/wandb/run-20260911_192039-uuwsh5cc/) | [8vm9tacs](../wandb_final/wandb/run-20260911_213957-8vm9tacs/) | [r4mtlj3y](../wandb_final/wandb/run-20260911_233307-r4mtlj3y/) |
|  | `taylor_green_spacetime_coeffs` | 5000 | 0 | [904fxoye](../wandb_final/wandb/run-20260912_003303-904fxoye/) | [gvxvbpxi](../wandb_final/wandb/run-20260912_222258-gvxvbpxi/) | [f9zo85x6](../wandb_final/wandb/run-20260912_235703-f9zo85x6/) |
|  |  | 5000 | 0.001 | [fg75wg1y](../wandb_final/wandb/run-20260911_145552-fg75wg1y/) | [j28pu92e](../wandb_final/wandb/run-20260911_210308-j28pu92e/) | [kv9drupk](../wandb_final/wandb/run-20260911_223613-kv9drupk/) |
|  |  | 5000 | 0.01 | [xnmltmr1](../wandb_final/wandb/run-20260911_151038-xnmltmr1/) | [aoq4q5dy](../wandb_final/wandb/run-20260911_211016-aoq4q5dy/) | [fd0j2vls](../wandb_final/wandb/run-20260911_230257-fd0j2vls/) |
|  |  | 5000 | 0.1 | [8hyuhx1e](../wandb_final/wandb/run-20260911_152341-8hyuhx1e/) | [3bavbenw](../wandb_final/wandb/run-20260911_213151-3bavbenw/) | [oz9giiwk](../wandb_final/wandb/run-20260911_232043-oz9giiwk/) |
|  |  | 5000 | 1 | [9muxpbjb](../wandb_final/wandb/run-20260911_193121-9muxpbjb/) | [9om4nz8f](../wandb_final/wandb/run-20260911_213957-9om4nz8f/) | [bmasdgge](../wandb_final/wandb/run-20260911_233709-bmasdgge/) |
|  | `merge_vortices_easier` | 500 | 0 | [spxlfy91](../wandb_final/wandb/run-20260912_120933-spxlfy91/) | [d5mrykhs](../wandb_final/wandb/run-20260912_195220-d5mrykhs/) | [qv9udj8i](../wandb_final/wandb/run-20260913_001305-qv9udj8i/) |
|  |  | 500 | 0.001 | [t8ps8yk8](../wandb_final/wandb/run-20260911_145614-t8ps8yk8/) | [1hun1hvn](../wandb_final/wandb/run-20260911_210308-1hun1hvn/) | [xa9wuku9](../wandb_final/wandb/run-20260911_223746-xa9wuku9/) |
|  |  | 500 | 0.01 | [cxpqw5p3](../wandb_final/wandb/run-20260911_151639-cxpqw5p3/) | [d2yg2531](../wandb_final/wandb/run-20260911_211014-d2yg2531/) | [khlpo0mm](../wandb_final/wandb/run-20260911_230257-khlpo0mm/) |
|  |  | 500 | 0.1 | [mvky47nl](../wandb_final/wandb/run-20260911_152510-mvky47nl/) | [d45hn6he](../wandb_final/wandb/run-20260911_213321-d45hn6he/) | [vrhgswty](../wandb_final/wandb/run-20260911_232043-vrhgswty/) |
|  |  | 500 | 1 | [6p24tu6w](../wandb_final/wandb/run-20260911_193326-6p24tu6w/) | [239bqm0q](../wandb_final/wandb/run-20260911_213955-239bqm0q/) | [2e6jvbot](../wandb_final/wandb/run-20260911_234315-2e6jvbot/) |
|  | `species_transport` | 10000 | 0 | [3lve36ss](../wandb_final/wandb/run-20260915_152237-3lve36ss/) | [1odurtu1](../wandb_final/wandb/run-20260915_152530-1odurtu1/) | [bsw06ns4](../wandb_final/wandb/run-20260915_152628-bsw06ns4/) |
|  |  | 10000 | 0.001 | [wbkq3f72](../wandb_final/wandb/run-20260911_145746-wbkq3f72/) | [idtu5hrg](../wandb_final/wandb/run-20260911_210431-idtu5hrg/) | [wfmshh1u](../wandb_final/wandb/run-20260911_224454-wfmshh1u/) |
|  |  | 10000 | 0.01 | [a2hjiy0y](../wandb_final/wandb/run-20260911_151821-a2hjiy0y/) | [6o2722r4](../wandb_final/wandb/run-20260911_212539-6o2722r4/) | [hef3epj4](../wandb_final/wandb/run-20260911_230257-hef3epj4/) |
|  |  | 10000 | 0.1 | [glp7wjjt](../wandb_final/wandb/run-20260911_153322-glp7wjjt/) | [nyq1fl1j](../wandb_final/wandb/run-20260911_213321-nyq1fl1j/) | [ybnvm7gt](../wandb_final/wandb/run-20260911_232521-ybnvm7gt/) |
|  |  | 10000 | 1 | [274fgfz6](../wandb_final/wandb/run-20260911_193908-274fgfz6/) | [glu0sz46](../wandb_final/wandb/run-20260911_214309-glu0sz46/) | [ba67nb31](../wandb_final/wandb/run-20260911_234312-ba67nb31/) |
|  | `forced_turb` | 10000 | 0.001 | [msc3p1s2](../wandb_final/wandb/run-20260916_024646-msc3p1s2/) | — | — |
|  |  | 10000 | 0.01 | [340bvcen](../wandb_final/wandb/run-20260916_034341-340bvcen/) | — | — |
|  |  | 10000 | 0.1 | [9jbh154b](../wandb_final/wandb/run-20260916_034607-9jbh154b/) | — | — |
|  |  | 10000 | 1 | [3kydzigh](../wandb_final/wandb/run-20260916_040527-3kydzigh/) | — | — |
|  |  | 5000 | 0 | [zld33s10](../wandb_final/wandb/run-20260916_024923-zld33s10/) | — | — |
|  |  | 1000 | 0 | [qnvzzzlw](../wandb_final/wandb/run-20260915_010920-qnvzzzlw/) | [p5q5t0zr](../wandb_final/wandb/run-20260915_022551-p5q5t0zr/) | [bnwf87gw](../wandb_final/wandb/run-20260915_022854-bnwf87gw/) |
|  |  | 500 | 0 | [gxofae1v](../wandb_final/wandb/run-20260914_180734-gxofae1v/) | [22l32gcl](../wandb_final/wandb/run-20260915_005624-22l32gcl/) | [t0gb5vnu](../wandb_final/wandb/run-20260915_022350-t0gb5vnu/) |
|  |  | 100 | 0 | [tln2277b](../wandb_final/wandb/run-20260914_174012-tln2277b/) | [xzp0bu81](../wandb_final/wandb/run-20260914_185738-xzp0bu81/) | [46r8qdzq](../wandb_final/wandb/run-20260915_020851-46r8qdzq/) |
