# Final results

New RAM-dataset batch: runs started 11–13 September 2026 in `../wandb`, all recorded at commit `eecb2394c41a7a9d8e1402bb379aa536eed4ea3d`. Older runs and values from `new_res.md` are not mixed into these averages.

Rows follow the current `train_div.sh`: no-div baseline and divergence-penalty weight $\lambda \in \{0.001, 0.01, 0.1, 1\}$. Each configuration targets seeds 1, 2, 3 and 500 epochs. The columns `ntrain` and `npoints` give the requested training-sample count and spatial point budget; spacetime layouts can expand this budget. `Div. order` is the logged RBF-FD polynomial degree, not an empirically measured convergence order. The new Geo-FNO order-2 runs are retained here because they belong to this batch; historical order-2 runs are excluded.

Metric entries are mean ± population standard deviation across seeds with finite final `test_loss`; NaN seeds are excluded and explicitly listed. A single finite seed has no standard deviation. For scalar seed values $z_1,\ldots,z_K$, where $K$ is the number of contributing seeds, the reported mean $\bar z$ and standard deviation $\sigma$ are:

$$
\bar z=\frac{1}{K}\sum_{j=1}^{K}z_j,\qquad
\sigma=\sqrt{\frac{1}{K}\sum_{j=1}^{K}(z_j-\bar z)^2}.
$$

`Time (s)` averages the logged `total_train_time` over all runs with that field, including NaN runs. It is not W&B runtime: Geo-FNO's timer includes final test evaluation (and its coefficient entry places it after OOD handling), while Transolver stops its timer before final evaluation.

Test and OOD losses are mean relative Euclidean errors of **field magnitudes**, not vector-field errors. For evaluation sample $i$, let $\widehat u_{i,q}$ and $u_{i,q}$ be the predicted and target component vectors at output index $q$. Define magnitude vectors $\widehat a_i=(\|\widehat u_{i,q}\|_2)_q$ and $a_i=(\|u_{i,q}\|_2)_q$. With $M$ evaluation samples, the logged loss is:

$$
\frac{1}{M}\sum_{i=1}^{M}\frac{\|\widehat a_i-a_i\|_2}{\|a_i\|_2}.
$$

Interior divergence is the maximum / median absolute discrete divergence over test samples and interior points (including time slices for spacetime datasets), followed by the seed aggregation above. Different orders produce different discrete diagnostics.

`NaN` means a logged numerical failure; `No summary` means an attempted run has metadata but no cached summary, not necessarily a NaN failure. Blank metric cells indicate no result; `—` indicates an absent metric in a populated result. `Disabled` means OOD evaluation was disabled for div training; `No OOD data` means the run logged `ood_available=false`. Unqualified seeds in the status column have finite test loss.

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
|  |  | div | 0.001 | 3 |  |  | Missing: 1, 2, 3 |  |  |  |  |
|  |  | div | 0.01 | 3 |  |  | Missing: 1, 2, 3 |  |  |  |  |
|  |  | div | 0.1 | 3 |  |  | Missing: 1, 2, 3 |  |  |  |  |
|  |  | div | 1 | 3 |  |  | Missing: 1, 2, 3 |  |  |  |  |
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
|  |  | div | 0.001 | 4 |  |  | No summary: 1, 2, 3 |  |  |  |  |
|  |  | div | 0.01 | 4 |  |  | No summary: 1, 2, 3 |  |  |  |  |
|  |  | div | 0.1 | 4 |  |  | No summary: 1, 2, 3 |  |  |  |  |
|  |  | div | 1 | 4 |  |  | No summary: 1, 2, 3 |  |  |  |  |
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
|  | **3D Species Transport** | no-div | 0 | 4 | 10000 | 7000 | No summary: 1, 2, 3 |  |  |  |  |
|  |  | div | 0.001 | 4 |  |  | No summary: 1, 2, 3 |  |  |  |  |
|  |  | div | 0.01 | 4 |  |  | No summary: 1, 2, 3 |  |  |  |  |
|  |  | div | 0.1 | 4 |  |  | No summary: 1, 2, 3 |  |  |  |  |
|  |  | div | 1 | 4 |  |  | No summary: 1, 2, 3 |  |  |  |  |
|  | **3D Homogeneous Forced Isotropic Turbulence** | no-div | 0 | 4 | 10000 | 7000 | Missing: 1, 2, 3 |  |  |  |  |
|  |  | div | 0.001 | 4 |  |  | Missing: 1, 2, 3 |  |  |  |  |
|  |  | div | 0.01 | 4 |  |  | Missing: 1, 2, 3 |  |  |  |  |
|  |  | div | 0.1 | 4 |  |  | Missing: 1, 2, 3 |  |  |  |  |
|  |  | div | 1 | 4 |  |  | Missing: 1, 2, 3 |  |  |  |  |

## Forced-turbulence training-size sweep

No runs from this new sweep are present in the cache. All rows are no-div, with 7,000 requested points and seeds 1, 2, 3. The 10,000-training-sample configurations are listed in the main table.

| Framework | ntrain | npoints | Div. order | Seeds / status | Time (s) | Test loss | Interior test div (max / median) | OOD loss |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| **Geo-FNO** | 100 | 7000 | 3 | Missing: 1, 2, 3 |  |  |  |  |
|  | 500 | 7000 | 3 | Missing: 1, 2, 3 |  |  |  |  |
|  | 1000 | 7000 | 3 | Missing: 1, 2, 3 |  |  |  |  |
|  | 5000 | 7000 | 3 | Missing: 1, 2, 3 |  |  |  |  |
|  | 7000 | 7000 | 3 | Missing: 1, 2, 3 |  |  |  |  |
| **Transolver** | 100 | 7000 | 4 | Missing: 1, 2, 3 |  |  |  |  |
|  | 500 | 7000 | 4 | Missing: 1, 2, 3 |  |  |  |  |
|  | 1000 | 7000 | 4 | Missing: 1, 2, 3 |  |  |  |  |
|  | 5000 | 7000 | 4 | Missing: 1, 2, 3 |  |  |  |  |
|  | 7000 | 7000 | 4 | Missing: 1, 2, 3 |  |  |  |  |

## Completion notes

- 330 new cached runs: 232 finite final test losses, 53 NaN test losses, and 45 without summaries. There are no duplicate seeds within a configuration.
- All species-transport attempts (both models) and all Transolver buoyancy-cavity attempts lack summaries: 15 runs per model/dataset combination. Their metric cells are blank, not fabricated NaNs.
- No new forced-turbulence runs are cached: 30 main-table runs and 30 training-size-sweep runs remain unaccounted for locally. Cache absence does not establish whether a cluster job ran.
- Geo-FNO baseline OOD losses are absent from all 30 available summaries, even where `ood_available=true`. Its code logs `ood_available` without an explicit step, then logs OOD at the previous epoch step; this ordering can drop the OOD metric. No OOD value is inferred from the test loss.
- Geo-FNO Taylor–Green coefficient baselines also lack summary time and divergence fields, which are logged after that same step increment. Their test losses are available.
- Available baseline summaries mark buoyancy-cavity and both Taylor–Green coefficient variants as lacking OOD data (Transolver buoyancy has no summary to inspect).
- Previously requested NaN placeholders are not carried over where these new runs provide actual finite measurements.

## Run provenance

Each link identifies the cached run directory; use `files/config.yaml` and `files/wandb-summary.json`, or `files/wandb-metadata.json` arguments where the config/summary is absent. Seed links include NaN and no-summary attempts. Training sizes and orders match the main table.

| Framework | Dataset ID | $\lambda$ | Seed 1 | Seed 2 | Seed 3 |
|---|---|---:|---|---|---|
| Geo-FNO | `backward_facing_step` | 0 | [ycxutvm3](../wandb/run-20260911_233838-ycxutvm3/) | [635kjgnn](../wandb/run-20260912_000649-635kjgnn/) | [8b91c797](../wandb/run-20260912_002007-8b91c797/) |
|  |  | 0.001 | [h7oj0a4x](../wandb/run-20260911_145108-h7oj0a4x/) | [ir1kjgo3](../wandb/run-20260911_180045-ir1kjgo3/) | [33310qmo](../wandb/run-20260911_201618-33310qmo/) |
|  |  | 0.01 | [74jn4i2m](../wandb/run-20260911_150222-74jn4i2m/) | [fhrzluqe](../wandb/run-20260911_201623-fhrzluqe/) | [yj07iw9i](../wandb/run-20260911_202332-yj07iw9i/) |
|  |  | 0.1 | [f8vj489k](../wandb/run-20260911_152007-f8vj489k/) | [983yerdp](../wandb/run-20260911_201623-983yerdp/) | [5cxxovm1](../wandb/run-20260911_220947-5cxxovm1/) |
|  |  | 1 | [5ffp5e5r](../wandb/run-20260911_170403-5ffp5e5r/) | [iu8aiuei](../wandb/run-20260911_201621-iu8aiuei/) | [peeaz7va](../wandb/run-20260911_232802-peeaz7va/) |
|  | `flow_cylinder_laminar` | 0 | [2m13e23v](../wandb/run-20260911_233400-2m13e23v/) | [vdw7cuzo](../wandb/run-20260912_000649-vdw7cuzo/) | [ttfvriio](../wandb/run-20260912_001705-ttfvriio/) |
|  |  | 0.001 | [cj9415e3](../wandb/run-20260911_144936-cj9415e3/) | [2k4n5qkb](../wandb/run-20260911_180046-2k4n5qkb/) | [opnsfg7s](../wandb/run-20260911_201623-opnsfg7s/) |
|  |  | 0.01 | [rwtg4ak3](../wandb/run-20260911_145745-rwtg4ak3/) | [78hv2jrf](../wandb/run-20260911_195130-78hv2jrf/) | [0pq71utq](../wandb/run-20260911_201750-0pq71utq/) |
|  |  | 0.1 | [by13yr9t](../wandb/run-20260911_151819-by13yr9t/) | [t1gn9lvk](../wandb/run-20260911_201623-t1gn9lvk/) | [50latx00](../wandb/run-20260911_215942-50latx00/) |
|  |  | 1 | [gj77wdpy](../wandb/run-20260911_153526-gj77wdpy/) | [e6vdv7a2](../wandb/run-20260911_201621-e6vdv7a2/) | [tenuodn3](../wandb/run-20260911_225447-tenuodn3/) |
|  | `flow_cylinder_shedding` | 0 | [99ns0put](../wandb/run-20260912_204212-99ns0put/) | [oumxu1rv](../wandb/run-20260912_212413-oumxu1rv/) | [zlv5gclw](../wandb/run-20260912_233900-zlv5gclw/) |
|  |  | 0.001 | [njtv50p4](../wandb/run-20260911_144936-njtv50p4/) | [9b2gs67a](../wandb/run-20260911_222604-9b2gs67a/) | [vqyfx19x](../wandb/run-20260912_165151-vqyfx19x/) |
|  |  | 0.01 | [qe4rqkfm](../wandb/run-20260911_150024-qe4rqkfm/) | [8aad13gp](../wandb/run-20260911_224329-8aad13gp/) | [9g7owsw9](../wandb/run-20260912_200011-9g7owsw9/) |
|  |  | 0.1 | [d1z8hba9](../wandb/run-20260911_151900-d1z8hba9/) | [fjanuags](../wandb/run-20260912_134238-fjanuags/) | [xi05tesn](../wandb/run-20260912_201913-xi05tesn/) |
|  |  | 1 | [fx8ywn9z](../wandb/run-20260911_153533-fx8ywn9z/) | [9uknzwsw](../wandb/run-20260912_164846-9uknzwsw/) | [jy6b4evq](../wandb/run-20260912_202111-jy6b4evq/) |
|  | `lid_cavity_flow` | 0 | [cnugxlp3](../wandb/run-20260911_233400-cnugxlp3/) | [mfk11cbh](../wandb/run-20260912_000649-mfk11cbh/) | [8dnhozi9](../wandb/run-20260912_001834-8dnhozi9/) |
|  |  | 0.001 | [gvswajsn](../wandb/run-20260911_145010-gvswajsn/) | [qy3rbqx6](../wandb/run-20260911_180045-qy3rbqx6/) | [x5mmzme5](../wandb/run-20260911_201623-x5mmzme5/) |
|  |  | 0.01 | [yvsn8dn5](../wandb/run-20260911_150148-yvsn8dn5/) | [jo0atf8r](../wandb/run-20260911_201623-jo0atf8r/) | [sse7hmwg](../wandb/run-20260911_202159-sse7hmwg/) |
|  |  | 0.1 | [w96njh22](../wandb/run-20260911_152001-w96njh22/) | [n3bvumaj](../wandb/run-20260911_201623-n3bvumaj/) | [rs81lrpi](../wandb/run-20260911_220947-rs81lrpi/) |
|  |  | 1 | [t4qxma0g](../wandb/run-20260911_170108-t4qxma0g/) | [6jinerk7](../wandb/run-20260911_201621-6jinerk7/) | [qa2aiue0](../wandb/run-20260911_232804-qa2aiue0/) |
|  | `buoyancy_cavity_flow` | 0 | [u20b6v2g](../wandb/run-20260913_015653-u20b6v2g/) | [g6e661vm](../wandb/run-20260913_020049-g6e661vm/) | [zc2993lc](../wandb/run-20260913_020449-zc2993lc/) |
|  |  | 0.001 | [7pc8agtw](../wandb/run-20260911_145110-7pc8agtw/) | [gryumiyo](../wandb/run-20260912_152422-gryumiyo/) | [cn0fnrwc](../wandb/run-20260912_231058-cn0fnrwc/) |
|  |  | 0.01 | [7wgfs8qe](../wandb/run-20260911_150325-7wgfs8qe/) | [8ajptlr6](../wandb/run-20260912_160551-8ajptlr6/) | [72kod1l1](../wandb/run-20260912_235355-72kod1l1/) |
|  |  | 0.1 | [46irk5n0](../wandb/run-20260911_152108-46irk5n0/) | [j93405lt](../wandb/run-20260912_164144-j93405lt/) | [ijbxm5o4](../wandb/run-20260913_000258-ijbxm5o4/) |
|  |  | 1 | [1pu0pgz5](../wandb/run-20260911_194543-1pu0pgz5/) | [m3sgoeq3](../wandb/run-20260912_214711-m3sgoeq3/) | [9ajf4q7j](../wandb/run-20260913_002801-9ajf4q7j/) |
|  | `taylor_green` | 0 | [pab42ujk](../wandb/run-20260911_234645-pab42ujk/) | [ex3985tz](../wandb/run-20260912_000821-ex3985tz/) | [pj57jrk6](../wandb/run-20260912_002145-pj57jrk6/) |
|  |  | 0.001 | [htalvo6t](../wandb/run-20260911_145140-htalvo6t/) | [cs8wu78w](../wandb/run-20260911_182203-cs8wu78w/) | [rjwakiom](../wandb/run-20260911_201618-rjwakiom/) |
|  |  | 0.01 | [pqxktgp0](../wandb/run-20260911_150525-pqxktgp0/) | [pqk0bh1n](../wandb/run-20260911_201623-pqk0bh1n/) | [8fzbkism](../wandb/run-20260911_202332-8fzbkism/) |
|  |  | 0.1 | [cerc1d4p](../wandb/run-20260911_152059-cerc1d4p/) | [ikn98daa](../wandb/run-20260911_201623-ikn98daa/) | [swo0z9zi](../wandb/run-20260911_221420-swo0z9zi/) |
|  |  | 1 | [28p66xrs](../wandb/run-20260911_170913-28p66xrs/) | [oqunmwpw](../wandb/run-20260911_201623-oqunmwpw/) | [x8zxzf76](../wandb/run-20260911_232802-x8zxzf76/) |
|  | `taylor_green_coeffs` | 0 | [a9ooy2wd](../wandb/run-20260911_234947-a9ooy2wd/) | [lo4au5wi](../wandb/run-20260912_000821-lo4au5wi/) | [9g4xmct5](../wandb/run-20260912_002554-9g4xmct5/) |
|  |  | 0.001 | [624dqz15](../wandb/run-20260911_145411-624dqz15/) | [b8dw35r3](../wandb/run-20260911_183621-b8dw35r3/) | [3uv3pdh9](../wandb/run-20260911_201618-3uv3pdh9/) |
|  |  | 0.01 | [6u5cwuv8](../wandb/run-20260911_150658-6u5cwuv8/) | [fxclt41s](../wandb/run-20260911_201623-fxclt41s/) | [49ruzhi5](../wandb/run-20260911_202332-49ruzhi5/) |
|  |  | 0.1 | [4frnyp99](../wandb/run-20260911_152059-4frnyp99/) | [itdr9bd6](../wandb/run-20260911_201623-itdr9bd6/) | [sjpjs0ix](../wandb/run-20260911_221801-sjpjs0ix/) |
|  |  | 1 | [0xkxhigv](../wandb/run-20260911_175128-0xkxhigv/) | [eye1c47x](../wandb/run-20260911_201623-eye1c47x/) | [1i4wpvq5](../wandb/run-20260911_232803-1i4wpvq5/) |
|  | `taylor_green_spacetime` | 0 | [tkxk8455](../wandb/run-20260912_000135-tkxk8455/) | [2mjpezul](../wandb/run-20260912_001026-2mjpezul/) | [v9m0rnyn](../wandb/run-20260912_002554-v9m0rnyn/) |
|  |  | 0.001 | [hcp20npc](../wandb/run-20260911_145549-hcp20npc/) | [wu95knfa](../wandb/run-20260911_190909-wu95knfa/) | [19n79ktf](../wandb/run-20260911_201623-19n79ktf/) |
|  |  | 0.01 | [0amwh0rl](../wandb/run-20260911_150912-0amwh0rl/) | [tjnlb9ej](../wandb/run-20260911_201618-tjnlb9ej/) | [wa1p27rs](../wandb/run-20260911_204128-wa1p27rs/) |
|  |  | 0.1 | [9dr2xxwx](../wandb/run-20260911_152059-9dr2xxwx/) | [vn67682j](../wandb/run-20260911_201623-vn67682j/) | [xuhry5ad](../wandb/run-20260911_221801-xuhry5ad/) |
|  |  | 1 | [sjc14d3b](../wandb/run-20260911_175611-sjc14d3b/) | [0asmxckc](../wandb/run-20260911_201623-0asmxckc/) | [vn412rw7](../wandb/run-20260911_232803-vn412rw7/) |
|  | `taylor_green_spacetime_coeffs` | 0 | [r7stmp01](../wandb/run-20260911_235558-r7stmp01/) | [n0yrjtzc](../wandb/run-20260912_001025-n0yrjtzc/) | [obknm49p](../wandb/run-20260912_002554-obknm49p/) |
|  |  | 0.001 | [6ehtv0cv](../wandb/run-20260911_145541-6ehtv0cv/) | [8ca714uq](../wandb/run-20260911_192409-8ca714uq/) | [fkl16fnr](../wandb/run-20260911_201623-fkl16fnr/) |
|  |  | 0.01 | [5ucid0l6](../wandb/run-20260911_150904-5ucid0l6/) | [h5cxl7p9](../wandb/run-20260911_201618-h5cxl7p9/) | [gcrrdd4y](../wandb/run-20260911_204432-gcrrdd4y/) |
|  |  | 0.1 | [q106r6bj](../wandb/run-20260911_152204-q106r6bj/) | [5e8uurf8](../wandb/run-20260911_201623-5e8uurf8/) | [malpt36k](../wandb/run-20260911_222437-malpt36k/) |
|  |  | 1 | [x9z79vx4](../wandb/run-20260911_175611-x9z79vx4/) | [pc0x5fw6](../wandb/run-20260911_201623-pc0x5fw6/) | [8vshmjid](../wandb/run-20260911_233305-8vshmjid/) |
|  | `merge_vortices_easier` | 0 | [aao4cuai](../wandb/run-20260912_000308-aao4cuai/) | [n6x347xd](../wandb/run-20260912_001158-n6x347xd/) | [y5oofven](../wandb/run-20260912_002656-y5oofven/) |
|  |  | 0.001 | [uea8ryex](../wandb/run-20260911_145620-uea8ryex/) | [g7jv0o3m](../wandb/run-20260911_194411-g7jv0o3m/) | [60kwn7o3](../wandb/run-20260911_201623-60kwn7o3/) |
|  |  | 0.01 | [b0l82o2r](../wandb/run-20260911_151038-b0l82o2r/) | [gviy8e31](../wandb/run-20260911_201618-gviy8e31/) | [sg2vjt18](../wandb/run-20260911_213320-sg2vjt18/) |
|  |  | 0.1 | [azwf5mek](../wandb/run-20260911_152509-azwf5mek/) | [soxmmkbw](../wandb/run-20260911_201623-soxmmkbw/) | [v1hl88bn](../wandb/run-20260911_222437-v1hl88bn/) |
|  |  | 1 | [12ucd5hg](../wandb/run-20260911_175741-12ucd5hg/) | [okpir40o](../wandb/run-20260911_201623-okpir40o/) | [iyqc0mv4](../wandb/run-20260911_233400-iyqc0mv4/) |
|  | `species_transport` | 0 | [0qmutkrp](../wandb/run-20260912_000308-0qmutkrp/) | [lys1koon](../wandb/run-20260912_001330-lys1koon/) | [zsqmf0d4](../wandb/run-20260912_002656-zsqmf0d4/) |
|  |  | 0.001 | [3wpvsn9k](../wandb/run-20260911_145753-3wpvsn9k/) | [3lqa62k7](../wandb/run-20260911_194543-3lqa62k7/) | [03xbpq7a](../wandb/run-20260911_201623-03xbpq7a/) |
|  |  | 0.01 | [jz82d80c](../wandb/run-20260911_151636-jz82d80c/) | [qb32n87l](../wandb/run-20260911_201618-qb32n87l/) | [baflxyd7](../wandb/run-20260911_213954-baflxyd7/) |
|  |  | 0.1 | [qsmmkez3](../wandb/run-20260911_153117-qsmmkez3/) | [0w4gmjdw](../wandb/run-20260911_201621-0w4gmjdw/) | [ozvzbuds](../wandb/run-20260911_223118-ozvzbuds/) |
|  |  | 1 | [w6iec4ej](../wandb/run-20260911_175911-w6iec4ej/) | [ea81gu7b](../wandb/run-20260911_201623-ea81gu7b/) | [g6tkq5hq](../wandb/run-20260911_233400-g6tkq5hq/) |
| Transolver | `backward_facing_step` | 0 | [5cbzvmiy](../wandb/run-20260911_234445-5cbzvmiy/) | [s2f6ow9m](../wandb/run-20260912_201720-s2f6ow9m/) | [hs1eqr8i](../wandb/run-20260912_230611-hs1eqr8i/) |
|  |  | 0.001 | [bjgsdnqe](../wandb/run-20260911_145104-bjgsdnqe/) | [s9blvm9i](../wandb/run-20260911_205751-s9blvm9i/) | [q1k32a9r](../wandb/run-20260911_214431-q1k32a9r/) |
|  |  | 0.01 | [dq0ug68q](../wandb/run-20260911_150330-dq0ug68q/) | [zkyxkcds](../wandb/run-20260911_210809-zkyxkcds/) | [5hqycsca](../wandb/run-20260911_225949-5hqycsca/) |
|  |  | 0.1 | [msoreepr](../wandb/run-20260911_152010-msoreepr/) | [kcxk4b8x](../wandb/run-20260911_213025-kcxk4b8x/) | [5tg0x07a](../wandb/run-20260911_230255-5tg0x07a/) |
|  |  | 1 | [djnzes7c](../wandb/run-20260911_185045-djnzes7c/) | [ez1yej4g](../wandb/run-20260911_213823-ez1yej4g/) | [rj3lvs5r](../wandb/run-20260911_232803-rj3lvs5r/) |
|  | `flow_cylinder_laminar` | 0 | [ikuf3pcl](../wandb/run-20260911_234320-ikuf3pcl/) | [1nkh0rur](../wandb/run-20260912_155617-1nkh0rur/) | [xp97nnpl](../wandb/run-20260912_223402-xp97nnpl/) |
|  |  | 0.001 | [i39xsi90](../wandb/run-20260911_144940-i39xsi90/) | [q1qr7bv0](../wandb/run-20260911_194934-q1qr7bv0/) | [oouiqcpj](../wandb/run-20260911_214309-oouiqcpj/) |
|  |  | 0.01 | [u51axn0p](../wandb/run-20260911_150018-u51axn0p/) | [idt8fit6](../wandb/run-20260911_210431-idt8fit6/) | [93x4f2z8](../wandb/run-20260911_224801-93x4f2z8/) |
|  |  | 0.1 | [hj59wiwm](../wandb/run-20260911_151903-hj59wiwm/) | [04ujce2f](../wandb/run-20260911_212539-04ujce2f/) | [bo64f588](../wandb/run-20260911_230255-bo64f588/) |
|  |  | 1 | [gpruhtf7](../wandb/run-20260911_153528-gpruhtf7/) | [jt57afgb](../wandb/run-20260911_213320-jt57afgb/) | [si8bo278](../wandb/run-20260911_232518-si8bo278/) |
|  | `flow_cylinder_shedding` | 0 | [2qkm11s9](../wandb/run-20260911_234446-2qkm11s9/) | [54136p96](../wandb/run-20260912_155815-54136p96/) | [y22ymvh5](../wandb/run-20260912_224101-y22ymvh5/) |
|  |  | 0.001 | [gdank786](../wandb/run-20260911_144940-gdank786/) | [ot8sz7h2](../wandb/run-20260911_195131-ot8sz7h2/) | [8etgosm6](../wandb/run-20260911_214309-8etgosm6/) |
|  |  | 0.01 | [dvm7tv86](../wandb/run-20260911_150056-dvm7tv86/) | [ki664fjv](../wandb/run-20260911_210431-ki664fjv/) | [nus5xyph](../wandb/run-20260911_225949-nus5xyph/) |
|  |  | 0.1 | [e9rgd7yf](../wandb/run-20260911_152002-e9rgd7yf/) | [5gjyyjai](../wandb/run-20260911_212539-5gjyyjai/) | [0gp726sz](../wandb/run-20260911_230255-0gp726sz/) |
|  |  | 1 | [ld914d5w](../wandb/run-20260911_164912-ld914d5w/) | [a42nbibb](../wandb/run-20260911_213522-a42nbibb/) | [sf959c8d](../wandb/run-20260911_232518-sf959c8d/) |
|  | `lid_cavity_flow` | 0 | [do40p8xq](../wandb/run-20260911_234446-do40p8xq/) | [5t2hmcp8](../wandb/run-20260912_160500-5t2hmcp8/) | [kklefqk4](../wandb/run-20260912_230016-kklefqk4/) |
|  |  | 0.001 | [z0fdlgvh](../wandb/run-20260911_145112-z0fdlgvh/) | [9nj6bc1k](../wandb/run-20260911_201954-9nj6bc1k/) | [3tb4qndw](../wandb/run-20260911_214309-3tb4qndw/) |
|  |  | 0.01 | [h87a15yl](../wandb/run-20260911_150150-h87a15yl/) | [0zf9xuyz](../wandb/run-20260911_210809-0zf9xuyz/) | [i3hykrq4](../wandb/run-20260911_225950-i3hykrq4/) |
|  |  | 0.1 | [ejr317wm](../wandb/run-20260911_152003-ejr317wm/) | [y49l16xp](../wandb/run-20260911_213025-y49l16xp/) | [emh50bl0](../wandb/run-20260911_230255-emh50bl0/) |
|  |  | 1 | [6u0m5miy](../wandb/run-20260911_180756-6u0m5miy/) | [ronkkkqy](../wandb/run-20260911_213654-ronkkkqy/) | [zj5m7uzt](../wandb/run-20260911_232807-zj5m7uzt/) |
|  | `buoyancy_cavity_flow` | 0 | [dr5cvy6p](../wandb/run-20260911_234647-dr5cvy6p/) | [x41prbl4](../wandb/run-20260912_202811-x41prbl4/) | [3sjhbspd](../wandb/run-20260912_234901-3sjhbspd/) |
|  |  | 0.001 | [uxnfptez](../wandb/run-20260911_145138-uxnfptez/) | [nqw1purh](../wandb/run-20260911_205922-nqw1purh/) | [glyn2av7](../wandb/run-20260911_214431-glyn2av7/) |
|  |  | 0.01 | [6vsk03ou](../wandb/run-20260911_150355-6vsk03ou/) | [o59ogu5c](../wandb/run-20260911_210809-o59ogu5c/) | [5np7yjqw](../wandb/run-20260911_230252-5np7yjqw/) |
|  |  | 0.1 | [33b8ikmj](../wandb/run-20260911_152100-33b8ikmj/) | [fd58bhvi](../wandb/run-20260911_213025-fd58bhvi/) | [wf85okv0](../wandb/run-20260911_230421-wf85okv0/) |
|  |  | 1 | [2bbrtx8e](../wandb/run-20260911_185212-2bbrtx8e/) | [y55gx5fm](../wandb/run-20260911_213823-y55gx5fm/) | [3mjdujqu](../wandb/run-20260911_232803-3mjdujqu/) |
|  | `taylor_green` | 0 | [c4wbtisw](../wandb/run-20260911_234646-c4wbtisw/) | [cq0ud9qu](../wandb/run-20260912_214511-cq0ud9qu/) | [v8xp1nfb](../wandb/run-20260912_235356-v8xp1nfb/) |
|  |  | 0.001 | [ougpjjp6](../wandb/run-20260911_145314-ougpjjp6/) | [q0k8266t](../wandb/run-20260911_210108-q0k8266t/) | [10bkljil](../wandb/run-20260911_214633-10bkljil/) |
|  |  | 0.01 | [ltxj4xkw](../wandb/run-20260911_150526-ltxj4xkw/) | [0ihoxrs3](../wandb/run-20260911_211016-0ihoxrs3/) | [1judymnw](../wandb/run-20260911_230253-1judymnw/) |
|  |  | 0.1 | [9gr3e052](../wandb/run-20260911_152109-9gr3e052/) | [990rpp1j](../wandb/run-20260911_213021-990rpp1j/) | [3ac14lkv](../wandb/run-20260911_230421-3ac14lkv/) |
|  |  | 1 | [t6imxk42](../wandb/run-20260911_185255-t6imxk42/) | [m7py8977](../wandb/run-20260911_213823-m7py8977/) | [0x97sc6y](../wandb/run-20260911_232803-0x97sc6y/) |
|  | `taylor_green_coeffs` | 0 | [gm655o1s](../wandb/run-20260911_235301-gm655o1s/) | [djye9c5u](../wandb/run-20260912_215522-djye9c5u/) | [bqbv4b1j](../wandb/run-20260912_235701-bqbv4b1j/) |
|  |  | 0.001 | [6a6v3v13](../wandb/run-20260911_145543-6a6v3v13/) | [88qwt8wj](../wandb/run-20260911_210108-88qwt8wj/) | [kwo39eiq](../wandb/run-20260911_214633-kwo39eiq/) |
|  |  | 0.01 | [d8953w0c](../wandb/run-20260911_150807-d8953w0c/) | [ysg3g12f](../wandb/run-20260911_211016-ysg3g12f/) | [1lzm9126](../wandb/run-20260911_230253-1lzm9126/) |
|  |  | 0.1 | [7dwr41si](../wandb/run-20260911_152100-7dwr41si/) | [5ljkauji](../wandb/run-20260911_213022-5ljkauji/) | [4ehiitsv](../wandb/run-20260911_230554-4ehiitsv/) |
|  |  | 1 | [wg5zpvn1](../wandb/run-20260911_191755-wg5zpvn1/) | [c857ct0n](../wandb/run-20260911_213957-c857ct0n/) | [u6e675x6](../wandb/run-20260911_232803-u6e675x6/) |
|  | `taylor_green_spacetime` | 0 | [staf5qhm](../wandb/run-20260912_001027-staf5qhm/) | [qq1lpgbs](../wandb/run-20260912_220304-qq1lpgbs/) | [sihyu01k](../wandb/run-20260912_235703-sihyu01k/) |
|  |  | 0.001 | [dkg38fuh](../wandb/run-20260911_145543-dkg38fuh/) | [fvjsrgqj](../wandb/run-20260911_210308-fvjsrgqj/) | [8jiw3pw8](../wandb/run-20260911_214633-8jiw3pw8/) |
|  |  | 0.01 | [zs3c7igz](../wandb/run-20260911_150916-zs3c7igz/) | [tht0cgh6](../wandb/run-20260911_211016-tht0cgh6/) | [vbjcjz4s](../wandb/run-20260911_230253-vbjcjz4s/) |
|  |  | 0.1 | [i7q7f69y](../wandb/run-20260911_152146-i7q7f69y/) | [r34jboxe](../wandb/run-20260911_213151-r34jboxe/) | [alzppz3f](../wandb/run-20260911_232043-alzppz3f/) |
|  |  | 1 | [uuwsh5cc](../wandb/run-20260911_192039-uuwsh5cc/) | [8vm9tacs](../wandb/run-20260911_213957-8vm9tacs/) | [r4mtlj3y](../wandb/run-20260911_233307-r4mtlj3y/) |
|  | `taylor_green_spacetime_coeffs` | 0 | [904fxoye](../wandb/run-20260912_003303-904fxoye/) | [gvxvbpxi](../wandb/run-20260912_222258-gvxvbpxi/) | [f9zo85x6](../wandb/run-20260912_235703-f9zo85x6/) |
|  |  | 0.001 | [fg75wg1y](../wandb/run-20260911_145552-fg75wg1y/) | [j28pu92e](../wandb/run-20260911_210308-j28pu92e/) | [kv9drupk](../wandb/run-20260911_223613-kv9drupk/) |
|  |  | 0.01 | [xnmltmr1](../wandb/run-20260911_151038-xnmltmr1/) | [aoq4q5dy](../wandb/run-20260911_211016-aoq4q5dy/) | [fd0j2vls](../wandb/run-20260911_230257-fd0j2vls/) |
|  |  | 0.1 | [8hyuhx1e](../wandb/run-20260911_152341-8hyuhx1e/) | [3bavbenw](../wandb/run-20260911_213151-3bavbenw/) | [oz9giiwk](../wandb/run-20260911_232043-oz9giiwk/) |
|  |  | 1 | [9muxpbjb](../wandb/run-20260911_193121-9muxpbjb/) | [9om4nz8f](../wandb/run-20260911_213957-9om4nz8f/) | [bmasdgge](../wandb/run-20260911_233709-bmasdgge/) |
|  | `merge_vortices_easier` | 0 | [spxlfy91](../wandb/run-20260912_120933-spxlfy91/) | [d5mrykhs](../wandb/run-20260912_195220-d5mrykhs/) | [qv9udj8i](../wandb/run-20260913_001305-qv9udj8i/) |
|  |  | 0.001 | [t8ps8yk8](../wandb/run-20260911_145614-t8ps8yk8/) | [1hun1hvn](../wandb/run-20260911_210308-1hun1hvn/) | [xa9wuku9](../wandb/run-20260911_223746-xa9wuku9/) |
|  |  | 0.01 | [cxpqw5p3](../wandb/run-20260911_151639-cxpqw5p3/) | [d2yg2531](../wandb/run-20260911_211014-d2yg2531/) | [khlpo0mm](../wandb/run-20260911_230257-khlpo0mm/) |
|  |  | 0.1 | [mvky47nl](../wandb/run-20260911_152510-mvky47nl/) | [d45hn6he](../wandb/run-20260911_213321-d45hn6he/) | [vrhgswty](../wandb/run-20260911_232043-vrhgswty/) |
|  |  | 1 | [6p24tu6w](../wandb/run-20260911_193326-6p24tu6w/) | [239bqm0q](../wandb/run-20260911_213955-239bqm0q/) | [2e6jvbot](../wandb/run-20260911_234315-2e6jvbot/) |
|  | `species_transport` | 0 | [m2faa885](../wandb/run-20260912_133642-m2faa885/) | [qt3xwkcr](../wandb/run-20260912_223058-qt3xwkcr/) | [bs3wpnma](../wandb/run-20260913_002901-bs3wpnma/) |
|  |  | 0.001 | [wbkq3f72](../wandb/run-20260911_145746-wbkq3f72/) | [idtu5hrg](../wandb/run-20260911_210431-idtu5hrg/) | [wfmshh1u](../wandb/run-20260911_224454-wfmshh1u/) |
|  |  | 0.01 | [a2hjiy0y](../wandb/run-20260911_151821-a2hjiy0y/) | [6o2722r4](../wandb/run-20260911_212539-6o2722r4/) | [hef3epj4](../wandb/run-20260911_230257-hef3epj4/) |
|  |  | 0.1 | [glp7wjjt](../wandb/run-20260911_153322-glp7wjjt/) | [nyq1fl1j](../wandb/run-20260911_213321-nyq1fl1j/) | [ybnvm7gt](../wandb/run-20260911_232521-ybnvm7gt/) |
|  |  | 1 | [274fgfz6](../wandb/run-20260911_193908-274fgfz6/) | [glu0sz46](../wandb/run-20260911_214309-glu0sz46/) | [ba67nb31](../wandb/run-20260911_234312-ba67nb31/) |
