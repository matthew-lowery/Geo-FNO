# W&B results

Standard deviations use `np.std()` with the default population convention. Test divergence is interior-only; each entry is `max / median`. `Total train time (s)` is the W&B `total_train_time` summary field; it is not the per-epoch `train_time` field. `Order` is the RBF-FD polynomial order $p$. Order-5 rows are averaged over seeds; order-2 divergence-loss rows are single seed-1 runs, while order-2 no-div rows average three repeated seed-1 runs.

| Problem | Run | Order | λ | Total train time (s) | Test loss | Interior test div (max / median) | OOD loss |
|---|---|---:|---:|---:|---:|---:|---:|
| backward_facing_step | div | 5 | 0.1 | 217.9 ± 3.03 | 1.064e-4 ± 9.36e-6 | 1.958 ± 0.00294 / 4.387e-4 ± 7.73e-6 | 1.059 ± 0.156 |
| backward_facing_step | no-div | 5 | 0 | 195.44 ± 0.851 | 9.949e-5 ± 1.73e-5 | 1.967 ± 0.00248 / 4.494e-4 ± 1.12e-5 | 1.013 ± 0.194 |
| backward_facing_step | div | 2 | 0.1 | 217.27 | 1.098e-4 | 1.457 / 4.025e-4 | 0.768 |
| backward_facing_step | div | 2 | 0.5 | 214.18 | 1.593e-4 | 1.398 / 4.318e-4 | 1.077 |
| backward_facing_step | div | 2 | 1 | 214.39 | 1.360e-3 | 1.035 / 7.074e-4 | 2.112 |
| backward_facing_step | no-div | 2 | 0 | 194.63 ± 1.44 | 1.239e-4 | 1.458 / 4.621e-4 | 0.866 |
| buoyancy_cavity_flow | div | 5 | 0.1 | 21664 ± 19.2 | 1.778e-3 ± 5.30e-6 | 2.333 ± 0.00881 / 0.01072 ± 1.07e-4 | — |
| buoyancy_cavity_flow | no-div | 5 | 0 | 21182 ± 19.3 | 1.421e-4 ± 6.24e-6 | 3.037 ± 0.00037 / 0.01977 ± 7.05e-5 | — |
| buoyancy_cavity_flow | div | 2 | 0.1 | 21613.33 | 2.159e-3 | 2.487 / 9.030e-3 | — |
| buoyancy_cavity_flow | div | 2 | 0.5 | 21578.73 | 4.791e-3 | 0.651 / 4.112e-3 | — |
| buoyancy_cavity_flow | div | 2 | 1 | 21576.74 | 5.487e-3 | 0.363 / 2.900e-3 | — |
| buoyancy_cavity_flow | no-div | 2 | 0 | 21158.15 ± 20.96 | 1.440e-4 | 5.006 / 0.01999 | — |
| flow_cylinder_laminar | div | 5 | 0.1 | 180.5 ± 0.755 | 1.960e-4 ± 4.70e-5 | 0.7021 ± 0.00070 / 1.028e-4 ± 8.65e-6 | 2.622 ± 2.71 |
| flow_cylinder_laminar | no-div | 5 | 0 | 175.81 ± 0.585 | 1.618e-4 ± 4.77e-5 | 0.7033 ± 0.00041 / 9.457e-5 ± 2.19e-5 | 0.645 ± 0.306 |
| flow_cylinder_laminar | div | 2 | 0.1 | 180.99 | 1.747e-4 | 0.878 / 1.060e-4 | 0.310 |
| flow_cylinder_laminar | div | 2 | 0.5 | 179.99 | 1.611e-4 | 0.876 / 1.125e-4 | 0.337 |
| flow_cylinder_laminar | div | 2 | 1 | 179.84 | 1.626e-4 | 0.872 / 1.231e-4 | 1.809 |
| flow_cylinder_laminar | no-div | 2 | 0 | 175.22 ± 0.17 | 1.581e-4 | 0.879 / 1.264e-4 | 0.554 |
| flow_cylinder_shedding | div | 5 | 0.001 | 11895 ± 10.8 | 4.255e-5 ± 1.08e-5 | 4.450 ± 0.00021 / 0.001433 ± 3.17e-6 | 0.662 ± 0.178 |
| flow_cylinder_shedding | no-div | 5 | 0 | 11414 ± 10.3 | 4.783e-5 ± 1.19e-5 | 4.450 ± 0.00057 / 0.001450 ± 1.41e-5 | 0.626 ± 0.325 |
| flow_cylinder_shedding | div | 2 | 0.1 | 11850.68 | 1.684e-4 | 5.172 / 1.563e-3 | 11.034 |
| flow_cylinder_shedding | div | 2 | 0.5 | 11863.24 | 5.882e-3 | 1.043 / 1.373e-3 | 429.156 |
| flow_cylinder_shedding | div | 2 | 1 | 11831.23 | 8.084e-3 | 0.838 / 1.211e-3 | 8.434 |
| flow_cylinder_shedding | no-div | 2 | 0 | 11397.83 ± 14.17 | 3.323e-5 | 5.659 / 1.534e-3 | 1.069 |
| forced_turb | no-div | 5 | 0 | 44815 ± 101 | 1.824e-4 ± 8.67e-6 | 0.4156 ± 0.00171 / 0.02749 ± 2.70e-6 | 0.0483 ± 0.0158 |
| lid_cavity_flow | div | 5 | 0.001 | 6741.3 ± 6.02 | 1.719e-4 ± 1.56e-5 | 21.47 ± 0.0098 / 0.5715 ± 2.08e-4 | 0.574 ± 0.111 |
| lid_cavity_flow | no-div | 5 | 0 | 6256.5 ± 7.2 | 1.509e-4 ± 2.04e-5 | 21.58 ± 0.0047 / 0.5734 ± 3.14e-4 | 0.590 ± 0.098 |
| lid_cavity_flow | div | 2 | 0.1 | 6721.26 | 1.185e-2 | 2.979 / 0.03542 | 65.526 |
| lid_cavity_flow | div | 2 | 0.5 | 6730.90 | 1.419e-2 | 0.620 / 0.01068 | 2.738 |
| lid_cavity_flow | div | 2 | 1 | 6702.51 | 1.500e-2 | 0.304 / 6.591e-3 | 0.435 |
| lid_cavity_flow | no-div | 2 | 0 | 6259.92 ± 5.00 | 1.297e-4 | 21.721 / 0.57496 | 0.591 |
| merge_vortices_easier | no-div | 5 | 0 | 303.46 ± 1.07 | 6.247e-4 ± 1.66e-4 | 1379 ± 477 / 0.03359 ± 2.24e-4 | 2413 ± 1310 |

## Issues

- All three `forced_turb` divergence-loss runs failed.
- No finished cached runs for Taylor–Green, Taylor–Green coefficients, either spacetime variant, or species transport.
- No buoyancy OOD loss was logged.
- `merge_vortices_easier` has extremely large maximum divergence and OOD error.
- `lid_cavity_flow` has a large maximum divergence.
- Laminar-cylinder OOD performance is highly variable, especially seed 3.
