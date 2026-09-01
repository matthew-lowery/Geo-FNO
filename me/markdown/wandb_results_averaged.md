# W&B results averaged over seeds

Standard deviations use `np.std()` with the default population convention. Test divergence is interior-only; each entry is `max / median`. `Total train time (s)` is the W&B `total_train_time` summary field; it is not the per-epoch `train_time` field.

| Problem | Run | λ | Total train time (s) | Test loss | Interior test div (max / median) | OOD loss |
|---|---|---:|---:|---:|---:|---:|
| backward_facing_step | div | 0.1 | 217.9 ± 3.03 | 1.064e-4 ± 9.36e-6 | 1.958 ± 0.00294 / 4.387e-4 ± 7.73e-6 | 1.059 ± 0.156 |
| backward_facing_step | no-div | 0 | 195.44 ± 0.851 | 9.949e-5 ± 1.73e-5 | 1.967 ± 0.00248 / 4.494e-4 ± 1.12e-5 | 1.013 ± 0.194 |
| buoyancy_cavity_flow | div | 0.1 | 21664 ± 19.2 | 1.778e-3 ± 5.30e-6 | 2.333 ± 0.00881 / 0.01072 ± 1.07e-4 | — |
| buoyancy_cavity_flow | no-div | 0 | 21182 ± 19.3 | 1.421e-4 ± 6.24e-6 | 3.037 ± 0.00037 / 0.01977 ± 7.05e-5 | — |
| flow_cylinder_laminar | div | 0.1 | 180.5 ± 0.755 | 1.960e-4 ± 4.70e-5 | 0.7021 ± 0.00070 / 1.028e-4 ± 8.65e-6 | 2.622 ± 2.71 |
| flow_cylinder_laminar | no-div | 0 | 175.81 ± 0.585 | 1.618e-4 ± 4.77e-5 | 0.7033 ± 0.00041 / 9.457e-5 ± 2.19e-5 | 0.645 ± 0.306 |
| flow_cylinder_shedding | div | 0.001 | 11895 ± 10.8 | 4.255e-5 ± 1.08e-5 | 4.450 ± 0.00021 / 0.001433 ± 3.17e-6 | 0.662 ± 0.178 |
| flow_cylinder_shedding | no-div | 0 | 11414 ± 10.3 | 4.783e-5 ± 1.19e-5 | 4.450 ± 0.00057 / 0.001450 ± 1.41e-5 | 0.626 ± 0.325 |
| forced_turb | no-div | 0 | 44815 ± 101 | 1.824e-4 ± 8.67e-6 | 0.4156 ± 0.00171 / 0.02749 ± 2.70e-6 | 0.0483 ± 0.0158 |
| lid_cavity_flow | div | 0.001 | 6741.3 ± 6.02 | 1.719e-4 ± 1.56e-5 | 21.47 ± 0.0098 / 0.5715 ± 2.08e-4 | 0.574 ± 0.111 |
| lid_cavity_flow | no-div | 0 | 6256.5 ± 7.2 | 1.509e-4 ± 2.04e-5 | 21.58 ± 0.0047 / 0.5734 ± 3.14e-4 | 0.590 ± 0.098 |
| merge_vortices_easier | no-div | 0 | 303.46 ± 1.07 | 6.247e-4 ± 1.66e-4 | 1379 ± 477 / 0.03359 ± 2.24e-4 | 2413 ± 1310 |

## Issues

- All three `forced_turb` divergence-loss runs failed.
- No finished cached runs for Taylor–Green, Taylor–Green coefficients, either spacetime variant, or species transport.
- No buoyancy OOD loss was logged.
- `merge_vortices_easier` has extremely large maximum divergence and OOD error.
- `lid_cavity_flow` has a large maximum divergence.
- Laminar-cylinder OOD performance is highly variable, especially seed 3.
