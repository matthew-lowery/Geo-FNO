# GPU Conda environment and forced-turbulence recovery

## Create the environment on Delta

From the Delta login node, with Conda available and the repository at `/u/rsharma15/Geo-FNO`:

```bash
cd /u/rsharma15/Geo-FNO/me
conda env create --prefix /u/rsharma15/.conda/envs/operator-benchmarks-gpu -f environment.yml
conda activate /u/rsharma15/.conda/envs/operator-benchmarks-gpu
python -m pip check
wandb login
```

[`environment.yml`](../environment.yml) is the Conda equivalent of a requirements file. It installs Python 3.10.18 and the project's dependencies in a separate environment; Conda installs its `pip` section automatically. [Conda environment documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

The file targets Linux x86-64 NVIDIA GPUs and pins the core versions recorded by the completed Delta runs: PyTorch 2.9.1, torchvision 0.24.1, and CUDA 12.8 wheels. The remaining direct dependencies are pinned too; transitive dependencies are resolved at installation. [Official PyTorch version/CUDA combinations](https://pytorch.org/get-started/previous-versions/#v291).

## Check CUDA on an allocated GPU

Run this from the activated environment. It requests a short GPU allocation; the login node itself need not expose a GPU.

```bash
srun --partition=gpuA100x4 --account=bgkk-delta-gpu --nodes=1 --ntasks=1 --gpus-per-node=1 --cpus-per-task=1 --mem=4G --time=00:05:00 \
  "$CONDA_PREFIX/bin/python" -c 'import torch, scipy, h5py, matplotlib, wandb, timm, einops; assert torch.cuda.is_available(), "CUDA unavailable"; x=torch.ones(4, device="cuda"); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), x.sum().item())'
```

Expected: PyTorch `2.9.1+cu128`, CUDA `12.8`, an A100 device name, and `4.0`. The CUDA libraries come with the Python packages; the GPU node supplies the NVIDIA driver.

## Preview and submit the remaining turbulence runs

```bash
cd /u/rsharma15/Geo-FNO/me
conda activate /u/rsharma15/.conda/envs/operator-benchmarks-gpu
export TRAIN_PYTHON="$CONDA_PREFIX/bin/python"
bash train_forced_recovery_explicit.sh --dry-run
bash train_forced_recovery_explicit.sh
```

[`train_forced_recovery_explicit.sh`](../train_forced_recovery_explicit.sh) contains 16 explicit training commands, ordered shortest to longest. Its default Python is `/u/rsharma15/.conda/envs/operator-benchmarks-gpu/bin/python`, overridable with `TRAIN_PYTHON`. It finds the training code relative to the script's own location. Each GPU job checks CUDA and package imports before training.

| Framework | Training samples | Divergence weight | Seeds | Hours per job |
|---|---:|---:|---|---:|
| Transolver | 10,000 | 0 | 3 | 10 |
| Geo-FNO | 5,000 | 0 | 2, 3 | 24 |
| Geo-FNO | 7,000 | 0 | 1, 2, 3 | 32 |
| Geo-FNO | 10,000 | 0 | 1, 2, 3 | 46 |
| Geo-FNO | 10,000 | 0.001 | 3 | 46 |
| Geo-FNO | 10,000 | 0.01 | 2, 3 | 46 |
| Geo-FNO | 10,000 | 0.1 | 2, 3 | 46 |
| Geo-FNO | 10,000 | 1 | 2, 3 | 46 |

Selection uses the September 23 `wandb_final_final_final` snapshot: 44 of 60 forced-turbulence configurations have finite final metrics. The 16 commands cover the remainder, including four partial attempts. Submission checks your Slurm queue and skips matching job names from this script, `train_recovery_explicit.sh`, and the older turbulence launchers. It stops if the queue cannot be read. This is a fixed snapshot list: completed jobs after that snapshot can be selected again once absent from the queue.

Newly submitted jobs train from the beginning. Settings remain 500 epochs and 7,000 points; Geo-FNO uses batch size 10 and divergence order 3, Transolver batch size 20 and order 4. Baselines require final OOD evaluation; divergence-penalty runs disable OOD. Training objectives and data processing are unchanged.

The hour limits use the measured full-size rates of about 236 seconds per epoch for Geo-FNO and 47 for Transolver. Multiply by 500 epochs and the training-size fraction of 10,000, add 35% headroom plus one hour, then round up. Full-size Geo-FNO training alone took roughly 32.5 hours, hence the 46-hour allocation.

Forced-turbulence data: `/u/rsharma15/pde_ml/code/op_dataset/forced_turb/`, containing `data.mat` and `data_ood.mat`. The script passes the parent `/u/rsharma15/pde_ml/code/op_dataset` as `--data-root`; the loader appends `forced_turb/`. Override that parent with `RAM_DATA_ROOT`. New outputs: `/u/rsharma15/operator-benchmarks/forced-recovery-20260923`, overridable with `RAM_RESULTS_ROOT`. Slurm text logs go to `out/` and `err/` beside the script, normally `/u/rsharma15/Geo-FNO/me/out/` and `/u/rsharma15/Geo-FNO/me/err/`.

Final metrics are written to W&B summary/history, printed as `METRICS` in stdout, and stored beside checkpoints as `.metrics.json`: total train time in seconds, test loss, interior divergence maximum/median, and OOD loss/metric for baselines. A missing or nonfinite required metric makes the trainer fail before reporting successful completion.
