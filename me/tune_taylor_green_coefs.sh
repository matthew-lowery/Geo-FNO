#!/bin/bash


sp() {
    local pycmd="$1"
    local hr="${2:-8}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpuA100x4,gpuA100x8
#SBATCH --account=bfel-delta-gpu
#SBATCH --job-name=myjob
#SBATCH --time=10:00:00
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --output=./out/%x_%A.out
#SBATCH --error=./err/%x_%A.err

module purge
export PATH=/u/mlowery/.conda/envs/gnot/bin:\$PATH
cd /u/mlowery/Geo-FNO/me/
$pycmd
EOF
}
### tune for tgtc and tgc
dir='/projects/bfel/mlowery/geo-fno-new'
for res1d in 10 15 20 25 30; do
sp "python3 ramansh_3d.py --dir=$dir --dataset='taylor_green_time_coeffs' --wandb --ntrain=500 --npoints=$npoints --dataset=$dataset --norm-grid --res1d=$res1d --width=32 --modes=$((res1d/2))"
done

for width in 64 128; do
sp "python3 ramansh_3d.py --dir=$dir --dataset='taylor_green_time_coeffs' --wandb --ntrain=500 --npoints=$npoints --dataset=$dataset --norm-grid --width=$width --res1d=15 --modes=7"
done

for res1d in 20 30 40 50 60; do
sp "python3 ramansh_2d_diff_grids.py --dir=$dir --dataset='taylor_green_coeffs' --wandb --ntrain=500 --npoints=$npoints --dataset=$dataset --norm-grid --res1d=$res1d --width=32 --modes=$((res1d/2))"
done

for width in 64 128; do
sp "python3 ramansh_2d_diff_grids.py --dir=$dir --dataset='taylor_green_coeffs' --wandb --ntrain=500 --npoints=$npoints --dataset=$dataset --norm-grid --width=$width"
done

