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
#SBATCH --account=bfbk-delta-gpu
#SBATCH --job-name=myjob
#SBATCH --time=20:00:00
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

#### need to specify how long each of these might take....
dataset='backward_facing_step'
for seed in 1 2 3; do
for ntrain in 10000; do
for npoints in all; do
sp "python3 ramansh_2d.py --wandb --calc-div --save --seed=$seed --ntrain=$ntrain --npoints=$npoints --dataset=$dataset --norm-grid --res1d=40 --width=64 --modes=12"
done
done
done

dataset='buoyancy_cavity_flow'
for seed in 1 2 3; do
for ntrain in 10000; do
for npoints in all; do
sp "python3 ramansh_2d.py --wandb --calc-div --save --seed=$seed --ntrain=$ntrain --npoints=$npoints --dataset=$dataset --norm-grid --res1d=40 --width=64 --modes=20"
done
done
done

#dataset='flow_cylinder_laminar'
#for seed in 1 2 3; do
#for ntrain in 7000 10000; do
#for npoints in all; do
#sp "python3 ramansh_2d.py --wandb --calc-div --save --seed=$seed --ntrain=$ntrain --npoints=$npoints --dataset=$dataset --norm-grid --res1d=50 --width=128 --modes=24"
#done
#done
#done

dataset='flow_cylinder_shedding'
for seed in 1 2 3; do
for ntrain in 10000; do
for npoints in all; do
sp "python3 ramansh_2d.py --wandb --calc-div --save --seed=$seed --ntrain=$ntrain --npoints=$npoints --dataset=$dataset --norm-grid --res1d=60 --width=64 --modes=28"
done
done
done

dataset='lid_cavity_flow'
for seed in 1 2 3; do
for ntrain in 10000; do
for npoints in all; do
sp "python3 ramansh_2d.py --wandb --calc-div --save --seed=$seed --ntrain=$ntrain --npoints=$npoints --dataset=$dataset --norm-grid --res1d=40 --width=64 --modes=20"
done
done
done

dataset='merge_vortices'
for seed in 1 2 3; do
for ntrain in 10000; do
for npoints in all; do
sp "python3 ramansh_2d.py --wandb --calc-div --save --seed=$seed --ntrain=$ntrain --npoints=$npoints --dataset=$dataset --norm-grid --res1d=60 --width=128 --modes=12"
done
done
done

dataset='taylor_green_exact'
for seed in 1 2 3; do
for ntrain in 10000; do
for npoints in all; do
sp "python3 ramansh_2d.py --wandb --calc-div --save --seed=$seed --ntrain=$ntrain --npoints=$npoints --dataset=$dataset --norm-grid --res1d=50 --width=64 --modes=20"
done
done
done

dataset='taylor_green_exact'
for seed in 1 2 3; do
for ntrain in 10000; do
for npoints in all; do
sp "python3 ramansh_2d.py --wandb --calc-div --save --seed=$seed --ntrain=$ntrain --npoints=$npoints --dataset=$dataset --norm-grid --res1d=50 --width=64 --modes=20"
done
done
done

