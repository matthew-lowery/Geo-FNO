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
#SBATCH --time=1:00:00
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

# 'backward_facing_step', s=40, width=64, modes=12
# 'buoyancy_cavity_flow', s=40, width=64, modes=20
#  'flow_cylinder_laminar', s=60, width=128, modes=24
#   'flow_cylinder_shedding', s=60, width=64, modes=28
# 'lid_cavity_flow', s=40, width=64, modes=20
# 'merge_vortices', s=60, width=128, modes=12
# 'taylor_green_numerical' s=60, width=64, modes=20
dataset='backward_facing_step'
for seed in 1; do
for ntrain in  100; do 
for npoints in 1000; do
sp "python3 ramansh.py --epochs=5 --wandb --calc-div --save --seed=$seed --ntrain=$ntrain --npoints=$npoints --dataset=$dataset --norm-grid --res1d=40 --width=64 --modes=12"
done
done
done
dataset='buoyancy_cavity_flow'
for seed in 1; do
for ntrain in  100; do 
for npoints in 1000; do
sp "python3 ramansh.py --epochs=5 --wandb --calc-div --save --seed=$seed --ntrain=$ntrain --npoints=$npoints --dataset=$dataset --norm-grid --res1d=40 --width=64 --modes=12"
done
done
done



#for npoints in 100 1000 5000 9566; do
#for npoints in 100 1000 5000 10514; do
#for npoints in 100 1000 5000 7477; do
#for npoints in 100 1000 5000 7359; do
#for npoints in 100 1000 5000 7359; do
#for npoints in 100 1000 5000 7477; do
#for npoints in 100 1000 5000 7000; do
#for npoints in 100 1000 2700; do
#
