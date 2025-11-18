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
#SBATCH --time=4:00:00
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

dataset='taylor_green_time'
for seed in 0; do
for ntrain in 1000; do 
for npoints in 2700; do
for res1d in 16; do
sp "python3 ramansh_3d.py --wandb --calc-div --save --seed=$seed --ntrain=$ntrain --npoints=$npoints --dataset=$dataset --norm-grid --res1d=$res1d --width=64 --modes=$((res1d/2))"
done
done
done
done

dataset='species_transport'
for seed in 0; do
for ntrain in 1000; do 
for npoints in 7000; do
for res1d in 10 16; do
sp "python3 ramansh_3d.py --wandb --calc-div --save --seed=$seed --ntrain=$ntrain --npoints=$npoints --dataset=$dataset --norm-grid --res1d=$res1d --width=64 --modes=$((res1d/2))"
done
done
done
done




#for npoints in 100 1000 5000 9520; do
#for npoints in 100 1000 5000 9566; do
#for npoints in 100 1000 5000 10514; do
#for npoints in 100 1000 5000 7477; do
#for npoints in 100 1000 5000 7359; do
#for npoints in 100 1000 5000 7359; do
#for npoints in 100 1000 5000 7477; do
#for npoints in 100 1000 5000 7000; do
#for npoints in 100 1000 2700; do
#
