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
#SBATCH --partition=gpuA100x4
#SBATCH --account=bfel-delta-gpu
#SBATCH --job-name=myjob
#SBATCH --time=$hr:00:00
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

### run species transport
res1d=20
modes=10
width=64
for seed in 1 2 3; do
for ntrain in 10000; do
sp "python3 ramansh_3d.py --calc-div --save --batch-size=20 --dataset='species_transport' --wandb --seed=$seed --width=$width --ntrain=$ntrain --norm-grid --res1d=20 --modes=10" 30
done
done
#
#### run tgt
#res1d=15
#modes=7
#width=64
#for seed in 1 2 3; do
#for ntrain in 5000 7000 10000; do
#sp "python3 ramansh_3d.py --calc-div --save --dataset='taylor_green_time' --wandb --seed=$seed --ntrain=$ntrain --norm-grid --res1d=$res1d --width=$width --modes=$modes"
#done
#done
#
#### run airfoil and bfs bc
#res1d=25
#modes=12
#width=64
#for seed in 1 2 3; do
#for ntrain in 5000 7000 10000; do
#sp "python3 ramansh_3d_airfoil.py --calc-div --save --wandb --seed=$seed --ntrain=$ntrain --norm-grid --res1d=$res1d --width=$width --modes=$modes"
#done
#done
#
#res1d=50
#modes=16
#width=64
#for seed in 1 2 3; do
#for ntrain in 5000 7000 10000; do
#sp "python3 ramansh_2d_backward_facing_step_bc.py --calc-div --save --wandb --seed=$seed --ntrain=$ntrain --norm-grid --res1d=$res1d --width=$width --modes=$modes"
#done
#done
#
