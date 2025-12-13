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

### need to specify the save dir also
#problems=("flow_cylinder_shedding" "flow_cylinder_laminar" "taylor_green_exact" "backward_facing_step" "backward_facing_step_ood" "merge_vortices_easier" "buoyancy_cavity_flow" "lid_cavity_flow")
#Ns=(10000  100  5000  500  500  500  10000  10000)

dir='/projects/bfel/mlowery/geo-fno-new'
divdir='/projects/bfel/mlowery/geo-fno-new_div'
modeldir='/projects/bfel/mlowery/geo-fno-new_models'
projname='ramansh_specific'
### currently need to rerun airfoil, taylor green time, make sure bfs_ood is good 
#
#dataset='flow_cylinder_laminar'
#ntrain=100
#for seed in 1 2 3; do
#sp "python3 ramansh_2d.py --project-name=$projname --div-folder=$divdir --model-folder=$modeldir --dir=$dir --wandb --calc-div --save --seed=$seed --ntrain=$ntrain --dataset=$dataset --norm-grid --res1d=60 --width=128 --modes=24"
#done
#
#dataset='flow_cylinder_shedding'
#ntrain=10000
#for seed in 1 2 3; do
#sp "python3 ramansh_2d.py --project-name=$projname --div-folder=$divdir --model-folder=$modeldir --dir=$dir --wandb --calc-div --save --seed=$seed --ntrain=$ntrain --dataset=$dataset --norm-grid --res1d=60 --width=64 --modes=28"
#done
#
#dataset='lid_cavity_flow'
#ntrain=10000
#for seed in 1 2 3; do
#sp "python3 ramansh_2d.py --project-name=$projname --div-folder=$divdir --model-folder=$modeldir --dir=$dir --wandb --calc-div --save --seed=$seed --ntrain=$ntrain --dataset=$dataset --norm-grid --res1d=40 --width=64 --modes=20"
#done
#
#dataset='merge_vortices_easier'
#ntrain=500
#for seed in 1 2 3; do
#sp "python3 ramansh_2d.py --project-name=$projname --div-folder=$divdir --model-folder=$modeldir --dir=$dir --wandb --calc-div --save --seed=$seed --ntrain=$ntrain --dataset=$dataset --norm-grid --res1d=60 --width=128 --modes=12"
#done
#
#dataset='buoyancy_cavity_flow'
#ntrain=10000
#for seed in 1 2 3; do
#sp "python3 ramansh_2d.py --project-name=$projname --div-folder=$divdir --model-folder=$modeldir --dir=$dir --wandb --calc-div --save --seed=$seed --ntrain=$ntrain --dataset=$dataset --norm-grid --res1d=40 --width=64 --modes=20"
#done
#
#dataset='backward_facing_step_ood'
#ntrain=500
#for seed in 1 2 3; do
#sp "python3 ramansh_2d.py --project-name=$projname --div-folder=$divdir --model-folder=$modeldir --dir=$dir --wandb --calc-div --save --seed=$seed --ntrain=$ntrain --dataset=$dataset --norm-grid --res1d=40 --width=64 --modes=12"
#done
#
#dataset='backward_facing_step'
#ntrain=500
#for seed in 1 2 3; do
#sp "python3 ramansh_2d.py --project-name=$projname --div-folder=$divdir --model-folder=$modeldir --dir=$dir --wandb --calc-div --save --seed=$seed --ntrain=$ntrain --dataset=$dataset --norm-grid --res1d=40 --width=64 --modes=12"
#done
#
#dataset='taylor_green_exact'
#ntrain=5000
#for seed in 1 2 3; do
#sp "python3 ramansh_2d.py --project-name=$projname --div-folder=$divdir --model-folder=$modeldir --dir=$dir --wandb --calc-div --save --seed=$seed --ntrain=$ntrain --dataset=$dataset --norm-grid --res1d=50 --width=64 --modes=20"
#done
#
res1d=25
modes=12
width=64
ntrain=7000
for seed in 1 2 3; do
sp "python3 ramansh_3d_airfoil.py --project-name=$projname --div-folder=$divdir --model-folder=$modeldir --dir=$dir --calc-div --save --wandb --seed=$seed --ntrain=$ntrain --norm-grid --res1d=$res1d --width=$width --modes=$modes" 12
done

res1d=15
modes=7
width=64
ntrain=500
for seed in 1 2 3; do
sp "python3 ramansh_3d.py --project-name=$projname --div-folder=$divdir --model-folder=$modeldir --dir=$dir --calc-div --save --dataset='taylor_green_time' --wandb --seed=$seed --ntrain=$ntrain --norm-grid --res1d=$res1d --width=$width --modes=$modes" 5
done

