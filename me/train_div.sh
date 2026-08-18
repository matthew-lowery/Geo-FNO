#!/bin/bash

set -euo pipefail

DIV_LOSS_WEIGHT=1.0

SEEDS=(1 2 3)
DATA_DIR='/projects/bfel/mlowery/geo-fno-new'
RUN_DIR="/projects/bfel/mlowery/geo-fno-div-loss/lambda-$DIV_LOSS_WEIGHT"
DIV_DIR="$RUN_DIR/divs"
MODEL_DIR="$RUN_DIR/models"
PROJECT_NAME='geo-fno_div_loss'

COMMON_ARGS="--project-name=$PROJECT_NAME --div-folder=$DIV_DIR --model-folder=$MODEL_DIR --dir=$DATA_DIR --epochs=500 --batch-size=20 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$DIV_LOSS_WEIGHT"

sp() {
    local pycmd="$1"
    local hours="$2"
    local job_name="$3"

    sbatch <<EOF
#!/bin/bash
#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpuA100x4,gpuA100x8
#SBATCH --account=bfel-delta-gpu
#SBATCH --job-name=$job_name
#SBATCH --time=${hours}:00:00
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

submit_2d() {
    local dataset="$1" ntrain="$2" res1d="$3" width="$4" modes="$5" hours="$6"
    for seed in "${SEEDS[@]}"; do
        sp "python3 ramansh_2d.py $COMMON_ARGS --seed=$seed --dataset=$dataset --ntrain=$ntrain --npoints=all --res1d=$res1d --width=$width --modes=$modes" "$hours" "div_$dataset"
    done
}

submit_3d() {
    local dataset="$1" ntrain="$2" res1d="$3" width="$4" modes="$5" hours="$6"
    for seed in "${SEEDS[@]}"; do
        sp "python3 ramansh_3d.py $COMMON_ARGS --seed=$seed --dataset=$dataset --ntrain=$ntrain --npoints=2700 --res1d=$res1d --width=$width --modes=$modes" "$hours" "div_$dataset"
    done
}

submit_airfoil() {
    local ntrain="$1" res1d="$2" width="$3" modes="$4" hours="$5"
    for seed in "${SEEDS[@]}"; do
        sp "python3 ramansh_3d_airfoil.py $COMMON_ARGS --seed=$seed --dataset=airfoil --ntrain=$ntrain --npoints=all --res1d=$res1d --width=$width --modes=$modes" "$hours" 'div_airfoil'
    done
}

submit_2d_diff_grids() {
    local dataset="$1" ntrain="$2" res1d="$3" width="$4" modes="$5" hours="$6"
    for seed in "${SEEDS[@]}"; do
        sp "python3 ramansh_2d_diff_grids.py $COMMON_ARGS --seed=$seed --dataset=$dataset --ntrain=$ntrain --npoints=all --res1d=$res1d --width=$width --modes=$modes" "$hours" "div_$dataset"
    done
}

# dataset, ntrain, res1d, width, modes, hours
submit_2d flow_cylinder_laminar 100 60 128 24 8
submit_2d flow_cylinder_shedding 10000 60 64 28 8
submit_2d lid_cavity_flow 10000 40 64 20 8
submit_2d backward_facing_step 500 40 64 12 8
submit_2d buoyancy_cavity_flow 10000 40 64 20 8
submit_2d taylor_green_exact 5000 50 64 20 8
submit_3d taylor_green_time 500 15 64 7 5
submit_2d merge_vortices_easier 500 60 128 12 8
submit_3d species_transport 10000 20 64 10 30
submit_airfoil 7000 25 64 12 12

submit_2d backward_facing_step_ood 500 40 64 12 8
submit_3d taylor_green_time_coeffs 500 15 64 7 10
submit_2d_diff_grids taylor_green_coeffs 500 40 64 12 10
