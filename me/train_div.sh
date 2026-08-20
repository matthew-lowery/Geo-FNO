#!/bin/bash

set -euo pipefail

DIV_LOSS_WEIGHTS=(1.0)
SEEDS=(1 2 3)

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

for div_loss_weight in "${DIV_LOSS_WEIGHTS[@]}"; do
for seed in 1; do
    sp "python3 ramansh_2d.py --project-name=geo-fno_div_loss --div-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/models --dir=/projects/bfel/mlowery/geo-fno-new --epochs=500 --batch-size=20 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=flow_cylinder_laminar --ntrain=100 --npoints=all --res1d=60 --width=128 --modes=24" 8 "div_flow_cylinder_laminar"

    sp "python3 ramansh_2d.py --project-name=geo-fno_div_loss --div-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/models --dir=/projects/bfel/mlowery/geo-fno-new --epochs=500 --batch-size=20 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=flow_cylinder_shedding --ntrain=10000 --npoints=all --res1d=60 --width=64 --modes=28" 8 "div_flow_cylinder_shedding"

    sp "python3 ramansh_2d.py --project-name=geo-fno_div_loss --div-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/models --dir=/projects/bfel/mlowery/geo-fno-new --epochs=500 --batch-size=20 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=lid_cavity_flow --ntrain=10000 --npoints=all --res1d=40 --width=64 --modes=20" 8 "div_lid_cavity_flow"

    sp "python3 ramansh_2d.py --project-name=geo-fno_div_loss --div-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/models --dir=/projects/bfel/mlowery/geo-fno-new --epochs=500 --batch-size=20 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=backward_facing_step --ntrain=500 --npoints=all --res1d=40 --width=64 --modes=12" 8 "div_backward_facing_step"

    sp "python3 ramansh_2d.py --project-name=geo-fno_div_loss --div-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/models --dir=/projects/bfel/mlowery/geo-fno-new --epochs=500 --batch-size=20 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=buoyancy_cavity_flow --ntrain=10000 --npoints=all --res1d=40 --width=64 --modes=20" 8 "div_buoyancy_cavity_flow"

    sp "python3 ramansh_2d.py --project-name=geo-fno_div_loss --div-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/models --dir=/projects/bfel/mlowery/geo-fno-new --epochs=500 --batch-size=20 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=taylor_green_exact --ntrain=5000 --npoints=all --res1d=50 --width=64 --modes=20" 8 "div_taylor_green_exact"

    sp "python3 ramansh_3d.py --project-name=geo-fno_div_loss --div-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/models --dir=/projects/bfel/mlowery/geo-fno-new --epochs=500 --batch-size=20 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=taylor_green_time --ntrain=500 --npoints=2700 --res1d=15 --width=64 --modes=7" 5 "div_taylor_green_time"

    sp "python3 ramansh_2d.py --project-name=geo-fno_div_loss --div-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/models --dir=/projects/bfel/mlowery/geo-fno-new --epochs=500 --batch-size=20 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=merge_vortices_easier --ntrain=500 --npoints=all --res1d=60 --width=128 --modes=12" 8 "div_merge_vortices_easier"

    sp "python3 ramansh_3d.py --project-name=geo-fno_div_loss --div-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/models --dir=/projects/bfel/mlowery/geo-fno-new --epochs=500 --batch-size=20 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=species_transport --ntrain=10000 --npoints=2700 --res1d=20 --width=64 --modes=10" 30 "div_species_transport"

    sp "python3 ramansh_3d_airfoil.py --project-name=geo-fno_div_loss --div-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/models --dir=/projects/bfel/mlowery/geo-fno-new --epochs=500 --batch-size=20 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=airfoil --ntrain=7000 --npoints=all --res1d=25 --width=64 --modes=12" 12 "div_airfoil"

    sp "python3 ramansh_2d.py --project-name=geo-fno_div_loss --div-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/models --dir=/projects/bfel/mlowery/geo-fno-new --epochs=500 --batch-size=20 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=backward_facing_step_ood --ntrain=500 --npoints=all --res1d=40 --width=64 --modes=12" 8 "div_backward_facing_step_ood"

    sp "python3 ramansh_3d.py --project-name=geo-fno_div_loss --div-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/models --dir=/projects/bfel/mlowery/geo-fno-new --epochs=500 --batch-size=20 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=taylor_green_time_coeffs --ntrain=500 --npoints=2700 --res1d=15 --width=64 --modes=7" 10 "div_taylor_green_time_coeffs"

    sp "python3 ramansh_2d_diff_grids.py --project-name=geo-fno_div_loss --div-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/models --dir=/projects/bfel/mlowery/geo-fno-new --epochs=500 --batch-size=20 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=taylor_green_coeffs --ntrain=500 --npoints=all --res1d=40 --width=64 --modes=12" 10 "div_taylor_green_coeffs"
done
done
