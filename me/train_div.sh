#!/bin/bash

set -euo pipefail

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

for seed in 2 3; do
    div_loss_weight=0.1
    sp "python3 ramansh_2d.py --project-name=geo-fno_div_loss --div-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/models --data-root=/projects/bgcs/mlowery/ram_dataset --epochs=500 --batch-size=20 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=flow_cylinder_laminar --ntrain=100 --npoints=1000 --res1d=60 --width=128 --modes=24" 2 "div_flow_cylinder_laminar"

    div_loss_weight=0.001
    sp "python3 ramansh_2d.py --project-name=geo-fno_div_loss --div-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/models --data-root=/projects/bgcs/mlowery/ram_dataset --epochs=500 --batch-size=20 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=flow_cylinder_shedding --ntrain=10000 --npoints=1000 --res1d=60 --width=64 --modes=28" 4 "div_flow_cylinder_shedding"

    div_loss_weight=0.001
    sp "python3 ramansh_2d.py --project-name=geo-fno_div_loss --div-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/models --data-root=/projects/bgcs/mlowery/ram_dataset --epochs=500 --batch-size=20 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=lid_cavity_flow --ntrain=10000 --npoints=1000 --res1d=40 --width=64 --modes=20" 2 "div_lid_cavity_flow"

    div_loss_weight=0.1
    sp "python3 ramansh_2d.py --project-name=geo-fno_div_loss --div-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/models --data-root=/projects/bgcs/mlowery/ram_dataset --epochs=500 --batch-size=20 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=backward_facing_step --ntrain=500 --npoints=1000 --res1d=40 --width=64 --modes=12" 2 "div_backward_facing_step"

    div_loss_weight=0.1
    sp "python3 ramansh_2d.py --project-name=geo-fno_div_loss --div-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/models --data-root=/projects/bgcs/mlowery/ram_dataset --epochs=500 --batch-size=20 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=buoyancy_cavity_flow --ntrain=10000 --npoints=5000 --res1d=40 --width=64 --modes=20" 7 "div_buoyancy_cavity_flow"

    div_loss_weight=0.01
    sp "python3 ramansh_2d.py --project-name=geo-fno_div_loss --div-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/models --data-root=/projects/bgcs/mlowery/ram_dataset --epochs=500 --batch-size=20 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=taylor_green --ntrain=5000 --npoints=500 --res1d=50 --width=64 --modes=20" 2 "div_taylor_green"

    div_loss_weight=0.1 # provisional: no Geo-FNO coefficient-map value was supplied
    sp "python3 ramansh_2d_diff_grids.py --project-name=geo-fno_div_loss --div-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/models --data-root=/projects/bgcs/mlowery/ram_dataset --epochs=500 --batch-size=20 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=taylor_green_coeffs --ntrain=5000 --npoints=500 --res1d=50 --width=64 --modes=20" 2 "div_taylor_green_coeffs"

    div_loss_weight=0.001
    sp "python3 ramansh_3d.py --project-name=geo-fno_div_loss --div-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/models --data-root=/projects/bgcs/mlowery/ram_dataset --epochs=500 --batch-size=20 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=taylor_green_spacetime --ntrain=5000 --npoints=500 --res1d=15 --width=64 --modes=7" 2 "div_taylor_green_spacetime"

    div_loss_weight=0.1
    sp "python3 ramansh_3d.py --project-name=geo-fno_div_loss --div-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/models --data-root=/projects/bgcs/mlowery/ram_dataset --epochs=500 --batch-size=20 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=taylor_green_spacetime_coeffs --ntrain=5000 --npoints=500 --res1d=15 --width=64 --modes=7" 2 "div_taylor_green_spacetime_coeffs"

    div_loss_weight=0.001
    sp "python3 ramansh_2d.py --project-name=geo-fno_div_loss --div-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/models --data-root=/projects/bgcs/mlowery/ram_dataset --epochs=500 --batch-size=20 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=merge_vortices_easier --ntrain=500 --npoints=500 --res1d=60 --width=128 --modes=12" 2 "div_merge_vortices_easier"

    div_loss_weight=0.001
    sp "python3 ramansh_3d.py --project-name=geo-fno_div_loss --div-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/models --data-root=/projects/bgcs/mlowery/ram_dataset --epochs=500 --batch-size=20 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=species_transport --ntrain=10000 --npoints=7000 --res1d=20 --width=64 --modes=10" 2 "div_species_transport"

    ### tune this
    sp "python3 ramansh_3d.py --project-name=geo-fno_div_loss --div-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/divs --model-folder=/projects/bfel/mlowery/geo-fno-div-loss/lambda-$div_loss_weight/models --data-root=/projects/bgcs/mlowery/ram_dataset --epochs=500 --batch-size=20 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=$div_loss_weight --seed=$seed --dataset=forced_turb --ntrain=10000 --npoints=7000 --res1d=20 --width=64 --modes=10" 2 "div_forced_turb"


done
