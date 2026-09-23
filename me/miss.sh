#!/bin/bash
set -euo pipefail

ME_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${TRAIN_PYTHON:-/u/mlowery/.conda/envs/gnot/bin/python}"
DATA_ROOT="${RAM_DATA_ROOT:-/projects/bgcs/mlowery/ram_dataset}"
RESULTS_ROOT="${RAM_RESULTS_ROOT:-/projects/bgcs/mlowery/operator-benchmarks/missing-recovery-20260922}"

sp() {
    local command="$1" hours="$2" job_name="$3"
    sbatch <<EOF
#!/bin/bash
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgcs-delta-gpu
#SBATCH --constraint=scratch
#SBATCH --job-name=$job_name
#SBATCH --time=${hours}:00:00
#SBATCH --output=$ME_DIR/out/%x_%A.out
#SBATCH --error=$ME_DIR/err/%x_%A.err

set -euo pipefail
module purge
cd $ME_DIR
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
$command
EOF
}

mkdir -p "$ME_DIR/out" "$ME_DIR/err"

sp "$PYTHON ramansh_2d_diff_grids.py --project-name=geo_div_loss --div-folder=$RESULTS_ROOT/geo/baseline/lambda-0/divs --model-folder=$RESULTS_ROOT/geo/baseline/lambda-0/models --data-root=$DATA_ROOT --epochs=500 --batch-size=20 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --no-ood --div-loss-weight=0 --seed=1 --dataset=taylor_green_coeffs --ntrain=5000 --npoints=500 --res1d=50 --width=64 --modes=20 --div-order=2" 2 miss_geo_tg_coeffs_s1
sp "$PYTHON ramansh_2d_diff_grids.py --project-name=geo_div_loss --div-folder=$RESULTS_ROOT/geo/baseline/lambda-0/divs --model-folder=$RESULTS_ROOT/geo/baseline/lambda-0/models --data-root=$DATA_ROOT --epochs=500 --batch-size=20 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --no-ood --div-loss-weight=0 --seed=2 --dataset=taylor_green_coeffs --ntrain=5000 --npoints=500 --res1d=50 --width=64 --modes=20 --div-order=2" 2 miss_geo_tg_coeffs_s2
sp "$PYTHON ramansh_2d_diff_grids.py --project-name=geo_div_loss --div-folder=$RESULTS_ROOT/geo/baseline/lambda-0/divs --model-folder=$RESULTS_ROOT/geo/baseline/lambda-0/models --data-root=$DATA_ROOT --epochs=500 --batch-size=20 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --no-ood --div-loss-weight=0 --seed=3 --dataset=taylor_green_coeffs --ntrain=5000 --npoints=500 --res1d=50 --width=64 --modes=20 --div-order=2" 2 miss_geo_tg_coeffs_s3

sp "$PYTHON -m transolver.train --project-name=trans_div_loss --div-folder=$RESULTS_ROOT/trans/baseline/lambda-0/divs --model-folder=$RESULTS_ROOT/trans/baseline/lambda-0/models --data-root=$DATA_ROOT --model=Transolver_Irregular_Mesh --epochs=500 --batch-size=20 --lr=1e-3 --weight_decay=1e-5 --mlp_ratio=1 --dropout=0.0 --ref=8 --gpu=0 --wandb --calc-div --save --norm-grid --div-loss-weight=0 --require-ood --seed=1 --dataset=taylor_green_spacetime --ntrain=5000 --npoints=500 --n-hidden=128 --n-layers=5 --n-heads=4 --slice-num=32 --div-order=4" 3 miss_trans_tg_spacetime_s1
sp "$PYTHON -m transolver.train --project-name=trans_div_loss --div-folder=$RESULTS_ROOT/trans/baseline/lambda-0/divs --model-folder=$RESULTS_ROOT/trans/baseline/lambda-0/models --data-root=$DATA_ROOT --model=Transolver_Irregular_Mesh --epochs=500 --batch-size=20 --lr=1e-3 --weight_decay=1e-5 --mlp_ratio=1 --dropout=0.0 --ref=8 --gpu=0 --wandb --calc-div --save --norm-grid --div-loss-weight=0 --require-ood --seed=2 --dataset=taylor_green_spacetime --ntrain=5000 --npoints=500 --n-hidden=128 --n-layers=5 --n-heads=4 --slice-num=32 --div-order=4" 3 miss_trans_tg_spacetime_s2
sp "$PYTHON -m transolver.train --project-name=trans_div_loss --div-folder=$RESULTS_ROOT/trans/baseline/lambda-0/divs --model-folder=$RESULTS_ROOT/trans/baseline/lambda-0/models --data-root=$DATA_ROOT --model=Transolver_Irregular_Mesh --epochs=500 --batch-size=20 --lr=1e-3 --weight_decay=1e-5 --mlp_ratio=1 --dropout=0.0 --ref=8 --gpu=0 --wandb --calc-div --save --norm-grid --div-loss-weight=0 --require-ood --seed=3 --dataset=taylor_green_spacetime --ntrain=5000 --npoints=500 --n-hidden=128 --n-layers=5 --n-heads=4 --slice-num=32 --div-order=4" 3 miss_trans_tg_spacetime_s3
