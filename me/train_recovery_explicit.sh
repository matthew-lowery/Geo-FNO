#!/bin/bash
set -euo pipefail

ME_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${TRAIN_PYTHON:-/u/mlowery/.conda/envs/gnot/bin/python}"
DATA_ROOT="${RAM_DATA_ROOT:-/projects/bgcs/mlowery/ram_dataset}"
RESULTS_ROOT="${RAM_RESULTS_ROOT:-/projects/bfel/mlowery/operator-benchmarks/recovery-20260918}"

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

# Transolver: remaining 10,000-sample forced-turbulence seeds.
sp "$PYTHON -m transolver.train --project-name=trans_div_loss --div-folder=$RESULTS_ROOT/trans/div/lambda-0/divs --model-folder=$RESULTS_ROOT/trans/div/lambda-0/models --data-root=$DATA_ROOT --model=Transolver_Irregular_Mesh --epochs=500 --batch-size=20 --lr=1e-3 --weight_decay=1e-5 --mlp_ratio=1 --dropout=0.0 --downsample=1 --ref=8 --unified_pos=0 --gpu=0 --wandb --calc-div --save --norm-grid --div-loss-weight=0 --seed=3 --dataset=forced_turb --ntrain=10000 --npoints=7000 --n-hidden=128 --n-layers=5 --n-heads=4 --slice-num=32 --div-order=4" 24 trans_forced_turb_n10000_l0_s3
sp "$PYTHON -m transolver.train --project-name=trans_div_loss --div-folder=$RESULTS_ROOT/trans/div/lambda-0.001/divs --model-folder=$RESULTS_ROOT/trans/div/lambda-0.001/models --data-root=$DATA_ROOT --model=Transolver_Irregular_Mesh --epochs=500 --batch-size=20 --lr=1e-3 --weight_decay=1e-5 --mlp_ratio=1 --dropout=0.0 --downsample=1 --ref=8 --unified_pos=0 --gpu=0 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=0.001 --no-ood --seed=1 --dataset=forced_turb --ntrain=10000 --npoints=7000 --n-hidden=128 --n-layers=5 --n-heads=4 --slice-num=32 --div-order=4" 24 trans_forced_turb_n10000_l001_s1
sp "$PYTHON -m transolver.train --project-name=trans_div_loss --div-folder=$RESULTS_ROOT/trans/div/lambda-0.01/divs --model-folder=$RESULTS_ROOT/trans/div/lambda-0.01/models --data-root=$DATA_ROOT --model=Transolver_Irregular_Mesh --epochs=500 --batch-size=20 --lr=1e-3 --weight_decay=1e-5 --mlp_ratio=1 --dropout=0.0 --downsample=1 --ref=8 --unified_pos=0 --gpu=0 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=0.01 --no-ood --seed=1 --dataset=forced_turb --ntrain=10000 --npoints=7000 --n-hidden=128 --n-layers=5 --n-heads=4 --slice-num=32 --div-order=4" 24 trans_forced_turb_n10000_l01_s1
sp "$PYTHON -m transolver.train --project-name=trans_div_loss --div-folder=$RESULTS_ROOT/trans/div/lambda-0.1/divs --model-folder=$RESULTS_ROOT/trans/div/lambda-0.1/models --data-root=$DATA_ROOT --model=Transolver_Irregular_Mesh --epochs=500 --batch-size=20 --lr=1e-3 --weight_decay=1e-5 --mlp_ratio=1 --dropout=0.0 --downsample=1 --ref=8 --unified_pos=0 --gpu=0 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=0.1 --no-ood --seed=1 --dataset=forced_turb --ntrain=10000 --npoints=7000 --n-hidden=128 --n-layers=5 --n-heads=4 --slice-num=32 --div-order=4" 24 trans_forced_turb_n10000_l01_s1
sp "$PYTHON -m transolver.train --project-name=trans_div_loss --div-folder=$RESULTS_ROOT/trans/div/lambda-1/divs --model-folder=$RESULTS_ROOT/trans/div/lambda-1/models --data-root=$DATA_ROOT --model=Transolver_Irregular_Mesh --epochs=500 --batch-size=20 --lr=1e-3 --weight_decay=1e-5 --mlp_ratio=1 --dropout=0.0 --downsample=1 --ref=8 --unified_pos=0 --gpu=0 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=1 --no-ood --seed=1 --dataset=forced_turb --ntrain=10000 --npoints=7000 --n-hidden=128 --n-layers=5 --n-heads=4 --slice-num=32 --div-order=4" 24 trans_forced_turb_n10000_l1_s1

# Geo-FNO: no-div forced turbulence, all three seeds.
sp "$PYTHON ramansh_3d.py --project-name=geo_div_loss --div-folder=$RESULTS_ROOT/geo/baseline/lambda-0/divs --model-folder=$RESULTS_ROOT/geo/baseline/lambda-0/models --data-root=$DATA_ROOT --epochs=500 --batch-size=10 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss-weight=0 --seed=1 --dataset=forced_turb --ntrain=10000 --npoints=7000 --res1d=20 --width=64 --modes=10 --div-order=3" 36 geo_forced_turb_n10000_l0_s1
sp "$PYTHON ramansh_3d.py --project-name=geo_div_loss --div-folder=$RESULTS_ROOT/geo/baseline/lambda-0/divs --model-folder=$RESULTS_ROOT/geo/baseline/lambda-0/models --data-root=$DATA_ROOT --epochs=500 --batch-size=10 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss-weight=0 --seed=2 --dataset=forced_turb --ntrain=10000 --npoints=7000 --res1d=20 --width=64 --modes=10 --div-order=3" 36 geo_forced_turb_n10000_l0_s2
sp "$PYTHON ramansh_3d.py --project-name=geo_div_loss --div-folder=$RESULTS_ROOT/geo/baseline/lambda-0/divs --model-folder=$RESULTS_ROOT/geo/baseline/lambda-0/models --data-root=$DATA_ROOT --epochs=500 --batch-size=10 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss-weight=0 --seed=3 --dataset=forced_turb --ntrain=10000 --npoints=7000 --res1d=20 --width=64 --modes=10 --div-order=3" 36 geo_forced_turb_n10000_l0_s3

# Species transport no-div OOD baselines, all three seeds.
sp "$PYTHON -m transolver.train --project-name=trans_div_loss --div-folder=$RESULTS_ROOT/trans/baseline/lambda-0/divs --model-folder=$RESULTS_ROOT/trans/baseline/lambda-0/models --data-root=$DATA_ROOT --model=Transolver_Irregular_Mesh --epochs=500 --batch-size=20 --lr=1e-3 --weight_decay=1e-5 --mlp_ratio=1 --dropout=0.0 --downsample=1 --ref=8 --unified_pos=0 --gpu=0 --wandb --calc-div --save --norm-grid --div-loss-weight=0 --seed=1 --dataset=species_transport --ntrain=10000 --npoints=7000 --n-hidden=128 --n-layers=5 --n-heads=4 --slice-num=32 --div-order=4" 12 trans_species_n10000_l0_s1
sp "$PYTHON -m transolver.train --project-name=trans_div_loss --div-folder=$RESULTS_ROOT/trans/baseline/lambda-0/divs --model-folder=$RESULTS_ROOT/trans/baseline/lambda-0/models --data-root=$DATA_ROOT --model=Transolver_Irregular_Mesh --epochs=500 --batch-size=20 --lr=1e-3 --weight_decay=1e-5 --mlp_ratio=1 --dropout=0.0 --downsample=1 --ref=8 --unified_pos=0 --gpu=0 --wandb --calc-div --save --norm-grid --div-loss-weight=0 --seed=2 --dataset=species_transport --ntrain=10000 --npoints=7000 --n-hidden=128 --n-layers=5 --n-heads=4 --slice-num=32 --div-order=4" 12 trans_species_n10000_l0_s2
sp "$PYTHON -m transolver.train --project-name=trans_div_loss --div-folder=$RESULTS_ROOT/trans/baseline/lambda-0/divs --model-folder=$RESULTS_ROOT/trans/baseline/lambda-0/models --data-root=$DATA_ROOT --model=Transolver_Irregular_Mesh --epochs=500 --batch-size=20 --lr=1e-3 --weight_decay=1e-5 --mlp_ratio=1 --dropout=0.0 --downsample=1 --ref=8 --unified_pos=0 --gpu=0 --wandb --calc-div --save --norm-grid --div-loss-weight=0 --seed=3 --dataset=species_transport --ntrain=10000 --npoints=7000 --n-hidden=128 --n-layers=5 --n-heads=4 --slice-num=32 --div-order=4" 12 trans_species_n10000_l0_s3
sp "$PYTHON ramansh_3d.py --project-name=geo_div_loss --div-folder=$RESULTS_ROOT/geo/baseline/lambda-0/divs --model-folder=$RESULTS_ROOT/geo/baseline/lambda-0/models --data-root=$DATA_ROOT --epochs=500 --batch-size=10 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss-weight=0 --seed=1 --dataset=species_transport --ntrain=10000 --npoints=7000 --res1d=20 --width=64 --modes=10 --div-order=3" 32 geo_species_n10000_l0_s1
sp "$PYTHON ramansh_3d.py --project-name=geo_div_loss --div-folder=$RESULTS_ROOT/geo/baseline/lambda-0/divs --model-folder=$RESULTS_ROOT/geo/baseline/lambda-0/models --data-root=$DATA_ROOT --epochs=500 --batch-size=10 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss-weight=0 --seed=2 --dataset=species_transport --ntrain=10000 --npoints=7000 --res1d=20 --width=64 --modes=10 --div-order=3" 32 geo_species_n10000_l0_s2
sp "$PYTHON ramansh_3d.py --project-name=geo_div_loss --div-folder=$RESULTS_ROOT/geo/baseline/lambda-0/divs --model-folder=$RESULTS_ROOT/geo/baseline/lambda-0/models --data-root=$DATA_ROOT --epochs=500 --batch-size=10 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss-weight=0 --seed=3 --dataset=species_transport --ntrain=10000 --npoints=7000 --res1d=20 --width=64 --modes=10 --div-order=3" 32 geo_species_n10000_l0_s3

# Geo-FNO OOD recovery: only datasets with an OOD file.
sp "$PYTHON ramansh_2d.py --project-name=geo_div_loss --div-folder=$RESULTS_ROOT/geo/baseline/lambda-0/divs --model-folder=$RESULTS_ROOT/geo/baseline/lambda-0/models --data-root=$DATA_ROOT --epochs=500 --batch-size=20 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss-weight=0 --seed=1 --dataset=flow_cylinder_shedding --ntrain=10000 --npoints=all --res1d=60 --width=64 --modes=28 --div-order=3" 6 geo_shedding_ood_s1
sp "$PYTHON ramansh_2d.py --project-name=geo_div_loss --div-folder=$RESULTS_ROOT/geo/baseline/lambda-0/divs --model-folder=$RESULTS_ROOT/geo/baseline/lambda-0/models --data-root=$DATA_ROOT --epochs=500 --batch-size=20 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss-weight=0 --seed=2 --dataset=flow_cylinder_shedding --ntrain=10000 --npoints=all --res1d=60 --width=64 --modes=28 --div-order=3" 6 geo_shedding_ood_s2
sp "$PYTHON ramansh_2d.py --project-name=geo_div_loss --div-folder=$RESULTS_ROOT/geo/baseline/lambda-0/divs --model-folder=$RESULTS_ROOT/geo/baseline/lambda-0/models --data-root=$DATA_ROOT --epochs=500 --batch-size=20 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss-weight=0 --seed=3 --dataset=flow_cylinder_shedding --ntrain=10000 --npoints=all --res1d=60 --width=64 --modes=28 --div-order=3" 6 geo_shedding_ood_s3
sp "$PYTHON ramansh_2d.py --project-name=geo_div_loss --div-folder=$RESULTS_ROOT/geo/baseline/lambda-0/divs --model-folder=$RESULTS_ROOT/geo/baseline/lambda-0/models --data-root=$DATA_ROOT --epochs=500 --batch-size=20 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss-weight=0 --seed=1 --dataset=taylor_green --ntrain=5000 --npoints=all --res1d=50 --width=64 --modes=20 --div-order=2" 3 geo_taylor_green_ood_s1
sp "$PYTHON ramansh_2d.py --project-name=geo_div_loss --div-folder=$RESULTS_ROOT/geo/baseline/lambda-0/divs --model-folder=$RESULTS_ROOT/geo/baseline/lambda-0/models --data-root=$DATA_ROOT --epochs=500 --batch-size=20 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss-weight=0 --seed=2 --dataset=taylor_green --ntrain=5000 --npoints=all --res1d=50 --width=64 --modes=20 --div-order=2" 3 geo_taylor_green_ood_s2
sp "$PYTHON ramansh_2d.py --project-name=geo_div_loss --div-folder=$RESULTS_ROOT/geo/baseline/lambda-0/divs --model-folder=$RESULTS_ROOT/geo/baseline/lambda-0/models --data-root=$DATA_ROOT --epochs=500 --batch-size=20 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss-weight=0 --seed=3 --dataset=taylor_green --ntrain=5000 --npoints=all --res1d=50 --width=64 --modes=20 --div-order=2" 3 geo_taylor_green_ood_s3
sp "$PYTHON ramansh_3d.py --project-name=geo_div_loss --div-folder=$RESULTS_ROOT/geo/baseline/lambda-0/divs --model-folder=$RESULTS_ROOT/geo/baseline/lambda-0/models --data-root=$DATA_ROOT --epochs=500 --batch-size=20 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss-weight=0 --seed=1 --dataset=taylor_green_spacetime --ntrain=5000 --npoints=all --res1d=15 --width=64 --modes=7 --div-order=2" 4 geo_taylor_green_spacetime_ood_s1
sp "$PYTHON ramansh_3d.py --project-name=geo_div_loss --div-folder=$RESULTS_ROOT/geo/baseline/lambda-0/divs --model-folder=$RESULTS_ROOT/geo/baseline/lambda-0/models --data-root=$DATA_ROOT --epochs=500 --batch-size=20 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss-weight=0 --seed=2 --dataset=taylor_green_spacetime --ntrain=5000 --npoints=all --res1d=15 --width=64 --modes=7 --div-order=2" 4 geo_taylor_green_spacetime_ood_s2
sp "$PYTHON ramansh_3d.py --project-name=geo_div_loss --div-folder=$RESULTS_ROOT/geo/baseline/lambda-0/divs --model-folder=$RESULTS_ROOT/geo/baseline/lambda-0/models --data-root=$DATA_ROOT --epochs=500 --batch-size=20 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss-weight=0 --seed=3 --dataset=taylor_green_spacetime --ntrain=5000 --npoints=all --res1d=15 --width=64 --modes=7 --div-order=2" 4 geo_taylor_green_spacetime_ood_s3
