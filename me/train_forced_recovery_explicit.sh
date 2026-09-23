#!/bin/bash
set -euo pipefail

ME_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${TRAIN_PYTHON:-/u/rsharma15/.conda/envs/operator-benchmarks-gpu/bin/python}"
DATA_ROOT="${RAM_DATA_ROOT:-/u/rsharma15/pde_ml/code/op_dataset}"
RESULTS_ROOT="${RAM_RESULTS_ROOT:-/u/rsharma15/operator-benchmarks/forced-recovery-20260923}"
dry_run=false
case "${1:-}" in
    --dry-run) dry_run=true ;;
    "") ;;
    *) echo "Usage: bash train_forced_recovery_explicit.sh [--dry-run]" >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || exit 2

active_jobs=""
if [[ "$dry_run" == false ]]; then
    PYTHON="$(command -v "$PYTHON")"
    [[ -r "$DATA_ROOT/forced_turb/data.mat" && -r "$DATA_ROOT/forced_turb/data_ood.mat" ]] || {
        echo "Missing forced_turb/data.mat or data_ood.mat under $DATA_ROOT" >&2
        exit 1
    }
    active_jobs="$(squeue --me --noheader --format=%200j | awk '{$1=$1; print}')"
    mkdir -p "$ME_DIR/out" "$ME_DIR/err"
fi

sp() {
    local command="$1" hours="$2" job_name="$3" legacy_name
    if [[ "$dry_run" == true ]]; then
        printf '%sh %s\n%s\n' "$hours" "$job_name" "$command"
        return
    fi
    [[ "$job_name" =~ ^(geo|trans)_forced_turb_n([0-9]+)_l([0-9p]+)_s([123])$ ]] || return 1
    legacy_name="${BASH_REMATCH[1]}_forced_turb_s${BASH_REMATCH[4]}_n${BASH_REMATCH[2]}_l${BASH_REMATCH[3]//p/.}"
    if [[ "$active_jobs"$'\n' == *"$job_name"$'\n'* || "$active_jobs"$'\n' == *"$legacy_name"$'\n'* ]]; then
        echo "Skip queued/running: $job_name"
        return
    fi
    sbatch <<EOF
#!/bin/bash
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgkk-delta-gpu
#SBATCH --constraint=scratch
#SBATCH --job-name=$job_name
#SBATCH --time=${hours}:00:00
#SBATCH --output=$ME_DIR/out/%x_%A.out
#SBATCH --error=$ME_DIR/err/%x_%A.err

set -euo pipefail
module purge
cd "$ME_DIR"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
"$PYTHON" -c 'import torch, scipy, h5py, matplotlib, wandb, timm, einops; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), flush=True)'
$command
EOF
}

# Unfinished cases in wandb_final_final_final, shortest to longest.
sp "$PYTHON -m transolver.train --project-name=trans_div_loss --div-folder=$RESULTS_ROOT/trans/baseline/lambda-0/divs --model-folder=$RESULTS_ROOT/trans/baseline/lambda-0/models --data-root=$DATA_ROOT --model=Transolver_Irregular_Mesh --epochs=500 --batch-size=20 --lr=1e-3 --weight_decay=1e-5 --mlp_ratio=1 --dropout=0.0 --ref=8 --gpu=0 --wandb --calc-div --save --norm-grid --div-loss-weight=0 --require-ood --seed=3 --dataset=forced_turb --ntrain=10000 --npoints=7000 --n-hidden=128 --n-layers=5 --n-heads=4 --slice-num=32 --div-order=4" 10 trans_forced_turb_n10000_l0_s3
sp "$PYTHON ramansh_3d.py --project-name=geo_div_loss --div-folder=$RESULTS_ROOT/geo/forced/lambda-0/divs --model-folder=$RESULTS_ROOT/geo/forced/lambda-0/models --data-root=$DATA_ROOT --epochs=500 --batch-size=10 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss-weight=0 --require-ood --seed=2 --dataset=forced_turb --ntrain=5000 --npoints=7000 --res1d=20 --width=64 --modes=10 --div-order=3" 24 geo_forced_turb_n5000_l0_s2
sp "$PYTHON ramansh_3d.py --project-name=geo_div_loss --div-folder=$RESULTS_ROOT/geo/forced/lambda-0/divs --model-folder=$RESULTS_ROOT/geo/forced/lambda-0/models --data-root=$DATA_ROOT --epochs=500 --batch-size=10 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss-weight=0 --require-ood --seed=3 --dataset=forced_turb --ntrain=5000 --npoints=7000 --res1d=20 --width=64 --modes=10 --div-order=3" 24 geo_forced_turb_n5000_l0_s3
sp "$PYTHON ramansh_3d.py --project-name=geo_div_loss --div-folder=$RESULTS_ROOT/geo/forced/lambda-0/divs --model-folder=$RESULTS_ROOT/geo/forced/lambda-0/models --data-root=$DATA_ROOT --epochs=500 --batch-size=10 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss-weight=0 --require-ood --seed=1 --dataset=forced_turb --ntrain=7000 --npoints=7000 --res1d=20 --width=64 --modes=10 --div-order=3" 32 geo_forced_turb_n7000_l0_s1
sp "$PYTHON ramansh_3d.py --project-name=geo_div_loss --div-folder=$RESULTS_ROOT/geo/forced/lambda-0/divs --model-folder=$RESULTS_ROOT/geo/forced/lambda-0/models --data-root=$DATA_ROOT --epochs=500 --batch-size=10 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss-weight=0 --require-ood --seed=2 --dataset=forced_turb --ntrain=7000 --npoints=7000 --res1d=20 --width=64 --modes=10 --div-order=3" 32 geo_forced_turb_n7000_l0_s2
sp "$PYTHON ramansh_3d.py --project-name=geo_div_loss --div-folder=$RESULTS_ROOT/geo/forced/lambda-0/divs --model-folder=$RESULTS_ROOT/geo/forced/lambda-0/models --data-root=$DATA_ROOT --epochs=500 --batch-size=10 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss-weight=0 --require-ood --seed=3 --dataset=forced_turb --ntrain=7000 --npoints=7000 --res1d=20 --width=64 --modes=10 --div-order=3" 32 geo_forced_turb_n7000_l0_s3
sp "$PYTHON ramansh_3d.py --project-name=geo_div_loss --div-folder=$RESULTS_ROOT/geo/baseline/lambda-0/divs --model-folder=$RESULTS_ROOT/geo/baseline/lambda-0/models --data-root=$DATA_ROOT --epochs=500 --batch-size=10 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss-weight=0 --require-ood --seed=1 --dataset=forced_turb --ntrain=10000 --npoints=7000 --res1d=20 --width=64 --modes=10 --div-order=3" 46 geo_forced_turb_n10000_l0_s1
sp "$PYTHON ramansh_3d.py --project-name=geo_div_loss --div-folder=$RESULTS_ROOT/geo/baseline/lambda-0/divs --model-folder=$RESULTS_ROOT/geo/baseline/lambda-0/models --data-root=$DATA_ROOT --epochs=500 --batch-size=10 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss-weight=0 --require-ood --seed=2 --dataset=forced_turb --ntrain=10000 --npoints=7000 --res1d=20 --width=64 --modes=10 --div-order=3" 46 geo_forced_turb_n10000_l0_s2
sp "$PYTHON ramansh_3d.py --project-name=geo_div_loss --div-folder=$RESULTS_ROOT/geo/baseline/lambda-0/divs --model-folder=$RESULTS_ROOT/geo/baseline/lambda-0/models --data-root=$DATA_ROOT --epochs=500 --batch-size=10 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss-weight=0 --require-ood --seed=3 --dataset=forced_turb --ntrain=10000 --npoints=7000 --res1d=20 --width=64 --modes=10 --div-order=3" 46 geo_forced_turb_n10000_l0_s3
sp "$PYTHON ramansh_3d.py --project-name=geo_div_loss --div-folder=$RESULTS_ROOT/geo/div/lambda-0.001/divs --model-folder=$RESULTS_ROOT/geo/div/lambda-0.001/models --data-root=$DATA_ROOT --epochs=500 --batch-size=10 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=0.001 --no-ood --seed=3 --dataset=forced_turb --ntrain=10000 --npoints=7000 --res1d=20 --width=64 --modes=10 --div-order=3" 46 geo_forced_turb_n10000_l0p001_s3
sp "$PYTHON ramansh_3d.py --project-name=geo_div_loss --div-folder=$RESULTS_ROOT/geo/div/lambda-0.01/divs --model-folder=$RESULTS_ROOT/geo/div/lambda-0.01/models --data-root=$DATA_ROOT --epochs=500 --batch-size=10 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=0.01 --no-ood --seed=2 --dataset=forced_turb --ntrain=10000 --npoints=7000 --res1d=20 --width=64 --modes=10 --div-order=3" 46 geo_forced_turb_n10000_l0p01_s2
sp "$PYTHON ramansh_3d.py --project-name=geo_div_loss --div-folder=$RESULTS_ROOT/geo/div/lambda-0.01/divs --model-folder=$RESULTS_ROOT/geo/div/lambda-0.01/models --data-root=$DATA_ROOT --epochs=500 --batch-size=10 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=0.01 --no-ood --seed=3 --dataset=forced_turb --ntrain=10000 --npoints=7000 --res1d=20 --width=64 --modes=10 --div-order=3" 46 geo_forced_turb_n10000_l0p01_s3
sp "$PYTHON ramansh_3d.py --project-name=geo_div_loss --div-folder=$RESULTS_ROOT/geo/div/lambda-0.1/divs --model-folder=$RESULTS_ROOT/geo/div/lambda-0.1/models --data-root=$DATA_ROOT --epochs=500 --batch-size=10 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=0.1 --no-ood --seed=2 --dataset=forced_turb --ntrain=10000 --npoints=7000 --res1d=20 --width=64 --modes=10 --div-order=3" 46 geo_forced_turb_n10000_l0p1_s2
sp "$PYTHON ramansh_3d.py --project-name=geo_div_loss --div-folder=$RESULTS_ROOT/geo/div/lambda-0.1/divs --model-folder=$RESULTS_ROOT/geo/div/lambda-0.1/models --data-root=$DATA_ROOT --epochs=500 --batch-size=10 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=0.1 --no-ood --seed=3 --dataset=forced_turb --ntrain=10000 --npoints=7000 --res1d=20 --width=64 --modes=10 --div-order=3" 46 geo_forced_turb_n10000_l0p1_s3
sp "$PYTHON ramansh_3d.py --project-name=geo_div_loss --div-folder=$RESULTS_ROOT/geo/div/lambda-1/divs --model-folder=$RESULTS_ROOT/geo/div/lambda-1/models --data-root=$DATA_ROOT --epochs=500 --batch-size=10 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=1 --no-ood --seed=2 --dataset=forced_turb --ntrain=10000 --npoints=7000 --res1d=20 --width=64 --modes=10 --div-order=3" 46 geo_forced_turb_n10000_l1_s2
sp "$PYTHON ramansh_3d.py --project-name=geo_div_loss --div-folder=$RESULTS_ROOT/geo/div/lambda-1/divs --model-folder=$RESULTS_ROOT/geo/div/lambda-1/models --data-root=$DATA_ROOT --epochs=500 --batch-size=10 --lr-fno=1e-3 --lr-phi=1e-4 --wandb --calc-div --save --norm-grid --div-loss --div-loss-weight=1 --no-ood --seed=3 --dataset=forced_turb --ntrain=10000 --npoints=7000 --res1d=20 --width=64 --modes=10 --div-order=3" 46 geo_forced_turb_n10000_l1_s3
