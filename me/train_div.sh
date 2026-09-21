#!/bin/bash
set -euo pipefail

ME_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${RAM_DATA_ROOT:-$ME_DIR/../../ram_dataset}"
if [[ -z "${RAM_DATA_ROOT:-}" && -d /projects/bgcs/mlowery/ram_dataset ]]; then
    DATA_ROOT=/projects/bgcs/mlowery/ram_dataset
fi
RESULTS_ROOT="${RAM_RESULTS_ROOT:-/projects/bgcs/mlowery/operator-benchmarks}"
PYTHON="${TRAIN_PYTHON:-/u/mlowery/.conda/envs/gnot/bin/python}"
mode=dry-run
phase_filter=all
model_filter=all
jobs=0
remaining=false
geo_ood_missing=false
dataset_filter=all
seed_filter=all
size_filter=all
original_arguments=("$@")

usage() {
    echo "Usage: bash train_div.sh [--dry-run|--check|--submit] [--remaining|--geo-ood-missing] [--dataset NAME] [--seed N] [--ntrain N] [--phase all|div|baseline|forced] [--model all|geo|trans]"
    echo "Overrides: RAM_DATA_ROOT, RAM_RESULTS_ROOT, TRAIN_PYTHON"
}

while (( $# )); do
    case "$1" in
        --dry-run) mode=dry-run; shift ;;
        --submit) mode=submit; shift ;;
        --check) mode=check; shift ;;
        --remaining) remaining=true; shift ;;
        --geo-ood-missing) geo_ood_missing=true; shift ;;
        --dataset) dataset_filter="${2:?missing dataset}"; shift 2 ;;
        --seed) seed_filter="${2:?missing seed}"; shift 2 ;;
        --ntrain) size_filter="${2:?missing ntrain}"; shift 2 ;;
        --phase) phase_filter="${2:?missing phase}"; shift 2 ;;
        --model) model_filter="${2:?missing model}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
case "$phase_filter" in all|div|baseline|forced) ;; *) usage >&2; exit 2 ;; esac
case "$model_filter" in all|geo|trans) ;; *) usage >&2; exit 2 ;; esac
if [[ "$remaining" == true ]]; then
    RESULTS_ROOT="${RAM_RESULTS_ROOT:-/projects/bgcs/mlowery/operator-benchmarks/rerun-20260914}"
fi
if [[ "$geo_ood_missing" == true ]]; then
    [[ "$remaining" == false ]] || { echo "Choose one rerun selection" >&2; exit 2; }
    [[ "$model_filter" == all || "$model_filter" == geo ]] || { echo "Geo-FNO only" >&2; exit 2; }
    [[ "$phase_filter" == all || "$phase_filter" == baseline ]] || { echo "No-div baselines only" >&2; exit 2; }
    model_filter=geo
    phase_filter=baseline
    RESULTS_ROOT="${RAM_RESULTS_ROOT:-/projects/bgcs/mlowery/operator-benchmarks/geo-ood-recovery-20260916}"
fi

# dataset, Geo entry, training count, points, Geo hours/batch/resolution/width/modes/order,
# Transolver hours/width/layers/heads/slices.
# Transolver forced turbulence has no reference entry: use its 3D species model
# settings. Longer allocations include headroom beyond measured epoch timings.
profiles() {
    cat <<'PROFILES'
flow_cylinder_laminar ramansh_2d.py 100 1000 2 20 60 128 24 3 3 128 5 8 32
flow_cylinder_shedding ramansh_2d.py 10000 1000 4 20 60 64 28 3 3 128 5 4 32
lid_cavity_flow ramansh_2d.py 10000 1000 2 20 40 64 20 3 3 128 5 4 16
backward_facing_step ramansh_2d.py 500 1000 2 20 40 64 12 3 3 128 5 8 16
buoyancy_cavity_flow ramansh_2d.py 10000 5000 8 20 40 64 20 3 8 128 5 4 32
taylor_green ramansh_2d.py 5000 500 2 20 50 64 20 2 3 128 5 4 32
taylor_green_coeffs ramansh_2d_diff_grids.py 5000 500 2 20 50 64 20 2 3 128 5 4 32
taylor_green_spacetime ramansh_3d.py 5000 500 2 20 15 64 7 2 3 128 5 4 32
taylor_green_spacetime_coeffs ramansh_3d.py 5000 500 2 20 15 64 7 2 3 128 5 4 32
merge_vortices_easier ramansh_2d.py 500 500 2 20 60 128 12 2 3 128 5 8 64
species_transport ramansh_3d.py 10000 7000 32 20 20 64 10 3 12 128 5 4 32
forced_turb ramansh_3d.py 10000 7000 36 10 20 64 10 3 24 128 5 4 32
PROFILES
}

data_files_exist() {
    local phase="$1" model="$2" suffix directory filename path
    local -a required
    for suffix in '' _ood; do
        [[ "$phase" != div || -z "$suffix" ]] || continue
        directory="$dataset"
        filename="data${suffix}.mat"
        case "$dataset" in
            flow_cylinder_laminar) directory=flow_cylinder; filename="data_laminar${suffix}.mat" ;;
            flow_cylinder_shedding) directory=flow_cylinder; filename="data_shedding${suffix}.mat" ;;
            taylor_green) directory=taylor_green; filename="data_ood.mat"; [[ -n "$suffix" ]] || filename=data_exact_matt.mat ;;
            taylor_green_coeffs) directory=taylor_green; filename="data_coeffs_ood.mat"; [[ -n "$suffix" ]] || filename=data_coeffs_matt.mat ;;
            taylor_green_spacetime*) directory=taylor_green; filename="data_time${suffix}.mat" ;;
        esac
        required=("$DATA_ROOT/$directory/$filename")
        if [[ "$dataset" == taylor_green_spacetime_coeffs ]]; then
            filename=data_coeffs_ood.mat
            [[ -n "$suffix" ]] || filename=data_coeffs_matt.mat
            required+=("$DATA_ROOT/taylor_green/$filename")
        fi
        for path in "${required[@]}"; do
            if [[ ! -f "$path" ]]; then
                echo "Skip $phase $model $dataset seed=$seed ntrain=$size lambda=$coef: missing $path" >&2
                return 1
            fi
        done
    done
}

launch() {
    local phase="$1" model="$2" seed="$3" coef="$4" size="$5"
    [[ "$dataset" != species_transport || "$phase" != div ]] || return 0
    [[ "$model_filter" == all || "$model_filter" == "$model" ]] || return 0
    [[ "$dataset_filter" == all || "$dataset_filter" == "$dataset" ]] || return 0
    [[ "$seed_filter" == all || "$seed_filter" == "$seed" ]] || return 0
    [[ "$size_filter" == all || "$size_filter" == "$size" ]] || return 0
    if [[ "$geo_ood_missing" == true ]]; then
        case "$dataset" in
            forced_turb|species_transport|buoyancy_cavity_flow) return 0 ;;
        esac
    fi
    if [[ "$remaining" == true ]]; then
        case "$dataset:$model:$phase" in
            forced_turb:*:*|species_transport:*:baseline|buoyancy_cavity_flow:geo:baseline|buoyancy_cavity_flow:trans:div|buoyancy_cavity_flow:trans:baseline) ;;
            *) return 0 ;;
        esac
    fi
    data_files_exist "$phase" "$model" || return 0
    local hours label result_dir
    local -a command
    label="${model}_${dataset}_s${seed}_n${size}_l${coef}"
    [[ "$remaining" != true ]] || label="rerun_${label}"
    [[ "$geo_ood_missing" != true ]] || label="oodfix_${label}"
    result_dir="$RESULTS_ROOT/$model/$phase/lambda-$coef"
    if [[ "$model" == geo ]]; then
        hours="$geo_hours"
        if [[ "$geo_ood_missing" == true ]]; then
            case "$dataset" in
                flow_cylinder_shedding) hours=6 ;;
                lid_cavity_flow|taylor_green_spacetime) hours=4 ;;
                taylor_green) hours=3 ;;
            esac
        fi
        command=("$PYTHON" "$geo_entry" "--batch-size=$geo_batch"
                 "--lr-fno=1e-3" "--lr-phi=1e-4" "--res1d=$resolution"
                 "--width=$width" "--modes=$modes" "--div-order=$geo_order")
    else
        hours="$trans_hours"
        command=("$PYTHON" -m transolver.train "--batch-size=20"
                 "--lr=1e-3" "--weight_decay=1e-5" "--n-hidden=$trans_width"
                 "--n-layers=$layers" "--n-heads=$heads" "--slice-num=$slices"
                 "--div-order=4" "--gpu=0")
    fi
    if [[ "$phase" == forced ]]; then
        hours=$(( (hours * size + 9999) / 10000 + 1 ))
    fi
    command+=("--dataset=$dataset" "--ntrain=$size" "--npoints=$points"
              "--seed=$seed" "--data-root=$DATA_ROOT" "--epochs=500"
              "--project-name=${model}_div_loss" "--div-loss-weight=$coef"
              "--div-folder=$result_dir/divs" "--model-folder=$result_dir/models"
              --wandb --calc-div --save --norm-grid)
    if [[ "$phase" == div ]]; then
        command+=(--div-loss --no-ood)
    else
        command+=(--require-ood)
    fi
    jobs=$((jobs + 1))
    if [[ "$mode" == dry-run ]]; then
        printf '%s %s %s hours=%s ' "$phase" "$model" "$label" "$hours"
        printf '%q ' "${command[@]}"
        printf '\n'
        return
    fi
    {
        printf '#!/bin/bash\n'
        printf '#SBATCH --mem=32g\n#SBATCH --nodes=1\n#SBATCH --ntasks-per-node=1\n'
        printf '#SBATCH --cpus-per-task=1\n#SBATCH --gpus-per-node=1\n'
        printf '#SBATCH --partition=gpuA100x4\n#SBATCH --account=bgcs-delta-gpu\n'
        printf '#SBATCH --constraint=scratch\n'
        printf '#SBATCH --job-name=%s\n#SBATCH --time=%s:00:00\n' "$label" "$hours"
        printf '#SBATCH --output=%s/out/%%x_%%j.out\n' "$ME_DIR"
        printf '#SBATCH --error=%s/err/%%x_%%j.err\n' "$ME_DIR"
        printf 'set -euo pipefail\nmodule purge\ncd %q\n' "$ME_DIR"
        printf 'export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\n'
        printf '%q ' "${command[@]}"
        printf '\n'
    } | sbatch
}

if [[ "$mode" == submit || "$mode" == check ]]; then
    [[ -d "$DATA_ROOT" ]] || { echo "Dataset root missing: $DATA_ROOT" >&2; exit 1; }
    [[ -x "$PYTHON" ]] || { echo "Python missing: $PYTHON (set TRAIN_PYTHON)" >&2; exit 1; }
    "$PYTHON" "$ME_DIR/check_training_plan.py" < <(
        bash "$ME_DIR/train_div.sh" "${original_arguments[@]}" --dry-run
    )
    [[ "$mode" != check ]] || exit 0
    command -v sbatch >/dev/null
    mkdir -p "$ME_DIR/out" "$ME_DIR/err"
fi

for phase in div baseline forced; do
    [[ "$phase_filter" == all || "$phase_filter" == "$phase" ]] || continue
    for seed in 1 2 3; do
        coefficients=(0)
        [[ "$phase" != div ]] || coefficients=(0.001 0.01 0.1 1)
        for coef in "${coefficients[@]}"; do
            while read -r dataset geo_entry training points geo_hours geo_batch resolution width modes geo_order trans_hours trans_width layers heads slices; do
                [[ "$phase" != forced || "$dataset" == forced_turb ]] || continue
                sizes=("$training")
                [[ "$phase" != forced ]] || sizes=(100 500 1000 5000 7000)
                for size in "${sizes[@]}"; do
                    for model in geo trans; do
                        launch "$phase" "$model" "$seed" "$coef" "$size"
                    done
                done
            done < <(profiles)
        done
    done
done
echo "$mode: $jobs jobs" >&2
