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


j
for dataset in "flow_cylinder_shedding" "flow_cylinder_laminar" "taylor_green_numerical" "taylor_green_exact" "backward_facing_step" "buoyancy_cavity_flow" "lid_cavity_flow" "merge_vortices"; do
for width in 16 32 64 128; do
sp "python3 -u ramansh.py --dataset=$dataset --norm-grid --width=$width --res1d=30"
done
for s in 30 40 50 60; do
sp "python3 -u ramansh.py --dataset=$dataset --modes=$((s/2)) --norm-grid --width=32 --res1d=30"
done
sp "python3 -u ramansh.py --dataset=$dataset --width=16 --res1d=30"
done

