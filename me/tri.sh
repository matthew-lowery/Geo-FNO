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
#SBATCH --partition=gpuA100x4,gpuA40x4,gpuA100x8
#SBATCH --account=bfbk-delta-gpu
#SBATCH --job-name=myjob
#SBATCH --time=4:00:00
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --output=./tribeee/%x_%A.out
#SBATCH --error=./tribeee/%x_%A.err

module purge
export PATH=/u/mlowery/.conda/envs/gnot/bin:\$PATH
cd /u/mlowery/Geo-FNO/elasticity/
$pycmd
EOF
}
## modes, res1d, width
for seed in 1 2 3 4; do
for w in 64; do
for m in 12; do
sp "python3 -u triangle.py --seed=$seed --width=$w --res1d=30 --modes=$m"
done
done
done

for seed in 1 2 3 4; do
for w in 64; do
for m in 12; do
sp "python3 -u triangle.py --seed=$seed --width=$w --res1d=25 --modes=$m"
done
done
done

