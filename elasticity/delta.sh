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
#SBATCH --partition=gpuH200x8
#SBATCH --account=bfbk-delta-gpu
#SBATCH --job-name=myjob
#SBATCH --time=12:00:00
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --output=./diffrrr/%x_%A.out
#SBATCH --error=./diffrrr/%x_%A.err

module purge
export PATH=/u/mlowery/.conda/envs/gnot/bin:\$PATH
cd /u/mlowery/Geo-FNO/elasticity/
$pycmd
EOF
}
## modes, res1d, width
for s in 1 2 3 4; do
for w in 64; do
for m in 6; do
sp "python3 -u diffr.py --seed=$s --width=$w --res1d=12 --modes=$m"
sp "python3 -u diffr.py --seed=$s --width=$w --res1d=10 --modes=$m"
done
done
done

