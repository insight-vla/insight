#!/bin/bash
#SBATCH --partition=viscam --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --account=viscam
#SBATCH --gres=gpu:h200:1
#SBATCH --job-name="pickplace_pour_05_10_h200"
#SBATCH --output=${USER_SCRATCH:-$HOME}/Workspace/openpi/slurm_logs/slurm-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${USER}@stanford.edu

# xArm side-grasp pickplace + flywheel-acquired pour: side-grasp pickplace
# base (xarm_pick_from_side_v5, 50 demos) + 20 flywheel-collected pour trials
# (40 episodes: pour-forward + return-upright, label-normalized). Validates
# the flywheel: policy absorbs new bootstrapped primitives without losing
# the originals.

echo "SLURM_JOBID="$SLURM_JOBID
set -x

export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_API_KEY="${WANDB_API_KEY:?set WANDB_API_KEY in env}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PATH=${USER_HOME:-$HOME}/.local/bin:$PATH
export HF_HOME=${USER_SCRATCH:-$HOME}/.cache/huggingface
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN in env}"
export OPENPI_DATA_HOME=${USER_SCRATCH:-$HOME}/.cache/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export UV_CACHE_DIR=${USER_SCRATCH:-$HOME}/.cache/uv

cd ${USER_SCRATCH:-$HOME}/Workspace/openpi

NORM_STATS_FILE="assets/xarm_pickplace_pour_05_10/maggie/xarm_pickplace_pour_05_10/norm_stats.json"
if [ ! -f "$NORM_STATS_FILE" ]; then
    uv run scripts/compute_norm_stats.py --config-name xarm_pickplace_pour_05_10
fi

uv run scripts/train.py xarm_pickplace_pour_05_10 \
    --exp-name=xarm_pickplace_pour_05_10_h200 \
    --batch-size 64 \
    --fsdp-devices 1 \
    --overwrite
