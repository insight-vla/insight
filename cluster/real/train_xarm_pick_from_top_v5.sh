#!/bin/bash
#SBATCH --partition=viscam --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --account=viscam
#SBATCH --gres=gpu:h200:1
#SBATCH --job-name="pick_from_top_v5_h200"
#SBATCH --output=${USER_SCRATCH:-$HOME}/Workspace/openpi/slurm_logs/slurm-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${USER}@stanford.edu

# xArm top-grasp pickplace v4 (50 teleop demos): base policy for twist flywheel.

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

NORM_STATS_FILE="assets/xarm_pick_from_top_v5/maggie/xarm_pick_from_top_v5_primitives_trimmed/norm_stats.json"
if [ ! -f "$NORM_STATS_FILE" ]; then
    uv run scripts/compute_norm_stats.py --config-name xarm_pick_from_top_v5
fi

uv run scripts/train.py xarm_pick_from_top_v5 \
    --exp-name=xarm_pick_from_top_v5_h200 \
    --batch-size 64 \
    --fsdp-devices 1 \
    --overwrite
