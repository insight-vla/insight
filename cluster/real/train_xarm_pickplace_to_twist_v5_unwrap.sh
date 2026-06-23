#!/bin/bash
#SBATCH --partition=viscam --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --account=viscam
#SBATCH --gres=gpu:h200:1
#SBATCH --job-name="pickplace_to_twist_v5_unwrap_h200"
#SBATCH --output=${USER_SCRATCH:-$HOME}/Workspace/openpi/slurm_logs/slurm-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${USER}@stanford.edu

# xArm pickplace + flywheel-acquired twist (v5): top-grasp pickplace v5 base
# (50 teleop demos with bowl matched to the pour/twist eval scene) + 20
# flywheel-collected twist demos (label-normalized to "twist open the cap").
# v5 successor to xarm_pickplace_to_twist_v4_unwrap.

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

NORM_STATS_FILE="assets/xarm_pickplace_to_twist_v5_unwrap/maggie/xarm_pickplace_to_twist_v5_unwrap/norm_stats.json"
if [ ! -f "$NORM_STATS_FILE" ]; then
    uv run scripts/compute_norm_stats.py --config-name xarm_pickplace_to_twist_v5_unwrap
fi

uv run scripts/train.py xarm_pickplace_to_twist_v5_unwrap \
    --exp-name=xarm_pickplace_to_twist_v5_unwrap_h200 \
    --batch-size 64 \
    --fsdp-devices 1 \
    --overwrite
