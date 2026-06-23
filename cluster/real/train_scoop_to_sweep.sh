#!/bin/bash
#SBATCH --partition=viscam --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --account=viscam
#SBATCH --gres=gpu:h200:1
#SBATCH --job-name="scoop_to_sweep"
#SBATCH --output=${USER_SCRATCH:-$HOME}/Workspace/xarm-openpi/slurm_logs/slurm-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${USER}@stanford.edu

# Train pi05 on the merged scoop+sweep dataset (50 episodes per task).
# Tests whether the existing scoop primitives can be extended with a new
# sweep primitive bootstrapped via the flywheel — single combined policy,
# trained from pi05 BASE (not from the existing 7D scoop checkpoint, since
# we truncated A's actions to 6D in the merge to match the flywheel format).

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$TMPDIR
set -x
echo "working directory = "$SLURM_SUBMIT_DIR

export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_API_KEY="${WANDB_API_KEY:?set WANDB_API_KEY in env}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PATH=${USER_HOME:-$HOME}/.local/bin:$PATH
export HF_HOME=${USER_SCRATCH:-$HOME}/.cache/huggingface
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN in env}"
export OPENPI_DATA_HOME=${USER_SCRATCH:-$HOME}/.cache/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

cd ${USER_SCRATCH:-$HOME}/Workspace/xarm-openpi

# Compute norm stats only if not already done.
NORM_STATS_FILE="assets/xarm_scoop_to_sweep_50_04_28/maggie/xarm_scoop_to_sweep_50_04_28/norm_stats.json"
if [ ! -f "$NORM_STATS_FILE" ]; then
    echo "Computing norm stats..."
    uv run scripts/compute_norm_stats.py --config-name xarm_scoop_to_sweep_50_04_28
else
    echo "Norm stats already exist, skipping..."
fi

# Train on 1x H200, batch 64. Single device so no FSDP sharding needed.
uv run scripts/train.py xarm_scoop_to_sweep_50_04_28 \
    --exp-name=xarm_scoop_to_sweep_50_04_28_h200_ah10_xarmrepo \
    --batch-size 64 \
    --fsdp-devices 1
