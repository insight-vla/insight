#!/bin/bash
#SBATCH --partition=viscam --qos=normal
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=93G
#SBATCH --account=viscam
#SBATCH --gres=gpu:l40s:2
#SBATCH --job-name="pi05_flip_prims"
#SBATCH --output=${USER_SCRATCH:-$HOME}/Workspace/openpi/slurm_logs/slurm-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${USER}@stanford.edu

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

cd ${USER_SCRATCH:-$HOME}/Workspace/openpi

# Compute norm stats only if not already done
NORM_STATS_FILE="assets/pi05_lego_oracle_flip_140_primitives/maggiewang/lego_oracle_flip_140_primitives_trimmed/norm_stats.json"
if [ ! -f "$NORM_STATS_FILE" ]; then
    echo "Computing norm stats..."
    uv run scripts/compute_norm_stats.py --config-name pi05_lego_oracle_flip_140_primitives
else
    echo "Norm stats already exist, skipping..."
fi

# Train on 2x L40S with FSDP
uv run scripts/train.py pi05_lego_oracle_flip_140_primitives --exp-name=pi05_oracle_flip_140_primitives --batch-size 32 --fsdp-devices 2
