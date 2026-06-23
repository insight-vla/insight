#!/bin/bash
#SBATCH --partition=viscam,svl --qos=normal
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=250G
#SBATCH --account=viscam
#SBATCH --gres=gpu:4
#SBATCH --exclude=viscam-hgx-2,viscam5,viscam1,viscam15,viscam14,viscam13,viscam10,svl5,svl1,svl6,viscam9,viscam-hgx-1,svl4,visionlab-dgx1,svl2,viscam12,svl3,viscam11
#SBATCH --job-name="pi05_libero_lora"
#SBATCH --output=${USER_SCRATCH:-$HOME}/Workspace/openpi/slurm_logs/slurm-%j.out
#SBATCH --mail-type=END,FAIL
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
NORM_STATS_FILE="assets/pi05_libero_lora/physical-intelligence/libero/norm_stats.json"
if [ ! -f "$NORM_STATS_FILE" ]; then
    echo "Computing norm stats..."
    uv run scripts/compute_norm_stats.py --config-name pi05_libero_lora
else
    echo "Norm stats already exist, skipping..."
fi

# Run training with reduced batch size and FSDP across 4 GPUs
uv run scripts/train.py pi05_libero_lora --exp-name=pi05_libero_lora_run1 --overwrite --batch-size 16 --fsdp-devices 4
