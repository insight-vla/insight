#!/bin/bash
#SBATCH --partition=viscam --qos=normal
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=viscam
#SBATCH --gres=gpu:1
#SBATCH --job-name="benchmark"
#SBATCH --output=${USER_SCRATCH:-$HOME}/Workspace/openpi/slurm_logs/benchmark-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@stanford.edu

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST

cd ${USER_SCRATCH:-$HOME}/Workspace/openpi

# Fix permissions for Docker container to read files
chmod -R 755 third_party/libero/

# Build Docker image (rebuild to pick up Dockerfile changes)
echo "Building Docker image..."
docker build -t libero -f examples/libero/Dockerfile .

# Config to benchmark (change these)
# CONFIG="pi05_lego_primitives_v6"
# EXP_NAME="pi05_lego_v6_02_03"
# STEP="15000"
CONFIG="pi05_red_blue_v6"
EXP_NAME="pi05_red_blue_v6_02_03"
STEP="5000"

CHECKPOINT_PATH="checkpoints/${CONFIG}/${EXP_NAME}/${STEP}"

echo "Benchmarking config: $CONFIG"
echo "Checkpoint: $CHECKPOINT_PATH"

# Environment setup for policy server
export HF_HOME=${USER_SCRATCH:-$HOME}/.cache/huggingface
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN in env}"
export OPENPI_DATA_HOME=${USER_SCRATCH:-$HOME}/.cache/openpi
export PATH=${USER_HOME:-$HOME}/.local/bin:$PATH

# Start policy server in background (runs on host with GPU)
echo "Starting policy server..."
uv run scripts/serve_policy.py --env LIBERO policy:checkpoint --policy.config $CONFIG --policy.dir $CHECKPOINT_PATH &
SERVER_PID=$!

# Wait for server to initialize
echo "Waiting for server to start..."
sleep 120

# Run benchmark in Docker container (LIBERO environment)
echo "Running benchmark in Docker..."
docker run --rm \
    --network=host \
    --gpus all \
    --user $(id -u):$(id -g) \
    -v ${USER_SCRATCH:-$HOME}/Workspace/openpi:/app \
    -e LIBERO_CONFIG_PATH=/tmp/libero \
    -e MUJOCO_GL=egl \
    -e PYOPENGL_PLATFORM=egl \
    -w /app \
    libero \
    /.venv/bin/python scripts/benchmark_primitives.py --num_episodes 10 --output_dir /app/data/libero/benchmarks --save_videos

# Cleanup
echo "Cleaning up..."
kill $SERVER_PID 2>/dev/null

echo "Done!"
