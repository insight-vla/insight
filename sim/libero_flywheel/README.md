# LIBERO simulation flywheel + LIBERO benchmark

This directory hosts InSight's LIBERO-side pipeline plus the upstream LIBERO benchmark example it's built on.

## What's here

- `vlm_flywheel/` — the InSight flywheel for LIBERO: VLM planner, primitive-gap proposer, oracle/completion checks, executor + recording stack. Files: `base_execution.py`, `flywheel_execution.py`, `reasoning.py`, `recording.py`, `prompts.py`, etc.
- `vlm_feedback_flywheel.py` — the unified flywheel entry point. Pick the task via `--args.task lego|drawer`. This is what produces the per-iteration acquisition data for the sim block-flip and drawer-close experiments.
- `test_primitives.py` — the Docker entry point and the script you'd use to run primitive-sequenced rollouts against a served VLA policy. Accepts `--args.task lego|drawer|mug` and various rollout flags.
- `data_processing/densely_label_dataset.py` — automatic primitive segmentation of LIBERO demos (the Stage-1 segmenter).
- `Dockerfile`, `compose.yml`, `requirements.txt` — the LIBERO sim container (LIBERO needs an older Python + Mujoco + PyTorch + GLX/EGL setup, so it lives in its own Docker image).
- `main.py` — a thinner entrypoint kept for the upstream LIBERO benchmark eval below.

## Running the InSight flywheel

```bash
# Start a policy server in one terminal:
uv run training/serve_policy.py policy:checkpoint \
  --policy.config pi05_lego_human_pickplace_flip \
  --policy.dir checkpoints/pi05_lego_human_pickplace_flip/<exp>/<step>

# Run the flywheel in another terminal (block-flip example):
uv run python sim/libero_flywheel/vlm_feedback_flywheel.py \
  --args.task lego \
  --args.seed 0 \
  --args.num-runs 5
```

For drawer-close, pass `--args.task drawer`. See `--help` for the full flag list (acceptance gates, VLM completion check, manual verdict, etc.).

## Running primitive sequences directly (no flywheel)

```bash
uv run python sim/libero_flywheel/test_primitives.py \
  --args.task drawer \
  --args.drawer-start-open
```

This is the Docker entrypoint; useful when you just want to roll out a fixed primitive sequence against a served policy.

---

## LIBERO benchmark eval (upstream openpi example)

This example also includes the upstream openpi LIBERO benchmark eval pipeline: https://github.com/Lifelong-Robot-Learning/LIBERO

Note: When updating `requirements.txt` in this directory, there is an additional flag `--extra-index-url https://download.pytorch.org/whl/cu113` that must be added to the `uv pip compile` command.

This example requires git submodules to be initialized:

```bash
git submodule update --init --recursive
```

## With Docker (recommended)

```bash
# Grant access to the X11 server:
sudo xhost +local:docker

# To run with the default checkpoint and task suite:
SERVER_ARGS="--env LIBERO" docker compose -f sim/libero_flywheel/compose.yml up --build

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx SERVER_ARGS="--env LIBERO" docker compose -f sim/libero_flywheel/compose.yml up --build
```

You can customize the loaded checkpoint by providing additional `SERVER_ARGS` (see `training/serve_policy.py`), and the LIBERO task by providing additional `CLIENT_ARGS` (passed through to `sim/libero_flywheel/test_primitives.py`, which is the Docker entrypoint).
For example:

```bash
# To load a custom checkpoint (path is relative to the repo root):
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir ./my_custom_checkpoint"

# To switch the task ("lego" block-flip / "drawer" / "mug"):
export CLIENT_ARGS="--args.task drawer"
```

## Without Docker (not recommended)

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.8 sim/libero_flywheel/.venv
source sim/libero_flywheel/.venv/bin/activate
uv pip sync sim/libero_flywheel/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# Run the simulation
python sim/libero_flywheel/main.py

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx python sim/libero_flywheel/main.py
```

Terminal window 2:

```bash
# Run the server
uv run training/serve_policy.py --env LIBERO
```

## Results

If you want to reproduce the following numbers, you can evaluate the checkpoint at `gs://openpi-assets/checkpoints/pi05_libero/`. This
checkpoint was trained in openpi with the `pi05_libero` config.

| Model | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
|-------|---------------|---------------|-------------|-----------|---------|
| π0.5 @ 30k (finetuned) | 98.8 | 98.2 | 98.0 | 92.4 | 96.85
