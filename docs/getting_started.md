# Getting started

A short, end-to-end walkthrough that takes you from a fresh clone to running
one of the paper experiments. For deeper material, see
[`installation.md`](installation.md), [`safety.md`](safety.md),
[`dataset_setup.md`](dataset_setup.md), and
[`checkpoint_setup.md`](checkpoint_setup.md).

## 1. Prerequisites

- Python 3.11 (Python 3.12+ is not supported &mdash; some pinned deps don't have
  wheels yet).
- [`uv`](https://docs.astral.sh/uv/) for environment management.
- A CUDA-12 capable GPU for training and inference (sim eval can be CPU-only
  with a remote policy server).
- Vertex AI credentials (`gcloud auth application-default login`) for the
  default Gemini provider. Optional but required for the flywheel itself.

## 2. Clone and install

```bash
git clone <repo-url> insight
cd insight
git submodule update --init --recursive    # fetches libero-insight

uv sync                  # core env (JAX, flax, lerobot, openpi-client, hardware drivers, openai, google-auth)
```

## 3. Set credentials

```bash
gcloud auth application-default login   # for the default Vertex AI Gemini provider
export VERTEX_PROJECT=<your-gcp-project> # override the project name baked into vlm_client.py
export WANDB_API_KEY=...                 # for training runs
export HF_TOKEN=...                      # for huggingface dataset access
```

If you prefer a `.env` file (gitignored), `python-dotenv` is in the
dependencies, but the training and serving entry points do **not** call
`load_dotenv` automatically — either `source .env` in your shell or invoke
`load_dotenv()` yourself.

## 4. Fetch a checkpoint

Pre-trained pi-0.5 checkpoints and InSight paper checkpoints are not bundled
with this repository. The canonical location is the cluster path documented in
[`checkpoint_setup.md`](checkpoint_setup.md):
`/viscam/projects/insight/checkpoints/<exp>/<run>/<step>/`. A HuggingFace Hub
mirror will be linked here when available.

Place checkpoints under `checkpoints/` (gitignored) following:

```
checkpoints/<config_name>/<exp_name>/<step>/
```

For example: `checkpoints/xarm_pick_from_top_v5/xarm_pick_from_top_v5_h200/15000/`.

## 5. Start a policy server

```bash
uv run python training/serve_policy.py policy:checkpoint \
  --policy.config xarm_pick_from_top_v5 \
  --policy.dir checkpoints/xarm_pick_from_top_v5/xarm_pick_from_top_v5_h200/15000
```

> The `policy:checkpoint` subcommand must come **before** any `--policy.*`
> flags &mdash; this is a tyro requirement.

## 6. Run a sim experiment

In a second terminal, run the LIBERO sim flywheel against the policy server:

```bash
# Block-flip (acquires rotate-block primitive)
uv run python sim/libero_flywheel/vlm_feedback_flywheel.py \
  --args.task lego --args.seed 0 \
  --args.num_runs 50 --args.target_successes 30 \
  --args.vlm gemini --args.record

# Drawer-close
uv run python sim/libero_flywheel/vlm_feedback_flywheel.py \
  --args.task drawer --args.seed 0 \
  --args.num_runs 50 --args.target_successes 30 \
  --args.vlm gemini --args.record
```

## 7. Run a real-hardware experiment

> Read [`safety.md`](safety.md) first. Keep an e-stop in hand.

```bash
# Pre-VLA twist acquisition (uses base top-pickplace policy, planner flags
# `twist` as a skill gap and runs the scripted single-axis controller).
bash real/runs/run_twist_pre_vla.sh
```

Once you have collected the new primitive data, preprocess and retrain:

```bash
uv run python training/preprocess/filter_normalize_twist.py
uv run python training/preprocess/merge_pickplace_twist.py
uv run python training/compute_norm_stats.py --config-name xarm_pickplace_to_twist_v5_unwrap
uv run python training/train.py xarm_pickplace_to_twist_v5_unwrap --exp-name=twist_post_vla --batch-size 32
```

Then evaluate with the retrained checkpoint via `run_twist_post_vla.sh`.

## 8. Where to go next

- Sim: [`sim/README.md`](../sim/README.md) (when present) + the experiments
  section of the top-level [`README.md`](../README.md).
- Real: [`real/README.md`](../real/README.md) (when present) +
  [`safety.md`](safety.md).
- Training internals: `src/openpi/training/sim_configs.py` and
  `src/openpi/training/xarm_configs.py` define every registered config.
