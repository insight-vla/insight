# Installation

InSight uses [`uv`](https://docs.astral.sh/uv/) for environment management.
Python 3.11 is required.

## Prerequisites

- Python **3.11** (3.12+ not supported by the JAX pin)
- CUDA 12 GPU recommended for training and inference (CPU works for sim
  development but is slow)
- For real-robot use: a UFactory xArm 6, two Intel RealSense cameras, and a
  parallel-jaw gripper. See [`safety.md`](safety.md) for safe operation.

## 1. Clone and initialize submodules

```bash
git clone <repo-url> insight
cd insight
git submodule update --init --recursive   # fetches third_party/libero
```

The `third_party/libero` submodule is the
[`libero-insight`](https://github.com/insight-vla/libero-insight) fork
with drawer-primitive BDDLs added on top of upstream LIBERO.

## 2. Main environment

```bash
uv sync                      # core deps (JAX, flax, lerobot, openpi-client, openai, google-auth)
uv sync --group rlds         # (optional) RLDS / DROID dataset support
```

The xArm hardware stack (`pyrealsense2`, `xarm-python-sdk`) is included in
the main `[project.dependencies]` block. If you only intend to run the JAX
training or LIBERO sim pipelines, those hardware deps are still pulled in
but the code paths are not exercised.

## 3. VLM credentials

The default `gemini` provider in `src/insight/vlm_client.py` uses Vertex AI
under the project name `gcp-maggie`. Authenticate with:

```bash
gcloud auth application-default login
# Override the project (or set it once):
export VERTEX_PROJECT=<your-gcp-project>
```

If you'd rather use the direct Gemini API (api key, no GCP), edit
`vlm_client.py` to set the `gemini` provider's `kind` from `"vertex"` to a
public-endpoint variant and then:

```bash
export GEMINI_API_KEY=...
```

For the OpenAI fallback (`set_vlm_provider("gpt")`):

```bash
export OPENAI_API_KEY=...
```

For W&B logging during training (optional):

```bash
export WANDB_API_KEY=...
```

## 4. Sanity check

After install, verify the imports resolve and a minimal entry point loads:

```bash
uv run python -c "from insight import prompts; from openpi.training import config"
uv run real/entry/run_flywheel.py --help | head -20
```

## Notes on the av==14.2.0 pin

`pyproject.toml` pins `av==14.2.0` because PyAV 14.3 / 14.4 are sdist-only
and Ubuntu 22.04 (jammy) ships ffmpeg 4.4 (PyAV 14.3+ needs ffmpeg 7).
Honor the pin even if `uv lock` suggests bumping.

## Where to go next

- [`getting_started.md`](getting_started.md) — first-time workflow walkthrough
- [`safety.md`](safety.md) — required reading for real-robot use
- [`dataset_setup.md`](dataset_setup.md) — where datasets live; how to
  re-build flywheel-curated datasets from raw rollouts
- [`checkpoint_setup.md`](checkpoint_setup.md) — where to put model
  checkpoints
