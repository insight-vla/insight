# Real (xArm) pipeline

The hardware pipeline for the InSight paper. Targets a **UFactory xArm 6** with
two RealSense cameras (exterior + wrist) and a parallel-jaw gripper.

> **Read [`../docs/safety.md`](../docs/safety.md) before powering on the arm.**

## Layout

- **`xarm_flywheel/`** &mdash; the planner + executor + oracle + hardware stack.
  - `args.py` &mdash; `FlywheelArgs` dataclass (tyro-driven CLI).
  - `runner.py` &mdash; rollout loop: known primitives via the VLA, skill gaps
    via the scripted single-axis controller.
  - `executor.py` &mdash; subclass of the `BaseExecutor` ABC in
    `src/insight/executor.py`. Per-primitive termination logic
    (progress-threshold + auto-advance + VLM completion check).
  - `hardware.py` &mdash; xArm SDK wrapper + RealSense capture + workspace + TCP
    speed clipping.
  - `oracle.py` &mdash; before/after VLM verdict.
  - `recording.py` &mdash; LeRobot-format dataset writer for accepted rollouts.
  - `skill_gap.py` &mdash; PREANALYZE pre-execution VLM call + chained-gap
    parameter passing.
  - `video.py` &mdash; rollout video composition (used for paper figures).
  - `main.py` &mdash; the actual `run(args)` function called by the tyro shim.
- **`entry/`** &mdash; tyro CLI shims and miscellaneous test scripts
  (`run_flywheel.py`, `run_primitives.py`, `inference*.py`,
  `collect_demo*.py`, `vlm_check.py`, `measure_bounds.py`).
- **`runs/`** &mdash; paper-experiment launch scripts.
- **`calibration/`** &mdash; `bounds_sweep.json` from the most recent workspace
  calibration.

## Always use `uv run`

Project memory `feedback_openpi_uv` &mdash; the openpi env is uv-managed and
will not work under raw `python3`. Always:

```bash
uv run python real/entry/run_flywheel.py ...
```

## Paper experiments

| Experiment            | Launch script                              | Notes                                                      |
| --------------------- | ------------------------------------------ | ---------------------------------------------------------- |
| Base top pickplace    | `runs/run_base_top_pickplace.sh`           | Sanity-check base skill retention                          |
| Base side pickplace   | `runs/run_base_side_pickplace.sh`          | Sanity-check base skill retention                          |
| Twist (acquisition)   | `runs/run_twist_pre_vla.sh`                | Base top-pickplace + scripted twist gap                    |
| Twist (evaluation)    | `runs/run_twist_post_vla.sh`               | Retrained VLA, twist now a known primitive                 |
| Pour (acquisition)    | `runs/run_pour_pre_vla.sh`                 | Two single-axis rotations (tilt forward + tilt back)       |
| Pour (evaluation)     | `runs/run_pour_post_vla.sh`                | Retrained VLA, pour now known                              |
| Twist-then-pour       | `runs/run_unified_twist_pour.sh`           | 14-primitive composition, unified VLA                      |

The **sweep** experiment uses scoop demos as base and acquires the lateral-push
primitive. The unified-skill SLURM training script for the resulting policy
lives at [`../cluster/real/train_scoop_to_sweep.sh`](../cluster/real/train_scoop_to_sweep.sh);
acquisition itself is launched via `entry/run_flywheel.py` with `--goal "sweep
the rocks"` and the scoop primitives passed through `--available-primitives`.

## Required VLM credentials

Every flywheel path goes through Gemini 3 Flash. The default `gemini` provider
uses Vertex AI: authenticate with `gcloud auth application-default login` (or
set `GOOGLE_APPLICATION_CREDENTIALS`) and override the project if needed via
`export VERTEX_PROJECT=<your-gcp-project>`. To use a direct Gemini API key
instead, edit `src/insight/vlm_client.py` to point at the public Gemini
endpoint and set `GEMINI_API_KEY`.

## Hardware-specific config

- Workspace bounds: [`calibration/bounds_sweep.json`](calibration/bounds_sweep.json).
  Override per-experiment with `--workspace-bounds`.
- Max TCP speed: pass `--max-tcp-speed-mm-s` (paper used 90 for most experiments,
  100 for twist). The `args.py` default is 0 (no override), which falls back to
  the 120 mm/s ceiling enforced in `hardware.py`.
- Camera serials are **hardcoded** in `real/xarm_flywheel/hardware.py` as
  `SERIAL_EXTERNAL` and `SERIAL_WRIST` (D435i exterior + wrist). Edit those
  constants for your own RealSense devices.

## Recording datasets

Acquisition scripts pass `--record-dataset-repo <hf-repo-id>` to write
accepted rollouts to a LeRobot dataset under
`$HF_LEROBOT_HOME/<repo-id>/` (defaults to `~/.cache/huggingface/lerobot/<repo-id>/`).
See [`../docs/dataset_setup.md`](../docs/dataset_setup.md).
