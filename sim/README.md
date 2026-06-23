# Sim pipeline (LIBERO)

This directory contains the simulation pipeline for the InSight paper:

- **`libero_flywheel/`** &mdash; the LIBERO flywheel runner. The unified entry
  is `vlm_feedback_flywheel.py`. It supports two tasks:
  - `--args.task lego` &mdash; block-flip from pick-and-place demos (paper
    experiment 1, Figure 3). Acquires the rotate-block primitive.
  - `--args.task drawer` &mdash; drawer-close from drawer-open demos (paper
    experiment 2). Acquires the push-drawer-closed primitive.
- **`libero_flywheel/data_processing/`** &mdash; dataset-labelling pipeline
  (Stage 1): primitive boundary detection, motion-caption generation,
  VLM-driven frame alignment.
- **`libero_flywheel/vlm_flywheel/`** &mdash; the executor + planner + oracle
  + recording stack.

> **Submodule required.** The sim pipeline depends on the
> [`libero-insight`](https://github.com/insight-vla/libero-insight) fork of
> LIBERO, which adds drawer-primitive BDDLs. Initialize via
> `git submodule update --init --recursive`; after init, the fork lives at
> `third_party/libero/`.

## Quickstart

1. Start a policy server pointing at a trained checkpoint:

   ```bash
   uv run python training/serve_policy.py policy:checkpoint \
     --policy.config pi05_lego_human_pickplace_flip \
     --policy.dir checkpoints/pi05_lego_human_pickplace_flip/my_run/30000
   ```

2. In another terminal, run the flywheel:

   ```bash
   uv run python sim/libero_flywheel/vlm_feedback_flywheel.py \
     --args.task lego \
     --args.seed 0 \
     --args.num_runs 50 \
     --args.target_successes 30 \
     --args.vlm gemini \
     --args.record
   ```

