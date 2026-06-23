# Safety notes — InSight on a real arm

InSight executes VLM-proposed motions on hardware. The defaults in this repo
were tuned for a **UFactory xArm 6** + parallel-jaw gripper + two RealSense
cameras (exterior + wrist). This document covers the **non-obvious** software
knobs and failure modes — not the obvious things (e-stop, clear workspace,
supervise the run).

> Research code, not safety-certified. Re-derive defaults for your setup.

---

## Software defaults you can change (or get wrong)

| Knob | Default | Where | Notes |
|---|---|---|---|
| Workspace bounds | empty (disabled) | `--workspace-bounds <xmin xmax ymin ymax zmin zmax>` (mm); `real/calibration/bounds_sweep.json` is a stored reference from the last `measure_bounds.py` run | The `args.py` default is empty, which means **no clipping**. You must pass `--workspace-bounds` to activate it. Twist uses `200 500 -230 170 200 450`. Always pass the **most conservative** bounds that still contain the task volume — this is the last line of defense against a runaway VLM-proposed magnitude. Re-measure via `real/entry/measure_bounds.py`. |
| TCP speed | 0 (disabled — falls back to 120 mm/s ceiling in `hardware.py`) | `--max-tcp-speed-mm-s`; clipped in `real/xarm_flywheel/hardware.py` | Paper scripts pass 90 mm/s for most experiments, 100 mm/s for twist. Applies to both acquisition (scripted single-axis) and eval (VLA chunks). Raise incrementally. |
| Interactive gates | varies | `--manual-verdict`, `--confirm-plan`, `--confirm-skill-gap` | Used in `run_twist_pre_vla.sh`. Pause before motion to sanity-check the planner; accept/reject each rollout interactively. |
| Progress warmup | off | `--progress-warmup-steps` | Avoids post-twist insta-fire of the progress channel (see [Known failure modes](#known-failure-modes-specific-to-insight)). |
| Gimbal-lock workarounds | off | opt-in flags | Slerp smoothing + gripper-only freeze + axis-angle SDK path. Turn on when the arm refuses to follow a rotation through ±180°. |

## Known failure modes specific to InSight

- **VLM proposes a hallucinated magnitude.** PREANALYZE asks for a signed
  magnitude in metres or degrees. Order-of-magnitude errors have been observed.
  The workspace clip is the only safeguard — don't weaken it.
- **Post-twist progress channel insta-fires.** The progress regressor is OOD
  on post-twist states. Gate with `--progress-warmup-steps 20` after twist.
- **Sweep / scoop are non-prehensile and contact-rich.** The end-effector
  presses into the tray. Use a low gripper force limit.
- **The oracle is a VLM too.** Wrong &gt;1% of the time. Spot-check eval
  numbers manually; for acquisition use `--manual-verdict` to override.
- **Camera shift silently degrades plans.** The PREANALYZE/oracle prompts
  assume the exterior view geometry from training. A camera bump between
  training and eval will not raise any error — every VLM call just gets worse.
- **RPY discontinuities at ±180°.** xArm conventions wrap at the seam. If a
  commanded rotation stalls mid-motion, switch to the axis-angle SDK path
  rather than fighting the wrap.

## Software abort

`Ctrl-C` in the executor terminal is caught by `real/xarm_flywheel/main.py`
and calls `hardware.close()` to release the RealSense pipelines. The arm
itself is **not** explicitly halted by the software path — it will hold
whatever pose it was commanded to last. This is **not** a substitute for
the physical e-stop, and the SIGINT can be missed entirely during a
blocking xArm SDK call.
