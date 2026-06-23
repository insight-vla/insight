"""Entry points for the xArm runtime.

``run_flywheel(args)`` — VLM-planned execution (used by ``real/entry/run_flywheel.py``).
``run_primitives(args)`` — fixed primitive sequence (used by ``real/entry/run_primitives.py``).

Both share the FPS resolution, hardware lifecycle, policy connection, and
optional dataset recorder via ``_setup_runtime``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import tyro

# Dataset-meta loader still lives in the legacy module; one of the few things
# we still consume from there.
import inference_primitives as ip
import vlm_check

from insight.keyboard import stop_requested

from .args import FlywheelArgs, PrimitivesArgs, RuntimeArgs
from .executor import XArmFlywheelExecutor
from .hardware import XArmHardware
from .recording import FlywheelRecorder
from .runner import XArmRunner

if TYPE_CHECKING:
    from openpi_client import websocket_client_policy


_DEFAULT_FPS = 20.0


@dataclasses.dataclass
class RuntimeContext:
    """Everything ``_setup_runtime`` produces, packaged so callers can pick
    fields by name instead of unpacking a tuple."""
    hardware: XArmHardware
    policy: "websocket_client_policy.WebsocketClientPolicy | None"
    dataset_durations: dict[str, int]
    recorder: FlywheelRecorder | None


def _resolve_fps_and_durations(args: RuntimeArgs) -> tuple[float, dict[str, int]]:
    """FPS precedence: ``--fps`` > dataset info.json > module default."""
    durations: dict[str, int] = {}
    dataset_fps: float | None = None
    if args.durations_from_dataset:
        durations, dataset_fps = ip.load_dataset_meta(
            args.durations_from_dataset, args.duration_percentile,
        )
    if args.fps > 0:
        fps = float(args.fps)
        logging.info("Using FPS=%.1f from --fps override", fps)
    elif dataset_fps is not None:
        fps = dataset_fps
        logging.info("Using FPS=%.1f from dataset info.json", fps)
    else:
        fps = _DEFAULT_FPS
        logging.info("Using FPS=%.1f (default; no dataset or override)", fps)
    return fps, durations


def _count_successful_trials(trials_log: Path, dataset_repo: str,
                             experiment_name: str = "") -> int:
    """Count previously-committed successful trials in this scope.

    Reads ``flywheel_trials.jsonl`` (per-trial outcome log; one line per
    trial with ``success: bool``) and returns the count of matching rows
    whose oracle/user verdict was success. Used to make ``--target-successes``
    span Ctrl+C → restart cycles in TRIAL units rather than lerobot-episode
    units (one trial may commit several episodes when
    --no-record-skill-gap-only is set).

    Scope rules (one of these must hold for a row to count):
    - ``dataset_repo`` matches (collection mode), OR
    - ``experiment_name`` matches (eval / non-recording mode).

    Returns 0 when neither scope key is set, so non-scoped runs start fresh
    every invocation instead of resuming against a shared log.

    Caveat: if you wipe the dataset cache but keep the trials log, this
    will over-count. Wipe both (or use a fresh scope tag) to truly start
    from zero.
    """
    if not trials_log.exists() or (not dataset_repo and not experiment_name):
        return 0
    n = 0
    with open(trials_log, "r") as f:
        for line in f:
            try:
                outcome = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not outcome.get("success"):
                continue
            if dataset_repo and outcome.get("dataset_repo") == dataset_repo:
                n += 1
            elif experiment_name and outcome.get("experiment_name") == experiment_name:
                n += 1
    return n


def _resolve_run_dir(args: RuntimeArgs) -> Path:
    """Resolve the per-run output directory and cache it on ``args``.

    Idempotent: subsequent calls return the same path so logging, the runner,
    and batch-summary writers all share one destination.
    """
    cached = getattr(args, "_resolved_run_dir", None)
    if cached is not None:
        return cached
    now = datetime.now()
    run_dir = (
        Path(args.video_dir)
        / now.strftime("%Y-%m-%d")
        / (args.run_name or f"run_{now.strftime('%Y%m%d_%H%M%S')}")
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    args._resolved_run_dir = run_dir  # type: ignore[attr-defined]
    return run_dir


def _setup_logging(args: RuntimeArgs, mode: str) -> Path:
    """Configure root logging to write to both stdout and a per-run log file.

    Log lives inside the run directory (date/run-name/) so all artifacts from
    one run cluster together. StreamHandler is bound to ``sys.stdout`` (not
    the default ``sys.stderr``) so log messages interleave correctly with
    ``input()`` prompts during per-step approval.
    """
    run_dir = _resolve_run_dir(args)
    log_path = run_dir / f"{mode}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        force=True,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path),
        ],
    )
    logging.info("Logging to: %s", log_path)

    # Snapshot the VLM prompts used by this run alongside the log so the
    # exact text the VLM saw is recoverable later (the prompt evolves
    # across batches and per-batch behavior is hard to debug without it).
    try:
        from insight import prompts as _insight_prompts
        # Only the prompts actually fired by the flywheel pipeline —
        # CHECK_GOAL_SYSTEM and DECIDE_NEXT_PRIMITIVE_SYSTEM are for
        # different flows and would just be noise here.
        prompt_names = (
            "PREANALYZE_TRANSLATION_SYSTEM",
            "TASK_COMPLETION_SYSTEM",
            "PRIMITIVE_DONE_SYSTEM",
        )
        with open(run_dir / "prompts.txt", "w") as f:
            for name in prompt_names:
                val = getattr(_insight_prompts, name, None)
                if val is None:
                    continue
                f.write(f"=== {name} ===\n{val}\n\n")
    except Exception as e:
        logging.warning("Could not snapshot prompts: %s", e)

    return log_path


def _setup_runtime(args: RuntimeArgs, mode: str) -> RuntimeContext:
    """Configure logging + VLM provider, build hardware + policy + recorder.

    ``mode`` is used to name the log file (e.g. ``flywheel_20260427_120000.log``)
    and to distinguish entry points in mixed-output directories.
    """
    _setup_logging(args, mode)

    vlm_check.set_provider(args.vlm_provider)
    if args.use_vlm_check:
        logging.info(
            "VLM check enabled: provider=%s warmup=%d interval=%d votes=%d "
            "consecutive=%d  |  Z safety: %.1fmm",
            args.vlm_provider, args.vlm_warmup_steps, args.vlm_check_interval_steps,
            args.vlm_num_votes, args.vlm_consecutive_required,
            args.z_completion_threshold_mm,
        )

    fps, dataset_durations = _resolve_fps_and_durations(args)
    hardware = XArmHardware(
        arm_ip=args.arm_ip, fps=fps,
        workspace_bounds=tuple(args.workspace_bounds),
        max_tcp_speed_mm_s=args.max_tcp_speed_mm_s,
        tool_offset_local_mm=tuple(args.tool_offset_mm),
        tool_tip_floor_z_mm=args.tool_tip_floor_z_mm,
        use_gripper=args.use_gripper,
        use_angle_axis_control=args.use_angle_axis_control,
    )

    policy = None
    if not args.dry_run:
        from openpi_client import websocket_client_policy
        policy = websocket_client_policy.WebsocketClientPolicy(
            host=args.host, port=args.port, api_key=args.api_key,
        )
        logging.info("Policy server: %s", policy.get_server_metadata())
        hardware.connect()
        logging.info("Current arm pose: %s", hardware.get_pose().tolist())
    else:
        logging.info("Dry run: skipping arm + policy connect.")

    recorder: FlywheelRecorder | None = None
    if args.record_dataset_repo and not args.dry_run:
        recorder = FlywheelRecorder(
            args.record_dataset_repo, fps=fps,
            record_gripper=args.use_gripper,
        )

    return RuntimeContext(
        hardware=hardware,
        policy=policy,
        dataset_durations=dataset_durations,
        recorder=recorder,
    )


def run_flywheel(args: FlywheelArgs) -> None:
    if not args.goal:
        raise SystemExit("Need --goal '<task description>'.")
    if not args.available_primitives:
        raise SystemExit("Need --available-primitives '<step1>' '<step2>' ...")

    ctx = _setup_runtime(args, mode="flywheel")
    executor = XArmFlywheelExecutor(args, ctx.hardware, ctx.policy, recorder=ctx.recorder)

    # Persistent per-trial log: appended every trial so partial batches survive
    # crashes. Path resolution (in priority order):
    # 1. record_dataset_repo set → lives inside the LeRobot dataset folder so
    #    demos + attempt log cluster for one task.
    # 2. experiment_name set → ``video_dir/<experiment_name>/`` so eval runs
    #    of the same goal across different checkpoints stay separate. Resume
    #    on Ctrl+C scopes to this file.
    # 3. neither set → legacy shared ``video_dir/flywheel_trials.jsonl``.
    #    Older test runs landed here; not recommended for new runs because
    #    rows from different experiments mix together.
    if args.record_dataset_repo:
        trials_log = (
            Path.home() / ".cache" / "huggingface" / "lerobot"
            / args.record_dataset_repo / "flywheel_trials.jsonl"
        )
    elif args.experiment_name:
        trials_log = Path(args.video_dir) / args.experiment_name / "flywheel_trials.jsonl"
    else:
        trials_log = Path(args.video_dir) / "flywheel_trials.jsonl"
    trials_log.parent.mkdir(parents=True, exist_ok=True)
    batch_outcomes: list[dict] = []  # collected for end-of-batch summary
    batch_started = datetime.now().isoformat(timespec="seconds")
    # Seed the success counter from the per-trial log so ``--target-successes``
    # is interpreted in TRIALS (one full plan run, regardless of how many
    # lerobot episodes commit per trial). Counting recorder.num_episodes()
    # would over-count when --no-record-skill-gap-only commits the lead-in
    # primitives alongside the skill gap (e.g. 3 episodes per trial for
    # twist's [move-to, close, twist]).
    successes = _count_successful_trials(
        trials_log, args.record_dataset_repo, args.experiment_name,
    )
    if successes > 0:
        logging.info("[BATCH] dataset %s already has %d successful trials logged; "
                     "resuming toward target.", args.record_dataset_repo, successes)
    run_idx = 0

    def _record_outcome(success: bool, reason: str, *,
                        interrupted: bool = False,
                        interrupt_outcome: str = "",
                        duration_s: float = 0.0) -> None:
        # Pull per-trial metrics off the executor (set during run_flywheel).
        # Defaults handle the "interrupted before any plan" case.
        prompt_time_s = float(getattr(executor, "_last_prompt_time_s", 0.0))
        vlm_time_s = float(getattr(executor, "_last_vlm_time_s", 0.0))
        oracle_time_s = float(getattr(executor, "_last_oracle_time_s", 0.0))
        return_home_time_s = float(getattr(executor, "_last_return_home_time_s", 0.0))
        # robot_time = wall-clock minus everything that isn't productive
        # manipulation: VLM thinking, oracle scoring, operator prompts, and
        # deterministic return-to-home motion. Floor at 0 to handle the
        # "interrupted before any motion" edge case where the deductions
        # could exceed the (tiny) wall-clock.
        robot_time_s = max(
            0.0,
            float(duration_s) - prompt_time_s - vlm_time_s
            - oracle_time_s - return_home_time_s,
        )
        outcome = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "trial_in_batch": run_idx + 1,
            "goal": args.goal,
            "dataset_repo": args.record_dataset_repo,
            "experiment_name": args.experiment_name,
            "success": bool(success),
            "reason": reason,
            "interrupted": interrupted,
            "interrupt_outcome": interrupt_outcome,
            "duration_s": round(float(duration_s), 2),
            "robot_time_s": round(robot_time_s, 2),
            "vlm_time_s": round(vlm_time_s, 2),
            "oracle_time_s": round(oracle_time_s, 2),
            "prompt_time_s": round(prompt_time_s, 2),
            "return_home_time_s": round(return_home_time_s, 2),
            "plan_length": int(getattr(executor, "_last_plan_length", 0)),
            "skill_gaps": list(getattr(executor, "_last_skill_gaps", [])),
            "num_skill_gaps": len(getattr(executor, "_last_skill_gaps", [])),
            "episodes_committed": int(getattr(executor, "_last_episodes_committed", 0)),
            "frames_committed": int(getattr(executor, "_last_frames_committed", 0)),
            "oracle_overridden": bool(getattr(executor, "_last_oracle_overridden", False)),
        }
        batch_outcomes.append(outcome)
        with open(trials_log, "a") as f:
            f.write(json.dumps(outcome) + "\n")

    try:
        for run_idx in range(args.num_runs):
            # Stop key persists across runs — a single 's'/'q' kills the whole
            # batch, not just the current run. Ctrl+C does the same via the
            # outer except below.
            if stop_requested():
                logging.info("\nStop requested — exiting batch.")
                break

            if args.num_runs > 1:
                target_str = f"/{args.target_successes}" if args.target_successes else ""
                logging.info(
                    "\n%s\n# RUN %d/%d  successes=%d%s\n%s",
                    "#" * 60, run_idx + 1, args.num_runs,
                    successes, target_str, "#" * 60,
                )
                # One prompt per trial: gives the user time to reset rocks
                # (run_idx > 0) or finalize scene setup (run_idx == 0). Fires
                # BEFORE executor.run_flywheel captures the scene for planning.
                msg = (
                    f"Reset rocks for trial {run_idx + 1}, then press ENTER to start... "
                    if run_idx > 0
                    else "Position rocks, then press ENTER to start trial 1... "
                )
                input(msg)

            t_trial_start = time.perf_counter()
            # Reset per-trial post-reject flag so a prior trial's rejection
            # doesn't poison this one.
            if hasattr(executor, "_trial_failed_skip_gaps"):
                executor._trial_failed_skip_gaps = False
            try:
                success = executor.run_flywheel(dataset_durations=ctx.dataset_durations)
            except KeyboardInterrupt:
                # Capture the in-progress trial's outcome before propagating —
                # the executor sets ``_last_trial_reason`` eagerly (before the
                # verdict prompt) so we have a useful reason even when Ctrl+C
                # interrupts the prompt itself.
                duration_s = time.perf_counter() - t_trial_start
                reason = getattr(executor, "_last_trial_reason", "") or "interrupted before verdict"
                # Ask the operator to classify the interrupt: spill-imminent
                # failures should count, deliberate batch aborts shouldn't.
                # Hammering Ctrl+C through the prompt defaults to failure
                # (the conservative read).
                try:
                    ans = input(
                        "\nCtrl+C — outcome? [F=failure / a=abort (don't count)]: "
                    ).strip().lower()
                except (EOFError, KeyboardInterrupt):
                    ans = "f"
                interrupt_outcome = "abort" if ans == "a" else "failure"
                if interrupt_outcome == "abort":
                    reason = f"{reason} [abort]"
                _record_outcome(
                    False, reason,
                    interrupted=True, interrupt_outcome=interrupt_outcome,
                    duration_s=duration_s,
                )
                raise
            duration_s = time.perf_counter() - t_trial_start
            reason = getattr(executor, "_last_trial_reason", "") or ""
            # If the user rejected a skill-gap mid-plan, override any oracle
            # "success" verdict with failure. The trial only ran cleanup
            # primitives (lower / open) after the rejection, no actual
            # task completion happened.
            if getattr(executor, "_trial_failed_skip_gaps", False):
                success = False
            if success:
                successes += 1
                logging.info("  => SUCCESS (%d total)", successes)
            else:
                logging.info("  => FAILED — %s", reason)

            _record_outcome(success, reason, duration_s=duration_s)

            if args.target_successes and successes >= args.target_successes:
                logging.info("\nReached target of %d successes. Stopping.",
                             args.target_successes)
                break
    except KeyboardInterrupt:
        logging.info("\nBatch aborted by user (Ctrl+C). Artifacts saved.")
    finally:
        if batch_outcomes:
            n_trials = len(batch_outcomes)
            n_success = sum(1 for o in batch_outcomes if o["success"])
            n_fail = n_trials - n_success
            logging.info(
                "\n%s\nBATCH COMPLETE: %d successes / %d failures out of %d trials\n%s",
                "=" * 60, n_success, n_fail, n_trials, "=" * 60,
            )
            if n_fail > 0:
                logging.info("Failure reasons:")
                for o in batch_outcomes:
                    if not o["success"]:
                        logging.info("  trial %d: %s",
                                     o["trial_in_batch"], o["reason"])
            # Per-batch summary file goes into the run dir alongside the log
            # and videos so all artifacts from this invocation cluster together.
            summary_path = _resolve_run_dir(args) / "batch_summary.json"
            total_duration_s = sum(float(o.get("duration_s", 0.0)) for o in batch_outcomes)
            total_robot_time_s = sum(float(o.get("robot_time_s", 0.0)) for o in batch_outcomes)
            total_vlm_time_s = sum(float(o.get("vlm_time_s", 0.0)) for o in batch_outcomes)
            total_prompt_time_s = sum(float(o.get("prompt_time_s", 0.0)) for o in batch_outcomes)
            success_robot_time_s = sum(
                float(o.get("robot_time_s", 0.0)) for o in batch_outcomes if o["success"]
            )
            total_episodes = sum(int(o.get("episodes_committed", 0)) for o in batch_outcomes)
            total_frames = sum(int(o.get("frames_committed", 0)) for o in batch_outcomes)
            n_oracle_override = sum(1 for o in batch_outcomes if o.get("oracle_overridden"))
            summary = {
                "batch_started": batch_started,
                "batch_finished": datetime.now().isoformat(timespec="seconds"),
                "goal": args.goal,
                "dataset_repo": args.record_dataset_repo,
                "num_runs_arg": args.num_runs,
                "target_successes_arg": args.target_successes,
                "n_trials": n_trials,
                "n_success": n_success,
                "n_fail": n_fail,
                "success_rate": (n_success / n_trials) if n_trials else 0.0,
                "total_duration_s": round(total_duration_s, 2),
                "total_robot_time_s": round(total_robot_time_s, 2),
                "total_vlm_time_s": round(total_vlm_time_s, 2),
                "total_prompt_time_s": round(total_prompt_time_s, 2),
                "success_robot_time_s": round(success_robot_time_s, 2),
                "mean_trial_duration_s": round(total_duration_s / n_trials, 2) if n_trials else 0.0,
                "mean_robot_time_s": round(total_robot_time_s / n_trials, 2) if n_trials else 0.0,
                "total_episodes_committed": total_episodes,
                "total_frames_committed": total_frames,
                "n_oracle_overridden": n_oracle_override,
                "outcomes": batch_outcomes,
            }
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            logging.info("Batch summary: %s", summary_path)
            logging.info("Per-trial log: %s", trials_log)
        ctx.hardware.close()


def run_primitives(args: PrimitivesArgs) -> None:
    if not args.primitives:
        raise SystemExit("Need --primitives '<step1>' '<step2>' ...")

    ctx = _setup_runtime(args, mode="primitives")
    runner = XArmRunner(args, ctx.hardware, ctx.policy, recorder=ctx.recorder)
    try:
        runner.run_sequence(list(args.primitives), dataset_durations=ctx.dataset_durations)
    except KeyboardInterrupt:
        logging.info("\nRun aborted by user (Ctrl+C). Artifacts saved.")
    finally:
        ctx.hardware.close()


# Back-compat alias for the existing ``run_flywheel.py`` shim.
run = run_flywheel


if __name__ == "__main__":
    run_flywheel(tyro.cli(FlywheelArgs))
