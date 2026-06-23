"""``XArmRunner`` — base class for primitive-sequence execution on the xArm.

Owns the bits that are shared between modes:

- The trained-policy execution loop (``_run_known_primitive``)
- Per-step dispatch (``execute_step`` — known only here; flywheel mode overrides)
- Run-output dir + per-primitive video artifacts
- The fixed-sequence entry point (``run_sequence``)

Subclassed by ``XArmFlywheelExecutor`` to add planning + skill-gap handling.
"""

from __future__ import annotations

import collections
import json
import logging
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from insight.executor import BaseExecutor
from insight.keyboard import start_stop_listener, stop_requested
from insight.planning import log_plan, resolve_step_durations
from insight.reasoning import check_primitive_done
from insight.rotation import quat_to_rpy_deg, rpy_to_quat_deg, slerp, unwrap_rpy_near
from insight.voting import TemporalConsistency, should_check_now

from .video import save_combined_video, save_debug_videos, save_primitive_video


def _save_continuous_video(
    run_dir: Path,
    frames: list[tuple[float, np.ndarray, np.ndarray]],
    name_suffix: str = "",
    high_quality: bool = False,
) -> None:
    """Write the uncut continuous-capture video to ``<run_dir>``.

    Frames are list of ``(timestamp_s, ext, wrist)`` from the background
    thread. We encode at the actual captured fps (timestamps tell us
    this — usually ~10 fps) so the result is wall-clock-faithful.
    Side-by-side ext|wrist composite, like the stepwise combined video.
    """
    import cv2
    import imageio.v2 as imageio

    if not frames:
        return
    composed = []
    for _t, ext, wrist in frames:
        # Resize both to 240 height so they stack cleanly. Same convention
        # as save_combined_video so the two videos look comparable.
        h = 240
        e = cv2.resize(ext, (int(ext.shape[1] * h / ext.shape[0]), h))
        w = cv2.resize(wrist, (int(wrist.shape[1] * h / wrist.shape[0]), h))
        composed.append(np.concatenate([e, w], axis=1))
    if len(frames) >= 2:
        elapsed = frames[-1][0] - frames[0][0]
        fps = max(1.0, (len(frames) - 1) / max(elapsed, 1e-3))
    else:
        fps = 10.0
    suffix = f"_{name_suffix}" if name_suffix else ""
    path = run_dir / f"all_primitives_continuous{suffix}.mp4"
    if high_quality:
        imageio.mimwrite(
            str(path), composed, fps=int(round(fps)), codec="libx264",
            output_params=["-crf", "17", "-pix_fmt", "yuv420p", "-preset", "slow"],
        )
    else:
        imageio.mimwrite(str(path), composed, fps=int(round(fps)), codec="libx264")
    logging.info("Saved continuous video (%d frames @ %.1f fps): %s",
                 len(composed), fps, path)


if TYPE_CHECKING:
    from openpi_client import websocket_client_policy
    from .args import RuntimeArgs
    from .hardware import XArmHardware
    from .recording import FlywheelRecorder


# (step_index, exterior_rgb, wrist_rgb)
FrameEntry = tuple[int, np.ndarray, np.ndarray, float]  # step, ext, wrist, progress


# Per-tick heartbeat log frequency for known-primitive execution: every
# Nth outer step plus the first few. Tuned for human-readable progress
# without spamming the terminal at 10 Hz.
_LOG_HEARTBEAT_EVERY_N_STEPS = 20
_LOG_HEARTBEAT_INITIAL_STEPS = 3

# Primitives whose training data was preprocessed with np.unwrap (in
# filter_normalize_*.py) to remove 2π discontinuities. At inference,
# state for these primitives must be unwrapped tick-to-tick so the policy
# sees the same continuous numerical convention it was trained on. Other
# primitives whose training data was wrapped (every standard pickplace
# primitive — they don't wrap during normal execution anyway) must NOT
# receive runtime unwrap: their training convention is wrapped, so
# unwrapping at inference creates a reverse convention mismatch that
# manifests as OOD-induced progress saturation in primitives that run
# immediately after twist (e.g. "lift upward" insta-terminating at
# progress=1.0 from the unwrapped state.rz ≈ 355° being OOD).
_UNWRAP_PRIMITIVES: frozenset[str] = frozenset({
    "twist open the cap",
})


class _SkipPrimitive(Exception):
    """User pressed 'n' at the per-step approval prompt to skip the rest of
    the current primitive and advance to the next plan step. Distinct from
    Ctrl+C (aborts the whole run) and hardware errors (abort the plan)."""


class _AbortTrial(Exception):
    """User rejected a skill-gap proposal ('f' at the confirm gate) — abort
    the rest of THIS trial's plan and let the runner mark the trial failed
    in the JSONL. Subsequent primitives in the same plan are predicated on
    the rejected gap having run, so executing them would do strange things.
    Distinct from Ctrl+C (whole batch) and _SkipPrimitive (one primitive)."""


class XArmRunner(BaseExecutor):
    """Run a sequence of known primitives via the trained policy.

    Concrete on its own (``execute_step`` only handles the known-primitive
    branch). Flywheel mode is a subclass that overrides ``execute_step`` to
    also dispatch skill gaps.
    """

    def __init__(self, args: "RuntimeArgs", hardware: "XArmHardware",
                 policy: "websocket_client_policy.WebsocketClientPolicy | None",
                 recorder: "FlywheelRecorder | None" = None) -> None:
        super().__init__(args)
        self.hardware = hardware
        self.policy = policy
        self.recorder = recorder
        self.run_dir: Path | None = None
        self._dataset_durations: dict[str, int] = {}
        self._combined_frames: list[tuple[str, np.ndarray, np.ndarray]] = []
        self._global_step = 0
        # Raw per-camera frame buffers — appended every tick of every primitive
        # so debug videos survive a mid-run Ctrl+C (whereas _combined_frames is
        # only extended at end-of-primitive and would lose the in-progress one).
        self._debug_ext_frames: list[np.ndarray] = []
        self._debug_wrist_frames: list[np.ndarray] = []
        # Per-trial timing buckets surfaced to main.py for paper reporting.
        # Wall-clock duration_s = vlm_time + oracle_time + prompt_time
        # + return_home_time + robot_time + small bookkeeping overhead.
        # The buckets are designed to split method-required cost from eval
        # scaffolding for clean paper comparisons:
        #   - vlm_time:        planner + skill-gap pre-analysis + per-step
        #                      done-checks. "Think time" in paper tables.
        #   - oracle_time:     end-of-trial success oracle. Eval scaffolding
        #                      shared across all methods, excluded from
        #                      "Think" in paper tables.
        #   - prompt_time:     confirm-plan / verdict-override / etc.
        #                      Operator overhead, excluded from "Wall".
        #   - return_home_time: deterministic hardware reset (no learned
        #                      behavior). Excluded from "Exec" so the
        #                      reported motion time is only productive work.
        #   - robot_time:      computed in main.py as duration - sum(above).
        #                      "Exec time" in paper tables.
        # Reset by the executor at trial boundaries.
        self._last_prompt_time_s = 0.0
        self._last_vlm_time_s = 0.0
        self._last_oracle_time_s = 0.0
        self._last_return_home_time_s = 0.0

    def _timed_input(self, prompt: str) -> str:
        """input(prompt) with elapsed time accumulated into ``_last_prompt_time_s``.

        Used so paper-reported robot time isn't inflated by user think-time at
        confirm-plan / confirm-skill-gap / manual-verdict prompts.
        """
        t0 = time.perf_counter()
        try:
            resp = input(prompt)
        except EOFError:
            resp = ""
        self._last_prompt_time_s += time.perf_counter() - t0
        return resp

    def _timed_vlm(self, fn, *args, **kwargs):
        """Call ``fn(*args, **kwargs)`` and accumulate elapsed time into
        ``_last_vlm_time_s``. Used for method-required VLM calls (planner,
        skill-gap pre-analysis, per-step done-checks) — i.e., the "Think"
        time bucket in paper tables. The end-of-trial success oracle uses
        ``_timed_oracle`` instead so it can be excluded from method-cost
        comparisons.
        """
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            self._last_vlm_time_s += time.perf_counter() - t0

    def _timed_oracle(self, fn, *args, **kwargs):
        """Call ``fn(*args, **kwargs)`` and accumulate elapsed time into
        ``_last_oracle_time_s``. Used for the end-of-trial success oracle
        only; tracked separately from VLM time because the oracle is eval
        scaffolding (shared by all methods), not method-required overhead.
        """
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            self._last_oracle_time_s += time.perf_counter() - t0

    def _timed_return_home(self, *args, **kwargs) -> None:
        """Wrap ``hardware.return_to_home`` with elapsed-time tracking into
        ``_last_return_home_time_s``. The return-to-home motion is a
        deterministic reset (canonical home pose via ``set_servo_angle``)
        — productive evaluation time, but not part of the method's learned
        manipulation. Excluded from "Exec" in paper tables.
        """
        t0 = time.perf_counter()
        try:
            self.hardware.return_to_home(*args, **kwargs)
        finally:
            self._last_return_home_time_s += time.perf_counter() - t0

    # ────────────────── Public entry ──────────────────

    def run_sequence(self, primitive_sequence: list[str],
                     dataset_durations: dict[str, int] | None = None) -> bool:
        """Execute ``primitive_sequence`` in order. Returns True on completion.

        No planning, no skill gaps. Each primitive is executed via the trained
        policy with the same auto-advance + VLM-done-check rules as flywheel mode.
        """
        if not primitive_sequence:
            raise SystemExit("primitive_sequence is empty.")
        self._dataset_durations = dataset_durations or {}
        log_plan(primitive_sequence, skill_gaps=[], step_notes=None,
                 header=f"Plan ({len(primitive_sequence)} primitives):")
        return self._prepare_and_run(
            primitive_sequence,
            skill_gaps=set(),
            plan_summary=str(primitive_sequence),
        )

    # ────────────────── Shared post-planning path ──────────────────

    def _prepare_and_run(self, primitive_sequence: list[str],
                         skill_gaps: set[str], plan_summary: str) -> bool:
        """Resolve durations, gate on dry-run/plan-only, set up the run dir,
        wait for ENTER, then delegate to ``_run_duration_plan``.

        Subclasses call this after they've produced a ``primitive_sequence``
        (either user-supplied for ``run_sequence`` or VLM-planned for
        ``run_flywheel``). Mode-specific logging (planner reasoning, plan
        header) belongs to the caller; this method handles only the common
        pre-execution path.
        """
        args = self.args
        duration_plan = resolve_step_durations(
            primitive_sequence,
            durations=self._dataset_durations,
            default=args.default_duration,
            fixed=args.fixed_duration if args.fixed_duration > 0 else None,
        )
        total = sum(d for _, d in duration_plan)
        logging.info("Total: %d steps (~%.1fs @ %.0fHz)",
                     total, total / self.hardware.fps, self.hardware.fps)

        if args.dry_run:
            logging.info("Dry run: exiting before execution.")
            return True
        if args.plan_only:
            logging.info("Plan-only: connections + plan validated, exiting before motion.")
            return True

        self.run_dir = self._setup_run_dir(plan_summary=plan_summary)
        # In batch mode (FlywheelArgs.num_runs > 1) the per-trial prompt is
        # owned by main.py, which fires it BEFORE this point so the user can
        # reset the scene before the VLM captures it. Skipping here avoids a
        # redundant second ENTER per trial.
        if getattr(self.args, "num_runs", 1) <= 1:
            self._timed_input("Press Enter to start, or Ctrl+C to abort... ")
        # Keyboard listener intentionally NOT started — it puts stdin in
        # cbreak mode, which intercepts characters destined for the manual-
        # verdict input() prompt. The result was that pressing 'n' at the
        # verdict prompt did nothing and the oracle's verdict was always
        # accepted, contaminating the dataset on false positives. Use Ctrl+C
        # to abort the batch — main.py captures the in-progress trial
        # outcome on KeyboardInterrupt and writes the batch summary.
        logging.info("Press Ctrl+C at any time to stop the batch.")

        return self._run_duration_plan(duration_plan, skill_gaps=skill_gaps)

    # ────────────────── Per-step dispatch (known only) ──────────────────

    def _maybe_run_deterministic(self, primitive: str, primitive_idx: int = 0,
                                  plan_total: int = 0) -> bool | None:
        """Handle non-VLA primitives that dispatch directly to hardware.

        The planner can include these in its plan as transition/reset steps;
        we intercept here before recording or the policy loop so they don't
        end up in the dataset and aren't gated on progress/duration. Frames
        captured during the motion ARE still added to the combined video so
        the run-wide playback shows the transition (e.g. between twist and
        pour in compositional tasks) — only the dataset commit is skipped.
        Returns True/False if handled, None to fall through to the VLA path.
        """
        if primitive.lower() != "return to home":
            return None
        logging.info("  [RETURN-HOME] driving arm to canonical home pose")
        captured = self._return_home_with_capture()
        if self.args.save_videos and captured and self.run_dir is not None:
            # Combined video: show the transition tagged like a known
            # primitive so the per-frame strip looks consistent.
            self._extend_combined(
                primitive, primitive_idx, plan_total, captured,
                is_skill_gap=False,
            )
            # Per-primitive video — same naming convention as other
            # successful primitives (no _INCOMPLETE suffix). Recorder is
            # deliberately bypassed since this isn't a learnable primitive.
            try:
                from .video import save_primitive_video
                save_primitive_video(
                    self.run_dir, primitive, primitive_idx, plan_total,
                    captured, fps=self.hardware.fps, is_skill_gap=False,
                    high_quality=getattr(self.args, "high_quality_videos", False),
                )
            except Exception as e:
                logging.warning("[RETURN-HOME] per-primitive video save failed: %s", e)
        return True

    def _return_home_with_capture(self) -> list[FrameEntry]:
        """Drive the arm home while capturing camera frames in the main
        thread. Returns the captured frames so they can be appended to the
        combined video.

        ``hardware.return_to_home`` blocks (``set_servo_angle(wait=True)``),
        so we run it on a daemon thread and poll the camera at the
        hardware fps until it finishes. Cameras are external to the arm so
        concurrent reads during a position-control move are safe.

        Timing: this helper is only called for the **in-plan** "return to
        home" primitive (a transition step the planner inserts mid-plan,
        e.g. between twist and pour). That motion IS productive method
        work, so we deliberately do NOT accumulate into
        ``_last_return_home_time_s`` — it falls through to ``robot_time_s``
        as the residual in main.py and counts toward paper Exec. The two
        eval-scaffolding paths that should be EXCLUDED from Exec
        (start-of-trial RTH in the executor, post-plan oracle-pose drive
        in the oracle mixin) bypass this helper and call
        ``_timed_return_home`` directly.
        """
        frames: list[FrameEntry] = []
        done = threading.Event()
        drive_error: list[BaseException] = []

        def _drive() -> None:
            try:
                self.hardware.return_to_home(
                    self.hardware.home_pose,
                    joint_speed_deg_s=self.args.home_joint_speed_deg_s,
                )
            except BaseException as e:  # noqa: BLE001 — re-raised after join
                drive_error.append(e)
            finally:
                done.set()

        thread = threading.Thread(target=_drive, daemon=True)
        thread.start()
        # Capture at the configured camera fps; clamp the period so a
        # mis-configured very-low fps still yields one frame per second.
        period = 1.0 / max(self.hardware.fps, 1.0)
        step = 0
        while not done.is_set():
            try:
                ext, wrist = self.hardware.capture_frames()
                frames.append((step, ext, wrist, 0.0))
                step += 1
            except Exception as e:
                logging.warning("[RETURN-HOME] frame capture failed mid-drive: %s", e)
            done.wait(period)
        thread.join()
        if drive_error:
            raise drive_error[0]
        return frames

    def execute_step(self, primitive: str, num_steps: int,
                     primitive_idx: int, plan_total: int,
                     is_skill_gap: bool) -> bool:
        """Dispatch one plan step. Base class only handles known primitives;
        the flywheel subclass overrides to add the skill-gap branch.
        """
        det = self._maybe_run_deterministic(primitive, primitive_idx, plan_total)
        if det is not None:
            return det
        if is_skill_gap:
            raise RuntimeError(
                f"{type(self).__name__} cannot execute skill gap {primitive!r}. "
                "Use XArmFlywheelExecutor for skill-gap support."
            )
        prim_frames: list[FrameEntry] | None = [] if self.args.save_videos else None
        if self.recorder is not None:
            self.recorder.start_primitive(primitive)
        ok = False
        user_skipped = False
        interrupted = False
        try:
            try:
                ok = self._run_known_primitive(primitive, num_steps, prim_frames)
            except _SkipPrimitive:
                logging.info("  [USER-SKIP] %r — advancing to next primitive", primitive)
                user_skipped = True
        except KeyboardInterrupt:
            # Save what we captured before re-raising — frames mutated in place
            # by the inner loop survive the interrupt via the finally below.
            interrupted = True
            raise
        finally:
            self._finalize_primitive_artifacts(
                primitive, primitive_idx, plan_total, prim_frames,
                ok=ok, user_skipped=user_skipped, interrupted=interrupted,
                is_skill_gap=is_skill_gap,
            )
        # Treat P90-cap and user-skip as soft successes at the plan level so the
        # next plan step still executes — recording is the only thing gated by ``ok``.
        return True

    # ────────────────── Shared inner loop ──────────────────

    def _run_duration_plan(self, duration_plan: list[tuple[str, int]],
                           skill_gaps: set[str]) -> bool:
        """Iterate ``duration_plan``, dispatch each primitive, save artifacts.

        Both ``run_sequence`` and ``run_flywheel`` end up here once their
        mode-specific setup is done.
        """
        self._combined_frames = []
        self._global_step = 0
        self._debug_ext_frames = []
        self._debug_wrist_frames = []
        # Start a background "continuous" capture so we can build a
        # wall-clock-faithful video that includes time spent inside VLM
        # calls (planner, pre-analysis, oracle) and VLA inference — the
        # stepwise video skips those.
        try:
            self.hardware.start_continuous_capture(fps=10.0)
        except Exception as e:
            logging.warning("Could not start continuous capture: %s", e)
        completed_cleanly = False
        try:
            for i, (primitive, num_steps) in enumerate(duration_plan):
                if stop_requested():
                    logging.info("Stop requested before primitive %d/%d — exiting plan.",
                                 i + 1, len(duration_plan))
                    break
                is_skill_gap = primitive in skill_gaps
                tag = "SKILL-GAP" if is_skill_gap else "known"
                logging.info("\n[%d/%d] [%s] %s  (%d steps)",
                             i + 1, len(duration_plan), tag, primitive, num_steps)

                ok = self.execute_step(primitive, num_steps, i, len(duration_plan), is_skill_gap)
                if not ok:
                    logging.error("Step '%s' failed; aborting plan.", primitive)
                    return False
            completed_cleanly = True
        finally:
            # Always save debug + combined videos, even on KeyboardInterrupt /
            # mid-primitive abort. Per-primitive videos for completed primitives
            # are already on disk by this point; this rescues whatever frames
            # were captured up to the interrupt. Ignore SIGINT during the saves
            # so a second Ctrl+C (impatient user) doesn't kill the encoder
            # mid-write and leave a truncated/empty file.
            # Stop the background continuous capture and save it as a
            # separate video. Done BEFORE the stepwise save so the
            # background thread doesn't keep growing the buffer while
            # we encode.
            try:
                continuous_frames = self.hardware.stop_continuous_capture()
            except Exception as e:
                logging.warning("Could not stop continuous capture: %s", e)
                continuous_frames = []

            if self.args.save_videos and self.run_dir is not None:
                hq = getattr(self.args, "high_quality_videos", False)
                prev_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
                try:
                    save_combined_video(
                        self.run_dir, self._combined_frames, self.hardware.fps,
                        name_suffix=self.args.experiment_name,
                        high_quality=hq,
                    )
                    # Continuous (uncut) video: includes time spent in
                    # VLM calls and VLA inference. Wall-clock-faithful
                    # at ~10 fps. Encoded as a side-by-side ext|wrist
                    # composite to match the stepwise video layout.
                    if continuous_frames:
                        _save_continuous_video(
                            self.run_dir, continuous_frames,
                            name_suffix=self.args.experiment_name,
                            high_quality=hq,
                        )
                    # Debug per-camera videos are redundant with the
                    # annotated per-primitive videos and only useful when
                    # deep-diagnosing — gated behind --save-debug-videos.
                    if self.args.save_debug_videos:
                        save_debug_videos(
                            self.run_dir,
                            self._debug_ext_frames, self._debug_wrist_frames,
                            self.hardware.fps,
                            high_quality=hq,
                        )
                finally:
                    signal.signal(signal.SIGINT, prev_sigint)
        if completed_cleanly:
            logging.info("\nSequence complete.")
        return True

    # ────────────────── Trained-policy primitive loop ──────────────────

    def _run_known_primitive(self, primitive: str, num_steps: int,
                             frames_out: list[FrameEntry] | None) -> bool:
        """Run the trained policy for one primitive with auto-advance + VLM/progress done-checks.

        Returns True if a success trigger fired (PROGRESS-DONE, VLM-DONE,
        Z-DONE, auto-advance) and False if the loop ran the full ``num_steps``
        without any trigger (P90-cap). Callers use the return value to gate
        dataset recording — P90-cap runs are inconclusive and shouldn't be
        used as VLA training demos.

        Loop body is a sequence of: pull-action → run state predicates
        (progress / VLM / z / auto-advance) → dispatch action. Each predicate
        is a small helper so this main method stays scannable.
        """
        args = self.args
        hw = self.hardware
        action_queue: collections.deque = collections.deque()
        consecutive_stop = 0
        vlm_streak = TemporalConsistency(required=max(1, args.vlm_consecutive_required))
        progress_streak = TemporalConsistency(required=max(1, args.progress_consecutive_required))

        is_move_to = primitive.lower().startswith("move gripper to")
        vlm_active = args.use_vlm_check and is_move_to
        z_safety_active = is_move_to and args.z_completion_threshold_mm > 0
        # Pure-gripper primitives (no EE motion) — auto-advance's goal-state delta
        # check fires immediately because the EE is supposed to stay stationary.
        # Disable auto-advance for these and rely on progress-check / VLM / P90.
        is_gripper_only = primitive.lower() in ("close gripper", "open gripper")

        # Per-primitive filter state for gimbal-lock rpy smoothing. Reset at
        # entry and on every transition out of the lock zone so we never
        # carry stale orientation across the singularity boundary.
        smoothed_quat: np.ndarray | None = None
        # Per-primitive previous state for rpy unwrap. ONLY applied to
        # primitives in _UNWRAP_PRIMITIVES (currently: twist); other
        # primitives received wrapped state in training and should also
        # receive wrapped state at inference.
        state_prev: np.ndarray | None = None
        primitive_needs_state_unwrap = primitive.lower() in _UNWRAP_PRIMITIVES

        for step in range(num_steps):
            t0 = time.perf_counter()
            ext, wrist = hw.capture_frames()
            self._debug_ext_frames.append(ext)
            self._debug_wrist_frames.append(wrist)

            if not action_queue:
                obs = hw.build_observation(primitive, ext, wrist)
                chunk = np.asarray(self.policy.infer(obs)["actions"], dtype=np.float32)
                for a in chunk[: args.replan_steps]:
                    action_queue.append(a)
            action = action_queue.popleft()

            frame_progress = float(action[7]) if len(action) >= 8 else float('nan')
            if frames_out is not None:
                frames_out.append((step, ext, wrist, frame_progress))
            if step % 10 == 0:
                self._log_step_heartbeat(step, num_steps, action)

            state = hw.get_pose()
            state_prev = self._apply_state_unwrap(state, state_prev, primitive_needs_state_unwrap)
            if self.recorder is not None:
                self.recorder.record_step(state, ext, wrist, hw.gripper_norm())

            goal = np.array(action[:6], dtype=np.float32)
            goal[3:6] = goal[3:6] / np.pi * 180
            smoothed_quat = self._apply_rpy_smoothing(goal, state, smoothed_quat)

            if self._check_progress_done(action, step, num_steps, is_gripper_only,
                                         primitive, progress_streak):
                return True

            if vlm_active and should_check_now(step, args.vlm_warmup_steps, args.vlm_check_interval_steps):
                verdict = self._timed_vlm(
                    check_primitive_done,
                    primitive, ext, wrist,
                    num_votes=args.vlm_num_votes, step_num=step,
                )
                if vlm_streak.update(verdict):
                    logging.info("  [VLM-DONE] '%s' at step %d/%d (streak=%d)",
                                 primitive, step, num_steps, vlm_streak.streak)
                    return True
                if verdict:
                    logging.info("  [VLM-MAYBE] '%s' streak=%d/%d at step %d/%d",
                                 primitive, vlm_streak.streak, vlm_streak.required, step, num_steps)

            if z_safety_active and self._check_z_done(state, primitive, step, num_steps):
                return True

            if args.auto_advance and not is_gripper_only and step >= args.min_steps:
                advanced, consecutive_stop = self._check_auto_advance(
                    goal, state, consecutive_stop,
                )
                if advanced:
                    return True

            if args.require_approval and step >= args.approval_warmup_steps:
                self._prompt_step_approval(primitive, step, num_steps, state, goal)

            hw.interpolate_action(state, goal, args.z_safety_floor, args.interp_divisor)

            # Gripper command from the policy. action[6] is in the [0=open, ~0.7=closed]
            # convention from collect_demo.py: raw_pos = 850 - 860*action[6].
            # Skipped when the end-effector isn't a gripper (e.g. scoop) — the
            # SDK call would error and the policy's action[6] is meaningless
            # in that setup.
            if hw.use_gripper and len(action) >= 7:
                gripper_target = 850 - 860 * float(action[6])
                hw.arm.set_gripper_position(gripper_target)

            time_left = hw.dt - (time.perf_counter() - t0)
            time.sleep(max(time_left, 0))

        logging.info("  [P90-CAP] '%s' ran full %d steps without VLM/Z trigger",
                     primitive, num_steps)
        return False

    # ────────────────── Inner-loop helpers ──────────────────

    def _log_step_heartbeat(self, step: int, num_steps: int,
                            action: np.ndarray) -> None:
        """Periodic state + action + grip + progress one-liner.

        Logs every 10th step. State vs action xyz divergence indicates clamp
        accumulation; grip pred vs actual divergence indicates grip slip;
        progress is the model's primitive-completion estimate."""
        hw = self.hardware
        state_now = hw.get_pose()
        grip_pred = float(action[6]) if len(action) >= 7 else float('nan')
        grip_actual_raw = hw.arm.get_gripper_position()[1] if hasattr(hw.arm, 'get_gripper_position') else None
        if grip_actual_raw is None:
            grip_actual_raw = 850
        grip_actual_norm = (grip_actual_raw - 850) / -860
        progress = float(action[7]) if len(action) >= 8 else float('nan')
        logging.info(
            "  step %3d/%d  prog=%.2f  grip %.2f→%.2f  "
            "state(x=%.0f y=%.0f z=%.0f rpy=%.1f,%.1f,%.1f)  "
            "action(x=%.0f y=%.0f z=%.0f rpy=%.1f,%.1f,%.1f)",
            step, num_steps, progress, grip_pred, grip_actual_norm,
            state_now[0], state_now[1], state_now[2],
            state_now[3], state_now[4], state_now[5],
            action[0], action[1], action[2],
            np.rad2deg(action[3]), np.rad2deg(action[4]), np.rad2deg(action[5]),
        )

    def _apply_state_unwrap(self, state: np.ndarray,
                            state_prev: np.ndarray | None,
                            needs_unwrap: bool) -> np.ndarray | None:
        """Apply per-primitive rpy unwrap to ``state`` (in place).

        Mirrors the ``filter_normalize_twist`` preprocess that np.unwraps
        training data, so the policy sees the same continuous numerical
        sequence across 2π boundaries at inference. Only fires for
        primitives in ``_UNWRAP_PRIMITIVES``; other primitives keep their
        training-time wrapped convention.

        Returns the new ``state_prev`` to be passed into the next tick.
        ``None`` when no unwrap is needed.
        """
        if not needs_unwrap:
            return None
        if state_prev is not None:
            for axis in (3, 4, 5):
                diff = state[axis] - state_prev[axis]
                if abs(diff) > 180:
                    state[axis] -= 360.0 * round(diff / 360.0)
        return state.copy()

    def _apply_rpy_smoothing(self, goal: np.ndarray, state: np.ndarray,
                             smoothed_quat: np.ndarray | None) -> np.ndarray | None:
        """Apply quaternion-slerp gimbal-lock smoothing to ``goal[3:6]`` (in place).

        At pitch≈±90° the ZYX-Euler chart is degenerate: equivalent physical
        orientations have multiple (roll, yaw) numerical representations, so
        the policy emits different rpy values for the same intended pose
        across ticks. Slerping in quaternion space collapses those equivalent
        representations (they map to ~the same quaternion) and tracks genuine
        rotation as the SO(3) geodesic — this only damps parameterization
        noise, not real motion.

        Returns the new ``smoothed_quat`` (or None when out of the lock zone).
        """
        args = self.args
        if args.rpy_smoothing_alpha >= 1.0 or abs(state[4]) < args.rpy_smoothing_pitch_threshold:
            return None
        command_q = rpy_to_quat_deg(goal[3:6])
        if smoothed_quat is None:
            # Seed from current state, not the (possibly noisy) command, so
            # the first in-lock tick doesn't latch a bad sample.
            smoothed_quat = rpy_to_quat_deg(state[3:6])
        smoothed_quat = slerp(smoothed_quat, command_q, args.rpy_smoothing_alpha)
        goal[3:6] = unwrap_rpy_near(quat_to_rpy_deg(smoothed_quat), state[3:6])
        return smoothed_quat

    def _check_progress_done(self, action: np.ndarray, step: int, num_steps: int,
                             is_gripper_only: bool, primitive: str,
                             progress_streak: TemporalConsistency) -> bool:
        """Check the policy's progress channel (action[7]) for primitive completion.

        Requires policy trained with progress + XarmOutputs returning >=8 dims.
        Warmup gate filters early-step progress glitches when the starting
        state is OOD (e.g. lift after twist) — bypassed for gripper-only
        primitives which legitimately finish in 2-3 steps.

        Logs ``[PROGRESS-DONE]`` and returns True on success-trigger; updates
        ``progress_streak`` in place either way.
        """
        args = self.args
        if not args.use_progress_check or len(action) < 8:
            return False
        progress = float(action[7])
        warmup = 0 if is_gripper_only else args.progress_warmup_steps
        if step < warmup:
            return False
        if not progress_streak.update(progress >= args.progress_threshold):
            return False
        logging.info("  [PROGRESS-DONE] '%s' at step %d/%d (progress=%.3f, streak=%d)",
                     primitive, step, num_steps, progress, progress_streak.streak)
        return True

    def _check_z_done(self, state: np.ndarray, primitive: str,
                      step: int, num_steps: int) -> bool:
        """Check whether the gripper (or tool tip) has descended below the
        z-completion threshold.

        When a tool offset is configured, the threshold applies to the
        FK-derived tool-tip world-Z (geometrically grounded). Otherwise it's
        the legacy TCP-Z heuristic. Logs ``[Z-DONE]`` on success-trigger.
        """
        args = self.args
        hw = self.hardware
        z_check = hw.tool_tip_z(state) if hw._has_tool_offset else state[2]
        if z_check >= args.z_completion_threshold_mm:
            return False
        label = "tip_z" if hw._has_tool_offset else "state_z"
        logging.info("  [Z-DONE] '%s' %s=%.2f < %.2f at step %d/%d",
                     primitive, label, z_check,
                     args.z_completion_threshold_mm, step, num_steps)
        return True

    def _check_auto_advance(self, goal: np.ndarray, state: np.ndarray,
                            consecutive_stop: int) -> tuple[bool, int]:
        """Auto-advance when BOTH translation AND rotation are stable.

        Rotation primitives (twist, pour, flip) keep xyz constant while
        orientation changes; without the rotation check they'd auto-advance
        immediately on the first xyz-stable tick. The rotation delta wraps
        to the nearest 360° so a principal-value flip (e.g. state=179°,
        goal=-179°) doesn't read as 358° of motion.

        Returns ``(advanced, new_consecutive_stop)``. ``advanced=True`` when
        the stop-step threshold has been reached and the loop should exit.
        """
        args = self.args
        goal_delta_mm = float(np.linalg.norm(goal[:3] - state[:3]))
        rot_diff = goal[3:6] - state[3:6]
        rot_diff = rot_diff - 360.0 * np.round(rot_diff / 360.0)
        goal_delta_deg = float(np.linalg.norm(rot_diff))
        pos_stable = goal_delta_mm < args.movement_threshold_mm
        rot_stable = goal_delta_deg < args.movement_threshold_deg
        if not (pos_stable and rot_stable):
            return False, 0
        consecutive_stop += 1
        if consecutive_stop < args.stop_steps:
            return False, consecutive_stop
        logging.info("  Auto-advance: pos<%.2fmm rot<%.2fdeg for %d steps",
                     args.movement_threshold_mm,
                     args.movement_threshold_deg, consecutive_stop)
        return True, consecutive_stop

    # ────────────────── Helpers ──────────────────

    def _finalize_primitive_artifacts(
        self,
        primitive: str,
        primitive_idx: int,
        plan_total: int,
        prim_frames: "list[FrameEntry] | None",
        *,
        ok: bool,
        user_skipped: bool,
        interrupted: bool,
        defer_recorder: bool = False,
        is_skill_gap: bool = False,
    ) -> None:
        """Commit (or discard) the recording, save the per-primitive video, and
        extend the combined video buffer.

        Designed to be called from a ``finally`` block so artifacts persist even
        when the inner primitive loop is interrupted (Ctrl+C). Filename suffix
        encodes the exit reason: ``_INTERRUPTED`` (Ctrl+C), ``_SKIPPED``
        (user pressed 'n'), ``_INCOMPLETE`` (P90-cap or skill-gap failure), or
        empty when the primitive succeeded.

        ``defer_recorder=True`` skips the save/discard call so the caller can
        gate commit on a later signal (used for skill gaps, where the success
        oracle runs after the whole plan finishes).
        """
        if self.recorder is not None and not defer_recorder:
            if ok:
                self.recorder.save_primitive()
            else:
                # Inconclusive / aborted / skipped → don't train VLA on it.
                self.recorder.discard_primitive()
        if ok:
            suffix = ""
        elif interrupted:
            suffix = "_INTERRUPTED"
        elif user_skipped:
            suffix = "_SKIPPED"
        else:
            suffix = "_INCOMPLETE"
        if self.args.save_videos and prim_frames and self.run_dir is not None:
            # Extend the combined buffer FIRST so the run-wide video survives
            # even if the per-primitive save below is interrupted by Ctrl+C.
            # Then ignore SIGINT during the per-primitive write so the file
            # isn't truncated by an impatient second Ctrl+C.
            self._extend_combined(
                primitive, primitive_idx, plan_total, prim_frames,
                is_skill_gap=is_skill_gap,
            )
            prev_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                save_primitive_video(
                    self.run_dir, primitive, primitive_idx, plan_total,
                    prim_frames, self.hardware.fps,
                    is_skill_gap=is_skill_gap, suffix=suffix,
                    high_quality=getattr(self.args, "high_quality_videos", False),
                )
            finally:
                signal.signal(signal.SIGINT, prev_sigint)

    def _prompt_step_approval(self, primitive: str, step: int, max_steps: int,
                              state: np.ndarray, goal: np.ndarray,
                              tag: str = "step") -> None:
        """Block on user ENTER before a single servo command.

        Shows current pose, commanded target, and the delta in mm + deg. Inputs:
        - ENTER: send the command
        - 'n': skip the rest of this primitive, advance to the next plan step
          (raises ``_SkipPrimitive``)
        - Ctrl+C: abort the entire run (propagates ``KeyboardInterrupt``)

        Note: ``goal`` is the long-term target — for skill gaps this is the
        full skill-gap end-pose, and a single ENTER advances roughly 5mm/tick
        toward it (interpolation damping + speed clamp). For known primitives
        ``goal`` is the per-tick policy target so motion ≈ delta in one tick.
        """
        delta = goal - state
        msg = (
            f"\n[{tag} {step}/{max_steps}] {primitive!r}\n"
            f"  current: x={state[0]:7.1f}  y={state[1]:7.1f}  z={state[2]:7.1f}  "
            f"rpy=({state[3]:6.1f}, {state[4]:6.1f}, {state[5]:6.1f})\n"
            f"  target:  x={goal[0]:7.1f}  y={goal[1]:7.1f}  z={goal[2]:7.1f}  "
            f"rpy=({goal[3]:6.1f}, {goal[4]:6.1f}, {goal[5]:6.1f})\n"
            f"  delta:   dx={delta[0]:+6.1f}  dy={delta[1]:+6.1f}  dz={delta[2]:+6.1f}mm  "
            f"dr=({delta[3]:+5.1f}, {delta[4]:+5.1f}, {delta[5]:+5.1f})°\n"
            f"  ENTER=send  'n'=skip primitive  Ctrl+C=abort run: "
        )
        response = self._timed_input(msg).strip().lower()
        if response == "n":
            raise _SkipPrimitive()

    def _capture_scene(self) -> tuple[np.ndarray, np.ndarray]:
        """Initial scene capture used for planning. ``dry_run`` returns blanks."""
        if self.args.dry_run:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            return blank, blank
        return self.hardware.capture_frames()

    def _next_trial_seq(self, batch_dir: Path) -> int:
        """Compute the trial sequence number — scoped per-experiment when
        possible so Ctrl+C and resume continues numbering across batches.

        Priority:
        1. ``record_dataset_repo`` set → count rows in the dataset's
           ``flywheel_trials.jsonl`` (which main.py keeps in the LeRobot
           cache dir).
        2. ``experiment_name`` set → count rows in the experiment's jsonl
           under ``video_dir/<experiment_name>/``.
        3. Neither set → fall back to counting ``trial_*`` subdirs in
           ``batch_dir`` (per-batch numbering only).
        """
        args = self.args
        jsonl: Path | None = None
        if getattr(args, "record_dataset_repo", ""):
            jsonl = (
                Path.home() / ".cache" / "huggingface" / "lerobot"
                / args.record_dataset_repo / "flywheel_trials.jsonl"
            )
        elif getattr(args, "experiment_name", ""):
            jsonl = Path(args.video_dir) / args.experiment_name / "flywheel_trials.jsonl"
        if jsonl is not None and jsonl.exists():
            try:
                with open(jsonl) as f:
                    completed = 0
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        # Aborts ("don't count this") shouldn't bump the
                        # trial counter — they're skipped in analyze too.
                        try:
                            row = json.loads(line)
                        except Exception:
                            continue
                        if row.get("interrupt_outcome") == "abort":
                            continue
                        completed += 1
                return completed + 1
            except Exception:
                pass  # fall through to dir-count
        return len([p for p in batch_dir.iterdir()
                    if p.is_dir() and p.name.startswith("trial_")]) + 1

    def _setup_run_dir(self, plan_summary: str) -> Path | None:
        """Reuse the run-output dir resolved at logging-setup time and write
        a metadata snapshot.

        Each trial gets its OWN ``trial_NN_HHMMSS/`` subdirectory under the
        batch run_dir. Before this, every trial in a batch reused the same
        directory and overwrote primitive_*.mp4 / all_primitives.mp4 — so
        the only recoverable video was the last trial's. Per-trial subdirs
        keep failures around for review (the pour batch on 2026-05-14 lost
        a failure case this way)."""
        args = self.args
        if not args.save_videos:
            return None
        # Reuse the dir already created in main._setup_logging so the .log file
        # and the videos cluster together. Falls back to fresh resolution if
        # this runner was invoked outside the main entry point (rare).
        batch_dir = getattr(args, "_resolved_run_dir", None)
        if batch_dir is None:
            now = datetime.now()
            batch_dir = (
                Path(args.video_dir)
                / now.strftime("%Y-%m-%d")
                / (args.run_name or f"run_{now.strftime('%Y%m%d_%H%M%S')}")
            )
            batch_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        # Sequence number scoped to the experiment (not the batch_dir) so
        # Ctrl+C and resume continues numbering instead of restarting at 01
        # in a new batch's empty subdir. Counts rows in the experiment's
        # per-trial jsonl, since main.py appends one row per completed
        # trial. Falls back to counting trial_* dirs in the current
        # batch_dir when no experiment is set.
        seq = self._next_trial_seq(batch_dir)
        run_dir = batch_dir / f"trial_{seq:02d}_{now.strftime('%H%M%S')}"
        # Same-second collision protection — if the path somehow already
        # exists (e.g., two trials started in the same second under a
        # restored experiment), bump the sequence until free.
        while run_dir.exists():
            seq += 1
            run_dir = batch_dir / f"trial_{seq:02d}_{now.strftime('%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "metadata.txt", "w") as f:
            f.write(f"Run:           {run_dir.name}\n")
            f.write(f"Timestamp:     {now.isoformat()}\n")
            f.write(f"Mode:          {type(self).__name__}\n")
            f.write(f"VLM provider:  {args.vlm_provider}\n")
            f.write(f"FPS:           {self.hardware.fps}\n")
            f.write(f"Z safety:      floor={args.z_safety_floor} done={args.z_completion_threshold_mm}\n")
            f.write(f"VLM check:     enabled={args.use_vlm_check} consecutive={args.vlm_consecutive_required}\n")
            f.write(f"\nPlan:\n{plan_summary}\n")
            f.write(f"\nCommand: {' '.join(sys.argv)}\n")
        logging.info("Saving run to: %s", run_dir)
        return run_dir

    def _extend_combined(self, primitive: str, primitive_idx: int, plan_total: int,
                         frames: list[FrameEntry],
                         *, is_skill_gap: bool = False) -> None:
        """Append per-primitive frames to the run-wide combined video buffer.

        Each appended entry is a structured label dict consumed by
        ``video._annotate`` for colored rendering: counter in light blue,
        ``[known]`` in green / ``[SKILL-GAP]`` in red, primitive name in
        off-white, step / progress / global-step right-aligned in dim grey.
        """
        counter = f"{primitive_idx + 1}/{plan_total}"
        tag = "primitive gap" if is_skill_gap else "known"
        for step, ext, wrist, progress in frames:
            parts = {
                "counter": counter,
                "tag": tag,
                "primitive": primitive,
                "step": step,
                "progress": progress,
                "global_step": self._global_step,
            }
            self._combined_frames.append((parts, ext, wrist))
            self._global_step += 1
