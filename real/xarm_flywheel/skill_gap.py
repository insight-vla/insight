"""Skill-gap execution: VLM pre-analysis + axis-appropriate motion control.

Mixin used by ``XArmFlywheelExecutor``. Provides:

- ``_execute_skill_gap`` — dispatches to translation or rotation execution
  after running pre-analysis + optional user confirmation.
- ``_execute_translation_servo`` — P-control servo loop with optional
  VLM-based mid-trajectory done-checks (sweep until cleared, move-to
  until centered).
- ``_execute_rotation_planned`` — controller-planned ``set_position``
  motion for smooth single-axis rotation arcs (twist, pour, flip).
- ``_preanalyze_translation_motion`` — VLM call returning the motion
  target; handles both translation (meters) and rotation (degrees) axes.

The asymmetry between translation and rotation paths is intentional:
translation benefits from mid-trajectory done-checks (the goal is a
region the gripper enters; the policy can't always predict when it's
"there enough"), while rotation is one-shot (joint range is finite,
partial rotation isn't a meaningful stop point).

The motion-target data type (``SkillGapMotion``) lives in
``insight.skill_gap`` so the same contract is used by both the LIBERO
sim flywheel and this xArm executor.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from insight.reasoning import (
    check_primitive_done,
    parse_signed_magnitude_m,
    preanalyze_translation,
)
from insight.images import WRIST_AXES_XARM, draw_wrist_axes
from insight.skill_gap import AXIS_INDEX, ROTATION_AXES, SkillGapMotion
from insight.voting import TemporalConsistency, should_check_now
from insight.rotation import (
    build_target_quat,
    quat_error_to_world_rotvec,
    quat_to_rpy_deg,
    rotate_vector_by_quat,
    rpy_to_quat_deg,
    unwrap_rpy_near,
)

from .runner import FrameEntry, _AbortTrial, _SkipPrimitive


# Below this magnitude, we treat the VLM's preanalyze translation as "no
# motion needed" — saves a P-control loop on tiny corrections.
_NEGLIGIBLE_DELTA_M = 0.005

# Translation gap below which the gripper is considered "at target" — used to
# trigger an extend in the P-control loop (the policy can't push past the
# physical stop, so reaching this gap means we're done with the current target).
_AT_TARGET_GAP_MM = 5.0
_AT_TARGET_GAP_DEG = 3.0  # angular tolerance for rotation skill gaps

# Tolerance below which a state coordinate is considered "at the workspace
# bound" — used to detect motion stall after bound-clamping fires.
_AT_BOUND_TOLERANCE_MM = 1.0

# Consecutive at-bound steps required to terminate the skill gap. 1 = exit
# the moment the arm reaches a bound on the motion axis. Minimizes the time
# the gripper spends pressing into whatever is at the bound (e.g., the bin
# wall when the VLM picks a wrong-direction sweep). At 50 mm/s caps and
# stable xArm pose readings, false positives from a single tick are rare.
_BOUND_HIT_STREAK = 1

# Backwards-compat alias for any external readers; new code should import
# from ``insight.skill_gap`` directly.
_SkillGapMotion = SkillGapMotion


class _SkillGapMixin:
    """Skill-gap execution methods. Mixed into ``XArmFlywheelExecutor``.

    Host class must provide: ``self.args``, ``self.hardware``,
    ``self.recorder``, ``self._debug_ext_frames``, ``self._debug_wrist_frames``,
    ``self._last_skill_gap_summary``, ``self._plan_context``,
    ``self._timed_vlm``, ``self._timed_input``, ``self._prompt_step_approval``.
    """

    # ────────────────── Skill gap (VLM pre-analysis + dispatch) ──────────────────

    def _execute_skill_gap(self, primitive: str,
                           frames_out: list[FrameEntry] | None) -> bool:
        """Skill-gap dispatcher: VLM pre-analysis, then axis-appropriate execution.

        Translation skill-gaps run a P-control servo loop with optional
        per-tick VLM done-checks (early exit when the goal is achieved
        before the predicted target). Rotation skill-gaps switch to
        controller-planned motion (mode 0 / set_position) so the trajectory
        matches the smooth web-teleop arc instead of the discrete-target
        servo chase. The asymmetry is intentional: translation benefits
        from mid-trajectory done-checks (sweep until cleared, move-to until
        centered), rotation is one-shot.

        Returns True on done; False on parse error, max-steps cap, or
        safety abort.
        """
        hw = self.hardware

        # Pre-analysis loop: when --confirm-skill-gap is on, the user can press
        # 'r' to re-call the VLM if the proposed motion is wrong. Each retry
        # captures fresh frames and re-runs preanalyze_translation. Up to
        # _MAX_RETRIES attempts; after that the user must accept or abort.
        _MAX_RETRIES = 2
        for attempt in range(_MAX_RETRIES + 1):
            initial_ext, initial_wrist = hw.capture_frames()
            try:
                motion = self._timed_vlm(
                    self._preanalyze_translation_motion,
                    primitive, initial_ext, initial_wrist,
                )
            except Exception as e:
                logging.error("  [SKILL-GAP] pre-analysis failed: %s", e)
                return False
            if motion is None:
                return True

            if not self.args.confirm_skill_gap:
                break

            if motion.is_rotation:
                summary = f"rotation along {motion.axis} by {motion.delta_deg:+.1f}° (planned set_position)"
            else:
                summary = f"translation along {motion.axis} by {motion.delta_m * 1000:+.1f}mm (servo P-control)"
            resp = self._timed_input(
                f"  [CONFIRM] About to run skill gap {primitive!r}: {summary}. "
                f"ENTER=proceed, r=retry VLM, f=fail trial, Ctrl+C=abort: "
            ).strip().lower()

            if resp in ("r", "retry"):
                if attempt < _MAX_RETRIES:
                    logging.info("  [SKILL-GAP] user requested retry; re-calling VLM (%d/%d)...",
                                 attempt + 1, _MAX_RETRIES)
                    continue
                logging.warning("  [SKILL-GAP] hit retry cap (%d); proceeding with current motion.", _MAX_RETRIES)
            elif resp in ("f", "fail", "x", "reject"):
                logging.warning(
                    "  [SKILL-GAP] user rejected motion; aborting trial. "
                    "Subsequent primitives depend on this gap, so the whole "
                    "trial is marked failed (no motion sent to robot)."
                )
                raise _AbortTrial("user rejected skill-gap motion")
            break  # ENTER (or anything else) -> accept

        if motion.is_rotation:
            return self._execute_rotation_planned(motion, primitive, frames_out)
        return self._execute_translation_servo(motion, primitive, frames_out)

    # ────────────────── Translation: servo + VLM done-check ──────────────────

    def _execute_translation_servo(self, motion: "_SkillGapMotion", primitive: str,
                                   frames_out: list[FrameEntry] | None) -> bool:
        """Translation skill-gap via servo-mode P-control toward an axis target.

        Drives the gripper along ``motion.axis`` toward the VLM-specified
        target. Extends past the target up to ``skill_gap_max_extends``
        times if the VLM done-check hasn't fired (handles VLM
        underestimating the magnitude). Early-exits when the workspace
        bound on the motion axis is hit; the post-plan oracle judges
        whether the partial motion counted.
        """
        args = self.args
        hw = self.hardware

        initial_pose = hw.get_pose()
        target_pose = initial_pose.copy()
        target_pose[motion.axis_idx] += motion.delta_native
        if target_pose[2] < args.z_safety_floor:
            logging.warning("  [SKILL-GAP] initial target z=%.1fmm below safety floor; clamping.",
                            target_pose[2])
            target_pose[2] = args.z_safety_floor

        logging.info("  [SKILL-GAP] %s start=%.2fmm  delta=%+.2fmm  initial target=%.2fmm",
                     motion.axis, initial_pose[motion.axis_idx],
                     motion.delta_native, target_pose[motion.axis_idx])

        vlm_streak = TemporalConsistency(required=max(1, args.vlm_consecutive_required))
        extends = 0
        was_at_target = False
        bound_hit_streak = 0

        for step in range(args.skill_gap_max_steps):
            t0 = time.perf_counter()
            ext, wrist = hw.capture_frames()
            # Skill-gap "progress" = step/max_steps (no policy progress channel
            # since this path is P-control, not VLA-driven). Matches FrameEntry's
            # 4-tuple shape so the per-primitive video annotator works.
            sg_progress = step / max(args.skill_gap_max_steps - 1, 1)
            if frames_out is not None:
                frames_out.append((step, ext, wrist, sg_progress))
            self._debug_ext_frames.append(ext)
            self._debug_wrist_frames.append(wrist)
            if self.recorder is not None:
                self.recorder.record_step(hw.get_pose(), ext, wrist, hw.gripper_norm())

            if args.use_vlm_check and should_check_now(
                step, args.vlm_warmup_steps, args.vlm_check_interval_steps,
            ):
                try:
                    verdict = self._timed_vlm(
                        check_primitive_done,
                        primitive, ext, wrist,
                        num_votes=args.vlm_num_votes, step_num=step,
                    )
                except Exception as e:
                    logging.warning("  [SKILL-GAP] VLM done-check failed at step %d: %s", step, e)
                    verdict = False
                if vlm_streak.update(verdict):
                    logging.info("  [SKILL-GAP-DONE] %r at step %d/%d (streak=%d)",
                                 primitive, step, args.skill_gap_max_steps, vlm_streak.streak)
                    return True
                if verdict:
                    logging.info("  [SKILL-GAP-MAYBE] streak=%d/%d at step %d/%d",
                                 vlm_streak.streak, vlm_streak.required,
                                 step, args.skill_gap_max_steps)

            try:
                state = hw.get_pose()
                if args.require_approval:
                    self._prompt_step_approval(primitive, step, args.skill_gap_max_steps,
                                               state, target_pose, tag="skill-gap step")
                # Linear interpolation along the motion axis. Capped by
                # max_tcp_speed_mm_s so a single tick can't overshoot.
                max_rate = args.max_tcp_speed_mm_s
                step_amount = max_rate * hw.dt if max_rate > 0 else float("inf")
                subgoal = state.copy()
                remaining = target_pose[motion.axis_idx] - state[motion.axis_idx]
                if abs(remaining) <= step_amount:
                    subgoal[motion.axis_idx] = target_pose[motion.axis_idx]
                else:
                    subgoal[motion.axis_idx] += motion.direction * step_amount
                hw.interpolate_action(state, subgoal, args.z_safety_floor, args.interp_divisor)
                new_state = hw.get_pose()
            except _SkipPrimitive:
                raise
            except Exception as e:
                # Most likely an xArm SDK error (collision fault, motion abort,
                # or boundary refusal). Log with context, attempt to recover the
                # arm state, and abort the primitive cleanly.
                logging.error("  [SKILL-GAP] motion failed at step %d/%d: %s",
                              step, args.skill_gap_max_steps, e)
                try:
                    if hw.arm is not None:
                        arm_state = hw.arm.get_state()
                        logging.error("  [SKILL-GAP] arm state=%s — calling clean_error()", arm_state)
                        hw.arm.clean_error()
                except Exception as cleanup_err:
                    logging.error("  [SKILL-GAP] clean_error failed: %s", cleanup_err)
                return False

            gap = (target_pose[motion.axis_idx] - new_state[motion.axis_idx]) * motion.direction
            is_at_target = gap < _AT_TARGET_GAP_MM
            if is_at_target and not was_at_target:
                extends += 1
                if extends > args.skill_gap_max_extends:
                    logging.warning("  [SKILL-GAP] reached target %d times without VLM done; "
                                    "stopping at step %d/%d.", extends, step, args.skill_gap_max_steps)
                    return False
                target_pose[motion.axis_idx] += motion.direction * motion.extend_mm
                if target_pose[2] < args.z_safety_floor:
                    logging.warning("  [SKILL-GAP] extended target z=%.1fmm below safety floor; "
                                    "clamping and stopping.", target_pose[2])
                    target_pose[2] = args.z_safety_floor
                    return False
                logging.info("  [SKILL-GAP] target reached (gap=%.1fmm), extending %s to %.1fmm "
                             "(extend %d/%d)", gap, motion.axis, target_pose[motion.axis_idx],
                             extends, args.skill_gap_max_extends)
                is_at_target = False
            was_at_target = is_at_target

            # Workspace bound-hit early termination — only on the motion axis.
            # Sitting at the z-floor (or any non-motion-axis bound) is not a
            # stall: it just means we started already pressed against a
            # non-motion bound (e.g. move-to-rocks ends at z=z_min).
            if hw._bounds:
                axis_idx = motion.axis_idx
                axis_min = hw._bounds[2 * axis_idx]
                axis_max = hw._bounds[2 * axis_idx + 1]
                at_motion_axis_bound = (
                    abs(new_state[axis_idx] - axis_min) < _AT_BOUND_TOLERANCE_MM
                    or abs(new_state[axis_idx] - axis_max) < _AT_BOUND_TOLERANCE_MM
                )
                if at_motion_axis_bound:
                    bound_hit_streak += 1
                    if bound_hit_streak >= _BOUND_HIT_STREAK:
                        logging.info(
                            "  [SKILL-GAP-BOUND] %s axis stalled at workspace bound "
                            "for %d steps; ending skill gap at step %d/%d (oracle will judge).",
                            motion.axis, bound_hit_streak, step, args.skill_gap_max_steps,
                        )
                        return False
                else:
                    bound_hit_streak = 0

            time_left = hw.dt - (time.perf_counter() - t0)
            time.sleep(max(time_left, 0))

        logging.warning("  [SKILL-GAP-CAP] %r hit max_steps=%d without VLM done.",
                        primitive, args.skill_gap_max_steps)
        return False

    # ────────────────── Rotation: planned motion (mode 0) ──────────────────

    def _execute_rotation_planned(self, motion: "_SkillGapMotion", primitive: str,
                                  frames_out: list[FrameEntry] | None) -> bool:
        """Execute a rotation skill-gap via xArm controller-planned motion.

        Switches to position-control (mode 0) and fires
        ``set_position(wait=False)`` with the world-axis-rotated target
        RPY. The xArm controller plans a smooth single-trajectory move
        (same path as Studio's web teleop jog buttons) while we record
        frames + poses at the configured fps and poll the quat-error gap
        until below ``_AT_TARGET_GAP_DEG``. Servo mode is restored on exit
        so subsequent primitives behave normally.

        One-shot: no extends past the target (joint 6 range is finite, and
        the VLM's predicted magnitude is the requested motion, not a lower
        bound) and no mid-trajectory VLM done-checks (the cap either turns
        or doesn't, partial rotation isn't a meaningful stop point).
        """
        args = self.args
        hw = self.hardware

        initial_pose = hw.get_pose()
        q_initial = rpy_to_quat_deg(initial_pose[3:6])
        # EE-local frame: "+dry" always means "tilt forward relative to the
        # gripper's current orientation" regardless of how the EE is currently
        # rotated (e.g., gripper-pointing-down for side-grasp pour). World-frame
        # produces opposite-sign motion on a flipped EE, which makes axis+sign
        # reasoning brittle. Local frame matches the VLM/user's intuitive
        # description ("tilt forward to pour").
        q_target = build_target_quat(q_initial, motion.axis, motion.delta_deg, frame="local")
        # Unwrap target RPY to the same branch as the arm's current RPY so
        # the controller doesn't plan a 360° joint swing for a physically
        # equivalent pose.
        target_rpy = unwrap_rpy_near(quat_to_rpy_deg(q_target), initial_pose[3:6])

        # Pivot-around-tool-tip: when ``tool_offset_mm`` is configured, compute
        # a TCP target such that the tool tip's world position stays fixed
        # before/after the rotation. Without this, the tool tip swings through
        # an arc of radius ||tool_offset|| during the rotation — for a side-
        # grasp pour, that means the bottle mouth lands ~10cm forward of the
        # bowl instead of over it. The post-rotation position is exact; the
        # mid-trajectory path may arc slightly depending on the controller's
        # interpolation, but for ≤90° rotations that residual is small enough
        # to keep the pour stream landing in the bowl.
        if hw._has_tool_offset:
            tool_offset_local = hw._tool_offset_local.astype(float)
            tip_world = initial_pose[:3] + rotate_vector_by_quat(q_initial, tool_offset_local)
            tcp_target = tip_world - rotate_vector_by_quat(q_target, tool_offset_local)
            logging.info("  [SKILL-GAP] %s rotation %+.1fdeg (planned, pivot=tip)  init RPY=%s  target RPY=%s  TCP %s -> %s  tip_world=%s",
                         motion.axis, motion.delta_deg,
                         np.array2string(initial_pose[3:6], precision=2),
                         np.array2string(target_rpy, precision=2),
                         np.array2string(initial_pose[:3], precision=1),
                         np.array2string(tcp_target, precision=1),
                         np.array2string(tip_world, precision=1))
        else:
            tcp_target = initial_pose[:3]
            logging.info("  [SKILL-GAP] %s rotation %+.1fdeg (planned, pivot=TCP)  init RPY=%s  target RPY=%s",
                         motion.axis, motion.delta_deg,
                         np.array2string(initial_pose[3:6], precision=2),
                         np.array2string(target_rpy, precision=2))

        hw.begin_planned_motion()
        try:
            try:
                code = hw.arm.set_position(
                    x=float(tcp_target[0]), y=float(tcp_target[1]), z=float(tcp_target[2]),
                    roll=float(target_rpy[0]), pitch=float(target_rpy[1]), yaw=float(target_rpy[2]),
                    speed=float(args.skill_gap_rotation_speed),
                    mvacc=float(args.skill_gap_rotation_mvacc),
                    wait=False,
                )
                if code != 0:
                    logging.error("  [SKILL-GAP] set_position returned code=%d, aborting rotation.", code)
                    return False

                # Poll until the target is reached. No fixed step cap — instead
                # a stall detector aborts only if the gap stops decreasing
                # (controller stuck), preventing an infinite hang while letting
                # slow-but-progressing trajectories complete naturally.
                _STALL_FRAMES = 100        # ~5s at 20Hz of no progress
                _STALL_GAP_EPSILON = 0.05  # deg — any improvement larger than this resets the stall counter
                reached = False
                last_gap = float('inf')
                no_progress_count = 0
                step = 0
                while True:
                    t0 = time.perf_counter()
                    ext, wrist = hw.capture_frames()
                    sg_progress = step / max(args.skill_gap_max_steps - 1, 1)
                    if frames_out is not None:
                        frames_out.append((step, ext, wrist, sg_progress))
                    self._debug_ext_frames.append(ext)
                    self._debug_wrist_frames.append(wrist)
                    if self.recorder is not None:
                        self.recorder.record_step(hw.get_pose(), ext, wrist, hw.gripper_norm())

                    state = hw.get_pose()
                    q_now = rpy_to_quat_deg(state[3:6])
                    gap = float(np.rad2deg(np.linalg.norm(quat_error_to_world_rotvec(q_target, q_now))))
                    if gap < _AT_TARGET_GAP_DEG:
                        reached = True
                        logging.info("  [SKILL-GAP] rotation target reached (gap=%.2fdeg).", gap)
                        break

                    # Stall detection: if gap isn't shrinking, controller is stuck.
                    if last_gap - gap > _STALL_GAP_EPSILON:
                        no_progress_count = 0
                    else:
                        no_progress_count += 1
                    last_gap = gap
                    if no_progress_count >= _STALL_FRAMES:
                        logging.warning("  [SKILL-GAP-STALL] %r gap=%.2fdeg not decreasing for %d frames; controller stuck.",
                                        primitive, gap, _STALL_FRAMES)
                        break

                    time_left = hw.dt - (time.perf_counter() - t0)
                    time.sleep(max(time_left, 0))
                    step += 1

                return reached
            except KeyboardInterrupt:
                # Planned motion runs autonomously on the controller — Ctrl+C
                # only interrupts our polling loop, the arm keeps executing
                # the trajectory. Issue a deceleration-stop (state=6) so the
                # arm halts gracefully before our handlers re-raise. Without
                # this, the user's Ctrl+C feels unresponsive (arm finishes
                # the rotation regardless).
                logging.info("  [SKILL-GAP] Ctrl+C — decelerating planned motion to a stop.")
                try:
                    hw.arm.set_state(6)
                    time.sleep(0.3)  # let the controller settle
                except Exception as stop_err:
                    logging.error("  [SKILL-GAP] decel-stop failed: %s", stop_err)
                raise
        finally:
            hw.end_planned_motion()

    # ────────────────── Pre-analysis parsing ──────────────────

    def _preanalyze_translation_motion(self, primitive: str, ext: np.ndarray,
                                       wrist: np.ndarray) -> _SkillGapMotion | None:
        """Run VLM pre-analysis and parse the response into a motion target.

        Translation axes (dx/dy/dz) read ``signed_magnitude_m`` (meters);
        rotation axes (drx/dry/drz) read ``signed_magnitude_deg`` (degrees).

        Returns ``None`` when the VLM reports goal already complete or proposes
        a near-zero magnitude — caller should treat as a no-op success.
        Raises ``ValueError`` for unparseable axis / magnitude.
        """
        logging.info("  [SKILL-GAP] pre-analyzing motion for %r ...", primitive)
        # Prefer the full-FOV wrist frame (camera's native 16:9) over the
        # policy-cropped 320x240 4:3 — the VLM does better axis selection
        # when it can see the full scene. Falls back to the passed-in
        # wrist if the hardware didn't cache one (e.g., tests).
        wrist_full = getattr(self.hardware, "last_wrist_full", None)
        wrist_for_vlm = wrist_full if wrist_full is not None else wrist
        # Downsize 640x360 → 320x180 — same width as the legacy 320x240
        # input, similar pixel count (57600 vs 76800), 16:9 aspect that
        # preserves the camera's native horizontal FOV. Avoids sending
        # 3x the bytes per VLM call.
        if wrist_for_vlm.shape[:2] != (180, 320):
            import cv2
            wrist_for_vlm = cv2.resize(
                wrist_for_vlm, (320, 180), interpolation=cv2.INTER_AREA,
            )
        # Overlay was removed — empirically the colored axis arrows
        # confused the VLM (it would associate "+y arrow points left"
        # with "the goal target is in the +y direction" regardless of
        # where the target actually was in the image). VLM now does
        # pure visual reasoning + motion-name → axis mapping.
        wrist_with_axes = wrist_for_vlm

        # Persist the exact images sent to the VLM so failed trials can be
        # inspected after the fact (e.g., to verify the overlay matches the
        # actual gripper pose, or to see what the VLM was looking at when
        # it picked a wrong axis).
        run_dir = getattr(self, "run_dir", None)
        if run_dir is not None:
            try:
                from PIL import Image as _Image
                safe = "".join(c if c.isalnum() or c in "_-" else "_"
                               for c in primitive)[:60]
                _Image.fromarray(ext).save(run_dir / f"preanalysis_ext_{safe}.png")
                _Image.fromarray(wrist_with_axes).save(
                    run_dir / f"preanalysis_wrist_{safe}.png"
                )
            except Exception as e:
                logging.warning("  [SKILL-GAP] could not save pre-analysis images: %s", e)

        pre = preanalyze_translation(
            primitive, ext, wrist_with_axes,
            goal=self.args.goal,
            plan_context=getattr(self, "_plan_context", None),
            prior_skill_gap=getattr(self, "_last_skill_gap_summary", None),
        )

        # No truncation on these fields — the VLM's reasoning is the most
        # important piece of evidence for paper analysis (e.g., whether the
        # axis was picked from a recipe shortcut vs from scene-grounded
        # geometric reasoning), so we log them in full. Previously these
        # were truncated to 100 chars and analysis required digging into
        # the raw JSON dump in the run folder.
        logging.info("  [SKILL-GAP] current=%s", str(pre.get("current_state", "")))
        logging.info("  [SKILL-GAP] target =%s", str(pre.get("target_state", "")))
        logging.info("  [SKILL-GAP] reason =%s", str(pre.get("reasoning", "")))

        if pre.get("already_complete"):
            logging.info("  [SKILL-GAP] VLM says goal already complete — skipping motion.")
            return None

        axis = str(pre.get("axis", "")).strip().lower()
        if axis not in AXIS_INDEX:
            raise ValueError(f"unparseable axis: {pre.get('axis')!r}")

        if axis in ROTATION_AXES:
            try:
                delta_deg = float(pre.get("signed_magnitude_deg", 0.0))
            except (TypeError, ValueError):
                raise ValueError(
                    f"unparseable signed_magnitude_deg={pre.get('signed_magnitude_deg')!r}"
                )
            if abs(delta_deg) < 1.0:  # < 1 deg is below noise / not worth executing
                logging.warning("  [SKILL-GAP] tiny rotation delta %.2f deg — treating as already complete.",
                                delta_deg)
                return None
            self._last_skill_gap_summary = (
                f"'{primitive}' rotation axis={axis} by {delta_deg:+.1f}°"
            )
            return _SkillGapMotion(axis=axis, delta_deg=delta_deg)

        # Translation
        delta_m = parse_signed_magnitude_m(pre.get("signed_magnitude_m", 0.0))
        if delta_m is None:
            raise ValueError(
                f"unparseable signed_magnitude_m={pre.get('signed_magnitude_m')!r}"
            )
        if abs(delta_m) < _NEGLIGIBLE_DELTA_M:
            logging.warning("  [SKILL-GAP] tiny VLM delta %.4f m — treating as already complete.",
                            delta_m)
            return None
        self._last_skill_gap_summary = (
            f"'{primitive}' translation axis={axis} by {delta_m * 1000:+.1f}mm"
        )
        return _SkillGapMotion(axis=axis, delta_m=delta_m)
