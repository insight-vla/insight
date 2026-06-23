"""Flywheel execution: plan, step dispatch, skill gap P-control, runner, main()."""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import re
import traceback
import json
from datetime import datetime, timedelta, timezone

import imageio
import numpy as np
from PIL import Image
from .env import (
    AVAILABLE_PRIMITIVES,
    DEFAULT_PRIMITIVE_STEPS,
    DEFAULT_SCENE_CONTEXT,
    PRIMITIVE_P90_DURATIONS,
    TaskPlan,
    _find_robot,
    _start_keyboard_listener,
    _stop_event,
    create_env,
    get_gripper_state,
    get_obs_images,
    resize_for_vlm,
    reset_gripper_pose,
    settle_physics,
    set_display_enabled,
    stop_requested,
)
from .vlm import parse_vlm_json, set_vlm_provider, vlm_with_images
from .reasoning import (
    _check_goal_achieved_sim,
    _evaluate_progress_sim,
    _evaluate_progress_v2,
    _generate_action_wrapper,
    _preanalyze_skill_gap,
)
from .control import (
    _K_P_ROT,
    _K_P_TRANS,
    _MAX_POS_HOLD_CMD,
    _MAX_TRANS_CMD,
    _build_target_quat,
    _get_ee_quat_wxyz,
    _quat_error_to_world_rotvec,
    _return_to_ee_pose,
    blocks_too_close,
    tilt_red_block_to_side,
)
from .recording import _make_recording_step, _save_raw_hdf5
from .curation import curate_batch
from .base_execution import LiberoSimExecutor, _save_adaptive_results
from .prompts import (
    _GENERATE_NEW_PRIMITIVE_SYSTEM_FLYWHEEL,
    _PLAN_TASK_SYSTEM_FLYWHEEL,
)


# =============================================================================
# Module-level constants
# =============================================================================
# Mutable per-run state (skill-gap progress, recording buffer, current plan)
# now lives on the LiberoFlywheelExecutor instance — see __init__.

# Behavior flags read by LiberoFlywheelExecutor.evaluate_progress and by
# reasoning._generate_action_wrapper. Kept at module level because they
# affect the choice of helper, not per-run state.
_USE_VLM_PROGRESS_CHECK = False
_USE_VLM_ACTION_GENERATION = False


def _new_skill_gap_state() -> dict:
    """Per-run skill-gap state. Created fresh in LiberoFlywheelExecutor.__init__."""
    return {
        "goal_complete": False,
        "preanalysis_hint": "",
        "preanalysis_drz_deg": None,
        "preanalysis_for": "",
        "rotation_complete": False,
        "target_rotation_deg": 90,
    }


def _new_raw_recording() -> dict:
    """Per-run HDF5 recording state. Created fresh in LiberoFlywheelExecutor.__init__."""
    return {"buffer": [], "last_obs": None, "primitive": "", "enabled": False}


def _new_current_plan() -> dict:
    """Per-run plan tracking. Created fresh in LiberoFlywheelExecutor.__init__."""
    return {"sequence": [], "step_notes": [], "current_idx": 0}


# =============================================================================
# Executor
# =============================================================================


class LiberoFlywheelExecutor(LiberoSimExecutor):
    """Flywheel mode for LIBERO sim.

    Subclasses ``LiberoSimExecutor`` to add:

    - **State** — per-run skill-gap, recording, and plan dicts as instance
      attributes.
    - **Method overrides** — ``execute_step`` / ``evaluate_progress`` /
      ``generate_action`` / ``execute_plan`` provide flywheel-specific
      behavior; ``GENERATE_PROMPT`` switches to the axis-aware FLYWHEEL
      prompt.
    - **Helpers** — ``_plan_task_with_notes`` and ``_execute_skill_gap`` are
      private methods that own the planning + P-control logic.
    - **Entry point** — ``run_flywheel_adaptive`` is the public top-level
      method (called by ``_run_single``).
    """

    GENERATE_PROMPT = _GENERATE_NEW_PRIMITIVE_SYSTEM_FLYWHEEL

    _MAX_REPLANS: int = 3
    _MAX_SKILL_GAP_STEPS: int = 800

    def __init__(self, args: "FlywheelArgs"):
        super().__init__(args)
        self._skill_gap_state = _new_skill_gap_state()
        self._raw_recording = _new_raw_recording()
        self._current_plan = _new_current_plan()
        # Iteration feedback (loaded from --feedback_dir in main()) gets spliced
        # into the pre-analysis system prompt; empty string = no injection.
        self._preanalyze_feedback: str = ""
        # Live cv2 display is a process-wide flag read by env.get_obs_images.
        set_display_enabled(args.display)

    # ────────────────── Per-step dispatch ──────────────────

    def execute_step(self, env, client, step, obs, output_dir,
                     primitives_tried, all_frames, all_datapoints):
        """Skill gaps → ``_execute_skill_gap`` (P-control, sim peg-direction
        goal check); known primitives → inherited base dispatch."""
        is_known = step in AVAILABLE_PRIMITIVES
        # HDF5 recording is scoped to skill-gap primitives (the new skills
        # being learned); known primitives don't get recorded.
        self._raw_recording["primitive"] = step if not is_known else ""
        self._current_plan["current_idx"] += 1

        if not is_known:
            return self._execute_skill_gap(
                env, client, step, obs, output_dir,
                primitives_tried, all_frames, all_datapoints,
            )
        return super().execute_step(
            env, client, step, obs, output_dir,
            primitives_tried, all_frames, all_datapoints,
        )

    def evaluate_progress(self, initial_img, current_img, primitive, goal, steps):
        """Flywheel uses sim ground-truth (peg direction) by default; the v2
        VLM variant is opt-in via ``_USE_VLM_PROGRESS_CHECK``. Both variants
        receive the executor's per-run skill-gap state to read/write."""
        if _USE_VLM_PROGRESS_CHECK:
            return _evaluate_progress_v2(
                initial_img, current_img, primitive, goal, steps,
                skill_gap_state=self._skill_gap_state,
            )
        return _evaluate_progress_sim(
            initial_img, current_img, primitive, goal, steps,
            skill_gap_state=self._skill_gap_state,
        )

    def generate_action(self, initial_img, current_img, wrist_img,
                        primitive, step, goal="", feedback=""):
        """Flywheel action generator — delegates to the peg-direction-aware
        wrapper, passing the executor's prompt variant and skill-gap state."""
        return _generate_action_wrapper(
            initial_img, current_img, wrist_img, primitive, step,
            goal=goal, feedback=feedback,
            system=self.GENERATE_PROMPT,
            skill_gap_state=self._skill_gap_state,
        )

    # ────────────────── Plan-level orchestration ──────────────────

    def execute_plan(self, env, client, plan, obs, output_dir,
                     primitives_tried, all_frames, all_datapoints, history):
        """Reset per-plan skill-gap state, then delegate to the inherited
        ``LiberoSimExecutor.execute_plan`` (which iterates and dispatches via
        ``self.execute_step``)."""
        self._current_plan["current_idx"] = 0
        self._skill_gap_state["rotation_complete"] = False
        self._skill_gap_state["goal_complete"] = False
        self._skill_gap_state["preanalysis_for"] = ""
        return super().execute_plan(
            env, client, plan, obs, output_dir,
            primitives_tried, all_frames, all_datapoints, history,
        )

    # ────────────────── Helpers ──────────────────

    def _plan_task_with_notes(self, goal, scene_image, history=None):
        """Plan a task using the FLYWHEEL prompt and capture step_notes onto
        ``self._current_plan``. Used by ``run_flywheel_adaptive`` for both the
        initial plan and replans."""
        scene_ctx = (getattr(self.args, "scene_context", "") or "").strip() \
            or DEFAULT_SCENE_CONTEXT
        system = _PLAN_TASK_SYSTEM_FLYWHEEL.format(
            scene_context=scene_ctx,
            primitives="\n".join(f"  - {p}" for p in AVAILABLE_PRIMITIVES),
        )
        history_context = ""
        if history:
            history_context = "\n\nPREVIOUS ACTIONS ALREADY COMPLETED:\n"
            for h in history:
                history_context += f"- {h['primitive']}: {h['result']}\n"
            history_context += "\nThe image shows the CURRENT state. Plan only the REMAINING steps."
        prompt = f"GOAL: {goal}{history_context}"
        logging.info(f"  [plan_task] SYSTEM:\n\033[36m{system}\033[0m")
        logging.info(f"  [plan_task] PROMPT:\n\033[36m{prompt}\033[0m")
        try:
            raw = vlm_with_images(prompt, [scene_image], max_tokens=4096, system=system)
            logging.info(f"  [plan_task] RAW VLM response:\n\033[32m{raw}\033[0m")
            data = parse_vlm_json(raw)
            notes = data.get("step_notes", [])
            seq = data.get("primitive_sequence", [])
            while len(notes) < len(seq):
                notes.append("")
            self._current_plan["step_notes"] = notes
            logging.info(f"Step notes: {notes}")
            return TaskPlan(
                goal=goal,
                primitive_sequence=seq,
                skill_gaps=data.get("skill_gaps", []),
                reasoning=data.get("reasoning", ""),
                confidence=data.get("confidence", 0.5),
                requires_new_primitive=data.get("requires_new_primitive", False),
            )
        except (ValueError, KeyError) as e:
            logging.error(f"Failed to parse plan: {e}")
            self._current_plan["step_notes"] = []
            return TaskPlan(goal, [], ["parse_error"], str(e), 0.0, True)

    def _execute_skill_gap(self, env, client, step, obs, output_dir,
                           primitives_tried, all_frames, all_datapoints):
        """Execute a skill gap via VLM pre-analysis + closed-loop P-control.

        Rotation gaps drive a quaternion target while holding position;
        translation gaps drive a position target while holding orientation.
        Falls back to the inherited (VLA) dispatch if the VLM hint can't be
        parsed into a motion command.
        """
        self._skill_gap_state["goal_complete"] = False
        self._skill_gap_state["rotation_complete"] = False

        # Trim previous-attempt frames of ANY prior skill-gap from the
        # recording buffer. Failed replans leave garbage in there, AND
        # the planner often gives each replan attempt a different name
        # (e.g., "rotate the block to be peg-up" vs "rotate block to
        # upright position"). Matching by exact name only catches some
        # replan duplicates. Instead, we track every skill-gap name we
        # see during this trial on the executor and trim entries
        # matching any of them — so only the LAST skill-gap attempt
        # remains in the buffer when the HDF5 gets written at end of
        # trial.
        if not hasattr(self, "_skill_gap_names_seen"):
            self._skill_gap_names_seen = set()
        self._skill_gap_names_seen.add(step)
        if self.args.record:
            prev_len = len(self._raw_recording["buffer"])
            self._raw_recording["buffer"] = [
                e for e in self._raw_recording["buffer"]
                if e["primitive"] not in self._skill_gap_names_seen
            ]
            if len(self._raw_recording["buffer"]) < prev_len:
                logging.info(
                    f"  [Skill gap] Trimmed {prev_len - len(self._raw_recording['buffer'])} "
                    f"frames of previous skill-gap attempts from buffer "
                    f"(matched {len(self._skill_gap_names_seen)} known skill-gap names)"
                )

        # VLM pre-analysis: determine what motion is needed. Pass the
        # executor's skill-gap state so the helper can stash the rotation
        # magnitude for the P-controller below.
        preanalysis = _preanalyze_skill_gap(
            env, obs, step, self.args.goal, save_dir=output_dir,
            skill_gap_state=self._skill_gap_state,
            feedback_block=self._preanalyze_feedback,
        )
        hint = preanalysis["hint"]
        self._skill_gap_state["preanalysis_hint"] = hint

        # Parse motion from hint (e.g., "primary: drx=-0.150" or "primary: dy=+0.100").
        match = re.search(r"(d[r]?[xyz])=([+-]?\d*\.?\d+)", hint) if hint else None
        if not match:
            # Pre-analysis failed (crashed in VLM call, returned empty
            # hint, or unparseable text). The old behavior was to fall
            # through to per-step VLA action generation, which burns
            # 100+ VLM calls picking inconsistent axes — worse than
            # just failing the trial. Abort instead and let main()
            # move on to the next seed.
            logging.warning(
                f"  [Skill gap] Pre-analysis produced no usable hint "
                f"({hint!r}); aborting trial."
            )
            step_result = {"success": False, "result_str": "preanalysis-failed",
                           "is_new": True, "num_frames": 0}
            return obs, True, step_result  # episode_done=True

        axis_name = match.group(1)
        direction = float(match.group(2))
        initial_pos = obs["robot0_eef_pos"][:3].copy()
        initial_quat = _get_ee_quat_wxyz(obs)

        if axis_name.startswith("dr"):
            axis_idx = {"drx": 3, "dry": 4, "drz": 5}[axis_name]
            target_deg = self._skill_gap_state.get("target_rotation_deg", 90)
            rotation_offset = np.zeros(7)
            rotation_offset[axis_idx] = np.radians(target_deg) * (1 if direction > 0 else -1)
            # frame="local" → axis is interpreted in the EE's current
            # frame (e.g. "rotate around X" means around the gripper's
            # current x-axis, not a fixed world axis). Matches the
            # intuitive flip-the-block semantics regardless of how the
            # gripper is oriented at planning time.
            target_quat = _build_target_quat(
                initial_quat, rotation_offset, frame="local",
            )
            target_pos = initial_pos  # hold position during rotation
            logging.info(f"  [Skill gap] Rotation: {axis_name}={np.degrees(rotation_offset[axis_idx]):.0f}° (EE local frame)")
        else:
            axis_idx = {"dx": 0, "dy": 1, "dz": 2}[axis_name]
            target_pos = initial_pos.copy()
            target_pos[axis_idx] += direction
            target_quat = initial_quat  # hold orientation during translation
            logging.info(f"  [Skill gap] Translation: {axis_name}={direction * 1000:.0f}mm")

        logging.info(f"  [Skill gap] From: [{initial_pos[0]:.4f}, {initial_pos[1]:.4f}, {initial_pos[2]:.4f}]")

        all_skill_frames: list = []
        goal_met = False
        episode_done = False

        # Determine gripper command from plan context: if the last gripper
        # action before this skill gap was 'close', stay closed (e.g. holding
        # a block during rotation).
        plan_seq = self._current_plan.get("sequence", [])
        plan_idx = self._current_plan.get("current_idx", 0)
        prior_steps = [s.lower() for s in plan_seq[:plan_idx]]
        grip_state = -1.0  # default open
        for s in prior_steps:
            if s == "close gripper":
                grip_state = 1.0
            elif s == "open gripper":
                grip_state = -1.0
        logging.info(
            f"  [Skill gap] Gripper from plan context: {grip_state:.0f} "
            f"({'closed' if grip_state > 0 else 'open'})"
        )

        rot_tol = np.radians(3.0)
        # Bumped from 0.05 → 0.10. At 0.05 rad/step the controller often
        # stalled on large EE-local rotations (the OSC_POSE controller's
        # per-tick error correction caps out and progress effectively
        # stops). 0.10 doubles the per-step rotation command without
        # causing visible instability. Still well below the 0.15 used
        # elsewhere in control.py (_MAX_ROT_CMD).
        rot_cmd_limit = 0.10

        # Stall detection: track the angular gap and abort if it doesn't
        # decrease by a meaningful amount for N consecutive steps. Without
        # this the loop runs the full _MAX_SKILL_GAP_STEPS even when the
        # controller is clearly not making progress.
        _STALL_FRAMES = 100  # ~5s at 20Hz of no progress
        _STALL_GAP_EPSILON_RAD = np.radians(0.5)
        last_gap_rad = float("inf")
        no_progress_count = 0

        for inner_step in range(self._MAX_SKILL_GAP_STEPS):
            current_pos = obs["robot0_eef_pos"][:3]
            current_quat = _get_ee_quat_wxyz(obs)

            pos_error = target_pos - current_pos
            rot_error_vec = _quat_error_to_world_rotvec(target_quat, current_quat)
            rot_error_mag = np.linalg.norm(rot_error_vec)

            action = np.zeros(7)
            if axis_name.startswith("dr") and rot_error_mag > rot_tol:
                action[:3] = np.clip(pos_error * _K_P_TRANS, -_MAX_POS_HOLD_CMD, _MAX_POS_HOLD_CMD)
            else:
                action[:3] = np.clip(pos_error * _K_P_TRANS, -_MAX_TRANS_CMD, _MAX_TRANS_CMD)
            for i in range(3):
                action[i + 3] = np.clip(rot_error_vec[i] * _K_P_ROT, -rot_cmd_limit, rot_cmd_limit)
            action[6] = grip_state

            img, _ = get_obs_images(obs)
            all_skill_frames.append(img)

            # Stall detection — applies only to rotation skill gaps.
            # IMPORTANT: only counts as a stall if we haven't already
            # converged. If the gap is already below ``rot_tol`` (e.g.
            # 0.1° when tol is 3°), the rotation succeeded — the gap
            # can't decrease further by definition, so we'd wrongly
            # flag a convergent rotation as stalled.
            if axis_name.startswith("dr"):
                if rot_error_mag < rot_tol:
                    # Rotation converged to its commanded target. We
                    # break out of the inner loop, but we do NOT set
                    # ``goal_met = True`` — "rotation reached target"
                    # is not the same as "task goal achieved". The
                    # trial-level oracle (BDDL + peg-direction) is the
                    # authority on whether the actual task succeeded.
                    logging.info(
                        f"  [Skill gap] Rotation converged at step {inner_step}: "
                        f"gap={np.degrees(rot_error_mag):.2f}° < tol={np.degrees(rot_tol):.1f}° "
                        f"(commanded rotation complete; task goal is judged separately)"
                    )
                    break
                if last_gap_rad - rot_error_mag > _STALL_GAP_EPSILON_RAD:
                    no_progress_count = 0
                    last_gap_rad = rot_error_mag
                else:
                    no_progress_count += 1
                if no_progress_count >= _STALL_FRAMES:
                    logging.warning(
                        f"  [Skill gap] STALLED at step {inner_step}: "
                        f"gap={np.degrees(rot_error_mag):.1f}° not decreasing "
                        f"(no progress for {_STALL_FRAMES} steps). Aborting rotation."
                    )
                    break

            if inner_step % 50 == 0:
                motion_mm = (current_pos - initial_pos) * 1000
                gap_deg = np.degrees(rot_error_mag)
                logging.info(
                    f"  [Skill gap] Step {inner_step}: "
                    f"motion=[{motion_mm[0]:.1f}, {motion_mm[1]:.1f}, {motion_mm[2]:.1f}]mm  "
                    f"rot_gap={gap_deg:.1f}°"
                )

            try:
                obs, _, done, _ = env.step(action.tolist())
            except ValueError as e:
                if "terminated episode" in str(e):
                    episode_done = True
                    break
                raise
            if done:
                episode_done = True
                goal_met = True
                break

            # Goal check — env.check_success() is the task-defined goal
            # (acceptable: it's the task definition, not auxiliary
            # privileged sensor state). The peg-direction fallback that
            # used to live here was removed because it reads block
            # joint qpos directly from the simulator, which is
            # privileged information not available in a real-world
            # deployment. Polling every 10 steps to limit overhead.
            if inner_step > 0 and inner_step % 10 == 0:
                try:
                    goal_met = env.check_success()
                except Exception:
                    pass
                if goal_met:
                    break

            # If near translation target but goal not met, extend in same direction.
            pos_dist = np.linalg.norm(pos_error)
            if not axis_name.startswith("dr") and pos_dist < 0.005:
                target_pos[axis_idx] += direction
                logging.info(f"  [Skill gap] Extending target {axis_name} by {direction * 1000:.0f}mm")

        all_frames.extend(all_skill_frames)

        if goal_met:
            logging.info("  [Skill gap] Goal achieved — success!")
            self._skill_gap_state["goal_complete"] = True
            self._skill_gap_state["rotation_complete"] = True
            step_result = {"success": True, "result_str": "P-control", "is_new": True,
                           "num_frames": len(all_skill_frames)}
        else:
            logging.info("  [Skill gap] Goal not achieved")
            step_result = {"success": False, "result_str": "P-control", "is_new": True,
                           "num_frames": len(all_skill_frames)}

        return obs, episode_done, step_result

    # ────────────────── Public entry ──────────────────

    def run_flywheel_adaptive(self, env, client, output_dir):
        """Top-level entry: tilt block (lego), record initial state, plan,
        execute the plan, and replan up to ``_MAX_REPLANS`` times until the
        goal is achieved or budget exhausted."""
        args = self.args
        logging.info(
            f"{'='*60}\nFLYWHEEL ADAPTIVE MODE\n"
            f"Goal: {args.goal}\nMax primitives: {args.max_primitives}\n{'='*60}"
        )

        all_datapoints: list = []
        all_frames: list = []
        history: list = []
        primitives_tried = 0
        goal_achieved = False
        error_msg = None
        original_env_step = env.step

        try:
            obs = settle_physics(env)

            # Respawn scene if red and blue lego blocks are too close.
            # When they spawn near each other, the grasp policy can
            # confuse them (approach blue when targeted at red, or bump
            # blue during the rotate primitive). Re-seed with a
            # different seed until the blocks are at least 10 cm apart,
            # up to a few retries.
            _MAX_RESPAWN_TRIES = 5
            # 80mm center-to-center — accounts for block width (~25mm)
            # plus enough clearance that the grasp won't confuse them
            # or rotate one into the other. 50mm was empirically still
            # too close (blocks visually overlapped in the wrist crop).
            _MIN_BLOCK_DIST_M = 0.08
            if any(t in args.goal.lower() for t in ("lego", "flip", "block")):
                respawn_seed = int(args.seed)
                tries = 0
                while blocks_too_close(env, _MIN_BLOCK_DIST_M) and tries < _MAX_RESPAWN_TRIES:
                    tries += 1
                    respawn_seed += 1000  # large jump so we don't reuse a neighboring random state
                    logging.info(
                        f"[Respawn] Blocks too close (<{_MIN_BLOCK_DIST_M*1000:.0f}mm) — "
                        f"reseeding to {respawn_seed} (attempt {tries}/{_MAX_RESPAWN_TRIES})."
                    )
                    np.random.seed(respawn_seed)
                    env.seed(respawn_seed)
                    obs = env.reset()
                    obs = settle_physics(env, obs)
                if tries == _MAX_RESPAWN_TRIES and blocks_too_close(env, _MIN_BLOCK_DIST_M):
                    logging.warning(
                        f"[Respawn] Gave up after {_MAX_RESPAWN_TRIES} retries; "
                        f"blocks still close. Proceeding anyway."
                    )

            # Tilt block for lego flip tasks only.
            if any(t in args.goal.lower() for t in ("lego", "flip", "block")):
                obs = tilt_red_block_to_side(env, obs)
                logging.info("Block tilted onto its side.")

            robot = _find_robot(env)
            initial_jpos = np.array(env.sim.data.qpos[robot._ref_joint_pos_indexes]).copy()
            initial_gripper_jpos = np.array(env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes]).copy()

            initial_img, _ = get_obs_images(obs)
            Image.fromarray(resize_for_vlm(initial_img)).save(output_dir / "initial_scene.png")

            if args.record:
                self._raw_recording["enabled"] = True
                self._raw_recording["last_obs"] = obs
                self._raw_recording["buffer"] = []
                env.step = _make_recording_step(env.step, self._raw_recording)

            logging.info(f"\n{'='*60}\n[PLANNING] Breaking down goal...\n{'='*60}")
            plan = self._plan_task_with_notes(args.goal, resize_for_vlm(initial_img))
            logging.info(f"Plan: {plan.primitive_sequence}")
            logging.info(f"Skill gaps: {plan.skill_gaps}")
            logging.info(f"Reasoning: {plan.reasoning}")
            logging.info(f"Requires new primitive: {plan.requires_new_primitive}")

            if not plan.primitive_sequence:
                logging.warning("Empty plan returned!")
                error_msg = "Empty plan"
            else:
                logging.info(f"\n{'='*60}\n[EXECUTION] Running plan...\n{'='*60}")
                self._current_plan["sequence"] = list(plan.primitive_sequence)
                self._current_plan["current_idx"] = 0

                replan_count = 0
                episode_done = False

                obs, primitives_tried, episode_done = self.execute_plan(
                    env, client, plan, obs, output_dir,
                    primitives_tried, all_frames, all_datapoints, history,
                )
                if self._skill_gap_state["goal_complete"]:
                    goal_achieved = True

                while not episode_done and not goal_achieved and not stop_requested():
                    logging.info("\nResetting gripper to initial pose...")
                    if args.record:
                        self._raw_recording["enabled"] = False
                    obs = reset_gripper_pose(env, obs, initial_jpos, initial_gripper_jpos)
                    # Wait long enough for the released block to fall
                    # (~0.3s from lift height) AND finish bouncing /
                    # settling on the table (up to ~1-2s for a lego
                    # with restitution). 100 steps × ~50ms control
                    # tick = ~5s, a safe upper bound that ensures the
                    # goal-check image shows a static final-state
                    # scene rather than a mid-air block. Without this
                    # settle, "open gripper" (only ~9 policy steps)
                    # barely lets the block start falling before the
                    # check fires, making both the BDDL pose check
                    # and any visual judgment unreliable.
                    obs = settle_physics(env, obs=obs, steps=100)
                    if args.record:
                        self._raw_recording["enabled"] = True
                        self._raw_recording["last_obs"] = obs

                    current_img, _ = get_obs_images(obs)
                    all_frames.append(current_img)
                    goal_achieved, reasoning = _check_goal_achieved_sim(
                        env, args.goal, current_img,
                        save_dir=output_dir, check_num=replan_count,
                        before_image=initial_img,
                    )
                    logging.info(f"Goal check: {'ACHIEVED' if goal_achieved else 'NOT achieved'} - {reasoning}")
                    if goal_achieved:
                        break

                    if replan_count >= self._MAX_REPLANS or primitives_tried >= args.max_primitives:
                        logging.info(
                            f"Stopping: replans={replan_count}/{self._MAX_REPLANS}, "
                            f"primitives={primitives_tried}/{args.max_primitives}"
                        )
                        break

                    replan_count += 1
                    logging.info(
                        f"\n{'='*60}\n[RE-PLANNING] Goal not achieved.\n"
                        f"Reason: {reasoning}\n"
                        f"Creating new plan... (attempt {replan_count}/{self._MAX_REPLANS})\n{'='*60}"
                    )

                    plan = self._plan_task_with_notes(args.goal, resize_for_vlm(current_img), history)
                    logging.info(f"New plan: {plan.primitive_sequence}")

                    if not plan.primitive_sequence:
                        logging.warning("Empty re-plan returned! Stopping.")
                        break

                    self._current_plan["sequence"] = list(plan.primitive_sequence)
                    self._current_plan["current_idx"] = 0
                    obs, primitives_tried, episode_done = self.execute_plan(
                        env, client, plan, obs, output_dir,
                        primitives_tried, all_frames, all_datapoints, history,
                    )
                    if self._skill_gap_state["goal_complete"]:
                        goal_achieved = True
                        break
                    if episode_done:
                        break

        except Exception as e:
            error_msg = str(e)
            logging.error(f"Error during flywheel adaptive run: {e}")
            traceback.print_exc()

        finally:
            if args.record:
                env.step = original_env_step
                if goal_achieved and self._raw_recording["buffer"]:
                    try:
                        _save_raw_hdf5(output_dir, self._raw_recording["buffer"])
                        logging.info("Saved raw HDF5 (goal achieved)")
                    except Exception as e:
                        logging.error(f"Failed to save HDF5: {e}")
                else:
                    logging.info(
                        f"Skipping HDF5 save (goal_achieved={goal_achieved}, "
                        f"buffer_len={len(self._raw_recording['buffer'])})"
                    )

            return _save_adaptive_results(
                output_dir, args.goal, goal_achieved, primitives_tried,
                history, all_frames, all_datapoints, error_msg,
            )


# =============================================================================
# Feedback injection
# =============================================================================

def _format_feedback_block(fb):
    """Format curation feedback into a prompt block."""
    parts = ["LESSONS FROM PREVIOUS ITERATIONS:"]
    if fb.get("tips_for_action_generation"):
        parts.append("Action generation: " + fb["tips_for_action_generation"])
    if fb.get("tips_for_grasp_strategy"):
        parts.append("Grasp strategy: " + fb["tips_for_grasp_strategy"])
    if fb.get("common_failure_modes"):
        parts.append("Avoid these failure modes: " + "; ".join(fb["common_failure_modes"]))
    return "\n".join(parts)


# =============================================================================
# Args & main
# =============================================================================

@dataclasses.dataclass
class FlywheelArgs:
    """Args for flywheel adaptive mode."""
    host: str = "localhost"
    port: int = 8000
    seed: int = 42
    task: str = "lego"  # "lego" or "drawer"
    output_dir: str = "data/libero/vlm_feedback_flywheel"
    goal: str = "flip the red lego block peg up"
    collect_data: bool = True
    max_primitives: int = 15
    vlm: str = "gpt"
    num_runs: int = 1
    target_successes: int = 0
    display: bool = False
    record: bool = False
    replan_steps: int = 30
    curate: bool = False
    curate_keep_ratio: float = 0.7
    curate_only: str = ""
    feedback_dir: str = ""
    use_vlm_completion_check: bool = False  # Enable VLM completion check for "move to" primitives (helps OOD cases like drawer)
    scene_context: str = ""
    """Scene description passed to the planner (DECIDE_NEXT_PRIMITIVE_SYSTEM).
    A richer description ("Tabletop with a red lego block and a green target
    plate; gripper grasps from above; goal is to flip the block peg-up")
    produces better primitive sequences than the default bare-bones
    "Robot arm with parallel jaw gripper." When empty, falls back to
    DEFAULT_SCENE_CONTEXT in env.py. Mirrors the --scene-context flag on
    the xArm flywheel."""
    experiment_name: str = ""
    """Scope tag for the per-trial outcome log. When set, the
    ``flywheel_trials.jsonl`` is written to
    ``data/libero/vlm_feedback_flywheel/<experiment_name>/`` and each
    row carries an ``experiment_name`` field, so multiple batches with
    different configs (e.g., with-recipes vs without-recipes ablations)
    don't pollute each other. Empty (default) = use the date-keyed
    path. Mirrors the same flag on the xArm flywheel."""


def _run_single(env, client, args, base_output, pst, run_seed, executor: "LiberoFlywheelExecutor"):
    """Execute a single flip run with its own output dir and log file.

    ``executor`` is constructed once in ``main`` and reused across runs so
    that feedback-injection mutations persist (the curated tips from one
    iteration carry over into the next)."""
    np.random.seed(run_seed)
    env.seed(run_seed)

    timestamp = datetime.now(pst).strftime("%Y-%m-%d_%H%M%S")
    output_dir = base_output / f"run_{timestamp}_seed{run_seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(output_dir / "log.txt")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(file_handler)

    # Reset per-run recording state on the executor for this seed.
    executor._raw_recording["enabled"] = False
    executor._raw_recording["buffer"] = []
    executor._raw_recording["last_obs"] = None
    executor._raw_recording["primitive"] = ""
    # Reset the per-trial set of skill-gap names seen so the buffer
    # trim in _execute_skill_gap doesn't carry across trials.
    executor._skill_gap_names_seen = set()

    try:
        result = executor.run_flywheel_adaptive(env, client, output_dir)
    finally:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()

    goal_achieved = result.get("goal_achieved", False) if result else False
    has_hdf5 = (output_dir / "demo.hdf5").exists()
    primitives_tried = result.get("primitives_tried", 0) if result else 0
    return goal_achieved, has_hdf5, primitives_tried, output_dir


def main(args: FlywheelArgs) -> None:
    import os
    import signal
    import sys

    def _hard_sigint(_sig, _frame):
        sys.stderr.write("\n*** SIGINT received — killing process. ***\n")
        os._exit(130)
    signal.signal(signal.SIGINT, _hard_sigint)

    from libero.libero import get_libero_path
    from openpi_client import websocket_client_policy as _wcp

    if args.curate_only:
        logging.basicConfig(level=logging.INFO, format="%(message)s",
                            handlers=[logging.StreamHandler()])
        if args.vlm != "gpt":
            set_vlm_provider(args.vlm)
        curate_batch(pathlib.Path(args.curate_only), args.curate_keep_ratio)
        return

    # Build the executor once. Behaviors that flywheel mode overrides
    # (execute_step, evaluate_progress, generate_action, GENERATE_PROMPT,
    # display) live on this object — see class above. Reused across runs
    # so curation feedback persists across iterations.
    executor = LiberoFlywheelExecutor(args)

    # Task-specific setup (must happen before base_output is set)
    if args.task == "drawer":
        if args.goal == "flip the red lego block peg up":
            args.goal = "close the top drawer"
        if args.output_dir == "data/libero/vlm_feedback_flywheel":
            args.output_dir = "data/libero/vlm_feedback_drawer"
        # Drawer push starts from OOD state (drawer open) — enable VLM completion check
        args.use_vlm_completion_check = True

    pst = timezone(timedelta(hours=-7))  # PDT (Mar-Nov)
    # When experiment_name is set, scope outputs under that name AND
    # date the batch (so multiple runs of the same experiment on
    # different days don't share a flywheel_trials.jsonl). Layout:
    #     <output_dir>/<experiment_name>/<date>/flywheel_trials.jsonl
    # When experiment_name is empty, fall back to the original
    # date-only layout:
    #     <output_dir>/<date>/flywheel_trials.jsonl
    date_str = datetime.now(pst).strftime("%Y-%m-%d")
    if args.experiment_name:
        base_output = pathlib.Path(args.output_dir) / args.experiment_name / date_str
    else:
        base_output = pathlib.Path(args.output_dir) / date_str
    base_output.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler()],
    )

    # Snapshot the VLM prompts used by this batch next to the trials log
    # so results stay reproducible across prompt edits (matches the
    # real-world flywheel under `real/xarm_flywheel/main.py`).
    try:
        from insight import prompts as _insight_prompts
        from . import prompts as _sim_prompts
        shared_names = (
            "PREANALYZE_TRANSLATION_SYSTEM",
            "TASK_COMPLETION_SYSTEM",
            "PRIMITIVE_DONE_SYSTEM",
        )
        sim_names = (
            "_PLAN_TASK_SYSTEM",
            "_PREANALYZE_SYSTEM",
            "_EVALUATE_PROGRESS_SYSTEM_V2",
            "_POSITION_CHECK_SYSTEM",
            "_GENERATE_NEW_PRIMITIVE_SYSTEM_FLYWHEEL",
        )
        with open(base_output / "prompts.txt", "w") as f:
            for name in shared_names:
                val = getattr(_insight_prompts, name, None)
                if val is not None:
                    f.write(f"=== insight.{name} ===\n{val}\n\n")
            for name in sim_names:
                val = getattr(_sim_prompts, name, None)
                if val is not None:
                    f.write(f"=== vlm_flywheel.prompts.{name} ===\n{val}\n\n")
    except Exception as e:
        logging.warning("Could not snapshot prompts: %s", e)

    if args.vlm != "gpt":
        set_vlm_provider(args.vlm)

    if args.feedback_dir:
        import json as _json
        fb_path = pathlib.Path(args.feedback_dir) / "feedback.json"
        if fb_path.exists():
            with open(fb_path) as _f:
                fb = _json.load(_f)
            executor._preanalyze_feedback = _format_feedback_block(fb)
            logging.info("Loaded feedback from %s", fb_path)

            action_fb = fb.get("tips_for_action_generation", "")
            if action_fb:
                # Shadow the class attribute on this instance — does not mutate prompts.py.
                executor.GENERATE_PROMPT = executor.GENERATE_PROMPT.replace(
                    "Respond with ONLY valid JSON:",
                    "\nLESSONS FROM PREVIOUS ITERATIONS:\n" + action_fb + "\n\nRespond with ONLY valid JSON:",
                    1,
                )
        else:
            logging.warning("Feedback file not found: %s", fb_path)

    if args.task == "drawer":
        bddl_file = (
            pathlib.Path(get_libero_path("bddl_files"))
            / "drawer_primitives" / "close_top_drawer.bddl"
        )
    else:
        bddl_file = (
            pathlib.Path(get_libero_path("bddl_files"))
            / "lego_primitives" / "wide_range" / "pick_blue_place_target_wide.bddl"
        )
    if not bddl_file.exists():
        logging.error(f"BDDL file not found: {bddl_file}")
        return

    _start_keyboard_listener()
    logging.info("Press 's' at any time to save and stop early.")
    logging.info("Creating LIBERO environment...")
    env = create_env(bddl_file, args.seed, resolution=512)

    logging.info(f"Connecting to policy server at {args.host}:{args.port}...")
    client = _wcp.WebsocketClientPolicy(args.host, args.port)

    # Persistent per-trial outcome log — appended after each trial so a
    # Ctrl+C'd batch still has analyzable data. Lives next to all the
    # per-run output dirs under ``base_output``. Each line is a JSON
    # record with success/fail and metadata. Symmetric in spirit
    # to the real-world ``flywheel_trials.jsonl``, but the sim-relevant
    # fields differ (e.g. ``primitives_tried`` is sim-only;
    # ``robot_time_s`` is real-only).
    trials_log = base_output / "flywheel_trials.jsonl"
    trials_log.parent.mkdir(parents=True, exist_ok=True)

    successes = 0
    run_idx = 0
    for run_idx in range(args.num_runs):
        run_seed = args.seed + run_idx

        _stop_event.clear()

        if args.num_runs > 1:
            logging.info(
                f"\n{'#'*60}\n"
                f"# RUN {run_idx + 1}/{args.num_runs}  seed={run_seed}  "
                f"successes={successes}"
                f"{'/' + str(args.target_successes) if args.target_successes else ''}\n"
                f"{'#'*60}"
            )

        goal_achieved, has_hdf5, primitives_tried, run_output_dir = _run_single(
            env, client, args, base_output, pst, run_seed, executor
        )

        # Task success = goal achieved. HDF5 saving is a separate
        # metric (data-acquisition efficiency, not task success). A
        # trial where the task is solved but no demo got saved still
        # counts as a successful trial for measuring policy
        # capability; we just don't add that trial to the training
        # set. The JSONL row carries both fields so paper analysis
        # can compute both rates from the same data.
        success = bool(goal_achieved)
        if success:
            successes += 1
            if args.num_runs > 1:
                hdf5_note = "" if has_hdf5 else " (no demo saved)"
                logging.info(f"  => SUCCESS ({successes} total){hdf5_note}")
        elif args.num_runs > 1:
            logging.info(f"  => FAILED (goal={goal_achieved}, hdf5={has_hdf5})")

        # Append per-trial outcome to the persistent log. Includes the
        # run-output dir so consumers can correlate with on-disk artifacts.
        outcome = {
            "trial_idx": run_idx,
            "seed": run_seed,
            "task": args.task,
            "experiment_name": args.experiment_name,
            "success": success,
            "goal_achieved": bool(goal_achieved),
            "has_hdf5": bool(has_hdf5),
            "primitives_tried": int(primitives_tried),
            "timestamp": datetime.now(pst).isoformat(timespec="seconds"),
            "run_output_dir": str(run_output_dir.relative_to(base_output)),
            "checkpoint_path": getattr(args, "checkpoint_path", ""),
        }
        with open(trials_log, "a") as f:
            f.write(json.dumps(outcome) + "\n")

        if args.target_successes and successes >= args.target_successes:
            logging.info(f"\nReached target of {args.target_successes} successes. Stopping.")
            break

        if stop_requested():
            if args.num_runs > 1:
                logging.info("\nStop requested by user. Stopping batch.")
            break

    if args.num_runs > 1:
        # Read back the JSONL we wrote per-trial and compute aggregate
        # stats. Keeping this in addition to the per-trial log means
        # paper analysis can either read the rich JSONL or just the
        # summary, depending on what's convenient.
        n_trials = run_idx + 1
        n_success = successes
        n_demo_saved = 0
        try:
            with open(trials_log) as f:
                rows = [json.loads(line) for line in f if line.strip()]
            n_demo_saved = sum(1 for r in rows if r.get("has_hdf5"))
        except Exception:
            pass
        success_rate = n_success / n_trials if n_trials else 0.0
        demo_rate = n_demo_saved / n_trials if n_trials else 0.0
        summary = {
            "experiment_name": args.experiment_name,
            "task": args.task,
            "date": date_str,
            "n_trials": n_trials,
            "n_success": n_success,
            "n_demo_saved": n_demo_saved,
            "success_rate": round(success_rate, 4),
            "demo_rate": round(demo_rate, 4),
        }
        summary_path = base_output / "batch_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logging.info(
            f"\n{'='*60}\n"
            f"BATCH COMPLETE\n"
            f"  Experiment:   {args.experiment_name or '(unnamed)'}\n"
            f"  Date:         {date_str}\n"
            f"  Trials:       {n_trials}\n"
            f"  Goal success: {n_success}/{n_trials} ({success_rate*100:.1f}%)\n"
            f"  Demos saved:  {n_demo_saved}/{n_trials} ({demo_rate*100:.1f}%)\n"
            f"  Output:       {base_output}\n"
            f"  Trials log:   {trials_log}\n"
            f"  Summary:      {summary_path}\n"
            f"{'='*60}"
        )

    if args.curate and successes >= 2:
        curate_batch(base_output, args.curate_keep_ratio)

    env.close()
