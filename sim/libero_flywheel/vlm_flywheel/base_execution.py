"""Base _execute_step/_execute_plan, run_adaptive, legacy modes, base main()."""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import traceback
from datetime import datetime, timedelta, timezone

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tyro

from insight.executor import BaseExecutor as _InsightBaseExecutor
from .env import (
    AVAILABLE_PRIMITIVES,
    DEFAULT_PRIMITIVE_STEPS,
    DEFAULT_SEQUENCE,
    Args,
    FlywheelDatapoint,
    PRIMITIVE_P90_DURATIONS,
    VLMFeedback,
    _find_robot,
    _start_keyboard_listener,
    create_env,
    get_gripper_state,
    get_obs_images,
    resize_for_policy,
    resize_for_vlm,
    reset_gripper_pose,
    settle_physics,
    stop_requested,
)
from .vlm import set_vlm_provider
import math


def _get_state(obs):
    """Get state vector matching test_primitives.py: EEF pose + gripper qpos."""
    q = obs["robot0_eef_quat"].copy()
    if q[3] > 1.0: q[3] = 1.0
    elif q[3] < -1.0: q[3] = -1.0
    den = np.sqrt(1.0 - q[3] * q[3])
    axis_angle = (q[:3] * 2.0 * math.acos(q[3])) / den if den > 1e-8 else np.zeros(3)
    return np.concatenate((obs["robot0_eef_pos"], axis_angle, obs["robot0_gripper_qpos"]))
from .reasoning import (
    analyze_execution,
    apply_correction_to_primitive,
    check_goal_achieved,
    get_action_correction,
    plan_task,
)


# =============================================================================
# New primitive action generation (real implementations)
# =============================================================================

# generate_new_primitive_action and evaluate_new_primitive_progress moved to
# insight.reasoning (env-agnostic VLM helpers, shared with the xArm
# pipeline). Re-imported here for backward compat with sim callers that
# import them via ``from vlm_flywheel import ...``.
from insight.reasoning import (  # noqa: E402  F401
    evaluate_new_primitive_progress,
    generate_new_primitive_action,
)


# =============================================================================
# VLM primitive completion check
# =============================================================================


def _vlm_check_primitive_done(primitive: str, img: np.ndarray, wrist: np.ndarray, save_dir=None, step_num=0, num_votes=3) -> bool:
    """Ask VLM whether the primitive has been completed. Uses majority voting to reduce false positives."""
    from insight.reasoning import check_primitive_done_verbose
    img_vlm = resize_for_vlm(img)
    wrist_vlm = resize_for_vlm(wrist)
    if save_dir is not None:
        from PIL import Image as PILImage
        PILImage.fromarray(img_vlm).save(save_dir / f"vlm_check_step{step_num}_scene.png")
        PILImage.fromarray(wrist_vlm).save(save_dir / f"vlm_check_step{step_num}_wrist.png")
    return check_primitive_done_verbose(primitive, img_vlm, wrist_vlm, num_votes=num_votes, step_num=step_num)["done"]


# =============================================================================
# Primitive execution
# =============================================================================

def run_primitive(env, client, primitive, max_steps, obs=None,
                  save_frames=True, replan_steps=30,
                  vlm_done_check_interval=0, min_vlm_check_step=30):
    """Execute primitive and return before/after images."""
    import collections
    if obs is None:
        obs = settle_physics(env)
    before_img, _ = get_obs_images(obs)
    frames = [before_img] if save_frames else []
    action_queue = collections.deque()
    for step in range(max_steps):
        img, wrist = get_obs_images(obs)
        if save_frames:
            frames.append(img)
        if not action_queue:
            action_chunk = client.infer({
                "observation/image": resize_for_policy(img),
                "observation/wrist_image": resize_for_policy(wrist),
                "observation/state": _get_state(obs),
                "prompt": primitive,
            })["actions"]
            for a in action_chunk[:replan_steps]:
                action_queue.append(a)
        action = action_queue.popleft()
        if step == 0:
            logging.info(f"  First action: [{action[0]:+.3f}, {action[1]:+.3f}, {action[2]:+.3f}] grip={action[6]:+.3f}")
        # Check progress prediction (8th dim) — stop primitive when done
        if len(action) > 7 and action[7] > 0.95 and step >= 30:
            logging.info(f"  Progress={action[7]:.2f} > 0.95 at step {step+1} — primitive done")
            action = action[:7]
            break
        action = action[:7]  # Strip progress before sending to env
        # VLM primitive completion check (every N steps, after min_vlm_check_step)
        if (vlm_done_check_interval > 0
            and step >= min_vlm_check_step
            and step % vlm_done_check_interval == 0):
            if _vlm_check_primitive_done(primitive, img, wrist, step_num=step):
                logging.info(f"  VLM says primitive done at step {step+1}")
                break
        try:
            obs, _, done, _ = env.step(action.tolist())
        except ValueError as e:
            if "terminated episode" in str(e):
                logging.info(f"  Episode terminated at step {step}")
                break
            raise
        if done:
            break
    after_img, _ = get_obs_images(obs)
    return before_img, after_img, frames, obs


def run_primitive_with_checkpoints(
    env, client, primitive, max_steps, obs=None,
    save_frames=True, collect_data=False, downsample_hz=2, replan_steps=30,
    vlm_done_check_interval=0, min_vlm_check_step=30,
):
    """Execute primitive with data collection at 10Hz."""
    import collections
    if obs is None:
        obs = settle_physics(env)
    before_img, wrist_before = get_obs_images(obs)
    frames = [before_img] if save_frames else []
    datapoints = []
    pending_obs = None
    accumulated_action = np.zeros(7)
    episode_done = False
    action_queue = collections.deque()

    for step in range(max_steps):
        if stop_requested():
            break
        img, wrist = get_obs_images(obs)
        if save_frames:
            frames.append(img)
        img_r, wrist_r = resize_for_policy(img), resize_for_policy(wrist)
        state = _get_state(obs)
        if not action_queue:
            action_chunk = client.infer({
                "observation/image": img_r,
                "observation/wrist_image": wrist_r,
                "observation/state": state,
                "prompt": primitive,
            })["actions"]
            for a in action_chunk[:replan_steps]:
                action_queue.append(a)
        action = action_queue.popleft()
        if collect_data:
            if step % downsample_hz == 0:
                pending_obs = {"image": img_r.copy(), "wrist": wrist_r.copy(), "state": state.copy()}
                accumulated_action = action[:7].copy()
            else:
                accumulated_action[:6] += action[:6]
                accumulated_action[6] = action[6]
            if (step + 1) % downsample_hz == 0 and pending_obs:
                datapoints.append(FlywheelDatapoint(
                    pending_obs["image"], pending_obs["wrist"], pending_obs["state"],
                    accumulated_action.copy(), "policy", primitive,
                    step_in_primitive=step // downsample_hz,
                ))
                pending_obs = None
        if step == 0:
            logging.info(f"  First action: [{action[0]:+.3f}, {action[1]:+.3f}, {action[2]:+.3f}] grip={action[6]:+.3f}")
        # Strip progress dim (8th) if present before sending to env
        action_env = action[:7]
        try:
            obs, _, done, _ = env.step(action_env.tolist())
        except ValueError as e:
            if "terminated episode" in str(e):
                episode_done = True
                logging.info(f"  Episode terminated at step {step}")
                break
            raise
        if done:
            episode_done = True
            logging.info(f"  Episode terminated at step {step}")
            break
        # Synchronous VLM primitive completion check (robot pauses during check)
        if (vlm_done_check_interval > 0
            and step >= min_vlm_check_step
            and step % vlm_done_check_interval == 0):
            if _vlm_check_primitive_done(primitive, img, wrist, step_num=step):
                logging.info(f"  VLM says primitive done at step {step+1}")
                break
    after_img, wrist_after = get_obs_images(obs)
    return before_img, after_img, wrist_before, wrist_after, frames, obs, datapoints, episode_done


# =============================================================================
# Save results
# =============================================================================

def _save_adaptive_results(output_dir, goal, goal_achieved, primitives_tried, history, all_frames, all_datapoints, error=None):
    """Save adaptive run results (called on success or crash)."""
    logging.info(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    logging.info(f"Goal achieved: {goal_achieved}")
    logging.info(f"Primitives tried: {primitives_tried}")
    if error:
        logging.info(f"Error: {error}")
    logging.info("\nHistory:")
    for h in history:
        logging.info(f"  {h['step']}. [{'OK' if h['success'] else 'FAIL'}] {h['primitive']}")

    if all_frames:
        try:
            frame_w = all_frames[0].shape[1]
            font_scale = frame_w / 256.0
            font_size = max(13, int(13 * font_scale))
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except (OSError, IOError):
                font = ImageFont.load_default()
            line_height = int(18 * font_scale)
            banner_height = line_height * 2 + int(6 * font_scale)

            def _truncate_text(text, max_w):
                if font.getlength(text) <= max_w:
                    return text
                while len(text) > 0 and font.getlength(text + "...") > max_w:
                    text = text[:-1]
                return text + "..."

            pad = int(6 * font_scale)
            text_pad = int(3 * font_scale)

            def _make_banner(line1, line2=""):
                banner = Image.new("RGB", (frame_w, banner_height), (0, 0, 0))
                draw = ImageDraw.Draw(banner)
                draw.text((pad, text_pad), _truncate_text(line1, frame_w - pad * 2), fill=(255, 255, 255), font=font)
                if line2:
                    draw.text((pad, text_pad + line_height), _truncate_text(line2, frame_w - pad * 2), fill=(200, 200, 200), font=font)
                return banner

            annotated_frames = []
            frame_idx = 0
            for h in history:
                n = h.get("num_frames", 0)
                tag = "VLM" if h.get("is_new", False) else "POLICY"
                line1 = f"[{h['step']}/{len(history)}] {h['primitive']}"
                line2 = f"({tag})  frame {{}}/{n}"
                for i in range(n):
                    if frame_idx < len(all_frames):
                        frame = all_frames[frame_idx]
                        banner = _make_banner(line1, line2.format(i + 1))
                        combined = Image.new("RGB", (frame_w, banner_height + frame.shape[0]))
                        combined.paste(banner, (0, 0))
                        combined.paste(Image.fromarray(frame), (0, banner_height))
                        annotated_frames.append(np.array(combined))
                    frame_idx += 1
            while frame_idx < len(all_frames):
                frame = all_frames[frame_idx]
                banner = _make_banner("")
                combined = Image.new("RGB", (frame_w, banner_height + frame.shape[0]))
                combined.paste(banner, (0, 0))
                combined.paste(Image.fromarray(frame), (0, banner_height))
                annotated_frames.append(np.array(combined))
                frame_idx += 1

            video_path = output_dir / "adaptive_full.mov"
            imageio.mimwrite(video_path, annotated_frames, fps=10, codec="libx264")
            logging.info(f"Saved full video ({len(annotated_frames)} frames): {video_path}")
        except Exception as e:
            logging.error(f"Failed to save video: {e}")

    summary = {"goal": goal, "goal_achieved": goal_achieved,
               "primitives_tried": primitives_tried, "history": history}
    if error:
        summary["error"] = str(error)
    with open(output_dir / "adaptive_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    datapoints_export = [{"index": i, "source": d.source, "primitive": d.primitive,
                          "action": d.action.tolist(), "vlm_feedback": d.vlm_feedback}
                         for i, d in enumerate(all_datapoints)]
    with open(output_dir / "datapoints.json", "w") as f:
        json.dump(datapoints_export, f, indent=2)

    logging.info(f"\nSaved to: {output_dir}")
    return summary


# =============================================================================
# Legacy modes
# =============================================================================

def run_single(env, client, args, output_dir):
    max_steps = PRIMITIVE_P90_DURATIONS.get(args.primitive, DEFAULT_PRIMITIVE_STEPS)
    logging.info(f"{'='*60}\nPrimitive: '{args.primitive}' ({max_steps} steps)\n{'='*60}")
    before, after, frames, _ = run_primitive(env, client, args.primitive, max_steps)
    before_vlm, after_vlm = resize_for_vlm(before), resize_for_vlm(after)
    logging.info("\nAnalyzing with VLM...")
    feedback = analyze_execution(before_vlm, after_vlm, args.primitive)
    logging.info(f"\nVLM FEEDBACK: {'SUCCESS' if feedback.success else 'FAIL'} ({feedback.confidence:.0%})")
    logging.info(f"  {feedback.description}")
    if feedback.correction:
        logging.info(f"  Correction: {feedback.correction}")
    Image.fromarray(before_vlm).save(output_dir / "before.png")
    Image.fromarray(after_vlm).save(output_dir / "after.png")
    if frames:
        imageio.mimwrite(output_dir / "execution.mov", frames, fps=10, codec="libx264")
    with open(output_dir / "feedback.json", "w") as f:
        json.dump(dataclasses.asdict(feedback), f, indent=2)
    logging.info(f"\nSaved to: {output_dir}")
    return feedback


def run_sequence(env, client, args, output_dir, primitives):
    logging.info(f"{'='*60}\nPRIMITIVE SEQUENCE\n{'='*60}")
    for i, p in enumerate(primitives):
        logging.info(f"  {i+1}. {p} ({PRIMITIVE_P90_DURATIONS.get(p, DEFAULT_PRIMITIVE_STEPS)} steps)")
    all_feedback, all_frames = [], []
    obs = settle_physics(env)
    failed_at = None
    for i, primitive in enumerate(primitives):
        max_steps = PRIMITIVE_P90_DURATIONS.get(primitive, DEFAULT_PRIMITIVE_STEPS)
        logging.info(f"\n{'='*60}\n[{i+1}/{len(primitives)}] '{primitive}'\n{'='*60}")
        before, after, frames, obs = run_primitive(env, client, primitive, max_steps, obs)
        all_frames.extend(frames)
        before_vlm, after_vlm = resize_for_vlm(before), resize_for_vlm(after)
        Image.fromarray(before_vlm).save(output_dir / f"p{i+1}_before.png")
        Image.fromarray(after_vlm).save(output_dir / f"p{i+1}_after.png")
        logging.info("Analyzing with VLM...")
        feedback = analyze_execution(before_vlm, after_vlm, primitive)
        all_feedback.append(feedback)
        status = "SUCCESS" if feedback.success else "FAIL"
        logging.info(f"\n  Result: {status} ({feedback.confidence:.0%}) - {feedback.description}")
        if not feedback.success:
            failed_at = i + 1
            if args.stop_on_fail:
                logging.info(f"\n  Stopping (failed at primitive {failed_at})")
                break
    if all_frames:
        imageio.mimwrite(output_dir / "full_sequence.mov", all_frames, fps=10, codec="libx264")
    overall = all(fb.success for fb in all_feedback)
    summary = {"primitives": primitives, "completed": len(all_feedback), "overall_success": overall, "failed_at": failed_at}
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"\n{'='*60}\nSUMMARY: {len(all_feedback)}/{len(primitives)} completed, {'SUCCESS' if overall else 'FAIL'}\n{'='*60}")
    return all_feedback


def run_with_retry(env, client, args, output_dir):
    max_steps = PRIMITIVE_P90_DURATIONS.get(args.primitive, DEFAULT_PRIMITIVE_STEPS)
    logging.info(f"{'='*60}\nRETRY LOOP: '{args.primitive}' (max {args.max_retries} attempts)\n{'='*60}")
    all_feedback = []
    current = args.primitive
    for attempt in range(1, args.max_retries + 1):
        logging.info(f"\n{'='*60}\nATTEMPT {attempt}: '{current}'\n{'='*60}")
        before, after, frames, _ = run_primitive(env, client, current, max_steps)
        before_vlm, after_vlm = resize_for_vlm(before), resize_for_vlm(after)
        logging.info("Analyzing with VLM...")
        feedback = analyze_execution(before_vlm, after_vlm, current)
        all_feedback.append(feedback)
        Image.fromarray(before_vlm).save(output_dir / f"attempt{attempt}_before.png")
        Image.fromarray(after_vlm).save(output_dir / f"attempt{attempt}_after.png")
        if frames:
            imageio.mimwrite(output_dir / f"attempt{attempt}.mov", frames, fps=10, codec="libx264")
        logging.info(f"\n{'SUCCESS' if feedback.success else 'FAIL'} ({feedback.confidence:.0%}) - {feedback.description}")
        if feedback.success and feedback.confidence >= 0.7:
            logging.info(f"\nSUCCESS after {attempt} attempt(s)!")
            break
        if feedback.correction and attempt < args.max_retries:
            logging.info(f"\nApplying correction: '{feedback.correction}'")
            current = apply_correction_to_primitive(current, feedback.correction)
            logging.info(f"New primitive: '{current}'")
    with open(output_dir / "summary.json", "w") as f:
        json.dump({"original": args.primitive, "attempts": len(all_feedback),
                   "final_success": all_feedback[-1].success}, f, indent=2)
    return all_feedback


def run_flywheel(env, client, args, output_dir):
    """Run full flywheel: plan -> execute -> correct -> collect data."""
    logging.info(f"{'='*60}\nFLYWHEEL MODE\nGoal: {args.goal}\n{'='*60}")
    all_datapoints = []
    all_frames = []
    log = []
    obs = settle_physics(env)
    initial_img, _ = get_obs_images(obs)
    initial_vlm = resize_for_vlm(initial_img)
    Image.fromarray(initial_vlm).save(output_dir / "initial_scene.png")
    logging.info("\n[PHASE 1] VLM Task Planning\n" + "-"*40)
    plan = plan_task(args.goal, initial_vlm)
    logging.info(f"Confidence: {plan.confidence:.0%}, Requires new primitive: {plan.requires_new_primitive}")
    logging.info(f"Reasoning: {plan.reasoning}")
    logging.info("\nPlanned sequence:")
    for i, p in enumerate(plan.primitive_sequence):
        logging.info(f"  {i+1}. {p} ({PRIMITIVE_P90_DURATIONS.get(p, DEFAULT_PRIMITIVE_STEPS)} steps)")
    if plan.skill_gaps:
        logging.info(f"\nSkill gaps: {plan.skill_gaps}")
    log.append({"phase": "planning", "plan": dataclasses.asdict(plan)})
    if not plan.primitive_sequence:
        logging.warning("VLM returned empty primitive sequence!")
        summary = {
            "goal": args.goal, "overall_success": False,
            "primitives_planned": 0, "primitives_completed": 0,
            "datapoints_total": 0, "skill_gaps": plan.skill_gaps, "log": log,
        }
        with open(output_dir / "flywheel_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        return summary
    logging.info("\n[PHASE 2] Execution with VLM Feedback\n" + "-"*40)
    primitives = plan.primitive_sequence[:]
    idx = 0
    retries_at_idx = 0
    MAX_RETRIES_PER_PRIMITIVE = 3
    while idx < len(primitives):
        primitive = primitives[idx]
        max_steps = PRIMITIVE_P90_DURATIONS.get(primitive, DEFAULT_PRIMITIVE_STEPS)
        logging.info(f"\n{'='*60}\n[{idx+1}/{len(primitives)}] '{primitive}'\n{'='*60}")
        before, after, wrist_before, wrist_after, frames, obs, datapoints, episode_done = run_primitive_with_checkpoints(
            env, client, primitive, max_steps, obs, collect_data=args.collect_data,
        )
        all_frames.extend(frames)
        all_datapoints.extend(datapoints)
        before_vlm, after_vlm = resize_for_vlm(before), resize_for_vlm(after)
        wrist_before_vlm, wrist_after_vlm = resize_for_vlm(wrist_before), resize_for_vlm(wrist_after)
        Image.fromarray(before_vlm).save(output_dir / f"p{idx+1}_before.png")
        Image.fromarray(after_vlm).save(output_dir / f"p{idx+1}_after.png")
        gripper_state = get_gripper_state(obs)
        logging.info(f"Analyzing with VLM... (gripper: {gripper_state})")
        feedback = analyze_execution(before_vlm, after_vlm, primitive, gripper_state,
                                      wrist_before_vlm, wrist_after_vlm)
        logging.info(f"\nResult: {'SUCCESS' if feedback.success else 'FAIL'} ({feedback.confidence:.0%})")
        logging.info(f"  {feedback.description}")
        prim_log = {"primitive": primitive, "index": idx,
                    "feedback": dataclasses.asdict(feedback), "datapoints": len(datapoints)}
        if episode_done:
            logging.info("Episode terminated by environment, stopping.")
            log.append({"phase": "execution", **prim_log, "episode_terminated": True})
            break
        if feedback.success:
            log.append({"phase": "execution", **prim_log})
            idx += 1
            retries_at_idx = 0
            continue
        logging.info("\n  Attempting correction...")
        _, wrist = get_obs_images(obs)
        correction = get_action_correction(after_vlm, resize_for_vlm(wrist), primitive, feedback.description)
        logging.info(f"  Suggestion: {correction.description}")
        logging.info(f"  Action: {[f'{x:.3f}' for x in correction.action_delta]}")
        prim_log["correction"] = dataclasses.asdict(correction)
        if correction.should_abort:
            logging.info("  VLM suggests abort. Stopping.")
            log.append({"phase": "execution", **prim_log, "aborted": True})
            break
        if correction.switch_to_primitive:
            retries_at_idx += 1
            if retries_at_idx > MAX_RETRIES_PER_PRIMITIVE:
                logging.info(f"  Max retries ({MAX_RETRIES_PER_PRIMITIVE}) exceeded, moving on")
                log.append({"phase": "execution", **prim_log})
                idx += 1
                retries_at_idx = 0
                continue
            logging.info(f"  Switching to: '{correction.switch_to_primitive}' (retry {retries_at_idx})")
            primitives[idx] = correction.switch_to_primitive
            log.append({"phase": "execution", **prim_log})
            continue
        if any(abs(x) > 0.001 for x in correction.action_delta):
            logging.info("  Applying correction...")
            pre_img, pre_wrist = get_obs_images(obs)
            pre_state = _get_state(obs)
            action = np.array(correction.action_delta)
            acc = np.zeros(7)
            for _ in range(2):
                try:
                    obs, _, _, _ = env.step(action.tolist())
                except ValueError as e:
                    if "terminated episode" in str(e):
                        logging.info("  Episode terminated during correction")
                        break
                    raise
                img, _ = get_obs_images(obs)
                all_frames.append(img)
                acc[:6] += action[:6]
                acc[6] = action[6]
            if args.collect_data:
                all_datapoints.append(FlywheelDatapoint(
                    resize_for_policy(pre_img), resize_for_policy(pre_wrist), pre_state,
                    acc.copy(), "vlm_correction", primitive, correction.description, -1,
                ))
        log.append({"phase": "execution", **prim_log})
        idx += 1
        retries_at_idx = 0
    logging.info("\n[PHASE 3] Summary\n" + "-"*40)
    overall = idx >= len(primitives)
    policy_pts = sum(1 for d in all_datapoints if d.source == "policy")
    vlm_pts = sum(1 for d in all_datapoints if d.source == "vlm_correction")
    if all_frames:
        imageio.mimwrite(output_dir / "flywheel_full.mov", all_frames, fps=10, codec="libx264")
    logging.info(f"Datapoints: {len(all_datapoints)} (policy: {policy_pts}, vlm: {vlm_pts})")
    logging.info(f"Completed: {idx}/{len(primitives)} primitives")
    logging.info(f"Overall: {'SUCCESS' if overall else 'INCOMPLETE'}")
    summary = {
        "goal": args.goal, "overall_success": overall,
        "primitives_planned": len(primitives), "primitives_completed": idx,
        "datapoints_total": len(all_datapoints), "skill_gaps": plan.skill_gaps, "log": log,
    }
    with open(output_dir / "flywheel_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    datapoints_export = [{"index": i, "source": d.source, "primitive": d.primitive,
                          "action": d.action.tolist(), "vlm_feedback": d.vlm_feedback}
                         for i, d in enumerate(all_datapoints)]
    with open(output_dir / "datapoints.json", "w") as f:
        json.dump(datapoints_export, f, indent=2)
    logging.info(f"\nSaved to: {output_dir}")
    return summary


# =============================================================================
# Adaptive mode
# =============================================================================

# =============================================================================
# Executor — LIBERO sim concrete implementation
# =============================================================================

class LiberoSimExecutor(_InsightBaseExecutor):
    """Adaptive (non-flywheel) execution for LIBERO sim.

    Inherits env-agnostic VLM behaviors (``evaluate_progress``, ``generate_action``,
    ``GENERATE_PROMPT``) from ``insight.executor.BaseExecutor`` and provides
    the LIBERO-specific dispatch and run loop.

    Flywheel mode is a subclass — see
    ``vlm_flywheel.flywheel_execution.LiberoFlywheelExecutor``.
    """

    # ────────────────── Per-step dispatch ──────────────────

    def execute_step(self, env, client, step, obs, output_dir,
                     primitives_tried, all_frames, all_datapoints):
        """Dispatch one plan step. Known primitives → trained policy via
        ``run_primitive_with_checkpoints``; skill gaps → VLM action loop via
        ``run_new_primitive_with_vlm``. Returns ``(obs, episode_done, step_result)``.
        """
        is_known_primitive = step in AVAILABLE_PRIMITIVES

        if is_known_primitive:
            max_steps = PRIMITIVE_P90_DURATIONS.get(step, DEFAULT_PRIMITIVE_STEPS)
            logging.info(f"Executing with policy ({max_steps} steps)...")
            # VLM completion check is opt-in for tasks where the primitive
            # operates on potentially OOD initial states (e.g. drawer-close).
            use_vlm_check = getattr(self.args, "use_vlm_completion_check", False)
            vlm_check_interval = 20 if (use_vlm_check and step.startswith("move gripper to")) else 0
            before, after, wrist_before, wrist_after, frames, obs, datapoints, episode_done = run_primitive_with_checkpoints(
                env, client, step, max_steps, obs, collect_data=self.args.collect_data,
                vlm_done_check_interval=vlm_check_interval,
            )
            num_frames = len(frames)
            all_frames.extend(frames)
            all_datapoints.extend(datapoints)
            before_vlm, after_vlm = resize_for_vlm(before), resize_for_vlm(after)
            wrist_before_vlm, wrist_after_vlm = resize_for_vlm(wrist_before), resize_for_vlm(wrist_after)
            gripper_state = get_gripper_state(obs)
            success = True
            result_str = f"DONE (gripper: {gripper_state})"
            logging.info(f"Result: {result_str}")
        else:
            logging.info("New primitive (skill gap) - executing with VLM-generated actions...")
            _, wrist_before = get_obs_images(obs)
            before, after, frames, obs, datapoints, episode_done, success = self.run_new_primitive_with_vlm(
                env, step, obs, max_steps=200,
                save_frames=True, collect_data=self.args.collect_data, goal=self.args.goal,
            )
            num_frames = len(frames)
            all_frames.extend(frames)
            all_datapoints.extend(datapoints)
            _, wrist_after = get_obs_images(obs)
            before_vlm, after_vlm = resize_for_vlm(before), resize_for_vlm(after)
            wrist_before_vlm, wrist_after_vlm = resize_for_vlm(wrist_before), resize_for_vlm(wrist_after)
            result_str = f"NEW_PRIMITIVE {'SUCCESS' if success else 'FAIL'}"
            logging.info(f"Result: {result_str}")
            if frames:
                step_num = primitives_tried + 1
                video_path = output_dir / f"step{step_num}_new_primitive.mov"
                imageio.mimwrite(video_path, frames, fps=10, codec="libx264")
                logging.info(f"Saved new primitive video: {video_path}")

        step_num = primitives_tried + 1
        Image.fromarray(before_vlm).save(output_dir / f"step{step_num}_a_before.png")
        Image.fromarray(after_vlm).save(output_dir / f"step{step_num}_b_after.png")
        Image.fromarray(wrist_before_vlm).save(output_dir / f"step{step_num}_c_wrist_before.png")
        Image.fromarray(wrist_after_vlm).save(output_dir / f"step{step_num}_d_wrist_after.png")

        return obs, episode_done, {
            "success": success,
            "result_str": result_str,
            "is_new": not is_known_primitive,
            "num_frames": num_frames,
        }

    # ────────────────── VLM action-generation loop for skill gaps ──────────────────

    def run_new_primitive_with_vlm(
        self, env, new_primitive, obs, max_steps=100,
        save_frames=True, collect_data=True, goal="",
        progress_check_interval=20, vlm_call_interval=5,
    ):
        """Execute a skill-gap primitive via VLM-generated actions.

        Calls ``self.evaluate_progress`` every ``progress_check_interval`` steps
        and ``self.generate_action`` every ``vlm_call_interval`` steps; between
        calls the last action is repeated. Subclasses override the methods to
        change progress-check / action-generation behavior.
        """
        logging.info("  Attempting new primitive with VLM-generated actions...")
        before_img, wrist_before = get_obs_images(obs)
        initial_vlm = resize_for_vlm(before_img, small=True)
        frames = [before_img] if save_frames else []
        datapoints = []
        episode_done = False
        current_feedback = ""
        last_action, last_desc = [0] * 7, ""

        for step in range(max_steps):
            if stop_requested():
                logging.info(f"    Early stop requested at step {step}")
                break
            img, wrist = get_obs_images(obs)
            if save_frames:
                frames.append(img)

            img_vlm = resize_for_vlm(img, small=True)
            wrist_vlm = resize_for_vlm(wrist, small=True)

            if step > 0 and step % progress_check_interval == 0:
                is_progressing, suggestion = self.evaluate_progress(
                    initial_vlm, img_vlm, new_primitive, goal, step,
                )
                if not is_progressing and suggestion:
                    current_feedback = suggestion
                elif is_progressing:
                    current_feedback = ""

            if step == 0 or step % vlm_call_interval == 0:
                action, desc, done = self.generate_action(
                    initial_vlm, img_vlm, wrist_vlm, new_primitive, step, goal, current_feedback,
                )
                if action is not None:
                    last_action, last_desc = action, desc
                else:
                    action, desc, done = last_action, last_desc, False
            else:
                action, desc, done = last_action, last_desc, False

            if step == 0:
                logging.info(f"    Step 0: {desc}")
                logging.info(
                    f"    Action: [{action[0]:+.3f}, {action[1]:+.3f}, {action[2]:+.3f}, "
                    f"{action[3]:+.3f}, {action[4]:+.3f}, {action[5]:+.3f}] grip={action[6]:+.3f}"
                )
            elif step % 10 == 0:
                logging.info(f"    Step {step}: {desc}")

            if collect_data:
                state = _get_state(obs)
                datapoints.append(FlywheelDatapoint(
                    resize_for_policy(img), resize_for_policy(wrist), state,
                    np.array(action), "vlm_generated", new_primitive,
                    vlm_feedback=desc, step_in_primitive=step,
                ))

            try:
                obs, _, env_done, _ = env.step(action)
            except ValueError as e:
                if "terminated episode" in str(e):
                    episode_done = True
                    logging.info(f"    Episode terminated at step {step}")
                    break
                raise
            if env_done:
                episode_done = True
                logging.info(f"    Episode terminated at step {step}")
                break

            if done:
                verify_img = resize_for_vlm(get_obs_images(obs)[0])
                is_progressing, suggestion = self.evaluate_progress(
                    initial_vlm, verify_img, new_primitive, goal, step,
                )
                if not is_progressing and suggestion:
                    logging.info(f"    VLM says done at step {step} but progress check disagrees - continuing")
                    current_feedback = suggestion
                    done = False
                else:
                    logging.info(f"    VLM says done at step {step} - verified by progress check")
                    break

        after_img, wrist_after = get_obs_images(obs)
        before_vlm, after_vlm = resize_for_vlm(before_img), resize_for_vlm(after_img)
        wrist_before_vlm, wrist_after_vlm = resize_for_vlm(wrist_before), resize_for_vlm(wrist_after)
        gripper_state = get_gripper_state(obs)
        eval_primitive = f"{new_primitive} (goal: {goal})" if goal else new_primitive
        feedback = analyze_execution(
            before_vlm, after_vlm, eval_primitive, gripper_state,
            wrist_before_vlm, wrist_after_vlm,
        )
        success = feedback.success
        logging.info(f"    Result: {'SUCCESS' if success else 'FAIL'} (gripper: {gripper_state}) - {feedback.description}")
        return before_img, after_img, frames, obs, datapoints, episode_done, success

    # ────────────────── Plan-level orchestration ──────────────────

    def execute_plan(self, env, client, plan, obs, output_dir,
                     primitives_tried, all_frames, all_datapoints, history):
        """Iterate ``plan.primitive_sequence`` and dispatch each step via
        ``self.execute_step``. Subclasses can override ``execute_step`` to
        intercept (e.g. skill-gap routing in flywheel mode)."""
        episode_done = False
        for plan_idx, step in enumerate(plan.primitive_sequence):
            if primitives_tried >= self.args.max_primitives or stop_requested():
                break
            logging.info(f"\n{'='*60}\n[Step {plan_idx+1}/{len(plan.primitive_sequence)}] '{step}'\n{'='*60}")
            obs, episode_done, step_result = self.execute_step(
                env, client, step, obs, output_dir,
                primitives_tried, all_frames, all_datapoints,
            )
            history.append({
                "step": primitives_tried + 1,
                "primitive": step,
                "is_new": step_result["is_new"],
                "result": step_result["result_str"],
                "success": step_result["success"],
                "num_frames": step_result["num_frames"],
            })
            primitives_tried += 1
            if episode_done:
                logging.info("Episode terminated by environment, stopping.")
                break
        return obs, primitives_tried, episode_done

    def run_adaptive(self, env, client, output_dir):
        """Top-level entry: plan a goal, execute with re-planning on failure."""
        args = self.args
        logging.info(
            f"{'='*60}\nADAPTIVE MODE (Plan-Execute)\n"
            f"Goal: {args.goal}\nMax primitives: {args.max_primitives}\n{'='*60}"
        )
        all_datapoints = []
        all_frames = []
        history = []
        primitives_tried = 0
        goal_achieved = False
        error_msg = None
        try:
            obs = settle_physics(env)
            robot = _find_robot(env)
            initial_jpos = np.array(env.sim.data.qpos[robot._ref_joint_pos_indexes]).copy()
            initial_gripper_jpos = np.array(env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes]).copy()
            initial_img, _ = get_obs_images(obs)
            Image.fromarray(resize_for_vlm(initial_img)).save(output_dir / "initial_scene.png")
            logging.info(f"\n{'='*60}\n[PLANNING] Breaking down goal...\n{'='*60}")
            plan = plan_task(args.goal, resize_for_vlm(initial_img))
            logging.info(f"Plan: {plan.primitive_sequence}")
            logging.info(f"Skill gaps: {plan.skill_gaps}")
            logging.info(f"Reasoning: {plan.reasoning}")
            logging.info(f"Requires new primitive: {plan.requires_new_primitive}")
            if not plan.primitive_sequence:
                logging.warning("Empty plan returned!")
                error_msg = "Empty plan"
            else:
                logging.info(f"\n{'='*60}\n[EXECUTION] Running plan...\n{'='*60}")
                replan_count = 0
                MAX_REPLANS = 3
                episode_done = False
                obs, primitives_tried, episode_done = self.execute_plan(
                    env, client, plan, obs, output_dir,
                    primitives_tried, all_frames, all_datapoints, history,
                )
                while not episode_done and not goal_achieved and not stop_requested():
                    logging.info("\nResetting gripper to initial pose...")
                    obs = reset_gripper_pose(env, obs, initial_jpos, initial_gripper_jpos)
                    current_img, _ = get_obs_images(obs)
                    goal_achieved, reasoning = check_goal_achieved(
                        args.goal, current_img, save_dir=output_dir, check_num=replan_count,
                    )
                    logging.info(f"Goal check: {'ACHIEVED' if goal_achieved else 'NOT achieved'} - {reasoning}")
                    if goal_achieved:
                        break
                    if replan_count >= MAX_REPLANS or primitives_tried >= args.max_primitives:
                        logging.info(f"Stopping: replans={replan_count}/{MAX_REPLANS}, primitives={primitives_tried}/{args.max_primitives}")
                        break
                    replan_count += 1
                    logging.info(
                        f"\n{'='*60}\n[RE-PLANNING] Goal not achieved.\n"
                        f"Reason: {reasoning}\nCreating new plan... (attempt {replan_count}/{MAX_REPLANS})\n{'='*60}"
                    )
                    plan = plan_task(args.goal, resize_for_vlm(current_img), history)
                    logging.info(f"New plan: {plan.primitive_sequence}")
                    if not plan.primitive_sequence:
                        logging.warning("Empty re-plan returned! Stopping.")
                        break
                    obs, primitives_tried, episode_done = self.execute_plan(
                        env, client, plan, obs, output_dir,
                        primitives_tried, all_frames, all_datapoints, history,
                    )
                    if episode_done:
                        break
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Error during adaptive run: {e}")
            traceback.print_exc()
        finally:
            return _save_adaptive_results(
                output_dir, args.goal, goal_achieved, primitives_tried,
                history, all_frames, all_datapoints, error_msg,
            )


# =============================================================================
# Main
# =============================================================================

def _main(args: Args) -> None:
    from libero.libero import get_libero_path
    from openpi_client import websocket_client_policy as _wcp

    np.random.seed(args.seed)

    pst = timezone(timedelta(hours=-7))  # PDT (Mar-Nov)
    now = datetime.now(pst)
    date_folder = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%Y-%m-%d_%H%M%S")
    output_dir = pathlib.Path(args.output_dir) / date_folder / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )

    if args.vlm != "gpt":
        set_vlm_provider(args.vlm)

    bddl_file = pathlib.Path(get_libero_path("bddl_files")) / "lego_primitives" / "wide_range" / "pick_blue_place_target_wide.bddl"
    if not bddl_file.exists():
        logging.error(f"BDDL file not found: {bddl_file}")
        return

    _start_keyboard_listener()
    logging.info("Press 's' at any time to save and stop early.")
    logging.info("Creating LIBERO environment...")
    env = create_env(bddl_file, args.seed)

    logging.info(f"Connecting to policy server at {args.host}:{args.port}...")
    client = _wcp.WebsocketClientPolicy(args.host, args.port)

    if args.adaptive:
        LiberoSimExecutor(args).run_adaptive(env, client, output_dir)
    elif args.flywheel:
        run_flywheel(env, client, args, output_dir)
    elif args.sequence:
        primitives = DEFAULT_SEQUENCE if args.sequence.lower() == "default" else [p.strip() for p in args.sequence.split(",")]
        run_sequence(env, client, args, output_dir, primitives)
    elif args.loop:
        run_with_retry(env, client, args, output_dir)
    else:
        run_single(env, client, args, output_dir)

    env.close()
