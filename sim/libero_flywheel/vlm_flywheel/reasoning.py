"""VLM reasoning: plan_task, pre-analysis, position check, progress eval, goal check."""

from __future__ import annotations

import json
import logging
import traceback

import numpy as np
from PIL import Image

import insight.vlm_client
from insight.reasoning import (
    analyze_execution,
    apply_correction_to_primitive,
    build_fixed_action_from_hint as _ic_build_fixed_action_from_hint,
    check_goal_achieved as _ic_check_goal_achieved,
    decide_next_primitive as _ic_decide_next_primitive,
    get_action_correction,
    parse_axis_direction as _parse_verified_axis_direction,
)
from .env import (
    AVAILABLE_PRIMITIVES,
    DEFAULT_SCENE_CONTEXT,
    ActionCorrection,
    TaskPlan,
    VLMFeedback,
    crop_around_red_block,
    get_gripper_state,
    get_obs_images,
    resize_for_vlm,
)
from .vlm import parse_vlm_json, vlm_chat, vlm_with_images
from .prompts import (
    _PLAN_TASK_SYSTEM,
    _EVALUATE_PROGRESS_SYSTEM,
    _POSITION_CHECK_SYSTEM,
    _POSITION_CHECK_SYSTEM_POS_ONLY,
    _EVALUATE_PROGRESS_SYSTEM_V2,
)
# NOTE: _PREANALYZE_SYSTEM (sim's world-frame lookup-table prompt) is no
# longer used by ``_preanalyze_skill_gap``; that path delegates to
# ``insight.reasoning.preanalyze_translation`` so sim + xArm share the
# exact same EE-local-frame prompt. The constant is still exported from
# ``vlm_flywheel/__init__.py`` for backward compatibility with any
# external callers that may reference it.
from .control import (
    _find_red_block_joint,
    _quat_to_euler_deg,
    _get_axis_info,
    _detect_red_mask_hsv,
    _draw_indicators,
    _draw_world_axes_only,
    _crop_around_block_hsv,
    _MAX_ROT_CMD,
    _MAX_TRANS_CMD,
)


# =============================================================================
# Base reasoning functions
# =============================================================================
# ``analyze_execution`` / ``get_action_correction`` / ``apply_correction_to_primitive``
# are env-agnostic and live in insight.reasoning; imported above and
# re-exported via __init__.py for sim callers.


def plan_task(goal, scene_image, history=None):
    """Plan how to achieve goal using available primitives.

    Thin sim-side wrapper around ``insight.reasoning.plan_task``: handles
    the sim-specific globals (DEFAULT_SCENE_CONTEXT, AVAILABLE_PRIMITIVES),
    appends the optional history block to the goal string, uses the basic
    sim-local ``_PLAN_TASK_SYSTEM`` prompt, and wraps the parsed dict in the
    sim ``TaskPlan`` dataclass. Flywheel-mode planning uses the more elaborate
    ``_plan_task_with_notes`` in flywheel_execution.py with the FLYWHEEL prompt.
    """
    from insight.reasoning import plan_task as _ic_plan_task

    history_context = ""
    if history:
        history_context = "\n\nPREVIOUS ACTIONS ALREADY COMPLETED:\n"
        for h in history:
            history_context += f"- {h['primitive']}: {h['result']}\n"
        history_context += "\nThe image shows the CURRENT state. Plan only the REMAINING steps."
    full_goal = f"{goal}{history_context}"

    try:
        data = _ic_plan_task(
            goal=full_goal,
            scene_image=scene_image,
            available_primitives=list(AVAILABLE_PRIMITIVES),
            scene_context=DEFAULT_SCENE_CONTEXT,
            system_template=_PLAN_TASK_SYSTEM,
            max_tokens=4096,
        )
        return TaskPlan(
            goal=goal,
            primitive_sequence=data.get("primitive_sequence", []),
            skill_gaps=data.get("skill_gaps", []),
            reasoning=data.get("reasoning", ""),
            confidence=data.get("confidence", 0.5),
            requires_new_primitive=data.get("requires_new_primitive", False),
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logging.error(f"Failed to parse plan: {e}")
        return TaskPlan(goal, [], ["parse_error"], str(e), 0.0, True)


def check_goal_achieved(goal, current_image, save_dir=None, check_num=0):
    """Sim-side goal check: pre-crops around the red lego block (if visible),
    saves both views, then delegates to ``insight.check_goal_achieved``."""
    cropped = crop_around_red_block(current_image)
    if cropped is not None:
        check_img = resize_for_vlm(cropped)
        logging.info("  [goal check] Using cropped close-up of red block")
    else:
        check_img = resize_for_vlm(current_image)
        logging.info("  [goal check] No red block found, using full image")
    if save_dir is not None:
        scene_img = resize_for_vlm(current_image) if cropped is not None else check_img
        Image.fromarray(scene_img).save(save_dir / f"goal_check_{check_num}_scene.png")
        Image.fromarray(check_img).save(save_dir / f"goal_check_{check_num}_crop.png")
    return _ic_check_goal_achieved(goal, check_img)


def decide_next_primitive(goal, current_image, history):
    """Sim-side wrapper that fills in LIBERO's primitive list + scene context."""
    return _ic_decide_next_primitive(
        goal, current_image, history,
        available_primitives=list(AVAILABLE_PRIMITIVES),
        scene_context=DEFAULT_SCENE_CONTEXT,
    )


# =============================================================================
# Second-flip reasoning
# =============================================================================

def _get_peg_direction_vector(env):
    """Return the peg direction as a unit vector in world frame, or None."""
    joint_name = _find_red_block_joint(env)
    if joint_name is None:
        return None
    qpos = env.sim.data.get_joint_qpos(joint_name)
    qw, qx, qy, qz = qpos[3], qpos[4], qpos[5], qpos[6]
    peg = -np.array([
        2 * (qx * qz + qw * qy),
        2 * (qy * qz - qw * qx),
        1 - 2 * (qx * qx + qy * qy),
    ])
    return peg


def _get_peg_direction_text(env):
    """Read block quaternion from sim and describe peg direction."""
    peg = _get_peg_direction_vector(env)
    if peg is None:
        return None
    if peg[2] > 0.7:
        return "Peg is pointing UP (+Z). Already near goal."
    if peg[2] < -0.7:
        return "Peg is pointing DOWN (-Z). Needs 180 deg flip."
    if abs(peg[0]) > abs(peg[1]):
        d = "+X" if peg[0] > 0 else "-X"
    else:
        d = "+Y" if peg[1] > 0 else "-Y"
    return f"Peg is pointing {d}. Block is on its side."


def _prepare_preanalysis_images(env, obs):
    """Prepare overhead and wrist images for pre-analysis VLM call.

    Both images are passed at full FOV with no crop. Earlier versions
    cropped the overhead around the red block (and the wrist around the
    red mask), but cropping discarded the surrounding context the VLM
    needed for rotation reasoning (gripper orientation cues, workspace
    edges, other objects). Mirrors the xArm side which sends full-FOV
    images to ``insight.reasoning.preanalyze_translation``.
    """
    import cv2
    img, wrist = get_obs_images(obs)
    crop_vlm = resize_for_vlm(img)
    wrist_vlm = resize_for_vlm(wrist)

    def _add_label(image, label):
        labeled = image.copy()
        cv2.putText(labeled, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(labeled, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return labeled

    crop_vlm = _add_label(crop_vlm, "IMAGE 1: Overhead")
    wrist_vlm = _add_label(wrist_vlm, "IMAGE 2: Wrist Camera")
    return crop_vlm, wrist_vlm


def _preanalyze_skill_gap(env, obs, primitive, goal, save_dir=None, plan_context=None,
                          skill_gap_state=None, feedback_block: str = ""):
    """Ask VLM to determine correct axis and direction before a skill gap.

    Delegates to ``insight.reasoning.preanalyze_translation`` — the SAME
    function the real-world xArm flywheel uses, with the SAME system
    prompt (``PREANALYZE_TRANSLATION_SYSTEM``). Critical consequence:
    rotation axes (drx/dry/drz) are interpreted in the GRIPPER's LOCAL
    frame here, matching the ``frame="local"`` target construction in
    ``_execute_skill_gap``:
      - drx (roll):  rotation around gripper's local X — sideways flip
      - dry (pitch): rotation around gripper's local Y — pour / forward tilt
      - drz (yaw):   rotation around gripper's local Z — unscrew / spin

    The old sim-specific ``_PREANALYZE_SYSTEM`` (world-frame lookup
    table) is no longer used here.

    ``skill_gap_state`` is the executor's per-run state dict; we write
    ``target_rotation_deg`` into it for the P-controller. The returned
    ``hint`` string keeps the legacy ``"primary: <axis>=<signed_value>"``
    format so the downstream regex parser in
    ``flywheel_execution._execute_skill_gap`` is unchanged.
    ``recommended_drz_deg`` is always ``None`` under the new prompt
    (it doesn't ask for a grasp orientation field).
    """
    if skill_gap_state is None:
        skill_gap_state = {}
    crop_vlm, wrist_vlm = _prepare_preanalysis_images(env, obs)

    if save_dir is not None:
        safe_name = primitive.replace(" ", "_")[:40]
        Image.fromarray(crop_vlm).save(save_dir / f"preanalysis_crop_{safe_name}.png")
        Image.fromarray(wrist_vlm).save(save_dir / f"preanalysis_wrist_{safe_name}.png")

    if feedback_block:
        logging.info(
            "  [Pre-analysis] NOTE: feedback_block ignored — sim now uses the "
            "shared insight.preanalyze_translation prompt which doesn't yet "
            "splice feedback. Remove --feedback-dir or extend insight prompt "
            "to re-enable."
        )

    logging.info(f"  [Pre-analysis] GOAL: {goal}")
    logging.info(f"  [Pre-analysis] PRIMITIVE: {primitive}")
    logging.info(f"  [Pre-analysis] Using shared insight prompt PREANALYZE_TRANSLATION_SYSTEM (EE local frame)")
    try:
        logging.info(f"  [Pre-analysis] Using model: {insight.vlm_client.get_model()}")
        from insight.reasoning import preanalyze_translation as _insight_preanalyze
        data = _insight_preanalyze(
            primitive=primitive,
            scene_image=crop_vlm,
            wrist_image=wrist_vlm,
            goal=goal,
            plan_context=plan_context,
            max_tokens=4096,
        )
        logging.info(f"  [Pre-analysis] PARSED: {data}")

        if data.get("already_complete"):
            logging.info("  [Pre-analysis] VLM says already_complete=true — skipping skill gap")
            skill_gap_state["target_rotation_deg"] = 0.0
            return {"hint": "", "recommended_drz_deg": None, "already_complete": True}

        axis = data.get("axis", "")
        if not axis:
            logging.warning("  [Pre-analysis] No axis returned, falling back to empty hint")
            return {"hint": "", "recommended_drz_deg": None}

        if axis.startswith("dr"):
            signed_deg = float(data.get("signed_magnitude_deg", 0.0))
            # FRANKA-SIM-ONLY ROTATION SIGN FLIP. Empirically the
            # rotation convention is swapped between Franka sim
            # (LIBERO/robosuite OSC_POSE) and xArm hardware: the same
            # VLM-issued ``+dry=+N°`` rotates the gripper in opposite
            # physical directions on the two robots. The shared
            # ``PREANALYZE_TRANSLATION_SYSTEM`` prompt describes the
            # xArm convention, so the VLM's output is correct for xArm
            # but inverted for sim. Negating the signed magnitude here
            # makes sim's physical motion match the VLM's intent.
            # This edit is SIM ONLY: xArm calls
            # ``insight.reasoning.preanalyze_translation`` directly and
            # does not pass through this function.
            signed_deg = -signed_deg
            skill_gap_state["target_rotation_deg"] = abs(signed_deg)
            value_str = f"{signed_deg:+.3f}"  # legacy regex expects a signed number
        else:
            signed_m = float(data.get("signed_magnitude_m", 0.0))
            value_str = f"{signed_m:+.4f}"

        hint = f"primary: {axis}={value_str}, also use other axes"
        logging.info(
            f"  [Pre-analysis] VLM reasoning: "
            f"{data.get('current_state', '?')} → {data.get('target_state', '?')}"
        )
        logging.info(f"  [Pre-analysis] hint={hint!r}")
        return {"hint": hint, "recommended_drz_deg": None}
    except Exception as e:
        logging.warning(f"  [Pre-analysis] Failed: {e}")
        traceback.print_exc()
    return {"hint": "", "recommended_drz_deg": None}


def _evaluate_gripper_position(env, obs, goal, next_steps=None, step_note="", save_dir=None, check_num=0, upcoming_action="", orientation_locked=False):
    """Ask VLM whether gripper positioning is suitable for the goal."""
    import cv2
    img, wrist = get_obs_images(obs)

    cropped = _crop_around_block_hsv(img)
    img_vlm = resize_for_vlm(cropped if cropped is not None else img)
    wrist_vlm = resize_for_vlm(wrist)

    if save_dir is not None:
        debug_dir = save_dir / "position_checks"
        debug_dir.mkdir(exist_ok=True)
        Image.fromarray(img_vlm).save(debug_dir / f"check{check_num}_scene.png")
        Image.fromarray(wrist_vlm).save(debug_dir / f"check{check_num}_wrist.png")

    ee_pos = obs["robot0_eef_pos"]
    ee_quat = obs["robot0_eef_quat"]
    ee_euler = _quat_to_euler_deg([ee_quat[3], ee_quat[0], ee_quat[1], ee_quat[2]])

    state_text = (
        f"Gripper position: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]"
        f"\nGripper orientation (roll, pitch, yaw deg): [{ee_euler[0]:.0f}, {ee_euler[1]:.0f}, {ee_euler[2]:.0f}]"
    )
    if not orientation_locked:
        state_text += f"\nNote: yaw (drz) controls gripper finger direction in the overhead view"

    next_steps_text = ""
    if next_steps:
        steps_str = " → ".join(next_steps)
        next_steps_text = f"\nNext steps after this: {steps_str}"

    skill_gap_context = ""
    if upcoming_action:
        skill_gap_context = f"\nAfter grasping, the robot will perform: {upcoming_action}. Ensure the gripper is positioned to enable this motion."
    elif next_steps:
        for ns in next_steps:
            if ns not in AVAILABLE_PRIMITIVES:
                skill_gap_context = f"\nAfter grasping, the robot will execute: \"{ns}\". Ensure the gripper is positioned to enable this."
                break

    orientation_note = ""
    if orientation_locked:
        orientation_note = (
            f"\nIMPORTANT: The gripper orientation is FIXED and cannot be changed."
            f" Do NOT suggest any rotation corrections."
            f" Only evaluate whether the gripper XY position is centered on the object"
            f" and the Z height is correct for grasping."
        )

    images = [img_vlm, wrist_vlm]
    image_desc = (
        "IMAGE 1: scene view showing gripper fingers and object."
        " IMAGE 2: wrist camera close-up of the object."
    )

    prompt = f"GOAL: {goal}\n{image_desc}\n{state_text}{next_steps_text}{skill_gap_context}{orientation_note}\nIs the gripper positioned well to achieve this goal?"

    system = _POSITION_CHECK_SYSTEM
    if orientation_locked:
        system = _POSITION_CHECK_SYSTEM_POS_ONLY

    logging.info(f"Position check prompt: \033[36m{prompt}\033[0m")
    try:
        raw = vlm_with_images(
            prompt, images,
            max_tokens=4096, system=system,
        )
        logging.info(f"  [Position check] RAW VLM response:\n\033[32m{raw}\033[0m")
        data = parse_vlm_json(raw)
        pos_ok = data.get("position_ok", True)
        if orientation_locked:
            orient_ok = True
        else:
            orient_ok = data.get("orientation_ok", True)
        well_positioned = pos_ok and orient_ok
        correction = data.get("correction", "")
        reasoning = data.get("reasoning", "")
    except Exception as e:
        logging.warning(f"Position check failed: {e}")
        return True, ""

    logging.info(f"Position check: pos={'OK' if pos_ok else 'BAD'} orient={'OK' if orient_ok else 'BAD'}{' (orient locked)' if orientation_locked else ''} — {reasoning}")
    if correction:
        logging.info(f"Suggested correction: {correction}")

    return well_positioned, correction


def _check_goal_achieved_sim(env, goal, current_image, save_dir=None, check_num=0,
                              before_image=None):
    """Sim goal check using BDDL + peg-direction sim ground truth.

    Why sim and hardware use different oracles:
    - **Sim** has reliable ground truth (block joint qpos). We use BDDL's
      ``env.check_success`` (task-definition success criterion) with a
      peg-direction fallback for the flip task whose BDDL goal is
      overspecified for placement at a target marker.
    - **Hardware** (xArm flywheel) uses a VLM oracle
      (``insight.reasoning.check_task_completion``) because real-world
      object pose isn't available.

    For sim demo collection in particular, the VLM oracle gives
    false positives that pollute the collected HDF5 dataset (we'd
    retrain on demos that weren't actually successful flips). The
    BDDL+peg-direction path uses ground truth and avoids this.

    The ``before_image`` arg is kept for caller-side compatibility
    but ignored here.
    """
    del before_image  # unused; kept for caller-side API compatibility
    if save_dir is not None:
        Image.fromarray(resize_for_vlm(current_image)).save(
            save_dir / f"goal_check_{check_num}_scene.png"
        )
    try:
        if env.check_success():
            logging.info("  [goal check] BDDL goal achieved!")
            return True, "BDDL goal condition satisfied"
    except Exception as e:
        logging.warning(f"  [goal check] env.check_success() raised: {e}")
    # Peg-direction fallback (lego flip): peg pointing approximately up
    # = task success even if BDDL placement predicate isn't satisfied.
    peg = _get_peg_direction_vector(env)
    if peg is not None:
        angle_from_up = np.degrees(np.arccos(np.clip(peg[2], -1, 1)))
        logging.info(f"  [goal check] Peg z-component: {peg[2]:.2f}, angle from +Z: {angle_from_up:.0f}°")
        if peg[2] > 0.85:
            return True, f"Peg pointing up (angle from +Z: {angle_from_up:.0f}°)"
    return False, "Goal not yet achieved (BDDL + peg-direction fallback both negative)"


# =============================================================================
# Helpers
# =============================================================================

# Axis-parsing helpers moved to insight.reasoning. ``_parse_verified_axis_direction``
# is the same function (env-agnostic regex); ``_build_fixed_action_from_hint``
# is a sim-side wrapper that fills in the LIBERO control-side per-step caps.

_AXIS_TO_INDEX = {"dx": 0, "dy": 1, "dz": 2, "drx": 3, "dry": 4, "drz": 5}


def _build_fixed_action_from_hint(hint):
    """Build a fixed 7-D OSC_POSE action from the pre-analysis hint string."""
    return _ic_build_fixed_action_from_hint(
        hint, max_rot_cmd=_MAX_ROT_CMD, max_trans_cmd=_MAX_TRANS_CMD,
    )


# =============================================================================
# Progress evaluation variants
# =============================================================================

def _evaluate_progress_sim(initial_img, current_img, new_primitive, goal, steps_so_far,
                           skill_gap_state=None):
    """Sim/proprioception-based progress check — no VLM call.

    Reads ``rotation_complete`` and writes ``goal_complete`` on the executor's
    ``skill_gap_state`` dict.
    """
    if skill_gap_state is None:
        skill_gap_state = {}
    if skill_gap_state.get("rotation_complete", False):
        logging.info("    [Progress Check - SIM] Rotation target reached (proprioception) — confirming complete")
        skill_gap_state["goal_complete"] = True
        return True, ""
    logging.info(f"    [Progress Check - SIM] Step {steps_so_far} — rotation not yet complete, continuing")
    return True, ""


def _evaluate_progress_v2(initial_img, current_img, new_primitive, goal, steps_so_far,
                          skill_gap_state=None):
    """VLM-based progress evaluation that also checks goal completion.

    Reads/writes ``rotation_complete`` / ``goal_complete`` / ``preanalysis_hint``
    on the executor's ``skill_gap_state`` dict.
    """
    if skill_gap_state is None:
        skill_gap_state = {}
    if skill_gap_state.get("rotation_complete", False):
        logging.info("    [Progress Check] Rotation target reached (proprioception) — confirming complete")
        skill_gap_state["goal_complete"] = True
        return True, ""

    hint = skill_gap_state.get("preanalysis_hint", "")
    verified_axis, verified_sign = _parse_verified_axis_direction(hint)
    hint_context = (
        f"\nPRE-ANALYSIS DETERMINED: {hint}. Stick with this direction unless "
        "you see CLEAR evidence it is wrong (object moving away from goal)."
        if hint else ""
    )
    prompt = f'GOAL: {goal}\nPRIMITIVE: "{new_primitive}"\nSTEPS SO FAR: {steps_so_far}{hint_context}'
    try:
        data = parse_vlm_json(vlm_with_images(
            prompt, [initial_img, current_img], max_tokens=4096,
            system=_EVALUATE_PROGRESS_SYSTEM_V2,
        ))
        making_progress = data.get("making_progress", True)
        goal_complete = data.get("goal_complete", False)
        current_state = data.get("current_state", "")
        goal_state = data.get("goal_state", "")
        gap = data.get("gap", "")
        suggested_axis = data.get("suggested_axis", "")
        suggested_direction = data.get("suggested_direction", "")
        suggested_motion = data.get("suggested_motion", "")
        position_correction = data.get("position_correction", "none") or "none"

        if verified_axis and suggested_axis == verified_axis:
            vlm_sign = "+" if suggested_direction.lstrip().startswith("+") else "-"
            if vlm_sign != verified_sign:
                import re
                mag_match = re.search(r'[\d.]+', suggested_direction)
                mag = mag_match.group(0) if mag_match else ("0.150" if verified_axis.startswith("dr") else "0.030")
                old_dir = suggested_direction
                suggested_direction = f"{verified_sign}{mag}"
                logging.info(f"    [Progress Check] OVERRIDING VLM direction: {suggested_axis}={old_dir} → {suggested_direction} (empirically verified)")

        feedback = f"Current: {current_state}. Goal: {goal_state}. Gap: {gap}."
        if suggested_axis:
            feedback += f" Primary axis: {suggested_axis}={suggested_direction}. You may also use other axes."
        if position_correction.lower() != "none":
            feedback += f" Position correction: {position_correction}."
            logging.info(f"    [Progress Check] Position correction: {position_correction}")
        feedback += f" Motion: {suggested_motion}"

        if not making_progress:
            logging.info("    [Progress Check] NOT making progress")
            logging.info(f"    [Progress Check] Current: {current_state}")
            logging.info(f"    [Progress Check] Goal: {goal_state}")
            logging.info(f"    [Progress Check] Gap: {gap}")
            logging.info(f"    [Progress Check] Suggested: {suggested_axis}={suggested_direction} ({suggested_motion})")
            return False, feedback

        if not goal_complete:
            logging.info(f"    [Progress Check] On track but NOT complete — {current_state}")
            return False, f"Goal not yet complete. {feedback}"

        logging.info(f"    [Progress Check] Goal COMPLETE — {current_state}")
        skill_gap_state["goal_complete"] = True
        return True, ""
    except Exception as e:
        logging.error(f"Failed to evaluate progress: {e}")
        return True, ""


def _generate_action_wrapper(initial_img, current_img, wrist_img, new_primitive, step,
                             goal="", feedback="", system=None, skill_gap_state=None):
    """Flywheel action generator wrapper.

    Reads goal-complete + pre-analysis hint from the executor's
    ``skill_gap_state`` dict (passed in by ``LiberoFlywheelExecutor.generate_action``).
    When ``skill_gap_state`` is None (legacy callers), defaults to a fresh
    empty state — the wrapper still runs but loses the early-stop / hint-replay
    behavior.

    ``system`` selects the prompt variant; defaults to the FLYWHEEL prompt.
    """
    if system is None:
        from .prompts import _GENERATE_NEW_PRIMITIVE_SYSTEM_FLYWHEEL
        system = _GENERATE_NEW_PRIMITIVE_SYSTEM_FLYWHEEL
    if skill_gap_state is None:
        skill_gap_state = {}
    from . import flywheel_execution

    if skill_gap_state.get("goal_complete"):
        skill_gap_state["goal_complete"] = False
        return [0, 0, 0, 0, 0, 0, 1.0], "Goal verified complete by progress check — stopping", True

    if not flywheel_execution._USE_VLM_ACTION_GENERATION:
        hint = skill_gap_state.get("preanalysis_hint", "")
        fixed = _build_fixed_action_from_hint(hint)
        if fixed is not None:
            if step % 10 == 0:
                logging.info(f"    [Action - fixed] Step {step}: replaying hint action from '{hint}'")
            return fixed, f"Fixed action from hint: {hint}", False
        logging.warning(f"    [Action - fixed] Could not parse hint '{hint}', falling back to VLM")

    goal_context = f"\nOVERALL GOAL: {goal}" if goal else ""
    if feedback and feedback.startswith("Goal not yet complete"):
        feedback_context = f"\n\nFEEDBACK: {feedback}\nKeep going with your current approach."
    elif feedback:
        feedback_context = f"\n\nFEEDBACK: {feedback}\nYour previous approach was NOT working. You MUST change your action."
    else:
        feedback_context = ""
    prompt = f'NEW PRIMITIVE: "{new_primitive}"{goal_context}\nSTEP: {step}{feedback_context}'

    raw_response = ""
    try:
        raw_response = vlm_with_images(
            prompt, [initial_img, current_img], max_tokens=4096,
            system=system,
        )
        data = parse_vlm_json(raw_response)
        action = data.get("action", [0] * 7)
        if not isinstance(action, list) or len(action) != 7:
            action = [0] * 7
        else:
            action = [float(x) for x in action]
            action[6] = 1.0 if action[6] >= 0 else -1.0
        desc = data.get("goal_analysis", "") or data.get("reasoning", "")
        return action, desc, data.get("done", False)
    except Exception as e:
        logging.error(f"Failed to generate action: {e}")
        if raw_response:
            logging.debug(f"Raw VLM response: {raw_response[:300]}")
        return None, str(e), False
