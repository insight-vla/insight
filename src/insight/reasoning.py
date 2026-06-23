"""Pure VLM-reasoning functions shared by sim and real pipelines.

Each function here is environment-agnostic: it takes RGB ndarrays + a primitive
label, calls the VLM, and returns a Python dict / bool. Environment-specific
glue (image resizing, saving debug frames, proprioception checks) lives in the
caller.

Callers are responsible for configuring the VLM provider (via
``insight.vlm_client.set_provider``) and the research-context preamble
(via ``insight.vlm_client.set_research_context``) before invoking these
functions.
"""

from __future__ import annotations

import json
import logging
import re

import numpy as np

from insight.prompts import (
    ACTION_CORRECTION_SYSTEM,
    ANALYZE_EXECUTION_SYSTEM,
    CHECK_GOAL_SYSTEM,
    DECIDE_NEXT_PRIMITIVE_SYSTEM,
    EVALUATE_PROGRESS_SYSTEM,
    GENERATE_NEW_PRIMITIVE_SYSTEM,
    PLAN_TASK_SYSTEM,
    PREANALYZE_NL_SYSTEM,
    PREANALYZE_TRANSLATION_SYSTEM,
    PRIMITIVE_DONE_SYSTEM,
    TASK_COMPLETION_SYSTEM,
)
from insight.types import ActionCorrection, TaskPlan, VLMFeedback
from insight.vlm_client import chat, parse_json, with_images


_log = logging.getLogger(__name__)


def check_primitive_done_verbose(
    primitive: str,
    exterior: np.ndarray,
    wrist: np.ndarray,
    num_votes: int = 1,
    step_num: int = 0,
    max_tokens: int = 256,
) -> dict:
    """Ask the VLM whether ``primitive`` is complete given the two camera views.

    Performs ``num_votes`` independent calls and majority-votes the verdict.
    Returns ``{"done": bool, "reasoning": str, "votes": list[bool]}``.
    """
    if num_votes < 1:
        raise ValueError(f"num_votes must be >= 1, got {num_votes}")
    prompt = f'Primitive: "{primitive}"\nIs this primitive complete?'
    votes: list[bool] = []
    reasoning = ""
    for i in range(num_votes):
        try:
            raw = with_images(prompt, [exterior, wrist], system=PRIMITIVE_DONE_SYSTEM, max_tokens=max_tokens)
            data = parse_json(raw)
            vote = bool(data.get("done", False))
            r = str(data.get("reasoning", ""))
            votes.append(vote)
            if i == 0:
                reasoning = r
                logging.info("  [VLM] step=%d vote=%s — %s", step_num, vote, r)
        except Exception as e:
            logging.warning("  [VLM] vote %d failed: %s", i, e)
            votes.append(False)
    done = sum(votes) > len(votes) / 2
    if len(set(votes)) > 1:
        logging.info("  [VLM] step=%d votes=%s -> %s", step_num, votes, done)
    return {"done": done, "reasoning": reasoning, "votes": votes}


def check_primitive_done(
    primitive: str,
    exterior: np.ndarray,
    wrist: np.ndarray,
    num_votes: int = 1,
    step_num: int = 0,
) -> bool:
    """Majority-vote primitive-done check. Returns just the verdict."""
    return check_primitive_done_verbose(primitive, exterior, wrist, num_votes, step_num)["done"]


def check_task_completion(
    task: str,
    before: np.ndarray,
    after: np.ndarray,
    max_tokens: int = 256,
    extra_context: str = "",
) -> dict:
    """Single-shot before/after success oracle for a whole task.

    Compares two RGB frames (gripper-at-home setup, only scene state differs)
    and asks the VLM whether the task was completed. Used to gate skill-gap
    recording — the modular analog of a sim oracle (e.g. ``is_close``).

    ``extra_context``: optional text appended to the prompt — used to feed
    ground-truth sensor readings (e.g. final gripper open/closed position)
    that are hard to judge from the after-image alone. Empty by default
    so existing callers stay unaffected.

    Returns ``{"completed": bool, "reasoning": str, "raw": str}``.
    """
    prompt = (
        "IMAGE 1 = BEFORE the action (gripper at home, scene in initial state).\n"
        "IMAGE 2 = AFTER the action.\n\n"
        f"Task: {task}.\n\n"
    )
    if extra_context:
        prompt += f"Additional sensor readings (ground truth, not visible in images): {extra_context}\n\n"
    prompt += (
        "Question: Was the task completed?\n\n"
        'Respond with valid JSON only:\n'
        '{"completed": true | false,\n'
        ' "reasoning": "<one sentence>"}'
    )
    raw = with_images(prompt, [before, after], max_tokens, TASK_COMPLETION_SYSTEM)
    try:
        data = parse_json(raw)
        completed = bool(data.get("completed", False))
        reasoning = str(data.get("reasoning", ""))
    except Exception as e:
        logging.warning("  [VLM-COMPLETION] parse failed: %s — raw=%r", e, raw)
        completed = False
        reasoning = f"parse error: {e}"
    return {"completed": completed, "reasoning": reasoning, "raw": raw}


# =============================================================================
# Skill-gap helpers (env-agnostic; called by BaseExecutor methods)
# =============================================================================

def generate_new_primitive_action(
    initial_img: np.ndarray,
    current_img: np.ndarray,
    wrist_img: np.ndarray,
    new_primitive: str,
    step: int,
    goal: str = "",
    feedback: str = "",
    system: str = GENERATE_NEW_PRIMITIVE_SYSTEM,
) -> tuple:
    """Generate a single 7-DOF action for a skill-gap (new) primitive.

    Returns ``(action: list[float, len=7], description: str, done: bool)``.
    On VLM error returns ``(None, error_message, False)`` so the caller can
    fall back to the previous action and avoid freezing the robot.

    Pure VLM logic — no environment access required. ``system`` selects the
    prompt variant; subclasses of ``BaseExecutor`` pass their ``GENERATE_PROMPT``
    class attribute.
    """
    goal_context = f"\nOVERALL GOAL: {goal}" if goal else ""
    feedback_context = (
        f"\n\nFEEDBACK: {feedback}\nYour previous approach was NOT working. "
        "You MUST change your action."
        if feedback
        else ""
    )
    prompt = f'NEW PRIMITIVE: "{new_primitive}"{goal_context}\nSTEP: {step}{feedback_context}'

    raw_response = ""
    try:
        raw_response = with_images(
            prompt,
            [initial_img, current_img, wrist_img],
            max_tokens=400,
            system=system,
        )
        data = parse_json(raw_response)
        action = data.get("action", [0] * 7)
        if not isinstance(action, list) or len(action) != 7:
            action = [0] * 7
        else:
            action = [float(x) for x in action]
            # Snap gripper to ±1 (VLM occasionally outputs 0, which would drop
            # objects mid-grasp).
            action[6] = 1.0 if action[6] >= 0 else -1.0
        desc = data.get("goal_analysis", "") or data.get("reasoning", "")
        return action, desc, data.get("done", False)
    except Exception as e:
        _log.error("Failed to generate action for new primitive: %s", e)
        if raw_response:
            _log.error("Raw VLM response (full):\n%s", raw_response)
        return None, str(e), False


def evaluate_new_primitive_progress(
    initial_img: np.ndarray,
    current_img: np.ndarray,
    new_primitive: str,
    goal: str,
    steps_so_far: int,
) -> tuple[bool, str]:
    """Evaluate whether the current trajectory is making progress toward the
    skill-gap goal. Returns ``(making_progress, feedback_string)``.

    On VLM error, returns ``(True, "")`` — fail-open so a transient API blip
    doesn't terminate an otherwise-correct execution.
    """
    prompt = f'GOAL: {goal}\nPRIMITIVE: "{new_primitive}"\nSTEPS SO FAR: {steps_so_far}'
    try:
        data = parse_json(
            with_images(
                prompt,
                [initial_img, current_img],
                max_tokens=500,
                system=EVALUATE_PROGRESS_SYSTEM,
            )
        )
        making_progress = data.get("making_progress", True)
        current_state = data.get("current_state", "")
        goal_state = data.get("goal_state", "")
        gap = data.get("gap", "")
        suggested_axis = data.get("suggested_axis", "")
        suggested_direction = data.get("suggested_direction", "")
        suggested_motion = data.get("suggested_motion", "")

        feedback = f"Current: {current_state}. Goal: {goal_state}. Gap: {gap}."
        if suggested_axis:
            feedback += f" USE AXIS: {suggested_axis}={suggested_direction}."
        feedback += f" Motion: {suggested_motion}"

        if not making_progress:
            _log.info("    [Progress Check] NOT making progress")
            _log.info("    [Progress Check] Current: %s", current_state)
            _log.info("    [Progress Check] Goal: %s", goal_state)
            _log.info("    [Progress Check] Gap: %s", gap)
            _log.info(
                "    [Progress Check] Suggested: %s=%s (%s)",
                suggested_axis, suggested_direction, suggested_motion,
            )
            return False, feedback
        _log.info("    [Progress Check] On track - %s", current_state)
        return True, ""
    except Exception as e:
        _log.error("Failed to evaluate progress: %s", e)
        return True, ""


def plan_task(
    goal: str,
    scene_image: np.ndarray,
    available_primitives: list[str],
    scene_context: str = "tabletop manipulation scene",
    system_template: str | None = None,
    max_tokens: int = 1024,
) -> dict:
    """Decompose a goal into a primitive sequence and identify skill gaps.

    Returns the parsed VLM response: ``{primitive_sequence, step_notes,
    skill_gaps, reasoning, confidence, requires_new_primitive}``. Caller is
    responsible for handling missing/malformed keys.

    ``system_template`` lets callers override the default prompt (defaults to
    ``insight.prompts.PLAN_TASK_SYSTEM``). Sim uses this to honor its
    runtime monkey-patch that swaps between the basic and flywheel variants
    of the planner prompt.
    """
    template = system_template if system_template is not None else PLAN_TASK_SYSTEM
    primitives_str = "\n".join(f"- {p}" for p in available_primitives)
    system = template.format(scene_context=scene_context, primitives=primitives_str)
    raw = with_images(f"GOAL: {goal}", [scene_image], system=system, max_tokens=max_tokens)
    return parse_json(raw)


def parse_signed_magnitude_m(value) -> float | None:
    """Coerce a ``signed_magnitude_m`` field from ``preanalyze_translation`` output
    into a float in meters.

    VLMs sometimes return a bare number, sometimes a string like ``"+0.150"``
    (still fine for ``float()``), and sometimes a string with units (``"50 mm"``,
    ``"0.15 m"``, ``"-150mm"``) which ``float()`` chokes on. Returns ``None`` if
    no number can be extracted.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    s = value.strip()
    try:
        return float(s)
    except ValueError:
        pass
    m = re.search(r"([+-]?\d*\.?\d+)\s*([a-zA-Z]*)", s)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2).lower()
    if unit == "mm":
        return val / 1000.0
    if unit == "cm":
        return val / 100.0
    if unit in ("", "m", "meter", "meters"):
        return val
    return None


def preanalyze_translation(
    primitive: str,
    scene_image: np.ndarray,
    wrist_image: np.ndarray,
    goal: str | None = None,
    plan_context: list[tuple[str, str]] | None = None,
    prior_skill_gap: str | None = None,
    max_tokens: int = 1024,
) -> dict:
    """Decompose a translation skill-gap into (axis, signed magnitude in meters).

    Designed for the xArm hardware skill-gap loop: the VLM looks at the start
    state and proposes a single-axis translation toward the goal. The caller
    converts to xArm units (m → mm), composes with the current pose, and runs
    P-control toward the resulting target.

    ``goal`` is the user-provided task description (e.g., from
    ``--goal``); spliced into the prompt as a ``GOAL:`` line so any
    user-specified hints (target distance, target rotation magnitude, ...)
    reach the VLM. ``plan_context`` is an optional list of
    ``(primitive_name, planner_note)`` tuples describing the surrounding
    plan so the VLM can interpret the current step in context. Both
    follow the sim flywheel's pre-analysis prompt format.

    Returns ``{axis: 'dx'|'dy'|'dz'|'drx'|'dry'|'drz', signed_magnitude_m: float,
    signed_magnitude_deg: float, already_complete: bool, current_state,
    target_state, reasoning}``. Caller picks the right magnitude field based
    on axis (translation vs. rotation) and handles ``already_complete=True``.
    """
    parts: list[str] = []
    if goal:
        parts.append(f"GOAL: {goal}")
    parts.append(f'PRIMITIVE: "{primitive}"')
    if plan_context:
        ctx_lines = [f"  - {name}: {note}" for name, note in plan_context if note]
        if ctx_lines:
            parts.append("PLAN CONTEXT (from planner):\n" + "\n".join(ctx_lines))
    if prior_skill_gap:
        parts.append(f"PRIOR SKILL GAP IN THIS PLAN: {prior_skill_gap}.")
    parts.append("What single-axis motion should the robot perform?")
    prompt = "\n".join(parts)

    raw = with_images(
        prompt,
        [scene_image, wrist_image],
        system=PREANALYZE_TRANSLATION_SYSTEM,
        max_tokens=max_tokens,
    )
    return parse_json(raw)


def preanalyze_motion(
    goal: str,
    scene_image: np.ndarray,
    wrist_image: np.ndarray,
    max_tokens: int = 1024,
) -> dict:
    """Ask the VLM to decompose the required motion (current vs. target state,
    translation or rotation, direction, magnitude) in natural language.

    Returns ``{current_state, target_state, motion_type, direction_nl,
    estimated_magnitude, reasoning}``. The output is descriptive, not
    structured by axis — for axis-tagged output use the LIBERO ``_PREANALYZE_SYSTEM``
    in ``vlm_flywheel/prompts.py`` (which assumes the OSC_POSE convention).
    """
    raw = with_images(f"GOAL: {goal}", [scene_image, wrist_image],
                      system=PREANALYZE_NL_SYSTEM, max_tokens=max_tokens)
    return parse_json(raw)


# =============================================================================
# Higher-level VLM reasoning (env-agnostic)
# =============================================================================

def analyze_execution(
    before: np.ndarray,
    after: np.ndarray,
    primitive: str,
    gripper_state: str | None = None,
    wrist_before: np.ndarray | None = None,
    wrist_after: np.ndarray | None = None,
) -> VLMFeedback:
    """Analyze whether ``primitive`` was executed correctly. Returns ``VLMFeedback``.

    On parse/API failure, returns a ``VLMFeedback`` with ``success=False`` and
    an ``error_type`` of ``"parse_error"`` or ``"api_error"`` so callers can
    distinguish transient failures from genuine primitive failures.
    """
    gripper_info = ""
    if gripper_state:
        gripper_info = f"\nGRIPPER SENSOR: {gripper_state}"
    images = [before, after]
    image_desc = (
        "IMAGE 1 (BEFORE): State before executing\n"
        "IMAGE 2 (AFTER): State after executing"
    )
    if wrist_before is not None and wrist_after is not None:
        images.extend([wrist_before, wrist_after])
        image_desc += (
            "\nIMAGE 3 (BEFORE - wrist): Close-up before"
            "\nIMAGE 4 (AFTER - wrist): Close-up after"
        )
    prompt = f'PRIMITIVE: "{primitive}"\n{image_desc}{gripper_info}'
    try:
        data = parse_json(with_images(prompt, images, system=ANALYZE_EXECUTION_SYSTEM))
        return VLMFeedback(
            success=bool(data.get("success", False)),
            confidence=float(data.get("confidence", 0.5)),
            description=str(data.get("description", "Unable to analyze")),
            error_type=data.get("error_type"),
            correction=data.get("correction"),
            next_primitive=data.get("next_primitive"),
        )
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        _log.error("Failed to parse VLM response: %s", e)
        return VLMFeedback(False, 0.0, "Parse error", error_type="parse_error")
    except Exception as e:
        _log.error("VLM API error: %s", e)
        return VLMFeedback(False, 0.0, f"API error: {e}", error_type="api_error")


def check_goal_achieved(
    goal: str,
    image: np.ndarray,
    max_tokens: int = 4096,
) -> tuple[bool, str]:
    """Single-image goal check. Returns ``(goal_achieved, reasoning)``.

    Caller is responsible for any task-specific cropping/preprocessing of
    ``image`` before passing it in.
    """
    try:
        data = parse_json(with_images(
            f"GOAL: {goal}", [image], max_tokens=max_tokens, system=CHECK_GOAL_SYSTEM,
        ))
        return data.get("goal_achieved", False), data.get("reasoning", "")
    except Exception as e:
        _log.error("Goal check failed: %s", e)
        return False, f"Error: {e}"


def decide_next_primitive(
    goal: str,
    image: np.ndarray,
    history: list[dict] | None,
    available_primitives: list[str],
    scene_context: str = "tabletop manipulation scene",
    max_tokens: int = 4096,
) -> tuple[str | None, str, bool, str | None]:
    """Pick the next primitive to execute given current state + history.

    Returns ``(next_primitive, reasoning, goal_achieved, new_primitive)``.
    ``next_primitive`` is one of ``available_primitives`` or ``None`` when
    stuck; ``new_primitive`` is a free-text description of a missing
    capability when the VLM proposes something not in the list.
    """
    system = DECIDE_NEXT_PRIMITIVE_SYSTEM.format(
        scene_context=scene_context,
        primitives="\n".join(f"  - {p}" for p in available_primitives),
    )
    history_str = ""
    if history:
        history_str = "\n\nPREVIOUS ATTEMPTS:\n"
        for h in history[-5:]:
            history_str += f"- Tried '{h['primitive']}': {h['result']}\n"
    prompt = f"GOAL: {goal}{history_str}"
    try:
        data = parse_json(with_images(prompt, [image], max_tokens=max_tokens, system=system))
        next_prim = data.get("next_primitive")
        new_primitive = data.get("new_primitive")
        if next_prim and next_prim not in available_primitives:
            _log.warning("VLM suggested unknown primitive '%s', treating as new primitive", next_prim)
            new_primitive = new_primitive or next_prim
            next_prim = None
        return (
            next_prim,
            data.get("reasoning", ""),
            data.get("goal_achieved", False),
            new_primitive,
        )
    except (json.JSONDecodeError, KeyError):
        return None, "Parse error", False, None
    except Exception as e:
        _log.error("VLM API error: %s", e)
        return None, f"API error: {e}", False, None


def get_action_correction(
    main_image: np.ndarray,
    wrist_image: np.ndarray,
    primitive: str,
    error: str,
    max_tokens: int = 4096,
) -> ActionCorrection:
    """Ask the VLM for a 7-DOF action delta to recover from an error.

    Returns ``ActionCorrection``. On parse/API failure, returns an aborting
    correction (``should_abort=True``) so the caller stops trying.
    """
    prompt = f'PRIMITIVE: "{primitive}"\nERROR: {error}'
    try:
        data = parse_json(with_images(
            prompt, [main_image, wrist_image],
            max_tokens=max_tokens, system=ACTION_CORRECTION_SYSTEM,
        ))
        action_delta = data.get("action_delta", [0] * 7)
        if not isinstance(action_delta, list) or len(action_delta) != 7:
            _log.warning("Invalid action_delta format: %s", action_delta)
            action_delta = [0] * 7
        else:
            action_delta = [float(x) for x in action_delta]
        return ActionCorrection(
            action_delta=action_delta,
            description=data.get("description", ""),
            confidence=data.get("confidence", 0.5),
            should_abort=data.get("should_abort", False),
            switch_to_primitive=data.get("switch_to_primitive"),
        )
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        _log.error("Failed to parse correction: %s", e)
        return ActionCorrection([0] * 7, str(e), 0.0, should_abort=True)
    except Exception as e:
        _log.error("VLM API error in correction: %s", e)
        return ActionCorrection([0] * 7, f"API error: {e}", 0.0, should_abort=True)


def apply_correction_to_primitive(primitive: str, correction: str, max_tokens: int = 2048) -> str:
    """Rewrite ``primitive`` to incorporate ``correction``. Returns the
    modified primitive string with any wrapping quotes/whitespace stripped."""
    prompt = (
        f'Robot attempted: "{primitive}"\n'
        f'Correction: "{correction}"\n\n'
        "Generate MODIFIED primitive command. Output ONLY the new command."
    )
    return chat([{"role": "user", "content": prompt}], max_tokens=max_tokens).strip().strip('"')


# =============================================================================
# Pre-analysis hint parsing (env-agnostic regex helpers)
# =============================================================================

def parse_axis_direction(hint: str) -> tuple[str | None, str | None]:
    """Extract ``(axis, sign)`` from a pre-analysis hint string.

    Looks for tokens like ``dz=+`` / ``drz=-`` and returns ``("dz", "+")`` etc.
    Returns ``(None, None)`` if no match.
    """
    m = re.search(r"(dr[xyz]|d[xyz])=([+-])", hint)
    if m:
        return m.group(1), m.group(2)
    return None, None


_AXIS_TO_INDEX = {"dx": 0, "dy": 1, "dz": 2, "drx": 3, "dry": 4, "drz": 5}


def build_fixed_action_from_hint(
    hint: str,
    max_rot_cmd: float,
    max_trans_cmd: float,
    n_dof: int = 7,
    grip_value: float = 1.0,
) -> list[float] | None:
    """Build an ``n_dof``-element action with one axis populated from ``hint``.

    Parses tokens like ``dz=+0.05`` or ``drx=-0.3`` and clamps the magnitude
    to the appropriate per-step cap. Returns ``None`` if no parseable token
    is present. Defaults assume the LIBERO/OSC_POSE 7-DOF layout
    ``[dx, dy, dz, drx, dry, drz, grip]``.
    """
    m = re.search(r"(dr[xyz]|d[xyz])=([+-]?[\d.]+)", hint)
    if not m:
        return None
    axis, value = m.group(1), float(m.group(2))
    idx = _AXIS_TO_INDEX.get(axis)
    if idx is None or idx >= n_dof - 1:
        return None
    cap = max_rot_cmd if axis.startswith("dr") else max_trans_cmd
    value = float(np.clip(value, -cap, cap))
    action = [0.0] * n_dof
    action[idx] = value
    action[n_dof - 1] = grip_value
    return action
