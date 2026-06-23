"""All VLM prompt strings used by the flywheel pipeline."""

# =============================================================================
# Axis conventions (shared by several prompts)
# =============================================================================

_GRASP_TOLERANCE = """\
Flag position_ok=false if the gripper needs adjustment to achieve a reliable grasp \
on the object. Consider both XY centering and Z height."""

_AXIS_CONVENTIONS = """\
World frame (Scene Camera — views the robot from across the table):
- +X = toward camera (DOWN on screen)
- +Y = RIGHT on screen
- +Z = UP (away from table)

Wrist camera (mounted on gripper, looking down at the object):
- TOP of wrist image = +X (toward scene camera)
- LEFT of wrist image = +Y (rightward in scene view)

Translations (WORLD):
  +dx = +X (toward camera), +dy = +Y (right), +dz = +Z (up)

Rotations are in the WORLD frame (OSC_POSE), right-hand rule about the named +axis.
  drx = rotation about +X, dry = rotation about +Y, drz = rotation about +Z

ROTATION LOOKUP TABLE — find your (current → target) pair, read the command:
  +X → +Y : +drz    +X → -Y : -drz    +X → +Z : -dry    +X → -Z : +dry
  -X → +Y : -drz    -X → -Y : +drz    -X → +Z : +dry    -X → -Z : -dry
  +Y → +X : -drz    +Y → -X : +drz    +Y → +Z : +drx    +Y → -Z : -drx
  -Y → +X : +drz    -Y → -X : -drz    -Y → +Z : -drx    -Y → -Z : +drx
  +Z → +X : +dry    +Z → -X : -dry    +Z → +Y : -drx    +Z → -Y : +drx
  -Z → +X : -dry    -Z → -X : +dry    -Z → +Y : +drx    -Z → -Y : -drx"""


# =============================================================================
# Base prompts — sourced from insight (shared with xArm pipeline)
# =============================================================================
# Local sim-only prompt: ``_PLAN_TASK_SYSTEM`` is the simple legacy planner
# variant used outside flywheel mode; the flywheel planner uses
# ``_PLAN_TASK_SYSTEM_FLYWHEEL`` (also re-exported from insight below).

from insight.prompts import (
    ACTION_CORRECTION_SYSTEM as _ACTION_CORRECTION_SYSTEM,
    ANALYZE_EXECUTION_SYSTEM as _ANALYZE_EXECUTION_SYSTEM,
    CHECK_GOAL_SYSTEM as _CHECK_GOAL_SYSTEM,
    DECIDE_NEXT_PRIMITIVE_SYSTEM as _DECIDE_NEXT_PRIMITIVE_SYSTEM,
    EVALUATE_PROGRESS_SYSTEM as _EVALUATE_PROGRESS_SYSTEM,
    GENERATE_NEW_PRIMITIVE_SYSTEM as _GENERATE_NEW_PRIMITIVE_SYSTEM,
)


_PLAN_TASK_SYSTEM = """You are a robot task planner. Scene: {scene_context}

AVAILABLE PRIMITIVES:
{primitives}

Break down the goal into a complete sequence of steps. Include ALL steps needed,
even if some require capabilities not in the available list.

Put every step in primitive_sequence, including new primitives that don't exist yet.
The skill_gaps list should name which steps in the sequence are new.

Example: if you need to push an object sideways (not in available list), include it:
primitive_sequence: ["move gripper to object", "push object left", "open gripper"]
skill_gaps: ["push object left"]

Respond with ONLY valid JSON:
{{"primitive_sequence": ["step1", "step2", ...],
  "skill_gaps": ["name of each new capability"],
  "reasoning": "explanation",
  "confidence": 0.0-1.0,
  "requires_new_primitive": true or false}}"""


# =============================================================================
# Second-flip prompts
# =============================================================================

# The flywheel-mode planner prompt is shared with the real-world xArm pipeline
# via insight. Sim's basic _PLAN_TASK_SYSTEM (line 50) stays local — it's
# the simpler legacy variant used outside flywheel mode.
from insight.prompts import PLAN_TASK_SYSTEM as _PLAN_TASK_SYSTEM_FLYWHEEL  # noqa: E402


_GENERATE_NEW_PRIMITIVE_SYSTEM_FLYWHEEL = f"""\
You control a robot to perform a NEW primitive via delta actions.

Focus on the OVERALL GOAL, not just the primitive name.
Ask yourself: what physical motion achieves the GOAL?

IMAGE 1: Initial state (before this primitive started)
IMAGE 2: Current state (now)
Compare IMAGE 1 vs IMAGE 2 to judge progress. Set done=true when complete.

{_AXIS_CONVENTIONS}

GRIPPER (last value):
- +1.0 = CLOSE/HOLD - use this to maintain grip on object
- -1.0 = OPEN/RELEASE
- If holding an object, you MUST use +1.0 or it will drop!

Actions: [dx, dy, dz, drx, dry, drz, gripper] (max +/-0.03m, +/-0.15rad per step)
Set done=true ONLY when the GOAL is visibly achieved in the image.

Respond with ONLY valid JSON:
{{"goal_analysis": "what motion achieves the GOAL?",
  "action": [dx, dy, dz, drx, dry, drz, grip],
  "done": true or false}}"""


_POSITION_CHECK_SYSTEM = f"""\
You evaluate whether a robot gripper is well-positioned to achieve a goal.

You receive two images:
- IMAGE 1: Scene view — shows the gripper fingers AND the object. Use this to judge
  whether the fingers are aligned with the object for a stable grasp.
- IMAGE 2: Wrist camera — close-up from the gripper looking down. Use this to see
  the object's shape and orientation in detail.

Check whether closing the fingers at this angle will produce a stable grasp on the
object's body. Consider the object's shape, the overall goal, and the planned
subsequent action.

{_GRASP_TOLERANCE}

{_AXIS_CONVENTIONS}

If correction is needed, specify a CONCRETE action (combine if multiple):
- For rotation: "rotate drz by +X degrees" or "rotate drz by -X degrees"
- For position: "move dx by +/-Xmm, dy by +/-Xmm, dz by +/-Xmm"

Respond with ONLY valid JSON:
{{"position_ok": true or false, "orientation_ok": true or false, "correction": "concrete action or empty string", "reasoning": "brief explanation"}}"""

_POSITION_CHECK_SYSTEM_POS_ONLY = f"""\
You evaluate whether a robot gripper's XY position and Z height are correct for grasping.

The gripper orientation is ALREADY SET and cannot be changed. Do NOT evaluate or comment on
the finger angle/rotation. Only check:
1. Is the gripper centered over the object? (XY position)
2. Is the height correct for closing the fingers on the object? (Z height)

IMPORTANT: The wrist camera is mounted on the gripper and ROTATES with it. After orientation
changes, the wrist camera axes (top/left/right) NO LONGER correspond to world X/Y. Do NOT
use the wrist camera to judge XY centering — use IMAGE 1 (scene view) for that, since its
axes are fixed. You may use the wrist camera to judge Z height (whether the object looks
close enough to grasp).

{_GRASP_TOLERANCE}

You receive two images:
- IMAGE 1: Scene view — shows the gripper and object from across the table. Use for XY position.
- IMAGE 2: Wrist camera — close-up from the gripper looking down. Use for Z height only.

{_AXIS_CONVENTIONS}

If the position needs correction, specify ONLY translation:
- "move dx by +/-Xmm, dy by +/-Xmm, dz by +/-Xmm"

Respond with ONLY valid JSON:
{{"position_ok": true or false, "orientation_ok": true, "correction": "concrete action or empty string", "reasoning": "brief explanation"}}"""


_PREANALYZE_SYSTEM = f"""\
Determine the robot motion needed to achieve the GOAL.
Focus on the GOAL and the images — ignore the primitive name.

You receive two images:
- IMAGE 1 (scene overview): Shows the full scene from across the table.
- IMAGE 2 (wrist camera): Close-up from the gripper looking down.

{_AXIS_CONVENTIONS}

THINK STEP BY STEP:
1) CURRENT STATE: Describe what you see in the images.
   - For objects: Where is it? What orientation is it in? Where is any distinguishing feature (peg, handle, lid) pointing?
   - For articulated objects (drawers, doors): Is it open or closed? Which direction does it slide/swing?
2) TARGET STATE: What does the goal require? Describe the desired position/orientation.
3) MOTION TYPE: Is this a ROTATION (object needs to turn/flip) or TRANSLATION (object/mechanism needs to slide/push/pull)?
4) AXIS AND DIRECTION:
   - For TRANSLATION: Which axis (dx, dy, dz) moves toward the target? Carefully check the scene image:
     On screen: down = +X (toward camera), up = -X (away), right = +Y, left = -Y, up from table = +Z
   - For ROTATION: Which axis (drx, dry, drz) turns toward the target?
     Use the lookup table to find the correct axis and sign from (current → target) axis mapping.
5) MAGNITUDE: Estimate distance in meters for translation (e.g., 0.150 = 150mm), or degrees for rotation.
6) GRASP ORIENTATION: If the robot needs to grasp the object for rotation, what gripper yaw (drz degrees)
   would give the most stable grip? (0 if not applicable)

Respond with ONLY valid JSON:
{{"current_state": "what you observe",
  "target_state": "what the goal requires",
  "needed_motion": "translate -dy or rotate drx->+Z etc",
  "axis": "dx or dy or dz or drx or dry or drz",
  "direction": "+0.150 or -0.150 (signed, with magnitude in meters or radians)",
  "estimated_rotation_deg": 0,
  "recommended_grasp_drz_deg": 0,
  "reasoning": "why this axis and direction achieves the goal"}}"""


_EVALUATE_PROGRESS_SYSTEM_V2 = f"""\
You evaluate a robot's progress on a NEW skill.

IMAGE 1: Initial state (before starting)
IMAGE 2: Current state
Compare the two images carefully.

CRITICAL: "making_progress" means moving TOWARD the goal, not just that motion
is occurring. If the object is moving AWAY from the goal, set making_progress=false
and suggest reversing the direction.

Step-by-step reasoning:
1. CURRENT STATE: Describe the object's orientation and position in IMAGE 2
2. GOAL STATE: What should it look like?
3. GAP: Is it closer to or further from the goal vs IMAGE 1? Note any position drift (height, lateral).
4. MOTION: What axis and direction (+/-) moves toward the goal? If there is also position drift, suggest a secondary correction.
5. COMPLETE: Has the goal been FULLY achieved? Be strict — partial progress is NOT complete.

{_AXIS_CONVENTIONS}

Respond with ONLY valid JSON:
{{"current_state": "describe object orientation and position",
  "goal_state": "describe target",
  "gap": "closer or further from goal?",
  "suggested_axis": "drx, dry, drz, dx, dy, or dz",
  "suggested_direction": "+0.150 or -0.150 (rotation) or +0.030 or -0.030 (translation)",
  "suggested_motion": "specific motion",
  "position_correction": "none, or e.g. dz=-0.030",
  "making_progress": true or false,
  "goal_complete": true or false}}"""


# =============================================================================
# Curation prompts
# =============================================================================

_COMPARE_SYSTEM = """You are evaluating robot manipulation trajectory quality.
You will see sequential frames from two trajectories of the same task.
Each frame is labeled with its position in the sequence (e.g. "Frame 3/12").
The frames are in chronological order — earlier frames show the start, later frames show the end.

Compare the two trajectories and pick the better one.

Evaluate on:
1. SMOOTHNESS: Does the motion flow continuously, or are there jerky jumps between frames?
2. STABILITY: Is the grasp firm throughout? Any signs of slippage, wobble, or near-drops?
3. EFFICIENCY: Did it achieve the goal with less wasted motion and fewer steps?
4. FINAL STATE: Is the block cleanly upright at the end?

Respond with ONLY valid JSON:
{"winner": "A" or "B", "confidence": 0.5-1.0, "reasoning": "brief explanation"}"""


_SUMMARIZE_SYSTEM = """You are analyzing robot trajectory quality patterns.
You have evaluated many pairs of robot manipulation trajectories performing the same task.
Below are your pairwise comparison notes explaining why one trajectory was better than another.

Distill these observations into actionable lessons for improving future trajectory collection.
Focus on concrete, specific patterns — not generic advice.

Respond with ONLY valid JSON:
{"patterns": ["specific pattern 1", "specific pattern 2", ...],
 "tips_for_action_generation": "concrete advice for the VLM that generates robot actions",
 "tips_for_grasp_strategy": "concrete advice about how to grasp the object",
 "common_failure_modes": ["failure mode 1", "failure mode 2", ...]}"""
