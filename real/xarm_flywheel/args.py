"""Args dataclasses for xArm runtime + flywheel + primitive-sequence modes.

``RuntimeArgs`` holds the fields shared by every entry point (connection,
duration resolution, control loop, VLM done-check, output). ``FlywheelArgs``
and ``PrimitivesArgs`` add their mode-specific fields on top.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class RuntimeArgs:
    # ── Connection ──────────────────────────────────────────────────
    host: str = "localhost"
    port: int = 8000
    api_key: str | None = None
    arm_ip: str = "192.168.1.219"

    # ── Duration resolution ─────────────────────────────────────────
    durations_from_dataset: str = ""
    """LeRobot repo_id to pull per-primitive p90 durations from."""

    duration_percentile: int = 90
    fixed_duration: int = 0
    """If >0, use this many steps for every primitive."""

    default_duration: int = 100
    """Fallback duration for primitives missing from the dataset."""

    # ── Outer loop / control ────────────────────────────────────────
    fps: float = 0.0
    """Outer-loop rate. 0 = read from dataset info.json. Must match training rate."""

    replan_steps: int = 10
    """Actions consumed from each policy chunk before re-querying."""

    interp_divisor: float = 1.0
    """Damping in interpolate_action. 1=no damping; higher=smoother/slower."""

    rpy_smoothing_alpha: float = 1.0
    """Slerp factor for commanded orientation when near gimbal lock. 1.0 =
    passthrough (no smoothing), lower values = smoother (each tick slerps
    only ``alpha`` of the way from the filter state toward the policy's
    command in SO(3)). Mitigates gimbal-lock-induced rpy variance: at
    pitch≈±90°, the policy outputs different RPY parameterizations of the
    same physical orientation across consecutive ticks; the IK solver picks
    different joint configurations per tick → visible joint-config flipping.
    Equivalent rpy parameterizations collapse to (essentially) the same
    quaternion, so slerping in quaternion space removes the per-tick
    parameterization variance without distorting genuine rotations. Only
    active when state pitch is within ``90 - rpy_smoothing_pitch_threshold``
    of ±90°; passthrough elsewhere. Try 0.2–0.3 for moderate smoothing on
    side-grasp tasks."""

    rpy_smoothing_pitch_threshold: float = 75.0
    """Absolute pitch (in degrees) above which ``rpy_smoothing_alpha`` becomes
    active. Default 75 means smoothing fires within 15° of ±90° gimbal lock.
    Raise to widen the smoothing zone, lower to narrow it. Has no effect when
    ``rpy_smoothing_alpha`` is 1.0 (passthrough)."""

    use_angle_axis_control: bool = False
    """Send servo commands as axis-angle (rotation vector) via
    ``set_servo_cartesian_aa`` instead of rpy via ``set_servo_cartesian``.
    Mitigates IK joint-config flipping at gimbal lock: equivalent rpy
    parameterizations of the same physical orientation collapse to the same
    quaternion → same rotation vector → consistent IK seed at the controller.
    Per-tick conversion: command rpy → quaternion → rotation vector (axis ×
    angle, radians). xyz/clamp/speed-limit logic is unchanged. Off by default
    for backward compatibility; turn on for side-grasp tasks where pitch
    sustained near ±90°."""

    z_safety_floor: float = 198.0
    """Minimum commanded Z in mm. Always applied; complements workspace_bounds."""

    home_joint_speed_deg_s: float = 12.0
    """Joint speed (deg/s) for return-to-home and oracle-pose drives. Lower
    = smoother / less jerky transitions between trials. xArm default for
    set_servo_angle is high; 12 deg/s is a comfortable "deliberate" pace."""

    tool_offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Tool tip position in EE-local frame (mm) — vector from the TCP origin
    to the actual tool tip, expressed in the gripper's local coordinates.
    [0,0,0] = TCP IS the tool tip (e.g. bare gripper). For an angled scoop or
    custom end-effector, run real/entry/measure_tool_offset.py once to derive
    this vector from a known calibration point."""

    tool_tip_floor_z_mm: float = 0.0
    """World-Z floor enforced on the tool tip via FK. When non-zero AND
    tool_offset_mm is configured, every commanded TCP pose is clamped so the
    tool tip's world-Z stays at or above this value, regardless of EE
    orientation. Replaces the orientation-blind TCP-Z floor for precise-contact
    tasks. 0 = disabled (uses TCP-Z floor only)."""

    workspace_bounds: tuple[float, ...] = ()
    """Cartesian workspace bounds in mm: (x_min, x_max, y_min, y_max, z_min, z_max).
    Empty = disabled. When set, the xArm enables reduced mode (hardware-side
    refusal of out-of-bounds motion) AND interpolate_action clamps every
    commanded pose into the box (software-side first-line defense). Run
    real/entry/measure_bounds.py to determine these for a new workspace."""

    max_tcp_speed_mm_s: float = 0.0
    """Cap end-effector velocity when reduced mode is active. 0 = no override
    (uses xArm default). Only applied when workspace_bounds is non-empty."""

    # ── Auto-advance for known primitives ───────────────────────────
    auto_advance: bool = True
    movement_threshold_mm: float = 1.0
    movement_threshold_deg: float = 0.3
    """Rotation delta below this counts as 'not moving' alongside the
    position threshold. Both must hold for ``stop_steps`` consecutive ticks
    before auto-advance fires. Without the rotation check, twist/pour/flip
    primitives auto-advance immediately because xyz stays put while the
    gripper is rotating. 0.3° leaves room above the recorded twist rate
    (~0.9°/tick at 20Hz from 18 dps) so an active rotation reads as motion;
    non-rotating primitives have rot delta ≈ 0° (well below) and still
    auto-advance normally."""
    stop_steps: int = 15
    min_steps: int = 30

    # ── VLM done-check ──────────────────────────────────────────────
    use_vlm_check: bool = False
    vlm_provider: str = "gemini"
    vlm_check_interval_steps: int = 10
    vlm_warmup_steps: int = 30
    vlm_num_votes: int = 1
    vlm_consecutive_required: int = 3

    z_completion_threshold_mm: float = 220.0
    """For 'move gripper to ...': break primitive when state_z < this. 0 disables."""

    # ── Progress-channel done-check (action[7], 0->1) ──────────────
    use_progress_check: bool = False
    """Enable primitive termination via the model's predicted progress channel (action[7]). Requires the policy to output >=8 dims and to have been trained with progress."""
    progress_threshold: float = 0.95
    """Progress value above which the primitive is considered done. 0.95 leaves a small margin."""
    progress_consecutive_required: int = 3
    """Number of consecutive steps with progress >= threshold required to fire [PROGRESS-DONE]. Filters single-frame spikes."""
    progress_warmup_steps: int = 0
    """Block [PROGRESS-DONE] from firing for the first N steps of a primitive.
    Use to filter early-step progress predictions when the starting state is
    out-of-distribution for the primitive's training data (e.g. the lift after
    a twist starts at an unusual rpy that the policy's progress head
    extrapolates wrong, predicting ~1.0 from step 0 before the arm has moved).
    0 disables (default). Bypassed for gripper-only primitives (close/open
    gripper) since they legitimately finish in a few steps."""

    # ── Output ──────────────────────────────────────────────────────
    save_videos: bool = True
    """Save the annotated per-primitive + combined videos. Default on
    because these are paper-figure friendly and small (~340 K each).
    Set ``--no-save-videos`` to disable all video output entirely."""

    save_debug_videos: bool = False
    """Save the raw per-camera ``debug_exterior.mp4`` / ``debug_wrist.mp4``
    (no composite, no annotation). The annotated per-primitive videos
    already contain the same camera frames, so these are redundant for
    paper / normal eval workflows and disabled by default. Enable when
    deep-diagnosing (e.g. checking whether a frame was even captured)
    or if you want raw footage to pipe into another tool without our
    title bar in the way."""

    high_quality_videos: bool = False
    """Encode all saved videos with near-visually-lossless settings
    (CRF 17, yuv420p, ``slow`` preset) instead of the default libx264
    settings. Roughly 2-3× larger files; pick this when you're producing
    paper figures / supplementary material. Default off for everyday
    eval runs where the smaller default is fine."""
    video_dir: str = "data/primitive_runs"
    run_name: str = ""
    """Subdir name under video_dir. Defaults to a timestamp."""

    experiment_name: str = ""
    """Scope tag for the per-trial outcome log. When set, the
    ``flywheel_trials.jsonl`` is written to
    ``data/primitive_runs/<experiment_name>/`` and each row carries an
    ``experiment_name`` field. This lets multiple runs with the same goal
    (e.g., post-VLA evals of the same primitive across different
    checkpoints) be analyzed in isolation rather than mixing into the
    legacy shared log. Resume-on-Ctrl+C scopes to the experiment too.
    Empty (default) = use the legacy shared path. Has no effect when
    ``record_dataset_repo`` is set (the dataset's own dir is used)."""

    record_dataset_repo: str = ""
    """LeRobot repo name (e.g. 'maggie/xarm_sweep_flywheel'). When set, each
    successful primitive becomes one episode in the dataset; failed primitives
    are discarded. Empty = no dataset recording."""

    use_gripper: bool = True
    """Whether the end-effector is a controllable gripper. When True (default):
    the hardware initializes the gripper at connect (set_gripper_mode +
    set_gripper_enable), the trained policy's gripper command is dispatched
    each tick, and the recorder includes gripper position + per-episode
    progress in the action vector (8D total). When False: all gripper
    interactions are skipped end-to-end — gripper init, command dispatch,
    and recording. Use False for non-gripper end-effectors (e.g., the scoop
    tool) where the parallel-jaw gripper API would error or pollute data."""

    dry_run: bool = False
    """Skip arm + policy connect entirely. Plans against a blank scene, so
    the resulting plan is structural-only — useful for verifying command
    parses, but not for validating planner behavior."""

    plan_only: bool = False
    """Connect to arm + cameras + policy, capture real scene, run real plan,
    log it, exit before any motion. Use this to validate everything works
    end-to-end without the arm moving."""

    require_approval: bool = False
    """Prompt for ENTER before EVERY servo command. Shows current pose, target
    pose, and the delta (mm/deg) so you can see exactly what's about to happen
    before approving. Maximum granularity safety mode — expect hundreds of
    prompts per run. Ctrl+C aborts."""

    confirm_skill_gap: bool = False
    """Prompt for ENTER before each SKILL-GAP primitive starts (one prompt per
    gap, not per step). Known primitives run unprompted — they're already
    trusted by the trained policy. Useful when first piloting a new gap motion
    where you want a heads-up before the unfamiliar P-control trajectory runs.
    Ctrl+C aborts."""

    approval_warmup_steps: int = 0
    """With --require-approval on, skip the per-step prompt for the first N
    steps of each KNOWN primitive (trained-policy execution). Skill-gap motion
    always prompts every step regardless — it's the genuinely novel behavior
    where every command matters. Useful when early known-primitive steps are
    clearly safe (high-altitude approach, etc.) and gating only matters near
    the end. 0 = approve from step 0 (default)."""


@dataclasses.dataclass
class PrimitivesArgs(RuntimeArgs):
    """Run a fixed primitive sequence (no planning, no skill gaps)."""

    primitives: tuple[str, ...] = (
        "move gripper to the rocks",
        "scoop the rocks",
        "lift upward",
    )
    """Ordered primitive prompts to execute. Override per task."""


@dataclasses.dataclass
class FlywheelArgs(RuntimeArgs):
    """Plan a goal into a primitive sequence + skill gaps, then execute."""

    goal: str = ""
    """Natural-language goal for the planner."""

    available_primitives: tuple[str, ...] = ()
    """Primitives the trained policy already knows. Plan steps not in this
    list are flagged as skill gaps."""

    scene_context: str = "Tabletop manipulation setup with a 6-DOF xArm 6 robot."
    """Brief scene description fed to the planner."""

    skill_gap_max_steps: int = 200
    """Hard cap on outer-loop steps for a single skill gap."""

    skill_gap_max_extends: int = 3
    """Times the P-controller will extend the target along the same axis."""

    skill_gap_rotation_speed: float = 100.0
    """xArm planned-motion linear speed for rotation skill gaps (mm/s).
    Defaults to 100. Lower for delicate motions where overshoot/jerk could
    spill held contents (e.g. pour: 30-50). Higher only if the rotation
    feels sluggish."""

    skill_gap_rotation_mvacc: float = 1000.0
    """xArm planned-motion acceleration for rotation skill gaps (mm/s²).
    Defaults to 1000. Lower for smoother starts/stops (e.g. pour: 200-400)."""

    num_runs: int = 1
    """Number of plan+execute trials to run back-to-back. Used to bootstrap a
    new primitive's training set — pause for a manual scene reset between
    trials, oracle gates whether each trial's recording is committed."""

    target_successes: int = 0
    """Stop the batch once this many oracle-confirmed successes are committed.
    0 = run all ``num_runs`` trials regardless. Set to (e.g.) 50 to collect a
    target episode count and exit early when reached."""

    manual_verdict: bool = True
    """After the oracle's verdict prints, prompt the user to confirm or
    override before committing the recording. ENTER accepts the oracle's
    verdict; 'y' forces success (commit), 'n' forces failure (discard).
    Set to False for fully-autonomous batch runs that trust the oracle.
    Ignored when ``use_oracle=False`` (the human is the sole judge so
    we always prompt)."""

    use_oracle: bool = True
    """Run the VLM success oracle. When True (default): capture the
    before + after frames, call ``insight.check_task_completion`` to get
    an automated verdict, then optionally let the user override per
    ``manual_verdict``. When False: skip the VLM call entirely and
    prompt the user for a y/n verdict directly — useful when the oracle
    is unreliable for a task or when you want to avoid VLM latency /
    token spend on cheap human-in-the-loop runs. The return-to-home
    (or oracle-pose) drive and the after-frame capture still happen so
    saved oracle_frames/ keeps debug images. Trial reason is logged as
    ``human-only: ...`` so paper analysis can distinguish oracle-judged
    vs human-judged trials."""

    confirm_plan: bool = False
    """After the planner emits a primitive sequence, prompt before any motion.
    ENTER = proceed; 'n' = mark this trial as a failure and advance to the
    next trial without running. Useful for batch collection where the planner
    occasionally picks the wrong axis or misses a step — reject the plan
    quickly so robot time isn't wasted on a doomed run. Rejected trials are
    logged with reason 'user-rejected plan' so paper analysis can split
    plan-quality failures from execution failures. Plan stats (plan_length,
    skill_gaps) are captured before the prompt, so the rejected trial still
    shows in flywheel_trials.jsonl with full plan info."""

    record_skill_gap_only: bool = True
    """Record only skill-gap primitives, not the known-primitive trajectories
    in the plan. When True (default), the dataset contains just the new
    skills being acquired. When False, the dataset also includes the
    LEAD-IN known primitives — the ones executed BEFORE the first skill
    gap in the plan (e.g., move-to-bottle, close-gripper). Post-gap
    primitives (e.g., lift, open) are still skipped because they're
    already in the trained policy's repertoire and don't depend on the
    new primitive's setup. Atomic commit/discard semantics still hold
    either way: if the plan's oracle verdict is failure, nothing commits."""

    return_home_for_oracle: bool = True
    """Drive to canonical home before capturing the post-plan ``after`` frame
    for the success oracle. Default True. Set False for tasks that finish in
    a configuration where driving home would disturb the scene (e.g. pour
    leaves the gripper tilted over the bowl — driving home would dump the
    bottle). When False, the oracle's after-frame is captured from wherever
    the plan finished, and the user resets the scene between trials so the
    next trial-start drives home safely."""

    oracle_pose: tuple[float, ...] = (-12.02, 0.20, -66.56, -0.64, 65.95, 168.92)
    """6-tuple of JOINT ANGLES in degrees — drive the arm here via
    ``set_servo_angle`` BEFORE capturing the oracle's after-frame, instead of
    canonical home. Joint-space (not Cartesian) targeting matches how
    ``return_to_home`` drives so we avoid the wrist-branch ambiguity that
    ``set_position`` hits at ±180° yaw (could over-rotate the long way
    around). Use a pose where the arm is fully out of the workspace camera's
    view so the oracle can see the scene unobstructed (e.g. capped vs
    uncapped bottle). Default is a measured "out of view" pose for the
    current xArm-6 + camera setup; measure your own with ``arm.get_servo_angle()``.
    Empty tuple = fall back to canonical home. Overrides
    ``return_home_for_oracle`` when non-empty."""
