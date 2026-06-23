"""VLM flywheel package — public re-exports.

Behavioral state lives on the executor classes
(``LiberoSimExecutor``/``LiberoFlywheelExecutor``); VLM-client state lives in
``insight.vlm_client``; display flag lives on ``env._display_enabled``.
"""

from __future__ import annotations

# ====================================================================
# Re-exports from sub-modules
# ====================================================================

# -- prompts (all prompt strings) ------------------------------------
from .prompts import (                                    # noqa: F401,E402
    _AXIS_CONVENTIONS,
    _ANALYZE_EXECUTION_SYSTEM,
    _PLAN_TASK_SYSTEM,
    _CHECK_GOAL_SYSTEM,
    _DECIDE_NEXT_PRIMITIVE_SYSTEM,
    _ACTION_CORRECTION_SYSTEM,
    _GENERATE_NEW_PRIMITIVE_SYSTEM,
    _EVALUATE_PROGRESS_SYSTEM,
    _PLAN_TASK_SYSTEM_FLYWHEEL,
    _GENERATE_NEW_PRIMITIVE_SYSTEM_FLYWHEEL,
    _POSITION_CHECK_SYSTEM,
    _POSITION_CHECK_SYSTEM_POS_ONLY,
    _PREANALYZE_SYSTEM,
    _EVALUATE_PROGRESS_SYSTEM_V2,
    _COMPARE_SYSTEM,
    _SUMMARIZE_SYSTEM,
)

# -- vlm (client infrastructure) ------------------------------------
from .vlm import (                                        # noqa: F401,E402
    set_vlm_provider, vlm_chat,
    vlm_with_images, parse_vlm_json,
    _VLM_PROVIDERS, _NEW_TOKEN_PARAM_MODELS, _RESEARCH_CONTEXT,
)

# -- env (environment, images, dataclasses, keyboard) ----------------
from .env import (                                        # noqa: F401,E402
    # Constants
    LIBERO_ENV_RESOLUTION,
    POLICY_IMAGE_SIZE, VLM_IMAGE_SIZE, VLM_IMAGE_SIZE_SMALL,
    DEFAULT_PRIMITIVE_STEPS, PRIMITIVE_P90_DURATIONS,
    AVAILABLE_PRIMITIVES, DEFAULT_SEQUENCE, DEFAULT_SCENE_CONTEXT,
    # Dataclasses
    VLMFeedback, TaskPlan, ActionCorrection, FlywheelDatapoint, Args,
    # Keyboard / stop
    _stop_event, _start_keyboard_listener, stop_requested,
    # Env helpers
    SimpleVisualizationWrapper, create_env, settle_physics,
    _find_robot, reset_gripper_pose,
    # Image helpers
    get_obs_images, resize_for_policy, resize_for_vlm,
    crop_around_red_block, encode_image_base64, get_gripper_state,
)

# -- reasoning -------------------------------------------------------
from .reasoning import (                                  # noqa: F401,E402
    analyze_execution, plan_task, check_goal_achieved,
    decide_next_primitive, get_action_correction,
    apply_correction_to_primitive,
    _get_peg_direction_vector, _get_peg_direction_text,
    _prepare_preanalysis_images, _preanalyze_skill_gap,
    _evaluate_gripper_position, _check_goal_achieved_sim,
    _parse_verified_axis_direction, _AXIS_TO_INDEX,
    _build_fixed_action_from_hint,
    _evaluate_progress_sim, _evaluate_progress_v2,
    _generate_action_wrapper,
)

# -- control ---------------------------------------------------------
from .control import (                                    # noqa: F401,E402
    _quat_multiply, _find_red_block_joint,
    _quat_to_euler_rad, _quat_to_euler_deg,
    _get_ee_euler_rad, _angle_diff, _quat_rotation_angle,
    _get_ee_quat_wxyz, _quat_conjugate, _quat_from_axis_angle,
    _quat_error_to_world_rotvec, _build_target_quat,
    _compute_peg_correction_quat,
    tilt_red_block_to_side,
    _K_P_ROT, _K_P_TRANS, _MAX_ROT_CMD, _MAX_TRANS_CMD, _MAX_POS_HOLD_CMD,
    _project_3d_to_pixel, _get_axis_info,
    _draw_axes_on_bgr, _draw_world_axes_only, _draw_indicators,
    _detect_red_mask_hsv, _crop_around_block_hsv,
    _run_vla_and_record, _replay_vla_with_rotation,
    _apply_concentrated_translation, _replay_with_correction,
    _parse_correction, _return_to_ee_pose,
)

# -- recording -------------------------------------------------------
from .recording import (                                  # noqa: F401,E402
    _make_recording_step, _save_raw_hdf5,
)

# -- base_execution --------------------------------------------------
# Helpers + entry points + the executor class. Plan-execute methods
# (execute_step / execute_plan / run_adaptive / run_new_primitive_with_vlm)
# live on LiberoSimExecutor; only env-agnostic helpers and standalone
# legacy modes are re-exported as module-level functions.
from .base_execution import (                             # noqa: F401,E402
    LiberoSimExecutor,
    run_primitive, run_primitive_with_checkpoints,
    _save_adaptive_results,
    run_single, run_sequence, run_with_retry, run_flywheel,
    _main,
)

# -- curation --------------------------------------------------------
from .curation import (                                   # noqa: F401,E402
    extract_frames, compare_trajectories, summarize_feedback,
    curate_batch, CurateArgs,
)

# -- flywheel execution ----------------------------------------------
# Public surface: the executor class (which owns ``run_flywheel_adaptive`` as
# a method) and the args dataclass. Module-level state and helpers were
# removed in favor of instance methods + state.
from .flywheel_execution import (                         # noqa: F401,E402
    LiberoFlywheelExecutor, FlywheelArgs,
)
