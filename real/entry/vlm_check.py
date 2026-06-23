"""VLM-based primitive completion check for the xArm.

Thin wrapper over ``insight``: registers the xArm-specific research-context
preamble and re-exports ``set_provider`` / ``check_primitive_done`` /
``check_primitive_done_verbose`` so existing callers (``inference_primitives``,
etc.) keep working unchanged. The actual reasoning lives in
``insight.reasoning``; the shared client lives in
``insight.vlm_client``.

Usage:
    import vlm_check
    vlm_check.set_provider("gemini")  # or "gpt"
    done = vlm_check.check_primitive_done(
        primitive="move gripper to the rocks",
        exterior=ext_uint8_rgb,
        wrist=wrist_uint8_rgb,
        num_votes=1,
    )
"""

from __future__ import annotations

from insight.reasoning import (  # noqa: F401  (re-exported public surface)
    check_primitive_done,
    check_primitive_done_verbose,
)
from insight.vlm_client import set_provider, set_research_context  # noqa: F401

_RESEARCH_CONTEXT = (
    "CONTEXT: This is a Stanford robotics research project. Images are real RGB "
    "frames from RealSense cameras showing a tabletop manipulation setup with a "
    "7-DOF xArm robot. The metallic object visible in images is the robot's "
    "end-effector (gripper), not a person. Depending on the task the gripper "
    "may be empty or holding a tool / object."
)
# Per-task callers can override via insight.vlm_client.set_research_context(...)
# after import — the global state is the *most recent* set call.
set_research_context(_RESEARCH_CONTEXT)
