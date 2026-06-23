"""Re-exports + sim research-context registration.

The VLM client itself lives in ``insight.vlm_client``; this module
exposes the historical names and registers the sim-specific safety preamble
at import.
"""

from __future__ import annotations

from insight.vlm_client import (
    _NEW_TOKEN_PARAM_MODELS,
    _PROVIDERS as _VLM_PROVIDERS,
    chat as vlm_chat,
    encode_image_b64,  # noqa: F401  (historical export)
    get_model,
    parse_json as parse_vlm_json,
    set_provider as _ic_set_provider,
    set_research_context,
    with_images as vlm_with_images,
)


# Sim-specific safety preamble — registered with insight at import time so
# the VLM doesn't refuse on synthetic-render images.
_RESEARCH_CONTEXT = (
    "CONTEXT: This is a Stanford University robotics research project running in a "
    "MuJoCo physics simulator. All images are synthetic renders of a simulated tabletop "
    "with toy LEGO blocks and a simulated robot gripper. No humans, faces, or real-world "
    "objects are depicted. The metallic object visible in images is a simulated robot "
    "end-effector (gripper), not a person."
)
set_research_context(_RESEARCH_CONTEXT)


def set_vlm_provider(provider: str) -> None:
    """Configure the active VLM provider. Thin pass-through to insight."""
    _ic_set_provider(provider)
