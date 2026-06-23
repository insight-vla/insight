"""Offline harness for the VLM task planner.

Loads the first frame of an exterior camera mp4, asks Gemini to decompose a goal
into a primitive sequence + skill gaps given the xArm's primitive vocabulary,
and prints the result. No arm or policy involved — useful for validating that
the VLM's plans are sensible before deploying to hardware.

Usage:
    uv run real/entry/test_planner.py --goal "scoop up the rocks and lift" \\
        --video data/example_videos/scooping_all_primitives_exterior.mp4
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

import imageio.v2 as imageio
import tyro

import vlm_check  # noqa: F401  (import registers xArm research context with insight)
from insight.reasoning import plan_task

# xArm primitive vocabulary (mirrors the inference_primitives.py defaults plus
# the verbs Gemini might propose for sweep). Override via --primitives.
DEFAULT_PRIMITIVES = (
    "move gripper to the rocks",
    "move gripper to the target",
    "close gripper",
    "open gripper",
    "lift upward",
    "lower gripper",
    "scoop the rocks",
    "sweep the rocks",
)


@dataclasses.dataclass
class Args:
    goal: str = "scoop up the rocks from the bin and lift them clear of the surface"
    """Natural-language description of the task to plan for."""
    video: Path = Path("data/example_videos/scooping_all_primitives_exterior.mp4")
    """Source mp4 — first frame is sent to the VLM."""
    frame: int = 0
    """Frame index to extract from the video (default: first frame)."""
    provider: str = "gemini"
    primitives: tuple[str, ...] = DEFAULT_PRIMITIVES
    scene_context: str = (
        "Tabletop manipulation setup with a 7-DOF xArm robot. Override with "
        "--scene-context \"...\" to describe the specific objects and surface "
        "(e.g. 'bin of rocks on white acrylic', 'a kitchen drawer with a handle', "
        "'a small block on a flat surface')."
    )


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    vlm_check.set_provider(args.provider)

    reader = imageio.get_reader(str(args.video))
    try:
        scene = reader.get_data(args.frame)
    finally:
        reader.close()

    print(f"\nGOAL: {args.goal}")
    print(f"PRIMITIVES: {list(args.primitives)}")
    print(f"FRAME: {args.video.name} #{args.frame}\n")

    result = plan_task(args.goal, scene, list(args.primitives), args.scene_context)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Args))
