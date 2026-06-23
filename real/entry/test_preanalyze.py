"""Offline harness for the VLM motion pre-analysis.

Loads start frames from exterior + wrist mp4s, asks Gemini to analyze the
required motion in natural language (current state, target state, direction,
magnitude) for a given goal. No arm or policy involved — useful for validating
that the VLM's motion descriptions are sensible before deploying to hardware.

Note: this uses the natural-language pre-analysis prompt, not the LIBERO
axis-tagged one. The xArm has different coordinate conventions than the sim,
so axis-tagged output (drx/dry/drz lookup) would be misleading without an
xArm-specific axis chart. Output is descriptive only.

Usage:
    uv run real/entry/test_preanalyze.py \\
        --goal "lower the gripper until the scoop touches the rocks" \\
        --ext-mp4 data/example_videos/scooping_all_primitives_exterior.mp4 \\
        --wrist-mp4 data/example_videos/scooping_all_primitives_wrist.mp4
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

import imageio.v2 as imageio
import tyro

import vlm_check  # noqa: F401  (import registers xArm research context with insight)
from insight.reasoning import preanalyze_motion


@dataclasses.dataclass
class Args:
    goal: str = "lower the gripper until the scoop touches the rocks"
    """Natural-language description of the desired motion."""
    ext_mp4: Path = Path("data/example_videos/scooping_all_primitives_exterior.mp4")
    wrist_mp4: Path = Path("data/example_videos/scooping_all_primitives_wrist.mp4")
    frame: int = 0
    """Frame index from both videos to use as the start state."""
    provider: str = "gemini"


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    vlm_check.set_provider(args.provider)

    reader_ext = imageio.get_reader(str(args.ext_mp4))
    reader_wrist = imageio.get_reader(str(args.wrist_mp4))
    try:
        ext = reader_ext.get_data(args.frame)
        wrist = reader_wrist.get_data(args.frame)
    finally:
        reader_ext.close()
        reader_wrist.close()

    print(f"\nGOAL: {args.goal}")
    print(f"EXT:   {args.ext_mp4.name} #{args.frame}")
    print(f"WRIST: {args.wrist_mp4.name} #{args.frame}\n")

    result = preanalyze_motion(args.goal, ext, wrist)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Args))
