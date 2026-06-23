"""Real-world flywheel-style execution on the xArm.

Plans a natural-language goal into a primitive sequence + skill gaps using a
VLM, then runs the resulting plan on hardware: trained VLA policy for known
primitives, VLM pre-analysis + P-control for translation skill gaps.

Implementation lives in ``real/xarm_flywheel/`` — this file is the tyro entry
shim so existing invocations keep working.

Usage:
    uv run real/entry/run_flywheel.py \\
        --goal "close the open top drawer" \\
        --available-primitives \\
            "move gripper to the top drawer handle" \\
            "close gripper" \\
            "open gripper" \\
        --scene-context "Kitchen cabinet with a top drawer in the open position" \\
        --use-vlm-check
"""

from __future__ import annotations

import tyro

from real.xarm_flywheel import FlywheelArgs, run


if __name__ == "__main__":
    run(tyro.cli(FlywheelArgs))
