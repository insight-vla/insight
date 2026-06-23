"""Run a fixed primitive sequence on the xArm — no planning, no skill gaps.

Sibling to ``run_flywheel.py``. Where flywheel mode VLM-plans a goal into a
sequence, this script takes the sequence directly via ``--primitives``. Same
hardware lifecycle, same trained-policy loop, same VLM-done-check, just no
planning.

Usage:
    uv run real/entry/run_primitives.py \\
        --primitives \\
            "move gripper to the rocks" \\
            "scoop the rocks" \\
            "lift upward" \\
        --durations-from-dataset maggie/xarm_scoop_100_primitives_trimmed
"""

from __future__ import annotations

import tyro

from real.xarm_flywheel import PrimitivesArgs, run_primitives


if __name__ == "__main__":
    run_primitives(tyro.cli(PrimitivesArgs))
