"""Interactive workspace-bounds measurement for the xArm.

Walks you through 6 axis-extreme positions in teach mode (you hand-guide the
arm), reads the pose at each, and prints the resulting bounds tuple ready to
paste into ``--workspace-bounds`` for ``run_flywheel.py`` / ``run_primitives.py``.

A safety margin (default 10mm) is subtracted inward on each face so the bound
is reached *before* the arm could actually hit anything.

Usage:
    uv run real/entry/measure_bounds.py
    uv run real/entry/measure_bounds.py --margin-mm 15 --output bounds.json
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from pathlib import Path

import numpy as np
import tyro
from xarm.wrapper import XArmAPI


_PROMPTS = [
    ("x_min", "MIN X (closest to robot base / back of workspace)"),
    ("x_max", "MAX X (farthest from base / front of workspace)"),
    ("y_min", "MIN Y (right side from robot's view)"),
    ("y_max", "MAX Y (left side from robot's view)"),
    ("z_min", "MIN Z (lowest safe height — just above acrylic, with clearance)"),
    ("z_max", "MAX Z (highest safe height — well clear of any structure above)"),
]


@dataclasses.dataclass
class Args:
    arm_ip: str = "192.168.1.219"
    margin_mm: float = 10.0
    """Safety margin shrunk inward on each face. 0 = use raw measurements."""

    output: str = ""
    """If set, write the bounds + raw measurements as JSON to this path."""


def _read_pose(arm: XArmAPI) -> np.ndarray:
    pose = list(arm.get_position()[1])
    return np.array(pose, dtype=np.float32)


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    arm = XArmAPI(args.arm_ip)
    arm.connect()
    if arm.get_state() != 0:
        arm.clean_error()
        time.sleep(0.5)
    arm.motion_enable(enable=True)

    # Teach mode (mode=2) lets you hand-guide the arm with no servo resistance.
    print("\nEntering TEACH MODE — you can now physically push the arm.")
    print("Brakes are released; support the arm if needed when first switching modes.\n")
    arm.set_mode(2)
    arm.set_state(0)
    time.sleep(0.5)

    raw: dict[str, list[float]] = {}
    try:
        for axis, description in _PROMPTS:
            input(f"\n>>> Move arm to {axis.upper()} extreme: {description}.\n"
                  f"    Press ENTER when the arm is in position... ")
            pose = _read_pose(arm)
            raw[axis] = pose.tolist()
            print(f"    Recorded pose: x=%.1f y=%.1f z=%.1f rpy=(%.1f, %.1f, %.1f)"
                  % tuple(pose))
    finally:
        # Restore normal position-control mode before disconnect so the arm
        # is left in a state ready for the next script.
        print("\nRestoring position-control mode...")
        arm.set_mode(0)
        arm.set_state(0)
        time.sleep(0.5)
        arm.disconnect()

    # Take the conservative box from the measurements: each axis-extreme prompt
    # constrains one face, but the user might have moved off-axis too — so for
    # safety, we use the most-conservative value across all six readings per axis.
    all_x = [v[0] for v in raw.values()]
    all_y = [v[1] for v in raw.values()]
    all_z = [v[2] for v in raw.values()]

    x_min_raw = min(all_x)
    x_max_raw = max(all_x)
    y_min_raw = min(all_y)
    y_max_raw = max(all_y)
    z_min_raw = min(all_z)
    z_max_raw = max(all_z)

    m = args.margin_mm
    bounds = (
        x_min_raw + m, x_max_raw - m,
        y_min_raw + m, y_max_raw - m,
        z_min_raw + m, z_max_raw - m,
    )

    print("\n" + "=" * 60)
    print("RAW MEASUREMENTS (mm):")
    print(f"  x: [{x_min_raw:.1f}, {x_max_raw:.1f}]   range = {x_max_raw - x_min_raw:.1f}")
    print(f"  y: [{y_min_raw:.1f}, {y_max_raw:.1f}]   range = {y_max_raw - y_min_raw:.1f}")
    print(f"  z: [{z_min_raw:.1f}, {z_max_raw:.1f}]   range = {z_max_raw - z_min_raw:.1f}")
    print(f"\nWITH {args.margin_mm:.0f}mm INWARD MARGIN:")
    print(f"  x: [{bounds[0]:.1f}, {bounds[1]:.1f}]")
    print(f"  y: [{bounds[2]:.1f}, {bounds[3]:.1f}]")
    print(f"  z: [{bounds[4]:.1f}, {bounds[5]:.1f}]")
    print("\nPaste into your run command:")
    print(f"  --workspace-bounds {bounds[0]:.1f} {bounds[1]:.1f} "
          f"{bounds[2]:.1f} {bounds[3]:.1f} {bounds[4]:.1f} {bounds[5]:.1f}")
    print("=" * 60)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "bounds": list(bounds),
                "margin_mm": args.margin_mm,
                "raw_measurements": raw,
            }, f, indent=2)
        print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
