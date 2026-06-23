"""Interactive tool-tip-offset calibration for the xArm.

Computes the EE-local tool offset vector ``t_local`` such that for any TCP
pose ``(p, R)``, the tool tip's world position is ``p + R @ t_local``.

How it works:
  1. The user puts the arm in teach mode and physically positions the tool tip
     at a calibration point with known world coordinates (e.g. center of the
     workspace, touching the acrylic at the measured contact Z).
  2. The script reads the current TCP pose ``(p_tcp, rpy_tcp)`` from the arm.
  3. Solves for ``t_local = R(rpy_tcp)^T @ (p_tip_world - p_tcp)``.
  4. Prints the offset, ready to paste into ``--tool-offset-mm tx ty tz``.

Once measured, the offset is constant across runs (it's a property of the
end-effector + tool, not the workspace). Re-run only if you change the tool
or how it's mounted.

Usage:
    uv run real/entry/measure_tool_offset.py \\
        --tip-x 457.0 --tip-y 0.0 --tip-z 211.2 \\
        --output tool_offset.json
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


def _R_from_xarm_rpy(rpy_rad: np.ndarray) -> np.ndarray:
    """xArm ZYX intrinsic Euler → rotation matrix mapping local → world."""
    roll, pitch, yaw = rpy_rad
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


@dataclasses.dataclass
class Args:
    arm_ip: str = "192.168.1.219"

    tip_x: float = 0.0
    tip_y: float = 0.0
    tip_z: float = 0.0
    """World coordinates (mm) of the calibration point you'll touch the tool tip
    to. Most reliable choice: a known location on the acrylic surface — pick
    a marker, measure its world XYZ once, then re-use those values for every
    calibration."""

    output: str = ""
    """If set, write tool offset + raw measurements as JSON to this path."""


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.tip_x == 0.0 and args.tip_y == 0.0 and args.tip_z == 0.0:
        raise SystemExit(
            "Set --tip-x/--tip-y/--tip-z to the world coordinates of the "
            "calibration point. Default (0,0,0) is rejected as a likely typo."
        )

    arm = XArmAPI(args.arm_ip)
    arm.connect()
    if arm.get_state() != 0:
        arm.clean_error()
        time.sleep(0.5)
    arm.motion_enable(enable=True)

    print("\nEntering TEACH MODE — you can now physically push the arm.")
    print("Brakes are released; support the arm if needed when first switching modes.\n")
    arm.set_mode(2)
    arm.set_state(0)
    time.sleep(0.5)

    print(f"Calibration point (world frame): "
          f"x={args.tip_x:.1f} y={args.tip_y:.1f} z={args.tip_z:.1f} mm")
    print("Position the TOOL TIP at this exact point. Take your time — orientation\n"
          "during this measurement does NOT need to match operating orientation; the\n"
          "computed offset is in EE-local frame and is invariant across orientations.")
    input("Press ENTER when the tool tip is at the calibration point: ")

    raw_pose = list(arm.get_position()[1])
    raw_pose[3] %= 360
    raw_pose[5] %= 360
    p_tcp = np.array(raw_pose[:3], dtype=np.float64)
    rpy_deg = np.array(raw_pose[3:6], dtype=np.float64)

    print(f"\nRecorded TCP pose: x=%.2f y=%.2f z=%.2f rpy=(%.2f, %.2f, %.2f)"
          % tuple(raw_pose))

    # Restore position-control mode before disconnect.
    print("\nRestoring position-control mode...")
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.5)
    arm.disconnect()

    # FK: tip_world = p_tcp + R @ t_local  →  t_local = R^T @ (tip_world - p_tcp)
    p_tip_world = np.array([args.tip_x, args.tip_y, args.tip_z], dtype=np.float64)
    R = _R_from_xarm_rpy(np.deg2rad(rpy_deg))
    t_local = R.T @ (p_tip_world - p_tcp)

    print("\n" + "=" * 60)
    print("TOOL OFFSET (EE-local frame, mm):")
    print(f"  tx = {t_local[0]:+.2f}")
    print(f"  ty = {t_local[1]:+.2f}")
    print(f"  tz = {t_local[2]:+.2f}")
    print(f"  |t| = {np.linalg.norm(t_local):.2f} mm  (tool length from TCP)")
    print(f"\nPaste into your run command:")
    print(f"  --tool-offset-mm {t_local[0]:.2f} {t_local[1]:.2f} {t_local[2]:.2f}")
    print("=" * 60)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "tool_offset_local_mm": t_local.tolist(),
                "calibration_point_world_mm": p_tip_world.tolist(),
                "tcp_pose_at_calibration": raw_pose,
            }, f, indent=2)
        print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
