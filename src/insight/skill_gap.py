"""Shared single-axis skill-gap motion data type.

Both the LIBERO sim flywheel and the xArm flywheel describe skill-gap
motions as a single dominant axis (one of dx/dy/dz/drx/dry/drz) plus a
signed magnitude. This module hosts the shared dataclass + axis-name
constants so both implementations agree on the contract.

What lives here (env-agnostic):
- ``SkillGapMotion`` — the (axis, magnitude) tuple with unit-conversion
  accessors.
- ``AXIS_INDEX`` — name → pose-vector index map.
- ``ROTATION_AXES`` — set of rotation axis names (for ``is_rotation``).

What does NOT live here (env-specific):
- The P-control loop, action dispatch, frame capture — those stay in
  each environment's executor.
- Tuning constants for individual control loops (e.g., xArm's
  ``_AT_TARGET_GAP_MM``, sim's max-step caps) — those stay near the
  loops that use them.
"""

from __future__ import annotations

import dataclasses


# Pose-vector layout: [x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg].
# Same convention works for sim (the wrapper handles the conversion to
# the underlying obs/action format) and xArm (native units).
AXIS_INDEX: dict[str, int] = {
    "dx": 0, "dy": 1, "dz": 2, "drx": 3, "dry": 4, "drz": 5,
}
ROTATION_AXES: frozenset[str] = frozenset({"drx", "dry", "drz"})


@dataclasses.dataclass(frozen=True)
class SkillGapMotion:
    """Single-axis motion parsed from VLM pre-analysis output.

    Translation: ``axis`` in ``{dx, dy, dz}``, ``delta_m`` carries the
    magnitude in meters.
    Rotation:    ``axis`` in ``{drx, dry, drz}``, ``delta_deg`` carries it
    in degrees.

    Use ``is_rotation`` to branch dispatch. ``delta_native`` returns the
    magnitude in the unit native to a 6-vector pose (translation: mm;
    rotation: deg), which is what ``target_pose[axis_idx] += delta_native``
    expects.
    """
    axis: str               # dx/dy/dz/drx/dry/drz
    delta_m: float = 0.0    # signed magnitude in meters (translation)
    delta_deg: float = 0.0  # signed magnitude in degrees (rotation)

    @property
    def axis_idx(self) -> int:
        return AXIS_INDEX[self.axis]

    @property
    def is_rotation(self) -> bool:
        return self.axis in ROTATION_AXES

    @property
    def delta_native(self) -> float:
        """Magnitude in the unit native to the pose vector.

        Translation primitives: millimeters. Rotation primitives: degrees.
        Lets callers do ``target_pose[axis_idx] += delta_native`` uniformly.
        """
        return self.delta_deg if self.is_rotation else self.delta_m * 1000.0

    # Legacy alias kept so the xArm P-control loop reads naturally for
    # translation; for rotation primitives the value is in degrees, not
    # millimeters.
    delta_mm = delta_native

    @property
    def direction(self) -> float:
        return 1.0 if self.delta_native > 0 else -1.0

    @property
    def extend_mm(self) -> float:
        return abs(self.delta_native)
