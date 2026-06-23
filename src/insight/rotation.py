"""Robot-agnostic rotation utilities for skill-gap execution.

Shared by the sim flywheel (LIBERO/robosuite OSC) and the real-hardware
flywheel (xArm cartesian servo). Keeps quaternion math + skill-gap
target construction in one place so both pipelines reason about
rotations the same way.

Quaternion convention: ``[w, x, y, z]`` (scalar-first). This matches what
the LIBERO/mujoco/robosuite stack uses throughout. The xArm SDK speaks
RPY in degrees, but we store target as quaternion internally and only
convert to/from RPY at the controller boundary.

Skill-gap convention:
- Axis labels ``"drx" / "dry" / "drz"`` map to base-frame world axes
  ``[1,0,0] / [0,1,0] / [0,0,1]`` respectively. Composition is
  left-multiplied (q_target = R_world * q_current), matching LIBERO's
  ``_build_target_quat``.
- Magnitudes are signed degrees. Positive = right-hand-rule around the
  base axis.
"""

from __future__ import annotations

import numpy as np


__all__ = [
    "AXIS_VECTORS",
    "quat_multiply",
    "quat_conjugate",
    "quat_from_axis_angle",
    "quat_to_axis_angle",
    "quat_normalize",
    "quat_error_to_world_rotvec",
    "build_target_quat",
    "rpy_to_quat_deg",
    "quat_to_rpy_deg",
    "unwrap_rpy_near",
    "slerp",
]


AXIS_VECTORS = {
    "drx": np.array([1.0, 0.0, 0.0]),
    "dry": np.array([0.0, 1.0, 0.0]),
    "drz": np.array([0.0, 0.0, 1.0]),
}


# ──────────────── Quaternion primitives ────────────────


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 * q2. Both inputs ``[w, x, y, z]``."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def rotate_vector_by_quat(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate 3D vector ``v`` by unit quaternion ``q`` ([w,x,y,z])."""
    v_quat = np.array([0.0, float(v[0]), float(v[1]), float(v[2])])
    rotated = quat_multiply(quat_multiply(q, v_quat), quat_conjugate(q))
    return rotated[1:]


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])


def quat_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Build a unit quaternion from axis-angle. Axis need not be unit length."""
    n = np.linalg.norm(axis)
    if n < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    a = np.asarray(axis, dtype=float) / n
    h = angle_rad / 2.0
    s = np.sin(h)
    return np.array([np.cos(h), s * a[0], s * a[1], s * a[2]])


def quat_to_axis_angle(q: np.ndarray) -> tuple[np.ndarray, float]:
    """Decode a unit quaternion to (axis, angle_rad). Shortest-path."""
    q = quat_normalize(np.asarray(q, dtype=float))
    if q[0] < 0:
        q = -q
    sin_ha = np.linalg.norm(q[1:])
    if sin_ha < 1e-10:
        return np.array([1.0, 0.0, 0.0]), 0.0
    angle = 2.0 * np.arctan2(sin_ha, q[0])
    axis = q[1:] / sin_ha
    return axis, angle


def quat_error_to_world_rotvec(q_target: np.ndarray, q_current: np.ndarray) -> np.ndarray:
    """World-frame rotation needed to go from q_current to q_target,
    expressed as an axis-angle vector (length = angle in radians).

    Computed as q_err = q_target * q_current^{-1}, decoded to axis*angle.
    """
    q_err = quat_multiply(np.asarray(q_target), quat_conjugate(np.asarray(q_current)))
    if q_err[0] < 0:
        q_err = -q_err
    sin_ha = np.linalg.norm(q_err[1:])
    if sin_ha < 1e-8:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(sin_ha, q_err[0])
    axis = q_err[1:] / sin_ha
    return axis * angle


# ──────────────── Skill-gap target construction ────────────────


def build_target_quat(q_initial: np.ndarray, axis: str, angle_deg: float,
                      frame: str = "world") -> np.ndarray:
    """Apply a single-axis rotation to ``q_initial``.

    Two composition modes:
      - ``frame="world"`` (default): q_target = R(axis, angle) * q_initial.
        ``axis`` refers to a fixed world/base axis. Predictable in absolute
        space; sign conventions don't depend on current EE orientation.
      - ``frame="local"``: q_target = q_initial * R(axis, angle).
        ``axis`` refers to the EE's current local axis. Matches "tilt
        forward/sideways relative to the gripper" intuition regardless of
        how the EE is currently oriented. Use when reasoning about pour /
        flip-style motions on a non-canonically-oriented EE.

    Args:
        q_initial: starting orientation, ``[w, x, y, z]``.
        axis: one of ``"drx" / "dry" / "drz"``.
        angle_deg: signed magnitude in degrees.
        frame: ``"world"`` or ``"local"``.
    """
    if axis not in AXIS_VECTORS:
        raise ValueError(f"axis must be one of {list(AXIS_VECTORS)}, got {axis!r}")
    if frame not in ("world", "local"):
        raise ValueError(f"frame must be 'world' or 'local', got {frame!r}")
    if abs(angle_deg) < 1e-6:
        return np.asarray(q_initial, dtype=float).copy()
    q_rot = quat_from_axis_angle(AXIS_VECTORS[axis], np.deg2rad(angle_deg))
    q_initial = np.asarray(q_initial, dtype=float)
    if frame == "local":
        return quat_normalize(quat_multiply(q_initial, q_rot))
    return quat_normalize(quat_multiply(q_rot, q_initial))


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two unit quaternions, ``t`` in [0, 1]."""
    q0 = quat_normalize(np.asarray(q0, dtype=float))
    q1 = quat_normalize(np.asarray(q1, dtype=float))
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return quat_normalize(q0 + t * (q1 - q0))
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_t0 = np.sin(theta_0)
    s0 = np.sin(theta_0 - theta) / sin_t0
    s1 = np.sin(theta) / sin_t0
    return s0 * q0 + s1 * q1


# ──────────────── xArm-style RPY conversion ────────────────
# The xArm SDK uses extrinsic XYZ Euler angles (roll, pitch, yaw) in degrees.
# These helpers stay here so both sim and hardware can speak the same units
# at the controller boundary without each repo rolling its own conversion.


def rpy_to_quat_deg(rpy_deg: np.ndarray) -> np.ndarray:
    """Convert ``[roll, pitch, yaw]`` in degrees to a quaternion ``[w, x, y, z]``.

    Convention: extrinsic XYZ Euler — apply Rx then Ry then Rz to a vector
    when written as ``Rz * Ry * Rx * v``. Matches xArm's set_position /
    set_servo_cartesian RPY.
    """
    r, p, y = np.deg2rad(np.asarray(rpy_deg, dtype=float))
    q_x = quat_from_axis_angle([1, 0, 0], r)
    q_y = quat_from_axis_angle([0, 1, 0], p)
    q_z = quat_from_axis_angle([0, 0, 1], y)
    return quat_normalize(quat_multiply(q_z, quat_multiply(q_y, q_x)))


def quat_to_rpy_deg(q: np.ndarray) -> np.ndarray:
    """Convert a quaternion ``[w, x, y, z]`` to extrinsic-XYZ ``[roll, pitch, yaw]``
    in degrees. Matches the xArm SDK convention.

    Returns principal values: roll/yaw in ``[-180, 180]``, pitch in ``[-90, 90]``.
    For controller commands across the principal-value boundary (e.g. when
    the arm reports roll outside ``[-180, 180]``), wrap each subgoal RPY with
    ``unwrap_rpy_near`` so consecutive commands don't differ by 360°.
    """
    w, x, y, z = quat_normalize(np.asarray(q, dtype=float))
    # Standard xyz Euler from quaternion. Atan2 + asin avoid singularity branches.
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sin_p = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sin_p, -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.rad2deg([roll, pitch, yaw])


def unwrap_rpy_near(rpy_deg: np.ndarray, reference_deg: np.ndarray) -> np.ndarray:
    """Add ±360° to each RPY component so it lies within 180° of the reference.

    Keeps commanded values continuous when the arm's reported pose is outside
    the principal range. Without this, ``quat_to_rpy_deg`` returns values in
    ``[-180, 180]`` while the xArm controller may report (and plan from)
    values like ``+182°``, producing a 360° joint swing for the same physical
    pose. Apply to every per-tick subgoal RPY before commanding.
    """
    rpy = np.asarray(rpy_deg, dtype=float).copy()
    ref = np.asarray(reference_deg, dtype=float)
    for i in range(len(rpy)):
        diff = rpy[i] - ref[i]
        rpy[i] -= 360.0 * round(diff / 360.0)
    return rpy
