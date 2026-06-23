"""Closed-loop control, quaternion math, VLA replay, correction parsing, visualization."""

from __future__ import annotations

import logging
import re

import numpy as np

from .env import (
    FlywheelDatapoint,
    get_obs_images,
    resize_for_policy,
    resize_for_vlm,
    settle_physics,
    stop_requested,
)


# =============================================================================
# Math/quaternion helpers — re-exported from ``insight.rotation`` so the sim
# and the real-hardware xArm flywheel share a single implementation. The
# underscore-prefixed names are preserved as thin wrappers so existing call
# sites in this package keep working without churn.
# =============================================================================

from insight.rotation import (
    quat_multiply as _quat_multiply_arr,
    quat_conjugate as _quat_conjugate,
    quat_from_axis_angle as _quat_from_axis_angle,
    quat_error_to_world_rotvec as _quat_error_to_world_rotvec,
    quat_to_rpy_deg as _quat_to_rpy_deg_shared,
    build_target_quat as _build_target_quat_single_axis,
)


def _quat_multiply(q1, q2):
    """Hamilton product of [w,x,y,z] quaternions; returns a list (legacy API)."""
    return list(_quat_multiply_arr(np.asarray(q1, float), np.asarray(q2, float)))


def _find_red_block_joint(env):
    """Return the MuJoCo joint name for the red lego block free joint."""
    model = env.sim.model
    for i in range(model.njnt):
        name = model.joint_id2name(i)
        if name and "upside_down_red_lego_block" in name:
            return name
    return None


def _find_blue_block_joint(env):
    """Return the MuJoCo joint name for the blue lego block free joint, if any."""
    model = env.sim.model
    for i in range(model.njnt):
        name = model.joint_id2name(i)
        if name and "blue_lego_block" in name:
            return name
    return None


def blocks_too_close(env, min_dist: float = 0.10) -> bool:
    """Check if red and blue lego blocks are within ``min_dist`` (meters) of
    each other in the xy plane.

    Used to decide whether to re-seed the scene at trial start — when the
    blocks spawn very close together, the grasp policy can confuse them
    (it might approach blue when targeted at red, or rotate red and bump
    blue out of position). Returns False if either joint isn't present
    (no rejection in that case)."""
    red_name = _find_red_block_joint(env)
    blue_name = _find_blue_block_joint(env)
    if red_name is None or blue_name is None:
        return False
    red_xy = env.sim.data.get_joint_qpos(red_name)[:2]
    blue_xy = env.sim.data.get_joint_qpos(blue_name)[:2]
    dist = float(np.linalg.norm(np.asarray(red_xy) - np.asarray(blue_xy)))
    return dist < min_dist


def _quat_to_euler_rad(w, x, y, z):
    """Convert quaternion (w,x,y,z) to [roll, pitch, yaw] in radians.

    Wrapper around ``insight.rotation.quat_to_rpy_deg``; converts back to
    radians at the boundary so existing radians-based callers keep working.
    """
    rpy_deg = _quat_to_rpy_deg_shared(np.array([w, x, y, z]))
    return np.deg2rad(rpy_deg)


def _quat_to_euler_deg(q):
    """Convert [w,x,y,z] quaternion to [roll, pitch, yaw] in degrees (tuple)."""
    return tuple(_quat_to_rpy_deg_shared(np.asarray(q, float)))


def _get_ee_euler_rad(obs):
    """Get gripper [roll, pitch, yaw] in radians from observation."""
    eq = obs["robot0_eef_quat"]  # [x,y,z,w] in robosuite
    return _quat_to_euler_rad(eq[3], eq[0], eq[1], eq[2])


def _angle_diff(target, current):
    """Signed angle difference, normalized to [-pi, pi]."""
    diff = target - current
    return (diff + np.pi) % (2 * np.pi) - np.pi


def _quat_rotation_angle(q1, q2):
    """Total rotation angle (degrees) between two quaternions [x,y,z,w]."""
    q1_conj = np.array([-q1[0], -q1[1], -q1[2], q1[3]])
    w_rel = (q2[3]*q1_conj[3] - q2[0]*q1_conj[0]
             - q2[1]*q1_conj[1] - q2[2]*q1_conj[2])
    return np.degrees(2 * np.arccos(np.clip(abs(w_rel), 0, 1)))


# =============================================================================
# Quaternion P-control helpers
# =============================================================================

def _get_ee_quat_wxyz(obs):
    """Get EE quaternion in [w,x,y,z] from observation ([x,y,z,w] in robosuite)."""
    eq = obs["robot0_eef_quat"]
    return np.array([eq[3], eq[0], eq[1], eq[2]])


def _build_target_quat(q_initial, offsets, frame: str = "world"):
    """Build a target quaternion by applying rotation offsets.

    offsets: array of length 7 (or 6) — indices 3,4,5 are drx,dry,drz in rad.
    frame: ``"world"`` (default, legacy callers) treats drx/dry/drz as
        fixed world axes. ``"local"`` treats them as axes of the EE's
        current local frame — "rotate around x" means around the
        gripper's current x-axis regardless of world orientation, which
        matches the intuitive "tilt forward/sideways relative to the
        gripper" semantics needed for flip-style skill-gap motions.

    Composes as: q_target = R(z) ∘ R(y) ∘ R(x) ∘ q_initial, with each
    axis interpreted per ``frame``.

    Default kept as world so the legacy ``_replay_with_correction``
    call site (which was written assuming world-frame deltas) keeps
    working unchanged; pass ``frame="local"`` explicitly from the
    skill-gap rotation site.
    """
    q = np.asarray(q_initial, dtype=float).copy()
    for axis_idx, axis_label in [(3, "drx"), (4, "dry"), (5, "drz")]:
        angle_rad = offsets[axis_idx] if axis_idx < len(offsets) else 0.0
        if abs(angle_rad) > 1e-8:
            q = _build_target_quat_single_axis(
                q, axis_label, float(np.rad2deg(angle_rad)), frame=frame,
            )
    return q


def _compute_peg_correction_quat(peg_direction, q_ee_current):
    """Compute EE target quaternion that brings the peg to +Z.

    Given the current peg direction (world frame) and EE quaternion,
    computes the world-frame rotation that maps peg → +Z, then applies
    it to q_ee_current to get the target EE quaternion.

    Returns (target_quat, correction_angle_deg, correction_axis) or
    (None, 0, None) if peg is already up.
    """
    peg = np.asarray(peg_direction, dtype=float)
    peg = peg / np.linalg.norm(peg)
    up = np.array([0.0, 0.0, 1.0])

    cos_angle = np.clip(np.dot(peg, up), -1.0, 1.0)
    angle = np.arccos(cos_angle)

    if angle < np.radians(5.0):
        return None, 0.0, None

    # Rotation axis = peg × up (world frame)
    axis = np.cross(peg, up)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-10:
        # peg is anti-parallel to up (pointing straight down) — pick arbitrary horizontal axis
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = axis / axis_norm

    q_correction = _quat_from_axis_angle(axis, angle)
    q_target = np.array(_quat_multiply(list(q_correction), list(q_ee_current)))
    q_target = q_target / np.linalg.norm(q_target)

    logging.info(
        f"  [Rotation correction] Peg=[{peg[0]:.2f}, {peg[1]:.2f}, {peg[2]:.2f}], "
        f"{np.degrees(angle):.0f}° from +Z"
    )
    logging.info(
        f"  [Rotation correction] Axis=[{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}], "
        f"angle={np.degrees(angle):.0f}°, "
        f"target euler={list(_quat_to_euler_deg(q_target))}"
    )

    return q_target, np.degrees(angle), axis


# =============================================================================
# P-control constants
# =============================================================================

_K_P_ROT = 3.0
_MAX_ROT_CMD = 0.15

_K_P_TRANS = 5.0
_MAX_TRANS_CMD = 0.15
_MAX_POS_HOLD_CMD = 0.08  # Balance between counteracting rotation drift and smooth motion


# =============================================================================
# Block tilting
# =============================================================================

def tilt_red_block_to_side(env, obs):
    """Tilt the red block ~90 deg about a random horizontal axis."""
    joint_name = _find_red_block_joint(env)
    if joint_name is None:
        logging.warning("Could not find red block joint — skipping tilt")
        return obs

    qpos = env.sim.data.get_joint_qpos(joint_name).copy()
    pos = qpos[:3]
    quat = list(qpos[3:7])

    axis_angle = np.random.uniform(0, 2 * np.pi)
    tilt_angle = np.pi / 2 + np.random.uniform(-0.15, 0.15)
    half = tilt_angle / 2
    ax = np.cos(axis_angle) * np.sin(half)
    ay = np.sin(axis_angle) * np.sin(half)
    q_tilt = [np.cos(half), ax, ay, 0.0]

    z_spin = np.random.uniform(-np.pi, np.pi)
    hz = z_spin / 2
    q_spin = [np.cos(hz), 0.0, 0.0, np.sin(hz)]

    new_quat = _quat_multiply(_quat_multiply(quat, q_tilt), q_spin)
    pos[2] += 0.015
    new_qpos = np.concatenate([pos, new_quat])
    env.sim.data.set_joint_qpos(joint_name, new_qpos)
    env.sim.forward()
    obs = settle_physics(env, obs, steps=100)
    logging.info(f"Tilt axis angle: {np.degrees(axis_angle):.0f} deg, tilt: {np.degrees(tilt_angle):.0f} deg, spin: {np.degrees(z_spin):.0f} deg")
    return obs


# =============================================================================
# Drawing/visualization
# =============================================================================

def _project_3d_to_pixel(env, point_3d, img_shape, camera_name="agentview"):
    """Project a 3D world point to 2D pixel coordinates."""
    cam_id = env.sim.model.camera_name2id(camera_name)
    cam_pos = env.sim.data.cam_xpos[cam_id]
    cam_mat = env.sim.data.cam_xmat[cam_id].reshape(3, 3)
    fovy = env.sim.model.cam_fovy[cam_id]

    p_cam = cam_mat.T @ (point_3d - cam_pos)
    h, w = img_shape[:2]
    f = (h / 2.0) / np.tan(np.radians(fovy) / 2.0)

    raw_px = f * p_cam[0] / (-p_cam[2]) + w / 2.0
    raw_py = f * (-p_cam[1]) / (-p_cam[2]) + h / 2.0

    px = int(raw_px)
    py = int(h - 1 - raw_py)
    return px, py


def _get_axis_info(env, obs, img_shape):
    """Compute EE pixel position and axis directions for rotation indicators."""
    ee_pos = obs["robot0_eef_pos"] + np.array([0, 0, 0.025])
    small = 0.01
    cx, cy = _project_3d_to_pixel(env, ee_pos, img_shape)
    xx, xy = _project_3d_to_pixel(env, ee_pos + np.array([small, 0, 0]), img_shape)
    yx, yy = _project_3d_to_pixel(env, ee_pos + np.array([0, small, 0]), img_shape)

    x_dir = np.array([xx - cx, xy - cy], dtype=float)
    y_dir = np.array([yx - cx, yy - cy], dtype=float)
    y_norm = y_dir / max(np.linalg.norm(y_dir), 1e-6)

    origin_3d = ee_pos.copy()
    axis_len = 0.06
    ox, oy = _project_3d_to_pixel(env, origin_3d, img_shape)
    wx, wy = _project_3d_to_pixel(env, origin_3d + np.array([axis_len, 0, 0]), img_shape)
    vx, vy = _project_3d_to_pixel(env, origin_3d + np.array([0, axis_len, 0]), img_shape)
    zx, zy = _project_3d_to_pixel(env, origin_3d + np.array([0, 0, axis_len]), img_shape)
    world_axes = {
        "origin": (ox, oy),
        "X": (zx, zy),
        "Y": (vx, vy),
        "Z": (wx, wy),
    }

    return (cx, cy), x_dir, y_norm, world_axes


def _draw_axes_on_bgr(out, world_axes, scale=1.0, labels=None):
    """Draw world XYZ coordinate frame arrows on a BGR image (in-place)."""
    import cv2
    red = (60, 60, 255)
    green = (60, 210, 60)
    blue = (255, 120, 60)
    black = (0, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if labels is None:
        labels = {"X": "+X", "Y": "+Y", "Z": "+Z"}
    o = world_axes["origin"]
    axes_info = [("X", labels.get("X", "+X"), red), ("Y", labels.get("Y", "+Y"), green), ("Z", labels.get("Z", "+Z"), blue)]
    thickness_ax = max(1, int(2 * scale))
    font_scale_ax = 0.4 * scale
    for key, label, color in axes_info:
        if key not in world_axes:
            continue
        tip = world_axes[key]
        cv2.arrowedLine(out, o, tip, black, max(1, int(4 * scale)), cv2.LINE_AA, tipLength=0.2)
        cv2.arrowedLine(out, o, tip, color, thickness_ax, cv2.LINE_AA, tipLength=0.2)
        cv2.putText(out, label, (tip[0] + int(4 * scale), tip[1] + int(4 * scale)),
                    font, font_scale_ax, black, max(1, int(3 * scale)), cv2.LINE_AA)
        cv2.putText(out, label, (tip[0] + int(4 * scale), tip[1] + int(4 * scale)),
                    font, font_scale_ax, color, max(1, int(scale)), cv2.LINE_AA)


def _draw_world_axes_only(img, world_axes, scale=1.0):
    """Draw just the world XYZ coordinate frame arrows on an image."""
    import cv2
    out = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
    _draw_axes_on_bgr(out, world_axes, scale)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def _draw_indicators(img, center, x_dir, y_norm, scale=1.0, world_axes=None):
    """Draw drx arc, dry markers, and world XYZ frame on img."""
    import cv2
    cx, cy = center
    perp = np.array([-y_norm[1], y_norm[0]])
    x_angle_cv = np.degrees(np.arctan2(x_dir[1], x_dir[0]))

    out = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
    red = (60, 60, 255)
    green = (60, 210, 60)
    black = (0, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    if world_axes is not None:
        _draw_axes_on_bgr(out, world_axes, scale, labels={"X": "X fwd", "Y": "Y right", "Z": "Z up"})

    # drx arc
    arc_offset = int(35 * scale)
    arc_r = int(22 * scale)
    arc_cx = int(cx + perp[0] * arc_offset)
    arc_cy = int(cy + perp[1] * arc_offset)
    thickness_outer = max(1, int(4 * scale))
    thickness_inner = max(1, int(2 * scale))

    cv2.ellipse(out, (arc_cx, arc_cy), (arc_r, arc_r), x_angle_cv,
                190, 350, black, thickness_outer, cv2.LINE_AA)
    cv2.ellipse(out, (arc_cx, arc_cy), (arc_r, arc_r), x_angle_cv,
                190, 350, red, thickness_inner, cv2.LINE_AA)

    end_rad = np.radians(x_angle_cv + 190)
    tip_x = int(arc_cx + arc_r * np.cos(end_rad))
    tip_y = int(arc_cy + arc_r * np.sin(end_rad))
    tang = np.array([np.sin(end_rad), -np.cos(end_rad)])
    arrow_len = max(3, int(7 * scale))
    for sign in [-1, 1]:
        a = np.radians(sign * 35)
        d = np.array([tang[0]*np.cos(a) - tang[1]*np.sin(a),
                       tang[0]*np.sin(a) + tang[1]*np.cos(a)])
        b = (int(tip_x - d[0]*arrow_len), int(tip_y - d[1]*arrow_len))
        cv2.line(out, (tip_x, tip_y), b, black, max(1, int(3 * scale)), cv2.LINE_AA)
        cv2.line(out, (tip_x, tip_y), b, red, thickness_inner, cv2.LINE_AA)

    # drx label
    mid_rad = np.radians(x_angle_cv + 270)
    label_dist = arc_r + int(10 * scale)
    lx = int(arc_cx + label_dist * np.cos(mid_rad))
    ly = int(arc_cy + label_dist * np.sin(mid_rad))
    font_scale = 0.35 * scale
    cv2.putText(out, "+drx", (lx - int(18 * scale), ly + int(4 * scale)),
                font, font_scale, black, max(1, int(3 * scale)), cv2.LINE_AA)
    cv2.putText(out, "+drx", (lx - int(18 * scale), ly + int(4 * scale)),
                font, font_scale, red, max(1, int(scale)), cv2.LINE_AA)

    # dry symbol
    sym_offset = int(35 * scale)
    sym_r = int(10 * scale)
    p1 = (int(cx + y_norm[0] * sym_offset), int(cy + y_norm[1] * sym_offset))
    cv2.circle(out, p1, sym_r, black, max(1, int(3 * scale)), cv2.LINE_AA)
    cv2.circle(out, p1, sym_r, green, thickness_inner, cv2.LINE_AA)
    xr = int(sym_r * 0.6)
    cv2.line(out, (p1[0] - xr, p1[1] - xr), (p1[0] + xr, p1[1] + xr),
             black, max(1, int(3 * scale)), cv2.LINE_AA)
    cv2.line(out, (p1[0] - xr, p1[1] + xr), (p1[0] + xr, p1[1] - xr),
             black, max(1, int(3 * scale)), cv2.LINE_AA)
    cv2.line(out, (p1[0] - xr, p1[1] - xr), (p1[0] + xr, p1[1] + xr),
             green, thickness_inner, cv2.LINE_AA)
    cv2.line(out, (p1[0] - xr, p1[1] + xr), (p1[0] + xr, p1[1] - xr),
             green, thickness_inner, cv2.LINE_AA)

    # dry label
    lbl_x = int(p1[0] + perp[0] * int(-14 * scale))
    lbl_y = int(p1[1] + perp[1] * int(-14 * scale))
    cv2.putText(out, "+dry", (lbl_x - int(20 * scale), lbl_y + int(4 * scale)),
                font, font_scale, black, max(1, int(3 * scale)), cv2.LINE_AA)
    cv2.putText(out, "+dry", (lbl_x - int(20 * scale), lbl_y + int(4 * scale)),
                font, font_scale, green, max(1, int(scale)), cv2.LINE_AA)

    # EE dot
    cv2.circle(out, (int(cx), int(cy)), max(1, int(2 * scale)), black, -1, cv2.LINE_AA)
    cv2.circle(out, (int(cx), int(cy)), max(1, int(scale)), (255, 255, 255), -1, cv2.LINE_AA)

    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return out


# =============================================================================
# HSV detection + cropping
# =============================================================================

def _detect_red_mask_hsv(img):
    """Return binary mask of red pixels using HSV thresholds."""
    import cv2
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask1 = cv2.inRange(hsv, (0, 70, 70), (12, 255, 255))
    mask2 = cv2.inRange(hsv, (168, 70, 70), (180, 255, 255))
    return mask1 | mask2


def _crop_around_block_hsv(img, padding=None, min_half=None):
    """Crop image around the red block using HSV color detection."""
    import cv2
    red_mask = _detect_red_mask_hsv(img)
    if not red_mask.any():
        logging.info("  [crop] No red pixels found in HSV")
        return None
    ys, xs = np.where(red_mask > 0)
    cy = (ys.min() + ys.max()) // 2
    cx = (xs.min() + xs.max()) // 2
    logging.info(f"  [crop] Red block center: ({cx}, {cy}), red pixels: {len(xs)}")
    h, w = img.shape[:2]
    s = h / 256.0
    if padding is None:
        padding = int(60 * s)
    if min_half is None:
        min_half = int(60 * s)
    half = max(padding, min_half)
    y1 = max(0, cy - half)
    y2 = min(h, cy + half)
    x1 = max(0, cx - half)
    x2 = min(w, cx + half)
    return img[y1:y2, x1:x2]


# =============================================================================
# VLA record + replay + correction
# =============================================================================

def _run_vla_and_record(
    env, client, primitive, max_steps, obs,
    save_frames=True, collect_data=False, downsample_hz=2,
    force_grip_closed=False,
    early_stop_fn=None,
):
    """Run VLA normally and record all actions for potential replay.

    If force_grip_closed=True, override VLA grip to +1 (closed) every step.
    Useful for 'rotate block' where VLA may open gripper OOD.

    early_stop_fn: optional callable(env, obs) → bool.  Called every 10 steps
    (after stepping env); returns True to stop early (goal reached mid-run).
    """
    if obs is None:
        obs = settle_physics(env)

    before_img, wrist_before = get_obs_images(obs)
    frames = [before_img] if save_frames else []
    recorded_actions = []
    datapoints = []
    pending_obs = None
    accumulated_action = np.zeros(7)
    episode_done = False

    for step in range(max_steps):
        if stop_requested():
            break
        img, wrist = get_obs_images(obs)
        if save_frames:
            frames.append(img)
        img_r = resize_for_policy(img)
        wrist_r = resize_for_policy(wrist)
        state = np.concatenate((obs["robot0_joint_pos"], obs["robot0_gripper_qpos"][:1]))

        action = np.array(client.infer({
            "observation/image": img_r,
            "observation/wrist_image": wrist_r,
            "observation/state": state,
            "prompt": primitive,
        })["actions"][0])

        recorded_actions.append(action.copy())

        if force_grip_closed:
            action[6] = 1.0

        if step == 0:
            logging.info(
                f"  First action: [{action[0]:+.3f}, {action[1]:+.3f}, "
                f"{action[2]:+.3f}] grip={action[6]:+.3f}"
            )

        if collect_data:
            if step % downsample_hz == 0:
                pending_obs = {"image": img_r.copy(), "wrist": wrist_r.copy(), "state": state.copy()}
                accumulated_action = action.copy()
            else:
                accumulated_action[:6] += action[:6]
                accumulated_action[6] = action[6]
            if (step + 1) % downsample_hz == 0 and pending_obs:
                datapoints.append(FlywheelDatapoint(
                    pending_obs["image"], pending_obs["wrist"], pending_obs["state"],
                    accumulated_action.copy(), "policy", primitive,
                    step_in_primitive=step // downsample_hz,
                ))
                pending_obs = None

        try:
            obs, _, done, _ = env.step(action.tolist() if hasattr(action, 'tolist') else list(action))
        except ValueError as e:
            if "terminated episode" in str(e):
                episode_done = True
                break
            raise
        if done:
            episode_done = True
            break

        if early_stop_fn is not None and step % 10 == 9 and step > 20:
            if early_stop_fn(env, obs):
                logging.info(f"  [VLA] Early stop at step {step + 1}/{max_steps} (goal reached)")
                break

    after_img, wrist_after = get_obs_images(obs)
    return before_img, after_img, wrist_before, wrist_after, frames, obs, datapoints, episode_done, recorded_actions


def _replay_vla_with_rotation(
    env, recorded_actions, obs,
    target_quat=None,
    controlled_axes=None,
    translation_offset=None,
    rotation_additive=None,
    save_frames=True, collect_data=False, downsample_hz=2,
    force_grip_closed=False,
    early_stop_fn=None,
):
    """Replay VLA trajectory with quaternion P-control rotation correction.

    Uses quaternion error → world-frame rotation vector, avoiding Euler gimbal
    lock and axis confusion at non-zero yaw.

    Rotation correction modes (can be combined for different axes):
    - target_quat + controlled_axes: quaternion P-control on specified axes
      (indices 3,4,5 → world X,Y,Z). Non-controlled axes keep VLA output.
    - rotation_additive: {axis_idx: total_offset_rad} ramped over second half
      (adds to VLA outputs, preserving the learned trajectory dynamics)

    Translation offset (if provided) is ramped linearly over the second half of
    the replay, so the robot arrives at the corrected endpoint smoothly.

    early_stop_fn: optional callable(env, obs) → bool.  Called every 10 steps;
    returns True to stop early (goal reached mid-replay).
    """
    frames = []
    datapoints = []
    pending_obs = None
    accumulated_action = np.zeros(7)
    episode_done = False

    controlled_axes = controlled_axes or set()
    n_steps = len(recorded_actions)
    # Ramp over the second half of replay
    ramp_start = n_steps // 2
    ramp_len = n_steps - ramp_start
    has_trans = translation_offset is not None and np.any(np.abs(translation_offset) > 1e-6)
    has_rot_add = rotation_additive is not None and any(abs(v) > 1e-6 for v in rotation_additive.values())

    if has_trans:
        # Per-step delta in meters — distributed evenly over the ramp
        trans_step_delta = translation_offset / ramp_len
        logging.info(f"  Blending translation [{translation_offset[0]*1000:.1f}, {translation_offset[1]*1000:.1f}, {translation_offset[2]*1000:.1f}]mm over steps {ramp_start}-{n_steps} ({trans_step_delta[0]*1000:.3f}, {trans_step_delta[1]*1000:.3f}, {trans_step_delta[2]*1000:.3f} mm/step)")

    if has_rot_add:
        rot_add_step = {i: v / ramp_len for i, v in rotation_additive.items()}
        axis_names = {3: "drx", 4: "dry", 5: "drz"}
        for i, v in rotation_additive.items():
            logging.info(f"  Blending additive {axis_names.get(i, f'axis{i}')} {np.degrees(v):.1f}° over steps {ramp_start}-{n_steps} ({np.degrees(v/ramp_len):.3f}°/step)")

    # Axes to log (union of P-control and additive)
    log_axes = controlled_axes | (set(rotation_additive.keys()) if has_rot_add else set())

    for step, orig_action in enumerate(recorded_actions):
        if stop_requested():
            break
        img, wrist = get_obs_images(obs)
        if save_frames:
            frames.append(img)

        action = orig_action.copy()

        # Quaternion P-control for controlled axes
        if controlled_axes and target_quat is not None:
            current_quat = _get_ee_quat_wxyz(obs)
            rot_error_vec = _quat_error_to_world_rotvec(target_quat, current_quat)
            # rot_error_vec[0]=errX(drx), [1]=errY(dry), [2]=errZ(drz)
            for i in controlled_axes:
                action[i] = np.clip(rot_error_vec[i - 3] * _K_P_ROT, -_MAX_ROT_CMD, _MAX_ROT_CMD)

        # Additive ramp for rotation (preserves VLA dynamics)
        if has_rot_add and step >= ramp_start:
            for i, delta in rot_add_step.items():
                action[i] += delta

        # Blend translation offset during second half of replay
        if has_trans and step >= ramp_start:
            action[:3] = action[:3] + trans_step_delta

        if step % 50 == 0:
            axis_names = {3: "drx", 4: "dry", 5: "drz"}
            if controlled_axes and target_quat is not None:
                current_quat_log = _get_ee_quat_wxyz(obs)
                rot_err = _quat_error_to_world_rotvec(target_quat, current_quat_log)
                parts = [f"err_{axis_names[i]}={np.degrees(rot_err[i-3]):.1f}°, "
                         f"{axis_names[i]}_cmd={action[i]:+.4f}"
                         for i in sorted(log_axes) if i in controlled_axes]
            else:
                parts = []
            # Log additive axes too
            for i in sorted(log_axes):
                if i not in controlled_axes:
                    parts.append(f"{axis_names[i]}_cmd={action[i]:+.4f} (additive)")
            current_euler = _get_ee_euler_rad(obs)
            euler_str = f"euler=[{np.degrees(current_euler[0]):.0f}, {np.degrees(current_euler[1]):.0f}, {np.degrees(current_euler[2]):.0f}]"
            status = ", ".join(parts) if parts else "no correction axes"
            logging.info(f"  Step {step}/{n_steps}: {status}, {euler_str}")

        if collect_data:
            img_r = resize_for_policy(img)
            wrist_r = resize_for_policy(wrist)
            state = np.concatenate((obs["robot0_joint_pos"], obs["robot0_gripper_qpos"][:1]))
            if step % downsample_hz == 0:
                pending_obs = {"image": img_r.copy(), "wrist": wrist_r.copy(), "state": state.copy()}
                accumulated_action = action.copy()
            else:
                accumulated_action[:6] += action[:6]
                accumulated_action[6] = action[6]
            if (step + 1) % downsample_hz == 0 and pending_obs:
                datapoints.append(FlywheelDatapoint(
                    pending_obs["image"], pending_obs["wrist"], pending_obs["state"],
                    accumulated_action.copy(), "vlm_replay", "replay_with_correction",
                    step_in_primitive=step // downsample_hz,
                ))
                pending_obs = None

        if force_grip_closed:
            action[6] = 1.0

        try:
            obs, _, done, _ = env.step(action.tolist() if hasattr(action, 'tolist') else list(action))
        except ValueError as e:
            if "terminated episode" in str(e):
                episode_done = True
                break
            raise
        if done:
            episode_done = True
            break

        if early_stop_fn is not None and step % 10 == 9 and step > 20:
            if early_stop_fn(env, obs):
                logging.info(f"  [Replay] Early stop at step {step + 1}/{n_steps} (goal reached)")
                break

    return frames, obs, datapoints, episode_done


def _return_to_ee_pose(env, obs, target_pos, target_quat, max_steps=400,
                       pos_tol=0.002, rot_tol_deg=3.0, save_frames=False,
                       max_rot_cmd=None, gripper_cmd=1.0):
    """Two-phase quaternion P-control the EE back to a saved pose.

    Uses quaternion error → world-frame rotation vector for the rotation
    commands, avoiding Euler-angle gimbal lock and axis confusion at
    non-zero yaw.

    Phase 1: Correct rotation (with gentle position hold) until rotation
             error is within ``rot_switch_deg``.
    Phase 2: Correct position while holding rotation.

    Args:
        target_quat: [w,x,y,z] target quaternion.
        save_frames: if True, return (frames, obs, episode_done).
        max_rot_cmd: override for _MAX_ROT_CMD (use lower value for
            slower, smoother rotation that reduces block slippage).

    Returns (obs, episode_done) or (frames, obs, episode_done) if save_frames.
    """
    rot_tol = np.radians(rot_tol_deg)
    rot_switch = np.radians(15.0)
    rot_cmd_limit = max_rot_cmd if max_rot_cmd is not None else _MAX_ROT_CMD
    episode_done = False
    phase = 1
    frames = [] if save_frames else None

    target_euler_deg = _quat_to_euler_deg(target_quat)
    logging.info(
        f"  [Return to pose] Target pos=[{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}], "
        f"euler=[{target_euler_deg[0]:.0f}, {target_euler_deg[1]:.0f}, {target_euler_deg[2]:.0f}]"
    )

    for step in range(max_steps):
        if stop_requested():
            break

        current_pos = obs["robot0_eef_pos"][:3]
        current_quat = _get_ee_quat_wxyz(obs)

        pos_error = target_pos - current_pos
        rot_error_vec = _quat_error_to_world_rotvec(target_quat, current_quat)
        rot_error_mag = np.linalg.norm(rot_error_vec)

        # Transition from phase 1 → 2 once rotation is close
        if phase == 1 and rot_error_mag < rot_switch:
            phase = 2
            logging.info(
                f"  [Return to pose] Step {step}: rotation settled ({np.degrees(rot_error_mag):.1f}°) "
                f"— switching to phase 2 (position + rotation)"
            )

        # Check full convergence
        if np.all(np.abs(pos_error) < pos_tol) and rot_error_mag < rot_tol:
            logging.info(f"  [Return to pose] Converged after {step} steps")
            break

        action = np.zeros(7)
        if phase == 1:
            # Gentle position hold during rotation to prevent large drift
            action[:3] = np.clip(pos_error * _K_P_TRANS, -_MAX_POS_HOLD_CMD, _MAX_POS_HOLD_CMD)
        else:
            action[:3] = np.clip(pos_error * _K_P_TRANS, -_MAX_TRANS_CMD, _MAX_TRANS_CMD)
        for i in range(3):
            action[i + 3] = np.clip(rot_error_vec[i] * _K_P_ROT, -rot_cmd_limit, rot_cmd_limit)
        action[6] = gripper_cmd  # use param: 1.0=closed (lego), -1.0=open (drawer push)

        if step % 50 == 0:
            pos_mm = pos_error * 1000
            current_euler = _get_ee_euler_rad(obs)
            current_euler_deg = np.degrees(current_euler)
            logging.info(
                f"  [Return to pose] Step {step} (phase {phase}): "
                f"pos_err=[{pos_mm[0]:.1f}, {pos_mm[1]:.1f}, {pos_mm[2]:.1f}]mm, "
                f"rot_err={np.degrees(rot_error_mag):.1f}°, "
                f"euler=[{current_euler_deg[0]:.0f}, {current_euler_deg[1]:.0f}, {current_euler_deg[2]:.0f}]"
            )

        try:
            obs, _, done, _ = env.step(action.tolist())
        except ValueError as e:
            if "terminated episode" in str(e):
                episode_done = True
                break
            raise
        if save_frames:
            img, _ = get_obs_images(obs)
            frames.append(img)
        if done:
            episode_done = True
            break

    final_pos = obs["robot0_eef_pos"][:3]
    final_quat = _get_ee_quat_wxyz(obs)
    pos_err = (target_pos - final_pos) * 1000
    rot_err_vec = _quat_error_to_world_rotvec(target_quat, final_quat)
    rot_err_deg = np.degrees(np.linalg.norm(rot_err_vec))
    final_euler_deg = _quat_to_euler_deg(final_quat)
    logging.info(
        f"  [Return to pose] Done (phase {phase}). Residual: pos=[{pos_err[0]:.1f}, {pos_err[1]:.1f}, {pos_err[2]:.1f}]mm, "
        f"rot={rot_err_deg:.1f}°, euler=[{final_euler_deg[0]:.0f}, {final_euler_deg[1]:.0f}, {final_euler_deg[2]:.0f}]"
    )
    if save_frames:
        return frames, obs, episode_done
    return obs, episode_done


def _apply_concentrated_translation(
    env, obs, translation, corrected_rot, save_frames=True,
):
    """Phase 2: Apply translation correction with closed-loop P-control."""
    target_pos = obs["robot0_eef_pos"][:3] + translation
    max_trans_steps = 100
    frames = []
    episode_done = False
    logging.info(
        f"  Applying translation correction (closed-loop): "
        f"target=[{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}], "
        f"delta=[{translation[0]*1000:.1f}, {translation[1]*1000:.1f}, {translation[2]*1000:.1f}]mm, "
        f"up to {max_trans_steps} steps"
    )

    for t_step in range(max_trans_steps):
        if stop_requested():
            break
        error = target_pos - obs["robot0_eef_pos"][:3]
        if np.all(np.abs(error) < 0.001):
            logging.info(f"  Translation converged after {t_step} steps")
            break

        action = np.zeros(7)
        action[:3] = np.clip(error * _K_P_TRANS, -_MAX_TRANS_CMD, _MAX_TRANS_CMD)
        action[6] = -1.0

        current_euler = _get_ee_euler_rad(obs)
        for i, target_final in corrected_rot.items():
            rot_error = _angle_diff(target_final, current_euler[i - 3])
            action[i] = np.clip(rot_error * _K_P_ROT, -_MAX_ROT_CMD, _MAX_ROT_CMD)

        try:
            obs, _, done, _ = env.step(action.tolist())
        except ValueError as e:
            if "terminated episode" in str(e):
                episode_done = True
                break
            raise
        if done:
            episode_done = True
            break

        if save_frames:
            img, _ = get_obs_images(obs)
            frames.append(img)

        if t_step % 20 == 19:
            err_mm = error * 1000
            logging.info(
                f"    Step {t_step+1}: error=[{err_mm[0]:.1f}, {err_mm[1]:.1f}, {err_mm[2]:.1f}]mm"
            )

    final_pos = obs["robot0_eef_pos"][:3]
    residual = target_pos - final_pos
    residual_mm = residual * 1000
    logging.info(
        f"  Final position after translation: "
        f"[{final_pos[0]:.4f}, {final_pos[1]:.4f}, {final_pos[2]:.4f}], "
        f"residual error=[{residual_mm[0]:.1f}, {residual_mm[1]:.1f}, {residual_mm[2]:.1f}]mm"
    )

    return frames, obs, episode_done, residual


def _replay_with_correction(
    env, recorded_actions, obs, total_offset,
    hold_rotation=False, force_grip_closed=False,
    additive_rotation=None,
    lock_axes=None,
    save_frames=True, collect_data=False, downsample_hz=2,
    early_stop_fn=None,
):
    """Replay recorded VLA actions with correction on specified axes.

    Uses quaternion P-control for rotation (avoids Euler gimbal lock).

    Rotation modes:
    - hold_rotation=True: P-control ALL rotation axes at initial + offset
    - lock_axes: set of axis indices (3,4,5) to always P-control, even if
      offset is 0 (e.g. lock_axes={5} to hold drz at initial yaw)
    - additive_rotation: dict {axis_idx: total_offset_rad} — additive ramp
      on those axes (preserves VLA dynamics). Axes in additive_rotation are
      excluded from P-control.
    """
    if obs is None:
        obs = settle_physics(env)

    additive_axes = set(additive_rotation.keys()) if additive_rotation else set()
    lock_axes = lock_axes or set()

    # Determine which axes to P-control
    controlled_axes = set()
    for i in range(3, 6):
        if i in additive_axes:
            continue
        if abs(total_offset[i]) > 1e-6 or hold_rotation or i in lock_axes:
            controlled_axes.add(i)

    # Build target quaternion from initial quat + world-frame offsets
    initial_quat = _get_ee_quat_wxyz(obs)
    initial_euler = _get_ee_euler_rad(obs)
    target_quat = _build_target_quat(initial_quat, total_offset)

    has_trans = np.any(np.abs(total_offset[:3]) > 1e-6)

    if controlled_axes:
        target_euler_deg = _quat_to_euler_deg(target_quat)
        axis_names = {3: "drx", 4: "dry", 5: "drz"}
        offsets_str = ", ".join(f"{axis_names[i]}={np.degrees(total_offset[i]):.0f}°" for i in sorted(controlled_axes))
        logging.info(f"  Quat P-control on {{{offsets_str}}} → target euler=[{target_euler_deg[0]:.0f}, {target_euler_deg[1]:.0f}, {target_euler_deg[2]:.0f}]")
    if controlled_axes or additive_rotation:
        logging.info(f"  Initial euler (deg): roll={np.degrees(initial_euler[0]):.0f}, pitch={np.degrees(initial_euler[1]):.0f}, yaw={np.degrees(initial_euler[2]):.0f}")
    if has_trans:
        logging.info(f"  Translation correction (mm): [{total_offset[0]*1000:.1f}, {total_offset[1]*1000:.1f}, {total_offset[2]*1000:.1f}] (blended into replay)")

    before_img, wrist_before = get_obs_images(obs)
    frames = [before_img] if save_frames else []

    translation = total_offset[:3] if has_trans else None
    replay_frames, obs, datapoints, episode_done = _replay_vla_with_rotation(
        env, recorded_actions, obs,
        target_quat=target_quat,
        controlled_axes=controlled_axes,
        translation_offset=translation,
        rotation_additive=additive_rotation,
        save_frames=save_frames, collect_data=collect_data, downsample_hz=downsample_hz,
        force_grip_closed=force_grip_closed,
        early_stop_fn=early_stop_fn,
    )
    frames.extend(replay_frames)

    after_img, wrist_after = get_obs_images(obs)
    return before_img, after_img, wrist_before, wrist_after, frames, obs, datapoints, episode_done, np.zeros(3)


# =============================================================================
# Correction parsing
# =============================================================================

def _parse_correction(correction: str) -> np.ndarray:
    """Parse VLM correction string into a delta action [dx,dy,dz,drx,dry,drz,grip]."""
    action = np.zeros(7)
    action[6] = -1.0

    axis_map = {"dx": 0, "dy": 1, "dz": 2, "drx": 3, "dry": 4, "drz": 5}

    for match in re.finditer(r"(d[rxyz]{1,2})\s+by\s+([+-]?\d+(?:\.\d+)?)\s*(deg|degrees|mm|rad)?", correction, re.IGNORECASE):
        axis_name = match.group(1).lower()
        value = float(match.group(2))
        unit = (match.group(3) or "").lower()

        if axis_name in axis_map:
            idx = axis_map[axis_name]
            if idx >= 3:
                if unit in ("deg", "degrees", ""):
                    value = np.radians(value)
            else:
                if unit in ("mm", ""):
                    value = value / 1000.0
            action[idx] = value

    return action
