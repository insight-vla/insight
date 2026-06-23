"""
XArm control with policy inference — **axis-angle state + dataset-style SO(3) deltas**
(must match ``real/entry/collect_demo_aa.py`` and ``xarm_scoop_aa``).

Deltas are **not** applied with ``relative=True`` on the arm API. They are integrated with the
same quaternion math as ``pose_delta_axis_angle`` in ``collect_demo_aa.py``, then sent as
**absolute** axis-angle poses (``set_servo_cartesian_aa(..., relative=False)``), with orientation
interpolated by **slerp** (not by splitting the axis-angle vector).

Policy server:

    uv run training/serve_policy.py policy:checkpoint \\
        --policy.config=xarm_scoop_aa \\
        --policy.dir=/path/to/checkpoint

Robot:

    uv run real/entry/inference_aa.py --host <ip> --port 8000
"""

import dataclasses
import logging
import sys
import threading
import time
from collections import deque

import numpy as np
import pyrealsense2 as rs
import tyro
from openpi_client import websocket_client_policy
from xarm.wrapper import XArmAPI


# =========================
# User inputs (align with collect_demo_aa.py)
# =========================
TASK_DESCRIPTION = "scoop up the rocks"

FPS = 5.0
DT = 1.0 / FPS
CONTROL_HZ = 60.0
PREDICTION_HORIZON = 20
MIN_EXECUTION_HORIZON = 10
ROBOT_DOF = 6

mutex = threading.Lock()
condition_variable = threading.Condition(mutex)

delay_init = 5
buffer_size = 5

SERIAL_EXTERNAL = "244222071219"
SERIAL_WRIST = "317222072257"

ARM_IP = "192.168.1.219"

policy: websocket_client_policy.WebsocketClientPolicy | None = None
arm: XArmAPI | None = None
camera_pipelines: dict[str, rs.pipeline] = {}

t = 0
observation_curr: dict | None = None
action_curr: np.ndarray | None = None

_run_prompt: str = TASK_DESCRIPTION


@dataclasses.dataclass
class Args:
    """Where to reach ``training/serve_policy.py`` (must load ``xarm_scoop_aa`` checkpoint)."""

    host: str = "localhost"
    port: int = 8000
    api_key: str | None = None
    prompt: str = TASK_DESCRIPTION


# ----- Quaternion / axis-angle (identical convention to collect_demo_aa.py) -----


def read_eef_state(api: XArmAPI) -> np.ndarray:
    """[x(mm), y, z, rx, ry, rz] axis-angle orientation, radians."""
    code_pose, pose_aa = api.get_position_aa(is_radian=True)
    if code_pose != 0:
        raise RuntimeError(f"get_position_aa failed with code={code_pose}")
    return np.array(list(pose_aa[:6]), dtype=np.float32)


def _axis_angle_to_quat(axis_angle: np.ndarray) -> np.ndarray:
    """Axis-angle vector (axis * angle) -> unit quaternion [w, x, y, z]."""
    v = np.asarray(axis_angle, dtype=np.float64)
    theta = np.linalg.norm(v)
    if theta < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = v / theta
    half = 0.5 * theta
    s = np.sin(half)
    return np.array([np.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float64)


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """Unit quaternion [w, x, y, z] -> axis-angle vector (axis * angle)."""
    q = np.asarray(q, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    if q[0] < 0:
        q = -q
    w = np.clip(q[0], -1.0, 1.0)
    v = q[1:]
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * np.arctan2(v_norm, w)
    axis = v / v_norm
    return axis * angle


def apply_pose_delta(prev_state: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Inverse of ``pose_delta_axis_angle``: absolute next pose from current + dataset delta.

    Uses ``q_curr = q_rel * q_prev`` with ``q_rel = axis_angle_to_quat(delta[3:6])``,
    matching ``q_rel = q_curr * q_prev^{-1}`` used when recording demos.
    """
    prev = np.asarray(prev_state, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    curr = np.zeros(6, dtype=np.float64)
    curr[:3] = prev[:3] + delta[:3]
    q_prev = _axis_angle_to_quat(prev[3:6])
    q_rel = _axis_angle_to_quat(delta[3:6])
    q_curr = _quat_multiply(q_rel, q_prev)
    curr[3:6] = _quat_to_axis_angle(q_curr)
    return curr.astype(np.float32)


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between unit quaternions."""
    q0 = q0 / (np.linalg.norm(q0) + 1e-12)
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if abs(dot) > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / (np.linalg.norm(q) + 1e-12)
    theta0 = np.arccos(dot)
    sin_theta0 = np.sin(theta0)
    q = (np.sin((1.0 - t) * theta0) / sin_theta0) * q0 + (np.sin(t * theta0) / sin_theta0) * q1
    return q / (np.linalg.norm(q) + 1e-12)


def _start_cameras() -> None:
    global camera_pipelines
    camera_pipelines = {}
    for serial in [SERIAL_EXTERNAL, SERIAL_WRIST]:
        pipeline = rs.pipeline()
        rs_cfg = rs.config()
        rs_cfg.enable_device(serial)
        rs_cfg.enable_stream(rs.stream.color, 320, 240, rs.format.rgb8, 30)

        try:
            pipeline.start(rs_cfg)
            camera_pipelines[serial] = pipeline
            logging.info("Started camera: %s", serial)
        except Exception as e:
            logging.error("Failed to start camera %s: %s", serial, e)
            sys.exit(1)


def get_observation() -> dict:
    assert arm is not None and camera_pipelines
    frames_exterior = camera_pipelines[SERIAL_EXTERNAL].wait_for_frames()
    frames_wrist = camera_pipelines[SERIAL_WRIST].wait_for_frames()

    wrist = frames_wrist.get_color_frame()
    exterior = frames_exterior.get_color_frame()

    a = np.asanyarray(wrist.get_data())
    b = np.asanyarray(exterior.get_data())

    state = read_eef_state(arm)

    return {
        "observation/exterior_image_1_left": b,
        "observation/wrist_image_left": a,
        "observation/state": state,
        "prompt": _run_prompt,
    }


def interpolate_delta_action(delta: np.ndarray, start_state: np.ndarray) -> None:
    """Integrate one policy delta like the demos, then move along a straight line in xyz + slerp in SO(3).

    Uses the same ``start_state`` the policy saw (from ``get_observation``), not a fresh read
    mid-path, so integration matches conditioning.
    """
    assert arm is not None
    start = np.asarray(start_state, dtype=np.float64)
    goal = apply_pose_delta(start, np.asarray(delta, dtype=np.float64))

    n_steps = max(1, int(DT * CONTROL_HZ))
    q_start = _axis_angle_to_quat(start[3:6])
    q_goal = _axis_angle_to_quat(goal[3:6])

    for i in range(1, n_steps + 1):
        t0 = time.perf_counter()
        alpha = i / n_steps
        pos = (1.0 - alpha) * start[:3] + alpha * goal[:3]
        q = _slerp(q_start, q_goal, alpha)
        aa = _quat_to_axis_angle(q)
        pose = np.concatenate([pos, aa]).astype(np.float32)
        print(pose)
        
        code = arm.set_servo_cartesian_aa(
            pose.tolist(),
            speed=100,
            mvacc=1000,
            is_radian=True,
            relative=False,
        )
        if code != 0:
            logging.warning("set_servo_cartesian_aa returned code=%s", code)
        

        time_left = (1.0 / CONTROL_HZ) - (time.perf_counter() - t0)
        time.sleep(max(time_left, 0.0))


def get_action(observation_next: dict) -> np.ndarray:
    global t, observation_curr, action_curr
    assert action_curr is not None

    with condition_variable:
        t += 1
        observation_curr = observation_next
        condition_variable.notify()
        action = action_curr[t - 1, :].copy()

    return action


def guided_inference(
    pol: websocket_client_policy.WebsocketClientPolicy,
    observation: dict,
    action_prev: np.ndarray,
    delay: int,
    time_since_last_inference: int,
) -> np.ndarray:
    H = PREDICTION_HORIZON
    i = np.arange(delay, H - time_since_last_inference)
    c = (H - time_since_last_inference - i) / (H - time_since_last_inference - delay + 1)

    W = np.ones(H)
    W[0:delay] = 1.0
    W[delay : H - time_since_last_inference] = c * (np.exp(c) - 1) / (np.exp(1) - 1)
    W[H - time_since_last_inference :] = 0.0

    T, robot_dof = action_prev.shape
    if T < H:
        action_prev = np.pad(action_prev, ((0, H - T), (0, 0)), mode="constant")

    v_pi = np.array(pol.infer(observation)["actions"])
    v_pi = v_pi[:H, :ROBOT_DOF]

    A = action_prev.copy()
    action_estimate = A * W[:, None] + v_pi * (1 - W[:, None])

    return action_estimate[:H, :ROBOT_DOF]


def inference_loop() -> None:
    global t, action_curr, observation_curr
    assert policy is not None and action_curr is not None and observation_curr is not None

    Q = deque([delay_init], maxlen=buffer_size)

    while True:
        with condition_variable:
            while t < MIN_EXECUTION_HORIZON:
                condition_variable.wait()

            time_since_last_inference = t
            action_prev = action_curr[time_since_last_inference:PREDICTION_HORIZON].copy()

            delay = max(Q)
            print("Delay: ", delay)
            obs = observation_curr.copy()

        action_new = guided_inference(policy, obs, action_prev, delay, time_since_last_inference)

        action_curr[: action_new.shape[0], :] = action_new
        t = t - time_since_last_inference
        Q.append(t)


def execution_loop() -> None:
    global t
    assert arm is not None and action_curr is not None

    while True:
        print("t:", t)
        t0 = time.perf_counter()

        observation = get_observation()
        state_at_obs = observation["observation/state"].copy()
        delta = get_action(observation)

        print("Delta (dxyz mm, daxis rad):", delta)

        interpolate_delta_action(delta, state_at_obs)

        time_left = DT - (time.perf_counter() - t0)
        time.sleep(max(time_left, 0.0))


def main(args: Args) -> None:
    global policy, arm, t, observation_curr, action_curr, _run_prompt

    logging.basicConfig(level=logging.INFO, force=True)
    _run_prompt = args.prompt

    policy = websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
    )
    logging.info("Connected to policy server; metadata: %s", policy.get_server_metadata())

    arm = XArmAPI(ARM_IP)
    arm.connect()

    if arm.get_state() != 0:
        arm.clean_error()
        time.sleep(0.5)

    arm.motion_enable(enable=True)
    arm.set_mode(1)
    arm.set_state(0)

    _start_cameras()

    t = 0
    observation_curr = get_observation()
    action_curr = np.array(policy.infer(observation_curr)["actions"], dtype=np.float32)

    print("Starting control system (SO(3) delta integration + absolute servo AA)...")

    infer_thread = threading.Thread(target=inference_loop, daemon=True)
    exec_thread = threading.Thread(target=execution_loop, daemon=True)

    infer_thread.start()
    exec_thread.start()

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main(tyro.cli(Args))
