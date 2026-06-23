"""
XArm control loop with policy inference over the network.

Run the policy on a GPU machine (or same machine) with:

    uv run training/serve_policy.py policy:checkpoint \\
        --policy.config=xarm_scoop \\
        --policy.dir=/path/to/checkpoint \\
        --default-prompt='scoop up the rocks'

Then on the robot / edge machine:

    uv run real/entry/inference.py --host <server_ip> --port 8000

Action / observation contract (must match ``real/entry/collect_demo.py``):

- ``joint_position`` and policy ``actions`` are the same 6D pose: ``[x, y, z, roll, pitch, yaw]``
  with xyz in mm (from ``get_position()``) and euler angles in **radians**.
- Images: ``exterior_image_1_left`` = external camera, ``wrist_image_left`` = wrist camera
  (same serials / order as ``read_cameras()``). PI05 does not need ``exterior_image_2_left`` at
  inference (training used zeros; the policy stack pads a third view).
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
# User inputs
# =========================
# Keep in sync with collect_demo.TASK_DESCRIPTION (LeRobot ``task`` field → training prompt).
TASK_DESCRIPTION = "scoop up the rocks"

FPS = 20.0
DT = 1.0 / FPS
CONTROL_HZ = 60.0  # multiple of 10
PREDICTION_HORIZON = 10
MIN_EXECUTION_HORIZON = 5
# Policy outputs 6D absolute pose; see XarmOutputs and collect_demo ``actions`` / ``joint_position``.
ROBOT_DOF = 6

mutex = threading.Lock()
condition_variable = threading.Condition(mutex)

delay_init = 5
buffer_size = 5

SCOOP_EFF_LENGTH = np.sqrt(160**2 + 85**2)
phi = np.arctan(85.0 / 160.0)

SERIAL_EXTERNAL = "244222071219"
SERIAL_WRIST = "317222072257"

ARM_IP = "192.168.1.219"

# Populated in main()
policy: websocket_client_policy.WebsocketClientPolicy | None = None
arm: XArmAPI | None = None
camera_pipelines: dict[str, rs.pipeline] = {}

t = 0
observation_curr: dict | None = None
action_curr: np.ndarray | None = None

# Set from CLI in ``main``; used by ``get_observation``.
_run_prompt: str = TASK_DESCRIPTION


@dataclasses.dataclass
class Args:
    """Where to reach `training/serve_policy.py` (websocket policy server)."""

    host: str = "localhost"
    port: int = 8000
    api_key: str | None = None
    # Should match the task string used when recording demos.
    prompt: str = TASK_DESCRIPTION

# =========================
# RealSense Camera Setup
# =========================
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


# =========================
# Observation
# =========================
def get_observation() -> dict:
    assert arm is not None and camera_pipelines
    frames_exterior = camera_pipelines[SERIAL_EXTERNAL].wait_for_frames()
    frames_wrist = camera_pipelines[SERIAL_WRIST].wait_for_frames()

    wrist = frames_wrist.get_color_frame()
    exterior = frames_exterior.get_color_frame()

    a = np.asanyarray(wrist.get_data())
    b = np.asanyarray(exterior.get_data())

    pose = arm.get_position()[1]
    pose[3] = pose[3] % 360
    pose[5] = pose[5] % 360
    angles_rad = (np.array(pose[3:6]) * np.pi / 180).tolist()
    state = np.array(pose[:3] + angles_rad, dtype=np.float32)

    return {
        "observation/exterior_image_1_left": b,
        "observation/wrist_image_left": a,
        "observation/state": state,
        "prompt": _run_prompt,
    }


# =========================
# Command Interpolation
# =========================
def interpolate_action(state: np.ndarray, goal: np.ndarray) -> None:
    assert arm is not None
    
    delta_increment = (goal - state) / (DT * CONTROL_HZ * 6)
    delta_increment[2] = delta_increment[2] * 0.6
    for _ in range(int(DT * CONTROL_HZ)):
        start_time = time.perf_counter()
        state = state + delta_increment
        command = state.copy()
        command[3] = (command[3] + 180) % 360 - 180
        command[5] = (command[5] + 180) % 360 - 180

        x, y, z, roll, pitch, yaw = command
        
        print("Interpolation:")
        print(x, y, z, roll, pitch, yaw)

        #command[2] = np.clip(z, 195, None)
        
        arm.set_servo_cartesian(command, speed=100, mvacc=1000)

        time_left = (1 / CONTROL_HZ) - (time.perf_counter() - start_time)
        time.sleep(max(time_left, 0))


# =========================
# Action Getter
# =========================
def get_action(observation_next: dict) -> np.ndarray:
    global t, observation_curr, action_curr
    assert action_curr is not None

    with condition_variable:
        t += 1

        observation_curr = observation_next
        condition_variable.notify()

        action = action_curr[t - 1, :].copy()

    return action


# =========================
# Guided Inference
# =========================
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
        action_prev = np.pad(action_prev, ((0, H - T), (0, 0)), mode='constant')

    v_pi = np.array(pol.infer(observation)["actions"])
    v_pi = v_pi[:H, :ROBOT_DOF]

    A = action_prev.copy()
    action_estimate = A * W[:, None] + v_pi * (1 - W[:, None])

    return action_estimate[:H, :ROBOT_DOF]


# =========================
# Inference Loop
# =========================
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


# =========================
# Execution Loop
# =========================
def execution_loop() -> None:
    global t
    assert arm is not None and action_curr is not None

    while True:
        print("t:", t)
        t0 = time.perf_counter()

        observation = get_observation()
        command = get_action(observation)

        cmd_joint_pose = command[:6].copy()
        cmd_joint_pose[3:6] = cmd_joint_pose[3:6] / np.pi * 180

        pose = arm.get_position()[1]
        pose[3] = pose[3] % 360
        pose[5] = pose[5] % 360
        state = np.array(pose, dtype=np.float32)

        print("Current pose:")
        print(pose)
        print("Command pose:")
        print(cmd_joint_pose)

        interpolate_action(state, cmd_joint_pose)

        time_left = DT - (time.perf_counter() - t0)
        time.sleep(max(time_left, 0))


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

    print("Starting control system...")

    infer_thread = threading.Thread(target=inference_loop, daemon=True)
    exec_thread = threading.Thread(target=execution_loop, daemon=True)

    infer_thread.start()
    exec_thread.start()

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main(tyro.cli(Args))
