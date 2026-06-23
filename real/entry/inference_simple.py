import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
import time
import tyro
import logging
import sys

import dataclasses
from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi_client import websocket_client_policy

FPS = 20.0
DT = 1.0 / FPS # your timestep
CONTROL_HZ = 40.0 # keep as a multiple of 10
# ACTION_ROLLOUT = 50
ACTION_ROLLOUT = 10

SERIAL_EXTERNAL = "244222071219"
SERIAL_WRIST = "317222072257"
TASK_DESCRIPTION = "sweep the rocks"
ARM_IP = "192.168.1.219"

# Populated in main()
policy: websocket_client_policy.WebsocketClientPolicy | None = None
arm: XArmAPI | None = None
camera_pipelines: dict[str, rs.pipeline] = {}
_run_prompt: str = TASK_DESCRIPTION

@dataclasses.dataclass
class Args:
    """Where to reach `training/serve_policy.py` (websocket policy server)."""

    host: str = "localhost"
    port: int = 8000
    api_key: str | None = None
    # Should match the task string used when recording demos.
    prompt: str = TASK_DESCRIPTION

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

def interpolate_action(state, goal):
    delta_increment = (goal - state) / (DT * CONTROL_HZ * 6)

    for i in range(int(DT * CONTROL_HZ)):
        start = time.perf_counter()
        command = state + delta_increment
        command[3] = (command[3]+ 180) % 360 -180
        command[5] = (command[5]+ 180) % 360 -180

        x, y, z, roll, pitch, yaw = command
        print(x, y, z, roll, pitch, yaw)
        if z < 198.0:
            print(f"!!!!!!!![Z-CLIP] {z:.2f} -> 198.00")
        command[2] = max(z, 198.0) # 193 for scoop

        arm.set_servo_cartesian(command, speed=100, mvacc=1000)

        time_left = (1 / CONTROL_HZ) - (time.perf_counter() - start)
        time.sleep(max(time_left,0))


def main(args: Args) -> None:
    global policy, arm, _run_prompt

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
    arm.set_collision_sensitivity(5)
    arm.set_mode(1)
    arm.set_state(0)
    _start_cameras()

    while True:
        observation = get_observation()

        print("Running inference")
        action = np.array(policy.infer(observation)["actions"], dtype=np.float32)
        
        count = 0

        init_joints = observation["observation/state"]
        
        while count < ACTION_ROLLOUT:
            # grab current state
            t0 = time.perf_counter()
            pose = arm.get_position()[1]
            pose[3] = pose[3] % 360
            pose[5] = pose[5] % 360
            state = np.array(pose, dtype=np.float32)
            
            # get the target angles
            cmd_joint_pose = np.array(action[count,:6])
            cmd_joint_pose[3:6] = cmd_joint_pose[3:6] / np.pi * 180
            
            # execute smooth motion to target via interpolation
            interpolate_action(state, cmd_joint_pose)

            count += 1
            time_left = DT - (time.perf_counter() - t0)
            
            time.sleep(max(time_left,0))

if __name__ == "__main__":
    main(tyro.cli(Args))


    