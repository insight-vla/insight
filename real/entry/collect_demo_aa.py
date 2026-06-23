import time
import sys
import select
import numpy as np
import pyrealsense2 as rs
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.constants import HF_LEROBOT_HOME
from xarm.wrapper import XArmAPI
from pathlib import Path

# ------------------------
# Config
# ------------------------
REPO_NAME = "maggie/xarm_scoop_aa"
FPS = 20.0
DT = 1.0 / FPS
ARM_IP = "192.168.1.219"

TASK_DESCRIPTION = "scoop up the rocks"

SERIAL_EXTERNAL = '244222071219'
SERIAL_WRIST = '317222072257'

START_FLAG = Path("/tmp/start_demo")
STOP_FLAG  = Path("/tmp/stop_demo")

if START_FLAG.exists():
    START_FLAG.unlink()

if STOP_FLAG.exists():
    STOP_FLAG.unlink()


# ------------------------
# Init robot + cameras
# ------------------------
arm = XArmAPI(ARM_IP)
arm.connect()

# Connect to cameras
camera_pipelines = {}

# Enable streams only for our specific serials
for serial in [SERIAL_EXTERNAL, SERIAL_WRIST]:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 320, 240, rs.format.rgb8, 30)
    
    try:
        pipeline.start(config)
        camera_pipelines[serial] = pipeline
        print(f"Started camera: {serial}")
    except Exception as e:
        print(f"Failed to start camera {serial}: {e}")
        sys.exit(1)

def read_cameras():
    # Fetch frames by serial number to ensure no swapping
    frames_exterior = camera_pipelines[SERIAL_EXTERNAL].wait_for_frames()
    frames_wrist = camera_pipelines[SERIAL_WRIST].wait_for_frames()
    wrist = frames_wrist.get_color_frame()
    exterior = frames_exterior.get_color_frame()
    wrist = np.asanyarray(wrist.get_data())
    exterior = np.asanyarray(exterior.get_data())
    exterior2 = np.zeros_like(exterior)
    return wrist, exterior, exterior2


def read_eef_state(arm):
    """Read 6D EEF state: [x(mm), y(mm), z(mm), rx(rad), ry(rad), rz(rad)]."""
    code_pose, pose_aa = arm.get_position_aa(is_radian=True)
    if code_pose != 0:
        raise RuntimeError(f"get_position_aa failed with code={code_pose}")

    return np.array(list(pose_aa[:6]), dtype=np.float32)


def _axis_angle_to_quat(axis_angle):
    """Axis-angle vector (axis * angle) -> unit quaternion [w, x, y, z]."""
    v = np.asarray(axis_angle, dtype=np.float64)
    theta = np.linalg.norm(v)
    if theta < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = v / theta
    half = 0.5 * theta
    s = np.sin(half)
    return np.array([np.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float64)


def _quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_multiply(q1, q2):
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


def _quat_to_axis_angle(q):
    """Unit quaternion [w, x, y, z] -> axis-angle vector (axis * angle)."""
    q = np.asarray(q, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    # Enforce shortest-path representation (angle in [0, pi]).
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


def pose_delta_axis_angle(prev_state, curr_state):
    """Compute 6D delta with proper SO(3) relative orientation."""
    prev = np.asarray(prev_state, dtype=np.float64)
    curr = np.asarray(curr_state, dtype=np.float64)
    dpos = curr[:3] - prev[:3]
    q_prev = _axis_angle_to_quat(prev[3:6])
    q_curr = _axis_angle_to_quat(curr[3:6])
    q_rel = _quat_multiply(q_curr, _quat_conjugate(q_prev))
    dori = _quat_to_axis_angle(q_rel)
    return np.array(np.concatenate([dpos, dori]), dtype=np.float32)

def timed_input(prompt, timeout, default="y"):
    print(prompt, end="", flush=True)
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if ready:
        return sys.stdin.readline().strip().lower()
    else:
        print(f"\nNo response after {timeout}s → defaulting to '{default}'")
        return default

# ------------------------
# Create dataset
# ------------------------

dataset_path = HF_LEROBOT_HOME / REPO_NAME

if dataset_path.exists(): 
    dataset = LeRobotDataset(
        root=dataset_path,
        repo_id=REPO_NAME,
    )
    print("Adding to existing dataset, waiting for signal.")
else:
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="xarm",
        fps=FPS,
        features={
            "exterior_image_1_left": {
                "dtype": "image",
                "shape": (240, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "exterior_image_2_left": { # this one is not used, put it as zeros or something
                "dtype": "image",
                "shape": (240, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (240, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (6,),  # We will use joint *velocity* actions here (6D)
                "names": ["actions"],
            },
        },
    )
    print("Dataset created, waiting for start signal.")

# ------------------------
# Collect episode
# ------------------------

recording = False
prev_data = None  # Buffer to hold the observation for time t

try:
    while True:
        if START_FLAG.exists() and not recording:
            START_FLAG.unlink()
            print("Starting demo")
            recording = True
            prev_data = None # Reset buffer for new demo

        if STOP_FLAG.exists() and recording:
            STOP_FLAG.unlink()
            print("Ending demo")
            
            recording = False
            prev_data = None # Clear buffer
            resp = timed_input("Save this demo? [y/n]: ", timeout=6, default="y")

            if resp == "y":
                dataset.save_episode()
                print("Episode saved")
            else:
                dataset = LeRobotDataset(root=dataset_path, repo_id=REPO_NAME)
                print("Episode discarded")

        if not recording:
            time.sleep(0.05)
            continue

        start = time.perf_counter()

        # 1) Capture current state at time t+1
        curr_state = read_eef_state(arm)
        wrist, base, base2 = read_cameras()

        # 2) If we have a previous observation, write (obs_t, action_t) where
        # action_t is the delta that moved from state_t to state_{t+1}
        if prev_data is not None:
            action_delta = pose_delta_axis_angle(prev_data["state"], curr_state)
            dataset.add_frame(
                {
                    "state": prev_data["state"][:6],
                    "actions": action_delta,
                    "exterior_image_1_left": prev_data["base"],
                    "exterior_image_2_left": prev_data["base2"],
                    "wrist_image_left": prev_data["wrist"],
                    "task": TASK_DESCRIPTION,
                }
            )

        # 3) Store current observation/state to pair with the next state's delta
        prev_data = {
            "state": curr_state,
            "wrist": wrist,
            "base": base,
            "base2": base2
        }

        # ---- Timing ----
        elapsed = time.perf_counter() - start
        time.sleep(max(0.0, DT - elapsed))

except KeyboardInterrupt:
    print("Shutting down")

finally:
    for p in camera_pipelines.values():
        p.stop()
    arm.disconnect()