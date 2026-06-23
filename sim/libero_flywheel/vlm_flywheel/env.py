"""Environment creation, image utilities, dataclasses."""

from __future__ import annotations

import base64
import dataclasses
import io
import logging
import os
import pathlib
import sys

import numpy as np
from PIL import Image

from openpi_client import image_tools


# =============================================================================
# Constants
# =============================================================================

LIBERO_ENV_RESOLUTION = 256
POLICY_IMAGE_SIZE = 224
VLM_IMAGE_SIZE = 512
VLM_IMAGE_SIZE_SMALL = 256
DEFAULT_PRIMITIVE_STEPS = 150

PRIMITIVE_P90_DURATIONS = {
    # Lego primitives
    "move gripper to the red lego block": 265,
    "move gripper to the blue lego block": 220,
    "close gripper": 11,
    "lift upward": 87,
    "lower gripper": 99,
    "move gripper to target": 36,
    "move gripper to the target zone": 60,
    "open gripper": 9,
    "rotate block": 149,
    # Drawer primitives
    "move gripper to the top drawer handle": 237,
    "pull the top drawer open": 59,
}

# Skill gaps being learned (excluded from AVAILABLE_PRIMITIVES so VLM flags them)
_SKILL_GAPS = {"rotate block", "push the top drawer closed"}

# Primitives the current policy actually knows (exclude skill gaps being learned)
AVAILABLE_PRIMITIVES = [p for p in PRIMITIVE_P90_DURATIONS.keys() if p not in _SKILL_GAPS]

DEFAULT_SEQUENCE = [
    "move gripper to the red lego block",
    "close gripper",
    "lift upward",
    "move gripper to the target zone",
    "lower gripper",
    "open gripper",
]

DEFAULT_SCENE_CONTEXT = "Robot arm with parallel jaw gripper."


# =============================================================================
# Data Classes
# =============================================================================

# Env-agnostic dataclasses (VLMFeedback, TaskPlan, ActionCorrection) live in
# insight.types and are re-exported here for sim callers.
from insight.types import ActionCorrection, TaskPlan, VLMFeedback  # noqa: F401,E402


@dataclasses.dataclass
class FlywheelDatapoint:
    """Single datapoint for flywheel training data (10Hz)."""
    observation_image: np.ndarray
    observation_wrist: np.ndarray
    observation_state: np.ndarray
    action: np.ndarray
    source: str
    primitive: str
    vlm_feedback: str | None = None
    step_in_primitive: int = 0


@dataclasses.dataclass
class Args:
    host: str = "localhost"
    port: int = 8000
    primitive: str = "move gripper to the red lego block"
    sequence: str | None = None
    seed: int = 42
    output_dir: str = "data/libero/vlm_feedback"
    loop: bool = False
    max_retries: int = 3
    stop_on_fail: bool = True
    flywheel: bool = False
    goal: str = "flip the red lego block peg up"
    collect_data: bool = True
    adaptive: bool = False
    max_primitives: int = 15
    vlm: str = "gpt"


# =============================================================================
# Early Stop (press 's' to save and quit)
# =============================================================================

_enable_display = False

# Keyboard early-stop lives in insight.keyboard — re-exported under the
# historical names so existing sim callers (and `vlm_flywheel._stop_event`)
# keep working.
from insight.keyboard import (  # noqa: E402
    _stop_event,
    request_stop as _request_stop,  # noqa: F401
    reset_stop as _reset_stop,  # noqa: F401
    start_stop_listener as _start_keyboard_listener,
    stop_requested,
)


# =============================================================================
# Environment
# =============================================================================

class SimpleVisualizationWrapper:
    """Wrapper that enables robot/gripper visualization."""

    def __init__(self, env):
        self.env = env
        self._enable_viz()

    def _enable_viz(self):
        if hasattr(self.env, "env") and hasattr(self.env.env, "visualize"):
            self.env.env.visualize(vis_settings={"env": True, "robots": True, "grippers": False})

    def reset(self):
        obs = self.env.reset()
        self._enable_viz()
        return obs

    def step(self, action):
        return self.env.step(action)

    def seed(self, seed):
        return self.env.seed(seed)

    @property
    def sim(self):
        return self.env.sim

    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        """Proxy any unhandled attribute lookups to the wrapped env.

        Notably surfaces ``check_success`` (the LIBERO BDDL goal check)
        which our goal-check path now relies on exclusively after the
        privileged-peg fallback was removed. Without this, the call
        path errors with ``'SimpleVisualizationWrapper' object has no
        attribute 'check_success'`` and goal achievement can never be
        detected. Underscore-prefixed names are excluded so we don't
        accidentally swallow attribute errors during pickling, repr,
        or other dunders that should raise normally.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.env, name)


def create_env(bddl_file: pathlib.Path, seed: int, resolution: int = LIBERO_ENV_RESOLUTION):
    """Create and initialize LIBERO environment."""
    from libero.libero.envs import OffScreenRenderEnv
    env = SimpleVisualizationWrapper(OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=resolution,
        camera_widths=resolution,
        horizon=3000,
    ))
    env.seed(seed)
    return env


def settle_physics(env, obs=None, steps: int = 50):
    """Wait for physics to settle.

    Robosuite raises ``ValueError("executing action in terminated
    episode")`` if env.step is called after done=True was returned.
    We swallow that and return the latest obs so the caller can fail
    the trial cleanly instead of crashing the batch.
    """
    if obs is None:
        obs = env.reset()
    for _ in range(steps):
        try:
            obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
        except ValueError as e:
            if "terminated episode" in str(e):
                logging.warning("settle_physics: env terminated, returning current obs")
                return obs
            raise
    return obs


def _find_robot(env):
    """Find the robosuite Robot object through the wrapper chain."""
    current = env
    while current is not None:
        if hasattr(current, "robots") and current.robots:
            return current.robots[0]
        current = getattr(current, "env", None)
    return None


def reset_gripper_pose(env, obs, initial_jpos, initial_gripper_jpos):
    """Reset robot arm to exact frame-0 pose. Objects stay where they are."""
    robot = _find_robot(env)
    if robot is None:
        logging.warning("Could not find robot — skipping reset")
        return obs

    env.sim.data.qpos[robot._ref_joint_pos_indexes] = initial_jpos
    env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes] = initial_gripper_jpos
    env.sim.data.qvel[robot._ref_joint_vel_indexes] = 0.0
    env.sim.data.qvel[robot._ref_gripper_joint_vel_indexes] = 0.0
    env.sim.forward()

    robot._load_controller()
    robot.controller.update_base_pose(robot.base_pos, robot.base_ori)

    obs = settle_physics(env, obs, steps=50)
    return obs


# =============================================================================
# Image Utilities
# =============================================================================

_display_checked = False
_display_works = False
_display_enabled = False  # toggled by set_display_enabled() at startup of flywheel mode


def set_display_enabled(enabled: bool) -> None:
    """Toggle the live cv2 display in ``get_obs_images``. Set once at process startup."""
    global _display_enabled
    _display_enabled = enabled


def _check_display_once():
    """Test if cv2.imshow works by running a subprocess (avoids Qt C-level abort)."""
    global _display_checked, _display_works
    if _display_checked:
        return _display_works
    _display_checked = True
    if not os.environ.get("DISPLAY"):
        logging.info("[display] No DISPLAY set, display disabled.")
        return False
    import subprocess
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import cv2; cv2.namedWindow('_test'); cv2.destroyAllWindows()"],
            timeout=5, capture_output=True,
        )
        _display_works = result.returncode == 0
        if not _display_works:
            stderr = result.stderr.decode(errors="replace").strip()
            logging.warning(f"[display] cv2 display test failed, disabling. stderr: {stderr[:200]}")
        else:
            logging.info("[display] Display available.")
    except Exception as e:
        logging.warning(f"[display] Display test error, disabling: {e}")
    return _display_works


def get_obs_images(obs) -> tuple[np.ndarray, np.ndarray]:
    """Extract and flip images from observation. Shows live display if enabled."""
    img = np.ascontiguousarray(obs["agentview_image"][::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
    if _display_enabled and _check_display_once():
        try:
            import cv2
            h = 256
            img_small = cv2.resize(img, (h, h))
            wrist_small = cv2.resize(wrist, (h, h))
            combined = np.concatenate([img_small, wrist_small], axis=1)
            cv2.imshow("Live", cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        except Exception:
            pass
    return img, wrist


def resize_for_policy(img: np.ndarray) -> np.ndarray:
    """Resize image for policy input."""
    img = np.ascontiguousarray(img[:, ::-1])
    return image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img, POLICY_IMAGE_SIZE, POLICY_IMAGE_SIZE)
    )


def resize_for_vlm(img: np.ndarray, small: bool = False) -> np.ndarray:
    """Resize image for VLM input."""
    size = VLM_IMAGE_SIZE_SMALL if small else VLM_IMAGE_SIZE
    return image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img, size, size)
    )


def crop_around_red_block(img: np.ndarray, padding: int = 40, min_size: int = 80) -> np.ndarray | None:
    """Crop image around the red lego block using color detection."""
    red_mask = (img[:, :, 0] > 150) & (img[:, :, 1] < 80) & (img[:, :, 2] < 80)
    if not red_mask.any():
        return None
    ys, xs = np.where(red_mask)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    h, w = img.shape[:2]
    cy, cx = (y_min + y_max) // 2, (x_min + x_max) // 2
    half = max((y_max - y_min) // 2 + padding, (x_max - x_min) // 2 + padding, min_size // 2)
    y1 = max(0, cy - half)
    y2 = min(h, cy + half)
    x1 = max(0, cx - half)
    x2 = min(w, cx + half)
    return img[y1:y2, x1:x2]


def encode_image_base64(img: np.ndarray) -> str:
    """Convert numpy image to base64 for VLM API."""
    buffer = io.BytesIO()
    Image.fromarray(img).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def get_gripper_state(obs) -> str:
    """Get human-readable gripper state from observation."""
    gripper_qpos = obs["robot0_gripper_qpos"][0]
    if gripper_qpos < 0.03:
        return "CLOSED"
    else:
        return "OPEN"
