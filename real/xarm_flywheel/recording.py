"""LeRobot dataset recording for the xArm flywheel.

Captures per-step ``(state, exterior_image, wrist_image, action)`` tuples in
the same schema as ``real/entry/collect_demo.py`` so flywheel-collected demos
can be fed straight back into VLA fine-tuning. Each successful primitive
becomes one LeRobot episode, tagged with the primitive name as ``task``.

The "action" at frame *t* is the *t+1* state (LeRobot convention); recording
buffers in-memory until ``save_primitive()`` commits, so failed/aborted
primitives are simply discarded rather than polluting the dataset.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

if TYPE_CHECKING:
    pass


# Schema. Action shape depends on whether the gripper is in use:
#   - record_gripper=True (default):  7D = 6 pose + 1 gripper
#   - record_gripper=False:           6D pose only (non-gripper end-effectors)
# Progress (the 8th column in pickplace v2 / collect_demo.py datasets) is
# DERIVED post-hoc as frame_idx / (length-1) per episode — it's purely a
# function of frame position, so adding it during recording would just
# couple the live loop to a value that's already implicit. The downstream
# filter/normalize/merge scripts append the progress column when assembling
# a training-ready dataset.
def _build_features(record_gripper: bool) -> dict:
    action_shape = (7,) if record_gripper else (6,)
    return {
        "exterior_image_1_left": {
            "dtype": "image",
            "shape": (240, 320, 3),
            "names": ["height", "width", "channel"],
        },
        "exterior_image_2_left": {  # legacy slot, written as zeros
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
            "shape": action_shape,
            "names": ["actions"],
        },
    }


class FlywheelRecorder:
    """Per-primitive episode buffer that commits to LeRobot only on success.

    Lifecycle (called by the executor around each plan step):

        recorder.start_primitive("sweep rocks to the right")
        for tick in primitive:
            recorder.record_step(state, exterior, wrist)
        if primitive_succeeded:
            recorder.save_primitive()
        else:
            recorder.discard_primitive()
    """

    def __init__(self, repo_name: str, fps: float, record_gripper: bool = True):
        self.repo_name = repo_name
        self.fps = fps
        self.record_gripper = bool(record_gripper)
        path = HF_LEROBOT_HOME / repo_name

        # ``LeRobotDataset.create`` lays down the directory before the first
        # episode commits, so a Ctrl+C between create and the first
        # save_episode leaves an "exists but empty" directory. ``meta/tasks.jsonl``
        # is only written once an episode is committed, so its presence is a
        # reliable proxy for "real data here". If the directory exists but
        # tasks.jsonl is missing, wipe and recreate — the alternative is the
        # LeRobotDataset constructor falling through to a Hub fetch and
        # 401'ing on an unpublished repo.
        tasks_path = path / "meta" / "tasks.jsonl"
        has_committed_data = path.exists() and tasks_path.exists()

        if has_committed_data:
            logging.info("Recorder: appending to existing dataset %s", repo_name)
            self.dataset = LeRobotDataset(root=path, repo_id=repo_name)
        else:
            if path.exists():
                import shutil
                logging.info("Recorder: %s exists but has no committed episodes "
                             "(probably a Ctrl+C before first save_episode); "
                             "wiping and recreating.", path)
                shutil.rmtree(path)
            logging.info("Recorder: creating new dataset %s (record_gripper=%s)",
                         repo_name, self.record_gripper)
            self.dataset = LeRobotDataset.create(
                repo_id=repo_name,
                robot_type="xarm",
                fps=fps,
                features=_build_features(self.record_gripper),
            )
        self._dataset_path = path

        self._current_primitive: str = ""
        self._buffer: list[dict] = []
        self._prev: dict | None = None  # last (state, ext, wrist, grip) seen

    # ────────────────── Lifecycle ──────────────────

    def start_primitive(self, name: str) -> None:
        """Begin buffering a new primitive episode. Drops any prior buffer."""
        self._current_primitive = name
        self._buffer = []
        self._prev = None

    def record_step(self, state: np.ndarray, exterior: np.ndarray,
                    wrist: np.ndarray, gripper_norm: float = 0.0) -> None:
        """Append one tick.

        ``state`` is 6D ``[x, y, z, roll_deg, pitch_deg, yaw_deg]`` from
        ``hw.get_pose()``; rotations are converted to radians here to match
        the dataset schema. ``gripper_norm`` is the current normalized
        gripper position (``hw.gripper_norm()``: 0=open, ~0.99=closed) —
        ignored when ``record_gripper=False``.

        Following the collect_demo.py convention, the recorded frame uses
        the *previous* tick's state+images and the current state as the
        "action" (= achieved next pose). When ``record_gripper=True`` the
        current gripper position is appended as the 7th action dimension.
        """
        state_rad = state.astype(np.float32).copy()
        state_rad[3:6] = state_rad[3:6] * np.pi / 180.0

        if self._prev is not None:
            if self.record_gripper:
                action = np.concatenate([
                    state_rad,
                    np.array([float(gripper_norm)], dtype=np.float32),
                ]).astype(np.float32)
            else:
                action = state_rad
            self._buffer.append({
                "state": self._prev["state"],
                "actions": action,
                "exterior_image_1_left": self._prev["ext"],
                "exterior_image_2_left": np.zeros_like(self._prev["ext"]),
                "wrist_image_left": self._prev["wrist"],
                "task": self._current_primitive,
            })
        self._prev = {"state": state_rad, "ext": exterior, "wrist": wrist}

    def save_primitive(self) -> int:
        """Commit the buffered frames as a LeRobot episode. Returns frame count."""
        n = len(self._buffer)
        if n == 0:
            logging.warning("Recorder: %r had no frames to save", self._current_primitive)
            return 0
        for frame in self._buffer:
            self.dataset.add_frame(frame)
        self.dataset.save_episode()
        logging.info("Recorder: saved %d frames as episode for %r",
                     n, self._current_primitive)
        self._buffer = []
        self._prev = None
        return n

    def discard_primitive(self) -> None:
        """Drop the current buffer without committing."""
        if self._buffer:
            logging.info("Recorder: discarded %d unsaved frames for %r",
                         len(self._buffer), self._current_primitive)
        self._buffer = []
        self._prev = None

    def pop_buffer(self) -> list[dict]:
        """Take ownership of the current buffer and clear recorder state.

        Used when the commit/discard decision needs to be deferred (e.g.,
        skill-gap recording gated on a post-hoc success oracle that runs
        after subsequent primitives). Caller holds the returned list and
        later passes it to ``commit_buffered`` or simply drops it on the floor.
        """
        frames = self._buffer
        self._buffer = []
        self._prev = None
        return frames

    def commit_buffered(self, frames: list[dict]) -> int:
        """Commit a previously-popped buffer as a LeRobot episode."""
        n = len(frames)
        if n == 0:
            logging.warning("Recorder: commit_buffered called with empty list")
            return 0
        for frame in frames:
            self.dataset.add_frame(frame)
        self.dataset.save_episode()
        task = frames[0].get("task", "?")
        logging.info("Recorder: committed %d deferred frames as episode for %r",
                     n, task)
        return n
