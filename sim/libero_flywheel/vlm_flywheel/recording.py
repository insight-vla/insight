"""HDF5 recording helpers for flywheel demo collection."""

from __future__ import annotations

import logging

import h5py
import imageio
import numpy as np
from PIL import Image


def _make_recording_step(original_step, recording_state: dict):
    """Wrap ``env.step`` to capture raw obs+action pairs for HDF5 recording.

    ``recording_state`` is the executor's recording dict (``self._raw_recording``)
    — we hold a reference so the wrapper sees mutations the executor makes
    (toggling ``enabled``, switching ``primitive``, resetting ``buffer``).
    """
    rec = recording_state

    def wrapper(action):
        if rec["enabled"] and rec["last_obs"] is not None and rec["primitive"]:
            obs = rec["last_obs"]

            def _resize(img):
                if img.shape[0] != 256:
                    return np.array(Image.fromarray(img).resize((256, 256)))
                return img.copy()

            rec["buffer"].append({
                "agentview_image": _resize(obs["agentview_image"]),
                "robot0_eye_in_hand_image": _resize(obs["robot0_eye_in_hand_image"]),
                "robot0_joint_pos": obs["robot0_joint_pos"].copy(),
                "robot0_gripper_qpos": obs["robot0_gripper_qpos"].copy(),
                "robot0_eef_pos": obs["robot0_eef_pos"].copy(),
                "robot0_eef_quat": obs["robot0_eef_quat"].copy(),
                "action": np.array(action, dtype=np.float32),
                "primitive": rec["primitive"],
            })
        result = original_step(action)
        if rec["enabled"]:
            rec["last_obs"] = result[0]
        return result

    return wrapper


def _save_raw_hdf5(output_dir, buffer):
    """Save recorded obs+action buffer to demo.hdf5, grouped by primitive."""
    if not buffer:
        return

    segments = []
    current_name = buffer[0]["primitive"]
    current_start = 0
    for i, entry in enumerate(buffer):
        if entry["primitive"] != current_name:
            segments.append((current_name, current_start, i))
            current_name = entry["primitive"]
            current_start = i
    segments.append((current_name, current_start, len(buffer)))

    last_segments = {}
    for name, start, end in segments:
        last_segments[name] = (start, end)

    hdf5_path = output_dir / "demo.hdf5"
    with h5py.File(str(hdf5_path), "w") as f:
        data_grp = f.create_group("data")
        demo_idx = 1
        for name, (start, end) in last_segments.items():
            entries = buffer[start:end]

            # Trim trailing low-motion steps (settling tail)
            ACTION_THRESH = 0.002  # ~2mm or ~0.1°
            while len(entries) > 1:
                a = entries[-1]["action"]
                if np.max(np.abs(a[:6])) > ACTION_THRESH:
                    break
                entries.pop()

            n = len(entries)
            if n == 0:
                continue

            demo_grp = data_grp.create_group(f"demo_{demo_idx}")
            obs_grp = demo_grp.create_group("obs")

            obs_grp.create_dataset(
                "agentview_image",
                data=np.stack([e["agentview_image"] for e in entries]),
            )
            obs_grp.create_dataset(
                "robot0_eye_in_hand_image",
                data=np.stack([e["robot0_eye_in_hand_image"] for e in entries]),
            )
            obs_grp.create_dataset(
                "robot0_joint_pos",
                data=np.stack([e["robot0_joint_pos"] for e in entries]),
            )
            obs_grp.create_dataset(
                "robot0_gripper_qpos",
                data=np.stack([e["robot0_gripper_qpos"] for e in entries]),
            )
            demo_grp.create_dataset(
                "actions",
                data=np.stack([e["action"] for e in entries]),
            )
            demo_grp.attrs["language_instruction"] = name

            video_path = output_dir / f"demo_{demo_idx}_{name.replace(' ', '_')}.mov"
            frames = [e["agentview_image"][::-1] for e in entries]
            imageio.mimwrite(str(video_path), frames, fps=10, codec="libx264")

            logging.info(
                f"  HDF5 demo_{demo_idx}: '{name}' — {n} steps, "
                f"img={entries[0]['agentview_image'].shape}, "
                f"action={entries[0]['action'].shape}"
            )
            demo_idx += 1

    logging.info(f"Saved raw HDF5: {hdf5_path} ({demo_idx - 1} demos)")
