"""
Convert raw LIBERO HDF5 demonstrations to LeRobot format.

Usage:
uv run sim/libero_flywheel/convert_libero_hdf5_to_lerobot.py --data_dir demonstration_data --repo_id your_username/lego_demos
"""

import shutil
from pathlib import Path

import h5py
import numpy as np
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from PIL import Image as PILImage


def main(
    data_dir: str,
    repo_id: str = "your_username/libero_lego_demos",
    *,
    push_to_hub: bool = False,
    fix_freeplay_mismatch: bool = False,
    downsample: bool = True,
):
    """Convert LIBERO HDF5 demonstrations to LeRobot format.

    Args:
        data_dir: Directory containing robosuite demonstration folders with demo.hdf5 files
        repo_id: Repository ID for the LeRobot dataset
        push_to_hub: Whether to push the dataset to Hugging Face Hub
        fix_freeplay_mismatch: If True, handle data where free-play actions were recorded but
            observations weren't (uses last N actions to match observation count)
        downsample: If True, take every other frame (20 Hz -> 10 Hz) to match standard LIBERO
    """
    data_path = Path(data_dir)

    # Find all demo.hdf5 files (check current dir and one level down)
    demo_files = list(data_path.glob("*/demo.hdf5"))
    if (data_path / "demo.hdf5").exists():
        demo_files.append(data_path / "demo.hdf5")
    print(f"Found {len(demo_files)} demonstration files")

    if len(demo_files) == 0:
        print(f"No demo.hdf5 files found in {data_dir}")
        return

    # Clean up any existing dataset
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)

    # Create LeRobot dataset
    # Use LIBERO naming convention (image, state, actions) instead of observation.* format
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=10,  # Match LIBERO dataset FPS
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),  # 7 DOF + gripper
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),  # 6 DOF + gripper
                # "shape": (8,),  # 6 DOF + gripper + progress (0→1)
                "names": ["action"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Process each demonstration file
    episode_idx = 0
    for demo_file in sorted(demo_files):
        folder_name = demo_file.parent.name
        print(f"\nProcessing {folder_name}")

        # Extract task description from folder name
        # Format: robosuite_ln_libero_tabletop_manipulation_TIMESTAMP_pick_up_the_red_lego_block_...
        # Find the task part after the timestamp
        parts = folder_name.split('_')
        task_start_idx = None
        for i, part in enumerate(parts):
            if part in ('pick', 'place', 'put', 'move', 'open', 'close', 'push', 'pull', 'lift', 'stack'):
                task_start_idx = i
                break

        if task_start_idx:
            task_description = ' '.join(parts[task_start_idx:])
        else:
            task_description = "pick and place lego"
        print(f"  Task: {task_description}")

        with h5py.File(demo_file, "r") as f:
            # Get list of demos in this file
            demo_keys = [key for key in f["data"].keys() if key.startswith("demo_")]
            print(f"  Found {len(demo_keys)} episodes in file")

            for demo_key in demo_keys:
                demo = f[f"data/{demo_key}"]

                # Skip episodes without observations
                if "obs" not in demo:
                    print(f"    Skipping {demo_key}: no observations")
                    continue

                # Check for language_instruction in HDF5 attributes (from primitive demos)
                if "language_instruction" in demo.attrs:
                    task_description = demo.attrs["language_instruction"]
                    print(f"    Using language from HDF5: {task_description}")

                # Skip episodes with empty task description
                if not task_description or task_description.strip() == "":
                    print(f"    Skipping {demo_key}: empty task description")
                    continue

                # Get episode length
                num_action_steps = demo["actions"].shape[0]
                action_offset = 0

                # Handle free-play mismatch if flag is set
                if fix_freeplay_mismatch:
                    # Check observation length to detect free-play mismatch
                    obs_len = None
                    for img_key in ["obs/agentview_image", "obs/frontview_image", "obs/image"]:
                        if img_key in demo:
                            obs_len = demo[img_key].shape[0]
                            break

                    # If observations shorter than actions, use only the last N actions (recording portion)
                    if obs_len and obs_len < num_action_steps:
                        action_offset = num_action_steps - obs_len
                        print(f"    WARNING: actions ({num_action_steps}) > obs ({obs_len}), using last {obs_len} steps")
                        num_steps = obs_len
                    else:
                        num_steps = num_action_steps
                else:
                    num_steps = num_action_steps

                # Downsample if enabled (20 Hz -> 10 Hz)
                step_stride = 2 if downsample else 1
                frame_indices = list(range(0, num_steps, step_stride))
                actual_frames = len(frame_indices)

                if downsample:
                    print(f"    {demo_key}: {num_steps} steps -> {actual_frames} frames (downsampled), task: {task_description}")
                else:
                    print(f"    {demo_key}: {num_steps} steps, task: {task_description}")

                # Add frames to dataset
                for frame_num, step_idx in enumerate(frame_indices):
                    # Get agentview/base image (try different possible names)
                    image = None
                    for img_key in ["obs/agentview_image", "obs/frontview_image", "obs/image"]:
                        if img_key in demo:
                            image = np.array(demo[img_key][step_idx])
                            break

                    if image is None:
                        # Find any image observation
                        image_keys = [k for k in demo["obs"].keys() if "image" in k.lower() and "eye" not in k.lower()]
                        if image_keys:
                            image = np.array(demo[f"obs/{image_keys[0]}"][step_idx])
                        else:
                            raise ValueError(f"No base image observation found in {demo_key}")

                    # Resize if needed (LIBERO default is 84x84 or 128x128)
                    if image.shape[:2] != (256, 256):
                        image = np.array(PILImage.fromarray(image).resize((256, 256)))

                    # Flip both axes (180° rotation) to match official LIBERO convention
                    image = image[::-1, ::-1]

                    # Ensure uint8 for Image type encoding
                    if image.dtype != np.uint8:
                        image = image.astype(np.uint8)

                    # Get wrist/eye-in-hand image
                    wrist_image = None
                    for wrist_key in ["obs/robot0_eye_in_hand_image", "obs/eye_in_hand_image", "obs/wrist_image"]:
                        if wrist_key in demo:
                            wrist_image = np.array(demo[wrist_key][step_idx])
                            break

                    if wrist_image is None:
                        # Find any eye_in_hand observation
                        wrist_keys = [k for k in demo["obs"].keys() if "eye" in k.lower() or "wrist" in k.lower()]
                        if wrist_keys:
                            wrist_image = np.array(demo[f"obs/{wrist_keys[0]}"][step_idx])
                        else:
                            # Use agentview as fallback if no wrist camera
                            print(f"    Warning: No wrist camera found, using agentview as fallback")
                            wrist_image = image.copy()

                    # Resize wrist image if needed
                    if wrist_image.shape[:2] != (256, 256):
                        wrist_image = np.array(PILImage.fromarray(wrist_image).resize((256, 256)))

                    # Flip both axes (180° rotation) to match official LIBERO convention
                    wrist_image = wrist_image[::-1, ::-1]

                    # Ensure uint8 for Image type encoding
                    if wrist_image.dtype != np.uint8:
                        wrist_image = wrist_image.astype(np.uint8)

                    # Get robot state: EEF pose (pos + axis-angle) + gripper
                    # Matches official Pi0.5 LIBERO format
                    eef_pos = None
                    for pos_key in ["obs/robot0_eef_pos", "obs/eef_pos"]:
                        if pos_key in demo:
                            eef_pos = np.array(demo[pos_key][step_idx])
                            break

                    eef_quat = None
                    for quat_key in ["obs/robot0_eef_quat", "obs/eef_quat"]:
                        if quat_key in demo:
                            eef_quat = np.array(demo[quat_key][step_idx])
                            break

                    # Fall back to joint_pos if no EEF data
                    if eef_pos is not None and eef_quat is not None:
                        # Convert quaternion to axis-angle (matches official LIBERO)
                        q = eef_quat.copy()
                        if q[3] > 1.0: q[3] = 1.0
                        elif q[3] < -1.0: q[3] = -1.0
                        den = np.sqrt(1.0 - q[3] * q[3])
                        if den < 1e-8:
                            axis_angle = np.zeros(3)
                        else:
                            axis_angle = (q[:3] * 2.0 * np.arccos(q[3])) / den
                        robot_state = np.concatenate([eef_pos, axis_angle])  # 6D
                    else:
                        # Fallback: use joint positions
                        joint_pos = None
                        for state_key in ["obs/robot0_joint_pos", "obs/joint_pos"]:
                            if state_key in demo:
                                joint_pos = np.array(demo[state_key][step_idx])
                                break
                        if joint_pos is None:
                            raise ValueError(f"No EEF or joint position found in {demo_key}")
                        robot_state = joint_pos

                    # Get gripper state
                    gripper = None
                    for gripper_key in ["obs/robot0_gripper_qpos", "obs/gripper_qpos"]:
                        if gripper_key in demo:
                            gripper = np.array(demo[gripper_key][step_idx])
                            break

                    if gripper is None:
                        gripper = np.array([0.0])  # Default if not found

                    # Combine state: eef_pos(3) + axis_angle(3) + gripper(2) = 8D
                    if len(gripper.shape) == 0:
                        gripper = np.array([gripper])
                    state = np.concatenate([robot_state, gripper])
                    if state.shape[0] > 8:
                        state = state[:8]  # Truncate to 8
                    elif state.shape[0] < 8:
                        state = np.pad(state, (0, 8 - state.shape[0]))  # Pad to 8

                    # Get action (with offset if fixing free-play mismatch)
                    action = np.array(demo["actions"][step_idx + action_offset])

                    # When downsampling, sum consecutive action pairs to preserve total motion
                    # Position/rotation deltas should be summed, gripper takes the later value
                    if downsample and (step_idx + 1 + action_offset) < len(demo["actions"]):
                        next_action = np.array(demo["actions"][step_idx + 1 + action_offset])
                        # Sum position (0:3) and rotation (3:6) deltas
                        action[:6] = action[:6] + next_action[:6]
                        # For gripper, take the second action's value (captures gripper changes)
                        action[6:] = next_action[6:]

                    if action.shape[0] > 7:
                        action = action[:7]  # Truncate to 7
                    elif action.shape[0] < 7:
                        action = np.pad(action, (0, 7 - action.shape[0]))  # Pad to 7

                    # # Append progress (0→1 within episode)
                    # progress = frame_num / max(actual_frames - 1, 1)
                    # action = np.append(action, progress)

                    # Add frame (using LIBERO naming convention)
                    dataset.add_frame({
                        "image": image,
                        "wrist_image": wrist_image,
                        "state": state.astype(np.float32),
                        "actions": action.astype(np.float32),
                        "task": task_description,
                    })

                # Mark episode as complete
                dataset.save_episode()
                episode_idx += 1

    print(f"\n{'='*60}")
    print(f"✓ Converted {episode_idx} episodes to LeRobot format")
    print(f"✓ Dataset saved to {output_path}")
    print(f"{'='*60}")

    if push_to_hub:
        print(f"\nPushing dataset to Hugging Face Hub: {repo_id}")
        dataset.push_to_hub()
        print("✓ Dataset pushed to Hub")


if __name__ == "__main__":
    tyro.cli(main)
