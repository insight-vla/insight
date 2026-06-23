#!/usr/bin/env python3
"""Test primitive execution in LIBERO using the Docker setup."""

import collections
import dataclasses
import logging
import math
import pathlib
import sys

import imageio
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from PIL import Image, ImageDraw, ImageFont
import tyro

LIBERO_ENV_RESOLUTION = 256


def _quat2axisangle(quat):
    """Convert quaternion to axis-angle. Matches official LIBERO/robosuite convention."""
    import math
    q = quat.copy()
    if q[3] > 1.0: q[3] = 1.0
    elif q[3] < -1.0: q[3] = -1.0
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den

# 90th percentile durations from training data (frames at 10Hz)
# These represent how long each primitive typically takes in the demos
PRIMITIVE_P90_DURATIONS = {
    "move gripper to the red lego block": 236,
    "move gripper to the blue lego block": 193,
    "close gripper": 8,
    "lift upward": 76,
    "lower gripper": 94,
    "move gripper to target": 33,
    "move gripper to the target zone": 49,
    "open gripper": 9,
    "rotate block": 149,
    "flip the red lego block peg up": 500,
    "pick up the red lego block and place it on the target zone": 700,
}
DEFAULT_DURATION = 200  # Fallback for unknown primitives


class SimpleVisualizationWrapper:
    """Simple wrapper that enables robot/gripper visualization (green line + red dot).

    Matches training data which has these visualization overlays.
    """

    def __init__(self, env):
        self.env = env
        self._vis_settings = {"env": True, "robots": True, "grippers": False}
        # Walk the env chain to find the actual robosuite env with visualize()
        self._inner_env = None
        e = env
        for _ in range(10):
            if hasattr(e, 'visualize'):
                self._inner_env = e
                break
            if hasattr(e, 'env'):
                e = e.env
            else:
                break
        if self._inner_env is None:
            logging.warning("SimpleVisualizationWrapper: could not find env with visualize() method")
        else:
            logging.info(f"SimpleVisualizationWrapper: found visualize() on {type(self._inner_env).__name__}")
        self._update_visualization()

    def _update_visualization(self):
        if self._inner_env is not None:
            self._inner_env.visualize(vis_settings=self._vis_settings)

    def reset(self):
        obs = self.env.reset()
        self._update_visualization()
        return obs

    def step(self, action):
        result = self.env.step(action)
        self._update_visualization()
        return result

    def seed(self, seed):
        return self.env.seed(seed)

    @property
    def sim(self):
        return self.env.sim

    def check_success(self):
        return self.env.check_success()

    def close(self):
        return self.env.close()


def annotate_frame(img: np.ndarray, text: str, step: int) -> np.ndarray:
    """Add text annotation above the frame showing current primitive."""
    # Convert to PIL Image
    original_img = Image.fromarray(img)
    img_width, img_height = original_img.size

    # Try to use a nice font
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Parse text to extract primitive number and description
    # Format: "[1/5] move gripper to the red lego block"
    if ']' in text:
        parts = text.split(']', 1)
        prim_num = parts[0] + ']'
        prim_desc = parts[1].strip()
    else:
        prim_num = ""
        prim_desc = text

    # Create temporary draw to measure text
    temp_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))

    # Wrap text if it's too long
    max_width = img_width - 20  # Leave 10px padding on each side

    # Simple word wrapping
    words = prim_desc.split()
    lines = []
    current_line = ""

    for word in words:
        test_line = current_line + " " + word if current_line else word
        bbox = temp_draw.textbbox((0, 0), test_line, font=font_large)
        if bbox[2] - bbox[0] <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    # Use FIXED height for text area to ensure all frames are same size
    line_height = temp_draw.textbbox((0, 0), "Ay", font=font_large)[3] - temp_draw.textbbox((0, 0), "Ay", font=font_large)[1]
    padding = 8
    # Reserve space for up to 3 lines of text (prim_num + 2 description lines)
    text_area_height = (line_height * 3) + (padding * 2) + 5

    # Create new canvas with dark blue-gray background
    new_height = img_height + text_area_height
    new_img = Image.new('RGB', (img_width, new_height), color=(30, 30, 40))

    # Paste original image below text area
    new_img.paste(original_img, (0, text_area_height))

    # Draw on the new image
    draw = ImageDraw.Draw(new_img)

    # Draw primitive number in yellow on the left
    y_pos = padding
    draw.text((padding, y_pos), prim_num, fill=(255, 200, 0), font=font_large)

    # Draw description lines in white
    y_pos += line_height + 2
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font_large)
        text_width = bbox[2] - bbox[0]
        x_pos = padding
        draw.text((x_pos, y_pos), line, fill=(255, 255, 255), font=font_large)
        y_pos += line_height

    # Add step counter in bottom right with better styling
    step_text = f"Step {step}"
    step_bbox = draw.textbbox((0, 0), step_text, font=font_small)
    step_width = step_bbox[2] - step_bbox[0]
    step_height = step_bbox[3] - step_bbox[1]

    # Position in bottom right
    step_padding = 6
    step_x = img_width - step_width - step_padding * 2
    step_y = new_height - step_height - step_padding * 2

    # Draw rounded rectangle background
    draw.rectangle(
        [step_x - step_padding, step_y - step_padding,
         img_width - step_padding, new_height - step_padding],
        fill=(0, 0, 0, 200),
        outline=(100, 100, 100)
    )
    draw.text((step_x, step_y), step_text, fill=(200, 200, 200), font=font_small)

    return np.array(new_img)


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    num_steps_wait: int = 70  # Physics settling steps (matches demo collection)
    num_steps_per_primitive: int = 0  # 0 = use p90 durations from training data, >0 = fixed steps
    p90_dataset: str = ""  # LeRobot dataset repo_id to compute durations from (overrides hardcoded values)
    duration_percentile: int = 90  # Percentile for primitive durations (90=p90, 100=max)
    dense_labels: str = ""  # Path to dense_labels.json — use exact demo durations when combined with train-init-demo
    video_out_path: str = "data/libero/videos"
    run_name: str = ""  # Optional name for this run (otherwise uses timestamp)
    batch_name: str = ""  # Parent folder to group multiple runs (e.g. batch_20260319_142611)
    checkpoint: str = ""  # Description of checkpoint being tested (for logging)
    task: str = "lego"  # Task to test: "lego", "mug", or "drawer"
    seed: int = -1  # Random seed by default, set to specific value for reproducibility
    flip: bool = True  # Test flip sequence
    oracle_flip: bool = False  # Test oracle flip (single primitive, block starts on side)
    tilt_block: bool = False  # Tilt red block to its side before running (for any mode)
    num_flips: int = 1  # Number of flip sequences to chain (1 or 2)
    pickplace_e2e: bool = False  # Test pick-place with single end-to-end prompt (no primitives)
    ood: bool = False  # Use out-of-distribution primitives instead of in-distribution
    auto_advance: bool = True  # Auto-advance to next primitive when movement stops
    movement_threshold: float = 0.01  # Action magnitude threshold for "stopped"
    stop_steps: int = 15  # Consecutive steps below threshold to trigger advance
    min_steps: int = 30  # Minimum steps before checking for stop
    snap_gripper: bool = False  # Snap gripper action to ±1.0 (matches training data collection)
    replan_steps: int = 20  # Actions to execute per inference call (1 = replan every step)
    train_init_hdf5: str = ""  # Path to training HDF5 — restore initial sim state from training demos
    train_init_demo: int = -1  # Specific demo index to use (-1 = random)
    lerobot_repo: str = ""  # LeRobot repo_id to load actual training video (e.g. maggiewang/lego_red_pickplace_e2e_trimmed)
    batch_seeds: str = ""  # Run batch eval over seed range, e.g. "1000-1020". Writes results.txt to batch folder.
    drawer_start_open: bool = False  # For drawer task: use close_top_drawer.bddl (drawer starts open)
    vlm_completion_check: bool = False  # Enable VLM completion check for "move to" primitives (for OOD cases)


def test_primitives(args: Args) -> bool:
    """Test executing primitives sequentially. Returns True if task completed."""
    from datetime import datetime

    # Compute p90 durations from dataset if specified
    if args.p90_dataset:
        logging.info(f"Computing p90 durations from {args.p90_dataset}...")
        import json
        from pathlib import Path
        episodes_path = Path.home() / ".cache/huggingface/lerobot" / args.p90_dataset / "meta/episodes.jsonl"
        tasks_path = Path.home() / ".cache/huggingface/lerobot" / args.p90_dataset / "meta/tasks.jsonl"
        # Load task names
        task_names = {}
        with open(tasks_path) as f:
            for line in f:
                t = json.loads(line)
                task_names[t["task_index"]] = t["task"]
        # Collect lengths per task
        from collections import defaultdict
        task_lengths = defaultdict(list)
        with open(episodes_path) as f:
            for line in f:
                ep = json.loads(line)
                for task in ep["tasks"]:
                    task_lengths[task].append(ep["length"])
        # Compute duration at specified percentile
        for task, lengths in task_lengths.items():
            if args.duration_percentile >= 100:
                dur = int(max(lengths))
            else:
                dur = int(np.percentile(lengths, args.duration_percentile))
            PRIMITIVE_P90_DURATIONS[task] = dur
            logging.info(f"  {task}: p{args.duration_percentile}={dur} (n={len(lengths)})")

    # Use random seed if not specified, otherwise use provided seed
    if args.seed < 0:
        args.seed = np.random.randint(0, 10000)
        logging.info(f"Using random seed: {args.seed}")
    np.random.seed(args.seed)

    # Select task and primitives based on flag
    if args.task == "mug":
        primitives = [
            "move gripper to the yellow and white mug",
            "close gripper",
            "lift upward",
            "move gripper to the right plate",
            "lower gripper and open to place object",
        ]
        bddl_file = pathlib.Path(get_libero_path("bddl_files")) / "libero_90" / "LIVING_ROOM_SCENE5_put_the_yellow_and_white_mug_on_the_right_plate.bddl"
    elif args.task == "drawer":
        if args.drawer_start_open:
            primitives = [
                "move gripper to the top drawer handle",
                "push the top drawer closed",
            ]
            args.vlm_completion_check = True  # OOD — need VLM to stop move-to
        else:
            primitives = [
                "move gripper to the top drawer handle",
                "close gripper",
                "pull the top drawer open",
            ]
        args.replan_steps = 40
        bddl_name = "close_top_drawer.bddl" if args.drawer_start_open else "open_top_drawer.bddl"
        bddl_file = pathlib.Path(get_libero_path("bddl_files")) / "drawer_primitives" / bddl_name
    else:  # default: lego
        if args.pickplace_e2e:
            primitives = [
                "pick up the red lego block and place it on the target zone",
            ]
            # Disable auto-advance — this is one long task, not a primitive chain
            args.auto_advance = False
            # Match official LIBERO eval: execute 5 actions per chunk instead of 1
            args.replan_steps = 10
        elif args.oracle_flip:
            primitives = [
                "flip the red lego block peg up",
            ]
            # Disable auto-advance — this is one long task, not a primitive chain
            args.auto_advance = False
            args.replan_steps = 30
        elif args.flip:
            flip_chain = [
                "move gripper to the red lego block",
                "close gripper",
                "lift upward",
                "rotate block",
                "lower gripper",
                "open gripper",
                "move gripper to target",
            ]
            primitives = flip_chain * args.num_flips
            # args.replan_steps = 5
            args.replan_steps = 10
            # args.replan_steps = 30
            # args.replan_steps = 1
        elif args.ood:
            # Out-of-distribution primitives (not trained on these)
            primitives = [
                "move gripper to the red lego block",
                "close gripper firmly",
                "lift upward 5 centimeters",
                "rotate wrist 180 degrees clockwise",
                "lower gripper and open to place object",
            ]
        else:
            # In-distribution primitives for pick-and-place sequence
            primitives = [
                "move gripper to the red lego block",
                "close gripper",
                "lift upward",
                "move gripper to the target zone",
                "lower gripper",
                "open gripper",
                "move gripper to target"
            ]
            args.replan_steps = 30
            # args.replan_steps = 5
        bddl_file = pathlib.Path(get_libero_path("bddl_files")) / "lego_primitives" / "wide_range" / "pick_blue_place_target_wide.bddl"
 
    if not bddl_file.exists():
        logging.error(f"BDDL file not found: {bddl_file}")
        return False

    # Create environment
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
        horizon=3000,  # Long horizon for testing multiple primitives
    )
    # Enable robot/gripper visualization to match training data (green line + dot)
    env = SimpleVisualizationWrapper(env)
    env.seed(args.seed)

    # Create output directory with run name or timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    date_folder = datetime.now().strftime("%Y-%m-%d")
    if args.run_name:
        run_dir = args.run_name
    else:
        run_dir = f"run_{timestamp}"

    if args.batch_name:
        video_out_path = pathlib.Path(args.video_out_path) / date_folder / args.batch_name / run_dir
    else:
        video_out_path = pathlib.Path(args.video_out_path) / date_folder / run_dir
    video_out_path.mkdir(parents=True, exist_ok=True)
    args.video_out_path = str(video_out_path)
    logging.info(f"Saving videos to: {video_out_path}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Get checkpoint info from server metadata
    server_metadata = client.get_server_metadata()
    checkpoint_config = server_metadata.get("checkpoint_config", "unknown")
    checkpoint_dir = server_metadata.get("checkpoint_dir", "unknown")
    checkpoint_info = f"{checkpoint_config} @ {checkpoint_dir}"
    if args.checkpoint:  # Override if manually specified
        checkpoint_info = args.checkpoint

    # Create metadata file
    metadata_file = video_out_path / "metadata.txt"
    with open(metadata_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("VLA Primitive Sequential Execution Test\n")
        f.write("=" * 80 + "\n")
        f.write(f"Run Name: {run_dir}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} PST\n")
        f.write(f"Checkpoint: {checkpoint_info}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Replan steps: {args.replan_steps}\n")
        f.write(f"Duration percentile: {args.duration_percentile}\n")
        f.write(f"P90 dataset: {args.p90_dataset or 'hardcoded'}\n")
        f.write(f"Auto advance: {args.auto_advance} (threshold={args.movement_threshold}, stop_steps={args.stop_steps}, min_steps={args.min_steps})\n")
        f.write(f"Tilt block: {args.tilt_block}\n")
        f.write(f"Snap gripper: {args.snap_gripper}\n")
        if args.num_steps_per_primitive > 0:
            f.write(f"Steps per primitive: {args.num_steps_per_primitive} (fixed)\n")
        else:
            f.write(f"Steps per primitive: p{args.duration_percentile} durations\n")
        f.write(f"Primitives: {'out-of-distribution' if args.ood else 'in-distribution'}\n")
        f.write(f"Command: {' '.join(sys.argv)}\n")
        f.write("\n")
        f.write("PRIMITIVES (executed sequentially):\n")
        for i, p in enumerate(primitives):
            f.write(f"  [{i+1}] {p}\n")
        f.write("=" * 80 + "\n")
    logging.info(f"Saved metadata to: {metadata_file}")

    logging.info("=" * 60)
    logging.info("Testing Primitive Execution")
    logging.info("=" * 60)

    # Reset environment
    obs = env.reset()

    # Stabilization + open gripper to match training data start state
    open_gripper_action = [0.0] * 6 + [-1.0]  # Last dim negative = open
    for _ in range(args.num_steps_wait):
        obs, _, _, _ = env.step(open_gripper_action)
    logging.info("Settled and gripper opened at start")

    # Restore sim state from training HDF5 (overfit test) — works for any mode
    if args.train_init_hdf5:
        import h5py
        with h5py.File(args.train_init_hdf5, 'r') as hf:
            demo_keys = sorted([k for k in hf['data'].keys() if k.startswith('demo_')])
            if args.train_init_demo >= 0:
                demo_key = f'demo_{args.train_init_demo}'
            else:
                demo_key = demo_keys[np.random.randint(len(demo_keys))]
            logging.info(f"Loading initial state from {demo_key} ({args.train_init_hdf5})")

            # Load exact primitive durations from dense labels if provided
            if args.dense_labels:
                import json as _json
                with open(args.dense_labels) as _f:
                    all_labels = _json.load(_f)
                # Map demo key to episode index (h5py alphabetical order)
                ep_idx = demo_keys.index(demo_key)
                if ep_idx < len(all_labels):
                    demo_durations = {}
                    for seg in all_labels[ep_idx]["segments"]:
                        demo_durations[seg["primitive_label"]] = seg["length"]
                    args._demo_durations = demo_durations
                    logging.info(f"  Loaded exact durations from dense labels: {demo_durations}")

            init_state = hf[f'data/{demo_key}/states'][0]  # First frame sim state
            if f'data/{demo_key}/obs/upside_down_red_lego_block_1_pos' in hf:
                block_pos = hf[f'data/{demo_key}/obs/upside_down_red_lego_block_1_pos'][0]
                block_quat = hf[f'data/{demo_key}/obs/upside_down_red_lego_block_1_quat'][0]
                logging.info(f"  Block pos: [{block_pos[0]:.4f}, {block_pos[1]:.4f}, {block_pos[2]:.4f}]")
                logging.info(f"  Block quat: [{block_quat[0]:.4f}, {block_quat[1]:.4f}, {block_quat[2]:.4f}, {block_quat[3]:.4f}]")
            # Save training demo video for comparison (raw HDF5, downsampled 2x)
            train_images = hf[f'data/{demo_key}/obs/agentview_image'][:]
            train_video_path = pathlib.Path(args.video_out_path) / f"training_raw_{demo_key}.mov"
            # Downsample to ~10Hz (raw is 20Hz) and flip to match display convention
            train_frames = [train_images[i][::-1, ::-1] for i in range(0, len(train_images), 2)]
            imageio.mimwrite(str(train_video_path), train_frames, fps=10, codec="libx264")
            logging.info(f"  Saved raw training demo video ({len(train_frames)} frames): {train_video_path}")
            # Also save from LeRobot dataset (actual training data after trimming)
            # Match by image content since episode ordering depends on conversion pipeline
            if args.lerobot_repo:
                try:
                    import pandas as pd
                    from PIL import Image
                    import io as _io
                    lerobot_path = pathlib.Path.home() / ".cache/huggingface/lerobot" / args.lerobot_repo
                    parquet_dir = lerobot_path / "data" / "chunk-000"
                    parquet_files = sorted(parquet_dir.glob("episode_*.parquet"))
                    # Get reference image from HDF5 (use frame ~1s in so block has settled)
                    ref_frame = min(20, len(train_images) - 1)
                    ref_img = np.array(Image.fromarray(train_images[ref_frame][::-1, ::-1]).resize((64, 64)))
                    # Find best matching episode by comparing first-frame images
                    best_idx, best_dist = -1, float("inf")
                    for pf in parquet_files:
                        df_peek = pd.read_parquet(pf, columns=["image"])
                        img_data = df_peek.iloc[0]["image"]
                        img_bytes = img_data["bytes"] if isinstance(img_data, dict) else img_data
                        ep_img = np.array(Image.open(_io.BytesIO(img_bytes)).resize((64, 64)))
                        dist = np.mean((ref_img.astype(float) - ep_img.astype(float)) ** 2)
                        ep_num = int(pf.stem.split("_")[1])
                        if dist < best_dist:
                            best_dist = dist
                            best_idx = ep_num
                    logging.info(f"  Matched {demo_key} -> episode_{best_idx:06d} (dist={best_dist:.1f})")
                    parquet_file = parquet_dir / f"episode_{best_idx:06d}.parquet"
                    df = pd.read_parquet(parquet_file)
                    lerobot_frames = []
                    wrist_frames = []
                    for _, row in df.iterrows():
                        img_data = row["image"]
                        img_bytes = img_data["bytes"] if isinstance(img_data, dict) else img_data
                        lerobot_frames.append(np.array(Image.open(_io.BytesIO(img_bytes))))
                        if "wrist_image" in row and row["wrist_image"] is not None:
                            w_data = row["wrist_image"]
                            w_bytes = w_data["bytes"] if isinstance(w_data, dict) else w_data
                            wrist_frames.append(np.array(Image.open(_io.BytesIO(w_bytes))))
                    lerobot_video_path = pathlib.Path(args.video_out_path) / f"training_lerobot_{demo_key}.mov"
                    imageio.mimwrite(str(lerobot_video_path), lerobot_frames, fps=10, codec="libx264")
                    logging.info(f"  Saved LeRobot training video ({len(lerobot_frames)} frames): {lerobot_video_path}")
                    if wrist_frames:
                        wrist_video_path = pathlib.Path(args.video_out_path) / f"training_lerobot_wrist_{demo_key}.mov"
                        imageio.mimwrite(str(wrist_video_path), wrist_frames, fps=10, codec="libx264")
                        logging.info(f"  Saved LeRobot wrist video ({len(wrist_frames)} frames): {wrist_video_path}")
                except Exception as e:
                    logging.warning(f"  Could not load LeRobot data: {e}")
        # Restore full sim state by setting qpos/qvel directly
        # Flattened state format: [time, qpos..., qvel..., act..., udd_state...]
        nq = env.sim.model.nq
        nv = env.sim.model.nv
        env.sim.data.time = init_state[0]
        env.sim.data.qpos[:] = init_state[1:1+nq]
        env.sim.data.qvel[:] = init_state[1+nq:1+nq+nv]
        env.sim.forward()
        # Reload controller after state change
        current = env
        while current is not None:
            if hasattr(current, "robots") and current.robots:
                robot = current.robots[0]
                robot._load_controller()
                robot.controller.update_base_pose(robot.base_pos, robot.base_ori)
                break
            current = getattr(current, "env", None)
        # Brief settling to stabilize
        for _ in range(40):
            obs, _, _, _ = env.step(open_gripper_action)
        logging.info(f"Restored training state from {demo_key}")

    # Tilt block to its side (oracle_flip always tilts; tilt_block flag for any mode)
    elif args.oracle_flip or args.tilt_block:
        from vlm_flywheel.control import tilt_red_block_to_side
        obs = tilt_red_block_to_side(env, obs)
        logging.info("Tilted red block to its side")

    # Check block position — discard seed if block is missing or out of workspace (lego tasks only)
    if args.task == "lego":
        from vlm_flywheel.control import _find_red_block_joint
        block_joint = _find_red_block_joint(env)
        if block_joint is not None:
            jnt_id = env.sim.model.joint_name2id(block_joint)
            jnt_addr = env.sim.model.jnt_qposadr[jnt_id]
            block_pos = env.sim.data.qpos[jnt_addr:jnt_addr+3]
            logging.info(f"Red block position: [{block_pos[0]:.3f}, {block_pos[1]:.3f}, {block_pos[2]:.3f}]")
            # Reachable workspace (empirical: good seeds x=[-0.2,0.08], y=[-0.12,0.1], z~0.92)
            if block_pos[2] < 0.8 or abs(block_pos[0]) > 0.25 or block_pos[1] > 0.15 or block_pos[1] < -0.25:
                logging.warning(f"Block out of workspace, discarding seed {args.seed}")
                env.close()
                return None  # None = discarded (vs True/False for success/failure)
        else:
            logging.warning(f"Red block not found in sim, discarding seed {args.seed}")
            env.close()
            return None

        # For flip/tilt modes, discard if block is already right-side up (peg pointing up)
        if args.oracle_flip or args.tilt_block:
            from vlm_flywheel.reasoning import _get_peg_direction_vector
            peg = _get_peg_direction_vector(env)
            if peg is not None and peg[2] > 0.85:
                logging.warning(f"Block already right-side up (peg z={peg[2]:.2f}), discarding seed {args.seed}")
                env.close()
                return None

    # Save initial joint positions for arm reset (between flip sequences + return at end)
    initial_jpos = None
    initial_gripper_jpos = None
    current = env
    while current is not None:
        if hasattr(current, "robots") and current.robots:
            robot = current.robots[0]
            initial_jpos = env.sim.data.qpos[robot._ref_joint_pos_indexes].copy()
            initial_gripper_jpos = env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes].copy()
            logging.info("Saved initial joint positions for arm reset")
            break
        current = getattr(current, "env", None)

    all_replay_images = []  # For combined video
    t = 0
    task_completed = False
    task_completion_step = None
    episode_terminated = False

    # Store first action chunk from each primitive for comparison
    primitive_first_actions = {}
    primitive_videos = {}  # Store frames for each primitive separately

    # Execute each primitive
    for prim_idx, prompt in enumerate(primitives):
        # Check if episode already terminated
        if episode_terminated:
            logging.info(f"\n[{prim_idx+1}/{len(primitives)}] Skipping (episode terminated): {prompt}")
            continue

        # Reset arm between flip sequences (after first "open gripper", before second "move to")
        if args.flip and prim_idx == 6 and initial_jpos is not None:
            logging.info("\n--- Resetting arm to initial pose between flips ---")
            current = env
            while current is not None:
                if hasattr(current, "robots") and current.robots:
                    robot = current.robots[0]
                    env.sim.data.qpos[robot._ref_joint_pos_indexes] = initial_jpos
                    env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes] = initial_gripper_jpos
                    env.sim.data.qvel[robot._ref_joint_vel_indexes] = 0.0
                    env.sim.data.qvel[robot._ref_gripper_joint_vel_indexes] = 0.0
                    env.sim.forward()
                    robot._load_controller()
                    robot.controller.update_base_pose(robot.base_pos, robot.base_ori)
                    break
                current = getattr(current, "env", None)
            for _ in range(args.num_steps_wait):
                obs, _, _, _ = env.step(open_gripper_action)
            logging.info("Arm reset and settled")



        logging.info(f"\n[{prim_idx+1}/{len(primitives)}] Primitive: {prompt}")

        primitive_images = []  # Frames for this primitive only
        consecutive_stop_steps = 0  # Track consecutive low-movement steps

        # Determine steps for this primitive
        if args.dense_labels and args.train_init_demo >= 0 and hasattr(args, '_demo_durations'):
            # Use exact duration from this demo's dense labels
            num_steps = args._demo_durations.get(prompt, PRIMITIVE_P90_DURATIONS.get(prompt, DEFAULT_DURATION))
            logging.info(f"  Running for {num_steps} steps (from demo labels)")
        elif args.num_steps_per_primitive > 0:
            num_steps = args.num_steps_per_primitive
        else:
            num_steps = PRIMITIVE_P90_DURATIONS.get(prompt, DEFAULT_DURATION)
            logging.info(f"  Running for {num_steps} steps (p{args.duration_percentile} duration)")

        # Execute this primitive for N steps
        action_queue = collections.deque()
        progress = None  # Track progress prediction across steps
        for step in range(num_steps):
            # Get preprocessed images
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
            )
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
            )

            # Annotate frame with current primitive and last progress prediction
            label = f"[{prim_idx+1}/{len(primitives)}] {prompt}"
            if progress is not None:
                label += f"  prog={progress:.2f}"
            annotated_img = annotate_frame(img, label, t)
            all_replay_images.append(annotated_img)
            primitive_images.append(annotated_img)

            # Get action: replan when queue is empty
            if not action_queue:
                # Prepare observation
                # State: EEF pos (3D) + axis-angle (3D) + gripper (2D) = 8D
                # Matches official Pi0.5 LIBERO format
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": np.concatenate((
                        obs["robot0_eef_pos"],  # 3D EEF position
                        _quat2axisangle(obs["robot0_eef_quat"]),  # 3D axis-angle
                        obs["robot0_gripper_qpos"],  # 2D gripper
                    )),
                    "prompt": prompt,
                }

                # Get action from policy
                action_chunk = client.infer(element)["actions"]
                # Queue up replan_steps actions from the chunk
                for a in action_chunk[:args.replan_steps]:
                    action_queue.append(a)

            action = action_queue.popleft()

            # Capture progress prediction (8th dim)
            progress = float(action[7]) if len(action) > 7 else None

            # Check progress prediction — stop primitive when done
            if progress is not None and progress > 0.95 and step >= args.min_steps:
                logging.info(f"  Progress={progress:.2f} > 0.95 at step {step+1} — primitive done")
                break

            # Strip progress dim before sending to env
            action = action[:7]

            # Snap gripper to ±1.0 if enabled
            if args.snap_gripper:
                action[6] = 1.0 if action[6] >= 0 else -1.0

            # Debug: Store first action of each primitive
            if step == 0:
                primitive_first_actions[prompt] = action.copy()

            # Verbose action logging every 50 steps
            if step % 50 == 0 or step < 5:
                prog_str = f" prog={progress:.2f}" if progress is not None else ""
                logging.info(f"  Step {step:4d}: pos=[{action[0]:+.4f}, {action[1]:+.4f}, {action[2]:+.4f}] "
                           f"rot=[{action[3]:+.4f}, {action[4]:+.4f}, {action[5]:+.4f}] "
                           f"grip={action[6]:+.4f}{prog_str}")

            # Check for auto-advance (movement stopped)
            if args.auto_advance and step >= args.min_steps:
                action_magnitude = np.linalg.norm(action[:6])  # pos + rot, ignore gripper
                if action_magnitude < args.movement_threshold:
                    consecutive_stop_steps += 1
                    if consecutive_stop_steps >= args.stop_steps:
                        logging.info(f"  Auto-advancing after {step+1} steps (movement stopped)")
                        break
                else:
                    consecutive_stop_steps = 0

            # VLM completion check for "move to" primitives (OOD cases)
            if (args.vlm_completion_check
                and prompt.startswith("move gripper to")
                and step >= args.min_steps
                and step % 20 == 0):
                from vlm_flywheel.vlm import set_vlm_provider
                from vlm_flywheel.base_execution import _vlm_check_primitive_done
                if not hasattr(args, '_vlm_initialized'):
                    set_vlm_provider("gemini")
                    args._vlm_initialized = True
                raw_img = obs["agentview_image"][::-1]  # vertical flip for VLM
                raw_wrist = obs["robot0_eye_in_hand_image"][::-1]
                if _vlm_check_primitive_done(prompt, raw_img, raw_wrist, save_dir=pathlib.Path(args.video_out_path), step_num=step):
                    logging.info(f"  VLM says primitive done at step {step+1}")
                    break

            # Execute
            try:
                obs, reward, done, info = env.step(action.tolist())
            except ValueError as e:
                if "terminated episode" in str(e):
                    logging.info(f"  Episode terminated unexpectedly at step {t}")
                    episode_terminated = True
                    done = True
                else:
                    raise

            if done:
                episode_terminated = True
                if not task_completed:
                    task_completed = True
                    task_completion_step = t
                    logging.info(f"  ✓ Task completed at step {t}!")
                # Can't continue stepping after episode terminates
                t += 1
                break

            t += 1

        # Save video for this primitive
        primitive_videos[prompt] = primitive_images
        prim_name = prompt.replace(" ", "_")[:40]  # Short filename
        prim_video_path = pathlib.Path(args.video_out_path) / f"primitive_{prim_idx+1}_{prim_name}.mov"
        imageio.mimwrite(prim_video_path, [np.asarray(x) for x in primitive_images], fps=10, codec="libx264")
        logging.info(f"  Executed {step+1} steps for this primitive")
        logging.info(f"  Saved video: {prim_video_path}")

        # Break out of primitive loop if episode terminated
        if episode_terminated:
            logging.info("  Episode terminated, stopping execution")
            break

    # Reset arm to initial pose (clears camera view for final frame)
    if initial_jpos is not None and not episode_terminated:
        logging.info("\n--- Resetting arm to initial pose ---")
        current = env
        while current is not None:
            if hasattr(current, "robots") and current.robots:
                robot = current.robots[0]
                env.sim.data.qpos[robot._ref_joint_pos_indexes] = initial_jpos
                env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes] = initial_gripper_jpos
                env.sim.data.qvel[robot._ref_joint_vel_indexes] = 0.0
                env.sim.data.qvel[robot._ref_gripper_joint_vel_indexes] = 0.0
                env.sim.forward()
                robot._load_controller()
                robot.controller.update_base_pose(robot.base_pos, robot.base_ori)
                break
            current = getattr(current, "env", None)
        for _ in range(args.num_steps_wait):
            obs, _, _, _ = env.step(open_gripper_action)
        # Capture final frame with arm out of the way
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
        )
        final_frame = annotate_frame(img, "[done] arm reset", t)
        all_replay_images.append(final_frame)
        logging.info("Arm reset and settled")

    # For oracle flip, check peg direction instead of BDDL success
    if args.oracle_flip or args.tilt_block:
        from vlm_flywheel.reasoning import _get_peg_direction_vector
        peg = _get_peg_direction_vector(env)
        if peg is not None and peg[2] > 0.85:
            task_completed = True
            logging.info(f"Peg pointing up (z={peg[2]:.2f}) — flip succeeded!")
        else:
            z = peg[2] if peg is not None else None
            logging.info(f"Peg NOT up (z={z}) — flip failed")

    # Save combined video with all primitives
    suffix = "success" if task_completed else "failure"
    video_path = pathlib.Path(args.video_out_path) / f"primitives_test_all_{suffix}.mov"
    imageio.mimwrite(video_path, [np.asarray(x) for x in all_replay_images], fps=10, codec="libx264")

    # Analyze action similarity
    logging.info("\n" + "=" * 60)
    logging.info("VLA Action Similarity Analysis")
    logging.info("=" * 60)

    prompts_list = list(primitive_first_actions.keys())
    for i in range(len(prompts_list)):
        for j in range(i+1, len(prompts_list)):
            p1, p2 = prompts_list[i], prompts_list[j]
            diff = np.abs(primitive_first_actions[p1] - primitive_first_actions[p2])
            mean_diff = diff.mean()

            logging.info(f"\n'{p1[:30]}...' vs '{p2[:30]}...'")
            logging.info(f"  Position diff: [{diff[0]:.4f}, {diff[1]:.4f}, {diff[2]:.4f}] (mean: {diff[:3].mean():.4f})")
            logging.info(f"  Rotation diff: [{diff[3]:.4f}, {diff[4]:.4f}, {diff[5]:.4f}] (mean: {diff[3:6].mean():.4f})")
            logging.info(f"  Gripper diff: {diff[6]:.4f}")
            logging.info(f"  Overall mean diff: {mean_diff:.6f}")

            if mean_diff < 0.001:
                logging.info(f"  ⚠️  NEARLY IDENTICAL - VLA ignoring prompt!")
            elif mean_diff < 0.01:
                logging.info(f"  ⚠️  Very similar - VLA mostly ignoring prompt")
            else:
                logging.info(f"  ✓ Actions differ")

    logging.info("\n" + "=" * 60)
    logging.info(f"Result: {suffix}")
    if task_completed:
        logging.info(f"Task completed at step: {task_completion_step}")
    logging.info(f"Video: {video_path}")
    logging.info(f"Total steps: {t}")
    logging.info(f"Primitives executed: {len(primitives)}")
    logging.info("=" * 60)

    return task_completed


def run_batch(args: Args) -> None:
    """Run batch eval over a seed range and save results.txt to the batch folder."""
    from datetime import datetime

    # Parse seed range
    parts = args.batch_seeds.split("-")
    seed_start, seed_end = int(parts[0]), int(parts[1])

    # Generate batch name
    if not args.batch_name:
        args.batch_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    date_folder = datetime.now().strftime("%Y-%m-%d")
    batch_dir = pathlib.Path(args.video_out_path) / date_folder / args.batch_name

    # Get checkpoint info from policy server
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    server_metadata = client.get_server_metadata()
    checkpoint_config = server_metadata.get("checkpoint_config", "unknown")
    checkpoint_dir = server_metadata.get("checkpoint_dir", "unknown")

    results = []  # (seed, True/False/None) where None = discarded
    for seed in range(seed_start, seed_end + 1):
        logging.info(f"\n{'='*60}")
        logging.info(f"=== SEED {seed} ===")
        logging.info(f"{'='*60}")

        run_args = dataclasses.replace(args, seed=seed, run_name=f"seed{seed}", batch_seeds="")
        try:
            result = test_primitives(run_args)
            results.append((seed, result))
            status = "success" if result is True else ("discarded" if result is None else "failure")
            logging.info(f"Seed {seed}: {status}")
        except Exception as e:
            logging.error(f"Seed {seed} crashed: {e}")
            results.append((seed, False))

    # Read actual replan_steps from first non-discarded seed's metadata
    actual_replan_steps = args.replan_steps
    for seed, result in results:
        if result is not None:
            meta_path = batch_dir / f"seed{seed}" / "metadata.txt"
            if meta_path.exists():
                for line in meta_path.read_text().splitlines():
                    if line.startswith("Replan steps:"):
                        actual_replan_steps = int(line.split(":")[1].strip())
                        break
            break

    # Write results.txt to batch folder
    batch_dir.mkdir(parents=True, exist_ok=True)
    valid = [(s, r) for s, r in results if r is not None]
    discarded = [(s, r) for s, r in results if r is None]
    successes = sum(1 for _, r in valid if r)
    total_valid = len(valid)
    results_path = batch_dir / "results.txt"
    with open(results_path, "w") as f:
        f.write(f"Batch eval: {args.batch_name}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} PST\n")
        f.write(f"Checkpoint: {checkpoint_config} @ {checkpoint_dir}\n")
        f.write(f"Seeds: {seed_start}-{seed_end}\n")
        f.write(f"Result: {successes}/{total_valid} ({100*successes//total_valid if total_valid else 0}%)")
        if discarded:
            f.write(f"  ({len(discarded)} discarded)")
        f.write(f"\n")
        f.write(f"\nSettings:\n")
        f.write(f"  tilt_block: {args.tilt_block}\n")
        f.write(f"  flip: {args.flip}\n")
        f.write(f"  oracle_flip: {args.oracle_flip}\n")
        f.write(f"  p90_dataset: {args.p90_dataset}\n")
        f.write(f"  duration_percentile: {args.duration_percentile}\n")
        f.write(f"  replan_steps: {actual_replan_steps}\n")
        f.write(f"\nPer-seed:\n")
        for seed, result in results:
            status = "success" if result is True else ("discarded" if result is None else "failure")
            f.write(f"  seed {seed}: {status}\n")

    logging.info(f"\n{'='*60}")
    logging.info(f"BATCH RESULT: {successes}/{total_valid} ({100*successes//total_valid if total_valid else 0}%)")
    if discarded:
        logging.info(f"  ({len(discarded)} seeds discarded: {[s for s, _ in discarded]})")
    logging.info(f"Results saved to: {results_path}")
    logging.info(f"{'='*60}")


def _quat2axisangle(quat):
    """Convert quaternion to axis-angle."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def main(args: Args) -> None:
    if args.batch_seeds:
        run_batch(args)
    else:
        test_primitives(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)
