#!/usr/bin/env python3
"""Test if VLA responds to different primitives when ALL start from same reset state."""

import dataclasses
import json
import logging
import math
import pathlib

import h5py
import imageio
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from PIL import Image, ImageDraw, ImageFont
import tyro

LIBERO_ENV_RESOLUTION = 256


class SimpleVisualizationWrapper:
    """Simple wrapper that enables robot/gripper visualization (green line + red dot).

    Matches training data which has these visualization overlays.
    """

    def __init__(self, env):
        self.env = env
        self._vis_settings = {"env": True, "robots": True, "grippers": True}
        self._update_visualization()

    def _update_visualization(self):
        if hasattr(self.env, 'env') and hasattr(self.env.env, 'visualize'):
            self.env.env.visualize(vis_settings=self._vis_settings)

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

    def set_init_state(self, init_state):
        return self.env.set_init_state(init_state)

    def get_sim_state(self):
        return self.env.get_sim_state()


def annotate_frame(img: np.ndarray, text: str, step: int) -> np.ndarray:
    """Add text annotation above the frame showing current primitive."""
    original_img = Image.fromarray(img)
    img_width, img_height = original_img.size

    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    if ']' in text:
        parts = text.split(']', 1)
        prim_num = parts[0] + ']'
        prim_desc = parts[1].strip()
    else:
        prim_num = ""
        prim_desc = text

    temp_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    max_width = img_width - 20

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

    line_height = temp_draw.textbbox((0, 0), "Ay", font=font_large)[3] - temp_draw.textbbox((0, 0), "Ay", font=font_large)[1]
    padding = 8
    text_area_height = (line_height * 3) + (padding * 2) + 5

    new_height = img_height + text_area_height
    new_img = Image.new('RGB', (img_width, new_height), color=(30, 30, 40))
    new_img.paste(original_img, (0, text_area_height))

    draw = ImageDraw.Draw(new_img)
    y_pos = padding
    draw.text((padding, y_pos), prim_num, fill=(255, 200, 0), font=font_large)

    y_pos += line_height + 2
    for line in lines:
        draw.text((padding, y_pos), line, fill=(255, 255, 255), font=font_large)
        y_pos += line_height

    step_text = f"Step {step}"
    step_bbox = draw.textbbox((0, 0), step_text, font=font_small)
    step_width = step_bbox[2] - step_bbox[0]
    step_height = step_bbox[3] - step_bbox[1]

    step_padding = 6
    step_x = img_width - step_width - step_padding * 2
    step_y = new_height - step_height - step_padding * 2

    draw.rectangle(
        [step_x - step_padding, step_y - step_padding,
         img_width - step_padding, new_height - step_padding],
        fill=(0, 0, 0, 200),
        outline=(100, 100, 100)
    )
    draw.text((step_x, step_y), step_text, fill=(200, 200, 200), font=font_small)

    return np.array(new_img)


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


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    num_steps_wait: int = 50  # Physics settling steps (matches demo collection)
    num_steps_per_primitive: int = 0  # If 0, use per-primitive p90 durations from training data
    video_out_path: str = "data/libero/videos"
    run_name: str = ""  # Optional name for this run (otherwise uses timestamp)
    checkpoint: str = ""  # Description of checkpoint being tested (for logging)
    task: str = "lego"  # Task to test: "lego" or "mug"
    seed: int = -1  # Random seed by default, set to specific value for reproducibility
    flip: bool = True  # Flip images 180° ([::-1, ::-1]) to match training data
                        # The red and blue only policy is False (and pre that), after should be true
    # Training data mode - use exact initial states from training demos
    training_data_hdf5: str = ""  # Path to raw HDF5 file with training demos
    demo_idx: int = 0  # Which demo to use from the HDF5 file


def test_primitives_from_reset(args: Args) -> None:
    """Test each primitive starting from the SAME reset state."""
    from datetime import datetime, timezone, timedelta

    # Use PST timezone (UTC-8)
    pst = timezone(timedelta(hours=-8))

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
    else:  # default: lego
        primitives = [
            # In-distribution (trained on these)
            "move gripper to the red lego block",
            "move gripper to the blue lego block",
            # "move gripper to target",
            # "close gripper",
            # "open gripper",
            # "lift upward",
            # "lower gripper",
            # # Out-of-distribution (not trained on these)
            # "rotate wrist 180 degrees clockwise",
            # "lower gripper to target",
            # "move gripper to the green target",
            # "pick up the red lego block",
        ]
        bddl_file = pathlib.Path(get_libero_path("bddl_files")) / "lego_primitives" / "wide_range" / "pick_blue_place_target_wide.bddl"

    # 90th percentile durations from training data (frames at 10Hz)
    PRIMITIVE_P90_DURATIONS = {
        "close gripper": 8,
        "lift upward": 127,
        "lower gripper": 60,
        "move gripper to target": 45,
        "move gripper to the blue lego block": 188,
        "move gripper to the red lego block": 234,
        "move gripper to the target zone": 89,
        "open gripper": 22,
    }
    DEFAULT_DURATION = 200  # Fallback for unknown primitives

    if not bddl_file.exists():
        logging.error(f"BDDL file not found: {bddl_file}")
        return

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION
    )
    # Wrap with visualization to match training data (green line + red dot on EE)
    env = SimpleVisualizationWrapper(env)
    env.seed(args.seed)

    # Create timestamped output directory with date folder (matches test_primitives.py)
    timestamp = datetime.now(pst).strftime("%Y%m%d_%H%M%S")
    date_folder = datetime.now(pst).strftime("%Y-%m-%d")
    if args.run_name:
        run_dir = args.run_name
    else:
        run_dir = f"run_{timestamp}"

    output_dir = pathlib.Path(args.video_out_path) / date_folder / run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Get checkpoint info from server metadata
    server_metadata = client.get_server_metadata()
    checkpoint_config = server_metadata.get("checkpoint_config", "unknown")
    checkpoint_dir = server_metadata.get("checkpoint_dir", "unknown")
    checkpoint_info = f"{checkpoint_config} @ {checkpoint_dir}"
    if args.checkpoint:  # Override if manually specified
        checkpoint_info = args.checkpoint

    # Define in-distribution vs out-of-distribution primitives
    in_distribution = [
        "move gripper to the red lego block",
        "move gripper to the blue lego block",
        "move gripper to target",
        "close gripper",
        "open gripper",
        "lift upward",
        "lower gripper",
    ]

    # Create summary file
    summary_file = output_dir / "summary.txt"
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("VLA Primitive Steerability Test")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    summary_lines.append("METADATA:")
    summary_lines.append(f"  Run: {run_dir}")
    summary_lines.append(f"  Timestamp: {datetime.now(pst).strftime('%Y-%m-%d %H:%M:%S')} PST")
    summary_lines.append(f"  Task: {args.task}")
    summary_lines.append(f"  Seed: {args.seed}")
    if args.num_steps_per_primitive > 0:
        summary_lines.append(f"  Steps per primitive: {args.num_steps_per_primitive} (fixed)")
    else:
        summary_lines.append(f"  Steps per primitive: p90 from training data")
    summary_lines.append("")
    summary_lines.append("CHECKPOINT:")
    summary_lines.append(f"  Config: {server_metadata.get('checkpoint_config', 'unknown')}")
    summary_lines.append(f"  Dir: {server_metadata.get('checkpoint_dir', 'unknown')}")
    summary_lines.append("")
    summary_lines.append("ENVIRONMENT:")
    summary_lines.append(f"  BDDL: {bddl_file}")
    if args.training_data_hdf5:
        summary_lines.append(f"  Initial state: FROM TRAINING DATA")
        summary_lines.append(f"    HDF5: {args.training_data_hdf5}")
        summary_lines.append(f"    Demo index: {args.demo_idx}")
    else:
        summary_lines.append(f"  Initial state: RANDOM (seed={args.seed})")
    summary_lines.append("")
    summary_lines.append("PRIMITIVES TESTED:")
    summary_lines.append(f"  In-distribution ({len([p for p in primitives if p in in_distribution])}):")
    for p in primitives:
        if p in in_distribution:
            summary_lines.append(f"    - {p}")
    summary_lines.append(f"  Out-of-distribution ({len([p for p in primitives if p not in in_distribution])}):")
    for p in primitives:
        if p not in in_distribution:
            summary_lines.append(f"    - {p}")
    summary_lines.append("")
    summary_lines.append("Each primitive starts from the SAME reset state.")
    summary_lines.append("If VLA is steerable, each video should show different behavior.")
    summary_lines.append("If VLA ignores language, all videos will show identical pick-and-place.")
    summary_lines.append("")

    logging.info("=" * 80)
    logging.info("Testing Primitives from SAME Reset State")
    logging.info("Each primitive starts from identical initial conditions")
    logging.info("=" * 80)

    primitive_first_actions = {}

    # Reset ONCE and save state - all primitives will start from this exact state
    if args.training_data_hdf5:
        # Load initial state from training data HDF5
        logging.info(f"Loading initial state from training data: {args.training_data_hdf5}")
        logging.info(f"Using demo index: {args.demo_idx}")

        with h5py.File(args.training_data_hdf5, "r") as f:
            demo_keys = sorted([k for k in f["data"].keys() if k.startswith("demo_")])
            if args.demo_idx >= len(demo_keys):
                logging.error(f"Demo index {args.demo_idx} out of range (only {len(demo_keys)} demos)")
                return

            demo_key = demo_keys[args.demo_idx]
            demo = f[f"data/{demo_key}"]

            # Get the initial state (first state in the trajectory)
            training_init_state = demo["states"][0]
            logging.info(f"Loaded initial state from {demo_key}, shape: {training_init_state.shape}")

            # Get model XML to recreate exact scene (object positions are baked in)
            model_xml = None
            if "model_file" in demo.attrs:
                model_xml = demo.attrs["model_file"]
                logging.info(f"Loaded model XML from demo ({len(model_xml)} chars)")

            # Get language instruction if available
            if "language_instruction" in demo.attrs:
                logging.info(f"Training task: {demo.attrs['language_instruction']}")

        # Reset env, then set full MuJoCo state (which includes object positions)
        # The state contains qpos for all joints including free joints for objects,
        # so set_state_from_flattened will restore exact object positions
        env.reset()
        logging.info("Applying training state (includes object positions in qpos)...")
        obs = env.set_init_state(training_init_state)

        # Run stabilization steps with zero action to let physics settle
        # This matches how training data collection started
        zero_action = [0.0] * 6 + [-1.0]  # zero motion + open gripper
        for _ in range(args.num_steps_wait):
            obs, _, _, _ = env.step(zero_action)

        # Save the stabilized state
        saved_state = env.get_sim_state()
        saved_obs = obs.copy()
        logging.info(f"Set environment to TRAINING DATA initial state (demo {args.demo_idx})")
    else:
        # Original behavior: random reset
        obs = env.reset()

        # Stabilization + open gripper to match training data start state
        open_gripper_action = [0.0] * 6 + [-1.0]
        for _ in range(args.num_steps_wait):
            obs, _, _, _ = env.step(open_gripper_action)

        # Save this state - we'll restore it for each primitive
        saved_state = env.get_sim_state()
        saved_obs = obs.copy()
        logging.info(f"Saved initial state. All primitives will start from this EXACT scene.")

    # Execute each primitive from the SAME saved state
    for prim_idx, prompt in enumerate(primitives):
        # Determine steps for this primitive (p90 from training data or override)
        if args.num_steps_per_primitive > 0:
            num_steps = args.num_steps_per_primitive
        else:
            num_steps = PRIMITIVE_P90_DURATIONS.get(prompt, DEFAULT_DURATION)

        logging.info(f"\n{'=' * 80}")
        logging.info(f"[{prim_idx+1}/{len(primitives)}] Testing primitive: '{prompt}' ({num_steps} steps)")
        logging.info(f"{'=' * 80}")

        # Reset first to clear internal done flag, then restore physics state
        env.reset()
        obs = env.set_init_state(saved_state)

        logging.info(f"Restored to saved initial state. Same scene as other primitives.")

        primitive_images = []

        # Execute this primitive for N steps
        for step in range(num_steps):
            # Get preprocessed images
            if args.flip:
                # Flip both axes (180° rotation) to match official LIBERO convention
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            else:
                # Flip vertical only
                img = np.ascontiguousarray(obs["agentview_image"][::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
            )
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
            )

            # Annotate frame
            annotated_img = annotate_frame(img, f"[{prim_idx+1}/5] {prompt}", step)
            primitive_images.append(annotated_img)

            # Prepare observation
            # NOTE: Training data uses joint_pos (7D) + gripper (1D) = 8D state
            # NOT EEF pos + axis-angle like other LIBERO examples
            element = {
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": np.concatenate((
                    obs["robot0_joint_pos"],  # 7D joint angles
                    obs["robot0_gripper_qpos"][:1],  # 1D gripper (truncated to match training)
                )),
                "prompt": prompt,
            }

            # Get action from policy
            action_chunk = client.infer(element)["actions"]
            action = action_chunk[0]

            # Store first action
            if step == 0:
                primitive_first_actions[prompt] = action.copy()
                logging.info(f"  First action: pos=[{action[0]:+.4f}, {action[1]:+.4f}, {action[2]:+.4f}] "
                           f"rot=[{action[3]:+.4f}, {action[4]:+.4f}, {action[5]:+.4f}] "
                           f"gripper={action[6]:+.4f}")

            # Execute
            obs, reward, done, info = env.step(action.tolist())

            if done:
                logging.info(f"  Task completed at step {step}!")
                break

        # Save video for this primitive
        prim_name = prompt.replace(" ", "_")[:40]
        prim_video_path = output_dir / f"reset_{prim_idx+1}_{prim_name}.mov"
        imageio.mimwrite(prim_video_path, [np.asarray(x) for x in primitive_images], fps=10, codec="libx264")
        logging.info(f"  Saved video: {prim_video_path}")

        # Add to summary
        summary_lines.append(f"VIDEO {prim_idx+1}: reset_{prim_idx+1}_{prim_name}.mov")
        summary_lines.append(f"  Prompt: \"{prompt}\"")
        summary_lines.append(f"  First action: pos=[{primitive_first_actions[prompt][0]:+.4f}, {primitive_first_actions[prompt][1]:+.4f}, {primitive_first_actions[prompt][2]:+.4f}]")
        summary_lines.append(f"                rot=[{primitive_first_actions[prompt][3]:+.4f}, {primitive_first_actions[prompt][4]:+.4f}, {primitive_first_actions[prompt][5]:+.4f}]")
        summary_lines.append(f"                gripper={primitive_first_actions[prompt][6]:+.4f}")
        summary_lines.append("")

    # Analyze action similarity
    logging.info("\n" + "=" * 80)
    logging.info("VLA Action Similarity Analysis")
    logging.info("(All primitives started from SAME initial state)")
    logging.info("=" * 80)

    prompts_list = list(primitive_first_actions.keys())
    all_diffs = []

    for i in range(len(prompts_list)):
        for j in range(i+1, len(prompts_list)):
            p1, p2 = prompts_list[i], prompts_list[j]
            diff = np.abs(primitive_first_actions[p1] - primitive_first_actions[p2])
            mean_diff = diff.mean()
            all_diffs.append(mean_diff)

            logging.info(f"\n[{i+1}] '{p1}'")
            logging.info(f"[{j+1}] '{p2}'")
            logging.info(f"  Position diff: [{diff[0]:.4f}, {diff[1]:.4f}, {diff[2]:.4f}] (mean: {diff[:3].mean():.4f})")
            logging.info(f"  Rotation diff: [{diff[3]:.4f}, {diff[4]:.4f}, {diff[5]:.4f}] (mean: {diff[3:6].mean():.4f})")
            logging.info(f"  Gripper diff: {diff[6]:.4f}")
            logging.info(f"  Mean diff: {mean_diff:.6f}")

            if mean_diff < 0.0001:
                logging.info(f"  ❌ IDENTICAL - VLA completely ignores language!")
            elif mean_diff < 0.001:
                logging.info(f"  ⚠️  Nearly identical - VLA mostly ignoring language")
            elif mean_diff < 0.01:
                logging.info(f"  🤔 Small difference - weak language sensitivity")
            else:
                logging.info(f"  ✓ Significant difference - VLA may respond to language")

    # Overall statistics
    logging.info("\n" + "=" * 80)
    logging.info("CONCLUSION")
    logging.info("=" * 80)

    all_actions = np.array([primitive_first_actions[p] for p in prompts_list])
    variance = np.var(all_actions, axis=0)
    mean_variance = variance.mean()
    mean_pairwise_diff = np.mean(all_diffs)

    logging.info(f"\nVariance across all {len(primitives)} prompts:")
    logging.info(f"  Position: {variance[:3].mean():.6f}")
    logging.info(f"  Rotation: {variance[3:6].mean():.6f}")
    logging.info(f"  Gripper: {variance[6]:.6f}")
    logging.info(f"  Overall: {mean_variance:.6f}")
    logging.info(f"  Mean pairwise diff: {mean_pairwise_diff:.6f}")

    # Build conclusion for summary
    summary_lines.append("=" * 80)
    summary_lines.append("ANALYSIS")
    summary_lines.append("=" * 80)
    summary_lines.append(f"Mean pairwise action difference: {mean_pairwise_diff:.6f}")
    summary_lines.append(f"Action variance across primitives: {mean_variance:.6f}")
    summary_lines.append("")

    if mean_variance < 0.00001:
        conclusion = "IDENTICAL - VLA produces IDENTICAL actions for all primitives"
        details = [
            "VLA is NOT language-sensitive at primitive level",
            "All 5 videos should show identical pick-and-place behavior",
            "Language prompts are completely ignored"
        ]
        logging.info("\n❌ RESULT: VLA produces IDENTICAL actions for all primitives")
        logging.info("   → VLA is NOT language-sensitive at primitive level")
        logging.info("   → All 5 videos should show identical pick-and-place behavior")
        logging.info("   → Language prompts are completely ignored")
    elif mean_variance < 0.001:
        conclusion = "WEAK - VLA shows WEAK language sensitivity"
        details = ["Small differences may be noise, not meaningful"]
        logging.info("\n🤔 RESULT: VLA shows WEAK language sensitivity")
        logging.info("   → Small differences may be noise, not meaningful")
    else:
        conclusion = "RESPONSIVE - VLA responds to different language prompts"
        details = ["Different primitives produce different behaviors"]
        logging.info("\n✓ RESULT: VLA responds to different language prompts")
        logging.info("   → Different primitives produce different behaviors")

    summary_lines.append(f"CONCLUSION: {conclusion}")
    for d in details:
        summary_lines.append(f"  - {d}")
    summary_lines.append("")
    summary_lines.append("=" * 80)

    # Write summary file
    with open(summary_file, 'w') as f:
        f.write("\n".join(summary_lines))

    logging.info("\n" + "=" * 80)
    logging.info(f"Results saved to: {output_dir}")
    logging.info(f"  - summary.txt (analysis and conclusions)")
    logging.info(f"  - reset_*.mov (video for each primitive)")
    logging.info("Compare the 5 videos to see if robot behavior differs")
    logging.info("=" * 80)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    tyro.cli(test_primitives_from_reset)
