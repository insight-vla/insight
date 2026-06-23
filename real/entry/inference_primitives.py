"""Run a primitive-conditioned policy on the real xArm.

Mirrors inference_simple.py's real-robot control path (RealSense cameras,
interpolate_action with Z floor, 20 Hz outer loop) but executes a sequence
of primitives by swapping the `prompt` field per primitive.

Durations per primitive default to the Nth percentile (default p90) of
per-task episode lengths from the training LeRobot dataset, matching the
LIBERO test_primitives.py convention. These can be overridden via
--fixed-duration or computed over a different percentile.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pyrealsense2 as rs
import tyro
from PIL import Image, ImageDraw, ImageFont
from xarm.wrapper import XArmAPI

from openpi_client import websocket_client_policy

import vlm_check
from insight.voting import TemporalConsistency

FPS = 20.0  # overridden in main() from dataset info.json or --fps
DT = 1.0 / FPS
CONTROL_HZ = 40.0
Z_SAFETY_FLOOR = 198.0

SERIAL_EXTERNAL = "244222071219"
SERIAL_WRIST = "317222072257"
ARM_IP = "192.168.1.219"

DATASET_HOME = Path.home() / ".cache/huggingface/lerobot"

# Populated in main()
policy: websocket_client_policy.WebsocketClientPolicy | None = None
arm: XArmAPI | None = None
camera_pipelines: dict[str, rs.pipeline] = {}


@dataclasses.dataclass
class Args:
    host: str = "localhost"
    port: int = 8000
    api_key: str | None = None

    primitives: tuple[str, ...] = (
        "move gripper to the rocks",
        "scoop the rocks",
        "lift upward",
    )
    """Ordered primitive prompts. Default is the scoop sequence; override with
    --primitives \"<step1>\" \"<step2>\" ... for any other task (e.g. drawer
    open/close, rotate block)."""

    durations_from_dataset: str = ""
    """LeRobot repo_id to pull per-primitive p90 durations from (e.g. 'maggie/xarm_scoop_100_primitives_trimmed')."""

    duration_percentile: int = 90
    """Percentile of per-task episode lengths to use as per-primitive duration."""

    fixed_duration: int = 0
    """If >0, use this many 20 Hz steps for every primitive regardless of dataset stats."""

    default_duration: int = 100
    """Fallback duration (steps) for primitives missing from the dataset."""

    replan_steps: int = 10
    """Actions executed from each policy chunk before re-querying. Should be <= action_horizon."""

    auto_advance: bool = True
    """Advance to next primitive early when goal-to-state distance stays small."""

    movement_threshold_mm: float = 1.0
    """Below this goal-to-state distance (xyz mm) the primitive is considered stopped."""

    stop_steps: int = 15
    """Consecutive below-threshold steps needed to trigger auto-advance."""

    min_steps: int = 30
    """Steps before auto-advance checks begin."""

    z_safety_floor: float = Z_SAFETY_FLOOR
    """Minimum commanded Z in mm. Set conservatively; never hard-contacts below this."""

    fps: float = 0.0
    """Outer-loop rate in Hz. 0 = auto-read from dataset info.json. Must match training rate."""

    interp_divisor: float = 1.0
    """Damping factor in interpolate_action. 1=no damping (arm reaches goal each step), 6=inference_simple.py default. Lower = more responsive / more jerky."""

    save_videos: bool = True
    """Record side-by-side (exterior | wrist) mp4s per primitive + one combined."""

    video_dir: str = "data/primitive_runs"
    """Root directory for saved videos and metadata."""

    run_name: str = ""
    """Subdir name under video_dir. Defaults to a timestamp."""

    dry_run: bool = False
    """Print primitives + durations and exit. No arm or policy connection."""

    use_vlm_check: bool = False
    """Enable VLM completion check on primitives starting with 'move gripper to'."""

    vlm_provider: str = "gemini"
    """VLM provider name (see real/entry/vlm_check.py: 'gemini' or 'gpt')."""

    vlm_check_interval_steps: int = 10
    """Steps between VLM completion polls (10 ≈ 0.5s @ 20Hz)."""

    vlm_warmup_steps: int = 30
    """Don't poll the VLM until this many steps into the primitive."""

    vlm_num_votes: int = 1
    """VLM calls per check. 1 = fastest; 3 = majority vote, blocks ~3x as long."""

    vlm_consecutive_required: int = 3
    """Number of consecutive True checks required to fire [VLM-DONE]. 1 = disable
    (advance on first True). Higher values filter one-off false positives at the
    cost of (N-1) * vlm_check_interval_steps extra steps before transition."""

    z_completion_threshold_mm: float = 220.0
    """Safety fallback for 'move gripper to ...': break primitive when state_z < this. 0 disables."""


def load_dataset_meta(repo_id: str, percentile: int) -> tuple[dict[str, int], float]:
    """Return (per-task durations at Nth percentile, dataset fps)."""
    root = DATASET_HOME / repo_id
    episodes_path = root / "meta/episodes.jsonl"
    info_path = root / "meta/info.json"
    if not episodes_path.exists():
        raise SystemExit(f"episodes.jsonl not found under {root}; is the dataset rsynced?")
    if not info_path.exists():
        raise SystemExit(f"info.json not found under {root}; is the dataset rsynced?")

    with open(info_path) as f:
        info = json.load(f)
    fps = float(info.get("fps", 0.0))
    if fps <= 0:
        raise SystemExit(f"info.json at {info_path} missing a usable 'fps' field")

    from collections import defaultdict
    lengths_by_task: dict[str, list[int]] = defaultdict(list)
    with open(episodes_path) as f:
        for line in f:
            ep = json.loads(line)
            for task in ep["tasks"]:
                lengths_by_task[task].append(ep["length"])

    durations: dict[str, int] = {}
    for task, lengths in lengths_by_task.items():
        if percentile >= 100:
            durations[task] = int(max(lengths))
        else:
            durations[task] = int(np.percentile(lengths, percentile))
    return durations, fps


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


def capture_frames() -> tuple[np.ndarray, np.ndarray]:
    """Grab the latest exterior + wrist RGB frames. Blocks briefly for the next frame."""
    assert camera_pipelines
    f_ext = camera_pipelines[SERIAL_EXTERNAL].wait_for_frames()
    f_wrist = camera_pipelines[SERIAL_WRIST].wait_for_frames()
    exterior = np.asanyarray(f_ext.get_color_frame().get_data()).copy()
    wrist = np.asanyarray(f_wrist.get_color_frame().get_data()).copy()
    return exterior, wrist


def build_observation(prompt: str, exterior: np.ndarray, wrist: np.ndarray) -> dict:
    assert arm is not None
    pose = arm.get_position()[1]
    pose[3] = pose[3] % 360
    pose[5] = pose[5] % 360
    angles_rad = (np.array(pose[3:6]) * np.pi / 180).tolist()
    state = np.array(pose[:3] + angles_rad, dtype=np.float32)
    return {
        "observation/exterior_image_1_left": exterior,
        "observation/wrist_image_left": wrist,
        "observation/state": state,
        "prompt": prompt,
    }


def _annotate(frame: np.ndarray, title: str) -> np.ndarray:
    """Add a black title bar above the frame."""
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    h_bar = 22
    img = Image.fromarray(frame)
    out = Image.new("RGB", (img.width, img.height + h_bar), (20, 20, 20))
    out.paste(img, (0, h_bar))
    draw = ImageDraw.Draw(out)
    draw.text((6, 3), title, fill=(255, 255, 255), font=font)
    return np.asarray(out)


def write_video(labeled_frames: list[tuple[str, np.ndarray, np.ndarray]], path: Path, fps: float) -> None:
    """Write a side-by-side (exterior | wrist) annotated mp4. Each frame carries its own label."""
    out = []
    for label, ext, wrist in labeled_frames:
        composite = np.hstack([ext, wrist])
        out.append(_annotate(composite, label))
    imageio.mimwrite(str(path), out, fps=int(round(fps)), codec="libx264")


def interpolate_action(state: np.ndarray, goal: np.ndarray, z_floor: float, divisor: float = 6.0) -> None:
    """Copy of inference_simple.py:interpolate_action with configurable Z floor and damping."""
    delta_increment = (goal - state) / (DT * CONTROL_HZ * divisor)

    for _ in range(int(DT * CONTROL_HZ)):
        loop_start = time.perf_counter()
        command = state + delta_increment
        command[3] = (command[3] + 180) % 360 - 180
        command[5] = (command[5] + 180) % 360 - 180

        x, y, z, roll, pitch, yaw = command
        if z < z_floor:
            print(f"[Z-CLIP] {z:.2f} -> {z_floor:.2f}")
        command[2] = max(z, z_floor)

        arm.set_servo_cartesian(command, speed=100, mvacc=1000)

        time_left = (1 / CONTROL_HZ) - (time.perf_counter() - loop_start)
        time.sleep(max(time_left, 0))


def run_primitive(
    primitive: str,
    num_steps: int,
    args: Args,
    frames_out: list[tuple[int, np.ndarray, np.ndarray]] | None = None,
) -> None:
    """Execute one primitive for up to num_steps with auto-advance check.

    If frames_out is not None, appends (step, exterior, wrist) each tick.
    """
    action_queue: collections.deque = collections.deque()
    consecutive_stop = 0
    vlm_streak = TemporalConsistency(required=max(1, args.vlm_consecutive_required))

    is_move_to = primitive.lower().startswith("move gripper to")
    vlm_active = args.use_vlm_check and is_move_to
    z_safety_active = is_move_to and args.z_completion_threshold_mm > 0

    for step in range(num_steps):
        t0 = time.perf_counter()

        exterior, wrist = capture_frames()
        if frames_out is not None:
            frames_out.append((step, exterior, wrist))

        if not action_queue:
            obs = build_observation(primitive, exterior, wrist)
            chunk = np.asarray(policy.infer(obs)["actions"], dtype=np.float32)
            for a in chunk[: args.replan_steps]:
                action_queue.append(a)

        action = action_queue.popleft()

        pose = arm.get_position()[1]
        pose[3] = pose[3] % 360
        pose[5] = pose[5] % 360
        state = np.array(pose, dtype=np.float32)

        goal = np.array(action[:6], dtype=np.float32)
        goal[3:6] = goal[3:6] / np.pi * 180

        if vlm_active and step >= args.vlm_warmup_steps and step % args.vlm_check_interval_steps == 0:
            verdict = vlm_check.check_primitive_done(
                primitive, exterior, wrist,
                num_votes=args.vlm_num_votes, step_num=step,
            )
            if vlm_streak.update(verdict):
                logging.info("  [VLM-DONE] '%s' at step %d/%d (streak=%d)",
                             primitive, step, num_steps, vlm_streak.streak)
                return
            if verdict:
                logging.info("  [VLM-MAYBE] '%s' streak=%d/%d at step %d/%d",
                             primitive, vlm_streak.streak, vlm_streak.required, step, num_steps)

        if z_safety_active and state[2] < args.z_completion_threshold_mm:
            logging.info("  [Z-DONE] '%s' state_z=%.2f < %.2f at step %d/%d",
                         primitive, state[2], args.z_completion_threshold_mm, step, num_steps)
            return

        if args.auto_advance and step >= args.min_steps:
            goal_delta_mm = float(np.linalg.norm(goal[:3] - state[:3]))
            if goal_delta_mm < args.movement_threshold_mm:
                consecutive_stop += 1
                if consecutive_stop >= args.stop_steps:
                    logging.info("  Auto-advance: goal-state delta < %.2fmm for %d steps",
                                 args.movement_threshold_mm, consecutive_stop)
                    return
            else:
                consecutive_stop = 0

        if step % 20 == 0 or step < 3:
            logging.info("  step %3d/%d  state_z=%.2f  goal_z=%.2f  goal_delta=%.2fmm",
                         step, num_steps, state[2], goal[2],
                         float(np.linalg.norm(goal[:3] - state[:3])))

        interpolate_action(state, goal, args.z_safety_floor, args.interp_divisor)

        time_left = DT - (time.perf_counter() - t0)
        time.sleep(max(time_left, 0))

    logging.info("  [P90-CAP] '%s' ran full %d steps without VLM/Z trigger", primitive, num_steps)


def main(args: Args) -> None:
    global policy, arm, FPS, DT
    logging.basicConfig(level=logging.INFO, force=True)

    if not args.primitives:
        raise SystemExit("Primitive list is empty.")

    # Resolve per-primitive durations and loop rate
    dataset_durations: dict[str, int] = {}
    dataset_fps: float | None = None
    if args.durations_from_dataset:
        dataset_durations, dataset_fps = load_dataset_meta(
            args.durations_from_dataset, args.duration_percentile
        )

    # FPS precedence: explicit --fps > dataset info.json > module default
    if args.fps > 0:
        FPS = float(args.fps)
        logging.info("Using FPS=%.1f from --fps override", FPS)
    elif dataset_fps is not None:
        FPS = dataset_fps
        logging.info("Using FPS=%.1f from dataset info.json", FPS)
    else:
        logging.info("Using FPS=%.1f (module default; no dataset or override)", FPS)
    DT = 1.0 / FPS

    if dataset_durations:
        logging.info("Loaded p%d durations from %s:",
                     args.duration_percentile, args.durations_from_dataset)
        for task, dur in sorted(dataset_durations.items(), key=lambda kv: -kv[1]):
            logging.info("  %-60s %4d steps  (%4.1fs @ %.0fHz)",
                         task, dur, dur / FPS, FPS)

    plan: list[tuple[str, int]] = []
    for p in args.primitives:
        if args.fixed_duration > 0:
            dur = args.fixed_duration
        elif p in dataset_durations:
            dur = dataset_durations[p]
        else:
            logging.warning("No dataset duration for '%s'; using default=%d", p, args.default_duration)
            dur = args.default_duration
        plan.append((p, dur))

    total_steps = sum(d for _, d in plan)
    logging.info("\nExecution plan: %d primitives, %d total %.0fHz steps (~%.1fs)",
                 len(plan), total_steps, FPS, total_steps / FPS)
    for i, (p, d) in enumerate(plan):
        logging.info("  [%d] %-60s %4d steps", i + 1, p, d)

    if args.use_vlm_check:
        vlm_check.set_provider(args.vlm_provider)
        logging.info("VLM check enabled: provider=%s warmup=%d interval=%d votes=%d consecutive=%d  |  Z safety: %.1fmm",
                     args.vlm_provider, args.vlm_warmup_steps, args.vlm_check_interval_steps,
                     args.vlm_num_votes, args.vlm_consecutive_required, args.z_completion_threshold_mm)

    if args.dry_run:
        logging.info("\nDry run: exiting before connecting to arm or policy.")
        return

    # Connect
    policy = websocket_client_policy.WebsocketClientPolicy(
        host=args.host, port=args.port, api_key=args.api_key
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

    current_pose = arm.get_position()[1]
    logging.info("Current arm pose: x=%.1f y=%.1f z=%.1f  rpy=(%.1f, %.1f, %.1f)",
                 *current_pose)

    run_dir: Path | None = None
    if args.save_videos:
        now = datetime.now()
        date_folder = now.strftime("%Y-%m-%d")
        stamp = now.strftime("%Y%m%d_%H%M%S")
        run_dir = Path(args.video_dir) / date_folder / (args.run_name or f"run_{stamp}")
        run_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Saving videos and metadata to: %s", run_dir)
        with open(run_dir / "metadata.txt", "w") as f:
            f.write(f"Run: {run_dir.name}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Server metadata: {policy.get_server_metadata()}\n")
            f.write(f"FPS: {FPS}\n")
            f.write(f"Replan steps: {args.replan_steps}\n")
            f.write(f"Auto-advance: {args.auto_advance} (thr={args.movement_threshold_mm}mm, stop={args.stop_steps}, min={args.min_steps})\n")
            f.write(f"Z safety floor: {args.z_safety_floor}\n")
            f.write(f"Interp divisor: {args.interp_divisor}\n")
            f.write(f"Duration source: {args.durations_from_dataset or 'default'} (p{args.duration_percentile})\n")
            f.write(f"VLM check: {args.use_vlm_check} (provider={args.vlm_provider}, "
                    f"warmup={args.vlm_warmup_steps}, interval={args.vlm_check_interval_steps}, "
                    f"votes={args.vlm_num_votes}, consecutive={args.vlm_consecutive_required}, "
                    f"z_done={args.z_completion_threshold_mm})\n")
            f.write(f"Plan:\n")
            for idx, (p, d) in enumerate(plan):
                f.write(f"  [{idx+1}] {p}  {d} steps\n")
            f.write(f"Command: {' '.join(sys.argv)}\n")

    input("Press Enter to start the primitive sequence, or Ctrl+C to abort... ")

    combined: list[tuple[str, np.ndarray, np.ndarray]] = []
    global_step = 0
    for i, (primitive, num_steps) in enumerate(plan):
        logging.info("\n[%d/%d] %s  (%d steps)", i + 1, len(plan), primitive, num_steps)
        prim_frames: list[tuple[int, np.ndarray, np.ndarray]] | None = [] if args.save_videos else None
        run_primitive(primitive, num_steps, args, frames_out=prim_frames)

        if args.save_videos and prim_frames is not None and run_dir is not None:
            slug = primitive.replace(" ", "_")[:40]
            prim_path = run_dir / f"primitive_{i+1:02d}_{slug}.mp4"
            prefix = f"[{i+1}/{len(plan)}] {primitive}"
            per_prim_labeled = [
                (f"{prefix}  step {step}", ext, wrist) for step, ext, wrist in prim_frames
            ]
            write_video(per_prim_labeled, prim_path, FPS)
            logging.info("  saved %s (%d frames)", prim_path.name, len(prim_frames))
            for step, ext, wrist in prim_frames:
                label = f"[{i+1}/{len(plan)}] {primitive}  step {step}  (global {global_step})"
                combined.append((label, ext, wrist))
                global_step += 1

    if args.save_videos and run_dir is not None and combined:
        combined_path = run_dir / "all_primitives.mp4"
        write_video(combined, combined_path, FPS)
        logging.info("Saved combined video: %s (%d frames)", combined_path.name, len(combined))

    logging.info("\nPrimitive sequence complete.")


if __name__ == "__main__":
    main(tyro.cli(Args))
