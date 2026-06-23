#!/usr/bin/env python3
"""Preprocess lego dataset to trim settling zeros from the start/end of each episode.
Keeps ALL primitives (unlike preprocess_red_blue_only.py which filters to red/blue only)."""

import json
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

# Config
# SOURCE_DATASET = Path.home() / ".cache/huggingface/lerobot/maggiewang/lego_primitives_01_30"
# OUTPUT_DATASET = Path.home() / ".cache/huggingface/lerobot/maggiewang/lego_primitives_v5_01_30_trimmed"
# SOURCE_DATASET = Path.home() / ".cache/huggingface/lerobot/maggiewang/lego_primitives_random_fixedflip"
# OUTPUT_DATASET = Path.home() / ".cache/huggingface/lerobot/maggiewang/lego_primitives_random_fixedflip_trimmed"
# SOURCE_DATASET = Path.home() / ".cache/huggingface/lerobot/maggiewang/lego_oracle_flip_140_primitives"
# OUTPUT_DATASET = Path.home() / ".cache/huggingface/lerobot/maggiewang/lego_oracle_flip_140_primitives_trimmed"
# SOURCE_DATASET = Path.home() / ".cache/huggingface/lerobot/maggiewang/lego_pickplace_tilted_150_primitives"
# OUTPUT_DATASET = Path.home() / ".cache/huggingface/lerobot/maggiewang/lego_pickplace_tilted_150_primitives_trimmed"
# SOURCE_DATASET = Path.home() / ".cache/huggingface/lerobot/maggiewang/lego_flywheel_flip_59_04_02"
# OUTPUT_DATASET = Path.home() / ".cache/huggingface/lerobot/maggiewang/lego_flywheel_flip_59_04_02_trimmed"
# SOURCE_DATASET = Path.home() / ".cache/huggingface/lerobot/maggiewang/lego_flywheel_flip_153_04_04"
# OUTPUT_DATASET = Path.home() / ".cache/huggingface/lerobot/maggiewang/lego_flywheel_flip_153_04_04_trimmed"
# SOURCE_DATASET = Path.home() / ".cache/huggingface/lerobot/maggiewang/lego_flywheel_flip_246_04_05"
# OUTPUT_DATASET = Path.home() / ".cache/huggingface/lerobot/maggiewang/lego_flywheel_flip_246_04_05_trimmed"
# SOURCE_DATASET = Path.home() / ".cache/huggingface/lerobot/maggiewang/drawer_open_top_50_04_07_primitives"
# OUTPUT_DATASET = Path.home() / ".cache/huggingface/lerobot/maggiewang/drawer_open_top_50_04_07_primitives_trimmed"
# SOURCE_DATASET = Path.home() / ".cache/huggingface/lerobot/maggiewang/drawer_open_top_50_04_09_primitives_progress"
# OUTPUT_DATASET = Path.home() / ".cache/huggingface/lerobot/maggiewang/drawer_open_top_50_04_09_primitives_progress_trimmed"
# SOURCE_DATASET = Path.home() / ".cache/huggingface/lerobot/maggiewang/drawer_flywheel_push_70_04_12"
# OUTPUT_DATASET = Path.home() / ".cache/huggingface/lerobot/maggiewang/drawer_flywheel_push_70_04_12_trimmed"
# SOURCE_DATASET = Path.home() / ".cache/huggingface/lerobot/maggiewang/xarm_scoop_100_primitives"
# OUTPUT_DATASET = Path.home() / ".cache/huggingface/lerobot/maggiewang/xarm_scoop_100_primitives_trimmed"
SOURCE_DATASET = Path.home() / ".cache/huggingface/lerobot/maggie/xarm_pick_up_top_30_primitives"
OUTPUT_DATASET = Path.home() / ".cache/huggingface/lerobot/maggie/xarm_pick_up_top_30_primitives_trimmed_v2"

# Threshold for detecting "settling" frames (position action magnitude)
SETTLING_THRESHOLD = 0.01  # Actions with position magnitude below this are trimmed


def compute_trim_indices(
    actions: np.ndarray,
    task: str,
    xarm: bool = False,
    delta_threshold: float = 0.5,  # leading-trim threshold (mm + rad); xarm only
    end_delta_threshold: float = 0.05,  # trailing-trim threshold; lower so we only cut truly-still frames (preserves precise end-of-motion frames like final approach)
) -> tuple[int, int]:
    """Compute trim start/end indices.

    libero: actions are deltas (dx, dy, dz, ...); we trim frames where the
            delta magnitude is below SETTLING_THRESHOLD.
    xarm:   actions are absolute pose targets, so the delta magnitude check
            is meaningless (norm always ~600mm). Instead we look at
            frame-to-frame *changes* in the action: a stationary teleop
            produces near-identical consecutive targets, so trim where the
            change between actions[i-1] and actions[i] is below
            ``delta_threshold`` (mm + rad). End trim uses a stricter
            (smaller) ``end_delta_threshold`` so precise final-approach
            frames are kept; only truly motionless trailing frames are cut.
    Gripper primitives ("close gripper" / "open gripper"): same logic in
    both modes — look at the gripper channel (action[6]) magnitude. For
    xarm (0=open, ~0.5+=closed) this trims leading-open / trailing-open
    settling frames.
    """
    trim_start = 0
    trim_end = len(actions)
    is_gripper_primitive = "close gripper" in task.lower() or "open gripper" in task.lower()

    if is_gripper_primitive:
        if xarm:
            # xarm: action[6] is in [0=open, ~0.565=closed]. The OPEN raw value
            # reads as 0.0105 due to encoder noise, so the magnitude check
            # (>0.01) would never trim the long "still open" prefix that VLM
            # segmentation leaves at the start of a close-gripper segment.
            #
            # Use SUSTAINED frame-to-frame motion as the criterion: trim until
            # we see N consecutive frames each with |diff| > thresh. This keeps
            # any frame where the gripper IS moving (even slowly), but rejects
            # single-frame sensor noise (e.g. 0.010 -> 0.013 jitter back at the
            # start that we mistook for motion before).
            if actions.shape[1] > 6:
                gdiffs = np.abs(np.diff(actions[:, 6]))  # len = n-1
                gthresh = 0.001       # tiny -- any real motion clears this
                sustain_n = 3         # require 3 consecutive frames of motion
                # Trim from start: first index where the next sustain_n diffs all > thresh
                for i in range(len(gdiffs) - sustain_n + 1):
                    if all(gdiffs[i + k] > gthresh for k in range(sustain_n)):
                        trim_start = i
                        break
                # Trim from end: last index where the prior sustain_n diffs all > thresh
                for i in range(len(gdiffs) - 1, sustain_n - 2, -1):
                    if all(gdiffs[i - k] > gthresh for k in range(sustain_n)):
                        trim_end = i + 2  # +2: diffs[i] = action[i+1]-action[i]
                        break
        else:
            # libero (bipolar gripper, action[6] in {-1,+1}): magnitude check is fine.
            for i in range(len(actions)):
                g = abs(actions[i, 6]) if actions.shape[1] > 6 else 0
                if g > SETTLING_THRESHOLD:
                    trim_start = i
                    break
            for i in range(len(actions) - 1, -1, -1):
                g = abs(actions[i, 6]) if actions.shape[1] > 6 else 0
                if g > SETTLING_THRESHOLD:
                    trim_end = i + 1
                    break
        return trim_start, trim_end

    # Non-gripper primitive: motion-based trim.
    if xarm:
        # Frame-to-frame change in absolute pose. Stationary teleop → near-zero change.
        diffs = np.linalg.norm(np.diff(actions[:, :6], axis=0), axis=1)  # len = n-1
        for i, d in enumerate(diffs):
            if d > delta_threshold:
                trim_start = i
                break
        # Use stricter (smaller) threshold for trailing trim — preserves the
        # last few frames of precise approach where motion is small but real.
        for i in range(len(diffs) - 1, -1, -1):
            if diffs[i] > end_delta_threshold:
                trim_end = i + 2  # +2 because diffs[i] = actions[i+1] - actions[i]
                break
    else:
        # Libero: actions are deltas, magnitude check works directly.
        for i in range(len(actions)):
            if np.linalg.norm(actions[i, :6]) > SETTLING_THRESHOLD:
                trim_start = i
                break
        for i in range(len(actions) - 1, -1, -1):
            if np.linalg.norm(actions[i, :6]) > SETTLING_THRESHOLD:
                trim_end = i + 1
                break

    return trim_start, trim_end


def main(recompute_progress: bool = False, xarm: bool = False,
         delta_threshold: float = 0.5, end_delta_threshold: float = 0.05,
         source: str = "", output: str = "",
         save_videos: int = 0,
         video_image_key: str = "exterior_image_1_left",
         video_fps: int = 10):
    """``source`` / ``output`` override the module-level SOURCE_DATASET /
    OUTPUT_DATASET constants when provided. Backwards-compatible: empty
    means use the hardcoded constants at the top of this file.

    ``save_videos`` (default 0) renders the trimmed range of the first N
    primitive episodes as labeled QA videos under
    ``OUTPUT_DATASET/qa_videos/`` so you can visually verify the trim.
    Each video is annotated with the primitive label and source frame
    indices. ``video_image_key`` selects which camera to render
    (default ``exterior_image_1_left`` for xArm; use ``image`` for libero).
    """
    global SOURCE_DATASET, OUTPUT_DATASET
    if source:
        SOURCE_DATASET = Path(source).expanduser()
    if output:
        OUTPUT_DATASET = Path(output).expanduser()
    print(f"Source: {SOURCE_DATASET}")
    print(f"Output: {OUTPUT_DATASET}")
    if recompute_progress:
        print("recompute_progress=True: action[7] will be rewritten to frame_index/(n-1) per segment")
    if xarm:
        print(f"xarm=True: trim non-gripper primitives by frame-to-frame action change "
              f"(start threshold={delta_threshold}, end threshold={end_delta_threshold})")

    # Create output directory
    if OUTPUT_DATASET.exists():
        print(f"Removing existing dataset at {OUTPUT_DATASET}")
        shutil.rmtree(OUTPUT_DATASET)
    OUTPUT_DATASET.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DATASET / "meta").mkdir(exist_ok=True)
    (OUTPUT_DATASET / "data").mkdir(exist_ok=True)

    # Get HuggingFace metadata from source parquet (need to preserve for proper image loading)
    sample_parquet = SOURCE_DATASET / "data/chunk-000/episode_000000.parquet"
    hf_metadata = pq.read_table(sample_parquet).schema.metadata

    # Load episodes metadata
    episodes = []
    with open(SOURCE_DATASET / "meta/episodes.jsonl") as f:
        for line in f:
            episodes.append(json.loads(line))

    # QA video setup. Render the first N DEMOS (each demo = consecutive
    # primitive episodes from the same source teleop, glued into one video)
    # so the user can eyeball boundaries by watching primitive labels
    # change in real time.
    qa_dataset = None
    qa_videos_dir = None
    demo_idx_for_ep_idx = {}  # episode["episode_index"] -> demo_idx
    if save_videos > 0:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        qa_dataset = LeRobotDataset(root=SOURCE_DATASET, repo_id="qa-source")
        qa_videos_dir = OUTPUT_DATASET / "qa_videos"
        qa_videos_dir.mkdir(parents=True, exist_ok=True)

        # Detect demo boundaries: each demo starts where its first-primitive
        # label appears. Assumes episodes are stored in source order (the
        # labeling pipeline produces them this way) and the first primitive
        # only occurs once per demo (true for pickplace / drawer / scoop).
        first_label = episodes[0]["tasks"][0] if episodes else None
        running_demo = -1
        for ep in episodes:
            if ep["tasks"][0] == first_label:
                running_demo += 1
            demo_idx_for_ep_idx[ep["episode_index"]] = running_demo
        n_demos = running_demo + 1
        print(f"\nWill render {min(save_videos, n_demos)} demo videos "
              f"(of {n_demos} total demos) to {qa_videos_dir}")

    print(f"\nProcessing {len(episodes)} episodes (keeping all primitives)")
    qa_demo_frames: list = []   # accumulator for the current demo's frames
    qa_current_demo: int = -1   # which demo we're currently accumulating

    # Process each episode
    new_episodes = []
    new_tasks = {}  # task_name -> task_index
    total_frames_before = 0
    total_frames_after = 0
    frames_trimmed = 0

    # Track new episode indices and chunks
    new_episode_idx = 0
    global_frame_idx = 0  # For updating the 'index' column
    chunk_size = 1000

    for ep in tqdm.tqdm(episodes, desc="Trimming episodes"):
        old_ep_idx = ep["episode_index"]
        old_chunk = old_ep_idx // chunk_size
        task = ep["tasks"][0]

        # Drop trailing fallback segments: "move gripper to target" is the
        # densely-label fallback label assigned to frames AFTER the last
        # expected primitive (e.g. operator returning the arm home after
        # release). These are post-task motion, not part of any primitive,
        # and including them inflates the trained policy's action distribution
        # with non-task motion.
        if task == "move gripper to target":
            continue

        # Assign task index
        if task not in new_tasks:
            new_tasks[task] = len(new_tasks)

        # Load parquet file
        parquet_path = SOURCE_DATASET / f"data/chunk-{old_chunk:03d}/episode_{old_ep_idx:06d}.parquet"
        if not parquet_path.exists():
            print(f"Warning: {parquet_path} not found, skipping")
            continue

        df = pd.read_parquet(parquet_path)
        total_frames_before += len(df)

        # Get actions and compute trim indices
        actions = np.stack(df["actions"].values)
        trim_start, trim_end = compute_trim_indices(actions, task, xarm=xarm, delta_threshold=delta_threshold, end_delta_threshold=end_delta_threshold)
        trimmed_count = trim_start + (len(df) - trim_end)
        frames_trimmed += trimmed_count

        # Trim the dataframe
        if trimmed_count > 0:
            df = df.iloc[trim_start:trim_end].reset_index(drop=True)

        if len(df) == 0:
            continue

        total_frames_after += len(df)

        # Update frame indices and episode index in dataframe
        df["frame_index"] = range(len(df))
        df["episode_index"] = new_episode_idx
        df["index"] = range(global_frame_idx, global_frame_idx + len(df))
        global_frame_idx += len(df)

        # Update task_index
        df["task_index"] = new_tasks[task]

        # Recompute progress (0→1) after trimming if actions have 8 dims and flag is set
        if recompute_progress:
            actions_col = df["actions"].values
            if len(actions_col) > 0 and len(actions_col[0]) >= 8:
                n = len(df)
                for i in range(n):
                    a = list(actions_col[i])
                    a[7] = i / max(n - 1, 1)
                    actions_col[i] = np.array(a, dtype=np.float32)
                df["actions"] = list(actions_col)

        # Save to new location
        new_chunk = new_episode_idx // chunk_size
        new_chunk_dir = OUTPUT_DATASET / f"data/chunk-{new_chunk:03d}"
        new_chunk_dir.mkdir(exist_ok=True)

        new_parquet_path = new_chunk_dir / f"episode_{new_episode_idx:06d}.parquet"
        # Use pyarrow to preserve HuggingFace metadata (pandas loses it)
        table = pa.Table.from_pandas(df)
        table = table.replace_schema_metadata(hf_metadata)
        pq.write_table(table, new_parquet_path)

        # Accumulate trimmed frames for the current demo's QA video. When the
        # demo index changes (new demo's first primitive), flush the previous
        # demo's accumulated frames as a single video. Limited to the first
        # ``save_videos`` demos.
        if qa_dataset is not None:
            ep_demo_idx = demo_idx_for_ep_idx.get(old_ep_idx, -1)
            # Flush previous demo on demo-boundary transition
            if ep_demo_idx != qa_current_demo:
                if qa_demo_frames and qa_current_demo >= 0 and qa_current_demo < save_videos:
                    try:
                        import imageio.v2 as imageio
                        video_path = qa_videos_dir / f"demo{qa_current_demo:03d}.mp4"
                        imageio.mimwrite(video_path, qa_demo_frames,
                                         fps=video_fps, codec="libx264", macro_block_size=1)
                        print(f"  [QA video] wrote {video_path.name} ({len(qa_demo_frames)} frames)")
                    except Exception as e:
                        print(f"  [QA video] failed to write demo {qa_current_demo}: {e}")
                qa_demo_frames = []
                qa_current_demo = ep_demo_idx
            # Render this primitive's trimmed range and append (only if within budget)
            if qa_current_demo < save_videos:
                try:
                    import cv2
                    ep_global_start = qa_dataset.episode_data_index["from"][old_ep_idx].item()
                    for fi in range(trim_start, trim_end):
                        sample = qa_dataset[ep_global_start + fi]
                        img = sample[video_image_key]
                        if hasattr(img, "numpy"):
                            img = img.numpy()
                        if img.dtype != np.uint8:
                            img = (img * 255).clip(0, 255).astype(np.uint8)
                        if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[0] < img.shape[-1]:
                            img = img.transpose(1, 2, 0)  # CHW -> HWC
                        img = np.ascontiguousarray(img)
                        label = f"demo {qa_current_demo}  {task[:48]}  src_frame {fi}"
                        cv2.putText(img, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                        cv2.putText(img, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                        qa_demo_frames.append(img)
                except Exception as e:
                    print(f"  [QA video] frame extraction failed for {task!r}: {e}")

        # Record new episode metadata
        new_episodes.append({
            "episode_index": new_episode_idx,
            "tasks": [task],
            "length": len(df),
        })

        new_episode_idx += 1

    # Flush the last demo's QA video (loop ended without a demo-boundary).
    if qa_dataset is not None and qa_demo_frames and 0 <= qa_current_demo < save_videos:
        try:
            import imageio.v2 as imageio
            video_path = qa_videos_dir / f"demo{qa_current_demo:03d}.mp4"
            imageio.mimwrite(video_path, qa_demo_frames,
                             fps=video_fps, codec="libx264", macro_block_size=1)
            print(f"  [QA video] wrote {video_path.name} ({len(qa_demo_frames)} frames)")
        except Exception as e:
            print(f"  [QA video] failed to write final demo {qa_current_demo}: {e}")

    print(f"\nFrames: {total_frames_before} -> {total_frames_after} ({frames_trimmed} trimmed)")
    print(f"Episodes: {len(new_episodes)}")

    # Write new metadata files

    # tasks.jsonl
    with open(OUTPUT_DATASET / "meta/tasks.jsonl", "w") as f:
        for task, idx in sorted(new_tasks.items(), key=lambda x: x[1]):
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")

    # episodes.jsonl
    with open(OUTPUT_DATASET / "meta/episodes.jsonl", "w") as f:
        for ep in new_episodes:
            f.write(json.dumps(ep) + "\n")

    # info.json - copy and modify from source
    with open(SOURCE_DATASET / "meta/info.json") as f:
        info = json.load(f)

    info["total_episodes"] = len(new_episodes)
    info["total_frames"] = total_frames_after
    info["total_tasks"] = len(new_tasks)
    info["total_chunks"] = (len(new_episodes) - 1) // chunk_size + 1
    info["splits"] = {"train": f"0:{len(new_episodes)}"}

    with open(OUTPUT_DATASET / "meta/info.json", "w") as f:
        json.dump(info, f, indent=4)

    # Compute and write episodes_stats.jsonl
    print("\nComputing episode stats...")
    with open(OUTPUT_DATASET / "meta/episodes_stats.jsonl", "w") as f:
        for ep in tqdm.tqdm(new_episodes, desc="Computing stats"):
            ep_idx = ep["episode_index"]
            chunk = ep_idx // chunk_size
            parquet_path = OUTPUT_DATASET / f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet"
            df = pd.read_parquet(parquet_path)

            col_stats = {}

            # Compute stats for each column
            for col in df.columns:
                values = df[col].values

                if col in ["frame_index", "episode_index", "index", "task_index"]:
                    # Integer columns
                    col_stats[col] = {
                        "min": [int(values.min())],
                        "max": [int(values.max())],
                        "mean": [float(values.mean())],
                        "std": [float(values.std())],
                        "count": [len(values)],
                    }
                elif col == "timestamp":
                    col_stats[col] = {
                        "min": [float(values.min())],
                        "max": [float(values.max())],
                        "mean": [float(values.mean())],
                        "std": [float(values.std())],
                        "count": [len(values)],
                    }
                elif isinstance(values[0], (np.ndarray, list)):
                    # Array column - compute per-dimension stats
                    arr = np.stack(values)
                    col_stats[col] = {
                        "min": arr.min(axis=0).tolist(),
                        "max": arr.max(axis=0).tolist(),
                        "mean": arr.mean(axis=0).tolist(),
                        "std": arr.std(axis=0).tolist(),
                        "count": [len(values)],
                    }

            f.write(json.dumps({"episode_index": ep_idx, "stats": col_stats}) + "\n")

    print(f"\nDone! New dataset at: {OUTPUT_DATASET}")
    print(f"\nTask distribution:")
    task_counts = defaultdict(int)
    for ep in new_episodes:
        task_counts[ep["tasks"][0]] += 1
    for task, count in sorted(task_counts.items(), key=lambda x: -x[1]):
        print(f"  {count}: {task}")


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
