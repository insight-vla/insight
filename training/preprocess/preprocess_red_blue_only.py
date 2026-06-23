#!/usr/bin/env python3
"""Preprocess lego dataset to only include 'move to red/blue block' primitives,
and trim settling zeros from the start of each episode."""

import json
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# Config
SOURCE_DATASET = Path.home() / ".cache/huggingface/lerobot/maggiewang/lego_pickplace_vlm_primitives_01_26"
OUTPUT_DATASET = Path.home() / ".cache/huggingface/lerobot/maggiewang/lego_red_blue_only"

# Only keep these primitives
KEEP_TASKS = [
    "move gripper to the red lego block",
    "move gripper to the blue lego block",
]

# Threshold for detecting "settling" frames (position action magnitude)
SETTLING_THRESHOLD = 0.01  # Actions with position magnitude below this are trimmed


def main():
    print(f"Source: {SOURCE_DATASET}")
    print(f"Output: {OUTPUT_DATASET}")

    # Create output directory
    OUTPUT_DATASET.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DATASET / "meta").mkdir(exist_ok=True)
    (OUTPUT_DATASET / "data").mkdir(exist_ok=True)

    # Load episodes metadata
    episodes = []
    with open(SOURCE_DATASET / "meta/episodes.jsonl") as f:
        for line in f:
            episodes.append(json.loads(line))

    # Filter to only red/blue episodes
    keep_episodes = []
    for ep in episodes:
        task = ep["tasks"][0]
        if task in KEEP_TASKS:
            keep_episodes.append(ep)

    print(f"\nFiltering: {len(episodes)} -> {len(keep_episodes)} episodes")

    # Process each episode
    new_episodes = []
    new_tasks = {}  # task_name -> task_index
    total_frames_before = 0
    total_frames_after = 0
    frames_trimmed = 0

    # Track new episode indices and chunks
    new_episode_idx = 0
    chunk_size = 1000

    for ep in keep_episodes:
        old_ep_idx = ep["episode_index"]
        old_chunk = old_ep_idx // chunk_size
        task = ep["tasks"][0]

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

        # Find first non-settling frame (trim start)
        # Action format: [x, y, z, rx, ry, rz, gripper]
        # Check position action magnitude (first 3 components)
        trim_start = 0
        trim_end = len(df)
        if "actions" in df.columns:
            actions = np.stack(df["actions"].values)

            # Trim from start
            for i in range(len(actions)):
                pos_magnitude = np.linalg.norm(actions[i, :3])
                if pos_magnitude > SETTLING_THRESHOLD:
                    trim_start = i
                    break

            # Trim from end
            for i in range(len(actions)-1, -1, -1):
                pos_magnitude = np.linalg.norm(actions[i, :3])
                if pos_magnitude > SETTLING_THRESHOLD:
                    trim_end = i + 1
                    break

        # Trim the dataframe
        trimmed_count = trim_start + (len(df) - trim_end)
        if trimmed_count > 0:
            df = df.iloc[trim_start:trim_end].reset_index(drop=True)
            frames_trimmed += trimmed_count

        if len(df) == 0:
            print(f"Warning: Episode {old_ep_idx} has no frames after trimming, skipping")
            continue

        total_frames_after += len(df)

        # Update frame indices and episode index in dataframe
        df["frame_index"] = range(len(df))
        df["episode_index"] = new_episode_idx

        # Update task_index
        df["task_index"] = new_tasks[task]

        # Save to new location
        new_chunk = new_episode_idx // chunk_size
        new_chunk_dir = OUTPUT_DATASET / f"data/chunk-{new_chunk:03d}"
        new_chunk_dir.mkdir(exist_ok=True)

        new_parquet_path = new_chunk_dir / f"episode_{new_episode_idx:06d}.parquet"
        df.to_parquet(new_parquet_path)

        # Record new episode metadata
        new_episodes.append({
            "episode_index": new_episode_idx,
            "tasks": [task],
            "length": len(df),
        })

        new_episode_idx += 1

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
        for ep in new_episodes:
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
    for task, count in task_counts.items():
        print(f"  {count}: {task}")


if __name__ == "__main__":
    main()
