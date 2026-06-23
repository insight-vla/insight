#!/usr/bin/env python3
"""Preprocess oracle flip dataset to trim leading/trailing idle frames.

Uses all 6 DOF (position + rotation) for the threshold check, not just position,
since flip demos have rotation-only phases that should be preserved.
"""

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
SOURCE_DATASET = Path.home() / ".cache/huggingface/lerobot/maggiewang/lego_pickplace_tilted_100_03_31"
OUTPUT_DATASET = Path.home() / ".cache/huggingface/lerobot/maggiewang/lego_pickplace_tilted_100_03_31_trimmed"

# Threshold for detecting idle frames (all 6 DOF magnitude below this = idle)
SETTLING_THRESHOLD = 0.01
CHUNK_SIZE = 1000


def compute_trim_indices(actions: np.ndarray) -> tuple[int, int]:
    """Compute trim start and end indices based on all 6 DOF action magnitudes."""
    trim_start = 0
    trim_end = len(actions)

    # Trim from start: find first frame where any position OR rotation exceeds threshold
    for i in range(len(actions)):
        magnitude = np.linalg.norm(actions[i, :6])  # All 6 DOF (pos + rot)
        if magnitude > SETTLING_THRESHOLD:
            trim_start = i
            break

    # Trim from end: find last frame where any position OR rotation exceeds threshold
    for i in range(len(actions) - 1, -1, -1):
        magnitude = np.linalg.norm(actions[i, :6])
        if magnitude > SETTLING_THRESHOLD:
            trim_end = i + 1
            break

    return trim_start, trim_end


def main():
    print(f"Source: {SOURCE_DATASET}")
    print(f"Output: {OUTPUT_DATASET}")

    if OUTPUT_DATASET.exists():
        print(f"Removing existing dataset at {OUTPUT_DATASET}")
        shutil.rmtree(OUTPUT_DATASET)
    OUTPUT_DATASET.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DATASET / "meta").mkdir(exist_ok=True)
    (OUTPUT_DATASET / "data").mkdir(exist_ok=True)

    # Get HuggingFace metadata from source parquet
    sample_parquet = list((SOURCE_DATASET / "data").glob("**/episode_*.parquet"))[0]
    hf_metadata = pq.read_table(sample_parquet).schema.metadata

    # Load episodes metadata
    episodes = []
    with open(SOURCE_DATASET / "meta/episodes.jsonl") as f:
        for line in f:
            episodes.append(json.loads(line))

    print(f"\nProcessing {len(episodes)} episodes")

    new_episodes = []
    new_tasks = {}
    total_frames_before = 0
    total_frames_after = 0
    frames_trimmed_start = 0
    frames_trimmed_end = 0
    new_episode_idx = 0
    global_frame_idx = 0

    for ep in tqdm.tqdm(episodes, desc="Trimming episodes"):
        old_ep_idx = ep["episode_index"]
        old_chunk = old_ep_idx // CHUNK_SIZE
        task = ep["tasks"][0]

        if task not in new_tasks:
            new_tasks[task] = len(new_tasks)

        parquet_path = SOURCE_DATASET / f"data/chunk-{old_chunk:03d}/episode_{old_ep_idx:06d}.parquet"
        if not parquet_path.exists():
            print(f"Warning: {parquet_path} not found, skipping")
            continue

        df = pd.read_parquet(parquet_path)
        total_frames_before += len(df)

        actions = np.stack(df["actions"].values)
        trim_start, trim_end = compute_trim_indices(actions)
        trimmed_start = trim_start
        trimmed_end = len(df) - trim_end
        frames_trimmed_start += trimmed_start
        frames_trimmed_end += trimmed_end

        if trimmed_start > 0 or trimmed_end > 0:
            df = df.iloc[trim_start:trim_end].reset_index(drop=True)

        if len(df) == 0:
            continue

        total_frames_after += len(df)

        # Update indices
        df["frame_index"] = range(len(df))
        df["episode_index"] = new_episode_idx
        df["index"] = range(global_frame_idx, global_frame_idx + len(df))
        df["task_index"] = new_tasks[task]
        global_frame_idx += len(df)

        # Save
        new_chunk = new_episode_idx // CHUNK_SIZE
        new_chunk_dir = OUTPUT_DATASET / f"data/chunk-{new_chunk:03d}"
        new_chunk_dir.mkdir(exist_ok=True)

        table = pa.Table.from_pandas(df)
        table = table.replace_schema_metadata(hf_metadata)
        pq.write_table(table, new_chunk_dir / f"episode_{new_episode_idx:06d}.parquet")

        new_episodes.append({
            "episode_index": new_episode_idx,
            "tasks": [task],
            "length": len(df),
        })
        new_episode_idx += 1

    print(f"\nFrames: {total_frames_before} -> {total_frames_after}")
    print(f"  Trimmed from start: {frames_trimmed_start} ({frames_trimmed_start/total_frames_before*100:.1f}%)")
    print(f"  Trimmed from end:   {frames_trimmed_end} ({frames_trimmed_end/total_frames_before*100:.1f}%)")
    print(f"  Total trimmed:      {frames_trimmed_start + frames_trimmed_end} ({(frames_trimmed_start + frames_trimmed_end)/total_frames_before*100:.1f}%)")

    # Write metadata
    with open(OUTPUT_DATASET / "meta/tasks.jsonl", "w") as f:
        for task, idx in sorted(new_tasks.items(), key=lambda x: x[1]):
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")

    with open(OUTPUT_DATASET / "meta/episodes.jsonl", "w") as f:
        for ep in new_episodes:
            f.write(json.dumps(ep) + "\n")

    with open(SOURCE_DATASET / "meta/info.json") as f:
        info = json.load(f)

    info["total_episodes"] = len(new_episodes)
    info["total_frames"] = total_frames_after
    info["total_tasks"] = len(new_tasks)
    info["total_chunks"] = (len(new_episodes) - 1) // CHUNK_SIZE + 1
    info["splits"] = {"train": f"0:{len(new_episodes)}"}

    with open(OUTPUT_DATASET / "meta/info.json", "w") as f:
        json.dump(info, f, indent=4)

    # Compute episode stats
    print("\nComputing episode stats...")
    with open(OUTPUT_DATASET / "meta/episodes_stats.jsonl", "w") as f:
        for ep in tqdm.tqdm(new_episodes, desc="Computing stats"):
            ep_idx = ep["episode_index"]
            chunk = ep_idx // CHUNK_SIZE
            parquet_path = OUTPUT_DATASET / f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet"
            df = pd.read_parquet(parquet_path)

            col_stats = {}
            for col in df.columns:
                values = df[col].values
                if col in ["frame_index", "episode_index", "index", "task_index"]:
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
    print(f"Episodes: {len(new_episodes)}")

    # Show per-episode trim stats
    print(f"\nEpisode length distribution:")
    lengths = [ep["length"] for ep in new_episodes]
    print(f"  Mean: {np.mean(lengths):.0f}, Median: {np.median(lengths):.0f}")
    print(f"  Min: {np.min(lengths)}, Max: {np.max(lengths)}")


if __name__ == "__main__":
    main()
