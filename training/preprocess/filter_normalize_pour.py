"""Filter + normalize the flywheel-collected pour dataset to a clean training set.

The flywheel pour dataset (``maggie/xarm_pour_05_10``) was collected with
``--record-skill-gap-only`` (default), so each successful trial commits the
TWO skill-gap episodes (forward-tilt to pour + back-to-upright). The VLM
planner phrases each of these in many different ways across trials:

    Forward (pour) variants:
        "tilt bottle to pour"
        "pour bottle contents"
        "empty bottle contents"
        "tilt bottle past horizontal"
        ...

    Return (back-upright) variants:
        "tilt bottle upright"
        "tilt bottle back upright"
        "return bottle to upright position"
        "orient bottle upright"
        ...

Without normalization, training would see ~1-2 episodes per phrasing — useless.

This script produces a clean derivative dataset with TWO canonical task labels:
    "tilt bottle forward to pour" — for all forward/pour-out variants
    "tilt bottle back upright"   — for all return/upright variants

Each variant is explicitly bucketed (no substring matching) so a stray label
like "tilt bottle off the table" wouldn't silently get folded in. An unknown
label raises an error — add it to the appropriate set after verifying its
intent.

The source dataset is NEVER modified. Output mirrors the source's LeRobot v2.1
layout — re-indexes episodes 0..N-1, re-numbers task indices, recomputes the
global index, rewrites info.json totals and per-episode stats. The 7D->8D
action padding from the flywheel recorder is also applied (matches pickplace
v4 schema). Pour data does NOT need rpy unwrap (verified: ry stays in
[-π/2, π/2] without crossing wrap boundaries).
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_DATASET = Path.home() / ".cache/huggingface/lerobot/maggie/xarm_pour_05_10"
OUTPUT_DATASET = Path.home() / ".cache/huggingface/lerobot/maggie/xarm_pour_05_10_clean"

# Forward-pour variants the VLM produces. Anything in this set rewrites to
# CANONICAL_POUR. Bucketing is explicit (no substring match) so unrelated
# labels don't silently get folded in.
POUR_FORWARD_VARIANTS = {
    "tilt bottle to pour",
    "tilt bottle forward to pour",
    "tilt bottle past horizontal",
    "tilt bottle beyond horizontal",
    "tilt bottle downwards to pour",
    "tilt bottle to empty",
    "tilt bottle to empty contents",
    "tilt bottle to pour contents",
    "pour bottle contents",
    "empty bottle contents",
    "empty bottle contents into bowl",
}

# Return-upright variants the VLM produces. Anything in this set rewrites
# to CANONICAL_RETURN.
POUR_RETURN_VARIANTS = {
    "tilt bottle upright",
    "tilt bottle back upright",
    "tilt bottle back to upright",
    "tilt bottle back to upright position",
    "tilt bottle to upright",
    "return bottle to upright position",
    "return bottle to vertical",
    "orient bottle upright",
}

CANONICAL_POUR = "tilt bottle forward to pour"
CANONICAL_RETURN = "tilt bottle back upright"


def main() -> None:
    print(f"Source: {SOURCE_DATASET}")
    print(f"Output: {OUTPUT_DATASET}")

    if not (SOURCE_DATASET / "meta/info.json").exists():
        raise FileNotFoundError(
            f"source dataset missing meta/info.json: {SOURCE_DATASET}"
        )

    src_episodes = []
    with open(SOURCE_DATASET / "meta/episodes.jsonl") as f:
        for line in f:
            src_episodes.append(json.loads(line))

    if OUTPUT_DATASET.exists():
        shutil.rmtree(OUTPUT_DATASET)
    OUTPUT_DATASET.mkdir(parents=True)
    (OUTPUT_DATASET / "meta").mkdir()
    (OUTPUT_DATASET / "data").mkdir()
    print(f"Source episodes: {len(src_episodes)}")

    src_counts: dict[str, int] = defaultdict(int)
    for ep in src_episodes:
        src_counts[ep["tasks"][0]] += 1
    print("Source task distribution:")
    for name, n in sorted(src_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {n}: {name!r}")

    # Bucket each episode and complain about unknowns.
    known_labels = POUR_FORWARD_VARIANTS | POUR_RETURN_VARIANTS
    unknown = sorted({ep["tasks"][0] for ep in src_episodes} - known_labels)
    if unknown:
        raise ValueError(
            f"Unknown task labels — add to POUR_FORWARD_VARIANTS or POUR_RETURN_VARIANTS "
            f"after verifying intent: {unknown}"
        )

    new_tasks: dict[str, int] = {CANONICAL_POUR: 0, CANONICAL_RETURN: 1}
    new_episodes: list[dict] = []
    total_frames = 0
    chunk_size = 1000

    for new_idx, ep in enumerate(src_episodes):
        old_ep_idx = ep["episode_index"]
        old_chunk = old_ep_idx // chunk_size
        src_task = ep["tasks"][0]
        if src_task in POUR_FORWARD_VARIANTS:
            task = CANONICAL_POUR
        else:
            task = CANONICAL_RETURN

        src_parquet = (
            SOURCE_DATASET / f"data/chunk-{old_chunk:03d}/episode_{old_ep_idx:06d}.parquet"
        )
        if not src_parquet.exists():
            print(f"  [WARN] missing parquet: {src_parquet}, skipping")
            continue

        df = pd.read_parquet(src_parquet).copy()
        df["frame_index"] = range(len(df))
        df["episode_index"] = new_idx
        df["task_index"] = new_tasks[task]

        # Augment 7D actions to 8D (append progress) so the merged dataset
        # matches pickplace v4 schema. The flywheel recorder writes
        # [x,y,z,rx,ry,rz,gripper] (7D); we synthesize a linear progress
        # ramp for the missing 8th column. NO rpy unwrap needed — pour
        # rotations live in ry and stay within [-π/2, π/2] without crossing
        # any wrap boundary (verified).
        n = len(df)
        progress = (np.arange(n, dtype=np.float32) / max(n - 1, 1)).astype(np.float32)
        old_actions = np.stack(df["actions"].values).astype(np.float32)  # (n, 7)
        progress_col = progress.reshape(-1, 1)
        new_actions = np.concatenate([old_actions, progress_col], axis=1)  # (n, 8)
        df["actions"] = list(new_actions)

        # Rewrite the per-frame "index" column so the merged global index
        # range is contiguous. LeRobot uses this as a flat frame index.
        # Start each episode at total_frames-so-far; recompute at write time.
        df["index"] = range(total_frames, total_frames + n)
        total_frames += n

        new_chunk = new_idx // chunk_size
        out_dir = OUTPUT_DATASET / f"data/chunk-{new_chunk:03d}"
        out_dir.mkdir(exist_ok=True)
        df.to_parquet(out_dir / f"episode_{new_idx:06d}.parquet")

        new_episodes.append(
            {"episode_index": new_idx, "tasks": [task], "length": n}
        )

    # tasks.jsonl
    with open(OUTPUT_DATASET / "meta/tasks.jsonl", "w") as f:
        for task, idx in sorted(new_tasks.items(), key=lambda kv: kv[1]):
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")

    # episodes.jsonl
    with open(OUTPUT_DATASET / "meta/episodes.jsonl", "w") as f:
        for ep in new_episodes:
            f.write(json.dumps(ep) + "\n")

    # info.json — base it on the source's, override totals + action shape.
    with open(SOURCE_DATASET / "meta/info.json") as f:
        info = json.load(f)
    info["total_episodes"] = len(new_episodes)
    info["total_frames"] = total_frames
    info["total_tasks"] = len(new_tasks)
    info["total_chunks"] = ((len(new_episodes) - 1) // chunk_size) + 1 if new_episodes else 0
    info["splits"] = {"train": f"0:{len(new_episodes)}"}
    # 7D→8D padding so the merged dataset's actions feature is 8D end-to-end.
    if "features" in info and "actions" in info["features"]:
        info["features"]["actions"] = dict(info["features"]["actions"])
        info["features"]["actions"]["shape"] = [8]
    with open(OUTPUT_DATASET / "meta/info.json", "w") as f:
        json.dump(info, f, indent=4)

    # episodes_stats.jsonl — recompute per-episode stats from the rewritten
    # parquets. Mirrors merge_pickplace_twist's stats schema.
    print("\nComputing episode stats...")
    with open(OUTPUT_DATASET / "meta/episodes_stats.jsonl", "w") as f:
        for ep in new_episodes:
            ep_idx = ep["episode_index"]
            chunk = ep_idx // chunk_size
            df = pd.read_parquet(
                OUTPUT_DATASET / f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet"
            )
            col_stats = {}
            for col in df.columns:
                values = df[col].values
                if col in ("frame_index", "episode_index", "index", "task_index"):
                    col_stats[col] = {
                        "min": [int(values.min())], "max": [int(values.max())],
                        "mean": [float(values.mean())], "std": [float(values.std())],
                        "count": [len(values)],
                    }
                elif col == "timestamp":
                    col_stats[col] = {
                        "min": [float(values.min())], "max": [float(values.max())],
                        "mean": [float(values.mean())], "std": [float(values.std())],
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

    print(f"\nDone. {len(new_episodes)} episodes, {total_frames} frames -> {OUTPUT_DATASET}")
    counts: dict[str, int] = defaultdict(int)
    for ep in new_episodes:
        counts[ep["tasks"][0]] += 1
    print("\nTask distribution:")
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {n}: {name!r}")


if __name__ == "__main__":
    main()
