"""Filter + normalize the flywheel-collected twist dataset to a clean training set.

The flywheel dataset (``maggie/xarm_twist_open_v1``) was collected with
``--no-record-skill-gap-only``, so each successful trial commits the
move-to-bottle and the skill-gap. The skill-gap labels also drift across
trials because the planner's per-step note (and the VLM's pre-analysis
reasoning) phrases the same primitive 13+ different ways:

    twist cap 180 degrees counter-clockwise
    twist open the cap
    twist the cap counter-clockwise
    unscrew cap / unscrew the cap / unscrew bottle cap / ...
    rotate gripper 180 degrees counter-clockwise
    twist the cap to detach
    ...

Without normalization, training would see ~2 episodes per phrasing — useless.

This script produces a clean derivative dataset with a single canonical task
label, dropping the move-to-bottle episodes (pickplace v2 already supplies
that primitive at the same fps with teleop motion style, no need to mix in
the trained policy's variant) and any anomalies (e.g., a one-off
"move gripper to the black tray" episode that snuck in from a put-down step
in some trial).

The source dataset is NEVER modified — episodes we drop or rewrite still
exist in the source on disk for inspection.

Output mirrors the source's LeRobot v2.1 layout — re-indexes episodes
0..N-1, re-numbers task indices, recomputes the global index, rewrites
info.json totals and per-episode stats.
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Original v1 source (preserved for reproducibility):
# SOURCE_DATASET = Path.home() / ".cache/huggingface/lerobot/maggie/xarm_twist_open_v1"
# OUTPUT_DATASET = Path.home() / ".cache/huggingface/lerobot/maggie/xarm_twist_open_30_clean"

# Previous v4 source (preserved for reproducibility):
# SOURCE_DATASET = Path.home() / ".cache/huggingface/lerobot/maggie/xarm_twist_open_v4"
# OUTPUT_DATASET = Path.home() / ".cache/huggingface/lerobot/maggie/xarm_twist_open_v4_clean_unwrap"

# v5 source: flywheel-collected against the xarm_pick_from_top_v5 policy with
# --record-skill-gap-only (default), so no move-to-bottle episodes to drop,
# only label normalization needed.
SOURCE_DATASET = Path.home() / ".cache/huggingface/lerobot/maggie/xarm_twist_open_v5"
OUTPUT_DATASET = Path.home() / ".cache/huggingface/lerobot/maggie/xarm_twist_open_v5_clean_unwrap"

# All twist/unscrew phrasings the planner + VLM produce for the same primitive.
# Anything in this set rewrites to CANONICAL_TWIST. Keep this list explicit
# (rather than pattern-matching on substrings) so a stray label like "twist
# the bottle off the table" wouldn't silently get folded in.
TWIST_VARIANTS = {
    # v1 collection phrasings (kept for reproducibility on older datasets)
    "twist cap 180 degrees counter-clockwise",
    "twist open the cap",
    "twist the cap counter-clockwise",
    "unscrew cap",
    "unscrew the cap",
    "unscrew cap 180 degrees",
    "twist cap counter-clockwise",
    "twist the cap 180 degrees counter-clockwise",
    "twist the cap to detach",
    "rotate gripper 180 degrees counter-clockwise",
    "twist cap counter-clockwise 180 degrees",
    "twist cap 180 degrees",
    "unscrew bottle cap",
    # v4 collection phrasings observed in the current dataset
    "rotate cap counter-clockwise",
    "rotate cap counter-clockwise 180 degrees",
    "rotate gripper counter-clockwise",
    "rotate gripper counter-clockwise 180 degrees",
    "rotate the cap counter-clockwise",
    "unscrew the cap 180 degrees",
    "unscrew cap counter-clockwise",
    "twist cap counter-clockwise 180 degrees",
    # v5 collection phrasings (new variants beyond v4)
    "rotate cap 180 degrees counter-clockwise",
    "rotate gripper 180 degrees counter-clockwise",
    "rotate gripper CCW 180 degrees",
    "unscrew cap 180 degrees counter-clockwise",
}
CANONICAL_TWIST = "twist open the cap"

# Pickplace v2 has 8D actions (pose + gripper + progress). The flywheel
# recorder writes 6D pose only — see TODO in recording.py to fix going
# forward. For the twist data we already have, synthesize the two missing
# columns:
#   gripper: held closed throughout the twist primitive (closed before
#            the skill-gap fires, released after lift). Empirical value
#            from the run telemetry: actual_raw≈368 → (850-368)/860 ≈ 0.57
#            in the policy's normalized convention. Matches pickplace's
#            "lift upward" gripper-action distribution.
#   progress: frame_idx / (length-1), monotonic 0→1 per episode.
_GRIPPER_CLOSED_ACTION = 0.57

# Tasks present in the source we deliberately drop:
# - "move gripper to the yellow bottle": pickplace v2 already has this primitive
#   from teleop demos at matching fps. Mixing trained-policy motion in could bias
#   the merged policy.
# - "move gripper to the black tray": one-off anomaly from a put-down step that
#   committed in an earlier trial. Not part of the twist primitive.
DROP_TASKS = {
    "move gripper to the yellow bottle",
    "move gripper to the black tray",
}


def main() -> None:
    print(f"Source: {SOURCE_DATASET}")
    print(f"Output: {OUTPUT_DATASET}")

    # Validate source BEFORE wiping output — if the source is missing or
    # the user pointed at the wrong path, we abort without losing whatever
    # was previously at OUTPUT_DATASET.
    if not (SOURCE_DATASET / "meta/info.json").exists():
        raise FileNotFoundError(
            f"source dataset missing meta/info.json: {SOURCE_DATASET}"
        )

    src_episodes = []
    with open(SOURCE_DATASET / "meta/episodes.jsonl") as f:
        for line in f:
            src_episodes.append(json.loads(line))

    # Wipe OUTPUT cleanly so a re-run doesn't leave orphan parquets from a
    # previous (potentially differently-sized) run alongside the fresh ones.
    # OUTPUT is a derived dataset, never a source.
    if OUTPUT_DATASET.exists():
        shutil.rmtree(OUTPUT_DATASET)
    OUTPUT_DATASET.mkdir(parents=True)
    (OUTPUT_DATASET / "meta").mkdir()
    (OUTPUT_DATASET / "data").mkdir()
    print(f"Source episodes: {len(src_episodes)}")

    # Source-side label distribution (sanity check before filter+normalize).
    src_counts: dict[str, int] = defaultdict(int)
    for ep in src_episodes:
        src_counts[ep["tasks"][0]] += 1
    print("Source task distribution:")
    for name, n in sorted(src_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {n}: {name!r}")

    # Filter: drop everything that isn't a twist variant. Anomalies and
    # move-to episodes go away (still preserved in source on disk).
    keep_episodes = [ep for ep in src_episodes if ep["tasks"][0] in TWIST_VARIANTS]
    print(f"\nAfter filter (kept twist variants only): {len(keep_episodes)}")

    # Normalize: collapse all variants to CANONICAL_TWIST.
    new_tasks: dict[str, int] = {CANONICAL_TWIST: 0}
    new_episodes: list[dict] = []
    total_frames = 0
    chunk_size = 1000

    for new_idx, ep in enumerate(keep_episodes):
        old_ep_idx = ep["episode_index"]
        old_chunk = old_ep_idx // chunk_size
        # Always rewrite to canonical — every kept episode is a twist variant.
        task = CANONICAL_TWIST

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
        # matches pickplace v4 schema. Also unwrap rotation columns so a
        # 180° twist that crosses the 0/2π boundary reads as a continuous
        # numerical sequence (e.g. 6.28 → 6.30) instead of wrapping to 0.
        # Without the unwrap the regression loss has to fit a -2π
        # discontinuity at each wrap frame, which manifests as elevated
        # training loss with high per-batch variance. v1/v2 twist datasets
        # dodged this by luck (rotations didn't cross 2π); v4 surfaces it.
        n = len(df)
        progress = (np.arange(n, dtype=np.float32) / max(n - 1, 1)).astype(np.float32)
        old_actions = np.stack(df["actions"].values).astype(np.float32)  # (n, 7)
        for axis in (3, 4, 5):
            old_actions[:, axis] = np.unwrap(old_actions[:, axis]).astype(np.float32)
        if "state" in df.columns:
            states = np.stack(df["state"].values).astype(np.float32)
            for axis in (3, 4, 5):
                states[:, axis] = np.unwrap(states[:, axis]).astype(np.float32)
            df["state"] = list(states)
        progress_col = progress.reshape(-1, 1)
        new_actions = np.concatenate([old_actions, progress_col], axis=1)  # (n, 8)
        df["actions"] = list(new_actions)

        new_chunk = new_idx // chunk_size
        out_dir = OUTPUT_DATASET / f"data/chunk-{new_chunk:03d}"
        out_dir.mkdir(exist_ok=True)
        df.to_parquet(out_dir / f"episode_{new_idx:06d}.parquet")

        new_episodes.append({
            "episode_index": new_idx,
            "tasks": [task],
            "length": len(df),
        })
        total_frames += len(df)

    # Recompute the global ``index`` field (cumulative frame index across episodes).
    print("Re-stamping global index field...")
    running = 0
    for ep in new_episodes:
        ep_idx = ep["episode_index"]
        chunk = ep_idx // chunk_size
        path = OUTPUT_DATASET / f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet"
        df = pd.read_parquet(path)
        df["index"] = range(running, running + len(df))
        df.to_parquet(path)
        running += len(df)

    with open(OUTPUT_DATASET / "meta/tasks.jsonl", "w") as f:
        for task, idx in sorted(new_tasks.items(), key=lambda kv: kv[1]):
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")

    with open(OUTPUT_DATASET / "meta/episodes.jsonl", "w") as f:
        for ep in new_episodes:
            f.write(json.dumps(ep) + "\n")

    with open(SOURCE_DATASET / "meta/info.json") as f:
        info = json.load(f)
    # Update the actions-feature shape to reflect the 8D augmentation
    # (was 6D in the source). Without this the merge sanity check would
    # see an apparent 6D source even though the parquet rows are 8D.
    if "features" in info and "actions" in info["features"]:
        info["features"] = dict(info["features"])
        info["features"]["actions"] = dict(info["features"]["actions"])
        info["features"]["actions"]["shape"] = [8]
    info["total_episodes"] = len(new_episodes)
    info["total_frames"] = total_frames
    info["total_tasks"] = len(new_tasks)
    info["total_chunks"] = ((len(new_episodes) - 1) // chunk_size) + 1 if new_episodes else 0
    info["splits"] = {"train": f"0:{len(new_episodes)}"}
    with open(OUTPUT_DATASET / "meta/info.json", "w") as f:
        json.dump(info, f, indent=4)

    print("Computing episode stats...")
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

    print()
    print(f"Done. {len(new_episodes)} episodes, {total_frames} frames -> {OUTPUT_DATASET}")
    print()
    counts: dict[str, int] = defaultdict(int)
    for ep in new_episodes:
        counts[ep["tasks"][0]] += 1
    for name, n in counts.items():
        print(f"  {n}: {name!r}")


if __name__ == "__main__":
    main()
