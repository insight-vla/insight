"""Merge side-grasp pickplace + flywheel-bootstrapped pour — pickplace->pour direction.

Validates whether a policy that already knows the side-grasp pickplace primitives
can absorb the NEW pour primitives (forward-tilt + back-upright) bootstrapped
via the flywheel without losing the originals.

────────────────────────────────────────────────────────────────────────
WATCH OUT — action dimension consistency between source datasets:
Both sources must be 8D (pose + gripper + progress). Pickplace v5 is 8D from
teleop; flywheel pour gets padded 7D→8D by filter_normalize_pour.py. Verified
at script start.
────────────────────────────────────────────────────────────────────────

Combines:
  A = maggie/xarm_pick_from_side_v5_primitives_trimmed (side-grasp pickplace
                                                          primitives @ 20fps,
                                                          5 task labels)
  B = maggie/xarm_pour_05_10_clean                     (filtered+canonicalized
                                                          flywheel pour data,
                                                          2 task labels: forward
                                                          + return upright)

Output: maggie/xarm_pickplace_pour_05_10
  - All A episodes preserved
  - All B episodes appended
  - Unified task table: 7 tasks (5 pickplace + 2 pour), task_index re-stamped
    per-frame to point at the unified tasks.jsonl.
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

DATASET_A = Path.home() / ".cache/huggingface/lerobot/maggie/xarm_pick_from_side_v5_primitives_trimmed"
DATASET_B = Path.home() / ".cache/huggingface/lerobot/maggie/xarm_pour_05_10_clean"
OUTPUT = Path.home() / ".cache/huggingface/lerobot/maggie/xarm_pickplace_pour_05_10"

CHUNK_SIZE = 1000


def load_meta(path: Path):
    with open(path / "meta/info.json") as f:
        info = json.load(f)
    with open(path / "meta/episodes.jsonl") as f:
        episodes = [json.loads(line) for line in f]
    tasks: dict[str, int] = {}
    with open(path / "meta/tasks.jsonl") as f:
        for line in f:
            entry = json.loads(line)
            tasks[entry["task"]] = entry["task_index"]
    return info, episodes, tasks


def main() -> None:
    print(f"A: {DATASET_A}")
    print(f"B: {DATASET_B}")
    print(f"Output: {OUTPUT}")

    # Validate sources + run sanity checks BEFORE wiping OUTPUT — if either
    # source is missing or the schemas don't match, we abort without losing
    # whatever was previously at OUTPUT.
    for src in (DATASET_A, DATASET_B):
        if not (src / "meta" / "info.json").exists():
            raise FileNotFoundError(
                f"source dataset missing meta/info.json: {src}. "
                f"Run filter_normalize_twist.py first if B is missing."
            )

    info_a, episodes_a, tasks_a = load_meta(DATASET_A)
    info_b, episodes_b, tasks_b = load_meta(DATASET_B)
    print(f"  A: {len(episodes_a)} episodes, tasks={list(tasks_a)}")
    print(f"  B: {len(episodes_b)} episodes, tasks={list(tasks_b)}")

    # Sanity-check action dim and fps consistency. If these don't match, the
    # merge would silently produce an inconsistent dataset.
    a_actions_shape = info_a["features"]["actions"]["shape"]
    b_actions_shape = info_b["features"]["actions"]["shape"]
    if a_actions_shape != b_actions_shape:
        raise ValueError(
            f"action shape mismatch: A={a_actions_shape}, B={b_actions_shape}."
        )
    if info_a["fps"] != info_b["fps"]:
        raise ValueError(
            f"fps mismatch: A={info_a['fps']}, B={info_b['fps']}."
        )
    merged_fps = info_a["fps"]
    print(f"  Merged fps: {merged_fps}")

    # All checks passed — now safe to wipe and rebuild OUTPUT.
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "meta").mkdir()
    (OUTPUT / "data").mkdir()

    # Use A's parquet schema metadata for output parquets.
    sample_parquet = next((DATASET_A / "data").glob("**/episode_*.parquet"))
    hf_metadata = pq.read_table(sample_parquet).schema.metadata

    # Unified task table: A's tasks first (preserve indices), then any new
    # tasks from B appended. The merged tasks.jsonl + per-frame task_index
    # stamping ensures each sample's task_index matches the unified table.
    merged_tasks: dict[str, int] = dict(tasks_a)
    for t in tasks_b:
        if t not in merged_tasks:
            merged_tasks[t] = len(merged_tasks)
            print(f"  New task from B: {t!r} -> task_index {merged_tasks[t]}")

    new_episodes: list[dict] = []
    total_frames = 0
    global_idx = 0

    def copy_episodes(src: Path, episodes: list[dict], label: str) -> None:
        """Copy episodes from ``src`` into the merged output.

        Re-stamps ``frame_index`` / ``episode_index`` / ``index`` / ``task_index``
        on each frame so the merged dataset's per-frame task_index matches the
        unified ``merged_tasks`` table (rather than each source's local index).
        """
        nonlocal total_frames, global_idx
        print(f"\nCopying {len(episodes)} episodes from {label}...")
        for ep in tqdm.tqdm(episodes, desc=f"  {label}"):
            old_idx = ep["episode_index"]
            task = ep["tasks"][0]
            new_idx = len(new_episodes)

            parquet_path = (
                src / f"data/chunk-{old_idx // CHUNK_SIZE:03d}/episode_{old_idx:06d}.parquet"
            )
            df = pd.read_parquet(parquet_path)
            total_frames += len(df)

            df["frame_index"] = range(len(df))
            df["episode_index"] = new_idx
            df["index"] = range(global_idx, global_idx + len(df))
            df["task_index"] = merged_tasks[task]
            global_idx += len(df)

            chunk_dir = OUTPUT / f"data/chunk-{new_idx // CHUNK_SIZE:03d}"
            chunk_dir.mkdir(exist_ok=True)
            table = pa.Table.from_pandas(df)
            table = table.replace_schema_metadata(hf_metadata)
            pq.write_table(table, chunk_dir / f"episode_{new_idx:06d}.parquet")

            new_episodes.append(
                {"episode_index": new_idx, "tasks": [task], "length": len(df)}
            )

    copy_episodes(DATASET_A, episodes_a, "A (pickplace v5 side-grasp)")
    copy_episodes(DATASET_B, episodes_b, "B (pour clean)")

    # tasks.jsonl
    with open(OUTPUT / "meta/tasks.jsonl", "w") as f:
        for task, idx in sorted(merged_tasks.items(), key=lambda kv: kv[1]):
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")

    # episodes.jsonl
    with open(OUTPUT / "meta/episodes.jsonl", "w") as f:
        for ep in new_episodes:
            f.write(json.dumps(ep) + "\n")

    # info.json — start from A's, override totals.
    info = dict(info_a)
    info["fps"] = merged_fps
    info["total_episodes"] = len(new_episodes)
    info["total_frames"] = total_frames
    info["total_tasks"] = len(merged_tasks)
    info["total_chunks"] = ((len(new_episodes) - 1) // CHUNK_SIZE) + 1 if new_episodes else 0
    info["splits"] = {"train": f"0:{len(new_episodes)}"}
    with open(OUTPUT / "meta/info.json", "w") as f:
        json.dump(info, f, indent=4)

    # episodes_stats.jsonl — recompute per-episode stats.
    print("\nComputing episode stats...")
    with open(OUTPUT / "meta/episodes_stats.jsonl", "w") as f:
        for ep in new_episodes:
            ep_idx = ep["episode_index"]
            chunk = ep_idx // CHUNK_SIZE
            df = pd.read_parquet(
                OUTPUT / f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet"
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

    print(f"\nDone. {len(new_episodes)} episodes, {total_frames} frames -> {OUTPUT}")
    counts: dict[str, int] = defaultdict(int)
    for ep in new_episodes:
        counts[ep["tasks"][0]] += 1
    print("\nTask distribution:")
    for name, n in counts.items():
        print(f"  {n}: {name!r}")


if __name__ == "__main__":
    main()
