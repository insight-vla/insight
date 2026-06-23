"""Merge top-grasp pickplace + twist + side-grasp pickplace + pour into one
unified-skill dataset. Sources collected 2026-05-10; current output is
xarm_unified_skills_05_11 with relabeled move-to-bottle (cap/body) — see
label-disambiguation block below. Bump the OUTPUT date suffix when
re-running with new labels or sources so prior datasets are preserved.

Validates whether a single policy can absorb BOTH grasp-type configurations
(top-grasp for twist, side-grasp for pour) plus the two flywheel-acquired
primitives (twist, pour) on top of the respective pickplace bases. The
flywheel paper's headline claim: skill acquisition is additive AND the
policy learns to dispatch between grasp types based on task context.

────────────────────────────────────────────────────────────────────────
Label disambiguation:
The same primitive name "move gripper to the yellow bottle" exists in
both top and side pickplace datasets, but the physical motions differ
(approach-from-above targeting the cap vs approach-from-side targeting
the bottle body). Without renaming, the policy would see identical
conditioning input (language + bottle image) with bimodal action targets
and couldn't disambiguate. We rename per source so the labels differ at
TWO positions — preposition AND head noun:
    top  → "move gripper above the yellow bottle cap"
    side → "move gripper to the side of the yellow bottle body"
Earlier experiments with "from the top"/"from the side" suffixes failed
because the labels differed only in 2-3 trailing tokens after an
identical shared prefix, producing near-identical pooled language
embeddings. The new labels diverge at the preposition (token 3:
"above" vs "to") AND at the head noun ("cap" vs "body"). The
prepositions carry real motor semantics — "above" describes a vertical
end-pose; "to the side of" describes a lateral end-pose — so the policy
inherits PaliGemma's pretrained spatial-preposition grounding. The
head-noun split adds object-level disambiguation (cap on top of bottle,
body in the middle).
The planner now picks the right one based on task context (twist needs
the cap; pour needs the body).

Other primitives (close gripper, lift upward, lower gripper, open gripper)
share labels across both grasp types — actions are mostly orientation-
trivial (jaw actuation or +Z translation) and the policy can dispatch
based on state input.
────────────────────────────────────────────────────────────────────────

Combines:
  A = maggie/xarm_pick_from_top_v5_primitives_trimmed       (top-grasp pickplace)
  B = maggie/xarm_twist_open_v5_clean_unwrap                (flywheel twist)
  C = maggie/xarm_pick_from_side_v5_primitives_trimmed      (side-grasp pickplace)
  D = maggie/xarm_pour_05_10_clean                          (flywheel pour)

Output: maggie/xarm_unified_skills_05_12
  - All A episodes (with move-to-bottle renamed → "above ... bottle cap")
  - All B episodes
  - All C episodes (with move-to-bottle renamed → "to the side of ... bottle body")
  - All D episodes
  - Unified task table: 10 task labels.
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

DATASET_A = Path.home() / ".cache/huggingface/lerobot/maggie/xarm_pick_from_top_v5_primitives_trimmed"
DATASET_B = Path.home() / ".cache/huggingface/lerobot/maggie/xarm_twist_open_v5_clean_unwrap"
DATASET_C = Path.home() / ".cache/huggingface/lerobot/maggie/xarm_pick_from_side_v5_primitives_trimmed"
DATASET_D = Path.home() / ".cache/huggingface/lerobot/maggie/xarm_pour_05_10_clean"
OUTPUT = Path.home() / ".cache/huggingface/lerobot/maggie/xarm_unified_skills_05_12"

CHUNK_SIZE = 1000

# Per-source label renaming. The move-to-bottle primitive has different
# physical motions in top vs side grasp; disambiguate by appending the
# grasp direction. Other primitives keep their labels (shared across
# grasp types since they're orientation-invariant or state-dispatched).
RENAMES_A: dict[str, str] = {
    # v5 top dataset uses the more descriptive "...to the top of..." phrasing,
    # but we keep the 05_11 unified label for cross-version policy comparison.
    "move gripper to the top of the yellow bottle": "move gripper above the yellow bottle cap",
}
RENAMES_B: dict[str, str] = {}  # twist labels are unique already
RENAMES_C: dict[str, str] = {
    "move gripper to the yellow bottle": "move gripper to the side of the yellow bottle body",
}
RENAMES_D: dict[str, str] = {}  # pour labels are unique already


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


def apply_rename(label: str, renames: dict[str, str]) -> str:
    return renames.get(label, label)


def main() -> None:
    sources = [
        (DATASET_A, "A (pickplace top v4)", RENAMES_A),
        (DATASET_B, "B (twist clean unwrap)", RENAMES_B),
        (DATASET_C, "C (pickplace side v5)", RENAMES_C),
        (DATASET_D, "D (pour clean)", RENAMES_D),
    ]
    print(f"Output: {OUTPUT}")
    for path, label, _ in sources:
        print(f"  {label}: {path}")

    for path, _, _ in sources:
        if not (path / "meta" / "info.json").exists():
            raise FileNotFoundError(
                f"source dataset missing meta/info.json: {path}. "
                f"Run the appropriate filter/preprocess scripts first."
            )

    metas = [load_meta(path) for path, _, _ in sources]
    print()
    for (info, eps, tks), (_, label, _) in zip(metas, sources):
        print(f"  {label}: {len(eps)} episodes, tasks={list(tks)}")

    # Action shape + fps consistency check.
    shapes = [m[0]["features"]["actions"]["shape"] for m in metas]
    fpses = [m[0]["fps"] for m in metas]
    if len(set(tuple(s) for s in shapes)) > 1:
        raise ValueError(f"action shape mismatch: {shapes}")
    if len(set(fpses)) > 1:
        raise ValueError(f"fps mismatch: {fpses}")
    merged_fps = fpses[0]
    print(f"\n  All sources: action shape={shapes[0]}, fps={merged_fps}")

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "meta").mkdir()
    (OUTPUT / "data").mkdir()

    sample_parquet = next((DATASET_A / "data").glob("**/episode_*.parquet"))
    hf_metadata = pq.read_table(sample_parquet).schema.metadata

    # Build unified task map by applying per-source renames and deduplicating.
    merged_tasks: dict[str, int] = {}
    for (info, eps, tks), (_, label, renames) in zip(metas, sources):
        for t in tks:
            new_t = apply_rename(t, renames)
            if new_t not in merged_tasks:
                merged_tasks[new_t] = len(merged_tasks)
                src_note = f"renamed from {t!r}" if new_t != t else "unchanged"
                print(f"  task[{merged_tasks[new_t]:2d}]: {new_t!r}  ({label}, {src_note})")

    new_episodes: list[dict] = []
    total_frames = 0
    global_idx = 0

    def copy_source(src: Path, episodes: list[dict], renames: dict[str, str], label: str) -> None:
        nonlocal total_frames, global_idx
        print(f"\nCopying {len(episodes)} episodes from {label}...")
        for ep in tqdm.tqdm(episodes, desc=f"  {label}"):
            old_idx = ep["episode_index"]
            old_task = ep["tasks"][0]
            new_task = apply_rename(old_task, renames)
            new_idx = len(new_episodes)

            parquet_path = (
                src / f"data/chunk-{old_idx // CHUNK_SIZE:03d}/episode_{old_idx:06d}.parquet"
            )
            df = pd.read_parquet(parquet_path)
            total_frames += len(df)

            df["frame_index"] = range(len(df))
            df["episode_index"] = new_idx
            df["index"] = range(global_idx, global_idx + len(df))
            df["task_index"] = merged_tasks[new_task]
            global_idx += len(df)

            chunk_dir = OUTPUT / f"data/chunk-{new_idx // CHUNK_SIZE:03d}"
            chunk_dir.mkdir(exist_ok=True)
            table = pa.Table.from_pandas(df)
            table = table.replace_schema_metadata(hf_metadata)
            pq.write_table(table, chunk_dir / f"episode_{new_idx:06d}.parquet")

            new_episodes.append(
                {"episode_index": new_idx, "tasks": [new_task], "length": len(df)}
            )

    for (_, episodes, _), (path, label, renames) in zip(metas, sources):
        copy_source(path, episodes, renames, label)

    with open(OUTPUT / "meta/tasks.jsonl", "w") as f:
        for task, idx in sorted(merged_tasks.items(), key=lambda kv: kv[1]):
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")

    with open(OUTPUT / "meta/episodes.jsonl", "w") as f:
        for ep in new_episodes:
            f.write(json.dumps(ep) + "\n")

    info = dict(metas[0][0])
    info["fps"] = merged_fps
    info["total_episodes"] = len(new_episodes)
    info["total_frames"] = total_frames
    info["total_tasks"] = len(merged_tasks)
    info["total_chunks"] = ((len(new_episodes) - 1) // CHUNK_SIZE) + 1 if new_episodes else 0
    info["splits"] = {"train": f"0:{len(new_episodes)}"}
    with open(OUTPUT / "meta/info.json", "w") as f:
        json.dump(info, f, indent=4)

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
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {n:4d}: {name!r}")


if __name__ == "__main__":
    main()
