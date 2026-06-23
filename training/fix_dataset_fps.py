"""One-off: rescale a LeRobot dataset's recorded fps from OLD to NEW.

Halves/scales every timestamp column in-place so the metadata fps and the
per-frame timestamps agree. Use this when a dataset was created with the
wrong fps in LeRobotDataset.create() and you don't want to re-run the
upstream pipeline. Only modifies timestamp-related fields; actions, state,
images, frame_index etc. are left untouched.

Run:
    uv run scripts/fix_dataset_fps.py \
        --root ~/.cache/huggingface/lerobot/maggie/xarm_pick_up_top_30_primitives \
        --old-fps 10 --new-fps 20
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm
import tyro


def main(root: Path, old_fps: int = 10, new_fps: int = 20) -> None:
    if old_fps == new_fps:
        print(f"old_fps == new_fps ({old_fps}), nothing to do")
        return

    scale = old_fps / new_fps  # multiply timestamps by this; e.g. 10/20=0.5

    # 1. info.json
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    if info.get("fps") != old_fps:
        print(f"WARNING: info.json fps={info.get('fps')} (expected {old_fps}). Aborting.")
        return
    info["fps"] = new_fps
    info_path.write_text(json.dumps(info, indent=4))
    print(f"  info.json fps: {old_fps} -> {new_fps}")

    # 2. parquets — rescale timestamp column, preserve hf metadata
    parquets = sorted((root / "data").rglob("*.parquet"))
    print(f"  Rescaling timestamp in {len(parquets)} parquets (x{scale})...")
    for p in tqdm.tqdm(parquets):
        table = pq.read_table(p)
        hf_meta = table.schema.metadata
        df = table.to_pandas()
        if "timestamp" not in df.columns:
            print(f"    {p.name}: no timestamp col, skipping")
            continue
        df["timestamp"] = df["timestamp"] * scale
        new_table = pa.Table.from_pandas(df)
        new_table = new_table.replace_schema_metadata(hf_meta)
        pq.write_table(new_table, p)

    # 3. episodes_stats.jsonl — rescale timestamp stats per episode
    stats_path = root / "meta" / "episodes_stats.jsonl"
    if stats_path.exists():
        out_lines = []
        with stats_path.open() as f:
            for line in f:
                row = json.loads(line)
                ts = row.get("stats", {}).get("timestamp")
                if ts is not None:
                    for k in ("min", "max", "mean", "std"):
                        if k in ts:
                            ts[k] = [v * scale for v in ts[k]]
                out_lines.append(json.dumps(row))
        stats_path.write_text("\n".join(out_lines) + "\n")
        print(f"  episodes_stats.jsonl: rescaled timestamp stats")

    print(f"\nDone. Dataset at {root} is now labeled as {new_fps}fps with consistent timestamps.")


if __name__ == "__main__":
    tyro.cli(main)
