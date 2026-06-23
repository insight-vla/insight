"""Stitch the 5 trimmed primitive episodes back into one full pickplace demo
per source demo, side-by-side exterior+wrist, with the primitive name burned
into each frame so you can see transitions.

Use to spot-check whether trimming made the boundaries between primitives
look awkward (sudden pose jumps, cut motion).

Run:
    uv run training/preprocess/visualize_trimmed_demos.py \
        --root ~/.cache/huggingface/lerobot/maggie/xarm_pick_up_top_30_primitives_trimmed_v2 \
        --num-demos 20 \
        --out data/demo_videos/pickplace_trim_v2_full
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import tyro
from PIL import Image, ImageDraw, ImageFont


PRIMS_PER_DEMO = 5  # move / close / lift / lower / open


def _decode(blob: dict) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(blob["bytes"])).convert("RGB"))


def _annotate(img: np.ndarray, label: str, frame_idx: int, total: int) -> np.ndarray:
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    txt = f"{label}  [{frame_idx + 1}/{total}]"
    # background bar so text is readable on any image
    draw.rectangle([(0, 0), (pil.width, 22)], fill=(0, 0, 0))
    draw.text((4, 3), txt, fill=(255, 255, 255), font=font)
    return np.asarray(pil)


def main(
    root: Path,
    num_demos: int = 20,
    out: Path = Path("data/demo_videos/pickplace_trim_v2_full"),
    fps: int = 20,
) -> None:
    eps = [json.loads(l) for l in open(root / "meta" / "episodes.jsonl")]
    if len(eps) % PRIMS_PER_DEMO != 0:
        print(f"WARNING: {len(eps)} episodes is not a multiple of {PRIMS_PER_DEMO}; "
              "stitching will be off if primitives are missing for some demos.")
    n_full_demos = len(eps) // PRIMS_PER_DEMO
    take = min(num_demos, n_full_demos)
    out.mkdir(parents=True, exist_ok=True)
    print(f"stitching {take} demos (of {n_full_demos}) into {out}")

    for d in range(take):
        ep_indices = list(range(d * PRIMS_PER_DEMO, (d + 1) * PRIMS_PER_DEMO))
        all_frames = []
        for ep_idx in ep_indices:
            ep = eps[ep_idx]
            chunk = ep["episode_index"] // 1000
            df = pd.read_parquet(
                root / f"data/chunk-{chunk:03d}/episode_{ep['episode_index']:06d}.parquet"
            )
            label = ep["tasks"][0]
            n = len(df)
            for i, row in enumerate(df.itertuples(index=False)):
                ext = _decode(getattr(row, "exterior_image_1_left"))
                wri = _decode(getattr(row, "wrist_image_left"))
                # match heights
                if ext.shape[0] != wri.shape[0]:
                    h = max(ext.shape[0], wri.shape[0])
                    ext = np.vstack([ext, np.zeros((h - ext.shape[0], ext.shape[1], 3), dtype=ext.dtype)]) if ext.shape[0] < h else ext
                    wri = np.vstack([wri, np.zeros((h - wri.shape[0], wri.shape[1], 3), dtype=wri.dtype)]) if wri.shape[0] < h else wri
                stitched = np.hstack([ext, wri])
                stitched = _annotate(stitched, label, i, n)
                all_frames.append(stitched)
        out_path = out / f"demo_{d:02d}.mp4"
        with imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8) as w:
            for f in all_frames:
                w.append_data(f)
        print(f"  demo {d:02d}: {len(all_frames)} frames @ {fps}fps "
              f"= {len(all_frames)/fps:.1f}s -> {out_path}")


if __name__ == "__main__":
    tyro.cli(main)
