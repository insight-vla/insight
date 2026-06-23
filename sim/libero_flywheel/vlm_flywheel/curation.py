"""Swiss tournament + feedback summarization for trajectory curation."""

from __future__ import annotations

import dataclasses
import glob
import json
import logging
import pathlib

import cv2
import numpy as np
import tyro

from .vlm import parse_vlm_json, vlm_with_images
from .prompts import _COMPARE_SYSTEM, _SUMMARIZE_SYSTEM


# ---------------------------------------------------------------------------
#  Frame extraction from demo MP4s
# ---------------------------------------------------------------------------

_SAMPLE_FPS = 1


def extract_frames(run_dir: pathlib.Path, primitive: str = "rotate block"):
    """Extract temporally-sampled frames from a primitive's demo MP4."""
    pattern = str(run_dir / f"demo_*_{primitive.replace(' ', '_')}.mp4")
    matches = glob.glob(pattern)
    if not matches:
        return None, 0
    cap = cv2.VideoCapture(matches[0])
    if not cap.isOpened():
        return None, 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(fps / _SAMPLE_FPS))
    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((frame, frame_idx))
        frame_idx += 1
    cap.release()
    if not frames:
        return None, 0
    labeled = []
    n_sampled = len(frames)
    for i, (frame, orig_idx) in enumerate(frames, 1):
        labeled.append(_label_frame(frame, f"Frame {i}/{n_sampled}"))
    return labeled, total_frames


def _label_frame(img: np.ndarray, text: str) -> np.ndarray:
    """Burn a label into the top-left corner of an image."""
    img = img.copy()
    h, w = img.shape[:2]
    font_scale = max(0.4, w / 640)
    thickness = max(1, int(w / 256))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(img, (0, 0), (tw + 8, th + 10), (0, 0, 0), -1)
    cv2.putText(img, text, (4, th + 6), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
#  VLM pairwise comparison
# ---------------------------------------------------------------------------

def compare_trajectories(frames_a, frames_b, primitive="rotate block"):
    """VLM pairwise comparison. Returns (winner, reasoning)."""
    n_a, n_b = len(frames_a), len(frames_b)
    prompt = (
        f'PRIMITIVE: "{primitive}"\n\n'
        f"TRAJECTORY A: first {n_a} images (labeled Frame 1/{n_a} through Frame {n_a}/{n_a})\n"
        f"TRAJECTORY B: next {n_b} images (labeled Frame 1/{n_b} through Frame {n_b}/{n_b})\n\n"
        f"All frames are in chronological order. Which trajectory is better?"
    )
    images = frames_a + frames_b
    try:
        data = parse_vlm_json(
            vlm_with_images(prompt, images, max_tokens=300, system=_COMPARE_SYSTEM)
        )
        winner = data.get("winner", "").upper()
        reasoning = data.get("reasoning", "")
        winner = winner if winner in ("A", "B") else "tie"
        return winner, reasoning
    except Exception as e:
        logging.warning(f"  [curate] VLM comparison failed: {e}")
        return "tie", ""


# ---------------------------------------------------------------------------
#  Feedback summarization
# ---------------------------------------------------------------------------

def summarize_feedback(comparisons, primitive="rotate block"):
    """Distill pairwise comparison reasoning into actionable feedback."""
    reasoning_lines = []
    for c in comparisons:
        if c.get("reasoning"):
            reasoning_lines.append(f"- {c['winner']} won: {c['reasoning']}")
    if not reasoning_lines:
        return None
    prompt = (
        f'TASK: "{primitive}"\n\n'
        f"PAIRWISE COMPARISON NOTES ({len(reasoning_lines)} comparisons):\n"
        + "\n".join(reasoning_lines)
    )
    try:
        return parse_vlm_json(
            vlm_with_images(prompt, [], max_tokens=500, system=_SUMMARIZE_SYSTEM)
        )
    except Exception as e:
        logging.warning(f"  [curate] Feedback summarization failed: {e}")
        return None


# ---------------------------------------------------------------------------
#  Swiss-style tournament
# ---------------------------------------------------------------------------

def curate_batch(base_output: pathlib.Path, keep_ratio: float = 0.7, num_rounds: int = 5):
    """Rank successful demos via Swiss tournament, symlink top ones to curated/."""
    runs = []
    for run_dir in sorted(base_output.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "curated":
            continue
        summary_path = run_dir / "adaptive_summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        if not summary.get("goal_achieved"):
            continue
        frames, n_total = extract_frames(run_dir)
        if frames is None:
            continue
        runs.append({"dir": run_dir, "frames": frames, "n_steps": n_total,
                      "score": 0.0, "wins": 0, "losses": 0})
    if len(runs) < 3:
        logging.info(f"  [curate] Only {len(runs)} successful runs — skipping curation")
        return
    logging.info(
        f"\n{'='*60}\n"
        f"[CURATION] Swiss tournament: {len(runs)} trajectories, {num_rounds} rounds\n"
        f"{'='*60}"
    )
    comparisons = []
    for round_num in range(1, num_rounds + 1):
        runs.sort(key=lambda r: (-r["score"], -r["wins"]))
        pairs = []
        for i in range(0, len(runs) - 1, 2):
            pairs.append((i, i + 1))
        if len(runs) % 2 == 1:
            runs[-1]["score"] += 0.5
        for i, j in pairs:
            winner, reasoning = compare_trajectories(runs[i]["frames"], runs[j]["frames"])
            if winner == "A":
                runs[i]["score"] += 1
                runs[i]["wins"] += 1
                runs[j]["losses"] += 1
            elif winner == "B":
                runs[j]["score"] += 1
                runs[j]["wins"] += 1
                runs[i]["losses"] += 1
            else:
                runs[i]["score"] += 0.5
                runs[j]["score"] += 0.5
            comparisons.append({
                "round": round_num,
                "a": runs[i]["dir"].name, "b": runs[j]["dir"].name,
                "winner": winner,
                "reasoning": reasoning,
            })
            logging.info(f"  Round {round_num}: {runs[i]['dir'].name} vs {runs[j]['dir'].name} → {winner}")
    runs.sort(key=lambda r: (-r["score"], -r["wins"]))
    keep_n = max(1, int(len(runs) * keep_ratio))
    curated_dir = base_output / "curated"
    curated_dir.mkdir(exist_ok=True)
    rankings = []
    for rank, r in enumerate(runs, 1):
        accepted = rank <= keep_n
        rankings.append({
            "rank": rank, "run": r["dir"].name,
            "score": r["score"], "wins": r["wins"], "losses": r["losses"],
            "n_steps": r["n_steps"], "accepted": accepted,
        })
        if accepted:
            link = curated_dir / r["dir"].name
            if not link.exists():
                link.symlink_to(r["dir"].resolve())
    with open(curated_dir / "rankings.json", "w") as f:
        json.dump({"total": len(runs), "kept": keep_n, "keep_ratio": keep_ratio,
                    "rankings": rankings, "comparisons": comparisons}, f, indent=2)
    logging.info(f"\n[CURATION] Kept {keep_n}/{len(runs)} demos → {curated_dir}")
    for r in rankings[:keep_n]:
        logging.info(f"  #{r['rank']}: {r['run']} (score={r['score']}, wins={r['wins']})")
    feedback_data = summarize_feedback(comparisons)
    if feedback_data:
        feedback_data["n_comparisons"] = len(comparisons)
        feedback_data["n_trajectories"] = len(runs)
        feedback_data["source_dir"] = str(base_output)
        with open(curated_dir / "feedback.json", "w") as f:
            json.dump(feedback_data, f, indent=2)
        logging.info(f"[CURATION] Feedback saved to {curated_dir / 'feedback.json'}")


# ---------------------------------------------------------------------------
#  Standalone CLI
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class CurateArgs:
    data_dir: str = ""
    keep_ratio: float = 0.7
    num_rounds: int = 5
    vlm: str = "gpt"


def main(args: CurateArgs) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.StreamHandler()])
    if not args.data_dir:
        logging.error("--args.data_dir is required")
        return
    if args.vlm != "gpt":
        from .vlm import set_vlm_provider
        set_vlm_provider(args.vlm)
    curate_batch(pathlib.Path(args.data_dir), args.keep_ratio, args.num_rounds)
