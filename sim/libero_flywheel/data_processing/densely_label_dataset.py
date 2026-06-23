#!/usr/bin/env python3
"""Densely label LIBERO dataset with primitive descriptions using VLM."""

import dataclasses
import json
import logging
import os
import pathlib
import shutil
from typing import List, Dict, Tuple

from dotenv import load_dotenv
import imageio
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME
import numpy as np
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
import torch
import tyro

load_dotenv()


@dataclasses.dataclass
class Args:
    dataset_repo_id: str = "physical-intelligence/libero"
    output_base_dir: str = "data/xarm/dense_labels"
    num_episodes: int = 10  # Start small for testing
    episode_indices: str = ""  # Comma-separated specific episode indices to process (e.g. "47,48,49"); empty = use num_episodes range
    segment_method: str = "gripper"  # "gripper", "action_change", "fixed_chunks", or "video" (VLM labels from video frames)
    chunk_size: int = 10  # For fixed_chunks method
    action_threshold: float = 0.05  # Threshold for detecting action changes
    min_segment_length: int = 5  # Minimum frames per segment
    max_segment_length: int = 50  # Maximum frames per segment
    save_videos: bool = True  # Save videos for each segment
    fps: int = 10  # FPS for saved videos
    run_name: str = ""  # Optional name for this run (otherwise uses timestamp)
    # LeRobot dataset creation options
    create_lerobot_dataset: bool = False  # Create a LeRobot dataset with primitive labels
    lerobot_repo_name: str = "maggiewang/libero_primitives"  # Name for the new dataset
    include_task_prefix: bool = False  # If True, prompt = "task: primitive", else just "primitive"
    include_original_episodes: bool = True  # Also include original task-labeled episodes (for mixed training)
    # Resume from existing labels (skip API calls)
    load_labels_from: str = ""  # Path to existing dense_labels.json to skip labeling and just create LeRobot dataset
    start_from_episode: int = 0  # Skip episodes before this index (for resuming after crash)
    end_at_episode: int = -1  # Stop at this episode index (-1 = process all)
    # Schema adapters for non-LIBERO datasets (e.g. xArm scoop hardware data)
    has_gripper: bool = True  # If False, skip gripper segmentation + post-processing (forces action_change method)
    gripper_close_threshold: float = 0.0  # Crossing point for open<->closed in action[6]; libero=0, xarm≈0.3
    primary_image_key: str = "image"  # Source-dataset key used to extract frames for VLM prompts / videos
    image_keys: str = ""  # Comma-separated list of image keys to mirror in the output LeRobot dataset (defaults to primary + "wrist_image")
    # Changepoint tuning (used for gripperless plan-driven split + mid-pipeline changepoint splits)
    changepoint_window: int = 5  # Smoothing window for variance-reduction changepoints
    changepoint_normalize: bool = False  # Z-score features per-dim before variance calc (helps when dim scales differ)
    changepoint_backtrack: str = "backward"  # "backward" | "forward" | "none"
    changepoint_feature: str = "actions"  # "actions" (velocity/delta) or "state" (EE pose trajectory — peaks align with visual arrivals, better for gripperless tasks)
    # VLM-driven labeling (gripperless, when action/state changepoints don't align with visual transitions)
    use_vlm_labels: bool = False  # Ask VLM to localize each plan primitive in time from video keyframes (instead of action/state changepoints)
    vlm_label_frame_subsample: int = 5  # Send every Nth frame to the VLM for coarse labeling pass
    vlm_label_snap_to_changepoint: bool = False  # After VLM returns frame boundaries, snap each to nearest state-based changepoint (±N frames)
    vlm_label_snap_radius: int = 10  # Snap radius in frames
    vlm_label_refine: bool = True  # Second VLM pass that zooms in on each boundary at frame-level granularity for precise placement
    vlm_label_refine_window: int = 15  # Half-window size (in frames) around each coarse boundary for the refine pass
    extra_image_keys: str = ""  # Comma-separated list of EXTRA camera keys (e.g. "wrist_image_left") to include alongside primary_image_key in VLM plan decomposition and labeling calls


def segment_episode_by_action_change(
    actions: torch.Tensor,
    threshold: float = 0.05,
    min_length: int = 5,
    max_length: int = 50,
) -> List[Tuple[int, int]]:
    """Segment episode into primitives based on action changes.

    Returns list of (start_idx, end_idx) tuples.
    """
    segments = []
    current_start = 0

    # Compute action velocity (change between consecutive actions)
    action_diffs = torch.diff(actions, dim=0).abs().mean(dim=-1)

    # Smooth with simple moving average
    window_size = 3
    smoothed_diffs = torch.nn.functional.avg_pool1d(
        action_diffs.unsqueeze(0).unsqueeze(0),
        kernel_size=window_size,
        stride=1,
        padding=window_size // 2
    ).squeeze()

    # Find segment boundaries
    for i in range(len(smoothed_diffs)):
        segment_length = i - current_start

        # Force segment if too long
        if segment_length >= max_length:
            segments.append((current_start, i))
            current_start = i
        # Start new segment if action change detected and min length met
        elif segment_length >= min_length and smoothed_diffs[i] > threshold:
            segments.append((current_start, i))
            current_start = i

    # Add final segment
    if current_start < len(actions) - 1:
        segments.append((current_start, len(actions) - 1))

    return segments


def segment_episode_fixed_chunks(
    episode_length: int,
    chunk_size: int = 10
) -> List[Tuple[int, int]]:
    """Segment episode into fixed-size chunks (like action horizon)."""
    segments = []
    for i in range(0, episode_length, chunk_size):
        end_idx = min(i + chunk_size, episode_length)
        segments.append((i, end_idx))
    return segments


def segment_episode_by_gripper(
    actions: torch.Tensor,
    states: torch.Tensor,
    min_segment_length: int = 5,
    gripper_motion_threshold: float = 0.002,  # Velocity threshold for detecting gripper in motion
    gripper_close_threshold: float = 0.0,  # Direction tiebreaker only — see docstring.
) -> List[Tuple[int, int, str]]:
    """Segment episode based on gripper ACTION COMMAND transitions.

    Strategy: a frame belongs to a gripper-action segment IF AND ONLY IF the
    gripper command is moving at that frame. Detection is purely from the
    command-velocity signal — connected regions where
    ``|action[6][t] - action[6][t-1]| > gripper_motion_threshold`` are gripper
    transitions; everything else is "move". Direction (close vs. open) is
    determined by the sign of the command change across the region.

    This means the gripper segment captures the FULL ramp (start to end of
    motion), not just frames after a threshold crossing. Pre-ramp and
    post-ramp frames where the command is stable get attributed to the
    surrounding "move" segments — which is exactly correct: the gripper
    isn't doing anything during stable-state frames.

    LIBERO convention: action[6] in {-1, +1} — sharp transitions, the velocity
    detector picks them up cleanly with default threshold.
    xArm convention:   action[6] in [0, ~0.5] — smooth ramps over ~10 frames,
    detector captures the full ramp window.

    ``gripper_close_threshold`` is no longer used for boundary placement
    (kept in the signature for backward compat); direction comes from the
    sign of the command change.

    Returns list of (start_idx, end_idx, segment_type) tuples.
    segment_type is one of: "move", "close_gripper", "open_gripper"
    """
    gripper_cmd = actions[:, 6]
    n_frames = len(gripper_cmd)

    # Frame-to-frame velocity of the gripper command. Use prepend so the
    # output matches gripper_cmd length and frame indices align directly.
    gripper_vel = torch.abs(torch.diff(gripper_cmd, prepend=gripper_cmd[:1]))
    in_transition = gripper_vel > gripper_motion_threshold

    # Find connected transition regions. Each is a contiguous run of frames
    # where the command is moving — that IS the gripper segment, by definition.
    transitions: List[Tuple[int, int, str]] = []
    i = 0
    while i < n_frames:
        if not in_transition[i]:
            i += 1
            continue
        start = i
        while i < n_frames and in_transition[i]:
            i += 1
        end = i  # exclusive
        # Direction: positive command change → closing; negative → opening.
        delta = gripper_cmd[end - 1].item() - gripper_cmd[max(start - 1, 0)].item()
        kind = "close_gripper" if delta > 0 else "open_gripper"
        # Ignore micro-transitions (single noise spikes that briefly exceed
        # the threshold). 3 frames is the same minimum the old code used.
        if end - start >= 3:
            transitions.append((start, end, kind))

    # Build segments by interleaving "move" segments between gripper transitions.
    segments: List[Tuple[int, int, str]] = []
    prev_end = 0
    for start, end, kind in transitions:
        if start > prev_end + min_segment_length:
            segments.append((prev_end, start, "move"))
        segments.append((start, end, kind))
        prev_end = end

    # Final trailing "move" segment if there's any frames left.
    if prev_end < n_frames - min_segment_length:
        segments.append((prev_end, n_frames, "move"))

    # If no segments created (no gripper transitions), treat whole episode as "move"
    if not segments:
        segments = [(0, n_frames, "move")]

    # Merge consecutive "move" segments
    merged = []
    for seg in segments:
        if merged and seg[2] == "move" and merged[-1][2] == "move":
            # Merge with previous move segment
            prev = merged[-1]
            merged[-1] = (prev[0], seg[1], "move")
        else:
            merged.append(seg)

    # Sub-divide long "move" segments based on vertical motion changes
    # This helps distinguish "move to object", "lift", "move to target", "lower"
    final_segments = []
    for seg in merged:
        start, end, seg_type = seg
        seg_length = end - start

        if seg_type == "move" and seg_length > 30:
            # Analyze vertical motion to find sub-segments
            seg_actions = actions[start:end, :3]  # Just position deltas

            # Compute cumulative vertical motion
            z_motion = seg_actions[:, 2]

            # Find where motion direction changes significantly
            # Simple approach: split at significant direction changes
            sub_segs = []
            sub_start = 0

            for i in range(10, len(z_motion), 5):  # Check every 5 frames
                # Look at recent vs upcoming motion
                recent_z = z_motion[max(0, i-10):i].sum().item()
                upcoming_z = z_motion[i:min(len(z_motion), i+10)].sum().item()

                # Direction change (going up vs going down)
                if (recent_z > 0.05 and upcoming_z < -0.05) or (recent_z < -0.05 and upcoming_z > 0.05):
                    if i - sub_start >= 10:  # Minimum sub-segment length
                        sub_segs.append((start + sub_start, start + i))
                        sub_start = i

            # Add final sub-segment
            if len(z_motion) - sub_start >= 10:
                sub_segs.append((start + sub_start, end))

            # If we found meaningful sub-segments, use them; otherwise keep original
            if len(sub_segs) > 1:
                for sub_start, sub_end in sub_segs:
                    final_segments.append((sub_start, sub_end, "move"))
            else:
                final_segments.append(seg)
        else:
            final_segments.append(seg)

    return final_segments if final_segments else merged


def _frame_to_hwc_uint8(frame, image_key: str) -> np.ndarray:
    """Convert a LeRobot frame's image tensor to HWC uint8 array.

    Handles both CHW (LIBERO convention) and HWC (xArm convention), plus
    float (0-1) and uint8 (0-255) inputs.
    """
    img = frame[image_key].numpy()
    # CHW -> HWC if first dim is 3 and last isn't
    if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
        img = img.transpose(1, 2, 0)
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    return img


def extract_keyframes(
    dataset: LeRobotDataset,
    episode_idx: int,
    segment_start: int,
    segment_end: int,
    num_keyframes: int = 15,
    image_key: str = "image",
) -> List[np.ndarray]:
    """Extract keyframes from a segment."""
    episode_data_start = dataset.episode_data_index['from'][episode_idx].item()

    # Get indices for keyframes (evenly spaced through segment)
    segment_length = segment_end - segment_start
    if segment_length <= num_keyframes:
        keyframe_indices = list(range(segment_start, segment_end))
    else:
        # Evenly space keyframes through the segment
        keyframe_indices = [
            segment_start + int(i * (segment_length - 1) / (num_keyframes - 1))
            for i in range(num_keyframes)
        ]

    keyframes = []
    for idx in keyframe_indices:
        frame = dataset[episode_data_start + idx]
        keyframes.append(_frame_to_hwc_uint8(frame, image_key))

    return keyframes


def extract_all_frames(
    dataset: LeRobotDataset,
    episode_idx: int,
    segment_start: int,
    segment_end: int,
    image_key: str = "image",
) -> List[np.ndarray]:
    """Extract all frames from a segment for video generation."""
    episode_data_start = dataset.episode_data_index['from'][episode_idx].item()

    frames = []
    for idx in range(segment_start, segment_end):
        frame = dataset[episode_data_start + idx]
        frames.append(_frame_to_hwc_uint8(frame, image_key))

    return frames


def annotate_frame(img: np.ndarray, primitive_label: str, frame_idx: int, total_frames: int, task_description: str = None) -> np.ndarray:
    """Add text annotation showing primitive label, frame number, and task description."""
    pil_img = Image.fromarray(img)
    img_width, img_height = pil_img.size

    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Strip quotes from label if present
    label = primitive_label.strip('"\'')

    # Create new image with space for text at top and bottom
    padding = 6
    top_height = 30
    bottom_height = 28 if task_description else 0  # Space for 2 lines
    new_img = Image.new('RGB', (img_width, img_height + top_height + bottom_height), color=(30, 30, 40))
    new_img.paste(pil_img, (0, top_height))

    draw = ImageDraw.Draw(new_img)

    # Draw primitive label at top
    draw.text((padding, padding), label, fill=(255, 200, 0), font=font_large)

    # Draw frame counter in top right
    frame_text = f"{frame_idx}/{total_frames}"
    bbox = draw.textbbox((0, 0), frame_text, font=font_small)
    text_width = bbox[2] - bbox[0]
    draw.text((img_width - text_width - padding, padding + 5),
              frame_text, fill=(200, 200, 200), font=font_small)

    # Draw task description at bottom (wrap to multiple lines if too long)
    if task_description:
        task_text = f"Task: {task_description}"
        max_width = img_width - 2 * padding
        line_height = 12

        # Wrap text to multiple lines
        words = task_text.split()
        lines = []
        current_line = ""
        for word in words:
            test_line = current_line + " " + word if current_line else word
            test_bbox = draw.textbbox((0, 0), test_line, font=font_small)
            if test_bbox[2] - test_bbox[0] <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)

        # Draw each line
        for i, line in enumerate(lines[:2]):  # Max 2 lines
            draw.text((padding, img_height + top_height + 2 + i * line_height), line, fill=(150, 200, 255), font=font_small)

    return np.array(new_img)


def find_action_changepoints(
    actions: torch.Tensor,
    seg_start: int,
    seg_end: int,
    n_splits: int,
    window: int = 5,
    normalize_features: bool = False,
    min_seg_len: int = None,
    backtrack: str = "backward",
    feature_tensor: torch.Tensor = None,
) -> List[int]:
    """Find changepoints in action profile within a segment.

    Recursive binary splitting: at each step, pick the sub-segment with the
    highest variance-reduction peak anywhere in the sequence and split it.
    Repeat until n_splits boundaries are placed (or no meaningful reduction
    remains). Each split is scored *within* its own sub-segment, so suppression
    between nearby candidates isn't needed.

    Args:
        actions: full episode actions tensor
        seg_start: start frame of segment
        seg_end: end frame of segment
        n_splits: number of split points to find (= num_primitives - 1)
        window: smoothing window size
        normalize_features: if True, z-score per-dim before variance calc.
            Helps when action dims have very different magnitudes (e.g. raw
            xArm meters vs radians).
        min_seg_len: minimum sub-segment length. Defaults to max(3, window).
        backtrack: "backward" (walk left to capture transition onset),
            "forward" (walk right to let segment finish before handing off), or
            "none" (use raw variance-peak frame).

    Returns: list of split frame indices (in original episode numbering)
    """
    # Use override feature tensor (e.g. state) if provided, else action deltas.
    source_tensor = feature_tensor if feature_tensor is not None else actions
    seg_source = source_tensor[seg_start:seg_end]
    seg_len = len(seg_source)
    min_seg = min_seg_len if min_seg_len is not None else max(3, window)
    if seg_len < min_seg * 2:
        return [seg_start + int(seg_len * (i + 1) / (n_splits + 1)) for i in range(n_splits)]

    # Feature: up to first 6 dims (translation + rotation)
    feat_dim = min(6, seg_source.shape[-1])
    full_features = seg_source[:, :feat_dim].float()

    # Smooth
    kernel = torch.ones(window) / window
    smoothed = []
    for dim in range(feat_dim):
        sig = full_features[:, dim]
        if len(sig) > window:
            padded = torch.nn.functional.pad(sig.unsqueeze(0).unsqueeze(0), (window // 2, window // 2), mode='replicate')
            s = torch.nn.functional.conv1d(padded, kernel.unsqueeze(0).unsqueeze(0)).squeeze()
            smoothed.append(s)
        else:
            smoothed.append(sig)
    full_features = torch.stack(smoothed, dim=1)

    # Optional per-dim z-score so one high-magnitude dim doesn't dominate.
    if normalize_features:
        mu = full_features.mean(dim=0, keepdim=True)
        sig = full_features.std(dim=0, keepdim=True) + 1e-6
        full_features = (full_features - mu) / sig

    def best_split_in(start: int, end: int) -> Tuple[int, float]:
        """Find best split within features[start:end] (local indices into full_features).
        Returns (local_split_idx, score) or (-1, -inf) if no valid split.
        """
        sub_len = end - start
        if sub_len < min_seg * 2:
            return -1, float("-inf")
        sub = full_features[start:end]
        total_var = sub.var(dim=0).sum().item()
        if total_var < 1e-6:
            return -1, float("-inf")
        best_idx = -1
        best_score = float("-inf")
        for t in range(min_seg, sub_len - min_seg):
            left_var = sub[:t].var(dim=0).sum().item()
            right_var = sub[t:].var(dim=0).sum().item()
            weighted_var = (t * left_var + (sub_len - t) * right_var) / sub_len
            score = total_var - weighted_var
            if score > best_score:
                best_score = score
                best_idx = t
        return (start + best_idx if best_idx >= 0 else -1), best_score

    def apply_backtrack(local_idx: int, start: int, end: int) -> int:
        """Refine the split inside [start, end] around local_idx."""
        if backtrack == "none":
            return local_idx
        sub = full_features[start:end]
        split_rel = local_idx - start
        left_mean = sub[:split_rel].mean(dim=0)
        right_mean = sub[split_rel:].mean(dim=0)
        idx = split_rel
        if backtrack == "backward":
            for bi in range(split_rel - 1, 0, -1):
                if torch.abs(sub[bi] - right_mean).mean() < torch.abs(sub[bi] - left_mean).mean():
                    idx = bi
                else:
                    break
        elif backtrack == "forward":
            for fi in range(split_rel + 1, len(sub)):
                if torch.abs(sub[fi] - left_mean).mean() < torch.abs(sub[fi] - right_mean).mean():
                    idx = fi
                else:
                    break
        return start + idx

    # Recursive binary splitting: keep a list of current sub-segments
    # along with their best-split candidates. At each iteration, pick the
    # sub-segment whose best-split has the highest score and apply it.
    sub_segments = [(0, seg_len)]
    cached_candidates = {sub_segments[0]: best_split_in(0, seg_len)}
    split_points = []

    for _ in range(n_splits):
        # Find sub-segment with highest-scoring split candidate
        best_sub = None
        best_split_idx = -1
        best_score = float("-inf")
        for sub in sub_segments:
            cand_idx, cand_score = cached_candidates[sub]
            if cand_score > best_score and cand_idx >= 0:
                best_score = cand_score
                best_sub = sub
                best_split_idx = cand_idx
        if best_sub is None or best_score <= 0:
            break
        sub_start, sub_end = best_sub
        # Effect size gate
        sub_feats = full_features[sub_start:sub_end]
        left_mean = full_features[sub_start:best_split_idx].mean(dim=0)
        right_mean = full_features[best_split_idx:sub_end].mean(dim=0)
        pooled_std = sub_feats.std(dim=0) + 1e-8
        effect_size = (torch.abs(left_mean - right_mean) / pooled_std).mean().item()
        if effect_size < 0.3:
            # Sub-segments too similar — drop this candidate and don't split here
            cached_candidates[best_sub] = (-1, float("-inf"))
            continue
        refined_split = apply_backtrack(best_split_idx, sub_start, sub_end)
        split_points.append(seg_start + refined_split)
        # Replace this sub-segment with its two halves + recompute their candidates
        sub_segments.remove(best_sub)
        del cached_candidates[best_sub]
        left_sub = (sub_start, refined_split)
        right_sub = (refined_split, sub_end)
        sub_segments.append(left_sub)
        sub_segments.append(right_sub)
        cached_candidates[left_sub] = best_split_in(*left_sub)
        cached_candidates[right_sub] = best_split_in(*right_sub)

    return sorted(split_points)


def split_merged_segment_from_video(
    client: OpenAI,
    dataset,
    episode_start: int,
    seg_start: int,
    seg_end: int,
    task_description: str,
    labels: List[str],
    model: str = "google/gemini-3-flash-preview",
    max_frames: int = 32,
    image_key: str = "image",
) -> List[Dict]:
    """Send frames from a merged gripper-closed segment to VLM to find transition boundaries.

    Args:
        labels: ordered list of primitives this segment could contain (e.g. ["lift upward", "rotate block"])

    Returns list of sub-segments: [{"start_frame": int, "end_frame": int, "primitive_label": str}, ...]
    """
    import base64
    import tempfile
    from io import BytesIO
    from PIL import Image as PILImage

    seg_length = seg_end - seg_start

    # Create video clip from segment frames
    frames = []
    for fi in range(seg_length):
        frames.append(_frame_to_hwc_uint8(dataset[episode_start + seg_start + fi], image_key))

    # Write to temp mp4
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    imageio.mimwrite(tmp_path, frames, fps=10, codec="libx264")
    with open(tmp_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode()
    os.unlink(tmp_path)

    labels_str = ", ".join(f'"{l}"' for l in labels)

    prompt = f"""This video shows a segment of a robot doing: "{task_description}"
The video is at 10fps. Frame 0 in the video = frame {seg_start} in the original episode. The segment is {seg_length} frames long (frames {seg_start}-{seg_end}).

This segment contains these primitives in order: {labels_str}

For each primitive, identify the start and end frame numbers in the ORIGINAL episode numbering (starting from {seg_start}).
Each primitive must be contiguous, cover the full segment, and appear in the order listed.
Watch the video carefully for changes in motion — a new primitive begins when the character of the motion changes.

Return JSON only: {{"segments": [{{"start_frame": N, "end_frame": N, "primitive_label": "..."}}, ...]}}"""

    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:video/mp4;base64,{video_b64}"}},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=8192,
        temperature=0.1,
    )

    resp = response.choices[0].message.content or ""
    resp = resp.strip()
    logging.info(f"    VLM split response: {resp[:500]}")

    import re
    resp = re.sub(r'^```(?:json)?\s*', '', resp)
    resp = re.sub(r'\s*```$', '', resp)
    json_match = re.search(r'\{.*\}', resp, re.DOTALL)
    if json_match:
        result = json.loads(json_match.group())
        segments = result.get("segments", [])
        if segments:
            # Fix boundaries: ensure contiguous and covers full range
            segments[0]["start_frame"] = seg_start
            segments[-1]["end_frame"] = seg_end
            for i in range(1, len(segments)):
                segments[i]["start_frame"] = segments[i - 1]["end_frame"]
            # Ensure labels match expected order
            for i, seg in enumerate(segments):
                if i < len(labels):
                    seg["primitive_label"] = labels[i]
            return segments
    return None


def label_episode_from_video(
    client: OpenAI,
    dataset,
    episode_start: int,
    episode_length: int,
    task_description: str,
    known_primitives: List[str],
    frame_subsample: int = 5,
    model: str = "google/gemini-3-flash-preview",
    image_key: str = "image",
) -> List[Dict]:
    """Use VLM to label an episode by sending video frames directly.

    Returns list of segments: [{"start_frame": int, "end_frame": int, "primitive_label": str}, ...]
    """
    import base64
    from io import BytesIO
    from PIL import Image as PILImage

    # Extract and subsample frames
    frame_indices = list(range(0, episode_length, frame_subsample))
    frames_b64 = []
    for fi in frame_indices:
        img_np = _frame_to_hwc_uint8(dataset[episode_start + fi], image_key)
        pil_img = PILImage.fromarray(img_np)
        buf = BytesIO()
        pil_img.save(buf, format="JPEG", quality=80)
        frames_b64.append(base64.b64encode(buf.getvalue()).decode())

    known_str = "\n".join(f"  - {p}" for p in known_primitives)

    prompt = f"""You are labeling a robot manipulation video. The video is at {10 // frame_subsample}fps (subsampled from 10fps, showing every {frame_subsample}th frame).
The original video is 10fps, so frame N in these images = frame N*{frame_subsample} in the original.

Task: "{task_description}"

Known primitive types:
{known_str}

For each primitive action in the video, identify:
1. The primitive label (from the known list, or create a new one if needed)
2. The start frame number (in ORIGINAL frame numbers, i.e., multiply the image index by {frame_subsample})
3. The end frame number (in ORIGINAL frame numbers)

Rules:
- Every frame must belong to exactly one primitive (no gaps, no overlaps)
- The last segment's end_frame should be {episode_length}
- For flipping tasks: use [move gripper to object, close gripper, lift upward, rotate block, open gripper]
- Short gripper transitions (close/open) may be just a few frames

Return JSON only: {{"segments": [{{"start_frame": 0, "end_frame": N, "primitive_label": "..."}}, ...]}}"""

    # Build message content with interleaved images
    content = [{"type": "text", "text": prompt}]
    for i, b64 in enumerate(frames_b64):
        content.append({"type": "text", "text": f"Frame {i * frame_subsample}:"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=2048,
        temperature=0.1,
    )

    resp_content = response.choices[0].message.content or ""
    resp_content = resp_content.strip()
    logging.info(f"  VLM video response: {resp_content[:300]}")

    # Parse JSON
    import re
    resp_content = re.sub(r'^```(?:json)?\s*', '', resp_content)
    resp_content = re.sub(r'\s*```$', '', resp_content)
    json_match = re.search(r'\{.*\}', resp_content, re.DOTALL)
    if json_match:
        result = json.loads(json_match.group())
    else:
        result = json.loads(resp_content)

    segments = result.get("segments", [])

    # Validate and fix segments
    if segments:
        # Ensure first segment starts at 0
        segments[0]["start_frame"] = 0
        # Ensure last segment ends at episode_length
        segments[-1]["end_frame"] = episode_length
        # Ensure no gaps between segments
        for i in range(1, len(segments)):
            segments[i]["start_frame"] = segments[i - 1]["end_frame"]

    return segments


def _encode_frame_b64(arr: np.ndarray) -> str:
    import base64
    from io import BytesIO
    from PIL import Image as PILImage
    pil_img = PILImage.fromarray(arr)
    buf = BytesIO()
    pil_img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def _extract_json(resp: str) -> dict:
    """Extract the last top-level JSON object in a VLM response. Handles reasoning prose before the block."""
    import re
    resp = resp.strip()
    # Strip trailing ``` and leading ```json if wrapped
    fenced = re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', resp, re.DOTALL)
    if fenced:
        return json.loads(fenced[-1])
    # Find the last balanced top-level {...} by scanning from right
    depth = 0
    end = -1
    for i in range(len(resp) - 1, -1, -1):
        c = resp[i]
        if c == '}':
            if depth == 0:
                end = i
            depth += 1
        elif c == '{':
            depth -= 1
            if depth == 0 and end > i:
                return json.loads(resp[i:end + 1])
    # Fallback: direct parse
    return json.loads(resp)


def _refine_boundary(
    client: OpenAI,
    dataset,
    episode_start: int,
    lo: int,
    hi: int,
    prev_label: str,
    next_label: str,
    task_description: str,
    image_key: str,
    extra_image_keys: List[str] = None,
    model: str = "google/gemini-3-flash-preview",
    states: torch.Tensor = None,
) -> int:
    """Zoom in on a boundary by sending every frame in [lo, hi] and asking the VLM
    to pick the exact transition frame between two specified primitives.

    When `states` (EE pose per frame) is provided, per-frame EE pose + velocity
    signals are injected into the prompt so the VLM can cross-reference visual
    evidence against precise physical motion.
    """
    cam_keys = [image_key] + list(extra_image_keys or [])
    n_frames = hi - lo
    if n_frames <= 1:
        return lo
    # Cap frames sent to avoid huge payloads
    MAX_FRAMES = 30
    if n_frames > MAX_FRAMES:
        stride = max(1, n_frames // MAX_FRAMES)
        frame_indices = list(range(lo, hi, stride))
    else:
        frame_indices = list(range(lo, hi))

    # Build per-frame EE signal lines when states are provided.
    ee_lines = []
    if states is not None and len(states) > 0:
        sdim = states.shape[-1]
        # Pre-compute a small context window so velocity at lo is well-defined
        for fi in frame_indices:
            pos_str = "?"
            vel_str = "?"
            if 0 <= fi < len(states):
                pose = states[fi].tolist()
                pos_str = f"x={pose[0]:+.3f} y={pose[1]:+.3f} z={pose[2]:+.3f}"
                if sdim >= 6:
                    pos_str += f" rx={pose[3]:+.2f} ry={pose[4]:+.2f} rz={pose[5]:+.2f}"
                # Velocity as delta over a 3-frame centered window (more stable than single-step)
                left = max(0, fi - 1)
                right = min(len(states) - 1, fi + 1)
                dv = (states[right] - states[left]) / max(right - left, 1)
                vel_str = f"dx={dv[0].item():+.3f} dy={dv[1].item():+.3f} dz={dv[2].item():+.3f}"
            ee_lines.append((fi, f"EE {pos_str} | Δ/frame {vel_str}"))

    prompt = f"""You are pinpointing the exact transition frame between two primitives in a robot demo.

Task: "{task_description}"
Previous primitive (before): "{prev_label}"
Next primitive (after): "{next_label}"

Each frame below provides (a) camera view(s) and (b) the end-effector pose plus per-frame delta (Δdx, Δdy, Δdz, Δrx, Δry, Δrz) in world coordinates.

A primitive boundary is, by definition, a point where the character of the robot's motion changes. Examples of motion-character changes (task-agnostic):
  - A component of velocity that was sustained approaches zero, or reverses sign.
  - The dominant axis of motion shifts from one to another (e.g. translation-dominant → rotation-dominant, or one translation axis → another).
  - A velocity magnitude rises or drops substantially after being steady.
  - An axis that had zero motion starts moving.

Procedure — THINK EXPLICITLY IN THIS ORDER:
1. Read the per-frame EE deltas across all shown frames. Identify the frame where the MOTION CHARACTER first changes — this is your EE-based boundary candidate. State which axis/quantity changed and at what frame.
2. Look at the visual frames. Identify the earliest frame that a person would confidently describe as the next primitive (not still ambiguous with the previous one).
3. Reconcile the two candidates. If they agree within ±2 frames, pick the EE-based frame (physical motion is precise). If they disagree by more, pick the frame where the EE-based change begins AND the visual matches — explain the disagreement in reasoning.

The boundary frame IS the first frame of the new primitive — at that frame the robot's motion is characteristically the new primitive.

## Output format — IMPORTANT
Output the JSON FIRST, then brief reasoning. This guarantees a parseable answer even if truncated.

```json
{{"boundary_frame": <int>, "reasoning": "<reference specific frame numbers, the EE axis/quantity that changed, and the visual confirmation>"}}
```
"""

    cam_labels = ["primary"] + [k for k in (extra_image_keys or [])]
    content = [{"type": "text", "text": prompt + f"\n\nFrames provided cover range [{lo}, {hi})."}]
    for idx, fi in enumerate(frame_indices):
        src = dataset[episode_start + fi]
        ee_line = ee_lines[idx][1] if ee_lines else ""
        for ci, key in enumerate(cam_keys):
            caption = f"Frame {fi} ({cam_labels[ci]})"
            if ee_line and ci == 0:
                caption += f" | {ee_line}"
            content.append({"type": "text", "text": caption + ":"})
            b64 = _encode_frame_b64(_frame_to_hwc_uint8(src, key))
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=16384,
        temperature=0.1,
    )
    resp = response.choices[0].message.content or ""
    logging.info(f"    [refine {prev_label!r}→{next_label!r}] VLM (first 800 chars): {resp[:800]}")

    try:
        result = _extract_json(resp)
    except Exception as e:
        logging.warning(f"    Refine JSON parse failed: {e}. Keeping original boundary {lo}.")
        return lo

    raw = result.get("boundary_frame")
    if raw is None:
        logging.warning(f"    No boundary_frame in response. Keeping lo={lo}.")
        return lo
    try:
        boundary = int(raw)
    except (TypeError, ValueError):
        logging.warning(f"    Non-integer boundary_frame={raw!r}. Keeping lo={lo}.")
        return lo
    # Clamp to window
    boundary = max(lo + 1, min(hi - 1, boundary))
    return boundary


def assign_plan_to_frames(
    client: OpenAI,
    dataset,
    episode_start: int,
    episode_length: int,
    task_description: str,
    plan: List[str],
    frame_subsample: int = 5,
    model: str = "google/gemini-3-flash-preview",
    image_key: str = "image",
    extra_image_keys: List[str] = None,
    refine_boundaries: bool = True,
    refine_window: int = 15,
    max_retries: int = 2,
    states: torch.Tensor = None,
) -> List[Dict]:
    """Two-pass VLM labeling:
      Pass 1 (coarse): localize each plan primitive to rough frame ranges using subsampled video.
      Pass 2 (refine, optional): for each adjacent boundary, send CONSECUTIVE frames in a small
        window around the rough boundary and ask the VLM to pick the exact transition frame.
    """
    cam_keys = [image_key] + list(extra_image_keys or [])

    # Pass 1: coarse keyframe-based localization
    frame_indices = list(range(0, episode_length, frame_subsample))
    frames_b64 = []  # list[list[str]]
    for fi in frame_indices:
        src = dataset[episode_start + fi]
        frames_b64.append([_encode_frame_b64(_frame_to_hwc_uint8(src, k)) for k in cam_keys])

    plan_str = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(plan))

    prompt = f"""You are localizing primitives in a robot manipulation video.

Task: "{task_description}"

## Plan (fixed, in order; each primitive appears exactly ONCE):
{plan_str}

## Video
{episode_length} total frames. Frames shown below are sampled every {frame_subsample}th frame, in order.
The image labeled "Frame N" corresponds to ORIGINAL frame N.

## What to do
1. For each primitive in the plan, briefly state TWO signatures you will look for:
   (a) Visual signature — what the end-effector is doing, what contact or direction characterizes it.
   (b) Motion signature on the per-frame Δ data — which translation/rotation axis dominates and in which direction. Derive this yourself from the primitive's name and task context (e.g. "lower" implies negative Δdz dominant; "move to X" implies |Δxy| dominant; "twist" implies |Δrz| dominant; "open/close gripper" implies no EE motion). DO NOT rely on a fixed table — reason from the primitive's name.
2. Walk through the sampled frames in order. For each, decide which primitive it best matches, cross-checking BOTH the image and the Δ-signature in the caption.
3. Output the boundary ORIGINAL frame numbers between primitives.

## Boundary rule (read carefully)
The boundary frame is the first frame that clearly belongs to the NEW primitive — i.e., if someone saw ONLY that frame, they would confidently assign it to the new primitive, not the previous one. A boundary is NOT where the new primitive is already well underway (too late) and NOT where it is barely visible (too early). Pick the earliest unambiguous frame.

The boundary should align with a SHIFT IN THE DOMINANT MOTION AXIS in the per-frame Δ data — the axis that was sustained during primitive A approaches zero or reverses sign, and the axis you expect for primitive B starts to dominate. Use this as a kinematic anchor; do not place a boundary in a region where the dominant axis still matches A's expected signature.

## Constraints
- Exactly {len(plan)} segments, one per plan primitive, in order.
- Segments must be contiguous (no gaps/overlaps).
- First segment starts at 0, last ends at {episode_length}.
- Boundary frames must be ORIGINAL frame numbers from the sampled set (multiples of {frame_subsample}, plus 0 and {episode_length}).

## Output format — IMPORTANT
Output the JSON block FIRST, then any reasoning after. This guarantees a parseable answer even if your response is truncated.

```json
{{"segments": [{{"start_frame": 0, "end_frame": N, "primitive_label": "..."}}, ...]}}
```
After the JSON, you may add brief reasoning (1–3 sentences per boundary)."""

    # Per-keyframe EE summary (position + per-frame delta + derived magnitudes
    # and dominant-axis tag) when states are available. The derived scalars let
    # the VLM apply its reasoning directly to a single readable label per frame
    # instead of mentally combining six raw delta components.
    def _ee_caption(fi: int) -> str:
        if states is None or len(states) == 0 or fi < 0 or fi >= len(states):
            return ""
        pose = states[fi].tolist()
        sdim = states.shape[-1]
        s = f" | EE x={pose[0]:+.3f} y={pose[1]:+.3f} z={pose[2]:+.3f}"
        if sdim >= 6:
            s += f" rx={pose[3]:+.2f} ry={pose[4]:+.2f} rz={pose[5]:+.2f}"
        left = max(0, fi - 1)
        right = min(len(states) - 1, fi + 1)
        dv = (states[right] - states[left]) / max(right - left, 1)
        dx, dy, dz = dv[0].item(), dv[1].item(), dv[2].item()
        s += f" | Δ dx={dx:+.3f} dy={dy:+.3f} dz={dz:+.3f}"
        # Derived translation magnitudes for quick reading.
        xy_mag = (dx * dx + dy * dy) ** 0.5
        z_mag = abs(dz)
        s += f" |Δxy|={xy_mag:.3f} |Δz|={z_mag:.3f}"
        # Per-axis rotation magnitudes when state has rpy.
        if sdim >= 6:
            drx, dry, drz = dv[3].item(), dv[4].item(), dv[5].item()
            rxy_mag = (drx * drx + dry * dry) ** 0.5
            rz_mag = abs(drz)
            s += f" |Δrxy|={rxy_mag:.2f} |Δrz|={rz_mag:.2f}"
        else:
            rxy_mag = rz_mag = 0.0
        # Dominant axis tag — picks among {+z, -z, xy, +rz, -rz, rxy, none}.
        # The threshold on "none" filters near-zero motion frames (e.g. settling).
        axes = {
            "+z" if dz > 0 else "-z": z_mag,
            "xy": xy_mag,
            "+rz" if (sdim >= 6 and drz > 0) else "-rz": rz_mag if sdim >= 6 else 0.0,
            "rxy": rxy_mag,
        }
        dominant, mag = max(axes.items(), key=lambda kv: kv[1])
        if mag < 1e-3:  # below this threshold, motion is negligible
            dominant = "none"
        s += f" dom={dominant}"
        return s

    extra_note = ""
    if states is not None:
        extra_note = (
            "\n\nEach keyframe caption below includes the EE (end-effector) pose, "
            "per-frame delta, derived magnitudes (|Δxy|, |Δz|, |Δrxy|, |Δrz|), and "
            "a `dom=` tag identifying the dominant motion axis among "
            "{+z, -z, xy, +rz, -rz, rxy, none}. Use these alongside the images — "
            "they are precise and unambiguous, while images can look alike across "
            "primitives. A clear shift in `dom` between consecutive frames is a "
            "strong boundary cue."
        )

    cam_labels = ["primary"] + [k for k in (extra_image_keys or [])]
    content = [{"type": "text", "text": prompt + extra_note + "\n\nEach original frame is followed by its companion camera view(s): " + ", ".join(cam_labels) + "."}]
    for i, cam_imgs in enumerate(frames_b64):
        orig_fi = i * frame_subsample
        ee_str = _ee_caption(orig_fi)
        for ci, b64 in enumerate(cam_imgs):
            caption = f"Frame {orig_fi} ({cam_labels[ci]})"
            if ee_str and ci == 0:
                caption += ee_str
            content.append({"type": "text", "text": caption + ":"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    # Retry if output count doesn't match plan
    segments = []
    for attempt in range(max_retries + 1):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=32768,  # high — Gemini "thinking" tokens count against this
            temperature=0.1,
        )
        resp = response.choices[0].message.content or ""
        logging.info(f"  [coarse pass, attempt {attempt+1}] VLM response (first 1500 chars): {resp[:1500]}")
        if not resp:
            finish = response.choices[0].finish_reason if response.choices else "?"
            logging.warning(f"  Empty response (finish_reason={finish}). Retrying.")
            continue
        try:
            result = _extract_json(resp)
            segments = result.get("segments", [])
        except Exception as e:
            logging.warning(f"  Coarse pass parse failed: {e}. Retrying.")
            segments = []
        if len(segments) == len(plan):
            break
        logging.warning(f"  Coarse pass returned {len(segments)} segments but plan has {len(plan)}. Retry {attempt+1}/{max_retries}.")

    if not segments:
        logging.error("  Coarse pass failed after retries — returning empty segmentation.")
        return []

    # Clamp: force count, contiguity, plan-label-order.
    if len(segments) != len(plan):
        # Distribute evenly as a sane fallback
        step = episode_length / len(plan)
        segments = [{"start_frame": int(i * step), "end_frame": int((i + 1) * step), "primitive_label": plan[i]}
                    for i in range(len(plan))]
        logging.warning("  Using even-split fallback for coarse segmentation.")
    else:
        segments[0]["start_frame"] = 0
        segments[-1]["end_frame"] = episode_length
        for i in range(1, len(segments)):
            segments[i]["start_frame"] = segments[i - 1]["end_frame"]
        for i, seg in enumerate(segments):
            seg["primitive_label"] = plan[i]

    # Pass 2: per-boundary refinement via state-trajectory changepoint within a local window.
    # VLM's coarse boundary defines the region of interest; state-variance picks the exact frame.
    # Physics is precise where VLMs are not.
    if refine_boundaries and len(segments) > 1 and states is not None:
        logging.info(f"  Refining {len(segments) - 1} boundaries via state changepoints (±{refine_window} frames)...")
        # Need actions argument for signature compatibility; we pass states directly via feature_tensor
        actions_dummy = torch.zeros((episode_length, 6))  # unused when feature_tensor is provided
        for i in range(len(segments) - 1):
            coarse_b = segments[i]["end_frame"]
            prev_start = segments[i]["start_frame"]
            next_end = segments[i + 1]["end_frame"]
            lo = max(prev_start + 1, coarse_b - refine_window)
            hi = min(next_end - 1, coarse_b + refine_window)
            if hi - lo < 4:
                continue
            splits = find_action_changepoints(
                actions_dummy, lo, hi, n_splits=1,
                window=3, normalize_features=True, backtrack="none",
                feature_tensor=states,
            )
            if splits:
                refined = splits[0]
                delta = refined - coarse_b
                logging.info(f"    Boundary {i} {segments[i]['primitive_label']!r}→{segments[i+1]['primitive_label']!r}: {coarse_b} → {refined} (Δ{delta:+d})")
                segments[i]["end_frame"] = refined
                segments[i + 1]["start_frame"] = refined
            else:
                logging.info(f"    Boundary {i}: no state changepoint found in [{lo}, {hi}); keeping coarse {coarse_b}")

    return segments


def decompose_task_to_primitives(
    client: OpenAI,
    task_description: str,
    known_primitives: List[str],
    keyframes: List[np.ndarray] = None,
    extra_keyframes: List[List[np.ndarray]] = None,
    has_gripper: bool = True,
) -> Tuple[List[str], List[str]]:
    """Use VLM to decompose high-level task into primitive sequence.

    Args:
        client: OpenAI client
        task_description: The task to decompose
        known_primitives: List of known primitive types to prefer
        keyframes: Optional HWC uint8 frames from a representative demo.
            When provided, the VLM grounds its plan in the visible motion
            (useful for non-standard tasks where the text alone would trigger
            a wrong prior — e.g. scooping getting "lower gripper" leaks).

    Returns:
        (primitives, new_primitive_types) - the sequence and any new types discovered
    """
    # Format known primitives for prompt
    known_str = "\n".join(f"  - {p}" for p in known_primitives)

    # Hardware flag — conditioned on whether the robot has a controllable gripper.
    hardware_note = ""
    if not has_gripper:
        hardware_note = (
            "\n\n## Hardware constraint\n"
            "This robot has no controllable gripper DoF. Any primitive that depends on gripper state "
            "(close, open, grasp, release) is not physically available. Positioning the end-effector "
            "is a single primitive (approach is not separable from lowering)."
        )

    prompt = f"""Decompose this robot manipulation task into atomic motion primitives.

Task: "{task_description}"

## Known primitive types (use these when applicable):
{known_str}{hardware_note}

## Rules:
1. Decompose only the motions the task explicitly describes.
2. Use known primitive types when applicable. Create new ones only if
   the task requires a motion not covered by existing primitives.
3. Be specific with object names (e.g., "the white mug" not "the mug").
4. Compound color descriptions like "yellow and white mug" refer to
   ONE object with multiple colors, not two separate objects.

## Examples:

Task: "pick up the red block and place it on the plate"
Primitives: ["move gripper to the red block", "close gripper", "lift upward", "move gripper to the plate", "lower gripper", "open gripper"]

Task: "put the black bowl in the bottom drawer and close it"
Primitives: ["move gripper to the black bowl", "close gripper", "lift upward", "move gripper to the bottom drawer", "lower gripper", "open gripper", "move gripper to the bottom drawer", "push the drawer closed"]

Task: "put the yellow and white mug in the microwave and close it"
Primitives: ["move gripper to the yellow and white mug", "close gripper", "lift upward", "move gripper to the microwave", "lower gripper", "open gripper", "move gripper to the microwave door", "push the microwave door closed"]

Task: "turn on the stove and put the pot on it"
Primitives: ["move gripper to the stove knob", "close gripper", "turn the stove knob on", "open gripper", "move gripper to the pot", "close gripper", "lift upward", "move gripper to the stove", "lower gripper", "open gripper"]

Task: "flip the red lego block peg up"
Primitives: ["move gripper to the red lego block", "close gripper", "lift upward", "rotate block", "open gripper"]

Return JSON: {{"primitives": [...], "new_primitive_types": [...]}}
- primitives: the full sequence for this task
- new_primitive_types: any primitive TYPES you used that aren't in the known list (e.g., "push [object] closed", "turn [object] [direction]")"""

    # Build message content — text only, or text + keyframes from a real demo.
    if keyframes:
        import base64
        from io import BytesIO
        from PIL import Image as PILImage

        def _enc(arr):
            p = PILImage.fromarray(arr)
            buf = BytesIO()
            p.save(buf, format="JPEG", quality=80)
            return base64.b64encode(buf.getvalue()).decode()

        content_msg = [{
            "type": "text",
            "text": prompt + "\n\n## Reference demo frames\nHere are keyframes from one real demonstration of this task, in time order. Use them to ground your decomposition in the motion that actually happens (don't hallucinate separate sub-steps that are really one continuous motion in the demo).",
        }]
        for i, img in enumerate(keyframes):
            b64 = _enc(img)
            content_msg.append({"type": "text", "text": f"Frame {i + 1}/{len(keyframes)} (primary):"})
            content_msg.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            if extra_keyframes:
                # extra_keyframes is parallel list-of-lists: [[cam0_frame_i, cam1_frame_i, ...], ...]
                for ci, extra_img in enumerate(extra_keyframes[i] if i < len(extra_keyframes) else []):
                    b64x = _enc(extra_img)
                    content_msg.append({"type": "text", "text": f"Frame {i + 1}/{len(keyframes)} (extra cam {ci}):"})
                    content_msg.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64x}"}})
    else:
        content_msg = prompt

    response = client.chat.completions.create(
        model="google/gemini-3-flash-preview",
        messages=[{
            "role": "user",
            "content": content_msg,
        }],
        max_tokens=8192,
        temperature=0.1
    )

    content = response.choices[0].message.content
    if content is None:
        content = ""
    content = content.strip()
    logging.info(f"  VLM response: {content[:200]}")

    # Try to extract JSON even if there's extra text (e.g. markdown code blocks)
    import re
    # Strip markdown code blocks if present
    content = re.sub(r'^```(?:json)?\s*', '', content)
    content = re.sub(r'\s*```$', '', content)
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
        result = json.loads(json_match.group())
    else:
        result = json.loads(content)

    primitives = result.get("primitives", [])
    new_primitive_types = result.get("new_primitive_types", [])

    # Fallback if parsing failed
    if not primitives:
        logging.warning(f"Failed to parse primitives, using generic sequence")
        primitives = [
            "move gripper to the object", "close gripper", "lift upward",
            "move gripper to the target", "lower gripper", "open gripper"
        ]
        new_primitive_types = []

    return primitives, new_primitive_types


def compute_action_summary(segment_actions: torch.Tensor) -> Dict:
    """Compute summary statistics for actions in a segment.

    Args:
        segment_actions: (num_frames, 7) tensor of actions
            [dx, dy, dz, drx, dry, drz, gripper]
            Gripper: -1 = closed, +1 = open (binary in LIBERO)

    Returns:
        Dictionary with action statistics for VLM interpretation.
    """
    actions = segment_actions.numpy()

    # Position changes (cumulative motion)
    total_dx = actions[:, 0].sum()
    total_dy = actions[:, 1].sum()
    total_dz = actions[:, 2].sum()

    # Gripper state - LIBERO uses binary -1 (closed) / +1 (open)
    gripper_start = actions[0, 6]
    gripper_end = actions[-1, 6]
    gripper_change = gripper_end - gripper_start

    # Detect gripper action based on state, not delta
    # LIBERO convention: -1 = open, +1 = closed (grasping)
    if gripper_start < 0 and gripper_end > 0:
        gripper_action = "closing"  # -1 -> +1 means closing to grasp
    elif gripper_start > 0 and gripper_end < 0:
        gripper_action = "opening"  # +1 -> -1 means opening to release
    elif gripper_end > 0:
        gripper_action = "closed"
    else:
        gripper_action = "open"

    # Determine dominant motion
    abs_motions = [abs(total_dx), abs(total_dy), abs(total_dz)]
    dominant_axis = ['X (left/right)', 'Y (forward/back)', 'Z (up/down)'][np.argmax(abs_motions)]

    # Use higher threshold for vertical motion (cumulative over segment)
    if total_dz > 0.3:
        vertical = "upward"
    elif total_dz < -0.3:
        vertical = "downward"
    else:
        vertical = "level"

    return {
        "total_motion": {
            "dx": float(total_dx),
            "dy": float(total_dy),
            "dz": float(total_dz),
        },
        "dominant_axis": dominant_axis,
        "gripper": {
            "start": float(gripper_start),
            "end": float(gripper_end),
            "change": float(gripper_change),
            "action": gripper_action,
        },
        "vertical_motion": vertical,
    }


def get_next_move_primitive(
    primitive_sequence: List[str],
    last_primitive_idx: int = 0,
) -> Tuple[str, int]:
    """Get the next 'move gripper to...' primitive from the sequence.

    Returns:
        (primitive_label, new_primitive_idx) - the label and updated index into sequence
    """
    # Find the next "move gripper to..." primitive in sequence
    for i in range(last_primitive_idx, len(primitive_sequence)):
        if primitive_sequence[i].startswith("move gripper to"):
            return primitive_sequence[i], i + 1

    # Fallback: if no more move primitives, use generic
    return "move gripper to target", last_primitive_idx


def get_last_move_primitive(
    primitive_sequence: List[str],
    current_idx: int,
) -> str:
    """Get the most recent 'move gripper to...' primitive before current index.

    Used after a retry to figure out what object we were trying to grasp.
    """
    for i in range(current_idx - 1, -1, -1):
        if primitive_sequence[i].startswith("move gripper to"):
            return primitive_sequence[i]
    return "move gripper to target"


def check_sequence_mismatch(
    expected_primitive: str,
    actual_segment_type: str,
    labels_so_far: List[str],
) -> Tuple[bool, str]:
    """Check if the actual segment type matches what we expected.

    Returns:
        (is_mismatch, mismatch_type)
        mismatch_type can be: "grasp_failed", "unexpected_open", "unexpected_close", etc.
    """
    expected_lower = expected_primitive.lower() if expected_primitive else ""

    # Expected "lift upward" but got "open_gripper" = failed grasp
    if "lift" in expected_lower and actual_segment_type == "open_gripper":
        return True, "grasp_failed"

    # Expected "lower gripper" but got "open_gripper" early = dropped object?
    if "lower" in expected_lower and actual_segment_type == "open_gripper":
        return True, "dropped_early"

    # Check recent history for close→open without lift (failed grasp pattern)
    if len(labels_so_far) >= 2:
        if ("close gripper" in labels_so_far[-1].lower() and
            actual_segment_type == "open_gripper" and
            not any("lift" in l.lower() for l in labels_so_far[-2:])):
            return True, "grasp_failed"

    return False, None


def _create_lerobot_dataset(args: Args, dataset: LeRobotDataset, all_labels: List[Dict], dataset_repo_id: str = None) -> None:
    """Create LeRobot dataset from existing labels."""
    import gc

    logging.info("\n" + "=" * 80)
    logging.info("Creating LeRobot Dataset with Primitive Labels")
    logging.info("=" * 80)

    # Clean up any existing dataset (unless resuming)
    lerobot_output_path = HF_LEROBOT_HOME / args.lerobot_repo_name
    if lerobot_output_path.exists() and args.start_from_episode == 0:
        logging.info(f"Removing existing dataset at {lerobot_output_path}")
        shutil.rmtree(lerobot_output_path)
    elif lerobot_output_path.exists():
        logging.info(f"Resuming - keeping existing dataset at {lerobot_output_path}")

    # Determine output image keys (mirror source schema). Default preserves LIBERO behavior.
    if args.image_keys:
        output_image_keys = [k.strip() for k in args.image_keys.split(",") if k.strip()]
    else:
        output_image_keys = ["image", "wrist_image"]

    # Source shapes — introspect to support non-LIBERO datasets (e.g. xArm)
    src_probe = dataset[0]
    src_state_dim = int(src_probe["state"].shape[-1]) if "state" in src_probe else 8
    src_action_dim = int(src_probe["actions"].shape[-1])
    output_action_dim = src_action_dim + 1  # +1 for progress (0→1 within primitive)

    # Probe image shapes per-key — avoids hardcoding LIBERO's 256x256 for xArm 240x320, etc.
    image_shapes = {}
    for key in output_image_keys:
        img = _frame_to_hwc_uint8(src_probe, key)
        image_shapes[key] = tuple(img.shape)  # (H, W, C)

    features_dict = {}
    for key in output_image_keys:
        features_dict[key] = {
            "dtype": "image",
            "shape": image_shapes[key],
            "names": ["height", "width", "channel"],
        }
    features_dict["state"] = {
        "dtype": "float32",
        "shape": (src_state_dim,),
        "names": ["state"],
    }
    features_dict["actions"] = {
        "dtype": "float32",
        "shape": (output_action_dim,),  # source action dims + 1 progress
        "names": ["actions"],
    }

    # Create new LeRobot dataset. Inherit fps from source so output timestamps
    # match the actual frame rate (xarm: 20fps; libero default: 10fps).
    src_fps = int(dataset.meta.info.get("fps", 10)) if hasattr(dataset, "meta") else 10
    logging.info(f"Creating new dataset: {args.lerobot_repo_name}")
    logging.info(f"  Image keys: {output_image_keys}  state dim: {src_state_dim}  action dim: {output_action_dim} (= {src_action_dim} + progress)  fps: {src_fps}")
    lerobot_dataset = LeRobotDataset.create(
        repo_id=args.lerobot_repo_name,
        robot_type="panda",
        fps=src_fps,
        features=features_dict,
        image_writer_threads=1,
        image_writer_processes=0,
    )

    # Process each episode
    total_lerobot_frames = 0
    total_lerobot_segments = 0
    total_original_frames = 0

    # Track total episodes processed across both phases for reload logic
    global_episode_counter = 0

    # First, add original task-labeled episodes if requested
    if args.include_original_episodes:
        logging.info("\nAdding original task-labeled episodes...")
        for ep_idx, ep_data in enumerate(all_labels):
            episode_idx = ep_data["episode_idx"]
            task_description = ep_data["task"]

            episode_start = dataset.episode_data_index['from'][episode_idx].item()
            episode_end = dataset.episode_data_index['to'][episode_idx].item()

            logging.info(f"  Adding original episode {episode_idx}: {task_description[:50]}...")

            ep_length = episode_end - episode_start
            for source_idx in range(episode_start, episode_end):
                source_frame = dataset[source_idx]
                # Progress for original episodes: 0→1 over whole episode
                progress = (source_idx - episode_start) / max(ep_length - 1, 1)
                actions_src = source_frame["actions"].numpy()[:src_action_dim]
                actions_with_progress = np.append(actions_src, progress).astype(np.float32)
                frame_out = {
                    "state": source_frame["state"].numpy(),
                    "actions": actions_with_progress,
                    "task": task_description,
                }
                for key in output_image_keys:
                    frame_out[key] = _frame_to_hwc_uint8(source_frame, key)
                lerobot_dataset.add_frame(frame_out)
                total_original_frames += 1
                del source_frame, frame_out

            lerobot_dataset.save_episode()
            # Clear the episode buffer to free memory after saving
            lerobot_dataset.clear_episode_buffer()
            global_episode_counter += 1

            # Free memory every 10 episodes
            if global_episode_counter % 10 == 0:
                gc.collect()
                logging.info(f"    [GC at global episode {global_episode_counter}]")

            # Reload source dataset every 200 episodes to clear HF dataset cache
            if global_episode_counter > 0 and global_episode_counter % 200 == 0 and dataset_repo_id:
                logging.info(f"    [Reloading source dataset at global episode {global_episode_counter} to free memory]")
                del dataset
                gc.collect()
                dataset = LeRobotDataset(dataset_repo_id)

        logging.info(f"  Added {len(all_labels)} original episodes ({total_original_frames} frames)")

    # Force garbage collection and reload source dataset before primitives phase
    logging.info("\n  [Clearing memory before primitive episodes phase]")
    del dataset
    gc.collect()
    dataset = LeRobotDataset(dataset_repo_id)

    # Then add primitive-labeled episodes
    logging.info("\nAdding primitive-labeled episodes...")
    if args.start_from_episode > 0:
        logging.info(f"  Starting from episode {args.start_from_episode}")
    if args.end_at_episode > 0:
        logging.info(f"  Ending at episode {args.end_at_episode}")
    for ep_idx, ep_data in enumerate(all_labels):
        # Skip episodes before start_from_episode
        if ep_idx < args.start_from_episode:
            continue
        # Stop at end_at_episode
        if args.end_at_episode > 0 and ep_idx >= args.end_at_episode:
            break

        episode_idx = ep_data["episode_idx"]
        task_description = ep_data["task"]
        segments = ep_data["segments"]

        logging.info(f"  Adding primitive episode {ep_idx}/{len(all_labels)} (source ep {episode_idx}): {len(segments)} segments")

        # Get source episode frame indices
        episode_start = dataset.episode_data_index['from'][episode_idx].item()

        # Process each segment as a SEPARATE episode
        for seg in segments:
            start_frame = seg["start_frame"]
            end_frame = seg["end_frame"]
            primitive_label = seg["primitive_label"]

            # Create the prompt (task for this segment)
            if args.include_task_prefix:
                prompt = f"{task_description}: {primitive_label}"
            else:
                prompt = primitive_label

            # Add frames for this segment
            seg_length = end_frame - start_frame
            for frame_offset in range(start_frame, end_frame):
                source_idx = episode_start + frame_offset
                source_frame = dataset[source_idx]

                # Compute progress within this primitive (0→1)
                progress = (frame_offset - start_frame) / max(seg_length - 1, 1)

                # Append progress as last action dimension (take first src_action_dim in case source already has progress)
                actions_src = source_frame["actions"].numpy()[:src_action_dim]
                actions_with_progress = np.append(actions_src, progress).astype(np.float32)
                frame_out = {
                    "state": source_frame["state"].numpy(),
                    "actions": actions_with_progress,
                    "task": prompt,
                }
                for key in output_image_keys:
                    frame_out[key] = _frame_to_hwc_uint8(source_frame, key)
                lerobot_dataset.add_frame(frame_out)
                del source_frame, frame_out

                total_lerobot_frames += 1

            total_lerobot_segments += 1

            # Save each segment as a separate episode
            lerobot_dataset.save_episode()
            lerobot_dataset.clear_episode_buffer()
            global_episode_counter += 1

            # Free memory every 50 segments
            if global_episode_counter % 50 == 0:
                gc.collect()
                logging.info(f"    [GC at segment {global_episode_counter}]")

        # Reload source dataset every 20 source episodes to clear HF dataset cache
        if ep_idx > 0 and ep_idx % 20 == 0 and dataset_repo_id:
            logging.info(f"    [Reloading source dataset at source episode {ep_idx} to free memory]")
            del dataset
            gc.collect()
            dataset = LeRobotDataset(dataset_repo_id)

    logging.info(f"\nLeRobot dataset created!")
    logging.info(f"  Original episodes: {len(all_labels) if args.include_original_episodes else 0} ({total_original_frames} frames)")
    logging.info(f"  Primitive segments (as episodes): {total_lerobot_segments} ({total_lerobot_frames} frames)")
    total_eps = (len(all_labels) if args.include_original_episodes else 0) + total_lerobot_segments
    logging.info(f"  Total episodes: {total_eps}")
    logging.info(f"  Total frames: {total_original_frames + total_lerobot_frames}")
    logging.info(f"  Output path: {lerobot_output_path}")
    logging.info(f"LeRobot dataset saved to: {lerobot_output_path}")


def densely_label_dataset(args: Args) -> None:
    """Main function to densely label LIBERO dataset."""

    logging.info("=" * 80)
    logging.info("Dense Labeling of LIBERO Dataset")
    logging.info("=" * 80)

    # Load dataset
    logging.info(f"\nLoading dataset: {args.dataset_repo_id}")
    dataset = LeRobotDataset(args.dataset_repo_id)
    logging.info(f"  Episodes: {dataset.num_episodes}")
    logging.info(f"  Frames: {dataset.num_frames}")

    # If loading from existing labels, skip to LeRobot dataset creation
    if args.load_labels_from:
        logging.info(f"\nLoading existing labels from: {args.load_labels_from}")
        with open(args.load_labels_from, 'r') as f:
            all_labels = json.load(f)
        logging.info(f"  Loaded {len(all_labels)} episodes")

        # Skip to LeRobot dataset creation
        if args.create_lerobot_dataset:
            _create_lerobot_dataset(args, dataset, all_labels, dataset_repo_id=args.dataset_repo_id)
        else:
            logging.warning("No action taken. Use --args.create-lerobot-dataset to create dataset from labels.")
        return

    # Create timestamped output directory
    from datetime import datetime
    now = datetime.now()
    date_dir = now.strftime("%Y-%m-%d")
    time_dir = now.strftime("%Y-%m-%d_%H%M%S")
    if args.run_name:
        time_dir = f"{time_dir}_{args.run_name}"
    output_dir = pathlib.Path(args.output_base_dir) / date_dir / time_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Output directory: {output_dir}")

    # Create videos subdirectory
    if args.save_videos:
        videos_dir = output_dir / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)

    # Save run configuration + invocation metadata
    import sys, socket, subprocess
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=pathlib.Path(__file__).parent, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_sha = None
    metadata = {
        "args": dataclasses.asdict(args),
        "invocation": {
            "command": " ".join(sys.argv),
            "argv": sys.argv,
            "cwd": str(pathlib.Path.cwd()),
            "timestamp": now.isoformat(),
            "host": socket.gethostname(),
            "python": sys.version.split()[0],
            "git_sha": git_sha,
            "env": {
                "USE_VERTEX_AI": os.environ.get("USE_VERTEX_AI", ""),
                "VERTEX_PROJECT": os.environ.get("VERTEX_PROJECT", ""),
                "GEMINI_API_KEY_set": bool(os.environ.get("GEMINI_API_KEY", "")),
                "OPENAI_API_KEY_set": bool(os.environ.get("OPENAI_API_KEY", "")),
            },
        },
    }
    config_file = output_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    logging.info(f"Saved config + invocation metadata to: {config_file}")

    # Initialize VLM client (Vertex AI via gcloud, Gemini AI Studio, or OpenAI)
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    use_vertex = os.environ.get("USE_VERTEX_AI", "")
    if use_vertex:
        import google.auth
        import google.auth.transport.requests
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        creds.refresh(google.auth.transport.requests.Request())
        project = os.environ.get("VERTEX_PROJECT", "gcp-maggie")
        client = OpenAI(
            api_key=creds.token,
            base_url=f"https://aiplatform.googleapis.com/v1beta1/projects/{project}/locations/global/endpoints/openapi",
        )
        logging.info(f"Using Vertex AI (project: {project})")
    elif gemini_key:
        client = OpenAI(
            api_key=gemini_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        logging.info("Using Gemini AI Studio")
    else:
        client = OpenAI()
        logging.info("Using OpenAI API")

    # Initialize primitive vocabulary (starts fresh each run)
    known_primitives = [
        "move gripper to the [object/location]",
        "close gripper",
        "open gripper",
        "lift upward",
        "lower gripper",
        "rotate block",
    ]
    all_discovered_primitives = []  # Track new primitive types discovered this run
    _task_sequence_cache = {}  # Cache decomposed sequences per task (avoids repeated API calls)

    # Process episodes
    all_labels = []

    num_episodes = dataset.num_episodes if args.num_episodes == -1 else min(args.num_episodes, dataset.num_episodes)
    if args.episode_indices.strip():
        target_indices = [int(s.strip()) for s in args.episode_indices.split(",") if s.strip()]
        target_indices = [i for i in target_indices if 0 <= i < dataset.num_episodes]
        logging.info(f"Restricting to specific episodes: {target_indices}")
    else:
        target_indices = list(range(num_episodes))
    for episode_idx in target_indices:
        logging.info(f"\n{'=' * 80}")
        logging.info(f"Episode {episode_idx + 1}/{args.num_episodes}")
        logging.info(f"{'=' * 80}")

        # Get episode info
        episode_start = dataset.episode_data_index['from'][episode_idx].item()
        episode_end = dataset.episode_data_index['to'][episode_idx].item()
        episode_length = episode_end - episode_start

        first_frame = dataset[episode_start]
        task_description = first_frame['task']

        logging.info(f"Task: {task_description}")
        logging.info(f"Episode length: {episode_length} frames")

        # Load all actions and states for this episode
        actions = torch.stack([
            dataset[episode_start + i]['actions']
            for i in range(episode_length)
        ])
        states = torch.stack([
            dataset[episode_start + i]['state']
            for i in range(episode_length)
        ])

        # Segment episode
        effective_segment_method = args.segment_method
        if not args.has_gripper and effective_segment_method == "gripper":
            logging.info("  has_gripper=False → forcing segment_method='action_change'")
            effective_segment_method = "action_change"

        if effective_segment_method == "gripper":
            # Gripper-based segmentation uses actual gripper state, not just action command
            segments_with_type = segment_episode_by_gripper(
                actions,
                states,
                min_segment_length=args.min_segment_length,
                gripper_close_threshold=args.gripper_close_threshold,
            )
            segments = [(s[0], s[1]) for s in segments_with_type]
            segment_types = [s[2] for s in segments_with_type]
        elif effective_segment_method == "action_change":
            segments = segment_episode_by_action_change(
                actions,
                threshold=args.action_threshold,
                min_length=args.min_segment_length,
                max_length=args.max_segment_length
            )
            segment_types = [None] * len(segments)
        else:  # fixed_chunks
            segments = segment_episode_fixed_chunks(episode_length, args.chunk_size)
            segment_types = [None] * len(segments)

        logging.info(f"Segmented into {len(segments)} primitives")

        # Decompose task into expected primitive sequence (cached per task)
        logging.info("\n  Decomposing task into primitive sequence...")
        logging.info(f"  Known primitives: {known_primitives}")
        if task_description in _task_sequence_cache:
            primitive_sequence = _task_sequence_cache[task_description]
            logging.info(f"  Using cached sequence ({len(primitive_sequence)} primitives)")
        else:
            # Pull ~12 keyframes from this episode so the VLM grounds the plan
            # in the visible motion (avoids LIBERO-biased "lower gripper" leaks for scoop).
            demo_keyframes = extract_keyframes(
                dataset, episode_idx, 0, episode_length,
                num_keyframes=12, image_key=args.primary_image_key,
            )
            extra_keys_list = [k.strip() for k in args.extra_image_keys.split(",") if k.strip()] if args.extra_image_keys else []
            extra_demo_keyframes = None
            if extra_keys_list:
                # Parallel list-of-lists aligned with demo_keyframes indices.
                extra_demo_keyframes = []
                for k in extra_keys_list:
                    extra_demo_keyframes.append(extract_keyframes(
                        dataset, episode_idx, 0, episode_length,
                        num_keyframes=12, image_key=k,
                    ))
                # Transpose to [[cam0_f0, cam1_f0, ...], [cam0_f1, ...], ...]
                extra_demo_keyframes = [list(row) for row in zip(*extra_demo_keyframes)]
            primitive_sequence = []
            for attempt in range(3):
                try:
                    primitive_sequence, new_primitives = decompose_task_to_primitives(
                        client, task_description, known_primitives,
                        keyframes=demo_keyframes,
                        extra_keyframes=extra_demo_keyframes,
                        has_gripper=args.has_gripper,
                    )
                    if primitive_sequence:
                        _task_sequence_cache[task_description] = primitive_sequence
                        if new_primitives:
                            logging.info(f"  New primitive types discovered: {new_primitives}")
                            for new_prim in new_primitives:
                                if new_prim not in known_primitives:
                                    known_primitives.append(new_prim)
                                    all_discovered_primitives.append(new_prim)
                        break
                except Exception as e:
                    logging.error(f"  Attempt {attempt+1} failed: {e}")
                    if "insufficient_quota" in str(e) or "429" in str(e):
                        logging.error("API quota exceeded - stopping.")
                        raise SystemExit(1)
        if primitive_sequence:
            logging.info(f"  Expected primitives ({len(primitive_sequence)}):")
            for i, prim in enumerate(primitive_sequence):
                logging.info(f"    {i+1}. {prim}")

        # VLM-driven end-to-end labeling: VLM picks frame ranges per plan primitive from video.
        vlm_segments = None
        if args.use_vlm_labels and primitive_sequence and len(primitive_sequence) >= 1:
            extra_keys_list = [k.strip() for k in args.extra_image_keys.split(",") if k.strip()] if args.extra_image_keys else None
            try:
                vlm_segments = assign_plan_to_frames(
                    client, dataset, episode_start, episode_length,
                    task_description, primitive_sequence,
                    frame_subsample=args.vlm_label_frame_subsample,
                    image_key=args.primary_image_key,
                    extra_image_keys=extra_keys_list,
                    refine_boundaries=args.vlm_label_refine,
                    refine_window=args.vlm_label_refine_window,
                    states=states,
                )
            except Exception as e:
                logging.error(f"  VLM plan-assignment failed: {e} — falling back to changepoints")
                vlm_segments = None
            if vlm_segments:
                # Optional: snap boundaries to nearest state changepoint (sharpens visual estimates with physical signal)
                if args.vlm_label_snap_to_changepoint and len(vlm_segments) > 1:
                    feat_tensor = states if args.changepoint_feature == "state" else None
                    cps = find_action_changepoints(
                        actions, 0, episode_length,
                        n_splits=max(len(vlm_segments) - 1, 1),
                        window=args.changepoint_window,
                        normalize_features=args.changepoint_normalize,
                        backtrack="none",
                        feature_tensor=feat_tensor,
                    )
                    radius = args.vlm_label_snap_radius
                    for i in range(len(vlm_segments) - 1):
                        b = vlm_segments[i]["end_frame"]
                        nearby = [c for c in cps if abs(c - b) <= radius]
                        if nearby:
                            snapped = min(nearby, key=lambda c: abs(c - b))
                            logging.info(f"    Snap boundary {b} → {snapped} (±{radius})")
                            vlm_segments[i]["end_frame"] = snapped
                            vlm_segments[i + 1]["start_frame"] = snapped
                segments = [(s["start_frame"], s["end_frame"]) for s in vlm_segments]
                segment_types = [None] * len(segments)
                logging.info(f"  VLM plan-assignment: {len(segments)} segments")
                for s in vlm_segments:
                    logging.info(f"    [{s['start_frame']:4d}-{s['end_frame']:4d}] {s['primitive_label']}")

        # For gripperless datasets (without VLM labels), override the segmentation with plan-driven changepoints.
        if vlm_segments is None and not args.has_gripper and primitive_sequence:
            n_plan = len(primitive_sequence)
            if n_plan >= 1:
                feat_tensor = states if args.changepoint_feature == "state" else None
                split_points = find_action_changepoints(
                    actions, 0, episode_length, n_plan - 1,
                    window=args.changepoint_window,
                    normalize_features=args.changepoint_normalize,
                    backtrack=args.changepoint_backtrack,
                    feature_tensor=feat_tensor,
                ) if n_plan > 1 else []
                boundaries = [0] + sorted(split_points) + [episode_length]
                segments = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
                segment_types = [None] * len(segments)
                logging.info(f"  Plan-driven changepoints (feature={args.changepoint_feature}, window={args.changepoint_window}, normalize={args.changepoint_normalize}, backtrack={args.changepoint_backtrack}): {len(segments)} segments at {boundaries}")

        # Label each segment
        episode_labels = {
            "episode_idx": episode_idx,
            "task": task_description,
            "episode_length": episode_length,
            "expected_primitives": primitive_sequence,
            "segments": []
        }

        # Track which primitive we're on in the sequence
        current_primitive_idx = 0
        labels_so_far = []  # Track labels for mismatch detection
        in_retry = False  # Track if we're currently in a retry sequence
        last_object_target = None  # The object we were trying to grasp before retry

        for seg_idx, (start_idx, end_idx) in enumerate(segments):
            segment_type = segment_types[seg_idx]
            logging.info(f"\n  Segment {seg_idx + 1}/{len(segments)}: frames {start_idx}-{end_idx} ({end_idx - start_idx} frames) [type: {segment_type}]")

            # Get expected next primitive for mismatch checking
            expected_next = primitive_sequence[current_primitive_idx] if current_primitive_idx < len(primitive_sequence) else None

            # Compute action summary for logging (for all segments)
            segment_actions = actions[start_idx:end_idx]
            action_summary = compute_action_summary(segment_actions) if segment_type == "move" else None
            if action_summary:
                logging.info(f"    Action summary: {action_summary['vertical_motion']}, gripper {action_summary['gripper']['action']}")

            # For gripper segments, use the detected type directly
            if segment_type in ("close_gripper", "open_gripper"):
                gripper_action = segment_type.replace("_", " ")  # "close gripper" or "open gripper"

                # Check for mismatch (e.g., expected "lift" but got "open gripper")
                is_mismatch, mismatch_type = check_sequence_mismatch(
                    expected_next, segment_type, labels_so_far
                )

                if is_mismatch and mismatch_type == "grasp_failed":
                    # Failed grasp - mark as retry and remember what we were trying to grasp
                    # DON'T advance current_primitive_idx - we need to retry from same point
                    primitive_label = "open gripper"  # Still label it as open gripper
                    last_object_target = get_last_move_primitive(primitive_sequence, current_primitive_idx)
                    # Reset index back to the "close gripper" we just failed at
                    # so we can re-try the sequence: move -> close -> lift
                    for i in range(current_primitive_idx - 1, -1, -1):
                        if primitive_sequence[i].lower() == "close gripper":
                            current_primitive_idx = i  # Go back to close gripper
                            break
                    in_retry = True
                    logging.info(f"    RETRY DETECTED: grasp failed, will re-approach '{last_object_target}'")
                    logging.info(f"    Label: '{primitive_label}' (retry - resetting to primitive {current_primitive_idx + 1})")
                else:
                    # Normal case - find matching primitive in sequence
                    primitive_label = gripper_action  # Default fallback
                    for i in range(current_primitive_idx, len(primitive_sequence)):
                        if gripper_action == primitive_sequence[i].lower():
                            primitive_label = primitive_sequence[i]
                            current_primitive_idx = i + 1
                            in_retry = False  # Successfully completed gripper action, exit retry
                            logging.info(f"    Label from plan: '{primitive_label}' (primitive {i+1})")

                            # Check if this is a long gripper segment that contains a push/pull action
                            # (e.g., close gripper + push drawer in one segment with gripper staying closed)
                            seg_length = end_idx - start_idx
                            if seg_length > 20 and current_primitive_idx < len(primitive_sequence):
                                next_expected = primitive_sequence[current_primitive_idx]
                                next_lower = next_expected.lower()
                                # If next primitive is a push/pull/turn action, subdivide
                                if any(action in next_lower for action in ["push", "pull", "turn"]):
                                    logging.info(f"    Long gripper segment ({seg_length} frames) with upcoming '{next_expected}'")
                                    # Mark for subdivision
                                    episode_labels["_subdivide_gripper_hint"] = {
                                        "seg_idx": seg_idx,
                                        "next_primitive": next_expected,
                                        "current_primitive_idx": current_primitive_idx
                                    }
                            break
                    else:
                        logging.info(f"    Label fallback: '{primitive_label}' (no match in sequence)")
            else:
                # For movement segments, use the VLM-generated plan as primary source
                # The action data is for logging/validation only

                if in_retry and last_object_target:
                    # We're in a retry - re-approach the same object
                    primitive_label = last_object_target
                    logging.info(f"    Label from retry context: '{primitive_label}' (re-approaching after failed grasp)")
                else:
                    # Check if previous segment was also a move (continuation)
                    prev_was_move = (
                        len(labels_so_far) > 0 and
                        labels_so_far[-1].startswith("move gripper to") and
                        seg_idx > 0 and
                        segment_types[seg_idx - 1] == "move"
                    )

                    if prev_was_move:
                        # Check if next expected primitive is still the same type (move/action)
                        # Don't continue if we should be switching to a different primitive
                        next_expected = primitive_sequence[current_primitive_idx] if current_primitive_idx < len(primitive_sequence) else None
                        prev_label = labels_so_far[-1].lower()

                        # If next expected is different AND matches the pattern after current
                        # (e.g., prev="move to door", next="push door closed"), don't continue
                        should_continue = True
                        if next_expected:
                            next_lower = next_expected.lower()
                            # Don't continue if:
                            # 1. We've been moving to something and next is a different action type
                            is_move_primitive = "move gripper to" in prev_label
                            next_is_action = any(word in next_lower for word in ["push", "pull", "turn", "lower", "lift"])
                            if is_move_primitive and next_is_action and not next_lower.startswith("move"):
                                should_continue = False
                                logging.info(f"    Breaking continuation: next expected is '{next_expected}'")

                        if should_continue:
                            # Continuation of previous movement - use same label
                            primitive_label = labels_so_far[-1]
                            logging.info(f"    Label from continuation: '{primitive_label}' (same movement)")
                        else:
                            # Don't continue, use next primitive from plan
                            primitive_label = next_expected
                            current_primitive_idx += 1
                            logging.info(f"    Label from plan (breaking continuation): '{primitive_label}' (primitive {current_primitive_idx})")
                    else:
                        # Get the next expected primitive from the plan
                        if current_primitive_idx < len(primitive_sequence):
                            expected = primitive_sequence[current_primitive_idx]

                            primitive_label = expected
                            current_primitive_idx += 1
                            logging.info(f"    Label from plan: '{primitive_label}' (primitive {current_primitive_idx})")

                            # After a successful lift, clear the retry state
                            if expected.lower() == "lift upward":
                                in_retry = False

                            # Check if this is a long segment that might contain multiple primitives
                            # (e.g., "move to door" + "push door closed" in one segment)
                            seg_length = end_idx - start_idx
                            if seg_length > 50 and current_primitive_idx < len(primitive_sequence):
                                next_expected = primitive_sequence[current_primitive_idx]
                                # If next primitive is also a movement-type action (push, turn, etc.),
                                # and there are no gripper actions until end of sequence,
                                # we should try to subdivide this segment
                                remaining = primitive_sequence[current_primitive_idx:]
                                no_gripper_actions = not any(
                                    "close gripper" in p.lower() or "open gripper" in p.lower()
                                    for p in remaining
                                )
                                if no_gripper_actions and ("push" in next_expected.lower() or "turn" in next_expected.lower()):
                                    # Subdivide segment: first half is "move to", second half is action
                                    # We'll add the next primitive as a separate sub-segment
                                    logging.info(f"    Long segment detected with remaining primitives: {remaining}")
                                    logging.info(f"    Will subdivide to include '{next_expected}'")
                                    # Mark this for subdivision in post-processing
                                    episode_labels["_subdivide_hint"] = {
                                        "seg_idx": seg_idx,
                                        "next_primitive": next_expected,
                                        "current_primitive_idx": current_primitive_idx
                                    }
                        else:
                            primitive_label = "move gripper to target"
                            logging.info(f"    Label fallback: '{primitive_label}' (end of sequence)")

            labels_so_far.append(primitive_label)

            episode_labels["segments"].append({
                "segment_idx": seg_idx,
                "start_frame": start_idx,
                "end_frame": end_idx,
                "length": end_idx - start_idx,
                "primitive_label": primitive_label,
                "action_summary": action_summary,
            })

        # Post-process: subdivide long gripper segments that contain push/pull actions
        if "_subdivide_gripper_hint" in episode_labels:
            hint = episode_labels.pop("_subdivide_gripper_hint")
            target_seg_idx = hint["seg_idx"]
            next_primitive = hint["next_primitive"]

            new_segments = []
            for seg in episode_labels["segments"]:
                if seg["segment_idx"] == target_seg_idx:
                    seg_length = seg["end_frame"] - seg["start_frame"]
                    # For gripper->push, first ~15% is gripper action, rest is push
                    split_point = seg["start_frame"] + max(10, int(seg_length * 0.15))

                    # First part: gripper action (close/open)
                    new_segments.append({
                        "segment_idx": seg["segment_idx"],
                        "start_frame": seg["start_frame"],
                        "end_frame": split_point,
                        "length": split_point - seg["start_frame"],
                        "primitive_label": seg["primitive_label"],
                        "action_summary": None,
                    })

                    # Second part: the push/pull action
                    new_segments.append({
                        "segment_idx": seg["segment_idx"] + 1,
                        "start_frame": split_point,
                        "end_frame": seg["end_frame"],
                        "length": seg["end_frame"] - split_point,
                        "primitive_label": next_primitive,
                        "action_summary": None,
                    })
                    logging.info(f"  Subdivided gripper segment {target_seg_idx} at frame {split_point}: '{seg['primitive_label']}' + '{next_primitive}'")
                else:
                    new_segments.append(seg)

            episode_labels["segments"] = new_segments

        # Post-process: subdivide long segments that contain multiple primitives
        if "_subdivide_hint" in episode_labels:
            hint = episode_labels.pop("_subdivide_hint")
            target_seg_idx = hint["seg_idx"]
            next_primitive = hint["next_primitive"]

            # Find the segment to subdivide
            new_segments = []
            for seg in episode_labels["segments"]:
                if seg["segment_idx"] == target_seg_idx:
                    # Subdivide this segment: first ~60% for move, last ~40% for action
                    seg_length = seg["end_frame"] - seg["start_frame"]
                    split_point = seg["start_frame"] + int(seg_length * 0.6)

                    # First part: move to target
                    new_segments.append({
                        "segment_idx": seg["segment_idx"],
                        "start_frame": seg["start_frame"],
                        "end_frame": split_point,
                        "length": split_point - seg["start_frame"],
                        "primitive_label": seg["primitive_label"],
                        "action_summary": seg.get("action_summary"),
                    })

                    # Second part: the action (push, turn, etc.)
                    new_segments.append({
                        "segment_idx": seg["segment_idx"] + 1,
                        "start_frame": split_point,
                        "end_frame": seg["end_frame"],
                        "length": seg["end_frame"] - split_point,
                        "primitive_label": next_primitive,
                        "action_summary": None,
                    })
                    logging.info(f"  Subdivided segment {target_seg_idx} at frame {split_point}: '{seg['primitive_label']}' + '{next_primitive}'")
                else:
                    new_segments.append(seg)

            episode_labels["segments"] = new_segments

        # Post-process: merge consecutive segments with same label
        merged_segments = []
        for seg in episode_labels["segments"]:
            label = seg["primitive_label"].strip('"\'').lower()
            if merged_segments:
                prev_label = merged_segments[-1]["primitive_label"].strip('"\'').lower()
                # Merge if same label (normalize for comparison)
                if label == prev_label:
                    # Extend previous segment
                    merged_segments[-1]["end_frame"] = seg["end_frame"]
                    merged_segments[-1]["length"] = merged_segments[-1]["end_frame"] - merged_segments[-1]["start_frame"]
                    logging.info(f"  Merged segment {seg['segment_idx']} into previous (same label: '{label}')")
                    continue
            merged_segments.append(seg)

        # Re-number segments after merging
        for i, seg in enumerate(merged_segments):
            seg["segment_idx"] = i

        episode_labels["segments"] = merged_segments
        logging.info(f"  After merging: {len(merged_segments)} segments")

        # Post-process: split gripper-closed segments when plan expects more primitives
        # Find the gripper close/open in the plan and count expected primitives between them
        plan_lower = [p.lower() for p in primitive_sequence]
        close_idx = next((i for i, p in enumerate(plan_lower) if p == "close gripper"), None)
        open_idx = next((i for i, p in enumerate(plan_lower) if p == "open gripper"), None)
        if args.has_gripper and close_idx is not None and open_idx is not None:
            expected_between = primitive_sequence[close_idx + 1:open_idx]
            # Count actual segments between close and open gripper in our labels
            seg_labels = [s["primitive_label"].lower() for s in episode_labels["segments"]]
            close_seg_idx = next((i for i, l in enumerate(seg_labels) if l == "close gripper"), None)
            open_seg_idx = next((i for i, l in enumerate(seg_labels) if l == "open gripper"), None)
            if close_seg_idx is not None and open_seg_idx is not None:
                actual_between = episode_labels["segments"][close_seg_idx + 1:open_seg_idx]
                if len(actual_between) < len(expected_between) and len(actual_between) > 0:
                    # Hybrid sub-split: gripper crossings already pinned the
                    # close/open boundaries from action data (exact, off by ≤1
                    # frame); the closed-gripper window is bounded. To localize
                    # the inner primitives (e.g. lift/move-to-bowl/lower for
                    # pickplace) we call the VLM on JUST that sub-window.
                    #
                    # Why not state-changepoints here: variance-reduction peaks
                    # don't always align with semantic primitive boundaries when
                    # teleop blends motions (XY decel + Z accel overlapping).
                    # Why not full-episode VLM: it sometimes drifts hundreds of
                    # frames misplacing close-gripper because it can't see the
                    # action data. Bounding VLM to the closed window prevents
                    # both failure modes.
                    #
                    # Falls back to changepoint sub-split if the sub-window VLM
                    # call fails or returns the wrong number of segments.
                    merged_start = actual_between[0]["start_frame"]
                    merged_end = actual_between[-1]["end_frame"]
                    n_splits = len(expected_between) - 1
                    sub_segments = None
                    try:
                        ep_global_start = dataset.episode_data_index['from'][episode_idx].item()
                        extra_keys = (
                            [k.strip() for k in args.extra_image_keys.split(",") if k.strip()]
                            if args.extra_image_keys else None
                        )
                        sub_states = (states[merged_start:merged_end]
                                      if states is not None else None)
                        logging.info(
                            f"  Sub-window VLM split: localizing {expected_between} "
                            f"in closed-gripper window [{merged_start}, {merged_end}] "
                            f"(length {merged_end - merged_start})"
                        )
                        sub_segments = assign_plan_to_frames(
                            client, dataset,
                            episode_start=ep_global_start + merged_start,
                            episode_length=merged_end - merged_start,
                            task_description=task_description,
                            plan=expected_between,
                            frame_subsample=args.vlm_label_frame_subsample,
                            image_key=args.primary_image_key,
                            extra_image_keys=extra_keys,
                            refine_boundaries=args.vlm_label_refine,
                            refine_window=args.vlm_label_refine_window,
                            states=sub_states,
                        )
                    except Exception as e:
                        logging.warning(f"  Sub-window VLM call failed ({e}); falling back to changepoints")
                        sub_segments = None

                    if sub_segments and len(sub_segments) == len(expected_between):
                        # Sub-VLM returns LOCAL frames (0..L); shift to episode frames.
                        for s in sub_segments:
                            s["start_frame"] = merged_start + s["start_frame"]
                            s["end_frame"] = merged_start + s["end_frame"]
                        boundaries = ([merged_start]
                                      + [s["end_frame"] for s in sub_segments[:-1]]
                                      + [merged_end])
                        split_points = boundaries[1:-1]
                        logging.info(f"  Sub-window VLM split: frames {merged_start}-{merged_end} into {expected_between}, boundaries at {split_points}")
                    else:
                        if sub_segments is not None:
                            logging.warning(
                                f"  Sub-window VLM returned {len(sub_segments)} segments, "
                                f"expected {len(expected_between)} — falling back to changepoints"
                            )
                        split_points = find_action_changepoints(
                            actions, merged_start, merged_end, n_splits,
                            window=args.changepoint_window,
                            normalize_features=args.changepoint_normalize,
                            backtrack=args.changepoint_backtrack,
                            feature_tensor=states if args.changepoint_feature == "state" else None,
                        )

                    if split_points:
                        logging.info(f"  Changepoint split: frames {merged_start}-{merged_end} into {[p for p in expected_between]}, splits at {split_points}")
                        # Rebuild
                        new_segments = []
                        for seg in episode_labels["segments"]:
                            if seg in actual_between:
                                if seg == actual_between[0]:
                                    boundaries = [merged_start] + split_points + [merged_end]
                                    for k in range(len(expected_between)):
                                        new_segments.append({
                                            "segment_idx": len(new_segments),
                                            "start_frame": boundaries[k],
                                            "end_frame": boundaries[k + 1],
                                            "length": boundaries[k + 1] - boundaries[k],
                                            "primitive_label": expected_between[k],
                                            "action_summary": None,
                                        })
                                        logging.info(f"    '{expected_between[k]}' frames {boundaries[k]}-{boundaries[k+1]}")
                            else:
                                seg["segment_idx"] = len(new_segments)
                                new_segments.append(seg)
                        episode_labels["segments"] = new_segments
                        merged_segments = new_segments
                        logging.info(f"  After changepoint split: {len(new_segments)} segments")

        # Post-process: find rotation onset and relabel segments accordingly (flip-specific)
        if args.has_gripper and "rotate block" in [p.lower() for p in primitive_sequence]:
            # Find gripper-closed range from ACTION DATA (not labels, which may be wrong)
            gripper_cmd = actions[:, 6]
            close_frame = None
            open_frame = None
            for fi in range(1, len(gripper_cmd)):
                if gripper_cmd[fi-1] < 0 and gripper_cmd[fi] > 0 and close_frame is None:
                    close_frame = fi
                if gripper_cmd[fi-1] > 0 and gripper_cmd[fi] < 0 and close_frame is not None:
                    open_frame = fi
                    break

            if close_frame is not None and open_frame is not None:
                # Find rotation onset: first frame where rotation is consistently above threshold
                # Use adaptive threshold: 15% of max rotation in the closed phase
                closed_actions = actions[close_frame:open_frame]
                rot_mags = torch.norm(closed_actions[:, 3:6], dim=1)
                rot_threshold = rot_mags.max().item() * 0.15
                onset_window = 3  # Must exceed threshold for this many consecutive frames to confirm

                rot_onset = None
                for fi in range(len(rot_mags) - onset_window):
                    if all(rot_mags[fi + k] > rot_threshold for k in range(onset_window)):
                        # Confirmed rotation — now backtrack to find where it actually started
                        # Walk backwards from fi to find first frame above a lower threshold
                        backtrack_thresh = rot_threshold * 0.5
                        onset = fi
                        for bi in range(fi - 1, -1, -1):
                            if rot_mags[bi] >= backtrack_thresh:
                                onset = bi
                            else:
                                break
                        rot_onset = close_frame + onset
                        break

                if rot_onset is not None:
                    logging.info(f"  Rotation onset at frame {rot_onset} (threshold={rot_threshold:.4f})")

                    # Rebuild segments with rotation onset as the "rotate block" start
                    final_segments = []
                    for seg in merged_segments:
                        seg_start = seg["start_frame"]
                        seg_end = seg["end_frame"]
                        label = seg["primitive_label"]

                        if label.lower().startswith("move gripper to") or seg_start >= open_frame or seg_end <= close_frame:
                            # Keep segments outside gripper-closed range unchanged
                            seg["segment_idx"] = len(final_segments)
                            final_segments.append(seg)
                        elif label.lower() == "close gripper":
                            seg["segment_idx"] = len(final_segments)
                            final_segments.append(seg)
                        elif seg_end <= rot_onset:
                            # Entirely before rotation — label as lift
                            seg["primitive_label"] = "lift upward"
                            seg["segment_idx"] = len(final_segments)
                            final_segments.append(seg)
                        elif seg_start >= rot_onset:
                            # Entirely after rotation onset — label as rotate
                            seg["primitive_label"] = "rotate block"
                            seg["segment_idx"] = len(final_segments)
                            final_segments.append(seg)
                        else:
                            # Rotation onset falls within this movement segment — split
                            if rot_onset > seg_start + 3:
                                final_segments.append({
                                    "segment_idx": len(final_segments),
                                    "start_frame": seg_start,
                                    "end_frame": rot_onset,
                                    "length": rot_onset - seg_start,
                                    "primitive_label": "lift upward",
                                    "action_summary": None,
                                })
                            final_segments.append({
                                "segment_idx": len(final_segments),
                                "start_frame": rot_onset,
                                "end_frame": seg_end,
                                "length": seg_end - rot_onset,
                                "primitive_label": "rotate block",
                                "action_summary": None,
                            })

                    # Ensure "open gripper" exists at open_frame
                    has_open = any(s["primitive_label"].lower() == "open gripper"
                                   and abs(s["start_frame"] - open_frame) < 10
                                   for s in final_segments)
                    if not has_open:
                        # Find the segment that contains open_frame and split it
                        new_final = []
                        for seg in final_segments:
                            if seg["start_frame"] < open_frame < seg["end_frame"]:
                                # Split: everything before open_frame keeps current label
                                if open_frame > seg["start_frame"]:
                                    new_final.append({
                                        "segment_idx": len(new_final),
                                        "start_frame": seg["start_frame"],
                                        "end_frame": open_frame,
                                        "length": open_frame - seg["start_frame"],
                                        "primitive_label": seg["primitive_label"],
                                        "action_summary": None,
                                    })
                                # Insert open gripper
                                new_final.append({
                                    "segment_idx": len(new_final),
                                    "start_frame": open_frame,
                                    "end_frame": seg["end_frame"],
                                    "length": seg["end_frame"] - open_frame,
                                    "primitive_label": "open gripper",
                                    "action_summary": None,
                                })
                            else:
                                seg["segment_idx"] = len(new_final)
                                new_final.append(seg)
                        final_segments = new_final

                    # Merge consecutive same-label segments
                    merged_final = []
                    for seg in final_segments:
                        if merged_final and seg["primitive_label"].lower() == merged_final[-1]["primitive_label"].lower():
                            merged_final[-1]["end_frame"] = seg["end_frame"]
                            merged_final[-1]["length"] = merged_final[-1]["end_frame"] - merged_final[-1]["start_frame"]
                        else:
                            seg["segment_idx"] = len(merged_final)
                            merged_final.append(seg)

                    episode_labels["segments"] = merged_final
                    logging.info(f"  After rotation onset re-split: {len(merged_final)} segments")
                else:
                    logging.info(f"  No rotation onset found in closed phase")

        # Save single video per episode with changing captions
        if args.save_videos:
            video_extra_keys = [k.strip() for k in args.extra_image_keys.split(",") if k.strip()] if args.extra_image_keys else []
            all_annotated_frames = []
            for seg in episode_labels["segments"]:
                start_idx = seg["start_frame"]
                end_idx = seg["end_frame"]
                primitive_label = seg["primitive_label"]

                segment_frames = extract_all_frames(
                    dataset, episode_idx, start_idx, end_idx,
                    image_key=args.primary_image_key,
                )
                extra_frames_per_cam = [
                    extract_all_frames(dataset, episode_idx, start_idx, end_idx, image_key=k)
                    for k in video_extra_keys
                ]
                # Annotate each frame with the current primitive label
                for i, frame in enumerate(segment_frames):
                    # Side-by-side concat: primary | extras (resize extras to primary height if needed)
                    if extra_frames_per_cam:
                        h = frame.shape[0]
                        tiles = [frame]
                        for cam_frames in extra_frames_per_cam:
                            ef = cam_frames[i]
                            if ef.shape[0] != h:
                                scale = h / ef.shape[0]
                                new_w = int(ef.shape[1] * scale)
                                ef = np.array(Image.fromarray(ef).resize((new_w, h)))
                            tiles.append(ef)
                        combined = np.concatenate(tiles, axis=1)
                    else:
                        combined = frame
                    annotated = annotate_frame(
                        combined, primitive_label,
                        start_idx + i + 1, episode_length,  # Global frame number
                        task_description=task_description
                    )
                    all_annotated_frames.append(annotated)

            # Save single video for entire episode
            video_filename = f"ep{episode_idx:03d}_labeled.mov"
            video_path = videos_dir / video_filename
            imageio.mimwrite(video_path, all_annotated_frames, fps=args.fps, codec="libx264")
            logging.info(f"    Saved video: {video_path.name} ({len(all_annotated_frames)} frames)")

        # Validity filter: drop episodes whose segments don't cover every
        # expected primitive from the planner. Common when a demo ended
        # without releasing the gripper (no open_gripper segment, so the
        # closed-gripper window can't be sub-window-split into lift/move/lower)
        # or when the operator's recording was truncated mid-trajectory.
        # Including incomplete episodes pollutes the trained policy with
        # mislabeled segments (e.g. a 500-frame "lift upward" that actually
        # contains lift+move-to-bowl+lower glued together).
        if primitive_sequence:
            expected_set = set(primitive_sequence)
            actual_set = {seg["primitive_label"] for seg in episode_labels["segments"]}
            missing = expected_set - actual_set
            if missing:
                logging.warning(
                    f"  Episode {episode_idx}: SKIPPED — segments missing expected primitives: {sorted(missing)} "
                    f"(got {len(episode_labels['segments'])} segments, expected ≥{len(primitive_sequence)})"
                )
                continue

        all_labels.append(episode_labels)

        # Save intermediate results
        output_file = output_dir / "dense_labels.json"
        with open(output_file, 'w') as f:
            json.dump(all_labels, f, indent=2)

        logging.info(f"\nSaved labels to: {output_file}")

    # Summary
    logging.info("\n" + "=" * 80)
    logging.info("SUMMARY")
    logging.info("=" * 80)

    total_segments = sum(len(ep["segments"]) for ep in all_labels)
    avg_segments_per_episode = total_segments / len(all_labels)

    logging.info(f"Labeled {len(all_labels)} episodes")
    logging.info(f"Total segments: {total_segments}")
    logging.info(f"Average segments per episode: {avg_segments_per_episode:.1f}")
    logging.info(f"\nLabels saved to: {output_dir / 'dense_labels.json'}")
    if args.save_videos:
        logging.info(f"Videos saved to: {output_dir / 'videos'}/")

    # Aggregate primitive vocabulary across all episodes
    logging.info("\n" + "=" * 80)
    logging.info("Primitive Vocabulary Analysis")
    logging.info("=" * 80)

    # Collect all unique primitives from expected sequences
    expected_primitives = set()
    for ep in all_labels:
        expected_primitives.update(ep.get("expected_primitives", []))

    # Collect all matched primitives from segments
    matched_primitives = {}
    for ep in all_labels:
        for seg in ep["segments"]:
            prim = seg["primitive_label"]
            matched_primitives[prim] = matched_primitives.get(prim, 0) + 1

    logging.info(f"\nExpected primitives vocabulary ({len(expected_primitives)} unique):")
    for prim in sorted(expected_primitives):
        logging.info(f"  - {prim}")

    logging.info(f"\nMatched primitives frequency:")
    for prim, count in sorted(matched_primitives.items(), key=lambda x: -x[1])[:15]:
        logging.info(f"  {count:3d}x  {prim}")

    # Log discovered primitives
    if all_discovered_primitives:
        logging.info(f"\nNew primitive types discovered this run ({len(all_discovered_primitives)}):")
        for prim in all_discovered_primitives:
            logging.info(f"  + {prim}")
    else:
        logging.info(f"\nNo new primitive types discovered (all tasks used base primitives)")

    # Save primitive vocabulary
    vocab_file = output_dir / "primitive_vocabulary.json"
    vocab_data = {
        "base_primitives": [
            "move gripper to the [object/location]",
            "close gripper",
            "open gripper",
            "lift upward",
            "lower gripper",
        ],
        "discovered_primitives": all_discovered_primitives,
        "final_vocabulary": known_primitives,
        "expected_primitives": sorted(list(expected_primitives)),
        "matched_primitives_freq": {k: v for k, v in sorted(matched_primitives.items(), key=lambda x: -x[1])},
        "total_unique_expected": len(expected_primitives),
        "total_unique_matched": len(matched_primitives)
    }
    with open(vocab_file, 'w') as f:
        json.dump(vocab_data, f, indent=2)
    logging.info(f"\nPrimitive vocabulary saved to: {vocab_file}")

    # Create LeRobot dataset if requested
    if args.create_lerobot_dataset:
        _create_lerobot_dataset(args, dataset, all_labels, dataset_repo_id=args.dataset_repo_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    tyro.cli(densely_label_dataset)
