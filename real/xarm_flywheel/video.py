"""Video / frame-annotation helpers for the xArm flywheel runner.

Composites the (exterior | wrist) side-by-side panel and burns a per-frame
title bar with colored segments:

- ``[counter]`` (e.g. ``[4/14]``) in **light blue**
- ``[known]`` in **green**, ``[SKILL-GAP]`` in **red**
- primitive name in off-white (primary text)
- step / progress on the right in dim grey
- thin progress bar across the bottom of the title bar (amber on slate)

Image format is xArm-specific (320×240 RGB twin-camera output) which is
why this stays in the xArm package rather than ``insight``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _ffmpeg_kwargs(high_quality: bool) -> dict:
    """imageio writer kwargs. ``high_quality=True`` produces near-lossless
    files suitable for paper figures (CRF 17, yuv420p, slow preset);
    otherwise defaults are used (faster encode, smaller files)."""
    if high_quality:
        return dict(
            codec="libx264",
            output_params=["-crf", "17", "-pix_fmt", "yuv420p", "-preset", "slow"],
        )
    return dict(codec="libx264")


# Font priority for the title bar. PT Sans (paratype package on Ubuntu) is
# the preferred face — cleaner spacing + warmer character shapes than DejaVu
# for video overlays. Falls back to DejaVu Sans Bold and finally PIL's
# bitmap default.
#
# To install on Debian/Ubuntu: ``sudo apt install fonts-paratype``. Verify
# with ``fc-list | grep -i 'pt sans'``.
_FONT_CANDIDATES: tuple[str, ...] = (
    "/usr/share/fonts/truetype/paratype/PTS75F.ttf",       # PT Sans Bold (fonts-paratype)
    "/usr/share/fonts/truetype/paratype/PTS55F.ttf",       # PT Sans Regular (fonts-paratype)
    "/usr/share/fonts/truetype/ptsans/PT_Sans-Web-Bold.ttf",
    "/usr/share/fonts/truetype/ptsans/PTSans-Bold.ttf",
    "/usr/share/fonts/opentype/ptsans/PTSans-Bold.otf",
    "/usr/share/fonts/truetype/ptsans/PT_Sans-Web-Regular.ttf",
    "/usr/share/fonts/truetype/ptsans/PTSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)


def _load_font(size: int = 14) -> ImageFont.ImageFont:
    """Return the first existing TrueType face from ``_FONT_CANDIDATES``.

    Module-cached at first call site (re-resolution per frame would be
    O(N_frames) syscalls). Falls back to PIL's bitmap default if no
    candidate is on disk.
    """
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


# One-shot font resolution at import — same face for every frame in the run.
_TITLE_FONT = _load_font(14)

# Title-bar palette. Tailwind-inspired so the colors look intentional rather
# than improvised. Tuples are (R, G, B) integer.
_BG_COLOR        = (30, 35, 48)        # #1E2330 — dark navy, easier on the eye than pure black
_TEXT_PRIMARY    = (241, 245, 249)     # slate-100 — primitive name
_TEXT_SECONDARY  = (148, 163, 184)     # slate-400 — step / progress / global step
_COUNTER_COLOR   = (147, 197, 253)     # sky-300  — [4/14]
_TAG_KNOWN_COLOR = (74, 222, 128)      # green-400 — [known]
_TAG_NEW_COLOR   = (248, 113, 113)     # red-400  — [SKILL-GAP]
# Progress bar is deliberately muted: the underlying model-progress signal
# can be wrong (notably lift-after-twist saturates at ~1.0 from step 0
# because the post-twist state is OOD). Loud amber would mis-sell "almost
# done" on those failure modes. A subtle slate fill on a slightly darker
# track stays informative without being authoritative.
_PROGRESS_TRACK  = (37, 47, 64)        # slate-800 — empty portion (close to bg)
_PROGRESS_FILL   = (100, 116, 139)     # slate-500 — filled portion (muted, neutral)
_SEPARATOR_COLOR = (148, 163, 184)     # slate-400 — pipe between primitive + total groups

_TOP_BAR_HEIGHT    = 24                # top strip: [counter] [tag] primitive (left-aligned)
_BOTTOM_BAR_HEIGHT = 24                # bottom strip: [bar] % step | total (right-aligned)
_TEXT_Y            = 4                 # vertical text offset inside each strip
_LEFT_PADDING      = 6
_RIGHT_PADDING     = 6

# Inline progress-bar geometry — rendered next to the percentage on the
# right side. Kept compact so it reads as a supplementary indicator, not
# a confident "you are here" claim (the underlying model-progress signal
# can be wrong, notably OOD post-twist).
_MINI_BAR_WIDTH  = 56
_MINI_BAR_HEIGHT = 6
_MINI_BAR_GAP_RIGHT = 6  # px between bar and `XX%` text
_MINI_BAR_GAP_LEFT  = 10  # px between `step N` text and bar


def _draw_segments(draw: ImageDraw.ImageDraw, x: int, y: int,
                   segments: list[tuple[str, tuple[int, int, int]]],
                   font: ImageFont.ImageFont) -> int:
    """Draw ``segments`` left-to-right starting at ``(x, y)``, advancing ``x``
    by each segment's measured width. Returns the new ``x`` after the last
    segment so callers can append more text inline."""
    for text, color in segments:
        draw.text((x, y), text, fill=color, font=font)
        bbox = draw.textbbox((x, y), text, font=font)
        x = bbox[2]
    return x


def _annotate(frame: np.ndarray, parts) -> np.ndarray:
    """Wrap ``frame`` with two annotation strips (top + bottom).

    Layout:
      ┌─ TOP 24px ──────────────────────────────────┐
      │ [counter] [tag] primitive (left-aligned)    │
      ├─────────────────────────────────────────────┤
      │            camera composite                 │
      ├─ BOTTOM 24px ───────────────────────────────┤
      │       [bar] %  step N  |  N/total           │  (right-aligned)
      └─────────────────────────────────────────────┘

    The split prevents the right-side progress group from overlapping a
    long primitive name (e.g., ``"move gripper to the top of the yellow
    bottle"``). Top strip is "what's running"; bottom strip is "where in
    time we are". Total padding height is 48 px (288 = 16·18 with a
    240-px camera composite, so imageio doesn't need to pad for the
    macro_block_size=16 codec requirement).

    ``parts`` may be:
    - a ``dict`` with keys ``counter`` / ``tag`` / ``primitive`` / ``step``
      / ``progress`` / ``global_step`` / ``total_frames`` (any subset).
    - a ``str`` (legacy): rendered as a single off-white line in the top
      strip without colors. Kept so older callers continue to work.
    """
    img = Image.fromarray(frame)
    new_h = _TOP_BAR_HEIGHT + img.height + _BOTTOM_BAR_HEIGHT
    out = Image.new("RGB", (img.width, new_h), _BG_COLOR)
    out.paste(img, (0, _TOP_BAR_HEIGHT))
    draw = ImageDraw.Draw(out)

    # Y coordinates of the text baseline inside each strip.
    top_y = _TEXT_Y
    bot_y = _TOP_BAR_HEIGHT + img.height + _TEXT_Y

    # Legacy string path — render in the top strip as a single off-white line.
    if isinstance(parts, str):
        draw.text((_LEFT_PADDING, top_y), parts,
                  fill=_TEXT_PRIMARY, font=_TITLE_FONT)
        return np.asarray(out)

    def _w(text: str) -> int:
        bbox = draw.textbbox((0, 0), text, font=_TITLE_FONT)
        return bbox[2] - bbox[0]

    # ────── Top strip: [counter] [tag] primitive ──────
    top_segments: list[tuple[str, tuple[int, int, int]]] = []
    counter = parts.get("counter")
    if counter:
        top_segments.append((f"[{counter}] ", _COUNTER_COLOR))
    tag = parts.get("tag")
    if tag:
        # Keyword color routing so renames don't break rendering: tags
        # containing "gap" or "new" → red; otherwise green. Covers all
        # legacy variants (SKILL-GAP, primitive-gap, VLA known, etc.).
        tag_lower = tag.lower()
        tag_color = (
            _TAG_NEW_COLOR
            if "gap" in tag_lower or "new" in tag_lower
            else _TAG_KNOWN_COLOR
        )
        top_segments.append((f"[{tag}] ", tag_color))
    x_left = _draw_segments(draw, _LEFT_PADDING, top_y, top_segments, _TITLE_FONT)

    # Primitive name. Truncate with ellipsis as a defensive guard against
    # exotic-long names (current camera width 640 px gives plenty of room
    # for typical primitives, so this branch rarely fires).
    primitive = parts.get("primitive", "")
    if primitive:
        avail = out.width - _RIGHT_PADDING - x_left
        if _w(primitive) <= avail:
            draw.text((x_left, top_y), primitive, fill=_TEXT_PRIMARY, font=_TITLE_FONT)
        elif avail >= 20:
            ellipsis = "…"
            lo, hi = 0, len(primitive)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if _w(primitive[:mid] + ellipsis) <= avail:
                    lo = mid
                else:
                    hi = mid - 1
            shown = primitive[:lo] + ellipsis if lo > 0 else ellipsis
            draw.text((x_left, top_y), shown, fill=_TEXT_PRIMARY, font=_TITLE_FONT)

    # ────── Bottom strip: [bar] %  step N  |  N/TOTAL (right-aligned) ──────
    # Drawn right-to-left so the geometry is easy to compute conditionally.
    step = parts.get("step")
    progress = parts.get("progress")
    global_step = parts.get("global_step")
    total_frames = parts.get("total_frames")

    x = out.width - _RIGHT_PADDING

    # Group 2: combined-video frame fraction (right-most).
    if global_step is not None:
        if total_frames is not None:
            text = f"{global_step}/{total_frames}"
        else:
            text = f"TOTAL {global_step}"
        x -= _w(text)
        draw.text((x, bot_y), text, fill=_TEXT_SECONDARY, font=_TITLE_FONT)
        if step is not None or progress is not None:
            sep = "  |  "
            x -= _w(sep)
            draw.text((x, bot_y), sep, fill=_SEPARATOR_COLOR, font=_TITLE_FONT)

    # Group 1: step, percentage, mini bar.
    if step is not None:
        text = f"step {step}"
        x -= _w(text)
        draw.text((x, bot_y), text, fill=_TEXT_SECONDARY, font=_TITLE_FONT)
        if progress is not None:
            x -= _MINI_BAR_GAP_LEFT

    if progress is not None:
        pct = max(0, min(100, int(round(float(progress) * 100))))
        pct_text = f"{pct}%"
        x -= _w(pct_text)
        draw.text((x, bot_y), pct_text, fill=_TEXT_SECONDARY, font=_TITLE_FONT)
        x -= _MINI_BAR_GAP_RIGHT
        bar_right = x
        bar_left = x - _MINI_BAR_WIDTH
        bar_top = bot_y + (_TITLE_FONT.size - _MINI_BAR_HEIGHT) // 2 + 1
        bar_bot = bar_top + _MINI_BAR_HEIGHT
        draw.rectangle([(bar_left, bar_top), (bar_right, bar_bot)], fill=_PROGRESS_TRACK)
        fill_w = int(_MINI_BAR_WIDTH * max(0.0, min(1.0, float(progress))))
        if fill_w > 0:
            draw.rectangle([(bar_left, bar_top), (bar_left + fill_w, bar_bot)],
                           fill=_PROGRESS_FILL)

    return np.asarray(out)


def write_video(labeled_frames: list, path: Path, fps: float,
                high_quality: bool = False) -> None:
    """Write a side-by-side (ext | wrist) annotated mp4.

    Each item in ``labeled_frames`` is ``(label, ext, wrist)`` where ``label``
    is either a structured ``dict`` (preferred, colorized) or a plain
    ``str`` (legacy, single-color). Pass ``high_quality=True`` for
    paper-figure encoding (CRF 17, yuv420p, slow preset).
    """
    composed = [_annotate(np.hstack([ext, wrist]), label) for label, ext, wrist in labeled_frames]
    imageio.mimwrite(str(path), composed, fps=int(round(fps)),
                     **_ffmpeg_kwargs(high_quality))


def save_primitive_video(
    run_dir: Path,
    primitive: str,
    primitive_idx: int,
    plan_total: int,
    frames: list[tuple[int, np.ndarray, np.ndarray, float]],
    fps: float,
    *,
    is_skill_gap: bool = False,
    suffix: str = "",
    high_quality: bool = False,
) -> Path:
    """Save the per-primitive video. ``frames`` is ``[(step, ext, wrist, progress), ...]``.

    ``is_skill_gap=True`` renders the ``[primitive-gap]`` tag in red (a
    primitive missing from the VLA's vocabulary, executed via low-level
    control); the default renders ``[known]`` in green. ``suffix`` (e.g.
    ``"_INCOMPLETE"``) is appended to the filename for the gap-failed case.
    """
    slug = primitive.replace(" ", "_")[:40]
    path = run_dir / f"primitive_{primitive_idx + 1:02d}_{slug}{suffix}.mp4"
    counter = f"{primitive_idx + 1}/{plan_total}"
    tag = "primitive gap" if is_skill_gap else "known"
    labeled = [
        (
            {
                "counter": counter,
                "tag": tag,
                "primitive": primitive,
                "step": step,
                "progress": progress,
            },
            ext, wrist,
        )
        for step, ext, wrist, progress in frames
    ]
    write_video(labeled, path, fps, high_quality=high_quality)
    logging.info("  saved %s (%d frames)", path.name, len(frames))
    return path


def save_combined_video(run_dir: Path, frames: list,
                        fps: float, name_suffix: str = "",
                        high_quality: bool = True) -> Path | None:
    """Write the run-wide ``all_primitives[<suffix>].mp4`` from already-labeled frames.

    Each frame is ``(label, ext, wrist)`` where ``label`` is the structured
    dict produced by ``XArmRunner._extend_combined`` (or a string for legacy
    callers).

    The total frame count is only known at this point (not when each label
    was constructed during execution), so we inject ``total_frames`` into
    every structured label here. ``_annotate`` then renders the combined-video
    counter as ``current/total`` (e.g. ``143/300``) instead of a bare
    ``TOTAL 143`` that gives no sense of position-in-video.

    Defaults to high-quality encoding because the combined video is the
    primary paper-facing artifact for each trial. Per-primitive,
    continuous, and debug videos default to faster/smaller encoding
    unless ``--high-quality-videos`` is passed.

    ``name_suffix`` is appended to the filename (e.g., the experiment name)
    so the mp4 is self-identifying when copied out of its run_dir.
    """
    if not frames:
        return None
    total = len(frames)
    for label, _ext, _wrist in frames:
        if isinstance(label, dict):
            label["total_frames"] = total
    suffix = f"_{name_suffix}" if name_suffix else ""
    path = run_dir / f"all_primitives{suffix}.mp4"
    write_video(frames, path, fps, high_quality=high_quality)
    logging.info("Saved combined video: %s (%d frames)", path.name, len(frames))
    return path


def save_debug_videos(run_dir: Path,
                      ext_frames: list[np.ndarray],
                      wrist_frames: list[np.ndarray],
                      fps: float,
                      high_quality: bool = False) -> None:
    """Save unlabeled per-camera videos for debugging.

    Writes ``debug_exterior.mp4`` and ``debug_wrist.mp4`` containing the raw
    camera frames captured during execution, with no titles/annotations and no
    side-by-side composite. Useful for review without the per-primitive labels
    cluttering the visual. Survives mid-run Ctrl+C if called from a finally
    block (the frame buffers are appended every tick).
    """
    if not ext_frames and not wrist_frames:
        return
    hq = _ffmpeg_kwargs(high_quality)
    if ext_frames:
        ext_path = run_dir / "debug_exterior.mp4"
        imageio.mimwrite(str(ext_path), ext_frames, fps=int(round(fps)), **hq)
        logging.info("Saved %s (%d frames)", ext_path.name, len(ext_frames))
    if wrist_frames:
        wrist_path = run_dir / "debug_wrist.mp4"
        imageio.mimwrite(str(wrist_path), wrist_frames, fps=int(round(fps)), **hq)
        logging.info("Saved %s (%d frames)", wrist_path.name, len(wrist_frames))
