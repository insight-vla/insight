"""Env-agnostic image helpers shared by sim + real pipelines."""

from __future__ import annotations

import numpy as np
from PIL import Image


def resize_for_vlm(img: np.ndarray, size: int = 512) -> np.ndarray:
    """Letterbox-resize ``img`` to ``size``×``size`` for VLM input.

    Preserves aspect ratio and pads with black. Output is uint8 RGB.
    """
    pil = Image.fromarray(img).convert("RGB")
    w, h = pil.size
    scale = size / max(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    pil = pil.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(pil, ((size - new_w) // 2, (size - new_h) // 2))
    return np.asarray(canvas, dtype=np.uint8)


# Direction of the gripper-local +X / +Y AXIS VECTORS in wrist-image
# pixel space (dx, dy). These are AXIS directions (not "tipping
# effects") — the VLM applies right-hand rule with its thumb along
# the arrow. +drz is implicit: rotation in the image plane (since
# +Z = camera direction, into screen). The wrist camera is rigidly
# mounted to the gripper, so these directions are CONSTANT; verify
# empirically by sending a small +drx / +dry command and watching
# the wrist image, then flip a sign here if RHR doesn't predict the
# observed motion.
WRIST_AXES_LIBERO = {
    "x": (0, -1),   # gripper-local +X axis points UP (forward) in wrist image
    "y": (-1, 0),   # gripper-local +Y axis points LEFT in wrist image
    # Right-handed: +x × +y = +z (OUT of screen, world-up for
    # downward-pointing gripper). Matches the +X forward, +Y left,
    # +Z up base-frame convention.
}

WRIST_AXES_XARM = {
    "x": (0, 1),    # placeholder — verify with a single +drx command
    "y": (-1, 0),
}


def draw_wrist_axes(
    img: np.ndarray,
    axes: dict | None = None,
) -> np.ndarray:
    """Draw curved rotation-effect arrows on a wrist-camera image.

    Three curved swooshes — one per rotation axis — replace the previous
    straight axis-arrow design. Straight arrows kept getting read as
    "axis vectors" by the VLM (i.e. "the +drx arrow points left, so
    rotate around the left-pointing line"), which is the wrong mental
    model. Curved arrows force the "tipping effect" reading: each
    swoosh shows the direction the gripper's top arcs under that
    rotation, in the image plane.

    - +drx: vertical ellipse arc on the LEFT side — rotation in the
            (image-UP, into-screen) plane. Drawn as a tilted oval to
            indicate it's an out-of-plane rotation; arrowhead leans
            toward image-LEFT to show the "tips left" effect.
    - +dry: horizontal ellipse arc at the TOP — rotation in the
            (image-RIGHT, into-screen) plane. Arrowhead leans UP for
            the "tips up" effect.
    - +drz: full CCW circle — pure in-plane rotation, no
            foreshortening needed.

    Args:
        img: HxWx3 uint8 RGB wrist image.
        axes: kept for backward compatibility but unused (curved
              swooshes don't need the per-axis (dx, dy) vectors).

    Returns:
        New image with rotation swooshes drawn. Original is not modified.
    """
    import cv2

    out = img.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.5, 0.0011 * min(h, w))

    def _label(text, pos, color):
        cv2.putText(out, text, pos, font, fs, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, text, pos, font, fs, color, 1, cv2.LINE_AA)

    # ----- +drx: straight red arrow pointing LEFT ------------------
    arrow_len = int(0.22 * min(h, w))
    cx_arr, cy_arr = int(0.32 * w), int(0.55 * h)
    # Arrow directions come from the caller-supplied axes (or the
    # default WRIST_AXES_LIBERO) — they represent gripper-local +x
    # and +y axis vectors in wrist image pixel space. Labels are
    # "+x" / "+y" (axis names) so the VLM applies right-hand rule
    # with thumb along the arrow.
    if axes is None:
        axes = WRIST_AXES_LIBERO
    red = (255, 60, 60)
    dx, dy = axes["x"]
    end = (cx_arr + int(dx * arrow_len), cy_arr + int(dy * arrow_len))
    cv2.arrowedLine(out, (cx_arr, cy_arr), end, (0, 0, 0), 5, cv2.LINE_AA, tipLength=0.25)
    cv2.arrowedLine(out, (cx_arr, cy_arr), end, red, 3, cv2.LINE_AA, tipLength=0.25)
    _label("+x", (end[0] + int(dx * 18 - 0.02 * w), end[1] + int(dy * 18) + 6), red)

    # ----- +y: straight green arrow --------------------------------
    green = (60, 220, 60)
    dx, dy = axes["y"]
    end = (cx_arr + int(dx * arrow_len), cy_arr + int(dy * arrow_len))
    cv2.arrowedLine(out, (cx_arr, cy_arr), end, (0, 0, 0), 5, cv2.LINE_AA, tipLength=0.25)
    cv2.arrowedLine(out, (cx_arr, cy_arr), end, green, 3, cv2.LINE_AA, tipLength=0.25)
    _label("+y", (end[0] + int(dx * 18 - 0.02 * w), end[1] + int(dy * 18) + 6), green)

    # ----- +drz: full CCW circle, bottom-right corner --------------
    blue = (90, 150, 255)
    arc_r = int(0.08 * min(h, w))
    arc_cx, arc_cy = int(0.85 * w), int(0.82 * h)
    cv2.ellipse(out, (arc_cx, arc_cy), (arc_r, arc_r), 0, 0, 270,
                (0, 0, 0), 5, cv2.LINE_AA)
    cv2.ellipse(out, (arc_cx, arc_cy), (arc_r, arc_r), 0, 0, 270,
                blue, 2, cv2.LINE_AA)
    tip = (arc_cx, arc_cy - arc_r)
    tip_back = (arc_cx + int(0.35 * arc_r), arc_cy - arc_r - int(0.35 * arc_r))
    cv2.arrowedLine(out, tip_back, tip, (0, 0, 0), 5, cv2.LINE_AA, tipLength=0.6)
    cv2.arrowedLine(out, tip_back, tip, blue, 2, cv2.LINE_AA, tipLength=0.6)
    _label("+z (out, up)", (arc_cx - int(1.1 * arc_r), arc_cy + int(0.15 * arc_r)), blue)
    return out
