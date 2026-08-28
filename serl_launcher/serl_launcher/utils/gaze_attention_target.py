"""Build ViT grounding targets from recorded operator gaze.

This is the mask-free replacement for the CGL target. Instead of scoring the
grounding query against a segmentation mask, it is scored against a
distribution built from where the operator actually looked.

Nothing here reads a mask or a phase label. That is the point: the masks in the
recordings were hand-annotated with SAM3, so a pipeline that consumed them
would invite the obvious question of why the gaze is needed at all. The
pick/place separation instead falls out of the gaze itself -- measured over the
41 recorded episodes, the operator's gaze lands inside the basket on 1.5% of
pre-grasp frames against 32.9% after the grasp, and 37 of 41 episodes contain
no pre-grasp basket look at all.

Three shape parameters, in token cells rather than pixels because the CGL loss
scores on the token grid:

``sigma_cells``
    The soft edge. 0.6 is the label-noise floor -- gaze labels carry ~4.4 px of
    error at 128x128, which is 0.55 cells at 224/patch-16 -- so a tighter
    target would be fitting noise.

``dilate_cells``
    How far from the gaze still counts as the same target. This is a
    morphological dilation, not a larger sigma: it makes everything within the
    radius equally correct, where a wider Gaussian would keep insisting the
    exact centre matters most. It is needed because during the approach the
    operator looks at the gripper rather than the ball, leaving the ball a
    median 0.77 cells outside the raw gaze blob.

``window``
    Temporal pooling. This buys robustness, not extent: sweeping 0 -> 8 frames
    moves the mass on every object by less than 0.003, because over 0.6 s the
    gaze barely crosses a token cell. What it does buy is that one drifted gaze
    sample is a wrong delta on its own but gets outvoted by its neighbours.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

# Settled by sweeping all 41 episodes; see the module docstring for what each
# one trades off. At dilate 1.0 the ball falls inside the target's strongest
# cells on 88% of frames while pre-grasp mass on the basket stays at 0.10
# against a 0.19 chance baseline, so the phase separation survives.
DEFAULT_SIGMA_CELLS = 0.6
DEFAULT_DILATE_CELLS = 1.0
DEFAULT_WINDOW = 5
DEFAULT_MAX_GAP = 3
DEFAULT_DECAY = 0.15
DEFAULT_SUPERSAMPLE = 4


_OFFSET_CACHE: Dict[str, Tuple[float, float]] = {}


def _session_offset(frame_dir: Path) -> Tuple[float, float]:
    """Per-session gaze calibration shift, in pixels, or (0, 0).

    Each recording session was eye-tracker-calibrated separately and they
    drifted against each other; the shift lives in
    `gaze_offset_correction.json` beside the frames rather than in code, so a
    session carries its own correction wherever it is read from.

    Only the drift *between* sessions is corrected. Every session shares a much
    larger offset toward the gripper, because the operator watches the contact
    region rather than the ball's centre -- that is behaviour, and removing it
    would move the target away from where they actually looked.
    """
    root = str(Path(frame_dir).parent)
    if root not in _OFFSET_CACHE:
        path = Path(root) / "gaze_offset_correction.json"
        shift = (0.0, 0.0)
        if path.exists():
            try:
                data = json.loads(path.read_text())
                shift = (float(data.get("dx_px", 0.0)), float(data.get("dy_px", 0.0)))
            except Exception:
                shift = (0.0, 0.0)
        _OFFSET_CACHE[root] = shift
    return _OFFSET_CACHE[root]


def read_gaze_xy(frame_dir: Path) -> Optional[np.ndarray]:
    """Normalised gaze in [0,1]^2, or None when this frame has no contact point.

    A missing or unparseable `gaze_contact.json` is the only failure mode; the
    `hit` field inside it is written unconditionally true and carries no
    information.

    Any per-session calibration shift is applied here, which is the single
    place every consumer -- target export, encoder training, the diagnostics --
    reads gaze through, so a correction cannot reach one of them and miss
    another.
    """
    path = Path(frame_dir) / "gaze_contact.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    uv = data.get("gaze_uv_in_realsense")
    size = data.get("realsense_size")
    if uv is None or size is None:
        return None
    width, height = float(size[0]), float(size[1])
    if width <= 0 or height <= 0:
        return None
    dx, dy = _session_offset(frame_dir)
    x = (float(uv[0]) + dx) / width
    y = (float(uv[1]) + dy) / height
    return np.array([min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)], dtype=np.float32)


def episode_gaze_index(frame_dirs: Dict[int, Path]) -> Dict[int, np.ndarray]:
    """{frame_id: gaze_xy} for the frames that have one."""
    points = {}
    for frame_id, frame_dir in frame_dirs.items():
        xy = read_gaze_xy(frame_dir)
        if xy is not None:
            points[int(frame_id)] = xy
    return points


def _walk(frame_id, points, direction, want, max_gap):
    """Up to `want` gaze samples walking one way from `frame_id`.

    Stops at a hole larger than `max_gap` so a pooled target never spans a cut
    the operator made when filtering, which would mix two unrelated moments.
    """
    found, previous, candidate = [], frame_id, frame_id + direction
    while len(found) < want and abs(candidate - frame_id) <= want + max_gap:
        if candidate in points:
            if abs(candidate - previous) > max_gap:
                break
            found.append(candidate)
            previous = candidate
        candidate += direction
    return found


def pool_gaze_points(
    frame_id: int,
    points: Dict[int, np.ndarray],
    window: int = DEFAULT_WINDOW,
    max_gap: int = DEFAULT_MAX_GAP,
    decay: float = DEFAULT_DECAY,
) -> Tuple[np.ndarray, np.ndarray]:
    """This frame's gaze plus its temporal neighbours, with weights.

    Aims for ``2 * window`` neighbours and compensates across sides: a frame
    near the start of an episode takes the shortfall from the future, and one
    near the end takes it from the past. Without that, boundary frames pool far
    fewer samples than interior ones and so get a systematically sharper,
    noisier target -- a difference in supervision that tracks position in the
    episode rather than anything about the scene.

    Weights decay with how far a neighbour was reached, not with which side it
    came from, so a compensated frame is weighted like any other.
    """
    pooled, weights = [points[frame_id]], [1.0]
    if window <= 0:
        return np.stack(pooled), np.asarray(weights, np.float32)

    total = 2 * window
    back = _walk(frame_id, points, -1, window, max_gap)
    forward = _walk(frame_id, points, 1, window, max_gap)
    shortfall = total - len(back) - len(forward)
    if shortfall > 0:
        # Only one side can be short at a time in practice, but ask both: the
        # extra call is free and it keeps the logic correct when a max_gap
        # break, not an episode edge, is what truncated a side.
        if len(back) < window:
            forward = _walk(frame_id, points, 1, len(forward) + shortfall, max_gap)
        elif len(forward) < window:
            back = _walk(frame_id, points, -1, len(back) + shortfall, max_gap)

    for side in (back, forward):
        for rank, candidate in enumerate(side, start=1):
            pooled.append(points[candidate])
            weights.append(float(np.exp(-decay * rank)))
    return np.stack(pooled), np.asarray(weights, np.float32)


def build_target_heatmap(
    pooled: np.ndarray,
    weights: np.ndarray,
    grid: Sequence[int],
    sigma_cells: float = DEFAULT_SIGMA_CELLS,
    dilate_cells: float = DEFAULT_DILATE_CELLS,
    supersample: int = DEFAULT_SUPERSAMPLE,
) -> np.ndarray:
    """Weighted Gaussians, dilated into a plateau, area-pooled to the token grid.

    Built at ``supersample`` times the token grid so sigma and the dilation
    radius can be fractions of a cell, which they must be: one cell spans 45 px
    of the 640-wide recording and the ball is under one cell across. Returns a
    (grid_h, grid_w) array summing to 1.
    """
    grid_h, grid_w = int(grid[0]), int(grid[1])
    scale = max(int(supersample), 1)
    fine_h, fine_w = grid_h * scale, grid_w * scale

    yy, xx = np.mgrid[0:fine_h, 0:fine_w].astype(np.float32)
    cx = np.asarray(pooled, np.float32)[:, 0] * (fine_w - 1)
    cy = np.asarray(pooled, np.float32)[:, 1] * (fine_h - 1)
    sigma = max(float(sigma_cells) * scale, 1e-3)
    blobs = np.exp(
        -((xx[None] - cx[:, None, None]) ** 2 + (yy[None] - cy[:, None, None]) ** 2)
        / (2.0 * sigma * sigma)
    )
    heatmap = (blobs * np.asarray(weights, np.float32)[:, None, None]).sum(axis=0)
    heatmap = heatmap.astype(np.float32)

    radius = int(round(float(dilate_cells) * scale))
    if radius > 0:
        disc = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        heatmap = cv2.dilate(heatmap, disc)

    heatmap = cv2.resize(heatmap, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
    return (heatmap / max(heatmap.sum(), 1e-8)).astype(np.float32)


def shift_points_for_crop(
    pooled: np.ndarray,
    offset_y: int,
    offset_x: int,
    padding: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Move gaze points with the random crop the images just went through.

    `random_crop` edge-pads by ``padding`` then slices the original size back
    out at ``(offset_y, offset_x)``, so a pixel at x lands at
    ``x + padding - offset_x``. The target has to follow or it drifts against
    the image by up to ``padding`` px -- on a 14x14 grid over a 128 px image
    that is most of a token cell, which is enough to teach the query the wrong
    location.
    """
    pooled = np.asarray(pooled, np.float32).copy()
    px = pooled[:, 0] * (width - 1) + padding - offset_x
    py = pooled[:, 1] * (height - 1) + padding - offset_y
    pooled[:, 0] = px / max(width - 1, 1)
    pooled[:, 1] = py / max(height - 1, 1)
    return np.clip(pooled, 0.0, 1.0)
