#!/usr/bin/env python3
"""Build ViT attention targets from recorded human gaze.

Replaces the mask-derived CGL target: instead of `front_camera_mask1`, the
grounding query is scored against a distribution pooled from the operator's
gaze over a short temporal window. A single gaze sample is a delta function --
it says "look here" but not "the object is this big" -- so pooling neighbouring
frames is what gives the target its extent.

The target lives on the ViT token grid, not on the image grid, because that is
what the CGL loss actually compares against. Sigma is therefore specified in
token cells: at 256/16 = 16x16, one cell spans 16 input pixels, and the ball is
roughly one cell across.

Writes one npz per episode plus optional review videos, and reports how the
pooled mass distributes over the recorded ball / basket / hand masks so the
pick-vs-place separation can be checked before any training runs.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from absl import app, flags

from serl_launcher.utils.gaze_attention_target import (
    DEFAULT_DECAY,
    DEFAULT_DILATE_CELLS,
    DEFAULT_MAX_GAP,
    DEFAULT_SIGMA_CELLS,
    DEFAULT_SUPERSAMPLE,
    DEFAULT_WINDOW,
    build_target_heatmap,
    pool_gaze_points,
    read_gaze_xy,
)

FLAGS = flags.FLAGS

# ============================== 编辑这里 ==============================
_RECORDED = "/home/ealin/workspaces/DexTacHil/data/recorded_data/tennis_ball_pick_and_place"
DEFAULT_ROOTS = [
    f"{_RECORDED}/tennis_ball_pick_and_place-2026-08-14_12-18-59",
    f"{_RECORDED}/tennis_ball_pick_and_place-2026-08-14_12-49-48",
]
# =====================================================================

DEFAULT_OUT = str(Path(__file__).resolve().parent / "gaze_attention_targets")

flags.DEFINE_multi_string("root", DEFAULT_ROOTS, "Recording root. Repeatable.")
flags.DEFINE_string("out_dir", DEFAULT_OUT, "Where to write targets and videos.")
flags.DEFINE_integer("grid", 14, "ViT token grid size. 14 matches the deployed vit_image_size=(224,224) with patch 16.")
flags.DEFINE_float(
    "sigma_cells", DEFAULT_SIGMA_CELLS,
    "Gaussian sigma in token cells. This, not the pooling window, is what "
    "controls the target's extent. 0.6 is the label-noise floor: gaze labels "
    "carry ~4.4 px of error at 128x128, which is 0.55 cells at 256/patch-16, "
    "so a tighter target would be fitting noise. Measured contrast against "
    "chance at 0.6 / 0.9 / 1.3 cells: ball 13x / 12x / 8x, and pick-vs-place "
    "basket separation 29x / 12x / 6x -- sharper is better on every axis "
    "until the noise floor.")
flags.DEFINE_integer(
    "window", DEFAULT_WINDOW,
    "Pool +/- this many gaze frames (8.3 Hz => 5 ~ 0.6 s). This buys "
    "robustness, not extent: sweeping 0 -> 8 moves the mass on every object by "
    "less than 0.003, because over 0.6 s the gaze barely crosses a token cell. "
    "What it does buy is that one drifted gaze sample becomes a wrong delta at "
    "window 0 but gets outvoted at window 5. Keep it small -- the operator "
    "first looks at the basket a median of 45 frames after the grasp, so a "
    "window anywhere near that would bleed place-phase mass into pick frames.")
flags.DEFINE_integer("max_gap", DEFAULT_MAX_GAP, "Stop pooling across a hole larger than this.")
flags.DEFINE_float("decay", DEFAULT_DECAY, "Exponential weight decay per pooled neighbour.")
flags.DEFINE_float(
    "cell_threshold", 0.04,
    "Mask occupancy above which a token cell counts as covered. Matches "
    "mask_grounding_cell_threshold in the agent so the diagnostic below "
    "measures what the CGL loss would actually see.")
flags.DEFINE_boolean("success_only", True, "Use only successful episodes.")
flags.DEFINE_integer("vis_episodes", 6, "Render review videos for the first N episodes. -1 for all.")
flags.DEFINE_float(
    "dilate_cells", DEFAULT_DILATE_CELLS,
    "Grow the pooled region outward by this many token cells before "
    "normalising. Gaze lands a median of 0.77 cells from the ball during the "
    "pick -- roughly one ball-diameter away -- because the operator looks at "
    "the contact region rather than the ball's centre, so an undilated target "
    "scores that as a miss. This is a morphological dilation, not a larger "
    "sigma: it makes everything within R equally correct (a plateau), where a "
    "wider Gaussian would instead keep insisting the exact centre matters "
    "most. Sigma stays as the soft edge beyond the plateau.")
flags.DEFINE_integer(
    "supersample", DEFAULT_SUPERSAMPLE,
    "Build the target at this multiple of the token grid, then area-downsample. "
    "Lets sigma and dilate_cells take sub-cell values, which they must: one "
    "cell spans 40 px of the 640-wide frame and the ball is only 0.85 cells "
    "across.")
flags.DEFINE_boolean("write_npz", True, "Write the per-episode target arrays.")


def _episodes(root: Path):
    meta = json.loads((root / "recording_metadata.json").read_text())
    for episode in meta["episode_ranges"]:
        if FLAGS.success_only and not episode.get("success"):
            continue
        frames = []
        for span in episode.get("kept_frame_ranges", []):
            frames.extend(range(int(span["start_frame"]), int(span["end_frame"]) + 1))
        if frames:
            yield int(episode["episode_index"]), sorted(frames)


def _cells(mask_path: Path, grid: int):
    """Downsample a mask to token-cell occupancy, then threshold."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    occupancy = cv2.resize(
        (mask > 127).astype(np.float32), (grid, grid), interpolation=cv2.INTER_AREA)
    return occupancy > FLAGS.cell_threshold


def _grasp_onset(root: Path, frame_ids, run_length: int = 5):
    """First frame of the first sustained run where the ball sits inside the hand.

    The per-frame containment test alone is not usable: it flickers false
    whenever the ball is occluded or the hand mask fragments, which relabels
    transport frames as pre-grasp and destroys the pick/place split. Requiring
    a run of `run_length` frames removes that.
    """
    run = 0
    for index, frame_id in enumerate(frame_ids):
        frame_dir = root / f"frame_{frame_id}"
        ball = cv2.imread(str(frame_dir / "ball_mask.png"), cv2.IMREAD_GRAYSCALE)
        hand = cv2.imread(str(frame_dir / "hand_mask.png"), cv2.IMREAD_GRAYSCALE)
        held = False
        if ball is not None and hand is not None:
            ys, xs = np.where(ball > 127)
            hy, hx = np.where(hand > 127)
            if xs.size and hx.size:
                held = (xs.min() >= hx.min() - 2 and ys.min() >= hy.min() - 2
                        and xs.max() <= hx.max() + 2 and ys.max() <= hy.max() + 2)
        run = run + 1 if held else 0
        if run >= run_length:
            return frame_ids[index - run_length + 1]
    return None


def _overlay(image, heatmap, pooled, weights):
    height, width = image.shape[:2]
    upsampled = cv2.resize(
        heatmap / max(heatmap.max(), 1e-8), (width, height), interpolation=cv2.INTER_NEAREST)
    colour = cv2.applyColorMap((upsampled * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    grey = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(grey, 0.45, colour, 0.55, 0)
    for point, weight in zip(pooled, weights):
        centre = (int(point[0] * (width - 1)), int(point[1] * (height - 1)))
        cv2.circle(blended, centre, 3, (255, 255, 255), 1, cv2.LINE_AA)
    centre = (int(pooled[0][0] * (width - 1)), int(pooled[0][1] * (height - 1)))
    cv2.circle(blended, centre, 6, (0, 255, 255), 2, cv2.LINE_AA)
    return blended


def _contours(image, frame_dir):
    canvas = image.copy()
    for name, colour in (("ball_mask.png", (0, 255, 0)),
                         ("basket_mask.png", (255, 128, 0)),
                         ("hand_mask.png", (0, 255, 255))):
        mask = cv2.imread(str(frame_dir / name), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        contours, _ = cv2.findContours(
            (mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, colour, 1)
    return canvas


def _writers(stem: Path, width: int, height: int, fps: float):
    """mp4 for archival, webm because that is what plays on the workstation."""
    import imageio.v2 as imageio
    return [
        imageio.get_writer(str(stem.with_suffix(".mp4")), fps=fps,
                           codec="libx264", quality=8,
                           macro_block_size=1, ffmpeg_log_level="error"),
        imageio.get_writer(str(stem.with_suffix(".webm")), fps=fps,
                           codec="libvpx-vp9", quality=8,
                           macro_block_size=1, ffmpeg_log_level="error"),
    ]


def main(_):
    out_dir = Path(FLAGS.out_dir)
    vis_dir = out_dir / "review"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    grid = FLAGS.grid
    rows = []
    rendered = 0

    for root_str in FLAGS.root:
        root = Path(root_str)
        for episode_index, frame_ids in _episodes(root):
            points = {}
            for frame_id in frame_ids:
                xy = read_gaze_xy(root / f"frame_{frame_id}")
                if xy is not None:
                    points[frame_id] = xy
            if not points:
                continue

            onset = _grasp_onset(root, sorted(points))
            render = FLAGS.vis_episodes < 0 or rendered < FLAGS.vis_episodes
            writers = None
            targets, kept_ids, stats = [], [], []

            for frame_id in sorted(points):
                frame_dir = root / f"frame_{frame_id}"
                pooled, weights = pool_gaze_points(
                    frame_id, points, FLAGS.window, FLAGS.max_gap, FLAGS.decay)
                heatmap = build_target_heatmap(
                    pooled, weights, (grid, grid), FLAGS.sigma_cells,
                    FLAGS.dilate_cells, FLAGS.supersample)
                targets.append(heatmap)
                kept_ids.append(frame_id)

                ball = _cells(frame_dir / "ball_mask.png", grid)
                basket = _cells(frame_dir / "basket_mask.png", grid)
                hand = _cells(frame_dir / "hand_mask.png", grid)
                # Effective support: 1/sum(p^2), the number of cells the target
                # spreads over. A near-delta (~1) ignores object extent and is
                # what a single un-pooled gaze sample gives; a diffuse blob
                # carries no localisation. This is the knob sigma and window
                # trade off against each other.
                support = 1.0 / float(np.sum(heatmap ** 2) + 1e-12)
                stats.append((
                    float(heatmap[ball].sum()) if ball is not None else np.nan,
                    float(heatmap[basket].sum()) if basket is not None else np.nan,
                    float(heatmap[hand].sum()) if hand is not None else np.nan,
                    len(pooled), support,
                ))

                if render:
                    image = cv2.imread(str(frame_dir / "color_image.jpg"))
                    if image is None:
                        continue
                    panel = np.hstack([
                        _contours(image, frame_dir),
                        _overlay(image, heatmap, pooled, weights),
                    ])
                    phase = "PLACE" if (onset is not None and frame_id >= onset) else "PICK"
                    cv2.putText(panel, f"ep{episode_index} f{frame_id} "
                                f"pooled={len(pooled)} {phase}",
                                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 1, cv2.LINE_AA)
                    if writers is None:
                        writers = _writers(
                            vis_dir / f"{root.name}_ep{episode_index:02d}",
                            panel.shape[1], panel.shape[0], 8.3)
                    for writer in writers:
                        writer.append_data(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))

            if writers is not None:
                for writer in writers:
                    writer.close()
                rendered += 1

            targets = np.stack(targets)
            if FLAGS.write_npz:
                np.savez_compressed(
                    out_dir / f"{root.name}_ep{episode_index:02d}.npz",
                    frame_ids=np.asarray(kept_ids, np.int32),
                    targets=targets,
                    grid=grid, sigma_cells=FLAGS.sigma_cells,
                    dilate_cells=FLAGS.dilate_cells,
                    window=FLAGS.window, decay=FLAGS.decay)

            stats = np.asarray(
                [(a, b, c, d, float(e)) for a, b, c, d, e in stats], np.float32)
            ordered = sorted(points)
            after = np.array([onset is not None and f >= onset for f in ordered])
            rows.append((root.name, episode_index, stats, after))

    if not rows:
        print("no episodes found")
        return

    every = np.concatenate([s for _, _, s, _ in rows])
    after = np.concatenate([a for _, _, _, a in rows])
    print(f"\n{len(rows)} episodes, {len(every)} frames with gaze")
    print(f"pooled points per frame: mean {every[:, 3].mean():.1f} "
          f"of {2 * FLAGS.window + 1} possible")
    print(f"effective support: mean {every[:, 4].mean():.1f} cells "
          f"of {grid * grid} (median {np.median(every[:, 4]):.1f})")
    print(f"\ntarget mass on each object (grid {grid}x{grid}, "
          f"sigma {FLAGS.sigma_cells} cells, dilate {FLAGS.dilate_cells} cells, "
          f"window +/-{FLAGS.window}, cell_threshold {FLAGS.cell_threshold})")
    print(f"{'phase':<12}{'frames':>8}{'ball':>9}{'basket':>9}{'hand':>9}")
    for name, selection in (("pick", ~after), ("place", after),
                            ("all", np.ones_like(after))):
        subset = every[selection]
        if not len(subset):
            continue
        print(f"{name:<12}{len(subset):>8}{np.nanmean(subset[:, 0]):>9.3f}"
              f"{np.nanmean(subset[:, 1]):>9.3f}{np.nanmean(subset[:, 2]):>9.3f}")
    print(f"\ntargets -> {out_dir}\nreview videos -> {vis_dir}")


if __name__ == "__main__":
    app.run(main)
