#!/usr/bin/env python3
"""Decide which SAM3 mask is the target in each frame, using gaze.

The masks come from text-prompted SAM3 (two words for the whole dataset); gaze
says which of them the operator was attending to. Neither step involves a
per-frame human label, and no pick classifier is trained.

Three rules, in the order they are applied:

* The ball's mask is dilated before the containment test. Gaze sits a median
  0.79 token cells from the ball's centre because the operator watches the
  gripper-ball contact region rather than the ball itself, so an undilated test
  would score most approach frames as misses. The dilation only widens the
  *test*; the mask written as the target keeps its original extent.
* Gaze inside neither mask defaults to the ball. Over the recordings this is
  the common case during the approach, when the gaze sits in the gap between
  gripper and ball, and the ball is what the operator is working toward.
* Gaze inside both counts as the basket. The two overlap only once the ball is
  being carried over the basket, and there the basket is the target.

Also completes the ball mask by fitting a circle to what is visible. A tennis
ball is round and its apparent size is the one monocular depth cue available,
so letting the fingers eat half the mask would corrupt exactly the signal that
the approach depends on.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

RECORDED = Path(
    "/home/ealin/workspaces/DexTacHil/data/recorded_data/tennis_ball_pick_and_place"
)
SAM_BALL = "sam_ball_mask.png"
SAM_BASKET = "sam_basket_mask.png"
OUT_BALL = "target_ball_mask.png"        # circle-completed ball
OUT_TARGET = "target_mask.png"           # the mask gaze selected, for CGL
OUT_JSON = "gaze_mask_choice.json"
PROTECTED = {"ball_mask.png", "basket_mask.png", "hand_mask.png",
             "mask1.png", "mask2.png"}
assert not ({OUT_BALL, OUT_TARGET} & PROTECTED)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", action="append", default=None,
                   help="Substring of a session directory name. Repeatable. "
                        "Defaults to the three 2026-08-25 sessions.")
    p.add_argument("--dilate_px", type=int, default=40,
                   help="Ball dilation for the containment test only, in pixels "
                        "of the 640-wide frame. 40 px is roughly one token cell "
                        "(46 px), the median gaze-to-ball distance.")
    p.add_argument("--min_ball_px", type=int, default=25,
                   help="Below this the ball mask is treated as absent rather "
                        "than as a sliver to fit a circle to.")
    p.add_argument("--complete_ball", type=int, default=1)
    p.add_argument("--latch", type=int, default=0,
                   help="Once the basket has been chosen on this many consecutive "
                        "frames, keep choosing it for the rest of the episode. Off "
                        "by default, which follows the gaze literally. The "
                        "operator's gaze genuinely alternates between the carried "
                        "ball and the basket during transport -- about six runs "
                        "per episode -- so without a latch neighbouring frames "
                        "supervise different objects, and with one the target "
                        "follows the task's single real transition instead.")
    return p.parse_args()


def circle_complete(mask, min_px):
    """Fill the ball back out to a circle fitted on the visible part.

    Returns (completed_mask, was_completed, radius_px). A ball hidden behind the
    fingers still has a visible arc, and an arc determines the circle; this is
    steadier than an amodal network and needs no training.
    """
    pixels = np.count_nonzero(mask)
    if pixels < min_px:
        return mask, False, 0.0
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask, False, 0.0
    points = np.vstack(contours).reshape(-1, 2).astype(np.float32)
    (cx, cy), radius = cv2.minEnclosingCircle(points)
    filled = np.zeros_like(mask, dtype=np.uint8)
    cv2.circle(filled, (int(round(cx)), int(round(cy))), int(round(radius)), 1, -1)
    filled = filled > 0
    # Only accept the completion when it actually adds area and stays plausible:
    # a wildly larger circle means the visible region was not an arc of the ball
    # (a stray detection, or two fragments) and the original is safer.
    if not filled.any() or np.count_nonzero(filled) > 6 * max(pixels, 1):
        return mask, False, 0.0
    return (filled | mask), True, float(radius)


def gaze_xy(frame_dir: Path):
    """Normalised gaze, with the session's calibration correction applied."""
    from serl_launcher.utils.gaze_attention_target import read_gaze_xy
    return read_gaze_xy(frame_dir)


def episodes(root: Path):
    meta = json.loads((root / "recording_metadata.json").read_text())
    for episode in meta.get("episode_ranges", []):
        if not episode.get("success"):
            continue
        frames = []
        for span in episode.get("kept_frame_ranges", []):
            frames.extend(range(int(span["start_frame"]), int(span["end_frame"]) + 1))
        if frames:
            yield int(episode["episode_index"]), sorted(frames)


def main():
    args = parse_args()
    sessions = args.session or ["2026-08-25_16-07-31", "2026-08-25_16-19-20",
                                "2026-08-25_17-28-19"]
    roots = [p for p in sorted(RECORDED.glob("tennis_ball_pick_and_place-*"))
             if any(s in p.name for s in sessions)]
    print(f"sessions: {[r.name[-19:] for r in roots]}", flush=True)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * args.dilate_px + 1,) * 2)
    tally = {"ball": 0, "basket": 0, "default_ball": 0, "both_basket": 0,
             "completed": 0, "no_ball": 0, "no_gaze": 0, "total": 0}
    radii = []
    per_episode = []

    for root in roots:
        for episode_index, frame_ids in episodes(root):
            choices = []
            basket_run = 0
            latched = False
            for frame_id in frame_ids:
                frame_dir = root / f"frame_{frame_id}"
                ball_raw = cv2.imread(str(frame_dir / SAM_BALL), cv2.IMREAD_GRAYSCALE)
                basket_raw = cv2.imread(str(frame_dir / SAM_BASKET),
                                        cv2.IMREAD_GRAYSCALE)
                if ball_raw is None or basket_raw is None:
                    continue
                tally["total"] += 1
                ball = ball_raw > 127
                basket = basket_raw > 127

                if args.complete_ball:
                    ball, completed, radius = circle_complete(ball, args.min_ball_px)
                    if completed:
                        tally["completed"] += 1
                        radii.append(radius)
                cv2.imwrite(str(frame_dir / OUT_BALL),
                            (ball.astype(np.uint8) * 255))

                xy = gaze_xy(frame_dir)
                if xy is None:
                    tally["no_gaze"] += 1
                    continue
                height, width = ball.shape
                px = int(np.clip(round(float(xy[0]) * (width - 1)), 0, width - 1))
                py = int(np.clip(round(float(xy[1]) * (height - 1)), 0, height - 1))

                ball_test = (cv2.dilate(ball.astype(np.uint8), kernel) > 0
                             if ball.any() else ball)
                in_ball = bool(ball_test[py, px])
                in_basket = bool(basket[py, px])

                if in_ball and in_basket:
                    choice = "basket"
                    tally["both_basket"] += 1
                elif in_basket:
                    choice = "basket"
                elif in_ball:
                    choice = "ball"
                else:
                    choice = "ball"
                    tally["default_ball"] += 1

                if args.latch > 0:
                    basket_run = basket_run + 1 if choice == "basket" else 0
                    if basket_run >= args.latch:
                        latched = True
                    if latched:
                        choice = "basket"
                tally[choice] += 1

                target = ball if choice == "ball" else basket
                if choice == "ball" and not ball.any():
                    tally["no_ball"] += 1
                cv2.imwrite(str(frame_dir / OUT_TARGET),
                            (target.astype(np.uint8) * 255))
                (frame_dir / OUT_JSON).write_text(json.dumps({
                    "choice": choice, "in_ball_dilated": in_ball,
                    "in_basket": in_basket, "dilate_px": args.dilate_px,
                    "gaze_px": [px, py], "ball_px": int(np.count_nonzero(ball)),
                }))
                choices.append((frame_id, choice))
            if choices:
                per_episode.append((root.name[-8:], episode_index, choices))

    total = max(tally["total"], 1)
    print(f"\n{total} frames")
    print(f"  选中球      {tally['ball']:>6}  ({tally['ball'] / total * 100:.1f}%)")
    print(f"  选中框      {tally['basket']:>6}  ({tally['basket'] / total * 100:.1f}%)")
    print(f"  其中: 落在两者之外 → 默认球   {tally['default_ball']:>6}  "
          f"({tally['default_ball'] / total * 100:.1f}%)")
    print(f"        同时落在两者 → 算框     {tally['both_basket']:>6}  "
          f"({tally['both_basket'] / total * 100:.1f}%)")
    print(f"  球 mask 补全 {tally['completed']:>6}  "
          f"({tally['completed'] / total * 100:.1f}%)"
          + (f"   半径中位 {np.median(radii):.1f} px" if radii else ""))
    print(f"  选了球但球 mask 为空 {tally['no_ball']:>6}")
    print(f"  无 gaze {tally['no_gaze']:>6}")

    # A switch that never flips, or flips many times, is the failure to look for.
    print(f"\n每个 episode 的 ball→basket 切换次数:")
    flips = []
    for name, index, choices in per_episode:
        seq = [c for _, c in choices]
        flips.append(sum(1 for a, b in zip(seq, seq[1:]) if a != b))
    flips = np.array(flips)
    print(f"  中位 {np.median(flips):.0f}   0 次的 episode {int((flips == 0).sum())}/{len(flips)}"
          f"   >4 次的 {int((flips > 4).sum())}")
    print(f"\n写入 {OUT_BALL} / {OUT_TARGET} / {OUT_JSON}")


if __name__ == "__main__":
    main()
