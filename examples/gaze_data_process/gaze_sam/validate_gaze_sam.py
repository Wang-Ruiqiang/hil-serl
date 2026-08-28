#!/usr/bin/env python3
"""Can SAM3 recover the task object from the gaze point alone?

This is the feasibility check for replacing hand-drawn masks with gaze-prompted
segmentation. Gaze is the ONLY input: no text prompt naming the objects, no
stored mask, nothing that a human labelled per frame. The recorded ball / hand /
basket masks are read solely to score the result, never to produce it -- if they
reached the prompt, the experiment would be measuring nothing.

Why this is worth measuring rather than assuming: the operator does not look at
the ball. Over the 41 recorded episodes their gaze falls inside the ball on
12.7% of pick-phase frames, inside the hand on 31.1%, and on neither on 56.4%,
sitting a median 0.68 cells from the ball and 0.51 from the hand -- usually in
the gap between gripper and ball, and sometimes on a tape marker stuck to the
table. A box centred there contains the gripper more centrally than the ball,
and the gripper is 22x the ball's area, so the obvious prompt has an obvious way
to fail.

The stratified table is the point. An aggregate IoU hides the only number that
decides feasibility: what happens on the 56.4% of frames where the gaze is on
neither object. `oracle` separates two very different failures -- SAM never
proposing the right mask, versus proposing it and the selection rule picking
another.
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
SLOTS = ("ball", "hand", "basket")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", action="append", default=None,
                   help="Recording root. Repeatable. Defaults to both recorded sessions.")
    p.add_argument("--episodes", type=int, default=3,
                   help="Episodes per root. -1 for all.")
    p.add_argument("--stride", type=int, default=10, help="Frame subsampling.")
    p.add_argument("--box_sizes", default="0.08,0.15,0.25",
                   help="Comma separated box side, as a fraction of image width.")
    p.add_argument("--pool_window", type=int, default=0,
                   help="Also prompt with a box covering the gaze of +/- this many "
                        "neighbouring frames, which is what makes the prompt robust "
                        "to a single drifted sample rather than merely wider.")
    p.add_argument("--no_points", action="store_true",
                   help="Skip point prompts. They behave nothing like box prompts -- "
                        "a box always returns something plausible and usually wrong, "
                        "while a point either lands on the object and segments it "
                        "cleanly or returns something unrelated -- so both belong in "
                        "the comparison.")
    p.add_argument("--confidence", type=float, default=0.3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "validation"))
    p.add_argument("--save_vis", type=int, default=12,
                   help="Render this many frames per strategy for eyeballing.")
    return p.parse_args()


def gaze_xy(frame_dir: Path):
    """Normalised gaze, or None. The only per-frame input this script may use."""
    path = frame_dir / "gaze_contact.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    uv, size = data.get("gaze_uv_in_realsense"), data.get("realsense_size")
    if uv is None or size is None or not size[0] or not size[1]:
        return None
    return float(uv[0]) / float(size[0]), float(uv[1]) / float(size[1])


def load_truth(frame_dir: Path):
    """Recorded masks. SCORING ONLY -- must never reach a prompt."""
    truth = {}
    for slot in SLOTS:
        mask = cv2.imread(str(frame_dir / f"{slot}_mask.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        truth[slot] = mask > 127
    return truth


def iou(a, b):
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 0.0


def episodes(root: Path, limit: int):
    meta = json.loads((root / "recording_metadata.json").read_text())
    count = 0
    for episode in meta.get("episode_ranges", []):
        if not episode.get("success"):
            continue
        frames = []
        for span in episode.get("kept_frame_ranges", []):
            frames.extend(range(int(span["start_frame"]), int(span["end_frame"]) + 1))
        if not frames:
            continue
        yield int(episode["episode_index"]), sorted(frames)
        count += 1
        if limit >= 0 and count >= limit:
            return


def build_processor(args):
    import torch
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    # enable_inst_interactivity builds the SAM2-style mask decoder that
    # model.predict_inst needs; without it only box prompts are reachable.
    model = build_sam3_image_model(device=args.device, load_from_HF=True,
                                   enable_inst_interactivity=True)
    processor = Sam3Processor(model, device=args.device,
                              confidence_threshold=args.confidence)
    # SAM3 runs its trunk in bfloat16; without this the first matmul raises a
    # dtype mismatch rather than falling back.
    autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if str(args.device).startswith("cuda")
                else torch.autocast(device_type="cpu", enabled=False))
    return processor, autocast, model


def segment_points(model, autocast, state, points):
    """Positive point prompts, in pixels. Returns (masks, scores)."""
    coords = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    labels = np.ones((len(coords),), dtype=np.int32)
    with autocast:
        masks, scores, _ = model.predict_inst(
            state, point_coords=coords, point_labels=labels, multimask_output=True)
    masks = np.asarray(masks)
    if masks.ndim == 4:
        masks = masks[:, 0]
    masks = masks if masks.dtype == bool else masks > 0.0
    return masks, np.asarray(scores, dtype=np.float32).reshape(-1)


def segment(processor, autocast, state, box):
    """Box is (cx, cy, w, h), normalised. Returns (masks, scores)."""
    processor.reset_all_prompts(state)
    with autocast:
        out = processor.add_geometric_prompt(list(box), True, state)
    masks = out["masks"]
    if masks is None or len(masks) == 0:
        return np.zeros((0, 1, 1), bool), np.zeros((0,), np.float32)
    masks = masks.cpu().numpy()[:, 0].astype(bool)
    scores = out["scores"].float().cpu().numpy()
    return masks, scores


def main():
    args = parse_args()
    roots = [Path(r) for r in (args.root or [])]
    if not roots:
        roots = sorted(RECORDED.glob("tennis_ball_pick_and_place-*"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = [float(s) for s in args.box_sizes.split(",") if s.strip()]

    from PIL import Image
    processor, autocast, model = build_processor(args)
    print(f"SAM3 ready. roots={[r.name for r in roots]}", flush=True)

    rows = []
    saved = {}
    for root in roots:
        for episode_index, frame_ids in episodes(root, args.episodes):
            points = {}
            for frame_id in frame_ids:
                xy = gaze_xy(root / f"frame_{frame_id}")
                if xy is not None:
                    points[frame_id] = xy
            for frame_id in sorted(points)[:: max(1, args.stride)]:
                frame_dir = root / f"frame_{frame_id}"
                truth = load_truth(frame_dir)
                if truth is None or not truth["ball"].any():
                    continue
                bgr = cv2.imread(str(frame_dir / "color_image.jpg"))
                if bgr is None:
                    continue
                height, width = bgr.shape[:2]
                gx, gy = points[frame_id]
                px, py = int(gx * (width - 1)), int(gy * (height - 1))
                # Where the gaze landed decides which stratum this frame scores in.
                if truth["ball"][py, px]:
                    stratum = "on_ball"
                elif truth["hand"][py, px]:
                    stratum = "on_hand"
                elif truth["basket"][py, px]:
                    stratum = "on_basket"
                else:
                    stratum = "neither"

                with autocast:
                    state = processor.set_image(
                        Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))

                prompts = [(f"box{size:g}", (gx, gy, size, size * width / height))
                           for size in sizes]
                if args.pool_window > 0:
                    window = [points[f] for f in range(frame_id - args.pool_window,
                                                       frame_id + args.pool_window + 1)
                              if f in points]
                    if len(window) > 1:
                        xs = [p[0] for p in window]
                        ys = [p[1] for p in window]
                        pad = 0.04
                        prompts.append((
                            f"pooled{args.pool_window}",
                            ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2,
                             max(max(xs) - min(xs), 0.02) + pad,
                             max(max(ys) - min(ys), 0.02) + pad)))

                jobs = [(name, "box", box) for name, box in prompts]
                if not args.no_points:
                    jobs.append(("point", "points", [(px, py)]))
                    if args.pool_window > 0:
                        neighbours = [
                            points[f] for f in range(frame_id - args.pool_window,
                                                     frame_id + args.pool_window + 1)
                            if f in points]
                        if len(neighbours) > 1:
                            jobs.append((f"points{args.pool_window}", "points",
                                         [(x * (width - 1), y * (height - 1))
                                          for x, y in neighbours]))

                for name, kind, payload in jobs:
                    if kind == "points":
                        masks, scores = segment_points(model, autocast, state, payload)
                    else:
                        masks, scores = segment(processor, autocast, state, payload)
                    if len(masks) == 0:
                        rows.append((name, stratum, 0.0, 0.0, "none", 0.0))
                        continue
                    per_candidate = {
                        slot: np.array([iou(m, truth[slot]) for m in masks])
                        for slot in SLOTS
                    }
                    top = int(np.argmax(scores))
                    best_slot_top = max(SLOTS, key=lambda s: per_candidate[s][top])
                    # oracle: the best any candidate could have scored against the
                    # ball. Separates "SAM never proposed it" from "we picked wrong".
                    oracle_ball = float(per_candidate["ball"].max())
                    rows.append((name, stratum,
                                 float(per_candidate["ball"][top]),
                                 oracle_ball, best_slot_top,
                                 float(masks[top].mean())))
                    key = (name, stratum)
                    if len(saved.get(key, [])) < max(0, args.save_vis) // 4:
                        panel = bgr.copy()
                        for slot, colour in (("ball", (0, 255, 0)),
                                             ("hand", (0, 255, 255)),
                                             ("basket", (255, 128, 0))):
                            contours, _ = cv2.findContours(
                                truth[slot].astype(np.uint8), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(panel, contours, -1, colour, 1)
                        contours, _ = cv2.findContours(
                            masks[top].astype(np.uint8), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(panel, contours, -1, (255, 255, 255), 2)
                        cv2.circle(panel, (px, py), 7, (0, 0, 255), 2)
                        cv2.putText(panel, f"{name} {stratum} "
                                    f"ballIoU={per_candidate['ball'][top]:.2f}",
                                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (255, 255, 255), 2)
                        saved.setdefault(key, []).append(panel)
            print(f"  {root.name} ep{episode_index}: {len(rows)} rows", flush=True)

    if not rows:
        print("no scorable frames")
        return

    names = sorted({r[0] for r in rows})
    strata = ["on_ball", "on_hand", "on_basket", "neither"]
    print(f"\n{len(rows)} 次分割\n")
    print("每格: 球的 IoU (top-score 候选)  /  oracle (最好的候选)")
    header = f"{'策略':<12}" + "".join(f"{s:>18}" for s in strata) + f"{'全部':>18}"
    print(header)
    for name in names:
        line = f"{name:<12}"
        for stratum in strata + [None]:
            subset = [r for r in rows
                      if r[0] == name and (stratum is None or r[1] == stratum)]
            if not subset:
                line += f"{'—':>18}"
                continue
            top = np.mean([r[2] for r in subset])
            oracle = np.mean([r[3] for r in subset])
            line += f"{top:>8.3f} /{oracle:>7.3f}"
        print(line)

    print(f"\n各分层的帧数: " + "  ".join(
        f"{s}={sum(1 for r in rows if r[1] == s and r[0] == names[0])}" for s in strata))
    print(f"\ntop-score 候选最像哪个物体:")
    for name in names:
        subset = [r for r in rows if r[0] == name]
        counts = {s: sum(1 for r in subset if r[4] == s) for s in SLOTS + ("none",)}
        total = max(len(subset), 1)
        print(f"  {name:<12}" + "  ".join(
            f"{k}={v / total * 100:.0f}%" for k, v in counts.items()))

    for (name, stratum), panels in saved.items():
        for index, panel in enumerate(panels):
            cv2.imwrite(str(out_dir / f"{name}_{stratum}_{index}.jpg"), panel)
    json.dump([{"strategy": r[0], "stratum": r[1], "ball_iou_top": r[2],
                "ball_iou_oracle": r[3], "top_matches": r[4], "top_area": r[5]}
               for r in rows], open(out_dir / "rows.json", "w"), indent=1)
    print(f"\n可视化和明细 -> {out_dir}")


if __name__ == "__main__":
    main()
