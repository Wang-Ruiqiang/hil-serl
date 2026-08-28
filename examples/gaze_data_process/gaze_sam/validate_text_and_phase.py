#!/usr/bin/env python3
"""Two questions the gaze+SAM3 pipeline stands or falls on.

**Mask quality.** SAM3 is given only a text prompt naming each object -- two
words for the whole dataset, not a per-frame annotation -- and scored against
the recorded masks. Gaze plays no part here; this measures whether text-prompted
segmentation is good enough to train a mask predictor on, which needs roughly
IoU 0.7.

**Phase quality.** Gaze decides which of the two masks is the current target,
in place of the pick classifier. The classifier needs its own labelled
pick-success frames; gaze needs nothing.

Results are split by phase because a failure's cost depends on it. The ball
disappears behind the fingers once grasped, so a low place-phase ball IoU is
expected and harmless -- the target there is the basket. A low *pick*-phase ball
IoU is not harmless, and that is the number to read first.

Ground-truth phase comes from the recorded masks (the ball's box inside the
hand's for five consecutive frames), independently of gaze and of SAM.
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=6, help="Per root. -1 for all.")
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--prompt_ball", default="tennis ball")
    p.add_argument("--prompt_basket", default="basket")
    p.add_argument("--confidence", type=float, default=0.3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "text_phase"))
    p.add_argument("--save_vis", type=int, default=16)
    return p.parse_args()


def gaze_xy(frame_dir: Path):
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


def truth_masks(frame_dir: Path):
    out = {}
    for slot in ("ball", "hand", "basket"):
        m = cv2.imread(str(frame_dir / f"{slot}_mask.png"), cv2.IMREAD_GRAYSCALE)
        if m is None:
            return None
        out[slot] = m > 127
    return out


def iou(a, b):
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else float("nan")


def episodes(root: Path, limit: int):
    meta = json.loads((root / "recording_metadata.json").read_text())
    count = 0
    for episode in meta.get("episode_ranges", []):
        if not episode.get("success"):
            continue
        frames = []
        for span in episode.get("kept_frame_ranges", []):
            frames.extend(range(int(span["start_frame"]), int(span["end_frame"]) + 1))
        if frames:
            yield int(episode["episode_index"]), sorted(frames)
            count += 1
            if limit >= 0 and count >= limit:
                return


def grasp_onset(root: Path, frame_ids, run_length: int = 5):
    """Ball's box inside the hand's for `run_length` consecutive frames.

    The single-frame test flickers false under occlusion and would relabel
    transport frames as pre-grasp, which is exactly the split being measured.
    """
    run = 0
    for index, frame_id in enumerate(frame_ids):
        truth = truth_masks(root / f"frame_{frame_id}")
        held = False
        if truth is not None and truth["ball"].any() and truth["hand"].any():
            ys, xs = np.where(truth["ball"])
            hy, hx = np.where(truth["hand"])
            held = (xs.min() >= hx.min() - 2 and ys.min() >= hy.min() - 2
                    and xs.max() <= hx.max() + 2 and ys.max() <= hy.max() + 2)
        run = run + 1 if held else 0
        if run >= run_length:
            return frame_ids[index - run_length + 1]
    return None


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(device=args.device, load_from_HF=True)
    processor = Sam3Processor(model, device=args.device,
                              confidence_threshold=args.confidence)
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    print(f"SAM3 ready. prompts: {args.prompt_ball!r} / {args.prompt_basket!r}",
          flush=True)

    def segment(state, prompt):
        with autocast:
            out = processor.set_text_prompt(prompt, state)
        masks = out.get("masks")
        if masks is None or len(masks) == 0:
            return None, 0.0
        masks = masks.cpu().numpy()[:, 0].astype(bool)
        scores = out["scores"].float().cpu().numpy().reshape(-1)
        # Union of everything above threshold: SAM3 returns one instance per
        # detection, and the recorded masks are per-object, not per-instance.
        keep = scores >= args.confidence
        if not keep.any():
            keep = np.zeros(len(scores), bool)
            keep[int(np.argmax(scores))] = True
        return np.any(masks[keep], axis=0), float(scores[keep].max())

    rows = []
    saved = 0
    for root in sorted(RECORDED.glob("tennis_ball_pick_and_place-*")):
        for episode_index, frame_ids in episodes(root, args.episodes):
            onset = grasp_onset(root, frame_ids)
            points = {f: gaze_xy(root / f"frame_{f}") for f in frame_ids}
            points = {f: v for f, v in points.items() if v is not None}
            latched = False
            for frame_id in sorted(points):
                frame_dir = root / f"frame_{frame_id}"
                truth = truth_masks(frame_dir)
                if truth is None:
                    continue
                bgr = cv2.imread(str(frame_dir / "color_image.jpg"))
                if bgr is None:
                    continue
                height, width = bgr.shape[:2]
                gx, gy = points[frame_id]
                px = int(np.clip(round(gx * (width - 1)), 0, width - 1))
                py = int(np.clip(round(gy * (height - 1)), 0, height - 1))
                # Gaze phase decision: latch on the first frame inside the basket.
                if truth["basket"][py, px]:
                    latched = True
                phase_truth = onset is not None and frame_id >= onset
                # The ball vanishing behind the fingers is a property of the
                # scene, not of the segmenter, so it is recorded rather than
                # silently counted as a miss.
                ball_visible = int(np.count_nonzero(truth["ball"])) >= 30

                if frame_id % max(1, args.stride):
                    rows.append((phase_truth, latched, ball_visible,
                                 float("nan"), float("nan"), 0.0, 0.0))
                    continue

                with autocast:
                    state = processor.set_image(
                        Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
                ball_mask, ball_score = segment(state, args.prompt_ball)
                basket_mask, basket_score = segment(state, args.prompt_basket)
                ball_iou = iou(ball_mask, truth["ball"]) if ball_mask is not None else 0.0
                basket_iou = (iou(basket_mask, truth["basket"])
                              if basket_mask is not None else 0.0)
                rows.append((phase_truth, latched, ball_visible,
                             ball_iou, basket_iou, ball_score, basket_score))

                if saved < args.save_vis and frame_id % (args.stride * 7) == 0:
                    panel = bgr.copy()
                    for slot, colour in (("ball", (0, 255, 0)),
                                         ("basket", (255, 128, 0))):
                        contours, _ = cv2.findContours(
                            truth[slot].astype(np.uint8), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(panel, contours, -1, colour, 1)
                    for mask, colour in ((ball_mask, (255, 255, 255)),
                                         (basket_mask, (0, 200, 255))):
                        if mask is None:
                            continue
                        contours, _ = cv2.findContours(
                            mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(panel, contours, -1, colour, 2)
                    cv2.circle(panel, (px, py), 7, (0, 0, 255), 2)
                    cv2.putText(panel,
                                f"{'PLACE' if phase_truth else 'PICK'} "
                                f"gaze->{'basket' if latched else 'ball'}  "
                                f"ballIoU={ball_iou:.2f} basketIoU={basket_iou:.2f}",
                                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (255, 255, 255), 2)
                    cv2.imwrite(str(out_dir / f"{root.name[-8:]}_ep{episode_index:02d}"
                                    f"_f{frame_id}.jpg"), panel)
                    saved += 1
            print(f"  {root.name[-8:]} ep{episode_index}: {len(rows)} rows", flush=True)

    if not rows:
        print("no frames")
        return
    table = np.array(rows, dtype=np.float32)
    phase = table[:, 0] > 0.5
    latch = table[:, 1] > 0.5
    visible = table[:, 2] > 0.5
    scored = ~np.isnan(table[:, 3])

    print(f"\n{int(scored.sum())} 帧跑了 SAM3 (共 {len(table)} 帧参与阶段统计)\n")
    print("=== 文本提示的 mask 质量 (对人工 mask 的 IoU) ===")
    print(f"{'':<26}{'球':>10}{'框':>10}{'帧数':>8}")
    for name, sel in (("pick 阶段", scored & ~phase),
                      ("place 阶段 (球可见)", scored & phase & visible),
                      ("place 阶段 (球被遮挡)", scored & phase & ~visible),
                      ("全部", scored)):
        s = table[sel]
        if not len(s):
            continue
        print(f"{name:<24}{np.nanmean(s[:, 3]):>10.3f}{np.nanmean(s[:, 4]):>10.3f}"
              f"{len(s):>8}")
    good = scored & ~phase
    if good.any():
        b = table[good][:, 3]
        print(f"\npick 阶段球 IoU 分布: "
              + " / ".join(f"{v:.2f}" for v in np.percentile(b, [10, 25, 50, 75, 90]))
              + "  (10/25/50/75/90 分位)")
        print(f"  IoU>0.7 的帧: {(b > 0.7).mean() * 100:.1f}%   "
              f"IoU>0.5: {(b > 0.5).mean() * 100:.1f}%   IoU<0.1: {(b < 0.1).mean() * 100:.1f}%")

    print("\n=== gaze 作为阶段判据 (对照真实抓取时刻) ===")
    tp = int((latch & phase).sum()); fp = int((latch & ~phase).sum())
    fn = int((~latch & phase).sum()); tn = int((~latch & ~phase).sum())
    print(f"  准确率 {((latch == phase).mean()) * 100:.1f}%")
    print(f"  pick 阶段判对 {tn / max(tn + fp, 1) * 100:.1f}%  "
          f"(误判成 place {fp}/{fp + tn})")
    print(f"  place 阶段判对 {tp / max(tp + fn, 1) * 100:.1f}%  "
          f"(滞后造成 {fn}/{tp + fn})")
    json.dump({"rows": table.tolist()}, open(out_dir / "rows.json", "w"))
    print(f"\n可视化 -> {out_dir}")


if __name__ == "__main__":
    main()
