#!/usr/bin/env python3
"""Generate ball / basket masks for every recorded frame with SAM3 text prompts.

Two words for the whole dataset stand in for per-frame human segmentation.
Measured against the hand-drawn masks on 651 frames: pick-phase ball IoU 0.921
(median 0.99, 94.4% above 0.7) and basket 0.995. The hand is deliberately not
generated -- no prompt reached usable quality on the dexterous hand (best 0.41,
0% above 0.7) -- and it is not needed: only mask1/mask2 reach the RL agent.

Writes `sam_ball_mask.png` / `sam_basket_mask.png` alongside the recordings.
The hand-drawn `ball_mask.png` / `basket_mask.png` are never touched: they are
the ground truth every number in this pipeline is measured against, and the
operator's own labelling time. An assert enforces that, because the obvious
filename is the destructive one.

Per-frame scores and the IoU against the hand-drawn masks go to
`sam_text_inference.json`, so the 5.6% of frames where SAM3 misses entirely can
be found and filtered later rather than silently training on them.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

RECORDED = Path(
    "/home/ealin/workspaces/DexTacHil/data/recorded_data/tennis_ball_pick_and_place"
)
# The hand-drawn masks. Writing to any of these would destroy the ground truth.
PROTECTED = {"ball_mask.png", "basket_mask.png", "hand_mask.png",
             "mask1.png", "mask2.png"}
OUTPUTS = {"ball": "sam_ball_mask.png", "basket": "sam_basket_mask.png"}
assert not (set(OUTPUTS.values()) & PROTECTED), "output would overwrite hand labels"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=-1, help="Per root. -1 for all.")
    p.add_argument("--prompt_ball", default="tennis ball")
    p.add_argument("--prompt_basket", default="basket")
    p.add_argument("--confidence", type=float, default=0.3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true",
                   help="Redo frames that already have output.")
    p.add_argument("--kept_only", action="store_true", default=True,
                   help="Only frames inside kept_frame_ranges, i.e. the operator's "
                        "own gaze review.")
    return p.parse_args()


def episodes(root: Path, limit: int, kept_only: bool):
    meta = json.loads((root / "recording_metadata.json").read_text())
    count = 0
    for episode in meta.get("episode_ranges", []):
        if not episode.get("success"):
            continue
        frames = []
        spans = (episode.get("kept_frame_ranges", []) if kept_only
                 else [{"start_frame": episode["start_frame"],
                        "end_frame": episode["end_frame"]}])
        for span in spans:
            frames.extend(range(int(span["start_frame"]), int(span["end_frame"]) + 1))
        if frames:
            yield int(episode["episode_index"]), sorted(frames)
            count += 1
            if limit >= 0 and count >= limit:
                return


def iou(a, b):
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else float("nan")


def main():
    args = parse_args()
    import torch
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(device=args.device, load_from_HF=True)
    processor = Sam3Processor(model, device=args.device,
                              confidence_threshold=args.confidence)
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    prompts = {"ball": args.prompt_ball, "basket": args.prompt_basket}
    print(f"SAM3 ready. prompts={prompts}", flush=True)

    def segment(state, prompt, shape):
        with autocast:
            out = processor.set_text_prompt(prompt, state)
        masks = out.get("masks")
        if masks is None or len(masks) == 0:
            return np.zeros(shape, bool), 0.0, 0
        masks = masks.cpu().numpy()[:, 0].astype(bool)
        scores = out["scores"].float().cpu().numpy().reshape(-1)
        keep = scores >= args.confidence
        if not keep.any():
            keep = np.zeros(len(scores), bool)
            keep[int(np.argmax(scores))] = True
        # Union over kept instances: SAM3 emits one per detection while the
        # recordings store one mask per object.
        return np.any(masks[keep], axis=0), float(scores[keep].max()), int(keep.sum())

    started = time.time()
    done = skipped = 0
    stats = {"ball": [], "basket": []}
    for root in sorted(RECORDED.glob("tennis_ball_pick_and_place-*")):
        for episode_index, frame_ids in episodes(root, args.episodes, args.kept_only):
            for frame_id in frame_ids:
                frame_dir = root / f"frame_{frame_id}"
                image_path = frame_dir / "color_image.jpg"
                if not image_path.exists():
                    continue
                record_path = frame_dir / "sam_text_inference.json"
                if record_path.exists() and not args.overwrite:
                    skipped += 1
                    continue
                bgr = cv2.imread(str(image_path))
                if bgr is None:
                    continue
                shape = bgr.shape[:2]
                with autocast:
                    state = processor.set_image(
                        Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
                record = {"prompts": prompts, "confidence": args.confidence}
                for slot, prompt in prompts.items():
                    mask, score, count = segment(state, prompt, shape)
                    name = OUTPUTS[slot]
                    assert name not in PROTECTED
                    cv2.imwrite(str(frame_dir / name),
                                (mask.astype(np.uint8) * 255))
                    record[slot] = {"score": score, "instances": count,
                                    "area": float(mask.mean())}
                    truth = cv2.imread(str(frame_dir / f"{slot}_mask.png"),
                                       cv2.IMREAD_GRAYSCALE)
                    if truth is not None:
                        value = iou(mask, truth > 127)
                        record[slot]["iou_vs_hand"] = value
                        if not np.isnan(value):
                            stats[slot].append(value)
                record_path.write_text(json.dumps(record))
                done += 1
                if done % 500 == 0:
                    rate = done / max(time.time() - started, 1e-6)
                    print(f"  {done} frames  {rate:.1f} fps", flush=True)
            print(f"{root.name[-8:]} ep{episode_index}: done={done} skipped={skipped}",
                  flush=True)

    elapsed = time.time() - started
    print(f"\n生成 {done} 帧, 跳过 {skipped} 帧, 用时 {elapsed / 60:.1f} 分钟")
    for slot in ("ball", "basket"):
        values = np.array(stats[slot], np.float32)
        if not len(values):
            continue
        print(f"  {slot:<8} IoU vs 人工: 平均 {values.mean():.3f}  "
              f"中位 {np.median(values):.3f}  >0.7 {(values > 0.7).mean() * 100:.1f}%  "
              f"<0.1 {(values < 0.1).mean() * 100:.1f}%   n={len(values)}")
    print("\n人工标注未被修改:", ", ".join(sorted(PROTECTED)))


if __name__ == "__main__":
    main()
