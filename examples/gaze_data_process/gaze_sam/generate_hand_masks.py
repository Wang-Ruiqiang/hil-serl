#!/usr/bin/env python3
"""Segment whatever the operator was looking at, from a point prompt.

The prompt is the recorded gaze pixel, so the output is "the object under the
fixation" rather than one fixed category. That serves two needs at once: on the
frames where gaze rests on the hand it is a hand mask for recordings that were
never hand-labelled, and on the frames where gaze falls between the fingers and
the ball -- 28% of them -- it is a real segment of the contact region, which is
what the third grounding query needs instead of a Gaussian blob.

SAM3's text prompts cannot segment this hand: measured against the manual masks,
the best prompt reached IoU 0.41 with 0% of frames above 0.7, while the same
mechanism gives 0.921 on the ball and 0.995 on the basket. A black multi-finger
hand against clutter is simply not what a text encoder is good at.

A point prompt is a different question -- "the object at this pixel" -- and the
pixel can be supplied without any annotation: a linear probe on the pretrained
ViT's own output vector predicts the hand centroid to ~1.4 px on the 2026-08-14
recordings, where manual masks exist to fit and check it. This script fits that
probe on 08-14, uses it to place a point on every frame of a target recording,
and asks SAM3 for the mask under that point.

--validate runs the whole chain on 08-14 itself and reports IoU against the
manual masks, which is the only honest way to decide whether the output is
usable before it is trained on.

Writes `sam_hand_mask.png`. The manual masks are never touched; an assert
enforces it, because the obvious filename is the destructive one.
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
RECORDED = Path("/home/ealin/workspaces/DexTacHil/data/recorded_data/tennis_ball_pick_and_place")
PROBE_SESSIONS = ["tennis_ball_pick_and_place-2026-08-14_12-18-59",
                  "tennis_ball_pick_and_place-2026-08-14_12-49-48"]
PROTECTED = {"ball_mask.png", "basket_mask.png", "hand_mask.png",
             "mask1.png", "mask2.png"}
OUTPUT = "sam_hand_mask.png"
assert OUTPUT not in PROTECTED, "output would overwrite hand labels"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", action="append", dest="roots", default=None,
                   help="Recording root to segment. Repeatable.")
    p.add_argument("--encoder",
                   default=str(REPO / "examples/encoder_training/runs/"
                              "tennis_ball_pick_and_place_vit_grounded_phase/best.msgpack"),
                   help="Encoder whose 256-D output the hand probe reads.")
    p.add_argument("--point_source", choices=("gaze", "probe"), default="probe",
                   help="gaze: prompt with the recorded fixation, so the mask is "
                        "whatever the operator was looking at. probe: prompt with "
                        "a learned hand-centroid prediction, which always targets "
                        "the hand but ignores where the operator actually looked.")
    p.add_argument("--validate", action="store_true",
                   help="Run on the 08-14 recordings and score IoU against the "
                        "manual hand masks instead of writing anything.")
    p.add_argument("--validate_n", type=int, default=200)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--confidence", type=float, default=0.3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_image(frame: Path, size=128):
    img = cv2.imread(str(frame / "color_image.jpg"))
    if img is None:
        return None
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)[:, :, ::-1]


def hand_centroid(frame: Path, size=128):
    m = cv2.imread(str(frame / "hand_mask.png"), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST) > 127
    if m.sum() < 4:
        return None
    ys, xs = np.nonzero(m)
    return np.array([xs.mean(), ys.mean()], np.float32)


def gaze_pixel(frame: Path, w: int, h: int):
    """The recorded fixation in full-resolution pixels, or None."""
    path = frame / "gaze_contact.json"
    if not path.exists():
        return None
    try:
        g = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    uv = g.get("gaze_uv_in_realsense")
    if not g.get("hit", False) or uv is None:
        return None
    gw, gh = g.get("realsense_size", [640, 480])
    return np.array([uv[0] / gw * w, uv[1] / gh * h], np.float32)


def build_embedder(encoder_path):
    import jax, jax.numpy as jnp, flax
    from serl_launcher.vision.vit import ViTImageEncoder
    enc = flax.serialization.msgpack_restore(open(encoder_path, "rb").read())["params"]
    pd = int(np.asarray(enc["grounding_query"]["query"]).shape[0])
    pt = int(np.asarray(enc["patch_embedding"]["kernel"]).shape[0])
    side = int(round((np.asarray(enc["pos_embedding"]).shape[1] - 1) ** 0.5)) * pt
    model = ViTImageEncoder(
        image_size=(side, side), patch_size=(pt, pt),
        hidden_dim=int(np.asarray(enc["patch_embedding"]["kernel"]).shape[-1]),
        num_layers=sum(1 for k in enc if k.startswith("encoder_block_")),
        num_heads=int(np.asarray(enc["grounding_query"]["key_projection"]["bias"]).shape[0]),
        mlp_dim=int(np.asarray(enc["encoder_block_0"]["mlp"]["Dense_0"]["kernel"]).shape[-1]),
        bottleneck_dim=256, pooling_method="spatial_learned_embeddings",
        num_spatial_blocks=8, use_grounding_query=True, grounding_phase_dim=pd)

    def embed(images, bs=64):
        out = []
        for i in range(0, len(images), bs):
            c = jnp.asarray(images[i:i + bs])
            kw = {}
            if pd > 1:
                ph = np.zeros((c.shape[0], 2), np.float32); ph[:, 0] = 1.0
                kw["phase"] = jnp.asarray(ph)
            out.append(np.asarray(model.apply({"params": enc}, c, train=False, **kw)))
        return np.concatenate(out)
    return embed


def fit_probe(embed, stride=8):
    imgs, cents = [], []
    for s in PROBE_SESSIONS:
        root = RECORDED / s
        frames = sorted(root.glob("frame_*"), key=lambda p: int(p.name.split("_")[1]))
        for f in frames[::stride]:
            c = hand_centroid(f)
            im = load_image(f)
            if c is None or im is None:
                continue
            imgs.append(im); cents.append(c)
    X = embed(np.stack(imgs)); Y = np.stack(cents)
    A = np.c_[X, np.ones(len(X))]
    W = np.linalg.solve(A.T @ A + 1e-2 * np.eye(A.shape[1]), A.T @ Y)
    err = np.linalg.norm(A @ W - Y, axis=1)
    print(f"[probe] fitted on {len(X)} frames, median error {np.median(err):.2f}px")
    return W


def main():
    args = parse_args()
    import torch
    sys.path.insert(0, "/home/ealin/workspaces/sam3")
    from validate_gaze_sam import build_processor, segment_points  # same dir

    embed = W = None
    if args.point_source == "probe":
        embed = build_embedder(args.encoder)
        W = fit_probe(embed)
    processor, autocast, model = build_processor(args)

    roots = ([RECORDED / s for s in PROBE_SESSIONS] if args.validate
             else [Path(r) if os.path.isabs(r) else RECORDED / r for r in (args.roots or [])])
    if not roots:
        raise SystemExit("give --root or --validate")

    ious, done, empty = [], 0, 0
    for root in roots:
        frames = sorted(root.glob("frame_*"), key=lambda p: int(p.name.split("_")[1]))
        if args.validate:
            frames = frames[::max(1, len(frames) // args.validate_n)][:args.validate_n]
        else:
            frames = frames[::args.stride]
        for f in frames:
            out_path = f / OUTPUT
            assert out_path.name not in PROTECTED
            if not args.validate and out_path.exists() and not args.overwrite:
                continue
            full = cv2.imread(str(f / "color_image.jpg"))
            if full is None:
                continue
            h, w = full.shape[:2]
            if args.point_source == "gaze":
                px = gaze_pixel(f, w, h)
                if px is None:
                    continue
            else:
                im = load_image(f)
                if im is None:
                    continue
                xy = np.r_[embed(im[None])[0], 1.0] @ W
                px = np.array([xy[0] / 128.0 * w, xy[1] / 128.0 * h], np.float32)
            from PIL import Image
            state = processor.set_image(Image.fromarray(full[:, :, ::-1]))
            masks, scores = segment_points(model, autocast, state, [px])
            if len(masks) == 0:
                empty += 1
                continue
            mask = masks[int(np.argmax(scores))]
            if args.validate:
                gt = cv2.imread(str(f / "hand_mask.png"), cv2.IMREAD_GRAYSCALE)
                if gt is None:
                    continue
                gt = gt > 127
                if args.point_source == "gaze":
                    # Only score frames where the fixation was actually on the
                    # hand. Elsewhere the prompt points at the ball or the table,
                    # and the hand mask is not the right answer to compare to.
                    gy, gx = int(np.clip(px[1], 0, gt.shape[0] - 1)), int(np.clip(px[0], 0, gt.shape[1] - 1))
                    if not gt[gy, gx]:
                        continue
                if gt.shape != mask.shape:
                    mask_r = cv2.resize(mask.astype(np.uint8), (gt.shape[1], gt.shape[0]),
                                        interpolation=cv2.INTER_NEAREST) > 0
                else:
                    mask_r = mask
                inter = np.logical_and(gt, mask_r).sum()
                union = np.logical_or(gt, mask_r).sum()
                ious.append(inter / max(union, 1))
            else:
                cv2.imwrite(str(out_path), (mask.astype(np.uint8) * 255))
            done += 1
        print(f"[{root.name}] {done} frames, {empty} with no mask")

    if args.validate and ious:
        a = np.asarray(ious)
        print(f"\n[validate] n={len(a)}  IoU 中位={np.median(a):.3f}  均值={a.mean():.3f}")
        print(f"           >0.7 的比例 {np.mean(a > 0.7):.1%}   >0.5 的比例 {np.mean(a > 0.5):.1%}")
        print(f"  参考: SAM3 文本提示对手 IoU 0.41, 0% 超过 0.7; 对球 0.921, 对框 0.995")
    print(f"\n人工标注未被修改: {', '.join(sorted(PROTECTED))}")


if __name__ == "__main__":
    main()
