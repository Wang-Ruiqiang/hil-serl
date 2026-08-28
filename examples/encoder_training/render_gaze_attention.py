#!/usr/bin/env python3
"""Render what a gaze-grounded encoder attends to, frame by frame.

Three panels: the frame with the recorded mask outlines, the gaze target the
grounding query was trained against, and the attention it actually produces.
The aggregate numbers cannot distinguish an attention that tracks the ball from
one parked in the average location; only watching it can.

Works on both recordings: the older one names its masks mask1/mask2 and has no
hand mask, so mask outlines are drawn from whatever exists rather than assumed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import train_encoder  # noqa: E402
from train_encoder import (  # noqa: E402
    MASK_FILE_ALIASES,
    TaskDemoFrameDataset,
    ViTPretrainModel,
    _frame_id,
    find_episode_demos,
    resolve_mask_path,
    split_demos,
)

OUTLINE = {"ball_mask.png": (0, 255, 0), "hand_mask.png": (0, 255, 255),
           "basket_mask.png": (255, 128, 0)}
PANEL = 360


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", type=Path, required=True)
    p.add_argument("--checkpoint", default="best.msgpack")
    p.add_argument("--split", default="test", choices=["test", "val", "train", "all"])
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--fps", type=float, default=8.3)
    p.add_argument("--out", default=None)
    return p.parse_args()


def heat(image, weights, label):
    height, width = image.shape[:2]
    upsampled = cv2.resize(weights / max(float(weights.max()), 1e-9),
                           (width, height), interpolation=cv2.INTER_NEAREST)
    colour = cv2.applyColorMap((upsampled * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    grey = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    panel = cv2.addWeighted(grey, 0.4, colour, 0.6, 0)
    cv2.putText(panel, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2)
    return panel


def main():
    args = parse_args()
    config = json.loads((args.run_dir / "config.json").read_text())
    source = config.get("grounding_source", "mask")
    data_root = Path(config["data_root"])
    stride = int(config["frame_stride"])
    # The run may have been trained on a subset of the sessions. Without
    # replaying that filter, find_episode_demos enumerates every session,
    # split_demos partitions a different list, and the "held-out" episodes are
    # not the ones held out -- some of them the encoder never saw at all, and
    # others it trained on.
    train_encoder._SESSION_FILTER = config.get("session_filter", "") or ""
    demos = find_episode_demos(data_root, stride,
                               kept_ranges_only=(source == "gaze"))
    train_demos, val_demos, test_demos = split_demos(
        demos, int(config["val_demos"]), int(config["test_demos"]),
        int(config["seed"]))
    chosen = {"train": train_demos, "val": val_demos, "test": test_demos,
              "all": demos}[args.split][: args.episodes]
    print(f"{args.split}: {len(chosen)} episodes", flush=True)

    params = serialization.msgpack_restore(
        (args.run_dir / args.checkpoint).read_bytes())["full_params"]
    image_size = int(config["image_size"])
    patch = int(config["patch_size"])
    grid = (image_size // patch, image_size // patch)
    model = ViTPretrainModel(
        num_targets=len(config["target_names"]),
        image_size=(image_size, image_size), patch_size=(patch, patch),
        hidden_dim=int(config["vit_hidden_dim"]),
        num_layers=int(config["vit_num_layers"]),
        num_heads=int(config["vit_num_heads"]),
        output_dim=int(config["output_dim"]),
        num_spatial_blocks=int(config["num_spatial_blocks"]),
        grounding_phase_dim=int(config.get("grounding_phase_dim", 0)),
        grounding_tactile_conditioned=bool(
            config.get("grounding_tactile_conditioned", False)))

    tactile_conditioned = bool(config.get("grounding_tactile_conditioned", False))

    @jax.jit
    def attention(images, tactile):
        # A tactile-conditioned query has no phase input at all; the encoder
        # derives its conditioning from the tactile frame itself, so leaving
        # this out makes the query raise rather than silently attend wrongly.
        out = model.apply({"params": params}, images, future_image=images,
                          state=jnp.zeros((images.shape[0], 7)), phase=None,
                          tactile=tactile, train=False)
        logits = out["grounding_logits"].reshape(images.shape[0], -1)
        return jax.nn.softmax(logits, axis=-1).reshape(images.shape[0], *grid)

    out_dir = Path(args.out) if args.out else args.run_dir / "attention_review"
    out_dir.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio

    for demo in chosen:
        dataset = TaskDemoFrameDataset(
            [demo], image_size=int(config["input_size"]),
            mask_files=list(config["mask_files"]),
            target_names=list(config["target_names"]),
            sample_stride=stride, grounding_source=source, token_grid=grid,
            gaze_sigma_cells=float(config.get("gaze_sigma_cells", 0.6)),
            gaze_dilate_cells=float(config.get("gaze_dilate_cells", 1.0)),
            gaze_window=int(config.get("gaze_window", 5)),
            gaze_max_gap=int(config.get("gaze_max_gap", 3)),
            gaze_decay=float(config.get("gaze_decay", 0.15)))
        images, targets, frames, tactiles = [], [], [], []
        for index in range(len(dataset)):
            sample = dataset[index]
            images.append(np.asarray(sample["image"]).transpose(1, 2, 0))
            targets.append(np.asarray(sample["gaze_target"]))
            frames.append(Path(sample["frame"]))
            tactiles.append(np.asarray(sample["tactile"]).transpose(1, 2, 0))
        if not images:
            continue
        maps = np.asarray(attention(
            jnp.asarray(np.stack(images)),
            jnp.asarray(np.stack(tactiles)) if tactile_conditioned else None))

        stem = out_dir / f"{demo.dataset_name[-8:]}_ep{demo.episode_index:02d}"
        writers = [
            imageio.get_writer(str(stem.with_suffix(".mp4")), fps=args.fps,
                               codec="libx264", quality=8, macro_block_size=1,
                               ffmpeg_log_level="error"),
            imageio.get_writer(str(stem.with_suffix(".webm")), fps=args.fps,
                               codec="libvpx-vp9", quality=8, macro_block_size=1,
                               ffmpeg_log_level="error"),
        ]
        for index, frame_dir in enumerate(frames):
            bgr = cv2.imread(str(frame_dir / "color_image.jpg"))
            if bgr is None:
                continue
            bgr = cv2.resize(bgr, (PANEL, PANEL))
            reference = bgr.copy()
            for name, colour in OUTLINE.items():
                path = resolve_mask_path(frame_dir, name)
                if path is None:
                    continue
                mask = cv2.resize(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE),
                                  (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
                contours, _ = cv2.findContours((mask > 127).astype(np.uint8),
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(reference, contours, -1, colour, 1)
            cv2.putText(reference, f"f{_frame_id(frame_dir)}", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            panel = np.hstack([
                reference,
                heat(bgr, targets[index], "gaze target"),
                heat(bgr, maps[index],
                     f"model  peak={maps[index].max():.3f}"),
            ])
            for writer in writers:
                writer.append_data(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
        for writer in writers:
            writer.close()
        print(f"  {stem.name}: {len(frames)} frames", flush=True)
    print(f"\n-> {out_dir}")


if __name__ == "__main__":
    main()
