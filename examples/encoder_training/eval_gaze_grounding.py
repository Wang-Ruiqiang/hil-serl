#!/usr/bin/env python3
"""Does a gaze-grounded encoder keep the pick/place separation on held-out episodes?

This is the go/no-go for the gaze route. A falling grounding KL only shows the
query fits the target on average; a model that emitted one fixed blob in the
average location would score well on aggregate object masses too. The question
that decides whether the route works is narrower:

    before the grasp, does the attention stay off the basket?

That is the property the phase one-hot was engineered to produce, and the whole
argument for dropping the one-hot is that gaze supplies it for free -- measured
on the recordings, the operator looks inside the basket on 1.5% of pre-grasp
frames against 32.9% after. If the trained query cannot reproduce that from a
single frame, the pick phase will drift toward the basket exactly as it did
before, and no amount of good average KL will save it.

The phase label here is computed from the recorded masks, independently of
anything the model or the target saw: the ball's bounding box sitting inside
the hand's for five consecutive frames. Five, not one -- the per-frame test
flickers false whenever the ball is occluded or the hand mask fragments, which
relabels transport frames as pre-grasp and destroys the split it is measuring.

Masks are read for scoring only and never reach the model.
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
    TaskDemoFrameDataset,
    ViTPretrainModel,
    _frame_id,
    find_episode_demos,
    split_demos,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best.msgpack")
    parser.add_argument(
        "--split", default="test", choices=["test", "val", "train", "all"])
    parser.add_argument(
        "--cell_threshold", type=float, default=0.04,
        help="Mask occupancy above which a token cell counts as covered. "
             "Matches mask_grounding_cell_threshold in the RL agent.")
    parser.add_argument("--run_length", type=int, default=5)
    parser.add_argument(
        "--render", type=Path, default=None,
        help="Write a review video per episode: the frame with mask contours, "
             "the gaze target the query was trained on, and the attention it "
             "actually produced. The numbers report averages; only this shows "
             "whether the attention tracks the ball frame by frame or just "
             "parks in the average location.")
    return parser.parse_args()


def _overlay(image, heatmap, label):
    height, width = image.shape[:2]
    upsampled = cv2.resize(
        heatmap / max(float(heatmap.max()), 1e-9), (width, height),
        interpolation=cv2.INTER_NEAREST)
    colour = cv2.applyColorMap((upsampled * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    grey = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    panel = cv2.addWeighted(grey, 0.4, colour, 0.6, 0)
    cv2.putText(panel, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def _contours(image, frame_dir, label):
    canvas = image.copy()
    for name, colour in (("ball_mask.png", (0, 255, 0)),
                         ("basket_mask.png", (255, 128, 0)),
                         ("hand_mask.png", (0, 255, 255))):
        mask = cv2.imread(str(frame_dir / name), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
        found, _ = cv2.findContours((mask > 127).astype(np.uint8),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, found, -1, colour, 1)
    cv2.putText(canvas, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _writers(stem: Path, fps: float = 8.3):
    """mp4 for archival, webm because that is what plays on the workstation."""
    import imageio.v2 as imageio
    return [
        imageio.get_writer(str(stem.with_suffix(".mp4")), fps=fps, codec="libx264",
                           quality=8, macro_block_size=1, ffmpeg_log_level="error"),
        imageio.get_writer(str(stem.with_suffix(".webm")), fps=fps, codec="libvpx-vp9",
                           quality=8, macro_block_size=1, ffmpeg_log_level="error"),
    ]


def _cells(path: Path, grid, threshold: float):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    occupancy = cv2.resize(
        (mask > 127).astype(np.float32), (grid[1], grid[0]),
        interpolation=cv2.INTER_AREA)
    return occupancy > threshold


def _grasp_onset(frames, run_length: int):
    """First frame of the first sustained run with the ball inside the hand."""
    run = 0
    for index, frame in enumerate(frames):
        ball = cv2.imread(str(frame / "ball_mask.png"), cv2.IMREAD_GRAYSCALE)
        hand = cv2.imread(str(frame / "hand_mask.png"), cv2.IMREAD_GRAYSCALE)
        held = False
        if ball is not None and hand is not None:
            ys, xs = np.where(ball > 127)
            hy, hx = np.where(hand > 127)
            if xs.size and hx.size:
                held = (xs.min() >= hx.min() - 2 and ys.min() >= hy.min() - 2
                        and xs.max() <= hx.max() + 2 and ys.max() <= hy.max() + 2)
        run = run + 1 if held else 0
        if run >= run_length:
            return _frame_id(frames[index - run_length + 1])
    return None


def main():
    args = parse_args()
    config = json.loads((args.run_dir / "config.json").read_text())
    source = config.get("grounding_source", "mask")
    if source != "gaze":
        print(f"warning: run was trained with grounding_source={source}")

    data_root = config.get("data_root")
    if not data_root:
        from task_configs import get_task_config
        data_root = get_task_config(config["exp_name"]).data_root
    stride = int(config["frame_stride"])
    # Must mirror training exactly. eval_encoder.py branches to find_phase_demos
    # when a phase scan file exists, which yields a different demo list and so a
    # different train/val/test partition -- the "held-out" episodes would then
    # not be the ones actually held out.
    # The run may have been trained on a subset of the sessions. Without
    # replaying that filter, find_episode_demos enumerates every session,
    # split_demos partitions a different list, and the "held-out" episodes are
    # not the ones held out -- some of them the encoder never saw at all, and
    # others it trained on.
    train_encoder._SESSION_FILTER = config.get("session_filter", "") or ""
    demos = find_episode_demos(
        Path(data_root), stride, kept_ranges_only=(source == "gaze"))
    train_demos, val_demos, test_demos = split_demos(
        demos, int(config["val_demos"]), int(config["test_demos"]),
        int(config["seed"]))
    chosen = {
        "train": train_demos, "val": val_demos, "test": test_demos,
        "all": demos,
    }[args.split]
    print(f"{args.split} split: {len(chosen)} episodes")

    payload = serialization.msgpack_restore(
        (args.run_dir / args.checkpoint).read_bytes())
    params = payload["full_params"]
    phase_dim = int(config.get("grounding_phase_dim", 0))
    image_size = int(config["image_size"])
    patch_size = int(config["patch_size"])
    grid = (image_size // patch_size, image_size // patch_size)

    model = ViTPretrainModel(
        num_targets=len(config["target_names"]),
        image_size=(image_size, image_size),
        patch_size=(patch_size, patch_size),
        hidden_dim=int(config["vit_hidden_dim"]),
        num_layers=int(config["vit_num_layers"]),
        num_heads=int(config["vit_num_heads"]),
        output_dim=int(config["output_dim"]),
        num_spatial_blocks=int(config["num_spatial_blocks"]),
        grounding_phase_dim=phase_dim,
        grounding_tactile_conditioned=bool(
            config.get("grounding_tactile_conditioned", False)),
    )

    tactile_conditioned = bool(config.get("grounding_tactile_conditioned", False))

    @jax.jit
    def attention_of(images, tactile):
        # A tactile-conditioned query takes no phase vector: the encoder derives
        # its conditioning from the tactile frame itself. Passing None for both
        # would make the query raise rather than silently attend wrongly.
        output = model.apply(
            {"params": params}, images,
            future_image=images, state=jnp.zeros((images.shape[0], 7)),
            phase=None, tactile=tactile, train=False)
        logits = output["grounding_logits"].reshape(images.shape[0], -1)
        return jax.nn.softmax(logits, axis=-1).reshape(
            images.shape[0], grid[0], grid[1])

    rows = []
    for demo in chosen:
        onset = _grasp_onset(demo.pooling_frames or demo.frames, args.run_length)
        dataset = TaskDemoFrameDataset(
            [demo], image_size=int(config["input_size"]),
            mask_files=list(config["mask_files"]),
            target_names=list(config["target_names"]),
            sample_stride=stride, grounding_source=source, token_grid=grid,
            # From the run's own config, not the module defaults: a run trained
            # with a different sigma/dilate would otherwise be scored against a
            # target it never saw, and the "target" column would silently
            # describe a different experiment than the "model" column.
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
        attention = np.asarray(attention_of(
            jnp.asarray(np.stack(images)),
            jnp.asarray(np.stack(tactiles)) if tactile_conditioned else None))
        writers = None
        if args.render is not None:
            args.render.mkdir(parents=True, exist_ok=True)
            writers = _writers(
                args.render / f"{demo.dataset_name}_ep{demo.episode_index:02d}")
        for index, frame in enumerate(frames):
            if writers is not None:
                colour_image = cv2.imread(str(frame / "color_image.jpg"))
                if colour_image is not None:
                    colour_image = cv2.resize(colour_image, (320, 240))
                    phase_label = ("PLACE" if (onset is not None
                                   and _frame_id(frame) >= onset) else "PICK")
                    panel = np.hstack([
                        _contours(colour_image, frame, f"f{_frame_id(frame)} {phase_label}"),
                        _overlay(colour_image, targets[index], "gaze target"),
                        _overlay(colour_image, attention[index], "model attention"),
                    ])
                    for writer in writers:
                        writer.append_data(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
            ball = _cells(frame / "ball_mask.png", grid, args.cell_threshold)
            hand = _cells(frame / "hand_mask.png", grid, args.cell_threshold)
            basket = _cells(frame / "basket_mask.png", grid, args.cell_threshold)
            if ball is None or hand is None or basket is None:
                continue
            after = onset is not None and _frame_id(frame) >= onset
            rows.append((
                after,
                attention[index][ball].sum(), attention[index][hand].sum(),
                attention[index][basket].sum(),
                targets[index][ball].sum(), targets[index][hand].sum(),
                targets[index][basket].sum(),
                ball.mean(), hand.mean(), basket.mean(),
            ))

        if writers is not None:
            for writer in writers:
                writer.close()

    if not rows:
        print("no scorable frames")
        return
    table = np.asarray(rows, dtype=np.float32)
    after = table[:, 0] > 0.5
    print(f"frames: {len(table)}  pick {int((~after).sum())}  "
          f"place {int(after.sum())}\n")
    print(f"{'':<10}{'ball':>18}{'hand':>18}{'basket':>18}")
    print(f"{'':<10}{'model':>8}{'target':>10}{'model':>8}{'target':>10}"
          f"{'model':>8}{'target':>10}")
    for name, selection in (("pick", ~after), ("place", after)):
        subset = table[selection]
        if not len(subset):
            continue
        print(f"{name:<10}"
              f"{subset[:, 1].mean():>8.3f}{subset[:, 4].mean():>10.3f}"
              f"{subset[:, 2].mean():>8.3f}{subset[:, 5].mean():>10.3f}"
              f"{subset[:, 3].mean():>8.3f}{subset[:, 6].mean():>10.3f}")
    print(f"{'chance':<10}{table[:, 7].mean():>8.3f}{'':>10}"
          f"{table[:, 8].mean():>8.3f}{'':>10}{table[:, 9].mean():>8.3f}")

    pick_basket = table[~after, 3].mean() if (~after).any() else float("nan")
    place_basket = table[after, 3].mean() if after.any() else float("nan")
    chance_basket = table[:, 9].mean()
    print(f"\npick basket {pick_basket:.3f} vs chance {chance_basket:.3f} "
          f"({chance_basket / max(pick_basket, 1e-6):.1f}x below)")
    print(f"place/pick basket separation: "
          f"{place_basket / max(pick_basket, 1e-6):.1f}x")


if __name__ == "__main__":
    main()
