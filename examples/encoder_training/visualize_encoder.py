"""Visualize the pretrained vit-grounded encoder on recorded demo frames.

For each frame this renders, side by side:

``grounding attention``
    The softmax over the grounding query's attention logits -- the exact map
    the CGL loss supervises during RL. It should sit on the ball.

``segmentation``
    The pretrain-only 1x1 head's per-target mask probabilities, which show
    whether the trunk represents all three objects.

Run after train_encoder.py to confirm the pretraining actually grounded
before spending robot time on an RL run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np

try:
    from .train_encoder import (
        TaskDemoFrameDataset,
        ViTPretrainModel,
        find_episode_demos,
        find_phase_demos,
        split_demos,
    )
except ImportError:  # Allows running this file directly from its directory.
    from train_encoder import (
        TaskDemoFrameDataset,
        ViTPretrainModel,
        find_episode_demos,
        find_phase_demos,
        split_demos,
    )


def overlay(image_rgb, heatmap):
    heatmap = cv2.resize(
        heatmap,
        (image_rgb.shape[1], image_rgb.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    heatmap = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)[:, :, ::-1]
    return cv2.addWeighted(image_rgb, 0.55, color, 0.45, 0)


def normalize_for_display(heatmap):
    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
    heatmap = np.maximum(heatmap, 0.0)
    robust_max = float(np.percentile(heatmap, 99.0))
    return np.clip(heatmap / (robust_max + 1e-6), 0.0, 1.0)


def add_panel_label(image_rgb, label):
    panel = image_rgb.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    thickness = 1
    (text_width, text_height), _ = cv2.getTextSize(label, font, scale, thickness)
    cv2.rectangle(
        panel,
        (0, 0),
        (min(panel.shape[1] - 1, text_width + 10), text_height + 10),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        panel,
        label,
        (5, text_height + 5),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return panel


def make_panel_grid(panels, columns=2):
    columns = max(1, int(columns))
    height = max(panel.shape[0] for panel in panels)
    width = max(panel.shape[1] for panel in panels)
    rows = (len(panels) + columns - 1) // columns
    grid = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, panel in enumerate(panels):
        row, column = divmod(index, columns)
        if panel.shape[:2] != (height, width):
            panel = cv2.resize(panel, (width, height), interpolation=cv2.INTER_LINEAR)
        grid[
            row * height : (row + 1) * height,
            column * width : (column + 1) * width,
        ] = panel
    return grid



def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True,
                        help="Directory written by train_encoder.py")
    parser.add_argument("--checkpoint", default="best.msgpack")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--max_frames", type=int, default=40)
    parser.add_argument("--frame_stride", type=int, default=25)
    parser.add_argument("--demos", type=int, default=2)
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "val", "train", "all"],
        help="Which split to render. Defaults to the held-out test demos, so "
             "the panels correspond to what eval_encoder.py reports.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = json.loads((args.run_dir / "config.json").read_text())
    if config.get("architecture") not in ("vit_grounded_v1", "vit_grounded_phase_v1"):
        raise ValueError(
            f"{args.run_dir} is not a vit-grounded run "
            f"(architecture={config.get('architecture')!r})"
        )
    phase_dim = int(config.get("grounding_phase_dim", 0))
    target_names = list(config["target_names"])
    mask_files = list(config["mask_files"])
    input_size = int(config["input_size"])
    image_size = int(config["image_size"])
    output_dir = args.output_dir or (args.run_dir / f"visualizations_{args.split}")
    output_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(config["data_root"])
    phase_scan = Path(config["phase_scan"]) if config.get("phase_scan") else None
    if phase_scan and phase_scan.is_file():
        demos = find_phase_demos(data_root, 1, phase_scan)
    else:
        demos = find_episode_demos(data_root, 1)
    if args.split != "all":
        train_demos, val_demos, test_demos = split_demos(
            demos,
            int(config["val_demos"]),
            int(config["test_demos"]),
            int(config["seed"]),
        )
        demos = {"train": train_demos, "val": val_demos, "test": test_demos}[args.split]
    print(f"{args.split} split: {len(demos)} demos, rendering the first {args.demos}")

    dataset = TaskDemoFrameDataset(
        demos[: args.demos],
        image_size=input_size,
        mask_files=mask_files,
        target_names=target_names,
        sample_stride=args.frame_stride,
    )

    payload = serialization.msgpack_restore(
        (args.run_dir / args.checkpoint).read_bytes()
    )
    params = payload.get("full_params")
    if params is None:
        raise ValueError(
            f"{args.checkpoint} has no 'full_params'; re-run train_encoder.py"
        )

    model = ViTPretrainModel(
        num_targets=len(target_names),
        image_size=(image_size, image_size),
        patch_size=(int(config["patch_size"]), int(config["patch_size"])),
        hidden_dim=int(config["vit_hidden_dim"]),
        num_layers=int(config["vit_num_layers"]),
        num_heads=int(config["vit_num_heads"]),
        output_dim=int(config["output_dim"]),
        num_spatial_blocks=int(config["num_spatial_blocks"]),
        grounding_phase_dim=phase_dim,
    )

    def run(image, pick_phase, place_phase):
        phase = (
            jnp.asarray([[pick_phase, place_phase]], jnp.float32)
            if phase_dim
            else None
        )
        return jax.device_get(
            model.apply(
                {"params": params}, jnp.asarray(image), phase=phase, train=False
            )
        )

    def softmax_map(output):
        logits = np.asarray(output["grounding_logits"][0], dtype=np.float32)
        attention = np.exp(logits - logits.max())
        return attention / max(attention.sum(), 1e-8)

    for index in range(min(args.max_frames, len(dataset))):
        sample = dataset[index]
        image = np.asarray(sample["image"], dtype=np.float32).transpose(1, 2, 0)[None]
        is_pick = float(sample["phase"] in ("pick", "all"))
        output = run(image, is_pick, 1.0 - is_pick)
        rgb = np.clip(image[0], 0, 255).astype(np.uint8)  # dataset is 0..255

        attention = softmax_map(output)
        peak = np.unravel_index(int(np.argmax(attention)), attention.shape)

        panels = [
            add_panel_label(rgb.copy(), f"RGB ({sample['phase']})"),
            add_panel_label(
                overlay(rgb, normalize_for_display(attention)),
                f"grounding attn peak={peak} p={attention.max():.3f}",
            ),
        ]
        if phase_dim:
            # Same pixels, phase flipped. Two clearly different maps here is
            # the whole point of the conditioning: it means the target is
            # chosen by the phase signal rather than guessed from the image.
            other = softmax_map(run(image, 1.0 - is_pick, is_pick))
            panels.append(
                add_panel_label(
                    overlay(rgb, normalize_for_display(other)),
                    "counterfactual: phase forced to "
                    + ("place" if is_pick else "pick"),
                )
            )
        segmentation = np.asarray(output["segmentation_logits"][0], dtype=np.float32)
        probability = 1.0 / (1.0 + np.exp(-segmentation))
        for channel, name in enumerate(target_names):
            panels.append(
                add_panel_label(
                    overlay(rgb, normalize_for_display(probability[channel])),
                    f"seg: {name}",
                )
            )

        grid = make_panel_grid(panels, columns=2)
        path = output_dir / f"frame_{index:04d}.png"
        cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"wrote {min(args.max_frames, len(dataset))} panels to {output_dir}")


if __name__ == "__main__":
    main()
