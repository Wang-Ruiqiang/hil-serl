"""Visualize and evaluate a trained RGB mask predictor on recorded frames.

The script samples frames from recorded demo data, runs ``train_mask_predictor``
checkpoints, compares predicted mask1/mask2 against saved ground-truth masks,
and saves side-by-side visualizations.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_mask_predictor import (  # noqa: E402
    DEFAULT_DATA_ROOTS,
    MASK_SLOTS,
    TinyUNet,
    discover_frame_dirs,
    has_supervision,
    read_mask_original,
    resize_mask,
)


DEFAULT_CHECKPOINT = str(SCRIPT_DIR / "mask_predictor_ckpt" / "best.pt")
DEFAULT_OUTPUT_DIR = str(SCRIPT_DIR / "mask_predictor_eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test RGB mask predictor and save prediction-vs-GT visualizations."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--data_root",
        action="append",
        default=None,
        help="Recorded data root. Can be repeated. Defaults to training defaults.",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--num_frames", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample_mode",
        choices=("random", "stride"),
        default="random",
        help="How to sample frames for visualization.",
    )
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--device", default=None, help="cuda/cpu; default auto.")
    parser.add_argument(
        "--save_raw_masks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save black/white predicted and GT mask png files for each sampled frame.",
    )
    return parser.parse_args()


def frame_label(frame_dir: Path) -> str:
    return f"{frame_dir.parent.name}/{frame_dir.name}"


def load_checkpoint(path: Path, device: torch.device) -> tuple[TinyUNet, dict[str, Any]]:
    checkpoint = torch.load(str(path), map_location="cpu")
    config = dict(checkpoint.get("config", {}))
    model = TinyUNet(
        input_channels=3,
        base_channels=int(config.get("base_channels", 32)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    print(
        f"[ckpt] {path} epoch={checkpoint.get('epoch', 'unknown')} "
        f"image_size={config.get('image_size', 128)}"
    )
    return model, config


def select_frames(frame_dirs: list[Path], args: argparse.Namespace) -> list[Path]:
    supervised = [frame_dir for frame_dir in frame_dirs if has_supervision(frame_dir)]
    if args.sample_mode == "stride":
        selected = supervised[:: max(1, int(args.stride))]
        return selected[: max(0, int(args.num_frames))]
    rng = random.Random(args.seed)
    if len(supervised) <= args.num_frames:
        return supervised
    return rng.sample(supervised, int(args.num_frames))


def prepare_input(
    bgr: np.ndarray,
    image_size: int,
) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    image = resized.astype(np.float32) / 255.0
    return torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32))[None]


def read_gt_masks(
    frame_dir: Path,
    image_shape: tuple[int, int],
    *,
    image_size: int,
    min_mask_pixels: int,
) -> np.ndarray:
    masks = [
        resize_mask(
            read_mask_original(frame_dir, slot, image_shape),
            image_size,
            min_mask_pixels,
        )
        for slot in MASK_SLOTS
    ]
    return np.stack(masks).astype(np.uint8)


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float | None:
    gt_positive = bool(np.any(gt_mask > 0))
    pred_positive = bool(np.any(pred_mask > 0))
    if not gt_positive:
        return 1.0 if not pred_positive else 0.0
    intersection = np.logical_and(pred_mask > 0, gt_mask > 0).sum()
    union = np.logical_or(pred_mask > 0, gt_mask > 0).sum()
    return float(intersection / max(1, union))


def blend_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = image.copy()
    active = mask > 0
    if np.any(active):
        color_arr = np.asarray(color, dtype=np.uint8)
        out[active] = (0.55 * out[active] + 0.45 * color_arr).astype(np.uint8)
    return out


def put_small_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.42,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def add_title_bar(image: np.ndarray, title: str, subtitle: str | None = None) -> np.ndarray:
    subtitles = [] if subtitle is None else str(subtitle).split("\n")
    bar_height = 66
    bar = np.full((bar_height, image.shape[1], 3), 35, dtype=np.uint8)
    put_small_text(bar, title, (6, 18), scale=0.45)
    for line_index, line in enumerate(subtitles):
        put_small_text(
            bar,
            line,
            (6, 38 + 16 * line_index),
            scale=0.32,
            color=(210, 210, 210),
        )
    return np.concatenate([bar, image], axis=0)


def make_panel(
    frame_dir: Path,
    bgr: np.ndarray,
    pred_masks: np.ndarray,
    gt_masks: np.ndarray,
    ious: dict[str, float],
) -> np.ndarray:
    image_size = pred_masks.shape[-1]
    base = cv2.resize(bgr, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    gt_overlay = base.copy()
    pred_overlay = base.copy()
    colors = {
        "mask1": (40, 220, 40),
        "mask2": (220, 40, 220),
    }
    for slot_index, slot in enumerate(MASK_SLOTS):
        gt_overlay = blend_mask(gt_overlay, gt_masks[slot_index], colors[slot])
        pred_overlay = blend_mask(pred_overlay, pred_masks[slot_index], colors[slot])

    diff = base.copy()
    for slot_index, slot in enumerate(MASK_SLOTS):
        pred = pred_masks[slot_index] > 0
        gt = gt_masks[slot_index] > 0
        false_pos = np.logical_and(pred, ~gt)
        false_neg = np.logical_and(~pred, gt)
        true_pos = np.logical_and(pred, gt)
        diff[true_pos] = (0.55 * diff[true_pos] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
        diff[false_pos] = (0.55 * diff[false_pos] + 0.45 * np.array([0, 0, 255])).astype(np.uint8)
        diff[false_neg] = (0.55 * diff[false_neg] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)

    header_height = 42
    header = np.full((header_height, image_size * 4, 3), 25, dtype=np.uint8)
    put_small_text(header, frame_label(frame_dir), (8, 18), scale=0.42)
    put_small_text(
        header,
        f"mask1 IoU={ious['mask1']:.3f}    mask2 IoU={ious['mask2']:.3f}",
        (8, 36),
        scale=0.38,
        color=(220, 220, 220),
    )

    columns = [
        add_title_bar(base, "RGB", "original image"),
        add_title_bar(gt_overlay, "GT", "saved mask1+mask2"),
        add_title_bar(pred_overlay, "PRED", "network output"),
        add_title_bar(diff, "DIFF", "green TP / red FP\nblue FN"),
    ]
    panel = np.concatenate(columns, axis=1)
    return np.concatenate([header, panel], axis=0)


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_raw_masks:
        (output_dir / "binary_masks").mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, config = load_checkpoint(checkpoint_path, device)
    image_size = int(config.get("image_size", 128))
    min_mask_pixels = int(config.get("min_mask_pixels", 8))

    data_roots = args.data_root or config.get("data_roots") or DEFAULT_DATA_ROOTS
    frame_dirs = discover_frame_dirs(list(data_roots))
    selected_frames = select_frames(frame_dirs, args)
    if not selected_frames:
        raise RuntimeError("No supervised frames found for evaluation.")

    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "data_roots": list(data_roots),
        "output_dir": str(output_dir),
        "mask_threshold": float(args.mask_threshold),
        "frames": [],
        "mean_iou": {},
    }
    all_ious: dict[str, list[float]] = {slot: [] for slot in MASK_SLOTS}

    with torch.no_grad():
        for sample_index, frame_dir in enumerate(tqdm(selected_frames, desc="eval")):
            bgr = cv2.imread(str(frame_dir / "color_image.jpg"), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"[skip] missing image: {frame_dir}")
                continue
            input_tensor = prepare_input(bgr, image_size).to(device)
            logits = model(input_tensor)
            pred_masks = (
                torch.sigmoid(logits)[0].detach().cpu().numpy() >= float(args.mask_threshold)
            ).astype(np.uint8)
            gt_masks = read_gt_masks(
                frame_dir,
                bgr.shape[:2],
                image_size=image_size,
                min_mask_pixels=min_mask_pixels,
            )

            ious = {
                slot: compute_iou(pred_masks[slot_index], gt_masks[slot_index])
                for slot_index, slot in enumerate(MASK_SLOTS)
            }
            for slot, iou in ious.items():
                if iou is not None:
                    all_ious[slot].append(float(iou))

            panel = make_panel(frame_dir, bgr, pred_masks, gt_masks, ious)
            output_path = output_dir / f"{sample_index:04d}_{frame_dir.parent.name}_{frame_dir.name}.jpg"
            cv2.imwrite(str(output_path), panel)

            if args.save_raw_masks:
                for slot_index, slot in enumerate(MASK_SLOTS):
                    pred_mask_path = (
                        output_dir
                        / "binary_masks"
                        / f"{sample_index:04d}_{frame_dir.parent.name}_{frame_dir.name}_{slot}.png"
                    )
                    gt_mask_path = (
                        output_dir
                        / "binary_masks"
                        / f"{sample_index:04d}_{frame_dir.parent.name}_{frame_dir.name}_{slot}_gt.png"
                    )
                    cv2.imwrite(str(pred_mask_path), pred_masks[slot_index] * 255)
                    cv2.imwrite(str(gt_mask_path), gt_masks[slot_index] * 255)

            summary["frames"].append(
                {
                    "frame_dir": str(frame_dir),
                    "image": str(output_path),
                    "ious": {slot: float(value) for slot, value in ious.items()},
                    "pred_pixels": {
                        slot: int(np.count_nonzero(pred_masks[slot_index]))
                        for slot_index, slot in enumerate(MASK_SLOTS)
                    },
                    "pred_mask_paths": {
                        slot: str(
                            output_dir
                            / "binary_masks"
                            / f"{sample_index:04d}_{frame_dir.parent.name}_{frame_dir.name}_{slot}.png"
                        )
                        for slot in MASK_SLOTS
                    },
                    "gt_pixels": {
                        slot: int(np.count_nonzero(gt_masks[slot_index]))
                        for slot_index, slot in enumerate(MASK_SLOTS)
                    },
                    "gt_mask_paths": {
                        slot: str(
                            output_dir
                            / "binary_masks"
                            / f"{sample_index:04d}_{frame_dir.parent.name}_{frame_dir.name}_{slot}_gt.png"
                        )
                        for slot in MASK_SLOTS
                    },
                }
            )

    summary["mean_iou"] = {
        slot: float(np.mean(values)) if values else float("nan")
        for slot, values in all_ious.items()
    }
    summary["mean_iou"]["all"] = float(
        np.mean([value for values in all_ious.values() for value in values])
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"[done] frames={len(summary['frames'])} "
        f"mask1_iou={summary['mean_iou']['mask1']:.3f} "
        f"mask2_iou={summary['mean_iou']['mask2']:.3f} "
        f"all_iou={summary['mean_iou']['all']:.3f} "
        f"output={output_dir}"
    )


if __name__ == "__main__":
    main()
