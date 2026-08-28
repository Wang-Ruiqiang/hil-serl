"""Train a lightweight RGB mask predictor for ball and basket masks.

Input frame format:
  <data_root>/frame_XXXX/color_image.jpg
  <data_root>/frame_XXXX/ball_mask.png  # tennis ball mask, optional/zero if hidden
  <data_root>/frame_XXXX/basket_mask.png  # basket mask, optional/zero if hidden

The model learns:
  RGB image -> two binary masks: [mask1, mask2]

This intentionally does not use gaze.  That lets older datasets with incorrect
gaze but correct SAM masks participate in training.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


DEFAULT_DATA_ROOTS = [
    "/media/user/data3/wrq/recorded_data/tennis_ball_pick/tennis_ball_pick-6-17-0",
    "/media/user/data3/wrq/recorded_data/tennis_ball_pick/tennis_ball_pick-6-23-1",
]
MASK_SLOTS = ("mask1", "mask2")
MASK_FILE_CANDIDATES = {
    "mask1": ("ball_mask.png", "mask1.png"),
    "mask2": ("basket_mask.png", "mask2.png"),
}
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = str(SCRIPT_DIR / "mask_predictor_ckpt")


@dataclass
class TrainConfig:
    data_roots: list[str]
    output_dir: str
    image_size: int = 128
    batch_size: int = 32
    num_epochs: int = 40
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    val_ratio: float = 0.15
    seed: int = 42
    num_workers: int = 4
    checkpoint_period: int = 10
    mask_bce_weight: float = 1.0
    mask_dice_weight: float = 1.0
    base_channels: int = 32
    min_mask_pixels: int = 8
    max_samples: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RGB image -> ball/basket mask predictor."
    )
    parser.add_argument(
        "--data_root",
        action="append",
        default=None,
        help=(
            "Recorded data root containing frame_* folders. Can be repeated. "
            "Defaults to tennis_ball_pick-6-17-0 and tennis_ball_pick-6-23-1."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for checkpoints and training metadata.",
    )
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--checkpoint_period",
        type=int,
        default=10,
        help="Save epoch_XXXX.pt every N epochs. Set <=0 to disable periodic saves.",
    )
    parser.add_argument("--mask_bce_weight", type=float, default=1.0)
    parser.add_argument("--mask_dice_weight", type=float, default=1.0)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument(
        "--min_mask_pixels",
        type=int,
        default=8,
        help="Masks with fewer foreground pixels after resize are treated as empty.",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def frame_id_from_dir(frame_dir: Path) -> int | None:
    if not frame_dir.name.startswith("frame_"):
        return None
    try:
        return int(frame_dir.name.split("_", 1)[1])
    except ValueError:
        return None


def discover_frame_dirs(data_roots: list[str]) -> list[Path]:
    frame_dirs: list[Path] = []
    for root_str in data_roots:
        root = Path(root_str).expanduser()
        if not root.exists():
            print(f"[warn] missing data_root={root}")
            continue
        for frame_dir in root.iterdir():
            if not frame_dir.is_dir() or frame_id_from_dir(frame_dir) is None:
                continue
            if (frame_dir / "color_image.jpg").exists():
                frame_dirs.append(frame_dir)
    frame_dirs.sort(key=lambda path: (str(path.parent), frame_id_from_dir(path) or -1))
    return frame_dirs


def read_mask_original(
    frame_dir: Path,
    slot: str,
    image_shape: tuple[int, int],
) -> np.ndarray:
    path = next(
        (frame_dir / name for name in MASK_FILE_CANDIDATES[slot] if (frame_dir / name).exists()),
        frame_dir / MASK_FILE_CANDIDATES[slot][0],
    )
    height, width = image_shape
    if not path.exists():
        return np.zeros((height, width), dtype=np.uint8)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros((height, width), dtype=np.uint8)
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


def resize_mask(mask: np.ndarray, image_size: int, min_mask_pixels: int) -> np.ndarray:
    resized = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    resized = (resized > 0).astype(np.float32)
    if int(resized.sum()) < int(min_mask_pixels):
        resized[...] = 0.0
    return resized


def has_supervision(frame_dir: Path) -> bool:
    return any(
        any((frame_dir / name).exists() for name in MASK_FILE_CANDIDATES[slot])
        for slot in MASK_SLOTS
    )


def split_frame_dirs(
    frame_dirs: list[Path],
    *,
    val_ratio: float,
    seed: int,
    max_samples: int | None,
) -> tuple[list[Path], list[Path]]:
    supervised = [frame_dir for frame_dir in frame_dirs if has_supervision(frame_dir)]
    if max_samples is not None:
        supervised = supervised[: int(max_samples)]
    rng = random.Random(seed)
    rng.shuffle(supervised)
    val_count = max(1, int(round(len(supervised) * val_ratio))) if len(supervised) > 1 else 0
    val_dirs = supervised[:val_count]
    train_dirs = supervised[val_count:]
    return train_dirs, val_dirs


class RecordedMaskDataset(Dataset):
    def __init__(
        self,
        frame_dirs: list[Path],
        *,
        image_size: int,
        min_mask_pixels: int,
        augment: bool,
    ) -> None:
        self.frame_dirs = frame_dirs
        self.image_size = int(image_size)
        self.min_mask_pixels = int(min_mask_pixels)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.frame_dirs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        frame_dir = self.frame_dirs[index]
        bgr = cv2.imread(str(frame_dir / "color_image.jpg"), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(frame_dir / "color_image.jpg")

        height, width = bgr.shape[:2]
        image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(
            image_rgb,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )

        masks = [
            resize_mask(
                read_mask_original(frame_dir, slot, (height, width)),
                self.image_size,
                self.min_mask_pixels,
            )
            for slot in MASK_SLOTS
        ]

        if self.augment and random.random() < 0.5:
            image_rgb = np.ascontiguousarray(image_rgb[:, ::-1])
            masks = [np.ascontiguousarray(mask[:, ::-1]) for mask in masks]

        image = image_rgb.astype(np.float32) / 255.0
        model_input = image.transpose(2, 0, 1).astype(np.float32)
        target_masks = np.stack(masks).astype(np.float32)

        return {
            "input": torch.from_numpy(model_input),
            "target_masks": torch.from_numpy(target_masks),
        }


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyUNet(nn.Module):
    def __init__(self, input_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        channels = int(base_channels)
        self.enc1 = ConvBlock(input_channels, channels)
        self.enc2 = ConvBlock(channels, channels * 2)
        self.enc3 = ConvBlock(channels * 2, channels * 4)
        self.enc4 = ConvBlock(channels * 4, channels * 8)
        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(channels * 8, channels * 4, 2, stride=2)
        self.dec3 = ConvBlock(channels * 8, channels * 4)
        self.up2 = nn.ConvTranspose2d(channels * 4, channels * 2, 2, stride=2)
        self.dec2 = ConvBlock(channels * 4, channels * 2)
        self.up1 = nn.ConvTranspose2d(channels * 2, channels, 2, stride=2)
        self.dec1 = ConvBlock(channels * 2, channels)
        self.mask_head = nn.Conv2d(channels, len(MASK_SLOTS), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d3 = self.up3(e4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.mask_head(d1)


def dice_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)
    intersection = (probs * targets).sum(dim=dims)
    union = probs.sum(dim=dims) + targets.sum(dim=dims)
    return (1.0 - (2.0 * intersection + eps) / (union + eps)).mean()


@torch.no_grad()
def compute_metrics(
    mask_logits: torch.Tensor,
    target_masks: torch.Tensor,
) -> dict[str, float]:
    pred_masks = (torch.sigmoid(mask_logits) > 0.5).float()
    metrics: dict[str, float] = {}
    ious: list[torch.Tensor] = []
    empty_hits: list[torch.Tensor] = []

    for slot_index, slot in enumerate(MASK_SLOTS):
        pred = pred_masks[:, slot_index]
        target = target_masks[:, slot_index]
        target_positive = target.sum(dim=(1, 2)) > 0
        target_empty = ~target_positive

        if target_positive.any():
            pred_pos = pred[target_positive]
            target_pos = target[target_positive]
            intersection = (pred_pos * target_pos).sum(dim=(1, 2))
            union = ((pred_pos + target_pos) > 0).float().sum(dim=(1, 2))
            slot_iou = intersection / union.clamp_min(1.0)
            metrics[f"{slot}_iou"] = float(slot_iou.mean().item())
            ious.append(slot_iou)
        else:
            metrics[f"{slot}_iou"] = float("nan")

        if target_empty.any():
            pred_empty = pred[target_empty].sum(dim=(1, 2)) == 0
            empty_acc = pred_empty.float()
            metrics[f"{slot}_empty_acc"] = float(empty_acc.mean().item())
            empty_hits.append(empty_acc)
        else:
            metrics[f"{slot}_empty_acc"] = float("nan")

    metrics["mask_iou"] = float(torch.cat(ious).mean().item()) if ious else float("nan")
    metrics["empty_acc"] = (
        float(torch.cat(empty_hits).float().mean().item()) if empty_hits else float("nan")
    )
    return metrics


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    config: TrainConfig,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals: dict[str, float] = {
        "loss": 0.0,
        "mask_bce": 0.0,
        "mask_dice": 0.0,
        "mask_iou": 0.0,
        "empty_acc": 0.0,
    }
    count = 0
    finite_counts = {"mask_iou": 0, "empty_acc": 0}

    for batch in tqdm(loader, leave=False, desc="train" if is_train else "val"):
        inputs = batch["input"].to(device, non_blocking=True)
        target_masks = batch["target_masks"].to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            mask_logits = model(inputs)
            mask_bce = F.binary_cross_entropy_with_logits(mask_logits, target_masks)
            mask_dice = dice_loss_with_logits(mask_logits, target_masks)
            loss = config.mask_bce_weight * mask_bce + config.mask_dice_weight * mask_dice
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        metrics = compute_metrics(mask_logits.detach(), target_masks)
        batch_size = int(inputs.shape[0])
        totals["loss"] += float(loss.detach().item()) * batch_size
        totals["mask_bce"] += float(mask_bce.detach().item()) * batch_size
        totals["mask_dice"] += float(mask_dice.detach().item()) * batch_size
        for key in ("mask_iou", "empty_acc"):
            if math.isfinite(metrics[key]):
                totals[key] += metrics[key] * batch_size
                finite_counts[key] += batch_size
        count += batch_size

    if count == 0:
        return totals

    return {
        "loss": totals["loss"] / count,
        "mask_bce": totals["mask_bce"] / count,
        "mask_dice": totals["mask_dice"] / count,
        "mask_iou": totals["mask_iou"] / max(1, finite_counts["mask_iou"]),
        "empty_acc": totals["empty_acc"] / max(1, finite_counts["empty_acc"]),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": asdict(config),
            "metrics": metrics,
            "mask_slots": MASK_SLOTS,
        },
        path,
    )


def mask_counts(frame_dirs: list[Path], config: TrainConfig) -> dict[str, dict[str, int]]:
    counts = {slot: {"nonempty": 0, "empty": 0} for slot in MASK_SLOTS}
    for frame_dir in frame_dirs:
        bgr = cv2.imread(str(frame_dir / "color_image.jpg"), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        height, width = bgr.shape[:2]
        for slot in MASK_SLOTS:
            mask = resize_mask(
                read_mask_original(frame_dir, slot, (height, width)),
                config.image_size,
                config.min_mask_pixels,
            )
            if np.any(mask > 0):
                counts[slot]["nonempty"] += 1
            else:
                counts[slot]["empty"] += 1
    return counts


def main() -> None:
    args = parse_args()
    data_roots = args.data_root or DEFAULT_DATA_ROOTS
    config = TrainConfig(
        data_roots=data_roots,
        output_dir=args.output_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        checkpoint_period=args.checkpoint_period,
        mask_bce_weight=args.mask_bce_weight,
        mask_dice_weight=args.mask_dice_weight,
        base_channels=args.base_channels,
        min_mask_pixels=args.min_mask_pixels,
        max_samples=args.max_samples,
    )

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    frame_dirs = discover_frame_dirs(config.data_roots)
    train_dirs, val_dirs = split_frame_dirs(
        frame_dirs,
        val_ratio=config.val_ratio,
        seed=config.seed,
        max_samples=config.max_samples,
    )
    if not train_dirs:
        raise RuntimeError(
            "No supervised frames found. Make sure frame_* folders contain ball_mask.png or basket_mask.png."
        )

    output_dir = Path(config.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))
    (output_dir / "splits.json").write_text(
        json.dumps(
            {
                "train": [str(path) for path in train_dirs],
                "val": [str(path) for path in val_dirs],
            },
            indent=2,
        )
    )

    print(f"[data] discovered_frames={len(frame_dirs)} supervised={len(train_dirs) + len(val_dirs)}")
    print(f"[data] train={len(train_dirs)} val={len(val_dirs)}")
    print(f"[data] train_mask_counts={mask_counts(train_dirs, config)}")
    print(f"[data] val_mask_counts={mask_counts(val_dirs, config)}")
    print(f"[output] {output_dir}")

    train_dataset = RecordedMaskDataset(
        train_dirs,
        image_size=config.image_size,
        min_mask_pixels=config.min_mask_pixels,
        augment=True,
    )
    val_dataset = RecordedMaskDataset(
        val_dirs,
        image_size=config.image_size,
        min_mask_pixels=config.min_mask_pixels,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyUNet(input_channels=3, base_channels=config.base_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_score = -float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(1, config.num_epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, config)
        val_metrics = run_epoch(model, val_loader, None, device, config) if len(val_dataset) else {}
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_iou={train_metrics['mask_iou']:.3f} "
            f"train_empty={train_metrics['empty_acc']:.3f} "
            f"val_loss={val_metrics.get('loss', float('nan')):.4f} "
            f"val_iou={val_metrics.get('mask_iou', float('nan')):.3f} "
            f"val_empty={val_metrics.get('empty_acc', float('nan')):.3f}"
        )

        save_checkpoint(output_dir / "last.pt", model, optimizer, config, epoch, row)
        if config.checkpoint_period > 0 and epoch % config.checkpoint_period == 0:
            periodic_path = output_dir / f"epoch_{epoch:04d}.pt"
            save_checkpoint(periodic_path, model, optimizer, config, epoch, row)
            print(f"[ckpt] saved {periodic_path.name}")

        val_iou = float(val_metrics.get("mask_iou", train_metrics["mask_iou"]))
        val_empty = float(val_metrics.get("empty_acc", train_metrics["empty_acc"]))
        score = val_iou + 0.25 * val_empty
        if score > best_score:
            best_score = score
            save_checkpoint(output_dir / "best.pt", model, optimizer, config, epoch, row)
            print(f"[ckpt] updated best.pt score={best_score:.4f}")

    print(f"[done] best_score={best_score:.4f} checkpoint={output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
