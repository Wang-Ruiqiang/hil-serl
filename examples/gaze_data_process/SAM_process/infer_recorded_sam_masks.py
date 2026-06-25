"""Automatically infer mask1/mask2 for recorded demos with SAM3.

This script is for generating masks on a new recorded dataset without manually
labeling every frame and without training/using a mask predictor network.

It runs SAM3 text-prompt segmentation on each frame:
  - mask1: default prompt "tennis ball"
  - mask2: default prompt "basket"

When a gaze point is available in ``gaze_contact.json``, SAM3 instances are
selected by preferring masks that contain or are closest to that gaze point.

Default target:
  /media/user/data3/wrq/recorded_data/tennis_ball_pick/tennis_ball_pick-6-23-1

Output:
  <frame_dir>/mask1.png
  <frame_dir>/mask2.png
  <frame_dir>/sam3_mask_inference.json

The script processes one recorded episode at a time according to
``recording_metadata.json`` and clears CUDA cache between episodes.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from label_recorded_sam_masks import (  # noqa: E402
    MASK_SLOTS,
    PromptState,
    load_gaze_point,
    load_sam3_processor,
    predict_sam3_mask,
)


DEFAULT_FRAME_ROOT = (
    "/media/user/data3/wrq/recorded_data/tennis_ball_pick/tennis_ball_pick-6-23-1"
)


@dataclass
class EpisodeSegment:
    episode_index: int
    start_frame: int
    end_frame: int
    frames: list[tuple[int, Path]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer recorded frame mask1/mask2 directly with SAM3."
    )
    parser.add_argument("--frame_root", default=DEFAULT_FRAME_ROOT)
    parser.add_argument("--image_name", default="color_image.jpg")
    parser.add_argument("--metadata_name", default="recording_metadata.json")
    parser.add_argument("--device", default=None, help="cuda/cpu; default auto.")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Default: <frame_root>/sam_masks/sam3_inferred",
    )
    parser.add_argument("--start_frame", type=int, default=None)
    parser.add_argument("--end_frame", type=int, default=None)
    parser.add_argument("--start_episode", type=int, default=None)
    parser.add_argument("--end_episode", type=int, default=None)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument(
        "--segment_by_demo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use recording_metadata.json episode_ranges and clear cache per episode.",
    )
    parser.add_argument(
        "--sam3_prompt_mask1",
        default="tennis ball",
        help="SAM3 text prompt for mask1.",
    )
    parser.add_argument(
        "--sam3_prompt_mask2",
        default="basket",
        help="SAM3 text prompt for mask2.",
    )
    parser.add_argument(
        "--sam3_prompt_mode",
        choices=("text", "text_box"),
        default="text",
        help="Batch inference uses text prompts. text_box behaves as text without manual boxes.",
    )
    parser.add_argument(
        "--sam3_confidence_threshold",
        type=float,
        default=0.3,
        help="SAM3 instance confidence threshold.",
    )
    parser.add_argument(
        "--sam3_checkpoint",
        default="",
        help="Optional local SAM3 checkpoint. Empty means SAM3 default/HF cache.",
    )
    parser.add_argument(
        "--sam3_no_hf_download",
        action="store_true",
        help="Do not auto-download SAM3 checkpoint from Hugging Face.",
    )
    parser.add_argument(
        "--sam3_select_by_gaze",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer SAM3 instances containing/closest to gaze point.",
    )
    parser.add_argument(
        "--sam3_autocast_bfloat16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA bfloat16 autocast for SAM3 inference.",
    )
    parser.add_argument(
        "--save_in_frame_dir",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write mask1.png/mask2.png into each frame_* folder.",
    )
    parser.add_argument(
        "--save_central_masks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also save frame_XXXXXX_mask*.png copies into --output_dir.",
    )
    parser.add_argument(
        "--write_legacy_rs_names",
        action="store_true",
        help="Also write rs_mask_obj0.png/rs_mask_obj1.png in each frame folder.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite existing mask1.png/mask2.png outputs.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip frames where both mask1.png and mask2.png already exist.",
    )
    parser.add_argument(
        "--save_overlay",
        action="store_true",
        help="Save debug overlay images into <output_dir>/overlays.",
    )
    return parser.parse_args()


def frame_id_from_dir(frame_dir: Path) -> int | None:
    if not frame_dir.name.startswith("frame_"):
        return None
    try:
        return int(frame_dir.name.split("_", 1)[1])
    except ValueError:
        return None


def discover_frames(args: argparse.Namespace) -> list[tuple[int, Path]]:
    frame_root = Path(args.frame_root).expanduser()
    frames: list[tuple[int, Path]] = []
    for frame_dir in frame_root.iterdir():
        if not frame_dir.is_dir():
            continue
        frame_id = frame_id_from_dir(frame_dir)
        if frame_id is None:
            continue
        if args.start_frame is not None and frame_id < args.start_frame:
            continue
        if args.end_frame is not None and frame_id > args.end_frame:
            continue
        if not (frame_dir / args.image_name).exists():
            continue
        frames.append((frame_id, frame_dir))
    frames.sort(key=lambda item: item[0])
    return frames


def read_recording_metadata(frame_root: Path, metadata_name: str) -> dict[str, Any] | None:
    path = frame_root / metadata_name
    if not path.exists():
        return None
    return json.loads(path.read_text())


def build_episode_segments(
    args: argparse.Namespace,
    frames: list[tuple[int, Path]],
) -> list[EpisodeSegment]:
    if not frames:
        return []

    if not args.segment_by_demo:
        return [
            EpisodeSegment(
                episode_index=0,
                start_frame=frames[0][0],
                end_frame=frames[-1][0],
                frames=frames,
            )
        ]

    frame_root = Path(args.frame_root).expanduser()
    metadata = read_recording_metadata(frame_root, args.metadata_name)
    if metadata is None:
        print(f"[warn] missing {frame_root / args.metadata_name}; using one segment.")
        return [
            EpisodeSegment(
                episode_index=0,
                start_frame=frames[0][0],
                end_frame=frames[-1][0],
                frames=frames,
            )
        ]

    segments: list[EpisodeSegment] = []
    for fallback_index, episode in enumerate(metadata.get("episode_ranges", [])):
        episode_index = int(episode.get("episode_index", fallback_index))
        if args.start_episode is not None and episode_index < args.start_episode:
            continue
        if args.end_episode is not None and episode_index > args.end_episode:
            continue
        start_frame = int(episode["start_frame"])
        end_frame = int(episode["end_frame"])
        segment_frames = [
            (frame_id, frame_dir)
            for frame_id, frame_dir in frames
            if start_frame <= frame_id <= end_frame
        ]
        if not segment_frames:
            continue
        segments.append(
            EpisodeSegment(
                episode_index=episode_index,
                start_frame=start_frame,
                end_frame=end_frame,
                frames=segment_frames,
            )
        )
        if args.max_episodes is not None and len(segments) >= args.max_episodes:
            break
    return segments


def read_image(frame_dir: Path, image_name: str) -> np.ndarray:
    image = cv2.imread(str(frame_dir / image_name), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(frame_dir / image_name)
    return image


def empty_mask(image_shape: tuple[int, int]) -> np.ndarray:
    return np.zeros(image_shape, dtype=np.uint8)


def bool_mask_to_uint8(mask: np.ndarray | None, image_shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return empty_mask(image_shape)
    mask = np.squeeze(mask).astype(bool)
    if mask.shape != image_shape:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (image_shape[1], image_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return mask.astype(np.uint8) * 255


def predict_slot(
    processor: Any,
    image: np.ndarray,
    slot: str,
    gaze_point: tuple[int, int] | None,
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, float | None]:
    state = PromptState()
    state.active_slot = slot
    return predict_sam3_mask(processor, image, state, args, gaze_point)


def save_masks(
    frame_id: int,
    frame_dir: Path,
    output_dir: Path,
    masks: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> None:
    if args.save_in_frame_dir:
        for slot, mask in masks.items():
            path = frame_dir / f"{slot}.png"
            if args.overwrite or not path.exists():
                cv2.imwrite(str(path), mask)
        if args.write_legacy_rs_names:
            legacy_paths = {
                "mask1": frame_dir / "rs_mask_obj0.png",
                "mask2": frame_dir / "rs_mask_obj1.png",
            }
            for slot, path in legacy_paths.items():
                if args.overwrite or not path.exists():
                    cv2.imwrite(str(path), masks[slot])

    if args.save_central_masks:
        output_dir.mkdir(parents=True, exist_ok=True)
        for slot, mask in masks.items():
            path = output_dir / f"frame_{frame_id:06d}_{slot}.png"
            if args.overwrite or not path.exists():
                cv2.imwrite(str(path), mask)


def save_metadata(
    frame_id: int,
    frame_dir: Path,
    output_dir: Path,
    metadata: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    if args.save_in_frame_dir:
        (frame_dir / "sam3_mask_inference.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"frame_{frame_id:06d}_sam3_mask_inference.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False)
    )


def save_overlay(
    frame_id: int,
    image: np.ndarray,
    masks: dict[str, np.ndarray],
    metadata: dict[str, Any],
    output_dir: Path,
) -> None:
    overlay = image.copy()
    colors = {
        "mask1": np.array([40, 220, 40], dtype=np.uint8),
        "mask2": np.array([220, 40, 220], dtype=np.uint8),
    }
    for slot, color in colors.items():
        mask = masks[slot] > 0
        overlay[mask] = (0.55 * overlay[mask] + 0.45 * color).astype(np.uint8)
    gaze = metadata.get("gaze_uv_in_realsense")
    if gaze is not None:
        cv2.drawMarker(
            overlay,
            (int(round(gaze[0])), int(round(gaze[1]))),
            (255, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
        )
    cv2.putText(
        overlay,
        (
            f"frame={frame_id} "
            f"m1={metadata['mask1']['score']} m2={metadata['mask2']['score']}"
        ),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(overlay_dir / f"frame_{frame_id:06d}_sam3_overlay.jpg"), overlay)


def frame_has_existing_masks(frame_dir: Path) -> bool:
    return (frame_dir / "mask1.png").exists() and (frame_dir / "mask2.png").exists()


def release_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def infer_episode(
    processor: Any,
    segment: EpisodeSegment,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, int]:
    stats = {
        "frames": 0,
        "mask1_found": 0,
        "mask2_found": 0,
        "mask1_empty": 0,
        "mask2_empty": 0,
        "missing_gaze": 0,
        "skipped_existing": 0,
    }

    for frame_id, frame_dir in tqdm(
        segment.frames,
        desc=f"episode {segment.episode_index}",
        leave=False,
    ):
        if args.skip_existing and frame_has_existing_masks(frame_dir):
            stats["skipped_existing"] += 1
            continue

        image = read_image(frame_dir, args.image_name)
        image_shape = image.shape[:2]
        gaze_point = load_gaze_point(frame_dir)
        if gaze_point is None:
            stats["missing_gaze"] += 1

        masks: dict[str, np.ndarray] = {}
        slot_metadata: dict[str, dict[str, Any]] = {}
        for slot in MASK_SLOTS:
            mask, score = predict_slot(processor, image, slot, gaze_point, args)
            mask_uint8 = bool_mask_to_uint8(mask, image_shape)
            masks[slot] = mask_uint8
            found = mask is not None and bool(np.any(mask_uint8 > 0))
            stats[f"{slot}_found" if found else f"{slot}_empty"] += 1
            slot_metadata[slot] = {
                "prompt": args.sam3_prompt_mask1 if slot == "mask1" else args.sam3_prompt_mask2,
                "score": None if score is None else float(score),
                "found": bool(found),
                "pixels": int(np.count_nonzero(mask_uint8)),
            }

        metadata = {
            "frame_id": int(frame_id),
            "frame_dir": str(frame_dir),
            "image_name": args.image_name,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "model": "sam3",
            "episode_index": int(segment.episode_index),
            "sam3_prompt_mode": args.sam3_prompt_mode,
            "sam3_select_by_gaze": bool(args.sam3_select_by_gaze),
            "gaze_uv_in_realsense": (
                [int(gaze_point[0]), int(gaze_point[1])] if gaze_point is not None else None
            ),
            "mask1": slot_metadata["mask1"],
            "mask2": slot_metadata["mask2"],
        }
        save_masks(frame_id, frame_dir, output_dir, masks, args)
        save_metadata(frame_id, frame_dir, output_dir, metadata, args)
        if args.save_overlay:
            save_overlay(frame_id, image, masks, metadata, output_dir)
        stats["frames"] += 1

    return stats


def main() -> None:
    args = parse_args()
    frame_root = Path(args.frame_root).expanduser()
    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else frame_root / "sam_masks" / "sam3_inferred"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = discover_frames(args)
    if not frames:
        raise RuntimeError(f"No frame_* directories with {args.image_name} found under {frame_root}")
    segments = build_episode_segments(args, frames)
    if not segments:
        raise RuntimeError("No episode segments found.")

    print(
        f"[data] frame_root={frame_root} frames={len(frames)} "
        f"episodes={len(segments)} output={output_dir}"
    )
    print(
        f"[sam3] mask1={args.sam3_prompt_mask1!r} "
        f"mask2={args.sam3_prompt_mask2!r} "
        f"select_by_gaze={args.sam3_select_by_gaze}"
    )

    processor = load_sam3_processor(args)
    summary: dict[str, Any] = {
        "frame_root": str(frame_root),
        "output_dir": str(output_dir),
        "model": "sam3",
        "sam3_prompt_mask1": args.sam3_prompt_mask1,
        "sam3_prompt_mask2": args.sam3_prompt_mask2,
        "sam3_select_by_gaze": bool(args.sam3_select_by_gaze),
        "episodes": [],
        "total": {
            "frames": 0,
            "mask1_found": 0,
            "mask2_found": 0,
            "mask1_empty": 0,
            "mask2_empty": 0,
            "missing_gaze": 0,
            "skipped_existing": 0,
        },
    }

    for segment in segments:
        print(
            f"[episode {segment.episode_index}] "
            f"frames={len(segment.frames)} range={segment.start_frame}-{segment.end_frame}"
        )
        stats = infer_episode(processor, segment, output_dir, args)
        for key, value in stats.items():
            summary["total"][key] += int(value)
        summary["episodes"].append(
            {
                "episode_index": int(segment.episode_index),
                "start_frame": int(segment.start_frame),
                "end_frame": int(segment.end_frame),
                **{key: int(value) for key, value in stats.items()},
            }
        )
        (output_dir / "sam3_inference_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False)
        )
        release_memory()

    print(
        "[done] "
        f"frames={summary['total']['frames']} "
        f"mask1_found={summary['total']['mask1_found']} "
        f"mask2_found={summary['total']['mask2_found']} "
        f"missing_gaze={summary['total']['missing_gaze']} "
        f"summary={output_dir / 'sam3_inference_summary.json'}"
    )


if __name__ == "__main__":
    main()
