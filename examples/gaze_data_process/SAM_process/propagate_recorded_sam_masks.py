"""Propagate saved keyframe SAM masks through recorded demo frames.

This script is intentionally separate from the SAM annotation script in this
folder, so SAM3 annotation can exit and release GPU memory before SAM2 video
propagation starts.

Input:
  - ``<frame_root>/frame_*/color_image.jpg``
  - keyframe masks saved by ``SAM_process/label_recorded_sam_masks.py`` in ``mask_dir``

Output:
  - propagated masks in ``<mask_dir>/propagated/frame_XXXXXX_mask1.png``
  - propagated masks in ``<mask_dir>/propagated/frame_XXXXXX_mask2.png``
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DEFAULT_FRAME_ROOT = (
    "/media/user/data3/wrq/recorded_data/tennis_ball_pick/tennis_ball_pick-6-23-1"
)
MASK_SLOTS = ("mask1", "mask2")
SAM2_OBJECT_IDS = {"mask1": 1, "mask2": 2}
SAM2_SLOTS_BY_OBJECT_ID = {value: key for key, value in SAM2_OBJECT_IDS.items()}


@dataclass
class MaskSeed:
    frame_id: int
    frame_dir: Path
    slot: str
    mask: np.ndarray


@dataclass
class PropagationSegment:
    episode_index: int
    start_frame: int
    end_frame: int
    frames: list[tuple[int, Path]]
    success: bool | None = None
    interrupted: bool | None = None


def frame_id_from_dir(frame_dir: Path) -> int | None:
    if not frame_dir.name.startswith("frame_"):
        return None
    try:
        return int(frame_dir.name.split("_", 1)[1])
    except ValueError:
        return None


def tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
        if hasattr(value, "is_floating_point") and value.is_floating_point():
            value = value.float()
        return value.cpu().numpy()
    return np.asarray(value)


def load_sam2_video_predictor(args: argparse.Namespace) -> Any:
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.sam2_checkpoint and args.sam2_model_cfg:
        from sam2.build_sam import build_sam2_video_predictor

        predictor = build_sam2_video_predictor(
            args.sam2_model_cfg,
            args.sam2_checkpoint,
            device=device,
        )
    else:
        try:
            from sam2.sam2_video_predictor import SAM2VideoPredictor

            if hasattr(SAM2VideoPredictor, "from_pretrained"):
                try:
                    predictor = SAM2VideoPredictor.from_pretrained(
                        args.sam2_repo_id,
                        device=device,
                    )
                except TypeError:
                    predictor = SAM2VideoPredictor.from_pretrained(args.sam2_repo_id)
            else:
                raise ImportError
        except ImportError:
            from sam2.build_sam import build_sam2_video_predictor_hf

            predictor = build_sam2_video_predictor_hf(
                args.sam2_repo_id,
                device=device,
            )
    print(f"[SAM2] loaded video predictor on {device}")
    return predictor


def prepare_sam2_video_frames(
    frames: list[tuple[int, Path]],
    image_name: str,
    video_dir: Path,
) -> None:
    video_dir.mkdir(parents=True, exist_ok=True)
    for local_idx, (_, frame_dir) in enumerate(frames):
        source = (frame_dir / image_name).resolve()
        target = video_dir / f"{local_idx:06d}.jpg"
        if target.exists() or target.is_symlink():
            try:
                if target.resolve() == source.resolve():
                    continue
            except FileNotFoundError:
                pass
            target.unlink()

        if source.suffix.lower() in (".jpg", ".jpeg"):
            try:
                os.symlink(source, target)
                continue
            except OSError:
                shutil.copy2(source, target)
                continue

        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(source)
        cv2.imwrite(str(target), image)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM2 propagation from saved keyframe mask1/mask2 annotations."
    )
    parser.add_argument("--frame_root", default=DEFAULT_FRAME_ROOT)
    parser.add_argument("--image_name", default="color_image.jpg")
    parser.add_argument(
        "--metadata_name",
        default="recording_metadata.json",
        help="Recording metadata file containing episode_ranges.",
    )
    parser.add_argument(
        "--segment_by_demo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use recording_metadata.json episode_ranges and propagate each demo "
            "independently so masks never cross demo boundaries."
        ),
    )
    parser.add_argument(
        "--mask_dir",
        default=None,
        help="Directory containing frame_XXXXXX_mask*.png. Default: <frame_root>/sam_masks",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Default: <mask_dir>/propagated; stores summary and optional central mask copies.",
    )
    parser.add_argument("--start_frame", type=int, default=None)
    parser.add_argument("--end_frame", type=int, default=None)
    parser.add_argument("--device", default=None, help="cuda/cpu; default auto.")
    parser.add_argument(
        "--sam2_repo_id",
        default="facebook/sam2-hiera-small",
        help="Used when SAM2VideoPredictor.from_pretrained is available.",
    )
    parser.add_argument(
        "--sam2_model_cfg",
        default=os.environ.get("SAM2_MODEL_CFG", ""),
        help="SAM2 yaml config, used with --sam2_checkpoint.",
    )
    parser.add_argument(
        "--sam2_checkpoint",
        default=os.environ.get("SAM2_CKPT", os.environ.get("SAM2_CHECKPOINT", "")),
        help="SAM2 checkpoint, used with --sam2_model_cfg.",
    )
    parser.add_argument(
        "--mask_threshold",
        type=float,
        default=0.0,
        help="Threshold applied to SAM2 video mask logits.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite existing propagated masks.",
    )
    parser.add_argument(
        "--save_in_frame_dir",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write propagated mask1.png/mask2.png into each frame_* folder.",
    )
    parser.add_argument(
        "--save_central_masks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also save frame_XXXXXX_mask*.png copies in --output_dir. By default "
            "--output_dir only stores propagation_summary.json."
        ),
    )
    parser.add_argument(
        "--separate_objects",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Propagate mask1 and mask2 separately to reduce GPU memory.",
    )
    parser.add_argument(
        "--chunked",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Optional low-memory fallback: propagate between neighboring keyframes "
            "in small chunks. By default, each demo is propagated as one video."
        ),
    )
    parser.add_argument(
        "--reverse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also propagate backward before the earliest keyframe seed.",
    )
    parser.add_argument(
        "--offload_video_to_cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use SAM2 CPU video offload to reduce GPU memory.",
    )
    parser.add_argument(
        "--offload_state_to_cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use SAM2 CPU state offload to reduce GPU memory more.",
    )
    parser.add_argument(
        "--sam2_autocast_bfloat16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA bfloat16 autocast during SAM2 propagation to reduce GPU memory.",
    )
    parser.add_argument(
        "--occlusion_mode",
        choices=("none", "keyframe", "until_next_visible"),
        default="keyframe",
        help=(
            "How to treat annotated keyframes where mask1/mask2 is missing. "
            "'keyframe' clears only that frame; 'until_next_visible' clears from "
            "that keyframe until the next keyframe where the object has a mask."
        ),
    )
    return parser.parse_args()


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


def build_propagation_segments(
    args: argparse.Namespace,
    frame_root: Path,
    frames: list[tuple[int, Path]],
) -> list[PropagationSegment]:
    if not args.segment_by_demo:
        return [
            PropagationSegment(
                episode_index=0,
                start_frame=frames[0][0],
                end_frame=frames[-1][0],
                frames=frames,
            )
        ]

    metadata = read_recording_metadata(frame_root, args.metadata_name)
    if metadata is None:
        print(
            f"[warn] {frame_root / args.metadata_name} not found; "
            "falling back to one full sequence."
        )
        return [
            PropagationSegment(
                episode_index=0,
                start_frame=frames[0][0],
                end_frame=frames[-1][0],
                frames=frames,
            )
        ]

    segments: list[PropagationSegment] = []
    for fallback_index, episode in enumerate(metadata.get("episode_ranges", [])):
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
            PropagationSegment(
                episode_index=int(episode.get("episode_index", fallback_index)),
                start_frame=start_frame,
                end_frame=end_frame,
                frames=segment_frames,
                success=bool(episode["success"]) if "success" in episode else None,
                interrupted=(
                    bool(episode["interrupted"]) if "interrupted" in episode else None
                ),
            )
        )

    if not segments:
        raise RuntimeError(
            f"No episode_ranges in {frame_root / args.metadata_name} overlap discovered frames."
        )
    return segments


def central_mask_path(mask_dir: Path, frame_id: int, slot: str) -> Path:
    return mask_dir / f"frame_{frame_id:06d}_{slot}.png"


def metadata_path(mask_dir: Path, frame_id: int) -> Path:
    return mask_dir / f"frame_{frame_id:06d}_masks.json"


def load_mask(mask_dir: Path, frame_id: int, frame_dir: Path, slot: str) -> np.ndarray | None:
    candidates = [
        central_mask_path(mask_dir, frame_id, slot),
        frame_dir / f"{slot}.png",
    ]
    for path in candidates:
        if not path.exists():
            continue
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is not None and np.any(mask > 0):
            return mask > 0
    return None


def is_annotation_keyframe(mask_dir: Path, frame_id: int) -> bool:
    if metadata_path(mask_dir, frame_id).exists():
        return True
    return any(central_mask_path(mask_dir, frame_id, slot).exists() for slot in MASK_SLOTS)


def collect_annotations(
    frames: list[tuple[int, Path]],
    mask_dir: Path,
) -> tuple[list[MaskSeed], dict[int, set[str]], dict[str, set[int]]]:
    seeds: list[MaskSeed] = []
    annotated_slots_by_frame: dict[int, set[str]] = {}
    absent_frame_ids_by_slot: dict[str, set[int]] = {slot: set() for slot in MASK_SLOTS}

    for frame_id, frame_dir in frames:
        present_slots: set[str] = set()
        for slot in MASK_SLOTS:
            mask = load_mask(mask_dir, frame_id, frame_dir, slot)
            if mask is None:
                continue
            present_slots.add(slot)
            seeds.append(MaskSeed(frame_id=frame_id, frame_dir=frame_dir, slot=slot, mask=mask))

        if is_annotation_keyframe(mask_dir, frame_id):
            annotated_slots_by_frame[frame_id] = present_slots
            for slot in MASK_SLOTS:
                if slot not in present_slots:
                    absent_frame_ids_by_slot[slot].add(frame_id)

    return seeds, annotated_slots_by_frame, absent_frame_ids_by_slot


def build_occluded_indices(
    frames: list[tuple[int, Path]],
    annotated_slots_by_frame: dict[int, set[str]],
    args: argparse.Namespace,
) -> dict[str, set[int]]:
    occluded_indices_by_slot: dict[str, set[int]] = {slot: set() for slot in MASK_SLOTS}
    if args.occlusion_mode == "none":
        return occluded_indices_by_slot

    frame_id_to_idx = {frame_id: idx for idx, (frame_id, _) in enumerate(frames)}
    annotated_frame_ids = sorted(
        frame_id for frame_id in annotated_slots_by_frame if frame_id in frame_id_to_idx
    )

    for slot in MASK_SLOTS:
        for frame_id in annotated_frame_ids:
            if slot in annotated_slots_by_frame[frame_id]:
                continue
            start_idx = frame_id_to_idx[frame_id]
            if args.occlusion_mode == "keyframe":
                occluded_indices_by_slot[slot].add(start_idx)
                continue

            next_visible_idx = len(frames)
            for next_frame_id in annotated_frame_ids:
                if next_frame_id <= frame_id:
                    continue
                if slot in annotated_slots_by_frame[next_frame_id]:
                    next_visible_idx = frame_id_to_idx[next_frame_id]
                    break
            occluded_indices_by_slot[slot].update(range(start_idx, next_visible_idx))

    return occluded_indices_by_slot


def clear_output_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for path in output_dir.glob("frame_*_mask*.png"):
        path.unlink()
    summary = output_dir / "propagation_summary.json"
    if summary.exists():
        summary.unlink()


def clear_video_dir(video_dir: Path) -> None:
    if not video_dir.exists():
        return
    for path in video_dir.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def release_cuda_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def save_mask(
    mask: np.ndarray,
    frame_id: int,
    frame_dir: Path,
    output_dir: Path,
    slot: str,
    *,
    save_in_frame_dir: bool,
    save_central_masks: bool,
) -> Path:
    mask_uint8 = np.squeeze(mask).astype(np.uint8) * 255
    path = frame_dir / f"{slot}.png"
    if save_central_masks:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"frame_{frame_id:06d}_{slot}.png"
        cv2.imwrite(str(path), mask_uint8)
    if save_in_frame_dir:
        frame_path = frame_dir / f"{slot}.png"
        cv2.imwrite(str(frame_path), mask_uint8)
        if not save_central_masks:
            path = frame_path
    return path


def sam2_autocast_context(args: argparse.Namespace) -> Any:
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.sam2_autocast_bfloat16 and str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def empty_mask_like_frame(frame_dir: Path, image_name: str) -> np.ndarray:
    image = cv2.imread(str(frame_dir / image_name), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(frame_dir / image_name)
    return np.zeros(image.shape[:2], dtype=bool)


def write_explicit_occlusions(
    frames: list[tuple[int, Path]],
    output_dir: Path,
    args: argparse.Namespace,
    occluded_indices_by_slot: dict[str, set[int]],
    saved_keys: set[tuple[int, str]],
) -> int:
    count = 0
    for slot, occluded_indices in occluded_indices_by_slot.items():
        for idx in sorted(occluded_indices):
            frame_id, frame_dir = frames[idx]
            key = (frame_id, slot)
            if key in saved_keys:
                continue
            empty = empty_mask_like_frame(frame_dir, args.image_name)
            save_mask(
                empty,
                frame_id,
                frame_dir,
                output_dir,
                slot,
                save_in_frame_dir=args.save_in_frame_dir,
                save_central_masks=args.save_central_masks,
            )
            saved_keys.add(key)
            count += 1
    return count


def propagate_slots(
    predictor: Any,
    frames: list[tuple[int, Path]],
    video_dir: Path,
    output_dir: Path,
    seeds: list[MaskSeed],
    args: argparse.Namespace,
    occluded_indices_by_slot: dict[str, set[int]],
    slots: tuple[str, ...],
) -> set[tuple[int, str]]:
    frame_id_to_idx = {frame_id: idx for idx, (frame_id, _) in enumerate(frames)}
    saved_keys: set[tuple[int, str]] = set()

    try:
        inference_state = predictor.init_state(
            video_path=str(video_dir),
            offload_video_to_cpu=args.offload_video_to_cpu,
            offload_state_to_cpu=args.offload_state_to_cpu,
            async_loading_frames=False,
        )
    except TypeError:
        inference_state = predictor.init_state(video_path=str(video_dir))

    selected_seeds = [seed for seed in seeds if seed.slot in slots]
    seed_indices: list[int] = []
    with sam2_autocast_context(args):
        for seed in selected_seeds:
            local_idx = frame_id_to_idx[seed.frame_id]
            seed_indices.append(local_idx)
            predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=local_idx,
                obj_id=SAM2_OBJECT_IDS[seed.slot],
                mask=seed.mask,
            )
            print(f"[seed] frame={seed.frame_id} idx={local_idx} slot={seed.slot}")

        if not seed_indices:
            return saved_keys

        def consume_outputs(*, reverse: bool) -> None:
            start_frame_idx = min(seed_indices) if reverse else None
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state,
                start_frame_idx=start_frame_idx,
                reverse=reverse,
            ):
                local_idx = int(out_frame_idx)
                if not 0 <= local_idx < len(frames):
                    continue
                frame_id, frame_dir = frames[local_idx]
                for obj_offset, obj_id in enumerate(out_obj_ids):
                    obj_id_int = int(obj_id.item()) if hasattr(obj_id, "item") else int(obj_id)
                    slot = SAM2_SLOTS_BY_OBJECT_ID.get(obj_id_int)
                    if slot not in slots:
                        continue
                    mask = tensor_to_numpy(out_mask_logits[obj_offset] > args.mask_threshold)
                    mask = np.squeeze(mask).astype(bool)
                    if local_idx in occluded_indices_by_slot.get(slot, set()):
                        mask = np.zeros_like(mask, dtype=bool)
                    save_mask(
                        mask,
                        frame_id,
                        frame_dir,
                        output_dir,
                        slot,
                        save_in_frame_dir=args.save_in_frame_dir,
                        save_central_masks=args.save_central_masks,
                    )
                    saved_keys.add((frame_id, slot))

        print(
            f"[sam2 propagate] slots={slots} direction=forward "
            f"frames={len(frames)}"
        )
        consume_outputs(reverse=False)
        if args.reverse and min(seed_indices) > 0:
            print(
                f"[sam2 propagate] slots={slots} direction=reverse "
                f"frames=0..{min(seed_indices)} count={min(seed_indices) + 1}"
            )
            consume_outputs(reverse=True)

    for seed in selected_seeds:
        save_mask(
            seed.mask,
            seed.frame_id,
            seed.frame_dir,
            output_dir,
            seed.slot,
            save_in_frame_dir=args.save_in_frame_dir,
            save_central_masks=args.save_central_masks,
        )
        saved_keys.add((seed.frame_id, seed.slot))

    return saved_keys


def propagate_slot_window(
    predictor: Any,
    frames: list[tuple[int, Path]],
    chunk_video_dir: Path,
    output_dir: Path,
    seed_inputs: list[MaskSeed],
    args: argparse.Namespace,
    occluded_indices: set[int],
    slot: str,
    start_idx: int,
    end_idx: int,
    *,
    reverse: bool,
) -> set[tuple[int, str]]:
    if start_idx > end_idx:
        return set()

    frame_id_to_idx = {frame_id: idx for idx, (frame_id, _) in enumerate(frames)}
    chunk_frames = frames[start_idx : end_idx + 1]
    clear_video_dir(chunk_video_dir)
    prepare_sam2_video_frames(chunk_frames, args.image_name, chunk_video_dir)

    try:
        inference_state = predictor.init_state(
            video_path=str(chunk_video_dir),
            offload_video_to_cpu=args.offload_video_to_cpu,
            offload_state_to_cpu=args.offload_state_to_cpu,
            async_loading_frames=False,
        )
    except TypeError:
        inference_state = predictor.init_state(video_path=str(chunk_video_dir))

    seed_indices: list[int] = []
    saved_keys: set[tuple[int, str]] = set()
    with sam2_autocast_context(args):
        for seed in seed_inputs:
            global_idx = frame_id_to_idx[seed.frame_id]
            if not start_idx <= global_idx <= end_idx:
                continue
            local_idx = global_idx - start_idx
            seed_indices.append(local_idx)
            predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=local_idx,
                obj_id=SAM2_OBJECT_IDS[slot],
                mask=seed.mask,
            )

        if not seed_indices:
            return saved_keys

        start_frame_idx = max(seed_indices) if reverse else min(seed_indices)
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=start_frame_idx,
            reverse=reverse,
        ):
            local_idx = int(out_frame_idx)
            if not 0 <= local_idx < len(chunk_frames):
                continue
            global_idx = start_idx + local_idx
            frame_id, frame_dir = frames[global_idx]
            for obj_offset, obj_id in enumerate(out_obj_ids):
                obj_id_int = int(obj_id.item()) if hasattr(obj_id, "item") else int(obj_id)
                if SAM2_SLOTS_BY_OBJECT_ID.get(obj_id_int) != slot:
                    continue
                mask = tensor_to_numpy(out_mask_logits[obj_offset] > args.mask_threshold)
                mask = np.squeeze(mask).astype(bool)
                if global_idx in occluded_indices:
                    mask = np.zeros_like(mask, dtype=bool)
                save_mask(
                    mask,
                    frame_id,
                    frame_dir,
                    output_dir,
                    slot,
                    save_in_frame_dir=args.save_in_frame_dir,
                    save_central_masks=args.save_central_masks,
                )
                saved_keys.add((frame_id, slot))

    del inference_state
    release_cuda_cache()
    return saved_keys


def propagate_slot_chunked(
    predictor: Any,
    frames: list[tuple[int, Path]],
    chunk_video_root: Path,
    output_dir: Path,
    seeds: list[MaskSeed],
    args: argparse.Namespace,
    occluded_indices_by_slot: dict[str, set[int]],
    slot: str,
) -> set[tuple[int, str]]:
    frame_id_to_idx = {frame_id: idx for idx, (frame_id, _) in enumerate(frames)}
    slot_seeds = sorted(
        [seed for seed in seeds if seed.slot == slot],
        key=lambda seed: frame_id_to_idx[seed.frame_id],
    )
    if not slot_seeds:
        return set()

    chunk_video_dir = chunk_video_root / f"chunk_{slot}"
    saved_keys: set[tuple[int, str]] = set()
    seed_global_indices = [frame_id_to_idx[seed.frame_id] for seed in slot_seeds]
    print(
        f"[chunked] {slot}: windows={max(1, len(slot_seeds) - 1)} "
        f"seeds={len(slot_seeds)}"
    )

    first_seed = slot_seeds[0]
    first_idx = seed_global_indices[0]
    if args.reverse and first_idx > 0:
        print(f"[chunk] {slot}: before first seed frames=0..{first_idx}")
        saved_keys.update(
            propagate_slot_window(
                predictor,
                frames,
                chunk_video_dir,
                output_dir,
                [first_seed],
                args,
                occluded_indices_by_slot.get(slot, set()),
                slot,
                0,
                first_idx,
                reverse=True,
            )
        )

    for seed_idx in range(len(slot_seeds) - 1):
        start_seed = slot_seeds[seed_idx]
        end_seed = slot_seeds[seed_idx + 1]
        start_idx = seed_global_indices[seed_idx]
        end_idx = seed_global_indices[seed_idx + 1]
        print(f"[chunk] {slot}: frames={start_idx}..{end_idx}")
        saved_keys.update(
            propagate_slot_window(
                predictor,
                frames,
                chunk_video_dir,
                output_dir,
                [start_seed, end_seed],
                args,
                occluded_indices_by_slot.get(slot, set()),
                slot,
                start_idx,
                end_idx,
                reverse=False,
            )
        )

    last_seed = slot_seeds[-1]
    last_idx = seed_global_indices[-1]
    if last_idx < len(frames) - 1:
        print(f"[chunk] {slot}: after last seed frames={last_idx}..{len(frames) - 1}")
        saved_keys.update(
            propagate_slot_window(
                predictor,
                frames,
                chunk_video_dir,
                output_dir,
                [last_seed],
                args,
                occluded_indices_by_slot.get(slot, set()),
                slot,
                last_idx,
                len(frames) - 1,
                reverse=False,
            )
        )

    for seed in slot_seeds:
        save_mask(
            seed.mask,
            seed.frame_id,
            seed.frame_dir,
            output_dir,
            seed.slot,
            save_in_frame_dir=args.save_in_frame_dir,
            save_central_masks=args.save_central_masks,
        )
        saved_keys.add((seed.frame_id, seed.slot))

    clear_video_dir(chunk_video_dir)
    return saved_keys


def segment_log_name(segment: PropagationSegment) -> str:
    return (
        f"episode={segment.episode_index} "
        f"frames={segment.start_frame}..{segment.end_frame} "
        f"actual={segment.frames[0][0]}..{segment.frames[-1][0]}"
    )


def propagate_segment(
    predictor: Any,
    segment: PropagationSegment,
    mask_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[set[tuple[int, str]], dict[str, set[int]], dict[str, Any]]:
    seeds, annotated_slots_by_frame, _ = collect_annotations(segment.frames, mask_dir)
    segment_summary: dict[str, Any] = {
        "episode_index": segment.episode_index,
        "start_frame": segment.start_frame,
        "end_frame": segment.end_frame,
        "actual_start_frame": segment.frames[0][0],
        "actual_end_frame": segment.frames[-1][0],
        "num_frames": len(segment.frames),
        "success": segment.success,
        "interrupted": segment.interrupted,
        "num_seed_masks": len(seeds),
        "num_annotated_keyframes": len(annotated_slots_by_frame),
        "num_saved_masks": 0,
        "skipped": False,
    }

    print(
        f"[segment] {segment_log_name(segment)} "
        f"num_frames={len(segment.frames)} seeds={len(seeds)} "
        f"keyframes={len(annotated_slots_by_frame)}"
    )
    if not seeds:
        print(f"[skip segment] {segment_log_name(segment)} has no saved mask seeds")
        segment_summary["skipped"] = True
        segment_summary["skip_reason"] = "no_seed_masks"
        return set(), {slot: set() for slot in MASK_SLOTS}, segment_summary

    occluded_indices_by_slot = build_occluded_indices(
        segment.frames,
        annotated_slots_by_frame,
        args,
    )
    for slot in MASK_SLOTS:
        print(
            f"[segment occlusion] episode={segment.episode_index} {slot}: "
            f"{len(occluded_indices_by_slot[slot])} frames mode={args.occlusion_mode}"
        )

    saved_keys: set[tuple[int, str]] = set()
    if args.chunked:
        chunk_video_root = mask_dir / "_sam2_video_chunks" / f"episode_{segment.episode_index:04d}"
        chunk_video_root.mkdir(parents=True, exist_ok=True)
        for slot in MASK_SLOTS:
            slot_seeds = [seed for seed in seeds if seed.slot == slot]
            if not slot_seeds:
                print(f"[skip] episode={segment.episode_index} no seeds for slot={slot}")
                continue
            print(f"[propagate slot chunked] episode={segment.episode_index} slot={slot}")
            saved_keys.update(
                propagate_slot_chunked(
                    predictor,
                    segment.frames,
                    chunk_video_root,
                    output_dir,
                    slot_seeds,
                    args,
                    occluded_indices_by_slot,
                    slot,
                )
            )
            release_cuda_cache()
    else:
        video_dir = mask_dir / "_sam2_video_frames" / f"episode_{segment.episode_index:04d}"
        clear_video_dir(video_dir)
        print(
            f"[prepare demo video] episode={segment.episode_index} "
            f"frames={len(segment.frames)} dir={video_dir}"
        )
        prepare_sam2_video_frames(segment.frames, args.image_name, video_dir)
        slot_groups = [(slot,) for slot in MASK_SLOTS] if args.separate_objects else [tuple(MASK_SLOTS)]
        for slots in slot_groups:
            slot_seeds = [seed for seed in seeds if seed.slot in slots]
            if not slot_seeds:
                print(f"[skip] episode={segment.episode_index} no seeds for slots={slots}")
                continue
            print(
                f"[propagate demo slots] episode={segment.episode_index} "
                f"frames={len(segment.frames)} slots={slots}"
            )
            saved_keys.update(
                propagate_slots(
                    predictor,
                    segment.frames,
                    video_dir,
                    output_dir,
                    slot_seeds,
                    args,
                    occluded_indices_by_slot,
                    slots,
                )
            )
            release_cuda_cache()
        clear_video_dir(video_dir)

    explicit_empty_count = write_explicit_occlusions(
        segment.frames,
        output_dir,
        args,
        occluded_indices_by_slot,
        saved_keys,
    )
    if explicit_empty_count:
        print(
            f"[occlusion] episode={segment.episode_index} "
            f"wrote explicit empty masks={explicit_empty_count}"
        )

    segment_summary["num_saved_masks"] = len(saved_keys)
    segment_summary["occluded_frames"] = {
        slot: [segment.frames[idx][0] for idx in sorted(indices)]
        for slot, indices in occluded_indices_by_slot.items()
    }
    return saved_keys, occluded_indices_by_slot, segment_summary


def save_summary(
    output_dir: Path,
    args: argparse.Namespace,
    frames: list[tuple[int, Path]],
    seeds: list[MaskSeed],
    annotated_slots_by_frame: dict[int, set[str]],
    occluded_frame_ids_by_slot: dict[str, set[int]],
    saved_keys: set[tuple[int, str]],
    segment_summaries: list[dict[str, Any]],
) -> None:
    summary = {
        "frame_root": str(Path(args.frame_root).expanduser()),
        "mask_dir": str(Path(args.mask_dir).expanduser() if args.mask_dir else ""),
        "output_dir": str(output_dir),
        "metadata_name": args.metadata_name,
        "segment_by_demo": args.segment_by_demo,
        "image_name": args.image_name,
        "num_frames": len(frames),
        "num_seed_masks": len(seeds),
        "num_annotated_keyframes": len(annotated_slots_by_frame),
        "num_saved_masks": len(saved_keys),
        "occlusion_mode": args.occlusion_mode,
        "occluded_frames": {
            slot: sorted(frame_ids)
            for slot, frame_ids in occluded_frame_ids_by_slot.items()
        },
        "segments": segment_summaries,
        "separate_objects": args.separate_objects,
        "chunked": args.chunked,
        "save_in_frame_dir": args.save_in_frame_dir,
        "save_central_masks": args.save_central_masks,
        "offload_video_to_cpu": args.offload_video_to_cpu,
        "offload_state_to_cpu": args.offload_state_to_cpu,
        "sam2_autocast_bfloat16": args.sam2_autocast_bfloat16,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (output_dir / "propagation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )


def main() -> None:
    args = parse_args()
    frame_root = Path(args.frame_root).expanduser()
    mask_dir = Path(args.mask_dir).expanduser() if args.mask_dir else frame_root / "sam_masks"
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else mask_dir / "propagated"

    frames = discover_frames(args)
    if not frames:
        raise RuntimeError(f"No frame_* directories with {args.image_name} found under {frame_root}")
    seeds, annotated_slots_by_frame, _ = collect_annotations(frames, mask_dir)
    if not seeds:
        raise RuntimeError(f"No saved mask1/mask2 keyframes found in {mask_dir}")
    segments = build_propagation_segments(args, frame_root, frames)

    if args.overwrite:
        clear_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[propagate] frames={len(frames)} seeds={len(seeds)} "
        f"keyframes={len(annotated_slots_by_frame)} segments={len(segments)} "
        f"output={output_dir} chunked={args.chunked} "
        f"segment_by_demo={args.segment_by_demo}"
    )

    predictor = load_sam2_video_predictor(args)
    all_saved_keys: set[tuple[int, str]] = set()
    all_occluded_frame_ids_by_slot: dict[str, set[int]] = {slot: set() for slot in MASK_SLOTS}
    segment_summaries: list[dict[str, Any]] = []

    for segment in segments:
        saved_keys, occluded_indices_by_slot, segment_summary = propagate_segment(
            predictor,
            segment,
            mask_dir,
            output_dir,
            args,
        )
        all_saved_keys.update(saved_keys)
        segment_summaries.append(segment_summary)
        for slot, indices in occluded_indices_by_slot.items():
            all_occluded_frame_ids_by_slot[slot].update(
                segment.frames[idx][0] for idx in indices
            )
        release_cuda_cache()

    save_summary(
        output_dir,
        args,
        frames,
        seeds,
        annotated_slots_by_frame,
        all_occluded_frame_ids_by_slot,
        all_saved_keys,
        segment_summaries,
    )
    print(f"[done] saved_masks={len(all_saved_keys)} output={output_dir}")


if __name__ == "__main__":
    main()
