"""Interactively label task-specific SAM masks for recorded demo frames.

This script is intentionally offline: it reads saved ``frame_*`` folders and
does not create a robot env or camera/tactile process.

Controls:
  - 1 / 2 / 3   select the task-specific mask slot
  - left drag   add a box prompt for the active mask
  - left click  add a positive point/polygon point for the active mask
  - right click add a negative point, or undo one polygon point in manual mode
  - s           save only the current frame's annotations
  - n / p       next / previous selected frame
  - c / z / d   clear active prompts / undo / delete active saved mask
  - q / Esc     save all pending annotations and quit
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_EXP_NAME = "tennis_ball_pick_and_place"
DEFAULT_FRAME_ROOT = (
    "/home/ealin/workspaces/DexTacHil/data/recorded_data/"
    "tennis_ball_pick_and_place/"
    "tennis_ball_pick_and_place-2026-08-14_12-18-59"
)
CONTROL_PANEL_WIDTH = 320
MASK_COLORS = {
    "mask1": (40, 220, 40),
    "hand_mask": (0, 165, 255),
    "mask2": (220, 40, 220),
    "candidate": (0, 220, 255),
}


@dataclass(frozen=True)
class MaskTaskSpec:
    """Mask slots and default text prompts for one experiment."""

    slots: tuple[str, ...]
    labels: tuple[str, ...]
    file_stems: dict[str, str]
    prompts: dict[str, str]


MASK_TASK_CONFIGS: dict[str, MaskTaskSpec] = {
    "tennis_ball_pick": MaskTaskSpec(
        slots=("mask1", "mask2"),
        labels=("ball", "basket"),
        file_stems={"mask1": "ball_mask", "mask2": "basket_mask"},
        prompts={"mask1": "tennis ball", "mask2": "basket"},
    ),
    "tennis_ball_pick_and_place": MaskTaskSpec(
        slots=("mask1", "hand_mask", "mask2"),
        labels=("ball", "hand", "basket"),
        file_stems={
            "mask1": "ball_mask",
            "hand_mask": "hand_mask",
            "mask2": "basket_mask",
        },
        prompts={
            "mask1": "tennis ball",
            "hand_mask": "robot end effector",
            "mask2": "basket",
        },
    ),
}
# Compatibility constant for older two-output helper scripts. The interactive
# labeler itself uses the task-specific ``MaskTaskSpec.slots`` above.
MASK_SLOTS = ("mask1", "mask2")


def mask_color(slot: str) -> tuple[int, int, int]:
    """Return a stable display color, including for future task-specific slots."""

    fallback_colors = (
        (40, 220, 40),
        (0, 165, 255),
        (220, 40, 220),
        (255, 120, 40),
        (120, 40, 220),
    )
    if slot in MASK_COLORS:
        return MASK_COLORS[slot]
    return fallback_colors[abs(hash(slot)) % len(fallback_colors)]


@dataclass
class SlotPrompt:
    points: list[list[float]] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    boxes: list[list[float]] = field(default_factory=list)
    polygon: list[tuple[int, int]] = field(default_factory=list)
    candidate_mask: np.ndarray | None = None
    candidate_score: float | None = None

    def clear(self) -> None:
        self.points.clear()
        self.labels.clear()
        self.boxes.clear()
        self.polygon.clear()
        self.candidate_mask = None
        self.candidate_score = None


@dataclass
class PromptState:
    slots: dict[str, SlotPrompt] = field(
        default_factory=lambda: {"mask1": SlotPrompt(), "mask2": SlotPrompt()}
    )
    slot_labels: dict[str, str] = field(default_factory=dict)
    active_slot: str = "mask1"
    drag_start: tuple[int, int] | None = None
    drag_current: tuple[int, int] | None = None

    @property
    def active(self) -> SlotPrompt:
        return self.slots[self.active_slot]

    @property
    def points(self) -> list[list[float]]:
        return self.active.points

    @property
    def labels(self) -> list[int]:
        return self.active.labels

    @property
    def boxes(self) -> list[list[float]]:
        return self.active.boxes

    @property
    def polygon(self) -> list[tuple[int, int]]:
        return self.active.polygon

    @property
    def candidate_mask(self) -> np.ndarray | None:
        return self.active.candidate_mask

    @candidate_mask.setter
    def candidate_mask(self, value: np.ndarray | None) -> None:
        self.active.candidate_mask = value

    @property
    def candidate_score(self) -> float | None:
        return self.active.candidate_score

    @candidate_score.setter
    def candidate_score(self, value: float | None) -> None:
        self.active.candidate_score = value

    def clear_active(self) -> None:
        self.active.clear()
        self.drag_start = None
        self.drag_current = None

    def clear_all(self) -> None:
        for slot in self.slots.values():
            slot.clear()
        self.drag_start = None
        self.drag_current = None

    def undo(self, mode: str) -> None:
        if mode == "manual":
            if self.polygon:
                self.polygon.pop()
            self.candidate_mask = None
            return

        if self.drag_start is not None:
            self.drag_start = None
            self.drag_current = None
            return
        if self.boxes:
            self.boxes.pop()
        elif self.points:
            self.points.pop()
            self.labels.pop()
            self.candidate_mask = None
            self.candidate_score = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Label task-specific masks for recorded frame_* demo data. "
            "The mask slots are selected from --exp_name."
        )
    )
    parser.add_argument(
        "--exp_name",
        default=DEFAULT_EXP_NAME,
        choices=tuple(MASK_TASK_CONFIGS),
        help="Experiment name used to select the number and names of mask slots.",
    )
    parser.add_argument(
        "--frame_root",
        default=DEFAULT_FRAME_ROOT,
        help="Recorded demo directory. The default is explicitly set above.",
    )
    parser.add_argument("--image_name", default="color_image.jpg")
    parser.add_argument(
        "--metadata_name",
        default="recording_metadata.json",
        help="Recording metadata containing episode_ranges for demo-wise keyframes.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Default: <frame_root>/sam_masks",
    )
    parser.add_argument("--start_frame", type=int, default=None)
    parser.add_argument("--end_frame", type=int, default=None)
    parser.add_argument(
        "--keyframes_per_demo",
        type=int,
        default=15,
        help=(
            "Uniformly select this many keyframes from each recorded demo. "
            "Set <=0 to use --stride sampling instead."
        ),
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Fallback global stride when --keyframes_per_demo <= 0 or metadata is missing.",
    )
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--display_scale", type=float, default=1.5)
    parser.add_argument(
        "--model",
        choices=("sam2", "sam3", "manual"),
        default="sam3",
        help="Use SAM2 prompts, SAM3 text prompts, or manual polygon masks.",
    )
    parser.add_argument("--device", default=None, help="cuda/cpu; default auto.")
    parser.add_argument(
        "--sam2_repo_id",
        default="facebook/sam2-hiera-small",
        help="Used when SAM2ImagePredictor.from_pretrained is available.",
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
        "--allow_manual_fallback",
        action="store_true",
        help="Fall back to manual polygon mode when SAM2 cannot be loaded.",
    )
    parser.add_argument(
        "--auto_gaze_prompt",
        action="store_true",
        help="Automatically add gaze_uv_in_realsense as a positive prompt on frame load.",
    )
    parser.add_argument(
        "--no_save_in_frame_dir",
        action="store_true",
        help="Only save into --output_dir, not into each frame_* folder.",
    )
    parser.add_argument(
        "--save_legacy_rs_names",
        action="store_true",
        help="Also save rs_mask_obj0.png and rs_mask_obj1.png in each frame dir.",
    )
    parser.add_argument(
        "--sam3_prompt_mask1",
        default=None,
        help="SAM3 text prompt used for mask1 when --sam3_prompt_mode uses text.",
    )
    parser.add_argument(
        "--sam3_prompt_mask2",
        default=None,
        help="SAM3 text prompt used for mask2 when --sam3_prompt_mode uses text.",
    )
    parser.add_argument(
        "--sam3_prompt_hand",
        default=None,
        help="SAM3 text prompt used for hand_mask when text prompting is enabled.",
    )
    parser.add_argument(
        "--sam3_prompt_mode",
        choices=("box", "text", "text_box"),
        default="text_box",
        help=(
            "SAM3 prompting mode. 'box' is manual box-only; 'text' is automatic "
            "text concept segmentation; 'text_box' combines text and manual box."
        ),
    )
    parser.add_argument(
        "--sam3_confidence_threshold",
        type=float,
        default=0.3,
        help="SAM3 instance confidence threshold.",
    )
    parser.add_argument(
        "--sam3_checkpoint",
        default=os.environ.get("SAM3_CKPT", os.environ.get("SAM3_CHECKPOINT", "")),
        help="Optional local SAM3 checkpoint. If empty, SAM3 may download from Hugging Face.",
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
        help="Prefer SAM3 instances that contain or are closest to gaze point.",
    )
    parser.add_argument(
        "--sam3_autocast_bfloat16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA bfloat16 autocast for SAM3 inference, matching SAM3 examples.",
    )
    return parser.parse_args()


def apply_task_config(args: argparse.Namespace) -> argparse.Namespace:
    """Attach task-specific mask slots and resolve optional prompt overrides."""

    spec = MASK_TASK_CONFIGS[args.exp_name]
    args.mask_slots = spec.slots
    args.mask_labels = dict(zip(spec.slots, spec.labels))
    args.mask_file_stems = dict(spec.file_stems)
    args.sam3_prompts = dict(spec.prompts)

    prompt_overrides = {
        "mask1": args.sam3_prompt_mask1,
        "hand_mask": args.sam3_prompt_hand,
        "mask2": args.sam3_prompt_mask2,
    }
    for slot, override in prompt_overrides.items():
        if slot in args.sam3_prompts and override:
            args.sam3_prompts[slot] = override
    return args


def frame_id_from_dir(frame_dir: Path) -> int | None:
    if not frame_dir.name.startswith("frame_"):
        return None
    try:
        return int(frame_dir.name.split("_", 1)[1])
    except ValueError:
        return None


def read_recording_metadata(frame_root: Path, metadata_name: str) -> dict[str, Any] | None:
    metadata_path = frame_root / metadata_name
    if not metadata_path.exists():
        return None
    try:
        return json.loads(metadata_path.read_text())
    except Exception as exc:
        print(f"[warn] failed to read {metadata_path}: {exc}")
        return None


def sample_evenly(items: list[tuple[int, Path]], count: int) -> list[tuple[int, Path]]:
    if count <= 0 or len(items) <= count:
        return items
    indices = np.linspace(0, len(items) - 1, count)
    selected_indices = sorted({int(round(index)) for index in indices})
    return [items[index] for index in selected_indices]


def sample_demo_keyframes(
    args: argparse.Namespace,
    frame_root: Path,
    frames: list[tuple[int, Path]],
) -> list[tuple[int, Path]] | None:
    if int(args.keyframes_per_demo) <= 0:
        return None

    metadata = read_recording_metadata(frame_root, args.metadata_name)
    if metadata is None:
        print(
            f"[warn] {frame_root / args.metadata_name} not found; "
            "falling back to global stride sampling."
        )
        return None

    selected: list[tuple[int, Path]] = []
    seen_frame_ids: set[int] = set()
    for fallback_index, episode in enumerate(metadata.get("episode_ranges", [])):
        start_frame = int(episode["start_frame"])
        end_frame = int(episode["end_frame"])
        episode_index = int(episode.get("episode_index", fallback_index))
        episode_frames = [
            (frame_id, frame_dir)
            for frame_id, frame_dir in frames
            if start_frame <= frame_id <= end_frame
        ]
        if not episode_frames:
            continue
        keyframes = sample_evenly(episode_frames, int(args.keyframes_per_demo))
        print(
            f"[demo keyframes] episode={episode_index} "
            f"range={start_frame}-{end_frame} "
            f"frames={len(episode_frames)} selected={len(keyframes)}"
        )
        for frame_id, frame_dir in keyframes:
            if frame_id in seen_frame_ids:
                continue
            selected.append((frame_id, frame_dir))
            seen_frame_ids.add(frame_id)

    if not selected:
        print(
            f"[warn] no episode_ranges in {frame_root / args.metadata_name} "
            "overlap discovered frames; falling back to global stride sampling."
        )
        return None
    return selected


def discover_frames(
    args: argparse.Namespace,
    *,
    apply_sampling: bool = True,
) -> list[tuple[int, Path]]:
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
    if apply_sampling:
        demo_keyframes = sample_demo_keyframes(args, frame_root, frames)
        if demo_keyframes is not None:
            frames = demo_keyframes
        else:
            stride = max(1, int(args.stride))
            frames = frames[::stride]
        if args.max_frames is not None:
            frames = frames[: max(0, int(args.max_frames))]
    return frames


def load_sam2_predictor(args: argparse.Namespace) -> Any:
    import torch

    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.sam2_checkpoint and args.sam2_model_cfg:
        from sam2.build_sam import build_sam2

        model = build_sam2(args.sam2_model_cfg, args.sam2_checkpoint, device=device)
        predictor = SAM2ImagePredictor(model)
    else:
        if not hasattr(SAM2ImagePredictor, "from_pretrained"):
            raise RuntimeError(
                "SAM2ImagePredictor.from_pretrained is unavailable in this SAM2 "
                "install. Pass --sam2_model_cfg and --sam2_checkpoint, or run "
                "with --model=manual."
            )
        try:
            predictor = SAM2ImagePredictor.from_pretrained(
                args.sam2_repo_id, device=device
            )
        except TypeError:
            predictor = SAM2ImagePredictor.from_pretrained(args.sam2_repo_id)
    print(f"[SAM2] loaded predictor on {device}")
    return predictor


def load_sam3_processor(args: argparse.Namespace) -> Any:
    import torch

    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.sam3_checkpoint or None
    model = build_sam3_image_model(
        device=device,
        checkpoint_path=checkpoint_path,
        load_from_HF=not args.sam3_no_hf_download,
    )
    processor = Sam3Processor(
        model,
        device=device,
        confidence_threshold=args.sam3_confidence_threshold,
    )
    print(
        "[SAM3] loaded processor "
        f"device={device} checkpoint={checkpoint_path or 'HF/default'}"
    )
    return processor


def read_image(frame_dir: Path, image_name: str) -> np.ndarray:
    image = cv2.imread(str(frame_dir / image_name), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(frame_dir / image_name)
    return image


def load_gaze_point(frame_dir: Path) -> tuple[int, int] | None:
    path = frame_dir / "gaze_contact.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if data.get("hit") is False:
            return None
        gaze_uv = data.get("gaze_uv_in_realsense")
        if gaze_uv is None or len(gaze_uv) < 2:
            return None
        return int(round(float(gaze_uv[0]))), int(round(float(gaze_uv[1])))
    except Exception:
        return None


def make_manual_mask(shape: tuple[int, int], polygon: list[tuple[int, int]]) -> np.ndarray | None:
    if len(polygon) < 3:
        return None
    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.asarray(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask > 0


def predict_sam2_mask(
    predictor: Any,
    image_bgr: np.ndarray,
    state: PromptState,
) -> tuple[np.ndarray | None, float | None]:
    if not state.points and not state.boxes:
        return None, None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    point_coords = None
    point_labels = None
    if state.points:
        point_coords = np.asarray(state.points, dtype=np.float32)
        point_labels = np.asarray(state.labels, dtype=np.int32)

    box = None
    if state.boxes:
        box = np.asarray(state.boxes[-1], dtype=np.float32)

    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        multimask_output=True,
    )
    best_idx = int(np.argmax(scores))
    return masks[best_idx].astype(bool), float(scores[best_idx])


def sam3_prompt_for_slot(args: argparse.Namespace, slot: str) -> str:
    prompts = getattr(args, "sam3_prompts", None)
    if prompts is not None:
        return prompts.get(slot, slot.replace("_", " "))
    # Compatibility for the older automatic SAM3 inference script.
    legacy_prompts = {
        "mask1": getattr(args, "sam3_prompt_mask1", "tennis ball"),
        "mask2": getattr(args, "sam3_prompt_mask2", "basket"),
        "hand_mask": getattr(args, "sam3_prompt_hand", "robot hand"),
    }
    return legacy_prompts.get(slot, slot.replace("_", " "))


def xyxy_to_normalized_cxcywh(
    box_xyxy: list[float],
    image_shape: tuple[int, int, int],
) -> list[float]:
    height, width = image_shape[:2]
    x0, y0, x1, y1 = box_xyxy
    x0, x1 = sorted((max(0.0, min(x0, width - 1.0)), max(0.0, min(x1, width - 1.0))))
    y0, y1 = sorted((max(0.0, min(y0, height - 1.0)), max(0.0, min(y1, height - 1.0))))
    box_w = max(1.0, x1 - x0)
    box_h = max(1.0, y1 - y0)
    center_x = x0 + box_w * 0.5
    center_y = y0 + box_h * 0.5
    return [
        center_x / float(width),
        center_y / float(height),
        box_w / float(width),
        box_h / float(height),
    ]


def tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
        if hasattr(value, "is_floating_point") and value.is_floating_point():
            value = value.float()
        return value.cpu().numpy()
    return np.asarray(value)


def distance_to_mask(mask: np.ndarray, point: tuple[int, int]) -> float:
    y, x = int(point[1]), int(point[0])
    height, width = mask.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        return float("inf")
    if mask[y, x]:
        return 0.0
    coords = np.argwhere(mask)
    if coords.size == 0:
        return float("inf")
    diff = coords.astype(np.float32) - np.asarray([[y, x]], dtype=np.float32)
    return float(np.sqrt(np.min(np.sum(diff * diff, axis=1))))


def choose_sam3_instance(
    masks: np.ndarray,
    scores: np.ndarray,
    gaze_point: tuple[int, int] | None,
    select_by_gaze: bool,
) -> int:
    if len(masks) == 0:
        return -1
    if not select_by_gaze or gaze_point is None:
        return int(np.argmax(scores))

    distances = np.asarray([distance_to_mask(mask, gaze_point) for mask in masks])
    containing = distances == 0.0
    if np.any(containing):
        containing_indices = np.flatnonzero(containing)
        local_best = int(np.argmax(scores[containing_indices]))
        return int(containing_indices[local_best])

    finite = np.isfinite(distances)
    if not np.any(finite):
        return int(np.argmax(scores))
    image_diag = max(1.0, float(np.sqrt(masks.shape[-2] ** 2 + masks.shape[-1] ** 2)))
    combined = scores.astype(np.float32) - (distances / image_diag).astype(np.float32)
    return int(np.argmax(combined))


def sam3_autocast_context(args: argparse.Namespace) -> Any:
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.sam3_autocast_bfloat16 and str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def predict_sam3_mask(
    processor: Any,
    image_bgr: np.ndarray,
    state: PromptState,
    args: argparse.Namespace,
    gaze_point: tuple[int, int] | None,
) -> tuple[np.ndarray | None, float | None]:
    from PIL import Image

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_rgb)
    prompt = sam3_prompt_for_slot(args, state.active_slot)
    effective_prompt_mode = args.sam3_prompt_mode
    if args.sam3_prompt_mode == "text_box" and not state.boxes:
        effective_prompt_mode = "text"

    with sam3_autocast_context(args):
        inference_state = processor.set_image(image)

        if effective_prompt_mode in ("text", "text_box"):
            inference_state = processor.set_text_prompt(prompt=prompt, state=inference_state)

        if effective_prompt_mode == "box" and not state.boxes:
            print("[SAM3] box mode needs a manual box. Drag a box with left mouse first.")
            return None, None

        if effective_prompt_mode in ("box", "text_box"):
            for box_xyxy in state.boxes:
                box = xyxy_to_normalized_cxcywh(box_xyxy, image_bgr.shape)
                inference_state = processor.add_geometric_prompt(
                    box=box,
                    label=True,
                    state=inference_state,
                )

        masks = tensor_to_numpy(inference_state.get("masks", np.empty((0,))))
        scores = tensor_to_numpy(inference_state.get("scores", np.empty((0,))))
    if masks.size == 0 or scores.size == 0:
        print(f"[SAM3] no mask found mode={args.sam3_prompt_mode} prompt={prompt!r}")
        return None, None

    masks = np.squeeze(masks).astype(bool)
    if masks.ndim == 2:
        masks = masks[None]
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    best_idx = choose_sam3_instance(
        masks,
        scores,
        gaze_point=gaze_point,
        select_by_gaze=args.sam3_select_by_gaze,
    )
    if best_idx < 0:
        return None, None
    print(
        f"[SAM3] mode={effective_prompt_mode} prompt={prompt!r} instances={len(masks)} "
        f"selected={best_idx} score={float(scores[best_idx]):.3f}"
    )
    return masks[best_idx], float(scores[best_idx])


def output_paths(
    output_dir: Path,
    frame_id: int,
    frame_dir: Path,
    slot: str,
    save_in_frame_dir: bool,
    save_legacy_rs_names: bool,
    file_stem: str,
) -> list[Path]:
    paths = [output_dir / f"frame_{frame_id:06d}_{file_stem}.png"]
    if save_in_frame_dir:
        paths.append(frame_dir / f"{file_stem}.png")
        if save_legacy_rs_names:
            legacy_names = {
                "mask1": "rs_mask_obj0.png",
                "mask2": "rs_mask_obj1.png",
            }
            if slot in legacy_names:
                paths.append(frame_dir / legacy_names[slot])
    return paths


def save_mask(
    mask: np.ndarray,
    frame_id: int,
    frame_dir: Path,
    output_dir: Path,
    slot: str,
    args: argparse.Namespace,
    state: PromptState,
    model_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_uint8 = (mask.astype(np.uint8) * 255)
    paths = output_paths(
        output_dir,
        frame_id,
        frame_dir,
        slot,
        save_in_frame_dir=not args.no_save_in_frame_dir,
        save_legacy_rs_names=args.save_legacy_rs_names,
        file_stem=args.mask_file_stems.get(slot, slot),
    )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), mask_uint8)

    metadata_path = output_dir / f"frame_{frame_id:06d}_masks.json"
    existing: dict[str, Any] = {}
    if metadata_path.exists():
        try:
            existing = json.loads(metadata_path.read_text())
        except Exception:
            existing = {}
    existing.update(
        {
            "frame_id": int(frame_id),
            "frame_dir": str(frame_dir),
            "image_name": args.image_name,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    existing[slot] = {
        "model": model_name,
        "score": state.candidate_score,
        "points": state.points,
        "labels": state.labels,
        "boxes": state.boxes,
        "polygon": [list(point) for point in state.polygon],
        "paths": [str(path) for path in paths],
    }
    metadata_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
    print(f"[save] frame={frame_id} {slot} -> {paths[0]}")


def load_saved_masks(
    output_dir: Path,
    frame_id: int,
    frame_dir: Path,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for slot in args.mask_slots:
        file_stem = args.mask_file_stems.get(slot, slot)
        candidates = [
            output_dir / f"frame_{frame_id:06d}_{file_stem}.png",
            frame_dir / f"{file_stem}.png",
            # Read annotations produced by older versions of this script.
            output_dir / f"frame_{frame_id:06d}_{slot}.png",
            frame_dir / f"{slot}.png",
        ]
        if args.save_legacy_rs_names:
            legacy_names = {
                "mask1": "rs_mask_obj0.png",
                "mask2": "rs_mask_obj1.png",
            }
            if slot in legacy_names:
                candidates.append(frame_dir / legacy_names[slot])
        for path in candidates:
            if not path.exists():
                continue
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                masks[slot] = mask > 0
                break
    return masks


def delete_mask(
    output_dir: Path,
    frame_id: int,
    frame_dir: Path,
    slot: str,
    args: argparse.Namespace,
) -> None:
    paths = output_paths(
        output_dir,
        frame_id,
        frame_dir,
        slot,
        save_in_frame_dir=not args.no_save_in_frame_dir,
        save_legacy_rs_names=args.save_legacy_rs_names,
        file_stem=args.mask_file_stems.get(slot, slot),
    )
    for path in paths:
        if path.exists():
            path.unlink()
    print(f"[delete] frame={frame_id} {slot}")


def blend_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    out = image.copy()
    color_arr = np.asarray(color, dtype=np.uint8)
    out[mask] = (out[mask].astype(np.float32) * (1.0 - alpha) + color_arr * alpha).astype(
        np.uint8
    )
    return out


def draw_control_panel(panel: np.ndarray, lines: list[str]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    thickness = 1
    y = 24
    for line in lines:
        (width, height), _ = cv2.getTextSize(line, font, scale, thickness)
        if width > panel.shape[1] - 20:
            scale_for_line = max(0.28, scale * (panel.shape[1] - 20) / width)
        else:
            scale_for_line = scale
        cv2.putText(
            panel,
            line,
            (12, y),
            font,
            scale_for_line,
            (235, 235, 235),
            thickness,
            cv2.LINE_AA,
        )
        y += 17
        if y >= panel.shape[0] - 8:
            break


def render_view(
    image: np.ndarray,
    state: PromptState,
    saved_masks: dict[str, np.ndarray],
    hidden_saved_slots: set[str],
    frame_id: int,
    frame_index: int,
    total_frames: int,
    gaze_point: tuple[int, int] | None,
    model_name: str,
    display_scale: float,
) -> np.ndarray:
    view = image.copy()
    for slot, mask in saved_masks.items():
        if slot in hidden_saved_slots:
            continue
        view = blend_mask(view, mask, mask_color(slot), alpha=0.38)

    for slot, prompt in state.slots.items():
        slot_color = mask_color(slot)
        if prompt.candidate_mask is not None:
            view = blend_mask(view, prompt.candidate_mask, slot_color, alpha=0.32)
        for point, label in zip(prompt.points, prompt.labels):
            color = slot_color if label == 1 else (0, 0, 255)
            cv2.circle(view, (int(point[0]), int(point[1])), 5, color, -1)
        for box in prompt.boxes:
            x0, y0, x1, y1 = [int(v) for v in box]
            thickness = 3 if slot == state.active_slot else 2
            cv2.rectangle(view, (x0, y0), (x1, y1), slot_color, thickness)
        if len(prompt.polygon) >= 1:
            for point in prompt.polygon:
                cv2.circle(view, point, 4, slot_color, -1)
            if len(prompt.polygon) >= 2:
                cv2.polylines(
                    view,
                    [np.asarray(prompt.polygon, dtype=np.int32)],
                    isClosed=False,
                    color=slot_color,
                    thickness=2,
                )

    if state.drag_start is not None and state.drag_current is not None:
        cv2.rectangle(
            view,
            state.drag_start,
            state.drag_current,
            mask_color(state.active_slot),
            2,
        )

    if gaze_point is not None:
        cv2.drawMarker(
            view,
            gaze_point,
            (255, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
        )

    score = "none" if state.candidate_score is None else f"{state.candidate_score:.3f}"
    prompt_status = "text+box" if model_name == "sam3" else model_name
    panel = np.full(
        (view.shape[0], CONTROL_PANEL_WIDTH, 3),
        (28, 32, 38),
        dtype=np.uint8,
    )
    panel_lines = [
        "SAM MASK LABELER",
        "------------------------------",
        f"frame: {frame_id} ({frame_index + 1}/{total_frames})",
        f"model: {model_name}  mode: {prompt_status}",
        f"active: {state.active_slot}",
        f"candidate score: {score}",
        "",
        "MASK SLOTS",
    ]
    panel_lines.extend(
        f"{index + 1}: {slot} ({state.slot_labels.get(slot, slot)})"
        for index, slot in enumerate(state.slots)
    )
    panel_lines.extend(
        [
            "",
            "CONTROLS",
            "Left drag  : add/refine box",
            "Left click : positive point",
            "Right click: negative point",
            "Middle click: undo",
            "",
            "c: clear active prompt",
            "z/u: undo prompt",
            "s: save current frame only",
            "d: delete active saved mask",
            "r: reload saved masks",
            "n/. : next frame",
            "p/, : previous frame",
            "g: add/use gaze point",
            "q / Esc: save all and quit",
        ]
    )
    draw_control_panel(panel, panel_lines)
    view = np.hstack((view, panel))

    if display_scale != 1.0:
        view = cv2.resize(
            view,
            None,
            fx=display_scale,
            fy=display_scale,
            interpolation=cv2.INTER_NEAREST,
        )
    return view


def window_flags() -> int:
    flags = cv2.WINDOW_NORMAL
    if hasattr(cv2, "WINDOW_KEEPRATIO"):
        flags |= cv2.WINDOW_KEEPRATIO
    if hasattr(cv2, "WINDOW_GUI_NORMAL"):
        flags |= cv2.WINDOW_GUI_NORMAL
    return flags


def scaled_point(x: int, y: int, display_scale: float, image_shape: tuple[int, int, int]) -> tuple[int, int]:
    if display_scale != 1.0:
        x = int(round(x / display_scale))
        y = int(round(y / display_scale))
    height, width = image_shape[:2]
    return max(0, min(x, width - 1)), max(0, min(y, height - 1))


def update_candidate(
    state: PromptState,
    args: argparse.Namespace,
    predictor: Any,
    image: np.ndarray,
    gaze_point: tuple[int, int] | None,
) -> None:
    if args.model == "manual":
        state.candidate_mask = make_manual_mask(image.shape[:2], state.polygon)
        state.candidate_score = None
        return
    if args.model == "sam3":
        mask, score = predict_sam3_mask(predictor, image, state, args, gaze_point)
        state.candidate_mask = mask
        state.candidate_score = score
        return
    mask, score = predict_sam2_mask(predictor, image, state)
    state.candidate_mask = mask
    state.candidate_score = score


def add_gaze_prompt(state: PromptState, gaze_point: tuple[int, int] | None) -> bool:
    if gaze_point is None:
        return False
    state.points.append([float(gaze_point[0]), float(gaze_point[1])])
    state.labels.append(1)
    state.candidate_mask = None
    state.candidate_score = None
    return True


def sam3_can_predict_from_text(args: argparse.Namespace) -> bool:
    return args.model == "sam3" and args.sam3_prompt_mode in ("text", "text_box")


def update_sam3_text_candidates(
    state: PromptState,
    args: argparse.Namespace,
    predictor: Any,
    image: np.ndarray,
    gaze_point: tuple[int, int] | None,
) -> None:
    old_slot = state.active_slot
    try:
        for slot in state.slots:
            state.active_slot = slot
            prompt = state.active
            if prompt.candidate_mask is not None or prompt.boxes:
                continue
            update_candidate(state, args, predictor, image, gaze_point)
    finally:
        state.active_slot = old_slot


def slot_has_annotation(prompt: SlotPrompt, args: argparse.Namespace) -> bool:
    if args.model == "manual":
        return len(prompt.polygon) >= 3 or prompt.candidate_mask is not None
    if args.model == "sam3" and args.sam3_prompt_mode in ("text", "text_box"):
        return True
    return bool(prompt.points or prompt.boxes or prompt.candidate_mask is not None)


def save_all_current_frame_annotations(
    state: PromptState,
    args: argparse.Namespace,
    predictor: Any,
    image: np.ndarray,
    gaze_point: tuple[int, int] | None,
    frame_id: int,
    frame_dir: Path,
    output_dir: Path,
    model_name: str,
) -> int:
    saved_count = 0
    old_slot = state.active_slot
    try:
        for slot in state.slots:
            state.active_slot = slot
            prompt = state.active
            if not slot_has_annotation(prompt, args):
                continue
            if prompt.candidate_mask is None:
                update_candidate(state, args, predictor, image, gaze_point)
            if prompt.candidate_mask is None:
                print(f"[skip] frame={frame_id} {slot}: no candidate mask")
                continue
            save_mask(
                prompt.candidate_mask,
                frame_id,
                frame_dir,
                output_dir,
                slot,
                args,
                state,
                model_name,
            )
            saved_count += 1
    finally:
        state.active_slot = old_slot
    if saved_count == 0:
        if args.model == "sam2":
            print(
                "[save skipped] SAM2 needs a point/box prompt first. "
                "Use --model=sam3 --sam3_prompt_mode=text_box for automatic text masks."
            )
        elif args.model == "sam3" and args.sam3_prompt_mode == "box":
            print(
                "[save skipped] SAM3 box mode needs a drawn box first. "
                "Use --sam3_prompt_mode=text_box for text auto masks."
            )
        else:
            print("[save skipped] no candidate mask was produced for this frame.")
    print(f"[save] frame={frame_id} saved_slots={saved_count}")
    return saved_count


def main() -> None:
    args = apply_task_config(parse_args())
    print(
        f"[task] exp_name={args.exp_name} "
        f"mask_slots={list(args.mask_slots)} "
        f"labels={args.mask_labels}"
    )
    frame_root = Path(args.frame_root).expanduser()
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else frame_root / "sam_masks"
    print(f"[data] frame_root={frame_root}")
    print(f"[data] output_dir={output_dir}")

    frames = discover_frames(args)
    if not frames:
        raise RuntimeError(f"No frame_* directories with {args.image_name} found under {frame_root}")

    predictor = None
    model_name = args.model
    if args.model == "sam2":
        try:
            predictor = load_sam2_predictor(args)
        except Exception as exc:
            if not args.allow_manual_fallback:
                raise
            print(f"[SAM2] failed to load ({type(exc).__name__}: {exc}); using manual mode.")
            args.model = "manual"
            model_name = "manual"
    elif args.model == "sam3":
        try:
            predictor = load_sam3_processor(args)
        except Exception as exc:
            if not args.allow_manual_fallback:
                raise
            print(f"[SAM3] failed to load ({type(exc).__name__}: {exc}); using manual mode.")
            args.model = "manual"
            model_name = "manual"
    print(
        f"[mode] model={args.model} sam3_prompt_mode={args.sam3_prompt_mode} "
        f"auto_text={sam3_can_predict_from_text(args)} "
        f"sam3_autocast_bfloat16={args.sam3_autocast_bfloat16}"
    )

    window_name = "label_recorded_sam_masks"
    cv2.namedWindow(window_name, window_flags())

    state = PromptState(
        slots={slot: SlotPrompt() for slot in args.mask_slots},
        slot_labels=args.mask_labels,
        active_slot=args.mask_slots[0],
    )
    frame_index = 0
    saved_masks: dict[str, np.ndarray] = {}
    image: np.ndarray | None = None
    gaze_point: tuple[int, int] | None = None
    hidden_saved_slots: set[str] = set()
    # Navigation stages edits in memory. Explicit `s` saves only the current
    # frame; the final flush saves every staged frame before exiting.
    pending_annotations: dict[
        int, tuple[PromptState, np.ndarray, tuple[int, int] | None]
    ] = {}
    final_flush_done = False
    dirty = False
    current_display_scale = max(0.1, float(args.display_scale))

    def stage_current_frame() -> None:
        if image is None:
            return
        pending_annotations[frame_index] = (
            copy.deepcopy(state),
            image.copy(),
            gaze_point,
        )

    def save_all_pending_annotations() -> int:
        nonlocal final_flush_done
        stage_current_frame()
        saved_count = 0
        for pending_index in sorted(pending_annotations):
            pending_state, pending_image, pending_gaze = pending_annotations[pending_index]
            pending_frame_id, pending_frame_dir = frames[pending_index]
            saved_count += save_all_current_frame_annotations(
                pending_state,
                args,
                predictor,
                pending_image,
                pending_gaze,
                pending_frame_id,
                pending_frame_dir,
                output_dir,
                model_name,
            )
        pending_annotations.clear()
        final_flush_done = True
        return saved_count

    def load_frame(index: int) -> None:
        nonlocal image, saved_masks, gaze_point, dirty
        frame_id, frame_dir = frames[index]
        image = read_image(frame_dir, args.image_name)
        saved_masks = load_saved_masks(output_dir, frame_id, frame_dir, args)
        gaze_point = load_gaze_point(frame_dir)
        hidden_saved_slots.clear()
        state.active_slot = args.mask_slots[0]
        state.clear_all()
        dirty = False
        if args.auto_gaze_prompt and args.model == "sam2":
            if add_gaze_prompt(state, gaze_point):
                dirty = True
        if sam3_can_predict_from_text(args):
            update_sam3_text_candidates(state, args, predictor, image, gaze_point)
        print(
            f"[frame] {index + 1}/{len(frames)} id={frame_id} "
            f"saved={sorted(saved_masks.keys())} gaze={gaze_point}"
        )

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        nonlocal dirty
        if image is None:
            return
        image_display_width = int(round(image.shape[1] * current_display_scale))
        if x >= image_display_width:
            return
        px, py = scaled_point(x, y, current_display_scale, image.shape)

        if args.model == "manual":
            if event == cv2.EVENT_LBUTTONDOWN:
                hidden_saved_slots.add(state.active_slot)
                state.polygon.append((px, py))
                dirty = True
            elif event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN):
                state.undo(args.model)
                dirty = True
            return

        if args.model == "sam3":
            if event == cv2.EVENT_LBUTTONDOWN:
                hidden_saved_slots.add(state.active_slot)
                state.drag_start = (px, py)
                state.drag_current = (px, py)
            elif event == cv2.EVENT_MOUSEMOVE and state.drag_start is not None:
                state.drag_current = (px, py)
            elif event == cv2.EVENT_LBUTTONUP and state.drag_start is not None:
                x0, y0 = state.drag_start
                x1, y1 = px, py
                if abs(x1 - x0) > 5 or abs(y1 - y0) > 5:
                    left, right = sorted((x0, x1))
                    top, bottom = sorted((y0, y1))
                    state.boxes.append([float(left), float(top), float(right), float(bottom)])
                    state.candidate_mask = None
                    state.candidate_score = None
                    dirty = True
                state.drag_start = None
                state.drag_current = None
            elif event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN):
                if event == cv2.EVENT_RBUTTONDOWN:
                    hidden_saved_slots.add(state.active_slot)
                state.undo(args.model)
                dirty = True
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            hidden_saved_slots.add(state.active_slot)
            state.drag_start = (px, py)
            state.drag_current = (px, py)
        elif event == cv2.EVENT_MOUSEMOVE and state.drag_start is not None:
            state.drag_current = (px, py)
        elif event == cv2.EVENT_LBUTTONUP and state.drag_start is not None:
            x0, y0 = state.drag_start
            x1, y1 = px, py
            if abs(x1 - x0) > 5 or abs(y1 - y0) > 5:
                left, right = sorted((x0, x1))
                top, bottom = sorted((y0, y1))
                state.boxes.append([float(left), float(top), float(right), float(bottom)])
            else:
                state.points.append([float(px), float(py)])
                state.labels.append(1)
            state.drag_start = None
            state.drag_current = None
            dirty = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            hidden_saved_slots.add(state.active_slot)
            state.points.append([float(px), float(py)])
            state.labels.append(0)
            dirty = True
        elif event == cv2.EVENT_MBUTTONDOWN:
            state.undo(args.model)
            dirty = True

    load_frame(frame_index)
    if image is not None:
        cv2.resizeWindow(
            window_name,
            max(
                100,
                int((image.shape[1] + CONTROL_PANEL_WIDTH) * current_display_scale),
            ),
            max(100, int(image.shape[0] * current_display_scale)),
        )
    cv2.setMouseCallback(window_name, on_mouse)

    try:
        while True:
            frame_id, frame_dir = frames[frame_index]
            if image is None:
                load_frame(frame_index)
            assert image is not None

            if dirty:
                update_candidate(state, args, predictor, image, gaze_point)
                dirty = False

            view = render_view(
                image,
                state,
                saved_masks,
                hidden_saved_slots,
                frame_id,
                frame_index,
                len(frames),
                gaze_point,
                model_name,
                current_display_scale,
            )
            cv2.imshow(window_name, view)
            key = cv2.waitKey(30) & 0xFF
            slot_by_key = {
                ord(str(index + 1)): slot
                for index, slot in enumerate(args.mask_slots)
            }

            if key in (27, ord("q")):
                break
            if key in (ord("n"), ord(".")):
                stage_current_frame()
                if frame_index >= len(frames) - 1:
                    saved_count = save_all_pending_annotations()
                    print(
                        f"[save] reached final frame; saved_slots={saved_count}; exiting"
                    )
                    break
                frame_index += 1
                load_frame(frame_index)
            elif key in (ord("p"), ord(",")):
                stage_current_frame()
                frame_index = max(frame_index - 1, 0)
                load_frame(frame_index)
            elif key == ord("c"):
                state.clear_active()
                hidden_saved_slots.add(state.active_slot)
            elif key in (ord("z"), ord("u")):
                state.undo(args.model)
                dirty = True
            elif key == ord("g"):
                if args.model == "manual":
                    print("[gaze] gaze prompt is only used in sam2/sam3 modes.")
                elif args.model == "sam3":
                    if gaze_point is not None:
                        state.candidate_mask = None
                        state.candidate_score = None
                        dirty = True
                        print(f"[gaze] SAM3 will prefer instance near {gaze_point}")
                    else:
                        print("[gaze] no valid gaze_contact.json point for this frame.")
                elif add_gaze_prompt(state, gaze_point):
                    dirty = True
                    print(f"[gaze] added positive prompt {gaze_point}")
                else:
                    print("[gaze] no valid gaze_contact.json point for this frame.")
            elif key in slot_by_key:
                state.active_slot = slot_by_key[key]
                if sam3_can_predict_from_text(args) and state.candidate_mask is None:
                    dirty = True
                print(f"[active] {state.active_slot}")
            elif key == ord("s"):
                saved_count = save_all_current_frame_annotations(
                    state,
                    args,
                    predictor,
                    image,
                    gaze_point,
                    frame_id,
                    frame_dir,
                    output_dir,
                    model_name,
                )
                pending_annotations.pop(frame_index, None)
                saved_masks = load_saved_masks(output_dir, frame_id, frame_dir, args)
                hidden_saved_slots.intersection_update(
                    set(args.mask_slots) - set(saved_masks)
                )
                print(
                    f"[save] current frame_index={frame_index} "
                    f"frame_id={frame_id} saved_slots={saved_count}"
                )
            elif key == ord("d"):
                delete_mask(output_dir, frame_id, frame_dir, state.active_slot, args)
                state.clear_active()
                hidden_saved_slots.add(state.active_slot)
                pending_annotations.pop(frame_index, None)
                saved_masks = load_saved_masks(output_dir, frame_id, frame_dir, args)
            elif key == ord("r"):
                saved_masks = load_saved_masks(output_dir, frame_id, frame_dir, args)
                print(f"[reload] frame={frame_id} saved={sorted(saved_masks.keys())}")
    finally:
        if not final_flush_done:
            saved_count = save_all_pending_annotations()
            print(f"[save] exit flush saved_slots={saved_count}")
        cv2.destroyWindow(window_name)


if __name__ == "__main__":
    main()
