"""Offline pretraining for the ``vit-grounded`` RGB encoder.

Trains the exact ``ViTImageEncoder`` module that HIL-RL loads, using only the
40 demos recorded at t=0 (plus, optionally, task-agnostic ImageNet weights).
No RL replay data is used here: reusing a previous run's buffer would make the
comparison circular.

Objective (the trunk is shared by all four terms):
  * grounding KL   -- the single grounding query must attend to the ball,
                      applied on pick-phase frames only, exactly matching the
                      CGL loss that keeps running during RL
  * segmentation   -- a pretrain-only 1x1 head predicts ball/hand/basket masks
                      from the patch tokens
  * geometry       -- object centers / areas decoded from the 256D output
                      vector, forcing position into the readout
  * inverse action -- predict the action from (v_t, v_t+1, state); the one term
                      that forces control-relevant information into the code

Deliberately absent: the temporal-invariance term from the old CNN encoder. It
rewarded representations for *not* changing between adjacent frames, which is
the opposite of what the RL critic needs.

Example:
  python examples/encoder_training/train_encoder.py \
      --exp_name tennis_ball_pick_and_place \
      --frame_stride 1 --epochs 60
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import flax.linen as nn
from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from .task_configs import get_task_config, list_task_names
except ImportError:  # Allows running this file directly from its directory.
    from task_configs import get_task_config, list_task_names

from serl_launcher.utils.gaze_attention_target import (
    DEFAULT_DECAY,
    DEFAULT_DILATE_CELLS,
    DEFAULT_MAX_GAP,
    DEFAULT_SIGMA_CELLS,
    DEFAULT_WINDOW,
    build_target_heatmap,
    episode_gaze_index,
    pool_gaze_points,
    shift_points_for_crop,
)
from serl_launcher.vision.encoder_utils import mask_supervision_loss
from serl_launcher.vision.vit import ViTImageEncoder


DEFAULT_PHASE_SCAN = Path(__file__).resolve().parent / "runs" / "pick_classifier_phase_scan.json"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "runs"

# Name of the ViT submodule inside this pretraining model. The RL agent calls
# the same module "task_encoder" (Flax names submodules passed as dataclass
# attributes after the attribute). That is fine: the checkpoint stores the
# subtree *contents* under "params", and create_pixels splices them in by the
# RL-side name, so only the internal structure has to match.
VIT_MODULE_NAME = "encoder_front_camera"

def frame_dirs(demo_dir: Path, stride: int) -> List[Path]:
    frames = sorted(demo_dir.glob("frame_*/color_image.jpg"), key=lambda p: int(p.parent.name.split("_")[-1]))
    return [p.parent for p in frames[:: max(1, stride)]]


@dataclass(frozen=True)
class DemoRecord:
    dataset_name: str
    dataset_dir: Path
    episode_index: int
    frames: List[Path]
    first_place_frame: int | None = None
    # Every frame of the episode that survived filtering, unstrided. `frames`
    # above is subsampled by --frame_stride, and gaze pooling walks frame ids
    # one at a time with a max_gap of 3, so indexing the strided list would
    # silently find no neighbours at any stride above 3 and turn the temporal
    # pooling into a no-op without erroring.
    pooling_frames: List[Path] = field(default_factory=list)


def _frame_id(frame: Path) -> int:
    return int(frame.name.split("_")[-1])


def _all_frame_dirs(recording_dir: Path) -> List[Path]:
    return sorted(
        [p.parent for p in recording_dir.glob("frame_*/color_image.jpg")],
        key=_frame_id,
    )


# Set by main() from --session_filter. A recording root usually holds several
# sessions recorded on different days, and they are not interchangeable: the
# eye tracker is calibrated per session, so mixing sessions mixes calibrations.
_SESSION_FILTER: str = ""


def _recording_dirs(root: Path) -> List[Path]:
    if (root / "recording_metadata.json").exists():
        return [root]
    dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    dirs = [p for p in dirs
            if (p / "recording_metadata.json").exists() or _all_frame_dirs(p)]
    if _SESSION_FILTER:
        kept = [p for p in dirs if _SESSION_FILTER in p.name]
        if not kept:
            raise ValueError(
                f"--session_filter={_SESSION_FILTER!r} matched none of "
                f"{[p.name for p in dirs]}"
            )
        dirs = kept
    return dirs


def _sample_frames(frames: List[Path], stride: int) -> List[Path]:
    return frames[:: max(1, stride)]


def find_demos(root: Path, stride: int) -> List[DemoRecord]:
    """Find ordinary demos, treating a recording folder as one demo."""
    demos = []
    for dataset_dir in _recording_dirs(root):
        frames = _sample_frames(_all_frame_dirs(dataset_dir), stride)
        if frames:
            demos.append(DemoRecord(dataset_dir.name, dataset_dir, 0, frames))
    if not demos:
        raise FileNotFoundError(f"No demo/frame_* directories found under {root}")
    return demos


def find_episode_demos(
    root: Path, stride: int, kept_ranges_only: bool = False
) -> List[DemoRecord]:
    """Read successful episode boundaries without using a phase classifier.

    ``kept_ranges_only`` restricts each episode to its ``kept_frame_ranges``,
    which is the operator's manual gaze review. It matters only for gaze
    grounding, and it matters a lot there: a frame rejected for gaze drift
    still has a ``gaze_contact.json``, so without this it would be trained on
    with a wrong target *and* pooled into its neighbours' targets, spreading
    the error past the frame that carries it.
    """
    demos: List[DemoRecord] = []
    skipped = []
    for dataset_dir in _recording_dirs(root):
        metadata_path = dataset_dir / "recording_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Episode-aware training requires recording_metadata.json in {dataset_dir}"
            )
        metadata = json.loads(metadata_path.read_text())
        all_frames = {_frame_id(frame): frame for frame in _all_frame_dirs(dataset_dir)}
        for episode in metadata.get("episode_ranges", []):
            episode_index = int(episode["episode_index"])
            if not bool(episode.get("success", False)) or bool(
                episode.get("interrupted", False)
            ):
                skipped.append((dataset_dir.name, episode_index))
                continue
            spans = []
            if kept_ranges_only:
                for span in episode.get("kept_frame_ranges", []):
                    spans.append((int(span["start_frame"]), int(span["end_frame"])))
                if not spans:
                    skipped.append((dataset_dir.name, episode_index))
                    continue
            else:
                spans.append((int(episode["start_frame"]), int(episode["end_frame"])))
            frames = [
                all_frames[i]
                for start, end in spans
                for i in range(start, end + 1)
                if i in all_frames
            ]
            sampled = _sample_frames(frames, stride)
            if sampled:
                demos.append(
                    DemoRecord(
                        dataset_name=dataset_dir.name,
                        dataset_dir=dataset_dir,
                        episode_index=episode_index,
                        frames=sampled,
                        pooling_frames=frames,
                    )
                )
    if skipped:
        print(f"episode-aware: skipped {len(skipped)} unsuccessful episodes: {skipped}")
    if not demos:
        raise FileNotFoundError(f"No successful episode metadata found under {root}")
    return demos


def _load_phase_scan(path: Path) -> Dict[Tuple[str, int], int]:
    if not path.exists():
        raise FileNotFoundError(
            f"Phase scan not found: {path}. Run test_pick_classifier_phase.py first."
        )
    payload = json.loads(path.read_text())
    transitions: Dict[Tuple[str, int], int] = {}
    for result in payload.get("results", []):
        if result.get("excluded"):
            continue
        first_positive = result.get("first_positive_frame")
        if first_positive is None:
            raise ValueError(
                "Phase scan contains an episode without a classifier transition: "
                f"{result.get('dataset')} episode {result.get('episode_index')}"
            )
        transitions[(str(result["dataset"]), int(result["episode_index"]))] = int(first_positive)
    if not transitions:
        raise ValueError(f"No usable phase transitions found in {path}")
    return transitions


def find_phase_demos(
    root: Path, stride: int, phase_scan_path: Path, kept_ranges_only: bool = False
) -> List[DemoRecord]:
    """Expand recording metadata into episode-level, phase-aware demos.

    ``kept_ranges_only`` mirrors find_episode_demos: it restricts each episode to
    its ``kept_frame_ranges``. It defaults to False because the frames the gaze
    filter dropped are dropped for gaze reasons, and this branch is supervised by
    masks, not gaze -- the 2026-08-20 phase run that succeeded used full episode
    ranges. The body already referenced this name without the signature declaring
    it, which made every call raise NameError.
    """
    transitions = _load_phase_scan(phase_scan_path)
    demos: List[DemoRecord] = []
    skipped = []
    for dataset_dir in _recording_dirs(root):
        metadata_path = dataset_dir / "recording_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Phase-aware training requires recording_metadata.json in {dataset_dir}"
            )
        metadata = json.loads(metadata_path.read_text())
        all_frames = {_frame_id(frame): frame for frame in _all_frame_dirs(dataset_dir)}
        for episode in metadata.get("episode_ranges", []):
            episode_index = int(episode["episode_index"])
            key = (dataset_dir.name, episode_index)
            if key not in transitions:
                skipped.append(key)
                continue
            spans = []
            if kept_ranges_only:
                for span in episode.get("kept_frame_ranges", []):
                    spans.append((int(span["start_frame"]), int(span["end_frame"])))
                if not spans:
                    skipped.append((dataset_dir.name, episode_index))
                    continue
            else:
                spans.append((int(episode["start_frame"]), int(episode["end_frame"])))
            frames = [
                all_frames[i]
                for start, end in spans
                for i in range(start, end + 1)
                if i in all_frames
            ]
            sampled = _sample_frames(frames, stride)
            if sampled:
                demos.append(
                    DemoRecord(
                        dataset_name=dataset_dir.name,
                        dataset_dir=dataset_dir,
                        episode_index=episode_index,
                        frames=sampled,
                        first_place_frame=transitions[key],
                    )
                )
    if skipped:
        print(f"phase-aware: skipped {len(skipped)} episodes without a valid scan entry: {skipped}")
    if not demos:
        raise FileNotFoundError(f"No phase-aware episodes found under {root}")
    return demos


def load_gaze_point(frame_dir: Path, gaze_json_name: str):
    path = frame_dir / gaze_json_name
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        if not payload.get("hit", False):
            return None
        point = payload.get("gaze_uv_in_realsense")
        size = payload.get("realsense_size", [640, 480])
        if point is None or len(point) != 2:
            return None
        return float(point[0]) / float(size[0]), float(point[1]) / float(size[1])
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


PHASE_DIM = 2  # (pick, place); see _batch_to_jax for why not 3


# Fallbacks are tried in order, so a session that carries the manual annotation
# keeps using it and only sessions without one fall through to the SAM3 output.
# The 2026-08-25 recordings were never hand-annotated: they have
# target_ball_mask.png (SAM3's ball, completed to a circle where the grasp
# occludes it) and sam_basket_mask.png, and no hand mask at all.
MASK_FILE_ALIASES = {
    "ball_mask.png": ("ball_mask.png", "mask1.png", "target_ball_mask.png"),
    "basket_mask.png": ("basket_mask.png", "mask2.png", "sam_basket_mask.png"),
    # sam_hand_mask.png comes from a SAM3 point prompt placed by a probe that
    # predicts the hand centroid from the encoder's own features (fitted on the
    # 2026-08-14 manual masks, 0.88 px). Measured against those manual masks it
    # reaches IoU 0.746 -- far from perfect, but the target is pooled onto a
    # 14x14 grid where the hand spans ~20 cells, so edge error costs little.
    # Text prompts were tried first and are unusable on this hand: 0.41, with
    # 0% of frames above 0.7.
    "hand_mask.png": ("hand_mask.png", "sam_hand_mask.png"),
}


def resolve_mask_path(frame: Path, name: str) -> Path | None:
    candidates = MASK_FILE_ALIASES.get(name, (name,))
    for candidate in candidates:
        path = frame / candidate
        if path.exists():
            return path
    return None



class TaskDemoFrameDataset(Dataset):
    """Build single-frame samples and phase-dependent semantic token gates."""

    def __init__(
        self,
        demos: Sequence[DemoRecord],
        *,
        image_size: int,
        mask_files: Sequence[str],
        target_names: Sequence[str],
        sample_stride: int,
        augment: bool = False,
        crop_padding: int = 4,
        grounding_source: str = "mask",
        gaze_ball_dilate_frac: float = 0.4,
        gaze_xy_jitter_px: float = 0.0,
        token_grid: Tuple[int, int] = (14, 14),
        gaze_sigma_cells: float = DEFAULT_SIGMA_CELLS,
        gaze_dilate_cells: float = DEFAULT_DILATE_CELLS,
        gaze_window: int = DEFAULT_WINDOW,
        gaze_max_gap: int = DEFAULT_MAX_GAP,
        gaze_decay: float = DEFAULT_DECAY,
    ):
        self.demos = list(demos)
        self.image_size = image_size
        self.grounding_source = str(grounding_source)
        self.gaze_ball_dilate_frac = float(gaze_ball_dilate_frac)
        self.gaze_xy_jitter_px = float(gaze_xy_jitter_px)
        self.token_grid = (int(token_grid[0]), int(token_grid[1]))
        self.gaze_sigma_cells = float(gaze_sigma_cells)
        self.gaze_dilate_cells = float(gaze_dilate_cells)
        self.gaze_window = int(gaze_window)
        self.gaze_max_gap = int(gaze_max_gap)
        self.gaze_decay = float(gaze_decay)
        # One gaze index per demo. Demos are episode-scoped here, which is what
        # keeps temporal pooling from ever spanning two episodes.
        self.gaze_points = (
            [
                episode_gaze_index(
                    {_frame_id(f): f for f in (demo.pooling_frames or demo.frames)}
                )
                for demo in self.demos
            ]
            if self.grounding_source in ("gaze", "gaze_hybrid")
            else []
        )
        self.mask_files = list(mask_files)
        self.target_names = list(target_names)
        self.augment = bool(augment)
        self.crop_padding = int(crop_padding)
        self.items = [
            (demo_index, frame_index)
            for demo_index, demo in enumerate(self.demos)
            for frame_index in range(0, len(demo.frames), max(1, sample_stride))
        ]

    def __len__(self):
        return len(self.items)

    def _load_image(self, frame: Path) -> torch.Tensor:
        bgr = cv2.imread(str(frame / "color_image.jpg"), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(frame / "color_image.jpg")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Match the env exactly: franka_env resizes the raw 640x480 frame to
        # the observation size with cv2's default INTER_LINEAR. Using
        # INTER_AREA here instead would hand the encoder systematically
        # different (unaliased) pixels than it sees during RL.
        rgb = cv2.resize(rgb, (self.image_size, self.image_size))
        # Raw 0..255, NOT /255. The RL encoder is handed uint8 pixels straight
        # out of the replay buffer and ViTImageEncoder's normalize_method
        # ("unit") does the /255 itself. Scaling here as well trains the trunk
        # on a 255x smaller input range than it sees during RL, which silently
        # destroys the checkpoint (measured: grounding inside-mass 0.857 at
        # [0,1] vs 0.178 at the [0,255] the RL wrapper actually feeds).
        return torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()

    def _load_tactile(self, frame: Path) -> torch.Tensor:
        """thumb|index depth, concatenated exactly as franka_env builds it.

        The env does `concatenate([thumb_depth, index_depth], axis=1)` and hands
        the encoder raw 0..255 uint8; matching both the order and the range here
        is what keeps the conditioner seeing the same input it will see on the
        robot. A missing file becomes zeros, which is also what no-contact looks
        like, so a dropped frame degrades to "not touching" rather than to noise.
        """
        panels = []
        for name in ("thumb_depth_image.png", "index_depth_image.png"):
            image = cv2.imread(str(frame / name), cv2.IMREAD_COLOR)
            if image is None:
                image = np.zeros((128, 128, 3), dtype=np.uint8)
            elif image.shape[:2] != (128, 128):
                image = cv2.resize(image, (128, 128))
            panels.append(image)
        canvas = np.concatenate(panels, axis=1)
        return torch.from_numpy(canvas.copy()).permute(2, 0, 1).float()

    def _load_masks(self, frame: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        masks = []
        valid = []
        for name in self.mask_files:
            path = resolve_mask_path(frame, name)
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path else None
            valid.append(mask is not None)
            if mask is None:
                mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
            else:
                mask = cv2.resize(
                    mask,
                    (self.image_size, self.image_size),
                    interpolation=cv2.INTER_NEAREST,
                )
            masks.append((mask > 127).astype(np.float32))
        return (
            torch.from_numpy(np.stack(masks, axis=0)),
            torch.from_numpy(np.asarray(valid, dtype=np.float32)),
        )

    @staticmethod
    def _load_action(frame: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        path = frame / "action.txt"
        if not path.exists():
            return torch.zeros(7, dtype=torch.float32), torch.tensor(0.0)
        action = np.asarray(np.loadtxt(path), dtype=np.float32).reshape(-1)
        if action.shape != (7,) or not np.all(np.isfinite(action)):
            return torch.zeros(7, dtype=torch.float32), torch.tensor(0.0)
        return torch.from_numpy(action), torch.tensor(1.0)

    @staticmethod
    def _load_state(frame: Path) -> torch.Tensor:
        path = frame / "robot_ee_pose.txt"
        if not path.exists():
            return torch.zeros(7, dtype=torch.float32)
        state = np.asarray(np.loadtxt(path), dtype=np.float32).reshape(-1)
        if state.shape != (7,) or not np.all(np.isfinite(state)):
            return torch.zeros(7, dtype=torch.float32)
        return torch.from_numpy(state)

    @staticmethod
    def _edge_pad_crop(tensor: torch.Tensor, offset_y: int, offset_x: int, padding: int):
        """Replicate serl_launcher.vision.data_augmentations.random_crop.

        That function edge-pads by ``padding`` on every side and then slices
        the original size back out at a random offset in [0, 2*padding]. RL
        applies it to every image key with one shared rng, so the image and
        its masks move together; the same has to hold here or the grounding
        target drifts against the image.
        """
        padded = torch.nn.functional.pad(
            tensor[None], (padding,) * 4, mode="replicate"
        )[0]
        height, width = tensor.shape[-2:]
        return padded[:, offset_y : offset_y + height, offset_x : offset_x + width]

    def __getitem__(self, index: int):
        demo_index, frame_index = self.items[index]
        demo = self.demos[demo_index]
        future_index = min(frame_index + 1, len(demo.frames) - 1)
        frame = demo.frames[frame_index]
        future_frame = demo.frames[future_index]
        target, mask_valid = self._load_masks(frame)
        future_target, future_mask_valid = self._load_masks(future_frame)
        action, action_valid = self._load_action(frame)
        transition_valid = float(future_index != frame_index) * float(action_valid)
        if demo.first_place_frame is None:
            phase = "all"
        else:
            phase = "pick" if _frame_id(frame) < demo.first_place_frame else "place"

        image = self._load_image(frame)
        future_image = self._load_image(future_frame)
        if self.augment:
            # One offset for the whole sample: image, its masks, and the future
            # frame all shift together. The future frame deliberately shares the
            # offset so the inverse-dynamics term sees real motion rather than
            # crop jitter.
            offsets = torch.randint(0, 2 * self.crop_padding + 1, (2,))
            offset_y, offset_x = int(offsets[0]), int(offsets[1])
            image = self._edge_pad_crop(image, offset_y, offset_x, self.crop_padding)
            future_image = self._edge_pad_crop(
                future_image, offset_y, offset_x, self.crop_padding
            )
            target = self._edge_pad_crop(target, offset_y, offset_x, self.crop_padding)
            future_target = self._edge_pad_crop(
                future_target, offset_y, offset_x, self.crop_padding
            )

        # Built after the crop, not before: the gaze points are shifted by the
        # same offset the images just took, so the target cannot drift against
        # the pixels it is supervising.
        gaze_target = np.zeros(self.token_grid, dtype=np.float32)
        gaze_valid = 0.0
        # Which query row this frame should select, and whether gaze said so
        # clearly enough to be worth teaching. Only set on the gaze_hybrid path.
        cond_label = 0.0
        cond_valid = 0.0
        if self.grounding_source == "gaze_mask":
            # target_mask.png is whichever SAM3 mask gaze selected, written by
            # gaze_sam/build_mask_targets.py. Cropped with the same offset as
            # the image so the supervision cannot drift against the pixels.
            selected = cv2.imread(str(frame / "target_mask.png"),
                                  cv2.IMREAD_GRAYSCALE)
            if selected is not None and np.count_nonzero(selected) > 0:
                selected = cv2.resize(selected, (self.image_size, self.image_size),
                                      interpolation=cv2.INTER_NEAREST)
                tensor = torch.from_numpy((selected > 127).astype(np.float32))[None]
                if self.augment:
                    tensor = self._edge_pad_crop(tensor, offset_y, offset_x,
                                                 self.crop_padding)
                cells = torch.nn.functional.adaptive_avg_pool2d(
                    tensor, self.token_grid)[0].numpy()
                cells = (cells > 0.04).astype(np.float32)
                if cells.sum() > 0:
                    gaze_target = cells / cells.sum()
                    gaze_valid = 1.0
        elif self.grounding_source == "gaze_hybrid":
            # Gaze picks the target, but never defines its shape.
            #
            #   gaze inside the DILATED ball  -> target is the ORIGINAL ball mask
            #   gaze inside the basket        -> target is the basket mask
            #   gaze on neither               -> target is a blob at the gaze point
            #
            # The dilation only widens the containment test, so an operator
            # looking at the contact seam a few pixels off the ball still
            # selects the ball -- and still gets the ball's true silhouette as
            # the target, not a square or a disc. Measured on the 2026-08-25
            # recordings: gaze lands on the ball 15.5%, the basket 32.6%, and
            # neither 51.8% of frames, and that last group sits a median 9.1 px
            # from the ball centre, i.e. on the fingers closing around it. That
            # group is why the blob branch exists: those frames are the contact
            # region, and discarding them would throw away half the data.
            frame_id = _frame_id(frame)
            points = self.gaze_points[demo_index]
            gaze_xy = points.get(frame_id)
            if gaze_xy is not None:
                gx = int(np.clip(gaze_xy[0] * self.image_size, 0, self.image_size - 1))
                gy = int(np.clip(gaze_xy[1] * self.image_size, 0, self.image_size - 1))

                def _mask(name):
                    path = resolve_mask_path(frame, name)
                    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path else None
                    if raw is None:
                        return None
                    raw = cv2.resize(raw, (self.image_size, self.image_size),
                                     interpolation=cv2.INTER_NEAREST)
                    return (raw > 127).astype(np.float32)

                ball = _mask("ball_mask.png")
                basket = _mask("basket_mask.png")
                chosen = None
                hand_first = _mask("hand_mask.png")
                # Hand before ball. Once the ball is grown by a fraction of its
                # radius the grown region covers the fingers closing on it, so a
                # fixation on the contact seam satisfies both tests. Checking the
                # ball first labelled those frames "ball", which put near
                # identical gaze positions in two different classes and left the
                # conditioner unable to commit: it answered 0.463 ball / 0.336
                # hand / 0.201 basket on ball frames, and that 0.201 is basket
                # attention during the grasp. Hand first also matches what the
                # operator is doing -- when the fingers are on the ball, the
                # thing being watched is the contact.
                if hand_first is not None and hand_first.sum() > 0:
                    if hand_first[gy, gx] > 0.5:
                        chosen = hand_first
                if chosen is None and ball is not None and ball.sum() > 0:
                    # Grow the ball by a fraction of its OWN radius, the way a
                    # 5 mm ball becomes 7 mm. An absolute pixel count would mean
                    # something different at every resolution and every distance:
                    # the first version used 40 px, a figure borrowed from masks
                    # stored at 640x480, and applied it at 128x128 -- a disc a
                    # third of the image wide, which swallowed 67% of all frames
                    # into the "looking at the ball" branch.
                    ball_radius = float(np.sqrt(ball.sum() / np.pi))
                    radius = max(1, int(round(ball_radius
                                              * self.gaze_ball_dilate_frac)))
                    kernel = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
                    if cv2.dilate(ball, kernel)[gy, gx] > 0.5:
                        chosen = ball            # original silhouette, not the dilated one
                hand = hand_first
                if chosen is None and basket is not None and basket.sum() > 0:
                    if basket[gy, gx] > 0.5:
                        chosen = basket
                # The hand gets its own branch rather than being swept into the
                # blob. Without it the ball's query row has to serve two
                # incompatible jobs -- a tight silhouette on ball frames and a
                # diffuse patch on contact frames -- and it showed: the ball
                # branch scored worst of the three (0.683 against 0.976 for the
                # basket) and the two query rows drifted together, cosine +0.68.
                # Neither the basket nor the hand is dilated; only the ball is,
                # because only the ball is small enough for a few pixels of gaze
                # error to miss it entirely.
                if chosen is None and hand is not None and hand.sum() > 0:
                    if hand[gy, gx] > 0.5:
                        chosen = hand

                if chosen is not None:
                    # The branch gaze took is exactly the question the query
                    # should be asking, so it can supervise the conditioner
                    # directly. Frames where gaze picked neither object are left
                    # unlabelled: their target is a blob near the fingers, and
                    # calling that "ball" or "basket" would be inventing a fact.
                    cond_label = (0.0 if chosen is ball
                                  else 1.0 if chosen is basket else 2.0)
                    cond_valid = 1.0
                    tensor = torch.from_numpy(chosen)[None]
                    if self.augment:
                        tensor = self._edge_pad_crop(tensor, offset_y, offset_x,
                                                     self.crop_padding)
                    cells = torch.nn.functional.adaptive_avg_pool2d(
                        tensor, self.token_grid)[0].numpy()
                    cells = (cells > 0.04).astype(np.float32)
                    if cells.sum() > 0:
                        gaze_target = cells / cells.sum()
                        gaze_valid = 1.0
                else:
                    # Nothing contains the fixation, so give it to the nearest
                    # object. Once the hand has a mask of its own this is a small
                    # and well-behaved set: 12.7% of frames, sitting a median
                    # 3.6 px from the nearest mask against 13.3 px from the
                    # second, and only 0.3% further than 15 px from anything.
                    # They are fixations on the seam between the fingers and the
                    # ball, not looks at empty table.
                    #
                    # A Gaussian blob at the gaze point used to cover this case.
                    # It was dropped because it is a much broader target than a
                    # silhouette, and mixing two target widths in one KL teaches
                    # the query to hedge between sharp and diffuse depending on
                    # which branch a frame happened to take.
                    best, best_d = None, None
                    for candidate in (hand_first, ball, basket):
                        if candidate is None or candidate.sum() <= 0:
                            continue
                        ys, xs = np.nonzero(candidate)
                        d = float(np.min(np.hypot(xs - gx, ys - gy)))
                        if best_d is None or d < best_d:
                            best, best_d = candidate, d
                    if best is not None:
                        cond_label = (0.0 if best is ball
                                      else 1.0 if best is basket else 2.0)
                        cond_valid = 1.0
                        tensor = torch.from_numpy(best)[None]
                        if self.augment:
                            tensor = self._edge_pad_crop(tensor, offset_y, offset_x,
                                                         self.crop_padding)
                        cells = torch.nn.functional.adaptive_avg_pool2d(
                            tensor, self.token_grid)[0].numpy()
                        cells = (cells > 0.04).astype(np.float32)
                        if cells.sum() > 0:
                            gaze_target = cells / cells.sum()
                            gaze_valid = 1.0
        elif self.grounding_source == "gaze":
            points = self.gaze_points[demo_index]
            frame_id = _frame_id(frame)
            if frame_id in points:
                pooled, pooled_weights = pool_gaze_points(
                    frame_id, points, self.gaze_window,
                    self.gaze_max_gap, self.gaze_decay,
                )
                if self.augment:
                    pooled = shift_points_for_crop(
                        pooled, offset_y, offset_x, self.crop_padding,
                        self.image_size, self.image_size,
                    )
                gaze_target = build_target_heatmap(
                    pooled, pooled_weights, self.token_grid,
                    self.gaze_sigma_cells, self.gaze_dilate_cells,
                )
                gaze_valid = 1.0

        # The conditioner reads the gaze position, not the target: it has to
        # decide *which question to ask*, and at RL time the same two numbers
        # arrive from the gaze predictor. -1 marks "no gaze this frame" so the
        # network can learn a distinct response instead of reading (0, 0) as a
        # real fixation in the top-left corner.
        gaze_xy_out = np.full((2,), -1.0, dtype=np.float32)
        if self.grounding_source in ("gaze", "gaze_hybrid"):
            _pt = self.gaze_points[demo_index].get(_frame_id(frame))
            if _pt is not None:
                _gx, _gy = float(_pt[0]), float(_pt[1])
                if self.augment:
                    # shift_points_for_crop works in normalized [0, 1]
                    # coordinates, not pixels: it multiplies by (width - 1),
                    # applies the crop offset, divides back and clips. Handing
                    # it pixels sent every value through that clip and collapsed
                    # the whole batch to ~0.03, which is why the conditioner sat
                    # at exactly ln 2 -- its input carried no information at all.
                    _shift = shift_points_for_crop(
                        np.asarray([[_gx, _gy]], dtype=np.float32),
                        offset_y, offset_x, self.crop_padding,
                        self.image_size, self.image_size,
                    )
                    _gx = float(_shift[0, 0])
                    _gy = float(_shift[0, 1])
                if self.augment and self.gaze_xy_jitter_px > 0:
                    # At RL time the conditioner is fed the predictor's estimate,
                    # not the eye tracker. Measured on held-out episodes, that
                    # estimate sits a median 8.7 px from the real fixation, so
                    # train the conditioner on inputs of the same accuracy rather
                    # than on a clean signal it will never see again.
                    _sd = self.gaze_xy_jitter_px / float(self.image_size)
                    _gx = float(np.clip(_gx + np.random.normal(0.0, _sd), 0.0, 1.0))
                    _gy = float(np.clip(_gy + np.random.normal(0.0, _sd), 0.0, 1.0))
                gaze_xy_out = np.asarray([_gx, _gy], dtype=np.float32)

        return {
            "gaze_xy": torch.from_numpy(gaze_xy_out),
            "cond_label": torch.tensor(float(cond_label), dtype=torch.float32),
            "cond_valid": torch.tensor(float(cond_valid), dtype=torch.float32),
            "gaze_target": torch.from_numpy(gaze_target),
            "gaze_valid": torch.tensor(gaze_valid, dtype=torch.float32),
            "tactile": self._load_tactile(frame),
            "image": image,
            "future_image": future_image,
            "target": target,
            "future_target": future_target,
            "mask_valid": mask_valid,
            "future_mask_valid": future_mask_valid,
            "action": action,
            "state": self._load_state(frame),
            "transition_valid": torch.tensor(transition_valid, dtype=torch.float32),
            # The grounding target follows the phase, mirroring RL: the CGL
            # loss there reads `front_camera_mask`, which the env wrapper fills
            # with mask1 (ball) during pick and mask2 (basket) during place.
            "pick_phase": torch.tensor(
                1.0 if phase in ("pick", "all") else 0.0, dtype=torch.float32
            ),
            "place_phase": torch.tensor(
                1.0 if phase == "place" else 0.0, dtype=torch.float32
            ),
            "phase": phase,
            "frame": str(frame),
        }


def split_demos(demos, val_count: int, test_count: int, seed: int):
    # Keep a non-empty training split for small pilot datasets. With five
    # demos, for example, this becomes 3 train / 1 validation / 1 test.
    n = len(demos)
    if n <= 1:
        val_count = test_count = 0
    elif n < 5:
        test_count = min(test_count, 1)
        val_count = min(val_count, max(0, n - test_count - 1))
    else:
        test_count = min(test_count, max(1, n // 5))
        val_count = min(val_count, max(1, n // 5))
    while n - val_count - test_count < 1 and (val_count or test_count):
        if val_count >= test_count and val_count:
            val_count -= 1
        elif test_count:
            test_count -= 1
    indices = list(range(len(demos)))
    random.Random(seed).shuffle(indices)
    test_indices = indices[:test_count]
    val_indices = indices[test_count : test_count + val_count]
    train_indices = indices[test_count + val_count :]
    if not train_indices:
        raise ValueError("Not enough demos for the requested validation/test split.")
    return [demos[i] for i in train_indices], [demos[i] for i in val_indices], [demos[i] for i in test_indices]


# ---------------------------------------------------------------- model


class ViTPretrainModel(nn.Module):
    """The RL ViT plus pretrain-only heads.

    Only the ``encoder_front_camera`` subtree is written into the RL agent;
    every head below is discarded after pretraining.
    """

    num_targets: int
    image_size: Tuple[int, int] = (224, 224)
    patch_size: Tuple[int, int] = (16, 16)
    hidden_dim: int = 192
    num_layers: int = 4
    num_heads: int = 6
    output_dim: int = 256
    num_spatial_blocks: int = 8
    action_dim: int = 7
    state_dim: int = 7
    # 0 = the original single constant grounding query (the model must infer
    # the phase from pixels). 2 = one query per phase, selected by the caller's
    # one-hot, so the phase is an input rather than something to be guessed.
    grounding_phase_dim: int = 0
    # Tactile drives the grounding query instead of a caller-supplied phase.
    # Nothing supervises what it should mean; the CGL loss is its only gradient.
    grounding_tactile_conditioned: bool = False
    # Gaze drives the grounding query. Same slot as the tactile conditioner, but
    # sourced from the signal the target itself is built from.
    grounding_gaze_conditioned: bool = False

    def setup(self):
        self.encoder = ViTImageEncoder(
            image_size=self.image_size,
            patch_size=self.patch_size,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            mlp_dim=self.hidden_dim * 2,
            bottleneck_dim=self.output_dim,
            pooling_method="spatial_learned_embeddings",
            num_spatial_blocks=self.num_spatial_blocks,
            use_grounding_query=True,
            grounding_phase_dim=self.grounding_phase_dim,
            grounding_tactile_conditioned=self.grounding_tactile_conditioned,
            grounding_gaze_conditioned=self.grounding_gaze_conditioned,
            normalize_method="unit",
            name=VIT_MODULE_NAME,
        )
        self.segmentation_head = nn.Conv(
            self.num_targets, (1, 1), name="segmentation_head"
        )
        self.geometry_head = nn.Dense(self.num_targets * 3, name="geometry_head")
        self.presence_head = nn.Dense(self.num_targets, name="presence_head")
        self.inverse_dense = nn.Dense(256, name="inverse_dense")
        self.inverse_action = nn.Dense(self.action_dim, name="inverse_action")

    def encode(
        self,
        image: jax.Array,
        train: bool,
        phase: Optional[jax.Array] = None,
        tactile: Optional[jax.Array] = None,
        gaze_xy: Optional[jax.Array] = None,
    ) -> Dict[str, jax.Array]:
        vector, spatial, grounding_logits, query_phase = self.encoder(
            image, train=train, encode=True, return_grounding=True,
            return_query_phase=True, phase=phase,
            tactile=tactile, gaze_xy=gaze_xy,
        )
        segmentation_logits = self.segmentation_head(spatial)
        # (B, h, w, C) -> (B, C, h, w) for mask_supervision_loss
        segmentation_logits = jnp.moveaxis(segmentation_logits, -1, 1)

        geometry_raw = self.geometry_head(vector).reshape(
            vector.shape[0], self.num_targets, 3
        )
        geometry = jnp.concatenate(
            (
                nn.tanh(geometry_raw[..., :2]),
                nn.sigmoid(geometry_raw[..., 2:3]),
            ),
            axis=-1,
        )
        return {
            "vector": vector,
            "spatial": spatial,
            "grounding_logits": grounding_logits,
            "segmentation_logits": segmentation_logits,
            "geometry_predictions": geometry,
            "presence_logits": self.presence_head(vector),
            "query_phase": query_phase,
        }

    def __call__(
        self,
        image: jax.Array,
        *,
        future_image: Optional[jax.Array] = None,
        state: Optional[jax.Array] = None,
        phase: Optional[jax.Array] = None,
        tactile: Optional[jax.Array] = None,
        gaze_xy: Optional[jax.Array] = None,
        train: bool = True,
    ) -> Dict[str, jax.Array]:
        output = self.encode(image, train, phase, tactile, gaze_xy)
        if future_image is not None:
            # The future frame is one step away, so it is still in the same
            # phase; reusing the phase here keeps the inverse-dynamics pair
            # consistent instead of asking two different queries.
            future = self.encode(future_image, train, phase, tactile, gaze_xy)
            if state is None:
                state = jnp.zeros(
                    (*output["vector"].shape[:-1], self.state_dim),
                    dtype=output["vector"].dtype,
                )
            inverse_input = jnp.concatenate(
                (output["vector"], future["vector"], state), axis=-1
            )
            hidden = nn.gelu(self.inverse_dense(inverse_input), approximate=False)
            output["inverse_action"] = self.inverse_action(hidden)
            output["future_vector"] = future["vector"]
        return output


# ---------------------------------------------------------------- losses


def _batch_to_jax(batch):
    def arr(key, dtype=np.float32):
        return jnp.asarray(np.asarray(batch[key], dtype=dtype))

    image = np.asarray(batch["image"], dtype=np.float32).transpose(0, 2, 3, 1)
    future_image = np.asarray(batch["future_image"], dtype=np.float32).transpose(
        0, 2, 3, 1
    )
    return {
        "image": jnp.asarray(image),
        "future_image": jnp.asarray(future_image),
        "target": arr("target"),
        "mask_valid": arr("mask_valid"),
        "action": arr("action"),
        "state": arr("state"),
        "transition_valid": arr("transition_valid"),
        "pick_phase": arr("pick_phase"),
        "place_phase": arr("place_phase"),
        "gaze_xy": arr("gaze_xy"),
        "cond_label": arr("cond_label"),
        "cond_valid": arr("cond_valid"),
        "gaze_target": arr("gaze_target"),
        "gaze_valid": arr("gaze_valid"),
        "tactile": jnp.asarray(
            np.asarray(batch["tactile"], dtype=np.float32).transpose(0, 2, 3, 1)
        ),
        # (B, 2) one-hot handed to the grounding query. Two dims, not three:
        # the RL state's third slot ("no target selected") never fires -- all
        # 8015 demo transitions carry [1,0,0] or [0,1,0] -- so the query table
        # would just hold a dead row. The RL wiring will slice state[..., -3:-1].
        "phase_onehot": jnp.stack(
            [arr("pick_phase"), arr("place_phase")], axis=-1
        ),
    }


def _mask_geometry(target_masks: jax.Array):
    """Return normalized xy center, area, and non-empty flags."""
    height, width = target_masks.shape[-2:]
    mass = jnp.sum(target_masks, axis=(-2, -1))
    present = (mass > 0.5).astype(jnp.float32)
    x_coords = jnp.linspace(-1.0, 1.0, width, dtype=jnp.float32)
    y_coords = jnp.linspace(-1.0, 1.0, height, dtype=jnp.float32)
    denominator = jnp.maximum(mass, 1e-6)
    center_x = jnp.sum(target_masks * x_coords[None, None, None, :], axis=(-2, -1)) / denominator
    center_y = jnp.sum(target_masks * y_coords[None, None, :, None], axis=(-2, -1)) / denominator
    area = jnp.mean(target_masks, axis=(-2, -1))
    return jnp.stack((center_x, center_y, area), axis=-1), present


def _weighted_mean(value: jax.Array, weight: jax.Array) -> jax.Array:
    return jnp.sum(value * weight) / jnp.maximum(jnp.sum(weight), 1.0)


def _grounding_kl_loss(
    grounding_logits: jax.Array,
    masks: jax.Array,
    weights: jax.Array,
):
    """KL of a per-object weighted attention target, in the RL-time CGL form.

    ``grounding_logits`` is (B, h, w). ``masks`` is (B, K, H, W) in [0, 1] and
    ``weights`` is (B, K) giving each object's share of the target probability
    mass; a zero weight drops that object for that sample.

    The weighting is per *object*, not per pixel, and that distinction matters.
    A single uniform distribution over the union of the masks would hand each
    object a share proportional to its area, and the hand is far bigger than
    the ball -- measured on one frame, 1479 px vs 103 px, i.e. 34 vs 4 cells on
    the 14x14 grid. The ball would receive under 11% of the target mass and be
    drowned by the hand. Here each mask is first normalized to a uniform
    distribution over its own cells, and only then mixed by ``weights``.
    """
    batch, grid_h, grid_w = grounding_logits.shape
    num_objects = masks.shape[1]

    cells = jax.image.resize(
        masks, (batch, num_objects, grid_h, grid_w), method="linear"
    )
    cells = (cells > 0.04).astype(jnp.float32)
    cells_flat = cells.reshape(batch, num_objects, -1)

    mass = jnp.sum(cells_flat, axis=-1, keepdims=True)
    present = (mass[..., 0] > 0).astype(jnp.float32)
    per_object = cells_flat / jnp.maximum(mass, 1e-8)

    # An object that is absent from the frame cannot take any mass; whatever it
    # would have held is redistributed over the objects that are present.
    effective = weights * present
    total = jnp.sum(effective, axis=-1, keepdims=True)
    valid = (total[:, 0] > 0).astype(jnp.float32)
    effective = effective / jnp.maximum(total, 1e-8)

    target = jnp.sum(per_object * effective[..., None], axis=1)

    logits_flat = grounding_logits.reshape(batch, -1)
    attention = jax.nn.softmax(logits_flat, axis=-1)
    kl = jnp.sum(
        target * (jnp.log(target + 1e-8) - jnp.log(attention + 1e-8)), axis=-1
    )

    union = (jnp.max(cells_flat * (weights > 0)[..., None], axis=1) > 0).astype(
        jnp.float32
    )
    inside = jnp.sum(attention * union, axis=-1)
    # Per-object attention mass, so the hand cannot mask a collapse on the ball.
    per_object_inside = jnp.sum(
        attention[:, None, :] * cells_flat, axis=-1
    )  # (B, K)
    return (
        jnp.where(valid > 0, kl, 0.0),
        jnp.where(valid > 0, inside, 0.0),
        valid,
        per_object_inside * present,
        present,
    )


def _grounding_kl_from_target(grounding_logits, target, valid):
    """KL against a target distribution supplied directly, not built from masks.

    Same form as the mask path and the same form RL uses at CGL time; the only
    difference is where the target came from.
    """
    batch = grounding_logits.shape[0]
    target_flat = target.reshape(batch, -1)
    target_flat = target_flat / jnp.maximum(
        jnp.sum(target_flat, axis=-1, keepdims=True), 1e-8
    )
    attention = jax.nn.softmax(grounding_logits.reshape(batch, -1), axis=-1)
    kl = jnp.sum(
        target_flat * (jnp.log(target_flat + 1e-8) - jnp.log(attention + 1e-8)),
        axis=-1,
    )
    return jnp.where(valid > 0, kl, 0.0)


def _per_object_attention(grounding_logits, masks):
    """Attention mass landing on each annotated object. Metrics only.

    The masks are hand-annotated, so nothing here may touch the gradient -- a
    gaze-supervised encoder that consumed them would be indistinguishable from
    just using the masks directly. They are read solely to answer "is the query
    actually looking at the ball", which the gaze target alone cannot report.
    """
    batch, grid_h, grid_w = grounding_logits.shape
    num_objects = masks.shape[1]
    cells = jax.image.resize(
        masks, (batch, num_objects, grid_h, grid_w), method="linear"
    )
    cells = (cells > 0.04).astype(jnp.float32).reshape(batch, num_objects, -1)
    attention = jax.nn.softmax(grounding_logits.reshape(batch, -1), axis=-1)
    inside = jnp.sum(attention[:, None, :] * cells, axis=-1)
    present = (jnp.sum(cells, axis=-1) > 0).astype(jnp.float32)
    return inside, present


def _compute_losses_gaze(output, batch, args):
    """Gaze-grounded pretraining: CGL against the gaze target, plus inverse dynamics.

    The mask-supervised heads (segmentation / geometry / presence) are off in
    this mode, so the trunk is shaped by exactly two things: the grounding KL,
    whose gradient reaches the patch tokens through the attention logits, and
    the inverse-dynamics term, which is what keeps the representation carrying
    control-relevant information rather than collapsing to a gaze detector.
    """
    valid = batch["gaze_valid"]
    grounding_per_sample = _grounding_kl_from_target(
        output["grounding_logits"], batch["gaze_target"], valid
    )
    grounding_loss = _weighted_mean(grounding_per_sample, valid)

    inside, present = _per_object_attention(
        output["grounding_logits"], batch["target"]
    )
    mask_valid = batch["mask_valid"]

    inverse_loss = _weighted_mean(
        jnp.mean(jnp.square(output["inverse_action"] - batch["action"]), axis=-1),
        batch["transition_valid"],
    )
    # The conditioner picks which query row the frame uses. Leaving that to the
    # grounding KL alone did not work: measured on the first gazehybrid encoder,
    # it returned (0.515, 0.485) for gaze on the ball and (0.471, 0.529) for gaze
    # on the basket -- a 0.044 spread, i.e. a permanent half-and-half blend. The
    # KL has an easier way out, because 67% of frames are supervised on or beside
    # the ball, so "always look at the ball" fits most of the data with the
    # conditioner left flat. This term hands it the answer directly, using the
    # branch gaze already took. It adds no information: that label IS where the
    # target came from.
    cond_loss = jnp.zeros((), dtype=jnp.float32)
    query_phase = output.get("query_phase")
    cond_weight = float(getattr(args, "cond_loss_weight", 0.0))
    if query_phase is not None and cond_weight > 0:
        probs = jnp.clip(query_phase.reshape(query_phase.shape[0], -1), 1e-6, 1.0)
        label = batch["cond_label"]
        n_rows = probs.shape[-1]
        onehot = jax.nn.one_hot(label.astype(jnp.int32), n_rows)
        picked = jnp.sum(probs * onehot, axis=-1)
        # Balance the two classes by their frequency in this batch. Without it
        # the conditioner just learns the marginal: measured on the first run,
        # 79.4% of labelled frames said "ball", and a constant "ball" output
        # scores a cross-entropy of ~0.6 -- almost exactly the 0.543 that run
        # plateaued at, with the gaze position doing no work at all.
        valid_c = batch["cond_valid"]
        counts = jnp.sum(onehot * valid_c[:, None], axis=0)
        per_class = (1.0 / n_rows) / jnp.maximum(counts, 1.0)
        cls_w = jnp.sum(onehot * per_class[None, :], axis=-1)
        cond_loss = jnp.sum(-jnp.log(picked) * valid_c * cls_w) / jnp.maximum(
            jnp.sum(valid_c * cls_w), 1e-6
        )

    total = (
        args.grounding_loss_weight * grounding_loss
        + args.inverse_loss_weight * inverse_loss
        + cond_weight * cond_loss
    )
    zero = jnp.zeros((), dtype=jnp.float32)
    # Channel order follows task_configs' target_names, and a task may declare
    # fewer than three. Indexing past the end must not silently report another
    # channel's value: JAX clamps out-of-range indices instead of raising, so
    # tennis_ball_pick (ball only) reported identical numbers for ball, hand
    # and basket, all three of them the ball.
    num_targets = inside.shape[1]

    def channel(index):
        if index >= num_targets:
            return zero
        return _weighted_mean(
            inside[:, index], valid * present[:, index] * mask_valid[:, index]
        )

    grounded = inside[:, 0] + (inside[:, 1] if num_targets > 1 else 0.0)
    return {
        "loss": total,
        "grounding_loss": grounding_loss,
        "grounding_inside": _weighted_mean(
            grounded, valid * present[:, 0] * mask_valid[:, 0]
        ),
        "inside_object": channel(0),
        "inside_hand": channel(1),
        "inside_basket": channel(2),
        "inside_pick": zero,
        "inside_place": zero,
        "segmentation_loss": zero,
        "segmentation_bce": zero,
        "segmentation_dice": zero,
        "geometry_loss": zero,
        "center_loss": zero,
        "presence_loss": zero,
        "inverse_loss": inverse_loss,
        "cond_loss": cond_loss,
    }


def compute_losses(output, batch, args):
    if getattr(args, "grounding_source", "mask") in ("gaze", "gaze_mask", "gaze_hybrid"):
        return _compute_losses_gaze(output, batch, args)

    segmentation = mask_supervision_loss(
        output["segmentation_logits"],
        batch["target"],
        dice_weight=args.dice_weight,
        channel_weights=batch["mask_valid"],
    )

    # The grounding target follows the phase, matching RL exactly: there the
    # CGL loss reads `front_camera_mask`, which the env wrapper fills with
    # mask1 (ball) while picking and mask2 (basket) while placing. Verified on
    # the demo buffer: front_camera_mask == mask1 on 262/262 pick frames and
    # == mask2 on 338/338 place frames.
    #
    # The hand is deliberately NOT in this target. The online mask predictor
    # only emits two slots (MASK_SLOTS = ball, basket), so no hand mask exists
    # at RL time; grounding the query on the hand here would just be undone by
    # the RL-time CGL loss, which would score hand attention as outside-mass.
    # The hand reaches the trunk through the segmentation and geometry heads
    # instead -- eval_encoder reports its center error to confirm that.
    pick = batch["pick_phase"]
    place = batch["place_phase"]
    # Target objects follow the phase, and the hand is grounded in both phases:
    #   pick  -> ball   + hand
    #   place -> basket + hand
    # The hand carries a smaller share than the phase target so it supports the
    # relation without dominating it. Channel order is (ball, hand, basket).
    object_weight = 1.0 - args.hand_grounding_weight
    hand_weight = args.hand_grounding_weight
    supervised = jnp.clip(pick + place, 0.0, 1.0)
    weights = jnp.stack(
        [
            object_weight * pick * batch["mask_valid"][:, 0],
            hand_weight * supervised * batch["mask_valid"][:, 1],
            object_weight * place * batch["mask_valid"][:, 2],
        ],
        axis=-1,
    )
    (
        grounding_per_sample,
        inside_per_sample,
        grounding_valid,
        per_object_inside,
        per_object_present,
    ) = _grounding_kl_loss(output["grounding_logits"], batch["target"], weights)

    grounding_weight = grounding_valid * supervised
    grounding_loss = _weighted_mean(grounding_per_sample, grounding_weight)
    grounding_inside = _weighted_mean(inside_per_sample, grounding_weight)
    # The phase target on its own -- directly comparable with the RL-time CGL
    # metric, which has no hand mask available and only scores this object.
    object_inside_per_sample = (
        per_object_inside[:, 0] * pick + per_object_inside[:, 2] * place
    )
    inside_object = _weighted_mean(object_inside_per_sample, grounding_weight)
    inside_hand = _weighted_mean(
        per_object_inside[:, 1], grounding_weight * per_object_present[:, 1]
    )
    inside_pick = _weighted_mean(object_inside_per_sample, grounding_weight * pick)
    inside_place = _weighted_mean(object_inside_per_sample, grounding_weight * place)

    geometry_target, present = _mask_geometry(batch["target"])
    geometry = output["geometry_predictions"]
    center_error = jnp.mean(
        jnp.square(geometry[..., :2] - geometry_target[..., :2]), axis=-1
    )
    center_loss = _weighted_mean(center_error, present * batch["mask_valid"])
    area_error = jnp.square(geometry[..., 2] - geometry_target[..., 2])
    area_loss = _weighted_mean(area_error, batch["mask_valid"])

    predicted_delta = geometry[:, :, None, :2] - geometry[:, None, :, :2]
    target_delta = (
        geometry_target[:, :, None, :2] - geometry_target[:, None, :, :2]
    )
    pair_valid = present[:, :, None] * present[:, None, :]
    pair_valid = pair_valid * (
        1.0 - jnp.eye(present.shape[-1], dtype=jnp.float32)[None]
    )
    relation_loss = _weighted_mean(
        jnp.mean(jnp.square(predicted_delta - target_delta), axis=-1), pair_valid
    )
    geometry_loss = center_loss + relation_loss + 0.25 * area_loss

    presence_loss = _weighted_mean(
        optax.sigmoid_binary_cross_entropy(output["presence_logits"], present),
        batch["mask_valid"],
    )

    inverse_loss = _weighted_mean(
        jnp.mean(jnp.square(output["inverse_action"] - batch["action"]), axis=-1),
        batch["transition_valid"],
    )

    total = (
        args.mask_loss_weight * segmentation["loss"]
        + args.grounding_loss_weight * grounding_loss
        + args.geometry_loss_weight * geometry_loss
        + args.presence_loss_weight * presence_loss
        + args.inverse_loss_weight * inverse_loss
    )
    return {
        "loss": total,
        "segmentation_loss": segmentation["loss"],
        "segmentation_bce": segmentation["bce"],
        "segmentation_dice": segmentation["dice"],
        "grounding_loss": grounding_loss,
        "grounding_inside": grounding_inside,
        "inside_object": inside_object,
        "inside_hand": inside_hand,
        "inside_pick": inside_pick,
        "inside_place": inside_place,
        "geometry_loss": geometry_loss,
        "center_loss": center_loss,
        "presence_loss": presence_loss,
        "inverse_loss": inverse_loss,
    }


# ---------------------------------------------------------------- train loop


def make_steps(model, optimizer, args):
    def forward(params, batch, train):
        return model.apply(
            {"params": params},
            batch["image"],
            future_image=batch["future_image"],
            state=batch["state"],
            phase=batch["phase_onehot"] if args.phase_conditioned else None,
            tactile=batch["tactile"] if args.grounding_tactile_conditioned else None,
            gaze_xy=batch["gaze_xy"] if args.grounding_gaze_conditioned else None,
            train=train,
            rngs={"dropout": jax.random.PRNGKey(0)},
        )

    @jax.jit
    def train_step(params, opt_state, batch):
        def loss_fn(current_params):
            output = forward(current_params, batch, True)
            losses = compute_losses(output, batch, args)
            return losses["loss"], losses

        (_, losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, losses

    @jax.jit
    def eval_step(params, batch):
        return compute_losses(forward(params, batch, False), batch, args)

    return train_step, eval_step


def run_epoch(loader, params, opt_state, train_step, eval_step, *, train):
    totals: Dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = _batch_to_jax(batch)
        if train:
            params, opt_state, losses = train_step(params, opt_state, batch)
        else:
            losses = eval_step(params, batch)
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        count += 1
    averages = {key: value / max(1, count) for key, value in totals.items()}
    return averages, params, opt_state


def save_checkpoint(path: Path, params, epoch: int):
    """Write the ViT subtree that create_pixels loads, plus the full tree."""
    payload = {
        "format_version": 1,
        "epoch": epoch,
        # load_encoder_checkpoint reads "params"; it must be the ViT subtree so
        # replace_named_param_subtree can drop it straight in.
        "params": jax.device_get(params[VIT_MODULE_NAME]),
        "full_params": jax.device_get(params),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialization.msgpack_serialize(payload))


# ---------------------------------------------------------- imagenet init


def load_imagenet_vit_params(npz_path: Path, model_params) -> dict:
    """Map a google/vision_transformer augreg .npz onto ViTImageEncoder.

    Only the Transformer trunk is transferred; the readout, grounding query,
    and bottleneck stay randomly initialized. Raises if the architecture does
    not match, rather than silently transferring nothing.
    """
    data = np.load(str(npz_path))
    encoder = model_params[VIT_MODULE_NAME]
    transferred = {}

    def take(name):
        if name not in data:
            raise KeyError(f"{npz_path} has no array named {name!r}")
        return jnp.asarray(data[name])

    patch = {"kernel": take("embedding/kernel"), "bias": take("embedding/bias")}
    if patch["kernel"].shape != encoder["patch_embedding"]["kernel"].shape:
        raise ValueError(
            "ImageNet patch embedding shape "
            f"{patch['kernel'].shape} != model {encoder['patch_embedding']['kernel'].shape}. "
            "Match --vit_hidden_dim / --patch_size to the checkpoint "
            "(ViT-S/16 is hidden_dim=384)."
        )
    transferred["patch_embedding"] = patch
    transferred["cls"] = take("cls")

    # Interpolate the position grid to this model's token count.
    source = take("Transformer/posembed_input/pos_embedding")
    target_tokens = encoder["pos_embedding"].shape[1]
    if source.shape[1] != target_tokens:
        source_grid = int(round((source.shape[1] - 1) ** 0.5))
        target_grid = int(round((target_tokens - 1) ** 0.5))
        cls_pos, grid_pos = source[:, :1], source[:, 1:]
        grid_pos = grid_pos.reshape(1, source_grid, source_grid, -1)
        grid_pos = jax.image.resize(
            grid_pos, (1, target_grid, target_grid, grid_pos.shape[-1]), method="bicubic"
        ).reshape(1, target_grid * target_grid, -1)
        source = jnp.concatenate((cls_pos, grid_pos), axis=1)
        print(f"  interpolated pos_embedding {source_grid}^2 -> {target_grid}^2")
    transferred["pos_embedding"] = source

    num_layers = len([k for k in encoder if k.startswith("encoder_block_")])
    for index in range(num_layers):
        prefix = f"Transformer/encoderblock_{index}"
        transferred[f"encoder_block_{index}"] = {
            "ln_0": {
                "scale": take(f"{prefix}/LayerNorm_0/scale"),
                "bias": take(f"{prefix}/LayerNorm_0/bias"),
            },
            "ln_1": {
                "scale": take(f"{prefix}/LayerNorm_2/scale"),
                "bias": take(f"{prefix}/LayerNorm_2/bias"),
            },
            "self_attention": {
                part: {
                    "kernel": take(
                        f"{prefix}/MultiHeadDotProductAttention_1/{part}/kernel"
                    ),
                    "bias": take(
                        f"{prefix}/MultiHeadDotProductAttention_1/{part}/bias"
                    ),
                }
                for part in ("query", "key", "value", "out")
            },
            "mlp": {
                "Dense_0": {
                    "kernel": take(f"{prefix}/MlpBlock_3/Dense_0/kernel"),
                    "bias": take(f"{prefix}/MlpBlock_3/Dense_0/bias"),
                },
                "Dense_1": {
                    "kernel": take(f"{prefix}/MlpBlock_3/Dense_1/kernel"),
                    "bias": take(f"{prefix}/MlpBlock_3/Dense_1/bias"),
                },
            },
        }
    transferred["encoder_norm"] = {
        "scale": take("Transformer/encoder_norm/scale"),
        "bias": take("Transformer/encoder_norm/bias"),
    }

    def check(node, reference, path=""):
        for key, value in node.items():
            if key not in reference:
                raise KeyError(f"model has no parameter at {path}/{key}")
            if isinstance(value, dict):
                check(value, reference[key], f"{path}/{key}")
            elif value.shape != reference[key].shape:
                raise ValueError(
                    f"shape mismatch at {path}/{key}: "
                    f"imagenet {value.shape} != model {reference[key].shape}"
                )

    check(transferred, encoder)
    merged = dict(encoder)
    merged.update(transferred)
    return merged


# ---------------------------------------------------------------- cli


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp_name", required=True, help=f"One of: {', '.join(list_task_names())}")
    parser.add_argument("--data_root", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    # Observation size the env produces; the dataset loads at this size so
    # pretraining and RL see the identical pixel pipeline.
    parser.add_argument("--input_size", type=int, default=128)
    # Size the ViT upsamples to internally before patchification.
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument(
        "--session_filter", default="",
        help="Substring a recording session's directory name must contain to be "
             "used, e.g. 2026-08-25. Empty uses every session under the data "
             "root. Sessions are separately eye-tracker-calibrated, so training "
             "on one batch and evaluating on another is a domain shift, not a "
             "held-out split.")
    parser.add_argument(
        "--grounding_source",
        choices=("mask", "gaze", "gaze_mask", "gaze_hybrid"), default="mask",
        help="What the grounding query is scored against. 'mask' is the "
             "existing vit-grounded recipe and is left untouched. 'gaze' scores "
             "against the operator's recorded gaze instead, and switches off "
             "every mask-supervised head -- the masks were hand-annotated with "
             "SAM3, so a pipeline that consumed them would invite the question "
             "of why the gaze is needed at all. In gaze mode the masks are read "
             "for metrics only and never reach the gradient. 'gaze_mask' scores "
             "against the text-prompted SAM3 mask that gaze selected for that "
             "frame: gaze decides *which* object, the mask supplies the extent "
             "gaze cannot. A gaze point sits a median 0.79 token cells off the "
             "ball because the operator watches the contact region, which is "
             "accurate enough to pick an object and far too coarse to outline "
             "one.")
    parser.add_argument("--gaze_sigma_cells", type=float, default=DEFAULT_SIGMA_CELLS)
    parser.add_argument(
        "--gaze_dilate_cells", type=float, default=DEFAULT_DILATE_CELLS,
        help="Grow the gaze region by this many token cells. Needed because "
             "during the approach the operator looks at the gripper, leaving "
             "the ball a median 0.77 cells outside the raw blob. Measured over "
             "all 41 episodes at 0 / 0.5 / 1.0 / 1.5 / 2.0 cells, the ball "
             "falls inside the target's strongest cells on 81 / 85 / 88 / 90 / "
             "92% of frames while pre-grasp mass on the basket rises 0.02 -> "
             "0.09 against a 0.19 chance baseline. 1.0 is the knee.")
    parser.add_argument(
        "--grounding_gaze_conditioned", type=int, default=0,
        help="Condition the grounding query on the gaze position (two numbers "
             "through a small MLP to a softmax over the query rows). Pairs with "
             "--grounding_source=gaze_hybrid: the question asked and the answer "
             "taught then come from the same signal, which is what the tactile "
             "conditioner could not guarantee.",
    )
    parser.add_argument(
        "--grounding_query_rows", type=int, default=2,
        help="How many questions the grounding query table holds when the query "
             "is gaze-conditioned. 3 gives the hand its own row instead of "
             "making the ball's row cover both the ball and the contact region.",
    )
    parser.add_argument(
        "--cond_loss_weight", type=float, default=0.0,
        help="Cross-entropy on the grounding query conditioner, using the "
             "branch gaze took as the label. Only frames where gaze clearly "
             "selected the ball or the basket are scored; the 51.8%% that fall "
             "on neither are left unlabelled rather than guessed.",
    )
    parser.add_argument(
        "--gaze_xy_jitter_px", type=float, default=0.0,
        help="Gaussian jitter, in 128px units, added to the gaze position fed to "
             "the conditioner during training. Set it to the gaze predictor's "
             "own error so the conditioner is trained on the accuracy it will "
             "be deployed with.",
    )
    parser.add_argument(
        "--gaze_ball_dilate_frac", type=float, default=0.4,
        help="Grow the ball by this fraction of its own radius for the "
             "containment test only -- 0.4 turns a 5 mm ball into 7 mm. The "
             "target always keeps the ball's original silhouette; this only "
             "decides whether a fixation just off the ball still counts as "
             "looking at it. Relative rather than absolute so it means the same "
             "thing at any resolution and any distance.",
    )
    parser.add_argument(
        "--grounding_tactile_conditioned", type=int, default=0,
        help="Condition the grounding query on tactile. Nothing tells the model "
             "what contact means -- the tactile frame is simply an input, and the "
             "CGL loss is the only gradient the conditioner receives. Contact is "
             "the one cue that separates two frames a single camera cannot: a "
             "hand holding the ball over the basket and a hand passing above it "
             "look the same, and the tactile sensors do not.")
    parser.add_argument("--gaze_window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--gaze_max_gap", type=int, default=DEFAULT_MAX_GAP)
    parser.add_argument("--gaze_decay", type=float, default=DEFAULT_DECAY)
    parser.add_argument("--vit_hidden_dim", type=int, default=192)
    parser.add_argument("--vit_num_layers", type=int, default=4)
    parser.add_argument("--vit_num_heads", type=int, default=6)
    parser.add_argument("--output_dim", type=int, default=256)
    parser.add_argument("--num_spatial_blocks", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dice_weight", type=float, default=1.0)
    parser.add_argument("--mask_loss_weight", type=float, default=1.0)
    parser.add_argument("--grounding_loss_weight", type=float, default=1.0)
    parser.add_argument("--geometry_loss_weight", type=float, default=1.0)
    parser.add_argument("--presence_loss_weight", type=float, default=0.1)
    # No temporal-invariance term: it rewarded representations for not
    # changing between adjacent frames, which fights the RL critic.
    parser.add_argument("--inverse_loss_weight", type=float, default=0.5)
    # stride 1 uses all 10921 legitimately-collected frames instead of 1/5.
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument(
        "--augment",
        type=lambda v: str(v).lower() not in ("0", "false", "no"),
        default=True,
        help="Apply the same random crop RL applies to every image key. "
             "Without it the trunk never sees a shifted input and, because a "
             "ViT's absolute pos_embedding is not translation equivariant, it "
             "degrades under RL's augmentation (measured: grounding "
             "inside-mass 0.986 clean vs 0.823 augmented).",
    )
    parser.add_argument(
        "--crop_padding",
        type=int,
        default=4,
        help="Must match the padding RL uses in make_batch_augmentation_func.",
    )
    parser.add_argument(
        "--phase_conditioned",
        type=lambda v: str(v).lower() not in ("0", "false", "no"),
        default=False,
        help="Select the grounding query by task phase instead of using one "
             "constant query. Without it the ViT has to infer the phase from "
             "pixels, and a 2D camera makes that unreliable exactly when it "
             "matters: a hand occluding the ball, or crossing in front of the "
             "grasp point, reads as a completed grasp and slides the attention "
             "onto the basket while nothing has been picked up. With it the "
             "phase is an input, so the map cannot move to the basket until "
             "the classifier's one-hot flips. Writes a checkpoint that only "
             "loads into an encoder built with grounding_phase_dim=2.",
    )
    parser.add_argument(
        "--hand_grounding_weight",
        type=float,
        default=0.2,
        help="Share of the attention target given to the hand; the phase's "
             "object gets the rest. Per-object, not per-pixel -- the hand is "
             "~8x the ball's area, so an unweighted union would bury the ball.",
    )
    parser.add_argument("--phase_scan", type=Path, default=DEFAULT_PHASE_SCAN)
    parser.add_argument("--imagenet_init", type=Path, default=None,
                        help="Path to a google/vision_transformer augreg .npz.")
    parser.add_argument("--val_demos", type=int, default=4)
    parser.add_argument("--test_demos", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    global _SESSION_FILTER
    _SESSION_FILTER = args.session_filter
    task = get_task_config(args.exp_name)
    data_root = args.data_root or task.data_root
    # Phase-conditioned runs land in their own directory so the unconditioned
    # checkpoint stays available for the comparison.
    if args.grounding_source in ("gaze", "gaze_mask", "gaze_hybrid") and args.phase_conditioned:
        # The gaze already carries the phase: the operator looks inside the
        # basket on 1.5% of pre-grasp frames against 32.9% after the grasp, and
        # 37 of 41 episodes contain no pre-grasp basket look at all. Feeding a
        # phase one-hot as well would mean the RL side has to produce one, and
        # the only sources for it are a mask or a pick classifier.
        print("grounding_source=gaze: forcing --phase_conditioned off")
        args.phase_conditioned = False
    default_name = f"{args.exp_name}_vit_grounded"
    if args.grounding_source in ("gaze", "gaze_mask", "gaze_hybrid"):
        default_name += "_" + args.grounding_source
    if args.phase_conditioned:
        default_name += "_phase"
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / default_name)
    mask_files = list(task.mask_files)
    target_names = list(task.target_names)

    if args.grounding_source in ("gaze", "gaze_mask", "gaze_hybrid"):
        # Episode-scoped demos, and deliberately not find_phase_demos: that one
        # reads pick_classifier_phase_scan.json, i.e. the pick classifier this
        # mode exists to remove. Episode scoping also stops the temporal pooling
        # from ever spanning two episodes.
        demos = find_episode_demos(
            data_root, args.frame_stride, kept_ranges_only=True
        )
    elif args.phase_scan and Path(args.phase_scan).is_file():
        demos = find_phase_demos(data_root, args.frame_stride, Path(args.phase_scan))
    else:
        demos = find_episode_demos(data_root, args.frame_stride)
    if not demos:
        raise ValueError(f"No demos found under {data_root}")
    train_demos, val_demos, test_demos = split_demos(
        demos, args.val_demos, args.test_demos, args.seed
    )
    print(
        f"demos: {len(demos)} total -> {len(train_demos)} train / "
        f"{len(val_demos)} val / {len(test_demos)} test"
    )

    token_grid = (args.image_size // args.patch_size, args.image_size // args.patch_size)
    dataset_kwargs = dict(
        image_size=args.input_size,
        mask_files=mask_files,
        target_names=target_names,
        sample_stride=args.frame_stride,
        grounding_source=args.grounding_source,
        gaze_ball_dilate_frac=args.gaze_ball_dilate_frac,
        gaze_xy_jitter_px=args.gaze_xy_jitter_px,
        token_grid=token_grid,
        gaze_sigma_cells=args.gaze_sigma_cells,
        gaze_dilate_cells=args.gaze_dilate_cells,
        gaze_window=args.gaze_window,
        gaze_max_gap=args.gaze_max_gap,
        gaze_decay=args.gaze_decay,
    )
    if args.grounding_source in ("gaze", "gaze_mask", "gaze_hybrid"):
        print(f"grounding_source={args.grounding_source}: token grid {token_grid}, "
              f"sigma {args.gaze_sigma_cells} cells, "
              f"dilate {args.gaze_dilate_cells} cells, window +/-{args.gaze_window}")
    # Augment the training split only. Validation and test stay clean so that
    # model selection and eval_encoder keep measuring the same thing across
    # runs; check_rl_grounding is what reports the augmented number, because
    # that is the setting RL actually trains in.
    train_dataset_kwargs = dict(
        dataset_kwargs, augment=args.augment, crop_padding=args.crop_padding
    )
    loader_kwargs = (
        {"multiprocessing_context": "spawn", "persistent_workers": True}
        if args.num_workers > 0
        else {}
    )
    train_loader = DataLoader(
        TaskDemoFrameDataset(train_demos, **train_dataset_kwargs),
        args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed),
        **loader_kwargs,
    )
    val_loader = (
        DataLoader(
            TaskDemoFrameDataset(val_demos, **dataset_kwargs),
            args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            **loader_kwargs,
        )
        if val_demos
        else None
    )
    print(f"train samples: {len(train_loader.dataset)}")

    model = ViTPretrainModel(
        num_targets=len(target_names),
        image_size=(args.image_size, args.image_size),
        patch_size=(args.patch_size, args.patch_size),
        hidden_dim=args.vit_hidden_dim,
        num_layers=args.vit_num_layers,
        num_heads=args.vit_num_heads,
        output_dim=args.output_dim,
        num_spatial_blocks=args.num_spatial_blocks,
        grounding_phase_dim=(
            args.grounding_query_rows if args.grounding_gaze_conditioned
            else PHASE_DIM if (args.phase_conditioned
                               or args.grounding_tactile_conditioned)
            else 0
        ),
        grounding_tactile_conditioned=bool(args.grounding_tactile_conditioned),
        grounding_gaze_conditioned=bool(args.grounding_gaze_conditioned),
    )
    dummy_image = jnp.zeros((1, args.input_size, args.input_size, 3), jnp.float32)
    params = model.init(
        jax.random.PRNGKey(args.seed),
        dummy_image,
        future_image=dummy_image,
        state=jnp.zeros((1, 7), jnp.float32),
        phase=(
            jnp.zeros((1, PHASE_DIM), jnp.float32)
            if args.phase_conditioned
            else None
        ),
        tactile=(
            jnp.zeros((1, 128, 256, 3), jnp.float32)
            if args.grounding_tactile_conditioned
            else None
        ),
        gaze_xy=(
            jnp.zeros((1, 2), jnp.float32)
            if args.grounding_gaze_conditioned
            else None
        ),
        train=False,
    )["params"]

    if args.imagenet_init:
        print(f"loading ImageNet ViT weights from {args.imagenet_init}")
        params = dict(params)
        params[VIT_MODULE_NAME] = load_imagenet_vit_params(args.imagenet_init, params)
        print("  ImageNet trunk transferred")

    optimizer = optax.adamw(
        learning_rate=args.learning_rate, weight_decay=args.weight_decay
    )
    opt_state = optimizer.init(params)
    train_step, eval_step = make_steps(model, optimizer, args)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                # Resolved values, not the raw (possibly None) CLI defaults --
                # downstream scripts read these back.
                "data_root": str(data_root),
                "output_dir": str(output_dir),
                "target_names": target_names,
                "mask_files": mask_files,
                "checkpoint_format": "flax_msgpack_v1",
                "encoder_version": "vit-grounded",
                "architecture": (
                    "vit_grounded_phase_v1"
                    if args.phase_conditioned
                    else "vit_grounded_v1"
                ),
                # Readers must build the encoder with a matching
                # grounding_phase_dim; the query param's leading axis differs.
                "grounding_phase_dim": (
                    args.grounding_query_rows
                    if args.grounding_gaze_conditioned
                    else PHASE_DIM
                    if (args.phase_conditioned or args.grounding_tactile_conditioned)
                    else 0
                ),
                "grounding_tactile_conditioned": bool(
                    args.grounding_tactile_conditioned),
                "grounding_gaze_conditioned": bool(args.grounding_gaze_conditioned),
                "gaze_ball_dilate_frac": float(args.gaze_ball_dilate_frac),
                "gaze_xy_jitter_px": float(args.gaze_xy_jitter_px),
                "cond_loss_weight": float(args.cond_loss_weight),
                "grounding_query_rows": int(args.grounding_query_rows),
                # Consumed by eval_encoder / check_rl_grounding to pick the
                # matching target. "ball_pick_only" was the previous scheme.
                "grounding_target": "phase_selected_mask_plus_hand",
                "vit_module_name": VIT_MODULE_NAME,
            },
            indent=2,
        )
    )

    # The RL encoder normalizes internally from raw 0..255 pixels. If this
    # ever prints a [0,1] range again, the checkpoint will load fine and then
    # be useless at RL time, so fail loudly instead.
    probe = next(iter(train_loader))["image"].numpy()
    print(f"[input check] image range [{probe.min():.1f}, {probe.max():.1f}] "
          f"(must be 0..255 to match the RL encoder)", flush=True)
    if probe.max() <= 1.5:
        raise RuntimeError(
            f"Dataset images are in [{probe.min():.3f}, {probe.max():.3f}]; the RL "
            "encoder is fed raw 0..255 pixels and normalizes internally. Training "
            "on a [0,1] range makes the checkpoint useless for RL."
        )

    best = float("inf")
    for epoch in range(args.epochs):
        train_losses, params, opt_state = run_epoch(
            train_loader, params, opt_state, train_step, eval_step, train=True
        )
        message = " ".join(f"{k}={v:.4f}" for k, v in sorted(train_losses.items()))
        print(f"[epoch {epoch}] train {message}", flush=True)
        if val_loader is not None:
            val_losses, _, _ = run_epoch(
                val_loader, params, opt_state, train_step, eval_step, train=False
            )
            message = " ".join(f"{k}={v:.4f}" for k, v in sorted(val_losses.items()))
            print(f"[epoch {epoch}] val   {message}", flush=True)
            score = val_losses["loss"]
        else:
            score = train_losses["loss"]
        save_checkpoint(output_dir / "latest.msgpack", params, epoch)
        if score < best:
            best = score
            save_checkpoint(output_dir / "best.msgpack", params, epoch)
            print(f"  new best {best:.4f} -> {output_dir / 'best.msgpack'}")
    save_checkpoint(output_dir / "final.msgpack", params, args.epochs - 1)
    print(f"done. checkpoints in {output_dir}")


if __name__ == "__main__":
    main()
