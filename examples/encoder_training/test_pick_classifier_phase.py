#!/usr/bin/env python3
"""Scan recorded pick-and-place episodes with the frozen pick classifier.

The scan is temporal: once the classifier first reaches the threshold inside an
episode, that frame is reported as the pick -> place transition and all later
frames are considered place frames.  The classifier is reset for every episode.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import jax
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from serl_launcher.networks.reward_classifier import load_classifier_func


DEFAULT_DATA_ROOT = (
    Path("/home/ealin/workspaces/DexTacHil/data/recorded_data")
    / "tennis_ball_pick_and_place"
)
DEFAULT_DATASETS = (
    DEFAULT_DATA_ROOT / "tennis_ball_pick_and_place-2026-08-14_12-18-59",
    DEFAULT_DATA_ROOT / "tennis_ball_pick_and_place-2026-08-14_12-49-48",
)
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "examples" / "reward_classifier" / "classifier_ckpt_ball_pick"
)
DEFAULT_OUTPUT = SCRIPT_DIR / "runs" / "pick_classifier_phase_scan.json"


@dataclass
class EpisodeResult:
    dataset: str
    episode_index: int
    start_frame: int
    end_frame: int
    num_frames: int
    threshold: float
    first_positive_frame: int | None
    first_positive_offset: int | None
    first_positive_probability: float | None
    max_probability: float
    max_probability_frame: int
    num_positive_frames: int
    excluded: bool = False
    exclusion_reason: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test pick classifier phase transitions on recorded episodes."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        default=None,
        help=(
            "Dataset directory. Repeat this option for multiple datasets. "
            "Defaults to the two 2026-08-14 recording folders."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="Pick classifier checkpoint directory.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Probability threshold for the first pick-success detection.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Number of frames passed to the classifier per JAX call.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="JSON file for the detailed scan results.",
    )
    parser.add_argument(
        "--include_invalid_last_episode",
        action="store_true",
        help="Include episode 31 from the 12-18-59 dataset instead of excluding it.",
    )
    return parser.parse_args()


def frame_id_from_dir(frame_dir: Path) -> int | None:
    if not frame_dir.name.startswith("frame_"):
        return None
    try:
        return int(frame_dir.name.split("_", 1)[1])
    except ValueError:
        return None


def discover_frames(frame_root: Path) -> dict[int, Path]:
    frames: dict[int, Path] = {}
    for frame_dir in frame_root.iterdir():
        if not frame_dir.is_dir():
            continue
        frame_id = frame_id_from_dir(frame_dir)
        if frame_id is None or not (frame_dir / "color_image.jpg").exists():
            continue
        frames[frame_id] = frame_dir
    return frames


def read_rgb(frame_dir: Path) -> np.ndarray:
    image = cv2.imread(str(frame_dir / "color_image.jpg"), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(frame_dir / "color_image.jpg")
    image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_LINEAR)
    return image[..., ::-1].astype(np.uint8, copy=False)


def read_tactile(frame_dir: Path) -> np.ndarray:
    def read_depth(path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(path)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[-1] == 4:
            image = image[..., :3]
        return cv2.resize(image.astype(np.uint8, copy=False), (128, 128))

    thumb = read_depth(frame_dir / "thumb_depth_image.png")
    index = read_depth(frame_dir / "index_depth_image.png")
    return np.concatenate((thumb, index), axis=1).astype(np.uint8, copy=False)


def read_observation(frame_dir: Path) -> dict[str, np.ndarray]:
    return {
        "front_camera": read_rgb(frame_dir),
        "tactile_data": read_tactile(frame_dir),
    }


def batch_observations(observations: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    # The classifier was trained with enable_stacking=True. Recorded frames are
    # single images, so add the singleton history dimension expected by it.
    return {
        key: np.stack([observation[key] for observation in observations], axis=0)[:, None, ...]
        for key in ("front_camera", "tactile_data")
    }


def predict_probabilities(
    classifier_func: Any,
    frame_dirs: list[Path],
    batch_size: int,
) -> np.ndarray:
    probabilities: list[np.ndarray] = []
    for start in range(0, len(frame_dirs), batch_size):
        batch_dirs = frame_dirs[start : start + batch_size]
        valid_count = len(batch_dirs)
        if valid_count < batch_size:
            # Keep the JAX input shape fixed. Otherwise every episode's final
            # short batch can trigger another expensive XLA compilation.
            batch_dirs = batch_dirs + [batch_dirs[-1]] * (batch_size - valid_count)
        observations = [read_observation(frame_dir) for frame_dir in batch_dirs]
        logits = np.asarray(classifier_func(batch_observations(observations)))
        probabilities.append(
            np.asarray(jax.nn.sigmoid(logits)).reshape(-1)[:valid_count]
        )
    return np.concatenate(probabilities, axis=0) if probabilities else np.empty(0, dtype=np.float32)


def load_metadata(frame_root: Path) -> dict[str, Any]:
    path = frame_root / "recording_metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing recording metadata: {path}")
    return json.loads(path.read_text())


def episode_result(
    dataset_name: str,
    episode: dict[str, Any],
    frame_dirs_by_id: dict[int, Path],
    probabilities: np.ndarray,
    threshold: float,
) -> EpisodeResult:
    start_frame = int(episode["start_frame"])
    end_frame = int(episode["end_frame"])
    frame_ids = sorted(
        frame_id
        for frame_id in frame_dirs_by_id
        if start_frame <= frame_id <= end_frame
    )
    positive_indices = np.flatnonzero(probabilities >= threshold)
    first_index = int(positive_indices[0]) if len(positive_indices) else None
    max_index = int(np.argmax(probabilities))
    return EpisodeResult(
        dataset=dataset_name,
        episode_index=int(episode["episode_index"]),
        start_frame=start_frame,
        end_frame=end_frame,
        num_frames=len(frame_ids),
        threshold=threshold,
        first_positive_frame=frame_ids[first_index] if first_index is not None else None,
        first_positive_offset=first_index,
        first_positive_probability=(
            float(probabilities[first_index]) if first_index is not None else None
        ),
        max_probability=float(probabilities[max_index]),
        max_probability_frame=frame_ids[max_index],
        num_positive_frames=int(np.sum(probabilities >= threshold)),
    )


def main() -> None:
    args = parse_args()
    if args.threshold < 0.0 or args.threshold > 1.0:
        raise ValueError("--threshold must be in [0, 1]")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")

    dataset_paths = [
        Path(path).expanduser().resolve()
        for path in (args.datasets if args.datasets else DEFAULT_DATASETS)
    ]
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Classifier checkpoint does not exist: {checkpoint_path}")

    print(f"[classifier] checkpoint={checkpoint_path}")
    print(f"[classifier] threshold={args.threshold:.3f} batch_size={args.batch_size}")

    # Build the classifier from one recorded frame, then reuse the frozen
    # function for every episode.
    first_frames = discover_frames(dataset_paths[0])
    if not first_frames:
        raise RuntimeError(f"No recorded frames found under {dataset_paths[0]}")
    first_frame_dir = first_frames[min(first_frames)]
    classifier = load_classifier_func(
        key=jax.random.PRNGKey(0),
        sample=batch_observations([read_observation(first_frame_dir)]),
        image_keys=["front_camera", "tactile_data"],
        checkpoint_path=str(checkpoint_path),
    )

    results: list[EpisodeResult] = []
    excluded_count = 0
    for dataset_path in dataset_paths:
        metadata = load_metadata(dataset_path)
        frames_by_id = discover_frames(dataset_path)
        episodes = metadata.get("episode_ranges", [])
        print(
            f"[dataset] {dataset_path.name} episodes={len(episodes)} "
            f"frames={len(frames_by_id)}"
        )
        for episode in episodes:
            episode_index = int(episode["episode_index"])
            is_invalid_last = (
                dataset_path.name.endswith("2026-08-14_12-18-59")
                and episode_index == 31
            )
            if is_invalid_last and not args.include_invalid_last_episode:
                excluded_count += 1
                print(
                    f"[exclude] dataset={dataset_path.name} episode={episode_index} "
                    "marked invalid/interrupted"
                )
                continue

            frame_ids = sorted(
                frame_id
                for frame_id in frames_by_id
                if int(episode["start_frame"]) <= frame_id <= int(episode["end_frame"])
            )
            frame_dirs = [frames_by_id[frame_id] for frame_id in frame_ids]
            probabilities = predict_probabilities(classifier, frame_dirs, args.batch_size)
            result = episode_result(
                dataset_path.name,
                episode,
                frames_by_id,
                probabilities,
                args.threshold,
            )
            results.append(result)
            if result.first_positive_frame is None:
                print(
                    f"[result] dataset={result.dataset} episode={result.episode_index} "
                    "first_positive=NONE "
                    f"max_prob={result.max_probability:.4f}@{result.max_probability_frame}"
                )
            else:
                print(
                    f"[result] dataset={result.dataset} episode={result.episode_index} "
                    f"first_positive=1 frame={result.first_positive_frame} "
                    f"offset={result.first_positive_offset} "
                    f"prob={result.first_positive_probability:.4f} "
                    f"max={result.max_probability:.4f}@{result.max_probability_frame}"
                )

    expected_count = 40
    print(
        f"[summary] valid_episodes={len(results)} excluded={excluded_count} "
        f"first_positive={sum(r.first_positive_frame is not None for r in results)} "
        f"missing_positive={sum(r.first_positive_frame is None for r in results)}"
    )
    if len(results) != expected_count:
        print(
            f"[summary] NOTE: expected_count={expected_count}, but metadata contains "
            f"{len(results)} valid episodes after the requested exclusion."
        )

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "threshold": args.threshold,
                "datasets": [str(path) for path in dataset_paths],
                "excluded_count": excluded_count,
                "results": [asdict(result) for result in results],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"[output] {output_path}")


if __name__ == "__main__":
    main()
