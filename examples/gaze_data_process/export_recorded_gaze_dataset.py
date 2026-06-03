#!/usr/bin/env python3
from __future__ import annotations

import json
import pickle as pkl
import random
from pathlib import Path

import cv2
import numpy as np
from absl import app, flags

from gaze_mapping.gaze_to_realsense_homography import EpisodeHomographyMap, load_homography_source


FLAGS = flags.FLAGS

# Edit these defaults instead of passing a long command line every time.
DEFAULT_METADATA = (
    "/media/user/data3/wrq/recorded_data/tennis_ball_pick/"
    "tennis_ball_pick-5-27-0/recording_metadata.json"
)
DEFAULT_OUT_DIR = str(Path(__file__).resolve().parent / "gaze_cls_data")
DEFAULT_HOMOGRAPHY = str(Path(__file__).resolve().parent / "gaze_mapping" / "gaze_episode_homographies.json")
DEFAULT_USE_HOMOGRAPHY = True
DEFAULT_REQUIRE_HOMOGRAPHY = True

flags.DEFINE_string(
    "metadata",
    DEFAULT_METADATA,
    "Path to recording_metadata.json.",
)
flags.DEFINE_string("out_dir", DEFAULT_OUT_DIR, "Output directory for pkl shards.")
flags.DEFINE_integer("shard_size", 5000, "Number of samples per pkl shard.")
flags.DEFINE_string("image_key", "front_camera", "Observation image key.")
flags.DEFINE_integer("resize_w", 128, "Output image width.")
flags.DEFINE_integer("resize_h", 128, "Output image height.")
flags.DEFINE_float("val_ratio", 0.2, "Validation split ratio.")
flags.DEFINE_integer("seed", 42, "Train/val split seed.")
flags.DEFINE_boolean("success_only", True, "Export only successful episodes.")
flags.DEFINE_string(
    "homography",
    DEFAULT_HOMOGRAPHY if DEFAULT_USE_HOMOGRAPHY else "",
    "Optional JSON mapping eye-tracker gaze pixels to RealSense image pixels. Defaults to the per-episode mapping file.",
)
flags.DEFINE_boolean(
    "require_homography",
    DEFAULT_REQUIRE_HOMOGRAPHY,
    "Skip samples if no required homography is available.",
)
flags.DEFINE_float(
    "max_homography_fit_error_px",
    100.0,
    "Skip episodes whose fitted homography mean error exceeds this threshold.",
)


def _metadata_frames(metadata):
    records = metadata.get("episode_ranges", [])
    if FLAGS.success_only:
        records = [rec for rec in records if bool(rec.get("success", False))]

    for rec in sorted(records, key=lambda r: int(r.get("episode_index", 0))):
        episode_index = int(rec.get("episode_index", 0))
        ranges = rec.get("kept_frame_ranges") or [
            {"start_frame": rec["start_frame"], "end_frame": rec["end_frame"]}
        ]
        for rng in ranges:
            start = int(rng["start_frame"])
            end = int(rng["end_frame"])
            if end < start:
                start, end = end, start
            for frame_id in range(start, end + 1):
                yield episode_index, frame_id


def _load_color_bgr(frame_dir: Path):
    path = frame_dir / "color_image.jpg"
    if not path.exists():
        return None
    return cv2.imread(str(path))


def _read_eye_gaze(frame_dir: Path):
    contact_path = frame_dir / "gaze_contact.json"
    if not contact_path.exists():
        return None

    try:
        contact = json.loads(contact_path.read_text())
    except Exception:
        return None

    gaze_uv = contact.get("gaze_uv_in_eye", None)
    if isinstance(gaze_uv, (list, tuple)) and len(gaze_uv) >= 2:
        x, y = gaze_uv[0], gaze_uv[1]
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return np.array([float(x), float(y)], np.float32)

    return None


def _normalize_xy(xy_pix, src_w, src_h):
    x = np.clip(float(xy_pix[0]) / max(1.0, src_w - 1), 0.0, 1.0)
    y = np.clip(float(xy_pix[1]) / max(1.0, src_h - 1), 0.0, 1.0)
    return np.array([x, y], np.float32)


def _map_eye_to_realsense_xy(xy_pix, homography, episode_index: int, dst_w: int, dst_h: int):
    if homography is None:
        return xy_pix

    if isinstance(homography, EpisodeHomographyMap):
        x, y = homography.transform_point(int(episode_index), xy_pix)
        hom = homography.get(int(episode_index))
        if hom is None or x is None or y is None:
            return xy_pix
        if hom.dst_size is not None:
            h_w, h_h = hom.dst_size
            x *= max(1.0, dst_w - 1) / max(1.0, h_w - 1)
            y *= max(1.0, dst_h - 1) / max(1.0, h_h - 1)
    else:
        x, y = homography.transform_point(xy_pix)
        if homography.dst_size is not None:
            h_w, h_h = homography.dst_size
            x *= max(1.0, dst_w - 1) / max(1.0, h_w - 1)
            y *= max(1.0, dst_h - 1) / max(1.0, h_h - 1)

    x = np.clip(float(x), 0.0, max(0.0, dst_w - 1))
    y = np.clip(float(y), 0.0, max(0.0, dst_h - 1))
    return np.array([x, y], np.float32)


def _has_episode_homography(homography, episode_index: int):
    if homography is None:
        return False
    if isinstance(homography, EpisodeHomographyMap):
        return homography.get(int(episode_index)) is not None
    return True


def _episode_homography_is_valid(homography, episode_index: int):
    if homography is None:
        return False
    ep_h = homography.get(int(episode_index)) if isinstance(homography, EpisodeHomographyMap) else homography
    if ep_h is None:
        return False
    if FLAGS.max_homography_fit_error_px <= 0:
        return True
    if ep_h.fit_mean_error_px is None:
        return True
    return float(ep_h.fit_mean_error_px) <= float(FLAGS.max_homography_fit_error_px)

def _split_train_val(samples, val_ratio, seed):
    rng = random.Random(seed)
    idxs = list(range(len(samples)))
    rng.shuffle(idxs)
    split = int(len(samples) * (1.0 - val_ratio))
    return [samples[i] for i in idxs[:split]], [samples[i] for i in idxs[split:]]


def _save_shards(samples, out_dir: Path, shard_size: int, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_shard in out_dir.glob(f"{prefix}_*.pkl"):
        old_shard.unlink()
    for shard_id, start in enumerate(range(0, len(samples), shard_size)):
        chunk = samples[start : start + shard_size]
        path = out_dir / f"{prefix}_{shard_id:05d}.pkl"
        with open(path, "wb") as f:
            pkl.dump(chunk, f)
        print(f"[dump] {path} ({len(chunk)} samples)")
    print(f"[done] {out_dir} total={len(samples)}")


def main(_):
    metadata_path = Path(FLAGS.metadata).expanduser().resolve()
    metadata = json.loads(metadata_path.read_text())
    frame_root = Path(metadata.get("frame_root", metadata_path.parent)).expanduser().resolve()
    out_dir = Path(FLAGS.out_dir).expanduser().resolve()
    expected_frames = list(_metadata_frames(metadata))
    if FLAGS.homography:
        homography = load_homography_source(FLAGS.homography)
    else:
        homography = None

    samples = []
    skipped_missing_image = 0
    skipped_missing_homography = 0
    skipped_missing_gaze = 0
    for episode_index, frame_id in expected_frames:
        if FLAGS.require_homography and not _has_episode_homography(homography, episode_index):
            skipped_missing_homography += 1
            continue
        if homography is not None and not _episode_homography_is_valid(homography, episode_index):
            skipped_missing_homography += 1
            continue

        frame_dir = frame_root / f"frame_{frame_id}"
        image_bgr = _load_color_bgr(frame_dir)
        if image_bgr is None:
            skipped_missing_image += 1
            continue

        src_h, src_w = image_bgr.shape[:2]
        xy_pix = _read_eye_gaze(frame_dir)
        if xy_pix is None:
            skipped_missing_gaze += 1
            continue
        xy_pix = _map_eye_to_realsense_xy(xy_pix, homography, episode_index, src_w, src_h)
        xy_norm = _normalize_xy(xy_pix, src_w, src_h)

        image_resized = cv2.resize(
            image_bgr,
            (FLAGS.resize_w, FLAGS.resize_h),
            interpolation=cv2.INTER_LINEAR,
        )
        image_rgb = image_resized[..., ::-1].copy()

        samples.append(
            {
                "observations": {FLAGS.image_key: image_rgb},
                "gaze_xy": xy_norm.astype(np.float32),
                "episode_index": np.int32(episode_index),
                "frame_id": np.int32(frame_id),
            }
        )

    if not samples:
        print("[warn] no samples exported.")
        return

    train_samples, val_samples = _split_train_val(
        samples,
        val_ratio=FLAGS.val_ratio,
        seed=FLAGS.seed,
    )
    print(f"[source] metadata={metadata_path}")
    print(f"[source] frame_root={frame_root}")
    if homography is not None:
        print(f"[source] homography={Path(FLAGS.homography).expanduser().resolve()}")
        print(f"[source] homography_scope={'per_episode' if isinstance(homography, EpisodeHomographyMap) else 'global'}")
        print(f"[source] max_homography_fit_error_px={FLAGS.max_homography_fit_error_px}")
    print(
        f"[export] expected_frames={len(expected_frames)} total={len(samples)} "
        f"skipped_missing_image={skipped_missing_image} "
        f"skipped_missing_homography={skipped_missing_homography} "
        f"skipped_missing_gaze={skipped_missing_gaze}"
    )
    print(f"[split] train={len(train_samples)}")
    print(f"[split] val  ={len(val_samples)}")

    _save_shards(train_samples, out_dir / "train", FLAGS.shard_size, "gaze_samples")
    _save_shards(val_samples, out_dir / "val", FLAGS.shard_size, "gaze_samples")


if __name__ == "__main__":
    app.run(main)
