#!/usr/bin/env python3
from __future__ import annotations

import json
import pickle as pkl
import random
from pathlib import Path

import cv2
import numpy as np
from absl import app, flags


FLAGS = flags.FLAGS

# ============================== 编辑这里 ==============================
_RECORDED = "/home/ealin/workspaces/DexTacHil/data/recorded_data/tennis_ball_pick_and_place"
DEFAULT_TRAIN_ROOTS = [
    f"{_RECORDED}/tennis_ball_pick_and_place-2026-08-14_12-18-59",
    f"{_RECORDED}/tennis_ball_pick_and_place-2026-08-14_12-49-48",
]
# =====================================================================

DEFAULT_METADATA = f"{DEFAULT_TRAIN_ROOTS[0]}/recording_metadata.json"
DEFAULT_OUT_DIR = str(Path(__file__).resolve().parent / "gaze_cls_data")

flags.DEFINE_string("metadata", DEFAULT_METADATA, "Path to recording_metadata.json.")
flags.DEFINE_multi_string(
    "train_root",
    DEFAULT_TRAIN_ROOTS,
    "Recording root to export into train split. Can be repeated.",
)
flags.DEFINE_multi_string(
    "val_root",
    [],
    "Recording root to export into val split. Can be repeated. If empty, train roots are randomly split by --val_ratio.",
)
flags.DEFINE_multi_string(
    "test_root",
    [],
    "Optional recording root to export into test split. Can be repeated.",
)
flags.DEFINE_string("out_dir", DEFAULT_OUT_DIR, "Output directory for pkl shards.")
flags.DEFINE_integer("shard_size", 5000, "Number of samples per pkl shard.")
flags.DEFINE_string("image_key", "front_camera", "Observation image key.")
flags.DEFINE_integer("resize_w", 128, "Output image width.")
flags.DEFINE_integer("resize_h", 128, "Output image height.")
flags.DEFINE_float("val_ratio", 0.2, "Validation split ratio.")
flags.DEFINE_integer("seed", 42, "Train/val split seed.")
flags.DEFINE_enum(
    "split_by",
    "episode",
    ["episode", "frame"],
    "How to divide train and val. 'episode' holds whole episodes out; 'frame' "
    "shuffles individual frames, which leaks: consecutive frames are ~0.1 s "
    "apart and nearly identical, so a frame-level split puts near-duplicates "
    "on both sides and the validation error stops measuring generalisation.",
)
flags.DEFINE_boolean("success_only", True, "Export only successful episodes.")
flags.DEFINE_boolean(
    "use_metadata_fallback",
    False,
    "Use --metadata as a single input when no split roots are desired.",
)


def _metadata_frames(metadata):
    records = metadata.get("episode_ranges", [])
    if FLAGS.success_only:
        records = [rec for rec in records if bool(rec.get("success", False))]

    for rec in sorted(records, key=lambda r: int(r.get("episode_index", 0))):
        episode_index = int(rec.get("episode_index", 0))
        ranges = rec.get("kept_frame_ranges") or [{"start_frame": rec["start_frame"], "end_frame": rec["end_frame"]}]
        for rng in ranges:
            start = int(rng["start_frame"])
            end = int(rng["end_frame"])
            if end < start:
                start, end = end, start
            for frame_id in range(start, end + 1):
                yield episode_index, frame_id


def _metadata_path_from_root(root: str | Path) -> Path:
    path = Path(root).expanduser().resolve()
    if path.name == "recording_metadata.json":
        return path
    return path / "recording_metadata.json"


def _load_color_bgr(frame_dir: Path):
    path = frame_dir / "color_image.jpg"
    if not path.exists():
        return None
    return cv2.imread(str(path))


def _read_realsense_gaze(frame_dir: Path):
    contact_path = frame_dir / "gaze_contact.json"
    if not contact_path.exists():
        return None, None

    try:
        contact = json.loads(contact_path.read_text())
    except Exception:
        return None, None

    gaze_uv = contact.get("gaze_uv_in_realsense", None)
    if not isinstance(gaze_uv, (list, tuple)) or len(gaze_uv) < 2:
        return None, None

    x, y = gaze_uv[0], gaze_uv[1]
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return None, None

    return np.array([float(x), float(y)], np.float32), contact.get("realsense_size")


def _normalize_xy(xy_pix, src_w, src_h):
    x = np.clip(float(xy_pix[0]) / max(1.0, src_w - 1), 0.0, 1.0)
    y = np.clip(float(xy_pix[1]) / max(1.0, src_h - 1), 0.0, 1.0)
    return np.array([x, y], np.float32)


def _rescale_gaze_to_image(xy_pix, gaze_size, dst_w, dst_h):
    xy_pix = np.asarray(xy_pix, dtype=np.float32).copy()
    if isinstance(gaze_size, (list, tuple)) and len(gaze_size) >= 2:
        gaze_w, gaze_h = float(gaze_size[0]), float(gaze_size[1])
        xy_pix[0] *= max(1.0, dst_w - 1) / max(1.0, gaze_w - 1)
        xy_pix[1] *= max(1.0, dst_h - 1) / max(1.0, gaze_h - 1)
    xy_pix[0] = np.clip(xy_pix[0], 0.0, max(0.0, dst_w - 1))
    xy_pix[1] = np.clip(xy_pix[1], 0.0, max(0.0, dst_h - 1))
    return xy_pix


def _split_train_val(samples, val_ratio, seed, split_by="episode"):
    """Hold out whole episodes by default; see the --split_by flag for why."""
    if split_by == "frame":
        rng = random.Random(seed)
        idxs = list(range(len(samples)))
        rng.shuffle(idxs)
        split = int(len(samples) * (1.0 - val_ratio))
        return [samples[i] for i in idxs[:split]], [samples[i] for i in idxs[split:]]

    # An episode is only unique together with the recording it came from.
    keys = sorted({(s["recording_root"], int(s["episode_index"])) for s in samples})
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_val = max(1, int(round(len(keys) * val_ratio)))
    val_keys = set(keys[:n_val])
    train, val = [], []
    for sample in samples:
        key = (sample["recording_root"], int(sample["episode_index"]))
        (val if key in val_keys else train).append(sample)
    print(f"[split] episodes total={len(keys)} val={len(val_keys)} train={len(keys)-len(val_keys)}")
    print(f"[split] held-out episodes: "
          f"{sorted((Path(r).name[-8:], e) for r, e in val_keys)}")
    return train, val


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


def _load_samples_from_metadata(metadata_path: Path, dataset_name: str):
    metadata = json.loads(metadata_path.read_text())
    frame_root = Path(metadata.get("frame_root", metadata_path.parent)).expanduser().resolve()
    expected_frames = list(_metadata_frames(metadata))

    samples = []
    skipped_missing_image = 0
    skipped_missing_gaze = 0
    for episode_index, frame_id in expected_frames:
        frame_dir = frame_root / f"frame_{frame_id}"
        image_bgr = _load_color_bgr(frame_dir)
        if image_bgr is None:
            skipped_missing_image += 1
            continue

        src_h, src_w = image_bgr.shape[:2]
        xy_pix, gaze_size = _read_realsense_gaze(frame_dir)
        if xy_pix is None:
            skipped_missing_gaze += 1
            continue

        xy_pix = _rescale_gaze_to_image(xy_pix, gaze_size, src_w, src_h)
        xy_norm = _normalize_xy(xy_pix, src_w, src_h)

        image_resized = cv2.resize(image_bgr, (FLAGS.resize_w, FLAGS.resize_h), interpolation=cv2.INTER_LINEAR)
        image_rgb = image_resized[..., ::-1].copy()
        samples.append(
            {
                "observations": {FLAGS.image_key: image_rgb},
                "gaze_xy": xy_norm.astype(np.float32),
                "episode_index": np.int32(episode_index),
                "frame_id": np.int32(frame_id),
                "recording_root": str(frame_root),
                "dataset_name": dataset_name,
            }
        )

    print(
        f"[source:{dataset_name}] metadata={metadata_path} expected={len(expected_frames)} "
        f"exported={len(samples)} skipped_missing_image={skipped_missing_image} "
        f"skipped_missing_gaze={skipped_missing_gaze}"
    )
    return samples


def _load_samples_from_roots(roots, split_name: str):
    all_samples = []
    for i, root in enumerate(roots):
        metadata_path = _metadata_path_from_root(root)
        if not metadata_path.exists():
            print(f"[warn:{split_name}] missing metadata, skip: {metadata_path}")
            continue
        dataset_name = f"{split_name}:{metadata_path.parent.name}"
        all_samples.extend(_load_samples_from_metadata(metadata_path, dataset_name))
    return all_samples


def main(_):
    out_dir = Path(FLAGS.out_dir).expanduser().resolve()

    if FLAGS.use_metadata_fallback:
        samples = _load_samples_from_metadata(Path(FLAGS.metadata).expanduser().resolve(), "metadata")
        train_samples, val_samples = _split_train_val(
            samples, FLAGS.val_ratio, FLAGS.seed, FLAGS.split_by)
        test_samples = []
    else:
        train_source_samples = _load_samples_from_roots(FLAGS.train_root, "train")
        explicit_val = len(FLAGS.val_root) > 0
        if explicit_val:
            train_samples = train_source_samples
            val_samples = _load_samples_from_roots(FLAGS.val_root, "val")
        else:
            train_samples, val_samples = _split_train_val(
                train_source_samples, FLAGS.val_ratio, FLAGS.seed, FLAGS.split_by)
        test_samples = _load_samples_from_roots(FLAGS.test_root, "test")

    if not train_samples and not val_samples and not test_samples:
        print("[warn] no samples exported.")
        return

    print(f"[split] train={len(train_samples)}")
    print(f"[split] val  ={len(val_samples)}")
    print(f"[split] test ={len(test_samples)}")

    _save_shards(train_samples, out_dir / "train", FLAGS.shard_size, "gaze_samples")
    _save_shards(val_samples, out_dir / "val", FLAGS.shard_size, "gaze_samples")
    if test_samples:
        _save_shards(test_samples, out_dir / "test", FLAGS.shard_size, "gaze_samples")


if __name__ == "__main__":
    app.run(main)
