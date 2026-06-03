#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np
from absl import app, flags

project_root = next(
    p for p in Path(__file__).resolve().parents if (p / "serl_launcher").exists()
)
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gaze_mapping.gaze_to_realsense_homography import EpisodeHomographyMap, load_homography_source


FLAGS = flags.FLAGS
DEFAULT_METADATA = (
    "/media/user/data3/wrq/recorded_data/tennis_ball_pick/"
    "tennis_ball_pick-5-27-0/recording_metadata.json"
)
DEFAULT_HOMOGRAPHY = str(Path(__file__).resolve().parent / "gaze_episode_homographies.json")
DEFAULT_LABELS = str(Path(__file__).resolve().parent / "gaze_homography_labels.json")

flags.DEFINE_string("metadata", DEFAULT_METADATA, "Path to recording_metadata.json.")
flags.DEFINE_string("frame_root", "", "Optional frame root override. Defaults to metadata frame_root.")
flags.DEFINE_string("labels", DEFAULT_LABELS, "Path to gaze_homography_labels.json.")
flags.DEFINE_string(
    "homography",
    DEFAULT_HOMOGRAPHY,
    "Path to gaze_episode_homographies.json or a global gaze_homography.json.",
)
flags.DEFINE_string("out_dir", "./gaze_homography_preview", "Directory to save preview images.")
flags.DEFINE_boolean("save_individual", True, "Save one image per labelled sample.")
flags.DEFINE_boolean("save_montage", True, "Save a montage image.")
flags.DEFINE_integer("max_samples", 200, "Maximum number of labelled samples to preview.")
flags.DEFINE_integer("display_scale", 2, "Display scale for preview images.")


def _load_metadata(metadata_path: Path):
    metadata = json.loads(metadata_path.read_text())
    frame_root = Path(FLAGS.frame_root).expanduser().resolve() if FLAGS.frame_root else Path(
        metadata.get("frame_root", metadata_path.parent)
    ).expanduser().resolve()
    return metadata, frame_root


def _load_rs_image(frame_root: Path, frame_id: int):
    for path in (
        frame_root / "frame_{}".format(frame_id) / "color_image.jpg",
        frame_root / "rs_images" / f"{frame_id}.jpg",
    ):
        if path.exists():
            image = cv2.imread(str(path))
            if image is not None:
                return image
    return None


def _draw_point(image, point, color, label, radius=6, thickness=2):
    if point is None:
        return image
    x, y = int(round(float(point[0]))), int(round(float(point[1])))
    cv2.circle(image, (x, y), radius, color, thickness, cv2.LINE_AA)
    cv2.putText(
        image,
        label,
        (x + 8, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )
    return image


def _render_sample(frame_root: Path, label: dict, homography_map):
    episode_index = int(label["episode_index"])
    frame_id = int(label["frame_id"])
    eye_uv = np.asarray(label["eye_uv"], dtype=np.float32)
    true_rs = np.asarray(label["rs_xy"], dtype=np.float32)
    homography = homography_map.get(episode_index) if isinstance(homography_map, EpisodeHomographyMap) else homography_map
    pred_rs = None
    if homography is not None:
        pred_rs = np.asarray(homography.transform_point(eye_uv), dtype=np.float32)

    image = _load_rs_image(frame_root, frame_id)
    if image is None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
    scale = max(1, int(FLAGS.display_scale))
    if scale > 1:
        image = cv2.resize(
            image,
            (image.shape[1] * scale, image.shape[0] * scale),
            interpolation=cv2.INTER_NEAREST,
        )
        true_rs = true_rs * scale
        if pred_rs is not None:
            pred_rs = pred_rs * scale

    image = _draw_point(image, true_rs, (0, 255, 0), "gt")
    image = _draw_point(image, pred_rs, (0, 0, 255), "pred")
    err = float(np.linalg.norm((pred_rs - true_rs) if pred_rs is not None else np.array([0.0, 0.0])))

    header = [
        f"ep={episode_index} frame={frame_id}",
        f"err_px={err:.2f}" if pred_rs is not None else "pred missing",
    ]
    bar_h = 54
    canvas = np.zeros((image.shape[0] + bar_h, image.shape[1], 3), dtype=np.uint8)
    canvas[: image.shape[0]] = image
    cv2.putText(canvas, header[0], (10, image.shape[0] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(canvas, header[1], (10, image.shape[0] + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return canvas, err


def _save_montage(images, path: Path):
    if not images:
        return
    h, w = images[0].shape[:2]
    cols = int(np.ceil(np.sqrt(len(images))))
    rows = int(np.ceil(len(images) / cols))
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for idx, image in enumerate(images):
        r = idx // cols
        c = idx % cols
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = image
    cv2.imwrite(str(path), canvas)


def main(_):
    if FLAGS.metadata is None:
        raise ValueError("Please pass --metadata=/path/to/recording_metadata.json")
    if FLAGS.labels is None:
        raise ValueError("Please pass --labels=/path/to/gaze_homography_labels.json")
    if FLAGS.homography is None:
        raise ValueError("Please pass --homography=/path/to/gaze_episode_homographies.json")

    metadata_path = Path(FLAGS.metadata).expanduser().resolve()
    _, frame_root = _load_metadata(metadata_path)
    labels = json.loads(Path(FLAGS.labels).expanduser().read_text())
    homography_map = load_homography_source(FLAGS.homography)

    out_dir = Path(FLAGS.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    images = []
    errs = []
    for idx, label in enumerate(labels[: FLAGS.max_samples]):
        image, err = _render_sample(frame_root, label, homography_map)
        images.append(image)
        errs.append(err)
        if FLAGS.save_individual:
            ep = int(label["episode_index"])
            frame_id = int(label["frame_id"])
            cv2.imwrite(str(out_dir / f"preview_ep{ep}_frame{frame_id}.png"), image)

    if FLAGS.save_montage:
        _save_montage(images, out_dir / "montage.png")

    if errs:
        print(f"[preview] samples={len(errs)} mean_px={float(np.mean(errs)):.2f} median_px={float(np.median(errs)):.2f}")
    print(f"[preview] wrote previews to {out_dir}")


if __name__ == "__main__":
    app.run(main)
