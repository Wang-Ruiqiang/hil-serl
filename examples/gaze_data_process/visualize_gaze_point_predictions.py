#!/usr/bin/env python3
from __future__ import annotations

import glob
import math
import os
import pickle as pkl
import sys
from pathlib import Path

import cv2
import numpy as np
from absl import app, flags


project_root = next(
    p for p in Path(__file__).resolve().parents if (p / "serl_launcher").exists()
)
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "serl_launcher"))

from serl_launcher.networks.gaze_point_predictor import load_gaze_point_predictor_func


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "data_dir",
    str((Path(__file__).resolve().parent / "gaze_cls_data" / "val")),
    "Directory with exported gaze pkl shards.",
)
flags.DEFINE_string(
    "checkpoint_dir",
    str((Path(__file__).resolve().parent / "gaze_heatmap_ckpt")),
    "Checkpoint directory to evaluate.",
)
flags.DEFINE_string(
    "out_dir",
    str((Path(__file__).resolve().parent / "gaze_heatmap_vis")),
    "Directory for rendered prediction images.",
)
flags.DEFINE_string("image_key", "front_camera", "Observation image key in pkl samples.")
flags.DEFINE_integer("num_samples", 100, "Number of random samples to visualize.")
flags.DEFINE_integer("seed", 0, "Random seed for selecting samples.")
flags.DEFINE_integer("batch_size", 64, "Prediction batch size.")
flags.DEFINE_integer("image_width", 128, "Model input image width.")
flags.DEFINE_integer("image_height", 128, "Model input image height.")
flags.DEFINE_string(
    "encoder_variant",
    "resnetv1-10",
    "Encoder backbone variant: resnetv1-10-frozen, resnetv1-18-frozen, or resnetv1-10.",
)
flags.DEFINE_integer("display_scale", 3, "Scale factor for rendered images.")
flags.DEFINE_integer("info_bar_height", 64, "Bottom info bar height in pixels after scaling.")
flags.DEFINE_float("gt_sigma_px", 4.0, "Gaussian sigma for rendering the ground-truth heatmap.")
flags.DEFINE_float(
    "eval_conf_sigma_px",
    12.0,
    "Pixel-distance sigma for evaluation-only gaze confidence from pred/true point distance.",
)
flags.DEFINE_float("heatmap_alpha", 0.45, "Heatmap overlay opacity.")
flags.DEFINE_boolean("save_individual", True, "Save one png per selected sample.")
flags.DEFINE_boolean("save_montage", True, "Save a montage png for selected samples.")
flags.DEFINE_boolean("pause_each", False, "Pause after each rendered sample and wait for user input.")


def _load_samples(data_dir: str):
    samples = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.pkl"))):
        with open(path, "rb") as f:
            samples.extend(pkl.load(f))
    if not samples:
        raise FileNotFoundError(f"No pkl samples found in {data_dir}")
    return samples


def _select_samples(samples, count: int, seed: int):
    rng = np.random.RandomState(seed)
    count = min(count, len(samples))
    idxs = rng.choice(len(samples), size=count, replace=False)
    return [samples[int(idx)] for idx in idxs]


def _predict(samples, predict_fn):
    images = np.asarray(
        [sample["observations"][FLAGS.image_key] for sample in samples],
        dtype=np.float32,
    )
    heatmaps = []
    gaze_conf = []
    for start in range(0, len(images), FLAGS.batch_size):
        batch = images[start : start + FLAGS.batch_size]
        outputs = predict_fn({FLAGS.image_key: batch})
        heatmaps.append(np.asarray(outputs["gaze_heat"]))
        gaze_conf.append(np.asarray(outputs["gaze_conf"]))
    return np.concatenate(heatmaps, axis=0), np.concatenate(gaze_conf, axis=0)


def _heatmap_to_xy(heatmap):
    height, width = heatmap.shape
    flat_idx = int(np.argmax(heatmap))
    y, x = divmod(flat_idx, width)
    return np.array(
        [
            x / max(1.0, float(width - 1)),
            y / max(1.0, float(height - 1)),
        ],
        dtype=np.float32,
    )


def _xy_to_heatmap(xy, height: int, width: int, sigma_px: float):
    xy = np.asarray(xy, dtype=np.float32)
    cx = float(xy[0]) * float(width - 1)
    cy = float(xy[1]) * float(height - 1)
    yy, xx = np.mgrid[0:height, 0:width]
    sigma = max(float(sigma_px), 1e-3)
    heatmap = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma))
    heatmap = heatmap / max(float(heatmap.max()), 1e-8)
    return heatmap.astype(np.float32)


def _to_pixel_xy(xy, width: int, height: int):
    xy = np.asarray(xy, dtype=np.float32)
    x = int(np.clip(round(float(xy[0]) * (width - 1)), 0, width - 1))
    y = int(np.clip(round(float(xy[1]) * (height - 1)), 0, height - 1))
    return x, y


def _distance_eval_score(true_xy, pred_xy, width: int, height: int, sigma_px: float):
    true_x, true_y = _to_pixel_xy(true_xy, width, height)
    pred_x, pred_y = _to_pixel_xy(pred_xy, width, height)
    dist_px = float(np.hypot(pred_x - true_x, pred_y - true_y))
    sigma = max(float(sigma_px), 1e-3)
    return float(np.exp(-(dist_px * dist_px) / (2.0 * sigma * sigma))), dist_px


def _overlay_heatmap(image, heatmap, alpha: float):
    heat_u8 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return cv2.addWeighted(image, 1.0 - alpha, colored, alpha, 0.0)


def _colorize_heatmap(heatmap):
    heat_u8 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)


def _draw_label(image, text: str, origin, color, font_scale=0.45, thickness=1):
    x, y = origin
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(image, (x, y - th - 5), (x + tw + 4, y + 3), (0, 0, 0), -1)
    cv2.putText(image, text, (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def _render_points(image_rgb, true_xy, pred_xy, title: str):
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = image.shape[:2]
    true_pix = _to_pixel_xy(true_xy, w, h)
    pred_pix = _to_pixel_xy(pred_xy, w, h)
    cv2.drawMarker(
        image,
        true_pix,
        (0, 255, 0),
        markerType=cv2.MARKER_CROSS,
        markerSize=10,
        thickness=2,
        line_type=cv2.LINE_AA,
    )
    cv2.drawMarker(
        image,
        pred_pix,
        (0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=10,
        thickness=2,
        line_type=cv2.LINE_AA,
    )
    _draw_label(image, title, (6, 18), (255, 255, 255), font_scale=0.5)
    return image


def _render_sample(sample, pred_heatmap, gaze_conf: float, eval_point_score: float):
    image_rgb = sample["observations"][FLAGS.image_key]
    true_xy = np.asarray(sample["gaze_xy"], dtype=np.float32)
    pred_xy = _heatmap_to_xy(pred_heatmap)
    point_panel = _render_points(image_rgb, true_xy, pred_xy, "realsense points")
    heatmap_panel = _colorize_heatmap(pred_heatmap)
    _draw_label(heatmap_panel, "pred heatmap", (6, 18), (255, 255, 255), font_scale=0.5)
    canvas = np.concatenate([point_panel, heatmap_panel], axis=1)

    scale = max(1, int(FLAGS.display_scale))
    if scale > 1:
        canvas = cv2.resize(
            canvas,
            (canvas.shape[1] * scale, canvas.shape[0] * scale),
            interpolation=cv2.INTER_NEAREST,
        )

    h, w = canvas.shape[:2]
    info_h = max(0, int(FLAGS.info_bar_height))
    if info_h <= 0:
        return canvas, pred_xy

    out = np.zeros((h + info_h, w, 3), dtype=np.uint8)
    out[:h] = canvas
    frame_id = sample.get("frame_id", "?")
    episode_index = sample.get("episode_index", "?")
    true_text = f"true xy=({true_xy[0]:.2f},{true_xy[1]:.2f})"
    pred_text = f"pred xy=({pred_xy[0]:.2f},{pred_xy[1]:.2f})"
    conf_text = f"gaze_conf={gaze_conf:.3f}"
    eval_text = f"eval_score={eval_point_score:.3f}"
    meta_text = f"ep={episode_index} frame={frame_id}"
    _draw_label(out, true_text, (6, h + 20), (0, 255, 0), font_scale=0.5)
    _draw_label(out, pred_text, (6, h + 42), (0, 0, 255), font_scale=0.5)
    _draw_label(out, conf_text, (max(6, w // 2), h + 20), (0, 255, 255), font_scale=0.5)
    _draw_label(out, eval_text, (max(6, w // 2), h + 42), (0, 200, 255), font_scale=0.5)
    _draw_label(out, meta_text, (max(6, w // 2), h + 62), (255, 255, 255), font_scale=0.5)
    return out, pred_xy


def _save_montage(images, path: Path):
    if not images:
        return
    h, w = images[0].shape[:2]
    cols = int(math.ceil(math.sqrt(len(images))))
    rows = int(math.ceil(len(images) / cols))
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, image in enumerate(images):
        r = i // cols
        c = i % cols
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = image
    cv2.imwrite(str(path), canvas)


def main(_):
    out_dir = Path(FLAGS.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = _load_samples(FLAGS.data_dir)
    selected = _select_samples(samples, FLAGS.num_samples, FLAGS.seed)

    predict_fn = load_gaze_point_predictor_func(
        key=np.asarray([0, 0], dtype=np.uint32),
        sample_observations={
            FLAGS.image_key: np.zeros((1, FLAGS.image_height, FLAGS.image_width, 3), np.float32)
        },
        image_keys=[FLAGS.image_key],
        checkpoint_path=str(Path(FLAGS.checkpoint_dir).expanduser().resolve()),
        encoder_variant=FLAGS.encoder_variant,
    )
    pred_heatmaps, gaze_conf_all = _predict(selected, predict_fn)

    rendered = []
    err_sum = 0.0
    err_count = 0
    max_sum = 0.0
    gaze_conf_sum = 0.0
    eval_score_sum = 0.0
    for i, (sample, heatmap, gaze_conf) in enumerate(
        zip(selected, pred_heatmaps, gaze_conf_all)
    ):
        heatmap = np.asarray(heatmap, dtype=np.float32)
        raw_peak = float(heatmap.max())
        heatmap = heatmap / max(float(heatmap.max()), 1e-8)
        true_xy = np.asarray(sample["gaze_xy"], dtype=np.float32)
        pred_xy = _heatmap_to_xy(heatmap)
        eval_point_score, dist_px = _distance_eval_score(
            true_xy,
            pred_xy,
            FLAGS.image_width,
            FLAGS.image_height,
            FLAGS.eval_conf_sigma_px,
        )
        gaze_conf = float(gaze_conf)
        image, pred_xy = _render_sample(sample, heatmap, gaze_conf, eval_point_score)
        frame_id = sample.get("frame_id", i)
        episode_index = sample.get("episode_index", "x")
        err_sum += float(np.abs(pred_xy - true_xy).sum())
        err_count += 1
        max_sum += raw_peak
        gaze_conf_sum += gaze_conf
        eval_score_sum += eval_point_score
        rendered.append(image)
        if FLAGS.save_individual:
            out_path = out_dir / f"sample_{i:03d}_ep{episode_index}_frame{frame_id}.png"
            cv2.imwrite(str(out_path), image)
        print(
            f"[vis][sample {i + 1}/{len(selected)}] "
            f"ep={episode_index} frame={frame_id} "
            f"true_xy={true_xy.tolist()} pred_xy={pred_xy.tolist()} "
            f"dist_px={dist_px:.2f} eval_point_score={eval_point_score:.4f} "
            f"raw_pred_peak={raw_peak:.4f} gaze_conf={gaze_conf:.4f}"
        )
        if FLAGS.pause_each:
            user_input = input("[vis][pause] Enter=next, q=quit: ").strip().lower()
            if user_input in {"q", "quit", "exit"}:
                print("[vis] stopped by user.")
                break

    if FLAGS.save_montage:
        _save_montage(rendered, out_dir / "montage.png")

    print(f"[vis] data_dir={FLAGS.data_dir}")
    print(f"[vis] checkpoint_dir={FLAGS.checkpoint_dir}")
    print(f"[vis] encoder_variant={FLAGS.encoder_variant}")
    print(f"[vis] wrote {len(rendered)} samples to {out_dir}")
    print(f"[vis] selected xy_l1={err_sum / max(1, err_count):.4f} valid_count={err_count}")
    print(f"[vis] mean_raw_pred_peak={max_sum / max(1, err_count):.4f}")
    print(f"[vis] mean_gaze_conf={gaze_conf_sum / max(1, err_count):.4f}")
    print(f"[vis] mean_eval_point_score={eval_score_sum / max(1, err_count):.4f}")


if __name__ == "__main__":
    app.run(main)
