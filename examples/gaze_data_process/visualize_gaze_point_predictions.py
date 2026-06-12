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


project_root = next(p for p in Path(__file__).resolve().parents if (p / "serl_launcher").exists())
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "serl_launcher"))

from serl_launcher.networks.gaze_point_predictor import load_gaze_point_predictor_func


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "frame_root",
    "/media/user/data3/wrq/recorded_data/tennis_ball_pick/tennis_ball_pick-5-27-0",
    "Recording root with frame_*/color_image.jpg. If empty, samples are loaded from --data_dir.",
)
flags.DEFINE_string(
    "data_dir",
    str((Path(__file__).resolve().parent / "gaze_cls_data" / "val")),
    "Optional directory with exported pkl shards. Used only when --frame_root is empty.",
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
flags.DEFINE_integer("num_samples", 100, "Number of random frames/samples to visualize.")
flags.DEFINE_integer("seed", 0, "Random seed for selecting frames/samples.")
flags.DEFINE_integer("batch_size", 64, "Prediction batch size.")
flags.DEFINE_integer("image_width", 128, "Model input image width.")
flags.DEFINE_integer("image_height", 128, "Model input image height.")
flags.DEFINE_string(
    "encoder_variant",
    "resnetv1-10",
    "Encoder backbone variant: resnetv1-10-frozen, resnetv1-18-frozen, or resnetv1-10.",
)
flags.DEFINE_integer("display_scale", 3, "Scale factor for rendered model-input images.")
flags.DEFINE_integer("info_bar_height", 52, "Bottom info bar height in pixels after scaling.")
flags.DEFINE_float("heatmap_alpha", 0.45, "Heatmap overlay opacity.")
flags.DEFINE_boolean("save_individual", True, "Save one png per selected frame/sample.")
flags.DEFINE_boolean("save_montage", True, "Save a montage png for selected frames/samples.")
flags.DEFINE_boolean("pause_each", False, "Pause after each rendered sample and wait for user input.")


def _frame_id(frame_dir: Path):
    try:
        return int(frame_dir.name.split("_", 1)[1])
    except Exception:
        return -1


def _load_frame_samples(frame_root: str):
    root = Path(frame_root).expanduser().resolve()
    samples = []
    for frame_dir in sorted(root.glob("frame_*"), key=_frame_id):
        image_path = frame_dir / "color_image.jpg"
        if not image_path.exists():
            continue
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            continue
        image_resized = cv2.resize(image_bgr, (FLAGS.image_width, FLAGS.image_height), interpolation=cv2.INTER_LINEAR)
        samples.append(
            {
                "observations": {FLAGS.image_key: image_resized[..., ::-1].copy()},
                "frame_id": np.int32(_frame_id(frame_dir)),
                "recording_root": str(root),
            }
        )
    if not samples:
        raise FileNotFoundError(f"No frame_*/color_image.jpg images found under {root}")
    return samples


def _load_pkl_samples(data_dir: str):
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
    images = np.asarray([sample["observations"][FLAGS.image_key] for sample in samples], dtype=np.float32)
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
    return np.array([x / max(1.0, float(width - 1)), y / max(1.0, float(height - 1))], dtype=np.float32)


def _to_pixel_xy(xy, width: int, height: int):
    xy = np.asarray(xy, dtype=np.float32)
    x = int(np.clip(round(float(xy[0]) * (width - 1)), 0, width - 1))
    y = int(np.clip(round(float(xy[1]) * (height - 1)), 0, height - 1))
    return x, y


def _colorize_heatmap(heatmap):
    heat_u8 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)


def _draw_label(image, text: str, origin, color, font_scale=0.45, thickness=1):
    x, y = origin
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(image, (x, y - th - 5), (x + tw + 4, y + 3), (0, 0, 0), -1)
    cv2.putText(image, text, (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def _render_sample(sample, pred_heatmap, gaze_conf: float):
    image_rgb = sample["observations"][FLAGS.image_key]
    pred_xy = _heatmap_to_xy(pred_heatmap)
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = image.shape[:2]
    pred_pix = _to_pixel_xy(pred_xy, w, h)
    cv2.drawMarker(
        image,
        pred_pix,
        (0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=14,
        thickness=2,
        line_type=cv2.LINE_AA,
    )
    _draw_label(image, "pred gaze point", (6, 18), (255, 255, 255), font_scale=0.5)

    heatmap_panel = _colorize_heatmap(pred_heatmap)
    _draw_label(heatmap_panel, "pred heatmap", (6, 18), (255, 255, 255), font_scale=0.5)
    canvas = np.concatenate([image, heatmap_panel], axis=1)

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
    recording = Path(str(sample.get("recording_root", ""))).name
    pred_text = f"pred xy=({pred_xy[0]:.2f},{pred_xy[1]:.2f})"
    conf_text = f"gaze_conf={float(gaze_conf):.3f}"
    meta_text = f"recording={recording} frame={frame_id}"
    _draw_label(out, pred_text, (6, h + 20), (0, 0, 255), font_scale=0.5)
    _draw_label(out, conf_text, (max(6, w // 2), h + 20), (0, 255, 255), font_scale=0.5)
    _draw_label(out, meta_text, (6, h + 42), (255, 255, 255), font_scale=0.5)
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

    if FLAGS.frame_root.strip():
        samples = _load_frame_samples(FLAGS.frame_root)
        data_source = FLAGS.frame_root
    else:
        samples = _load_pkl_samples(FLAGS.data_dir)
        data_source = FLAGS.data_dir

    selected = _select_samples(samples, FLAGS.num_samples, FLAGS.seed)
    predict_fn = load_gaze_point_predictor_func(
        key=np.asarray([0, 0], dtype=np.uint32),
        sample_observations={FLAGS.image_key: np.zeros((1, FLAGS.image_height, FLAGS.image_width, 3), np.float32)},
        image_keys=[FLAGS.image_key],
        checkpoint_path=str(Path(FLAGS.checkpoint_dir).expanduser().resolve()),
        encoder_variant=FLAGS.encoder_variant,
    )
    pred_heatmaps, gaze_conf_all = _predict(selected, predict_fn)

    rendered = []
    gaze_conf_sum = 0.0
    for i, (sample, heatmap, gaze_conf) in enumerate(zip(selected, pred_heatmaps, gaze_conf_all)):
        heatmap = np.asarray(heatmap, dtype=np.float32)
        raw_peak = float(heatmap.max())
        heatmap = heatmap / max(float(heatmap.max()), 1e-8)
        gaze_conf = float(gaze_conf)
        image, pred_xy = _render_sample(sample, heatmap, gaze_conf)
        frame_id = sample.get("frame_id", i)
        rendered.append(image)
        gaze_conf_sum += gaze_conf
        if FLAGS.save_individual:
            out_path = out_dir / f"sample_{i:03d}_frame{frame_id}.png"
            cv2.imwrite(str(out_path), image)
        print(
            f"[vis][sample {i + 1}/{len(selected)}] frame={frame_id} "
            f"pred_xy={pred_xy.tolist()} raw_pred_peak={raw_peak:.4f} gaze_conf={gaze_conf:.4f}"
        )
        if FLAGS.pause_each:
            user_input = input("[vis][pause] Enter=next, q=quit: ").strip().lower()
            if user_input in {"q", "quit", "exit"}:
                print("[vis] stopped by user.")
                break

    if FLAGS.save_montage:
        _save_montage(rendered, out_dir / "montage.png")

    print(f"[vis] data_source={data_source}")
    print(f"[vis] checkpoint_dir={FLAGS.checkpoint_dir}")
    print(f"[vis] encoder_variant={FLAGS.encoder_variant}")
    print(f"[vis] wrote {len(rendered)} samples to {out_dir}")
    print(f"[vis] mean_gaze_conf={gaze_conf_sum / max(1, len(rendered)):.4f}")


if __name__ == "__main__":
    app.run(main)
