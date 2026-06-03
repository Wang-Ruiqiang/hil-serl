#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
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

from gaze_mapping.gaze_to_realsense_homography import EpisodeHomographyMap, GazeHomography


FLAGS = flags.FLAGS
flags.DEFINE_string("metadata", "/media/user/data3/wrq/recorded_data/tennis_ball_pick/"
    "tennis_ball_pick-5-27-0/recording_metadata.json", "Path to recording_metadata.json.")
flags.DEFINE_string("frame_root", "", "Optional frame root override. Defaults to metadata frame_root.")
DEFAULT_MAPPING_DIR = Path(__file__).resolve().parent
DEFAULT_EPISODE_HOMOGRAPHY = DEFAULT_MAPPING_DIR / "gaze_episode_homographies.json"
flags.DEFINE_string(
    "out",
    str(DEFAULT_EPISODE_HOMOGRAPHY),
    "Output homography JSON path. Defaults to the per-episode mapping file.",
)
flags.DEFINE_string(
    "labels_out",
    str(DEFAULT_MAPPING_DIR / "gaze_homography_labels.json"),
    "Where to save clicked point labels.",
)
flags.DEFINE_string("labels_in", "", "Optional existing labels JSON. If set, only fit and save homographies.")
flags.DEFINE_string(
    "fit_scope",
    "per_episode",
    "Homography fit scope: per_episode or global.",
)
flags.DEFINE_integer("frames_per_episode", 20, "Number of frames to label per episode/demo.")
flags.DEFINE_integer("start_episode", 0, "First episode_index to label.")
flags.DEFINE_integer("num_episodes", 0, "Number of episodes to label. 0 means all episodes from start_episode.")
flags.DEFINE_boolean("success_only", True, "Use only successful episodes.")
flags.DEFINE_integer("max_width", 1800, "Max OpenCV display width.")
flags.DEFINE_integer("max_height", 900, "Max OpenCV display height.")
flags.DEFINE_integer("eye_width", 1280, "Eye-tracker image width for pupil_gaze fallback.")
flags.DEFINE_integer("eye_height", 720, "Eye-tracker image height for pupil_gaze fallback.")
flags.DEFINE_boolean("flip_eye_y", True, "Flip pupil norm_pos y for pupil_gaze fallback.")
flags.DEFINE_float("ransac_reproj_threshold", 8.0, "RANSAC reprojection threshold in RealSense pixels.")
flags.DEFINE_float(
    "max_label_error_px",
    100.0,
    "Drop labelled samples above this reprojection error before final refit. 0 disables.",
)


def _load_metadata(metadata_path: Path):
    metadata = json.loads(metadata_path.read_text())
    frame_root = Path(FLAGS.frame_root).expanduser().resolve() if FLAGS.frame_root else Path(
        metadata.get("frame_root", metadata_path.parent)
    ).expanduser().resolve()
    return metadata, frame_root


def _episode_frame_ids(metadata: dict):
    records = metadata.get("episode_ranges", [])
    if FLAGS.success_only:
        records = [rec for rec in records if bool(rec.get("success", False))]

    episodes = []
    for rec in sorted(records, key=lambda item: int(item.get("episode_index", 0))):
        ranges = rec.get("kept_frame_ranges") or [
            {"start_frame": rec["start_frame"], "end_frame": rec["end_frame"]}
        ]
        frame_ids = []
        for rng in ranges:
            start = int(rng["start_frame"])
            end = int(rng["end_frame"])
            if end < start:
                start, end = end, start
            frame_ids.extend(range(start, end + 1))
        episode_index = int(rec.get("episode_index", len(episodes)))
        if episode_index < FLAGS.start_episode:
            continue
        if frame_ids:
            episodes.append((int(rec.get("episode_index", len(episodes))), frame_ids))
    if FLAGS.num_episodes > 0:
        episodes = episodes[: FLAGS.num_episodes]
    return episodes


def _read_eye_gaze(frame_dir: Path):
    contact_path = frame_dir / "gaze_contact.json"
    if contact_path.exists():
        try:
            contact = json.loads(contact_path.read_text())
            uv = contact.get("gaze_uv_in_eye")
            if isinstance(uv, (list, tuple)) and len(uv) >= 2:
                return np.array([float(uv[0]), float(uv[1])], dtype=np.float32)
        except Exception:
            pass

    pupil_path = frame_dir / "pupil_gaze.json"
    if not pupil_path.exists():
        return None
    try:
        data = json.loads(pupil_path.read_text())
        payload = data.get("data", data)
        norm_pos = payload.get("norm_pos")
        if norm_pos is None or len(norm_pos) < 2:
            return None
        x_norm, y_norm = float(norm_pos[0]), float(norm_pos[1])
        x = x_norm * float(FLAGS.eye_width - 1)
        y_raw = (1.0 - y_norm) if FLAGS.flip_eye_y else y_norm
        y = y_raw * float(FLAGS.eye_height - 1)
        return np.array([x, y], dtype=np.float32)
    except Exception:
        return None


def _load_et_image(frame_root: Path, frame_dir: Path, frame_id: int):
    for path in (
        frame_root / "et_images_gaze" / f"{frame_id}.jpg",
        frame_root / "et_images" / f"{frame_id}.jpg",
        frame_dir / "eye_image.jpg",
    ):
        if path.exists():
            image = cv2.imread(str(path))
            if image is not None:
                return image
    return None


def _load_rs_image(frame_root: Path, frame_dir: Path, frame_id: int):
    for path in (
        frame_dir / "color_image.jpg",
        frame_root / "rs_images" / f"{frame_id}.jpg",
    ):
        if path.exists():
            image = cv2.imread(str(path))
            if image is not None:
                return image
    return None


def _draw_eye_gaze(image, uv):
    out = image.copy()
    x, y = int(round(float(uv[0]))), int(round(float(uv[1])))
    cv2.circle(out, (x, y), 12, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.circle(out, (x, y), 4, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.line(out, (x - 20, y), (x + 20, y), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.line(out, (x, y - 20), (x, y + 20), (0, 0, 255), 2, cv2.LINE_AA)
    return out


def _make_canvas(et_image, rs_image, eye_uv):
    et = _draw_eye_gaze(et_image, eye_uv)
    rs = rs_image.copy()
    target_h = min(max(et.shape[0], rs.shape[0]), FLAGS.max_height)
    et_scale = target_h / float(et.shape[0])
    rs_scale = target_h / float(rs.shape[0])
    et_show = cv2.resize(et, (int(round(et.shape[1] * et_scale)), target_h))
    rs_show = cv2.resize(rs, (int(round(rs.shape[1] * rs_scale)), target_h))
    cv2.putText(et_show, "Eye tracker gaze", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(rs_show, "Click RealSense gaze point", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    canvas = cv2.hconcat([et_show, rs_show])
    scale = min(FLAGS.max_width / float(canvas.shape[1]), FLAGS.max_height / float(canvas.shape[0]), 1.0)
    if scale < 0.999:
        canvas = cv2.resize(canvas, (int(round(canvas.shape[1] * scale)), int(round(canvas.shape[0] * scale))))
    fixed = np.zeros((FLAGS.max_height, FLAGS.max_width, 3), dtype=np.uint8)
    h, w = canvas.shape[:2]
    fixed[:h, :w] = canvas
    return fixed, {
        "rs_offset_x": et_show.shape[1] * scale,
        "rs_scale": rs_scale * scale,
    }


class GazePointLabelWindow:
    def __init__(self):
        self.window = "label RealSense gaze point"
        self.canvas = None
        self.layout = None
        self.clicked = []
        self.frame_id = None
        self.episode_index = None
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, FLAGS.max_width, FLAGS.max_height)
        cv2.setMouseCallback(self.window, self._on_mouse)

    def close(self):
        cv2.destroyWindow(self.window)

    def _on_mouse(self, event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN or self.layout is None:
            return
        if x < self.layout["rs_offset_x"]:
            return
        self.clicked.clear()
        self.clicked.append((float(x), float(y)))
        self._redraw(self.clicked[0])

    def _redraw(self, point=None):
        shown = self.canvas.copy()
        text = (
            f"episode={self.episode_index} frame={self.frame_id} | "
            "click RS point | s/Enter save | n skip | q quit"
        )
        cv2.putText(shown, text, (10, shown.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        if point is not None:
            cv2.circle(shown, (int(point[0]), int(point[1])), 7, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.imshow(self.window, shown)

    def get_point(self, et_image, rs_image, eye_uv, frame_id: int, episode_index: int):
        self.canvas, self.layout = _make_canvas(et_image, rs_image, eye_uv)
        self.clicked.clear()
        self.frame_id = frame_id
        self.episode_index = episode_index
        self._redraw()
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                raise KeyboardInterrupt("Labelling cancelled.")
            if key == ord("n"):
                return None
            if self.clicked and key in (ord("s"), 13, 32):
                x_show, y_show = self.clicked[0]
                x_rs = (x_show - self.layout["rs_offset_x"]) / max(1e-6, self.layout["rs_scale"])
                y_rs = y_show / max(1e-6, self.layout["rs_scale"])
                return np.array([x_rs, y_rs], dtype=np.float32)


def _select_frames(metadata: dict):
    selected = []
    for episode_index, frame_ids in _episode_frame_ids(metadata):
        frame_ids = list(frame_ids)
        if len(frame_ids) <= FLAGS.frames_per_episode:
            chosen = frame_ids
        else:
            idxs = np.linspace(0, len(frame_ids) - 1, num=FLAGS.frames_per_episode)
            chosen = [frame_ids[int(round(idx))] for idx in idxs]
        selected.extend((episode_index, frame_id) for frame_id in chosen)
        print(f"[select] episode={episode_index} selected={len(chosen)}/{len(frame_ids)}")
    return selected


def _collect_labels(metadata: dict, frame_root: Path):
    labels = []
    label_window = GazePointLabelWindow()
    try:
        for episode_index, frame_id in _select_frames(metadata):
            frame_dir = frame_root / f"frame_{frame_id}"
            eye_uv = _read_eye_gaze(frame_dir)
            if eye_uv is None:
                print(f"[skip] missing eye gaze: episode={episode_index} frame={frame_id}")
                continue
            et_image = _load_et_image(frame_root, frame_dir, frame_id)
            rs_image = _load_rs_image(frame_root, frame_dir, frame_id)
            if et_image is None or rs_image is None:
                print(f"[skip] missing image: episode={episode_index} frame={frame_id}")
                continue
            rs_xy = label_window.get_point(et_image, rs_image, eye_uv, frame_id, episode_index)
            if rs_xy is None:
                print(f"[skip] user skipped episode={episode_index} frame={frame_id}")
                continue
            labels.append(
                {
                    "episode_index": int(episode_index),
                    "frame_id": int(frame_id),
                    "eye_uv": eye_uv.tolist(),
                    "rs_xy": rs_xy.tolist(),
                    "rs_size": [int(rs_image.shape[1]), int(rs_image.shape[0])],
                }
            )
            print(f"[label] episode={episode_index} frame={frame_id} eye={eye_uv.tolist()} rs={rs_xy.tolist()}")
    finally:
        label_window.close()
    return labels


def _fit_one_homography(scope_name: str, labels: list[dict]):
    if len(labels) < 4:
        print(f"[fit][skip] scope={scope_name} needs >=4 labels, got {len(labels)}")
        return None

    src = np.asarray([label["eye_uv"] for label in labels], dtype=np.float32)
    dst = np.asarray([label["rs_xy"] for label in labels], dtype=np.float32)
    matrix, inliers = cv2.findHomography(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(FLAGS.ransac_reproj_threshold),
    )
    if matrix is None:
        print(f"[fit][skip] scope={scope_name} cv2.findHomography failed")
        return None

    mapped = cv2.perspectiveTransform(src.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    errors = np.linalg.norm(mapped - dst, axis=-1)
    keep_mask = np.ones(len(labels), dtype=bool)
    if float(FLAGS.max_label_error_px) > 0:
        keep_mask = errors <= float(FLAGS.max_label_error_px)
        kept = int(keep_mask.sum())
        if 4 <= kept < len(labels):
            src_kept = src[keep_mask]
            dst_kept = dst[keep_mask]
            refined_matrix, refined_inliers = cv2.findHomography(
                src_kept,
                dst_kept,
                method=cv2.RANSAC,
                ransacReprojThreshold=float(FLAGS.ransac_reproj_threshold),
            )
            if refined_matrix is not None:
                matrix = refined_matrix
                src = src_kept
                dst = dst_kept
                inliers = refined_inliers
                mapped = cv2.perspectiveTransform(src.reshape(-1, 1, 2), matrix).reshape(-1, 2)
                errors = np.linalg.norm(mapped - dst, axis=-1)
        elif kept < len(labels):
            print(
                f"[fit][warn] scope={scope_name} only {kept}/{len(labels)} labels below "
                f"max_label_error_px={FLAGS.max_label_error_px}; keeping original fit."
            )

    inlier_count = int(inliers.sum()) if inliers is not None else len(labels)
    dst_size = tuple(labels[0].get("rs_size", [640, 480]))
    print(
        f"[fit] scope={scope_name} labels={len(labels)} inliers={inlier_count} "
        f"mean_px={errors.mean():.2f} median_px={np.median(errors):.2f} max_px={errors.max():.2f}"
    )
    return GazeHomography(
        matrix=matrix.astype(np.float32),
        src_points=src[:4].astype(np.float32),
        dst_points=dst[:4].astype(np.float32),
        dst_size=(int(dst_size[0]), int(dst_size[1])),
        fit_label_count=len(labels),
        fit_inlier_count=inlier_count,
        fit_mean_error_px=float(errors.mean()),
        fit_median_error_px=float(np.median(errors)),
        fit_max_error_px=float(errors.max()),
    )


def _fit_global_homography(labels: list[dict]):
    homography = _fit_one_homography("global", labels)
    if homography is None:
        raise ValueError("No global homography was fitted. Need at least 4 labelled points.")
    return homography


def _fit_episode_homographies(labels: list[dict]):
    grouped = defaultdict(list)
    for label in labels:
        grouped[int(label["episode_index"])].append(label)

    episode_homographies = {}
    for episode_index in sorted(grouped):
        homography = _fit_one_homography(f"episode={episode_index}", grouped[episode_index])
        if homography is not None:
            episode_homographies[int(episode_index)] = homography
    if not episode_homographies:
        raise ValueError("No episode homographies were fitted. Need at least 4 labelled points per episode.")
    return EpisodeHomographyMap(episodes=episode_homographies)


def main(_):
    if FLAGS.labels_in:
        labels = json.loads(Path(FLAGS.labels_in).expanduser().read_text())
    else:
        if FLAGS.metadata is None:
            raise ValueError("Please pass --metadata=/path/to/recording_metadata.json")
        metadata_path = Path(FLAGS.metadata).expanduser().resolve()
        metadata, frame_root = _load_metadata(metadata_path)
        labels = _collect_labels(metadata, frame_root)
        Path(FLAGS.labels_out).expanduser().write_text(json.dumps(labels, indent=2, ensure_ascii=False))
        print(f"[labels] saved {len(labels)} labels to {FLAGS.labels_out}")

    fit_scope = FLAGS.fit_scope.strip().lower()
    if fit_scope == "global":
        homography = _fit_global_homography(labels)
        homography.save(FLAGS.out)
        print(f"[homography] saved global homography to {FLAGS.out}")
    elif fit_scope == "per_episode":
        episode_map = _fit_episode_homographies(labels)
        episode_map.save(FLAGS.out)
        print(f"[homography] saved {len(episode_map.episodes)} episode homographies to {FLAGS.out}")
    else:
        raise ValueError("--fit_scope must be one of: global, per_episode")


if __name__ == "__main__":
    app.run(main)
