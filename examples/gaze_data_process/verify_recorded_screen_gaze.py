#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from absl import app, flags


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "frame_root",
    "/media/user/data3/wrq/recorded_data/tennis_ball_pick/tennis_ball_pick-6-23-1",
    "Recorded frame root that contains frame_*/, et_images/, and recording_metadata.json.",
)
flags.DEFINE_integer("start_frame", 0, "First frame id to inspect.")
flags.DEFINE_integer("stride", 1, "Frame stride when pressing next.")
flags.DEFINE_integer("display_width", 640, "Width of each panel in the verification window.")
flags.DEFINE_boolean("save", False, "Save rendered verification images instead of only displaying them.")
flags.DEFINE_string("out_dir", "./gaze_screen_mapping_verify", "Output directory used when --save=true.")


def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _point_from_contact(contact: dict, key: str):
    value = contact.get(key)
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except Exception:
        return None


def _draw_cross(image, point, color, label: str):
    if point is None:
        return image
    vis = image.copy()
    x = int(round(point[0]))
    y = int(round(point[1]))
    h, w = vis.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return vis
    cv2.line(vis, (x - 18, y), (x + 18, y), color, 3, cv2.LINE_AA)
    cv2.line(vis, (x, y - 18), (x, y + 18), color, 3, cv2.LINE_AA)
    cv2.circle(vis, (x, y), 5, color, -1, cv2.LINE_AA)
    cv2.putText(
        vis,
        f"{label}=({x},{y})",
        (max(5, x + 12), max(24, y - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )
    return vis


def _draw_markers(image, points, color=(255, 255, 255)):
    if not isinstance(points, list):
        return image
    vis = image.copy()
    for idx, point in enumerate(points):
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        x, y = int(round(float(point[0]))), int(round(float(point[1])))
        cv2.circle(vis, (x, y), 9, color, 2, cv2.LINE_AA)
        cv2.putText(
            vis,
            str(idx),
            (x + 10, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return vis


def _resize_panel(image, width: int):
    h, w = image.shape[:2]
    if w == width:
        return image
    scale = float(width) / max(1, w)
    return cv2.resize(image, (width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_LINEAR)


def _label_panel(image, title: str):
    vis = image.copy()
    cv2.rectangle(vis, (0, 0), (min(vis.shape[1], 520), 34), (0, 0, 0), -1)
    cv2.putText(vis, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def _add_footer(canvas, frame_id: int, index: int | None = None, total: int | None = None):
    footer_h = 42
    footer = np.zeros((footer_h, canvas.shape[1], 3), dtype=np.uint8)
    progress = f"frame={frame_id}"
    if index is not None and total is not None:
        progress = f"{progress}  sample={index + 1}/{total}"
    help_text = "keys: n/space=next   p=prev   s=save current   q/esc=quit"
    cv2.putText(footer, progress, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(footer, help_text, (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([canvas, footer])


def _make_canvas(frame_root: Path, frame_id: int, index: int | None = None, total: int | None = None):
    frame_dir = frame_root / f"frame_{frame_id}"
    contact = _load_json(frame_dir / "gaze_contact.json")
    if contact is None:
        return None, "missing gaze_contact.json"

    et_img = cv2.imread(str(frame_root / "et_images" / f"{frame_id}.jpg"))
    rs_img = cv2.imread(str(frame_dir / "color_image.jpg"))
    if et_img is None:
        return None, "missing et image"
    if rs_img is None:
        return None, "missing realsense image"

    eye_xy = _point_from_contact(contact, "gaze_uv_in_eye")
    rs_xy = _point_from_contact(contact, "gaze_uv_in_realsense")
    if rs_xy is not None:
        rs_size = contact.get("realsense_size")
        if isinstance(rs_size, (list, tuple)) and len(rs_size) >= 2:
            rs_w, rs_h = float(rs_size[0]), float(rs_size[1])
            h, w = rs_img.shape[:2]
            rs_xy = (
                rs_xy[0] * max(1.0, w - 1) / max(1.0, rs_w - 1),
                rs_xy[1] * max(1.0, h - 1) / max(1.0, rs_h - 1),
            )

    et_vis = _draw_markers(et_img, contact.get("marker_points_eye"), color=(255, 255, 255))
    et_vis = _draw_cross(et_vis, eye_xy, (0, 0, 255), "eye_gaze")
    rs_vis = _draw_markers(rs_img, contact.get("marker_points_realsense"), color=(255, 255, 255))
    rs_vis = _draw_cross(rs_vis, rs_xy, (0, 0, 255), "rs_gaze")

    et_panel = _label_panel(_resize_panel(et_vis, FLAGS.display_width), f"eye tracker frame={frame_id}")
    rs_panel = _label_panel(_resize_panel(rs_vis, FLAGS.display_width), "realsense mapped gaze")

    target_h = max(et_panel.shape[0], rs_panel.shape[0])
    panels = []
    for panel in (et_panel, rs_panel):
        if panel.shape[0] < target_h:
            pad = np.zeros((target_h - panel.shape[0], panel.shape[1], 3), dtype=np.uint8)
            panel = np.vstack([panel, pad])
        panels.append(panel)
    return _add_footer(np.hstack(panels), frame_id, index=index, total=total), None


def _available_frame_ids(frame_root: Path):
    ids = []
    for path in frame_root.glob("frame_*/gaze_contact.json"):
        try:
            ids.append(int(path.parent.name.split("_", 1)[1]))
        except Exception:
            continue
    return sorted(ids)


def main(_):
    frame_root = Path(FLAGS.frame_root).expanduser().resolve()
    frame_ids = [frame_id for frame_id in _available_frame_ids(frame_root) if frame_id >= FLAGS.start_frame]
    if not frame_ids:
        print(f"[verify] no frames with gaze_contact.json found under {frame_root}")
        return

    out_dir = Path(FLAGS.out_dir).expanduser().resolve()
    if FLAGS.save:
        out_dir.mkdir(parents=True, exist_ok=True)

    index = 0
    print("[verify] keys: n/space=next, p=prev, s=save current, q/esc=quit")
    while 0 <= index < len(frame_ids):
        frame_id = frame_ids[index]
        canvas, error = _make_canvas(frame_root, frame_id, index=index, total=len(frame_ids))
        if canvas is None:
            print(f"[verify] frame={frame_id} skipped: {error}")
            index += max(1, FLAGS.stride)
            continue

        if FLAGS.save:
            out_path = out_dir / f"verify_{frame_id:06d}.jpg"
            cv2.imwrite(str(out_path), canvas)
            print(f"[verify] wrote {out_path}")
            index += max(1, FLAGS.stride)
            continue

        cv2.imshow("screen gaze mapping verify", canvas)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("p"):
            index = max(0, index - max(1, FLAGS.stride))
        elif key == ord("s"):
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"verify_{frame_id:06d}.jpg"
            cv2.imwrite(str(out_path), canvas)
            print(f"[verify] wrote {out_path}")
        else:
            index += max(1, FLAGS.stride)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    app.run(main)
