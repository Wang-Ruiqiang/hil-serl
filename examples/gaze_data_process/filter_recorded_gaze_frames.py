#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from absl import app, flags


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "root",
    "/media/user/data3/wrq/recorded_data/tennis_ball_pick/tennis_ball_pick-6-23-1",
    "Recording root containing frame_* folders and recording_metadata.json.",
)
flags.DEFINE_string(
    "metadata",
    None,
    "Optional path to recording_metadata.json. Defaults to <root>/recording_metadata.json.",
)
flags.DEFINE_integer("start_frame", None, "Optional frame id to start browsing from.")
flags.DEFINE_integer("max_width", 1800, "Max display width.")
flags.DEFINE_integer("max_height", 1000, "Max display height.")
flags.DEFINE_boolean("move_frames", True, "Move rejected frame_* folders to _filtered_out_frames when saving.")
flags.DEFINE_boolean("overwrite_metadata", True, "Backup and replace recording_metadata.json with filtered frame metadata.")


def _frame_id(frame_dir: Path) -> int:
    return int(frame_dir.name.split("_", 1)[1])


def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _load_metadata(root: Path, metadata_path: Path):
    if metadata_path.exists():
        return json.loads(metadata_path.read_text())
    frame_ids = sorted(_frame_id(p) for p in root.glob("frame_*") if p.is_dir())
    if not frame_ids:
        raise FileNotFoundError(f"No frame_* folders under {root}")
    return {
        "frame_root": str(root),
        "episode_ranges": [
            {
                "episode_index": 0,
                "start_frame": frame_ids[0],
                "end_frame": frame_ids[-1],
                "success": True,
                "num_frames": len(frame_ids),
            }
        ],
    }


def _metadata_base_episode_ranges(metadata):
    ranges = metadata.get("episode_ranges_original") or metadata.get("episode_ranges", [])
    return [dict(rec) for rec in ranges]


def _ranges_to_episode_by_frame(metadata):
    by_frame = {}
    for rec in _metadata_base_episode_ranges(metadata):
        start = int(rec["start_frame"])
        end = int(rec["end_frame"])
        if end < start:
            start, end = end, start
        for fid in range(start, end + 1):
            by_frame[fid] = rec
    return by_frame


def _point(contact: dict, key: str):
    value = contact.get(key)
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except Exception:
        return None


def _load_contact(frame_dir: Path):
    contact = _load_json(frame_dir / "gaze_contact.json")
    if not isinstance(contact, dict):
        return None
    if _point(contact, "gaze_uv_in_realsense") is None:
        return None
    return contact


def _valid_frame_dirs(frame_dirs: list[Path]):
    valid = []
    invalid_ids = set()
    for frame_dir in frame_dirs:
        fid = _frame_id(frame_dir)
        if _load_contact(frame_dir) is None:
            invalid_ids.add(fid)
        else:
            valid.append(frame_dir)
    return valid, invalid_ids


def _draw_cross(image, point, color, label: str):
    if point is None:
        return image
    out = image.copy()
    x, y = int(round(point[0])), int(round(point[1]))
    h, w = out.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return out
    cv2.line(out, (x - 22, y), (x + 22, y), color, 3, cv2.LINE_AA)
    cv2.line(out, (x, y - 22), (x, y + 22), color, 3, cv2.LINE_AA)
    cv2.circle(out, (x, y), 5, color, -1, cv2.LINE_AA)
    cv2.putText(
        out,
        f"{label}=({x},{y})",
        (max(5, x + 14), max(28, y - 14)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )
    return out


def _draw_markers(image, points, color=(255, 255, 255)):
    if not isinstance(points, list):
        return image
    out = image.copy()
    for idx, point in enumerate(points):
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        x, y = int(round(float(point[0]))), int(round(float(point[1])))
        cv2.circle(out, (x, y), 10, color, 2, cv2.LINE_AA)
        cv2.putText(out, str(idx), (x + 12, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return out


def _resize_panel(image, width: int):
    h, w = image.shape[:2]
    if w == width:
        return image
    scale = float(width) / max(1, w)
    return cv2.resize(image, (width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_LINEAR)


def _label_panel(image, title: str):
    out = image.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 620), 38), (0, 0, 0), -1)
    cv2.putText(out, title, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _fit_canvas(image):
    h, w = image.shape[:2]
    scale = min(FLAGS.max_width / float(w), FLAGS.max_height / float(h), 1.0)
    if scale >= 0.999:
        return image
    return cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_LINEAR)


def _make_preview(root: Path, frame_dir: Path, idx: int, total: int, rejected: bool, episode):
    fid = _frame_id(frame_dir)
    contact = _load_contact(frame_dir)
    if contact is None:
        canvas = np.zeros((480, 960, 3), dtype=np.uint8)
        cv2.putText(canvas, f"frame={fid} has no valid gaze_contact.json", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return canvas

    et_img = cv2.imread(str(root / "et_images" / f"{fid}.jpg"))
    rs_img = cv2.imread(str(frame_dir / "color_image.jpg"))
    if et_img is None:
        et_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(et_img, "missing eye tracker image", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    if rs_img is None:
        rs_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(rs_img, "missing realsense image", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    eye_xy = _point(contact, "gaze_uv_in_eye")
    rs_xy = _point(contact, "gaze_uv_in_realsense")
    rs_size = contact.get("realsense_size")
    if rs_xy is not None and isinstance(rs_size, (list, tuple)) and len(rs_size) >= 2:
        src_w, src_h = float(rs_size[0]), float(rs_size[1])
        h, w = rs_img.shape[:2]
        rs_xy = (
            rs_xy[0] * max(1.0, w - 1) / max(1.0, src_w - 1),
            rs_xy[1] * max(1.0, h - 1) / max(1.0, src_h - 1),
        )

    et_vis = _draw_markers(et_img, contact.get("marker_points_eye"))
    et_vis = _draw_cross(et_vis, eye_xy, (0, 0, 255), "eye_gaze")
    rs_vis = _draw_markers(rs_img, contact.get("marker_points_realsense"))
    rs_vis = _draw_cross(rs_vis, rs_xy, (0, 0, 255), "rs_gaze")

    et_panel = _label_panel(_resize_panel(et_vis, 720), f"eye tracker frame={fid}")
    rs_panel = _label_panel(_resize_panel(rs_vis, 720), "realsense mapped gaze")
    target_h = max(et_panel.shape[0], rs_panel.shape[0])
    panels = []
    for panel in (et_panel, rs_panel):
        if panel.shape[0] < target_h:
            pad = np.zeros((target_h - panel.shape[0], panel.shape[1], 3), dtype=np.uint8)
            panel = np.vstack([panel, pad])
        panels.append(panel)
    canvas = np.hstack(panels)

    footer_h = 74
    footer = np.zeros((footer_h, canvas.shape[1], 3), dtype=np.uint8)
    status_color = (0, 0, 255) if rejected else (0, 220, 0)
    ep_text = "episode=?"
    if episode is not None:
        ep_text = f"episode={episode.get('episode_index', '?')} success={episode.get('success', '?')}"
    cv2.putText(
        footer,
        f"{'REJECT' if rejected else 'KEEP'} frame={fid} sample={idx + 1}/{total} {ep_text}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        status_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        footer,
        "keys: k/Right/space keep next | x reject next | r toggle reject | j/Left prev",
        (8, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        footer,
        "range: b set start | e reject start..current | s save | q save quit | Esc quit no save",
        (8, 67),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        1,
        cv2.LINE_AA,
    )
    return _fit_canvas(np.vstack([canvas, footer]))


def _contiguous_ranges(frame_ids):
    if not frame_ids:
        return []
    frame_ids = sorted(frame_ids)
    ranges = []
    start = prev = frame_ids[0]
    for fid in frame_ids[1:]:
        if fid == prev + 1:
            prev = fid
            continue
        ranges.append((start, prev))
        start = prev = fid
    ranges.append((start, prev))
    return ranges


def _range_records(frame_ids):
    return [
        {"start_frame": int(start), "end_frame": int(end), "num_frames": int(end - start + 1)}
        for start, end in _contiguous_ranges(frame_ids)
    ]


def _filtered_metadata(metadata, kept_ids, rejected_ids, auto_rejected_ids):
    kept_ids = set(map(int, kept_ids))
    rejected_ids = set(map(int, rejected_ids))
    auto_rejected_ids = set(map(int, auto_rejected_ids))
    original_ranges = _metadata_base_episode_ranges(metadata)
    new_ranges = []

    for default_ep_idx, source in enumerate(original_ranges):
        start = int(source["start_frame"])
        end = int(source["end_frame"])
        if end < start:
            start, end = end, start
        all_ids = range(start, end + 1)
        kept_in_episode = [fid for fid in all_ids if fid in kept_ids]
        rejected_in_episode = [fid for fid in all_ids if fid in rejected_ids]

        rec = dict(source)
        rec["episode_index"] = int(source.get("episode_index", default_ep_idx))
        rec["start_frame"] = int(start)
        rec["end_frame"] = int(end)
        rec["success"] = bool(source.get("success", True))
        rec["num_frames"] = len(kept_in_episode)
        rec["kept_frame_ranges"] = _range_records(kept_in_episode)
        rec["filtered_out_frame_ranges"] = _range_records(rejected_in_episode)
        new_ranges.append(rec)

    out = dict(metadata)
    out["episode_ranges_original"] = original_ranges
    out["episode_ranges"] = new_ranges
    out["num_episodes"] = len(new_ranges)
    out["filtered_out_frames"] = sorted(rejected_ids)
    out["auto_filtered_missing_realsense_gaze_frames"] = sorted(auto_rejected_ids)
    out["num_auto_filtered_missing_realsense_gaze_frames"] = len(auto_rejected_ids)
    out["num_filtered_out_frames"] = len(rejected_ids)
    out["num_kept_frames"] = len(kept_ids)
    return out


def _save_filter(root: Path, metadata_path: Path, metadata, all_frame_dirs, valid_frame_dirs, rejected_ids, auto_rejected_ids):
    rejected_ids = set(map(int, rejected_ids))
    valid_ids = {_frame_id(p) for p in valid_frame_dirs}
    kept_ids = {fid for fid in valid_ids if fid not in rejected_ids}

    if FLAGS.overwrite_metadata:
        backup = metadata_path.with_suffix(metadata_path.suffix + ".pre_filter_backup")
        if metadata_path.exists() and not backup.exists():
            shutil.copy2(metadata_path, backup)
            print(f"[filter] metadata backup: {backup}")
        filtered = _filtered_metadata(metadata, kept_ids, rejected_ids, auto_rejected_ids)
        metadata_path.write_text(json.dumps(filtered, indent=2, ensure_ascii=False))
        print(f"[filter] wrote filtered metadata: {metadata_path}")

    if FLAGS.move_frames and rejected_ids:
        out_dir = root / "_filtered_out_frames"
        out_dir.mkdir(exist_ok=True)
        moved = 0
        moved_mirror = 0
        for frame_dir in all_frame_dirs:
            fid = _frame_id(frame_dir)
            if fid not in rejected_ids:
                continue
            if frame_dir.exists():
                dst = out_dir / frame_dir.name
                if not dst.exists():
                    shutil.move(str(frame_dir), str(dst))
                    moved += 1

            for mirror_name in ("et_images_gaze", "et_images", "rs_images"):
                src = root / mirror_name / f"{fid}.jpg"
                if not src.exists():
                    continue
                mirror_out = out_dir / mirror_name
                mirror_out.mkdir(exist_ok=True)
                dst = mirror_out / src.name
                if dst.exists():
                    continue
                shutil.move(str(src), str(dst))
                moved_mirror += 1
        print(f"[filter] moved {moved} rejected frame folders to {out_dir}")
        print(f"[filter] moved {moved_mirror} mirror images to {out_dir}/<mirror_dir>/")

    print(
        f"[filter] kept={len(kept_ids)} manual_rejected={len(rejected_ids - auto_rejected_ids)} "
        f"auto_missing_gaze={len(auto_rejected_ids)} total_rejected={len(rejected_ids)}"
    )


def main(_):
    root = Path(FLAGS.root).expanduser().resolve()
    metadata_path = Path(FLAGS.metadata).expanduser().resolve() if FLAGS.metadata else root / "recording_metadata.json"
    metadata = _load_metadata(root, metadata_path)
    all_frame_dirs = sorted([p for p in root.glob("frame_*") if p.is_dir()], key=_frame_id)
    if not all_frame_dirs:
        raise FileNotFoundError(f"No frame_* folders under {root}")

    valid_frame_dirs, auto_rejected = _valid_frame_dirs(all_frame_dirs)
    if not valid_frame_dirs:
        raise ValueError(f"No frames with valid gaze_uv_in_realsense found under {root}")

    by_frame = _ranges_to_episode_by_frame(metadata)
    rejected = set(map(int, metadata.get("filtered_out_frames", [])))
    rejected.update(auto_rejected)

    start_idx = 0
    if FLAGS.start_frame is not None:
        ids = [_frame_id(p) for p in valid_frame_dirs]
        start_idx = min(range(len(ids)), key=lambda i: abs(ids[i] - FLAGS.start_frame))

    print(
        f"[filter] total_frames={len(all_frame_dirs)} valid_gaze_frames={len(valid_frame_dirs)} "
        f"auto_missing_gaze={len(auto_rejected)}"
    )
    print("[filter] browse only frames with gaze_uv_in_realsense; missing-gaze frames are auto rejected.")

    idx = start_idx
    range_start = None
    dirty = False
    win = "Filter recorded screen gaze frames"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    while 0 <= idx < len(valid_frame_dirs):
        frame_dir = valid_frame_dirs[idx]
        fid = _frame_id(frame_dir)
        preview = _make_preview(root, frame_dir, idx, len(valid_frame_dirs), fid in rejected, by_frame.get(fid))
        cv2.imshow(win, preview)
        key = cv2.waitKey(0) & 0xFF

        if key == 27:
            print("[filter] quit without saving")
            break
        if key == ord("q"):
            _save_filter(root, metadata_path, metadata, all_frame_dirs, valid_frame_dirs, rejected, auto_rejected)
            dirty = False
            break
        if key == ord("s"):
            _save_filter(root, metadata_path, metadata, all_frame_dirs, valid_frame_dirs, rejected, auto_rejected)
            dirty = False
        elif key in (ord("k"), ord(" "), 83):
            idx = min(idx + 1, len(valid_frame_dirs) - 1)
        elif key in (ord("j"), 81):
            idx = max(idx - 1, 0)
        elif key == ord("x"):
            rejected.add(fid)
            dirty = True
            idx = min(idx + 1, len(valid_frame_dirs) - 1)
        elif key == ord("r"):
            if fid in auto_rejected:
                print(f"[filter] frame {fid} is auto-rejected because it has no valid RealSense gaze.")
            elif fid in rejected:
                rejected.remove(fid)
            else:
                rejected.add(fid)
            dirty = True
        elif key == ord("b"):
            range_start = fid
            print(f"[filter] range start set to frame {range_start}")
        elif key == ord("e"):
            if range_start is None:
                print("[filter] press b first to set a range start")
            else:
                lo, hi = sorted((range_start, fid))
                for p in valid_frame_dirs:
                    pfid = _frame_id(p)
                    if lo <= pfid <= hi:
                        rejected.add(pfid)
                print(f"[filter] rejected valid-gaze range {lo}..{hi}")
                range_start = None
                dirty = True

    cv2.destroyWindow(win)
    if dirty:
        print("[filter] unsaved changes remain. Press q or s next time to save them.")


if __name__ == "__main__":
    app.run(main)
