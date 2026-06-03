import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from absl import app, flags


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "root",
    "/media/user/data3/wrq/recorded_data/tennis_ball_pick/tennis_ball_pick-5-27-0",
    "Recording root containing frame_* folders and recording_metadata.json.",
)
flags.DEFINE_string(
    "metadata",
    None,
    "Optional path to recording_metadata.json. Defaults to <root>/recording_metadata.json.",
)
flags.DEFINE_integer("start_frame", None, "Optional frame id to start browsing from.")
flags.DEFINE_integer("max_width", 1600, "Max display width.")
flags.DEFINE_integer("max_height", 900, "Max display height.")
flags.DEFINE_boolean(
    "move_frames",
    True,
    "Move rejected frame_* folders to _filtered_out_frames when saving.",
)
flags.DEFINE_boolean(
    "overwrite_metadata",
    True,
    "Backup and replace recording_metadata.json with filtered frame metadata.",
)


def _frame_id(frame_dir: Path) -> int:
    return int(frame_dir.name.split("_", 1)[1])


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


def _read_gaze_uv(frame_dir: Path, image_shape):
    path = frame_dir / "pupil_gaze.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        payload = data.get("data", data)
        norm_pos = payload.get("norm_pos")
        if norm_pos is None or len(norm_pos) < 2:
            return None
        h, w = image_shape[:2]
        x_norm, y_norm = float(norm_pos[0]), float(norm_pos[1])
        u = int(round(x_norm * (w - 1)))
        v = int(round((1.0 - y_norm) * (h - 1)))
        if not (0 <= u < w and 0 <= v < h):
            return None
        return u, v
    except Exception:
        return None


def _draw_gaze(image, uv):
    if uv is None:
        return image
    out = image.copy()
    u, v = uv
    cv2.circle(out, (u, v), 14, (0, 0, 255), 3)
    cv2.circle(out, (u, v), 4, (0, 255, 255), -1)
    cv2.line(out, (u - 22, v), (u + 22, v), (0, 0, 255), 2)
    cv2.line(out, (u, v - 22), (u, v + 22), (0, 0, 255), 2)
    cv2.putText(
        out,
        f"gaze ({u},{v})",
        (max(0, u + 12), max(24, v - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    return out


def _load_et_view(root: Path, frame_dir: Path, fid: int):
    gaze_img = root / "et_images_gaze" / f"{fid}.jpg"
    if gaze_img.exists():
        img = cv2.imread(str(gaze_img))
        if img is not None:
            return img

    et_img = root / "et_images" / f"{fid}.jpg"
    if et_img.exists():
        img = cv2.imread(str(et_img))
        if img is not None:
            return _draw_gaze(img, _read_gaze_uv(frame_dir, img.shape))
    return None


def _load_rs_view(frame_dir: Path):
    img = cv2.imread(str(frame_dir / "color_image.jpg"))
    return img


def _fit_image(image, max_width, max_height):
    h, w = image.shape[:2]
    scale = min(max_width / float(w), max_height / float(h), 1.0)
    if scale >= 0.999:
        return image
    return cv2.resize(image, (int(round(w * scale)), int(round(h * scale))))


def _draw_header(image, lines, rejected):
    out = image.copy()
    color = (0, 0, 255) if rejected else (0, 220, 0)
    pad = 8
    line_h = 26
    max_w = 0
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        max_w = max(max_w, tw)
    cv2.rectangle(out, (0, 0), (max_w + 2 * pad, line_h * len(lines) + pad), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(
            out,
            line,
            (pad, 22 + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color if i == 0 else (255, 255, 0),
            2,
        )
    return out


def _make_preview(root: Path, frame_dir: Path, idx, total, rejected, episode):
    fid = _frame_id(frame_dir)
    et = _load_et_view(root, frame_dir, fid)
    rs = _load_rs_view(frame_dir)
    panels = []
    if et is not None:
        panels.append(("ET gaze", et))
    if rs is not None:
        panels.append(("Robot RGB", rs))
    if not panels:
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(canvas, "No image for this frame", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        panels.append(("missing", canvas))

    target_h = min(max(p[1].shape[0] for p in panels), 720)
    resized = []
    for title, img in panels:
        scale = target_h / float(img.shape[0])
        panel = cv2.resize(img, (int(round(img.shape[1] * scale)), target_h))
        cv2.putText(panel, title, (10, target_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        resized.append(panel)
    canvas = cv2.hconcat(resized) if len(resized) > 1 else resized[0]
    ep_text = "episode=?"
    if episode is not None:
        ep_text = f"episode={episode.get('episode_index', '?')} success={episode.get('success', '?')}"
    lines = [
        f"{'REJECT' if rejected else 'KEEP'} frame={fid} ({idx + 1}/{total}) {ep_text}",
        "keys: k/Right keep next | x reject next | r toggle reject | j/Left prev",
        "range: b set start | e reject start..current | s save | q save quit | Esc quit no save",
    ]
    canvas = _draw_header(canvas, lines, rejected)
    return _fit_image(canvas, FLAGS.max_width, FLAGS.max_height)


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


def _metadata_base_episode_ranges(metadata):
    ranges = metadata.get("episode_ranges_original") or metadata.get("episode_ranges", [])
    return [dict(rec) for rec in ranges]


def _range_records(frame_ids):
    return [
        {
            "start_frame": int(start),
            "end_frame": int(end),
            "num_frames": int(end - start + 1),
        }
        for start, end in _contiguous_ranges(frame_ids)
    ]


def _filtered_metadata(metadata, kept_ids, rejected_ids):
    kept_ids = set(map(int, kept_ids))
    rejected_ids = set(map(int, rejected_ids))
    original_ranges = _metadata_base_episode_ranges(metadata)
    new_ranges = []

    for default_ep_idx, source in enumerate(original_ranges):
        start = int(source["start_frame"])
        end = int(source["end_frame"])
        if end < start:
            start, end = end, start
        all_ids = range(start, end + 1)
        kept_in_episode = [fid for fid in all_ids if fid in kept_ids]
        rejected_in_episode = [fid for fid in range(start, end + 1) if fid in rejected_ids]

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
    out["filtered_out_frames"] = sorted(map(int, rejected_ids))
    out["num_filtered_out_frames"] = len(rejected_ids)
    out["num_kept_frames"] = len(kept_ids)
    return out


def _save_filter(root: Path, metadata_path: Path, metadata, frame_dirs, rejected_ids):
    rejected_ids = set(map(int, rejected_ids))
    kept_ids = {_frame_id(p) for p in frame_dirs if _frame_id(p) not in rejected_ids}
    if FLAGS.overwrite_metadata:
        backup = metadata_path.with_suffix(metadata_path.suffix + ".pre_filter_backup")
        if metadata_path.exists() and not backup.exists():
            shutil.copy2(metadata_path, backup)
            print(f"[filter] metadata backup: {backup}")
        filtered = _filtered_metadata(metadata, kept_ids, rejected_ids)
        metadata_path.write_text(json.dumps(filtered, indent=2, ensure_ascii=False))
        print(f"[filter] wrote filtered metadata: {metadata_path}")

    if FLAGS.move_frames and rejected_ids:
        out_dir = root / "_filtered_out_frames"
        out_dir.mkdir(exist_ok=True)
        moved = 0
        moved_mirror = 0
        for frame_dir in frame_dirs:
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
        print(f"[filter] moved {moved} frame folders to {out_dir}")
        print(f"[filter] moved {moved_mirror} mirror images to {out_dir}/<mirror_dir>/")
    print(f"[filter] kept={len(kept_ids)} rejected={len(rejected_ids)}")


def main(_):
    root = Path(FLAGS.root).expanduser().resolve()
    metadata_path = Path(FLAGS.metadata).expanduser().resolve() if FLAGS.metadata else root / "recording_metadata.json"
    metadata = _load_metadata(root, metadata_path)
    frame_dirs = sorted([p for p in root.glob("frame_*") if p.is_dir()], key=_frame_id)
    if not frame_dirs:
        raise FileNotFoundError(f"No frame_* folders under {root}")

    by_frame = _ranges_to_episode_by_frame(metadata)
    rejected = set(map(int, metadata.get("filtered_out_frames", [])))
    start_idx = 0
    if FLAGS.start_frame is not None:
        ids = [_frame_id(p) for p in frame_dirs]
        start_idx = min(range(len(ids)), key=lambda i: abs(ids[i] - FLAGS.start_frame))

    idx = start_idx
    range_start = None
    dirty = False
    win = "Filter recorded gaze frames"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    while 0 <= idx < len(frame_dirs):
        frame_dir = frame_dirs[idx]
        fid = _frame_id(frame_dir)
        preview = _make_preview(root, frame_dir, idx, len(frame_dirs), fid in rejected, by_frame.get(fid))
        cv2.imshow(win, preview)
        key = cv2.waitKey(0) & 0xFF

        if key in (27,):
            print("[filter] quit without saving")
            break
        if key == ord("q"):
            _save_filter(root, metadata_path, metadata, frame_dirs, rejected)
            dirty = False
            break
        if key == ord("s"):
            _save_filter(root, metadata_path, metadata, frame_dirs, rejected)
            dirty = False
        elif key in (ord("k"), ord(" "), 83):
            idx = min(idx + 1, len(frame_dirs) - 1)
        elif key in (ord("j"), 81):
            idx = max(idx - 1, 0)
        elif key == ord("x"):
            rejected.add(fid)
            dirty = True
            idx = min(idx + 1, len(frame_dirs) - 1)
        elif key == ord("r"):
            if fid in rejected:
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
                for p in frame_dirs:
                    pfid = _frame_id(p)
                    if lo <= pfid <= hi:
                        rejected.add(pfid)
                print(f"[filter] rejected range {lo}..{hi}")
                range_start = None
                dirty = True

    cv2.destroyWindow(win)
    if dirty:
        print("[filter] unsaved changes remain. Press q or s next time to save them.")


if __name__ == "__main__":
    app.run(main)
