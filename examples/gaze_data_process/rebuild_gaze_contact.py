#!/usr/bin/env python3
"""Recover gaze_contact.json for frames the recorder failed to write.

EpisodeDataRecorder writes nothing when it cannot build the eye->screen
mapping, and `hit` is hardcoded True, so a *missing file* is the only failure
mode -- no frame in these recordings has hit=false.

The dominant cause is not the operator looking away. Of the frames that failed
marker detection, 84.6% had three of the four ArUco markers detected and only
1.2% had none: the screen was in view the whole time, one corner just did not
decode on that frame. `detect_gaze_display_markers` requires all four and
returns None otherwise, so a single missed corner discards the frame.

That is recoverable, because each detected marker carries four corners, not
one. Three markers give twelve point correspondences where a homography needs
four. This script therefore rebuilds the mapping itself instead of calling
detect_gaze_display_markers, and only falls back to the recorder's behaviour
when too few markers are present.

Recovery, in order of preference:

  markers4 / markers3 / markers2
      Homography from the corners of however many markers decoded. Validated
      against frames where all four decoded, by dropping some and comparing:
          3 markers  median 1.80 px error, p90 5.88   (ball is ~37 px wide)
          2 markers  median 6.47 px error, p90 25.92
      Two markers is noticeably worse and is off by default.

  interpolated
      No usable markers here, but a frame shortly before and after has them.
      The screen is rigid and the head moves slowly at 10 Hz, so the four
      marker corners are interpolated linearly across the gap and the
      homography rebuilt from them. Validated the same way:
          1-frame gap  median 1.30 px      5-frame  3.39      9-frame  5.39
      Corners are interpolated rather than the homography matrix, whose
      entries are projective and do not interpolate meaningfully.

Not recoverable: a gaze whose pupil norm_pos lies outside [0, 1] left the eye
camera altogether, so there is no point to project.

Frames already moved into _filtered_out_frames are searched too, and anything
recovered is moved back beside the kept frames (with its et_images /
et_images_gaze / rs_images mirrors) so the labelling tool can see it.

Every rebuilt file records how it was made -- `source`, `n_markers`,
`marker_ids`, and for interpolation `rebuilt_from` / `interp_gap` /
`marker_drift_px` -- so a suspect frame can be judged and dropped by hand.

Dry run by default; --apply writes and moves.

    python examples/gaze_data_process/rebuild_gaze_contact.py
    python examples/gaze_data_process/rebuild_gaze_contact.py --apply
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from absl import app, flags

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "serl_robot_infra"))
from franka_env.gaze.display_markers import (  # noqa: E402
    ARUCO_DICT_ID,
    MARKER_IDS,
    marker_points_for_size,
)

# ============================== 编辑这里 ==============================
ROOTS = [
    "/home/ealin/workspaces/DexTacHil/data/recorded_data/tennis_ball_pick_and_place/"
    "tennis_ball_pick_and_place-2026-08-14_12-18-59",
    "/home/ealin/workspaces/DexTacHil/data/recorded_data/tennis_ball_pick_and_place/"
    "tennis_ball_pick_and_place-2026-08-14_12-49-48",
]
# =====================================================================

FLAGS = flags.FLAGS
flags.DEFINE_string("root", None, "Override ROOTS with a single recording directory.")
flags.DEFINE_boolean("apply", False, "Write files and move frames back. Otherwise report only.")
flags.DEFINE_integer("min_markers", 3,
                     "Fewest decoded markers accepted for a corner homography. "
                     "3 costs ~1.8px; 2 costs ~6.5px and is a last resort.")
flags.DEFINE_integer("max_interp", 10,
                     "Longest marker-less run to bridge by interpolating corners. 0 disables.")
flags.DEFINE_float("max_drift", 60.0,
                   "Skip an interpolation whose bracketing marker sets differ by more than "
                   "this (px in the eye image) -- the head moved too fast to interpolate.")
flags.DEFINE_boolean("restore_metadata", True,
                     "Restore recording_metadata.json from .pre_filter_backup so the "
                     "recovered frames are visible to the labelling tool again.")

FILTERED_DIR = "_filtered_out_frames"
MIRRORS = ("et_images", "et_images_gaze", "rs_images")


# ------------------------------------------------------------- geometry

def marker_corner_reference(root: Path):
    """Where each marker's four corners sit in the RealSense frame.

    draw_gaze_display_markers centres a marker_size square on each point from
    marker_points_for_size, so the corners are the centre plus/minus half that
    size, in the aruco corner order (TL, TR, BR, BL).
    """
    meta_path = root / "recording_metadata.json"
    backup = root / "recording_metadata.json.pre_filter_backup"
    size = [640, 480]
    centres = None
    for path in (backup, meta_path):
        if path.exists():
            meta = json.loads(path.read_text())
            size = meta.get("gaze_realsense_size", size)
            if meta.get("gaze_marker_points_realsense"):
                centres = np.asarray(meta["gaze_marker_points_realsense"], np.float32)
                break
    width, height = int(size[0]), int(size[1])
    if centres is None:
        centres = marker_points_for_size(width, height)
    marker_size = max(54, int(round(min(width, height) * 0.105)))
    half = marker_size / 2.0
    reference = {}
    for marker_id, (cx, cy) in zip(MARKER_IDS, centres):
        reference[int(marker_id)] = np.asarray(
            [[cx - half, cy - half], [cx + half, cy - half],
             [cx + half, cy + half], [cx - half, cy + half]], np.float32)
    return reference, centres, (width, height)


def eye_uv(image_shape, norm_pos):
    """Recorder convention: x scales directly, y is flipped."""
    h, w = image_shape[:2]
    try:
        x_norm, y_norm = float(norm_pos[0]), float(norm_pos[1])
    except (TypeError, ValueError, IndexError):
        return None
    u = int(round(x_norm * (w - 1)))
    v = int(round((1.0 - y_norm) * (h - 1)))
    return (u, v) if 0 <= u < w and 0 <= v < h else None


def detect_corners(image, detector):
    """{marker_id: 4x2 corners} for the markers that decoded."""
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    if ids is None:
        return {}
    found = {}
    for marker_corners, marker_id in zip(corners, ids.reshape(-1)):
        if int(marker_id) in MARKER_IDS:
            found[int(marker_id)] = marker_corners.reshape(4, 2).astype(np.float32)
    return found


def project(uv, homography, width, height):
    point = cv2.perspectiveTransform(
        np.asarray(uv, np.float32).reshape(1, 1, 2), homography)[0, 0]
    return [float(np.clip(point[0], 0, width - 1)),
            float(np.clip(point[1], 0, height - 1))]


def record_from(uv, homography, centres, width, height, source, extra):
    return {
        "hit": True,
        "source": source,
        "gaze_uv_in_eye": [float(uv[0]), float(uv[1])],
        "gaze_uv_in_realsense": project(uv, homography, width, height),
        "realsense_size": [width, height],
        "marker_points_realsense": np.asarray(centres, np.float32).tolist(),
        "homography_eye_to_realsense": homography.tolist(),
        **extra,
    }


# ------------------------------------------------------------- per recording

def locate(root: Path, fid: int):
    """A frame may still be in place or already moved into _filtered_out_frames."""
    here = root / f"frame_{fid}"
    if here.is_dir():
        return here, False
    there = root / FILTERED_DIR / f"frame_{fid}"
    if there.is_dir():
        return there, True
    return None, False


def find_eye_image(root: Path, fid: int):
    for base in (root, root / FILTERED_DIR):
        path = base / "et_images" / f"{fid}.jpg"
        if path.exists():
            image = cv2.imread(str(path))
            if image is not None:
                return image
    return None


def move_back(root: Path, fid: int):
    src = root / FILTERED_DIR / f"frame_{fid}"
    if src.is_dir():
        dst = root / f"frame_{fid}"
        if not dst.exists():
            shutil.move(str(src), str(dst))
    for mirror in MIRRORS:
        src = root / FILTERED_DIR / mirror / f"{fid}.jpg"
        if src.exists():
            out = root / mirror
            out.mkdir(exist_ok=True)
            dst = out / f"{fid}.jpg"
            if not dst.exists():
                shutil.move(str(src), str(dst))


def rebuild(root: Path, args):
    reference, centres, (width, height) = marker_corner_reference(root)
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID),
        cv2.aruco.DetectorParameters())

    meta_source = root / "recording_metadata.json.pre_filter_backup"
    if not meta_source.exists():
        meta_source = root / "recording_metadata.json"
    meta = json.loads(meta_source.read_text())
    ranges = meta.get("episode_ranges_original") or meta.get("episode_ranges", [])
    ids = []
    for entry in ranges:
        ids.extend(range(int(entry["start_frame"]), int(entry["end_frame"]) + 1))
    ids = sorted(set(ids))

    corners_by_frame = {}   # fid -> corners dict, for interpolation anchors
    pending = {}            # fid -> (frame_dir, uv, corners)
    stats = {"already_ok": 0, "markers4": 0, "markers3": 0, "markers2": 0,
             "interpolated": 0, "gaze_offscreen": 0, "too_few_markers": 0,
             "gap_too_long": 0, "drift_too_large": 0, "missing": 0}

    for fid in ids:
        frame_dir, _ = locate(root, fid)
        if frame_dir is None:
            stats["missing"] += 1
            continue
        contact = frame_dir / "gaze_contact.json"
        if contact.exists():
            stats["already_ok"] += 1
            try:
                existing = json.loads(contact.read_text())
                if "marker_corners_eye" in existing:
                    corners_by_frame[fid] = {
                        int(k): np.asarray(v, np.float32)
                        for k, v in existing["marker_corners_eye"].items()}
            except (OSError, ValueError, json.JSONDecodeError):
                pass
            continue
        image = find_eye_image(root, fid)
        if image is None:
            stats["missing"] += 1
            continue
        try:
            payload = json.loads((frame_dir / "pupil_gaze.json").read_text()).get("data", {})
        except (OSError, ValueError, json.JSONDecodeError):
            payload = {}
        uv = eye_uv(image.shape, payload.get("norm_pos") or [])
        if uv is None:
            stats["gaze_offscreen"] += 1
            continue
        found = detect_corners(image, detector)
        if found:
            corners_by_frame[fid] = found
        pending[fid] = (frame_dir, uv, found)

    # Anchors for interpolation: any frame whose markers decoded, recovered or not.
    for fid in ids:
        if fid in corners_by_frame:
            continue
        frame_dir, _ = locate(root, fid)
        if frame_dir is None or not (frame_dir / "gaze_contact.json").exists():
            continue
        image = find_eye_image(root, fid)
        if image is not None:
            found = detect_corners(image, detector)
            if found:
                corners_by_frame[fid] = found

    anchors = sorted(corners_by_frame)
    anchor_array = np.asarray(anchors)
    recovered = []

    for fid, (frame_dir, uv, found) in sorted(pending.items()):
        usable = {m: c for m, c in found.items() if m in reference}
        record = None
        if len(usable) >= max(2, args["min_markers"]):
            src = np.concatenate([usable[m] for m in sorted(usable)])
            dst = np.concatenate([reference[m] for m in sorted(usable)])
            homography, _ = cv2.findHomography(src, dst, method=0)
            if homography is not None:
                key = f"markers{min(4, len(usable))}"
                stats[key] = stats.get(key, 0) + 1
                record = record_from(
                    uv, homography, centres, width, height,
                    f"screen_marker_homography_corners{len(usable)}",
                    {"n_markers": len(usable), "marker_ids": sorted(usable),
                     "marker_corners_eye": {str(m): usable[m].tolist() for m in sorted(usable)}})
        if record is None and len(usable) < max(2, args["min_markers"]):
            if args["max_interp"] <= 0:
                stats["too_few_markers"] += 1
                continue
            position = int(np.searchsorted(anchor_array, fid))
            left = anchors[position - 1] if position > 0 else None
            right = anchors[position] if position < len(anchors) else None
            if left is None or right is None or (right - left - 1) > args["max_interp"]:
                stats["gap_too_long"] += 1
                continue
            shared = sorted(set(corners_by_frame[left]) & set(corners_by_frame[right])
                            & set(reference))
            if len(shared) < 3:
                stats["too_few_markers"] += 1
                continue
            a = np.concatenate([corners_by_frame[left][m] for m in shared])
            b = np.concatenate([corners_by_frame[right][m] for m in shared])
            drift = float(np.max(np.linalg.norm(b - a, axis=1)))
            if drift > args["max_drift"]:
                stats["drift_too_large"] += 1
                continue
            t = (fid - left) / float(right - left)
            dst = np.concatenate([reference[m] for m in shared])
            homography, _ = cv2.findHomography(a + (b - a) * t, dst, method=0)
            if homography is None:
                stats["too_few_markers"] += 1
                continue
            stats["interpolated"] += 1
            record = record_from(
                uv, homography, centres, width, height,
                "screen_marker_homography_interpolated",
                {"n_markers": len(shared), "marker_ids": shared,
                 "rebuilt_from": [left, right], "interp_gap": right - left - 1,
                 "interp_t": round(t, 4), "marker_drift_px": round(drift, 2)})
        if record is None:
            stats["too_few_markers"] += 1
            continue
        recovered.append(fid)
        if args["apply"]:
            (frame_dir / "gaze_contact.json").write_text(json.dumps(record, indent=2))
            move_back(root, fid)

    total = len(ids)
    before = stats["already_ok"]
    print(f"  frames {total}   已有 contact {before} ({100*before/total:.1f}%)")
    for key, label in (("markers4", "4 个 marker 角点"), ("markers3", "3 个 marker 角点"),
                       ("markers2", "2 个 marker 角点"), ("interpolated", "邻帧插值")):
        if stats.get(key):
            print(f"    恢复 · {label:<16}{stats[key]:>5}")
    print(f"    gaze 出画,无法恢复      {stats['gaze_offscreen']:>5}")
    print(f"    marker 太少             {stats['too_few_markers']:>5}")
    print(f"    空洞过长                {stats['gap_too_long']:>5}")
    print(f"    marker 漂移过大          {stats['drift_too_large']:>5}")
    if stats["missing"]:
        print(f"    帧目录/眼图缺失          {stats['missing']:>5}")
    after = before + len(recovered)
    print(f"    -> 可用率 {100*before/total:.1f}% → {100*after/total:.1f}%  (+{len(recovered)} 帧)")

    if args["apply"] and args["restore_metadata"]:
        backup = root / "recording_metadata.json.pre_filter_backup"
        current = root / "recording_metadata.json"
        if backup.exists():
            keep = root / "recording_metadata.json.after_auto_filter"
            if current.exists() and not keep.exists():
                shutil.copy2(current, keep)
            shutil.copy2(backup, current)
            print(f"    metadata 已从备份还原 (过滤后的版本存到 {keep.name})")
    return len(recovered)


def main(_):
    args = {"apply": FLAGS.apply, "min_markers": FLAGS.min_markers,
            "max_interp": FLAGS.max_interp, "max_drift": FLAGS.max_drift,
            "restore_metadata": FLAGS.restore_metadata}
    roots = [FLAGS.root] if FLAGS.root else ROOTS
    total = 0
    for entry in roots:
        root = Path(entry).expanduser().resolve()
        print(f"\n######## {root.name}")
        total += rebuild(root, args)
    if FLAGS.apply:
        print(f"\n恢复 {total} 帧,已移回主目录。source 字段区分来源:")
        print("  screen_marker_homography               录制时原生")
        print("  screen_marker_homography_cornersN      N 个 marker 的角点重建")
        print("  screen_marker_homography_interpolated  邻帧插值")
        print("\n接着手动筛选:  python examples/gaze_data_process/filter_recorded_gaze_frames.py")
    else:
        print(f"\n试运行:可恢复 {total} 帧,未写入或移动任何文件。加 --apply 才生效。")


if __name__ == "__main__":
    app.run(main)
