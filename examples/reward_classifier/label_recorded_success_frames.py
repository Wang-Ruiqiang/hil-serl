#!/usr/bin/env python3

import json
import re
from pathlib import Path

import cv2
from absl import app, flags


FLAGS = flags.FLAGS

flags.DEFINE_multi_string("frame_root", None, "Recorded data root(s) with frame_xxx folders.")
flags.DEFINE_string("classifier_task", "default", "Label task name, used only for metadata.")
flags.DEFINE_string("image_name", "color_image.jpg", "Image filename in each frame folder.")
flags.DEFINE_string("label_name", "is_record_success.txt", "Label filename to write.")
flags.DEFINE_bool("reset_existing_labels", True, "Overwrite missing/existing labels with 0 first.")
flags.DEFINE_integer("display_width", 960, "Display image width.")
flags.DEFINE_bool("show_tactile", True, "Show thumb/index tactile panels when available.")
flags.DEFINE_integer("tactile_width", 480, "Tactile panel width.")
flags.DEFINE_string("range_name", "classifier_ranges.json", "Range json filename. Use none to disable.")
flags.DEFINE_bool("export_only_manual_ranges", False, "Only write manually selected ranges.")


def _frame_number(path: Path) -> int:
    match = re.search(r"frame_(\d+)", path.name)
    return int(match.group(1)) if match else 10**12


def _range_name():
    value = FLAGS.range_name.strip()
    return "" if value.lower() in {"none", "null", "off", "false"} else value


def _find_frames(root: Path):
    frames = [
        frame_dir
        for frame_dir in root.iterdir()
        if frame_dir.is_dir()
        and frame_dir.name.startswith("frame_")
        and (frame_dir / FLAGS.image_name).exists()
    ]
    return sorted(frames, key=_frame_number)


def _load_episode_ranges(root: Path, frames):
    metadata_path = root / "recording_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        episodes = metadata.get("episode_ranges", [])
        if episodes:
            return [
                {
                    "episode_index": int(episode.get("episode_index", index)),
                    "start_frame": int(episode["start_frame"]),
                    "end_frame": int(episode["end_frame"]),
                }
                for index, episode in enumerate(episodes)
            ]
    if not frames:
        return []
    return [
        {
            "episode_index": 0,
            "start_frame": _frame_number(frames[0]),
            "end_frame": _frame_number(frames[-1]),
        }
    ]


def _read_label(frame_dir: Path) -> int:
    label_path = frame_dir / FLAGS.label_name
    if not label_path.exists():
        return 0
    return 1 if label_path.read_text().strip() == "1" else 0


def _write_label(frame_dir: Path, value: int):
    (frame_dir / FLAGS.label_name).write_text(f"{int(value)}\n")


def _initialise_labels(frames):
    for frame_dir in frames:
        if FLAGS.reset_existing_labels or not (frame_dir / FLAGS.label_name).exists():
            _write_label(frame_dir, 0)


def _read_manual_ranges(root: Path):
    range_name = _range_name()
    if not range_name:
        return {}
    range_path = root / range_name
    if not range_path.exists():
        return {}
    data = json.loads(range_path.read_text())
    return {
        int(item["episode_index"]): {
            "start_frame": int(item["start_frame"]),
            "end_frame": int(item["end_frame"]),
        }
        for item in data.get("ranges", [])
        if item.get("manual", False)
    }


def _episode_index_for_frame(frame_id: int, episodes):
    for episode in episodes:
        if int(episode["start_frame"]) <= frame_id <= int(episode["end_frame"]):
            return int(episode["episode_index"])
    return None


def _write_range_file(root: Path, frames, manual_ranges):
    range_name = _range_name()
    if not range_name:
        return
    ranges = []
    for episode in _load_episode_ranges(root, frames):
        episode_index = int(episode["episode_index"])
        manual_range = manual_ranges.get(episode_index)
        if manual_range is None:
            if FLAGS.export_only_manual_ranges:
                continue
            manual_range = {
                "start_frame": int(episode["start_frame"]),
                "end_frame": int(episode["end_frame"]),
            }
            manual = False
        else:
            manual = True
        start = int(manual_range["start_frame"])
        end = int(manual_range["end_frame"])
        if end < start:
            start, end = end, start
        ranges.append(
            {
                "episode_index": episode_index,
                "episode_start_frame": int(episode["start_frame"]),
                "episode_end_frame": int(episode["end_frame"]),
                "start_frame": start,
                "end_frame": end,
                "manual": manual,
                "num_frames": end - start + 1,
            }
        )
    output = {
        "classifier_task": FLAGS.classifier_task,
        "label_name": FLAGS.label_name,
        "range_name": range_name,
        "ranges": ranges,
    }
    output_path = root / range_name
    output_path.write_text(json.dumps(output, indent=2))
    print(f"[range] wrote {output_path} ranges={len(ranges)}")


def _read_tactile_image(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]
    return image.astype("uint8", copy=False)


def _make_tactile_panel(frame_dir: Path, height: int):
    if not FLAGS.show_tactile:
        return None
    thumb = _read_tactile_image(frame_dir / "thumb_heat_map.jpg")
    index = _read_tactile_image(frame_dir / "index_heat_map.jpg")
    if thumb is None or index is None:
        return None
    thumb = cv2.resize(thumb, (height, height), interpolation=cv2.INTER_NEAREST)
    index = cv2.resize(index, (height, height), interpolation=cv2.INTER_NEAREST)
    panel = cv2.resize(cv2.hconcat([thumb, index]), (FLAGS.tactile_width, height))
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 35), (0, 0, 0), -1)
    cv2.putText(panel, "tactile: thumb | index", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    return panel


def _make_display_image(frame_dir: Path, root: Path, idx: int, total: int, episode_index, manual_range):
    image = cv2.imread(str(frame_dir / FLAGS.image_name))
    if image is None:
        return None
    h, w = image.shape[:2]
    if FLAGS.display_width > 0 and w != FLAGS.display_width:
        scale = FLAGS.display_width / float(w)
        image = cv2.resize(image, (FLAGS.display_width, max(1, int(h * scale))))
    label = _read_label(frame_dir)
    status = "SUCCESS" if label else "FAIL/UNMARKED"
    color = (0, 255, 0) if label else (0, 0, 255)
    range_text = (
        f"episode={episode_index} range={manual_range['start_frame']}..{manual_range['end_frame']}"
        if manual_range is not None
        else f"episode={episode_index} range=full episode"
    )
    cv2.rectangle(image, (0, 0), (image.shape[1], 112), (0, 0, 0), -1)
    cv2.putText(image, f"{root.name} | {frame_dir.name} | {idx + 1}/{total} | {status}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    cv2.putText(image, f"label file: {FLAGS.label_name}", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(image, range_text, (12, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv2.putText(image, "1/s=success | 0/f=fail | n/space=next | p=prev | [ ]=range | q=quit", (12, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
    tactile_panel = _make_tactile_panel(frame_dir, image.shape[0])
    return cv2.hconcat([image, tactile_panel]) if tactile_panel is not None else image


def _label_root(root: Path):
    frames = _find_frames(root)
    if not frames:
        print(f"[warn] no frames in {root}")
        return
    episodes = _load_episode_ranges(root, frames)
    episode_by_index = {int(episode["episode_index"]): episode for episode in episodes}
    manual_ranges = _read_manual_ranges(root)
    _initialise_labels(frames)

    idx = 0
    while 0 <= idx < len(frames):
        frame_dir = frames[idx]
        frame_id = _frame_number(frame_dir)
        episode_index = _episode_index_for_frame(frame_id, episodes)
        image = _make_display_image(
            frame_dir,
            root,
            idx,
            len(frames),
            episode_index,
            manual_ranges.get(episode_index),
        )
        if image is None:
            idx += 1
            continue
        cv2.imshow("label recorded success frames", image)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("1"), ord("s")):
            _write_label(frame_dir, 1)
            idx += 1
        elif key in (ord("0"), ord("f")):
            _write_label(frame_dir, 0)
            idx += 1
        elif key in (ord("n"), ord(" "), 83):
            idx += 1
        elif key in (ord("p"), 81):
            idx = max(0, idx - 1)
        elif key == ord("[") and episode_index is not None:
            episode = episode_by_index[episode_index]
            manual_ranges.setdefault(episode_index, dict(episode))["start_frame"] = frame_id
            print(f"[range] episode={episode_index} start={frame_id}")
        elif key == ord("]") and episode_index is not None:
            episode = episode_by_index[episode_index]
            manual_ranges.setdefault(episode_index, dict(episode))["end_frame"] = frame_id
            print(f"[range] episode={episode_index} end={frame_id}")
        elif key in (ord("q"), 27):
            break

    success_count = sum(_read_label(frame_dir) for frame_dir in frames)
    _write_range_file(root, frames, manual_ranges)
    print(f"[done] {root.name}: success={success_count} total={len(frames)}")


def main(_):
    if not FLAGS.frame_root:
        raise ValueError("--frame_root is required")
    for root_str in FLAGS.frame_root:
        root = Path(root_str).expanduser()
        if root.exists():
            _label_root(root)
        else:
            print(f"[warn] missing root: {root}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    app.run(main)
