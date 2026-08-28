#!/usr/bin/env python3

import json
import re
from pathlib import Path

import cv2
from absl import app, flags


FLAGS = flags.FLAGS

flags.DEFINE_multi_string(
    "frame_root",
    [
        "/home/ealin/workspaces/DexTacHil/data/recorded_data/tennis_ball_pick_and_place/tennis_ball_pick_and_place-2026-08-14_12-18-59",
        "/home/ealin/workspaces/DexTacHil/data/recorded_data/tennis_ball_pick_and_place/tennis_ball_pick_and_place-2026-08-14_12-49-48",
    ],
    "Recorded data root(s) containing frame_xxx folders.",
)
flags.DEFINE_string(
    "classifier_task",
    "place",
    "Classifier label task: pick or place. Controls default label/range filenames.",
)
flags.DEFINE_string(
    "image_name",
    "color_image.jpg",
    "Image filename inside each frame_xxx folder.",
)
flags.DEFINE_string(
    "label_name",
    "",
    "Success label filename to create/update. Empty chooses by classifier_task.",
)
flags.DEFINE_bool(
    "reset_existing_labels",
    True,
    "If true, overwrite all existing labels with 0 before labeling.",
)
flags.DEFINE_integer(
    "display_width",
    960,
    "Width used for the annotation display window.",
)
flags.DEFINE_bool(
    "show_tactile",
    True,
    "Show thumb/index tactile depth images next to the RGB image.",
)
flags.DEFINE_integer(
    "tactile_width",
    480,
    "Width used for the tactile display panel.",
)
flags.DEFINE_string(
    "range_name",
    "",
    "Range json written after labeling. Empty chooses by classifier_task; use 'none' to disable.",
)
flags.DEFINE_bool(
    "export_only_manual_ranges",
    True,
    "If true, only episodes with manually-set '[' and ']' ranges are written.",
)


def _frame_number(path: Path) -> int:
    match = re.search(r"frame_(\d+)", path.name)
    return int(match.group(1)) if match else 10**12


def _classifier_task() -> str:
    task = FLAGS.classifier_task.lower().strip()
    if task not in ("pick", "place"):
        raise ValueError("--classifier_task must be 'pick' or 'place'")
    return task


def _label_name() -> str:
    if FLAGS.label_name:
        return FLAGS.label_name
    if _classifier_task() == "pick":
        return "is_recorded_pick_success.txt"
    return "is_recorded_success.txt"


def _range_name() -> str:
    if FLAGS.range_name:
        range_name = FLAGS.range_name.strip()
        if range_name.lower() in ("none", "null", "off", "false"):
            return ""
        return range_name
    if _classifier_task() == "pick":
        return "pick_classifier_ranges.json"
    return "place_classifier_ranges.json"


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
        try:
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
        except Exception as exc:
            print(f"[warn] failed to read {metadata_path}: {exc}")

    frame_ids = [_frame_number(frame_dir) for frame_dir in frames]
    if not frame_ids:
        return []
    ranges = []
    start = frame_ids[0]
    prev = frame_ids[0]
    episode_index = 0
    for frame_id in frame_ids[1:]:
        if frame_id != prev + 1:
            ranges.append(
                {
                    "episode_index": episode_index,
                    "start_frame": int(start),
                    "end_frame": int(prev),
                }
            )
            episode_index += 1
            start = frame_id
        prev = frame_id
    ranges.append(
        {
            "episode_index": episode_index,
            "start_frame": int(start),
            "end_frame": int(prev),
        }
    )
    return ranges


def _read_label(frame_dir: Path) -> int:
    label_path = frame_dir / _label_name()
    try:
        return 1 if label_path.read_text().strip() == "1" else 0
    except OSError:
        return 0


def _write_label(frame_dir: Path, value: int):
    (frame_dir / _label_name()).write_text(f"{int(value)}\n")


def _initialise_labels(frames):
    for frame_dir in frames:
        label_path = frame_dir / _label_name()
        if FLAGS.reset_existing_labels or not label_path.exists():
            _write_label(frame_dir, 0)


def _range_label_path(root: Path):
    range_name = _range_name()
    if not range_name:
        return None
    return root / range_name


def _read_manual_ranges(root: Path):
    range_path = _range_label_path(root)
    if range_path is None or not range_path.exists():
        return {}
    try:
        data = json.loads(range_path.read_text())
    except Exception as exc:
        print(f"[warn] failed to read existing range file {range_path}: {exc}")
        return {}
    manual_ranges = {}
    for item in data.get("ranges", []):
        if not item.get("manual", False):
            continue
        manual_ranges[int(item["episode_index"])] = {
            "start_frame": int(item["start_frame"]),
            "end_frame": int(item["end_frame"]),
        }
    return manual_ranges


def _episode_index_for_frame(frame_id: int, episodes):
    for episode in episodes:
        if int(episode["start_frame"]) <= frame_id <= int(episode["end_frame"]):
            return int(episode["episode_index"])
    return None


def _write_pick_range_file(root: Path, frames, manual_ranges):
    range_name = _range_name()
    if not range_name:
        return
    output_ranges = []
    for episode in _load_episode_ranges(root, frames):
        start_frame = int(episode["start_frame"])
        end_frame = int(episode["end_frame"])
        manual_range = manual_ranges.get(int(episode["episode_index"]))
        if manual_range is not None:
            range_start = max(start_frame, int(manual_range["start_frame"]))
            range_end = min(end_frame, int(manual_range["end_frame"]))
            manual = True
        else:
            if FLAGS.export_only_manual_ranges:
                continue
            range_start = start_frame
            range_end = end_frame
            manual = False
        if range_end < range_start:
            range_start, range_end = range_end, range_start
        output_ranges.append(
            {
                "episode_index": int(episode["episode_index"]),
                "episode_start_frame": start_frame,
                "episode_end_frame": end_frame,
                "start_frame": int(range_start),
                "end_frame": int(range_end),
                "manual": manual,
                "num_frames": int(range_end - range_start + 1),
            }
        )

    output_path = _range_label_path(root)
    output = {
        "classifier_task": _classifier_task(),
        "label_name": _label_name(),
        "range_name": range_name,
        "semantics": (
            f"For {_classifier_task()} classifier export, keep frames start_frame..end_frame inclusive. "
            "Use '[' to set the current episode range start and ']' to set range end. "
            "Unset episodes are skipped when export_only_manual_ranges=True."
        ),
        "ranges": output_ranges,
    }
    output_path.write_text(json.dumps(output, indent=2))
    print(f"[range] wrote {output_path} ranges={len(output_ranges)}")


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
    thumb = _read_tactile_image(frame_dir / "thumb_depth_image.png")
    index = _read_tactile_image(frame_dir / "index_depth_image.png")
    if thumb is None or index is None:
        thumb = _read_tactile_image(frame_dir / "thumb_heat_map.jpg")
        index = _read_tactile_image(frame_dir / "index_heat_map.jpg")
    if thumb is None or index is None:
        return None

    panel = cv2.resize(
        cv2.hconcat(
            [
                cv2.resize(thumb, (height, height), interpolation=cv2.INTER_NEAREST),
                cv2.resize(index, (height, height), interpolation=cv2.INTER_NEAREST),
            ]
        ),
        (FLAGS.tactile_width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 35), (0, 0, 0), -1)
    cv2.putText(
        panel,
        "tactile: thumb | index",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


def _make_display_image(
    frame_dir: Path,
    root: Path,
    sample_idx: int,
    total: int,
    episode_index=None,
    manual_range=None,
):
    image = cv2.imread(str(frame_dir / FLAGS.image_name))
    if image is None:
        return None

    label = _read_label(frame_dir)
    h, w = image.shape[:2]
    if FLAGS.display_width > 0 and w != FLAGS.display_width:
        scale = FLAGS.display_width / float(w)
        image = cv2.resize(
            image,
            (FLAGS.display_width, max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )

    status = "SUCCESS" if label == 1 else "FAIL/UNMARKED"
    status_color = (0, 255, 0) if label == 1 else (0, 0, 255)
    frame_text = (
        f"{root.name} | {frame_dir.name} | sample={sample_idx + 1}/{total} | {status}"
    )
    label_text = f"task={_classifier_task()} label file: {_label_name()}"
    range_text = (
        f"episode={episode_index} range="
        f"{manual_range['start_frame']}..{manual_range['end_frame']}"
        if manual_range is not None
        else f"episode={episode_index} range=default full episode"
    )
    keys_text = (
        "keys: 1/s/k=success | 0/f/x=fail | n/right/space=next | "
        "p/left=prev | [=range start | ]=range end | q/esc=quit"
    )

    cv2.rectangle(image, (0, 0), (image.shape[1], 120), (0, 0, 0), -1)
    cv2.putText(
        image,
        frame_text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        status_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        label_text,
        (12, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        range_text,
        (12, 83),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        keys_text,
        (12, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    tactile_panel = _make_tactile_panel(frame_dir, image.shape[0])
    if tactile_panel is not None:
        image = cv2.hconcat([image, tactile_panel])
    return image


def _label_root(root: Path):
    frames = _find_frames(root)
    if not frames:
        print(f"[warn] no frames with {FLAGS.image_name}: {root}")
        return

    episodes = _load_episode_ranges(root, frames)
    episode_by_index = {int(episode["episode_index"]): episode for episode in episodes}
    manual_ranges = _read_manual_ranges(root)
    _initialise_labels(frames)
    print(f"[label] root={root}")
    print(f"[label] frames={len(frames)} task={_classifier_task()} label={_label_name()}")
    print(f"[label] range_file={_range_label_path(root)}")

    idx = 0
    while 0 <= idx < len(frames):
        frame_dir = frames[idx]
        frame_id = _frame_number(frame_dir)
        episode_index = _episode_index_for_frame(frame_id, episodes)
        manual_range = manual_ranges.get(episode_index)
        image = _make_display_image(
            frame_dir,
            root,
            idx,
            len(frames),
            episode_index=episode_index,
            manual_range=manual_range,
        )
        if image is None:
            idx += 1
            continue

        cv2.imshow("label recorded success frames", image)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("1"), ord("s"), ord("k")):
            _write_label(frame_dir, 1)
            idx += 1
        elif key in (ord("0"), ord("f"), ord("x")):
            _write_label(frame_dir, 0)
            idx += 1
        elif key in (ord("n"), ord(" "), 83):
            idx += 1
        elif key in (ord("p"), 81):
            idx = max(0, idx - 1)
        elif key == ord("["):
            if episode_index is not None:
                episode = episode_by_index[episode_index]
                current = manual_ranges.setdefault(
                    episode_index,
                    {
                        "start_frame": int(episode["start_frame"]),
                        "end_frame": int(episode["end_frame"]),
                    },
                )
                current["start_frame"] = frame_id
                print(f"[range] episode={episode_index} start={frame_id}")
        elif key == ord("]"):
            if episode_index is not None:
                episode = episode_by_index[episode_index]
                current = manual_ranges.setdefault(
                    episode_index,
                    {
                        "start_frame": int(episode["start_frame"]),
                        "end_frame": int(episode["end_frame"]),
                    },
                )
                current["end_frame"] = frame_id
                print(f"[range] episode={episode_index} end={frame_id}")
        elif key in (ord("q"), 27):
            break

    success_count = sum(_read_label(frame_dir) for frame_dir in frames)
    _write_pick_range_file(root, frames, manual_ranges)
    print(f"[done] {root.name}: success={success_count} total={len(frames)}")


def main(_):
    for root_str in FLAGS.frame_root:
        root = Path(root_str).expanduser()
        if not root.exists():
            print(f"[warn] missing root: {root}")
            continue
        _label_root(root)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    app.run(main)
