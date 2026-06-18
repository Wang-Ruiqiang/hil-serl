#!/usr/bin/env python3

import re
from pathlib import Path

import cv2
from absl import app, flags


FLAGS = flags.FLAGS

flags.DEFINE_multi_string(
    "frame_root",
    [
        "/media/user/data3/wrq/recorded_data/tennis_ball_pick/tennis_ball_pick-6-17-0",
    ],
    "Recorded data root(s) containing frame_xxx folders.",
)
flags.DEFINE_string(
    "image_name",
    "color_image.jpg",
    "Image filename inside each frame_xxx folder.",
)
flags.DEFINE_string(
    "label_name",
    "is_recorded_success.txt",
    "Success label filename to create/update inside each frame_xxx folder.",
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


def _frame_number(path: Path) -> int:
    match = re.search(r"frame_(\d+)", path.name)
    return int(match.group(1)) if match else 10**12


def _find_frames(root: Path):
    frames = [
        frame_dir
        for frame_dir in root.iterdir()
        if frame_dir.is_dir()
        and frame_dir.name.startswith("frame_")
        and (frame_dir / FLAGS.image_name).exists()
    ]
    return sorted(frames, key=_frame_number)


def _read_label(frame_dir: Path) -> int:
    label_path = frame_dir / FLAGS.label_name
    try:
        return 1 if label_path.read_text().strip() == "1" else 0
    except OSError:
        return 0


def _write_label(frame_dir: Path, value: int):
    (frame_dir / FLAGS.label_name).write_text(f"{int(value)}\n")


def _initialise_labels(frames):
    for frame_dir in frames:
        label_path = frame_dir / FLAGS.label_name
        if FLAGS.reset_existing_labels or not label_path.exists():
            _write_label(frame_dir, 0)


def _make_display_image(frame_dir: Path, root: Path, sample_idx: int, total: int):
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
    keys_text = (
        "keys: 1/s/k=success | 0/f/x=fail | n/right/space=next | "
        "p/left=prev | q/esc=quit"
    )

    cv2.rectangle(image, (0, 0), (image.shape[1], 70), (0, 0, 0), -1)
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
        keys_text,
        (12, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return image


def _label_root(root: Path):
    frames = _find_frames(root)
    if not frames:
        print(f"[warn] no frames with {FLAGS.image_name}: {root}")
        return

    _initialise_labels(frames)
    print(f"[label] root={root}")
    print(f"[label] frames={len(frames)} label={FLAGS.label_name}")

    idx = 0
    while 0 <= idx < len(frames):
        frame_dir = frames[idx]
        image = _make_display_image(frame_dir, root, idx, len(frames))
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
        elif key in (ord("q"), 27):
            break

    success_count = sum(_read_label(frame_dir) for frame_dir in frames)
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
