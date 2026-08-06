#!/usr/bin/env python3

import argparse
import csv
import math
import os
import re
from pathlib import Path

import cv2
import numpy as np


DEFAULT_RECORDING = (
    "/home/wrq/workspaces/HK_TACEXO_WANG/hoh_data/flip_object/"
    "flip_object_hoh_2026_07_30_03"
)


def frame_id(path: Path) -> int:
    match = re.search(r"frame_(\d+)$", path.name)
    return int(match.group(1)) if match else 10**12


def iter_frame_dirs(recording: Path):
    return sorted(
        [
            child
            for child in recording.iterdir()
            if child.is_dir() and child.name.startswith("frame_")
        ],
        key=frame_id,
    )


def read_image(path: Path, size):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    return cv2.resize(img, size)


def make_pair_tile(frame_dir: Path, image_size, label_height=34):
    front = read_image(frame_dir / "color_image.jpg", image_size)
    wrist = read_image(frame_dir / "color_image2.jpg", image_size)
    pair = np.hstack([front, wrist])

    label = np.full((label_height, pair.shape[1], 3), 245, dtype=np.uint8)
    fid = frame_id(frame_dir)
    text = f"frame_{fid}    left: color_image/front    right: color_image2/wrist"
    cv2.putText(
        label,
        text,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    return np.vstack([label, pair])


def make_contact_sheets(frame_dirs, out_dir: Path, image_size, cols, max_rows):
    tiles = [make_pair_tile(frame_dir, image_size) for frame_dir in frame_dirs]
    if not tiles:
        return []

    tile_h, tile_w = tiles[0].shape[:2]
    rows_per_page = max_rows
    tiles_per_page = cols * rows_per_page
    page_paths = []

    for page_idx in range(math.ceil(len(tiles) / tiles_per_page)):
        page_tiles = tiles[page_idx * tiles_per_page : (page_idx + 1) * tiles_per_page]
        rows = math.ceil(len(page_tiles) / cols)
        sheet = np.full((rows * tile_h, cols * tile_w, 3), 255, dtype=np.uint8)

        for i, tile in enumerate(page_tiles):
            r = i // cols
            c = i % cols
            y0 = r * tile_h
            x0 = c * tile_w
            sheet[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile

        page_path = out_dir / f"contact_sheet_page_{page_idx:02d}.jpg"
        cv2.imwrite(str(page_path), sheet)
        page_paths.append(page_path)

    return page_paths


def make_video(frame_dirs, out_path: Path, image_size, fps):
    if not frame_dirs:
        return None

    first = make_pair_tile(frame_dirs[0], image_size)
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_path}")

    try:
        for frame_dir in frame_dirs:
            writer.write(make_pair_tile(frame_dir, image_size))
    finally:
        writer.release()
    return out_path


def write_index_csv(frame_dirs, out_path: Path, cols, max_rows):
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame_id", "frame_dir", "contact_sheet_page", "row", "col"],
        )
        writer.writeheader()
        for i, frame_dir in enumerate(frame_dirs):
            page = i // (cols * max_rows)
            idx_in_page = i % (cols * max_rows)
            writer.writerow(
                {
                    "frame_id": frame_id(frame_dir),
                    "frame_dir": str(frame_dir),
                    "contact_sheet_page": page,
                    "row": idx_in_page // cols,
                    "col": idx_in_page % cols,
                }
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", default=DEFAULT_RECORDING)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth frame.")
    parser.add_argument("--cols", type=int, default=2)
    parser.add_argument("--max_rows", type=int, default=30)
    parser.add_argument("--image_width", type=int, default=320)
    parser.add_argument("--image_height", type=int, default=240)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--no_video", action="store_true")
    args = parser.parse_args()

    recording = Path(args.recording).expanduser().resolve()
    if not recording.exists():
        raise FileNotFoundError(recording)
    if args.stride <= 0:
        raise ValueError("--stride must be positive")

    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else recording / "frame_sequence"
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_dirs = iter_frame_dirs(recording)[:: args.stride]
    image_size = (args.image_width, args.image_height)

    page_paths = make_contact_sheets(
        frame_dirs,
        out_dir,
        image_size=image_size,
        cols=args.cols,
        max_rows=args.max_rows,
    )
    write_index_csv(frame_dirs, out_dir / "frame_index.csv", args.cols, args.max_rows)

    video_path = None
    if not args.no_video:
        video_path = make_video(frame_dirs, out_dir / "frame_sequence.mp4", image_size, args.fps)

    print(f"[done] recording={recording}")
    print(f"[done] frames={len(frame_dirs)} stride={args.stride}")
    print(f"[done] output_dir={out_dir}")
    for path in page_paths:
        print(f"[done] contact_sheet={path}")
    if video_path is not None:
        print(f"[done] video={video_path}")
    print(f"[done] index={out_dir / 'frame_index.csv'}")


if __name__ == "__main__":
    main()
