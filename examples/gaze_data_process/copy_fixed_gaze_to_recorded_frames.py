#!/usr/bin/env python3
"""Copy one recorded gaze JSON to every frame in another recorded dataset.

This is useful for building negative/augmentation data where the RGB state is
new, but the intended gaze point should be fixed to a known object location.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path


DEFAULT_SOURCE_GAZE = (
    "/media/user/data3/wrq/recorded_data/tennis_ball_pick/"
    "tennis_ball_pick-6-23-1/frame_433/gaze_contact.json"
)
DEFAULT_TARGET_ROOT = (
    "/media/user/data3/wrq/recorded_data/tennis_ball_pick/"
    "tennis_ball_pick-6-25-1"
)


def frame_number(path: Path) -> int:
    match = re.search(r"frame_(\d+)$", path.name)
    return int(match.group(1)) if match else 10**12


def discover_frame_dirs(root: Path) -> list[Path]:
    return sorted(
        [
            frame_dir
            for frame_dir in root.iterdir()
            if frame_dir.is_dir() and frame_dir.name.startswith("frame_")
        ],
        key=frame_number,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy one gaze_contact.json to every frame_* directory."
    )
    parser.add_argument(
        "--source_gaze_json",
        default=DEFAULT_SOURCE_GAZE,
        help="Source gaze JSON. Default: tennis_ball_pick-6-23-1/frame_117/gaze_contact.json",
    )
    parser.add_argument(
        "--target_root",
        default=DEFAULT_TARGET_ROOT,
        help="Target recorded dataset root containing frame_* directories.",
    )
    parser.add_argument(
        "--gaze_json_name",
        default="gaze_contact.json",
        help="Output JSON filename inside each target frame directory.",
    )
    parser.add_argument(
        "--backup_existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Backup existing target gaze JSON before overwriting.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be written without modifying files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.source_gaze_json).expanduser()
    target_root = Path(args.target_root).expanduser()

    if not source_path.exists():
        raise FileNotFoundError(f"Missing source gaze JSON: {source_path}")
    if not target_root.exists():
        raise FileNotFoundError(f"Missing target root: {target_root}")

    source_data = json.loads(source_path.read_text())
    required_keys = ("gaze_uv_in_realsense", "realsense_size")
    missing_keys = [key for key in required_keys if key not in source_data]
    if missing_keys:
        raise ValueError(f"{source_path} is missing required keys: {missing_keys}")

    frame_dirs = discover_frame_dirs(target_root)
    if not frame_dirs:
        raise RuntimeError(f"No frame_* directories found under {target_root}")

    backup_suffix = datetime.now().strftime(".backup_%Y%m%d_%H%M%S")
    written = 0
    backed_up = 0
    for frame_dir in frame_dirs:
        target_path = frame_dir / args.gaze_json_name
        if args.dry_run:
            print(f"[dry-run] write {target_path}")
            continue

        if target_path.exists() and args.backup_existing:
            backup_path = target_path.with_name(target_path.name + backup_suffix)
            shutil.copy2(target_path, backup_path)
            backed_up += 1

        target_path.write_text(
            json.dumps(source_data, indent=2, ensure_ascii=False) + "\n"
        )
        written += 1

    gaze_uv = source_data.get("gaze_uv_in_realsense")
    print(f"[source] {source_path}")
    print(f"[source] gaze_uv_in_realsense={gaze_uv}")
    print(f"[target] {target_root}")
    print(
        f"[done] frames={len(frame_dirs)} written={written} "
        f"backed_up={backed_up} dry_run={args.dry_run}"
    )


if __name__ == "__main__":
    main()
