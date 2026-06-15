#!/usr/bin/env python3

import copy
import datetime
import importlib
import os
import pickle as pkl
import re
import sys
from pathlib import Path

import numpy as np
from absl import app, flags
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
ROBOT_INFRA_DIR = REPO_ROOT / "serl_robot_infra"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EXAMPLES_DIR))
sys.path.insert(0, str(ROBOT_INFRA_DIR))


FLAGS = flags.FLAGS

flags.DEFINE_multi_string(
    "frame_root",
    [
        "/media/user/data3/wrq/recorded_data/tennis_ball_pick/tennis_ball_pick-6-11-0",
        "/media/user/data3/wrq/recorded_data/tennis_ball_pick/tennis_ball_pick-6-12-0",
    ],
    "Recorded data root(s) containing frame_xxx folders.",
)
flags.DEFINE_string("exp_name", "tennis_ball_pick", "Experiment name.")
flags.DEFINE_integer("enable_tactile", 1, "Whether to include tactile_data in observations.")
flags.DEFINE_string(
    "config_module",
    "",
    "Experiment config module. Defaults to experiments.<exp_name>.config.",
)
flags.DEFINE_string(
    "robot_urdf_path",
    "",
    "Robot URDF used by read_utils. Defaults to examples/urdf/fr3_moveit_servo.urdf.",
)
flags.DEFINE_string(
    "label_name",
    "is_recorded_success.txt",
    "Success label filename inside each frame_xxx folder.",
)
flags.DEFINE_string(
    "output_dir",
    "",
    "Output classifier data directory. Defaults to reward_classifier/classifier_data_pick(_no_tactile).",
)
flags.DEFINE_integer("batch_size", 500, "Number of transitions per pickle dump.")


def _frame_number(path: Path) -> int:
    match = re.search(r"frame_(\d+)", path.name)
    return int(match.group(1)) if match else 10**12


def _frame_dirs(root: Path):
    frames = [
        frame_dir
        for frame_dir in root.iterdir()
        if frame_dir.is_dir()
        and frame_dir.name.startswith("frame_")
        and (frame_dir / "color_image.jpg").exists()
        and (frame_dir / FLAGS.label_name).exists()
    ]
    return sorted(frames, key=_frame_number)


def _read_success_label(frame_dir: Path) -> int:
    return 1 if (frame_dir / FLAGS.label_name).read_text().strip() == "1" else 0


def _default_output_dir() -> Path:
    if FLAGS.output_dir:
        return Path(FLAGS.output_dir).expanduser()
    suffix = "classifier_data_pick" if FLAGS.enable_tactile else "classifier_data_pick_no_tactile"
    return SCRIPT_DIR / suffix


def _save_batch(batch_data, file_path: Path):
    if not batch_data:
        return
    with file_path.open("ab") as f:
        pkl.dump(batch_data, f)
    print(f"[dump] {file_path} ({len(batch_data)} transitions)")


def _classifier_image_keys_from_config():
    module_name = FLAGS.config_module or f"experiments.{FLAGS.exp_name}.config"
    try:
        config_module = importlib.import_module(module_name)
        config = config_module.TrainConfig()
        env = config.get_environment(
            fake_env=True,
            save_video=False,
            classifier=False,
            enable_tactile=bool(FLAGS.enable_tactile),
        )
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()
        image_keys = getattr(config, "classifier_keys", None) or getattr(config, "image_keys", None)
        if image_keys:
            return list(image_keys)
    except Exception as exc:
        print(f"[warn] failed to read classifier_keys from {module_name}: {exc}")

    image_keys = ["front_camera"]
    if FLAGS.enable_tactile:
        image_keys.append("tactile_data")
    return image_keys


def _transition_from_frames(current_frame: Path, next_frame: Path, robot_urdf_path: str, image_keys):
    from examples.utils import read_utils

    obs, is_success, frame_action = read_utils.get_frame_data(
        str(current_frame),
        robot_urdf_path,
        bool(FLAGS.enable_tactile),
        FLAGS.exp_name,
        image_keys=image_keys,
    )
    next_obs, _, _ = read_utils.get_frame_data(
        str(next_frame),
        robot_urdf_path,
        bool(FLAGS.enable_tactile),
        FLAGS.exp_name,
        image_keys=image_keys,
    )
    action = np.asarray(frame_action, dtype=np.float32).copy()
    return copy.deepcopy(
        dict(
            observations=obs,
            next_observations=next_obs,
            actions=action,
            rewards=0,
            masks=1.0,
            dones=0,
        )
    ), int(is_success)


def main(_):
    robot_urdf_path = FLAGS.robot_urdf_path or str(
        REPO_ROOT / "examples" / "urdf" / "fr3_moveit_servo.urdf"
    )
    image_keys = _classifier_image_keys_from_config()
    output_dir = _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    success_file = output_dir / f"success_images_{uuid}.pkl"
    failure_file = output_dir / f"failure_images_{uuid}.pkl"

    successes = []
    failures = []
    total_frames = 0
    skipped = 0

    print(f"[source] config_module={FLAGS.config_module or f'experiments.{FLAGS.exp_name}.config'}")
    print(f"[source] classifier_image_keys={image_keys}")
    print(f"[source] robot_urdf={robot_urdf_path}")

    roots = [Path(root).expanduser() for root in FLAGS.frame_root]
    for root in roots:
        if not root.exists():
            print(f"[warn] missing root: {root}")
            continue
        frames = _frame_dirs(root)
        print(f"[source] {root} frames={len(frames)}")
        if len(frames) < 2:
            continue

        for i in tqdm(range(len(frames) - 1), desc=root.name):
            current_frame = frames[i]
            next_frame = frames[i + 1]
            if _frame_number(next_frame) <= _frame_number(current_frame):
                skipped += 1
                continue
            try:
                transition, is_success = _transition_from_frames(
                    current_frame,
                    next_frame,
                    robot_urdf_path,
                    image_keys,
                )
            except Exception as exc:
                skipped += 1
                print(f"[warn] skipped {current_frame}: {exc}")
                continue

            total_frames += 1
            if is_success:
                successes.append(transition)
            else:
                failures.append(transition)

            if len(successes) >= FLAGS.batch_size:
                _save_batch(successes, success_file)
                successes = []
            if len(failures) >= FLAGS.batch_size:
                _save_batch(failures, failure_file)
                failures = []

    _save_batch(successes, success_file)
    _save_batch(failures, failure_file)

    print(f"[done] output_dir={output_dir}")
    print(f"[done] success_file={success_file}")
    print(f"[done] failure_file={failure_file}")
    print(f"[done] exported={total_frames} skipped={skipped}")


if __name__ == "__main__":
    app.run(main)
