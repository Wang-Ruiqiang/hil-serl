#!/usr/bin/env python3

import copy
import datetime
import importlib
import json
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

flags.DEFINE_multi_string("frame_root", None, "Recorded data root(s) with frame_xxx folders.")
flags.DEFINE_string("exp_name", "tennis_ball_pick", "Experiment name.")
flags.DEFINE_string("classifier_task", "auto", "Task/stage name used for default output dirs.")
flags.DEFINE_integer("enable_tactile", 1, "Whether to include tactile_data.")
flags.DEFINE_string("config_module", "", "Defaults to experiments.<exp_name>.config.")
flags.DEFINE_string(
    "robot_urdf_path",
    "",
    "Robot URDF path. Defaults to examples/urdf/denso_robot_with_ati_4.urdf.",
)
flags.DEFINE_string("label_name", "is_record_success.txt", "Success label filename.")
flags.DEFINE_string("output_dir", "", "Output classifier data dir.")
flags.DEFINE_integer("batch_size", 500, "Transitions per pickle dump.")
flags.DEFINE_string("range_name", "classifier_ranges.json", "Optional range json. Use none to disable.")
flags.DEFINE_boolean("disable_image_crop", False, "Use full frame images instead of task crop.")


def _frame_number(path: Path) -> int:
    match = re.search(r"frame_(\d+)", path.name)
    return int(match.group(1)) if match else 10**12


def _range_name():
    value = FLAGS.range_name.strip()
    return "" if value.lower() in {"none", "null", "off", "false"} else value


def _classifier_task():
    task = FLAGS.classifier_task.strip()
    if task and task != "auto":
        return task
    if FLAGS.exp_name == "twist_bottle_cap":
        return "bottle_twist"
    if FLAGS.exp_name == "lid_grip":
        return "lid_grip"
    if FLAGS.exp_name == "tube_insertion":
        return "tube_insertion"
    if FLAGS.exp_name in {"tennis_ball_pick", "tennis_ball_place"}:
        return "pick"
    return FLAGS.exp_name


def _frame_dirs(root: Path):
    frames = [
        frame_dir
        for frame_dir in root.iterdir()
        if frame_dir.is_dir()
        and frame_dir.name.startswith("frame_")
        and (frame_dir / "color_image.jpg").exists()
    ]
    return sorted(frames, key=_frame_number)


def _has_label(frame_dir: Path):
    return (frame_dir / FLAGS.label_name).exists()


def _read_label(frame_dir: Path):
    return 1 if (frame_dir / FLAGS.label_name).read_text().strip() == "1" else 0


def _load_allowed_frame_ids(root: Path):
    range_name = _range_name()
    if not range_name:
        return None
    range_path = root / range_name
    if not range_path.exists():
        return None
    data = json.loads(range_path.read_text())
    allowed = set()
    for item in data.get("ranges", []):
        allowed.update(range(int(item["start_frame"]), int(item["end_frame"]) + 1))
    print(f"[source] range_filter={range_path} allowed_frames={len(allowed)}")
    return allowed


def _default_output_dir():
    if FLAGS.output_dir:
        return Path(FLAGS.output_dir).expanduser()
    suffix = _classifier_task()
    tactile_suffix = "" if FLAGS.enable_tactile else "_no_tactile"
    return SCRIPT_DIR / f"classifier_data_{suffix}{tactile_suffix}"


def _classifier_image_keys_from_config():
    module_name = FLAGS.config_module or f"experiments.{FLAGS.exp_name}.config"
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
    return list(config.classifier_keys or config.image_keys)


def _save_batch(batch_data, file_path: Path):
    if not batch_data:
        return
    with file_path.open("ab") as f:
        pkl.dump(batch_data, f)
    print(f"[dump] {file_path} ({len(batch_data)} transitions)")


def _transition_from_frames(current_frame, next_frame, robot_urdf_path, image_keys):
    from examples.utils import read_utils

    obs, is_success, action = read_utils.get_frame_data(
        str(current_frame),
        robot_urdf_path,
        bool(FLAGS.enable_tactile),
        FLAGS.exp_name,
        image_keys=image_keys,
        disable_image_crop=FLAGS.disable_image_crop,
        label_name=FLAGS.label_name,
    )
    next_obs, _, _ = read_utils.get_frame_data(
        str(next_frame),
        robot_urdf_path,
        bool(FLAGS.enable_tactile),
        FLAGS.exp_name,
        image_keys=image_keys,
        disable_image_crop=FLAGS.disable_image_crop,
        label_name=FLAGS.label_name,
    )
    return copy.deepcopy(
        dict(
            observations=obs,
            next_observations=next_obs,
            actions=np.asarray(action[:7], dtype=np.float32),
            rewards=0,
            masks=1.0,
            dones=0,
        )
    ), int(is_success)


def main(_):
    if not FLAGS.frame_root:
        raise ValueError("--frame_root is required")
    robot_urdf_path = FLAGS.robot_urdf_path or str(
        REPO_ROOT / "examples" / "urdf" / "denso_robot_with_ati_4.urdf"
    )
    image_keys = _classifier_image_keys_from_config()
    output_dir = _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    success_file = output_dir / f"success_images_{timestamp}.pkl"
    failure_file = output_dir / f"failure_images_{timestamp}.pkl"
    successes = []
    failures = []
    exported = 0
    skipped = 0

    print(f"[source] exp_name={FLAGS.exp_name} classifier_task={_classifier_task()}")
    print(f"[source] label_name={FLAGS.label_name} range_name={_range_name() or 'disabled'}")
    print(f"[source] image_keys={image_keys}")
    print(f"[source] robot_urdf={robot_urdf_path}")

    for root_str in FLAGS.frame_root:
        root = Path(root_str).expanduser()
        if not root.exists():
            print(f"[warn] missing root: {root}")
            continue
        frames = _frame_dirs(root)
        allowed = _load_allowed_frame_ids(root)
        for i in tqdm(range(max(0, len(frames) - 1)), desc=root.name):
            current_frame = frames[i]
            next_frame = frames[i + 1]
            if not _has_label(current_frame):
                skipped += 1
                continue
            if allowed is not None and _frame_number(current_frame) not in allowed:
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
            exported += 1
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
    print(f"[done] exported={exported} skipped={skipped}")


if __name__ == "__main__":
    app.run(main)
