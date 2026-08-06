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
REPO_ROOT = SCRIPT_DIR.parent
ROBOT_INFRA_DIR = REPO_ROOT / "serl_robot_infra"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROBOT_INFRA_DIR))

from experiments.mappings import NEW_MAPPING

FLAGS = flags.FLAGS

DEFAULT_FRAME_ROOT = Path("/home/wrq/workspaces/HK_TACEXO_WANG/bc_data")

flags.DEFINE_multi_string(
    "frame_root",
    None,
    f"Recorded data root(s) with frame_xxx folders. Defaults to all recordings under {DEFAULT_FRAME_ROOT}.",
)
flags.DEFINE_string("exp_name", "flip_object", "Experiment name.")
flags.DEFINE_integer("enable_tactile", 0, "Whether to include tactile_data.")
flags.DEFINE_string("config_module", "", "Defaults to experiments.<exp_name>.config.")
flags.DEFINE_string(
    "robot_urdf_path",
    "",
    "Robot URDF path. Defaults to examples/urdf/denso_robot_with_ati_4.urdf.",
)
flags.DEFINE_string(
    "output_dir",
    "bc_data",
    "Output directory for demo pickle. Default writes to examples/bc_data/<exp_name>.",
)
flags.DEFINE_string("output_name", "", "Output filename.")
flags.DEFINE_boolean("success_only", True, "Only export successful episodes from metadata.")
flags.DEFINE_boolean("disable_image_crop", False, "Use full frame images instead of task crop.")
flags.DEFINE_string("label_name", "is_record_success.txt", "Per-frame success label filename.")
flags.DEFINE_boolean(
    "stop_at_first_success_label",
    True,
    "End each exported episode at the first frame whose success label is 1.",
)
flags.DEFINE_integer("max_transitions", 0, "Debug limit. 0 means no limit.")


def _frame_number(path: Path) -> int:
    match = re.search(r"frame_(\d+)", path.name)
    return int(match.group(1)) if match else 10**12


def _frame_dir(root: Path, frame_id: int) -> Path:
    return root / f"frame_{int(frame_id)}"


def _existing_frame_ids(root: Path, start: int, end: int):
    frame_ids = []
    for frame_id in range(start, end + 1):
        if _frame_dir(root, frame_id).exists():
            frame_ids.append(frame_id)
    return frame_ids


def _stack_obs_horizon_one(obs):
    return {key: np.asarray(value)[None] for key, value in obs.items()}


def _read_metadata(root: Path):
    metadata_path = root / "recording_metadata.json"
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text())


def _all_frame_ranges(root: Path):
    frame_dirs = sorted(
        [
            frame_dir
            for frame_dir in root.iterdir()
            if frame_dir.is_dir() and frame_dir.name.startswith("frame_")
        ],
        key=_frame_number,
    )
    if not frame_dirs:
        return []
    return [
        {
            "episode_index": 0,
            "success": True,
            "start_frame": _frame_number(frame_dirs[0]),
            "end_frame": _frame_number(frame_dirs[-1]),
        }
    ]


def _episodes(root: Path):
    metadata = _read_metadata(root)
    if metadata is None:
        print(f"[warn] {root} has no recording_metadata.json; using all contiguous frames.")
        return _all_frame_ranges(root)
    episodes = metadata.get("episode_ranges", [])
    stage_export_ranges = {
        "tennis_ball_place": ("place_range", "place"),
        "twist_bottle_cap": ("twist_range", "twist"),
    }
    if FLAGS.exp_name in stage_export_ranges:
        range_key, export_stage = stage_export_ranges[FLAGS.exp_name]
        stage_episodes = []
        for episode in episodes:
            stage_range = episode.get(range_key)
            if stage_range is None:
                stage_episodes.append(episode)
                continue
            stage_episode = dict(episode)
            stage_episode["full_start_frame"] = episode.get("start_frame")
            stage_episode["full_end_frame"] = episode.get("end_frame")
            stage_episode["start_frame"] = stage_range["start_frame"]
            stage_episode["end_frame"] = stage_range["end_frame"]
            stage_episode["export_stage"] = export_stage
            stage_episodes.append(stage_episode)
        return stage_episodes
    return episodes


def _default_frame_roots():
    if not DEFAULT_FRAME_ROOT.exists():
        return []
    prefixes = {
        "tennis_ball_pick": ("ball_pick_", "tennis_ball_pick_"),
        "tennis_ball_place": ("ball_place_", "tennis_ball_place_"),
        "lid_grip": ("lid_grip_",),
        "twist_bottle_cap": ("twist_bottle_cap_",),
        "tube_insertion": ("tube_insertion_",),
        "flip_object": ("flip_object_",),
    }.get(FLAGS.exp_name, (f"{FLAGS.exp_name}_",))
    search_roots = [DEFAULT_FRAME_ROOT]
    task_root = DEFAULT_FRAME_ROOT / FLAGS.exp_name
    if task_root.exists():
        search_roots.insert(0, task_root)

    frame_roots = []
    for search_root in search_roots:
        frame_roots.extend(
            [
                child
                for child in search_root.iterdir()
                if child.is_dir()
                and child.name.startswith(prefixes)
                and (child / "recording_metadata.json").exists()
            ]
        )
    return sorted(
        set(frame_roots)
    )


def _image_keys_from_config():
    if FLAGS.config_module:
        config_module = importlib.import_module(FLAGS.config_module)
        config = config_module.TrainConfig()
    else:
        assert FLAGS.exp_name in NEW_MAPPING, f"Experiment {FLAGS.exp_name} not found."
        config = NEW_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(
        fake_env=True,
        save_video=False,
        classifier=False,
        enable_tactile=bool(FLAGS.enable_tactile),
    )
    return list(config.image_keys)


def _read_success(frame_dir: Path) -> bool:
    label_path = frame_dir / FLAGS.label_name
    if not label_path.exists():
        return False
    return label_path.read_text().strip() == "1"


def _transition_from_frames(current_frame, next_frame, robot_urdf_path, image_keys, episode_index, done):
    from examples.utils import read_utils

    obs, _, action = read_utils.get_frame_data(
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
    reward = np.float32(1.0 if done else 0.0)
    return copy.deepcopy(
        dict(
            observations=_stack_obs_horizon_one(obs),
            actions=np.asarray(action[:7], dtype=np.float32),
            next_observations=_stack_obs_horizon_one(next_obs),
            rewards=reward,
            masks=np.float32(1.0 - float(done)),
            dones=bool(done),
            infos={
                "succeed": bool(done),
                "episode_id": int(episode_index),
                "frame_idx": _frame_number(current_frame),
                "next_frame_idx": _frame_number(next_frame),
                "source_root": str(current_frame.parent),
            },
        )
    )


def _iter_episode_pairs(root: Path, episode):
    if FLAGS.success_only and not bool(episode.get("success", False)):
        return
    start = int(episode["start_frame"])
    end = int(episode["end_frame"])
    if end <= start:
        return
    frame_ids = _existing_frame_ids(root, start, end)
    if len(frame_ids) < 2:
        return
    episode_index = int(episode.get("episode_index", 0))
    terminal_frame = frame_ids[-1]
    if FLAGS.stop_at_first_success_label:
        for frame_id in frame_ids:
            if _read_success(_frame_dir(root, frame_id)):
                terminal_frame = frame_id
                break
        frame_ids = [frame_id for frame_id in frame_ids if frame_id <= terminal_frame]
        if len(frame_ids) < 2:
            return
    for frame_id, next_frame_id in zip(frame_ids, frame_ids[1:]):
        current_frame = _frame_dir(root, frame_id)
        next_frame = _frame_dir(root, next_frame_id)
        done = bool(episode.get("success", False)) and next_frame_id == terminal_frame
        yield current_frame, next_frame, episode_index, done
        if done:
            return


def main(_):
    frame_roots = [Path(root).expanduser() for root in FLAGS.frame_root] if FLAGS.frame_root else _default_frame_roots()
    if not frame_roots:
        raise ValueError(f"No frame roots found. Pass --frame_root or add recordings under {DEFAULT_FRAME_ROOT}.")
    robot_urdf_path = FLAGS.robot_urdf_path or str(
        REPO_ROOT / "examples" / "urdf" / "denso_robot_with_ati_4.urdf"
    )
    image_keys = _image_keys_from_config()
    output_dir = (SCRIPT_DIR / FLAGS.output_dir).resolve()
    if FLAGS.output_dir == "bc_data":
        output_dir = output_dir / FLAGS.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    transitions = []
    episode_count = 0
    skipped = 0
    for root in frame_roots:
        if not root.exists():
            print(f"[warn] missing root: {root}")
            continue
        episodes = _episodes(root)
        episode_count += len([e for e in episodes if not FLAGS.success_only or e.get("success", False)])
        pairs = [item for episode in episodes for item in _iter_episode_pairs(root, episode)]
        print(f"[source] {root} episodes={len(episodes)} pairs={len(pairs)}")
        for current_frame, next_frame, episode_index, done in tqdm(pairs, desc=root.name):
            try:
                transitions.append(
                    _transition_from_frames(
                        current_frame,
                        next_frame,
                        robot_urdf_path,
                        image_keys,
                        episode_index,
                        done,
                    )
                )
            except Exception as exc:
                skipped += 1
                print(f"[warn] skipped {current_frame}: {exc}")
            if FLAGS.max_transitions > 0 and len(transitions) >= FLAGS.max_transitions:
                break
        if FLAGS.max_transitions > 0 and len(transitions) >= FLAGS.max_transitions:
            break

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_name = FLAGS.output_name or f"{FLAGS.exp_name}_{episode_count}_recorded_demos_{timestamp}.pkl"
    output_path = output_dir / output_name
    with output_path.open("wb") as f:
        pkl.dump(transitions, f)

    done_count = sum(int(transition["dones"]) for transition in transitions)
    reward_sum = float(sum(float(transition["rewards"]) for transition in transitions))
    print(f"[done] output={output_path}")
    print(f"[done] transitions={len(transitions)} dones={done_count} reward_sum={reward_sum:.1f} skipped={skipped}")


if __name__ == "__main__":
    app.run(main)
