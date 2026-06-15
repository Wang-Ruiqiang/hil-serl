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
import cv2
from absl import app, flags
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ROBOT_INFRA_DIR = REPO_ROOT / "serl_robot_infra"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROBOT_INFRA_DIR))


FLAGS = flags.FLAGS

flags.DEFINE_multi_string(
    "frame_root",
    [
        "/media/user/data3/wrq/recorded_data/tennis_ball_pick/tennis_ball_pick-6-11-0",
        "/media/user/data3/wrq/recorded_data/tennis_ball_pick/tennis_ball_pick-6-12-0",
    ],
    "Recorded data root(s) containing frame_xxx folders and recording_metadata.json.",
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
    "output_dir",
    "examples/demo_data",
    "Output directory for the generated RL demo pickle.",
)
flags.DEFINE_string(
    "output_name",
    "",
    "Output pickle filename. Defaults to <exp_name>_<episodes>_recorded_demos_<timestamp>.pkl.",
)
flags.DEFINE_boolean(
    "use_filtered_ranges",
    True,
    "Use kept_frame_ranges from recording_metadata.json when available.",
)
flags.DEFINE_boolean(
    "success_only",
    True,
    "Only export episodes marked success=True in recording_metadata.json.",
)
flags.DEFINE_integer(
    "max_transitions",
    0,
    "Optional debug limit. 0 means export all transitions.",
)


def _frame_number(path: Path) -> int:
    match = re.search(r"frame_(\d+)", path.name)
    return int(match.group(1)) if match else 10**12


def _frame_dir(root: Path, frame_id: int) -> Path:
    return root / f"frame_{int(frame_id)}"


def _stack_obs_horizon_one(obs):
    return {key: np.asarray(value)[None] for key, value in obs.items()}


def _read_numeric_file(path: Path, default=None, dtype=np.float32):
    if not path.exists():
        if default is None:
            raise FileNotFoundError(f"Missing numeric file: {path}")
        return np.asarray(default, dtype=dtype)
    values = np.loadtxt(path, dtype=dtype)
    return np.asarray(values, dtype=dtype).reshape(-1)


def _read_rgb(path: Path, size=(128, 128)):
    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        raise FileNotFoundError(f"Missing or unreadable RGB image: {path}")
    image_bgr = cv2.resize(image_bgr, size, interpolation=cv2.INTER_LINEAR)
    return image_bgr[..., ::-1].astype(np.uint8)


def _read_tactile(frame_dir: Path):
    thumb_path = frame_dir / "thumb_heat_map.jpg"
    index_path = frame_dir / "index_heat_map.jpg"
    thumb = cv2.imread(str(thumb_path))
    index = cv2.imread(str(index_path))
    if thumb is None:
        raise FileNotFoundError(f"Missing or unreadable tactile image: {thumb_path}")
    if index is None:
        raise FileNotFoundError(f"Missing or unreadable tactile image: {index_path}")
    thumb = cv2.resize(thumb, (128, 128), interpolation=cv2.INTER_LINEAR)
    index = cv2.resize(index, (128, 128), interpolation=cv2.INTER_LINEAR)
    return cv2.hconcat([thumb, index]).astype(np.uint8)


def _read_ee_pose(frame_dir: Path, robot_urdf_path: str):
    ee_pose_path = frame_dir / "robot_ee_pose.txt"
    if ee_pose_path.exists():
        ee_pose = _read_numeric_file(ee_pose_path, dtype=np.float32)
        if ee_pose.shape[0] >= 7:
            return ee_pose[:3], ee_pose[3:7]

    from examples.utils import kinematics_utils

    joints = _read_numeric_file(frame_dir / "right_arm_joint.txt", dtype=np.float64)
    tcp_pos, tcp_ori = kinematics_utils.comupute_forward_kinematics(
        joints,
        robot_urdf_path,
    )
    return np.asarray(tcp_pos, dtype=np.float32), np.asarray(tcp_ori, dtype=np.float32)


def _read_frame_data(frame_dir: Path, robot_urdf_path: str, image_keys):
    obs = {}
    for image_key in image_keys:
        if image_key == "front_camera":
            obs[image_key] = _read_rgb(frame_dir / "color_image.jpg")
        elif image_key == "tactile_data":
            obs[image_key] = _read_tactile(frame_dir)
        else:
            raise ValueError(f"Unsupported image key for RL demo export: {image_key}")

    tcp_pos, tcp_ori = _read_ee_pose(frame_dir, robot_urdf_path)
    hand_state = _read_numeric_file(
        frame_dir / "hand_state.txt",
        default=[0.0],
        dtype=np.float32,
    )[:1]
    obs["state"] = np.concatenate(
        [
            np.asarray(tcp_pos, dtype=np.float32).reshape(-1),
            np.asarray(tcp_ori, dtype=np.float32).reshape(-1),
            np.asarray(hand_state, dtype=np.float32).reshape(-1),
        ],
        axis=0,
    ).astype(np.float32)

    action = _read_numeric_file(
        frame_dir / "action.txt",
        default=np.zeros(7, dtype=np.float32),
        dtype=np.float32,
    )
    return obs, action[:7].astype(np.float32)


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
            "kept_frame_ranges": [
                {
                    "start_frame": _frame_number(frame_dirs[0]),
                    "end_frame": _frame_number(frame_dirs[-1]),
                    "num_frames": len(frame_dirs),
                }
            ],
        }
    ]


def _episodes_from_metadata(root: Path):
    metadata = _read_metadata(root)
    if metadata is None:
        print(f"[warn] {root} has no recording_metadata.json; falling back to all frames.")
        return _all_frame_ranges(root)

    episodes = metadata.get("episode_ranges", [])
    if not FLAGS.use_filtered_ranges:
        for episode in episodes:
            episode["kept_frame_ranges"] = [
                {
                    "start_frame": int(episode["start_frame"]),
                    "end_frame": int(episode["end_frame"]),
                    "num_frames": int(episode["end_frame"]) - int(episode["start_frame"]) + 1,
                }
            ]
    return episodes


def _image_keys_from_config():
    module_name = FLAGS.config_module or f"experiments.{FLAGS.exp_name}.config"
    try:
        config_module = importlib.import_module(module_name)
        config = config_module.TrainConfig()
        if hasattr(config, "get_image_keys"):
            return list(config.get_image_keys(bool(FLAGS.enable_tactile)))
        if hasattr(config, "image_keys"):
            return list(config.image_keys)
    except Exception as exc:
        print(
            f"[info] could not import {module_name} only for image_keys ({exc}); "
            "using the task default image keys."
        )

    image_keys = ["front_camera"]
    if FLAGS.enable_tactile:
        image_keys.append("tactile_data")
    return image_keys


def _transition_from_frames(
    current_frame: Path,
    next_frame: Path,
    *,
    robot_urdf_path: str,
    image_keys,
    reward: float,
    done: bool,
    episode_index: int,
):
    obs, action = _read_frame_data(current_frame, robot_urdf_path, image_keys)
    next_obs, _ = _read_frame_data(next_frame, robot_urdf_path, image_keys)

    action = np.asarray(action, dtype=np.float32)
    if action.shape != (7,):
        raise ValueError(f"expected action shape (7,), got {action.shape} from {current_frame}")

    info = {
        "succeed": bool(done and reward > 0.0),
        "frame_idx": _frame_number(current_frame),
        "next_frame_idx": _frame_number(next_frame),
        "episode_id": int(episode_index),
        "source_root": str(current_frame.parent),
        "grasp_penalty": 0.0,
        "robot_arm_penalty": 0.0,
    }
    return copy.deepcopy(
        dict(
            observations=_stack_obs_horizon_one(obs),
            actions=action,
            next_observations=_stack_obs_horizon_one(next_obs),
            rewards=np.float32(reward),
            masks=np.float32(1.0 - float(done)),
            dones=bool(done),
            infos=info,
            grasp_penalty=np.float32(0.0),
            robot_arm_penalty=np.float32(0.0),
        )
    )


def _iter_episode_pairs(root: Path, episode):
    kept_ranges = episode.get("kept_frame_ranges") or []
    if FLAGS.success_only and not bool(episode.get("success", False)):
        return
    if not kept_ranges:
        return

    episode_index = int(episode.get("episode_index", 0))
    last_range_index = len(kept_ranges) - 1
    for range_index, frame_range in enumerate(kept_ranges):
        start = int(frame_range["start_frame"])
        end = int(frame_range["end_frame"])
        if end <= start:
            continue
        for frame_id in range(start, end):
            current_frame = _frame_dir(root, frame_id)
            next_frame = _frame_dir(root, frame_id + 1)
            if not current_frame.exists() or not next_frame.exists():
                continue
            is_last_transition = (
                range_index == last_range_index
                and frame_id == end - 1
                and bool(episode.get("success", False))
            )
            yield current_frame, next_frame, episode_index, is_last_transition


def main(_):
    robot_urdf_path = FLAGS.robot_urdf_path or str(
        REPO_ROOT / "examples" / "urdf" / "fr3_moveit_servo.urdf"
    )
    image_keys = _image_keys_from_config()
    output_dir = (REPO_ROOT / FLAGS.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    roots = [Path(root).expanduser() for root in FLAGS.frame_root]
    valid_roots = [root for root in roots if root.exists()]
    missing_roots = [root for root in roots if not root.exists()]
    for root in missing_roots:
        print(f"[warn] missing root: {root}")

    print(f"[source] exp_name={FLAGS.exp_name}")
    print(f"[source] image_keys={image_keys}")
    print(f"[source] robot_urdf={robot_urdf_path}")
    print(f"[source] use_filtered_ranges={FLAGS.use_filtered_ranges}")

    transitions = []
    episode_count = 0
    skipped = 0
    for root in valid_roots:
        episodes = _episodes_from_metadata(root)
        episode_count += sum(
            1
            for episode in episodes
            if (not FLAGS.success_only or bool(episode.get("success", False)))
        )
        pair_count = sum(1 for episode in episodes for _ in _iter_episode_pairs(root, episode))
        print(f"[source] {root} episodes={len(episodes)} transition_pairs={pair_count}")

        pbar = tqdm(
            (item for episode in episodes for item in _iter_episode_pairs(root, episode)),
            total=pair_count,
            desc=root.name,
        )
        for current_frame, next_frame, episode_index, is_last_transition in pbar:
            try:
                transition = _transition_from_frames(
                    current_frame,
                    next_frame,
                    robot_urdf_path=robot_urdf_path,
                    image_keys=image_keys,
                    reward=1.0 if is_last_transition else 0.0,
                    done=bool(is_last_transition),
                    episode_index=episode_index,
                )
            except Exception as exc:
                skipped += 1
                print(f"[warn] skipped {current_frame}: {exc}")
                continue
            transitions.append(transition)
            if FLAGS.max_transitions > 0 and len(transitions) >= FLAGS.max_transitions:
                break
        if FLAGS.max_transitions > 0 and len(transitions) >= FLAGS.max_transitions:
            break

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_name = FLAGS.output_name or (
        f"{FLAGS.exp_name}_{episode_count}_demos_{timestamp}.pkl"
    )
    output_path = output_dir / output_name
    with output_path.open("wb") as f:
        pkl.dump(transitions, f)

    done_count = sum(int(transition["dones"]) for transition in transitions)
    reward_sum = float(sum(float(transition["rewards"]) for transition in transitions))
    print(f"[done] output={output_path}")
    print(f"[done] transitions={len(transitions)} dones={done_count} reward_sum={reward_sum:.1f} skipped={skipped}")


if __name__ == "__main__":
    app.run(main)
