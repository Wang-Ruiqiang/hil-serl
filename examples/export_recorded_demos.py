#!/usr/bin/env python3

import copy
import datetime
import importlib
import inspect
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

from serl_launcher.utils.gaze_mask_utils import (  # noqa: E402
    gaze_phase_onehot,
    select_gaze_target_mask,
)

FLAGS = flags.FLAGS

flags.DEFINE_multi_string(
    "frame_root",
    [
        "/home/ealin/workspaces/DexTacHil/data/recorded_data/tennis_ball_pick_and_place/tennis_ball_pick_and_place-2026-08-14_12-18-59",
    ],
    "Recorded data root(s) containing frame_xxx folders and recording_metadata.json.",
)
flags.DEFINE_string("exp_name", "tennis_ball_pick_and_place", "Experiment name.")
flags.DEFINE_integer("enable_tactile", 1, "Whether to include tactile_data in observations.")
flags.DEFINE_string(
    "gaze_json_name",
    "gaze_contact.json",
    "JSON file inside each frame folder that stores realsense gaze coordinates.",
)
flags.DEFINE_boolean(
    "require_gaze_hit",
    False,
    "Skip frames whose gaze JSON has hit=false.",
)
flags.DEFINE_enum(
    "state_gaze_slot",
    "phase",
    ["phase", "gaze_xy"],
    "What the last two state columns carry. Both are two numbers, so the "
    "observation space is identical either way -- only the meaning differs, "
    "and it has to match what the encoder's grounding query was trained to "
    "read. 'phase' writes the [mask1, mask2] one-hot the pick_classifier "
    "pipeline uses. 'gaze_xy' writes the recorded fixation, normalised over "
    "the full RealSense frame exactly as train_encoder.py does, for encoders "
    "built with --grounding_gaze_conditioned. Frames without a usable "
    "fixation get (-1, -1), the same out-of-frame value append_gaze_xy_to_state "
    "uses, so the conditioner can tell 'no gaze' from a corner.",
)
flags.DEFINE_boolean(
    "disable_image_crop",
    True,
    "Export full-frame RGB images and normalize gaze in full-frame coordinates.",
)
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
flags.DEFINE_string(
    "reward_label_name",
    "",
    "Frame success label used for demo reward/done. Empty chooses by exp_name.",
)
flags.DEFINE_integer(
    "max_transitions",
    0,
    "Optional debug limit. 0 means export all transitions.",
)
flags.DEFINE_integer(
    "max_episodes",
    30,
    "Maximum successful episodes to export across each root. 0 means all.",
)
flags.DEFINE_boolean(
    "use_gaze_target_mask",
    True,
    "Add gaze-selected mask image and phase one-hot to exported observations.",
)
flags.DEFINE_string(
    "gaze_target_mask_key",
    "front_camera_mask",
    "Observation key for the gaze-selected mask image.",
)
flags.DEFINE_string(
    "mask1_key",
    "front_camera_mask1",
    "Observation key for the first predicted/recorded target mask.",
)
flags.DEFINE_string(
    "mask2_key",
    "front_camera_mask2",
    "Observation key for the second predicted/recorded target mask.",
)
flags.DEFINE_string(
    "hand_mask_key",
    "front_camera_hand_mask",
    "Optional observation key for the recorded hand mask.",
)
flags.DEFINE_string(
    "phase_scan_path",
    "examples/encoder_training/runs/pick_classifier_phase_scan.json",
    "Classifier phase scan used to assign pick/place state. Empty uses recorded gaze.",
)
flags.DEFINE_integer(
    "gaze_target_mask_dilation",
    2,
    "Dilation radius in exported 128x128 mask pixels when checking gaze hits. "
    "Defaults to the same 2 train_rlpd.py uses, so a demo frame and the "
    "identical frame online resolve to the same slot.",
)
def _frame_number(path: Path) -> int:
    match = re.search(r"frame_(\d+)", path.name)
    return int(match.group(1)) if match else 10**12


def _frame_dir(root: Path, frame_id: int) -> Path:
    return root / f"frame_{int(frame_id)}"


def _stack_obs_horizon_one(obs):
    return {key: np.asarray(value)[None] for key, value in obs.items()}


def _zero_action_rpy(action):
    action = np.asarray(action, dtype=np.float32).copy()
    action[..., 3:6] = 0.0
    return action


def _read_numeric_file(path: Path, default=None, dtype=np.float32):
    if not path.exists():
        if default is None:
            raise FileNotFoundError(f"Missing numeric file: {path}")
        return np.asarray(default, dtype=dtype)
    values = np.loadtxt(path, dtype=dtype)
    return np.asarray(values, dtype=dtype).reshape(-1)


def _reward_label_name() -> str:
    if FLAGS.reward_label_name:
        return FLAGS.reward_label_name
    if FLAGS.exp_name == "tennis_ball_pick":
        return "is_recorded_pick_success.txt"
    return "is_recorded_success.txt"


def _read_recorded_success(frame_dir: Path) -> bool:
    label_path = frame_dir / _reward_label_name()
    if not label_path.exists() and _reward_label_name() == "is_recorded_success.txt":
        label_path = frame_dir / "is_record_success.txt"
    if not label_path.exists():
        return False
    return label_path.read_text().strip() == "1"


def _read_rgb(path: Path, size=(128, 128)):
    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        raise FileNotFoundError(f"Missing or unreadable RGB image: {path}")
    image_bgr = cv2.resize(image_bgr, size, interpolation=cv2.INTER_LINEAR)
    return image_bgr[..., ::-1].astype(np.uint8)


def _read_mask(mask_path: Path, target_shape=(128, 128)):
    if not mask_path.exists():
        return np.zeros(target_shape, dtype=bool)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros(target_shape, dtype=bool)
    mask = cv2.resize(
        mask,
        (int(target_shape[1]), int(target_shape[0])),
        interpolation=cv2.INTER_NEAREST,
    )
    return mask > 0


def _mask_paths(frame_dir: Path):
    frame_id = _frame_number(frame_dir)
    central_names = {
        "mask1": (f"frame_{frame_id:06d}_ball_mask.png", f"frame_{frame_id:06d}_mask1.png"),
        "hand": (f"frame_{frame_id:06d}_hand_mask.png",),
        "mask2": (f"frame_{frame_id:06d}_basket_mask.png", f"frame_{frame_id:06d}_mask2.png"),
    }
    root = frame_dir.parent
    return {
        "mask1": [
            frame_dir / "ball_mask.png",
            frame_dir / "mask1.png",
            # The 2026-08-25 sessions name it target_ball_mask.png. Without
            # this the export silently finds nothing and writes blank masks --
            # which is exactly how new30_demos.pkl ended up with all three
            # mask images empty in all 5822 transitions.
            frame_dir / "target_ball_mask.png",
            frame_dir / "rs_mask_obj0.png",
            *(root / "sam_masks" / name for name in central_names["mask1"]),
            *(root / "sam_masks" / "propagated" / name for name in central_names["mask1"]),
        ],
        "hand": [
            frame_dir / "hand_mask.png",
            frame_dir / "sam_hand_mask.png",
            *(root / "sam_masks" / name for name in central_names["hand"]),
            *(root / "sam_masks" / "propagated" / name for name in central_names["hand"]),
        ],
        "mask2": [
            frame_dir / "basket_mask.png",
            frame_dir / "mask2.png",
            frame_dir / "sam_basket_mask.png",
            frame_dir / "rs_mask_obj1.png",
            *(root / "sam_masks" / name for name in central_names["mask2"]),
            *(root / "sam_masks" / "propagated" / name for name in central_names["mask2"]),
        ],
    }


def _first_existing_mask(frame_dir: Path, slot: str, target_shape=(128, 128)):
    for mask_path in _mask_paths(frame_dir)[slot]:
        if mask_path.exists():
            return _read_mask(mask_path, target_shape)
    return np.zeros(target_shape, dtype=bool)


def _phase_onehot(selected_index):
    # Delegate to the shared implementation so this cannot drift from the
    # width the env wrappers append -- a local copy is how demos would end up
    # one column wider than the observation space they are replayed into.
    return gaze_phase_onehot(selected_index)


def _read_gaze_target_mask(frame_dir: Path, target_shape=(128, 128)):
    """Pick the mask the recorded fixation lands on, using the wrapper's rule.

    Delegated rather than reimplemented, for the same reason _phase_onehot is:
    a local copy drifts. The copy this replaced dilated by a different default
    and broke ties by mask area, so a demo frame whose gaze sat between the two
    objects went to the basket -- 60x the ball's area -- while the identical
    frame online went to whichever was nearer.
    """
    try:
        gaze_xy = _read_recorded_gaze_xy(frame_dir)
    except Exception:
        return np.zeros(target_shape, dtype=bool), None

    mask_probs = np.stack(
        [
            _first_existing_mask(frame_dir, slot, target_shape).astype(np.float32)
            for slot in ("mask1", "mask2")
        ],
        axis=0,
    )
    selected, info = select_gaze_target_mask(
        mask_probs,
        None,
        target_shape=target_shape,
        dilation_px=FLAGS.gaze_target_mask_dilation,
        gaze_xy_norm=gaze_xy,
        return_info=True,
    )
    return selected > 0.5, info["selected_mask_index"]


def _mask_to_image(mask):
    return np.repeat((np.asarray(mask, dtype=np.uint8)[..., None] * 255), 3, axis=-1)


def _read_slot_mask_image(frame_dir: Path, slot: str, target_shape=(128, 128)):
    return _mask_to_image(_first_existing_mask(frame_dir, slot, target_shape))


def _read_front_camera_mask(
    frame_dir: Path,
    rgb_image,
    target_shape=(128, 128),
    selected_index=None,
):
    if selected_index in (0, 1):
        slot = "mask1" if selected_index == 0 else "mask2"
        selected_mask = _first_existing_mask(frame_dir, slot, target_shape)
    else:
        selected_mask, selected_index = _read_gaze_target_mask(frame_dir, target_shape)
    return _mask_to_image(selected_mask), _phase_onehot(selected_index)


def _read_tactile_depth(path: Path):
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Missing or unreadable tactile depth image: {path}")
    if depth.ndim == 2:
        depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
    elif depth.ndim == 3 and depth.shape[-1] == 4:
        depth = depth[..., :3]
    return cv2.resize(depth.astype(np.uint8, copy=False), (128, 128), interpolation=cv2.INTER_LINEAR)


def _read_tactile(frame_dir: Path):
    thumb = _read_tactile_depth(frame_dir / "thumb_depth_image.png")
    index = _read_tactile_depth(frame_dir / "index_depth_image.png")
    return np.concatenate([thumb, index], axis=1).astype(np.uint8)


def _front_camera_bounds(exp_name: str, realsense_size):
    width, height = int(realsense_size[0]), int(realsense_size[1])
    return 0.0, 0.0, float(width), float(height)


def _read_recorded_gaze_xy(frame_dir: Path):
    gaze_path = frame_dir / FLAGS.gaze_json_name
    if not gaze_path.exists():
        raise FileNotFoundError(f"Missing gaze json: {gaze_path}")

    gaze_data = json.loads(gaze_path.read_text())
    if FLAGS.require_gaze_hit and not bool(gaze_data.get("hit", False)):
        raise ValueError(f"Invalid gaze hit in {gaze_path}")

    gaze_uv = gaze_data.get("gaze_uv_in_realsense")
    realsense_size = gaze_data.get("realsense_size")
    if gaze_uv is None or realsense_size is None:
        raise ValueError(
            f"{gaze_path} must contain gaze_uv_in_realsense and realsense_size"
        )

    x, y = float(gaze_uv[0]), float(gaze_uv[1])
    x0, y0, x1, y1 = _front_camera_bounds(FLAGS.exp_name, realsense_size)
    x_norm = (x - x0) / max(x1 - x0, 1e-6)
    y_norm = (y - y0) / max(y1 - y0, 1e-6)
    return np.asarray(
        [np.clip(x_norm, 0.0, 1.0), np.clip(y_norm, 0.0, 1.0)],
        dtype=np.float32,
    )


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


def _read_frame_data(
    frame_dir: Path,
    robot_urdf_path: str,
    image_keys,
    phase_selected_index=None,
):
    obs = {}
    rgb_image = None
    gaze_phase = None
    for image_key in image_keys:
        if image_key == "front_camera":
            rgb_image = _read_rgb(frame_dir / "color_image.jpg")
            obs[image_key] = rgb_image
        elif image_key == "tactile_data":
            obs[image_key] = _read_tactile(frame_dir)
        elif image_key == FLAGS.gaze_target_mask_key:
            if rgb_image is None:
                rgb_image = _read_rgb(frame_dir / "color_image.jpg")
            obs[image_key], gaze_phase = _read_front_camera_mask(
                frame_dir,
                rgb_image,
                selected_index=phase_selected_index,
            )
        elif image_key == FLAGS.mask1_key:
            obs[image_key] = _read_slot_mask_image(frame_dir, "mask1")
        elif image_key == FLAGS.mask2_key:
            obs[image_key] = _read_slot_mask_image(frame_dir, "mask2")
        elif image_key == FLAGS.hand_mask_key:
            obs[image_key] = _read_slot_mask_image(frame_dir, "hand")
        else:
            raise ValueError(f"Unsupported image key for RL demo export: {image_key}")
    if FLAGS.gaze_target_mask_key in image_keys and gaze_phase is None:
        if rgb_image is None:
            rgb_image = _read_rgb(frame_dir / "color_image.jpg")
        obs[FLAGS.gaze_target_mask_key], gaze_phase = _read_front_camera_mask(
            frame_dir,
            rgb_image,
            selected_index=phase_selected_index,
        )

    tcp_pos, tcp_ori = _read_ee_pose(frame_dir, robot_urdf_path)
    hand_state = _read_numeric_file(
        frame_dir / "hand_state.txt",
        default=[0.0],
        dtype=np.float32,
    )[:1]
    state_parts = [
        np.asarray(tcp_pos, dtype=np.float32).reshape(-1),
        np.asarray(tcp_ori, dtype=np.float32).reshape(-1),
        np.asarray(hand_state, dtype=np.float32).reshape(-1),
    ]
    if FLAGS.state_gaze_slot == "gaze_xy":
        # _read_recorded_gaze_xy already returns the fixation normalised over
        # the full frame -- _front_camera_bounds is (0, 0, w, h) and the env
        # resizes the whole frame rather than cropping it -- so this is the
        # same convention train_encoder.py fed the grounding query offline and
        # the same one gaze_xy_norm_from_heatmap produces online.
        # hit=False means the homography did not place the fixation in the
        # frame, so the stored uv is stale. train_encoder.py drops those frames
        # outright; the slot has to agree, or the query is conditioned on a
        # position the operator was never looking at.
        gaze_slot = np.asarray([-1.0, -1.0], dtype=np.float32)
        try:
            gaze_json = json.loads(
                (frame_dir / FLAGS.gaze_json_name).read_text()
            )
            if bool(gaze_json.get("hit", False)):
                gaze_slot = _read_recorded_gaze_xy(frame_dir)
        except Exception:
            pass
        state_parts.append(np.asarray(gaze_slot, dtype=np.float32).reshape(-1)[:2])
    elif gaze_phase is not None:
        state_parts.append(gaze_phase)
    obs["state"] = np.concatenate(state_parts, axis=0).astype(np.float32)
    action = _read_numeric_file(
        frame_dir / "action.txt",
        default=np.zeros(7, dtype=np.float32),
        dtype=np.float32,
    )
    return obs, _zero_action_rpy(action[:7])


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
    for episode in episodes:
        if not FLAGS.use_filtered_ranges or not episode.get("kept_frame_ranges"):
            episode["kept_frame_ranges"] = [
                {
                    "start_frame": int(episode["start_frame"]),
                    "end_frame": int(episode["end_frame"]),
                    "num_frames": int(episode["end_frame"]) - int(episode["start_frame"]) + 1,
                }
            ]
    return episodes


def _load_phase_switches(path: str):
    if not path:
        return {}
    phase_path = Path(path).expanduser()
    if not phase_path.is_absolute():
        phase_path = REPO_ROOT / phase_path
    if not phase_path.is_file():
        raise FileNotFoundError(f"Phase scan does not exist: {phase_path}")
    payload = json.loads(phase_path.read_text())
    switches = {}
    for result in payload.get("results", []):
        first_place_frame = result.get("first_positive_frame")
        if result.get("excluded") or first_place_frame is None:
            continue
        switches[(str(result["dataset"]), int(result["episode_index"]))] = int(
            first_place_frame
        )
    return switches


def _image_keys_from_config():
    module_name = FLAGS.config_module or f"experiments.{FLAGS.exp_name}.config"
    try:
        config_module = importlib.import_module(module_name)
        config = config_module.TrainConfig()
        if hasattr(config, "get_image_keys"):
            signature = inspect.signature(config.get_image_keys)
            kwargs = {"enable_tactile": bool(FLAGS.enable_tactile)}
            if "use_gaze_target_mask" in signature.parameters:
                kwargs["use_gaze_target_mask"] = bool(FLAGS.use_gaze_target_mask)
            image_keys = list(config.get_image_keys(**kwargs))
            if FLAGS.use_gaze_target_mask and FLAGS.gaze_target_mask_key not in image_keys:
                image_keys.append(FLAGS.gaze_target_mask_key)
            if FLAGS.use_gaze_target_mask:
                for mask_key in (FLAGS.mask1_key, FLAGS.mask2_key):
                    if mask_key not in image_keys:
                        image_keys.append(mask_key)
            return image_keys
        if hasattr(config, "image_keys"):
            image_keys = list(config.image_keys)
            if FLAGS.use_gaze_target_mask and FLAGS.gaze_target_mask_key not in image_keys:
                image_keys.append(FLAGS.gaze_target_mask_key)
            if FLAGS.use_gaze_target_mask:
                for mask_key in (FLAGS.mask1_key, FLAGS.mask2_key):
                    if mask_key not in image_keys:
                        image_keys.append(mask_key)
            return image_keys
    except Exception as exc:
        print(
            f"[info] could not import {module_name} only for image_keys ({exc}); "
            "using the task default image keys."
        )

    image_keys = ["front_camera"]
    if FLAGS.enable_tactile:
        image_keys.append("tactile_data")
    if FLAGS.use_gaze_target_mask:
        image_keys.append(FLAGS.gaze_target_mask_key)
        image_keys.extend([FLAGS.mask1_key, FLAGS.mask2_key])
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
    phase_switch_frame=None,
):
    def selected_index(frame):
        if phase_switch_frame is None:
            return None
        return int(_frame_number(frame) >= phase_switch_frame)

    obs, action = _read_frame_data(
        current_frame,
        robot_urdf_path,
        image_keys,
        phase_selected_index=selected_index(current_frame),
    )
    next_obs, _ = _read_frame_data(
        next_frame,
        robot_urdf_path,
        image_keys,
        phase_selected_index=selected_index(next_frame),
    )

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
    transition = dict(
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
    return copy.deepcopy(transition)


def _iter_episode_pairs(root: Path, episode, phase_switches):
    kept_ranges = episode.get("kept_frame_ranges") or []
    if FLAGS.success_only and not bool(episode.get("success", False)):
        return
    if not kept_ranges:
        return

    episode_index = int(episode.get("episode_index", 0))
    phase_switch_frame = phase_switches.get((root.name, episode_index))
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
            is_final_transition = (
                range_index == len(kept_ranges) - 1 and frame_id + 1 == end
            )
            is_success_transition = bool(episode.get("success", False)) and (
                _read_recorded_success(current_frame)
                or _read_recorded_success(next_frame)
                or is_final_transition
            )
            yield (
                current_frame,
                next_frame,
                episode_index,
                is_success_transition,
                phase_switch_frame,
            )
            if is_success_transition:
                return


def main(_):
    robot_urdf_path = FLAGS.robot_urdf_path or str(
        REPO_ROOT / "examples" / "urdf" / "fr3_moveit_servo.urdf"
    )
    image_keys = _image_keys_from_config()
    phase_switches = _load_phase_switches(FLAGS.phase_scan_path)
    output_dir = (REPO_ROOT / FLAGS.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    roots = [Path(root).expanduser() for root in FLAGS.frame_root]
    valid_roots = [root for root in roots if root.exists()]
    missing_roots = [root for root in roots if not root.exists()]
    for root in missing_roots:
        print(f"[warn] missing root: {root}")

    print(f"[source] exp_name={FLAGS.exp_name}")
    print(f"[source] reward_label_name={_reward_label_name()}")
    print(f"[source] image_keys={image_keys}")
    print(f"[source] robot_urdf={robot_urdf_path}")
    print(f"[source] use_filtered_ranges={FLAGS.use_filtered_ranges}")
    print(f"[source] disable_image_crop={FLAGS.disable_image_crop}")
    print(f"[source] phase_scan_path={FLAGS.phase_scan_path or '<recorded gaze>'}")

    transitions = []
    episode_count = 0
    skipped = 0
    for root in valid_roots:
        episodes = _episodes_from_metadata(root)
        selected_episodes = [
            episode
            for episode in episodes
            if (not FLAGS.success_only or bool(episode.get("success", False)))
        ]
        if FLAGS.max_episodes > 0:
            selected_episodes = selected_episodes[: FLAGS.max_episodes]
        episode_count += len(selected_episodes)
        pair_count = sum(
            1
            for episode in selected_episodes
            for _ in _iter_episode_pairs(root, episode, phase_switches)
        )
        print(
            f"[source] {root} episodes={len(episodes)} "
            f"selected_episodes={len(selected_episodes)} transition_pairs={pair_count}"
        )

        pbar = tqdm(
            (
                item
                for episode in selected_episodes
                for item in _iter_episode_pairs(root, episode, phase_switches)
            ),
            total=pair_count,
            desc=root.name,
        )
        for (
            current_frame,
            next_frame,
            episode_index,
            is_last_transition,
            phase_switch_frame,
        ) in pbar:
            try:
                transition = _transition_from_frames(
                    current_frame,
                    next_frame,
                    robot_urdf_path=robot_urdf_path,
                    image_keys=image_keys,
                    reward=1.0 if is_last_transition else 0.0,
                    done=bool(is_last_transition),
                    episode_index=episode_index,
                    phase_switch_frame=phase_switch_frame,
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

    if transitions:
        first_obs = transitions[0]["observations"]
        first_state = np.asarray(first_obs["state"])
        if FLAGS.gaze_target_mask_key in first_obs:
            mask_image = np.asarray(first_obs[FLAGS.gaze_target_mask_key])
            print(
                f"[check] {FLAGS.gaze_target_mask_key} shape={mask_image.shape} "
                f"active_pixels={int(np.count_nonzero(mask_image))}"
            )
        for mask_key in (FLAGS.mask1_key, FLAGS.hand_mask_key, FLAGS.mask2_key):
            if mask_key in first_obs:
                mask_image = np.asarray(first_obs[mask_key])
                print(
                    f"[check] {mask_key} shape={mask_image.shape} "
                    f"active_pixels={int(np.count_nonzero(mask_image))}"
                )
        print(
            f"[check] state_shape={first_state.shape} "
            f"phase_onehot={first_state[..., -2:]}"
        )
        first_action = np.asarray(transitions[0]["actions"], dtype=np.float32)
        print(f"[check] first_action_rpy={first_action[3:6]}")

    done_count = sum(int(transition["dones"]) for transition in transitions)
    reward_sum = float(sum(float(transition["rewards"]) for transition in transitions))
    print(f"[done] output={output_path}")
    print(f"[done] transitions={len(transitions)} dones={done_count} reward_sum={reward_sum:.1f} skipped={skipped}")


if __name__ == "__main__":
    app.run(main)
