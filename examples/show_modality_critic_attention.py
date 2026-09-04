#!/usr/bin/env python3

import csv
import json
import os
import sys
import time
import ast
from collections.abc import Mapping
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".3")

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (
    REPO_ROOT,
    REPO_ROOT / "examples",
    REPO_ROOT / "serl_launcher",
    REPO_ROOT / "serl_robot_infra",
):
    path = str(path)
    if path not in sys.path:
        sys.path.insert(0, path)

DEFAULT_FRAME_ROOT = (
    "/home/ealin/workspaces/DexTacHil/data/recorded_data/tennis_ball_pick/"
    "tennis_ball_pick-7-14-1"
)
DEFAULT_CHECKPOINT_PATH = str(
    REPO_ROOT
    / "examples"
    / "experiments"
    / "tennis_ball_pick_and_place"
    # / "2026-07-21_resnet_maskhead_maskobs_SUCCESS"
    / "2026-08-13_vit_noattn_maskobs_fail"
)
DEFAULT_ROBOT_URDF_PATH = str(
    REPO_ROOT / "examples" / "urdf" / "fr3_moveit_servo.urdf"
)

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags
from flax.training import checkpoints

from experiments.mappings import NEW_MAPPING
from serl_launcher.utils.gaze_mask_utils import (
    add_gaze_mask_image_to_obs,
    compute_all_index_target_mask_fields,
    compute_index_target_mask_fields,
    gaze_phase_onehot,
    load_mask_predictor,
)
from serl_launcher.utils.launcher import make_gaze_sac_pixel_agent_hybrid_single_arm


FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", "tennis_ball_pick_and_place", "Experiment name.")
flags.DEFINE_enum(
    "encoder_type",
    "vit",
    ["vit", "pretrained_resnet"],
    "Encoder used by the checkpoint: vit or pretrained_resnet.",
)
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("frame_root", DEFAULT_FRAME_ROOT, "Recorded frame root.")
flags.DEFINE_string("checkpoint_path", DEFAULT_CHECKPOINT_PATH, "Checkpoint directory.")
flags.DEFINE_integer(
    "checkpoint_step",
    -1,
    "Checkpoint step to load. Use -1 for latest.",
)
flags.DEFINE_boolean(
    "skip_restore",
    False,
    "Do not restore RL checkpoint; useful for checking raw pretrained/backbone attention.",
)
flags.DEFINE_integer("image_size", 128, "Network image size.")
flags.DEFINE_integer("max_frames", 100, "Max frames to render. Use 0 for all.")
flags.DEFINE_integer("frame_stride", 10, "Process every Nth recorded frame.")
flags.DEFINE_integer("start_frame", 0, "Skip frames with id smaller than this.")
flags.DEFINE_integer("enable_tactile", 1, "Whether tactile_data is part of obs.")
flags.DEFINE_integer(
    "gaze_target_mask_dilation",
    0,
    "Dilation used when deciding whether recorded gaze hits mask1/mask2.",
)
flags.DEFINE_enum(
    "mask_selection_mode",
    "pick_only",
    [
        "auto",
        "recorded_gaze",
        "pick_only",
        "place_only",
        "phase_ranges",
    ],
    "How this offline viewer builds front_camera_mask.",
)
flags.DEFINE_string(
    "phase_ranges_path",
    "",
    "JSON file with offline phase frame ranges. Supports {'place': [[start, end]], "
    "{'pick': [[start, end]], 'place': [[start, end]]}, or [[start, end]] for place.",
)
flags.DEFINE_string(
    "mask_predictor_checkpoint_path",
    str(REPO_ROOT / "examples" / "gaze_data_process" / "SAM_process" / "mask_predictor_ckpt" / "best.pt"),
    "Mask predictor checkpoint used by pick_only/place_only modes.",
)
flags.DEFINE_string("gaze_json_name", "gaze_contact.json", "Recorded gaze json name.")
flags.DEFINE_string(
    "attention_keys",
    "front_camera,tactile_data",
    "Comma-separated image keys whose critic attention should be rendered.",
)
flags.DEFINE_float(
    "viewer_mask_grounding_threshold",
    0.05,
    "High-resolution mask threshold used by this viewer for offline grounding metrics.",
)
flags.DEFINE_float(
    "viewer_mask_grounding_cell_threshold",
    0.01,
    "Minimum per-cell mask occupancy used by this viewer for offline grounding metrics.",
)
flags.DEFINE_enum(
    "attention_display_mode",
    "both",
    ["heatmap", "prob", "both"],
    "Render raw min-max attention heatmaps, softmax probability maps, or both.",
)
flags.DEFINE_float(
    "attention_logit_floor",
    0.0,
    "For logits heatmaps, values at or below this raw value render as no attention.",
)
flags.DEFINE_float(
    "attention_probability_floor",
    0.01,
    "For probability maps, probabilities below this value render as no attention.",
)
flags.DEFINE_integer("output_scale", 3, "Scale rendered panels for easier viewing.")
flags.DEFINE_float("text_scale", 0.42, "Panel text scale after output scaling.")
flags.DEFINE_integer(
    "attention_panels_per_row",
    2,
    "Number of attention panels per output row.",
)
flags.DEFINE_string(
    "output_dir",
    "modality_critic_attention_outputs",
    "Folder for rendered attention images.",
)
flags.DEFINE_string("robot_urdf_path", DEFAULT_ROBOT_URDF_PATH, "Robot URDF path.")


def print_green(message):
    print(f"\033[92m {message}\033[00m")


def print_red(message):
    print(f"\033[91m {message}\033[00m")


def frame_id(frame_dir: Path):
    return int(frame_dir.name.replace("frame_", ""))


def list_frame_dirs(frame_root: Path):
    frame_dirs = []
    for frame_dir in frame_root.glob("frame_*"):
        if not frame_dir.is_dir():
            continue
        try:
            fid = frame_id(frame_dir)
        except ValueError:
            continue
        if fid >= FLAGS.start_frame:
            frame_dirs.append(frame_dir)
    frame_dirs = sorted(frame_dirs, key=frame_id)
    if FLAGS.frame_stride > 1:
        frame_dirs = frame_dirs[:: FLAGS.frame_stride]
    if FLAGS.max_frames > 0:
        frame_dirs = frame_dirs[: FLAGS.max_frames]
    return frame_dirs


def read_numeric_file(path: Path, default=None, dtype=np.float32):
    if not path.exists():
        if default is None:
            raise FileNotFoundError(f"Missing numeric file: {path}")
        return np.asarray(default, dtype=dtype).reshape(-1)
    return np.asarray(np.loadtxt(path, dtype=dtype), dtype=dtype).reshape(-1)


def read_rgb(frame_dir: Path):
    image_bgr = cv2.imread(str(frame_dir / "color_image.jpg"))
    if image_bgr is None:
        raise FileNotFoundError(f"Missing color_image.jpg in {frame_dir}")
    image_bgr = cv2.resize(
        image_bgr,
        (FLAGS.image_size, FLAGS.image_size),
        interpolation=cv2.INTER_LINEAR,
    )
    return image_bgr[..., ::-1].astype(np.uint8)


def read_tactile_image(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Missing tactile image: {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]
    return cv2.resize(
        image.astype(np.uint8, copy=False),
        (FLAGS.image_size, FLAGS.image_size),
        interpolation=cv2.INTER_LINEAR,
    )


def read_tactile(frame_dir: Path):
    candidates = [
        (frame_dir / "thumb_depth_image.png", frame_dir / "index_depth_image.png"),
        (frame_dir / "thumb_heat_map.jpg", frame_dir / "index_heat_map.jpg"),
    ]
    for thumb_path, index_path in candidates:
        if thumb_path.exists() and index_path.exists():
            thumb = read_tactile_image(thumb_path)
            index = read_tactile_image(index_path)
            return np.concatenate([thumb, index], axis=1).astype(np.uint8)
    raise FileNotFoundError(f"Missing tactile images in {frame_dir}")


def read_mask(mask_path: Path, target_shape):
    if not mask_path.exists():
        return np.zeros(target_shape, dtype=bool)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros(target_shape, dtype=bool)
    mask = cv2.resize(
        mask,
        (target_shape[1], target_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return mask > 0


def mask_paths(frame_dir: Path):
    fid = frame_id(frame_dir)
    central_names = {
        "mask1": (f"frame_{fid:06d}_ball_mask.png", f"frame_{fid:06d}_mask1.png"),
        "mask2": (f"frame_{fid:06d}_basket_mask.png", f"frame_{fid:06d}_mask2.png"),
    }
    root = frame_dir.parent
    return {
        "mask1": [
            frame_dir / "ball_mask.png",
            frame_dir / "mask1.png",
            frame_dir / "rs_mask_obj0.png",
            *(root / "sam_masks" / name for name in central_names["mask1"]),
            *(root / "sam_masks" / "propagated" / name for name in central_names["mask1"]),
        ],
        "mask2": [
            frame_dir / "basket_mask.png",
            frame_dir / "mask2.png",
            frame_dir / "rs_mask_obj1.png",
            *(root / "sam_masks" / name for name in central_names["mask2"]),
            *(root / "sam_masks" / "propagated" / name for name in central_names["mask2"]),
        ],
    }


def first_existing_mask(frame_dir: Path, slot: str, target_shape):
    for mask_path in mask_paths(frame_dir)[slot]:
        if mask_path.exists():
            return read_mask(mask_path, target_shape)
    return np.zeros(target_shape, dtype=bool)


def has_recorded_mask(frame_dir: Path, slot: str):
    return any(mask_path.exists() for mask_path in mask_paths(frame_dir)[slot])


def has_any_recorded_mask(frame_dirs, slot: str):
    return any(has_recorded_mask(frame_dir, slot) for frame_dir in frame_dirs)


def dilate_mask(mask, radius: int):
    if radius <= 0:
        return mask
    kernel_size = int(radius) * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def mask_bbox(mask):
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def mask_bbox_inside(inner_mask, outer_mask, margin_px=2):
    inner_bbox = mask_bbox(inner_mask)
    outer_bbox = mask_bbox(outer_mask)
    if inner_bbox is None or outer_bbox is None:
        return False
    ix0, iy0, ix1, iy1 = inner_bbox
    ox0, oy0, ox1, oy1 = outer_bbox
    margin_px = int(margin_px)
    return (
        ix0 >= ox0 - margin_px
        and iy0 >= oy0 - margin_px
        and ix1 <= ox1 + margin_px
        and iy1 <= oy1 + margin_px
    )


def read_recorded_gaze_xy(frame_dir: Path):
    gaze_path = frame_dir / FLAGS.gaze_json_name
    if not gaze_path.exists():
        return None
    gaze_data = json.loads(gaze_path.read_text())
    gaze_uv = gaze_data.get("gaze_uv_in_realsense")
    realsense_size = gaze_data.get("realsense_size")
    if gaze_uv is None or realsense_size is None:
        return None
    width, height = float(realsense_size[0]), float(realsense_size[1])
    return np.asarray(
        [
            np.clip(float(gaze_uv[0]) / max(width, 1e-6), 0.0, 1.0),
            np.clip(float(gaze_uv[1]) / max(height, 1e-6), 0.0, 1.0),
        ],
        dtype=np.float32,
    )


def phase_onehot(selected_index):
    # Delegate to the shared implementation so this cannot drift from the
    # width the env wrappers append -- a local copy is how demos would end up
    # one column wider than the observation space they are replayed into.
    return gaze_phase_onehot(selected_index)


def zero_action_rpy(action):
    action = np.asarray(action, dtype=np.float32).copy()
    if action.shape[-1] >= 6:
        action[..., 3:6] = 0.0
    return action


def read_gaze_target_mask(frame_dir: Path, target_shape):
    gaze_xy = read_recorded_gaze_xy(frame_dir)
    if gaze_xy is None:
        return np.zeros(target_shape, dtype=bool), None, "none"

    masks = [
        first_existing_mask(frame_dir, "mask1", target_shape),
        first_existing_mask(frame_dir, "mask2", target_shape),
    ]
    height, width = target_shape
    gaze_x = int(round(float(gaze_xy[0]) * (width - 1)))
    gaze_y = int(round(float(gaze_xy[1]) * (height - 1)))
    search_masks = [dilate_mask(mask, FLAGS.gaze_target_mask_dilation) for mask in masks]
    candidate_indices = [
        index for index, mask in enumerate(search_masks) if bool(mask[gaze_y, gaze_x])
    ]

    if not candidate_indices:
        selected_index = 0
        selected = masks[selected_index]
        selected_slot = "mask1"
    elif len(candidate_indices) == 1:
        selected_index = candidate_indices[0]
        selected = masks[selected_index]
        selected_slot = f"mask{selected_index + 1}"
    elif 0 in candidate_indices and 1 in candidate_indices and mask_bbox_inside(
        masks[0],
        masks[1],
    ):
        selected_index = 1
        selected = masks[1]
        selected_slot = "mask2"
    elif candidate_indices:
        selected_index = max(candidate_indices, key=lambda index: int(masks[index].sum()))
        selected = masks[selected_index]
        selected_slot = f"mask{selected_index + 1}"

    return selected, selected_index, selected_slot


def read_front_camera_mask(frame_dir: Path, rgb_image, target_shape):
    selected_mask, selected_index, selected_slot = read_gaze_target_mask(
        frame_dir,
        target_shape,
    )
    mask_image = np.repeat(
        (selected_mask.astype(np.uint8)[..., None] * 255),
        3,
        axis=-1,
    )
    return mask_image, phase_onehot(selected_index), selected_slot


def read_mask_image(frame_dir: Path, slot: str, target_shape):
    mask = first_existing_mask(frame_dir, slot, target_shape)
    return np.repeat((mask.astype(np.uint8)[..., None] * 255), 3, axis=-1)


def set_phase_in_obs(obs, selected_index):
    obs = dict(obs)
    phase = phase_onehot(selected_index)
    state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
    if state.shape[0] >= 11:
        state = np.concatenate([state[:-3], phase], axis=0)
    else:
        state = np.concatenate([state, phase], axis=0)
    obs["state"] = state.astype(np.float32)
    return obs


def selected_slot_to_phase_index(selected_slot):
    if selected_slot == "mask1":
        return 0
    if selected_slot == "mask2":
        return 1
    return 2


def infer_actor_state_dim(agent):
    try:
        kernel = agent.state.params["modules_actor"]["encoder"]["Dense_0"]["kernel"]
    except Exception:
        return None
    return int(kernel.shape[0])


def align_obs_state_dim(obs, expected_dim, selected_slot):
    if expected_dim is None:
        return obs
    obs = dict(obs)
    state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
    if state.shape[0] == expected_dim:
        return obs
    if state.shape[0] + 3 == expected_dim:
        state = np.concatenate(
            [state, phase_onehot(selected_slot_to_phase_index(selected_slot))],
            axis=0,
        )
    elif state.shape[0] > expected_dim:
        state = state[:expected_dim]
    else:
        state = np.pad(state, (0, expected_dim - state.shape[0]))
    obs["state"] = state.astype(np.float32)
    return obs


def apply_predicted_mask_to_obs(obs, mask_predictor, selected_index, target_shape):
    if mask_predictor is None:
        raise RuntimeError("force_mask mode requires a loaded mask predictor.")
    all_fields = compute_all_index_target_mask_fields(
        obs,
        mask_predictor,
        target_shape,
    )
    for slot_name, image_key in (
        ("mask1", "front_camera_mask1"),
        ("mask2", "front_camera_mask2"),
    ):
        slot_fields = all_fields.get(slot_name)
        if slot_fields is None:
            continue
        obs = add_gaze_mask_image_to_obs(
            obs,
            gaze_target_mask=slot_fields["gaze_target_mask"],
            image_key=image_key,
            reference_key="front_camera",
        )
    fields = compute_index_target_mask_fields(
        obs,
        mask_predictor,
        target_shape,
        selected_mask_index=selected_index,
    )
    obs = add_gaze_mask_image_to_obs(
        obs,
        gaze_target_mask=fields["gaze_target_mask"],
        image_key="front_camera_mask",
        reference_key="front_camera",
    )
    obs = set_phase_in_obs(obs, selected_index)
    return obs, fields.get("selected_mask_slot", f"mask{selected_index + 1}")


def apply_recorded_mask_selection_to_obs(obs, frame_dir: Path, selected_index, target_shape):
    obs = dict(obs)
    selected_slot = f"mask{int(selected_index) + 1}"
    for slot_name, image_key in (
        ("mask1", "front_camera_mask1"),
        ("mask2", "front_camera_mask2"),
    ):
        if image_key not in obs:
            obs[image_key] = read_mask_image(frame_dir, slot_name, target_shape)
    obs["front_camera_mask"] = obs.get(
        f"front_camera_mask{int(selected_index) + 1}",
        read_mask_image(frame_dir, selected_slot, target_shape),
    )
    obs = set_phase_in_obs(obs, selected_index)
    return obs, selected_slot


def normalize_phase_ranges(raw_ranges):
    if raw_ranges is None:
        return []
    ranges = []
    for item in raw_ranges:
        if isinstance(item, dict):
            start = item.get("start", item.get("from"))
            end = item.get("end", item.get("to"))
        else:
            start, end = item[:2]
        ranges.append((int(start), int(end)))
    return ranges


def load_phase_ranges(path):
    if not path:
        return {"pick": [], "place": [], "none": []}
    path = Path(path).expanduser().resolve()
    with path.open("r") as f:
        data = json.load(f)
    if isinstance(data, list):
        data = {"place": data}
    return {
        "pick": normalize_phase_ranges(data.get("pick")),
        "place": normalize_phase_ranges(data.get("place")),
        "none": normalize_phase_ranges(data.get("none")),
    }


def frame_in_ranges(fid, ranges):
    return any(start <= fid <= end for start, end in ranges)


def phase_ranges_selected_index(fid, ranges):
    if frame_in_ranges(fid, ranges.get("place", [])):
        return 1
    if frame_in_ranges(fid, ranges.get("pick", [])):
        return 0
    if frame_in_ranges(fid, ranges.get("none", [])):
        return 2
    return 0


def resolve_mask_selection_mode(exp_name):
    if FLAGS.mask_selection_mode != "auto":
        if FLAGS.mask_selection_mode == "force_mask1":
            print_green("[warn] mask_selection_mode=force_mask1 is deprecated; use pick_only.")
            return "pick_only"
        if FLAGS.mask_selection_mode == "force_mask2":
            print_green("[warn] mask_selection_mode=force_mask2 is deprecated; use place_only.")
            return "place_only"
        return FLAGS.mask_selection_mode
    if exp_name == "tennis_ball_pick":
        return "pick_only"
    return "recorded_gaze"


def read_ee_pose(frame_dir: Path):
    ee_pose_path = frame_dir / "robot_ee_pose.txt"
    if ee_pose_path.exists():
        ee_pose = read_numeric_file(ee_pose_path, dtype=np.float32)
        if ee_pose.shape[0] >= 7:
            return ee_pose[:3], ee_pose[3:7]

    from examples.utils import kinematics_utils

    joints = read_numeric_file(frame_dir / "right_arm_joint.txt", dtype=np.float64)
    tcp_pos, tcp_ori = kinematics_utils.comupute_forward_kinematics(
        joints,
        FLAGS.robot_urdf_path,
    )
    return np.asarray(tcp_pos, dtype=np.float32), np.asarray(tcp_ori, dtype=np.float32)


def read_frame_observation_and_action(frame_dir: Path, image_keys):
    obs = {}
    selected_slot = "none"
    gaze_phase = None
    rgb_image = None
    target_shape = (FLAGS.image_size, FLAGS.image_size)
    for image_key in image_keys:
        if image_key == "front_camera":
            rgb_image = read_rgb(frame_dir)
            obs[image_key] = rgb_image
        elif image_key == "tactile_data":
            obs[image_key] = read_tactile(frame_dir)
        elif image_key == "front_camera_mask":
            if rgb_image is None:
                rgb_image = read_rgb(frame_dir)
            obs[image_key], gaze_phase, selected_slot = read_front_camera_mask(
                frame_dir,
                rgb_image,
                target_shape,
            )
        elif image_key == "front_camera_mask1":
            obs[image_key] = read_mask_image(frame_dir, "mask1", target_shape)
        elif image_key == "front_camera_mask2":
            obs[image_key] = read_mask_image(frame_dir, "mask2", target_shape)
        else:
            raise ValueError(f"Unsupported image_key={image_key}")
    if "front_camera_mask" in image_keys and gaze_phase is None:
        if rgb_image is None:
            rgb_image = read_rgb(frame_dir)
        obs["front_camera_mask"], gaze_phase, selected_slot = read_front_camera_mask(
            frame_dir,
            rgb_image,
            target_shape,
        )

    tcp_pos, tcp_ori = read_ee_pose(frame_dir)
    hand_state = read_numeric_file(
        frame_dir / "hand_state.txt",
        default=[0.0],
        dtype=np.float32,
    )[:1]
    state_parts = [
        np.asarray(tcp_pos, dtype=np.float32).reshape(-1),
        np.asarray(tcp_ori, dtype=np.float32).reshape(-1),
        np.asarray(hand_state, dtype=np.float32).reshape(-1),
    ]
    if gaze_phase is not None:
        state_parts.append(gaze_phase)
    obs["state"] = np.concatenate(state_parts, axis=0).astype(np.float32)

    action = read_numeric_file(
        frame_dir / "action.txt",
        default=np.zeros(7, dtype=np.float32),
        dtype=np.float32,
    )
    if action.shape[0] < 7:
        action = np.pad(action, (0, 7 - action.shape[0]))
    return obs, zero_action_rpy(action[:7]), selected_slot


def attention_to_prob(attention_map):
    values = np.asarray(attention_map, dtype=np.float32)
    while values.ndim > 2:
        values = values.reshape((-1, *values.shape[-2:]))[0]
    flat = values.reshape(-1)
    flat = flat - np.max(flat)
    prob = np.exp(flat)
    prob = prob / max(float(np.sum(prob)), 1e-8)
    return prob.reshape(values.shape)


def normalize_heatmap(heatmap, floor_value=None):
    heatmap = np.asarray(heatmap, dtype=np.float32)
    while heatmap.ndim > 2:
        heatmap = heatmap.reshape((-1, *heatmap.shape[-2:]))[0]
    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
    if floor_value is None:
        heatmap = heatmap - np.min(heatmap)
    else:
        heatmap = np.maximum(heatmap - float(floor_value), 0.0)
    denom = np.max(heatmap)
    if denom > 1e-8:
        heatmap = heatmap / denom
    return heatmap


def resize_mask_to_attention(mask_image, attention_shape):
    mask = np.asarray(mask_image, dtype=np.float32)
    while mask.ndim > 3:
        mask = mask.reshape((-1, *mask.shape[-3:]))[0]
    if mask.ndim == 3:
        mask = np.max(mask, axis=-1)
    elif mask.ndim != 2:
        raise ValueError(f"Unsupported mask image shape: {mask_image.shape}")
    if np.max(mask) > 1.0:
        mask = mask / 255.0
    mask = mask > float(FLAGS.viewer_mask_grounding_threshold)

    attention_h, attention_w = int(attention_shape[0]), int(attention_shape[1])
    source_h, source_w = mask.shape
    block_h = int(np.ceil(source_h / max(attention_h, 1)))
    block_w = int(np.ceil(source_w / max(attention_w, 1)))
    padded_h = attention_h * block_h
    padded_w = attention_w * block_w
    padded = np.pad(
        mask,
        ((0, padded_h - source_h), (0, padded_w - source_w)),
        mode="constant",
        constant_values=False,
    )
    pooled = padded.reshape(attention_h, block_h, attention_w, block_w)
    occupancy = np.mean(pooled.astype(np.float32), axis=(1, 3))
    return (
        occupancy > float(FLAGS.viewer_mask_grounding_cell_threshold)
    ).astype(np.float32)


def attention_mask_metrics(attention_map, mask_image):
    attention_prob = attention_to_prob(attention_map)
    mask = resize_mask_to_attention(mask_image, attention_prob.shape)
    mask_binary = mask > 0.5
    mask_cell_fraction = float(np.mean(mask_binary))
    if not np.any(mask_binary):
        return {
            "mask_mass": 0.0,
            "mask_grounding_loss": np.nan,
            "mask_cell_fraction": mask_cell_fraction,
        }
    mask_mass = float(np.sum(attention_prob * mask_binary.astype(np.float32)))
    gaze_distribution = mask_binary.astype(np.float32)
    gaze_distribution /= max(float(np.sum(gaze_distribution)), 1e-8)
    mask_grounding_loss = float(
        np.sum(
            gaze_distribution
            * (
                np.log(gaze_distribution + 1e-8)
                - np.log(attention_prob + 1e-8)
            )
        )
    )
    return {
        "mask_mass": mask_mass,
        "mask_grounding_loss": mask_grounding_loss,
        "mask_cell_fraction": mask_cell_fraction,
    }


def draw_text_panel(image, lines):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = FLAGS.text_scale
    thickness = 1
    margin = 6
    line_gap = 4
    text_sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    panel_width = max(width for width, _ in text_sizes) + 2 * margin
    panel_height = (
        sum(height for _, height in text_sizes)
        + line_gap * (len(lines) - 1)
        + 2 * margin
    )
    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (min(panel_width, image.shape[1] - 1), min(panel_height, image.shape[0] - 1)),
        (0, 0, 0),
        -1,
    )
    image[:] = cv2.addWeighted(overlay, 0.55, image, 0.45, 0.0)
    y = margin
    for line, (_, height) in zip(lines, text_sizes):
        y += height
        cv2.putText(
            image,
            line,
            (margin, y),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += line_gap


def render_attention_panel(image_rgb, attention_map, title, extra_lines):
    heatmap = normalize_heatmap(
        attention_map,
        floor_value=FLAGS.attention_logit_floor,
    )
    heatmap = cv2.resize(
        heatmap,
        (image_rgb.shape[1], image_rgb.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    heatmap_color = cv2.applyColorMap(
        (255.0 * heatmap).astype(np.uint8),
        cv2.COLORMAP_JET,
    )
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(image_bgr, 0.62, heatmap_color, 0.38, 0.0)
    if FLAGS.output_scale > 1:
        overlay = cv2.resize(
            overlay,
            (overlay.shape[1] * FLAGS.output_scale, overlay.shape[0] * FLAGS.output_scale),
            interpolation=cv2.INTER_NEAREST,
        )
    draw_text_panel(overlay, [title, *extra_lines])
    return overlay


def render_attention_probability_panel(image_rgb, attention_map, title, extra_lines):
    probability = attention_to_prob(attention_map)
    probability = np.where(
        probability >= float(FLAGS.attention_probability_floor),
        probability,
        0.0,
    )
    probability = cv2.resize(
        probability,
        (image_rgb.shape[1], image_rgb.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    probability_color = cv2.applyColorMap(
        np.clip(255.0 * probability / max(float(np.max(probability)), 1e-8), 0, 255).astype(
            np.uint8
        ),
        cv2.COLORMAP_JET,
    )
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(image_bgr, 0.62, probability_color, 0.38, 0.0)
    if FLAGS.output_scale > 1:
        overlay = cv2.resize(
            overlay,
            (overlay.shape[1] * FLAGS.output_scale, overlay.shape[0] * FLAGS.output_scale),
            interpolation=cv2.INTER_NEAREST,
        )
    draw_text_panel(overlay, [title, *extra_lines])
    return overlay


def render_feature_vector_panel(feature_vector, title, extra_lines):
    """Render the actual vector sent from one image branch before the MLP.

    After spatial pooling and Dense fusion there is no one-to-one pixel
    correspondence anymore, so this is intentionally shown as a vector strip
    instead of an RGB overlay.
    """
    vector = np.asarray(feature_vector, dtype=np.float32).reshape(-1)
    width, height = 512, 180
    panel = np.full((height, width, 3), 245, dtype=np.uint8)
    margin_x, margin_y = 18, 18
    usable_w = width - 2 * margin_x
    usable_h = height - 2 * margin_y
    scale = max(float(np.max(np.abs(vector))), 1e-8)
    points = []
    for index, value in enumerate(vector):
        x = margin_x + int(index * (usable_w - 1) / max(vector.size - 1, 1))
        y = margin_y + int((0.5 - 0.45 * float(value) / scale) * usable_h)
        points.append((x, int(np.clip(y, margin_y, height - margin_y))))
    if points:
        cv2.polylines(
            panel,
            [np.asarray(points, dtype=np.int32)],
            isClosed=False,
            color=(40, 40, 40),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    zero_y = margin_y + usable_h // 2
    cv2.line(panel, (margin_x, zero_y), (width - margin_x, zero_y), (180, 180, 180), 1)
    cv2.line(panel, (margin_x, margin_y), (margin_x, height - margin_y), (180, 180, 180), 1)
    if FLAGS.output_scale > 1:
        panel = cv2.resize(
            panel,
            (panel.shape[1] * FLAGS.output_scale, panel.shape[0] * FLAGS.output_scale),
            interpolation=cv2.INTER_NEAREST,
        )
    draw_text_panel(panel, [title, *extra_lines])
    return panel


def panel_base_image(obs, attention_key, attention_kind):
    if attention_kind == "mask_encoder_feature":
        if "front_camera_mask" in obs:
            return obs["front_camera_mask"]
        if "front_camera_mask1" in obs:
            return obs["front_camera_mask1"]
    return obs[attention_key]


def is_vit_encoder(config):
    return getattr(config, "encoder_type", "") in ("vit", "vit-small")


def effective_mask_suppress_beta(config):
    return 1.0 if getattr(config, "encoder_type", "") == "resnet-pretrained" else 0.0


def effective_use_mask_feature_head(config):
    return getattr(config, "encoder_type", "") == "resnet-pretrained"


def attention_kind_name(config, attention_key, *, return_raw_attention):
    if is_vit_encoder(config) and return_raw_attention:
        return "vit_patch_feature"
    if attention_key == "front_camera" and not return_raw_attention:
        return "fused_feature"
    return "raw_feature"


def create_attention_agent(
    config,
    sample_obs,
    attention_key,
    *,
    return_raw_attention,
    return_mask_encoder_attention=False,
    return_feature_debug=False,
):
    return make_gaze_sac_pixel_agent_hybrid_single_arm(
        seed=FLAGS.seed,
        sample_obs=sample_obs,
        sample_action=np.zeros(7, dtype=np.float32),
        image_keys=config.image_keys,
        encoder_type=config.encoder_type,
        discount=config.discount,
        gaze_regularization_weight=0.0,
        gaze_heatmap_size=(FLAGS.image_size, FLAGS.image_size),
        gaze_region_radius=0,
        gaze_attention_image_key=attention_key,
        mask_feature_gate_alpha=getattr(config, "mask_feature_gate_alpha", 0.9),
        mask_feature_min_gate=getattr(config, "mask_feature_min_gate", 0.4),
        mask_pick_place_phase_control=getattr(
            config,
            "mask_pick_place_phase_control",
            False,
        ),
        return_raw_attention=return_raw_attention,
        return_mask_encoder_attention=return_mask_encoder_attention,
        return_feature_debug=return_feature_debug,
    )


def resolve_checkpoint_path():
    checkpoint_root = Path(FLAGS.checkpoint_path).expanduser().resolve()
    if not checkpoint_root.exists():
        raise FileNotFoundError(
            f"checkpoint_path does not exist: {checkpoint_root}\n"
            "Please pass the exact run directory that contains checkpoint_* folders."
        )
    if not checkpoint_root.is_dir():
        raise NotADirectoryError(
            f"checkpoint_path is not a directory: {checkpoint_root}"
        )

    step = None if FLAGS.checkpoint_step < 0 else FLAGS.checkpoint_step
    if step is None:
        resolved_checkpoint = checkpoints.latest_checkpoint(str(checkpoint_root))
        if resolved_checkpoint is None:
            raise FileNotFoundError(
                f"No checkpoint found under: {checkpoint_root}\n"
                "Expected at least one checkpoint_* directory/file. "
                "Use --checkpoint_path=<run_dir> or --skip_restore=True."
            )
        return checkpoint_root, resolved_checkpoint, step

    resolved_checkpoint = checkpoint_root / f"checkpoint_{step}"
    if not resolved_checkpoint.exists():
        raise FileNotFoundError(
            f"Requested checkpoint step does not exist: {resolved_checkpoint}\n"
            "Check --checkpoint_step or use --checkpoint_step=-1 for latest."
        )
    return checkpoint_root, str(resolved_checkpoint), step


def critic_input_dim_from_agent(agent):
    try:
        kernel = agent.state.params["modules_critic"]["network"]["VmapMLP_0"][
            "Dense_0"
        ]["kernel"]
    except Exception:
        return None
    shape = tuple(np.asarray(kernel).shape)
    return int(shape[-2]) if len(shape) >= 2 else None


def actor_input_dim_from_agent(agent):
    try:
        kernel = agent.state.params["modules_actor"]["network"]["Dense_0"]["kernel"]
    except Exception:
        return None
    shape = tuple(np.asarray(kernel).shape)
    return int(shape[-2]) if len(shape) >= 2 else None


def checkpoint_param_shape(checkpoint_dir, param_path):
    metadata_path = Path(checkpoint_dir) / "_METADATA"
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text()).get("tree_metadata", {})
    for key, value in metadata.items():
        try:
            parsed_key = ast.literal_eval(key)
        except Exception:
            continue
        if parsed_key == param_path:
            return tuple(value["value_metadata"]["write_shape"])
    return None


def restore_agent(agent, *, label, allow_fallback=False):
    if FLAGS.skip_restore:
        print_green(
            f"skip_restore=True: using initialized agent for {label} "
            "with loaded ResNet backbone only."
        )
        return agent
    checkpoint_root, resolved_checkpoint, step = resolve_checkpoint_path()
    current_critic_dim = critic_input_dim_from_agent(agent)
    current_actor_dim = actor_input_dim_from_agent(agent)
    ckpt_critic_shape = checkpoint_param_shape(
        resolved_checkpoint,
        (
            "params",
            "modules_critic",
            "network",
            "VmapMLP_0",
            "Dense_0",
            "kernel",
        ),
    )
    ckpt_actor_shape = checkpoint_param_shape(
        resolved_checkpoint,
        ("params", "modules_actor", "network", "Dense_0", "kernel"),
    )
    print_green(
        f"restoring {label} checkpoint="
        f"{resolved_checkpoint if resolved_checkpoint is not None else 'None'}"
    )
    print_green(
        f"restore dims {label}: "
        f"current_actor_in={current_actor_dim} ckpt_actor_shape={ckpt_actor_shape} "
        f"current_critic_in={current_critic_dim} ckpt_critic_shape={ckpt_critic_shape}"
    )
    if (
        ckpt_critic_shape is not None
        and current_critic_dim is not None
        and int(ckpt_critic_shape[-2]) != int(current_critic_dim)
    ):
        raise ValueError(
            "Checkpoint/model critic input dimension mismatch before restore: "
            f"label={label} current={current_critic_dim} "
            f"checkpoint={ckpt_critic_shape[-2]} checkpoint_path={resolved_checkpoint}. "
            "This means the viewer was launched with a different experiment/config "
            "or structural flags than the checkpoint was trained with."
        )
    try:
        restored_state = checkpoints.restore_checkpoint(
            str(checkpoint_root),
            agent.state,
            step=step,
        )
    except Exception as exc:
        if not allow_fallback:
            raise
        print_red(
            f"[warn] restore failed for {label}; using initialized raw encoder. "
            f"reason={exc}"
        )
        return agent
    return agent.replace(state=restored_state)


def critic_attention_for_frame(agent, obs, action):
    critic_action = np.asarray(action, dtype=np.float32)[..., :-1]
    attention_map = agent.forward_gaze_attention(
        jax.device_put(obs),
        jax.device_put(critic_action),
        rng=None,
        train=False,
    )
    attention_map = np.asarray(jax.device_get(attention_map))
    if attention_map.ndim > 2:
        attention_map = attention_map.reshape((-1, *attention_map.shape[-2:]))[0]
    return attention_map


def critic_feature_debug_for_frame(agent, obs, action):
    critic_action = np.asarray(action, dtype=np.float32)[..., :-1]
    feature_debug = agent.forward_feature_debug(
        jax.device_put(obs),
        jax.device_put(critic_action),
    )
    return jax.device_get(feature_debug)


def critic_fused_attribution_for_frame(agent, obs, action, image_key):
    critic_action = np.asarray(action, dtype=np.float32)[..., :-1]
    attribution = agent.forward_fused_attribution(
        jax.device_put(obs),
        jax.device_put(critic_action),
        image_key=image_key,
    )
    return np.asarray(jax.device_get(attribution))


def find_param_by_suffix(tree, suffix):
    """Find a checkpoint parameter without depending on Flax module prefixes."""
    def visit(node, path=()):
        if isinstance(node, Mapping):
            for key, value in node.items():
                result = visit(value, path + (str(key),))
                if result is not None:
                    return result
        elif hasattr(node, "shape") and "/".join(path).endswith(suffix):
            return np.asarray(node)
        return None

    return visit(tree)


def fused_feature_proxy(feature_debug, agent, image_key):
    """Approximate the fused vector's spatial contribution using ckpt weights."""
    debug = feature_debug["features"][image_key]
    raw_map = np.asarray(debug["raw_spatial"], dtype=np.float32)
    head_map = debug.get("head_spatial")
    raw_vector = debug.get("raw_vector")
    head_vector = debug.get("head_vector")
    if head_map is None or raw_vector is None or head_vector is None:
        return raw_map

    raw_vector = np.asarray(raw_vector, dtype=np.float32).reshape(-1)
    head_vector = np.asarray(head_vector, dtype=np.float32).reshape(-1)
    kernel = find_param_by_suffix(
        agent.state.params,
        "mask_visual_fusion_front_camera_proj/kernel",
    )
    if kernel is not None and kernel.ndim == 2 and kernel.shape[0] >= 2 * raw_vector.size:
        raw_contribution = raw_vector @ kernel[: raw_vector.size]
        head_contribution = head_vector @ kernel[raw_vector.size : 2 * raw_vector.size]
        raw_weight = float(np.mean(np.abs(raw_contribution)))
        head_weight = float(np.mean(np.abs(head_contribution)))
    else:
        raw_weight = float(np.linalg.norm(raw_vector))
        head_weight = float(np.linalg.norm(head_vector))

    def normalize_map(value):
        value = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
        return value / max(float(np.mean(value)), 1e-8)

    return raw_weight * normalize_map(raw_map) + head_weight * normalize_map(head_map)


def pad_to_same_height(panels):
    max_height = max(panel.shape[0] for panel in panels)
    padded = []
    for panel in panels:
        if panel.shape[0] == max_height:
            padded.append(panel)
            continue
        pad = np.zeros((max_height - panel.shape[0], panel.shape[1], 3), dtype=panel.dtype)
        padded.append(np.concatenate([panel, pad], axis=0))
    return padded


def pad_to_same_width(panels):
    max_width = max(panel.shape[1] for panel in panels)
    padded = []
    for panel in panels:
        if panel.shape[1] == max_width:
            padded.append(panel)
            continue
        pad = np.zeros((panel.shape[0], max_width - panel.shape[1], 3), dtype=panel.dtype)
        padded.append(np.concatenate([panel, pad], axis=1))
    return padded


def make_panel_grid(panels, panels_per_row):
    if not panels:
        raise ValueError("No panels to render.")
    panels_per_row = max(1, int(panels_per_row))
    rows = []
    for start in range(0, len(panels), panels_per_row):
        row_panels = pad_to_same_height(panels[start : start + panels_per_row])
        rows.append(np.concatenate(row_panels, axis=1))
    rows = pad_to_same_width(rows)
    return np.concatenate(rows, axis=0)


def main(_):
    if FLAGS.exp_name not in NEW_MAPPING:
        raise ValueError(f"Unknown exp_name={FLAGS.exp_name}")
    config = NEW_MAPPING[FLAGS.exp_name]()
    # The checkpoint architecture is selected explicitly at runtime. The task
    # config still supplies observation and phase/mask behavior.
    config.encoder_type = (
        "resnet-pretrained"
        if FLAGS.encoder_type == "pretrained_resnet"
        else "vit"
    )
    mask_selection_mode = resolve_mask_selection_mode(FLAGS.exp_name)
    config.image_keys = list(
        config.get_image_keys(
            enable_tactile=bool(FLAGS.enable_tactile),
            use_gaze_target_mask=True,
        )
    )
    attention_keys = [
        key.strip()
        for key in FLAGS.attention_keys.split(",")
        if key.strip() in config.image_keys
        and key.strip()
        not in {"front_camera_mask", "front_camera_mask1", "front_camera_mask2"}
    ]
    if not attention_keys:
        raise ValueError(f"No valid attention_keys in {FLAGS.attention_keys}")
    phase_control_enabled = bool(
        getattr(config, "mask_pick_place_phase_control", False)
    )
    if mask_selection_mode == "place_only" and not phase_control_enabled:
        raise ValueError(
            "mask_selection_mode=place_only requires an experiment with "
            "mask_pick_place_phase_control=True. Did you mean "
            "--exp_name=tennis_ball_pick_and_place?"
        )

    frame_root = Path(FLAGS.frame_root).expanduser().resolve()
    output_dir = Path(FLAGS.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_dirs = list_frame_dirs(frame_root)
    if not frame_dirs:
        raise FileNotFoundError(f"No frame_* dirs found under {frame_root}")
    resolved_checkpoint = None
    if not FLAGS.skip_restore:
        _checkpoint_root, resolved_checkpoint, _checkpoint_step = resolve_checkpoint_path()
    has_recorded_mask1 = has_any_recorded_mask(frame_dirs, "mask1")

    print_green(f"frame_root={frame_root}")
    print_green(f"checkpoint_path={Path(FLAGS.checkpoint_path).expanduser().resolve()}")
    if resolved_checkpoint is not None:
        print_green(f"resolved_checkpoint={resolved_checkpoint}")
    print_green(f"image_keys={config.image_keys}")
    print_green(f"attention_keys={attention_keys}")
    print_green(f"mask_selection_mode={mask_selection_mode}")
    if mask_selection_mode == "phase_ranges":
        print_green(f"phase_ranges_path={FLAGS.phase_ranges_path}")
    print_green(f"encoder_type={FLAGS.encoder_type} ({config.encoder_type})")
    print_green(f"mask_suppress_beta={effective_mask_suppress_beta(config)}")
    print_green(
        "mask_pick_place_phase_control="
        f"{phase_control_enabled}"
    )
    print_green(f"use_mask_feature_head={effective_use_mask_feature_head(config)}")
    print_green("use_mask_encoder=True")
    print_green(f"has_recorded_mask1={has_recorded_mask1}")
    print_green(
        "mask_feature_gate_alpha="
        f"{getattr(config, 'mask_feature_gate_alpha', 0.9)}"
    )
    print_green(
        "mask_feature_min_gate="
        f"{getattr(config, 'mask_feature_min_gate', 0.4)}"
    )
    print_green("mask_grounding_key=auto(front_camera_mask1 else front_camera_mask)")
    print_green(f"viewer_mask_grounding_threshold={FLAGS.viewer_mask_grounding_threshold}")
    print_green(
        "viewer_mask_grounding_cell_threshold="
        f"{FLAGS.viewer_mask_grounding_cell_threshold}"
    )
    print_green(f"attention_display_mode={FLAGS.attention_display_mode}")
    print_green(f"frames={len(frame_dirs)} output_dir={output_dir}")

    phase_ranges = (
        load_phase_ranges(FLAGS.phase_ranges_path)
        if mask_selection_mode == "phase_ranges"
        else None
    )
    if mask_selection_mode == "phase_ranges":
        print_green(f"phase_ranges={phase_ranges}")

    sample_obs, _, _ = read_frame_observation_and_action(frame_dirs[0], config.image_keys)
    mask_predictor = None
    if mask_selection_mode in ("pick_only", "place_only"):
        mask_predictor = load_mask_predictor(
            sample_obs,
            config.image_keys,
            FLAGS.mask_predictor_checkpoint_path,
            preferred_key="front_camera",
            log_fn=print_green,
        )
        selected_index = 0 if mask_selection_mode == "pick_only" else 1
        sample_obs, _ = apply_predicted_mask_to_obs(
            sample_obs,
            mask_predictor,
            selected_index,
            (FLAGS.image_size, FLAGS.image_size),
        )
    attention_specs = []
    if is_vit_encoder(config):
        for attention_key in attention_keys:
            attention_specs.append(
                {
                    "key": attention_key,
                    "kind": "vit_feature",
                    "return_raw_attention": False,
                    "return_mask_encoder_attention": False,
                    "return_feature_debug": True,
                    "agent_kind": "feature_debug",
                    "allow_restore_fallback": False,
                }
            )
    else:
        for attention_key in attention_keys:
            attention_specs.append(
                {
                    "key": attention_key,
                    "kind": "raw_feature",
                    "return_raw_attention": False,
                    "return_mask_encoder_attention": False,
                    "return_feature_debug": True,
                    "agent_kind": "feature_debug",
                    "allow_restore_fallback": False,
                }
            )
            if attention_key == "front_camera":
                attention_specs.extend(
                    [
                        {
                            "key": attention_key,
                            "kind": "mask_head_feature",
                            "return_raw_attention": False,
                            "return_mask_encoder_attention": False,
                            "return_feature_debug": True,
                            "agent_kind": "feature_debug",
                            "allow_restore_fallback": False,
                        },
                        {
                            "key": attention_key,
                            "kind": "fused_attribution",
                            "return_raw_attention": False,
                            "return_mask_encoder_attention": False,
                            "return_feature_debug": True,
                            "agent_kind": "feature_debug",
                            "allow_restore_fallback": False,
                        },
                    ]
                )
    if not attention_specs:
        raise ValueError("No attention specs were created.")

    agents = {}
    expected_state_dims = {}
    for spec in attention_specs:
        attention_key = spec["key"]
        attention_kind = spec["kind"]
        agent_kind = spec.get("agent_kind", attention_kind)
        agent_id = (attention_key, agent_kind)
        start = time.time()
        if agent_id in agents:
            agent = agents[agent_id]
        else:
            agent = create_attention_agent(
                config,
                sample_obs,
                attention_key,
                return_raw_attention=spec["return_raw_attention"],
                return_mask_encoder_attention=spec["return_mask_encoder_attention"],
                return_feature_debug=spec.get("return_feature_debug", False),
            )
        label = f"{attention_key}/{attention_kind}"
        if agent_id not in agents:
            agent = restore_agent(
                agent,
                label=label,
                allow_fallback=spec["allow_restore_fallback"],
            )
            agent = jax.device_put(jax.tree_util.tree_map(jnp.array, agent))
            agents[agent_id] = agent
        expected_state_dims[agent_id] = infer_actor_state_dim(agent)
        print_green(
            f"loaded attention agent key={label} "
            f"state_dim={expected_state_dims[agent_id]} "
            f"time={time.time() - start:.2f}s"
        )

    summary_path = output_dir / "attention_summary.csv"
    with summary_path.open("w", newline="") as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow(
            [
                "frame_id",
                "selected_mask",
                "attention_key",
                "attention_kind",
                "attention_shape",
                "peak_cell_x",
                "peak_cell_y",
                "peak_prob",
                "mask_mass",
                "mask_grounding_loss",
                "mask_cell_fraction",
            ]
        )
        metric_rows = []

        for index, frame_dir in enumerate(frame_dirs, start=1):
            fid = frame_id(frame_dir)
            try:
                obs, action, selected_slot = read_frame_observation_and_action(
                    frame_dir,
                    config.image_keys,
                )
                if mask_selection_mode in ("pick_only", "place_only"):
                    selected_index = 0 if mask_selection_mode == "pick_only" else 1
                    obs, selected_slot = apply_predicted_mask_to_obs(
                        obs,
                        mask_predictor,
                        selected_index,
                        (FLAGS.image_size, FLAGS.image_size),
                    )
                elif mask_selection_mode == "phase_ranges":
                    selected_index = phase_ranges_selected_index(fid, phase_ranges)
                    obs, selected_slot = apply_recorded_mask_selection_to_obs(
                        obs,
                        frame_dir,
                        selected_index,
                        (FLAGS.image_size, FLAGS.image_size),
                    )
                panels = []
                for spec in attention_specs:
                    attention_key = spec["key"]
                    attention_kind = spec["kind"]
                    agent_id = (attention_key, spec.get("agent_kind", attention_kind))
                    frame_obs = align_obs_state_dim(
                        obs,
                        expected_state_dims.get(agent_id),
                        selected_slot,
                    )
                    feature_vector = None
                    full_pre_mlp_vector = None
                    if spec.get("return_feature_debug", False):
                        if attention_kind == "fused_attribution":
                            feature_debug = critic_feature_debug_for_frame(
                                agents[agent_id],
                                frame_obs,
                                action,
                            )
                            attention_map = fused_feature_proxy(
                                feature_debug,
                                agents[agent_id],
                                attention_key,
                            )
                        else:
                            feature_debug = critic_feature_debug_for_frame(
                                agents[agent_id],
                                frame_obs,
                                action,
                            )
                            image_debug = feature_debug["features"][attention_key]
                            map_name = {
                                "raw_feature": "raw_spatial",
                                "mask_head_feature": "head_spatial",
                                "vit_feature": "raw_spatial",
                            }[attention_kind]
                            attention_map = np.asarray(image_debug[map_name])
                    else:
                        attention_map = critic_attention_for_frame(
                            agents[agent_id],
                            frame_obs,
                            action,
                        )
                    if attention_map is not None and attention_map.ndim > 2:
                        attention_map = attention_map.reshape(
                            (-1, *attention_map.shape[-2:])
                        )[0]
                    if attention_map is not None:
                        attention_prob = attention_to_prob(attention_map)
                        peak_y, peak_x = np.unravel_index(
                            int(np.argmax(attention_prob)),
                            attention_prob.shape,
                        )
                        peak_prob = float(attention_prob[peak_y, peak_x])
                    else:
                        attention_prob = None
                        peak_y = peak_x = -1
                        peak_prob = np.nan
                    metrics = {
                        "mask_mass": np.nan,
                        "mask_grounding_loss": np.nan,
                        "mask_cell_fraction": np.nan,
                    }
                    metric_lines = []
                    if (
                        attention_map is not None
                        and attention_key == "front_camera"
                        and "front_camera_mask1" in obs
                    ):
                        metrics = attention_mask_metrics(
                            attention_map,
                            obs["front_camera_mask1"],
                        )
                        metric_lines = [
                            f"mask_mass={metrics['mask_mass']:.3f}",
                            (
                                "mask_loss=nan"
                                if np.isnan(metrics["mask_grounding_loss"])
                                else f"mask_loss={metrics['mask_grounding_loss']:.3f}"
                            ),
                        ]
                        metric_rows.append(
                            {
                                "frame_id": fid,
                                "selected_mask": selected_slot,
                                "attention_key": attention_key,
                                "attention_kind": attention_kind,
                                **metrics,
                            }
                        )
                    writer.writerow(
                        [
                            fid,
                            selected_slot,
                            attention_key,
                            attention_kind,
                            "none" if attention_map is None else tuple(attention_map.shape),
                            int(peak_x),
                            int(peak_y),
                            f"{peak_prob:.8f}",
                            f"{metrics['mask_mass']:.8f}",
                            (
                                "nan"
                                if np.isnan(metrics["mask_grounding_loss"])
                                else f"{metrics['mask_grounding_loss']:.8f}"
                            ),
                            f"{metrics['mask_cell_fraction']:.8f}",
                        ]
                    )
                    panel_lines = [
                        f"selected_mask={selected_slot}",
                        *metric_lines,
                    ]
                    base_image = panel_base_image(obs, attention_key, attention_kind)
                    panel_lines = []
                    if attention_map is not None and FLAGS.attention_display_mode in ("heatmap", "both"):
                        panel = render_attention_panel(
                            base_image,
                            attention_map,
                            f"frame={fid} {attention_key} {attention_kind} logits",
                            panel_lines,
                        )
                        panels.append(panel)
                    if attention_map is not None and FLAGS.attention_display_mode in ("prob", "both"):
                        panel = render_attention_probability_panel(
                            base_image,
                            attention_map,
                            f"frame={fid} {attention_key} {attention_kind} prob",
                            panel_lines,
                        )
                        panels.append(panel)

                comparison = make_panel_grid(panels, FLAGS.attention_panels_per_row)
                output_path = output_dir / f"frame_{fid:06d}_modality_attention.jpg"
                if not cv2.imwrite(str(output_path), comparison):
                    raise RuntimeError(f"cv2.imwrite failed: {output_path}")
                if index == 1 or index % 10 == 0:
                    print_green(f"[{index}/{len(frame_dirs)}] saved {output_path.name}")
            except Exception as exc:
                print_red(f"[skip] frame={fid}: {exc}")

    loss_summary_path = output_dir / "attention_loss_summary.csv"
    with loss_summary_path.open("w", newline="") as loss_summary_file:
        writer = csv.writer(loss_summary_file)
        writer.writerow(
            [
                "attention_key",
                "attention_kind",
                "frames",
                "mean_mask_mass",
                "mean_mask_grounding_loss",
                "mean_mask_cell_fraction",
            ]
        )
        grouped = {}
        for row in metric_rows:
            grouped.setdefault((row["attention_key"], row["attention_kind"]), []).append(row)
        for (attention_key, attention_kind), rows in grouped.items():
            losses = np.asarray(
                [row["mask_grounding_loss"] for row in rows],
                dtype=np.float32,
            )
            masses = np.asarray([row["mask_mass"] for row in rows], dtype=np.float32)
            fractions = np.asarray(
                [row["mask_cell_fraction"] for row in rows],
                dtype=np.float32,
            )
            writer.writerow(
                [
                    attention_key,
                    attention_kind,
                    len(rows),
                    f"{float(np.nanmean(masses)):.8f}",
                    f"{float(np.nanmean(losses)):.8f}",
                    f"{float(np.nanmean(fractions)):.8f}",
                ]
            )

    print_green(f"done. summary={summary_path}")
    print_green(f"done. loss_summary={loss_summary_path}")


if __name__ == "__main__":
    app.run(main)
