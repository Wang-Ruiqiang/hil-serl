#!/usr/bin/env python3

import csv
import json
import os
import sys
import time
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
    "tennis_ball_pick-7-14-0"
)
DEFAULT_CHECKPOINT_PATH = str(
    REPO_ROOT
    / "examples"
    / "experiments"
    / "tennis_ball_pick"
    / "2026-7-14_0_ball_pick_rl_run"
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
    load_mask_predictor,
)
from serl_launcher.utils.launcher import make_gaze_sac_pixel_agent_hybrid_single_arm


FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", "tennis_ball_pick", "Experiment name.")
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
    "auto",
    ["auto", "recorded_gaze", "force_mask1", "force_mask2"],
    "How this offline viewer builds front_camera_mask.",
)
flags.DEFINE_string(
    "mask_predictor_checkpoint_path",
    str(REPO_ROOT / "examples" / "gaze_data_process" / "SAM_process" / "mask_predictor_ckpt" / "best.pt"),
    "Mask predictor checkpoint used by force_mask1/force_mask2 modes.",
)
flags.DEFINE_float(
    "mask_suppress_beta",
    1.0,
    "Feature-space suppression strength for front_camera_mask2.",
)
flags.DEFINE_boolean(
    "use_mask_feature_head",
    True,
    "Render the trainable mask feature head instead of raw ResNet feature energy.",
)
flags.DEFINE_float(
    "mask_feature_gate_alpha",
    0.25,
    "Signed feature gating strength from the trainable mask feature head.",
)
flags.DEFINE_float(
    "mask_feature_min_gate",
    0.1,
    "Minimum multiplicative gate for non-selected RGB features.",
)
flags.DEFINE_integer(
    "mask_feature_hidden_dim",
    128,
    "Hidden channel count for the trainable mask feature head.",
)
flags.DEFINE_boolean(
    "use_mask_encoder",
    True,
    "Encode front_camera_mask1 with a small CNN and concatenate it as an extra modality.",
)
flags.DEFINE_integer(
    "mask_encoder_latent_dim",
    64,
    "Output dimension of the small CNN mask encoder.",
)
flags.DEFINE_boolean(
    "show_mask_encoder_feature",
    True,
    "Render the small CNN mask encoder's last convolutional feature energy.",
)
flags.DEFINE_string("gaze_json_name", "gaze_contact.json", "Recorded gaze json name.")
flags.DEFINE_string(
    "attention_keys",
    "front_camera,tactile_data",
    "Comma-separated image keys whose critic attention should be rendered.",
)
flags.DEFINE_boolean(
    "compare_raw_mask_feature",
    True,
    "Render both raw spatial feature energy and trainable mask feature head.",
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
        "mask1": f"frame_{fid:06d}_mask1.png",
        "mask2": f"frame_{fid:06d}_mask2.png",
    }
    root = frame_dir.parent
    return {
        "mask1": [
            frame_dir / "mask1.png",
            frame_dir / "rs_mask_obj0.png",
            root / "sam_masks" / central_names["mask1"],
            root / "sam_masks" / "propagated" / central_names["mask1"],
        ],
        "mask2": [
            frame_dir / "mask2.png",
            frame_dir / "rs_mask_obj1.png",
            root / "sam_masks" / central_names["mask2"],
            root / "sam_masks" / "propagated" / central_names["mask2"],
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
    phase = np.zeros((3,), dtype=np.float32)
    if selected_index == 0:
        phase[0] = 1.0
    elif selected_index == 1:
        phase[1] = 1.0
    else:
        phase[2] = 1.0
    return phase


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


def resolve_mask_selection_mode(exp_name):
    if FLAGS.mask_selection_mode != "auto":
        return FLAGS.mask_selection_mode
    if exp_name == "tennis_ball_pick":
        return "force_mask1"
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


def panel_base_image(obs, attention_key, attention_kind):
    if attention_kind == "mask_encoder_feature":
        if "front_camera_mask" in obs:
            return obs["front_camera_mask"]
        if "front_camera_mask1" in obs:
            return obs["front_camera_mask1"]
    return obs[attention_key]


def create_attention_agent(
    config,
    sample_obs,
    attention_key,
    *,
    use_mask_feature_head,
    return_raw_attention,
    return_mask_encoder_attention=False,
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
        mask_suppress_beta=FLAGS.mask_suppress_beta,
        use_mask_feature_head=use_mask_feature_head,
        mask_feature_gate_alpha=FLAGS.mask_feature_gate_alpha,
        mask_feature_min_gate=FLAGS.mask_feature_min_gate,
        mask_feature_hidden_dim=FLAGS.mask_feature_hidden_dim,
        use_mask_encoder=FLAGS.use_mask_encoder,
        mask_encoder_latent_dim=FLAGS.mask_encoder_latent_dim,
        mask_pick_place_phase_control=getattr(
            config,
            "mask_pick_place_phase_control",
            False,
        ),
        return_raw_attention=return_raw_attention,
        return_mask_encoder_attention=return_mask_encoder_attention,
        mask_grounding_key=getattr(config, "mask_grounding_key", "front_camera_mask1"),
    )


def restore_agent(agent, *, label, allow_fallback=False):
    if FLAGS.skip_restore:
        print_green(
            f"skip_restore=True: using initialized agent for {label} "
            "with loaded ResNet backbone only."
        )
        return agent
    step = None if FLAGS.checkpoint_step < 0 else FLAGS.checkpoint_step
    if step is None:
        resolved_checkpoint = checkpoints.latest_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path)
        )
    else:
        resolved_checkpoint = os.path.join(
            os.path.abspath(FLAGS.checkpoint_path),
            f"checkpoint_{step}",
        )
    print_green(
        f"restoring {label} checkpoint="
        f"{resolved_checkpoint if resolved_checkpoint is not None else 'None'}"
    )
    try:
        restored_state = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
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

    frame_root = Path(FLAGS.frame_root).expanduser().resolve()
    output_dir = Path(FLAGS.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_dirs = list_frame_dirs(frame_root)
    if not frame_dirs:
        raise FileNotFoundError(f"No frame_* dirs found under {frame_root}")
    has_recorded_mask1 = has_any_recorded_mask(frame_dirs, "mask1")
    show_mask_encoder_feature = bool(
        FLAGS.show_mask_encoder_feature
        and FLAGS.use_mask_encoder
        and has_recorded_mask1
    )

    print_green(f"frame_root={frame_root}")
    print_green(f"checkpoint_path={Path(FLAGS.checkpoint_path).expanduser().resolve()}")
    print_green(f"image_keys={config.image_keys}")
    print_green(f"attention_keys={attention_keys}")
    print_green(f"mask_selection_mode={mask_selection_mode}")
    print_green(f"mask_suppress_beta={FLAGS.mask_suppress_beta}")
    print_green(
        "mask_pick_place_phase_control="
        f"{getattr(config, 'mask_pick_place_phase_control', False)}"
    )
    print_green(f"use_mask_feature_head={FLAGS.use_mask_feature_head}")
    print_green(f"use_mask_encoder={FLAGS.use_mask_encoder}")
    print_green(f"has_recorded_mask1={has_recorded_mask1}")
    print_green(f"show_mask_encoder_feature={show_mask_encoder_feature}")
    if FLAGS.show_mask_encoder_feature and FLAGS.use_mask_encoder and not has_recorded_mask1:
        print_green("No recorded mask1 files found; skipping mask_encoder_feature panels.")
    print_green(f"compare_raw_mask_feature={FLAGS.compare_raw_mask_feature}")
    print_green(f"mask_feature_gate_alpha={FLAGS.mask_feature_gate_alpha}")
    print_green(f"mask_feature_min_gate={FLAGS.mask_feature_min_gate}")
    print_green(f"mask_feature_hidden_dim={FLAGS.mask_feature_hidden_dim}")
    print_green(f"mask_encoder_latent_dim={FLAGS.mask_encoder_latent_dim}")
    print_green(f"mask_grounding_key={getattr(config, 'mask_grounding_key', 'front_camera_mask1')}")
    print_green(f"viewer_mask_grounding_threshold={FLAGS.viewer_mask_grounding_threshold}")
    print_green(
        "viewer_mask_grounding_cell_threshold="
        f"{FLAGS.viewer_mask_grounding_cell_threshold}"
    )
    print_green(f"attention_display_mode={FLAGS.attention_display_mode}")
    print_green(f"frames={len(frame_dirs)} output_dir={output_dir}")

    sample_obs, _, _ = read_frame_observation_and_action(frame_dirs[0], config.image_keys)
    mask_predictor = None
    if mask_selection_mode in ("force_mask1", "force_mask2"):
        mask_predictor = load_mask_predictor(
            sample_obs,
            config.image_keys,
            FLAGS.mask_predictor_checkpoint_path,
            preferred_key="front_camera",
            log_fn=print_green,
        )
        selected_index = 0 if mask_selection_mode == "force_mask1" else 1
        sample_obs, _ = apply_predicted_mask_to_obs(
            sample_obs,
            mask_predictor,
            selected_index,
            (FLAGS.image_size, FLAGS.image_size),
        )
    attention_specs = []
    for attention_key in attention_keys:
        if FLAGS.compare_raw_mask_feature:
            attention_specs.append(
                {
                    "key": attention_key,
                    "kind": "raw_feature",
                    "use_mask_feature_head": FLAGS.use_mask_feature_head,
                    "return_raw_attention": True,
                    "return_mask_encoder_attention": False,
                    "allow_restore_fallback": False,
                }
            )
        if attention_key == "front_camera" and FLAGS.use_mask_feature_head:
            attention_specs.append(
                {
                    "key": attention_key,
                    "kind": "fused_feature",
                    "use_mask_feature_head": True,
                    "return_raw_attention": False,
                    "return_mask_encoder_attention": False,
                    "allow_restore_fallback": False,
                }
            )
        if (
            attention_key == "front_camera"
            and show_mask_encoder_feature
        ):
            attention_specs.append(
                {
                    "key": attention_key,
                    "kind": "mask_encoder_feature",
                    "use_mask_feature_head": FLAGS.use_mask_feature_head,
                    "return_raw_attention": False,
                    "return_mask_encoder_attention": True,
                    "allow_restore_fallback": False,
                }
            )
        elif not FLAGS.compare_raw_mask_feature:
            attention_specs.append(
                {
                    "key": attention_key,
                    "kind": "raw_feature",
                    "use_mask_feature_head": FLAGS.use_mask_feature_head,
                    "return_raw_attention": True,
                    "return_mask_encoder_attention": False,
                    "allow_restore_fallback": False,
                }
            )
    if not attention_specs:
        raise ValueError("No attention specs were created.")

    agents = {}
    expected_state_dims = {}
    for spec in attention_specs:
        attention_key = spec["key"]
        attention_kind = spec["kind"]
        start = time.time()
        agent = create_attention_agent(
            config,
            sample_obs,
            attention_key,
            use_mask_feature_head=spec["use_mask_feature_head"],
            return_raw_attention=spec["return_raw_attention"],
            return_mask_encoder_attention=spec["return_mask_encoder_attention"],
        )
        label = f"{attention_key}/{attention_kind}"
        agent = restore_agent(
            agent,
            label=label,
            allow_fallback=spec["allow_restore_fallback"],
        )
        expected_state_dims[(attention_key, attention_kind)] = infer_actor_state_dim(agent)
        agent = jax.device_put(jax.tree_util.tree_map(jnp.array, agent))
        agents[(attention_key, attention_kind)] = agent
        print_green(
            f"loaded attention agent key={label} "
            f"state_dim={expected_state_dims[(attention_key, attention_kind)]} "
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
                if mask_selection_mode in ("force_mask1", "force_mask2"):
                    selected_index = 0 if mask_selection_mode == "force_mask1" else 1
                    obs, selected_slot = apply_predicted_mask_to_obs(
                        obs,
                        mask_predictor,
                        selected_index,
                        (FLAGS.image_size, FLAGS.image_size),
                    )
                panels = []
                for spec in attention_specs:
                    attention_key = spec["key"]
                    attention_kind = spec["kind"]
                    frame_obs = align_obs_state_dim(
                        obs,
                        expected_state_dims.get((attention_key, attention_kind)),
                        selected_slot,
                    )
                    attention_map = critic_attention_for_frame(
                        agents[(attention_key, attention_kind)],
                        frame_obs,
                        action,
                    )
                    attention_prob = attention_to_prob(attention_map)
                    peak_y, peak_x = np.unravel_index(
                        int(np.argmax(attention_prob)),
                        attention_prob.shape,
                    )
                    peak_prob = float(attention_prob[peak_y, peak_x])
                    metrics = {
                        "mask_mass": np.nan,
                        "mask_grounding_loss": np.nan,
                        "mask_cell_fraction": np.nan,
                    }
                    metric_lines = []
                    if attention_key == "front_camera" and "front_camera_mask1" in obs:
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
                            tuple(attention_map.shape),
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
                        f"attn_shape={tuple(attention_map.shape)}",
                        f"peak=({int(peak_x)},{int(peak_y)}) p={peak_prob:.3f}",
                        f"selected_mask={selected_slot}",
                        *metric_lines,
                    ]
                    base_image = panel_base_image(obs, attention_key, attention_kind)
                    if FLAGS.attention_display_mode in ("heatmap", "both"):
                        panel = render_attention_panel(
                            base_image,
                            attention_map,
                            f"frame={fid} {attention_key} {attention_kind} logits",
                            panel_lines,
                        )
                        panels.append(panel)
                    if FLAGS.attention_display_mode in ("prob", "both"):
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
