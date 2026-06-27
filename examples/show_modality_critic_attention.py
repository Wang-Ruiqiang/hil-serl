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
    "/media/user/data3/wrq/recorded_data/tennis_ball_pick/"
    "tennis_ball_pick-6-23-1"
)
DEFAULT_CHECKPOINT_PATH = str(
    REPO_ROOT
    / "examples"
    / "experiments"
    / "tennis_ball_pick"
    / "2026-6-26_0_ball_pick_rl_run"
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
flags.DEFINE_integer("image_size", 128, "Network image size.")
flags.DEFINE_integer("max_frames", 24, "Max frames to render. Use 0 for all.")
flags.DEFINE_integer("frame_stride", 25, "Process every Nth recorded frame.")
flags.DEFINE_integer("start_frame", 0, "Skip frames with id smaller than this.")
flags.DEFINE_integer("enable_tactile", 1, "Whether tactile_data is part of obs.")
flags.DEFINE_integer(
    "gaze_target_mask_dilation",
    8,
    "Dilation used when deciding whether recorded gaze hits mask1/mask2.",
)
flags.DEFINE_integer(
    "mask1_input_dilation",
    6,
    "Dilation radius applied to mask1 before building front_camera_masked.",
)
flags.DEFINE_integer(
    "mask2_input_dilation",
    0,
    "Dilation radius applied to mask2 before building front_camera_masked.",
)
flags.DEFINE_string("gaze_json_name", "gaze_contact.json", "Recorded gaze json name.")
flags.DEFINE_string(
    "attention_keys",
    "front_camera,tactile_data,front_camera_masked",
    "Comma-separated image keys whose critic attention should be rendered.",
)
flags.DEFINE_integer("output_scale", 3, "Scale rendered panels for easier viewing.")
flags.DEFINE_float("text_scale", 0.42, "Panel text scale after output scaling.")
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


def read_front_camera_masked(frame_dir: Path, rgb_image, target_shape):
    selected_mask, selected_index, selected_slot = read_gaze_target_mask(
        frame_dir,
        target_shape,
    )
    input_dilation = (
        FLAGS.mask1_input_dilation
        if selected_index == 0
        else FLAGS.mask2_input_dilation
    )
    selected_mask = dilate_mask(selected_mask, input_dilation)
    masked_rgb = (
        rgb_image.astype(np.float32) * selected_mask.astype(np.float32)[..., None]
    ).astype(np.uint8)
    return masked_rgb, phase_onehot(selected_index), selected_slot


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
        elif image_key == "front_camera_masked":
            if rgb_image is None:
                rgb_image = read_rgb(frame_dir)
            obs[image_key], gaze_phase, selected_slot = read_front_camera_masked(
                frame_dir,
                rgb_image,
                target_shape,
            )
        else:
            raise ValueError(f"Unsupported image_key={image_key}")
    if "front_camera_masked" in image_keys and gaze_phase is None:
        if rgb_image is None:
            rgb_image = read_rgb(frame_dir)
        obs["front_camera_masked"], gaze_phase, selected_slot = read_front_camera_masked(
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
    return obs, action[:7].astype(np.float32), selected_slot


def attention_to_prob(attention_map):
    values = np.asarray(attention_map, dtype=np.float32)
    while values.ndim > 2:
        values = values.reshape((-1, *values.shape[-2:]))[0]
    flat = values.reshape(-1)
    flat = flat - np.max(flat)
    prob = np.exp(flat)
    prob = prob / max(float(np.sum(prob)), 1e-8)
    return prob.reshape(values.shape)


def normalize_heatmap(heatmap):
    heatmap = np.asarray(heatmap, dtype=np.float32)
    while heatmap.ndim > 2:
        heatmap = heatmap.reshape((-1, *heatmap.shape[-2:]))[0]
    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
    heatmap = heatmap - np.min(heatmap)
    denom = np.max(heatmap)
    if denom > 1e-8:
        heatmap = heatmap / denom
    return heatmap


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
    heatmap = normalize_heatmap(attention_map)
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


def create_attention_agent(config, sample_obs, attention_key):
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
    )


def restore_agent(agent):
    step = None if FLAGS.checkpoint_step < 0 else FLAGS.checkpoint_step
    restored_state = checkpoints.restore_checkpoint(
        os.path.abspath(FLAGS.checkpoint_path),
        agent.state,
        step=step,
    )
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


def main(_):
    if FLAGS.exp_name not in NEW_MAPPING:
        raise ValueError(f"Unknown exp_name={FLAGS.exp_name}")
    config = NEW_MAPPING[FLAGS.exp_name]()
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
    ]
    if not attention_keys:
        raise ValueError(f"No valid attention_keys in {FLAGS.attention_keys}")

    frame_root = Path(FLAGS.frame_root).expanduser().resolve()
    output_dir = Path(FLAGS.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_dirs = list_frame_dirs(frame_root)
    if not frame_dirs:
        raise FileNotFoundError(f"No frame_* dirs found under {frame_root}")

    print_green(f"frame_root={frame_root}")
    print_green(f"checkpoint_path={Path(FLAGS.checkpoint_path).expanduser().resolve()}")
    print_green(f"image_keys={config.image_keys}")
    print_green(f"attention_keys={attention_keys}")
    print_green(f"frames={len(frame_dirs)} output_dir={output_dir}")

    sample_obs, _, _ = read_frame_observation_and_action(frame_dirs[0], config.image_keys)
    agents = {}
    for attention_key in attention_keys:
        start = time.time()
        agent = create_attention_agent(config, sample_obs, attention_key)
        agent = restore_agent(agent)
        agent = jax.device_put(jax.tree_util.tree_map(jnp.array, agent))
        agents[attention_key] = agent
        print_green(f"loaded attention agent key={attention_key} time={time.time() - start:.2f}s")

    summary_path = output_dir / "attention_summary.csv"
    with summary_path.open("w", newline="") as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow(
            [
                "frame_id",
                "selected_mask",
                "attention_key",
                "attention_shape",
                "peak_cell_x",
                "peak_cell_y",
                "peak_prob",
            ]
        )

        for index, frame_dir in enumerate(frame_dirs, start=1):
            fid = frame_id(frame_dir)
            try:
                obs, action, selected_slot = read_frame_observation_and_action(
                    frame_dir,
                    config.image_keys,
                )
                panels = []
                for attention_key in attention_keys:
                    attention_map = critic_attention_for_frame(
                        agents[attention_key],
                        obs,
                        action,
                    )
                    attention_prob = attention_to_prob(attention_map)
                    peak_y, peak_x = np.unravel_index(
                        int(np.argmax(attention_prob)),
                        attention_prob.shape,
                    )
                    peak_prob = float(attention_prob[peak_y, peak_x])
                    writer.writerow(
                        [
                            fid,
                            selected_slot,
                            attention_key,
                            tuple(attention_map.shape),
                            int(peak_x),
                            int(peak_y),
                            f"{peak_prob:.8f}",
                        ]
                    )
                    panel = render_attention_panel(
                        obs[attention_key],
                        attention_map,
                        f"frame={fid} {attention_key}",
                        [
                            f"attn_shape={tuple(attention_map.shape)}",
                            f"peak=({int(peak_x)},{int(peak_y)}) p={peak_prob:.3f}",
                            f"selected_mask={selected_slot}",
                        ],
                    )
                    panels.append(panel)

                comparison = np.concatenate(pad_to_same_height(panels), axis=1)
                output_path = output_dir / f"frame_{fid:06d}_modality_attention.jpg"
                if not cv2.imwrite(str(output_path), comparison):
                    raise RuntimeError(f"cv2.imwrite failed: {output_path}")
                if index == 1 or index % 10 == 0:
                    print_green(f"[{index}/{len(frame_dirs)}] saved {output_path.name}")
            except Exception as exc:
                print_red(f"[skip] frame={fid}: {exc}")

    print_green(f"done. summary={summary_path}")


if __name__ == "__main__":
    app.run(main)
