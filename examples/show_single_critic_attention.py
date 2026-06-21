#!/usr/bin/env python3

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".2")

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
    "tennis_ball_pick-6-17-0"
)
DEFAULT_CHECKPOINT_PATH = str(
    REPO_ROOT
    / "examples"
    / "experiments"
    / "tennis_ball_pick"
    / "2026-6-19_0_ball_pick_rl_run"
)
DEFAULT_ROBOT_URDF_PATH = str(
    REPO_ROOT / "examples" / "urdf" / "fr3_moveit_servo.urdf"
)
DEFAULT_GAZE_PREDICTOR_CHECKPOINT_PATH = str(
    REPO_ROOT / "examples" / "gaze_data_process" / "gaze_heatmap_ckpt"
)

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags
from flax.training import checkpoints

from experiments.mappings import NEW_MAPPING
from serl_launcher.utils.launcher import (
    make_gaze_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_hybrid_single_arm,
)
from serl_launcher.utils.gaze_utils import (
    compute_gaze_heatmap_fields,
    gaze_xy_norm_from_heatmap,
    load_gaze_predictor,
)


FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", "tennis_ball_pick", "Experiment name.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("frame_root", DEFAULT_FRAME_ROOT, "Recorded frame root.")
flags.DEFINE_string("checkpoint_path", DEFAULT_CHECKPOINT_PATH, "Checkpoint directory.")
flags.DEFINE_integer("checkpoint_step", 11000, "Checkpoint step to load.")
flags.DEFINE_string(
    "gaze_predictor_checkpoint_path",
    DEFAULT_GAZE_PREDICTOR_CHECKPOINT_PATH,
    "Frozen gaze predictor checkpoint directory. Empty string disables gaze overlay.",
)
flags.DEFINE_string("robot_urdf_path", DEFAULT_ROBOT_URDF_PATH, "Robot URDF path.")
flags.DEFINE_integer("enable_tactile", 1, "Whether to include tactile input.")
flags.DEFINE_boolean("use_gaze_cgl", True, "Use gaze-CGL critic agent.")
flags.DEFINE_float("gaze_regularization_weight", 0.2, "Gaze auxiliary loss weight.")
flags.DEFINE_integer("image_size", 128, "Network RGB image size.")
flags.DEFINE_integer("max_frames", 200, "Max frames to process. Use 0 for all frames.")
flags.DEFINE_integer("frame_stride", 1, "Process every Nth frame.")
flags.DEFINE_integer("start_frame", 0, "Skip frames with id smaller than this.")
flags.DEFINE_integer("output_scale", 4, "Scale saved overlays for easier viewing.")
flags.DEFINE_float("text_scale", 0.45, "Overlay text font scale after output scaling.")
flags.DEFINE_string("output_dir", "critic_attention_outputs", "Output folder.")


def print_green(message):
    print(f"\033[92m {message}\033[00m")


def print_red(message):
    print(f"\033[91m {message}\033[00m")


def timed(label, fn):
    start = time.time()
    value = fn()
    print_green(f"{label}: {time.time() - start:.3f}s")
    return value


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
    image = cv2.resize(
        image,
        (FLAGS.image_size, FLAGS.image_size),
        interpolation=cv2.INTER_LINEAR,
    )
    return image.astype(np.uint8, copy=False)


def read_tactile(frame_dir: Path):
    thumb_path = frame_dir / "thumb_depth_image.png"
    index_path = frame_dir / "index_depth_image.png"
    if not thumb_path.exists() or not index_path.exists():
        thumb_path = frame_dir / "thumb_heat_map.jpg"
        index_path = frame_dir / "index_heat_map.jpg"
    thumb = read_tactile_image(thumb_path)
    index = read_tactile_image(index_path)
    return np.concatenate([thumb, index], axis=1).astype(np.uint8)


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
    for image_key in image_keys:
        if image_key == "front_camera":
            obs[image_key] = read_rgb(frame_dir)
        elif image_key == "tactile_data":
            obs[image_key] = read_tactile(frame_dir)
        else:
            raise ValueError(f"Unsupported image_key={image_key}")

    tcp_pos, tcp_ori = read_ee_pose(frame_dir)
    hand_state = read_numeric_file(
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

    action = read_numeric_file(
        frame_dir / "action.txt",
        default=np.zeros(7, dtype=np.float32),
        dtype=np.float32,
    )
    if action.shape[0] < 7:
        action = np.pad(action, (0, 7 - action.shape[0]))
    return obs, action[:7].astype(np.float32)


def batch_observation(obs):
    return jax.tree_util.tree_map(lambda x: np.asarray(x)[None], obs)


def normalize_heatmap(heatmap):
    heatmap = np.asarray(heatmap, dtype=np.float32)
    while heatmap.ndim > 2:
        heatmap = heatmap[0]
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
    line_gap = 5

    text_sizes = [
        cv2.getTextSize(line, font, scale, thickness)[0]
        for line in lines
    ]
    panel_width = max(width for width, _ in text_sizes) + 2 * margin
    panel_height = (
        sum(height for _, height in text_sizes)
        + line_gap * (len(lines) - 1)
        + 2 * margin
    )

    panel = image.copy()
    cv2.rectangle(
        panel,
        (0, 0),
        (min(panel_width, image.shape[1] - 1), min(panel_height, image.shape[0] - 1)),
        (0, 0, 0),
        -1,
    )
    image[:] = cv2.addWeighted(panel, 0.45, image, 0.55, 0.0)

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


def heatmap_peak_xy(heatmap):
    xy_norm = gaze_xy_norm_from_heatmap(heatmap)
    if xy_norm is None:
        return None
    heatmap = np.asarray(heatmap)
    while heatmap.ndim > 2:
        heatmap = heatmap[0]
    y, x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    return int(x), int(y), float(np.max(heatmap)), xy_norm


def render_heatmap_overlay(image_rgb, heatmap, title, lines=()):
    heatmap = normalize_heatmap(heatmap)
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
    overlay = cv2.addWeighted(image_bgr, 0.65, heatmap_color, 0.35, 0.0)
    if FLAGS.output_scale > 1:
        overlay = cv2.resize(
            overlay,
            (
                overlay.shape[1] * FLAGS.output_scale,
                overlay.shape[0] * FLAGS.output_scale,
            ),
            interpolation=cv2.INTER_NEAREST,
        )
    draw_text_panel(
        overlay,
        [title, *lines],
    )
    return overlay


def render_attention(image_rgb, attention_map, fid):
    return render_heatmap_overlay(
        image_rgb,
        attention_map,
        f"frame={fid} critic attention",
        [f"attn_shape={tuple(np.asarray(attention_map).shape)}"],
    )


def render_gaze_prediction(image_rgb, gaze_heatmap, fid):
    lines = [f"gaze_shape={tuple(np.asarray(gaze_heatmap).shape)}"]
    peak = heatmap_peak_xy(gaze_heatmap)
    if peak is not None:
        peak_x, peak_y, peak_value, _ = peak
        lines.append(f"peak=({peak_x},{peak_y}) val={peak_value:.3f}")
    return render_heatmap_overlay(
        image_rgb,
        gaze_heatmap,
        f"frame={fid} gaze predictor",
        lines,
    )


def create_agent(config, sample_obs):
    agent_factory = (
        make_gaze_sac_pixel_agent_hybrid_single_arm
        if FLAGS.use_gaze_cgl
        else make_sac_pixel_agent_hybrid_single_arm
    )
    agent_kwargs = {}
    if FLAGS.use_gaze_cgl:
        agent_kwargs = {
            "gaze_regularization_weight": FLAGS.gaze_regularization_weight,
            "gaze_heatmap_size": sample_obs["front_camera"].shape[:2],
        }
    return agent_factory(
        seed=FLAGS.seed,
        sample_obs=sample_obs,
        sample_action=np.zeros(7, dtype=np.float32),
        image_keys=config.image_keys,
        encoder_type=config.encoder_type,
        discount=config.discount,
        **agent_kwargs,
    )


def critic_attention_for_frame(agent, obs, action):
    critic_action = np.asarray(action, dtype=np.float32)
    critic_action = critic_action[..., :-1]
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


def main(_):
    if FLAGS.exp_name not in NEW_MAPPING:
        raise ValueError(f"Unknown exp_name={FLAGS.exp_name}")
    config = NEW_MAPPING[FLAGS.exp_name]()
    if hasattr(config, "get_image_keys"):
        config.image_keys = list(config.get_image_keys(bool(FLAGS.enable_tactile)))
    elif not hasattr(config, "image_keys"):
        config.image_keys = ["front_camera", "tactile_data"] if FLAGS.enable_tactile else ["front_camera"]

    frame_root = Path(FLAGS.frame_root).expanduser().resolve()
    output_dir = Path(FLAGS.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_dirs = list_frame_dirs(frame_root)
    if not frame_dirs:
        raise FileNotFoundError(f"No frame_* dirs found under {frame_root}")

    print_green(f"frame_root={frame_root}")
    print_green(f"output_dir={output_dir}")
    print_green(f"frames={len(frame_dirs)} image_keys={config.image_keys}")

    sample_obs, _ = read_frame_observation_and_action(frame_dirs[0], config.image_keys)
    gaze_predictor = None
    if FLAGS.gaze_predictor_checkpoint_path:
        gaze_predictor = timed(
            "load gaze predictor",
            lambda: load_gaze_predictor(
                sample_obs,
                config.image_keys,
                FLAGS.gaze_predictor_checkpoint_path,
                log_fn=print_green,
            ),
        )
    devices = jax.local_devices()
    sharding = jax.sharding.PositionalSharding(devices)
    agent = timed("create agent", lambda: create_agent(config, sample_obs))
    agent = jax.device_put(
        jax.tree_util.tree_map(jnp.array, agent),
        sharding.replicate(),
    )
    ckpt = timed(
        "restore checkpoint",
        lambda: checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
            step=FLAGS.checkpoint_step,
        ),
    )
    agent = agent.replace(state=ckpt)

    if not hasattr(agent, "forward_gaze_attention"):
        raise ValueError("Current agent does not support critic attention.")

    summary_path = output_dir / "attention_summary.txt"
    with summary_path.open("w") as summary_file:
        summary_file.write(
            "frame_id,attention_shape,gaze_shape,gaze_peak,action\n"
        )
        for index, frame_dir in enumerate(frame_dirs, start=1):
            fid = frame_id(frame_dir)
            try:
                obs, action = read_frame_observation_and_action(frame_dir, config.image_keys)
                attention_map = critic_attention_for_frame(agent, obs, action)
                critic_overlay = render_attention(
                    obs["front_camera"],
                    attention_map,
                    fid,
                )

                gaze_shape = ""
                gaze_peak = ""
                saved_path = None
                if gaze_predictor is not None:
                    gaze_heatmap = compute_gaze_heatmap_fields(
                        obs,
                        gaze_predictor,
                        obs["front_camera"].shape[:2],
                    )["gaze_heatmap"]
                    gaze_shape = tuple(np.asarray(gaze_heatmap).shape)
                    peak = heatmap_peak_xy(gaze_heatmap)
                    if peak is not None:
                        peak_x, peak_y, peak_value, _ = peak
                        gaze_peak = f"({peak_x},{peak_y},{peak_value:.6f})"
                    gaze_overlay = render_gaze_prediction(
                        obs["front_camera"],
                        gaze_heatmap,
                        fid,
                    )

                    comparison = np.concatenate([critic_overlay, gaze_overlay], axis=1)
                    comparison_path = output_dir / f"frame_{fid:06d}_attention_vs_gaze.jpg"
                    if not cv2.imwrite(str(comparison_path), comparison):
                        raise RuntimeError(f"cv2.imwrite failed: {comparison_path}")
                    saved_path = comparison_path
                summary_file.write(
                    f"{fid},{tuple(attention_map.shape)},{gaze_shape},{gaze_peak},"
                    f"{np.array2string(action, precision=5, separator=' ')}\n"
                )
                if index == 1 or index % 25 == 0:
                    if saved_path is not None:
                        print_green(
                            f"[{index}/{len(frame_dirs)}] saved {saved_path.name}"
                        )
            except Exception as exc:
                print_red(f"[skip] frame={fid}: {exc}")

    print_green(f"done. summary={summary_path}")


if __name__ == "__main__":
    app.run(main)
