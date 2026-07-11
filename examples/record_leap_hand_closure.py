#!/usr/bin/env python3

import csv
import datetime
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from absl import app, flags


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ROBOT_INFRA_DIR = REPO_ROOT / "serl_robot_infra"
for path in (REPO_ROOT, SCRIPT_DIR, ROBOT_INFRA_DIR):
    path = str(path)
    if path not in sys.path:
        sys.path.insert(0, path)

from experiments.tennis_ball_pick.config import EnvConfig
from experiments.tennis_ball_pick.wrapper import RAMEnv


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "output_root",
    "/home/ealin/workspaces/DexTacHil/data/leap_hand_joint_recordings",
    "Root directory for saved RGB/tactile/leap-hand joint frames.",
)
flags.DEFINE_float("hz", 10.0, "Sampling frequency.")
flags.DEFINE_integer(
    "max_frames",
    0,
    "Optional frame limit. 0 means record until Ctrl+C.",
)
flags.DEFINE_float(
    "warmup_sec",
    0.5,
    "Short wait after env creation before the first sample.",
)


def _timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _as_uint8_image(image):
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _copy_tactile_images(env):
    tactile = {}
    if not getattr(env, "enable_tactile", False):
        return tactile

    with env.tac_thumb_lock:
        tactile["thumb_raw"] = np.asarray(env.thumb_raw_img).copy()
        tactile["thumb_depth"] = np.asarray(env.thumb_depth_img).copy()
    with env.tac_index_lock:
        tactile["index_raw"] = np.asarray(env.index_raw_img).copy()
        tactile["index_depth"] = np.asarray(env.index_depth_img).copy()

    if getattr(env, "enable_dm_tac_middle", False):
        with env.tac_middle_lock:
            tactile["middle_raw"] = np.asarray(env.middle_raw_img).copy()
            tactile["middle_depth"] = np.asarray(env.middle_depth_img).copy()

    tactile["tactile_data"] = np.concatenate(
        [tactile["thumb_depth"], tactile["index_depth"]],
        axis=1,
    )
    return tactile


def _read_leap_joints(env):
    try:
        joints = env.ros_interface.get_current_leap_position()
    except Exception as exc:
        print(f"\n[warn] failed to read leap hand position, using cached value: {exc}")
        joints = getattr(env, "curr_leap_hand_pos", np.zeros(16, dtype=np.float32))
    joints = np.asarray(joints, dtype=np.float32).reshape(-1)
    env.curr_leap_hand_pos = joints.copy()
    try:
        hand_state = float(env._hand_progress_scalar(joints))
        env.hand_state = hand_state
    except Exception:
        hand_state = float(getattr(env, "hand_state", 0.0))
    return joints, hand_state


def _sample_frame(env, frame_id):
    image_obs, record_images = env.get_im(return_full_images=True)
    front_rgb = record_images.get("front_camera", image_obs.get("front_camera"))

    if front_rgb is None:
        raise RuntimeError("front_camera image is missing from environment output.")

    leap_joints, hand_state = _read_leap_joints(env)
    return {
        "frame_id": int(frame_id),
        "timestamp": time.time(),
        "front_rgb": _as_uint8_image(front_rgb).copy(),
        "tactile": _copy_tactile_images(env),
        "leap_joints": leap_joints.copy(),
        "hand_state": float(hand_state),
    }


def _write_image_rgb(path: Path, image_rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    image_rgb = _as_uint8_image(image_rgb)
    if image_rgb.ndim == 3 and image_rgb.shape[-1] >= 3:
        cv2.imwrite(str(path), image_rgb[..., :3][..., ::-1])
    else:
        cv2.imwrite(str(path), image_rgb)


def _write_image_bgr(path: Path, image_bgr):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), _as_uint8_image(image_bgr))


def _save_frames(frames, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    joints_csv = output_dir / "leap_hand_joints.csv"

    with joints_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["frame_id", "timestamp", "hand_state"]
            + [f"joint_{i:02d}" for i in range(16)]
        )
        for frame in frames:
            writer.writerow(
                [
                    frame["frame_id"],
                    f"{frame['timestamp']:.9f}",
                    f"{frame['hand_state']:.9f}",
                ]
                + [f"{v:.9f}" for v in frame["leap_joints"]]
            )

    for frame in frames:
        frame_dir = output_dir / f"frame_{frame['frame_id']}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        _write_image_rgb(frame_dir / "color_image.jpg", frame["front_rgb"])
        np.savetxt(
            frame_dir / "leap_hand_joints.txt",
            frame["leap_joints"],
            fmt="%.9f",
        )
        np.savetxt(
            frame_dir / "hand_state.txt",
            np.asarray([frame["hand_state"]], dtype=np.float32),
            fmt="%.9f",
        )
        (frame_dir / "timestamp.txt").write_text(
            f"{frame['timestamp']:.9f}\n",
            encoding="utf-8",
        )

        tactile = frame["tactile"]
        if "thumb_raw" in tactile:
            _write_image_bgr(frame_dir / "thumb_raw_image.jpg", tactile["thumb_raw"])
        if "thumb_depth" in tactile:
            _write_image_bgr(frame_dir / "thumb_depth_image.png", tactile["thumb_depth"])
        if "index_raw" in tactile:
            _write_image_bgr(frame_dir / "index_raw_image.jpg", tactile["index_raw"])
        if "index_depth" in tactile:
            _write_image_bgr(frame_dir / "index_depth_image.png", tactile["index_depth"])
        if "middle_raw" in tactile:
            _write_image_bgr(frame_dir / "middle_raw_image.jpg", tactile["middle_raw"])
        if "middle_depth" in tactile:
            _write_image_bgr(frame_dir / "middle_depth_image.png", tactile["middle_depth"])

    joint_array = np.stack([frame["leap_joints"] for frame in frames], axis=0)
    np.save(output_dir / "leap_hand_joints.npy", joint_array)
    metadata = {
        "created_at": datetime.datetime.now().isoformat(),
        "num_frames": len(frames),
        "hz": float(FLAGS.hz),
        "files": {
            "per_frame_rgb": "frame_<id>/color_image.jpg",
            "per_frame_tactile": (
                "frame_<id>/thumb_depth_image.png, "
                "frame_<id>/index_depth_image.png"
            ),
            "per_frame_joints": "frame_<id>/leap_hand_joints.txt",
            "joints_csv": str(joints_csv.name),
            "joints_npy": "leap_hand_joints.npy",
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def main(_):
    if FLAGS.hz <= 0:
        raise ValueError("--hz must be positive.")

    output_dir = Path(FLAGS.output_root).expanduser() / f"leap_hand_closure_{_timestamp()}"
    config = EnvConfig()
    config.ENABLE_TACTILE = True
    config.DISPLAY_IMAGE = True
    config.ENABLE_DATA_RECORDING = False
    config.ENABLE_GAZE_COLLECTION = False

    env = None
    frames = []
    period = 1.0 / float(FLAGS.hz)

    try:
        print("[record_leap_hand_closure] creating RAMEnv without reset/step...")
        env = RAMEnv(fake_env=False, save_video=False, config=config)
        if FLAGS.warmup_sec > 0:
            time.sleep(float(FLAGS.warmup_sec))
        print("[record_leap_hand_closure] recording. Press Ctrl+C to stop and save.")

        frame_id = 0
        while FLAGS.max_frames <= 0 or frame_id < FLAGS.max_frames:
            start = time.time()
            frame = _sample_frame(env, frame_id)
            frames.append(frame)
            joints_preview = np.array2string(
                frame["leap_joints"],
                precision=4,
                suppress_small=True,
                max_line_width=200,
            )
            print(
                f"\rframe={frame_id:06d} hand_state={frame['hand_state']:.4f} "
                f"joints={joints_preview}",
                end="",
                flush=True,
            )
            frame_id += 1
            elapsed = time.time() - start
            time.sleep(max(0.0, period - elapsed))
    except KeyboardInterrupt:
        print("\n[record_leap_hand_closure] Ctrl+C received; saving buffered frames...")
    finally:
        if frames:
            _save_frames(frames, output_dir)
            print(f"[record_leap_hand_closure] saved {len(frames)} frames to {output_dir}")
        else:
            print("[record_leap_hand_closure] no frames captured; nothing saved.")
        if env is not None:
            env.close()


if __name__ == "__main__":
    app.run(main)
