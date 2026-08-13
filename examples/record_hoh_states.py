import copy
import datetime
import importlib
import json
import os
import re
import sys
import time

import cv2
import numpy as np
from absl import app, flags
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../serl_launcher"))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", "flip_object", "Experiment name.")
flags.DEFINE_float("hz", 10.0, "Recording frequency in Hz.")
flags.DEFINE_integer(
    "enable_tactile",
    0,
    "Whether to enable tactile sensors. Use -1 to follow the task config default.",
)
flags.DEFINE_string(
    "record_root",
    "/home/wrq/workspaces/HK_TACEXO_WANG/hoh_data",
    "Root directory for HOH recordings.",
)
flags.DEFINE_string(
    "record_task_name",
    "",
    "Task name used in auto-created folder names. Defaults to exp_name.",
)
flags.DEFINE_string("record_source", "hoh", "Source tag used in auto-created folder names.")
flags.DEFINE_string(
    "frame_save_path",
    "",
    "Directory for raw frame_xxx data. If empty, auto-create one under --record_root.",
)
flags.DEFINE_bool("display_image", True, "Display camera/tactile images while recording.")
flags.DEFINE_bool(
    "enable_wrist_camera",
    True,
    "Initialize and record the wrist RealSense camera.",
)
flags.DEFINE_integer("max_frames", 0, "Stop after this many frames. 0 means record until Ctrl+C.")
flags.DEFINE_bool("save_on_interrupt", True, "Save buffered frames after Ctrl+C.")
flags.DEFINE_bool("save_video", True, "Save one MP4 per available camera when recording exits.")
flags.DEFINE_string("video_name", "hoh_recording.mp4", "Output video filename.")
flags.DEFINE_float(
    "video_fps",
    0.0,
    "Output video FPS. Use 0 to follow --hz.",
)
flags.DEFINE_float(
    "startup_timeout",
    15.0,
    "Seconds to wait for robot joint state, TCP pose, and LeapHand state before recording.",
)


EXP_CONFIG_MODULES = {
    "tennis_ball_pick": "experiments.tennis_ball_pick.config",
    "tennis_ball_place": "experiments.tennis_ball_pick.config_place",
    "lid_grip": "experiments.twist_bottle_cap.config_lid_grip",
    "twist_bottle_cap": "experiments.twist_bottle_cap.config",
    "tube_insertion": "experiments.tube_insertion.config",
    "flip_object": "experiments.flip_object.config",
}


def _default_record_task_name(exp_name):
    aliases = {
        "tennis_ball_pick": "ball_pick",
        "tennis_ball_place": "ball_place",
        "lid_grip": "lid_grip",
        "twist_bottle_cap": "twist_bottle_cap",
        "tube_insertion": "tube_insertion",
        "flip_object": "flip_object",
    }
    return aliases.get(exp_name, exp_name)


def _default_frame_save_path(exp_name):
    date = datetime.datetime.now().strftime("%Y_%m_%d")
    task_name = FLAGS.record_task_name or _default_record_task_name(exp_name)
    prefix = f"{task_name}_{FLAGS.record_source}_{date}"
    task_root = os.path.join(FLAGS.record_root, exp_name)
    os.makedirs(task_root, exist_ok=True)

    existing_indices = []
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    for name in os.listdir(task_root):
        match = pattern.match(name)
        if match:
            existing_indices.append(int(match.group(1)))

    next_index = max(existing_indices, default=0) + 1
    return os.path.abspath(os.path.join(task_root, f"{prefix}_{next_index:02d}"))


def _load_task_module(exp_name):
    if exp_name not in EXP_CONFIG_MODULES:
        supported = ", ".join(sorted(EXP_CONFIG_MODULES))
        raise ValueError(f"Unsupported exp_name={exp_name}. Supported: {supported}")
    return importlib.import_module(EXP_CONFIG_MODULES[exp_name])


def _make_env(config_module, frame_save_path):
    env_config = config_module.EnvConfig()
    if FLAGS.enable_tactile >= 0:
        env_config.ENABLE_TACTILE = bool(FLAGS.enable_tactile)
    if not FLAGS.enable_wrist_camera:
        env_config.REALSENSE_CAMERAS = {
            name: camera_config
            for name, camera_config in env_config.REALSENSE_CAMERAS.items()
            if name != "wrist_camera"
        }
    if not env_config.REALSENSE_CAMERAS:
        raise ValueError("No RealSense cameras are enabled for HOH recording.")
    env_config.DISPLAY_IMAGE = bool(FLAGS.display_image)

    print(
        "[record_hoh_states] cameras="
        f"{list(env_config.REALSENSE_CAMERAS)} tactile={env_config.ENABLE_TACTILE}"
    )

    return config_module.RAMEnv(
        fake_env=False,
        save_video=False,
        config=env_config,
        record_data=True,
        frame_save_path=frame_save_path,
    )


def _read_current_state(env):
    env.cur_position, env.cur_oritation = env.ros_interface.get_current_robot_ee()
    joint_position = env.ros_interface.get_current_joint()
    if joint_position is None:
        raise RuntimeError(
            "No /joint_states message has been received yet. "
            "Check that the Denso ROS controller/state publisher is running."
        )
    env.joint_position = np.asarray(joint_position, dtype=np.float32).copy()
    env.curr_leap_hand_pos = np.asarray(
        env.ros_interface.get_current_leap_position(), dtype=np.float32
    ).copy()
    env.hand_state = env._hand_progress_scalar(env.curr_leap_hand_pos)
    return {
        "tcp_pos": np.asarray(env.cur_position, dtype=np.float32).copy(),
        "tcp_ori": np.asarray(env.cur_oritation, dtype=np.float32).copy(),
        "arm_joint": env.joint_position.copy(),
        "hand_joint": env.curr_leap_hand_pos.copy(),
        "hand_state": float(np.asarray(env.hand_state).item()),
    }


def _wait_for_initial_state(env):
    deadline = time.time() + FLAGS.startup_timeout
    last_error = None
    while time.time() < deadline:
        try:
            state = _read_current_state(env)
            print("[record_hoh_states] Initial robot and hand state received.")
            return state
        except Exception as exc:
            last_error = exc
            time.sleep(0.1)
    raise RuntimeError(
        f"Timed out after {FLAGS.startup_timeout:.1f}s waiting for initial HOH state. "
        f"Last error: {last_error}"
    )


def _state_record(frame_id, timestamp, state, prev_state):
    arm_delta = np.zeros_like(state["arm_joint"])
    hand_delta = np.zeros_like(state["hand_joint"])
    if prev_state is not None:
        arm_delta = state["arm_joint"] - prev_state["arm_joint"]
        hand_delta = state["hand_joint"] - prev_state["hand_joint"]

    return {
        "frame_id": int(frame_id),
        "timestamp": float(timestamp),
        "tcp_pos": state["tcp_pos"].tolist(),
        "tcp_ori": state["tcp_ori"].tolist(),
        "arm_joint": state["arm_joint"].tolist(),
        "hand_joint": state["hand_joint"].tolist(),
        "arm_joint_delta": arm_delta.tolist(),
        "hand_joint_delta": hand_delta.tolist(),
        "hand_state": float(state["hand_state"]),
    }


def _write_hoh_metadata(
    frame_save_path, exp_name, hz, records, interrupted, video_paths=None
):
    metadata = {
        "exp_name": exp_name,
        "record_type": "hoh_state_recording",
        "frame_root": frame_save_path,
        "hz": float(hz),
        "total_frames": len(records),
        "interrupted": bool(interrupted),
        "video_paths": video_paths or [],
        "created_at": datetime.datetime.now().isoformat(),
    }
    metadata_path = os.path.join(frame_save_path, "hoh_metadata.json")
    states_path = os.path.join(frame_save_path, "hoh_states.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with open(states_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    for record in records:
        frame_dir = os.path.join(frame_save_path, f"frame_{record['frame_id']}")
        if not os.path.isdir(frame_dir):
            continue
        with open(os.path.join(frame_dir, "hoh_state.json"), "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        np.savetxt(os.path.join(frame_dir, "arm_joint_delta.txt"), record["arm_joint_delta"])
        np.savetxt(os.path.join(frame_dir, "hand_joint_delta.txt"), record["hand_joint_delta"])

    print(f"[record_hoh_states][metadata] saved {metadata_path}")
    print(f"[record_hoh_states][states] saved {states_path}")


def _save_camera_video(frame_save_path, records, camera_name, image_name, video_name):
    video_path = os.path.abspath(os.path.join(frame_save_path, video_name))
    fps = FLAGS.video_fps if FLAGS.video_fps > 0 else FLAGS.hz
    image_paths = []
    for record in records:
        image_path = os.path.join(
            frame_save_path, f"frame_{record['frame_id']}", image_name
        )
        if os.path.isfile(image_path):
            image_paths.append(image_path)
    if not image_paths:
        print(f"[record_hoh_states][video] {camera_name} camera not recorded; skipped")
        return None

    first_frame = cv2.imread(image_paths[0], cv2.IMREAD_COLOR)
    if first_frame is None:
        raise RuntimeError(f"Failed to read camera frame: {image_paths[0]}")
    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {video_path}")
    try:
        written_frames = 0
        for image_path in image_paths:
            frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
            written_frames += 1
    finally:
        writer.release()
    print(
        f"[record_hoh_states][video] saved {camera_name}: {video_path} "
        f"frames={written_frames} fps={fps:g}"
    )
    return video_path


def _save_recording_videos(frame_save_path, records):
    video_name = FLAGS.video_name.strip()
    if not video_name:
        raise ValueError("--video_name must not be empty")
    if not video_name.lower().endswith(".mp4"):
        video_name += ".mp4"
    stem = os.path.splitext(video_name)[0]
    camera_outputs = (
        ("front", "color_image.jpg", f"front_{stem}.mp4"),
        ("wrist", "color_image2.jpg", f"wrist_{stem}.mp4"),
    )
    video_paths = []
    for camera_name, image_name, output_name in camera_outputs:
        path = _save_camera_video(
            frame_save_path, records, camera_name, image_name, output_name
        )
        if path is not None:
            video_paths.append(path)
    if not video_paths:
        raise RuntimeError("No camera frames were available for video export.")
    return video_paths


def main(_):
    if FLAGS.hz <= 0:
        raise ValueError("--hz must be positive")

    config_module = _load_task_module(FLAGS.exp_name)
    frame_save_path = FLAGS.frame_save_path or _default_frame_save_path(FLAGS.exp_name)
    os.makedirs(frame_save_path, exist_ok=True)
    print(f"[record_hoh_states][recording] frame_save_path={frame_save_path}")
    print("[record_hoh_states] Initializing cameras, tactile sensors, and ROS interfaces...")
    env = _make_env(config_module, frame_save_path)
    _wait_for_initial_state(env)
    print("[record_hoh_states] Start dragging/operating the robot. Press Ctrl+C to save.")

    records = []
    prev_state = None
    interrupted = False
    period = 1.0 / FLAGS.hz
    pbar_total = FLAGS.max_frames if FLAGS.max_frames > 0 else None
    pbar = tqdm(total=pbar_total, dynamic_ncols=True)

    try:
        while FLAGS.max_frames <= 0 or env.frame_count < FLAGS.max_frames:
            start = time.time()
            state = _read_current_state(env)
            env.current_action = np.zeros(env.action_space.shape, dtype=np.float32)
            frame_id = int(env.frame_count)
            buffered_frames_before = len(env.joint_buffer)
            env.save_training_frame()
            if len(env.joint_buffer) <= buffered_frames_before:
                print("[record_hoh_states][warn] skipped one frame because buffering failed")
                elapsed = time.time() - start
                time.sleep(max(0.0, period - elapsed))
                continue
            env.frame_count += 1
            records.append(_state_record(frame_id, start, state, prev_state))
            prev_state = copy.deepcopy(state)
            pbar.update(1)
            pbar.set_description(f"frames={env.frame_count}")

            elapsed = time.time() - start
            time.sleep(max(0.0, period - elapsed))
    except KeyboardInterrupt:
        interrupted = True
        print("\n[record_hoh_states][interrupt] Ctrl+C received.")
    finally:
        pbar.close()
        try:
            if records and (not interrupted or FLAGS.save_on_interrupt):
                print("[record_hoh_states][save] writing images, tactile data, and states...")
                env.save_all_data_on_exit()
                video_paths = []
                if FLAGS.save_video:
                    try:
                        video_paths = _save_recording_videos(frame_save_path, records)
                    except Exception as exc:
                        print(f"[record_hoh_states][video][error] {exc}")
                _write_hoh_metadata(
                    frame_save_path,
                    FLAGS.exp_name,
                    FLAGS.hz,
                    records,
                    interrupted=interrupted,
                    video_paths=video_paths,
                )
            elif interrupted and not FLAGS.save_on_interrupt:
                print("[record_hoh_states][interrupt] skipped save because --nosave_on_interrupt was set")
            else:
                print("[record_hoh_states] no frames recorded")
        finally:
            env.close()


if __name__ == "__main__":
    app.run(main)
