import copy
import datetime
import importlib
import json
import os
import re
import sys
import time

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
    -1,
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
flags.DEFINE_integer("max_frames", 0, "Stop after this many frames. 0 means record until Ctrl+C.")
flags.DEFINE_bool("save_on_interrupt", True, "Save buffered frames after Ctrl+C.")
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
    env_config.DISPLAY_IMAGE = bool(FLAGS.display_image)

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


def _write_hoh_metadata(frame_save_path, exp_name, hz, records, interrupted):
    metadata = {
        "exp_name": exp_name,
        "record_type": "hoh_state_recording",
        "frame_root": frame_save_path,
        "hz": float(hz),
        "total_frames": len(records),
        "interrupted": bool(interrupted),
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
                _write_hoh_metadata(
                    frame_save_path,
                    FLAGS.exp_name,
                    FLAGS.hz,
                    records,
                    interrupted=interrupted,
                )
            elif interrupted and not FLAGS.save_on_interrupt:
                print("[record_hoh_states][interrupt] skipped save because --nosave_on_interrupt was set")
            else:
                print("[record_hoh_states] no frames recorded")
        finally:
            env.close()


if __name__ == "__main__":
    app.run(main)
