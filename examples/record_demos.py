import os
import sys
from tqdm import tqdm
import numpy as np
import copy
import datetime
import re
import shutil
import json
from absl import app, flags
import time

# 提前输入export PYTHONPATH=$(pwd)/../serl_robot_infra:$PYTHONPATH
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../serl_robot_infra'))
# sys.path.insert(0, project_root)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../serl_launcher'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.mappings import NEW_MAPPING
from examples.utils.runtime import KeyReader

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "flip_object", "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 20, "Number of successful demos to collect.")
flags.DEFINE_integer("enable_tactile", 1, "evaluate pick or place task.")
flags.DEFINE_integer(
    "max_episode_steps",
    0,
    "Maximum episode length while recording. Use 0 to disable the limit.",
)
flags.DEFINE_boolean("record_data", True, "Save raw frame data for later demo/classifier export.")
flags.DEFINE_boolean("classifier", True, "Load reward classifier while recording demos.")
flags.DEFINE_string(
    "frame_save_path",
    "",
    "Directory for raw frame_xxx data when --record_data is enabled. If empty, auto-create one under --record_root.",
)
flags.DEFINE_string(
    "record_root",
    "/home/wrq/workspaces/HK_TACEXO_WANG/bc_data",
    "Root directory for auto-named raw recorded demo folders.",
)
flags.DEFINE_string(
    "record_task_name",
    "",
    "Task name used in auto-created folder names. Defaults to a short name derived from --exp_name.",
)
flags.DEFINE_string(
    "record_source",
    "teleop",
    "Source tag used in auto-created folder names.",
)
flags.DEFINE_string(
    "success_label_name",
    "is_record_success.txt",
    "Per-frame success label filename.",
)
flags.DEFINE_boolean(
    "save_on_interrupt",
    True,
    "Save buffered raw frames when Ctrl+C interrupts recording. Disable for fast abort.",
)


def _get_unwrapped_env(env):
    return getattr(env, "unwrapped", env)


def _format_debug_array(value, precision=4):
    array = np.asarray(value).reshape(-1)
    return np.array2string(array, precision=precision, suppress_small=True)


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
    return os.path.abspath(
        os.path.join(task_root, f"{prefix}_{next_index:02d}")
    )


def _recording_paths(frame_save_path):
    return {
        "raw_frames": os.path.abspath(frame_save_path),
        "metadata": os.path.join(os.path.abspath(frame_save_path), "recording_metadata.json"),
        "path_note": os.path.join(os.path.abspath(frame_save_path), "recording_path.txt"),
        "per_frame_success_label": FLAGS.success_label_name,
    }


def _write_recording_path_note(frame_save_path):
    paths = _recording_paths(frame_save_path)
    os.makedirs(paths["raw_frames"], exist_ok=True)
    with open(paths["path_note"], "w", encoding="utf-8") as f:
        for key, value in paths.items():
            f.write(f"{key}: {value}\n")
    return paths


def _print_recording_paths(paths):
    print("\n" + "=" * 80)
    print("[record_demos][recording paths]")
    print(f"  raw frames: {paths['raw_frames']}")
    print(f"  metadata: {paths['metadata']}")
    print(f"  path note: {paths['path_note']}")
    print(f"  success label per frame: {paths['per_frame_success_label']}")
    print("=" * 80 + "\n")


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


MULTI_STAGE_RECORDING = {
    "tennis_ball_place": ("pick", "place"),
    "twist_bottle_cap": ("lid_grip", "twist"),
}


def _initial_record_stage(exp_name):
    stages = MULTI_STAGE_RECORDING.get(exp_name)
    return stages[0] if stages else exp_name


def _record_stage_from_info(exp_name, info):
    stages = MULTI_STAGE_RECORDING.get(exp_name)
    if not stages:
        return exp_name
    return stages[0] if info.get("is_pick", True) else stages[1]


def _iter_frame_dirs(frame_root):
    if not frame_root or not os.path.isdir(frame_root):
        return []

    def frame_index(path):
        match = re.search(r"frame_(\d+)$", os.path.basename(path))
        return int(match.group(1)) if match else float("inf")

    return sorted(
        (
            os.path.join(frame_root, name)
            for name in os.listdir(frame_root)
            if os.path.isdir(os.path.join(frame_root, name))
            and name.startswith("frame_")
        ),
        key=frame_index,
    )


def _terminal_success_by_frame(episode_records):
    labels = {}
    for episode in episode_records:
        if episode.get("success", False):
            labels[int(episode["end_frame"])] = 1
    return labels


def _frame_record_map(frame_stage_records):
    return {
        int(record["frame_id"]): record
        for record in frame_stage_records
        if record.get("frame_id") is not None
    }


def _copy_if_exists(src, dst):
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copyfile(src, dst)


def _postprocess_recorded_frames(frame_root, episode_records, frame_success_labels=None, frame_stage_records=None):
    terminal_labels = _terminal_success_by_frame(episode_records)
    frame_success_labels = frame_success_labels or {}
    stage_records = _frame_record_map(frame_stage_records or [])
    tactile_aliases = {
        "thumb_raw_image.jpg": "rthumb_raw_image.jpg",
        "thumb_heat_map.jpg": "rthumb_deform_image.jpg",
        "index_raw_image.jpg": "rindex_raw_image.jpg",
        "index_heat_map.jpg": "rindex_deform_image.jpg",
        "middle_raw_image.jpg": "rmiddle_raw_image.jpg",
        "middle_heat_map.jpg": "rmiddle_deform_image.jpg",
    }

    for frame_dir in _iter_frame_dirs(frame_root):
        match = re.search(r"frame_(\d+)$", os.path.basename(frame_dir))
        if not match:
            continue
        frame_id = int(match.group(1))
        stage_record = stage_records.get(frame_id, {})
        label = frame_success_labels.get(
            frame_id,
            stage_record.get("reward", terminal_labels.get(frame_id, 0)),
        )
        with open(os.path.join(frame_dir, FLAGS.success_label_name), "w", encoding="utf-8") as f:
            f.write(f"{label}\n")
        for src_name, dst_name in tactile_aliases.items():
            _copy_if_exists(
                os.path.join(frame_dir, src_name),
                os.path.join(frame_dir, dst_name),
            )


def _mark_episode_start(env, reason):
    root_env = _get_unwrapped_env(env)
    if hasattr(root_env, "mark_episode_start_for_recording"):
        start = root_env.mark_episode_start_for_recording()
        print(f"[record_demos][mark] start={start} @{reason}")


def _collect_episode_range(env):
    root_env = _get_unwrapped_env(env)
    if not hasattr(root_env, "end_episode_and_collect"):
        return None
    return root_env.end_episode_and_collect()


def _current_recorded_frame_id(env):
    root_env = _get_unwrapped_env(env)
    if not hasattr(root_env, "frame_count"):
        return None
    frame_id = int(root_env.frame_count) - 1
    return frame_id if frame_id >= 0 else None


def _classifier_success_label(reward, info):
    try:
        return float(np.asarray(reward).item())
    except Exception:
        if "succeed" in info:
            return 1.0 if bool(info["succeed"]) else 0.0
        return 0.0


def _stage_ranges_for_episode(frame_range, frame_stage_records):
    if frame_range is None:
        return {}
    start_frame, end_frame = frame_range
    records = [
        record
        for record in frame_stage_records
        if start_frame <= int(record.get("frame_id", -1)) <= end_frame
    ]
    if not records:
        return {}

    ranges = {}
    for stage in ("pick", "place", "lid_grip", "twist"):
        stage_frames = [
            int(record["frame_id"])
            for record in records
            if record.get("stage") == stage
        ]
        if stage_frames:
            ranges[f"{stage}_range"] = {
                "start_frame": min(stage_frames),
                "end_frame": max(stage_frames),
            }

    place_start_frames = [
        int(record["frame_id"])
        for record in records
        if record.get("stage_after") == "place"
    ]
    if place_start_frames:
        place_start = min(place_start_frames)
        ranges["place_range"] = {
            "start_frame": place_start,
            "end_frame": int(end_frame),
        }
    twist_start_frames = [
        int(record["frame_id"])
        for record in records
        if record.get("stage_after") == "twist"
    ]
    if twist_start_frames:
        twist_start = min(twist_start_frames)
        ranges["twist_range"] = {
            "start_frame": twist_start,
            "end_frame": int(end_frame),
        }
    return ranges


def _append_episode_record(
    episode_records,
    frame_range,
    success=False,
    interrupted=False,
    frame_stage_records=None,
):
    if frame_range is None:
        return False
    start_frame, end_frame = frame_range
    episode_record = {
        "episode_index": len(episode_records),
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "success": bool(success),
        "interrupted": bool(interrupted),
        "num_frames": int(end_frame - start_frame + 1),
    }
    episode_record.update(_stage_ranges_for_episode(frame_range, frame_stage_records or []))
    episode_records.append(episode_record)
    return True


def _write_recording_metadata(
    env,
    exp_name,
    successes_needed,
    success_count,
    episode_records,
    frame_stage_records=None,
):
    root_env = _get_unwrapped_env(env)
    if hasattr(root_env, "write_recording_metadata"):
        metadata_path = root_env.write_recording_metadata(
            exp_name,
            successes_needed,
            success_count,
            episode_records,
        )
        if frame_stage_records is not None:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            metadata["frame_stage_records"] = frame_stage_records
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
        print(f"[record_demos][metadata] saved {metadata_path}")


def main(_):
    assert FLAGS.exp_name in NEW_MAPPING, 'Experiment folder not found.'
    config = NEW_MAPPING[FLAGS.exp_name]()
    classifier_enabled = FLAGS.classifier
    frame_save_path = FLAGS.frame_save_path or _default_frame_save_path(FLAGS.exp_name)
    if FLAGS.record_data:
        recording_paths = _write_recording_path_note(frame_save_path)
        _print_recording_paths(recording_paths)
    if FLAGS.exp_name == "flip_object":
        print(
            "[record_demos][flip_object] hand state test: "
            "press i to advance state1->state2->state3->state4->state5->state1, "
            "press o to move backward within the sequence."
        )
    env = config.get_environment(
        fake_env=False,
        save_video=False,
        classifier=classifier_enabled,
        enable_tactile=FLAGS.enable_tactile,
        record_data=FLAGS.record_data,
        frame_save_path=frame_save_path,
    )
    base_env = _get_unwrapped_env(env)
    if not hasattr(base_env, "max_episode_length"):
        raise AttributeError(
            "Recording environment does not expose max_episode_length."
        )
    base_env.max_episode_length = (
        float("inf") if FLAGS.max_episode_steps <= 0 else FLAGS.max_episode_steps
    )
    print(
        "[record_demos] max episode length: "
        f"{'disabled' if FLAGS.max_episode_steps <= 0 else FLAGS.max_episode_steps}"
    )

    obs, info = env.reset()
    if FLAGS.record_data:
        _mark_episode_start(env, "first reset")
    transitions = []
    success_count = 0
    episode_count = 0
    success_needed = FLAGS.successes_needed
    count_episodes_for_progress = FLAGS.exp_name == "flip_object"
    pbar = tqdm(total=success_needed)
    trajectory = []
    returns = 0
    episode_records = []
    frame_success_labels = {}
    frame_stage_records = []
    current_stage = _initial_record_stage(FLAGS.exp_name)
    
    key_reader = KeyReader()
    key_reader.start()
    interrupted = False
    try:
        while (episode_count if count_episodes_for_progress else success_count) < success_needed:
            actions = np.zeros(env.action_space.sample().shape)
            # actions[1] = -0.1
            stage_before = current_stage
            next_obs, rew, done, truncated, info = env.step(actions)
            manual_success = False
            manual_failure = False
            current_stage = _record_stage_from_info(FLAGS.exp_name, info)
            # print("reward = ", rew)
            print(f"obs[state] =  {obs['state']}")
            
            key = key_reader.get_key_nowait()
            while key is not None:
                if key == '1':
                    done = True
                    manual_success = True
                    info = dict(info)
                    rew = 1.0
                    info['succeed'] = True
                elif key == '2':
                    done = True
                    manual_success = False
                    manual_failure = True
                    info = dict(info)
                    rew = 0.0
                    info['succeed'] = False
                key = key_reader.get_key_nowait()
            if FLAGS.record_data:
                frame_id = _current_recorded_frame_id(env)
                if frame_id is not None:
                    if manual_success:
                        reward_label = 1.0
                    elif manual_failure:
                        reward_label = 0.0
                    else:
                        reward_label = _classifier_success_label(rew, info)
                    frame_success_labels[frame_id] = reward_label
                    frame_stage_records.append(
                        {
                            "frame_id": int(frame_id),
                            "stage": stage_before,
                            "stage_after": current_stage,
                            "reward": reward_label,
                            "is_pick": bool(
                                stage_before == _initial_record_stage(FLAGS.exp_name)
                            ),
                        }
                    )
            returns += rew
            if "intervene_action" in info:
                actions = info["intervene_action"]
            
            print(
                "[record_demos][step] "
                f"obs_state={_format_debug_array(obs['state'])} "
                f"transition_action={_format_debug_array(actions)} "
                f"reward={rew} done={done}"
            )
            # force_end = _stdin_key_pressed("1")
            # if force_end:
            #     done = True
            #     info = dict(info)  # 防止底层是只读映射
            #     info["succeed"] = True
            
            transition = copy.deepcopy(
                dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=rew,
                    masks=1.0 - done,
                    dones=done,
                )
            )
            if 'grasp_penalty' in info:
                transition['grasp_penalty']= info['grasp_penalty']
            if 'robot_arm_penalty' in info:
                transition['robot_arm_penalty']= info['robot_arm_penalty']
                
            print("info['robot_arm_penalty'] = ", info.get('robot_arm_penalty', None))
            print("info['grasp_penalty'] = ", info.get('grasp_penalty', None))
            trajectory.append(transition)
            # if "is_pick" in info:
            #     is_pick = info["is_pick"]
            # else:
            #     is_pick = True
            
            pbar.set_description(f"Return: {returns}")

            obs = next_obs
            if done:
                succeeded = bool(info.get("succeed", False))
                if succeeded:
                    # time.sleep(0.5)
                    # actions = np.zeros(env.action_space.sample().shape)
                    # # actions[6] = 1.0
                    # stable_obs, _, _, _, _ = env.step(actions)
                    # trajectory[-1]["next_observations"] = stable_obs
                    for transition in trajectory:
                        transitions.append(copy.deepcopy(transition))
                    success_count += 1
                    pbar.update(1)
                elif count_episodes_for_progress:
                    pbar.update(1)
                if FLAGS.record_data:
                    frame_range = _collect_episode_range(env)
                    if _append_episode_record(
                        episode_records,
                        frame_range,
                        success=succeeded,
                        interrupted=False,
                        frame_stage_records=frame_stage_records,
                    ):
                        print(f"[record_demos][recording] episode range={frame_range}")
                episode_count += 1
                trajectory = []
                returns = 0
                if FLAGS.exp_name == "tube_insertion":
                    env.unwrapped.open_hand(steps=20, step_time=0.05)
                    time.sleep(1.5)
                elif FLAGS.exp_name == "tennis_ball_pick":
                    env.move_up()
                input("reset env")
                obs, info = env.reset()
                current_stage = _initial_record_stage(FLAGS.exp_name)
                if FLAGS.record_data:
                    _mark_episode_start(env, "reset")
    except KeyboardInterrupt:
        interrupted = True
        print("\n[record_demos][interrupt] Ctrl+C received, shutting down cleanly...")
    finally:
        key_reader.stop()
        if FLAGS.record_data and trajectory:
            frame_range = _collect_episode_range(env)
            _append_episode_record(
                episode_records,
                frame_range,
                success=False,
                interrupted=True,
                frame_stage_records=frame_stage_records,
            )
        save_owner = env if hasattr(env, "save_all_data_on_exit") else _get_unwrapped_env(env)
        if FLAGS.record_data and hasattr(save_owner, "save_all_data_on_exit"):
            if interrupted and not FLAGS.save_on_interrupt:
                print("[record_demos][interrupt] skipped raw frame save because --nosave_on_interrupt was set")
            else:
                print("[record_demos][save] writing buffered raw frames, this can take a while...")
                save_owner.save_all_data_on_exit()
                _postprocess_recorded_frames(
                    frame_save_path,
                    episode_records,
                    frame_success_labels=frame_success_labels,
                    frame_stage_records=frame_stage_records,
                )
                _write_recording_metadata(
                    env,
                    FLAGS.exp_name,
                    success_needed,
                    success_count,
                    episode_records,
                    frame_stage_records=frame_stage_records,
                )
        if hasattr(env, "keyboard_process") and env.keyboard_process.is_alive():
            print("Shutting down keyboard process...")
            env.keyboard_process.terminate()
            env.keyboard_process.join()
        env.close()
            
    if FLAGS.record_data:
        _print_recording_paths(_recording_paths(frame_save_path))
    print(
        "[record_demos][done] raw frames and metadata saved. "
        "Run export_recorded_demos.py to generate demo pkl files."
    )

if __name__ == "__main__":
    app.run(main)
