import os
import sys
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
import time
import sys, threading, queue, termios, tty, select
import json
from pathlib import Path

# 提前输入export PYTHONPATH=$(pwd)/../serl_robot_infra:$PYTHONPATH
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../serl_robot_infra'))
# sys.path.insert(0, project_root)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../serl_launcher'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.mappings import NEW_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "tennis_ball_pick", "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 10, "Number of successful demos to collect.")
flags.DEFINE_integer("enable_tactile", 1, "evaluate pick or place task.")
flags.DEFINE_boolean("record_data", True, "Save robot/camera/tactile frame data while recording demos.")
flags.DEFINE_boolean("record_gaze", False, "Collect Pupil gaze/world frames while recording demos.")
flags.DEFINE_boolean("classifier", True, "Load JAX reward classifier during demo recording.")


# def _stdin_key_pressed(target_char="1"):
#     """若用户按下 target_char（默认 '1'）则返回 True。否则 False。"""
#     # 检查是否有可读的输入（不阻塞）
#     if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
#         ch = sys.stdin.read(1)
#         return ch == '1'
#     return False

class KeyReader(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.q = queue.Queue()
        self._stop = threading.Event()
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)  # 立即读取，无需回车

    def run(self):
        try:
            while not self._stop.is_set():
                if sys.stdin in select.select([sys.stdin], [], [], 0.01)[0]:
                    ch = sys.stdin.read(1)
                    self.q.put(ch)
        finally:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def get_key_nowait(self):
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self._stop.set()


def _get_unwrapped_env(env):
    return getattr(env, "unwrapped", env)


def _int_like(x):
    try:
        if isinstance(x, (list, tuple)):
            return int(_int_like(x[0]) if len(x) else 0)
        if isinstance(x, np.ndarray):
            return int(x.reshape(-1)[0])
        return int(x)
    except Exception:
        return int(x)


def _mark_episode_start(env, reason):
    root_env = _get_unwrapped_env(env)
    if hasattr(root_env, "mark_episode_start_for_recording"):
        start = root_env.mark_episode_start_for_recording()
        if start is not None:
            print(f"[record_demos][mark] start={start} @{reason}")
            return
    if hasattr(root_env, "global_frame_id"):
        root_env._cur_ep_start = int(root_env.global_frame_id)
        print(f"[record_demos][mark] start={root_env._cur_ep_start} @{reason}")


def _collect_episode_range(env):
    root_env = _get_unwrapped_env(env)
    if not hasattr(root_env, "end_episode_and_collect"):
        return None
    rng = root_env.end_episode_and_collect()
    if rng is None:
        return None
    if isinstance(rng, np.ndarray):
        if rng.size < 2:
            return None
        flat = rng.reshape(-1)
        return int(flat[0]), int(flat[1])
    if isinstance(rng, (list, tuple)) and len(rng) >= 2:
        return int(rng[0]), int(rng[1])
    return None


def _write_recording_metadata(env, exp_name, successes_needed, success_count, episode_records):
    root_env = _get_unwrapped_env(env)
    frame_root = Path(getattr(root_env, "frame_root", getattr(root_env, "frame_save_path", "./")))
    metadata = {
        "exp_name": exp_name,
        "successes_needed": int(successes_needed),
        "success_count": int(success_count),
        "frame_root": str(frame_root),
        "total_frames": int(getattr(root_env, "global_frame_id", -1)),
        "num_episodes": len(episode_records),
        "episode_ranges": episode_records,
        "created_at": datetime.datetime.now().isoformat(),
    }
    if hasattr(root_env, "gaze_marker_points_realsense"):
        metadata["gaze_display_markers"] = bool(getattr(root_env, "gaze_display_markers", False))
        metadata["gaze_marker_points_realsense"] = np.asarray(
            root_env.gaze_marker_points_realsense,
            dtype=float,
        ).tolist()
        metadata["gaze_realsense_size"] = [
            int(getattr(root_env, "gaze_rs_save_width", 640)),
            int(getattr(root_env, "gaze_rs_save_height", 480)),
        ]
    frame_root.mkdir(parents=True, exist_ok=True)
    meta_path = frame_root / "recording_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"[record_demos][metadata] saved {len(episode_records)} episode ranges to {meta_path}")
    return meta_path


def main(_):
    assert FLAGS.exp_name in NEW_MAPPING, 'Experiment folder not found.'
    collect_gaze = bool(FLAGS.record_gaze)
    config = NEW_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(
        fake_env=False,
        save_video=False,
        classifier=FLAGS.classifier,
        enable_tactile=FLAGS.enable_tactile,
        record_data=FLAGS.record_data or collect_gaze,
        record_gaze=collect_gaze,
    )
    
    fd = sys.stdin.fileno()
    old_term_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    
    obs, info = env.reset()
    if FLAGS.record_data or collect_gaze:
        _mark_episode_start(env, "first reset")
    transitions = []
    success_count = 0
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    trajectory = []
    returns = 0
    episode_records = []
    
    key_reader = KeyReader()
    key_reader.start()
    try:
        while success_count < success_needed:
            actions = np.zeros(env.action_space.sample().shape)
            # actions[1] = -0.1
            next_obs, rew, done, truncated, info = env.step(actions)
            # print("reward = ", rew)
            # print(f"obs[state] =  {obs['state']}")
            
            key = key_reader.get_key_nowait()
            while key is not None:
                if key == '1':
                    done = True
                    info = dict(info)
                    rew = 1.0
                    info['succeed'] = True
                key = key_reader.get_key_nowait()
            returns += rew
            if "intervene_action" in info:
                actions = info["intervene_action"]
            
            # print("actions taken: ", actions)
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
                    infos=copy.deepcopy(info),
                )
            )
            if 'grasp_penalty' in info:
                transition['grasp_penalty']= info['grasp_penalty']
            if 'robot_arm_penalty' in info:
                transition['robot_arm_penalty']= info['robot_arm_penalty']
                
            # print("info['robot_arm_penalty'] = ", info.get('robot_arm_penalty', None))
            # print("info['grasp_penalty'] = ", info.get('grasp_penalty', None))
            trajectory.append(transition)
            # if "is_pick" in info:
            #     is_pick = info["is_pick"]
            # else:
            #     is_pick = True
            
            pbar.set_description(f"Return: {returns}")

            obs = next_obs
            if done:
                if info["succeed"]:
                    # time.sleep(0.5)
                    # actions = np.zeros(env.action_space.sample().shape)
                    # # actions[6] = 1.0
                    # stable_obs, _, _, _, _ = env.step(actions)
                    # trajectory[-1]["next_observations"] = stable_obs
                    for transition in trajectory:
                        transitions.append(copy.deepcopy(transition))
                    success_count += 1
                    pbar.update(1)

                if FLAGS.record_data or collect_gaze:
                    try:
                        rng = _collect_episode_range(env)
                        if rng is not None:
                            episode_records.append(
                                {
                                    "episode_index": len(episode_records),
                                    "start_frame": int(rng[0]),
                                    "end_frame": int(rng[1]),
                                    "success": bool(info.get("succeed", False)),
                                    "num_frames": int(rng[1] - rng[0] + 1),
                                }
                            )
                            print(
                                f"[record_demos][recording] collected episode range: {rng}"
                            )
                        else:
                            print("[record_demos][recording] rng is None (episode had no frames?)")
                    except Exception as exc:
                        print(f"[record_demos][WARN] end_episode_and_collect failed: {exc}")

                trajectory = []
                returns = 0
                if FLAGS.exp_name == "tube_insertion":
                    env.unwrapped.open_hand(steps=20, step_time=0.05)
                    time.sleep(1.5)
                # elif FLAGS.exp_name == "tennis_ball_pick":
                #     env.move_up()
                input("reset env")
                obs, info = env.reset()
                if FLAGS.record_data or collect_gaze:
                    _mark_episode_start(env, "reset")
    finally:
        root_env = _get_unwrapped_env(env)
        save_owner = env if hasattr(env, "save_all_data_on_exit") else root_env
        if (FLAGS.record_data or collect_gaze) and hasattr(save_owner, "save_all_data_on_exit"):
            try:
                save_owner.save_all_data_on_exit()
            except Exception as exc:
                print(f"[record_demos][WARN] save_all_data_on_exit failed: {exc}")
        if FLAGS.record_data or collect_gaze:
            try:
                _write_recording_metadata(
                    env,
                    FLAGS.exp_name,
                    success_needed,
                    success_count,
                    episode_records,
                )
            except Exception as exc:
                print(f"[record_demos][WARN] write recording metadata failed: {exc}")
        key_reader.stop()
        if hasattr(env, "keyboard_process") and env.keyboard_process.is_alive():
            print("Shutting down keyboard process...")
            env.keyboard_process.terminate()
            env.keyboard_process.join()
        if hasattr(env, "close"):
            env.close()
            
    if not os.path.exists("./demo_data"):
        os.makedirs("./demo_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./demo_data/{FLAGS.exp_name}_{success_needed}_demos_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {success_needed} demos to {file_name}")

if __name__ == "__main__":
    app.run(main)
