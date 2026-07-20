import os
import sys
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
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
flags.DEFINE_string("exp_name", "tennis_ball_place", "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 20, "Number of successful demos to collect.")
flags.DEFINE_integer("enable_tactile", 1, "evaluate pick or place task.")
flags.DEFINE_boolean("record_data", False, "Save raw frame data for later demo/classifier export.")
flags.DEFINE_boolean("classifier", True, "Load reward classifier while recording demos.")
flags.DEFINE_string(
    "frame_save_path",
    "",
    "Directory for raw frame_xxx data when --record_data is enabled.",
)


def _get_unwrapped_env(env):
    return getattr(env, "unwrapped", env)


def _default_frame_save_path(exp_name):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.abspath(os.path.join("recorded_data", exp_name, timestamp))


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


def _append_episode_record(episode_records, frame_range, success=False, interrupted=False):
    if frame_range is None:
        return False
    start_frame, end_frame = frame_range
    episode_records.append(
        {
            "episode_index": len(episode_records),
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "success": bool(success),
            "interrupted": bool(interrupted),
            "num_frames": int(end_frame - start_frame + 1),
        }
    )
    return True


def _write_recording_metadata(env, exp_name, successes_needed, success_count, episode_records):
    root_env = _get_unwrapped_env(env)
    if hasattr(root_env, "write_recording_metadata"):
        metadata_path = root_env.write_recording_metadata(
            exp_name,
            successes_needed,
            success_count,
            episode_records,
        )
        print(f"[record_demos][metadata] saved {metadata_path}")


def main(_):
    assert FLAGS.exp_name in NEW_MAPPING, 'Experiment folder not found.'
    config = NEW_MAPPING[FLAGS.exp_name]()
    frame_save_path = FLAGS.frame_save_path or _default_frame_save_path(FLAGS.exp_name)
    env = config.get_environment(
        fake_env=False,
        save_video=False,
        classifier=FLAGS.classifier,
        enable_tactile=FLAGS.enable_tactile,
        record_data=FLAGS.record_data,
        frame_save_path=frame_save_path,
    )

    obs, info = env.reset()
    if FLAGS.record_data:
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
            print(f"obs[state] =  {obs['state']}")
            
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
            
            print("actions taken: ", actions)
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
                if FLAGS.record_data:
                    frame_range = _collect_episode_range(env)
                    if _append_episode_record(
                        episode_records,
                        frame_range,
                        success=succeeded,
                        interrupted=False,
                    ):
                        print(f"[record_demos][recording] episode range={frame_range}")
                trajectory = []
                returns = 0
                if FLAGS.exp_name == "tube_insertion":
                    env.unwrapped.open_hand(steps=20, step_time=0.05)
                    time.sleep(1.5)
                elif FLAGS.exp_name == "tennis_ball_pick":
                    env.move_up()
                input("reset env")
                obs, info = env.reset()
                if FLAGS.record_data:
                    _mark_episode_start(env, "reset")
    finally:
        key_reader.stop()
        if FLAGS.record_data:
            frame_range = _collect_episode_range(env)
            _append_episode_record(
                episode_records,
                frame_range,
                success=False,
                interrupted=True,
            )
        save_owner = env if hasattr(env, "save_all_data_on_exit") else _get_unwrapped_env(env)
        if FLAGS.record_data and hasattr(save_owner, "save_all_data_on_exit"):
            save_owner.save_all_data_on_exit()
            _write_recording_metadata(
                env,
                FLAGS.exp_name,
                success_needed,
                success_count,
                episode_records,
            )
        if hasattr(env, "keyboard_process") and env.keyboard_process.is_alive():
            print("Shutting down keyboard process...")
            env.keyboard_process.terminate()
            env.keyboard_process.join()
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
