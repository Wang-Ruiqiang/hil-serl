import os
import sys
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
import time
import json
import cv2
import gymnasium as gym
import pinocchio as pin
import jax
import jax.numpy as jnp
import re
from scipy.spatial.transform import Rotation as R

# 提前输入export PYTHONPATH=$(pwd)/../serl_robot_infra:$PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../serl_robot_infra'))
sys.path.insert(0, project_root)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../serl_launcher'))
sys.path.insert(0, project_root)

from serl_launcher.networks.reward_classifier import load_classifier_func

from examples.utils import read_utils

from experiments.mappings import NEW_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "tennis_ball_pick", "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 25, "Number of successful demos to collect.")
flags.DEFINE_string("data_dir", "/home/ruiqiang/workspaces/HK_TACEXO_WANG/recorded_data/pick_demo_data", "demo data dir")
# flags.DEFINE_string("data_dir", "/home/qiangqiang/workspaces/data/2025-4-3/test_data", "demo data dir")
flags.DEFINE_string("robot_urdf_path", "/home/ruiqiang/workspaces/HK_TACEXO_WANG/hil-serl/examples/urdf/denso_robot_with_ati_4.urdf", "robot urdf dir")
flags.DEFINE_boolean("is_pick_task", True, "read exist data or not.")
flags.DEFINE_integer("enable_tactile", 1, "evaluate pick or place task.")

# camera_keys = ["front_camera", "side_camera"]
# classifier_keys = ["front_camera", "side_camera"]

def save_batch_to_pickle(batch_data, file_path):
    """
    将批次数据追加保存到 pickle 文件。
    :param batch_data: 要保存的批次数据
    :param file_path: 保存路径
    """
    with open(file_path, "ab") as f: 
        pkl.dump(batch_data, f)
        print(f"Saved batch of {len(batch_data)} transitions to {file_path}")

def compute_reward(obs, classifier):

    sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
    prob = sigmoid(classifier(obs)).item()
    success = prob > 0.8
    reward = 1 if success else 0
    # state = obs["state"]
    # ee_pos = state[0, :3] if state.ndim > 1 else state[:3]
    # if ee_pos[1] > -0.13 and ee_pos[2] < 0.14:
    #     reward -= 0.05
    return reward

def main(_):

    assert FLAGS.exp_name in NEW_MAPPING, 'Experiment folder not found.'
    config = NEW_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=True, save_video=False, classifier=True, enable_tactile=FLAGS.enable_tactile)
    terminate = False
    transitions = []
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    trajectory = []
    returns = 0
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,))
        
    if not os.path.exists("./demo_data"):
        os.makedirs("./demo_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # file_name = f"./demo_data/{FLAGS.exp_name}_{success_needed}_demos_{uuid}.pkl"
    batch_size = 500
    
    actions = np.zeros(action_space.sample().shape) 
    data_dir = FLAGS.data_dir
    # print("env.observation_space.sample().shape = ", env.observation_space.sample()["front_camera"].shape)
    
    print("config.classifier_keys = ", config.classifier_keys)
    if FLAGS.exp_name == "twist_bottle_cap":
        classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=config.classifier_keys,
                image_key_weights=config.classifier_key_weights,
                checkpoint_path=os.path.abspath("classifier_ckpt_pick_bottle_twist/"),
            )
    elif FLAGS.exp_name == "tube_insertion":
        if FLAGS.enable_tactile:
            classifier = load_classifier_func(
                    key=jax.random.PRNGKey(0),
                    sample=env.observation_space.sample(),
                    image_keys=config.classifier_keys,
                    image_key_weights=config.classifier_key_weights,
                    checkpoint_path=os.path.abspath("classifier_ckpt_tube_insertion/"),
                )
        else:
            classifier = load_classifier_func(
                    key=jax.random.PRNGKey(0),
                    sample=env.observation_space.sample(),
                    image_keys=config.classifier_keys,
                    image_key_weights=config.classifier_key_weights,
                    checkpoint_path=os.path.abspath("classifier_ckpt_tube_insertion_no_tactile/"),
                )
    elif FLAGS.exp_name == "tennis_ball_pick":
        if FLAGS.is_pick_task:
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=config.classifier_keys,
                image_key_weights=config.classifier_key_weights,
                checkpoint_path=os.path.abspath("classifier_ckpt_ball_pick/"),
            )
        else:
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=config.classifier_keys,
                image_key_weights=config.classifier_key_weights,
                checkpoint_path=os.path.abspath("classifier_ckpt_ball_place/"),
            )

    

    tcp_ori_list = []
    is_pick = True
    for collect_data_dir in sorted(os.listdir(data_dir)):
        collect_data_path = os.path.join(data_dir, collect_data_dir)
        if not os.path.isdir(collect_data_path):
            continue

        frame_dirs = sorted(
            [os.path.join(collect_data_path, d) for d in os.listdir(collect_data_path) if os.path.isdir(os.path.join(collect_data_path, d))],
            key=lambda folder: int(re.search(r'frame_(\d+)', os.path.basename(folder)).group(1)) if re.search(r'frame_(\d+)', os.path.basename(folder)) else float('inf')
        )

        # clip_marks_json = os.path.join(collect_data_path, 'clip_marks.json')
        if FLAGS.exp_name == "twist_bottle_cap" or FLAGS.exp_name == "tube_insertion":
            print("clip_marks")
            is_pick = False
            clip_marks_json = os.path.join(collect_data_path, 'clip_marks.json')

        elif FLAGS.exp_name == "tennis_ball_pick":
            if FLAGS.is_pick_task:
                print("clip_marks_pick")
                is_pick = True
                clip_marks_json = os.path.join(collect_data_path, 'clip_marks_pick.json')
            else:
                print("clip_marks_place")
                is_pick = False
                clip_marks_json = os.path.join(collect_data_path, 'clip_marks_place.json')
            
        with open(clip_marks_json, 'r') as f:
            clip_marks = json.load(f)

        
        history_obs = read_utils.ObsHistoryBuffer(obs_horizon=3)
        history_next_obs = read_utils.ObsHistoryBuffer(obs_horizon=3)
        for clip in clip_marks:
            start_frame = int(clip['start'].split('_')[-1])
            end_frame = int(clip['end'].split('_')[-1])
            # print("start_frame = ", start_frame)
            # print("end_frame = ", end_frame)
            # clip_length = end_frame - start_frame + 1 # include frame 0

            # for i in range(50):
            
            for i in list(range(start_frame, end_frame+1)):
                current_frame_path = os.path.join(collect_data_path, frame_dirs[i])
                next_frame_path = os.path.join(collect_data_path, frame_dirs[i + 1]) if i < end_frame else current_frame_path

                obs, is_record_success, grip_action = read_utils.get_frame_data(current_frame_path, FLAGS.robot_urdf_path, FLAGS.enable_tactile)
                if i == end_frame:
                    next_obs = obs
                else:
                    next_frame_path = os.path.join(collect_data_path, frame_dirs[i + 1])
                    next_obs, _, _ = read_utils.get_frame_data(next_frame_path, FLAGS.robot_urdf_path, FLAGS.enable_tactile)
                # print("obs state shape = ", obs["state"].shape)
                # input("debug")
                tcp_ori = obs["state"][3:7]  # 四元数部分
                tcp_ori_list.append(tcp_ori)

                if i == start_frame:
                    history_obs.reset(obs)
                    history_next_obs.reset(next_obs)
                else:
                    history_obs.append(obs)
                    history_next_obs.append(next_obs)
                stacked_obs = history_obs.get_success_fail_obs()
                stacked_next_obs = history_next_obs.get_success_fail_obs()
                # print("stacked_obs['front_camera'].shape = ", stacked_obs['front_camera'].shape)
                # print("stacked_next_obs['front_camera'].shape = ", stacked_obs['front_camera'].shape)
                # print("obs keys:", obs.keys())
                # if is_pick:
                #     reward = compute_reward(obs, classifier_pick)
                # else:
                reward = compute_reward(obs, classifier)

                done = reward or terminate

                if FLAGS.exp_name == "tennis_ball_pick":
                    ACTION_SCALE = (0.02, 0.02, 0.02)
                elif FLAGS.exp_name == "tube_insertion":
                    ACTION_SCALE = (0.005, 0.005, 0.05)
                delta_pos = next_obs["state"][:3] - obs["state"][:3]
                actions[:3] = delta_pos / ACTION_SCALE[0]
                low  = np.array([-1, -1, -1, -1, -1, -1, -1], dtype=np.float32)
                high = np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
                actions = np.clip(actions, low, high)
                
                current_quat = obs["state"][3:7]  # wxyz
                next_quat = next_obs["state"][3:7]

                current_euler = R.from_quat([current_quat[1], current_quat[2], current_quat[3], current_quat[0]]).as_euler("xyz")
                next_euler = R.from_quat([next_quat[1], next_quat[2], next_quat[3], next_quat[0]]).as_euler("xyz")

                delta_euler = next_euler - current_euler
                # actions[3:6] = delta_euler
                actions[3:6] = 0.0
                actions[6] = grip_action

                transition = copy.deepcopy(
                    dict(
                        observations=obs,
                        next_observations=next_obs,
                        actions=actions,
                        rewards=reward,
                        masks=1.0 - done,
                        dones=done,
                    )
                )

                trajectory.append(transition)
                returns += reward
        
                pbar.set_description(f"Return: {returns}")

                obs = next_obs
                if i == end_frame:
                    for transition in trajectory:
                        transitions.append(copy.deepcopy(transition))
                    pbar.update(1)
                    trajectory = []
                    # returns = 0
                    terminate = False
                    # if len(transitions) >= batch_size:
                    #     save_batch_to_pickle(transitions, file_name)
                    #     transitions = []

    # if transitions:
    #     save_batch_to_pickle(transitions, file_name)
    print("record_finished")
    # env.close()
    if not os.path.exists("./demo_data"):
        os.makedirs("./demo_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if FLAGS.exp_name == "twist_bottle_cap" or FLAGS.exp_name == "tube_insertion":
        file_name = f"./demo_data/{FLAGS.exp_name}_{success_needed}_demos_{uuid}.pkl"
    elif FLAGS.exp_name == "tennis_ball_pick":
        if FLAGS.is_pick_task:
            file_name = f"./demo_data/{FLAGS.exp_name}_pick_{success_needed}_demos_{uuid}.pkl"
        else:
            file_name = f"./demo_data/{FLAGS.exp_name}_{success_needed}_demos_{uuid}.pkl"
            
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {success_needed} demos to {file_name}")

    tcp_ori_array = np.stack(tcp_ori_list, axis=0)

    # 分别提取每一维并打印 min/max
    component_names = ["w", "x", "y", "z"]
    for i in range(4):
        min_val = np.min(tcp_ori_array[:, i])
        max_val = np.max(tcp_ori_array[:, i])
        print(f"tcp_ori {component_names[i]}: min = {min_val:.6f}, max = {max_val:.6f}")

if __name__ == "__main__":
    app.run(main)