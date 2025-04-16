import os
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


from serl_launcher.networks.reward_classifier import load_classifier_func

from examples.utils import read_utils

from experiments.mappings import NEW_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "tennis_ball_pick", "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 100, "Number of successful demos to collect.")
flags.DEFINE_string("data_dir", "/home/qiangqiang/workspaces/data/2025-4-3/demo_data", "demo data dir")
# flags.DEFINE_string("data_dir", "/home/qiangqiang/workspaces/data/2025-4-3/test_data", "demo data dir")
flags.DEFINE_string("robot_urdf_path", "/home/qiangqiang/workspaces/HK_TACTEXO_DATA/denso_robot_with_ati_4.urdf", "robot urdf dir")

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

def comupute_reward(obs, classifier):

    sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
    classifier_output = sigmoid(classifier(obs))

    # 使用索引提取标量值
    classifier_score = classifier_output[0]
    return int(classifier_score > 0.45)

def main(_):

    assert FLAGS.exp_name in NEW_MAPPING, 'Experiment folder not found.'
    config = NEW_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=True)
    terminate = False
    transitions = []
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    trajectory = []
    returns = 0

    action_space = gym.spaces.Box(
            np.ones((23,), dtype=np.float32) * -1,
            np.ones((23,), dtype=np.float32),
        )
    
    # action_space = gym.spaces.Box(
    #         np.ones((7,), dtype=np.float32) * -1,
    #         np.ones((7,), dtype=np.float32),
    #     )
    if not os.path.exists("./demo_data"):
        os.makedirs("./demo_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./demo_data/{FLAGS.exp_name}_{success_needed}_demos_{uuid}.pkl"
    batch_size = 500
    
    actions = np.zeros(action_space.sample().shape) 
    data_dir = FLAGS.data_dir
    # print("env.observation_space.sample().shape = ", env.observation_space.sample()["front_camera"].shape)

    classifier = load_classifier_func(
        key=jax.random.PRNGKey(0),
        sample=env.observation_space.sample(),
        image_keys=config.classifier_keys,
        checkpoint_path=os.path.abspath("classifier_ckpt/"),
    )

    tcp_ori_list = []
    for collect_data_dir in sorted(os.listdir(data_dir)):
        collect_data_path = os.path.join(data_dir, collect_data_dir)
        if not os.path.isdir(collect_data_path):
            continue

        frame_dirs = sorted(
            [os.path.join(collect_data_path, d) for d in os.listdir(collect_data_path) if os.path.isdir(os.path.join(collect_data_path, d))],
            key=lambda folder: int(re.search(r'frame_(\d+)', os.path.basename(folder)).group(1)) if re.search(r'frame_(\d+)', os.path.basename(folder)) else float('inf')
        )

        clip_marks_json = os.path.join(collect_data_path, 'clip_marks.json')
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
                next_frame_path = os.path.join(collect_data_path, frame_dirs[i + 1])
                if not os.path.isdir(current_frame_path) or not os.path.isdir(next_frame_path):
                    continue

                obs, is_record_success = read_utils.get_frame_data(current_frame_path, FLAGS.robot_urdf_path)
                tcp_ori = obs["state"][3:7]  # 四元数部分
                tcp_ori_list.append(tcp_ori)
                next_obs, _ = read_utils.get_frame_data(next_frame_path, FLAGS.robot_urdf_path)

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
                reward = comupute_reward(obs, classifier)

                done = reward or terminate
                actions[:3] = next_obs["state"][:3]  # xyz坐标
                actions[3:7] = next_obs["state"][3:7]  # 四元数姿态
                actions[7:] = next_obs["state"][7:]  # leap_hand

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
                if (done and reward) or i == end_frame:
                    for transition in trajectory:
                        transitions.append(copy.deepcopy(transition))
                    pbar.update(1)
                    trajectory = []
                    returns = 0
                    terminate = False
                    if len(transitions) >= batch_size:
                        save_batch_to_pickle(transitions, file_name)
                        transitions = []

    if transitions:
        save_batch_to_pickle(transitions, file_name)
    print("record_finished")

    tcp_ori_array = np.stack(tcp_ori_list, axis=0)

    # 分别提取每一维并打印 min/max
    component_names = ["w", "x", "y", "z"]
    for i in range(4):
        min_val = np.min(tcp_ori_array[:, i])
        max_val = np.max(tcp_ori_array[:, i])
        print(f"tcp_ori {component_names[i]}: min = {min_val:.6f}, max = {max_val:.6f}")

if __name__ == "__main__":
    app.run(main)