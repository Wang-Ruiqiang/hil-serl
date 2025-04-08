import copy
import json
import cv2
import os
from tqdm import tqdm
import numpy as np
import gymnasium as gym
from examples.utils import kinematics_utils
import re
from collections import deque

class ObsHistoryBuffer:
    def __init__(self, obs_horizon=3, image_keys=("front_camera", "side_camera"), proprio_key="state"):
        self.obs_horizon = obs_horizon
        self.image_keys = image_keys
        self.proprio_key = proprio_key
        self.buffer = deque(maxlen=obs_horizon)

    def reset(self, first_obs):
        self.buffer.clear()
        for _ in range(self.obs_horizon):
            self.buffer.append(copy.deepcopy(first_obs))

    def append(self, obs):
        self.buffer.append(copy.deepcopy(obs))

    def get_stacked_obs(self):
        stacked_obs = {}
        for key in self.image_keys:
            frames = [o[key] for o in self.buffer]  # list of (H, W, 3)
            stacked_obs[key] = np.concatenate(frames, axis=-1)  # (H, W, 9)

        if self.proprio_key is not None:
            vecs = [o[self.proprio_key] for o in self.buffer]  # list of (23,)
            stacked_obs[self.proprio_key] = np.concatenate(vecs, axis=-1)  # (69,)

        return stacked_obs
    
    def get_success_fail_obs(self):
        stacked_obs = {}
        for key in self.image_keys:
            frames = [o[key] for o in self.buffer]  # list of (H, W, 3)
            stacked_obs[key] = np.stack(frames, axis=0)

        if self.proprio_key is not None:
            vecs = [o[self.proprio_key] for o in self.buffer]  # list of (23,)
            stacked_obs[self.proprio_key] = np.stack(vecs, axis=0)  # (69,)

        return stacked_obs
    

def get_frame_data(frame_path, robot_urdf_path):
    color_image_path = os.path.join(frame_path, "color_image.jpg")
    color_image_path2 = os.path.join(frame_path, "color_image2.jpg")
    # depth_image_path = os.path.join(frame_path, "depth_image.png")
    # depth_image_path2 = os.path.join(frame_path, "depth_image2.png")
    color_image = cv2.imread(color_image_path) if os.path.exists(color_image_path) else None
    color_image2 = cv2.imread(color_image_path2) if os.path.exists(color_image_path) else None
    # depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED) if os.path.exists(depth_image_path) else None
    # depth_image2 = cv2.imread(depth_image_path2, cv2.IMREAD_UNCHANGED) if os.path.exists(depth_image_path) else None
    joint_file_path = os.path.join(frame_path, "right_arm_joint.txt")
    record_success_failed_file = os.path.join(frame_path, "is_record_success.txt")
    hand_joint = None
    is_record_success = np.loadtxt(record_success_failed_file, dtype=int)

    if os.path.exists(joint_file_path):

        with open(joint_file_path, "r") as f:
            all_joint_values = np.array([float(x.strip()) for x in f.readlines()])
            # Change the order of robot arm joint data
            # wrist_joint_index = [2,0,1,3,4,5]
            # all_joint_values[:6] = all_joint_values[wrist_joint_index]

            hand_joint = all_joint_values[6:]
    tcp_pos, tcp_ori = kinematics_utils.comupute_forward_kinematics(all_joint_values, robot_urdf_path)

    state_flattened = np.concatenate([
        np.array(tcp_pos, dtype=np.float32).flatten(),
        np.array(tcp_ori, dtype=np.float32).flatten(),
        np.array(hand_joint, dtype=np.float32).flatten()
    ])

    resized_image = cv2.resize(color_image, (320,240))
    resized_image2 = cv2.resize(color_image2, (320,240))
    front_camera_image = resized_image[..., ::-1]
    side_camera_image = resized_image2[..., ::-1]

    obs = {
        "front_camera": front_camera_image,
        "side_camera": side_camera_image,
        "state": state_flattened
    }
    # print("state_flattened = ", state_flattened)
    # cv2.imwrite("front_camera_image.jpg", front_camera_image)
    # input("enter")
    return obs, is_record_success

def read_data(robot_urdf_path, is_evaluate_classifier=False):
    data = []
    action_space = gym.spaces.Box(
        np.ones((23,), dtype=np.float32) * -1,
        np.ones((23,), dtype=np.float32),
    )
    # action_space = gym.spaces.Box(
    #     np.ones((7,), dtype=np.float32) * -1,
    #     np.ones((7,), dtype=np.float32),
    # )
    actions = np.zeros(action_space.sample().shape)
    if is_evaluate_classifier:
        data_dir = "/home/qiangqiang/workspaces/data/2025-4-3/test_data"
    else:
        data_dir = "/home/qiangqiang/workspaces/data/2025-4-3/demo_data"
    for collect_data_dir in sorted(os.listdir(data_dir)):
        collect_data_path = os.path.join(data_dir, collect_data_dir)
        if not os.path.isdir(collect_data_path):
            continue

        # 获取 collect_data_path 目录下所有名为 frame_xxx 的子目录，并按照 xxx 数值大小排序。
        frame_dirs = sorted(
            [os.path.join(collect_data_path, d) for d in os.listdir(collect_data_path) if os.path.isdir(os.path.join(collect_data_path, d))],
            key=lambda folder: int(re.search(r'frame_(\d+)', os.path.basename(folder)).group(1)) if re.search(r'frame_(\d+)', os.path.basename(folder)) else float('inf')
        )
        clip_marks_json = os.path.join(collect_data_path, 'clip_marks.json')
        with open(clip_marks_json, 'r') as f:
                    clip_marks = json.load(f)


        for clip in clip_marks:
            start_frame = int(clip['start'].split('_')[-1])
            end_frame = int(clip['end'].split('_')[-1])
            
            for i in list(range(start_frame, end_frame+1)):
            # for i in range(len(frame_dirs) - 1):
                current_frame_path = os.path.join(collect_data_path, frame_dirs[i])
                next_frame_path = os.path.join(collect_data_path, frame_dirs[i + 1])
                if not os.path.isdir(current_frame_path) or not os.path.isdir(next_frame_path):
                    continue

                obs, is_record_success= get_frame_data(current_frame_path, robot_urdf_path)
                next_obs, _ = get_frame_data(next_frame_path, robot_urdf_path)

                transition = copy.deepcopy(
                    dict(
                        observations=obs,
                        next_observations=next_obs,
                        actions=actions,
                        is_record_success=is_record_success,
                        rewards=0,
                        masks=1.0,
                        dones=0,
                    )
                )
                data.append(transition)

    return data
