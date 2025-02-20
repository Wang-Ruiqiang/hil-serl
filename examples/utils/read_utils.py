import copy
import json
import cv2
import os
from tqdm import tqdm
import numpy as np
from examples.utils import kinematics_utils

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
    record_success_failed_file = os.path.join(frame_path, "record_success_failed.txt")
    robot_joint = None
    hand_joint = None
    is_record_success = np.loadtxt(record_success_failed_file, dtype=int)
    if os.path.exists(joint_file_path):
        with open(joint_file_path, "r") as f:
            all_joint_values = np.array([float(x.strip()) for x in f.readlines()])
            robot_joint = all_joint_values[:6]
            hand_joint = all_joint_values[6:]
    tcp_pos, tcp_ori = kinematics_utils.comupute_forward_kinematics(all_joint_values, robot_urdf_path)

    state_flattened = np.concatenate([
        np.array(tcp_pos, dtype=np.float32).flatten(),
        np.array(tcp_ori, dtype=np.float32).flatten(),
        np.array(hand_joint, dtype=np.float32).flatten()
    ])

    resized_image = cv2.resize(color_image, (128,128))
    resized_image2 = cv2.resize(color_image2, (128,128))
    front_camera_image = resized_image[..., ::-1]
    side_camera_image = resized_image2[..., ::-1]

    obs = {
        "front_camera": front_camera_image,
        "side_camera": side_camera_image,
        "state": state_flattened
    }
    return obs, is_record_success

def read_data(env, robot_urdf_path):
    data = []
    actions = np.zeros(env.action_space.sample().shape) 
    data_dir = "/home/ruiqiang/workspace/HK_TACTEXO_DATA/wrq_project_data"
    for collect_data_dir in sorted(os.listdir(data_dir)):
        collect_data_path = os.path.join(data_dir, collect_data_dir)
        if not os.path.isdir(collect_data_path):
            continue

        frame_dirs = sorted(os.listdir(collect_data_path))

        for i in range(len(frame_dirs) - 1):
        # for i in list(range(start_frame, end_frame+1)):
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
                    rewards=0,
                    masks=1.0,
                    dones=0,
                )
            )
            data.append(transition)
            break
    return data
