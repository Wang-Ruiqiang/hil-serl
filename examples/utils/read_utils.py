import copy
import json
import cv2
import os
import numpy as np
import gymnasium as gym
from utils import kinematics_utils
import re
from collections import deque
from scipy.spatial.transform import Rotation as R

palm_lower2denso_end_tf = np.array([
    [1.00000000e+00, -3.26589794e-07, 0.00000000e+00, -6.00952496e-02],
    [-3.26589379e-07, -9.99998732e-01, 1.59265292e-03, -3.39726879e-02],
    [-5.20144187e-10, -1.59265292e-03, -9.99998732e-01, -1.69276725e-01],
    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

# IMAGE_CROP = {
#     "front_camera": lambda img: img[60:340, 140:420],
#     "wrist_camera": lambda img: img[0:480, 120:600],
# }

TENNIS_BALL_PICK_IMAGE_CROP = {
    "front_camera": lambda img: img[0:460, 60:520],
}

TUBE_INSERTION_IMAGE_CROP = {
    "front_camera": lambda img: img[60:340, 140:420],
    "wrist_camera": lambda img: img[0:480, 120:600],
}

BOTTLE_TWIST_IMAGE_CROP = {
    "front_camera": lambda img: img[35:375, 160:500],
    "wrist_camera": lambda img: img[0:480, 120:600],
}

TUBE_INSERTION_CLASSIFIER_IMAGE_CROP = {
    "front_classifier": lambda img: img[240:360, 230:350],
    "wrist_classifier": lambda img: img[50:280, 270:500],
}

resize_dim = (128, 128)
tactile_resize_dim = (128, 128)



class ObsHistoryBuffer:
    # def __init__(self, obs_horizon=3, image_keys=("front_camera", "side_camera"), proprio_key="state"):
    def __init__(self, obs_horizon=3, image_keys=("front_camera",), proprio_key="state"):
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
            vecs = [o[self.proprio_key] for o in self.buffer]
            stacked_obs[self.proprio_key] = np.stack(vecs, axis=0)

        return stacked_obs
    

def get_frame_data(frame_path, robot_urdf_path, enable_tactile=False, exp_name="tennis_ball_pick"):
    color_image_path = os.path.join(frame_path, "color_image.jpg")
    color_image_path_wrist = os.path.join(frame_path, "color_image2.jpg")
    index_heat_map_path = os.path.join(frame_path, "index_heat_map.jpg")
    thumb_heat_map_path = os.path.join(frame_path, "thumb_heat_map.jpg")
    # index_heat_map_path = os.path.join(frame_path, "thumb_heat_map.jpg")
    # thumb_heat_map_path = os.path.join(frame_path, "index_heat_map.jpg")
    middle_heat_map_path = os.path.join(frame_path, "middle_heat_map.jpg")
    # color_image_path2 = os.path.join(frame_path, "color_image2.jpg")
    # depth_image_path = os.path.join(frame_path, "depth_image.png")
    # depth_image_path2 = os.path.join(frame_path, "depth_image2.png")
    color_image = cv2.imread(color_image_path) if os.path.exists(color_image_path) else None
    color_image_wrist = cv2.imread(color_image_path_wrist) if os.path.exists(color_image_path_wrist) else None
    # color_image2 = cv2.imread(color_image_path2) if os.path.exists(color_image_path) else None
    # depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED) if os.path.exists(depth_image_path) else None
    # depth_image2 = cv2.imread(depth_image_path2, cv2.IMREAD_UNCHANGED) if os.path.exists(depth_image_path) else None
    if enable_tactile:
        index_heat_map_image = cv2.imread(index_heat_map_path) if os.path.exists(index_heat_map_path) else None
        thumb_heat_map_image = cv2.imread(thumb_heat_map_path) if os.path.exists(thumb_heat_map_path) else None
        middle_heat_map_image = cv2.imread(middle_heat_map_path) if os.path.exists(middle_heat_map_path) else None

        index_heat_map_image = cv2.resize(index_heat_map_image, tactile_resize_dim, interpolation=cv2.INTER_LINEAR)
        thumb_heat_map_image = cv2.resize(thumb_heat_map_image, tactile_resize_dim, interpolation=cv2.INTER_LINEAR)
        middle_heat_map_image = cv2.resize(middle_heat_map_image, tactile_resize_dim, interpolation=cv2.INTER_LINEAR)
        heatmap_canvas = cv2.hconcat([thumb_heat_map_image, index_heat_map_image])

    joint_file_path = os.path.join(frame_path, "right_arm_joint.txt")
    action_file_path = os.path.join(frame_path, "action.txt")
        
    record_success_failed_file = os.path.join(frame_path, "is_record_success.txt")
    hand_joint = None
    is_record_success = np.loadtxt(record_success_failed_file, dtype=int)
    
    hand_state = np.loadtxt(os.path.join(frame_path, "hand_state.txt"), dtype=float) if os.path.exists(os.path.join(frame_path, "hand_state.txt")) else 0.0

    action = np.zeros(7, dtype=np.float32)
    if os.path.exists(action_file_path):
        with open(action_file_path, "r") as f:
            action = np.array([float(x.strip()) for x in f.readlines()])
    
    if os.path.exists(joint_file_path):
        with open(joint_file_path, "r") as f:
            all_joint_values = np.array([float(x.strip()) for x in f.readlines()])
            hand_joint = all_joint_values[6:]
    
    tcp_pos, tcp_ori = kinematics_utils.comupute_forward_kinematics(all_joint_values, robot_urdf_path)
    tcp_pos, tcp_ori = kinematics_utils.apply_transformation(tcp_pos, tcp_ori, palm_lower2denso_end_tf)

    # if exp_name == "twist_bottle_cap":
    #     state_flattened = np.concatenate([
    #         np.array(tcp_pos, dtype=np.float32).flatten(),
    #         np.array(tcp_ori, dtype=np.float32).flatten(),
    #     ])
    # else:
    # state_flattened = np.concatenate([
    #     np.array(tcp_pos, dtype=np.float32).flatten(),
    #     np.array(tcp_ori, dtype=np.float32).flatten(),
    #     np.array(hand_state, dtype=np.float32).flatten(),
    # ])
    state_flattened = np.concatenate([
        np.array(tcp_pos, dtype=np.float32).flatten(),
        np.array(tcp_ori, dtype=np.float32).flatten(),
        np.array(hand_state, dtype=np.float32).flatten(),
    ])
    if exp_name == "tennis_ball_pick" or exp_name == "tennis_ball_place":
        IMAGE_CROP = TENNIS_BALL_PICK_IMAGE_CROP
        CLASSIFIER_IMAGE_CROP = {}
    elif exp_name == "tube_insertion":
        IMAGE_CROP = TUBE_INSERTION_IMAGE_CROP
        CLASSIFIER_IMAGE_CROP = TUBE_INSERTION_CLASSIFIER_IMAGE_CROP
    elif exp_name == "twist_bottle_cap" or exp_name == "lid_grip":
        IMAGE_CROP = BOTTLE_TWIST_IMAGE_CROP
        CLASSIFIER_IMAGE_CROP = BOTTLE_TWIST_IMAGE_CROP
        
    cropped_front = IMAGE_CROP["front_camera"](color_image) if "front_camera" in IMAGE_CROP else color_image
    cropped_wrist = IMAGE_CROP["wrist_camera"](color_image_wrist) if "wrist_camera" in IMAGE_CROP else color_image_wrist

    cropped_front_classifier = CLASSIFIER_IMAGE_CROP["front_classifier"](color_image) if "front_classifier" in CLASSIFIER_IMAGE_CROP else color_image
    # cropped_wrist_classifier = CLASSIFIER_IMAGE_CROP["wrist_classifier"](color_image_wrist) if "wrist_classifier" in CLASSIFIER_IMAGE_CROP else color_image_wrist

    resized_image = cv2.resize(cropped_front, resize_dim)
    resized_image_wrist = cv2.resize(cropped_wrist, resize_dim)
    
    resized_image_front_classifier = cv2.resize(cropped_front_classifier, resize_dim)
    # resized_image_wrist_classifier = cv2.resize(cropped_wrist_classifier, resize_dim)
    
    front_camera_image = resized_image[..., ::-1]
    wrist_camera_image = resized_image_wrist[..., ::-1]
    front_classifier_image = resized_image_front_classifier[..., ::-1]
    # wrist_classifier_image = resized_image_wrist_classifier[..., ::-1]
    
    if not enable_tactile:
        if exp_name == "tennis_ball_pick" or exp_name == "tennis_ball_place":
            obs = {
                "front_camera": front_camera_image,
                "state": state_flattened
            }
        elif exp_name == "tube_insertion":
            obs = {
                "front_camera": front_camera_image,
                "wrist_camera": wrist_camera_image,
                # "front_classifier": front_classifier_image,
                "state": state_flattened
            }
        elif exp_name == "twist_bottle_cap" or exp_name == "lid_grip":
            obs = {
                "front_camera": front_camera_image,
                "wrist_camera": wrist_camera_image,
                "state": state_flattened
            }
    else:
        if exp_name == "tennis_ball_pick" or exp_name == "tennis_ball_place":
            obs = {
                "front_camera": front_camera_image,
                "tactile_data": heatmap_canvas,
                "state": state_flattened
            }
        elif exp_name == "tube_insertion":
            obs = {
                "front_camera": front_camera_image,
                "wrist_camera": wrist_camera_image,
                # "front_classifier": front_classifier_image,
                "tactile_data": heatmap_canvas,
                "state": state_flattened
            }
        elif exp_name == "twist_bottle_cap" or exp_name == "lid_grip":
            obs = {
                "front_camera": front_camera_image,
                "wrist_camera": wrist_camera_image,
                "tactile_data": heatmap_canvas,
                "state": state_flattened
            }
    # debug_imshow(obs)
    # print("state_flattened = ", state_flattened)
    # cv2.imwrite("front_camera_image.jpg", front_camera_image)
    # input("enter")
    return obs, int(is_record_success), action


def read_data(robot_urdf_path, enable_tactile=False):
    data = []
    clip_ranges = []
    global_idx = 0
    # action_space = gym.spaces.Box(
    #     np.ones((23,), dtype=np.float32) * -1,
    #     np.ones((23,), dtype=np.float32),
    # )
    low  = np.concatenate([np.ones(6, dtype=np.float32) * -1, [0]])
    high = np.ones(7, dtype=np.float32)
    action_space = gym.spaces.Box(low, high, dtype=np.float32)
    
    actions = np.zeros(action_space.sample().shape)
    data_dir = "/home/wrq/workspaces/HK_TACEXO_WANG/recorded_data/test_data/"
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
            clip_start_idx = global_idx
            
            for i in list(range(start_frame, end_frame+1)):
            # for i in range(len(frame_dirs) - 1):
                current_frame_path = os.path.join(collect_data_path, frame_dirs[i])
                if i == end_frame:
                    next_frame_path = current_frame_path
                else:
                    next_frame_path = os.path.join(collect_data_path, frame_dirs[i + 1])
                
                if not os.path.isdir(current_frame_path) or not os.path.isdir(next_frame_path):
                    continue

                obs, is_record_success= get_frame_data(current_frame_path, robot_urdf_path, enable_tactile)
                if i == end_frame:
                    next_obs = obs
                else:
                    next_obs, _ = get_frame_data(next_frame_path, robot_urdf_path, enable_tactile)
                # print("next_obs['state'][3:7] = ", next_obs["state"][3:7])

                delta_pos = next_obs["state"][:3] - obs["state"][:3]
                actions[:3] = delta_pos

                current_quat = obs["state"][3:7]  # wxyz
                next_quat = next_obs["state"][3:7]

                current_euler = R.from_quat([current_quat[1], current_quat[2], current_quat[3], current_quat[0]]).as_euler("xyz")
                next_euler = R.from_quat([next_quat[1], next_quat[2], next_quat[3], next_quat[0]]).as_euler("xyz")

                delta_euler = next_euler - current_euler
                actions[3:6] = delta_euler

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
                global_idx += 1
            clip_end_idx = global_idx - 1 
            clip_ranges.append((clip_start_idx, clip_end_idx))

    return data, clip_ranges