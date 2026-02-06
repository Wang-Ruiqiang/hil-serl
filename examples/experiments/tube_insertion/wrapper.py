import copy
import time
from scipy.spatial.transform import Rotation as R
import numpy as np
import requests
from typing import OrderedDict
import gymnasium as gym

from denso_env.envs.denso_env import DensoEnv
from denso_env.camera.rs_capture import RSCapture
from denso_env.camera.video_capture import VideoCapture

robot_urdf_path = "/home/wrq/workspaces/HK_TACEXO_WANG/hm_denso_wrq_ws/src/hm_denso/hm_denso_description/urdf/denso_robot_with_ati_4.urdf"

class RAMEnv(DensoEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = kwargs.get("config", {})
        self.exp_name = self.config.EXP_NAME

    def init_cameras(self, name_serial_dict=None):
        """Init both wrist cameras."""
        if self.cap is not None:  # close cameras if they are already open
            self.close_cameras()

        self.cap = OrderedDict()
        for cam_name, kwargs in name_serial_dict.items():
            if cam_name == "front_classifier":
                self.cap["front_classifier"] = self.cap["front_camera"]
            else:
                cap = VideoCapture(
                    RSCapture(name=cam_name, **kwargs)
                )
                self.cap[cam_name] = cap
                
        # for cam_name, kwargs in extra_cameras_dict.items():
        #     cap = VideoCapture(
        #         RSCapture(name=cam_name, **kwargs)
        #     )
        #     self.cap[cam_name] = cap
                

    def open_hand(self, steps=20, step_time=0.05):
        self._send_leap_hand_command(self.gripper_open_joint, steps=steps, step_time=step_time)
        time.sleep(steps * step_time + 0.5)
        # self.get_im()

    def reset(self, joint_reset=False, **kwargs):
        print("RAMEnv reset")
        self.last_gripper_act = time.time()
        # if self.save_video:
        #     self.save_video_recording()
            
        hand_joint_msg = self.ros_interface.get_current_leap_position()
        self.curr_leap_hand_pos = np.asarray(hand_joint_msg, dtype=np.float32).copy()
        self._send_leap_hand_command(self.gripper_open_joint, steps=20, step_time=0.05)
        time.sleep(1)
        self.curr_leap_hand_pos = np.array(self.gripper_open_joint, dtype=np.float32)
        
        # z_init = np.random.uniform(0.08, 0.14)
        # init_pos = np.array([0.7, -0.1458, z_init])
        # init_pos = np.array([0.7, -0.1458, 0.0809])
        init_pos = np.array([0.7, -0.1458, 0.1209])
        init_ori = np.array([0, 1, 0, 0])
        init_arm_action = np.concatenate([init_pos, init_ori])
        self.ros_interface.publish_arm_action(init_arm_action)
        self._segmented_init(self.curr_leap_hand_pos)

        time.sleep(5)
        
        # cur_position, cur_orientation = self.ros_interface.get_current_robot_ee()
        # curr_pose = np.concatenate((cur_position, cur_orientation), axis=0)
        self.curr_path_length = 0
        self._update_cur_position(init_arm_action, wait_threshold=0.05)
        obs = self._get_obs()
        self.terminate = False
        return obs, {}
    
class GripperPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, exp_name="tennis_ball_pick", penalty=-0.05):
        super().__init__(env)
        self.penalty = penalty
        self.last_hand_pos = None
        self.exp_name = exp_name

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        robot_height = observation["state"]["tcp_pos"][2]
        hand_state = observation["state"]["gripper_pose"]
        print("robot_height: ", robot_height)
        # print("robot_height: ", robot_height, " hand_state: ", hand_state)
        if robot_height > 0.07 and action[-1] > 0.3 and hand_state < 0.7:
            info["grasp_penalty"] = self.penalty
        elif hand_state > 0.85 and action[-1] < -0.3:
            info["grasp_penalty"] = self.penalty
        else:
            info["grasp_penalty"] = 0.0
        return observation, reward, terminated, truncated, info
        

class RobotArmPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=-0.05):
        super().__init__(env)
        self.penalty = penalty
        self.last_hand_pos = None

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        robot_height = observation["state"]["tcp_pos"][2]
        info["robot_arm_penalty"] = 0.0
        return observation, reward, terminated, truncated, info