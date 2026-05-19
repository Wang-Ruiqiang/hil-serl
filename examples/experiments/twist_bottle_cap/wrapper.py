import copy
import time
from scipy.spatial.transform import Rotation as R
import numpy as np
import requests

from franka_env.envs.franka_env import FrankaEnv
import gymnasium as gym

from examples.utils import kinematics_utils

class RAMEnv(FrankaEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = kwargs.get("config", {})
        self.exp_name = self.config.EXP_NAME


    def reset(self, joint_reset=False, **kwargs):
        print("RAMEnv reset")
        self.last_gripper_act = time.time()
        # if self.save_video:
        #     self.save_video_recording()
            
        hand_joint_msg = self.ros_interface.get_current_leap_position()
        self.curr_leap_hand_pos = np.asarray(hand_joint_msg, dtype=np.float32).copy()
        # print("self.curr_leap_hand_pos reset before= ", self.curr_leap_hand_pos)
        self._send_leap_hand_command(self.gripper_open_joint, steps=20, step_time=0.05)
        time.sleep(1)
        self.curr_leap_hand_pos = np.array(self.gripper_open_joint, dtype=np.float32)
        # print("self.curr_leap_hand_pos reset = ", self.curr_leap_hand_pos)

        # init_pos = np.array([0.5580441126924457, 0, 0.25153528])
        # init_ori = np.array([0, 1, 0, 0])
        # init_arm_action = np.concatenate([init_pos, init_ori])
        # self.ros_interface.publish_arm_action(init_arm_action)



        cur_position, cur_orientation = self.ros_interface.get_current_robot_ee()
        self.curpos = np.concatenate((cur_position, cur_orientation), axis=0)
        # init_pos = np.array([x_init, y_init, z_init])
        init_pos = np.array([0.51977625898067087, -0.030797684551726014, 0.4622486952647027])
        init_ori = np.array([0, 1, 0, 0], dtype=np.float32)
        init_arm_action = np.concatenate([init_pos, init_ori])
        self.ros_interface.arm_interpolate_and_publish(self.curpos, init_arm_action, 0.02, 200)
        max_steps = 15
        self._segmented_init(self.curr_leap_hand_pos, max_steps)

        time.sleep(5)
        self.cmd_pose = np.concatenate([init_pos.copy(), init_ori.copy()], axis=0)

        self.curr_path_length = 0

        # reset 时强制把相位历史清零（避免初始误判到 0.9+）
        self._prev_hand_progress = 0.0
        # 如果你还保留 turns（可选）
        self._turns = 0

        self._update_cur_position(init_arm_action)
        self.gripper_open_joint_np = self.curr_leap_hand_pos.copy()
        # print("self.cur_position = ", self.cur_position)
        # self.save_training_frame()
        obs = self._get_obs()
        # requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)
        self.terminate = False
        return obs, {}
    

class GripperPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=-0.05):
        super().__init__(env)
        self.penalty = penalty
        self.last_hand_pos = None

    def step(self, action, is_pick=True):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        robot_height = observation["state"][0, 2]
        hand_state = observation["state"][0, 7]
        # print("hand_state = ", hand_state)
        # print("robot_height = ", robot_height)
        # print("action = ", action)
        # print("action[-1] = ", action[-1])
        # print("is_pick = ", is_pick)
        # if is_pick and robot_height > 0.195 and hand_state > 0.15:
        #     info["grasp_penalty"] = self.penalty
        if (robot_height < 0.18 and action[-1] > 0.3):
            info["grasp_penalty"] = self.penalty
        else:
            info["grasp_penalty"] = 0.0
        # hand_state = observation["state"][0, 7]
        # if "intervene_action" in info:
        #     action = info["intervene_action"]
        # if hand_state > 4:
        #     info["grasp_penalty"] = self.penalty
        # else:
            # info["grasp_penalty"] = 0.0
        return observation, reward, terminated, truncated, info
    
class RobotArmPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=-0.05):
        super().__init__(env)
        self.penalty = penalty
        self.last_hand_pos = None

    def step(self, action, is_pick=True):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action, is_pick)
        robot_height = observation["state"][0, 2]
        if is_pick and robot_height > 0.20 and action[2] > 0:
            info["robot_arm_penalty"] = self.penalty
        elif robot_height < 0.18:
            info["robot_arm_penalty"] = self.penalty
        else:
            info["robot_arm_penalty"] = 0.0
        return observation, reward, terminated, truncated, info