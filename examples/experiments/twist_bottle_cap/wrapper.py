import copy
import time
from scipy.spatial.transform import Rotation as R
import numpy as np
import requests

from denso_env.envs.denso_env import DensoEnv

from examples.utils import kinematics_utils

robot_urdf_path = "/home/ruiqiang/workspaces/HK_TACEXO_WANG/hm_denso_wrq_ws/src/hm_denso/hm_denso_description/urdf/denso_robot_with_ati_4.urdf"

class RAMEnv(DensoEnv):
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

        # init_pos = np.array([0.5580441126924457, -2.3018392461026157e-05, 0.25153528])
        # init_ori = np.array([-0.03244228, 0.99039508, 0.12396424, -0.05194187])
        init_pos = np.array([0.65, -0.05, 0.22153528])
        # init_pos = np.array([0.5580441126924457, -2.3018392461026157e-05, 0.45153528])
        init_ori = np.array([0, 1, 0, 0])
        # init_ori = np.array([0, 0.7071, -0.7071, 0])
        init_arm_action = np.concatenate([init_pos, init_ori])
        self.ros_interface.publish_arm_action(init_arm_action)
        self._segmented_init(self.curr_leap_hand_pos)

        time.sleep(5)

        self.curr_path_length = 0
        # self.ros_interface.reset_cur_pose()
        self._update_cur_position(init_arm_action)
        self.gripper_open_joint_np = self.curr_leap_hand_pos.copy()
        # print("self.cur_position = ", self.cur_position)
        # self.save_training_frame()
        obs = self._get_obs()
        # requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)
        self.terminate = False
        return obs, {}