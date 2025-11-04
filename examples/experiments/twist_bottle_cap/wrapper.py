import copy
import time
from scipy.spatial.transform import Rotation as R
import numpy as np
import requests

from denso_env.envs.denso_env import DensoEnv

from examples.utils import kinematics_utils

robot_urdf_path = "/home/ruiqiang/workspaces/HK_TACEXO_WANG/hm_denso_wrq_ws/src/hm_denso/hm_denso_description/urdf/denso_robot_with_ati_4.urdf"
palm_lower2denso_end_tf = np.array([
    [1.00000000e+00, -3.26589794e-07, 0.00000000e+00, -6.00952496e-02],
    [-3.26589379e-07, -9.99998732e-01, 1.59265292e-03, -3.39726879e-02],
    [-5.20144187e-10, -1.59265292e-03, -9.99998732e-01, -1.69276725e-01],
    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

# gripper_open_joint = [
#     2.989728450775146484, 3.231437253952026367, 3.438389015197753906, 3.96806390762329102,    #index
#     2.904854822158813477, 3.202951908111572266, 3.466796636581420898, 3.969689750671386719,   #middle
#     3.218291759490966797, 3.238233327865600586, 2.867010116577148438, 3.325670242309570312,
#     4.512019824981689453, 3.3605515193939208984, 3.374757766723632812, 3.397184896469116211    #thumb
# ]

gripper_open_joint = np.array([
    3.160000324249267578, 4.144815921783447266, 2.949845075607299805, 3.945398569107055664,
    3.067961692810058594, 4.141747951507568359, 3.003534317016601562, 3.874835491180419922,
    3.218291759490966797, 3.238233327865600586, 2.867010116577148438, 3.325670242309570312,
    4.781418323516845703, 3.385495662689208984, 2.001844882965087891, 4.227651119232177734
])

# T_palm_lower_to_end_link = np.array([
#     [ 9.99994927e-01, -3.18530179e-03,  5.11967137e-22, -3.40506487e-02],
#     [-3.18529775e-03, -9.99993659e-01,  1.59265292e-03,  6.04734432e-02],
#     [-5.07308019e-06, -1.59264484e-03, -9.99998732e-01, -1.69126301e-01],
#     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
# ])

class RAMEnv(DensoEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.should_regrasp = False


    def reset(self, joint_reset=False, **kwargs):
        print("RAMEnv reset")
        self.last_gripper_act = time.time()
        if self.save_video:
            self.save_video_recording()

        # # if True:
        # if self.should_regrasp:
        #     self.regrasp()
        #     self.should_regrasp = False

        # self._recover()
        # self.go_to_reset(joint_reset=False)
        # self._recover()
        # obs, info =  self.env.reset(**kwargs)
        hand_joint_msg = self.ros_interface.get_current_leap_position()
        self.curr_leap_hand_pos = np.asarray(hand_joint_msg, dtype=np.float32).copy()
        # print("self.curr_leap_hand_pos reset before= ", self.curr_leap_hand_pos)
        self._send_leap_hand_command(gripper_open_joint, steps=20, step_time=0.05)
        time.sleep(1)
        self.curr_leap_hand_pos = np.array(gripper_open_joint, dtype=np.float32)
        # print("self.curr_leap_hand_pos reset = ", self.curr_leap_hand_pos)

        # init_pos = np.array([0.55513753, 0.04267503, 0.18153528])
        # init_ori = np.array([-0.03244228, 0.99039508, 0.12396424, -0.05194187])
        init_pos = np.array([0.5580441126924457, -2.3018392461026157e-05, 0.25153528])
        init_ori = np.array([0, 1, 0, 0])
        # init_ori = np.array([0, 0.7071, -0.7071, 0])
        init_arm_action = np.concatenate([init_pos, init_ori])
        self.ros_interface.publish_arm_action(init_arm_action)

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