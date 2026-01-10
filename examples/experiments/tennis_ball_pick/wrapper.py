import copy
import time
from scipy.spatial.transform import Rotation as R
import numpy as np
from typing import OrderedDict

from denso_env.envs.denso_env import DensoEnv

from denso_env.camera.rs_capture import RSCapture
from denso_env.camera.video_capture import VideoCapture

robot_urdf_path = "/home/wrq/workspaces/HK_TACEXO_WANG/hm_denso_wrq_ws/src/hm_denso/hm_denso_description/urdf/denso_robot_with_ati_4.urdf"

class RAMEnv(DensoEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.should_regrasp = False

    def init_cameras(self, name_serial_dict=None, extra_cameras_dict=None):
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
                
        for cam_name, kwargs in extra_cameras_dict.items():
            cap = VideoCapture(
                RSCapture(name=cam_name, **kwargs)
            )
            self.cap[cam_name] = cap
            

    def move_up(self):
        print("move up to avoid collision")
        pos = self.cur_position.copy()
        pos[2] += 0.02
        ori = self.cur_oritation.copy()
        nextpos = np.concatenate((pos, ori), axis=0)
        self.ros_interface.publish_arm_action(nextpos)
        time.sleep(2.0)

    def reset(self, joint_reset=False, **kwargs):
        print("RAMEnv reset")
        self.last_gripper_act = time.time()
        # if self.save_video:
        #     self.save_video_recording()

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
        self._send_leap_hand_command(self.gripper_open_joint, steps=20, step_time=0.05)
        time.sleep(1)
        self.curr_leap_hand_pos = np.array(self.gripper_open_joint, dtype=np.float32)
        # print("self.curr_leap_hand_pos reset = ", self.curr_leap_hand_pos)

        # x_init = np.random.uniform(0.55, 0.65)
        # y_init = np.random.uniform(-0.08, -0.18)
        # z_init = np.random.uniform(0.16, 0.20)
        # init_pos = np.array([x_init, y_init, z_init])
        init_pos = np.array([0.65513753, -0.2067503, 0.16153528])
        # init_pos = np.array([0.60513753, -0.1567503, 0.18153528])
        # init_pos = np.array([0.70513753, -0.3067503, 0.15153528])
        init_ori = np.array([0, 1, 0, 0])
        init_arm_action = np.concatenate([init_pos, init_ori])
        self.ros_interface.publish_arm_action(init_arm_action)
        self._close_open_pose_init(self.curr_leap_hand_pos)

        time.sleep(5)

        self.curr_path_length = 0
        # self.ros_interface.reset_cur_pose()
        self._update_cur_position(init_arm_action, wait_threshold=0.02)
        # print("self.cur_position = ", self.cur_position)
        # self.save_training_frame()
        obs = self._get_obs()
        # requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)
        self.terminate = False
        return obs, {}