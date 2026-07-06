import copy
import time
from scipy.spatial.transform import Rotation as R
import numpy as np
from typing import OrderedDict
import gymnasium as gym

from franka_env.envs.franka_env import FrankaEnv

from franka_env.camera.rs_capture import RSCapture
from franka_env.camera.video_capture import VideoCapture

class RAMEnv(FrankaEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
                
        # for cam_name, kwargs in extra_cameras_dict.items():
        #     cap = VideoCapture(
        #         RSCapture(name=cam_name, **kwargs)
        #     )
        #     self.cap[cam_name] = cap
            

    def move_up(self):
        print("move up to avoid collision")
        pos = self.cur_position.copy()
        pos[2] += 0.02
        ori = self.cur_orientation.copy()
        nextpos = np.concatenate((pos, ori), axis=0)
        self.ros_interface.publish_arm_action(nextpos)
        time.sleep(2.0)
        self.get_im()

    def reset(self, joint_reset=False, **kwargs):
        print("RAMEnv reset")

        hand_joint_msg = self.ros_interface.get_current_leap_position()
        self.curr_leap_hand_pos = np.asarray(hand_joint_msg, dtype=np.float32).copy()
        # print("self.curr_leap_hand_pos reset before= ", self.curr_leap_hand_pos)
        self._send_leap_hand_command(self.gripper_open_joint, steps=20, step_time=0.05)
        time.sleep(1)
        self.curr_leap_hand_pos = np.array(self.gripper_open_joint, dtype=np.float32)
        # print("self.curr_leap_hand_pos reset = ", self.curr_leap_hand_pos)
        cur_position, cur_orientation = self.ros_interface.get_current_robot_ee()
        self.curpos = np.concatenate((cur_position, cur_orientation), axis=0)
        # init_pos = np.array([x_init, y_init, z_init])
        init_pos = np.array([0.55977625898067087, -0.090797684551726014, 0.4022486952647027])
        init_ori = np.array([0, 1, 0, 0], dtype=np.float32)
        init_arm_action = np.concatenate([init_pos, init_ori])
        self.ros_interface.arm_interpolate_and_publish(self.curpos, init_arm_action, 0.02, 200)

        self._close_open_pose_init(self.curr_leap_hand_pos)

        time.sleep(5)

        self.cmd_pose = np.concatenate([init_pos.copy(), init_ori.copy()], axis=0)

        self.curr_path_length = 0
        # self.ros_interface.reset_cur_pose()
        self._update_cur_position(init_arm_action, wait_threshold=0.05)
        # print("self.cur_position = ", self.cur_position)
        # self.save_training_frame()
        obs = self._get_obs()
        # requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)
        self.terminate = False
        return obs, {}
    
class GripperPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, exp_name="tennis_ball_pick_and_place", penalty=-0.05):
        super().__init__(env)
        self.penalty = penalty
        self.last_hand_pos = None
        self.exp_name = exp_name

    def step(self, action):
        """Keep the grasp_penalty info key for training compatibility."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        info["grasp_penalty"] = 0.0
        return observation, reward, terminated, truncated, info
