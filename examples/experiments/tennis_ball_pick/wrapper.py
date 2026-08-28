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
        distance = 0.06
        steps = 100
        step_time = 0.02
        print(f"move up to verify grasp, distance={distance}")
        cur_position, cur_orientation = self.ros_interface.get_current_robot_ee()
        start_pose = np.concatenate((cur_position, cur_orientation), axis=0)
        pos = cur_position.copy()
        pos[2] += distance
        nextpos = np.concatenate((pos, cur_orientation), axis=0)
        self.ros_interface.arm_interpolate_and_publish(
            start_pose,
            nextpos,
            step_time=step_time,
            steps=steps,
        )
        self.cmd_pose = nextpos.copy()
        self._update_cur_position(nextpos, wait_threshold=0.05)
        self.get_im()

    def reset(self, joint_reset=False, **kwargs):
        print("RAMEnv reset")
        self.data_count = 0
        self.last_gripper_act = time.time()
        if self.save_video:
            self.save_video_recording(self.video_count)

        hand_joint_msg = self.ros_interface.get_current_leap_position()
        self.curr_leap_hand_pos = np.asarray(hand_joint_msg, dtype=np.float32).copy()
        self._send_leap_hand_command(self.gripper_open_joint, steps=20, step_time=0.05)
        time.sleep(1)
        self.curr_leap_hand_pos = np.array(self.gripper_open_joint, dtype=np.float32)

        # Temporary target-pose-only control: reset and RL actions both use
        # the Cartesian /target_pose path.
        cur_position, cur_orientation = self.ros_interface.get_current_robot_ee()
        self.curpos = np.concatenate((cur_position, cur_orientation), axis=0)
        init_pos = np.array(
            [0.55977625898067087, -0.140797684551726014, 0.4022486952647027],
            dtype=np.float32,
        )
        init_ori = np.array([0, 1, 0, 0], dtype=np.float32)
        init_arm_action = np.concatenate([init_pos, init_ori])
        self.ros_interface.arm_interpolate_and_publish(
            self.curpos,
            init_arm_action,
            step_time=0.02,
            steps=200,
        )
        self._close_open_pose_init(self.curr_leap_hand_pos)

        self.cmd_pose = init_arm_action.copy()
        self.nextpos = self.cmd_pose.copy()

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
        """Keep the grasp_penalty info key for training compatibility."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        info["grasp_penalty"] = 0.0
        return observation, reward, terminated, truncated, info
