import time
import numpy as np

from denso_env.envs.denso_env import DensoEnv
import gymnasium as gym

robot_urdf_path = "/home/wrq/workspaces/HK_TACEXO_WANG/hm_denso_wrq_ws/src/hm_denso/hm_denso_description/urdf/denso_robot_with_ati_4.urdf"

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

        # init_pos = np.array([0.6580441126924457, -0.10, 0.20153528])
        # init_ori = np.array([-0.03244228, 0.99039508, 0.12396424, -0.05194187])
        # init_pos = np.array([0.65, -0.05, 0.22153528])
        init_pos = np.array([0.5580441126924457, 0, 0.25153528])
        init_ori = np.array([0, 1, 0, 0])
        # init_ori = np.array([0, 0.7071, -0.7071, 0])
        init_arm_action = np.concatenate([init_pos, init_ori])
        self.ros_interface.publish_arm_action(init_arm_action)
        max_steps = 15
        self._segmented_init(self.curr_leap_hand_pos, max_steps)

        time.sleep(5)

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
