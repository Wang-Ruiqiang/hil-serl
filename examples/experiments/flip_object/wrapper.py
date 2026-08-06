import numpy as np
import time

from denso_env.envs.denso_env import DensoEnv


class RAMEnv(DensoEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = kwargs.get("config", {})
        self.exp_name = self.config.EXP_NAME

    def open_hand(self, steps=20, step_time=0.05):
        self._send_leap_hand_command(self.gripper_open_joint, steps=steps, step_time=step_time)
        time.sleep(steps * step_time + 0.5)

    def reset(self, joint_reset=False, **kwargs):
        print("RAMEnv reset")
        self.last_gripper_act = time.time()

        hand_joint_msg = self.ros_interface.get_current_leap_position()
        self.curr_leap_hand_pos = np.asarray(hand_joint_msg, dtype=np.float32).copy()
        self._send_leap_hand_command(self.gripper_open_joint, steps=15, step_time=0.05)
        time.sleep(1)
        self.curr_leap_hand_pos = np.asarray(self.gripper_open_joint, dtype=np.float32).copy()

        init_pos = np.array([0.7, -0.1458, 0.0709])
        init_ori = np.array([0, 1, 0, 0])
        init_arm_action = np.concatenate([init_pos, init_ori])
        self.ros_interface.publish_arm_action(init_arm_action)
        self._segmented_init(self.curr_leap_hand_pos, max_steps=10)

        time.sleep(5)

        self.curr_path_length = 0
        self._update_cur_position(init_arm_action, wait_threshold=0.05)
        self.terminate = False
        return self._get_obs(), {}
