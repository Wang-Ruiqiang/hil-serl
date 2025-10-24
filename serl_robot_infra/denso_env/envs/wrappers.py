import time
from gymnasium import Env, spaces
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
import copy
from denso_env.envs.keyboard_expert import KeyboardExpert
import requests
from scipy.spatial.transform import Rotation as R
from denso_env.envs.denso_env import DensoEnv
from typing import List

sigmoid = lambda x: 1 / (1 + np.exp(-x))

class HumanClassifierWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        if done:
            while True:
                try:
                    rew = int(input("Success? (1/0)"))
                    assert rew == 0 or rew == 1
                    break
                except:
                    continue
        info['succeed'] = rew
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
class MultiCameraBinaryRewardClassifierWrapper(gym.Wrapper):
    """
    This wrapper uses the camera images to compute the reward,
    which is not part of the observation space
    """

    def __init__(self, env: Env, reward_classifier_func, target_hz = None):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.target_hz = target_hz
        self.log_file_path = "classifier_test.txt"
        self.is_pick = True  # whether the task is pick or place, used for classifier

    def compute_reward(self, obs):
        if self.reward_classifier_func is not None:
            with open(self.log_file_path, "w") as f:
                log_msg = f"obs = {obs}\n"
                f.write(log_msg)
                f.flush()
            return self.reward_classifier_func(obs, self.is_pick)
        return 0

    def step(self, action):
        start_time = time.time()
        obs, rew, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(obs)
        if rew <= 0:
            done = 0
        else:
            done = 1

        info['succeed'] = bool(rew)
        info['is_pick'] = self.is_pick
        if self.target_hz is not None:
            time.sleep(max(0, 1/self.target_hz - (time.time() - start_time)))
        
        if done and self.is_pick:
            self.is_pick = False  # switch to place task after pick is done
        elif done and not self.is_pick:
            self.is_pick = True
        # if done:
        #     self.env.move_up()
        # print("reward = ", rew)
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['succeed'] = False
        return obs, info
    
    
class MultiStageBinaryRewardClassifierWrapper(gym.Wrapper):
    def __init__(self, env: Env, reward_classifier_func: List[callable]):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.received = [False] * len(reward_classifier_func)
    
    def compute_reward(self, obs):
        rewards = [0] * len(self.reward_classifier_func)
        for i, classifier_func in enumerate(self.reward_classifier_func):
            if self.received[i]:
                continue

            logit = classifier_func(obs).item()
            if sigmoid(logit) >= 0.75:
                self.received[i] = True
                rewards[i] = 1

        reward = sum(rewards)
        return reward

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(obs)
        done = (done or all(self.received)) # either environment done or all rewards satisfied
        info['succeed'] = all(self.received)
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.received = [False] * len(self.reward_classifier_func)
        info['succeed'] = False
        return obs, info


class Quat2EulerWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to euler angles
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["tcp_pose"]
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        return observation


class Quat2R2Wrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to rotation matrix
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(9,)
        )

    def observation(self, observation):
        tcp_pose = observation["state"]["tcp_pose"]
        r = R.from_quat(tcp_pose[3:]).as_matrix()
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], r[..., :2].flatten())
        )
        return observation


class DualQuat2EulerWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to euler angles
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["left/tcp_pose"].shape == (7,)
        assert env.observation_space["state"]["right/tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["left/tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )
        self.observation_space["state"]["right/tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["left/tcp_pose"]
        observation["state"]["left/tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        tcp_pose = observation["state"]["right/tcp_pose"]
        observation["state"]["right/tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        return observation
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info
    

class KeyboardIntervention(gym.ActionWrapper):
    def __init__(self, env, action_indices=None):
        super().__init__(env)

        self.expert = KeyboardExpert()
        self.action_indices = action_indices
        self.exp_name = env.config.EXP_NAME

        if self.exp_name == "tennis_ball_pick":
            self.gripper_open_joint = self.env.gripper_open_joint
            # self.gripper_open_joint = [
            #     2.989728450775146484, 3.231437253952026367, 3.438389015197753906, 3.96806390762329102,    #index
            #     2.904854822158813477, 3.202951908111572266, 3.466796636581420898, 3.969689750671386719,   #middle
            #     3.218291759490966797, 3.238233327865600586, 2.867010116577148438, 3.325670242309570312,
            #     4.512019824981689453, 3.3605515193939208984, 3.374757766723632812, 3.397184896469116211    #thumb
            # ]
            
            self.gripper_closen_joint = self.env.gripper_close_joint
            
            # self.gripper_close_joint = [
            #     3.546563625335693359, 4.127942085266113281, 3.413689804077148438, 3.641670465469360352,
            #     3.626330614089965820, 3.529689788818359375, 2.931437253952026367, 3.782796621322631836,
            #     3.838019847869873047, 3.532757759094238281, 3.535825729370117188, 3.413107156753540039,
            #     4.661767482757568359, 3.366175127029418945, 3.374757766723632812, 3.397184896469116211
            # ]
        elif self.exp_name == "twist_bottle_cap":
            self.gripper_open_joint = self.env.gripper_open_joint
            # self.gripper_open_joint = [
            #     2.989728450775146484, 3.231437253952026367, 3.438389015197753906, 3.96806390762329102,    #index
            #     2.904854822158813477, 3.202951908111572266, 3.466796636581420898, 3.969689750671386719,   #middle
            #     3.218291759490966797, 3.238233327865600586, 2.867010116577148438, 3.325670242309570312,
            #     4.512019824981689453, 3.3605515193939208984, 3.374757766723632812, 3.397184896469116211    #thumb
            # ]

            self.gripper_closen_joint = self.env.gripper_close_joint
            # self.gripper_close_joint = [
            #     3.546563625335693359, 4.127942085266113281, 3.413689804077148438, 3.641670465469360352,
            #     3.626330614089965820, 3.529689788818359375, 2.931437253952026367, 3.782796621322631836,
            #     3.838019847869873047, 3.532757759094238281, 3.535825729370117188, 3.413107156753540039,
            #     4.661767482757568359, 3.366175127029418945, 3.374757766723632812, 3.397184896469116211
            # ]

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        expert_a = self.expert.get_action()
        intervened = False
        
        # 人为输入动作非0,触发干预
        if np.linalg.norm(expert_a) > 0.001:
            intervened = True

        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        if intervened:
            new_action = np.zeros(7, dtype=np.float32)
            new_action[:3] = expert_a[:3]
            new_action[5] = expert_a[3]
            if expert_a[4] > 0.8 :
                # hand_joint = self.gripper_close_joint
                new_action[6] = 1.0
            if expert_a[5] > 0.8 :
                # hand_joint = self.gripper_open_joint
                new_action[6] = -1.0

            return new_action, True

        return action, False

    def step(self, action):
        new_action, replaced = self.action(action)
        
        obs, rew, done, truncated, info = self.env.step(new_action)

        if replaced:
            info["intervene_action"] = new_action

        return obs, rew, done, truncated, info
    

class GripperCloseEnv(gym.ActionWrapper):
    """
    Use this wrapper to task that requires the gripper to be closed
    """

    def __init__(self, env):
        super().__init__(env)
        ub = self.env.action_space
        assert ub.shape == (7,)
        self.action_space = Box(ub.low[:6], ub.high[:6])

    def action(self, action: np.ndarray) -> np.ndarray:
        new_action = np.zeros((7,), dtype=np.float32)
        new_action[:6] = action.copy()
        return new_action

    def step(self, action):
        new_action = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if "intervene_action" in info:
            info["intervene_action"] = info["intervene_action"][:6]
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    

class GripperPenaltyWrapper(gym.RewardWrapper):
    def __init__(self, env, penalty=0.1):
        super().__init__(env)
        assert env.action_space.shape == (7,)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["state"][0, 0]
        return obs, info

    def reward(self, reward: float, action) -> float:
        if (action[6] < -0.5 and self.last_gripper_pos > 0.95) or (
            action[6] > 0.5 and self.last_gripper_pos < 0.95
        ):
            return reward - self.penalty
        else:
            return reward

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]
        reward = self.reward(reward, action)
        self.last_gripper_pos = observation["state"][0, 0]
        return observation, reward, terminated, truncated, info

class DualGripperPenaltyWrapper(gym.RewardWrapper):
    def __init__(self, env, penalty=0.1):
        super().__init__(env)
        assert env.action_space.shape == (14,)
        self.penalty = penalty
        self.last_gripper_pos_left = 0 #TODO: this assume gripper starts opened
        self.last_gripper_pos_right = 0 #TODO: this assume gripper starts opened
    
    def reward(self, reward: float, action) -> float:
        if (action[6] < -0.5 and self.last_gripper_pos_left==0):
            reward -= self.penalty
            self.last_gripper_pos_left = 1
        elif (action[6] > 0.5 and self.last_gripper_pos_left==1):
            reward -= self.penalty
            self.last_gripper_pos_left = 0
        if (action[13] < -0.5 and self.last_gripper_pos_right==0):
            reward -= self.penalty
            self.last_gripper_pos_right = 1
        elif (action[13] > 0.5 and self.last_gripper_pos_right==1):
            reward -= self.penalty
            self.last_gripper_pos_right = 0
        return reward
    
    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]
        reward = self.reward(reward, action)
        return observation, reward, terminated, truncated, info