import time
from gymnasium import Env, spaces
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
import copy
from franka_env.envs.keyboard_expert import KeyboardExpert
from franka_env.envs.spacemouse_expert import SpaceMouseExpert
import requests
from scipy.spatial.transform import Rotation as R
from franka_env.envs.franka_env import FrankaEnv
from typing import List
import jax.numpy as jnp

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
    Camera-based reward wrapper for tennis-ball tasks.

    Supported classifier outputs:
    - tennis_ball_pick: scalar 0/1 reward.
    - tennis_ball_pick_and_place: scalar 0/1 place reward.
    """

    SUPPORTED_EXPS = {"tennis_ball_pick", "tennis_ball_pick_and_place"}

    def __init__(self, env: Env, reward_classifier_func, target_hz=None):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.target_hz = target_hz
        self.config = env.config
        self.exp_name = self.config.EXP_NAME
        self.is_pick = True
        if self.exp_name not in self.SUPPORTED_EXPS:
            raise ValueError(
                "MultiCameraBinaryRewardClassifierWrapper currently supports only "
                f"{sorted(self.SUPPORTED_EXPS)}, got {self.exp_name}."
            )

    def compute_reward(self, obs):
        if self.reward_classifier_func is not None:
            return self.reward_classifier_func(obs, self.is_pick)
        return 0

    def _parse_classifier_output(self, classifier_output, info):
        if not isinstance(classifier_output, dict):
            return classifier_output, False, False

        reward = classifier_output.get("reward", 0)
        pick_success = bool(classifier_output.get("pick_success", False))
        place_success = bool(classifier_output.get("place_success", False))
        for key, value in classifier_output.items():
            if key != "reward":
                info[key] = value
        return reward, pick_success, place_success

    def _step_pick(self, env_done, reward):
        task_success = bool(reward >= 1)
        done = bool(env_done or task_success)
        reward = 1 if task_success else 0
        return reward, done, task_success

    def _step_pick_and_place(self, env_done, reward, pick_success, place_success):
        if self.is_pick:
            if pick_success:
                self.is_pick = False
            task_success = False
            done = bool(env_done)
            reward = 0
            return reward, done, task_success

        task_success = bool(reward >= 1 or place_success)
        done = bool(env_done or task_success)
        reward = 1 if task_success else 0
        return reward, done, task_success

    def step(self, action):
        start_time = time.time()
        obs, env_rew, env_done, truncated, info = self.env.step(action)
        classifier_output = self.compute_reward(obs)
        classifier_reward, pick_success, place_success = self._parse_classifier_output(
            classifier_output,
            info,
        )

        if self.exp_name == "tennis_ball_pick":
            rew, done, task_success = self._step_pick(env_done, classifier_reward)
        else:
            rew, done, task_success = self._step_pick_and_place(
                env_done,
                classifier_reward,
                pick_success,
                place_success,
            )

        info["succeed"] = task_success
        info["is_pick"] = self.is_pick
        info["classifier_reward"] = classifier_reward
        info["rl_reward"] = rew
        info.setdefault("env_reward", env_rew)
        if self.target_hz is not None:
            time.sleep(max(0, 1 / self.target_hz - (time.time() - start_time)))
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        self.is_pick = True
        obs, info = self.env.reset(**kwargs)
        info["succeed"] = False
        info["is_pick"] = self.is_pick
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

        self.exp_name = env.config.EXP_NAME
        self.expert = KeyboardExpert(self.exp_name)
        self.action_indices = action_indices

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
            # new_action[:3] = expert_a[:3]
            # new_action[3:6] = expert_a[3:6]
            new_action[:] = expert_a[:]

            return new_action, True

        return action, False

    def step(self, action):
        new_action, replaced = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action

        return obs, rew, done, truncated, info

    def close(self):
        self.expert.close()
        if hasattr(self.env, "close"):
            self.env.close()
    

class SpacemouseIntervention(gym.ActionWrapper):
    def __init__(self, env, action_indices=None):
        super().__init__(env)
        self.expert = SpaceMouseExpert()
        self.left = False
        self.right = False
        self.action_indices = action_indices
        self.translation_deadband = 0.03

    def action(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        Input:
        - action: policy or inner-wrapper action
        Output:
        - action: SpaceMouse action if nonzero; else input action
        """
        expert_a, buttons = self.expert.get_action()
        self.left = bool(buttons[0]) if len(buttons) > 0 else False
        self.right = bool(buttons[1]) if len(buttons) > 1 else False
        translation_active = bool(np.linalg.norm(expert_a[:3]) > self.translation_deadband)
        intervened = bool(translation_active or self.left or self.right)

        if not intervened:
            return action, False

        new_action = np.zeros_like(action, dtype=np.float32)
        arm_dims = min(6, new_action.shape[0], expert_a.shape[0])
        new_action[:arm_dims] = expert_a[:arm_dims]

        if self.left:
            new_action[6] = -1.0
        elif self.right:
            new_action[6] = 1.0

        if self.action_indices is not None:
            filtered_action = np.zeros_like(new_action)
            filtered_action[self.action_indices] = new_action[self.action_indices]
            new_action = filtered_action

        return new_action, True

    def step(self, action):
        new_action, replaced = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["spacemouse_left"] = self.left
        info["spacemouse_right"] = self.right
        return obs, rew, done, truncated, info

    def close(self):
        self.expert.close()
        if hasattr(self.env, "close"):
            self.env.close()


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
