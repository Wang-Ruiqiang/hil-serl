import os, sys
import jax
import jax.numpy as jnp
import numpy as np

from serl_robot_infra.denso_env.envs.wrappers import (
    MultiCameraBinaryRewardClassifierWrapper,
    KeyboardIntervention,
)

from denso_env.envs.denso_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.tennis_ball_pick.wrapper import RAMEnv

class EnvConfig(DefaultEnvConfig):
    SERVER_URL = "http://127.0.0.2:5000/"
    REALSENSE_CAMERAS = {
        "front_camera": {
            "serial_number": "242422303461",
            "dim": (640, 480),
            "exposure": 40000,
            "depth": True,
        },
        # "side_camera": {
        #     "serial_number": "234222300515",
        #     "dim": (640, 480),
        #     "exposure": 40000,
        # },
    }
    EXTRA_REALSENSE_CAMERAS = {
        "side_camera": {
            "serial_number": "234222300515",
            "dim": (640, 480),
            "exposure": 40000,
            "depth": True,
        },
    }
    IMAGE_CROP = {
        "front_camera": lambda img: img[150:450, 350:1100],
        "side_camera": lambda img: img[100:500, 400:900],
    }
    TARGET_POSE = np.array([1.55513753, -0.14267503, 0.18153528, -0.03244228, 0.99039508, 0.12396424, -0.05194187])
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.02
    RANDOM_RZ_RANGE = 0.05
    # ACTION_SCALE = (0.01, 0.06, 1)
    ACTION_SCALE = (0.05, 0.05, 0.05)
    DISPLAY_IMAGE = True
    MAX_EPISODE_LENGTH = 100
    REWARD_THRESHOLD = np.array([0.01, 0.005, 0.01, 1, 1, 1])  # [x, y, z, roll, pitch, yaw]
    GRIPPER_CLOSE_JOINT = np.array([
        3.546563625335693359, 4.127942085266113281, 3.413689804077148438, 3.641670465469360352,
        3.626330614089965820, 3.529689788818359375, 2.931437253952026367, 3.782796621322631836,
        3.838019847869873047, 3.532757759094238281, 3.535825729370117188, 3.413107156753540039,
        4.661767482757568359, 3.366175127029418945, 3.260291767120361328, 3.566796636581420898
    ], dtype=np.float32)
    
    GRIPPER_OPEN_JOINT = np.array([
        2.989728450775146484, 3.231437253952026367, 3.438389015197753906, 3.96806390762329102,    #index
        2.904854822158813477, 3.202951908111572266, 3.466796636581420898, 3.969689750671386719,   #middle
        3.218291759490966797, 3.238233327865600586, 2.867010116577148438, 3.325670242309570312,
        4.512019824981689453, 3.3605515193939208984, 3.374757766723632812, 3.397184896469116211    #thumb
    ], dtype=np.float32)
    IS_ARM_ONLY = True
    ENABLE_TACTILE = True
    TACT_BASE_PATH = '/home/ruiqiang/workspaces/HK_TacExo/9DTact/shape_reconstruction/'
    EXP_NAME = "tennis_ball_pick"


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["front_camera", "tactile_data"]
    # image_keys = ["front_camera", "side_camera"]
    classifier_keys = ["front_camera", "tactile_data"]
    classifier_key_weights = {"front_camera": 1.0, "tactile_data": 0.5}
    state_weights = np.concatenate(
        [
            np.full(6, 1.0, dtype=np.float32),  # arm joints
            np.full(1, 1.0, dtype=np.float32),  # leaphand joints
        ]
    )
    proprio_keys = ["tcp_pos", "tcp_ori", "gripper_pose"]
    # proprio_keys = ["tcp_pos", "tcp_ori"]
    # classifier_keys = ["front_camera", "side_camera"]
    buffer_period = 1000
    checkpoint_period = 1000 
    steps_per_update = 100
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-fixed-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env_config = EnvConfig()

        env = RAMEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=env_config,
        )
        # env = GripperCloseEnv(env)
        if not fake_env:
            env = KeyboardIntervention(env)
        # env = RelativeFrame(env)
        # env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        if classifier:
            # print("classifier path = ", os.path.abspath("../../classifier_ckpt/"))
            classifier_pick = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("../../classifier_ckpt_pick/"),
            )

            # classifier_place = load_classifier_func(
            #     key=jax.random.PRNGKey(0),
            #     sample=env.observation_space.sample(),
            #     image_keys=self.classifier_keys,
            #     checkpoint_path=os.path.abspath("../../classifier_ckpt/"),
            # )
            classifier_pick = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("/home/ruiqiang/workspaces/HK_TACEXO_WANG/hil-serl/examples/classifier_ckpt_pick/"),
            )
            
            classifier_normal = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                image_key_weights=self.classifier_key_weights,
                checkpoint_path=os.path.abspath("/home/ruiqiang/workspaces/HK_TACEXO_WANG/hil-serl/examples/classifier_ckpt"),
            )
            # input("debug")
            def reward_func(obs, is_pick=True):
                # print("classifier obs = ", classifier(obs))
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                # if is_pick:
                #     print("classifier = classifier_pick")
                #     classifier = classifier_pick
                # else:
                #     print("classifier = classifier_place")
                #     classifier = classifier_normal
                classifier = classifier_pick
                print("sigmoid(classifier(obs) = ", sigmoid(classifier(obs)))
                # added check for z position to further robustify classifier, but should work without as well
                # return int(sigmoid(classifier(obs)).item() > 0.95)
            
                prob = sigmoid(classifier(obs)).item()
                success = prob > 0.3
                # if classifier == classifier_pick:
                #     reward = 0.3 if success else 0
                # else:
                reward = 1 if success else 0
                state = obs["state"]
                ee_pos = state[0, :3] if state.ndim > 1 else state[:3]
                gripper_pose = state[0, -1] if state.ndim > 1 else state[-1]
                if ee_pos[1] > -0.13 and ee_pos[2] < 0.14:
                    reward -= 0.01
                if ee_pos[2] < 0.02:
                    reward -= 0.05
                # if not is_pick and -0.30 < ee_pos[1] < -0.04 and gripper_pose < 0.8:
                #     reward -= 0.05
                return reward

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        return env