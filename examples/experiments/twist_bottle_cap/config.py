import os
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
from experiments.twist_bottle_cap.wrapper import RAMEnv

class EnvConfig(DefaultEnvConfig):
    SERVER_URL = "http://127.0.0.2:5000/"
    REALSENSE_CAMERAS = {
        "front_camera": {
            "serial_number": "242422303461",
            "dim": (640, 480),
            "exposure": 40000,
            "depth": True,
        },
        "wrist_camera": {
            "serial_number": "218622271185",
            "dim": (640, 480),
            "exposure": 40000,
            "depth": True,
        },
    }
    EXTRA_REALSENSE_CAMERAS = {
        "side_camera": {
            "serial_number": "234222300515",
            "dim": (640, 480),
            "exposure": 40000,
            "depth": True,
        },
    }
    # WRIST_REALSENSE_CAMERAS = {
    #     "wrist_camera": {
    #         "serial_number": "218622271185",
    #         "dim": (640, 480),
    #         "exposure": 40000,
    #         "depth": True,
    #     },
    # }
    IMAGE_CROP = {
        "front_camera": lambda img: img[150:450, 350:1100],
        "side_camera": lambda img: img[100:500, 400:900],
    }
    # TARGET_POSE = np.array([0.5881241235410154,-0.03578590131997776,0.27843494179085326, np.pi, 0, 0])
    TARGET_POSE = np.array([1.55513753, -0.14267503, 0.18153528, -0.03244228, 0.99039508, 0.12396424, -0.05194187])
    # GRASP_POSE = np.array([0.5857508505445138,-0.22036261105675414,0.2731021902359492, np.pi, 0, 0])
    # RESET_POSE = TARGET_POSE + np.array([0, 0, 0.05, 0, 0.05, 0])
    # ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.03, 0.02, 0.01, 0.01, 0.1, 0.4])
    # ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.03, 0.02, 0.05, 0.01, 0.1, 0.4])
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.02
    RANDOM_RZ_RANGE = 0.05
    # ACTION_SCALE = (0.01, 0.06, 1)
    ACTION_SCALE = (0.05, 0.05, 0.05)
    DISPLAY_IMAGE = True
    MAX_EPISODE_LENGTH = 100
    REWARD_THRESHOLD = np.array([0.01, 0.005, 0.01, 1, 1, 1])  # [x, y, z, roll, pitch, yaw]
    # COMPLIANCE_PARAM = {
    #     "translational_stiffness": 2000,
    #     "translational_damping": 89,
    #     "rotational_stiffness": 150,
    #     "rotational_damping": 7,
    #     "translational_Ki": 0,
    #     "translational_clip_x": 0.0075,
    #     "translational_clip_y": 0.0016,
    #     "translational_clip_z": 0.0055,
    #     "translational_clip_neg_x": 0.002,
    #     "translational_clip_neg_y": 0.0016,
    #     "translational_clip_neg_z": 0.005,
    #     "rotational_clip_x": 0.01,
    #     "rotational_clip_y": 0.025,
    #     "rotational_clip_z": 0.005,
    #     "rotational_clip_neg_x": 0.01,
    #     "rotational_clip_neg_y": 0.025,
    #     "rotational_clip_neg_z": 0.005,
    #     "rotational_Ki": 0,
    # }
    # PRECISION_PARAM = {
    #     "translational_stiffness": 2000,
    #     "translational_damping": 89,
    #     "rotational_stiffness": 250,
    #     "rotational_damping": 9,
    #     "translational_Ki": 0.0,
    #     "translational_clip_x": 0.1,
    #     "translational_clip_y": 0.1,
    #     "translational_clip_z": 0.1,
    #     "translational_clip_neg_x": 0.1,
    #     "translational_clip_neg_y": 0.1,
    #     "translational_clip_neg_z": 0.1,
    #     "rotational_clip_x": 0.5,
    #     "rotational_clip_y": 0.5,
    #     "rotational_clip_z": 0.5,
    #     "rotational_clip_neg_x": 0.5,
    #     "rotational_clip_neg_y": 0.5,
    #     "rotational_clip_neg_z": 0.5,
    #     "rotational_Ki": 0.0,
    # }
    IS_ARM_ONLY = True
    ENABLE_TACTILE = True
    TACT_BASE_PATH = '/home/ruiqiang/workspaces/HK_TacExo/9DTact/shape_reconstruction/'
    EXP_NAME = "twist_bottle_cap"


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["front_camera", "wrist_camera", "tactile_data"]
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
            # classifier_pick = load_classifier_func(
            #     key=jax.random.PRNGKey(0),
            #     sample=env.observation_space.sample(),
            #     image_keys=self.classifier_keys,
            #     checkpoint_path=os.path.abspath("../../classifier_ckpt_pick/"),
            # )

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
            classifier_bottle_twist = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                image_key_weights=self.classifier_key_weights,
                checkpoint_path=os.path.abspath("/home/ruiqiang/workspaces/HK_TACEXO_WANG/hil-serl/examples/classifier_ckpt_bottle_twist"),
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
                classifier = classifier_bottle_twist
                print("sigmoid(classifier(obs) = ", sigmoid(classifier(obs)))
                # added check for z position to further robustify classifier, but should work without as well
                # return int(sigmoid(classifier(obs)).item() > 0.95)
            
                prob = sigmoid(classifier(obs)).item()
                success = prob > 0.8
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