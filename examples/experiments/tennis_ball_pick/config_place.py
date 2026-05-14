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
from experiments.tennis_ball_pick.wrapper import RAMEnv, GripperPenaltyWrapper

class EnvConfig(DefaultEnvConfig):
    SERVER_URL = "http://127.0.0.2:5000/"
    REALSENSE_CAMERAS = {
        "front_camera": {
            "serial_number": "242422303461",
            "dim": (640, 480),
            "exposure": 40000,
            "depth": True,
        },
        # "wrist_camera": {
        #     "serial_number": "218622271185",
        #     "dim": (640, 480),
        #     "exposure": 40000,
        #     "depth": True,
        # },
        # "front_classifier": {
        #     "serial_number": "318122301393",
        #     "dim": (640, 480),
        #     "exposure": 40000,
        # },
    }
    EXTRA_REALSENSE_CAMERAS = {
        "front_camera_2": {
            "serial_number": "318122301393",
            "dim": (640, 480),
            "exposure": 40000,
            "depth": True,
        },
    }
    IMAGE_CROP = {
        "front_camera": lambda img: img[0:460, 60:520],
        # "front_classifier": lambda img: img[240:360, 210:330],
    }
    TARGET_POSE = np.array([1.55513753, -0.14267503, 0.18153528, -0.03244228, 0.99039508, 0.12396424, -0.05194187])
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.02
    RANDOM_RZ_RANGE = 0.05
    # ACTION_SCALE = (0.01, 0.06, 1)
    ACTION_SCALE = (0.03, 0.03, 0.03)
    DISPLAY_IMAGE = True
    MAX_EPISODE_LENGTH = 100
    REWARD_THRESHOLD = np.array([0.01, 0.005, 0.01, 1, 1, 1])  # [x, y, z, roll, pitch, yaw]
    GRIPPER_CLOSE_JOINT = np.array([
        3.1584663, 4.4301367, 3.3057287, 3.3241363,
        3.406330614089965820, 3.202951908111572266, 3.466796636581420898, 3.969689750671386719,
        3.218291759490966797, 3.238233327865600586, 2.867010116577148438, 3.325670242309570312,
        4.6126804, 3.118583, 3.028078, 3.4085052,
    ], dtype=np.float32)
    GRIPPER_OPEN_JOINT = np.array([
        3.209087848663330078, 4.022466754913330078, 3.210621833801269531, 3.652039194107055664,
        3.406330614089965820, 3.202951908111572266, 3.466796636581420898, 3.969689750671386719,
        3.218291759490966797, 3.238233327865600586, 2.867010116577148438, 3.325670242309570312,
        4.644893646240234375, 3.060291767120361328, 2.528000354766845703, 3.739845275878906250,
    ], dtype=np.float32)
    
    
    ENABLE_TACTILE = True
    TACT_BASE_PATH = '/home/wrq/workspaces/HK_TACEXO_WANG/9DTact/shape_reconstruction/'
    EXP_NAME = "tennis_ball_place"


class TrainConfig(DefaultTrainingConfig):
    state_weights = np.concatenate(
        [
            np.full(6, 1.0, dtype=np.float32),  # arm joints
            np.full(1, 1.0, dtype=np.float32),  # leaphand joints
        ]
    )
    proprio_keys = ["tcp_pos", "tcp_ori", "gripper_pose"]
    batch_size = 128
    buffer_period = 1000
    checkpoint_period = 1000
    steps_per_update = 100
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-fixed-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False, enable_tactile=True):
        env_config = EnvConfig()
        env_config.ENABLE_TACTILE = enable_tactile

        if enable_tactile:
            self.image_keys = ["front_camera", "tactile_data"]
            self.classifier_keys = ["front_camera", "tactile_data"]
            self.classifier_key_weights = {"front_camera": 1.0, "tactile_data": 1.0}
            self.image_weights = {"front_camera": 1.0, "tactile_data": 1.0}
        else:
            self.image_keys = ["front_camera"]
            self.classifier_keys = ["front_camera"]
            self.classifier_key_weights = {"front_camera": 1.0}
            self.image_weights = {"front_camera": 1.0}
            
        
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
            if enable_tactile:
                classifier_pick = load_classifier_func(
                    key=jax.random.PRNGKey(0),
                    sample=env.observation_space.sample(),
                    image_keys=self.classifier_keys,
                    checkpoint_path=os.path.abspath("/home/wrq/workspaces/HK_TACEXO_WANG/hil-serl/examples/tennis_ball_pick_classifier/classifier_ckpt_ball_pick/"),
                )
                classifier_place = load_classifier_func(
                    key=jax.random.PRNGKey(0),
                    sample=env.observation_space.sample(),
                    image_keys=self.classifier_keys,
                    checkpoint_path=os.path.abspath("../../tennis_ball_pick_classifier/classifier_ckpt_ball_place/"),
                )
            else:
                classifier_pick = load_classifier_func(
                    key=jax.random.PRNGKey(0),
                    sample=env.observation_space.sample(),
                    image_keys=self.classifier_keys,
                    checkpoint_path=os.path.abspath("/home/wrq/workspaces/HK_TACEXO_WANG/hil-serl/examples/tennis_ball_pick_classifier/classifier_ckpt_ball_pick_no_tactile/"),
                )
                classifier_place = load_classifier_func(
                    key=jax.random.PRNGKey(0),
                    sample=env.observation_space.sample(),
                    image_keys=self.classifier_keys,
                    checkpoint_path=os.path.abspath("../../tennis_ball_pick_classifier/classifier_ckpt_ball_place_no_tactile/"),
                )
            
            # input("debug")
            def reward_func(obs, is_pick=True):
                # print("classifier obs = ", classifier(obs))
                if is_pick:
                    print("classifier = classifier_pick")
                    classifier = classifier_pick
                else:
                    print("classifier = classifier_place")
                    classifier = classifier_place
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                print("sigmoid(classifier(obs) = ", sigmoid(classifier(obs)))
                # added check for z position to further robustify classifier, but should work without as well
                # return int(sigmoid(classifier(obs)).item() > 0.95)
            
                prob = sigmoid(classifier(obs)).item()
                if is_pick:
                    success = prob > 0.9
                    reward = 1 if success else 0
                else:
                    success = prob > 0.9
                    reward = 1 if success else 0
                return reward

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        env = GripperPenaltyWrapper(env, exp_name=env_config.EXP_NAME, penalty=-0.02)
        return env