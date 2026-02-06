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
from experiments.twist_bottle_cap.wrapper import RAMEnv, GripperPenaltyWrapper, RobotArmPenaltyWrapper

class EnvConfig(DefaultEnvConfig):
    SERVER_URL = "http://127.0.0.2:5000/"
    REALSENSE_CAMERAS = {
        "front_camera": {
            "serial_number": "242422303461",
            "dim": (640, 480),
            # "exposure": 40000,
            "depth": True,
        },
        "wrist_camera": {
            "serial_number": "218622273562",
            "dim": (640, 480),
            # "exposure": 40000,
            "depth": True,
        },
    }
    EXTRA_REALSENSE_CAMERAS = {
        "front_camera_2": {
            "serial_number": "318122301393",
            "dim": (640, 480),
            # "exposure": 40000,
            "depth": True,
        },
    }
    IMAGE_CROP = {
        "front_camera": lambda img: img[35:375, 160:500],
        "wrist_camera": lambda img: img[0:480, 120:600],
    }
    # TARGET_POSE = np.array([0.5881241235410154,-0.03578590131997776,0.27843494179085326, np.pi, 0, 0])
    TARGET_POSE = np.array([1.55513753, -0.14267503, 0.18153528, -0.03244228, 0.99039508, 0.12396424, -0.05194187])
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.02
    RANDOM_RZ_RANGE = 0.05
    # ACTION_SCALE = (0.01, 0.06, 1)
    ACTION_SCALE = (0.025, 0.025, 0.025)
    DISPLAY_IMAGE = True
    MAX_EPISODE_LENGTH = 100
    REWARD_THRESHOLD = np.array([0.01, 0.005, 0.01, 1, 1, 1])  # [x, y, z, roll, pitch, yaw]
    
    # 1-4 index, 5-8 middle, 9-12 ring, 13-16 thumb
    GRIPPER_CLOSE_JOINT = np.array([
        3.341010093688964844, 4.459281921386718750, 3.118582963943481445, 3.745980978012084961,
        3.406330614089965820, 3.202951908111572266, 3.466796636581420898, 3.969689750671386719,
        3.218291759490966797, 3.238233327865600586, 2.867010116577148438, 3.325670242309570312,
        4.709321022033691406, 3.160000324249267578, 2.954447031021118164, 3.816544294357299805
    ], dtype=np.float32)
            
    GRIPPER_TWIST_JOINT = np.array([
        2.659922599792480469, 4.605010509490966797, 3.118582963943481445, 3.745980978012084961,
        3.406330614089965820, 3.202951908111572266, 3.466796636581420898, 3.969689750671386719,
        3.218291759490966797, 3.238233327865600586, 2.867010116577148438, 3.325670242309570312,
        5.062136650085449219, 2.856272220611572266, 2.954447031021118164, 3.816544294357299805
    ], dtype=np.float32)
    
    GRIPPER_OPEN_JOINT = np.array([
        2.989728450775146484, 3.231437253952026367, 3.438389015197753906, 3.96806390762329102,
        3.406330614089965820, 3.529689788818359375, 3.438389015197753906, 3.969689750671386719,
        3.218291759490966797, 3.238233327865600586, 2.867010116577148438, 3.325670242309570312,
        4.670971393585205078, 3.207553863525390625, 2.396078109741210938, 3.879437446594238281
    ], dtype=np.float32)
    ENABLE_TACTILE = True
    TACT_BASE_PATH = '/home/wrq/workspaces/HK_TACEXO_WANG/9DTact/shape_reconstruction/'
    EXP_NAME = "twist_bottle_cap"
    LOOP_CONTROL = True


class TrainConfig(DefaultTrainingConfig):
    # image_keys = ["front_camera", "wrist_camera", "tactile_data"]
    # classifier_keys = ["front_camera", "wrist_camera"]
    # classifier_key_weights = {"front_camera": 1.0, "wrist_camera": 1.0}
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
    batch_size = 128
    checkpoint_period = 1000
    steps_per_update = 100
    encoder_type = "resnet-pretrained"

    def get_environment(self, fake_env=False, save_video=False, classifier=False, enable_tactile=False):
        env_config = EnvConfig()
        env_config.ENABLE_TACTILE = enable_tactile
        if enable_tactile:
            # self.image_keys = ["front_camera", "wrist_camera", "tactile_data"]
            # self.classifier_keys = ["front_camera", "wrist_camera","tactile_data"]
            # self.classifier_key_weights = {"front_camera": 1.0, "wrist_camera": 1.0, "tactile_data": 0.5}
            self.image_keys = ["front_camera", "wrist_camera", "tactile_data"]
            self.classifier_keys = ["front_camera", "tactile_data"]
            self.classifier_keys_grip_lid = ["front_camera", "wrist_camera", "tactile_data"]
            # self.classifier_key_weights = {"front_camera": 1.0, "tactile_data": 1.0}
            # self.image_weights = {"front_camera": 1.0, "wrist_camera": 1.0, "tactile_data": 1.0}
        else:
            self.image_keys = ["front_camera", "wrist_camera"]
            self.classifier_keys = ["front_camera", "wrist_camera"]
            # self.classifier_key_weights = {"front_camera": 1.0, "wrist_camera": 1.0}
            # self.image_weights = {"front_camera": 1.0, "wrist_camera": 1.0}
            
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
        env = GripperPenaltyWrapper(env, penalty=-0.02)
        env = RobotArmPenaltyWrapper(env, penalty=-0.02)
        if classifier:
            if enable_tactile:
                classifier_bottle_twist = load_classifier_func(
                    key=jax.random.PRNGKey(0),
                    sample=env.observation_space.sample(),
                    image_keys=self.classifier_keys,
                    checkpoint_path=os.path.abspath("/home/wrq/workspaces/HK_TACEXO_WANG/hil-serl/examples/classifier_ckpt_bottle_twist"),
                )
                classifier_lid_grip = load_classifier_func(
                    key=jax.random.PRNGKey(0),
                    sample=env.observation_space.sample(),
                    image_keys=self.classifier_keys_grip_lid,
                    checkpoint_path=os.path.abspath("/home/wrq/workspaces/HK_TACEXO_WANG/hil-serl/examples/classifier_ckpt_lid_grip"),
                )
            else:
                classifier_bottle_twist = load_classifier_func(
                    key=jax.random.PRNGKey(0),
                    sample=env.observation_space.sample(),
                    image_keys=self.classifier_keys,
                    checkpoint_path=os.path.abspath("/home/wrq/workspaces/HK_TACEXO_WANG/hil-serl/examples/classifier_ckpt_bottle_twist_no_tactile"),
                )
                classifier_lid_grip = load_classifier_func(
                    key=jax.random.PRNGKey(0),
                    sample=env.observation_space.sample(),
                    image_keys=self.classifier_keys,
                    checkpoint_path=os.path.abspath("/home/wrq/workspaces/HK_TACEXO_WANG/hil-serl/examples/classifier_ckpt_lid_grip_no_tactile"),
                )


            def reward_func(obs, is_pick=True):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))

                if is_pick:
                    print("classifier = classifier_pick")
                    print("sigmoid(classifier(obs) = ", sigmoid(classifier_lid_grip(obs)))
                    prob = sigmoid(classifier_lid_grip(obs)).item()
                else:
                    print("classifier = classifier_place")
                    print("sigmoid(classifier(obs) = ", sigmoid(classifier_bottle_twist(obs)))
                    prob = sigmoid(classifier_bottle_twist(obs)).item()

                # print("sigmoid(classifier(obs) = ", sigmoid(classifier(obs)))
                # prob = sigmoid(classifier(obs)).item()
                if is_pick:
                    success = prob > 0.6
                    if success:
                        env.unwrapped.stop_cur_command()
                        reward = 1
                    else:
                        reward = 0
                else:
                    success = prob > 0.9
                    reward = 1 if success else 0
                return reward
                # success = prob > 0.5
                # reward = 1 if success else 0
                # return reward

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        
        if not fake_env:
            env = KeyboardIntervention(env)
        return env