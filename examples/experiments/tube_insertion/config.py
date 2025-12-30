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
from experiments.tube_insertion.wrapper import RAMEnv

class EnvConfig(DefaultEnvConfig):
    SERVER_URL = "http://127.0.0.2:5000/"
    REALSENSE_CAMERAS = {
        "front_camera": {
            "serial_number": "318122301393",
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
        "front_classifier": {
            "serial_number": "318122301393",
            "dim": (640, 480),
            "exposure": 40000,
        },
        
    }
    EXTRA_REALSENSE_CAMERAS = {
        "side_camera": {
            "serial_number": "242422303461",
            "dim": (640, 480),
            "exposure": 40000,
            "depth": True,
        },
    }
    # IMAGE_CROP = {
    #     "front_camera": lambda img: img[242:370, 232:360],
    #     "wrist_camera": lambda img: img[0:480, 120:600],
    # }
    
    # "wrist_classifier": lambda img: img[50:280, 270:500],
    # "front_classifier": lambda img: img[240:360, 230:350],   D405 ROI
    # "front_camera": lambda img: img[60:340, 140:420],  D405 ROI
    IMAGE_CROP = {
        "front_camera": lambda img: img[57:337, 115:395],
        "wrist_camera": lambda img: img[0:480, 120:600],
        "front_classifier": lambda img: img[240:360, 210:330],
    }
    # TARGET_POSE = np.array([0.5881241235410154,-0.03578590131997776,0.27843494179085326, np.pi, 0, 0])
    TARGET_POSE = np.array([1.55513753, -0.14267503, 0.18153528, -0.03244228, 0.99039508, 0.12396424, -0.05194187])
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.02
    RANDOM_RZ_RANGE = 0.05
    # ACTION_SCALE = (0.01, 0.06, 1)
    # ACTION_SCALE = (0.01, 0.01, 0.05)
    ACTION_SCALE = (0.005, 0.005, 0.05)
    DISPLAY_IMAGE = True
    MAX_EPISODE_LENGTH = 100
    REWARD_THRESHOLD = np.array([0.01, 0.005, 0.01, 1, 1, 1])  # [x, y, z, roll, pitch, yaw]
    
    # 1-4 index, 5-8 middle, 9-12 ring, 13-16 thumb
    
    GRIPPER_CLOSE_JOINT = np.array([
        3.121650934219360352, 4.624951839447021484, 3.364019870758056641, 3.166718912124633789,
        3.406330614089965820, 3.202951908111572266, 3.466796636581420898, 3.969689750671386719,
        3.218291759490966797, 3.238233327865600586, 2.867010116577148438, 3.325670242309570312,
        4.580466747283935547, 3.147728681564331055, 3.663146018981933594, 2.846116971969604492,
    ], dtype=np.float32)
    
    GRIPPER_OPEN_JOINT = np.array([
        3.209087848663330078, 4.422466754913330078, 3.210621833801269531, 3.252039194107055664,
        3.406330614089965820, 3.202951908111572266, 3.466796636581420898, 3.969689750671386719,
        3.218291759490966797, 3.238233327865600586, 2.867010116577148438, 3.325670242309570312,
        4.644893646240234375, 3.060291767120361328, 2.528000354766845703, 3.739845275878906250,
    ], dtype=np.float32)
    
    GRIPPER_TWIST_JOINT = np.array([
        2.359922599792480469, 4.624951839447021484, 3.364019870758056641, 3.166718912124633789,
        3.406330614089965820, 3.202951908111572266, 3.466796636581420898, 3.969689750671386719,
        3.218291759490966797, 3.238233327865600586, 2.867010116577148438, 3.325670242309570312,
        3.982136650085449219, 3.147728681564331055, 3.663146018981933594, 2.846116971969604492
    ], dtype=np.float32)
    
    ENABLE_TACTILE = True
    TACT_BASE_PATH = '/home/ruiqiang/workspaces/HK_TacExo/9DTact/shape_reconstruction/'
    EXP_NAME = "tube_insertion"


class TrainConfig(DefaultTrainingConfig):
    state_weights = np.concatenate(
        [
            np.full(6, 1.0, dtype=np.float32),  # arm joints
            np.full(1, 1.0, dtype=np.float32),  # leaphand joints
        ]
    )
    proprio_keys = ["tcp_pos", "tcp_ori", "gripper_pose"]
    buffer_period = 1000
    checkpoint_period = 1000
    steps_per_update = 100
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-fixed-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False, enable_tactile=True):
        env_config = EnvConfig()
        env_config.ENABLE_TACTILE = enable_tactile
        if enable_tactile:
            self.image_keys = ["front_camera", "wrist_camera", "tactile_data"]
            self.classifier_keys = ["front_classifier", "wrist_camera", "tactile_data"]
            self.classifier_key_weights = {"front_classifier": 1.0, "wrist_camera": 1.0, "tactile_data": 1.0}
            self.image_weights = {"front_camera": 1.0, "wrist_camera": 1.0, "tactile_data": 1.0}
        else:
            self.image_keys = ["front_camera", "wrist_camera"]
            self.classifier_keys = ["front_classifier", "wrist_camera"]
            self.classifier_key_weights = {"front_classifier": 1.0, "wrist_camera": 1.0}
            self.image_weights = {"front_camera": 1.0, "wrist_camera": 1.0, "tactile_data": 1.0}
            
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
                classifier_insert = load_classifier_func(
                    key=jax.random.PRNGKey(0),
                    sample=env.observation_space.sample(),
                    image_keys=self.classifier_keys,
                    image_key_weights=self.classifier_key_weights,
                    checkpoint_path=os.path.abspath("/home/ruiqiang/workspaces/HK_TACEXO_WANG/hil-serl/examples/classifier_ckpt_tube_insertion/"),
                )
                classifier_pick = load_classifier_func(
                    key=jax.random.PRNGKey(0),
                    sample=env.observation_space.sample(),
                    image_keys=self.classifier_keys,
                    image_key_weights=self.classifier_key_weights,
                    checkpoint_path=os.path.abspath("/home/ruiqiang/workspaces/HK_TACEXO_WANG/hil-serl/examples/classifier_ckpt_tube_pick/"),
                )
            else:
                classifier_insert = load_classifier_func(
                    key=jax.random.PRNGKey(0),
                    sample=env.observation_space.sample(),
                    image_keys=self.classifier_keys,
                    image_key_weights=self.classifier_key_weights,
                    checkpoint_path=os.path.abspath("/home/ruiqiang/workspaces/HK_TACEXO_WANG/hil-serl/examples/classifier_ckpt_tube_insertion_no_tactile/"),
                )
                classifier_pick = load_classifier_func(
                    key=jax.random.PRNGKey(0),
                    sample=env.observation_space.sample(),
                    image_keys=self.classifier_keys,
                    image_key_weights=self.classifier_key_weights,
                    checkpoint_path=os.path.abspath("/home/ruiqiang/workspaces/HK_TACEXO_WANG/hil-serl/examples/classifier_ckpt_tube_pick_no_tactile/"),
                )
            # input("debug")
            def reward_func(obs, is_pick=True):
                # print("classifier obs = ", classifier(obs))
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                if is_pick:
                    print("classifier = classifier_pick")
                    classifier = classifier_pick
                else:
                    print("classifier = classifier_place")
                    classifier = classifier_insert
                print("sigmoid(classifier(obs) = ", sigmoid(classifier(obs)))
                # added check for z position to further robustify classifier, but should work without as well
                # return int(sigmoid(classifier(obs)).item() > 0.95)
            
                prob = sigmoid(classifier(obs)).item()
                
                if is_pick:
                    if prob > 0.4:
                        success = 1
                    else:
                        success = 0
                    reward = 0.2 if success else 0
                else:
                    if prob > 0.70:
                        success = 1
                    else:
                        success = 0
                    reward = 1 if success else 0
                # state = obs["state"]
                # ee_pos = state[0, :3] if state.ndim > 1 else state[:3]
                # if ee_pos[2] < 0.16:
                #     reward -= 0.05
                return reward

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        return env