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
from experiments.flip_object.wrapper import RAMEnv


class EnvConfig(DefaultEnvConfig):
    SERVER_URL = "http://127.0.0.2:5000/"
    REALSENSE_CAMERAS = {
        "front_camera": {
            "serial_number": "318122301393",
            "dim": (640, 480),
            "depth": True,
            "color_format": "yuyv",
        },
        "wrist_camera": {
            "serial_number": "218622273562",
            "dim": (640, 480),
            "depth": True,
            "color_format": "yuyv",
        },
    }
    IMAGE_CROP = {
        "front_camera": lambda img: img[35:375, 100:440],
        "wrist_camera": lambda img: img[0:480, 0:640],
    }
    TARGET_POSE = np.array(
        [1.55513753, -0.14267503, 0.18153528, -0.03244228, 0.99039508, 0.12396424, -0.05194187]
    )
    ACTION_SCALE = (0.025, 0.025, 0.025)
    DISPLAY_IMAGE = True
    DISPLAY_TACTILE_RAW = False
    MAX_EPISODE_LENGTH = 200
    REWARD_THRESHOLD = np.array([0.01, 0.005, 0.01, 1, 1, 1])
    ENABLE_TACTILE = True
    USE_THREE_FINGER_TACTILE = True
    TACT_BASE_PATH = "/home/wrq/workspaces/HK_TACEXO_WANG/9DTact/shape_reconstruction/"
    EXP_NAME = "flip_object"
    LOOP_CONTROL = True
    REWARD_CLASSIFIER_THRESHOLD = 0.9
    CLASSIFIER_CHECKPOINT_PATH = (
        "/home/wrq/workspaces/HK_TACEXO_WANG/hil-serl/examples/"
        "flip_object_classifier/classifier_ckpt_flip_object"
    )
    CLASSIFIER_NO_TACTILE_CHECKPOINT_PATH = (
        "/home/wrq/workspaces/HK_TACEXO_WANG/hil-serl/examples/"
        "flip_object_classifier/classifier_ckpt_flip_object_no_tactile"
    )

    # 1-4 index, 5-8 middle, 9-12 ring, 13-16 thumb
    # GRIPPER_OPEN_JOINT = np.array([
    #     3.149728450775146484, 3.231437253952026367, 3.438389015197753906, 3.96806390762329102,
    #     3.146330614089965820, 3.239689788818359375, 3.438389015197753906, 3.969689750671386719,
    #     3.218291759490966797, 3.238233327865600586, 2.867010116577148438, 3.325670242309570312,
    #     4.670971393585205078, 3.207553863525390625, 1.896078109741210938, 3.149437446594238281,
    # ], dtype=np.float32)

    # Backup before 2026-07-30_03 state update.
    # OLD_FLIP_OBJECT_STATE_1_JOINT = GRIPPER_OPEN_JOINT
    # OLD_FLIP_OBJECT_STATE_2_JOINT = np.array([
    #     3.147728681564331055, 4.229185104370117188, 2.896155834197998047, 3.827281951904296875,
    #     3.146330614089965820, 3.529689788818359375, 3.438389015197753906, 3.969689750671386719,
    #     3.218291759490966797, 3.238233327865600586, 2.867010116577148438, 3.325670242309570312,
    #     4.733864784240722656, 3.066427707672119141, 2.572485685348510742, 3.718369483947753906,
    # ], dtype=np.float32)
    # OLD_FLIP_OBJECT_STATE_3_JOINT = np.array([
    #     3.147495670318603516, 4.124874114990234375, 2.744291543960571289, 4.828971385955810547,
    #     3.146330614089965820, 3.529689788818359375, 3.438389015197753906, 3.969689750671386719,
    #     3.218291759490966797, 3.238233327865600586, 2.867010116577148438, 3.325670242309570312,
    #     4.700117111206054688, 3.066427707672119141, 2.992796421051025391, 3.336408138275146484,
    # ], dtype=np.float32)
    # OLD_FLIP_OBJECT_STATE_4_JOINT = np.array([
    #     3.147495670318603516, 4.124874114990234375, 2.744291543960571289, 4.828971385955810547,
    #     3.189146041870117188, 4.011359691619873047, 3.937728643417358398, 3.969359683990478516,
    #     3.218291759490966797, 3.238233327865600586, 2.867010116577148438, 3.325670242309570312,
    #     4.752272605895996094, 3.124718904495239258, 3.267379045486450195, 3.192214012145996094,
    # ], dtype=np.float32)
    # OLD_FLIP_OBJECT_STATE_5_JOINT = np.array([
    #     3.147495670318603516, 3.624874114990234375, 2.744291543960571289, 4.828971385955810547,
    #     3.149146041870117188, 4.211359691619873047, 4.137728643417358398, 4.169359683990478516,
    #     3.218291759490966797, 3.238233327865600586, 2.867010116577148438, 3.325670242309570312,
    #     4.752272605895996094, 3.124718904495239258, 3.267379045486450195, 3.792214012145996094,
    # ], dtype=np.float32)

    # From flip_object_hoh_2026_07_30_03/frame_494.
    FLIP_OBJECT_STATE_1_JOINT = np.array([
        3.149728450775146484, 3.631437253952026367, 3.438389015197753906, 3.96806390762329102,
        3.146330614089965820, 3.639689788818359375, 3.438389015197753906, 3.969689750671386719,
        3.218291759490966797, 3.238233327865600586, 2.867010116577148438, 3.325670242309570312,
        4.670971393585205078, 3.207553863525390625, 1.896078109741210938, 4.279437446594238281,
    ], dtype=np.float32)

    # grip object: index/thumb from frame_148, middle and ring unchanged.
    FLIP_OBJECT_STATE_2_JOINT = np.array([
        3.173806190490722656, 4.548253059387207031, 2.851670265197753906, 3.597184896469116211,
        3.146330614089965820, 3.629689788818359375, 3.438389015197753906, 3.969689750671386719,
        3.218291759490966797, 3.238233327865600586, 2.867010116577148438, 3.325670242309570312,
        4.703185081481933594, 3.296524763107299805, 2.872485685348510742, 3.718369483947753906,
    ], dtype=np.float32)

    # 3.147495670318603516, 4.124874114990234375, 2.744291543960571289, 4.828971385955810547
    # first flip with index: frame_162, middle and ring unchanged.
    FLIP_OBJECT_STATE_3_JOINT = np.array([
        3.225961685180664062, 4.575495624542236328, 2.794913053512573242, 4.314874553680419922,
        3.146330614089965820, 3.629689788818359375, 3.438389015197753906, 3.969689750671386719,
        3.218291759490966797, 3.238233327865600586, 2.867010116577148438, 3.325670242309570312,
        4.703185081481933594, 3.296524763107299805, 2.872485685348510742, 3.718369483947753906,
    ], dtype=np.float32)

    # hold with middle: only middle/thumb from frame_433.
    FLIP_OBJECT_STATE_4_JOINT = np.array([
        3.225961685180664062, 4.575495624542236328, 2.794913053512573242, 4.714874553680419922,
        3.138524770736694336, 4.443942546844482422, 3.009670257568359375, 4.348835468292236328,
        3.218291759490966797, 3.238233327865600586, 2.867010116577148438, 3.325670242309570312,
        4.703185081481933594, 3.296524763107299805, 2.872485685348510742, 3.718369483947753906,
    ], dtype=np.float32)

    #  5-8 3.086369276046752930, 4.690913200378417969, 2.952913045883178711, 4.456213951110839844
    # flip: only middle/thumb from frame_478; index and ring unchanged.
    FLIP_OBJECT_STATE_5_JOINT = np.array([
        3.147495670318603516, 3.624874114990234375, 2.744291543960571289, 4.528971385955810547,
        3.149146041870117188, 4.211359691619873047, 3.937728643417358398, 4.369359683990478516,
        3.218291759490966797, 3.238233327865600586, 2.867010116577148438, 3.325670242309570312,
        4.963301467895507812, 3.296524763107299805, 2.969786882400512695, 3.742913007736206055,
    ], dtype=np.float32)

     # thumb   4.663301467895507812, 3.296524763107299805, 2.969786882400512695, 3.742913007736206055,

    FLIP_OBJECT_HAND_STATES = np.stack(
        [
            FLIP_OBJECT_STATE_1_JOINT,
            FLIP_OBJECT_STATE_2_JOINT,
            FLIP_OBJECT_STATE_3_JOINT,
            FLIP_OBJECT_STATE_4_JOINT,
            FLIP_OBJECT_STATE_5_JOINT
        ],
        axis=0,
    ).astype(np.float32)


class TrainConfig(DefaultTrainingConfig):
    state_weights = np.concatenate(
        [
            np.full(6, 1.0, dtype=np.float32),
            np.full(1, 1.0, dtype=np.float32),
        ]
    )
    proprio_keys = ["tcp_pos", "tcp_ori", "gripper_pose"]
    buffer_period = 1000
    batch_size = 128
    checkpoint_period = 1000
    steps_per_update = 100
    encoder_type = "resnet-pretrained"

    def get_environment(
        self,
        fake_env=False,
        save_video=False,
        classifier=False,
        enable_tactile=None,
        record_data=False,
        frame_save_path=None,
    ):
        env_config = EnvConfig()
        if enable_tactile is not None:
            env_config.ENABLE_TACTILE = bool(enable_tactile)
        enable_tactile = env_config.ENABLE_TACTILE

        if enable_tactile:
            self.image_keys = ["front_camera", "wrist_camera", "tactile_data"]
            self.classifier_keys = ["front_camera", "wrist_camera", "tactile_data"]
        else:
            self.image_keys = ["front_camera", "wrist_camera"]
            self.classifier_keys = ["front_camera", "wrist_camera"]

        env = RAMEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=env_config,
            record_data=record_data,
            frame_save_path=frame_save_path,
        )
        if not fake_env:
            env = KeyboardIntervention(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        if classifier:
            checkpoint_path = os.path.abspath(
                env_config.CLASSIFIER_CHECKPOINT_PATH
                if enable_tactile
                else env_config.CLASSIFIER_NO_TACTILE_CHECKPOINT_PATH
            )
            if not os.path.isdir(checkpoint_path):
                raise FileNotFoundError(
                    f"Flip-object classifier checkpoint not found: {checkpoint_path}"
                )
            print(
                f"[flip_object][classifier] loading {checkpoint_path} "
                f"threshold={env_config.REWARD_CLASSIFIER_THRESHOLD}"
            )
            classifier_flip = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=checkpoint_path,
            )

            def reward_func(obs, is_pick=True):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                prob = sigmoid(classifier_flip(obs)).item()
                print("sigmoid(classifier_flip_object(obs)) = ", prob)
                return 1 if prob > env_config.REWARD_CLASSIFIER_THRESHOLD else 0

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        return env
