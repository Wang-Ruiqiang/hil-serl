import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

from serl_robot_infra.franka_env.envs.wrappers import (
    MultiCameraBinaryRewardClassifierWrapper,
    KeyboardIntervention,
    SpacemouseIntervention,
)

from franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.wrappers.gaze_derived_observation import GazeDerivedObservationWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.tennis_ball_pick.wrapper import RAMEnv, GripperPenaltyWrapper


REPO_ROOT = Path(__file__).resolve().parents[3]
REWARD_CLASSIFIER_DIR = REPO_ROOT / "examples" / "reward_classifier"


def _reward_classifier_ckpt(name):
    return str((REWARD_CLASSIFIER_DIR / name).resolve())


class EnvConfig(DefaultEnvConfig):
    SERVER_URL = "http://127.0.0.2:5000/"
    REALSENSE_CAMERAS = {
        #denso
        # "front_camera": {
        #     "serial_number": "242422303461",
        #     "dim": (640, 480),
        #     "exposure": 40000,
        #     "depth": True,
        # },
        #franka
        # "front_camera": {
        #     "serial_number": "036522072607",
        #     "dim": (640, 480),
        #     "exposure": 40000,
        #     "depth": True,
        # },

        #franka
        # "front_camera": {
        #     "serial_number": "151422254571",
        #     "dim": (640, 480),
        #     "exposure": 40000,
        #     "depth": True,
        # },
        "front_camera": {
            "serial_number": "234222300515",
            "dim": (640, 480),
            "exposure": 40000,
            "depth": True,
        },
    }
    TARGET_POSE = np.array([1.55513753, -0.14267503, 0.18153528, -0.03244228, 0.99039508, 0.12396424, -0.05194187])
    RESET_JOINT = np.array([
        -0.18156941995945866,
        0.1296803145320053,
        0.0,
        -1.9407152154140208,
        0.0,
        2.1285232352375885,
        -0.23116278766225817,
    ], dtype=np.float32)
    RESET_JOINT_DURATION = 4.0
    RESET_JOINT_STEPS = 80
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.02
    RANDOM_RZ_RANGE = 0.05
    ACTION_SCALE = (0.004, 0.004, 0.004)
    CMD_POSE_RESYNC_THRESHOLD = 0.05
    DISPLAY_IMAGE = True
    GAZE_DISPLAY_MARKERS = True
    GAZE_RS_SAVE_WIDTH = 640
    GAZE_RS_SAVE_HEIGHT = 480
    MAX_EPISODE_LENGTH = 200
    REWARD_THRESHOLD = np.array([0.01, 0.005, 0.01, 1, 1, 1])  # [x, y, z, roll, pitch, yaw]

    # 1-4 index, 5-8 middle, 9-12 ring, 13-16 thumb
    GRIPPER_CLOSE_JOINT = np.array([
        3.1584663, 4.4301367, 3.4057287, 3.4241363,
        3.406330614089965820, 3.202951908111572266, 3.466796636581420898, 3.969689750671386719,
        3.218291759490966797, 3.238233327865600586, 2.867010116577148438, 3.325670242309570312,
        4.6126804, 3.118583, 3.128078, 3.5085052,
    ], dtype=np.float32)

    # GRIPPER_CLOSE_JOINT = np.array([
    #     3.078699350, 4.512971401, 2.707476139, 4.020563602,
    #     3.406330614089965820, 3.202951908111572266, 3.466796636581420898, 3.969689750671386719,
    #     3.218291759490966797, 3.238233327865600586, 2.867010116577148438, 3.325670242309570312,
    #     4.594272614, 3.063359737, 3.178408146, 3.528155804,
    # ], dtype=np.float32)


    GRIPPER_OPEN_JOINT = np.array([
        3.209087848663330078, 4.022466754913330078, 3.210621833801269531, 3.652039194107055664,
        3.406330614089965820, 3.202951908111572266, 3.466796636581420898, 3.969689750671386719,
        3.218291759490966797, 3.238233327865600586, 2.867010116577148438, 3.325670242309570312,
        4.644893646240234375, 3.060291767120361328, 2.528000354766845703, 3.739845275878906250,
    ], dtype=np.float32)
    
    
    ENABLE_TACTILE = True
    TACT_BASE_PATH = '/home/user/franka_ros2_ws/src/tact9d/tact9d/shape_reconstruction/'
    DM_TAC_DEPTH_SCALE = 3
    USE_SPACEMOUSE = True
    EXP_NAME = "tennis_ball_pick"


class TrainConfig(DefaultTrainingConfig):
    proprio_keys = ["tcp_pos", "tcp_ori", "gripper_pose"]
    batch_size = 128
    log_period = 10
    buffer_period = 1000
    checkpoint_period = 1000
    steps_per_update = 100
    # Set to "vit" to train a lightweight ViT encoder for RGB/tactile images.
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-fixed-gripper"
    mask_pick_place_phase_control = False
    mask_suppress_beta = 1.0
    use_mask_feature_head = True
    mask_feature_gate_alpha = 0.9
    mask_feature_min_gate = 0.4
    mask_feature_hidden_dim = 128
    use_mask_encoder = True
    mask_encoder_latent_dim = 64
    mask_grounding_threshold = 0.05
    mask_grounding_cell_threshold = 0.04

    def get_image_keys(
        self,
        enable_tactile=True,
        use_gaze_target_mask=True,
    ):
        image_keys = ["front_camera"]
        if enable_tactile:
            image_keys.append("tactile_data")
        if use_gaze_target_mask:
            image_keys.extend(["front_camera_mask1", "front_camera_mask2"])
        return image_keys

    def get_classifier_keys(self, enable_tactile=True):
        return self.get_image_keys(
            enable_tactile,
            use_gaze_target_mask=False,
        )

    def get_environment(
        self,
        fake_env=False,
        save_video=False,
        classifier=False,
        enable_tactile=True,
        record_data=False,
        record_gaze=False,
        gaze_predictor_checkpoint_path=str(
            REPO_ROOT / "examples" / "gaze_data_process" / "gaze_heatmap_ckpt"
        ),
        mask_predictor_checkpoint_path=str(
            REPO_ROOT
            / "examples"
            / "gaze_data_process"
            / "SAM_process"
            / "mask_predictor_ckpt"
            / "best.pt"
        ),
        mask_selection_mode="gaze",
        pick_classifier_checkpoint_path=_reward_classifier_ckpt("classifier_ckpt_ball_pick"),
        pick_classifier_threshold=0.95,
        gaze_target_mask_dilation=0,
        frame_save_path=None,
    ):
        env_config = EnvConfig()
        env_config.ENABLE_TACTILE = enable_tactile
        env_config.ENABLE_DATA_RECORDING = bool(record_data)
        env_config.ENABLE_GAZE_COLLECTION = bool(record_gaze)
        if frame_save_path is not None:
            env_config.GAZE_FRAME_SAVE_PATH = frame_save_path

        self.image_keys = self.get_image_keys(
            enable_tactile,
            use_gaze_target_mask=True,
        )
        self.classifier_keys = self.get_classifier_keys(enable_tactile)
            
        
        env = RAMEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=env_config,
        )
        # env = GripperCloseEnv(env)
        if not fake_env:
            env = KeyboardIntervention(env)
            if getattr(env_config, "USE_SPACEMOUSE", False):
                env = SpacemouseIntervention(env)
        # env = RelativeFrame(env)
        # env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = GazeDerivedObservationWrapper(
            env,
            use_gaze_target_mask=True,
            gaze_predictor_checkpoint_path=gaze_predictor_checkpoint_path,
            mask_predictor_checkpoint_path=mask_predictor_checkpoint_path,
            mask_selection_mode=mask_selection_mode,
            pick_classifier_checkpoint_path=pick_classifier_checkpoint_path,
            pick_classifier_threshold=pick_classifier_threshold,
            gaze_target_mask_dilation=gaze_target_mask_dilation,
            log_fn=print,
        )
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        if classifier:
            sample = env.observation_space.sample()
            print("[debug] classifier image keys:", self.classifier_keys)
            for image_key in self.classifier_keys:
                image_sample = sample.get(image_key)
                if image_sample is None:
                    print(f"[debug] sample[{image_key}] is missing")
                    continue
                print(
                    f"[debug] sample[{image_key}] "
                    f"shape={image_sample.shape}, dtype={image_sample.dtype}"
                )
            if enable_tactile:
                reward_classifier = load_classifier_func(
                    key=jax.random.PRNGKey(0),
                    sample=sample,
                    image_keys=self.classifier_keys,
                    checkpoint_path=_reward_classifier_ckpt("classifier_ckpt_ball_pick"),
                )
            else:
                reward_classifier = load_classifier_func(
                    key=jax.random.PRNGKey(0),
                    sample=sample,
                    image_keys=self.classifier_keys,
                    checkpoint_path=_reward_classifier_ckpt("classifier_ckpt_ball_pick_no_tactile"),
                )
            
            # input("debug")
            def reward_func(obs, is_pick=True):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                prob = sigmoid(reward_classifier(obs)).item()
                print("sigmoid(reward_classifier(obs)) = ", prob)
                # added check for z position to further robustify classifier, but should work without as well
                # return int(sigmoid(classifier(obs)).item() > 0.95)
                return int(prob > 1)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        env = GripperPenaltyWrapper(env, exp_name=env_config.EXP_NAME, penalty=-0.02)
        return env
