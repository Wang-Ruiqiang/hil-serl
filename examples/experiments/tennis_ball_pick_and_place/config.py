import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

from serl_robot_infra.franka_env.envs.wrappers import (
    MultiCameraBinaryRewardClassifierWrapper,
    KeyboardIntervention,
    SpacemouseIntervention,
    ArmActionSubspaceWrapper,
)

from franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.wrappers.gaze_derived_observation import GazeDerivedObservationWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.tennis_ball_pick_and_place.wrapper import RAMEnv, GripperPenaltyWrapper


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
        -0.1662740508,
        0.0850178096,
        0.0055932945,
        -2.0390907424,
        -0.0005582517,
        2.1241071588,
        -0.1604075928,
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
    MAX_EPISODE_LENGTH = 250
    REWARD_THRESHOLD = np.array([0.01, 0.005, 0.01, 1, 1, 1])  # [x, y, z, roll, pitch, yaw]

    # 1-4 index, 5-8 middle, 9-12 ring, 13-16 thumb
    GRIPPER_CLOSE_JOINT = np.array([
        3.1584663, 4.4301367, 3.4057287, 3.4241363,
        3.406330614089965820, 3.202951908111572266, 3.466796636581420898, 3.969689750671386719,
        3.218291759490966797, 3.238233327865600586, 2.867010116577148438, 3.325670242309570312,
        4.6126804, 3.118583, 3.128078, 3.5085052,
    ], dtype=np.float32)
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
    # Raw pyspacemouse readings top out around 0.6 on this device, so human
    # demonstrations only ever covered the inner part of the [-1, 1] action
    # box while the policy operates near its tanh saturation (|a| ~= 0.9 at
    # 48k steps). Scale the device up so a full deflection reaches the same
    # range the policy can. This does NOT touch action_scale, so previously
    # recorded demos remain physically valid -- only the values a human
    # produces from here on change. Verify with examples/spacemouse_test.py.
    SPACEMOUSE_GAIN = 1.6
    # franka_env ignores action[3:6] (rpy is pinned to a constant quaternion)
    # and train_rlpd zeroes those slots before storing a transition, so the
    # critic never sees a nonzero value there while the actor keeps sampling
    # one. Drop them from the action space entirely; the SpaceMouse rpy
    # interface is untouched and simply no longer reaches the policy.
    MASK_RPY_ACTION = True
    EXP_NAME = "tennis_ball_pick_and_place"


class TrainConfig(DefaultTrainingConfig):
    proprio_keys = ["tcp_pos", "tcp_ori", "gripper_pose"]
    batch_size = 128
    log_period = 10
    buffer_period = 1000
    checkpoint_period = 2000
    steps_per_update = 100
    # Frozen ResNet RGB backbone with pick-phase mask2 suppression, mask1 head,
    # raw/head fusion, and a separate lightweight CNN for tactile heatmaps.
    #   front_camera      -> frozen ResNet10 -> mask2 suppression -> mask1
    #                        feature head -> raw/head Dense fusion -> 256D
    #   front_camera_mask -> small mask CNN                       ->  64D
    #   tactile_data      -> SharedTactileCNNEncoder              -> 256D
    #   state             -> Dense + LayerNorm + tanh             ->  64D
    encoder_type = "resnet-pretrained"
    # "cnn": small shared tactile CNN. "resnet": original frozen-ResNet tactile
    # branch, kept so the two can be timed against each other.
    tactile_encoder_type = "cnn"
    # Freeze the pretrained ViT trunk and the grounding query; the spatial
    # readout and bottleneck stay trainable, mirroring the resnet baseline.
    # Justified by the run to 72.9k steps, where inside stayed at 0.94-0.96
    # throughout (no drift to correct) while td sat at ~1e-4, i.e. TD was
    # barely shaping the trunk anyway. Freezing also switches the RL-time CGL
    # loss off, which matters now that pretraining grounds the hand as well:
    # the RL target has no hand mask and would push that attention back off.
    freeze_encoder = True
    observation_horizon = 1
    wandb_project = "tennis_ball_pick-and-place-gazemask-8-28-0"
    setup_mode = "single-arm-fixed-gripper"
    pick_reward_threshold = 0.8
    place_reward_threshold = 0.8
    manual_failure_penalty = 0.0
    mask_pick_place_phase_control = True
    # Values from the 2026-7-21 run that completed pick-and-place. The 0.9/0.4
    # pair that was here came from the ViT commit and never ran with the mask
    # feature head enabled.
    mask_feature_gate_alpha = 1.0
    mask_feature_min_gate = 0.1

    def get_image_keys(
        self,
        enable_tactile=True,
        use_gaze_target_mask=True,
    ):
        image_keys = ["front_camera"]
        if enable_tactile:
            image_keys.append("tactile_data")
        if use_gaze_target_mask:
            image_keys.extend(
                [
                    "front_camera_mask",
                    "front_camera_mask1",
                    "front_camera_mask2",
                ]
            )
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
        mask_selection_mode="pick_classifier",
        encoder_type=None,
        pick_classifier_checkpoint_path=_reward_classifier_ckpt("classifier_ckpt_ball_pick"),
        pick_classifier_threshold=0.8,
        gaze_target_mask_dilation=0,
        frame_save_path=None,
    ):
        env_config = EnvConfig()
        env_config.ENABLE_TACTILE = enable_tactile
        env_config.ENABLE_DATA_RECORDING = bool(record_data)
        env_config.ENABLE_GAZE_COLLECTION = bool(record_gaze)
        if frame_save_path is not None:
            env_config.GAZE_FRAME_SAVE_PATH = frame_save_path

        # vit-gaze carries no mask observation and no phase one-hot. Its
        # grounding query was pretrained on the operator's recorded gaze, so
        # at RL time there is nothing to predict a mask for: no SAM, no mask
        # predictor, no gaze predictor, no pick classifier. The observation is
        # front_camera + tactile_data + an 8-wide state.
        active_encoder_type = encoder_type or self.encoder_type
        use_gaze_target_mask = active_encoder_type != "vit-gaze"
        self.image_keys = self.get_image_keys(
            enable_tactile,
            use_gaze_target_mask=use_gaze_target_mask,
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
                env = SpacemouseIntervention(
                    env, gain=getattr(env_config, "SPACEMOUSE_GAIN", 1.0)
                )
        # Sits outside the intervention wrappers on purpose: they keep speaking
        # the 7-dim language (so the SpaceMouse rpy interface survives) and this
        # is where the action space narrows to what the robot actually obeys.
        if getattr(env_config, "MASK_RPY_ACTION", False):
            env = ArmActionSubspaceWrapper(env)
        # env = RelativeFrame(env)
        # env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        if use_gaze_target_mask:
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
        else:
            # Skipped entirely rather than run with use_gaze_target_mask=False:
            # everything this wrapper does -- loading the gaze and mask
            # predictors, adding the three mask images, appending the phase
            # one-hot -- is exactly what vit-gaze removes. Not applying it is
            # also what keeps the state 8 wide instead of 10.
            print("[vit-gaze] GazeDerivedObservationWrapper skipped: "
                  "no mask predictor, no gaze predictor, no pick classifier, "
                  "no phase one-hot")
        env = ChunkingWrapper(
            env,
            obs_horizon=self.observation_horizon,
            act_exec_horizon=None,
        )
        if classifier:
            place_only_reward = active_encoder_type == "vit-gaze"
            sample = env.observation_space.sample()
            # Reward classifiers were trained with horizon=1; keep only the
            # latest observation for classifier inference as well.
            classifier_sample = {
                key: np.asarray(value)[-1:]
                for key, value in sample.items()
            }
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
                if not place_only_reward:
                    pick_reward_classifier = load_classifier_func(
                        key=jax.random.PRNGKey(0),
                        sample=classifier_sample,
                        image_keys=self.classifier_keys,
                        checkpoint_path=_reward_classifier_ckpt("classifier_ckpt_ball_pick"),
                    )
                place_reward_classifier = load_classifier_func(
                    key=jax.random.PRNGKey(0),
                    sample=classifier_sample,
                    image_keys=self.classifier_keys,
                    checkpoint_path=_reward_classifier_ckpt("classifier_ckpt_ball_place"),
                )
            else:
                if not place_only_reward:
                    pick_reward_classifier = load_classifier_func(
                        key=jax.random.PRNGKey(0),
                        sample=classifier_sample,
                        image_keys=self.classifier_keys,
                        checkpoint_path=_reward_classifier_ckpt("classifier_ckpt_ball_pick_no_tactile"),
                    )
                place_reward_classifier = load_classifier_func(
                    key=jax.random.PRNGKey(0),
                    sample=classifier_sample,
                    image_keys=self.classifier_keys,
                    checkpoint_path=_reward_classifier_ckpt("classifier_ckpt_ball_place_no_tactile"),
                )
            
            # input("debug")
            def reward_func(obs, is_pick=True):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                classifier_obs = {
                    key: np.asarray(value)[-1:]
                    for key, value in obs.items()
                }
                if not place_only_reward and is_pick:
                    pick_prob = sigmoid(
                        pick_reward_classifier(classifier_obs)
                    ).item()
                    print("sigmoid(pick_reward_classifier(obs)) = ", pick_prob)
                    pick_success = pick_prob > self.pick_reward_threshold
                    return {
                        "reward": 0,
                        "pick_success": pick_success,
                        "place_success": False,
                        "pick_prob": pick_prob,
                    }
                place_prob = sigmoid(place_reward_classifier(classifier_obs)).item()
                print("sigmoid(place_reward_classifier(obs)) = ", place_prob)
                place_success = place_prob > self.place_reward_threshold
                return {
                    "reward": int(place_success),
                    "pick_success": False,
                    "place_success": place_success,
                    "place_prob": place_prob,
                }

            env = MultiCameraBinaryRewardClassifierWrapper(
                env,
                reward_func,
                start_in_pick_phase=not place_only_reward,
            )
            if place_only_reward:
                print("[vit-gaze] reward classifier: place-only (pick classifier not loaded)")
        env = GripperPenaltyWrapper(env, exp_name=env_config.EXP_NAME, penalty=-0.02)
        return env
