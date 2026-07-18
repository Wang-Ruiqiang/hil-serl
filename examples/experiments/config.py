from abc import abstractmethod
from typing import List

class DefaultTrainingConfig:
    """Default training configuration. """

    agent: str = "drq"
    max_traj_length: int = 100
    batch_size: int = 64
    cta_ratio: int = 3
    discount: float = 0.97

    max_steps: int = 250000
    replay_buffer_capacity: int = 150000

    random_steps: int = 100
    training_starts: int = 100
    steps_per_update: int = 50

    log_period: int = 1000
    eval_period: int = 2000

    # "resnet" for ResNet10 from scratch and "resnet-pretrained" for frozen ResNet10 with pretrained weights
    encoder_type: str = "resnet-pretrained"
    demo_path: str = None
    checkpoint_period: int = 0
    buffer_period: int = 0

    eval_checkpoint_step: int = 0
    eval_n_trajs: int = 5

    image_keys: List[str] = None
    classifier_keys: List[str] = None
    proprio_keys: List[str] = None
    state_weights: List[float] | None = None

    # Gaze/mask-conditioned visual feature settings.
    # These only affect the gaze/mask SAC agent when observations include
    # front_camera_mask. Plain vision+tactile SAC ignores these values.

    # Enable a small trainable mask feature head on top of front_camera
    # features. Its output is used for fused-feature visualization and
    # mask_grounding_loss.
    use_mask_feature_head: bool = True

    # Strength of the trainable mask feature gate. The learned gate can both
    # amplify selected regions and suppress less useful regions.
    mask_feature_gate_alpha: float = 1.0

    # Lower bound for the trainable gate multiplier. Smaller values suppress
    # non-selected regions harder; larger values preserve more global context.
    mask_feature_min_gate: float = 0.1

    # Hidden channel count inside the trainable mask feature head. This is a model
    # capacity setting; most tasks should leave it at the default.
    mask_feature_hidden_dim: int = 128

    # Encode the selected binary mask with a small CNN and concatenate that
    # vector as an additional policy/critic modality.
    use_mask_encoder: bool = True
    mask_encoder_latent_dim: int = 64

    # Weight for the auxiliary loss that pulls the trainable mask feature map into
    # front_camera_mask. 0 disables this auxiliary supervision.
    mask_grounding_weight: float = 0.0

    # Optional step-based decay for mask_grounding_weight. This lets the mask
    # strongly shape mask features early, then reduce its influence after the
    # mask feature head has learned a stable target region. Set step <= 0 to disable.
    mask_grounding_decay_step: int = 0
    mask_grounding_decay_weight: float = 0.0

    # Threshold used after resizing front_camera_mask to the critic attention
    # resolution when computing mask_grounding_loss.
    mask_grounding_threshold: float = 0.05

    # Minimum high-resolution mask occupancy required to mark one low-resolution
    # attention cell as positive. For example, 0.01 means at least 1% of the
    # source pixels in that attention cell must belong to the mask.
    mask_grounding_cell_threshold: float = 0.01

    # "single-arm-learned-gripper", "dual-arm-learned-gripper" for with learned gripper, 
    # "single-arm-fixed-gripper", "dual-arm-fixed-gripper" for without learned gripper (i.e. pregrasped)
    setup_mode: str = "single-arm-fixed-gripper"

    @abstractmethod
    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        raise NotImplementedError
    
    @abstractmethod
    def process_demos(self, demo):
        raise NotImplementedError
    
