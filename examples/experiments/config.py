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

    # Visual encoder backbone for non-mask image keys.
    #   "resnet": train ResNet10 from scratch.
    #   "resnet-pretrained": frozen ImageNet ResNet10 + trainable readout head.
    #   "vit" / "vit-small": trainable lightweight ViT-style image encoder.
    #   "vit-grounded": ViT trunk with a spatial-learned-embeddings readout
    #     and a grounding query supervised by the CGL mask loss; the trunk is
    #     fine-tuned by TD + CGL together (never frozen).
    # Mask images, when present, go through the gaze agent's small mask CNN,
    # not through this visual encoder.
    encoder_type: str = "resnet-pretrained"
    # Only read by "vit-grounded"; empty means train the trunk from scratch.
    encoder_checkpoint_path: str | None = None
    demo_path: str = None
    checkpoint_period: int = 0
    buffer_period: int = 0

    eval_checkpoint_step: int = 0
    eval_n_trajs: int = 5

    # Terminal reward assigned only when the actor operator presses "2" to
    # mark the current episode as an unrecoverable manual failure. A value of
    # 0 disables the extra penalty; max-length timeouts do not use this value.
    manual_failure_penalty: float = 0.0

    image_keys: List[str] = None
    classifier_keys: List[str] = None
    proprio_keys: List[str] = None
    state_weights: List[float] | None = None

    # "single-arm-learned-gripper", "dual-arm-learned-gripper" for with learned gripper, 
    # "single-arm-fixed-gripper", "dual-arm-fixed-gripper" for without learned gripper (i.e. pregrasped)
    setup_mode: str = "single-arm-fixed-gripper"

    @abstractmethod
    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        raise NotImplementedError
    
    @abstractmethod
    def process_demos(self, demo):
        raise NotImplementedError
    
