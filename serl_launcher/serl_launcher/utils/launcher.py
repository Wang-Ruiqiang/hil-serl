# !/usr/bin/env python3

import jax
from jax import nn
import jax.numpy as jnp
from typing import Dict

from agentlace.trainer import TrainerConfig

from serl_launcher.common.typing import Batch, PRNGKey
from serl_launcher.common.wandb import WandBLogger
from serl_launcher.agents.continuous.bc import BCAgent
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_single_gaze import SACAgentHybridSingleArmGaze
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from serl_launcher.vision.data_augmentations import (
    batched_random_crop,
    batched_temporal_random_crop,
)

##############################################################################


def make_bc_agent(
    seed, 
    sample_obs, 
    sample_action, 
    image_keys=("image",), 
    encoder_type="resnet-pretrained",
):
    return BCAgent.create(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [512, 512, 512],
            "dropout_rate": 0.25,
        },
        policy_kwargs={
            "tanh_squash_distribution": False,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        use_proprio=True,
        encoder_type=encoder_type,
        image_keys=image_keys,
        augmentation_function=make_batch_augmentation_func(image_keys),
    )


def make_sac_pixel_agent(
    seed,
    sample_obs,
    sample_action,
    image_keys=("image",),
    encoder_type="resnet-pretrained",
    reward_bias=0.0,
    target_entropy=None,
    discount=0.97,
    state_weights=None,
):
    agent = SACAgent.create_pixels(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        encoder_type=encoder_type,
        use_proprio=True,
        state_weights=state_weights,
        image_keys=image_keys,
        policy_kwargs={
            "tanh_squash_distribution": True,
            # "tanh_squash_distribution": False,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 0.5,
            # "std_max": 1,
        },
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
            # "hidden_dims": [512, 512, 512],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
            # "hidden_dims": [512, 512, 512],
        },
        temperature_init=1e-2,
        discount=discount,
        backup_entropy=False,
        critic_ensemble_size=2,
        critic_subsample_size=None,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        augmentation_function=make_batch_augmentation_func(image_keys),
    )
    return agent


def make_sac_pixel_agent_hybrid_single_arm(
    seed,
    sample_obs,
    sample_action,
    image_keys=("image",),
    encoder_type="resnet-pretrained",
    reward_bias=0.0,
    target_entropy=None,
    discount=0.97,
):
    agent = SACAgentHybridSingleArm.create_pixels(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        grasp_critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=discount,
        backup_entropy=False,
        critic_ensemble_size=2,
        critic_subsample_size=None,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        augmentation_function=make_batch_augmentation_func(image_keys),
    )
    return agent


def make_gaze_sac_pixel_agent_hybrid_single_arm(
    seed,
    sample_obs,
    sample_action,
    image_keys=("image",),
    encoder_type="resnet-pretrained",
    encoder_checkpoint_path=None,
    freeze_encoder=False,
    tactile_encoder_type="cnn",
    vit_image_size=(224, 224),
    vit_hidden_dim=192,
    vit_num_layers=4,
    vit_num_heads=6,
    reward_bias=0.0,
    target_entropy=None,
    discount=0.97,
    gaze_regularization_weight=0.0,
    gaze_heatmap_key="gaze_heatmap",
    gaze_heatmap_size=(128, 128),
    gaze_valid_threshold=1e-8,
    gaze_region_radius=1,
    gaze_attention_image_key="front_camera",
    mask_feature_gate_alpha=1.0,
    mask_feature_min_gate=0.1,
    mask_pick_place_phase_control=False,
    return_raw_attention=False,
    return_mask_encoder_attention=False,
    return_feature_debug=False,
):
    augmentation_image_keys = image_keys
    latest_only_augmentation_keys = ()
    if encoder_type == "vit-gaze":
        # No mask keys: vit-gaze has no mask observation and no CGL target, so
        # front_camera and tactile_data are the only images to crop. The shared
        # rng that keeps the RGB and its masks aligned in the branch below is
        # irrelevant here for the same reason.
        augmentation_image_keys = tuple(
            key for key in image_keys if key in ("front_camera", "tactile_data")
        )
        latest_only_augmentation_keys = ("tactile_data",)
    elif encoder_type == "vit-grounded":
        # front_camera_mask feeds the mask CNN branch and front_camera_mask1 is
        # the CGL grounding target, so both must be cropped with the RGB --
        # data_augmentation_fn reuses one rng across keys, which is what keeps
        # the crops aligned. Leaving mask1 out shifts the supervision target by
        # up to 4px against the image, roughly half a token cell.
        # front_camera_mask2 is genuinely unused by this pipeline (no feature
        # suppression), so it stays out.
        augmentation_image_keys = tuple(
            key
            for key in image_keys
            if key in (
                "front_camera",
                "front_camera_mask",
                "front_camera_mask1",
                "tactile_data",
            )
        )
        latest_only_augmentation_keys = ("tactile_data",)
    agent = SACAgentHybridSingleArmGaze.create_pixels(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        encoder_type=encoder_type,
        encoder_checkpoint_path=encoder_checkpoint_path,
        freeze_encoder=freeze_encoder,
        tactile_encoder_type=tactile_encoder_type,
        vit_image_size=vit_image_size,
        vit_hidden_dim=vit_hidden_dim,
        vit_num_layers=vit_num_layers,
        vit_num_heads=vit_num_heads,
        use_proprio=True,
        image_keys=image_keys,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        grasp_critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=discount,
        backup_entropy=False,
        critic_ensemble_size=2,
        critic_subsample_size=None,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        augmentation_function=make_batch_augmentation_func(
            augmentation_image_keys,
            temporal_consistent=False,
            latest_only_keys=latest_only_augmentation_keys,
        ),
        gaze_regularization_weight=gaze_regularization_weight,
        gaze_heatmap_key=gaze_heatmap_key,
        gaze_heatmap_size=gaze_heatmap_size,
        gaze_valid_threshold=gaze_valid_threshold,
        gaze_region_radius=gaze_region_radius,
        gaze_attention_image_key=gaze_attention_image_key,
        mask_feature_gate_alpha=mask_feature_gate_alpha,
        mask_feature_min_gate=mask_feature_min_gate,
        mask_pick_place_phase_control=mask_pick_place_phase_control,
        return_raw_attention=return_raw_attention,
        return_mask_encoder_attention=return_mask_encoder_attention,
        return_feature_debug=return_feature_debug,
    )
    return agent


def make_sac_pixel_agent_hybrid_dual_arm(
    seed,
    sample_obs,
    sample_action,
    image_keys=("image",),
    encoder_type="resnet-pretrained",
    reward_bias=0.0,
    target_entropy=None,
    discount=0.97,
):
    agent = SACAgentHybridDualArm.create_pixels(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        grasp_critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=discount,
        backup_entropy=False,
        critic_ensemble_size=2,
        critic_subsample_size=None,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        augmentation_function=make_batch_augmentation_func(image_keys),
    )
    return agent


def linear_schedule(step):
    init_value = 10.0
    end_value = 50.0
    decay_steps = 15_000


    linear_step = jnp.minimum(step, decay_steps)
    decayed_value = init_value + (end_value - init_value) * (linear_step / decay_steps)
    return decayed_value
    
def make_batch_augmentation_func(
    image_keys,
    *,
    temporal_consistent=False,
    latest_only_keys=(),
) -> callable:

    latest_only_keys = frozenset(latest_only_keys)

    def data_augmentation_fn(rng, observations):
        for pixel_key in image_keys:
            image = observations[pixel_key]
            if pixel_key in latest_only_keys:
                if image.ndim != 5:
                    raise ValueError(
                        f"Expected BTHWC for latest-only augmentation, got "
                        f"{pixel_key}={image.shape}."
                    )
                latest = batched_random_crop(
                    image[:, -1],
                    rng,
                    padding=4,
                    num_batch_dims=1,
                )
                augmented = image.at[:, -1].set(latest)
            elif temporal_consistent:
                augmented = batched_temporal_random_crop(image, rng, padding=4)
            else:
                augmented = batched_random_crop(
                    image,
                    rng,
                    padding=4,
                    num_batch_dims=2,
                )
            observations = observations.copy(
                add_or_replace={
                    pixel_key: augmented
                }
            )
        return observations
    
    def augment_batch(batch: Batch, rng: PRNGKey) -> Batch:
        rng, obs_rng, next_obs_rng = jax.random.split(rng, 3)
        obs = data_augmentation_fn(obs_rng, batch["observations"])
        next_obs = data_augmentation_fn(next_obs_rng, batch["next_observations"])
        batch = batch.copy(
            add_or_replace={
                "observations": obs,
                "next_observations": next_obs,
            }
        )
        return batch
    
    return augment_batch


def make_trainer_config(port_number: int = 5588, broadcast_port: int = 5589):
    return TrainerConfig(
        port_number=port_number,
        broadcast_port=broadcast_port,
        request_types=["send-stats"],
    )


def make_wandb_logger(
    project: str = "hil-serl",
    description: str = "serl_launcher",
    debug: bool = False,
):
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": project,
            "exp_descriptor": description,
            "tag": description,
        }
    )
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant={},
        debug=debug,
    )
    return wandb_logger
