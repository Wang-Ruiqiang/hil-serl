from functools import partial
from typing import Iterable, Optional

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp

from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.common.typing import Batch, Params, PRNGKey
from serl_launcher.networks.actor_critic_nets import Critic, Policy, ensemblize
from serl_launcher.networks.gaze_attention import GazeAttentionCritic
from serl_launcher.networks.lagrange import GeqLagrangeMultiplier
from serl_launcher.networks.mlp import MLP


class SACAgentHybridSingleArmGaze(SACAgentHybridSingleArm):
    """Hybrid SAC agent with critic-side CGL attention supervision.

    This class keeps the base SAC update structure intact. The only behavioral
    change is inside critic_loss_fn, where an optional gaze auxiliary loss
    encourages front-camera critic attention to cover the gaze heatmap.
    """

    def forward_gaze_attention(
        self,
        observations,
        actions: jax.Array,
        rng: PRNGKey,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ):
        if train:
            assert rng is not None, "Must specify rng when training"
        _, attention_map = self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            actions,
            name="critic",
            rngs={"dropout": rng} if train else {},
            train=train,
            return_attention=True,
        )
        return attention_map

    def _gaze_heatmap_per_sample(self, batch: Batch):
        key = self.config["gaze_heatmap_key"]
        if key in batch:
            return batch[key], True
        if key in batch["observations"]:
            return batch["observations"][key], True
        height, width = self.config["gaze_heatmap_size"]
        return jnp.zeros((batch["rewards"].shape[0], height, width)), False

    def _cgl_coverage_kl(self, gaze_heatmap, attention_map):
        batch_size, gaze_h, gaze_w = gaze_heatmap.shape
        attention_map = jax.image.resize(
            attention_map[..., None],
            (batch_size, gaze_h, gaze_w, 1),
            method="bilinear",
        )[..., 0]
        attention_probs = jax.nn.softmax(
            attention_map.reshape(batch_size, -1),
            axis=-1,
        )

        gaze_heatmap = jnp.maximum(gaze_heatmap, 0.0)
        gaze_max = jnp.max(gaze_heatmap.reshape(batch_size, -1), axis=-1, keepdims=True)
        gaze_flat = gaze_heatmap.reshape(batch_size, -1)
        gaze_flat = jnp.where(
            gaze_flat >= self.config["gaze_cgl_threshold"] * (gaze_max + 1e-8),
            gaze_flat,
            0.0,
        )
        gaze_sum = jnp.sum(gaze_flat, axis=-1, keepdims=True)
        gaze_probs = gaze_flat / (gaze_sum + 1e-8)
        kl = jnp.sum(
            gaze_probs * (jnp.log(gaze_probs + 1e-8) - jnp.log(attention_probs + 1e-8)),
            axis=-1,
        )
        has_gaze = jnp.squeeze(gaze_sum > 1e-8, axis=-1)
        return jnp.where(has_gaze, kl, 0.0), has_gaze.astype(jnp.float32)

    def critic_loss_fn(self, batch, params: Params, rng: PRNGKey):
        batch_size = batch["rewards"].shape[0]
        actions = batch["actions"][..., :-1]

        rng, next_action_sample_key = jax.random.split(rng)
        next_actions, next_actions_log_probs = self._compute_next_actions(
            batch, next_action_sample_key
        )

        target_next_qs = self.forward_target_critic(
            batch["next_observations"],
            next_actions[..., :-1],
            rng=rng,
        )

        if self.config["critic_subsample_size"] is not None:
            rng, subsample_key = jax.random.split(rng)
            subsample_idcs = jax.random.randint(
                subsample_key,
                (self.config["critic_subsample_size"],),
                0,
                self.config["critic_ensemble_size"],
            )
            target_next_qs = target_next_qs[subsample_idcs]

        target_next_min_q = target_next_qs.min(axis=0)
        chex.assert_shape(target_next_min_q, (batch_size,))

        target_q = (
            batch["rewards"]
            + batch["robot_arm_penalty"]
            + self.config["discount"] * batch["masks"] * target_next_min_q
        )
        chex.assert_shape(target_q, (batch_size,))

        if self.config["backup_entropy"]:
            temperature = self.forward_temperature()
            target_q = target_q - temperature * next_actions_log_probs

        predicted_qs = self.forward_critic(
            batch["observations"], actions, rng=rng, grad_params=params
        )

        chex.assert_shape(
            predicted_qs, (self.config["critic_ensemble_size"], batch_size)
        )
        target_qs = target_q[None].repeat(self.config["critic_ensemble_size"], axis=0)
        chex.assert_equal_shape([predicted_qs, target_qs])
        td_loss = jnp.mean((predicted_qs - target_qs) ** 2)

        gaze_heatmap, has_gaze_heatmap = self._gaze_heatmap_per_sample(batch)
        gaze_heatmap = jnp.reshape(
            gaze_heatmap,
            (batch_size, *self.config["gaze_heatmap_size"]),
        )
        attention_map = self.forward_gaze_attention(
            batch["observations"],
            actions,
            rng=rng,
            grad_params=params,
        )

        gaze_weight = self.config["gaze_regularization_weight"]
        cgl_loss_per_sample, valid_gaze = self._cgl_coverage_kl(gaze_heatmap, attention_map)
        valid_gaze_count = jnp.maximum(jnp.sum(valid_gaze), 1.0)
        has_active_gaze_aux = has_gaze_heatmap and gaze_weight > 0.0
        gaze_aux_loss = (
            jnp.sum(cgl_loss_per_sample * valid_gaze)
            / valid_gaze_count
        )
        if not has_active_gaze_aux:
            gaze_aux_loss = jnp.asarray(0.0, dtype=td_loss.dtype)

        weighted_gaze_aux_loss = gaze_weight * gaze_aux_loss
        gaze_to_td_ratio = weighted_gaze_aux_loss / (td_loss + 1e-8)
        critic_loss = td_loss + weighted_gaze_aux_loss

        info = {
            "critic_loss": critic_loss,
            "critic_td_loss": td_loss,
            "predicted_qs": jnp.mean(predicted_qs),
            "target_qs": jnp.mean(target_qs),
            "rewards": batch["rewards"].mean(),
            "gaze_aux_available": jnp.asarray(float(has_active_gaze_aux)),
            "gaze_aux_loss": gaze_aux_loss,
            "gaze_weight": jnp.asarray(gaze_weight, dtype=td_loss.dtype),
            "weighted_gaze_aux_loss": weighted_gaze_aux_loss,
            "gaze_to_td_ratio": gaze_to_td_ratio,
            "gaze_cgl_kl": jnp.sum(cgl_loss_per_sample * valid_gaze) / valid_gaze_count,
            "gaze_valid_fraction": jnp.mean(valid_gaze),
        }

        return critic_loss, info

    @classmethod
    def create_pixels(
        cls,
        rng: PRNGKey,
        observations,
        actions: jnp.ndarray,
        encoder_type: str = "resnet-pretrained",
        use_proprio: bool = False,
        critic_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        grasp_critic_network_kwargs: dict = {
            "hidden_dims": [128, 128],
        },
        policy_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        policy_kwargs: dict = {
            "tanh_squash_distribution": True,
            "std_parameterization": "uniform",
        },
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        temperature_init: float = 1.0,
        image_keys: Iterable[str] = ("image",),
        augmentation_function: Optional[callable] = None,
        gaze_regularization_weight: float = 0.0,
        gaze_heatmap_key: str = "gaze_heatmap",
        gaze_heatmap_size: tuple = (128, 128),
        gaze_cgl_threshold: float = 1e-4,
        gaze_attention_image_key: str = "front_camera",
        **kwargs,
    ):
        policy_network_kwargs["activate_final"] = True
        critic_network_kwargs["activate_final"] = True

        if encoder_type == "resnet":
            from serl_launcher.vision.resnet_v1 import resnetv1_configs

            encoders = {
                image_key: resnetv1_configs["resnetv1-10"](
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                    bottleneck_dim=256,
                    name=f"encoder_{image_key}",
                )
                for image_key in image_keys
            }
        elif encoder_type == "resnet-pretrained":
            from serl_launcher.vision.resnet_v1 import (
                PreTrainedResNetEncoder,
                resnetv1_configs,
            )

            pretrained_encoder = resnetv1_configs["resnetv1-10-frozen"](
                pre_pooling=True,
                name="pretrained_encoder",
            )
            encoders = {
                image_key: PreTrainedResNetEncoder(
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                    bottleneck_dim=256,
                    pretrained_encoder=pretrained_encoder,
                    name=f"encoder_{image_key}",
                )
                for image_key in image_keys
            }
        else:
            raise NotImplementedError(f"Unknown encoder type: {encoder_type}")

        from serl_launcher.common.encoding import EncodingWrapper

        encoder_def = EncodingWrapper(
            encoder=encoders,
            use_proprio=use_proprio,
            enable_stacking=True,
            image_keys=image_keys,
            # TODO: If gaze is collected on a different camera in a future task,
            # pass that camera key through the launcher/config instead of front_camera.
            attention_image_key=gaze_attention_image_key,
        )

        encoders = {
            "critic": encoder_def,
            "actor": encoder_def,
            "grasp_critic": encoder_def,
        }

        critic_backbone = partial(MLP, **critic_network_kwargs)
        critic_backbone = ensemblize(critic_backbone, critic_ensemble_size)(
            name="critic_ensemble"
        )
        critic_def = partial(
            GazeAttentionCritic,
            encoder=encoders["critic"],
            network=critic_backbone,
        )(name="critic")

        grasp_critic_backbone = MLP(**grasp_critic_network_kwargs)
        grasp_critic_def = partial(
            Critic, encoder=encoders["grasp_critic"], network=grasp_critic_backbone
        )(name="grasp_critic")

        policy_def = Policy(
            encoder=encoders["actor"],
            network=MLP(**policy_network_kwargs),
            action_dim=actions.shape[-1],
            **policy_kwargs,
            name="actor",
        )

        temperature_def = GeqLagrangeMultiplier(
            init_value=temperature_init,
            constraint_shape=(),
            constraint_type="geq",
            name="temperature",
        )

        agent = cls.create(
            rng,
            observations,
            actions,
            actor_def=policy_def,
            critic_def=critic_def,
            grasp_critic_def=grasp_critic_def,
            temperature_def=temperature_def,
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            image_keys=image_keys,
            augmentation_function=augmentation_function,
            gaze_regularization_weight=gaze_regularization_weight,
            gaze_heatmap_key=gaze_heatmap_key,
            gaze_heatmap_size=gaze_heatmap_size,
            gaze_cgl_threshold=gaze_cgl_threshold,
            **kwargs,
        )

        if "pretrained" in encoder_type:
            from serl_launcher.utils.train_utils import load_resnet10_params

            agent = load_resnet10_params(agent, image_keys)

        return agent
