from functools import partial
from typing import FrozenSet, Iterable, Optional, Tuple

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
    """Hybrid SAC agent with mask-conditioned features and CGL grounding loss.

    This class keeps the base SAC update structure intact. The only behavioral
    change is inside critic_loss_fn, where optional auxiliary losses apply the
    CGL KL term to the front-camera mask feature head.
    """

    _VISUAL_AUX_PARAM_MARKERS = (
        "mask_feature_head",
    )

    @staticmethod
    def _path_keys(path) -> Tuple[str, ...]:
        return tuple(str(getattr(entry, "key", entry)) for entry in path)

    @classmethod
    def _is_visual_aux_param_path(cls, path) -> bool:
        keys = cls._path_keys(path)
        if any(key in ("actor", "grasp_critic", "temperature") for key in keys):
            return False
        return any(
            marker in key
            for key in keys
            for marker in cls._VISUAL_AUX_PARAM_MARKERS
        )

    @classmethod
    def _stop_visual_aux_params(cls, params: Params) -> Params:
        return jax.tree_util.tree_map_with_path(
            lambda path, value: (
                jax.lax.stop_gradient(value)
                if cls._is_visual_aux_param_path(path)
                else value
            ),
            params,
        )

    @classmethod
    def _stop_non_visual_aux_params(cls, params: Params) -> Params:
        return jax.tree_util.tree_map_with_path(
            lambda path, value: (
                value
                if cls._is_visual_aux_param_path(path)
                else jax.lax.stop_gradient(value)
            ),
            params,
        )

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

    def forward_cgl_attention(
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
            return_cgl_attention=True,
        )
        return attention_map

    def _gaze_heatmap_per_sample(self, batch: Batch):
        key = self.config["gaze_heatmap_key"]
        if key in batch:
            return batch[key], True
        if key in batch["observations"]:
            gaze_heatmap = batch["observations"][key]
            if gaze_heatmap.ndim == 5:
                gaze_heatmap = gaze_heatmap[:, -1, ..., 0]
            elif gaze_heatmap.ndim == 4:
                gaze_heatmap = gaze_heatmap[..., 0]
            gaze_heatmap = gaze_heatmap.astype(jnp.float32)
            gaze_heatmap = jnp.where(
                jnp.max(gaze_heatmap) > 1.0,
                gaze_heatmap / 255.0,
                gaze_heatmap,
            )
            return gaze_heatmap, True
        height, width = self.config["gaze_heatmap_size"]
        return jnp.zeros((batch["rewards"].shape[0], height, width)), False

    def _gaze_region_tracking_loss(self, gaze_heatmap, attention_map):
        batch_size, gaze_h, gaze_w = gaze_heatmap.shape
        _, attention_h, attention_w = attention_map.shape
        attention_probs = jax.nn.softmax(
            attention_map.reshape(batch_size, -1),
            axis=-1,
        )

        gaze_heatmap = jnp.maximum(gaze_heatmap, 0.0)
        gaze_flat = gaze_heatmap.reshape(batch_size, -1)
        gaze_max = jnp.max(gaze_flat, axis=-1)
        gaze_peak = jnp.argmax(gaze_flat, axis=-1)
        gaze_y = gaze_peak // gaze_w
        gaze_x = gaze_peak % gaze_w

        scale_y = 0.0 if gaze_h <= 1 else (attention_h - 1) / float(gaze_h - 1)
        scale_x = 0.0 if gaze_w <= 1 else (attention_w - 1) / float(gaze_w - 1)
        attention_y = jnp.rint(gaze_y.astype(jnp.float32) * scale_y).astype(jnp.int32)
        attention_x = jnp.rint(gaze_x.astype(jnp.float32) * scale_x).astype(jnp.int32)

        radius = self.config["gaze_region_radius"]
        grid_y = jnp.arange(attention_h)[None, :, None]
        grid_x = jnp.arange(attention_w)[None, None, :]
        region_mask = (
            (jnp.abs(grid_y - attention_y[:, None, None]) <= radius)
            & (jnp.abs(grid_x - attention_x[:, None, None]) <= radius)
        )
        region_mask = region_mask.reshape(batch_size, -1).astype(attention_probs.dtype)
        region_coverage = jnp.sum(attention_probs * region_mask, axis=-1)
        region_loss = -jnp.log(region_coverage + 1e-8)

        has_gaze = gaze_max > self.config["gaze_valid_threshold"]
        return (
            jnp.where(has_gaze, region_loss, 0.0),
            jnp.where(has_gaze, region_coverage, 0.0),
            has_gaze.astype(jnp.float32),
        )

    def _mask_to_attention_shape(self, mask, attention_map):
        mask = jnp.asarray(mask, dtype=jnp.float32)
        if mask.ndim == 5:
            mask = jnp.max(mask, axis=(1, 4))
        elif mask.ndim == 4:
            mask = jnp.max(mask, axis=-1)
        elif mask.ndim != 3:
            raise ValueError(f"Unsupported mask shape for grounding loss: {mask.shape}")

        batch_size, attention_h, attention_w = attention_map.shape
        mask = jnp.where(jnp.max(mask) > 1.0, mask / 255.0, mask)
        mask = mask > self.config["mask_grounding_threshold"]

        source_h, source_w = mask.shape[-2:]
        block_h = (source_h + attention_h - 1) // attention_h
        block_w = (source_w + attention_w - 1) // attention_w
        padded_h = attention_h * block_h
        padded_w = attention_w * block_w
        mask = jnp.pad(
            mask,
            (
                (0, 0),
                (0, padded_h - source_h),
                (0, padded_w - source_w),
            ),
            mode="constant",
            constant_values=False,
        )
        mask = mask.reshape(batch_size, attention_h, block_h, attention_w, block_w)
        occupancy = jnp.mean(mask.astype(attention_map.dtype), axis=(2, 4))
        mask = (
            occupancy > self.config["mask_grounding_cell_threshold"]
        ).astype(attention_map.dtype)
        return mask

    def _pick_phase_gate_per_sample(self, observations, batch_size: int, dtype):
        if not self.config.get("mask_pick_place_phase_control", False):
            return jnp.ones((batch_size,), dtype=dtype)
        state = observations.get("state")
        if state is None:
            return jnp.ones((batch_size,), dtype=dtype)
        state = jnp.asarray(state)
        if state.shape[-1] < 3:
            return jnp.ones((batch_size,), dtype=dtype)
        if state.ndim == 3:
            phase = state[:, -1, -3:]
        else:
            phase = state[:, -3:]
        return phase[..., 0].astype(dtype)

    def _mask_grounding_loss(self, observations, attention_map):
        key = (
            "front_camera_mask1"
            if "front_camera_mask1" in observations
            else "front_camera_mask"
        )
        batch_size = attention_map.shape[0]
        if key not in observations:
            zeros = jnp.zeros((batch_size,), dtype=attention_map.dtype)
            return zeros, zeros, zeros, zeros, zeros

        mask = self._mask_to_attention_shape(observations[key], attention_map)
        mask = mask.astype(attention_map.dtype)
        logits_flat = attention_map.reshape(attention_map.shape[0], -1)
        mask_flat = mask.reshape(mask.shape[0], -1)

        attention_probs = jax.nn.softmax(
            logits_flat,
            axis=-1,
        )
        mask_sum = jnp.sum(mask_flat, axis=-1, keepdims=True)
        valid_mask = (mask_sum[:, 0] > 0).astype(attention_map.dtype)
        phase_gate = self._pick_phase_gate_per_sample(
            observations,
            batch_size,
            attention_map.dtype,
        )
        valid_mask = valid_mask * phase_gate
        gaze_distribution = mask_flat / jnp.maximum(mask_sum, 1e-8)
        cgl_loss = jnp.sum(
            gaze_distribution
            * (
                jnp.log(gaze_distribution + 1e-8)
                - jnp.log(attention_probs + 1e-8)
            ),
            axis=-1,
        )
        mask_mass = jnp.sum(attention_probs * mask_flat, axis=-1)
        outside_mass = jnp.sum(attention_probs * (1.0 - mask_flat), axis=-1)

        mask_feature_entropy = -jnp.sum(
            attention_probs * jnp.log(attention_probs + 1e-8),
            axis=-1,
        ) / jnp.log(attention_probs.shape[-1])
        return (
            jnp.where(valid_mask > 0, cgl_loss, 0.0),
            jnp.where(valid_mask > 0, mask_mass, 0.0),
            jnp.where(valid_mask > 0, outside_mass, 0.0),
            jnp.where(valid_mask > 0, mask_feature_entropy, 0.0),
            valid_mask,
        )

    def _critic_td_loss(self, batch, params: Params, rng: PRNGKey):
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
        return td_loss, predicted_qs, target_qs, actions

    def critic_loss_fn(self, batch, params: Params, rng: PRNGKey, train_step=None):
        td_params = self._stop_visual_aux_params(params)
        td_loss, predicted_qs, target_qs, _actions = self._critic_td_loss(
            batch, td_params, rng
        )

        info = {
            "critic_loss": td_loss,
            "critic_td_loss": td_loss,
            "predicted_qs": jnp.mean(predicted_qs),
            "target_qs": jnp.mean(target_qs),
            "rewards": batch["rewards"].mean(),
        }

        return td_loss, info

    def visual_aux_loss_fn(self, batch, params: Params, rng: PRNGKey, train_step=None):
        batch_size = batch["rewards"].shape[0]
        actions = batch["actions"][..., :-1]
        visual_params = self._stop_non_visual_aux_params(params)

        reference_td_loss, _predicted_qs, _target_qs, _actions = self._critic_td_loss(
            batch,
            jax.tree_util.tree_map(jax.lax.stop_gradient, params),
            rng,
        )
        attention_map = self.forward_gaze_attention(
            batch["observations"],
            actions,
            rng=rng,
            grad_params=visual_params,
        )
        cgl_attention_map = self.forward_cgl_attention(
            batch["observations"],
            actions,
            rng=rng,
            grad_params=visual_params,
        )

        gaze_weight = self.config["gaze_regularization_weight"]
        if gaze_weight > 0.0:
            gaze_heatmap, has_gaze_heatmap = self._gaze_heatmap_per_sample(batch)
            gaze_heatmap = jnp.reshape(
                gaze_heatmap,
                (batch_size, *self.config["gaze_heatmap_size"]),
            )
            gaze_loss_per_sample, gaze_region_coverage, valid_gaze = (
                self._gaze_region_tracking_loss(gaze_heatmap, attention_map)
            )
            has_active_gaze_aux = has_gaze_heatmap
        else:
            gaze_loss_per_sample = jnp.zeros(
                (batch_size,),
                dtype=reference_td_loss.dtype,
            )
            gaze_region_coverage = jnp.zeros(
                (batch_size,),
                dtype=reference_td_loss.dtype,
            )
            valid_gaze = jnp.zeros((batch_size,), dtype=reference_td_loss.dtype)
            has_active_gaze_aux = False
        valid_gaze_count = jnp.maximum(jnp.sum(valid_gaze), 1.0)
        gaze_aux_loss = (
            jnp.sum(gaze_loss_per_sample * valid_gaze)
            / valid_gaze_count
        )
        if not has_active_gaze_aux:
            gaze_aux_loss = jnp.asarray(0.0, dtype=reference_td_loss.dtype)

        weighted_gaze_aux_loss = gaze_weight * gaze_aux_loss
        gaze_to_td_ratio = weighted_gaze_aux_loss / (reference_td_loss + 1e-8)

        (
            mask_cgl_loss_per_sample,
            mask_grounding_coverage,
            mask_grounding_outside_mass,
            mask_feature_entropy,
            valid_mask,
        ) = self._mask_grounding_loss(batch["observations"], cgl_attention_map)
        valid_mask_count = jnp.maximum(jnp.sum(valid_mask), 1.0)
        mask_cgl_loss = (
            jnp.sum(mask_cgl_loss_per_sample * valid_mask) / valid_mask_count
        )
        mask_grounding_loss = mask_cgl_loss
        mask_grounding_aux_loss = mask_grounding_loss
        mask_grounding_to_td_ratio = mask_grounding_aux_loss / (reference_td_loss + 1e-8)

        visual_aux_loss = weighted_gaze_aux_loss + mask_grounding_aux_loss

        info = {
            "visual_aux_loss": visual_aux_loss,
            "visual_aux_reference_td_loss": reference_td_loss,
            "gaze_aux_available": jnp.asarray(float(has_active_gaze_aux)),
            "gaze_aux_loss": gaze_aux_loss,
            "gaze_weight": jnp.asarray(gaze_weight, dtype=reference_td_loss.dtype),
            "weighted_gaze_aux_loss": weighted_gaze_aux_loss,
            "gaze_to_td_ratio": gaze_to_td_ratio,
            "gaze_region_coverage": jnp.sum(gaze_region_coverage) / valid_gaze_count,
            "gaze_valid_fraction": jnp.mean(valid_gaze),
            "mask_grounding_loss": mask_grounding_loss,
            "mask_grounding_cgl_loss": mask_cgl_loss,
            "mask_grounding_mass_loss": mask_cgl_loss,
            "weighted_mask_grounding_loss": mask_grounding_aux_loss,
            "mask_grounding_aux_loss": mask_grounding_aux_loss,
            "mask_grounding_to_td_ratio": mask_grounding_to_td_ratio,
            "mask_grounding_coverage": (
                jnp.sum(mask_grounding_coverage) / valid_mask_count
            ),
            "mask_grounding_outside_mass": (
                jnp.sum(mask_grounding_outside_mass) / valid_mask_count
            ),
            "mask_feature_entropy": (
                jnp.sum(mask_feature_entropy) / valid_mask_count
            ),
            "mask_grounding_valid_fraction": jnp.mean(valid_mask),
        }

        return visual_aux_loss, info

    def loss_fns(self, batch, train_step=None):
        losses = {
            "critic": partial(self.critic_loss_fn, batch, train_step=train_step),
            "grasp_critic": partial(self.grasp_critic_loss_fn, batch),
            "actor": partial(self.policy_loss_fn, batch),
            "temperature": partial(self.temperature_loss_fn, batch),
        }
        if self.config.get("use_visual_aux", False):
            losses["visual_aux"] = partial(
                self.visual_aux_loss_fn,
                batch,
                train_step=train_step,
            )
        return losses

    @partial(jax.jit, static_argnames=("pmap_axis", "networks_to_update"))
    def update(
        self,
        batch: Batch,
        *,
        pmap_axis: Optional[str] = None,
        networks_to_update: FrozenSet[str] = frozenset(
            {"actor", "critic", "visual_aux", "grasp_critic", "temperature"}
        ),
        **kwargs,
    ):
        return super().update(
            batch,
            pmap_axis=pmap_axis,
            networks_to_update=networks_to_update,
            **kwargs,
        )

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
        gaze_valid_threshold: float = 1e-8,
        gaze_region_radius: int = 1,
        gaze_attention_image_key: str = "front_camera",
        mask_feature_gate_alpha: float = 1.0,
        mask_feature_min_gate: float = 0.1,
        mask_pick_place_phase_control: bool = False,
        return_raw_attention: bool = False,
        return_mask_encoder_attention: bool = False,
        **kwargs,
    ):
        policy_network_kwargs["activate_final"] = True
        critic_network_kwargs["activate_final"] = True
        use_pretrained_resnet_mask_pipeline = encoder_type == "resnet-pretrained"
        mask_suppress_beta = 1.0 if use_pretrained_resnet_mask_pipeline else 0.0
        use_mask_feature_head = use_pretrained_resnet_mask_pipeline
        mask_feature_hidden_dim = 128
        use_mask_encoder = True
        mask_encoder_latent_dim = 64
        mask_grounding_threshold = 0.05
        mask_grounding_cell_threshold = 0.04
        use_visual_aux = use_pretrained_resnet_mask_pipeline
        if use_visual_aux:
            kwargs.setdefault(
                "visual_aux_optimizer_kwargs",
                kwargs.get("critic_optimizer_kwargs", {"learning_rate": 3e-4}),
            )
        else:
            # A pure ViT run must not even create a visual-aux optimizer.
            kwargs.pop("visual_aux_optimizer_kwargs", None)

        image_keys = tuple(image_keys)
        head_mask_key = (
            "front_camera_mask1"
            if "front_camera_mask1" in image_keys
            else "front_camera_mask"
            if "front_camera_mask" in image_keys
            else None
        )
        mask_cnn_key = (
            "front_camera_mask"
            if "front_camera_mask" in image_keys
            else head_mask_key
        )
        mask_target_pairs = (
            (("front_camera", head_mask_key),)
            if use_mask_feature_head and head_mask_key in image_keys
            else ()
        )
        mask_encoder_pairs = (
            (("front_camera", mask_cnn_key),)
            if mask_cnn_key in image_keys
            else ()
        )
        mask_suppress_pairs = (
            (("front_camera", "front_camera_mask2"),)
            if use_pretrained_resnet_mask_pipeline
            and "front_camera_mask2" in image_keys
            else ()
        )
        mask_target_keys = {mask_key for _, mask_key in mask_target_pairs}
        mask_encoder_keys = {mask_key for _, mask_key in mask_encoder_pairs}
        mask_suppress_keys = {mask_key for _, mask_key in mask_suppress_pairs}
        mask_observation_keys = {
            key
            for key in image_keys
            if key in {"front_camera_mask", "front_camera_mask1", "front_camera_mask2"}
        }
        encoder_image_keys = [
            key
            for key in image_keys
            if key not in mask_observation_keys
            and key not in mask_target_keys
            and key not in mask_encoder_keys
            and key not in mask_suppress_keys
        ]

        if encoder_type == "resnet":
            from serl_launcher.vision.resnet_v1 import resnetv1_configs

            encoders = {
                image_key: resnetv1_configs["resnetv1-10"](
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                    bottleneck_dim=256,
                    name=f"encoder_{image_key}",
                )
                for image_key in encoder_image_keys
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
                for image_key in encoder_image_keys
            }
        elif encoder_type in ("vit", "vit-small"):
            from serl_launcher.vision.vit import ViTImageEncoder

            encoders = {
                image_key: ViTImageEncoder(
                    patch_size=(16, 16),
                    hidden_dim=192,
                    num_layers=4,
                    num_heads=6,
                    mlp_dim=384,
                    bottleneck_dim=256,
                    pooling_method="mean",
                    dropout_rate=0.0,
                    attention_dropout_rate=0.0,
                    normalize_method="unit",
                    name=f"encoder_{image_key}",
                )
                for image_key in encoder_image_keys
            }
        else:
            raise NotImplementedError(f"Unknown encoder type: {encoder_type}")

        from serl_launcher.common.encoding import EncodingWrapper

        encoder_def = EncodingWrapper(
            encoder=encoders,
            use_proprio=use_proprio,
            enable_stacking=True,
            image_keys=encoder_image_keys,
            # TODO: If gaze is collected on a different camera in a future task,
            # pass that camera key through the launcher/config instead of front_camera.
            attention_image_key=gaze_attention_image_key,
            mask_target_pairs=mask_target_pairs,
            mask_encoder_pairs=mask_encoder_pairs,
            mask_suppress_pairs=mask_suppress_pairs,
            mask_suppress_beta=mask_suppress_beta,
            use_mask_feature_head=use_mask_feature_head,
            mask_feature_gate_alpha=mask_feature_gate_alpha,
            mask_feature_min_gate=mask_feature_min_gate,
            mask_feature_hidden_dim=mask_feature_hidden_dim,
            use_mask_encoder=use_mask_encoder,
            mask_encoder_latent_dim=mask_encoder_latent_dim,
            mask_pick_place_phase_control=mask_pick_place_phase_control,
            return_raw_attention=return_raw_attention,
            return_mask_encoder_attention=return_mask_encoder_attention,
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
            gaze_valid_threshold=gaze_valid_threshold,
            gaze_region_radius=gaze_region_radius,
            mask_suppress_beta=mask_suppress_beta,
            use_mask_feature_head=use_mask_feature_head,
            mask_feature_hidden_dim=mask_feature_hidden_dim,
            use_mask_encoder=use_mask_encoder,
            mask_encoder_latent_dim=mask_encoder_latent_dim,
            mask_grounding_threshold=mask_grounding_threshold,
            mask_grounding_cell_threshold=mask_grounding_cell_threshold,
            use_visual_aux=use_visual_aux,
            **kwargs,
        )

        if encoder_type == "resnet-pretrained":
            from serl_launcher.utils.train_utils import load_resnet10_params

            agent = load_resnet10_params(agent, encoder_image_keys)

        return agent
