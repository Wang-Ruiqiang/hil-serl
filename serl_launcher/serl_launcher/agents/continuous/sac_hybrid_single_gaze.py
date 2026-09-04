from functools import partial
from typing import FrozenSet, Iterable, Optional, Tuple

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp

from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.common.typing import Batch, Data, Params, PRNGKey
from serl_launcher.networks.actor_critic_nets import Critic, Policy, ensemblize
from serl_launcher.networks.gaze_attention import GazeAttentionCritic
from serl_launcher.networks.lagrange import GeqLagrangeMultiplier
from serl_launcher.networks.mlp import MLP
from serl_launcher.utils.gaze_mask_utils import PHASE_ONEHOT_DIM


class SACAgentHybridSingleArmGaze(SACAgentHybridSingleArm):
    """Hybrid SAC agent with mask-conditioned features and CGL grounding loss.

    This class keeps the base SAC update structure intact. The only behavioral
    change is inside critic_loss_fn, where optional auxiliary losses apply the
    CGL KL term to the front-camera mask feature head.
    """

    # Default markers reproduce the ResNet baseline exactly: the mask feature
    # head is trained ONLY by the visual-aux (CGL) loss, everything else ONLY
    # by the TD loss. Encoder pipelines that want CGL to also shape the visual
    # backbone add that backbone to "shared" via create_pixels.
    _VISUAL_AUX_PARAM_MARKERS = (
        "mask_feature_head",
    )

    @staticmethod
    def _path_keys(path) -> Tuple[str, ...]:
        return tuple(str(getattr(entry, "key", entry)) for entry in path)

    def _aux_only_markers(self) -> Tuple[str, ...]:
        return tuple(
            self.config.get(
                "aux_only_param_markers",
                self._VISUAL_AUX_PARAM_MARKERS,
            )
        )

    def _shared_aux_markers(self) -> Tuple[str, ...]:
        return tuple(self.config.get("shared_aux_param_markers", ()))

    def _frozen_markers(self) -> Tuple[str, ...]:
        return tuple(self.config.get("frozen_param_markers", ()))

    @classmethod
    def _path_matches(cls, path, markers: Tuple[str, ...]) -> bool:
        if not markers:
            return False
        keys = cls._path_keys(path)
        return any(marker in key for key in keys for marker in markers)

    def _is_aux_only_param_path(self, path) -> bool:
        """Params the TD loss must not touch (visual-aux trains them alone)."""
        return self._path_matches(path, self._aux_only_markers())

    def _is_aux_trainable_param_path(self, path) -> bool:
        """Params the visual-aux loss is allowed to train (aux-only + shared)."""
        if self._is_aux_only_param_path(path):
            return True
        return self._path_matches(path, self._shared_aux_markers())

    def _stop_visual_aux_params(self, params: Params) -> Params:
        """Freeze aux-only params inside the TD loss.

        Shared params are deliberately left alive here: for the ViT pipeline
        the backbone is trained by TD and CGL together. `apply_gradients` sums
        the per-loss updates, so overlapping parameter sets are fine.
        """
        frozen = self._frozen_markers()
        return jax.tree_util.tree_map_with_path(
            lambda path, value: (
                jax.lax.stop_gradient(value)
                if self._is_aux_only_param_path(path)
                or self._path_matches(path, frozen)
                else value
            ),
            params,
        )

    def _stop_non_visual_aux_params(self, params: Params) -> Params:
        """Freeze everything the visual-aux loss is not allowed to train."""
        return jax.tree_util.tree_map_with_path(
            lambda path, value: (
                value
                if self._is_aux_trainable_param_path(path)
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

    @partial(jax.jit, static_argnames=("argmax", "return_distribution"))
    def sample_actions_with_attention(
        self,
        observations: Data,
        *,
        seed: Optional[PRNGKey] = None,
        argmax: bool = False,
        return_distribution: bool = False,
    ):
        """Return the actor action and active encoder attention in one forward.

        ``return_distribution`` additionally hands back the pre-tanh Gaussian.
        Its loc/scale are the only way to tell a saturated policy from a merely
        confident one -- both emit |a| near 1 -- and taking them from this same
        forward avoids a second pass through the encoder in the control loop.
        """
        distribution, attention_map = self.state.apply_fn(
            {"params": self.state.params},
            observations,
            name="actor",
            train=False,
            return_attention=True,
        )
        actions = distribution.mode() if argmax else distribution.sample(seed=seed)
        if return_distribution:
            return actions, attention_map, distribution
        return actions, attention_map

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

    def forward_feature_debug(
        self,
        observations,
        actions: jax.Array,
        *,
        grad_params: Optional[Params] = None,
    ):
        _, feature_debug = self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            actions,
            name="critic",
            train=False,
            return_feature_debug=True,
        )
        return feature_debug

    def forward_fused_attribution(
        self,
        observations,
        actions: jax.Array,
        image_key: str = "front_camera",
        *,
        grad_params: Optional[Params] = None,
    ):
        """Gradient attribution of the final image vector before the MLP.

        This produces a spatial attribution map over the input image. It is
        not a reconstruction of the pooled vector; it measures which image
        locations affect the final fused image representation.
        """
        params = grad_params or self.state.params

        image = jnp.asarray(observations[image_key], dtype=jnp.float32)

        def fused_score(image_value):
            # Recorded RGB/tactile observations are uint8. Differentiate with
            # respect to a float copy while keeping all other observations
            # unchanged.
            obs = dict(observations)
            obs[image_key] = image_value
            _, debug = self.state.apply_fn(
                {"params": params},
                obs,
                actions,
                name="critic",
                train=False,
                return_feature_debug=True,
            )
            vector = debug["features"][image_key]["pre_mlp_vector"]
            return jnp.mean(jnp.square(vector))

        gradients = jax.grad(fused_score)(image)
        if gradients.ndim == 4:
            gradients = gradients[0]
        return jnp.mean(jnp.abs(gradients), axis=-1)

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

        if self.config.get("mask_grounding_align", "block") == "resize":
            # ViT tokens tile the image uniformly (224/16 = 14 cells of 9.14
            # source px each), so the mask has to land on that same uniform
            # grid. The block path below instead pads the image up to a
            # multiple of the grid and cuts fixed blocks, which drifts
            # progressively toward the bottom-right: at 128 -> 14 the blocks
            # are 10px and the last row/column is pure padding, i.e. always
            # empty. Measured on real demos with identical logits, the block
            # path scored inside=0.386 / hit=0.162 where this path scored
            # inside=0.987 / hit=1.000 -- the two agree on 98% of cells, but
            # the ball only occupies ~2% of them.
            resized = jax.image.resize(
                mask.astype(attention_map.dtype)[:, None],
                (batch_size, 1, attention_h, attention_w),
                method="linear",
            )[:, 0]
            return (
                resized > self.config["mask_grounding_cell_threshold"]
            ).astype(attention_map.dtype)

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
        if state.shape[-1] < PHASE_ONEHOT_DIM:
            return jnp.ones((batch_size,), dtype=dtype)
        if state.ndim == 3:
            phase = state[:, -1, -PHASE_ONEHOT_DIM:]
        else:
            phase = state[:, -PHASE_ONEHOT_DIM:]
        return phase[..., 0].astype(dtype)

    def _mask_grounding_loss(self, observations, attention_map):
        # "front_camera_mask" holds whichever slot the phase classifier
        # selected: mask1 (ball) while picking, mask2 (basket) while placing.
        # vit-grounded points the grounding query at it so the query learns
        # "attend to the current phase's target" instead of "find the ball",
        # which left the place phase with no attention supervision at all.
        # resnet-pretrained keeps "front_camera_mask1" so the baseline is
        # unchanged.
        preferred = self.config.get("mask_grounding_key") or "front_camera_mask1"
        if preferred in observations:
            key = preferred
        elif "front_camera_mask1" in observations:
            key = "front_camera_mask1"
        else:
            key = "front_camera_mask"
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

    def _freeze(self, params: Params) -> Params:
        """Detach frozen params before any loss sees them.

        The encoder is shared by actor, critic and grasp_critic, so stopping
        gradients inside `critic_loss_fn` alone still leaves the actor and the
        grasp critic free to move the trunk. Applied here it covers every loss.
        """
        markers = self._frozen_markers()
        if not markers:
            return params
        return jax.tree_util.tree_map_with_path(
            lambda path, value: (
                jax.lax.stop_gradient(value)
                if self._path_matches(path, markers)
                else value
            ),
            params,
        )

    def loss_fns(self, batch, train_step=None):
        def frozen(fn):
            return lambda params, *args, **kwargs: fn(
                self._freeze(params), *args, **kwargs
            )

        losses = {
            "critic": frozen(partial(self.critic_loss_fn, batch, train_step=train_step)),
            "grasp_critic": frozen(partial(self.grasp_critic_loss_fn, batch)),
            "actor": frozen(partial(self.policy_loss_fn, batch)),
            "temperature": partial(self.temperature_loss_fn, batch),
        }
        if self.config.get("use_visual_aux", False):
            losses["visual_aux"] = frozen(
                partial(
                    self.visual_aux_loss_fn,
                    batch,
                    train_step=train_step,
                )
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
        encoder_checkpoint_path: Optional[str] = None,
        freeze_encoder: bool = False,
        tactile_encoder_type: str = "cnn",
        vit_image_size: tuple = (224, 224),
        vit_hidden_dim: int = 192,
        vit_num_layers: int = 4,
        vit_num_heads: int = 6,
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
        use_vit_grounded_pipeline = encoder_type == "vit-grounded"
        # vit-gaze is vit-grounded's encoder with the mask removed from every
        # role it played. The pretrained grounding query was supervised by the
        # operator's recorded gaze instead of a segmentation mask, so at RL
        # time there is no mask predictor, no pick classifier, and no phase
        # one-hot: the observation is RGB + tactile + an 8-wide state. The ViT
        # is frozen and the readout is what RL trains, which is structurally
        # the same arrangement as the resnet-pretrained baseline that works.
        use_vit_gaze_pipeline = encoder_type == "vit-gaze"
        use_vit_pipeline = use_vit_grounded_pipeline or use_vit_gaze_pipeline
        # Freezing covers the ViT trunk and the grounding query, but NOT the
        # spatial readout or the bottleneck: the working resnet baseline is
        # exactly "frozen trunk + trainable SpatialLearnedEmbeddings", and that
        # readout is where the RL losses do their selecting.
        # vit-gaze forces the freeze rather than trusting the caller to pair
        # it with freeze_encoder=True. The two are not independent choices:
        # CGL is permanently off in this pipeline, so an unfrozen trunk would
        # drift the grounding query's inputs out from under it with nothing to
        # pull them back -- the failure would be silent, and it would look
        # exactly like a normal run that never learns.
        freeze_vit_trunk = (
            bool(freeze_encoder) and use_vit_pipeline
        ) or use_vit_gaze_pipeline
        frozen_param_markers = (
            (
                "patch_embedding",
                "pos_embedding",
                "encoder_block_",
                "encoder_norm",
                "cls",
                "grounding_query",
                # The conditioners belong with the query, not with the readout.
                # They decide WHICH of the query table's rows gets asked, so a
                # trainable conditioner in front of a frozen table is the same
                # thing as a trainable query. Leaving them out cost the
                # 2026-09-02 run its gaze conditioning: after 152k steps
                # gaze_conditioner/Dense_0's bias had moved 261% of its
                # pretrained magnitude under TD alone, and the query stopped
                # following the fixation -- the attention answered "ball" while
                # the gaze, the selected mask and the operator were all on the
                # basket. The phase-conditioned encoder never showed this
                # because it has no conditioner: its one-hot goes straight from
                # the observation into the einsum with the frozen table, with
                # nothing learnable in between.
                "gaze_conditioner",
                "tactile_conditioner",
            )
            if freeze_vit_trunk
            else ()
        )
        mask_suppress_beta = 1.0 if use_pretrained_resnet_mask_pipeline else 0.0
        use_mask_feature_head = use_pretrained_resnet_mask_pipeline
        mask_feature_hidden_dim = 128
        # The mask CNN obs branch stays on for the mask pipelines: it is part
        # of the configuration that works, and keeping it makes the ResNet/ViT
        # comparison a pure backbone swap. vit-gaze cannot have it -- no mask
        # exists at RL time -- so its encoding vector is
        # [ViT 256, tactile 256, state 64] instead of [.., mask 64, ..].
        use_mask_encoder = not use_vit_gaze_pipeline
        mask_encoder_latent_dim = 64
        mask_grounding_threshold = 0.05
        mask_grounding_cell_threshold = 0.04
        # CGL mask grounding runs for the ResNet baseline (supervising its
        # mask feature head) and for vit-grounded (supervising the ViT's
        # grounding query, and through it the ViT trunk).
        # vit-gaze is absent here on purpose, and not only because its trunk is
        # frozen: CGL needs a per-frame target on RL's own distribution, and the
        # only sources for one are a mask predictor or a gaze predictor. Both
        # are exactly what this pipeline exists to remove. The grounding the
        # query has is what offline pretraining gave it, and freezing is what
        # keeps it -- an unfrozen trunk with no CGL would drift the query's
        # inputs out from under it with nothing to pull them back.
        use_visual_aux = (
            use_pretrained_resnet_mask_pipeline or use_vit_grounded_pipeline
        ) and not freeze_vit_trunk
        aux_only_param_markers = ("mask_feature_head",)
        shared_aux_param_markers = ()
        if use_vit_grounded_pipeline:
            # grounding_query: trained only by CGL.
            # task_encoder (the ViT trunk): trained by TD *and* CGL.
            # NOTE: "task_encoder" is the attribute name on
            # ViTGroundedEncodingWrapper. Flax names a submodule passed in as a
            # dataclass attribute after the attribute, ignoring the name= given
            # at construction, so these markers must track the attribute name.
            aux_only_param_markers = ("grounding_query",)
            shared_aux_param_markers = ("task_encoder",)
        # `_pick_phase_gate_per_sample` reads this out of `self.config`, so it
        # has to be forwarded into `cls.create` -- passing it to EncodingWrapper
        # alone (which is all the resnet path needs, for mask suppression and
        # gating) leaves the CGL loss ungated. Restricted to vit-grounded on
        # purpose: the resnet-pretrained baseline has always run CGL ungated,
        # and it must stay bit-identical for the comparison.
        # The grounding target now follows the phase, so gating the loss to
        # pick frames would throw away exactly the place-phase supervision this
        # is meant to add. During a phase transition the wrapper leaves
        # front_camera_mask empty, and an empty mask already zeroes the sample
        # through `valid_mask`, so no gate is needed.
        cgl_pick_phase_gate = False
        mask_grounding_key = (
            "front_camera_mask"
            if use_vit_grounded_pipeline
            else "front_camera_mask1"
        )
        # "resize" matches how the offline pretraining discretizes the ball
        # mask, and is the geometrically correct mapping onto a ViT token grid.
        # The resnet baseline keeps "block" so its behaviour stays identical.
        mask_grounding_align = "resize" if use_vit_grounded_pipeline else "block"
        if use_visual_aux:
            default_aux_lr = 1e-4 if use_vit_grounded_pipeline else 3e-4
            kwargs.setdefault(
                "visual_aux_optimizer_kwargs",
                {"learning_rate": default_aux_lr},
            )
        else:
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
            if use_mask_encoder and mask_cnn_key in image_keys
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
        if tactile_encoder_type not in ("cnn", "resnet"):
            raise ValueError(
                "tactile_encoder_type must be 'cnn' or 'resnet', got "
                f"{tactile_encoder_type!r}"
            )
        # "cnn" routes tactile_data through the small SharedTactileCNNEncoder;
        # "resnet" keeps the original frozen-ResNet tactile branch.
        use_resnet_tactile_cnn = (
            (use_pretrained_resnet_mask_pipeline or use_vit_grounded_pipeline)
            and tactile_encoder_type == "cnn"
            and "tactile_data" in encoder_image_keys
        )
        if use_resnet_tactile_cnn:
            encoder_image_keys = [
                key for key in encoder_image_keys if key != "tactile_data"
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
        elif encoder_type in ("vit-grounded", "vit-gaze"):
            if "front_camera" not in encoder_image_keys:
                raise ValueError(
                    f"{encoder_type} requires front_camera in the image "
                    "observation keys."
                )
            from serl_launcher.common.encoding import ViTGroundedEncodingWrapper
            from serl_launcher.vision.encoder_utils import (
                load_encoder_checkpoint_config,
            )
            from serl_launcher.vision.mask_cnn import MaskCNNEncoder
            from serl_launcher.vision.tactile_cnn import SharedTactileCNNEncoder
            from serl_launcher.vision.vit import ViTImageEncoder

            # Whether the grounding query is selected by phase is a property of
            # the pretrained checkpoint, not a free choice: a phase-conditioned
            # run stores a (2, 1, D) query table and an unconditioned one stores
            # (1, 1, D). Read it from the checkpoint's own config so the two can
            # never be paired wrongly -- guessing here would either crash on
            # load or, worse, load a partial subtree and run with a random
            # query.
            grounding_phase_dim = 0
            grounding_tactile = False
            if encoder_checkpoint_path:
                checkpoint_config = load_encoder_checkpoint_config(
                    encoder_checkpoint_path
                )
                grounding_phase_dim = int(
                    checkpoint_config.get("grounding_phase_dim", 0)
                )
                grounding_tactile_cfg = bool(
                    checkpoint_config.get("grounding_tactile_conditioned", False))
                grounding_gaze_cfg = bool(
                    checkpoint_config.get("grounding_gaze_conditioned", False))
                # The width only has to match a one-hot when a one-hot is what
                # feeds the query. A gaze-conditioned query reads two state
                # columns holding a position and maps them, through its own
                # head, onto however many rows the table has -- three, once the
                # hand gets a row of its own. Tactile is the same: its
                # conditioner reads an image, not the state.
                if (not (grounding_tactile_cfg or grounding_gaze_cfg)
                        and grounding_phase_dim not in (0, PHASE_ONEHOT_DIM)):
                    raise ValueError(
                        f"encoder checkpoint declares grounding_phase_dim="
                        f"{grounding_phase_dim}, but the state carries a "
                        f"{PHASE_ONEHOT_DIM}-wide phase one-hot."
                    )
                grounding_tactile = bool(
                    checkpoint_config.get("grounding_tactile_conditioned", False))
                # Gaze conditioning reads the same two state columns a phase
                # one-hot would, but they hold the gaze position instead. The
                # observation wrapper is told to write them that way; see
                # condition_on_gaze_xy in GazeDerivedObservationWrapper.
                grounding_gaze = bool(
                    checkpoint_config.get("grounding_gaze_conditioned", False))
                if use_vit_gaze_pipeline:
                    # A phase-conditioned checkpoint would make the query read
                    # state[..., -2:], and under vit-gaze those two columns are
                    # proprioception, not a phase one-hot -- the query would
                    # silently mix its two rows by whatever the gripper pose
                    # happened to be. Refuse rather than run that.
                    # Tactile conditioning is the one accepted exception: the
                    # query's conditioning then comes from the tactile frame
                    # inside the encoder, not from state columns that hold
                    # proprioception under this pipeline.
                    if grounding_phase_dim != 0 and not (grounding_tactile or grounding_gaze):
                        raise ValueError(
                            "vit-gaze requires an unconditioned grounding query, "
                            f"but {encoder_checkpoint_path} declares "
                            f"grounding_phase_dim={grounding_phase_dim}. The "
                            "gaze pipeline carries no phase one-hot in its state."
                        )
                    source = checkpoint_config.get("grounding_source", "mask")
                    # Both gaze-supervised recipes are accepted. "gaze" scores the
                    # query against a blob built from the gaze point; "gaze_mask"
                    # scores it against the SAM3 mask that gaze selected, which is
                    # the object's own outline rather than a Gaussian near it.
                    # Neither puts a mask into the observation, so the RL side is
                    # identical either way -- what differs is only what the frozen
                    # query was taught to look at.
                    if source not in ("gaze", "gaze_mask", "gaze_hybrid"):
                        raise ValueError(
                            "vit-gaze expects an encoder pretrained with "
                            "grounding_source=gaze or gaze_mask, but "
                            f"{encoder_checkpoint_path} declares "
                            f"grounding_source={source!r}."
                        )

            # Upsample 128x128 -> 224x224 so a pretrained-compatible 16x16
            # patch yields a 14x14 token grid (the ball is only ~7px wide at
            # 128, i.e. sub-patch without this). 224/16 is also ViT-S/16's
            # native ImageNet grid, so pretrained pos_embedding needs no
            # interpolation. Measured at parity with the ResNet baseline.
            task_encoder = ViTImageEncoder(
                image_size=vit_image_size,
                patch_size=(16, 16),
                hidden_dim=vit_hidden_dim,
                num_layers=vit_num_layers,
                num_heads=vit_num_heads,
                mlp_dim=vit_hidden_dim * 2,
                bottleneck_dim=256,
                pooling_method="spatial_learned_embeddings",
                num_spatial_blocks=8,
                use_grounding_query=True,
                grounding_phase_dim=grounding_phase_dim,
                grounding_tactile_conditioned=grounding_tactile,
                grounding_gaze_conditioned=grounding_gaze,
                dropout_rate=0.0,
                attention_dropout_rate=0.0,
                normalize_method="unit",
                name="encoder_front_camera",
            )
            encoder_def = ViTGroundedEncodingWrapper(
                task_encoder=task_encoder,
                grounding_phase_conditioned=(
                    grounding_phase_dim > 0
                    and not grounding_tactile
                    and not grounding_gaze),
                grounding_tactile_conditioned=grounding_tactile,
                grounding_gaze_conditioned=grounding_gaze,
                mask_encoder=(
                    MaskCNNEncoder(
                        latent_dim=mask_encoder_latent_dim,
                        name="mask_cnn_front_camera",
                    )
                    if use_mask_encoder and mask_cnn_key is not None
                    else None
                ),
                tactile_encoder=(
                    SharedTactileCNNEncoder(
                        feature_dim=64,
                        output_dim=256,
                        name="tactile_encoder",
                    )
                    if "tactile_data" in image_keys
                    else None
                ),
                use_proprio=use_proprio,
                enable_stacking=True,
                task_image_key="front_camera",
                mask_image_key=mask_cnn_key or "front_camera_mask",
            )
        else:
            raise NotImplementedError(f"Unknown encoder type: {encoder_type}")

        if not use_vit_pipeline:
            from serl_launcher.common.encoding import EncodingWrapper
            from serl_launcher.vision.tactile_cnn import SharedTactileCNNEncoder

            encoder_def = EncodingWrapper(
                encoder=encoders,
                use_proprio=use_proprio,
                tactile_encoder=(
                    SharedTactileCNNEncoder(
                        feature_dim=64,
                        output_dim=256,
                        name="tactile_encoder",
                    )
                    if use_resnet_tactile_cnn
                    else None
                ),
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
            aux_only_param_markers=aux_only_param_markers,
            shared_aux_param_markers=shared_aux_param_markers,
            mask_pick_place_phase_control=cgl_pick_phase_gate,
            mask_grounding_align=mask_grounding_align,
            mask_grounding_key=mask_grounding_key,
            frozen_param_markers=frozen_param_markers,
            **kwargs,
        )

        if encoder_type == "resnet-pretrained":
            from serl_launcher.utils.train_utils import load_resnet10_params

            agent = load_resnet10_params(agent, encoder_image_keys)
        elif encoder_type in ("vit-grounded", "vit-gaze") and encoder_checkpoint_path:
            from serl_launcher.vision.encoder_utils import (
                load_encoder_checkpoint,
                replace_named_param_subtree,
            )

            vit_params, vit_epoch = load_encoder_checkpoint(encoder_checkpoint_path)
            params, loaded_paths = replace_named_param_subtree(
                agent.state.params,
                module_name="task_encoder",
                replacement=vit_params,
                allow_partial=True,
            )
            target_params, _ = replace_named_param_subtree(
                agent.state.target_params,
                module_name="task_encoder",
                replacement=vit_params,
                allow_partial=True,
            )
            agent = agent.replace(
                state=agent.state.replace(
                    params=params,
                    target_params=target_params,
                )
            )
            regime = (
                "trunk + grounding query FROZEN, readout still trained by TD "
                "(CGL off)"
                if freeze_vit_trunk
                else "trunk fine-tuned by TD + CGL"
            )
            if grounding_tactile:
                regime += "; grounding query conditioned on TACTILE"
            elif grounding_gaze:
                # Checked before the phase branch: a gaze-conditioned encoder
                # also has grounding_phase_dim > 0, so the old ordering printed
                # PHASE-CONDITIONED for every gaze run.
                regime += (
                    f"; grounding query GAZE-CONDITIONED on state[..., -2:] "
                    f"({grounding_phase_dim} query rows), and those two columns "
                    "are withheld from the policy's proprio projection"
                )
            elif grounding_phase_dim:
                regime += "; grounding query PHASE-CONDITIONED on state[..., -2:]"
            else:
                regime += "; grounding query unconditioned (phase inferred from pixels)"
            print(
                f"Loaded pretrained ViT ({encoder_type}; {regime}) "
                f"epoch={vit_epoch} path={encoder_checkpoint_path} "
                f"modules={loaded_paths}"
            )
        elif encoder_type == "vit-gaze":
            # Unlike vit-grounded, there is no fallback here worth allowing: an
            # untrained query with CGL permanently off would never ground at
            # all, and the run would look healthy while attending to nothing.
            raise ValueError(
                "vit-gaze requires --encoder_checkpoint_path. Its grounding "
                "query is trained only offline (CGL is off during RL, because "
                "generating per-frame targets would need exactly the mask or "
                "gaze predictor this pipeline removes), so starting from "
                "scratch would leave it permanently ungrounded."
            )
        elif encoder_type == "vit-grounded":
            print(
                "[vit-grounded] no encoder_checkpoint_path given; "
                "training the ViT trunk from scratch"
            )

        return agent
