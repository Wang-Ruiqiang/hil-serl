from typing import Dict, Iterable, Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange, repeat


class EncodingWrapper(nn.Module):
    """
    Encodes observations into a single flat encoding, adding additional
    functionality for adding proprioception and stopping the gradient.

    Args:
        encoder: The encoder network.
        use_proprio: Whether to concatenate proprioception (after encoding).
    """

    encoder: nn.Module
    use_proprio: bool
    proprio_latent_dim: int = 64
    enable_stacking: bool = False
    image_keys: Iterable[str] = ("image",)
    attention_image_key: Optional[str] = "front_camera"
    mask_target_pairs: Tuple[Tuple[str, str], ...] = ()
    mask_encoder_pairs: Tuple[Tuple[str, str], ...] = ()
    mask_suppress_pairs: Tuple[Tuple[str, str], ...] = ()
    mask_suppress_beta: float = 1.0
    use_mask_feature_head: bool = False
    mask_feature_hidden_dim: int = 128
    mask_feature_gate_alpha: float = 1.0
    mask_feature_min_gate: float = 0.1
    use_mask_encoder: bool = True
    mask_encoder_latent_dim: int = 64
    mask_pick_place_phase_control: bool = False
    return_raw_attention: bool = False
    return_mask_encoder_attention: bool = False
    # state_weights: Optional[Iterable[float]] = None

    def _mask_for_spatial(self, mask: jnp.ndarray, spatial_features: jnp.ndarray):
        """Resize a mask observation to a spatial feature map as [..., H, W, 1]."""
        mask = jnp.asarray(mask, dtype=jnp.float32)
        if spatial_features.ndim == 3:
            if mask.ndim == 4:
                mask = jnp.max(mask, axis=(0, 3))
            elif mask.ndim == 3:
                mask = jnp.max(mask, axis=-1)
            elif mask.ndim != 2:
                raise ValueError(f"Unsupported unbatched mask shape: {mask.shape}")
            mask = mask[..., None]
            target_shape = (*spatial_features.shape[:2], 1)
        elif spatial_features.ndim == 4:
            if mask.ndim == 5:
                mask = jnp.max(mask, axis=(1, 4))
            elif mask.ndim == 4:
                mask = jnp.max(mask, axis=-1)
            elif mask.ndim == 3:
                pass
            else:
                raise ValueError(f"Unsupported batched mask shape: {mask.shape}")
            mask = mask[..., None]
            target_shape = (spatial_features.shape[0], *spatial_features.shape[1:3], 1)
        else:
            raise ValueError(f"Unsupported spatial feature shape: {spatial_features.shape}")
        mask = jax.image.resize(mask, target_shape, method="linear")
        mask = jnp.where(jnp.max(mask) > 1.0, mask / 255.0, mask)
        return jnp.clip(mask, 0.0, 1.0)

    def _project_pooled_feature(self, pooled_feature, output_dim: int, name: str):
        pooled_feature = nn.Dense(output_dim, name=f"{name}_proj")(pooled_feature)
        pooled_feature = nn.LayerNorm(name=f"{name}_ln")(pooled_feature)
        return nn.tanh(pooled_feature)

    def _global_feature_from_spatial(
        self,
        spatial_features: jnp.ndarray,
        output_dim: int,
        name: str,
    ):
        global_feature = jnp.mean(spatial_features, axis=(-3, -2))
        return self._project_pooled_feature(global_feature, output_dim, name=name)

    def _suppress_spatial_feature(
        self,
        spatial_features: jnp.ndarray,
        mask: jnp.ndarray,
    ):
        mask = self._mask_for_spatial(mask, spatial_features)
        suppress_gate = 1.0 - self.mask_suppress_beta * mask
        suppress_gate = jnp.clip(suppress_gate, 0.0, 1.0)
        return spatial_features * suppress_gate

    def _pick_phase_gate(self, observations, dtype=jnp.float32):
        if not self.mask_pick_place_phase_control:
            return jnp.asarray(1.0, dtype=dtype)
        state = observations.get("state")
        if state is None:
            return jnp.asarray(1.0, dtype=dtype)
        state = jnp.asarray(state)
        if state.shape[-1] < 3:
            return jnp.asarray(1.0, dtype=dtype)
        if state.ndim >= 3:
            phase = state[:, -1, -3:] if state.ndim == 3 else state[..., -1, -3:]
        else:
            phase = state[..., -3:]
        return phase[..., 0].astype(dtype)

    def _broadcast_gate(self, gate, target):
        gate = jnp.asarray(gate, dtype=target.dtype)
        while gate.ndim < target.ndim:
            gate = gate[..., None]
        return gate

    def _mask_encoder_feature(
        self,
        mask: jnp.ndarray,
        name: str,
        return_spatial_attention: bool = False,
    ):
        mask = jnp.asarray(mask, dtype=jnp.float32)
        if self.enable_stacking:
            if mask.ndim == 4:
                mask = rearrange(mask, "T H W C -> H W (T C)")
            elif mask.ndim == 5:
                mask = rearrange(mask, "B T H W C -> B H W (T C)")
        if mask.ndim == 2:
            mask = mask[..., None]
        elif mask.ndim == 3:
            mask = jnp.max(mask, axis=-1, keepdims=True)
        elif mask.ndim == 4:
            mask = jnp.max(mask, axis=-1, keepdims=True)
        else:
            raise ValueError(f"Unsupported mask encoder input shape: {mask.shape}")
        mask = jnp.where(jnp.max(mask) > 1.0, mask / 255.0, mask)
        mask = jnp.clip(mask, 0.0, 1.0)

        hidden = nn.Conv(16, (5, 5), strides=(2, 2), padding="SAME", name=f"{name}_conv1")(mask)
        hidden = nn.relu(hidden)
        hidden = nn.Conv(32, (3, 3), strides=(2, 2), padding="SAME", name=f"{name}_conv2")(hidden)
        hidden = nn.relu(hidden)
        hidden = nn.Conv(64, (3, 3), strides=(2, 2), padding="SAME", name=f"{name}_conv3")(hidden)
        hidden = nn.relu(hidden)
        attention_map = jnp.mean(jnp.square(hidden), axis=-1)
        hidden = jnp.mean(hidden, axis=(-3, -2))
        hidden = nn.Dense(self.mask_encoder_latent_dim, name=f"{name}_proj")(hidden)
        hidden = nn.LayerNorm(name=f"{name}_ln")(hidden)
        hidden = nn.tanh(hidden)
        if return_spatial_attention:
            return hidden, attention_map
        return hidden

    def _fuse_feature_list(self, features, output_dim: int, name: str):
        if len(features) == 1:
            return features[0]
        fused = jnp.concatenate(features, axis=-1)
        fused = nn.Dense(output_dim, name=f"{name}_proj")(fused)
        fused = nn.LayerNorm(name=f"{name}_ln")(fused)
        return nn.tanh(fused)

    def _mask_feature_head_outputs(
        self,
        spatial_features: jnp.ndarray,
        output_dim: int,
        name: str,
    ):
        no_batch_dim = spatial_features.ndim == 3
        if no_batch_dim:
            spatial_features = spatial_features[None]

        hidden = nn.Conv(
            self.mask_feature_hidden_dim,
            (1, 1),
            name=f"{name}_hidden",
        )(spatial_features)
        hidden = nn.relu(hidden)
        feature_delta = nn.Conv(
            spatial_features.shape[-1],
            (1, 1),
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name=f"{name}_feature_delta",
        )(hidden)
        task_features = spatial_features + feature_delta
        mask_feature_logits = nn.Conv(
            1,
            (1, 1),
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name=f"{name}_logits",
        )(hidden)[..., 0]
        mask_feature_gate = nn.sigmoid(mask_feature_logits)[..., None]
        # Use a signed gate instead of pure residual amplification.
        # Zero-init logits give sigmoid=0.5, so gate=1.0 and the pretrained
        # features are unchanged at initialization. During training, selected
        # mask features can be amplified while other regions are suppressed.
        feature_gate = 1.0 + self.mask_feature_gate_alpha * (
            2.0 * mask_feature_gate - 1.0
        )
        feature_gate = jnp.maximum(feature_gate, self.mask_feature_min_gate)
        task_features = task_features * feature_gate

        global_feature = jnp.mean(task_features, axis=(-3, -2))
        global_feature = self._project_pooled_feature(
            global_feature,
            output_dim,
            name=f"{name}_global",
        )

        if no_batch_dim:
            task_features = task_features[0]
            mask_feature_logits = mask_feature_logits[0]
            global_feature = global_feature[0]

        return global_feature, task_features, mask_feature_logits

    @nn.compact
    def __call__(
        self,
        observations: Dict[str, jnp.ndarray],
        train=False,
        stop_gradient=False,
        is_encoded=False,
        return_attention=False,
        return_cgl_attention=False,
    ) -> jnp.ndarray:
        # encode images with encoder
        encoded = []
        attention_maps = []
        cgl_attention_maps = []
        selected_attention_map = None
        selected_cgl_attention_map = None
        mask_target_pairs = dict(self.mask_target_pairs)
        mask_encoder_pairs = dict(self.mask_encoder_pairs)
        mask_suppress_pairs = dict(self.mask_suppress_pairs)
        mask_keys = (
            set(mask_target_pairs.values())
            | set(mask_encoder_pairs.values())
            | set(mask_suppress_pairs.values())
        )
        for image_key in self.image_keys:
            if image_key in mask_keys:
                continue
            image = observations[image_key]
            if not is_encoded:
                if self.enable_stacking:
                    # Combine stacking and channels into a single dimension
                    if len(image.shape) == 4:
                        image = rearrange(image, "T H W C -> H W (T C)")
                    if len(image.shape) == 5:
                        image = rearrange(image, "B T H W C -> B H W (T C)")
            needs_spatial_features = (
                return_attention
                or return_cgl_attention
                or image_key in mask_target_pairs
            )
            needs_spatial_features = needs_spatial_features or image_key in mask_suppress_pairs
            mask_key = mask_target_pairs.get(image_key)
            suppress_mask_key = mask_suppress_pairs.get(image_key)
            if needs_spatial_features:
                image, spatial_features = self.encoder[image_key](
                    image,
                    train=train,
                    encode=not is_encoded,
                    return_spatial=True,
                )
                raw_attention_map = jnp.mean(jnp.square(spatial_features), axis=-1)
                suppressed_spatial_features = spatial_features
                if suppress_mask_key is not None and suppress_mask_key in observations:
                    pick_phase_gate = self._pick_phase_gate(
                        observations,
                        dtype=spatial_features.dtype,
                    )
                    pick_phase_spatial_gate = self._broadcast_gate(
                        pick_phase_gate,
                        spatial_features,
                    )
                    suppress_input_features = self._suppress_spatial_feature(
                        spatial_features,
                        observations[suppress_mask_key],
                    )
                    suppressed_spatial_features = (
                        pick_phase_spatial_gate * suppress_input_features
                        + (1.0 - pick_phase_spatial_gate) * spatial_features
                    )
                mask_feature_map = None
                mask_head_global_feature = None
                mask_head_task_features = None
                pick_phase_gate = None
                attention_spatial_features = suppressed_spatial_features
                if self.use_mask_feature_head and image_key in mask_target_pairs:
                    pick_phase_gate = self._pick_phase_gate(
                        observations,
                        dtype=spatial_features.dtype,
                    )
                    (
                        mask_head_global_feature,
                        mask_head_task_features,
                        mask_feature_map,
                    ) = self._mask_feature_head_outputs(
                        suppressed_spatial_features,
                        image.shape[-1],
                        name=f"mask_feature_head_{image_key}",
                    )
                    spatial_head_gate = self._broadcast_gate(
                        pick_phase_gate,
                        mask_head_task_features,
                    )
                    mask_head_task_features = (
                        spatial_head_gate * mask_head_task_features
                        + (1.0 - spatial_head_gate) * suppressed_spatial_features
                    )
                    vector_head_gate = self._broadcast_gate(
                        pick_phase_gate,
                        mask_head_global_feature,
                    )
                    mask_head_global_feature = vector_head_gate * mask_head_global_feature
                    # Keep the mask2-suppressed raw branch for hand/arm context,
                    # and add the mask-head branch for target-focused features.
                    # This spatial blend is only used for attention visualization;
                    # the RL input below fuses the two pooled vector branches.
                    attention_spatial_features = (
                        spatial_head_gate
                        * (suppressed_spatial_features + mask_head_task_features)
                        + (1.0 - spatial_head_gate) * spatial_features
                    )
                if return_attention:
                    attention_map = raw_attention_map
                    if not self.return_raw_attention:
                        attention_map = jnp.mean(
                            jnp.square(attention_spatial_features),
                            axis=-1,
                        )
                    attention_maps.append(attention_map)
                    if image_key == self.attention_image_key:
                        selected_attention_map = attention_map
                if return_cgl_attention:
                    cgl_attention_map = (
                        mask_feature_map
                        if mask_feature_map is not None
                        else raw_attention_map
                    )
                    cgl_attention_maps.append(cgl_attention_map)
                    if image_key == self.attention_image_key:
                        selected_cgl_attention_map = cgl_attention_map
            else:
                image = self.encoder[image_key](image, train=train, encode=not is_encoded)

            if stop_gradient:
                image = jax.lax.stop_gradient(image)

            if image_key in mask_target_pairs:
                mask_key = mask_target_pairs[image_key]
                if mask_key not in observations:
                    raise KeyError(
                        f"Mask feature head for {image_key} requires observation key {mask_key}."
                    )
                fused_features = [
                    self._global_feature_from_spatial(
                        suppressed_spatial_features,
                        image.shape[-1],
                        name=f"mask2_suppressed_raw_global_{image_key}",
                    )
                ]
                if mask_head_global_feature is not None:
                    fused_features.append(mask_head_global_feature)
                if fused_features:
                    fused_image = self._fuse_feature_list(
                        fused_features,
                        image.shape[-1],
                        name=f"mask_visual_fusion_{image_key}",
                    )
                    if pick_phase_gate is None:
                        image = fused_image
                    else:
                        output_head_gate = self._broadcast_gate(
                            pick_phase_gate,
                            fused_image,
                        )
                        image = output_head_gate * fused_image + (
                            1.0 - output_head_gate
                        ) * image
                if stop_gradient:
                    image = jax.lax.stop_gradient(image)

            encoded.append(image)

            if (
                self.use_mask_encoder
                and image_key in mask_encoder_pairs
                and mask_encoder_pairs[image_key] in observations
            ):
                mask_encoder_output = self._mask_encoder_feature(
                    observations[mask_encoder_pairs[image_key]],
                    name=f"mask_encoder_{image_key}",
                    return_spatial_attention=(
                        return_attention and self.return_mask_encoder_attention
                    ),
                )
                if return_attention and self.return_mask_encoder_attention:
                    mask_encoder_feature, mask_encoder_attention = mask_encoder_output
                    selected_attention_map = mask_encoder_attention
                    attention_maps.append(mask_encoder_attention)
                    encoded.append(mask_encoder_feature)
                else:
                    encoded.append(mask_encoder_output)

        if self.use_proprio:
            # project state to embeddings as well
            state = observations["state"]
            state = jnp.asarray(state)
            if self.enable_stacking:
                # Combine stacking and channels into a single dimension
                if len(state.shape) == 2:
                    state = rearrange(state, "T C -> (T C)")
                    encoded = [feature.reshape(-1) for feature in encoded]
                if len(state.shape) == 3:
                    state = rearrange(state, "B T C -> B (T C)")
            
            # if self.state_weights is not None:
            #     weights = jnp.asarray(self.state_weights, dtype=state.dtype)
            #     feature_dim = state.shape[-1]
            #     base_weights = jnp.ones((feature_dim,), dtype=state.dtype)
            #     limit = min(feature_dim, weights.shape[0])
            #     base_weights = base_weights.at[:limit].set(weights[:limit])
            #     state = state * base_weights
            state = nn.Dense(
                self.proprio_latent_dim, kernel_init=nn.initializers.xavier_uniform()
            )(state)
            state = nn.LayerNorm()(state)
            state = nn.tanh(state)
            encoded.append(state)

        encoded = jnp.concatenate(encoded, axis=-1)
        # print(f"Concatenated encoded shape: {encoded.shape}")

        if return_attention or return_cgl_attention:
            selected_map = (
                selected_cgl_attention_map
                if return_cgl_attention
                else selected_attention_map
            )
            maps = cgl_attention_maps if return_cgl_attention else attention_maps
            attention_map = selected_map
            if attention_map is None:
                # TODO: For future multi-camera tasks, explicitly pass the gaze camera
                # key instead of falling back to averaged attention.
                attention_map = jnp.mean(jnp.stack(maps, axis=0), axis=0)
            return encoded, attention_map

        return encoded
