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
    mask_pool_pairs: Tuple[Tuple[str, str], ...] = ()
    use_mask_pooling: bool = True
    mask_spatial_gate_alpha: float = 0.0
    use_mask_feature_head: bool = False
    mask_feature_hidden_dim: int = 128
    mask_feature_gate_alpha: float = 1.0
    mask_feature_min_gate: float = 0.1
    return_raw_attention: bool = False
    use_modality_gate: bool = False
    modality_gate_hidden_dim: int = 128
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

    def _mask_pool_feature(
        self,
        spatial_features: jnp.ndarray,
        mask: jnp.ndarray,
        output_dim: int,
        name: str,
    ):
        mask = self._mask_for_spatial(mask, spatial_features)
        weighted = spatial_features * mask
        numerator = jnp.sum(weighted, axis=(-3, -2))
        denominator = jnp.maximum(jnp.sum(mask, axis=(-3, -2)), 1e-6)
        target_feature = numerator / denominator
        target_feature = nn.Dense(output_dim, name=f"{name}_proj")(target_feature)
        target_feature = nn.LayerNorm(name=f"{name}_ln")(target_feature)
        return nn.tanh(target_feature)

    def _project_pooled_feature(self, pooled_feature, output_dim: int, name: str):
        pooled_feature = nn.Dense(output_dim, name=f"{name}_proj")(pooled_feature)
        pooled_feature = nn.LayerNorm(name=f"{name}_ln")(pooled_feature)
        return nn.tanh(pooled_feature)

    def _fuse_feature_list(self, features, output_dim: int, name: str):
        if len(features) == 1:
            return features[0]
        fused = jnp.concatenate(features, axis=-1)
        fused = nn.Dense(output_dim, name=f"{name}_proj")(fused)
        fused = nn.LayerNorm(name=f"{name}_ln")(fused)
        return nn.tanh(fused)

    def _apply_modality_gate(self, modality_features):
        if len(modality_features) <= 1:
            return jnp.concatenate(modality_features, axis=-1)
        gate_input = jnp.concatenate(modality_features, axis=-1)
        hidden = nn.Dense(
            self.modality_gate_hidden_dim,
            name="modality_gate_hidden",
        )(gate_input)
        hidden = nn.relu(hidden)
        gate_logits = nn.Dense(
            len(modality_features),
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="modality_gate_logits",
        )(hidden)
        gate_weights = jax.nn.softmax(gate_logits, axis=-1) * len(modality_features)
        gated_features = [
            feature * gate_weights[..., index : index + 1]
            for index, feature in enumerate(modality_features)
        ]
        return jnp.concatenate(gated_features, axis=-1)

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

        mask_feature_probs = jax.nn.softmax(
            mask_feature_logits.reshape(mask_feature_logits.shape[0], -1),
            axis=-1,
        ).reshape(mask_feature_logits.shape)
        target_feature = jnp.sum(
            task_features * mask_feature_probs[..., None],
            axis=(-3, -2),
        )
        target_feature = self._project_pooled_feature(
            target_feature,
            output_dim,
            name=f"{name}_target",
        )

        if no_batch_dim:
            task_features = task_features[0]
            mask_feature_logits = mask_feature_logits[0]
            global_feature = global_feature[0]
            target_feature = target_feature[0]

        return global_feature, target_feature, task_features, mask_feature_logits

    @nn.compact
    def __call__(
        self,
        observations: Dict[str, jnp.ndarray],
        train=False,
        stop_gradient=False,
        is_encoded=False,
        return_attention=False,
    ) -> jnp.ndarray:
        # encode images with encoder
        encoded = []
        attention_maps = []
        selected_attention_map = None
        mask_pool_pairs = dict(self.mask_pool_pairs)
        mask_keys = set(mask_pool_pairs.values())
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
            needs_spatial_features = return_attention or image_key in mask_pool_pairs
            mask_key = mask_pool_pairs.get(image_key)
            spatial_attention_kwargs = {}
            if (
                mask_key is not None
                and self.mask_spatial_gate_alpha != 0.0
                and mask_key in observations
            ):
                spatial_attention_kwargs = {
                    "spatial_attention_mask": observations[mask_key],
                    "spatial_attention_alpha": self.mask_spatial_gate_alpha,
                }
            if needs_spatial_features:
                image, spatial_features = self.encoder[image_key](
                    image,
                    train=train,
                    encode=not is_encoded,
                    return_spatial=True,
                    **spatial_attention_kwargs,
                )
                raw_attention_map = jnp.mean(jnp.square(spatial_features), axis=-1)
                mask_feature_map = None
                mask_head_target_feature = None
                if self.use_mask_feature_head and image_key in mask_pool_pairs:
                    (
                        _mask_head_global_feature,
                        mask_head_target_feature,
                        spatial_features,
                        mask_feature_map,
                    ) = self._mask_feature_head_outputs(
                        spatial_features,
                        image.shape[-1],
                        name=f"mask_feature_head_{image_key}",
                    )
                if return_attention:
                    attention_map = raw_attention_map
                    if mask_feature_map is not None and not self.return_raw_attention:
                        attention_map = mask_feature_map
                    attention_maps.append(attention_map)
                    if image_key == self.attention_image_key:
                        selected_attention_map = attention_map
            else:
                image = self.encoder[image_key](image, train=train, encode=not is_encoded)

            if stop_gradient:
                image = jax.lax.stop_gradient(image)

            if image_key in mask_pool_pairs:
                mask_key = mask_pool_pairs[image_key]
                if mask_key not in observations:
                    raise KeyError(
                        f"Mask pooling for {image_key} requires observation key {mask_key}."
                    )
                fused_features = []
                if mask_head_target_feature is not None:
                    fused_features.append(mask_head_target_feature)
                if self.use_mask_pooling:
                    target_feature = self._mask_pool_feature(
                        spatial_features,
                        observations[mask_key],
                        image.shape[-1],
                        name=f"mask_pool_{image_key}",
                    )
                    fused_features.append(target_feature)
                if fused_features:
                    image = self._fuse_feature_list(
                        fused_features,
                        image.shape[-1],
                        name=f"mask_visual_fusion_{image_key}",
                    )
                if stop_gradient:
                    image = jax.lax.stop_gradient(image)

            encoded.append(image)

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

        if self.use_modality_gate:
            encoded = self._apply_modality_gate(encoded)
        else:
            encoded = jnp.concatenate(encoded, axis=-1)
        # print(f"Concatenated encoded shape: {encoded.shape}")

        if return_attention:
            attention_map = selected_attention_map
            if attention_map is None:
                # TODO: For future multi-camera tasks, explicitly pass the gaze camera
                # key instead of falling back to averaged attention.
                attention_map = jnp.mean(jnp.stack(attention_maps, axis=0), axis=0)
            return encoded, attention_map

        return encoded
