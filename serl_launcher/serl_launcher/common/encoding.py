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
    # image_weights: Optional[Dict[str, float]] = None
    # state_weights: Optional[Iterable[float]] = None

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
        for image_key in self.image_keys:
            image = observations[image_key]
            if not is_encoded:
                if self.enable_stacking:
                    # Combine stacking and channels into a single dimension
                    if len(image.shape) == 4:
                        image = rearrange(image, "T H W C -> H W (T C)")
                    if len(image.shape) == 5:
                        image = rearrange(image, "B T H W C -> B H W (T C)")
            if return_attention:
                image, spatial_features = self.encoder[image_key](
                    image,
                    train=train,
                    encode=not is_encoded,
                    return_spatial=True,
                )
                attention_maps.append(jnp.mean(jnp.square(spatial_features), axis=-1))
            else:
                image = self.encoder[image_key](image, train=train, encode=not is_encoded)

            if stop_gradient:
                image = jax.lax.stop_gradient(image)

            # if self.image_weights is not None:
            #     image = image * self.image_weights.get(image_key, 1.0)
            encoded.append(image)

        encoded = jnp.concatenate(encoded, axis=-1)
        # print(f"Concatenated encoded shape: {encoded.shape}")

        if self.use_proprio:
            # project state to embeddings as well
            state = observations["state"]
            state = jnp.asarray(state)
            if self.enable_stacking:
                # Combine stacking and channels into a single dimension
                if len(state.shape) == 2:
                    state = rearrange(state, "T C -> (T C)")
                    encoded = encoded.reshape(-1)
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
            encoded = jnp.concatenate([encoded, state], axis=-1)

        if return_attention:
            attention_map = jnp.mean(jnp.stack(attention_maps, axis=0), axis=0)
            return encoded, attention_map

        return encoded
