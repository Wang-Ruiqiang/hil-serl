"""Small CNN that turns a binary target mask image into a feature vector.

This mirrors the inline mask branch inside ``EncodingWrapper``
(``_mask_encoder_feature``) but lives in its own module so that new encoder
pipelines can reuse it without touching ``EncodingWrapper``, whose parameter
names must stay byte-identical for the ``resnet-pretrained`` baseline.
"""

from __future__ import annotations

from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp


class MaskCNNEncoder(nn.Module):
    """Encode a mask image into ``latent_dim`` features.

    The mask arrives as a uint8 RGB image (the mask painted into three equal
    channels), so it is collapsed to one channel and rescaled to [0, 1] first.
    """

    latent_dim: int = 64

    @nn.compact
    def __call__(
        self,
        mask: jnp.ndarray,
        *,
        return_spatial: bool = False,
    ):
        mask = jnp.asarray(mask, dtype=jnp.float32)
        no_batch_dim = mask.ndim == 3
        if no_batch_dim:
            mask = mask[None]
        if mask.ndim != 4:
            raise ValueError(f"MaskCNNEncoder expects HWC or BHWC, got {mask.shape}")

        mask = jnp.max(mask, axis=-1, keepdims=True)
        mask = jnp.where(jnp.max(mask) > 1.0, mask / 255.0, mask)
        mask = jnp.clip(mask, 0.0, 1.0)

        hidden = nn.Conv(16, (5, 5), strides=(2, 2), padding="SAME", name="conv1")(mask)
        hidden = nn.relu(hidden)
        hidden = nn.Conv(32, (3, 3), strides=(2, 2), padding="SAME", name="conv2")(hidden)
        hidden = nn.relu(hidden)
        hidden = nn.Conv(64, (3, 3), strides=(2, 2), padding="SAME", name="conv3")(hidden)
        hidden = nn.relu(hidden)

        spatial = jnp.mean(jnp.square(hidden), axis=-1)
        pooled = jnp.mean(hidden, axis=(-3, -2))
        pooled = nn.Dense(self.latent_dim, name="proj")(pooled)
        pooled = nn.LayerNorm(name="ln")(pooled)
        pooled = nn.tanh(pooled)

        if no_batch_dim:
            pooled = pooled[0]
            spatial = spatial[0]
        if return_spatial:
            return pooled, spatial
        return pooled
