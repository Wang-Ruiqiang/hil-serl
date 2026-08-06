from typing import Any, Literal, Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp


class ViTMLPBlock(nn.Module):
    """Feed-forward block used inside a Transformer encoder block."""

    mlp_dim: int
    output_dim: int
    dropout_rate: float = 0.0
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        x = nn.Dense(self.mlp_dim, dtype=self.dtype, name="Dense_0")(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not train)(x)
        x = nn.Dense(self.output_dim, dtype=self.dtype, name="Dense_1")(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not train)(x)
        return x


class ViTEncoderBlock(nn.Module):
    """A ViT-style Transformer encoder block."""

    hidden_dim: int
    num_heads: int
    mlp_dim: int
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        y = nn.LayerNorm(epsilon=1e-6, dtype=self.dtype, name="ln_0")(x)
        y = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
            out_features=self.hidden_dim,
            dropout_rate=self.attention_dropout_rate,
            deterministic=not train,
            dtype=self.dtype,
            name="self_attention",
        )(y)
        y = nn.Dropout(self.dropout_rate, deterministic=not train)(y)
        x = x + y

        y = nn.LayerNorm(epsilon=1e-6, dtype=self.dtype, name="ln_1")(x)
        y = ViTMLPBlock(
            mlp_dim=self.mlp_dim,
            output_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype,
            name="mlp",
        )(y, train=train)
        return x + y


class ViTImageEncoder(nn.Module):
    """Minimal Vision Transformer encoder compatible with SERL image encoders.

    This follows the original ViT recipe closely: split an image into patches,
    linearly project each patch, add a learned class token plus learned position
    embeddings, process the token sequence with Transformer encoder blocks, and
    pool the resulting tokens into a vector for actor/critic MLPs.
    """

    image_size: Optional[Tuple[int, int]] = None
    patch_size: Tuple[int, int] = (16, 16)
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    mlp_dim: int = 256
    bottleneck_dim: Optional[int] = 256
    pooling_method: Literal["cls", "mean"] = "mean"
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0
    normalize_method: Literal["imagenet", "unit", "none"] = "unit"
    dtype: Any = jnp.float32

    def _normalize(self, observations: jnp.ndarray) -> jnp.ndarray:
        x = observations.astype(jnp.float32)
        if self.normalize_method == "none":
            return x
        x = x / 255.0
        if self.normalize_method == "unit":
            return x * 2.0 - 1.0
        if self.normalize_method == "imagenet":
            channels = x.shape[-1]
            if channels % 3 != 0:
                return x * 2.0 - 1.0
            repeats = channels // 3
            mean = jnp.tile(jnp.array([0.485, 0.456, 0.406]), repeats)
            std = jnp.tile(jnp.array([0.229, 0.224, 0.225]), repeats)
            return (x - mean) / std
        raise ValueError(f"Unknown ViT normalize_method: {self.normalize_method}")

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        train: bool = True,
        encode: bool = True,
        return_spatial: bool = False,
    ):
        spatial_features = None
        if not encode:
            x = observations
        else:
            no_batch_dim = observations.ndim == 3
            if no_batch_dim:
                observations = observations[None]
            elif observations.ndim != 4:
                raise ValueError(f"ViTImageEncoder expects HWC or BHWC, got {observations.shape}")

            x = self._normalize(observations)
            patch_h, patch_w = self.patch_size
            x = nn.Conv(
                features=self.hidden_dim,
                kernel_size=(patch_h, patch_w),
                strides=(patch_h, patch_w),
                padding="VALID",
                dtype=self.dtype,
                name="patch_embedding",
            )(x)
            spatial_tokens = x
            height, width, _ = x.shape[-3:]
            x = x.reshape((x.shape[0], height * width, self.hidden_dim))

            cls = self.param(
                "cls",
                nn.initializers.zeros,
                (1, 1, self.hidden_dim),
                self.dtype,
            )
            cls = jnp.tile(cls, (x.shape[0], 1, 1))
            x = jnp.concatenate([cls, x], axis=1)

            pos_embedding = self.param(
                "pos_embedding",
                nn.initializers.normal(stddev=0.02),
                (1, x.shape[1], self.hidden_dim),
                self.dtype,
            )
            x = x + pos_embedding
            x = nn.Dropout(self.dropout_rate, deterministic=not train)(x)

            for idx in range(self.num_layers):
                x = ViTEncoderBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    mlp_dim=self.mlp_dim,
                    dropout_rate=self.dropout_rate,
                    attention_dropout_rate=self.attention_dropout_rate,
                    dtype=self.dtype,
                    name=f"encoder_block_{idx}",
                )(x, train=train)

            x = nn.LayerNorm(epsilon=1e-6, dtype=self.dtype, name="encoder_norm")(x)
            patch_tokens = x[:, 1:, :]
            spatial_features = patch_tokens.reshape((x.shape[0], height, width, self.hidden_dim))
            if self.pooling_method == "cls":
                x = x[:, 0]
            elif self.pooling_method == "mean":
                x = jnp.mean(patch_tokens, axis=1)
            else:
                raise ValueError(f"Unknown ViT pooling_method: {self.pooling_method}")

            if no_batch_dim:
                x = x[0]
                spatial_features = spatial_features[0]

        if self.bottleneck_dim is not None:
            x = nn.Dense(self.bottleneck_dim, name="bottleneck_dense")(x)
            x = nn.LayerNorm(name="bottleneck_ln")(x)
            x = nn.tanh(x)

        if return_spatial:
            if spatial_features is None:
                raise ValueError("return_spatial=True requires encode=True for ViTImageEncoder.")
            return x, spatial_features
        return x


vit_configs = {
    "vit-small": ViTImageEncoder,
}
