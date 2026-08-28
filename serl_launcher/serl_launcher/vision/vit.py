from typing import Any, Literal, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from serl_launcher.vision.data_augmentations import resize
from serl_launcher.vision.resnet_v1 import SpatialLearnedEmbeddings


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


class ViTGroundingQuery(nn.Module):
    """A learned task query that cross-attends to the patch tokens.

    Returns a summary vector plus the raw attention logits over the patch
    grid. The logits are what the CGL mask-grounding loss supervises, so the
    attention is written out explicitly instead of relying on
    ``nn.SelfAttention`` (which does not expose its weights).

    With ``phase_dim > 0`` the query is selected from a table of one query per
    task phase instead of being a single constant. That is the difference
    between "look at whatever this image suggests the target is" and "look at
    the target for the phase I am being told I am in".

    The unconditioned form has to infer the phase from pixels, and it gets that
    wrong in exactly the situations where it is most expensive: with a 2D
    camera, a hand occluding the ball or merely passing in front of the grasp
    point looks like a completed grasp, so the attention slides onto the basket
    while the robot has not picked anything up. A hard one-hot makes that
    impossible -- the einsum below selects exactly one row, so the map cannot
    move to the basket until the phase signal itself flips.

    Only the query is conditioned. Keys and values stay shared, because the
    visual content of a patch does not depend on which phase the task is in;
    only the question being asked of it does.
    """

    feature_dim: int
    num_heads: int = 4
    phase_dim: int = 0
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, patch_tokens: jnp.ndarray, phase: Optional[jnp.ndarray] = None):
        if self.feature_dim % self.num_heads:
            raise ValueError("feature_dim must be divisible by num_heads")
        batch, num_patches, _ = patch_tokens.shape
        head_dim = self.feature_dim // self.num_heads

        # phase_dim == 0 keeps the table at (1, 1, D) -- the same name and the
        # same shape the unconditioned checkpoints were written with, so they
        # still load into this module.
        query_table = self.param(
            "query",
            nn.initializers.normal(stddev=0.02),
            (max(int(self.phase_dim), 1), 1, self.feature_dim),
            self.dtype,
        )
        if self.phase_dim > 0:
            if phase is None:
                raise ValueError(
                    "ViTGroundingQuery was built with phase_dim="
                    f"{self.phase_dim} but called without a phase vector."
                )
            phase = jnp.asarray(phase, dtype=query_table.dtype)
            if phase.ndim != 2 or phase.shape[-1] != self.phase_dim:
                raise ValueError(
                    f"phase must have shape (batch, {self.phase_dim}), got "
                    f"{phase.shape}"
                )
            # A hard one-hot selects a single row exactly; a soft phase blends,
            # which is what makes this degrade gracefully rather than break if
            # the classifier ever emits something other than a one-hot.
            query = jnp.einsum("bp,pqd->bqd", phase, query_table)
        else:
            query = jnp.broadcast_to(query_table, (batch, 1, self.feature_dim))
        query = nn.LayerNorm(name="query_ln")(query)
        keys = nn.LayerNorm(name="patch_ln")(patch_tokens)

        query_heads = nn.DenseGeneral(
            (self.num_heads, head_dim), axis=-1, name="query_projection"
        )(query)
        key_heads = nn.DenseGeneral(
            (self.num_heads, head_dim), axis=-1, name="key_projection"
        )(keys)
        value_heads = nn.DenseGeneral(
            (self.num_heads, head_dim), axis=-1, name="value_projection"
        )(keys)

        logits = jnp.einsum("bqhd,bshd->bhqs", query_heads, key_heads)
        logits = logits / jnp.sqrt(jnp.asarray(head_dim, dtype=logits.dtype))
        weights = nn.softmax(logits, axis=-1)
        attended = jnp.einsum("bhqs,bshd->bqhd", weights, value_heads)
        attended = attended.reshape(batch, 1, self.feature_dim)
        summary = query + nn.Dense(self.feature_dim, name="output_projection")(
            attended
        )
        # Average the heads so the grounding loss sees one map per image.
        grounding_logits = jnp.mean(logits, axis=1)[:, 0, :]
        return summary[:, 0, :], grounding_logits


class TactileQueryConditioner(nn.Module):
    """Turn a tactile frame into a soft choice over the grounding query rows.

    The output goes into the same slot a phase one-hot would, so the query is
    conditioned on contact without anyone stating what contact means. Nothing
    supervises this head directly: the CGL loss is the only gradient it sees,
    so whatever it learns to read out of the tactile image is whatever helps
    predict where the operator was looking.

    Softmax rather than a free vector because the query table has one row per
    mode, and a convex mixture keeps the conditioned query inside the span of
    those rows -- an unconstrained vector could scale the query arbitrarily and
    change the attention's sharpness rather than its target.
    """

    out_dim: int
    features: int = 16
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, tactile: jnp.ndarray) -> jnp.ndarray:
        x = tactile.astype(jnp.float32)
        if x.ndim == 3:
            x = x[None]
        # Raw 0..255 like every other image the encoder is handed.
        x = x / 255.0 * 2.0 - 1.0
        for index, stride in enumerate((4, 2, 2)):
            x = nn.Conv(self.features * (index + 1), (3, 3), strides=(stride, stride),
                        padding="SAME", dtype=self.dtype, name=f"conv_{index}")(x)
            x = nn.relu(x)
        x = jnp.mean(x, axis=(-3, -2))
        x = nn.Dense(self.out_dim, dtype=self.dtype, name="head")(x)
        return jax.nn.softmax(x, axis=-1)


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
    pooling_method: Literal["cls", "mean", "spatial_learned_embeddings"] = "mean"
    num_spatial_blocks: int = 8
    use_grounding_query: bool = False
    # >0 selects the grounding query by task phase instead of using a single
    # constant query. See ViTGroundingQuery for why. 0 = unconditioned, which
    # is what every checkpoint written before this option existed used.
    grounding_phase_dim: int = 0
    # Derive the grounding query's conditioning from tactile instead of taking
    # it from the caller. Contact is the one signal that separates two frames
    # the camera cannot: a hand holding the ball above the basket and a hand
    # passing over it look alike, and tactile does not.
    grounding_tactile_conditioned: bool = False
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
        return_grounding: bool = False,
        phase: Optional[jnp.ndarray] = None,
        tactile: Optional[jnp.ndarray] = None,
    ):
        spatial_features = None
        grounding_logits = None
        if not encode:
            x = observations
        else:
            no_batch_dim = observations.ndim == 3
            if no_batch_dim:
                observations = observations[None]
            elif observations.ndim != 4:
                raise ValueError(f"ViTImageEncoder expects HWC or BHWC, got {observations.shape}")

            if (
                self.image_size is not None
                and observations.shape[-3:-1] != tuple(self.image_size)
            ):
                # Upsampling before patchification keeps a 16x16 patch grid
                # compatible with pretrained ViT weights while still giving a
                # fine enough token grid for small objects. This mirrors what
                # ResNetEncoder already does with its own image_size field.
                observations = resize(observations, tuple(self.image_size))

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

            query_summary = None
            if self.use_grounding_query:
                query_phase = phase
                if (
                    self.grounding_tactile_conditioned
                    and self.grounding_phase_dim > 0
                    and tactile is not None
                ):
                    query_phase = TactileQueryConditioner(
                        out_dim=self.grounding_phase_dim,
                        dtype=self.dtype,
                        name="tactile_conditioner",
                    )(tactile)
                if self.grounding_phase_dim > 0 and query_phase is not None:
                    query_phase = jnp.asarray(query_phase)
                    if query_phase.ndim == 1:
                        # An unbatched observation carries an unbatched phase.
                        query_phase = query_phase[None]
                query_summary, grounding_logits = ViTGroundingQuery(
                    feature_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    phase_dim=self.grounding_phase_dim,
                    dtype=self.dtype,
                    name="grounding_query",
                )(patch_tokens, query_phase)
                grounding_logits = grounding_logits.reshape(
                    (x.shape[0], height, width)
                )

            if self.pooling_method == "cls":
                x = x[:, 0]
            elif self.pooling_method == "mean":
                x = jnp.mean(patch_tokens, axis=1)
            elif self.pooling_method == "spatial_learned_embeddings":
                # Position-aware readout: one learned spatial weight map per
                # (channel, feature) pair, trained by the RL losses. This is
                # the same readout the working ResNet pipeline uses; plain
                # mean pooling throws the patch positions away.
                x = SpatialLearnedEmbeddings(
                    height=height,
                    width=width,
                    channel=self.hidden_dim,
                    num_features=self.num_spatial_blocks,
                    name="spatial_learned_embeddings",
                )(spatial_features)
                x = nn.Dropout(0.1, deterministic=not train)(x)
            else:
                raise ValueError(f"Unknown ViT pooling_method: {self.pooling_method}")

            if query_summary is not None:
                x = jnp.concatenate((x, query_summary), axis=-1)

            if no_batch_dim:
                x = x[0]
                spatial_features = spatial_features[0]
                if grounding_logits is not None:
                    grounding_logits = grounding_logits[0]

        if self.bottleneck_dim is not None:
            x = nn.Dense(self.bottleneck_dim, name="bottleneck_dense")(x)
            x = nn.LayerNorm(name="bottleneck_ln")(x)
            x = nn.tanh(x)

        if return_grounding:
            if spatial_features is None:
                raise ValueError(
                    "return_grounding=True requires encode=True for ViTImageEncoder."
                )
            if grounding_logits is None:
                raise ValueError(
                    "return_grounding=True requires use_grounding_query=True."
                )
            return x, spatial_features, grounding_logits
        if return_spatial:
            if spatial_features is None:
                raise ValueError("return_spatial=True requires encode=True for ViTImageEncoder.")
            return x, spatial_features
        return x


vit_configs = {
    "vit-small": ViTImageEncoder,
}
