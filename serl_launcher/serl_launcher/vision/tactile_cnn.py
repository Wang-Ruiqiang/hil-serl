"""Lightweight tactile heatmap encoder for HIL-RL."""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


class TactileResidualBlock(nn.Module):
    features: int
    stride: int = 1

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        residual = x
        x = nn.Conv(
            self.features,
            (3, 3),
            strides=(self.stride, self.stride),
            padding="SAME",
            use_bias=False,
            name="conv_0",
        )(x)
        x = nn.GroupNorm(
            num_groups=min(8, self.features),
            epsilon=1e-5,
            name="group_norm_0",
        )(x)
        x = nn.gelu(x, approximate=False)
        x = nn.Conv(
            self.features,
            (3, 3),
            padding="SAME",
            use_bias=False,
            name="conv_1",
        )(x)
        x = nn.GroupNorm(
            num_groups=min(8, self.features),
            epsilon=1e-5,
            name="group_norm_1",
        )(x)
        if residual.shape[-1] != self.features or self.stride != 1:
            residual = nn.Conv(
                self.features,
                (1, 1),
                strides=(self.stride, self.stride),
                use_bias=False,
                name="residual_projection",
            )(residual)
            residual = nn.GroupNorm(
                num_groups=min(8, self.features),
                epsilon=1e-5,
                name="residual_group_norm",
            )(residual)
        return nn.gelu(x + residual, approximate=False)


class TactileSensorBackbone(nn.Module):
    """Shared feature extractor for one 128x128 tactile sensor image."""

    feature_dim: int = 64

    @nn.compact
    def __call__(self, image: jax.Array) -> jax.Array:
        x = nn.Conv(
            16,
            (5, 5),
            strides=(2, 2),
            padding="SAME",
            use_bias=False,
            name="stem",
        )(image)
        x = nn.GroupNorm(num_groups=8, epsilon=1e-5, name="stem_group_norm")(x)
        x = nn.gelu(x, approximate=False)
        x = TactileResidualBlock(32, stride=2, name="block_0")(x)
        x = TactileResidualBlock(
            self.feature_dim,
            stride=2,
            name="block_1",
        )(x)
        return TactileResidualBlock(
            self.feature_dim,
            stride=1,
            name="block_2",
        )(x)


class SharedTactileCNNEncoder(nn.Module):
    """Encode two side-by-side tactile heatmaps with one shared small CNN.

    Each sensor feature map is summarized by spatial-softmax coordinates plus
    average and maximum channel responses. The two sensor summaries are then
    projected to the same 256D interface previously produced by tactile ResNet.
    """

    feature_dim: int = 64
    output_dim: int = 256

    def setup(self):
        self.sensor_backbone = TactileSensorBackbone(
            feature_dim=self.feature_dim,
            name="shared_sensor_backbone",
        )
        self.output_projection = nn.Dense(
            self.output_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            name="output_projection",
        )
        self.output_layer_norm = nn.LayerNorm(name="output_layer_norm")

    @staticmethod
    def _normalize_image(image: jax.Array) -> jax.Array:
        image = jnp.asarray(image, dtype=jnp.float32)
        image = jnp.where(jnp.max(image) > 1.0, image / 255.0, image)
        return jnp.clip(image, 0.0, 1.0)

    @staticmethod
    def _spatial_softmax_coordinates(feature_map: jax.Array) -> jax.Array:
        batch, height, width, channels = feature_map.shape
        logits = jnp.moveaxis(feature_map, -1, 1).reshape(
            batch,
            channels,
            height * width,
        )
        weights = nn.softmax(logits, axis=-1)
        grid_y, grid_x = jnp.meshgrid(
            jnp.linspace(-1.0, 1.0, height),
            jnp.linspace(-1.0, 1.0, width),
            indexing="ij",
        )
        expected_x = jnp.sum(weights * grid_x.reshape(1, 1, -1), axis=-1)
        expected_y = jnp.sum(weights * grid_y.reshape(1, 1, -1), axis=-1)
        return jnp.concatenate((expected_x, expected_y), axis=-1)

    def _sensor_vector(
        self,
        feature_map: jax.Array,
        sensor_image: jax.Array,
    ) -> jax.Array:
        coordinates = self._spatial_softmax_coordinates(feature_map)
        average_response = jnp.mean(feature_map, axis=(-3, -2))
        maximum_response = jnp.max(feature_map, axis=(-3, -2))
        raw_average = jnp.mean(sensor_image, axis=(-3, -2))
        raw_maximum = jnp.max(sensor_image, axis=(-3, -2))
        raw_rms = jnp.sqrt(jnp.mean(jnp.square(sensor_image), axis=(-3, -2)) + 1e-8)
        return jnp.concatenate(
            (
                coordinates,
                average_response,
                maximum_response,
                raw_average,
                raw_maximum,
                raw_rms,
            ),
            axis=-1,
        )

    def __call__(self, image: jax.Array, *, train: bool = False) -> jax.Array:
        del train
        no_batch_dim = image.ndim == 3
        if no_batch_dim:
            image = image[None]
        elif image.ndim != 4:
            raise ValueError(
                "SharedTactileCNNEncoder expects HWC or BHWC, got "
                f"{image.shape}"
            )
        if image.shape[-1] < 1:
            raise ValueError(f"Tactile image has no channels: {image.shape}")
        if image.shape[-2] % 2 != 0:
            raise ValueError(
                "Tactile image width must contain two equal sensor panels, got "
                f"{image.shape[-2]}"
            )

        image = self._normalize_image(image[..., -3:])
        split = image.shape[-2] // 2
        first_sensor = image[..., :split, :]
        second_sensor = image[..., split:, :]

        # Calling the same module twice shares all convolutional parameters.
        first_feature = self.sensor_backbone(first_sensor)
        second_feature = self.sensor_backbone(second_sensor)
        sensor_vectors = jnp.concatenate(
            (
                self._sensor_vector(first_feature, first_sensor),
                self._sensor_vector(second_feature, second_sensor),
            ),
            axis=-1,
        )
        output = self.output_projection(sensor_vectors)
        output = self.output_layer_norm(output)
        output = nn.tanh(output)
        return output[0] if no_batch_dim else output
