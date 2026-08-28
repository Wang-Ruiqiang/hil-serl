from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from serl_launcher.common.common import default_init


class GazeAttentionCritic(nn.Module):
    """Critic that can expose encoder spatial attention for gaze supervision."""

    encoder: Optional[nn.Module]
    network: nn.Module
    init_final: Optional[float] = None

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        train: bool = False,
        return_attention: bool = False,
        return_cgl_attention: bool = False,
        return_feature_debug: bool = False,
    ):
        if jnp.ndim(actions) == 3:
            if return_attention or return_cgl_attention:
                raise ValueError(
                    "return_attention is only supported for 2D actions."
                )
            return jax.vmap(
                lambda action: self(observations, action, train=train),
                in_axes=1,
                out_axes=-1,
            )(actions)

        if self.encoder is None:
            obs_enc = observations
            attention_map = None
        elif return_attention or return_cgl_attention or return_feature_debug:
            if return_feature_debug:
                obs_enc, attention_map = self.encoder(
                    observations,
                    train=train,
                    return_feature_debug=True,
                )
            else:
                obs_enc, attention_map = self.encoder(
                    observations,
                    train=train,
                    return_attention=return_attention,
                    return_cgl_attention=return_cgl_attention,
                )
        else:
            obs_enc = self.encoder(observations)
            attention_map = None

        inputs = jnp.concatenate([obs_enc, actions], -1)
        outputs = self.network(inputs, train)
        if self.init_final is not None:
            value = nn.Dense(
                1,
                kernel_init=nn.initializers.uniform(-self.init_final, self.init_final),
            )(outputs)
        else:
            value = nn.Dense(1, kernel_init=default_init())(outputs)
        q_value = jnp.squeeze(value, -1)

        if return_attention or return_cgl_attention or return_feature_debug:
            return q_value, attention_map
        return q_value
