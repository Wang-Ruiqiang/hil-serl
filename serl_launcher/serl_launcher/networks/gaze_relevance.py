from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from serl_launcher.common.common import default_init


class GazeRegularizedCritic(nn.Module):
    """Critic with an additional gaze relevance gate head.

    The Q-value path is intentionally the same shape as the standard critic.
    The extra head predicts how much a gaze auxiliary signal should matter for
    this state-action pair.
    """

    encoder: Optional[nn.Module]
    network: nn.Module
    init_final: Optional[float] = None
    relevance_hidden_dim: int = 128

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        train: bool = False,
        return_gaze_relevance: bool = False,
        return_attention: bool = False,
    ):
        if jnp.ndim(actions) == 3:
            if return_gaze_relevance or return_attention:
                raise ValueError(
                    "return_gaze_relevance/return_attention are only supported for 2D actions."
                )
            return jax.vmap(
                lambda a: self(observations, a, train=train),
                in_axes=1,
                out_axes=-1,
            )(actions)

        if self.encoder is None:
            obs_enc = observations
            attention_map = None
        elif return_attention:
            obs_enc, attention_map = self.encoder(
                observations,
                train=train,
                return_attention=True,
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

        relevance = nn.Dense(self.relevance_hidden_dim, kernel_init=default_init())(
            inputs
        )
        relevance = nn.LayerNorm()(relevance)
        relevance = nn.tanh(relevance)
        relevance_logit = nn.Dense(1, kernel_init=default_init())(relevance)
        relevance_logit = jnp.squeeze(relevance_logit, -1)

        if not return_gaze_relevance:
            if return_attention:
                return q_value, attention_map
            return q_value

        if return_attention:
            return q_value, relevance_logit, attention_map

        return q_value, relevance_logit
