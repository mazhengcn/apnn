import dataclasses
from collections.abc import Callable

import haiku as hk
import jax
import jax.numpy as jnp


@dataclasses.dataclass
class MLP(hk.Module):
    """MLP"""

    num_layer: int
    latent_dim: int
    activation: Callable[[jax.Array], jax.Array] = jax.nn.gelu
    keep_dim: bool = False

    def __call__(self, x: jax.Array) -> jax.Array:
        for _ in range(self.num_layer - 1):
            x = hk.Linear(self.latent_dim)(x)
            x = self.activation(x)
        x = hk.Linear(1)

        if self.keep_dim:
            x = jnp.squeeze(x, axis=-1)
        return x
