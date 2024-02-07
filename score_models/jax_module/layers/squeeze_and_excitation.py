import jax.numpy as jnp
from jax import nn as jnn
from jax import random
import flax.linen as nn


class SqueezeAndExcite(nn.Module):
    channels: int
    hidden_units: int

    def setup(self):
        self.excite_network = nn.Sequential(
            [
                nn.Dense(self.hidden_units),
                nn.relu,
                nn.Dense(self.channels),
            ]
        )

    def __call__(self, x):
        B, C, H, W = x.shape
        z = jnp.mean(x, axis=(2, 3))  # Squeeze operation is a global average
        z = self.excite_network(z)  # compute channel importance
        s = jnn.sigmoid(z).reshape((B, C, 1, 1))
        return s * x  # scale channels
