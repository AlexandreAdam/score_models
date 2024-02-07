import jax.numpy as jnp
import equinox as eqx
from jax.nn import sigmoid


class SqueezeAndExcite(eqx.Module):
    """
    Implementation of the Squeeze and Excite module, a form of channel attention
    originally described in Hu et al (2019). Squeeze-and-Excitation Networks, https://arxiv.org/abs/1709.01507
    """
    excite_network: eqx.Module

    def __init__(self, channels, hidden_units, *, key):
        keys = eqx.split_key(key, 3)
        self.excite_network = eqx.nn.Sequential([
            eqx.nn.Linear(channels, hidden_units, key=keys[0]),
            eqx.nn.ReLU(),
            eqx.nn.Linear(hidden_units, channels, key=keys[1])
        ])

    def __call__(self, x):
        B, C, H, W = x.shape
        # Squeeze operation is a global average
        z = jnp.mean(x, axis=(2, 3))
        # Compute channel importance
        z = self.excite_network(z)
        s = sigmoid(z).reshape((B, C, 1, 1))
        return s * x  # Scale channels

