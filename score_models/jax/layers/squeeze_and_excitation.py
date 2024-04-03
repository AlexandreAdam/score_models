import jax
import jax.numpy as jnp
import equinox as eqx
from jax.nn import relu, sigmoid
from jaxtyping import PRNGKeyArray

class SqueezeAndExcite(eqx.Module):
    """
    Implementation of the Squeeze and Excite module in JAX,
    a form of channel attention originally described in Hu et al (2019).
    Squeeze-and-Excitation Networks, https://arxiv.org/abs/1709.01507
    """
    excite_network: eqx.nn.Sequential

    def __init__(self, channels, hidden_units, *, key: PRNGKeyArray):
        key1, key2 = jax.random.split(key)
        self.excite_network = eqx.nn.Sequential([
            eqx.nn.Linear(channels, hidden_units, key=key1),
            relu,
            eqx.nn.Linear(hidden_units, channels, key=key2),
        ])

    def __call__(self, x):
        B, C, H, W = x.shape
        z = jnp.mean(x, axis=(2, 3))  # Squeeze operation is a global average
        z = self.excite_network(z)  # Compute channel importance
        s = sigmoid(z).reshape(B, C, 1, 1)
        return s * x  # Scale channels

