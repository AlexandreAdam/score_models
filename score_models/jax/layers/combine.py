import jax.numpy as jnp
import equinox as eqx
from .conv_layers import conv1x1
from jaxtyping import PRNGKeyArray, Array

class Combine(eqx.Module):
    Conv_0: eqx.Module
    method: str

    def __init__(self, in_ch: int, out_ch: int, method: str = 'cat', dimensions: int = 2, *, key: PRNGKeyArray):
        self.Conv_0 = conv1x1(in_ch, out_ch, dimensions=dimensions, key=key)
        assert method in ["cat", "sum"], f'Method {method} not recognized.'
        self.method = method

    def __call__(self, x: jnp.ndarray, y: Array) -> Array:
        h = self.Conv_0(x)
        if self.method == 'cat':
            return jnp.concatenate([h, y], axis=0)
        elif self.method == 'sum':
            return h + y

