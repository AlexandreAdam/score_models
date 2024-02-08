import jax.numpy as jnp
import equinox as eqx
from .conv_layers import conv1x1

class Combine(eqx.Module):
    Conv_0: eqx.Module
    method: str

    def __init__(self, in_ch: int, out_ch: int, method: str = 'cat', dimensions: int = 2):
        self.Conv_0 = conv1x1(in_ch, out_ch, dimensions=dimensions)
        assert method in ["cat", "sum"], f'Method {method} not recognized.'
        self.method = method

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        h = self.Conv_0(x)
        if self.method == 'cat':
            return jnp.concatenate([h, y], axis=1)
        elif self.method == 'sum':
            return h + y

