import jax.numpy as jnp
import equinox as eqx
from jax import vmap
from numpy import pi


class GaussianFourierProjection(eqx.Module):
    W: jnp.ndarray

    def __init__(self, embed_dim: int, scale: float = 30.0):
        self.W = eqx.static(jnp.random.randn(embed_dim // 2) * scale)

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        t_proj = t[:, None] * self.W[None, :] * 2 * pi
        return jnp.concatenate([jnp.sin(t_proj), jnp.cos(t_proj)], axis=1)


class PositionalEncoding(eqx.Module):
    W: jnp.ndarray

    def __init__(self, channels: int, embed_dim: int, scale: float = 30.0):
        self.W = eqx.static(jnp.random.randn(embed_dim // 2, channels) * scale)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        def single_matmul(w, x):
            return jnp.dot(w, x)

        x_proj = vmap(single_matmul, in_axes=(0, None))(self.W, x) * 2 * pi
        return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=1)
