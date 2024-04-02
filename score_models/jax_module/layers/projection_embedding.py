from numpy import pi
import jax
import jax.numpy as jnp
import equinox as eqx
from jax.lax import stop_gradient
from jaxtyping import PRNGKeyArray


class GaussianFourierProjection(eqx.Module):
    W: jax.Array

    def __init__(self, embed_dim: int, scale: float = 30.0, *, key: PRNGKeyArray):
        self.W = jax.random.normal(key, shape=(embed_dim // 2,)) * scale

    def __call__(self, t: jnp.ndarray) -> jax.Array:
        # Mark W as non-trainable
        t_proj = t[:, None] * stop_gradient(self.W[None, :]) * 2 * pi
        return jnp.concatenate([jnp.sin(t_proj), jnp.cos(t_proj)], axis=1)


class PositionalEncoding(eqx.Module):
    W: jax.Array

    def __init__(self, channels: int, embed_dim: int, scale: float = 30.0, *, key: PRNGKeyArray):
        self.W = jax.random.normal(key, shape=(embed_dim // 2, channels)) * scale

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x_proj = jax.vmap(jnp.dot, in_axes=(None, 0))(self.W, x) * 2 * pi
        return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=1)
