import jax.numpy as jnp
from distrax import Independent, Normal, Distribution
from jaxtyping import Array


__all__ = ["VPSDE"]


class VPSDE:
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20, T: float = 1.0, epsilon: float = 1e-5):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
        self.epsilon = epsilon

    def beta(self, t: Array) -> Array:
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def sigma(self, t: Array) -> Array:
        return self.marginal_prob_scalars(t)[1]

    def prior(self, shape) -> Distribution:
        mu = jnp.zeros(shape)
        return Independent(Normal(loc=mu, scale=1.), reinterpreted_batch_ndims=len(shape))

    def diffusion(self, t: Array, x: Array) -> Array:
        _, *D = x.shape
        return jnp.sqrt(self.beta(t)).reshape(-1, *[1]*len(D))

    def drift(self, t: Array, x: Array) -> Array:
        _, *D = x.shape
        return -0.5 * self.beta(t).reshape(-1, *[1]*len(D)) * x

    def marginal_prob_scalars(self, t: Array) -> tuple[Array, Array]:
        """
        See equation (33) in Song et al 2020. (https://arxiv.org/abs/2011.13456)
        """
        log_coeff = 0.5 * (self.beta_max - self.beta_min) * t**2 + self.beta_min * t  # integral of beta(t)
        std = jnp.sqrt(1. - jnp.exp(-log_coeff))
        return jnp.exp(-0.5 * log_coeff), std

