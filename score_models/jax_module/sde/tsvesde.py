from jax import grad, vmap
from distrax import Independent, Normal, Distribution
from jaxtyping import Array
import jax.numpy as jnp
import jax.nn as nn


__all__ = ["TSVESDE"]

class TSVESDE:
    def __init__(self, sigma_min: float, sigma_max: float, t_star: float, beta: float, T:float=1.0, epsilon:float=0.0, beta_fn="relu", alpha=30):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.beta = beta
        self.t_star = t_star
        self.T = T
        self.epsilon = epsilon

        if beta_fn == "relu":
            self.beta_fn = lambda t: - self.beta * nn.relu(t / self.T - self.t_star)
        elif beta_fn == "swish" or beta_fn == "silu":
            self.beta_fn = lambda t: - self.beta * nn.silu(alpha * (t / self.T - self.t_star)) / alpha
        elif beta_fn == "hardswish":
            self.beta_fn = lambda t: - self.beta * nn.hard_swish(alpha * (t / self.T - self.t_star)) / alpha
        self.beta_fn_dot = vmap(grad(self.beta_fn))

    def scale(self, t: Array) -> Array:
        return jnp.exp(self.beta_fn(t))

    def sigma(self, t: Array) -> Array:
        smin = jnp.log(self.sigma_min)
        smax = jnp.log(self.sigma_max)
        log_coeff = self.beta_fn(t) + (smax - smin) * t / self.T + smin
        return jnp.exp(log_coeff)

    def prior(self, shape) -> Distribution:
        mu = jnp.zeros(shape)
        sigma_max = jnp.exp(-self.beta * (1. - self.t_star) + jnp.log(self.sigma_max))
        return Independent(Normal(loc=mu, scale=sigma_max), reinterpreted_batch_ndims=len(shape))

    def marginal_prob_scalars(self, t: Array) -> tuple[Array, Array]:
        return self.scale(t), self.sigma(t)

    def diffusion(self, t: Array, x: Array) -> Array:
        _, *D = x.shape
        return self.sigma(t).reshape(-1, *[1]*len(D)) * jnp.sqrt(2 * (jnp.log(self.sigma_max) - jnp.log(self.sigma_min)))

    def drift(self, t: Array, x: Array) -> Array:
        _, *D = x.shape
        return self.beta_fn_dot(t).reshape(-1, *[1]*len(D)) * x

