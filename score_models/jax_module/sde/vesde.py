import jax.numpy as jnp
from distrax import Independent, Normal, Distribution
from jaxtyping import Array


__all__ = ["VESDE"]

class VESDE:
    def __init__(self, sigma_min: float, sigma_max: float, T: float = 1.0, epsilon: float = 0.0):
        """
        Variance Exploding stochastic differential equation 
        
        Args:
            sigma_min (float): The minimum value of the standard deviation of the noise term.
            sigma_max (float): The maximum value of the standard deviation of the noise term.
            T (float, optional): The time horizon for the VESDE. Defaults to 1.0.
            device (str, optional): The device to use for computation. Defaults to DEVICE.
        """
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.T = T
        self.epsilon = epsilon

    def sigma(self, t: Array) -> Array:
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** (t / self.T)

    def prior(self, shape, mu=None) -> Distribution:
        """
        Technically, VESDE does not change the mean of the 0 temperature distribution, 
        so I give the option to provide for more accuracy. In practice, 
        sigma_max is chosen large enough to make this choice irrelevant
        """
        if mu is None:
            mu = jnp.zeros(shape)
        else:
            assert mu.shape == shape
        return Independent(Normal(loc=mu, scale=self.sigma_max), reinterpreted_batch_ndims=len(shape))

    def marginal_prob_scalars(self, t: Array) -> tuple[Array, Array]:
        return jnp.ones_like(t), self.sigma(t)

    def diffusion(self, t: Array, x: Array) -> Array:
        _, *D = x.shape  # broadcast diffusion coefficient to x shape
        return self.sigma(t).reshape(-1, *[1]*len(D)) * jnp.sqrt(2 * (jnp.log(self.sigma_max) - jnp.log(self.sigma_min)))

    def drift(self, t: Array, x: Array) -> Array:
        return jnp.zeros_like(x)

