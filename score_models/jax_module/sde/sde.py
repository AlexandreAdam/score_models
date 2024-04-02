from abc import ABC, abstractmethod
from typing import Tuple

import jax.numpy as jnp
from jax import random
from jax.distrax import Distribution
from jaxtyping import PRNGKeyArray, Array

class SDE(ABC):
    """
    Abstract class for some SDE info important for the score models in JAX.
    """
    def __init__(self, T=1.0, epsilon=0., **kwargs):
        """
        The time index in the diffusion is defined in the range [epsilon, T]. 
        """
        super().__init__()
        self.T = T
        self.epsilon = epsilon
    
    @abstractmethod
    def sigma(self, t) -> Array:
        pass
    
    @abstractmethod
    def prior(self, shape) -> Distribution:
        """
        High temperature distribution
        """
        pass
    
    @abstractmethod
    def diffusion(self, t: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        pass

    @abstractmethod
    def drift(self, t: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        pass
    
    @abstractmethod
    def marginal_prob_scalars(self, t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns scaling functions for the mean and the standard deviation of the marginals
        """
        pass

    def sample_marginal(self, t: jnp.ndarray, x0: jnp.ndarray, key) -> jnp.ndarray:
        """
        Sample from the marginal at time t given some initial condition x0
        """
        _, *D = x0.shape
        z = random.normal(key, x0.shape)
        mu_t, sigma_t = self.marginal_prob_scalars(t)
        return mu_t.reshape(-1, *[1]*len(D)) * x0 + sigma_t.reshape(-1, *[1]*len(D)) * z

    def marginal_prob(self, t: jnp.ndarray, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        _, *D = x.shape
        m_t, sigma_t = self.marginal_prob_scalars(t)
        mean = m_t.reshape(-1, *[1]*len(D)) * x
        std = sigma_t.reshape(-1, *[1]*len(D))
        return mean, std

