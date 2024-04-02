from abc import ABC, abstractmethod
from typing import Tuple

from jax import random
from distrax import Distribution
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
    def diffusion(self, t: Array, x: Array) -> Array:
        pass

    @abstractmethod
    def drift(self, t: Array, x: Array) -> Array:
        pass
    
    @abstractmethod
    def marginal_prob_scalars(self, t: Array) -> Tuple[Array, Array]:
        """
        Returns scaling functions for the mean and the standard deviation of the marginals
        """
        pass

    def sample_marginal(self, t: Array, x0: Array, key: PRNGKeyArray) -> Array:
        """
        Sample from the marginal at time t given some initial condition x0
        """
        _, *D = x0.shape
        z = random.normal(key, x0.shape)
        mu_t, sigma_t = self.marginal_prob_scalars(t)
        return mu_t.reshape(-1, *[1]*len(D)) * x0 + sigma_t.reshape(-1, *[1]*len(D)) * z

    def marginal_prob(self, t: Array, x: Array) -> Tuple[Array, Array]:
        _, *D = x.shape
        m_t, sigma_t = self.marginal_prob_scalars(t)
        mean = m_t.reshape(-1, *[1]*len(D)) * x
        std = sigma_t.reshape(-1, *[1]*len(D))
        return mean, std

