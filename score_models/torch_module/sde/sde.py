from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch.distributions import Normal, Independent
from torch.distributions import Distribution
from torch import Tensor

class SDE(ABC):
    """
    Abstract class for some SDE info important for the score models
    """
    def __init__(self, T=1.0, epsilon=0., **kwargs):
        """
        The time index in the diffusion is defined in the range [epsilon, T]. 
        """
        super().__init__()
        self.T = T
        self.epsilon = epsilon
    
    @abstractmethod
    def sigma(self, t) -> Tensor:
        ...
    
    @abstractmethod
    def prior(self, shape) -> Distribution:
        """
        High temperature distribution
        """
        ...
    
    @abstractmethod
    def diffusion(self, t:Tensor, x: Tensor) -> Tensor:
        ...

    @abstractmethod
    def drift(self, t, x) -> Tensor:
        ...
    
    @abstractmethod
    def marginal_prob_scalars(self, t) -> Tuple[Tensor, Tensor]:
        """
        Returns scaling functions for the mean and the standard deviation of the marginals
        """
        ...

    def sample_marginal(self, t: Tensor, x0: Tensor) -> Tensor:
        """
        Sample from the marginal at time t given some initial condition x0
        """
        _, *D = x0.shape
        z = torch.randn_like(x0)
        mu_t, sigma_t = self.marginal_prob_scalars(t)
        return mu_t.view(-1, *[1]*len(D)) * x0 + sigma_t.view(-1, *[1]*len(D)) * z

    def marginal_prob(self, t, x):
        _, *D = x.shape
        m_t, sigma_t = self.marginal_prob_scalars(t)
        mean = m_t.view(-1, *[1]*len(D)) * x
        std = sigma_t.view(-1, *[1]*len(D))
        return mean, std


