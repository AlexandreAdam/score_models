from abc import ABC, abstractmethod

import torch
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
    def mu(self, t) -> Tensor:
        ...
    
    @abstractmethod
    def prior(self, shape) -> Distribution:
        """
        High temperature distribution
        """
        ...
    
    @abstractmethod
    def diffusion(self, x: Tensor, t: Tensor) -> Tensor:
        ...

    @abstractmethod
    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        ...
    
    def sample_marginal(self, x0: Tensor, t: Tensor) -> Tensor:
        """
        Sample from the marginal at time t given some initial condition x0
        """
        _, *D = x0.shape
        z = torch.randn_like(x0)
        mu_t, sigma_t = self.marginal_prob_scalars(t)
        return mu_t.view(-1, *[1]*len(D)) * x0 + sigma_t.view(-1, *[1]*len(D)) * z

    def marginal_prob(self, x: Tensor, t: Tensor) -> Tensor:
        _, *D = x.shape
        m_t, sigma_t = self.marginal_prob_scalars(t)
        mean = m_t.view(-1, *[1]*len(D)) * x
        std = sigma_t.view(-1, *[1]*len(D))
        return mean, std

    def marginal_prob_scalars(self, t: Tensor) -> tuple[Tensor, Tensor]:
        return self.mu(t), self.sigma(t)
