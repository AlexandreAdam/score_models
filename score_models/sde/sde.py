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
    def mu(self, t) -> Tensor:
        ...
    
    @abstractmethod
    def sigma(self, t) -> Tensor:
        ...
        
    @abstractmethod
    def prior(self, shape) -> Distribution:
        """
        High temperature prior distribution. Typically a Gaussian distribution.
        """
        ...
    
    @abstractmethod
    def diffusion(self, t:Tensor, x: Tensor) -> Tensor:
        """
        Diffusion coefficient of the SDE.
        """
        ...

    @abstractmethod
    def drift(self, t, x) -> Tensor:
        """
        Drift coefficient of the SDE.
        """
        ...

    def perturbation_scalars(self, t) -> Tuple[Tensor, Tensor]:
        return self.mu(t), self.sigma(t)
    
    def perturbation_kernel(self, t: Tensor, x0: Tensor) -> Tensor:
        """
        Sample from the marginal at time t using the Gaussian perturbation kernel
        and the reparametrization trick.
        """
        _, *D = x0.shape
        mu_t = self.mu(t).view(-1, *[1]*len(D))
        sigma_t = self.sigma(t).view(-1, *[1]*len(D))
        z = torch.randn_like(x0)
        return mu_t * x0 + sigma_t * z
    
    # Backward compatibility
    def sample_time_marginal(self, t: Tensor, x0: Tensor) -> Tensor:
        return self.perturbation_kernel(t, x0)

    def marginal_prob_scalars(self, t) -> Tuple[Tensor, Tensor]:
        return self.perturbation_scalars(t)
