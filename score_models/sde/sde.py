from abc import ABC, abstractmethod
from typing import Tuple

from torch.distributions import Normal, Independent
from torch.distributions import Distribution
from torch import Tensor

class SDE(ABC):
    """
    Abstract class for some SDE info important for the score models
    """
    def __init__(self):
        super().__init__()
    
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
    def marginal(self, t: Tensor, x0: Tensor) -> Tensor:
        """
        Sample from the marginal at time t given some initial condition x0
        """
        ...

    @abstractmethod
    def marginal_prob_scalars(self, t) -> Tuple[Tensor, Tensor]:
        """
        Returns scaling functions for the mean and the standard deviation of the marginals
        """
        ...

    def marginal_prob(self, t, x):
        _, *D = x.shape
        m_t, sigma_t = self.marginal_prob_scalars(t)
        mean = m_t.view(-1, *[1]*len(D)) * x
        std = sigma_t.view(-1, *[1]*len(D))
        return mean, std


