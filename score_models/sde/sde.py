from abc import ABC, abstractmethod
from typing import Tuple

import torch
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
    def prior(self, dimensions) -> Tensor:
        """
        Sample from the high temperature prior
        """
        ...

    @abstractmethod
    def diffusion(self, t:Tensor, x: Tensor) -> Tensor:
        ...

    @abstractmethod
    def drift_f(self, t, x) -> Tensor:
        ...
    
    @abstractmethod
    def marginal(self, x0: Tensor, t: Tensor) -> Tensor:
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
        std = sigma_t.view(-1, *[1]*len(D)) * torch.ones_like(x)
        return mean, std


