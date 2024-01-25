from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch.distributions import Normal, Independent
from torch import Tensor
from score_models.utils import DEVICE


class SDE(ABC):
    """
    Abstract class for some SDE info important for the score models
    """

    def __init__(self, t_min=0.0, t_max=1.0, **kwargs):
        """
        The time index in the diffusion is defined in the range [t_min, t_max].
        """
        super().__init__()
        self.t_min = t_min
        self.t_max = t_max

    @property
    def DT(self):
        return self.t_max - self.t_min

    @abstractmethod
    def sigma(self, t: Tensor) -> Tensor:
        """perturbation kernel standard deviation"""
        ...

    @abstractmethod
    def mu(self, t: Tensor) -> Tensor:
        """perturbation kernel mean"""
        ...

    @abstractmethod
    def diffusion(self, t: Tensor, x: Tensor) -> Tensor:
        """diffusion coefficient for SDE. This is the term in front of dw"""
        ...

    @abstractmethod
    def drift(self, t: Tensor, x: Tensor) -> Tensor:
        """drift coefficient for SDE. This is the term in front of dt"""
        ...

    def p0t(self, shape, t: Tensor, x0: Optional[Tensor] = None, device=DEVICE):
        """perturbation kernel"""
        if x0 is None:
            x0 = torch.zeros(shape).to(device)

        mu = x0 * self.mean(t)
        scale = self.sigma(t)
        return Independent(Normal(loc=mu, scale=scale, validate_args=False), len(shape))

    def prior(self, shape, x0: Optional[Tensor] = None, device=DEVICE):
        """
        High temperature (t=1) distribution
        """
        return self.p0t(shape, torch.ones(1).to(device), x0, device)

    def marginal_prob_scalars(self, t: Tensor) -> tuple[Tensor, Tensor]:
        return self.mu(t), self.sigma(t)

    def sample_marginal(self, t: Tensor, x0: Tensor) -> Tensor:
        """
        Sample from the marginal at time t given some initial condition x0
        """
        _, *D = x0.shape
        z = torch.randn_like(x0)
        mu_t, sigma_t = self.marginal_prob_scalars(t)
        return mu_t.view(-1, *[1] * len(D)) * x0 + sigma_t.view(-1, *[1] * len(D)) * z

    def marginal_prob(self, t, x):
        _, *D = x.shape
        m_t, sigma_t = self.marginal_prob_scalars(t)
        mean = m_t.view(-1, *[1] * len(D)) * x
        std = sigma_t.view(-1, *[1] * len(D))
        return mean, std
