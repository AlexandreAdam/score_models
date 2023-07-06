import torch
from torch import Tensor
from .sde import SDE
from ..utils import DEVICE


class VPSDE(SDE):
    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20,
    ):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
    
    def beta(self, t: Tensor):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def sigma(self, t: Tensor) -> Tensor:
        return self.marginal_prob_scalars(t)[1]
        
    def prior(self, dimensions: list):
        return torch.randn(dimensions)

    def marginal(self, x0: Tensor, t: Tensor) -> Tensor:
        _, *D = x0.shape
        z = torch.randn_like(x0)
        mu_t, sigma_t = self.marginal_prob_scalars(t)
        return mu_t.view(-1, *[1]*len(D)) * x0 + sigma_t.view(-1, *[1]*len(D)) * z
    
    def diffusion(self, t: Tensor, x: Tensor) -> Tensor:
        _, *D = x.shape
        return torch.sqrt(self.beta(t)).view(-1, *[1]*len(D))

    def drift(self, t: Tensor, x: Tensor) -> Tensor:
        _, *D = x.shape
        return -0.5 * self.beta(t).view(-1, *[1]*len(D)) * x

    def marginal_prob_scalars(self, t: Tensor) -> tuple[Tensor, Tensor]:
        """
        See equation (33) in Song et al 2020. (https://arxiv.org/abs/2011.13456)
        """
        log_coeff = torch.exp(-0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min)
        std = torch.sqrt(1. - torch.exp(2. * log_coeff))
        return torch.exp(log_coeff), std

