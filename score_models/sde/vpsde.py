import torch
from torch import Tensor
from .sde import SDE
from torch.distributions import Independent, Normal
from score_models.utils import DEVICE


class VPSDE(SDE):
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20, **kwargs):
        super().__init__(**kwargs)
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t: Tensor):
        return self.beta_min + (self.beta_max - self.beta_min) * t / self.t_max

    def alpha(self, t: Tensor):
        return 0.5 * (self.beta_max - self.beta_min) * (t / self.t_max) ** 2 + self.beta_min * (
            t / self.t_max
        )

    def sigma(self, t: Tensor) -> Tensor:
        return torch.sqrt(1.0 - torch.exp(-self.alpha(t)))

    def mu(self, t: Tensor) -> Tensor:
        return torch.exp(-0.5 * self.alpha(t))

    def diffusion(self, t: Tensor, x: Tensor) -> Tensor:
        _, *D = x.shape
        return torch.sqrt(self.beta(t)).view(-1, *[1] * len(D))

    def drift(self, t: Tensor, x: Tensor) -> Tensor:
        _, *D = x.shape
        return -0.5 * self.beta(t).view(-1, *[1] * len(D)) * x
