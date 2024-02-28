from .sde import SDE
import torch
from torch import Tensor


class subVPSDE(SDE):
    """
    subVariance Preserving SDE coefficients
    """

    def __init__(self, beta_min, beta_max, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t: Tensor):
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def alpha(self, t: Tensor):
        return 0.5 * (self.beta_max - self.beta_min) * t**2 + self.beta_min * t

    def sigma(self, t: Tensor):
        return 1.0 - torch.exp(-self.alpha(t))

    def mu(self, t: Tensor):
        return torch.exp(-0.5 * self.alpha(t))

    def diffusion(self, t: Tensor, x: Tensor) -> Tensor:
        _, *D = x.shape
        return torch.sqrt(
            self.beta(t).view(-1, *[1] * len(D)) * (1 - torch.exp(-2 * self.alpha(t)))
        )

    def drift(self, t: Tensor, x: Tensor) -> Tensor:
        _, *D = x.shape
        return -0.5 * self.beta(t).view(-1, *[1] * len(D)) * x
