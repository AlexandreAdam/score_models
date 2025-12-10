from typing import Literal

from torch import Tensor
from torch.distributions import Independent, Normal
from torch.func import vmap, grad
from .sde import SDE
from ..utils import DEVICE
import torch
import numpy as np


PI_OVER_2 = np.pi / 2


__all__ = ["VPSDE"]


class VPSDE(SDE):
    def __init__(
        self,
        beta_min: float = 1e-2,
        beta_max: float = 20,
        T: float = 1.0,
        epsilon: float = 1e-3,
        **kwargs,
    ):
        """
        The is the original noise schedule from Ho et al. 2020 and Song et al. 2021.
        See equation (33) in Song et al 2020. (https://arxiv.org/abs/2011.13456).
        
        Args:
            beta_min (float): Coefficient of the linear VP noise schedule, control minimum amount of noise.
            beta_max (float): Coefficient of the linear VP noise schedule, control rescaling of the data space.
            T (float, optional): The time horizon for the VPSDE. Defaults to 1.0.
            epsilon (float, optional): The initial time for the VPSDE. Defaults to 1e-3.

        """
        super().__init__(T, epsilon, **kwargs)
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.hyperparameters.update(
            {"beta_min": beta_min, "beta_max": beta_max}
        )

    def beta_primitive(self, t: Tensor) -> Tensor:
        """
        See equation (33) in Song et al 2020. (https://arxiv.org/abs/2011.13456)
        """
        t = t / self.T
        return 0.5 * (self.beta_max - self.beta_min) * t**2 + self.beta_min * t

    def beta(self, t: Tensor):
        return vmap(grad(self.beta_primitive))(t)

    def mu(self, t: Tensor) -> Tensor:
        return torch.exp(-0.5 * self.beta_primitive(t))

    def sigma(self, t: Tensor) -> Tensor:
        return (1 - self.mu(t) ** 2).sqrt()

    def prior(self, shape, device=DEVICE):
        mu = torch.zeros(shape).to(device)
        return Independent(Normal(loc=mu, scale=1.0, validate_args=False), len(shape))

    def diffusion(self, t: Tensor, x: Tensor) -> Tensor:
        _, *D = x.shape
        beta = self.beta(t).view(-1, *[1] * len(D))
        return beta.sqrt()

    def drift(self, t: Tensor, x: Tensor) -> Tensor:
        _, *D = x.shape
        beta = self.beta(t).view(-1, *[1] * len(D))
        return -0.5 * beta * x

    def beta_primitive_inverse(self, beta: Tensor) -> Tensor:
        beta_diff = self.beta_max - self.beta_min
        return ((self.beta_min**2 + 2 * beta_diff * beta)**(1/2) - self.beta_min) / beta_diff

    def sigma_inverse(self, sigma: Tensor) -> Tensor:
        beta = - 2 * torch.log(torch.sqrt(1 - sigma**2))
        return self.beta_primitive_inverse(beta) * self.T
    
    def c_skip(self, t: Tensor) -> Tensor:
        """
        Formula (181) in appendix C.1.2 of Karras et al. 2022 (https://arxiv.org/pdf/2206.00364)
        """
        return torch.ones_like(t)
    
    def c_out(self, t: Tensor) -> Tensor:
        """
        Formula (181) in appendix C.1.2 of Karras et al. 2022 (https://arxiv.org/pdf/2206.00364),
        modified accordingly our own definition of sigma and mu
        """
        return self.sigma(t) / self.mu(t) # Not quite sigma(t) of Karras, but essentially the same
    
    def c_in(self, t: Tensor) -> Tensor:
        """
        Formula (181) in appendix C.1.2 of Karras et al. 2022 (https://arxiv.org/pdf/2206.00364),
        modified accordingly our own definition of sigma and mu
        """
        sigma = self.sigma(t) / self.mu(t)
        return 1 / (sigma**2 + 1)**(1/2)
 
