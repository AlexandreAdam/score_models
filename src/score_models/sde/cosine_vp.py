from typing import Literal

from torch import Tensor
from torch.distributions import Independent, Normal
from torch.func import vmap, grad
from .sde import SDE
from ..utils import DEVICE
import torch
import numpy as np


PI_OVER_2 = np.pi / 2

__all__ = ["CosineVPSDE"]


class CosineVPSDE(SDE):
    def __init__(
        self,
        beta_max: float = 100,
        T: float = 1.0,
        epsilon: float = 1e-3, # Very important for EDM formulation, otherwise can be set to 0
        **kwargs,
    ):
        """
        Args:
            T (float, optional): The time horizon for the VPSDE. Defaults to 1.0.
            epsilon (float, optional): The initial time for the VPSDE. Defaults to 0.

        Notes:
            - The "cosine" schedule is the one defined in Nichol & Dhariwal 2021. (https://arxiv.org/abs/2102.09672)
            but reformulated in continuous time. beta_max controls the clipping of the gradient to avoid
            numerical instability as t -> T.
            - Suggest making beta_max much larger for the cosine schedule to avoid sharp deviations in the mu function.
            After all, I am not using a manual clipping of beta, rather I make a patchwork between cosine and a linear schedule.
        """
        super().__init__(T, epsilon, **kwargs)
        self.beta_max = beta_max
        self.hyperparameters.update({"beta_max": beta_max})

    def beta_primitive(self, t: Tensor, *args) -> Tensor:
        """
        See equation (17) in Nichol & Dhariwal 2021. (https://arxiv.org/abs/2102.09672).
        The primitive of the beta function is the log of \bar{alpha} in their notation.

        To implement the clipping, we use beta_max to control the maximum drift value in the diffusion.
        The derivative of log(\bar{\alpha}}) is beta(t) = 2/pi * arctan(pi*t/2), 
        which we can invert to get the time index at which the drift reaches beta_max.
        """
        t = t / self.T
        return torch.where(
                t < 2/np.pi * np.arctan(self.beta_max / np.pi), # analytical inversion of the beta schedule
                - 2 * torch.log(torch.cos(PI_OVER_2 * t)), # Cosine schedule for the primitive of beta
                self.beta_max * t, # Linear schedule for regime where cosine is clipped
                )

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

    def beta_primitive_inverse(self, beta: Tensor, *args) -> Tensor:
        """
        The inverse of the beta primitive function.
        """
        return torch.where(
            beta < self.beta_max * 2 / np.pi * np.arctan(self.beta_max / np.pi),
            2 / np.pi * torch.arccos(torch.exp(-0.5 * beta)),
            beta / self.beta_max,
        )

    def sigma_inverse(self, sigma: Tensor) -> Tensor:
        beta = -2 * torch.log(torch.sqrt(1 - sigma**2))
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
        return self.sigma(t)
    
    def c_in(self, t: Tensor) -> Tensor:
        """
        Formula (181) in appendix C.1.2 of Karras et al. 2022 (https://arxiv.org/pdf/2206.00364),
        modified accordingly our own definition of sigma and mu
        """
        sigma = self.sigma(t)
        return 1 / (sigma**2 + 1)**(1/2)
  
