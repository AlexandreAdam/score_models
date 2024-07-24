from typing import Literal

import torch
from torch import Tensor
from torch.distributions import Independent, Normal
from torch.func import vmap, grad
import numpy as np

from .sde import SDE
from ..utils import DEVICE

PI_OVER_2 = np.pi / 2


class VPSDE(SDE):
    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20,
        T: float = 1.0,
        epsilon: float = 1e-3,
        schedule: Literal["cosine", "linear"] = "linear",
        **kwargs
    ):
        """
        Args:
            beta_min (float): Coefficient of the linear VP noise schedule, control minimum amount of noise.
            beta_max (float): Coefficient of the linear VP noise schedule, control rescaling of the data space.
            T (float, optional): The time horizon for the VPSDE. Defaults to 1.0.
            epsilon (float, optional): The initial time for the VPSDE. Defaults to 1e-3.
            schedule (str, optional): The VP noise schedule. Defaults to "cosine".
        
        Notes:
            - The "cosine" schedule is the one defined in Nichol & Dhariwal 2021. (https://arxiv.org/abs/2102.09672)
            but reformulated in continuous time. beta_max controls the clipping of the gradient to avoid 
            numerical instability as t -> T.
            - The "linear" schedule is the original noise schedule from Ho et al. 2020 and Song et al. 2021. 
            See equation (33) in Song et al 2020. (https://arxiv.org/abs/2011.13456).
            - Suggest making beta_max much larger for the cosine schedule to avoid sharp deviations in the mu function.
            After all, I am not using a manual clipping of beta, rather I make a patchwork between cosine and a linear schedule.
        """
        super().__init__(T, epsilon)
        self.beta_min = beta_min
        self.beta_max = beta_max
        
        if schedule == "cosine":
            def beta_primitive(t: Tensor, beta_max, *args) -> Tensor:
                """
                See equation (17) in Nichol & Dhariwal 2021. (https://arxiv.org/abs/2102.09672).
                The primitive of the beta function is the log of \bar{alpha} in their notation.
                
                To implement the clipping discussed in their paper,
                we instead use beta_max to control the maximum drift value in the diffusion.
                The derivative of log(\bar{\alpha}}) is
                beta(t) = 2/pi * arctan(pi*t/2), which we can invert to get the time index
                at which the drift reaches beta_max.
                """
                return torch.where(
                        t < 2/np.pi * np.arctan(beta_max / np.pi), # analytical inversion of the beta schedule
                        - 2 * torch.log(torch.cos(PI_OVER_2 * t)), # Cosine schedule for the primitive of beta
                        beta_max * t, # Linear schedule for regime where cosine is clipped
                        )

        elif schedule == "linear":
            def beta_primitive(t: Tensor, beta_max, beta_min) -> Tensor:
                """
                See equation (33) in Song et al 2020. (https://arxiv.org/abs/2011.13456)
                """
                return 0.5 * (beta_max - beta_min) * t**2 + beta_min * t
            
        else:
            raise ValueError(f"Unknown noise schedule {schedule}")
        self._beta_primitive = beta_primitive
    
    def beta_primitive(self, t: Tensor) -> Tensor:
        return self._beta_primitive(t/self.T, self.beta_max, self.beta_min)
    
    def beta(self, t: Tensor):
        return vmap(grad(self.beta_primitive))(t)

    def mu(self, t: Tensor) -> Tensor:
        return torch.exp( - 0.5 * self.beta_primitive(t))

    def sigma(self, t: Tensor) -> Tensor:
        return (1 - self.mu(t)**2).sqrt()
        
    def prior(self, shape, device=DEVICE):
        mu = torch.zeros(shape).to(device)
        return Independent(Normal(loc=mu, scale=1., validate_args=False), len(shape))

    def diffusion(self, t: Tensor, x: Tensor) -> Tensor:
        _, *D = x.shape
        beta = self.beta(t).view(-1, *[1]*len(D))
        return beta.sqrt()

    def drift(self, t: Tensor, x: Tensor) -> Tensor:
        _, *D = x.shape
        beta = self.beta(t).view(-1, *[1]*len(D))
        return - 0.5 * beta * x
