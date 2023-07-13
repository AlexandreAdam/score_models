import torch
from .sde import SDE
from torch import Tensor
import numpy as np
from torch.distributions import Normal, Independent


class VESDE(SDE):
    def __init__(
            self,
            sigma_min: float,
            sigma_max: float,
    ):
        """
        Variance Exploding stochastic differential equation 
        
        Args:
            sigma_min (float): The minimum value of the standard deviation of the noise term.
            sigma_max (float): The maximum value of the standard deviation of the noise term.
            T (float, optional): The time horizon for the VESDE. Defaults to 1.0.
            device (str, optional): The device to use for computation. Defaults to DEVICE.
        """
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def sigma(self, t: Tensor) -> Tensor:
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    
    def prior(self, shape, mu=None):
        """
        Technically, VESDE does not change the mean of the 0 temperature distribution, 
        so I give the option to provide for more accuracy. In practice, 
        sigma_max is chosen large enough to make this choice irrelevant
        """
        if mu is None:
            mu = torch.zeros(shape)
        else:
            assert mu.shape == shape 
        return Independent(Normal(loc=mu, scale=self.sigma_max, validate_args=False), 1)
    
    def marginal(self, t: Tensor, x0: Tensor) -> Tensor:
        _, *D = x0.shape
        z = torch.randn_like(x0)
        _, sigma_t = self.marginal_prob_scalars(t)
        return x0 + sigma_t.view(-1, *[1]*len(D)) * z
    
    def marginal_prob_scalars(self, t) -> tuple[Tensor, Tensor]:
        return torch.ones_like(t), self.sigma(t)

    def diffusion(self, t: Tensor, x: Tensor) -> Tensor:
        _, *D = x.shape # broadcast diffusion coefficient to x shape
        return self.sigma(t).view(-1, *[1]*len(D)) * np.sqrt(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)))

    def drift(self, t: Tensor, x: Tensor) -> Tensor:
        return torch.zeros_like(x)


