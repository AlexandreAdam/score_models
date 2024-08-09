import torch
from .sde import SDE
from torch import Tensor
from torch.distributions import Normal, Independent
from score_models.utils import DEVICE
import numpy as np


class VESDE(SDE):
    def __init__(
        self, sigma_min: float, sigma_max: float, T: float = 1.0, epsilon: float = 0.0, **kwargs
    ):
        """
        Variance Exploding stochastic differential equation

        Args:
            sigma_min (float): The minimum value of the standard deviation of the noise term.
            sigma_max (float): The maximum value of the standard deviation of the noise term.
            T (float, optional): The time horizon for the VESDE. Defaults to 1.0.
            device (str, optional): The device to use for computation. Defaults to DEVICE.
        """
        super().__init__(T, epsilon)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.hyperparameters.update({"sigma_min": sigma_min, "sigma_max": sigma_max})

    def mu(self, t: Tensor) -> Tensor:
        return torch.ones_like(t)

    def sigma(self, t: Tensor) -> Tensor:
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** (t / self.T)

    def t_sigma(self, sigma: Tensor) -> Tensor:
        return torch.log(torch.as_tensor(sigma / self.sigma_min, device=DEVICE)) / torch.log(
            torch.as_tensor(self.sigma_max / self.sigma_min, device=DEVICE)
        )

    def diffusion(self, t: Tensor, x: Tensor) -> Tensor:
        _, *D = x.shape  # broadcast diffusion coefficient to x shape
        # Analytical derivative of the sigma**2 function, square rooted at the end
        prefactor = np.sqrt(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)))
        return prefactor * self.sigma(t).view(-1, *[1] * len(D))

    def drift(self, t: Tensor, x: Tensor) -> Tensor:
        return torch.zeros_like(x)

    def prior(self, shape, mean=None, device=DEVICE):
        if mean is None:
            mean = torch.zeros(shape).to(device)
        return Independent(Normal(loc=mean, scale=self.sigma_max, validate_args=False), len(shape))
