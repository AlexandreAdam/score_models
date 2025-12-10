from typing import Literal

import torch
from torch import Tensor
from torch.distributions import Independent, Normal
from torch.func import vmap, grad
import numpy as np

from .sde import SDE
from ..utils import DEVICE

PI_OVER_2 = np.pi / 2

__all__ = ["TrigSDE"]


class TrigSDE(SDE):
    def __init__(
        self,
        T: float = 1.0,
        epsilon: float = 0,
        **kwargs,
    ):
        super().__init__(T, epsilon, **kwargs)


    def mu(self, t: Tensor) -> Tensor:
        ...

    def sigma(self, t: Tensor) -> Tensor:
        ...

    def prior(self, shape, device=DEVICE):
        ...

    def diffusion(self, t: Tensor, x: Tensor) -> Tensor:
        ...

    def drift(self, t: Tensor, x: Tensor) -> Tensor:
        ...

    def inv_beta_primitive(self, beta: Tensor, beta_max, *args) -> Tensor:
        ...

    def sigma_inverse(self, sigma: Tensor) -> Tensor:
        ...
