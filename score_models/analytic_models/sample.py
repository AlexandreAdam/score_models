import torch
import torch.nn as nn
from torch import vmap

from ..solver import RK4_ODE


class SampleScoreModel(nn.Module):
    """
    A score model based on individual samples.

    This score model class is based on individual samples. The score at a given
    point is the average of the scores of the individual samples. The scores
    are calculated as the difference between the sample and the point, weighted
    by the inverse of the variance of the noise at that point:

    .. math::

        W_i = \\exp\\left(\\frac{-(x - x_i)^2}{\\sigma(t)^2 + \\sigma_{\\min}^2}\\right) \\
        \\nabla_x \\log p(x) = \\frac{1}{\\sum_i W_i} \\sum_i W_i \\frac{x - x_i}{\\sigma(t)^2 + \\sigma_{\\min}^2}

    Args:
        sde (SDE): The stochastic differential equation for the score model.
        samples (Tensor): The samples to use for the score model.
        sigma_min (float, optional): The minimum value of the standard deviation of the noise term. Defaults to 0.0.

    """

    def __init__(
        self,
        sde,
        samples,
        sigma_min=0.0,
    ):
        super().__init__()
        self.sde = sde
        self.samples = samples
        self.sigma_min = sigma_min

    def single_score(self, t, x):
        t_scale = self.sde.sigma(t)
        W = torch.sum(
            -0.5 * (self.samples - x) ** 2 / (t_scale**2 + self.sigma_min**2), dim=-1, keepdim=True
        )
        W = torch.exp(W - W.max())
        W = torch.nan_to_num(W)
        W /= W.sum()
        return t_scale * torch.sum(W * (self.samples - x) / (t_scale**2 + self.sigma_min**2), dim=0)

    @torch.no_grad()
    def forward(self, t, x, *args, **kwargs):
        return vmap(self.single_score)(t, x)
