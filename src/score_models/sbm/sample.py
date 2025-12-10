import torch
from torch import Tensor

from ..sde import SDE
from ..architectures import NullNet
from .score_model import ScoreModel


class SampleScoreModel(ScoreModel):
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
        sde: SDE,
        samples: Tensor,
        sigma_min: Tensor = 0.0,
        **kwargs,
    ):
        super().__init__(net=NullNet(isenergy=False), sde=sde, path=None, checkpoint=None, **kwargs)
        self.sde = sde
        self.samples = samples
        self.sigma_min = sigma_min

    @torch.no_grad()
    def score(self, t: Tensor, x: Tensor, *args, **kwargs):
        B, *D = x.shape
        K, *D = self.samples.shape
        sigma_t = self.sde.sigma(t[0]) # This is wrong, should be calculated for all t, not assume t[0]
        mu_t = self.sde.mu(t[0])
        W = torch.sum(
            -0.5
            * (self.samples.unsqueeze(0) * mu_t - x.unsqueeze(1)) ** 2  # B, K, *D
            / (sigma_t**2 + self.sigma_min**2),
            dim=tuple(range(2, 2 + len(D))),
            keepdim=True,
        )  # B, K, *[1]*len(D)
        W = torch.exp(W - torch.max(W, dim=1, keepdim=True).values)
        W = torch.nan_to_num(W)
        W = W / torch.sum(W, dim=1, keepdim=True)
        scores = torch.sum(
            W
            * (self.samples.unsqueeze(0) * mu_t - x.unsqueeze(1))
            / (sigma_t**2 + self.sigma_min**2),
            dim=1,
        )  # B, *D
        return scores
    
    def reparametrized_score(self, t: Tensor, x: Tensor, *args, **kwargs):
        raise RuntimeError("Reparametrized score is not defined for sample models.")

    def save(self, *args, **kwargs):
        raise RuntimeError("Sample models cannot be saved.")

    def load(self, *args, **kwargs):
        raise RuntimeError("Sample models cannot be loaded.")
