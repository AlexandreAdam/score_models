from typing import Optional

import torch
from torch import Tensor

from ..sde import SDE
from .energy_model import EnergyModel
from ..architectures import NullNet


class MVGEnergyModel(EnergyModel):
    """
    A multivariate gaussian score model.

    A multivariate gaussian energy model, which can be used as a single
    multivariate gaussian or a mixture of them. if the ``mean`` is 1D, then it
    is a single gaussian, if it is 2D, then it is a mixture of gaussians.

    Args:
        sde: The SDE that the score model is associated with.
        mean: The mean of the gaussian(s).
        cov: The covariance of the gaussian(s).
        w: The weights of the mixture of gaussians (if a mixture). Default is equal weight.
    """

    def __init__(self, sde: SDE, mean: Tensor, cov: Tensor, w: Optional[Tensor] = None, **kwargs):
        super().__init__(net=NullNet(isenergy=True), sde=sde, path=None, checkpoint=None, **kwargs)
        self.sde = sde
        self.mean = mean
        self.cov = cov
        if mean.dim() == 1:
            self.energy = self.energy_single
            self.w = torch.tensor(1.0, dtype=self.mean.dtype, device=self.mean.device)
        elif mean.dim() == 2:
            self.energy = self.energy_mixture
            if w is None:
                self.w = (
                    torch.ones(self.mean.shape[0], dtype=self.mean.dtype, device=self.mean.device)
                    / self.mean.shape[0]
                )
            else:
                self.w = w
            self.w *= (torch.ones_like(self.w) * 2 * torch.pi) ** (self.mean.shape[0] / 2)
        else:
            raise ValueError("mean must be 1D (single Gaussian) or 2D (mixture of Gaussians)")

    def ll(self, t: Tensor, x: Tensor, mu: Tensor, cov: Tensor, w: Tensor):
        r = (x.squeeze() - self.sde.mu(t) * mu).flatten()
        cov_t = self.sde.mu(t) ** 2 * cov + self.sde.sigma(t) ** 2 * torch.eye(
            cov.shape[-1], dtype=cov.dtype, device=cov.device
        )
        icov = torch.linalg.inv(cov_t)
        logdet = torch.logdet(cov_t)
        ll = -0.5 * (r @ icov @ r.reshape(1, -1).T) - 0.5 * logdet + torch.log(w)
        return ll

    def energy_single(self, t: Tensor, x: Tensor, *args, **kwargs):
        """MVG energy for a single gaussian"""
        return -self.ll(t, x, self.mean, self.cov, self.w)

    def energy_mixture(self, t: Tensor, x: Tensor, *args, **kwargs):
        """MVG energy for a mixture of gaussians"""
        ll = torch.vmap(self.ll, in_dims=(None, None, 0, 0, 0))(t, x, self.mean, self.cov, self.w)
        return -torch.logsumexp(ll, dim=0)

    def unnormalized_energy(self, t: Tensor, x: Tensor, *args, **kwargs):
        raise RuntimeError("Unnormalized energy should not be called for MVG models.")

    def reparametrized_score(self, t, x, *args, **kwargs):
        raise RuntimeError("Reparametrized score should not be called for MVG models.")
