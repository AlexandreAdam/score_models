from typing import Optional

import torch
import numpy as np
from torch import Tensor

from .energy_model import EnergyModel
from .score_model import ScoreModel
from ..sde import SDE
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

        cov: The covariance of the gaussian(s). If cov.shape == mean.shape, this
            is a diagonal covariance. Otherwise, it is a full covariance matrix
            where if mean has shape (M, *D) (or just (*D,) for single MVG) then
            the covariance matrix should have shape (M, prod(D), prod(D)) (or
            just (prod(D), prod(D)) for single MVG).

        w: The weights of the mixture of gaussians (if a mixture). Default is
            equal weight.
    """

    def __init__(self, sde: SDE, mean: Tensor, cov: Tensor, w: Optional[Tensor] = None, **kwargs):
        super().__init__(net=NullNet(isenergy=True), sde=sde, path=None, checkpoint=None, **kwargs)
        self.sde = sde
        self.mean = mean
        self.cov = cov
        self.diag = mean.shape == cov.shape
        if mean.dim() == 1:
            self.mixture = False
            self.w = torch.tensor(1.0, dtype=self.mean.dtype, device=self.mean.device)
        elif mean.dim() == 2:
            self.mixture = True
            if w is None:
                self.w = (
                    torch.ones(self.mean.shape[0], dtype=self.mean.dtype, device=self.mean.device)
                    / self.mean.shape[0]
                )
            else:
                self.w = w
        else:
            raise ValueError("mean must be 1D (single Gaussian) or 2D (mixture of Gaussians)")

    @property
    def diag(self):
        return self._diag

    @diag.setter
    def diag(self, value):
        self._diag = value
        self.ll = self.ll_diag if value else self.ll_full

    @property
    def mixture(self):
        return self._mixture

    @mixture.setter
    def mixture(self, value):
        self._mixture = value
        self.energy = self.energy_mixture if value else self.energy_single

    def ll_diag(self, t: Tensor, x: Tensor, mu: Tensor, cov: Tensor, w: Tensor):
        r = (x.squeeze() - self.sde.mu(t) * mu).flatten()
        cov_t = self.sde.mu(t) ** 2 * cov + self.sde.sigma(t) ** 2
        icov = 1 / cov_t
        logdet = torch.sum(torch.log(2 * torch.pi * cov_t))
        ll = -0.5 * torch.sum(r**2 * icov) - 0.5 * logdet + torch.log(w)
        return ll

    def ll_full(self, t: Tensor, x: Tensor, mu: Tensor, cov: Tensor, w: Tensor):
        r = (x.squeeze() - self.sde.mu(t) * mu).flatten()
        cov_t = self.sde.mu(t) ** 2 * cov + self.sde.sigma(t) ** 2 * torch.eye(
            cov.shape[-1], dtype=cov.dtype, device=cov.device
        )
        icov = torch.linalg.inv(cov_t)
        logdet = torch.logdet(2 * torch.pi * cov_t)
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
    
    def save(self, *args, **kwargs):
        raise NotImplementedError("MVG models cannot yet be saved.")
    
    def load(self, *args, **kwargs):
        raise NotImplementedError("MVG models cannot yet be loaded.")


class MVGScoreModel(ScoreModel):
    """
    A multivariate gaussian score model.

    A multivariate gaussian score model, which can be used as a single
    multivariate gaussian or a mixture of them. Make sure to set mixture=True if
    using a mixture model. The gaussians may also be diagonal or have full
    covariance matrices, this will be automatically determined if the shape of
    ``cov`` is equal to the shape of ``mean``.

    Args:

        sde: The SDE that the score model is associated with.

        mean: The mean of the gaussian(s).

        cov: The covariance of the gaussian(s). If cov.shape == mean.shape, this
            is a diagonal covariance. Otherwise, it is a full covariance matrix
            where if mean has shape (M, *D) (or just (*D,) for single MVG) then
            the covariance matrix should have shape (M, prod(D), prod(D)) (or
            just (prod(D), prod(D)) for single MVG).

        mixture: Whether the model is a mixture of gaussians. Default is False.

        w: The weights of the mixture of gaussians (if a mixture). Default is
            equal weight.
    """

    def __init__(
        self,
        sde: SDE,
        mean: Tensor,
        cov: Tensor,
        mixture: bool = False,
        w: Optional[Tensor] = None,
        **kwargs
    ):
        super().__init__(net=NullNet(isenergy=False), sde=sde, path=None, checkpoint=None, **kwargs)
        self.sde = sde
        self.mean = mean
        self.cov = cov
        self.diag = mean.shape == cov.shape
        self.mixture = mixture
        if mixture:
            assert mean.dim() > 1, "mean must be at least 2D for a mixture of Gaussians"
            if w is None:
                self.w = torch.ones(mean.shape[0], dtype=mean.dtype, device=mean.device)
            else:
                self.w = w
            self.w = self.w / self.w.sum()

    @property
    def score_fn(self):
        if self.diag and self.mixture:
            return self.score_diag_mixture
        elif self.diag and not self.mixture:
            return self.score_diag_single
        elif not self.diag and self.mixture:
            return self.score_full_mixture
        return self.score_full_single

    def score_diag_single(self, t: Tensor, x: Tensor, *args, cov: Tensor, icov: Tensor, **kwargs):
        mu_t = self.sde.mu(t[0])
        r = mu_t * self.mean - x
        score = icov * r
        return score

    def _gamma_diag(self, r, cov, icov):
        B, K, *D = r.shape
        logdet = torch.sum(
            torch.log(2 * torch.pi * cov), dim=tuple(range(1, len(cov.shape)))
        ).reshape(1, K)
        logw = torch.log(self.w).reshape(1, K)
        logd = torch.sum(r**2 * icov.unsqueeze(0), dim=tuple(range(2, len(r.shape))))
        ll = -0.5 * logd - 0.5 * logdet + logw
        gamma = torch.exp(ll - torch.logsumexp(ll, dim=1, keepdim=True))
        return gamma

    def score_diag_mixture(self, t: Tensor, x: Tensor, *args, cov: Tensor, icov: Tensor, **kwargs):
        B, *D = x.shape
        mu_t = self.sde.mu(t[0])
        r = mu_t * self.mean.unsqueeze(0) - x.unsqueeze(1)
        gamma = self._gamma_diag(r, cov, icov).reshape(B, -1, *[1] * len(D))
        score = torch.sum(gamma * icov.unsqueeze(0) * r, dim=1)
        return score

    def score_full_single(self, t: Tensor, x: Tensor, *args, cov: Tensor, icov: Tensor, **kwargs):
        mu_t = self.sde.mu(t[0])
        r = mu_t * self.mean - x
        score = torch.vmap(lambda r_i: icov @ r_i.reshape(-1, 1))(r)
        return score.reshape(*x.shape)

    def _gamma_full(self, r, cov, icov):
        B, K, *D = r.shape
        logdet = torch.logdet(2 * torch.pi * cov).reshape(1, K)
        logw = torch.log(self.w).reshape(1, K)
        sub_logd = torch.vmap(
            lambda r_i_k, icov_k: r_i_k.reshape(1, -1) @ icov_k @ r_i_k.reshape(-1, 1)
        )
        logd = torch.vmap(sub_logd, in_dims=(0, None))(r, icov).reshape(B, K)
        ll = -0.5 * logd - 0.5 * logdet + logw
        gamma = torch.exp(ll - torch.logsumexp(ll, dim=1, keepdim=True))
        return gamma

    def score_full_mixture(self, t: Tensor, x: Tensor, *args, cov: Tensor, icov: Tensor, **kwargs):
        B, *D = x.shape
        mu_t = self.sde.mu(t[0])
        r = mu_t * self.mean.unsqueeze(0) - x.unsqueeze(1)
        gamma = self._gamma_full(r, cov, icov).reshape(B, -1, *[1] * len(D))
        sub_score = torch.vmap(
            lambda r_i_k, icov_k, gamma_k: gamma_k * icov_k @ r_i_k.reshape(-1, 1)
        )
        score = torch.vmap(sub_score, in_dims=(0, None, 0))(r, icov, gamma)
        return score.sum(dim=1).reshape(*x.shape)

    def score(self, t, x, *args, **kwargs):
        mu_t = self.sde.mu(t[0])
        sigma_t = self.sde.sigma(t[0])
        if self.diag:
            cov = mu_t**2 * self.cov + sigma_t**2
            icov = 1 / cov
        else:
            cov = mu_t**2 * self.cov + sigma_t**2 * torch.eye(
                self.cov.shape[-1], dtype=self.cov.dtype, device=self.cov.device
            )
            icov = torch.linalg.inv(cov)
        return self.score_fn(t, x, *args, cov=cov, icov=icov, **kwargs)

    def reparametrized_score(self, t, x, *args, **kwargs):
        raise RuntimeError("Reparametrized score should not be called for MVG models.")
    
    def save(self, *args, **kwargs):
        raise NotImplementedError("MVG models cannot yet be saved.")

    def load(self, *args, **kwargs):
        raise NotImplementedError("MVG models cannot yet be loaded.")
