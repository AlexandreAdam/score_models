import torch
import torch.nn as nn


class MVGScoreModel(nn.Module):
    def __init__(self, sde, mean, cov, w=None):
        super().__init__()
        self.sde = sde
        self.mean = mean
        self.cov = cov
        self.hyperparameters = {"nn_is_energy": True}
        if mean.dim() == 1:
            self.forward = self.forward_single
            self.w = torch.tensor(1.0, dtype=self.mean.dtype, device=self.mean.device)
        elif mean.dim() == 2:
            self.forward = self.forward_mixture
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

    def ll(self, t, x, mu, cov, w):
        r = x.squeeze() - mu
        cov_t = self.sde.mu(t) ** 2 * cov + self.sde.sigma(t) ** 2 * torch.eye(
            cov.shape[-1], dtype=cov.dtype, device=cov.device
        )
        icov = torch.linalg.inv(cov_t)
        det = torch.linalg.det(cov_t)
        ll = -0.5 * (r @ icov @ r.reshape(1, -1).T) - 0.5 * torch.log(det) + torch.log(w)
        return ll.unsqueeze(0)

    def forward_single(self, t, x, **kwargs):
        return -self.ll(t, x, self.mean, self.cov, self.w) * self.sde.sigma(t)

    def forward_mixture(self, t, x, **kwargs):
        ll = torch.vmap(self.ll, in_dims=(None, None, 0, 0, 0))(t, x, self.mean, self.cov, self.w)
        return -torch.logsumexp(ll, dim=0) * self.sde.sigma(t)
