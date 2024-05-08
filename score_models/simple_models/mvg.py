import torch
import torch.nn as nn


class MVGScoreModel(nn.Module):
    def __init__(self, sde, mean, cov):
        super().__init__()
        self.sde = sde
        self.mean = mean
        self.cov = cov
        self.hyperparameters = {"nn_is_energy": True}

    def forward(self, t, x, **kwargs):
        t_mu = self.sde.mu(t)
        t_scale = self.sde.sigma(t)

        r = (x.squeeze() - t_mu * self.mean).flatten()
        icov = torch.linalg.inv(t_mu**2 * self.cov + t_scale**2 * torch.eye(self.cov.shape[-1], device=self.cov.device))
        nll = 0.5 * (r @ icov @ r.reshape(1, r.shape[0]).T)
        return nll.unsqueeze(0) * t_scale
