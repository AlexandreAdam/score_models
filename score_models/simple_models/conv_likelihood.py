import torch
import torch.nn as nn
from torch import vmap
from torch.func import grad


class ConvolvedLikelihood(nn.Module):
    def __init__(self, sde, y, sigma_y, A=None, f=None, AAT=None):
        assert (A is not None) + (f is not None) == 1, "Either A or f must be provided (not both)"
        assert (A is not None) + (
            AAT is not None
        ) == 1, "Either A or AAT must be provided (not both)"
        super().__init__()
        self.sde = sde
        self.y = y
        self.sigma_y = sigma_y
        self.A = A
        self.f = f
        if AAT is None:
            self.AAT = A @ A.T
        else:
            self.AAT = AAT
        self.hyperparameters = {"nn_is_energy": True}

    def forward(self, t, xt, **kwargs):
        r = self.y - self.A @ xt.squeeze()
        sigma = torch.inverse(self.sigma_y + self.sde.sigma(t) ** 2 * self.AAT)
        ll = 0.5 * (r @ sigma @ r.reshape(1, r.shape[0]).T)
        return ll.unsqueeze(0) * self.sde.sigma(t)


class ConvolvedLikelihood_diag(nn.Module):
    def __init__(self, sde, y, sigma_y, AAT, f):
        super().__init__()
        self.sde = sde
        self.y = y
        self.sigma_y = sigma_y
        self.AAT = AAT
        self.f = f
        self.hyperparameters = {"nn_is_energy": True}

    def forward(self, t, xt, **kwargs):
        r = self.y - self.f(xt.squeeze())
        sigma = 1 / (self.sigma_y + self.sde.sigma(t) ** 2 * self.AAT)
        return 0.5 * torch.sum(r**2 * sigma).unsqueeze(0) * self.sde.sigma(t)
