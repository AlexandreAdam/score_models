import torch
import torch.nn as nn
from torch import vmap
from torch.func import grad


class ConvolvedLikelihood(nn.Module):
    def __init__(self, sde, y, sigma_y, A=None, f=None, AAT=None, diag=False):
        assert (A is not None) != (f is not None), "Either A or f must be provided (not both)"
        assert (A is not None) | (AAT is not None), "Either A or AAT must be provided"
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
        assert self.sigma_y.shape == self.AAT.shape, "sigma_y and AAT must have the same shape"
        self.diag = diag
        self.hyperparameters = {"nn_is_energy": True}

    @property
    def diag(self):
        return self._diag

    @diag.setter
    def diag(self, value):
        self._diag = value
        self.forward = self.diag_forward if value else self.full_forward

    def diag_forward(self, t, xt, **kwargs):
        r = self.y - self.f(xt.squeeze())
        sigma = 1 / (self.sigma_y + self.sde.sigma(t) ** 2 * self.AAT)
        ll = 0.5 * torch.sum(r**2 * sigma)
        return ll.unsqueeze(0) * self.sde.sigma(t)

    def full_forward(self, t, xt, **kwargs):
        r = self.y - self.A @ xt.squeeze()
        sigma = torch.inverse(self.sigma_y + self.sde.sigma(t) ** 2 * self.AAT)
        ll = 0.5 * (r @ sigma @ r.reshape(1, r.shape[0]).T)
        return ll.unsqueeze(0) * self.sde.sigma(t)
