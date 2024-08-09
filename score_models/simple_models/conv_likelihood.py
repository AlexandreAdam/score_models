from typing import Callable, Union, Tuple, Optional
import torch
from torch import Tensor
import torch.nn as nn
from torch import vmap
from torch.func import grad
import numpy as np
import matplotlib.pyplot as plt


class ConvolvedLikelihood(nn.Module):
    @torch.no_grad()
    def __init__(self, sde, y, Sigma_y, x_shape, A=None, f=None, AAT=None, diag=False):
        assert (A is not None) != (f is not None), "Either A or f must be provided (not both)"
        assert (A is not None) | (AAT is not None), "Either A or AAT must be provided"
        super().__init__()
        self.sde = sde
        self.y = y
        self.Sigma_y = Sigma_y
        self.x_shape = x_shape
        self.y_shape = y.shape
        if A is not None:
            self.A = A.reshape(np.prod(self.y_shape), np.prod(x_shape))
        else:
            self.A = A
        self.f = f
        if AAT is None:
            self.AAT = self.A @ self.A.T
        else:
            self.AAT = AAT

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
        sigma = 1 / (self.Sigma_y + self.sde.sigma(t) ** 2 * self.AAT)
        ll = 0.5 * torch.sum(r**2 * sigma).unsqueeze(0)
        return ll.unsqueeze(0) * self.sde.sigma(t)

    def full_forward(self, t, xt, **kwargs):
        r = self.y.reshape(-1) - self.A @ xt.reshape(-1)
        sigma = torch.linalg.inv(self.Sigma_y + self.sde.sigma(t[0]) ** 2 * self.AAT)
        ll = 0.5 * (r @ sigma @ r.reshape(1, r.shape[0]).T)
        return ll.unsqueeze(0) * self.sde.sigma(t)
