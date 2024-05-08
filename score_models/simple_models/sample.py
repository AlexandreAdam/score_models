import torch
import torch.nn as nn
from torch.nn.functional import avg_pool2d, conv2d
from torch.func import grad
from torch import vmap
import numpy as np
from torch.func import jacrev

# from matrixkernel import matrix_to_kernel, kernel_to_matrix
from score_models import RK4_ODE
from scipy.fft import next_fast_len


class SampleScoreModel(nn.Module):
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
        # W = torch.nan_to_num(W)
        W /= W.sum()
        return t_scale * torch.sum(W * (self.samples - x) / (t_scale**2 + self.sigma_min**2), dim=0)

    @torch.no_grad()
    def forward(self, t, x, **kwargs):
        return vmap(self.single_score)(t, x)
