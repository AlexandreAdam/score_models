import torch
from torch import nn
from numpy import pi


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, t):
        t_proj = t[:, None] * self.W[None, :] * 2 * pi
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)
