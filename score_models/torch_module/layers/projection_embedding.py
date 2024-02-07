import torch
from torch.func import vmap
from torch import nn
from numpy import pi

left_matmul = vmap(torch.matmul, in_dims=(None,  0))


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, t):
        t_proj = t[:, None] * self.W[None, :] * 2 * pi
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=1)


class PositionalEncoding(nn.Module):
    """ More classical encoding of a vector """
    def __init__(self, channels, embed_dim, scale=30.):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2, channels) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = left_matmul(self.W, x) * 2 * pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=1)

