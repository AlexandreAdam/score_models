"""
Code ported from Yang Song's repo https://github.com/yang-song/score_sde_pytorch/blob/main/models/layers.py
"""
from torch import nn
from score_models.definitions import default_init
from .conv2dsame import Conv2dSame
import torch


def conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1.):
    """3x3 convolution with DDPM initialization."""
    conv = Conv2dSame(in_planes, out_planes, kernel_size=3, stride=stride, dilation=dilation, bias=bias)
    return conv


def conv1x1(in_planes, out_planes, stride=1, bias=True, init_scale=1.):
    """1x1 convolution with DDPM initialization."""
    conv = Conv2dSame(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)
    return conv


class NIN(nn.Module):
    """
    Equivalent to a 1x1 conv, act upon the channel dimension
    """
    def __init__(self, in_dim, num_units, init_scale=0.1):
        super().__init__()
        self.W = nn.Parameter(default_init(scale=init_scale)((num_units, in_dim)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        y = torch.einsum("ij, ...j -> ...i", self.W, x) + self.b
        return y.permute(0, 3, 1, 2)


class DDPMResnetBlock(nn.Module):
    """The ResNet Blocks used in DDPM."""
    def __init__(self, act, in_ch, out_ch=None, temb_dim=None, conv_shortcut=False, dropout=0.1):
        super().__init__()
        if out_ch is None:
            out_ch = in_ch
        self.GroupNorm_0 = nn.GroupNorm(num_groups=32, num_channels=in_ch, eps=1e-6)
        self.act = act
        self.Conv_0 = conv3x3(in_ch, out_ch)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
            nn.init.zeros_(self.Dense_0.bias)

        self.GroupNorm_1 = nn.GroupNorm(num_groups=32, num_channels=out_ch, eps=1e-6)
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=0.)
        if in_ch != out_ch:
            if conv_shortcut:
                self.Conv_2 = conv3x3(in_ch, out_ch)
            else:
                self.NIN_0 = NIN(in_ch, out_ch)
        self.out_ch = out_ch
        self.in_ch = in_ch
        self.conv_shortcut = conv_shortcut

    def forward(self, x, temb=None):
        B, C, H, W = x.shape
        assert C == self.in_ch
        out_ch = self.out_ch if self.out_ch else self.in_ch
        h = self.act(self.GroupNorm_0(x))
        h = self.Conv_0(h)
        # Add bias to each feature map conditioned on the time embedding
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None]
        h = self.act(self.GroupNorm_1(h))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)
        if C != out_ch:
            if self.conv_shortcut:
                x = self.Conv_2(x)
            else:
                x = self.NIN_0(x)
        return x + h
