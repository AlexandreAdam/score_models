import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from .spectral_normalization import SpectralNorm

"""
Implements same padding behavior as in Tensorflow
"""


class Conv2dSame(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: Union[str, int] = 0,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            spectral_norm: bool = False,
            device=None,
            dtype=None
    ):
        super(Conv2dSame, self).__init__()
        if spectral_norm:
            sp_norm = SpectralNorm
        else:
            sp_norm = lambda x: x
        self.conv = sp_norm(nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding="valid" if stride > 1 else "same",
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        ))
        self.stride = stride
        self.dilation = dilation
        self.kernel_size = kernel_size

    def forward(self, x):
        if self.stride > 1:
            # compute some padding
            b, c, h, w = x.shape
            h_o, w_o = h // self.stride, w // self.stride
            p0 = ((h_o - 1) * self.stride + 1 + self.dilation * (self.kernel_size -1) - h) // 2
            p1 = ((w_o - 1) * self.stride + 1 + self.dilation * (self.kernel_size -1) - w) // 2
            x = F.pad(x, (p0, p0+1, p1, p1+1)) # It might be controversial to do this, but this is how it is done in Tensorflow...
        x = self.conv(x)
        return x


class ConvTranspose2dSame(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: Union[str, int] = 0,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            spectral_norm: bool = False,
            device=None,
            dtype=None
    ):
        super(ConvTranspose2dSame, self).__init__()
        if spectral_norm:
            sp_norm = SpectralNorm
        else:
            sp_norm = lambda x: x
        self.conv = sp_norm(nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding=dilation * (kernel_size - 1) // 2,
            output_padding=stride - 1,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        ))
        self.stride = stride
        self.dilation = dilation
        self.kernel_size = kernel_size

    def forward(self, x):
        x = self.conv(x)
        return x
