from typing import Optional, Callable

import torch
from torch import nn
from score_models.definitions import default_init
from .conv_layers import conv1x1, conv3x3
from .up_or_downsampling import *


SQRT2 = 1.41421356237

__all__ = ['ResnetBlockBigGANpp']


class ResnetBlockBigGANpp(nn.Module):
    def __init__(
            self,
            act: Callable,
            in_ch: int,
            out_ch: Optional[int] = None,
            temb_dim: Optional[int] = None,
            up:  bool = False,
            down: bool = False,
            dropout: float = 0.,
            fir: bool = False,
            fir_kernel: tuple[int] = (1, 3, 3, 1),
            skip_rescale: bool = True,
            init_scale: float = 0.,
            factor: int = 2,
            dimensions: int = 2
    ):
        super().__init__()
        assert dimensions in [1, 2, 3]
        self.dimensions = dimensions
        out_ch = out_ch if out_ch is not None else in_ch
        self.up = up
        self.down = down
        self.fir = fir
        self.fir_kernel = fir_kernel
        self.skip_rescale = skip_rescale
        self.act = act
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.factor = factor 

        self.GroupNorm_0 = nn.GroupNorm(num_groups=max(min(in_ch // 4, 32), 1), num_channels=in_ch, eps=1e-6)
        self.Conv_0 = conv3x3(in_ch, out_ch, dimensions=dimensions)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            with torch.no_grad():
                self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
                nn.init.zeros_(self.Dense_0.bias)

        self.GroupNorm_1 = nn.GroupNorm(num_groups=max(min(out_ch // 4, 32), 1), num_channels=out_ch, eps=1e-6)
        self.Dropout_0 = nn.Dropout(dropout)
        # suppress skip connection at initialization
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale, dimensions=dimensions)
        if in_ch != out_ch or up or down:
            self.Conv_2 = conv1x1(in_ch, out_ch, dimensions=dimensions)

    def forward(self, x, temb=None):
        B, *_ = x.shape
        h = self.act(self.GroupNorm_0(x))

        if self.up:
            if self.fir:
                h = upsample(h, self.fir_kernel, factor=self.factor, dimensions=self.dimensions)
                x = upsample(x, self.fir_kernel, factor=self.factor, dimensions=self.dimensions)
            else:
                h = naive_upsample(h, factor=self.factor, dimensions=self.dimensions)
                x = naive_upsample(x, factor=self.factor, dimensions=self.dimensions)
        elif self.down:
            if self.fir:
                h = downsample(h, self.fir_kernel, factor=self.factor, dimensions=self.dimensions)
                x = downsample(x, self.fir_kernel, factor=self.factor, dimensions=self.dimensions)
            else:
                h = naive_downsample(h, factor=self.factor, dimensions=self.dimensions)
                x = naive_downsample(x, factor=self.factor, dimensions=self.dimensions)

        h = self.Conv_0(h)
        # Add bias to each feature map conditioned on the time embedding
        if temb is not None:
            h += self.Dense_0(self.act(temb)).view(B, -1, *[1]*self.dimensions)
        h = self.act(self.GroupNorm_1(h))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)

        if self.in_ch != self.out_ch or self.up or self.down:
            x = self.Conv_2(x)

        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / SQRT2

