import torch
from torch import nn
from .conv1dsame import Conv1dSame
from ..definitions import default_init
from .up_or_downsampling1d import upsample_1d, naive_upsample_1d, downsample_1d, naive_downsample_1d, Conv1d


SQRT2 = 1.41421356237


def conv3(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1.):
    """3x3 convolution with DDPM initialization."""
    conv = Conv1dSame(in_planes, out_planes, kernel_size=3, stride=stride, dilation=dilation, bias=bias)
    conv.conv.weight.data *= init_scale
    conv.conv.bias.data *= init_scale
    return conv


def conv1(in_planes, out_planes, stride=1, bias=True, init_scale=1.):
    """1x1 convolution with DDPM initialization."""
    conv = Conv1dSame(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)
    with torch.no_grad():
        conv.conv.weight.data = default_init(init_scale)(conv.conv.weight.data.shape)
        nn.init.zeros_(conv.conv.bias)
    return conv


class ResnetBlockBigGANpp1d(nn.Module):
    def __init__(
            self,
            act,
            in_ch,
            out_ch=None,
            temb_dim=None,
            up=False,
            down=False,
            dropout=0.1,
            fir=False,
            fir_kernel=(1, 3, 3, 1),
            skip_rescale=True,
            init_scale=0.
    ):
        super().__init__()

        out_ch = out_ch if out_ch else in_ch
        self.GroupNorm_0 = nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6)
        self.up = up
        self.down = down
        self.fir = fir
        self.fir_kernel = fir_kernel

        self.Conv_0 = conv3(in_ch, out_ch)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            with torch.no_grad():
                self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
                nn.init.zeros_(self.Dense_0.bias)

        self.GroupNorm_1 = nn.GroupNorm(num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6)
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3(out_ch, out_ch, init_scale=init_scale)
        if in_ch != out_ch or up or down:
            self.Conv_2 = conv1(in_ch, out_ch)

        self.skip_rescale = skip_rescale
        self.act = act
        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(self, x, temb=None):
        h = self.act(self.GroupNorm_0(x))

        if self.up:
            if self.fir:
                h = upsample_1d(h, self.fir_kernel, factor=2)
                x = upsample_1d(x, self.fir_kernel, factor=2)
            else:
                h = naive_upsample_1d(h, factor=2)
                x = naive_upsample_1d(x, factor=2)
        elif self.down:
            if self.fir:
                h = downsample_1d(h, self.fir_kernel, factor=2)
                x = downsample_1d(x, self.fir_kernel, factor=2)
            else:
                h = naive_downsample_1d(h, factor=2)
                x = naive_downsample_1d(x, factor=2)

        h = self.Conv_0(h)
        # Add bias to each feature map conditioned on the time embedding
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None]
        h = self.act(self.GroupNorm_1(h))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)

        if self.in_ch != self.out_ch or self.up or self.down:
            x = self.Conv_2(x)

        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / SQRT2