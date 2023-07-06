import torch
import torch.nn.functional as F
from torch import nn

from .up_or_downsampling1d import upsample_conv_1d, conv_downsample_1d
from .up_or_downsampling2d import upsample_conv_2d, conv_downsample_2d
from .up_or_downsampling3d import upsample_conv_3d, conv_downsample_3d


class StyleGANConv(nn.Module):
    """Conv2d layer with optimal upsampling and downsampling (StyleGAN2)."""
    def __init__(self, in_ch, out_ch, kernel, up=False, down=False,
                 resample_kernel=(1, 3, 3, 1),
                 use_bias=True,
                 kernel_init=None,
                 dimensions=2
                 ):
        super().__init__()
        assert not (up and down)
        assert kernel >= 1 and kernel % 2 == 1
        assert dimensions in [1, 2, 3]
        self.weight = nn.Parameter(torch.zeros(out_ch, in_ch, *[kernel]*dimensions))
        if kernel_init is not None:
            self.weight.data = kernel_init(self.weight.data.shape)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_ch))

        self.up = up
        self.down = down
        self.resample_kernel = resample_kernel
        self.kernel = kernel
        self.use_bias = use_bias
        self.dimensions = dimensions
        
        if dimensions == 1:
            self.upsample = upsample_conv_1d
            self.downsample = conv_downsample_1d
            self.conv = F.conv1d
        elif dimensions == 2:
            self.upsample = upsample_conv_2d
            self.downsample = conv_downsample_2d
            self.conv = F.conv2d
        elif dimensions == 3:
            self.upsample = upsample_conv_3d
            self.downsample = conv_downsample_3d
            self.conv = F.conv3d

    def forward(self, x):
        if self.up:
            x = self.upsample(x, self.weight, k=self.resample_kernel)
        elif self.down:
            x = self.downsample(x, self.weight, k=self.resample_kernel)
        else:
            x = self.conv(x, self.weight, stride=1, padding=self.kernel // 2)
        if self.use_bias:
            x = x + self.bias.reshape(1, -1, *[1]*self.dimensions)
        return x


