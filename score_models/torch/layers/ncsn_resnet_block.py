"""
Code ported from Yang Song's repo https://github.com/yang-song/score_sde_pytorch/blob/main/models/layers.py
with hacked padding
"""
from torch import nn
from functools import partial
from .conv2dsame import Conv2dSame


def conv1x1(in_planes, out_planes, stride=1, bias=True, dilation=1):
    """1x1 convolution. Same as NCSNv1/v2."""
    conv = Conv2dSame(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias, dilation=dilation)
    return conv


def conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1):
    """3x3 convolution with PyTorch initialization. Same as NCSNv1/NCSNv2."""
    conv = Conv2dSame(in_planes, out_planes, stride=stride, bias=bias, dilation=dilation, kernel_size=3)
    return conv


class ConvMeanPool(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True):
        super().__init__()
        self.conv = Conv2dSame(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)

    def forward(self, inputs):
        output = self.conv(inputs)
        output = sum([output[:, :, ::2, ::2], output[:, :, 1::2, ::2], output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
        return output


class NCSNResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, resample=None, act=nn.ELU(), normalization=nn.InstanceNorm2d, dilation=1):
        super().__init__()
        self.non_linearity = act
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resample = resample
        self.normalization = normalization
        if resample == 'down':
            if dilation > 1:
                self.conv1 = conv3x3(input_dim, input_dim, dilation=dilation)
                self.normalize2 = normalization(input_dim)
                self.conv2 = conv3x3(input_dim, output_dim, dilation=dilation)
                conv_shortcut = partial(conv3x3, dilation=dilation)
            else:
                self.conv1 = conv3x3(input_dim, input_dim)
                self.normalize2 = normalization(input_dim)
                self.conv2 = ConvMeanPool(input_dim, output_dim, 3)
                conv_shortcut = partial(ConvMeanPool, kernel_size=1)

        elif resample is None:
            if dilation > 1:
                conv_shortcut = partial(conv3x3, dilation=dilation)
                self.conv1 = conv3x3(input_dim, output_dim, dilation=dilation)
                self.normalize2 = normalization(output_dim)
                self.conv2 = conv3x3(output_dim, output_dim, dilation=dilation)
            else:
                # conv_shortcut = nn.Conv2d ### Something wierd here.
                conv_shortcut = partial(conv1x1)
                self.conv1 = conv3x3(input_dim, output_dim)
                self.normalize2 = normalization(output_dim)
                self.conv2 = conv3x3(output_dim, output_dim)
        else:
            raise Exception('invalid resample value')

        if output_dim != input_dim or resample is not None:
            self.shortcut = conv_shortcut(input_dim, output_dim)

        self.normalize1 = normalization(input_dim)

    def forward(self, x):
        output = self.normalize1(x)
        output = self.non_linearity(output)
        output = self.conv1(output)
        output = self.normalize2(output)
        output = self.non_linearity(output)
        output = self.conv2(output)

        if self.output_dim == self.input_dim and self.resample is None:
            shortcut = x
        else:
            shortcut = self.shortcut(x)
        return shortcut + output
