import torch
from torch import nn
from score_models.definitions import default_init
from .conv1dsame import Conv1dSame
from .conv2dsame import Conv2dSame
from .conv3dsame import Conv3dSame

CONVS = {1: Conv1dSame, 2: Conv2dSame, 3: Conv3dSame} 

__all__ = ("conv1x1", "conv3x3")


def conv1x1(in_planes, out_planes, stride=1, bias=True, init_scale=1., dimensions=2):
    """1x1 convolution with DDPM initialization."""
    conv = CONVS[dimensions](in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)
    with torch.no_grad():
        conv.conv.weight.data = default_init(init_scale)(conv.conv.weight.data.shape)
        nn.init.zeros_(conv.conv.bias)
    return conv


def conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1., dimensions=2):
    """3x3 convolution with DDPM initialization."""
    conv = CONVS[dimensions](in_planes, out_planes, kernel_size=3, stride=stride, dilation=dilation, bias=bias)
    with torch.no_grad():
        conv.conv.weight.data = default_init(init_scale)(conv.conv.weight.data.shape)
        nn.init.zeros_(conv.conv.bias)
    return conv

