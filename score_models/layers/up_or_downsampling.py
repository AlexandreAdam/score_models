from .up_or_downsampling1d import *
from .up_or_downsampling2d import *
from .up_or_downsampling3d import *

__all__ = ["naive_upsample", "naive_downsample", 
           "upsample", "downsample",
           "conv_downsample", "upsample_conv"
           ]
 
def naive_upsample(x, factor=2, dimensions=2):
    if dimensions == 1:
        return naive_upsample_1d(x, factor)
    elif dimensions == 2:
        return naive_upsample_2d(x, factor)
    elif dimensions == 3:
        return naive_upsample_3d(x, factor)

def naive_downsample(x, factor=2, dimensions=2):
    if dimensions == 1:
        return naive_downsample_1d(x, factor)
    elif dimensions == 2:
        return naive_downsample_2d(x, factor)
    elif dimensions == 3:
        return naive_downsample_3d(x, factor)
 
def upsample(x, k=None, factor=2, gain=1, dimensions=2):
    if dimensions == 1:
        return upsample_1d(x, k, factor, gain)
    elif dimensions == 2:
        return upsample_2d(x, k, factor, gain)
    elif dimensions == 3:
        return upsample_3d(x, k, factor, gain)

def downsample(x, k=None, factor=2, gain=1, dimensions=2):
    if dimensions == 1:
        return downsample_1d(x, k, factor, gain)
    elif dimensions == 2:
        return downsample_2d(x, k, factor, gain)
    elif dimensions == 3:
        return downsample_3d(x, k, factor, gain)

def upsample_conv(x, w, k=None, factor=2, gain=1, dimensions=2):
    if dimensions == 1:
        return upsample_conv_1d(x, w, k, factor, gain)
    elif dimensions == 2:
        return upsample_conv_2d(x, w, k, factor, gain)
    elif dimensions == 3:
        return upsample_conv_3d(x, w, k, factor, gain)

def conv_downsample(x, w, k=None, factor=2, gain=1, dimensions=2):
    if dimensions == 1:
        return conv_downsample_1d(x, w, k, factor, gain)
    elif dimensions == 2:
        return conv_downsample_2d(x, w, k, factor, gain)
    elif dimensions == 3:
        return conv_downsample_3d(x, w, k, factor, gain)
