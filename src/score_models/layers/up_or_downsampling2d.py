"""Layers used for up-sampling or down-sampling images.
Many functions are ported from https://github.com/NVlabs/stylegan2.

Code ported from https://github.com/yang-song/score_sde_pytorch/blob/main/models/up_or_down_sampling.py
"""
import torch
import torch.nn.functional as F
import numpy as np
from .upfirdn2d import upfirdn2d

__all__ = ["naive_upsample_2d", "naive_downsample_2d", 
           "upsample_2d", "downsample_2d",
           "conv_downsample_2d", "upsample_conv_2d"
           ]

def naive_upsample_2d(x, factor=2):
    _N, C, H, W = x.shape
    x = torch.reshape(x, (-1, C, H, 1, W, 1))
    x = x.repeat(1, 1, 1, factor, 1, factor)
    return torch.reshape(x, (-1, C, H * factor, W * factor))


def naive_downsample_2d(x, factor=2):
    _N, C, H, W = x.shape
    x = torch.reshape(x, (-1, C, H // factor, factor, W // factor, factor))
    return torch.mean(x, dim=(3, 5))


def upsample_conv_2d(x, w, k=None, factor=2, gain=1):
    """Fused `upsample_2d()` followed by `tf.nn.conv2d()`.
       Padding is performed only once at the beginning, not between the
       operations.
       The fused op is considerably more efficient than performing the same
       calculation
       using standard TensorFlow ops. It supports gradients of arbitrary order.
       Args:
         x: Input tensor of the shape `[N, C, H, W]` 
         w: Weight tensor of the shape `[filterH, filterW, inChannels,
           outChannels]`. Grouped convolution can be performed by `inChannels =
           x.shape[0] // numGroups`.
         k: FIR filter of the shape `[firH, firW]` or `[firN]`
           (separable). The default is `[1] * factor`, which corresponds to
           nearest-neighbor upsampling.
         factor: Integer upsampling factor (default: 2).
         gain:  Scaling factor for signal magnitude (default: 1.0).
       Returns:
         Tensor of the shape `[N, C, H * factor, W * factor]`
    """

    assert isinstance(factor, int) and factor >= 1

    # Check weight shape.
    assert len(w.shape) == 4
    convH = w.shape[2]
    convW = w.shape[3]
    inC = w.shape[1]
    outC = w.shape[0]
    assert convW == convH

    # Setup filter kernel.
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * (gain * (factor ** 2))
    p = (k.shape[0] - factor) - (convW - 1)

    stride = (factor, factor)

    # Determine data dimensions.
    stride = [factor, factor]
    output_shape = ((x.shape[2] - 1) * factor + convH, (x.shape[3] - 1) * factor + convW)
    output_padding = (output_shape[0] - (x.shape[2] - 1) * stride[0] - convH,
                      output_shape[1] - (x.shape[3] - 1) * stride[1] - convW)
    assert output_padding[0] >= 0 and output_padding[1] >= 0
    num_groups = x.shape[1]  // inC

    w = torch.reshape(w, (num_groups, -1, inC, convH, convW))
    w = torch.flip(w, dims=(1, 2)).permute(0, 2, 1, 3, 4)
    w = torch.reshape(w, (num_groups * inC, -1, convH, convW))
    
    x = F.conv_transpose2d(x, w, stride=stride, output_padding=output_padding, padding=0)
    return upfirdn2d(x, torch.tensor(k, device=x.device),
                     pad=((p + 1) // 2 + factor - 1, p // 2 + 1))


def conv_downsample_2d(x, w, k=None, factor=2, gain=1):
    """Fused `tf.nn.conv2d()` followed by `downsample_2d()`.
      Padding is performed only once at the beginning, not between the operations.
      The fused op is considerably more efficient than performing the same
      calculation
      using standard TensorFlow ops. It supports gradients of arbitrary order.
      Args:
          x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
            C]`.
          w:            Weight tensor of the shape `[filterH, filterW, inChannels,
            outChannels]`. Grouped convolution can be performed by `inChannels =
            x.shape[0] // numGroups`.
          k:            FIR filter of the shape `[firH, firW]` or `[firN]`
            (separable). The default is `[1] * factor`, which corresponds to
            average pooling.
          factor:       Integer downsampling factor (default: 2).
          gain:         Scaling factor for signal magnitude (default: 1.0).
      Returns:
          Tensor of the shape `[N, C, H // factor, W // factor]` or
          `[N, H // factor, W // factor, C]`, and same datatype as `x`.
    """

    assert isinstance(factor, int) and factor >= 1
    _outC, _inC, convH, convW = w.shape
    assert convW == convH
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * gain
    p = (k.shape[0] - factor) + (convW - 1)
    s = [factor, factor]
    x = upfirdn2d(x, torch.tensor(k, device=x.device),
                  pad=((p + 1) // 2, p // 2))
    return F.conv2d(x, w, stride=s, padding=0)


def _setup_kernel(k):
    k = np.asarray(k, dtype=np.float32)
    if k.ndim == 1:
      k = np.outer(k, k)
    k /= np.sum(k)
    assert k.ndim == 2
    assert k.shape[0] == k.shape[1]
    return k


def upsample_2d(x, k=None, factor=2, gain=1):
    r"""Upsample a batch of 2D images with the given filter.
      Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]`
      and upsamples each image with the given filter. The filter is normalized so
      that
      if the input pixels are constant, they will be scaled by the specified
      `gain`.
      Pixels outside the image are assumed to be zero, and the filter is padded
      with
      zeros so that its shape is a multiple of the upsampling factor.
      Args:
          x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
            C]`.
          k:            FIR filter of the shape `[firH, firW]` or `[firN]`
            (separable). The default is `[1] * factor`, which corresponds to
            nearest-neighbor upsampling.
          factor:       Integer upsampling factor (default: 2).
          gain:         Scaling factor for signal magnitude (default: 1.0).
      Returns:
          Tensor of the shape `[N, C, H * factor, W * factor]`
    """
    assert isinstance(factor, int) and factor >= 1
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * (gain * (factor ** 2))
    p = k.shape[0] - factor
    return upfirdn2d(x, torch.tensor(k, device=x.device),
                     up=factor, pad=((p + 1) // 2 + factor - 1, p // 2))


def downsample_2d(x, k=None, factor=2, gain=1):
    r"""Downsample a batch of 2D images with the given filter.
      Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]`
      and downsamples each image with the given filter. The filter is normalized
      so that
      if the input pixels are constant, they will be scaled by the specified
      `gain`.
      Pixels outside the image are assumed to be zero, and the filter is padded
      with
      zeros so that its shape is a multiple of the downsampling factor.
      Args:
          x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
            C]`.
          k:            FIR filter of the shape `[firH, firW]` or `[firN]`
            (separable). The default is `[1] * factor`, which corresponds to
            average pooling.
          factor:       Integer downsampling factor (default: 2).
          gain:         Scaling factor for signal magnitude (default: 1.0).
      Returns:
          Tensor of the shape `[N, C, H // factor, W // factor]`
    """

    assert isinstance(factor, int) and factor >= 1
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * gain
    p = k.shape[0] - factor
    return upfirdn2d(x, torch.tensor(k, device=x.device),
                     down=factor, pad=((p + 1) // 2, p // 2))
