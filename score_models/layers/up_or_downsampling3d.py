import torch
import torch.nn.functional as F
import numpy as np
from .upfirdn3d import upfirdn3d

__all__ = ["naive_upsample_3d", "naive_downsample_3d", 
           "upsample_3d", "downsample_3d",
           "conv_downsample_3d", "upsample_conv_3d"
           ]


def naive_upsample_3d(x, factor=2):
    print("hello")
    _N, C, H, W, D = x.shape
    x = torch.reshape(x, (-1, C, H, 1, W, 1, D, 1))
    x = x.repeat(1, 1, 1, factor, 1, factor, 1, factor)
    return torch.reshape(x, (-1, C, H * factor, W * factor, D * factor))


def naive_downsample_3d(x, factor=2):
    _N, C, H, W, D = x.shape
    x = torch.reshape(x, (-1, C, H // factor, factor, W // factor, factor, D // factor, factor))
    return torch.mean(x, dim=(3, 5, 7))


def upsample_conv_3d(x, w, k=None, factor=2, gain=1):
    """Fused `upsample_3d()` followed by `tf.nn.conv3d()`.
       Padding is performed only once at the beginning, not between the
       operations.
       The fused op is considerably more efficient than performing the same
       calculation using standard ops. It supports gradients of arbitrary order.
       Args:
         x:            Input tensor of the shape `[N, C, H, W, D]`
         w:            Weight tensor of the shape `[filterH, filterW, filterD, inChannels,
           outChannels]`. Grouped convolution can be performed by `inChannels =
           x.shape[0] // numGroups`.
         k:            FIR filter of the shape `[firH, firW, firD]` or `[firN]`
           (separable). The default is `[1] * factor`, which corresponds to
           nearest-neighbor upsampling.
         factor:       Integer upsampling factor (default: 2).
         gain:         Scaling factor for signal magnitude (default: 1.0).
       Returns:
         Tensor of the shape `[N, C, H * factor, W * factor, D * factor]` and same datatype as `x`.
    """

    assert isinstance(factor, int) and factor >= 1
    outC, inC, convH, convW, convD = w.shape
    assert convW == convH
    assert convW == convD

    # Setup filter kernel.
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * (gain * (factor ** 3))
    p = (k.shape[0] - factor) - (convW - 1)

    # Determine data dimensions.
    stride = [factor, factor, factor]
    output_shape = ((x.shape[2] - 1) * factor + convH, (x.shape[3] - 1) * factor + convW, (x.shape[4] - 1) * factor + convD)
    output_padding = (output_shape[0] - (x.shape[2] - 1) * stride[0] - convH,
                      output_shape[1] - (x.shape[3] - 1) * stride[1] - convW,
                      output_shape[2] - (x.shape[3] - 1) * stride[2] - convD)
    assert output_padding[0] >= 0 and output_padding[1] >= 0 and output_padding[2] >= 0
    num_groups = x.shape[1] // inC

    w = torch.reshape(w, (num_groups, -1, inC, convH, convW, convD))
    w = torch.flip(w, [3, 4, 5]).permute(0, 2, 1, 3, 4, 5)
    w = torch.reshape(w, (num_groups * inC, -1, convH, convW, convD))

    x = F.conv_transpose3d(x, w, stride=stride, output_padding=output_padding, padding=0)
    return upfirdn3d(x, torch.tensor(k, device=x.device), pad=((p + 1) // 2 + factor - 1, p // 2 + 1))


def conv_downsample_3d(x, w, k=None, factor=2, gain=1):
    """Fused `tf.nn.conv3d()` followed by `downsample_3d()`.
      Padding is performed only once at the beginning, not between the operations.
      The fused op is considerably more efficient than performing the same
      calculation using standard ops. It supports gradients of arbitrary order.
      Args:
          x:            Input tensor of the shape `[N, C, H, W, D]`.
          w:            Weight tensor of the shape `[filterH, filterW, filterD, inChannels,
            outChannels]`. Grouped convolution can be performed by `inChannels =
            x.shape[0] // numGroups`.
          k:            FIR filter of the shape `[firH, firW, firD]` or `[firN]`
            (separable). The default is `[1] * factor`, which corresponds to
            average pooling.
          factor:       Integer downsampling factor (default: 2).
          gain:         Scaling factor for signal magnitude (default: 1.0).
      Returns:
          Tensor of the shape `[N, C, H // factor, W // factor, D // factor]` and same datatype as `x`.
    """
    assert isinstance(factor, int) and factor >= 1
    _outC, _inC, convH, convW, convD = w.shape
    assert convW == convH
    assert convW == convD
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * gain
    p = (k.shape[0] - factor) + (convW - 1)
    s = [factor, factor, factor]
    x = upfirdn3d(x, torch.tensor(k, device=x.device), pad=((p + 1) // 2, p // 2))
    return F.conv3d(x, w, stride=s, padding=0)


def _setup_kernel(k):
    k = np.asarray(k, dtype=np.float32)
    if k.ndim == 1:
        m = np.outer(k, k)
        k = np.multiply.outer(m, k)
    k /= np.sum(k)
    assert k.ndim == 3
    assert k.shape[0] == k.shape[1]
    assert k.shape[0] == k.shape[2]
    return k


def upsample_3d(x, k=None, factor=2, gain=1):
    r"""Upsample a batch of 3D images with the given filter.
      Accepts a batch of 3D images of the shape `[N, C, H, W, D]`
      and upsamples each image with the given filter. The filter is normalized so
      that if the input pixels are constant, they will be scaled by the specified
      `gain`.
      Pixels outside the image are assumed to be zero, and the filter is padded
      with zeros so that its shape is a multiple of the upsampling factor.
      Args:
          x:            Input tensor of the shape `[N, C, H, W, D]`.
          k:            FIR filter of the shape `[firH, firW, firD]` or `[firN]`
            (separable). The default is `[1] * factor`, which corresponds to
            nearest-neighbor upsampling.
          factor:       Integer upsampling factor (default: 2).
          gain:         Scaling factor for signal magnitude (default: 1.0).
      Returns:
          Tensor of the shape `[N, C, H * factor, W * factor, D * factor]`
    """
    assert isinstance(factor, int) and factor >= 1
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * (gain * (factor ** 3))
    p = k.shape[0] - factor
    return upfirdn3d(x, torch.tensor(k, device=x.device), up=factor, pad=((p + 1) // 2 + factor - 1, p // 2))


def downsample_3d(x, k=None, factor=2, gain=1):
    r"""Downsample a batch of 3D images with the given filter.
      Accepts a batch of 3D images of the shape `[N, C, H, W, D]`.
      and downsamples each image with the given filter. The filter is normalized
      so that if the input pixels are constant, they will be scaled by the specified
      `gain`.
      Pixels outside the image are assumed to be zero, and the filter is padded
      with zeros so that its shape is a multiple of the downsampling factor.
      Args:
          x:            Input tensor of the shape `[N, C, H, W, D]`
          k:            FIR filter of the shape `[firH, firW, firD]` or `[firN]`
            (separable). The default is `[1] * factor`, which corresponds to
            average pooling.
          factor:       Integer downsampling factor (default: 2).
          gain:         Scaling factor for signal magnitude (default: 1.0).
      Returns:
          Tensor of the shape `[N, C, H // factor, W // factor, D // factor]`
    """

    assert isinstance(factor, int) and factor >= 1
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * gain
    p = k.shape[0] - factor
    return upfirdn3d(x, torch.tensor(k, device=x.device), down=factor, pad=((p + 1) // 2, p // 2))


