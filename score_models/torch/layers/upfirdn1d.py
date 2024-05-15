import torch
from torch.nn import functional as F


def upfirdn1d(input, kernel, up=1, down=1, pad=(0, 0)):
    out = upfirdn1d_native(
        input, kernel, up, down, pad[0], pad[1]
    )
    return out


def upfirdn1d_native(x, kernel, up, down, pad_0, pad_1):
    _, channel, in_h = x.shape
    kernel_h = kernel.shape[0]
    
    # Interweave zeros between input pixels (if upsampling)
    out = x.view(-1, in_h, 1, 1)
    out = F.pad(out, [0, 0, 0, up - 1])
    out = out.view(-1, up*in_h, 1)

    # Pad with zeros at the boundaries 
    out = F.pad(
        out,
        [
            0, 
            0, 
            max(pad_0, 0), 
            max(pad_1, 0)
        ]
    )
    out = out[
        :,
        max(-pad_0, 0) : out.shape[1] - max(-pad_1, 0),
        :,
    ]

    # Reshape for convolution (NHI -> NIH)
    out = out.permute(0, 2, 1)
    out = out.view([-1, 1, up * in_h + pad_0 + pad_1])

    # Flip spatial + reshape kernel for convolution (H -> OIH).
    w = torch.flip(kernel, [0]).view(1, 1, kernel_h)
    out = F.conv1d(out, w)
    
    # Permute back to NHO
    out_h = up*in_h + pad_0 + pad_1 - kernel_h + 1
    out = out.view(-1, 1, out_h)
    out = out.permute(0, 2, 1)
    
    # Downsample (if needed)
    out = out[:, ::down, :]

    out_h = (in_h * up + pad_0 + pad_1 - kernel_h) // down + 1
    return out.view(-1, channel, out_h)

