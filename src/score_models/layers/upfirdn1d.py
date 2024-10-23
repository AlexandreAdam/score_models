import torch
from torch.nn import functional as F


def upfirdn1d(input, kernel, up=1, down=1, pad=(0, 0)):
    out = upfirdn1d_native(
        input, kernel, up, down, pad[0], pad[1]
    )
    return out


def upfirdn1d_native(
    input, kernel, up, down, pad_0, pad_1
):
    _, channel, in_h = input.shape
    input = input.reshape(-1, in_h, 1)

    _, in_h, minor = input.shape
    kernel_h = kernel.shape[0]

    out = input.view(-1, in_h, 1, minor)
    out = F.pad(out, [0, 0, 0, up - 1])
    out = out.view(-1, in_h * up, minor)

    out = F.pad(
        out, [0, 0, max(pad_0, 0), max(pad_1, 0)]
    )
    out = out[
        :,
        max(-pad_0, 0) : out.shape[1] - max(-pad_1, 0),
        :,
    ]

    out = out.permute(0, 2, 1)
    out = out.reshape(
        [-1, 1, in_h * up + pad_0 + pad_1]
    )
    w = torch.flip(kernel, [0]).view(1, 1, kernel_h)
    out = F.conv1d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up + pad_0 + pad_1 - kernel_h + 1,
    )
    out = out.permute(0, 2, 1)
    out = out[:, ::down, :]

    out_h = (in_h * up + pad_0 + pad_1 - kernel_h) // down + 1

    return out.view(-1, channel, out_h)


if __name__ == '__main__':
    from scipy.signal import upfirdn
    print(upfirdn([1, 1], [1, 2, 3], up=1, down=2))
    input = torch.arange(1, 4).view(1, 1, -1).float()
    kernel = torch.ones(2)
    print(upfirdn1d_native(input, kernel, 1, 2, 1, 0).squeeze())
