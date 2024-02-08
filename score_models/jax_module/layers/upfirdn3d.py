import jax.numpy as jnp
from jax import lax


def upfirdn3d(input, kernel, up=1, down=1, pad=(0, 0)):
    out = upfirdn3d_native(
        input,
        kernel,
        up,
        up,
        up,
        down,
        down,
        down,
        pad[0],
        pad[1],
        pad[0],
        pad[1],
        pad[0],
        pad[1],
    )
    return out


def upfirdn3d_native(
    input,
    kernel,
    up_x,
    up_y,
    up_z,
    down_x,
    down_y,
    down_z,
    pad_x0,
    pad_x1,
    pad_y0,
    pad_y1,
    pad_z0,
    pad_z1,
):
    _, channel, in_h, in_w, in_d = input.shape
    input = input.reshape(-1, in_h, in_w, in_d, 1)

    _, in_h, in_w, in_d, minor = input.shape
    kernel_h, kernel_w, kernel_d = kernel.shape

    out = input.reshape(-1, in_h, 1, in_w, 1, in_d, 1, minor)
    out = jnp.pad(
        out,
        [
            (0, 0),
            (0, up_y - 1),
            (0, 0),
            (0, up_x - 1),
            (0, 0),
            (0, up_z - 1),
            (0, 0),
            (0, 0),
        ],
    )
    out = out.reshape(-1, in_h * up_y, in_w * up_x, in_d * up_z, minor)

    out = jnp.pad(
        out,
        [
            (0, 0),
            (max(pad_x0, 0), max(pad_x1, 0)),
            (max(pad_y0, 0), max(pad_y1, 0)),
            (max(pad_z0, 0), max(pad_z1, 0)),
            (0, 0),
        ],
    )
    out = out[
        :,
        max(-pad_x0, 0) : out.shape[1] - max(-pad_x1, 0),
        max(-pad_y0, 0) : out.shape[2] - max(-pad_y1, 0),
        max(-pad_z0, 0) : out.shape[3] - max(-pad_z1, 0),
        :,
    ]

    out = out.transpose((0, 4, 1, 2, 3))
    out = out.reshape(
        [
            -1,
            1,
            in_h * up_y + pad_y0 + pad_y1,
            in_w * up_x + pad_x0 + pad_x1,
            in_d * up_z + pad_z0 + pad_z1,
        ]
    )
    w = jnp.flip(kernel, [0, 1, 2]).reshape(1, 1, kernel_h, kernel_w, kernel_d)
    out = lax.conv_general_dilated(
        out,
        w,
        window_strides=(down_y, down_x, down_z),
        padding="VALID",
        lhs_dilation=(up_y, up_x, up_z),
        rhs_dilation=(1, 1, 1),
        dimension_numbers=("NCHWD", "IODHW", "NCHWD"),
    )

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
    out_d = (in_d * up_z + pad_z0 + pad_z1 - kernel_d) // down_z + 1

    return out.reshape(-1, channel, out_h, out_w, out_d)