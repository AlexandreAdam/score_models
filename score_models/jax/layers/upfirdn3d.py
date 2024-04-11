import jax.numpy as jnp
from jax import vmap
from jax import lax
from functools import partial


def upfirdn3d(x, kernel, up=1, down=1, pad=(0, 0)):
    out = upfirdn3d_native(
        x,
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
    x,
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
    channel, in_h, in_w, in_d = x.shape
    kernel_h, kernel_w, kernel_d = kernel.shape

    # Interweave zeros between input pixels.
    out = x.reshape(channel, in_h, 1, in_w, 1, in_d, 1)
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
        ],
    )
    out = out.reshape(channel, in_h * up_y, in_w * up_x, in_d * up_z)

    # Pad with zeros around the image.
    out = jnp.pad(
        out,
        [
            (0, 0),
            (max(pad_x0, 0), max(pad_x1, 0)),
            (max(pad_y0, 0), max(pad_y1, 0)),
            (max(pad_z0, 0), max(pad_z1, 0)),
        ],
    )

    # Add dimensions for convolution (COIHWD -> OIHWD).
    out = out.reshape(
        [
            channel,
            1,
            1,
            in_h * up_y + pad_y0 + pad_y1,
            in_w * up_x + pad_x0 + pad_x1,
            in_d * up_z + pad_z0 + pad_z1,
        ]
    )
    # Flip x-y dimensions and reshape to 3+2D convolution kernel expected by lax.conv.
    w = jnp.flip(kernel, [0, 1]).reshape(1, 1, kernel_h, kernel_w, kernel_d) # OIHWD layout
    out = vmap(partial(lax.conv, window_strides=(1, 1, 1), padding="VALID"), in_axes=(0, None))(out, w)
    
    # Crop down to the output size.
    out = out.reshape(
        channel,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
        in_d * up_z + pad_z0 + pad_z1 - kernel_d + 1,
    )
    out = out[:, ::down_y, ::down_x, ::down_z]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
    out_d = (in_d * up_z + pad_z0 + pad_z1 - kernel_d) // down_z + 1

    return out.reshape(channel, out_h, out_w, out_d)

