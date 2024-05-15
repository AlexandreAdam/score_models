import jax.numpy as jnp
from jax import lax


def upfirdn2d(x, kernel, up=1, down=1, pad=(0, 0)):
    out = upfirdn2d_native(
        x, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1]
    )
    return out


def upfirdn2d_native(
    x, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
    channel, in_h, in_w = x.shape
    kernel_h, kernel_w = kernel.shape

    # Interweave zeros between input pixels.
    out = x.reshape(channel, in_h, 1, in_w, 1)
    out = jnp.pad(out, [(0, 0), (0, 0), (0, up_y - 1), (0, 0), (0, up_x - 1)])
    out = out.reshape(channel, up_y*in_h, up_x*in_w, 1)

    # Pad with zeros at the boundaries
    out = jnp.pad(
        out,
        [
            (0, 0),
            (max(pad_y0, 0), max(pad_y1, 0)),
            (max(pad_x0, 0), max(pad_x1, 0)),
            (0, 0)
        ]
    )
    out = out[
            :,
            max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
            max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
            :
        ]


    # Reshape for convolution (NHWI -> NIHW).
    out = out.transpose([0, 3, 1, 2])
    out = out.reshape(
        [
            -1, 
            1, 
            in_h * up_y + pad_y0 + pad_y1, 
            in_w * up_x + pad_x0 + pad_x1
            ]
    )
    
    # Flip spatial + reshape kernel for convolution (HW -> OIHW).
    w = jnp.flip(kernel, [0, 1]).reshape(1, 1, kernel_h, kernel_w)
    out = lax.conv(out, w, window_strides=(1, 1), padding='VALID')
    
    # Permute back to NHWI
    out = out.reshape(
        -1,
        1,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.transpose(0, 2, 3, 1)
    
    # Downsample (if needed)
    out = out[:, ::down_y, ::down_x]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
    return out.reshape(channel, out_h, out_w)

