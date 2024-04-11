import jax.numpy as jnp
from jax import vmap
from functools import partial
from jax import lax


def upfirdn1d(input, kernel, up=1, down=1, pad=(0, 0)):
    out = upfirdn1d_native(input, kernel, up, down, pad[0], pad[1])
    return out


def upfirdn1d_native(x, kernel, up, down, pad_0, pad_1):
    channel, in_h = x.shape
    kernel_h = kernel.shape[0]

    # Interweave zeros between input pixels.
    out = x.reshape(channel, in_h, 1)
    out = jnp.pad(out, [(0, 0), (0, up - 1), (0, 0)])
    out = out.reshape(channel, in_h * up)

    # Pad with zeros at the boundaries
    out = jnp.pad(out, [(0, 0), (max(pad_0, 0), max(pad_1, 0))])
    out = out[
        :,
        max(-pad_0, 0) : out.shape[1] - max(-pad_1, 0),
    ]


    # Add dimensions for convolution (COIH -> OIH).
    out = out.reshape([channel, 1, 1, in_h * up + pad_0 + pad_1])
    w = jnp.flip(kernel, [0]).reshape(1, 1, kernel_h)
    out = vmap(partial(lax.conv, window_strides=[1], padding="VALID"), in_axes=[0, None])(out, w)
    
    # Crop to the output size
    out = out.reshape(
        channel,
        in_h * up + pad_0 + pad_1 - kernel_h + 1,
    )
    out = out[:, ::down]

    out_h = (in_h * up + pad_0 + pad_1 - kernel_h) // down + 1
    return out.reshape(channel, out_h)


if __name__ == "__main__":
    from scipy.signal import upfirdn

    print(upfirdn([1, 1], [1, 2, 3], up=1, down=2))
    input = jnp.arange(1, 4).reshape(1, 1, -1).astype(jnp.float32)
    kernel = jnp.ones(2)
    print(upfirdn1d_native(input, kernel, 1, 2, 1, 0).squeeze())
