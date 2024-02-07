import jax.numpy as jnp
from jax import lax


def upfirdn1d(input, kernel, up=1, down=1, pad=(0, 0)):
    out = upfirdn1d_native(input, kernel, up, down, pad[0], pad[1])
    return out


def upfirdn1d_native(input, kernel, up, down, pad_0, pad_1):
    _, channel, in_h = input.shape
    input = input.reshape(-1, in_h, 1)

    _, in_h, minor = input.shape
    kernel_h = kernel.shape[0]

    out = input.reshape(-1, in_h, 1, minor)
    out = jnp.pad(out, [(0, 0), (0, up - 1), (0, 0), (0, 0)])
    out = out.reshape(-1, in_h * up, minor)

    out = jnp.pad(out, [(0, 0), (max(pad_0, 0), max(pad_1, 0)), (0, 0)])
    out = out[
        :,
        max(-pad_0, 0) : out.shape[1] - max(-pad_1, 0),
        :,
    ]

    out = out.transpose((0, 2, 1))
    out = out.reshape([-1, 1, in_h * up + pad_0 + pad_1])
    w = jnp.flip(kernel, [0]).reshape(1, 1, kernel_h)
    out = lax.conv_general_dilated(
        out,
        w,
        window_strides=[1],
        padding="VALID",
        dimension_numbers=("NCH", "IOC", "NCH"),
    )
    out = out.reshape(
        -1,
        minor,
        in_h * up + pad_0 + pad_1 - kernel_h + 1,
    )
    out = out.transpose((0, 2, 1))
    out = out[:, ::down, :]

    out_h = (in_h * up + pad_0 + pad_1 - kernel_h) // down + 1

    return out.reshape(-1, channel, out_h)


if __name__ == "__main__":
    from scipy.signal import upfirdn

    print(upfirdn([1, 1], [1, 2, 3], up=1, down=2))
    input = jnp.arange(1, 4).reshape(1, 1, -1).astype(jnp.float32)
    kernel = jnp.ones(2)
    print(upfirdn1d_native(input, kernel, 1, 2, 1, 0).squeeze())
