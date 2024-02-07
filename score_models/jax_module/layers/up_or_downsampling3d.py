import jax.numpy as jnp
from jax import lax

__all__ = [
    "naive_upsample_3d",
    "naive_downsample_3d",
    "upsample_3d",
    "downsample_3d",
    "conv_downsample_3d",
    "upsample_conv_3d",
]


def naive_upsample_3d(x, factor=2):
    _N, C, H, W, D = x.shape
    x = jnp.reshape(x, (-1, C, H, 1, W, 1, D, 1))
    x = jnp.tile(x, (1, 1, 1, factor, 1, factor, 1, factor))
    return jnp.reshape(x, (-1, C, H * factor, W * factor, D * factor))


def naive_downsample_3d(x, factor=2):
    _N, C, H, W, D = x.shape
    x = jnp.reshape(
        x, (-1, C, H // factor, factor, W // factor, factor, D // factor, factor)
    )
    return jnp.mean(x, axis=(3, 5, 7))


def upsample_conv_3d(x, w, k=None, factor=2, gain=1):
    assert isinstance(factor, int) and factor >= 1
    outC, inC, convH, convW, convD = w.shape
    assert convW == convH
    assert convW == convD

    if k is None:
        k = jnp.array([1] * factor)
    k = _setup_kernel(k) * (gain * (factor**3))
    p = (k.shape[0] - factor) - (convW - 1)

    stride = (factor, factor, factor)
    output_shape = (
        (x.shape[2] - 1) * factor + convH,
        (x.shape[3] - 1) * factor + convW,
        (x.shape[4] - 1) * factor + convD,
    )
    output_padding = (
        (output_shape[0] - (x.shape[2] - 1) * stride[0] - convH),
        (output_shape[1] - (x.shape[3] - 1) * stride[1] - convW),
        (output_shape[2] - (x.shape[3] - 1) * stride[2] - convD),
    )
    assert output_padding[0] >= 0 and output_padding[1] >= 0 and output_padding[2] >= 0
    num_groups = x.shape[1] // inC

    w = jnp.reshape(w, (num_groups, -1, inC, convH, convW, convD))
    w = jnp.flip(w, (3, 4, 5)).transpose((0, 2, 1, 3, 4, 5))
    w = jnp.reshape(w, (num_groups * inC, -1, convH, convW, convD))

    x = lax.conv_transpose(
        x,
        w,
        strides=stride,
        padding=((p + 1) // 2 + factor - 1, p // 2 + 1),
        output_padding=output_padding,
    )
    return upfirdn3d(
        x, jnp.array(k, device=x.device), pad=((p + 1) // 2 + factor - 1, p // 2 + 1)
    )


def conv_downsample_3d(x, w, k=None, factor=2, gain=1):
    assert isinstance(factor, int) and factor >= 1
    _outC, _inC, convH, convW, convD = w.shape
    assert convW == convH
    assert convW == convD

    if k is None:
        k = jnp.array([1] * factor)
    k = _setup_kernel(k) * gain
    p = (k.shape[0] - factor) + (convW - 1)
    s = (factor, factor, factor)

    x = upfirdn3d(x, jnp.array(k, device=x.device), pad=((p + 1) // 2, p // 2))
    return lax.conv_general_dilated(x, w, window_strides=s, padding="VALID")


def _setup_kernel(k):
    k = jnp.asarray(k, dtype=jnp.float32)
    if k.ndim == 1:
        m = jnp.outer(k, k)
        k = jnp.multiply.outer(m, k)
    k /= jnp.sum(k)
    assert k.ndim == 3
    assert k.shape[0] == k.shape[1]
    assert k.shape[0] == k.shape[2]
    return k


def upsample_3d(x, k=None, factor=2, gain=1):
    assert isinstance(factor, int) and factor >= 1
    if k is None:
        k = jnp.array([1] * factor)
    k = _setup_kernel(k) * (gain * (factor**3))
    p = k.shape[0] - factor
    return upfirdn3d(
        x,
        jnp.array(k, device=x.device),
        up=factor,
        pad=((p + 1) // 2 + factor - 1, p // 2),
    )


def downsample_3d(x, k=None, factor=2, gain=1):
    assert isinstance(factor, int) and factor >= 1
    if k is None:
        k = jnp.array([1] * factor)
    k = _setup_kernel(k) * gain
    p = k.shape[0] - factor
    return upfirdn3d(
        x, jnp.array(k, device=x.device), down=factor, pad=((p + 1) // 2, p // 2)
    )
