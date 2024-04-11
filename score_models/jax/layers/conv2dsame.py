from typing import Union
import jax.numpy as jnp
import equinox as eqx
from flax.linen import SpectralNorm
from jaxtyping import PRNGKeyArray, Array
from typing import Optional


__all__ = ["Conv2dSame"]


class Conv2dSame(eqx.Module):
    conv: Union[eqx.Module, SpectralNorm]
    stride: int
    dilation: int
    kernel_size: int

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        spectral_norm: bool = False,
        *,
        key: PRNGKeyArray,
    ):
        conv_layer = eqx.nn.Conv(
            num_spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            use_bias=bias,
            key=key
        )
        self.conv = SpectralNorm(conv_layer) if spectral_norm else conv_layer
        self.stride = stride
        self.dilation = dilation
        self.kernel_size = kernel_size

    def __call__(self, x: Array, key: Optional[PRNGKeyArray] = None) -> Array:
        effective_kernel_size = (self.kernel_size - 1) * self.dilation + 1
        pad_total = max(effective_kernel_size - self.stride, 0)
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = jnp.pad(x, [(0, 0), (pad_beg, pad_end), (pad_beg, pad_end)], mode='constant')
        x = self.conv(x)
        return x

# class ConvTransposed2dSame(eqx.Module):
    # conv: Union[eqx.Module, SpectralNorm]
    # stride: int
    # dilation: int
    # kernel_size: int

    # def __init__(
        # self,
        # in_channels: int,
        # out_channels: int,
        # kernel_size: int,
        # stride: int = 1,
        # padding: int = 0,
        # dilation: int = 1,
        # groups: int = 1,
        # bias: bool = True,
        # spectral_norm: bool = False,
        # *,
        # key: PRNGKeyArray,
    # ):
        # conv_layer = eqx.nn.ConvTranspose2d(
            # in_channels=in_channels,
            # out_channels=out_channels,
            # kernel_size=kernel_size,
            # stride=stride,
            # padding=0,
            # output_padding=0,
            # dilation=dilation,
            # groups=groups,
            # use_bias=bias,
            # key=key
        # )
        # self.conv = SpectralNorm(conv_layer) if spectral_norm else conv_layer
        # self.stride = stride
        # self.dilation = dilation
        # self.kernel_size = kernel_size

    # def __call__(self, x: Array, key: Optional[PRNGKeyArray] = None) -> Array:
        # return self.conv(x)

