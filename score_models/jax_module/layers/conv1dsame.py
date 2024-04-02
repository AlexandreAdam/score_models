import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Union
from flax.linen import SpectralNorm

class Conv1dSame(eqx.Module):
    conv: Union[eqx.nn.Conv1d, SpectralNorm]
    stride: int
    dilation: int
    kernel_size: int

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        key: jax.random.PRNGKey,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        padding: int = 0,
        bias: bool = True,
        spectral_norm: bool = False,
    ):
        conv_layer = eqx.nn.Conv1d(
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

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        effective_kernel_size = (self.kernel_size - 1) * self.dilation + 1
        pad_total = max(effective_kernel_size - self.stride, 0)
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = jnp.pad(x, [(0, 0), (0, 0), (pad_beg, pad_end)], mode='constant')
        x = self.conv(x)
        return x


class ConvTransposed1dSame(eqx.Module):
    conv: Union[eqx.nn.ConvTranspose1d, SpectralNorm]
    stride: int
    dilation: int
    kernel_size: int

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        key: jax.random.PRNGKey,
        stride: int = 1,
        dilation: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        spectral_norm: bool = False,
    ):
        conv_layer = eqx.nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            output_padding=stride - 1,
            dilation=dilation,
            groups=groups,
            use_bias=bias,
            key=key
        )
        self.conv = SpectralNorm(conv_layer) if spectral_norm else conv_layer
        self.stride = stride
        self.dilation = dilation
        self.kernel_size = kernel_size

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        input_length = x.shape[-1]
        effective_kernel_size = (self.kernel_size - 1) * self.dilation + 1
        pad_along_length = max(0, (input_length - 1) * self.stride + effective_kernel_size - input_length)
        padding = pad_along_length // 2
        output_padding = (input_length - 1) * self.stride + effective_kernel_size - input_length - 2 * padding
        x = jnp.pad(x, [(0, 0), (0, 0), (padding, padding + output_padding)], mode='constant')
        x = self.conv(x)
        return x

