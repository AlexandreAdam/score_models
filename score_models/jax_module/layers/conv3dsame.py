import jax.numpy as jnp
import equinox as eqx
from typing import Union
from .spectral_normalization import SpectralNorm

class Conv3dSame(eqx.Module):
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
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        spectral_norm: bool = False,
    ):
        conv_layer = eqx.nn.Conv3D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding="SAME" if stride == 1 else "VALID",
            dilation=dilation,
            groups=groups,
            use_bias=bias,
            padding_mode=padding_mode,
            bias_init=eqx.init.zeros if bias else None,
        )
        self.conv = SpectralNorm(conv_layer) if spectral_norm else conv_layer
        self.stride = stride
        self.dilation = dilation
        self.kernel_size = kernel_size

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.conv(x)
        return x

class ConvTransposed3dSame(eqx.Module):
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
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        spectral_norm: bool = False,
    ):
        conv_layer = eqx.nn.ConvTranspose3D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation * (kernel_size - 1) // 2,
            output_padding=stride - 1,
            dilation=dilation,
            groups=groups,
            use_bias=bias,
            bias_init=eqx.init.zeros if bias else None,
        )
        self.conv = SpectralNorm(conv_layer) if spectral_norm else conv_layer
        self.stride = stride
        self.dilation = dilation
        self.kernel_size = kernel_size

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.conv(x)
        return x

