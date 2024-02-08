from typing import Optional, Callable
import jax.numpy as jnp
import equinox as eqx
from jax.nn import elu
from functools import partial
from .conv2dsame import Conv2dSame
"""
Code ported from Yang Song's repo https://github.com/yang-song/score_sde_pytorch/blob/main/models/layers.py
with hacked padding
"""


def conv1x1(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    bias: bool = True,
    dilation: int = 1,
):
    """1x1 convolution."""
    return Conv2dSame(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=bias,
        dilation=dilation,
    )


def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    bias: bool = True,
    dilation: int = 1,
):
    """3x3 convolution."""
    return Conv2dSame(
        in_planes,
        out_planes,
        stride=stride,
        bias=bias,
        dilation=dilation,
        kernel_size=3,
    )


class ConvMeanPool(eqx.Module):
    conv: eqx.Module

    def __init__(
        self, input_dim: int, output_dim: int, kernel_size: int = 3, biases: bool = True
    ):
        self.conv = Conv2dSame(
            input_dim, output_dim, kernel_size, stride=1, bias=biases
        )

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        output = self.conv(inputs)
        output = (
            output[:, :, ::2, ::2]
            + output[:, :, 1::2, ::2]
            + output[:, :, ::2, 1::2]
            + output[:, :, 1::2, 1::2]
        ) / 4.0
        return output


class NCSNResidualBlock(eqx.Module):
    conv1: eqx.Module
    normalize2: eqx.Module
    conv2: eqx.Module
    shortcut: Optional[eqx.Module]
    normalize1: eqx.Module
    non_linearity: Callable[[jnp.ndarray], jnp.ndarray]
    input_dim: int
    output_dim: int
    resample: Optional[str]

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        resample: Optional[str] = None,
        act: Callable[[jnp.ndarray], jnp.ndarray] = elu,
        normalization: Callable = eqx.nn.InstanceNorm,
        dilation: int = 1,
    ):
        self.non_linearity = act
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resample = resample
        self.normalize1 = normalization(input_dim)

        if resample == "down":
            conv_shortcut = (
                partial(ConvMeanPool, kernel_size=1)
                if dilation == 1
                else partial(conv3x3, dilation=dilation)
            )
            self.conv1 = conv3x3(input_dim, input_dim, dilation=dilation)
            self.normalize2 = normalization(input_dim)
            self.conv2 = (
                ConvMeanPool(input_dim, output_dim, 3)
                if dilation == 1
                else conv3x3(input_dim, output_dim, dilation=dilation)
            )
        elif resample is None:
            conv_shortcut = (
                partial(conv1x1)
                if dilation == 1
                else partial(conv3x3, dilation=dilation)
            )
            self.conv1 = conv3x3(input_dim, output_dim, dilation=dilation)
            self.normalize2 = normalization(output_dim)
            self.conv2 = conv3x3(output_dim, output_dim, dilation=dilation)
        else:
            raise ValueError("Invalid resample value")

        if output_dim != input_dim or resample is not None:
            self.shortcut = conv_shortcut(input_dim, output_dim)
        else:
            self.shortcut = None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        output = self.non_linearity(self.normalize1(x))
        output = self.conv1(output)
        output = self.non_linearity(self.normalize2(output))
        output = self.conv2(output)

        shortcut = x if self.shortcut is None else self.shortcut(x)
        return shortcut + output