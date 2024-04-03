"""
Code ported from Yang Song's repo https://github.com/yang-song/score_sde_pytorch/blob/main/models/layers.py
"""
from typing import Optional, Callable
import jax.numpy as jnp
import equinox as eqx
from .conv1dsame import Conv1dSame
from .conv2dsame import Conv2dSame
from .conv3dsame import Conv3dSame
from ..definitions import default_init
from jaxtyping import PRNGKeyArray, Array
import jax

CONVS = {1: Conv1dSame, 2: Conv2dSame, 3: Conv3dSame}


def conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1, dimensions=2, *, key: PRNGKeyArray):
    conv = CONVS[dimensions](
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        dilation=dilation,
        bias=bias,
        key=key,
    )
    return conv


def conv1x1(in_planes, out_planes, stride=1, bias=True, dimensions=2, *, key: PRNGKeyArray):
    conv = CONVS[dimensions](
        in_planes, out_planes, kernel_size=1, stride=stride, bias=bias, key=key
    )
    return conv


class NIN(eqx.Module):
    W: jnp.ndarray
    b: jnp.ndarray

    def __init__(self, in_dim, num_units, init_scale=0.1, *, key: PRNGKeyArray):
        self.W = default_init(scale=init_scale)(shape=(num_units, in_dim), key=key)
        self.b = jnp.zeros(num_units)

    def __call__(self, x):
        B, C, *D = x.shape
        spatial_dims = list(range(2, 2 + len(D)))
        x = x.transpose((0, *spatial_dims, 1))
        y = jnp.einsum("ij,...j->...i", self.W, x) + self.b
        return y.transpose((0, -1, *spatial_dims))


class DDPMResnetBlock(eqx.Module):
    GroupNorm_0: eqx.nn.GroupNorm
    activation: Callable
    Conv_0: eqx.nn.Conv
    Dense_0: Optional[eqx.nn.Linear]
    GroupNorm_1: eqx.nn.GroupNorm
    Dropout_0: eqx.nn.Dropout
    Conv_1: eqx.nn.Conv
    Conv_2: Optional[eqx.nn.Conv]
    NIN_0: Optional[NIN]
    out_ch: int
    in_ch: int
    conv_shortcut: bool

    def __init__(
        self,
        act,
        in_ch,
        out_ch=None,
        temb_dim=None,
        conv_shortcut=False,
        dropout=0.1,
        dimensions=2,
        *,
        key: PRNGKeyArray
    ):
        self.activation = act
        out_ch = out_ch if out_ch is not None else in_ch
        
        key_conv0, key_conv1, key_conv2, key_dense0 = jax.random.split(key, 4)
        self.GroupNorm_0 = eqx.nn.GroupNorm(groups=min(in_ch // 4, 32), channels=in_ch)
        self.Conv_0 = conv3x3(in_ch, out_ch, dimensions=dimensions, key=key_conv0)
        if temb_dim is not None:
            self.Dense_0 = eqx.nn.Linear(temb_dim, out_ch, key=key_dense0)
        else:
            self.Dense_0 = None
        self.GroupNorm_1 = eqx.nn.GroupNorm(groups=min(out_ch // 4, 32), channels=out_ch)
        self.Dropout_0 = eqx.nn.Dropout(p=dropout)
        self.Conv_1 = conv3x3(out_ch, out_ch, dimensions=dimensions, key=key_conv1)
        if in_ch != out_ch:
            if conv_shortcut:
                self.Conv_2 = conv3x3(in_ch, out_ch, dimensions=dimensions, key=key_conv2)
                self.NIN_0 = None
            else:
                self.NIN_0 = NIN(in_ch, out_ch, key=key_conv2)
                self.Conv_2 = None
        else: # Have to initialize it to None in jax...
            self.Conv_2 = None
            self.NIN_0 = None
        self.out_ch = out_ch
        self.in_ch = in_ch
        self.conv_shortcut = conv_shortcut

    def __call__(self, x: Array, temb: Optional[Array] = None):
        B, C, *D = x.shape
        h = self.activation(self.GroupNorm_0(x))
        h = self.Conv_0(h)
        if temb is not None and self.Dense_0 is not None:
            temb_act = self.activation(temb)
            h += self.Dense_0(temb_act).reshape((B, -1) + (1,) * len(D))
        h = self.activation(self.GroupNorm_1(h))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)
        if C != self.out_ch:
            if self.conv_shortcut and self.Conv_2 is not None:
                x = self.Conv_2(x)
            elif self.NIN_0 is not None:
                x = self.NIN_0(x)
        return x + h
