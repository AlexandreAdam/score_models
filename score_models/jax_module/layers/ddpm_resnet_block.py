"""
Code ported from Yang Song's repo https://github.com/yang-song/score_sde_pytorch/blob/main/models/layers.py
"""
import jax.numpy as jnp
import equinox as eqx
from jax.nn.initializers import zeros
from .conv1dsame import Conv1dSame
from .conv2dsame import Conv2dSame
from .conv3dsame import Conv3dSame
from score_models.definitions import default_init

CONVS = {1: Conv1dSame, 2: Conv2dSame, 3: Conv3dSame}


def conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1, dimensions=2):
    conv = CONVS[dimensions](
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        dilation=dilation,
        bias=bias,
    )
    return conv


def conv1x1(in_planes, out_planes, stride=1, bias=True, dimensions=2):
    conv = CONVS[dimensions](
        in_planes, out_planes, kernel_size=1, stride=stride, bias=bias
    )
    return conv


class NIN(eqx.Module):
    W: jnp.ndarray
    b: jnp.ndarray

    def __init__(self, in_dim, num_units, init_scale=0.1):
        self.W = eqx.Parameter(default_init(scale=init_scale)((num_units, in_dim)))
        self.b = eqx.Parameter(jnp.zeros(num_units))

    def __call__(self, x):
        B, C, *D = x.shape
        spatial_dims = list(range(2, 2 + len(D)))
        x = x.transpose((0, *spatial_dims, 1))
        y = jnp.einsum("ij,...j->...i", self.W, x) + self.b
        return y.transpose((0, -1, *spatial_dims))


class DDPMResnetBlock(eqx.Module):
    GroupNorm_0: eqx.Module
    Conv_0: eqx.Module
    Dense_0: Optional[eqx.Module] = None
    GroupNorm_1: eqx.Module
    Dropout_0: eqx.Module
    Conv_1: eqx.Module
    Conv_2: Optional[eqx.Module] = None
    NIN_0: Optional[eqx.Module] = None
    act: Callable
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
    ):
        self.act = act
        out_ch = out_ch if out_ch is not None else in_ch
        self.GroupNorm_0 = eqx.nn.GroupNorm(
            num_groups=min(in_ch // 4, 32), num_channels=in_ch
        )
        self.Conv_0 = conv3x3(in_ch, out_ch, dimensions=dimensions)
        if temb_dim is not None:
            self.Dense_0 = eqx.nn.Linear(
                temb_dim, out_ch, kernel_init=default_init(), bias_init=zeros
            )
        self.GroupNorm_1 = eqx.nn.GroupNorm(
            num_groups=min(out_ch // 4, 32), num_channels=out_ch
        )
        self.Dropout_0 = eqx.nn.Dropout(rate=dropout)
        self.Conv_1 = conv3x3(out_ch, out_ch, dimensions=dimensions)
        if in_ch != out_ch:
            if conv_shortcut:
                self.Conv_2 = conv3x3(in_ch, out_ch, dimensions=dimensions)
            else:
                self.NIN_0 = NIN(in_ch, out_ch)
        self.out_ch = out_ch
        self.in_ch = in_ch
        self.conv_shortcut = conv_shortcut

    def __call__(self, x, temb=None):
        B, C, *D = x.shape
        h = self.act(self.GroupNorm_0(x))
        h = self.Conv_0(h)
        if temb is not None and self.Dense_0 is not None:
            temb_act = self.act(temb)
            h += self.Dense_0(temb_act).reshape((B, -1) + (1,) * len(D))
        h = self.act(self.GroupNorm_1(h))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)
        if C != self.out_ch:
            if self.conv_shortcut and self.Conv_2 is not None:
                x = self.Conv_2(x)
            elif self.NIN_0 is not None:
                x = self.NIN_0(x)
        return x + h
