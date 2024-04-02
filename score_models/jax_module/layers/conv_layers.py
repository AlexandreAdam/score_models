import jax
from jax.nn.initializers import zeros
from .conv1dsame import Conv1dSame
from .conv2dsame import Conv2dSame
from .conv3dsame import Conv3dSame
from ..definitions import default_init
from jaxtyping import PRNGKeyArray, Array

CONVS = {1: Conv1dSame, 2: Conv2dSame, 3: Conv3dSame}


def conv1x1(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    bias: bool = True,
    init_scale: float = 1.0,
    dimensions: int = 2,
    *,
    key: PRNGKeyArray
):
    """1x1 convolution with DDPM initialization."""
    conv = CONVS[dimensions](
        in_planes, out_planes, kernel_size=1, stride=stride, bias=bias
    )
    conv.weight = default_init(init_scale)(conv.weight.shape, key=key)
    if bias:
        conv.bias = zeros(jax.random.PRNGKey(0), conv.bias.shape)
    return conv


def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    bias: bool = True,
    dilation: int = 1,
    init_scale: float = 1.0,
    dimensions: int = 2,
    *,
    key: PRNGKeyArray
):
    """3x3 convolution with DDPM initialization."""
    conv = CONVS[dimensions](
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        dilation=dilation,
        bias=bias,
    )
    conv.weight = default_init(init_scale)(conv.weight.shape, key=key)
    if bias:
        conv.bias = zeros(jax.random.PRNGKey(0), conv.bias.shape)
    return conv
