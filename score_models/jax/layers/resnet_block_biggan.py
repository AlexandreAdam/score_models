import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray
from jax.nn import relu
from typing import Optional, Callable
from .conv_layers import conv1x1, conv3x3
from .up_or_downsampling import upsample, downsample, naive_upsample, naive_downsample
from jaxtyping import Array

SQRT2 = 1.41421356237


class ResnetBlockBigGANpp(eqx.Module):
    activation: Callable
    dimensions: int
    in_ch: int
    out_ch: int
    up: bool
    down: bool
    fir: bool
    fir_kernel: tuple
    skip_rescale: bool
    GroupNorm_0: eqx.nn.GroupNorm
    Conv_0: eqx.nn.Conv
    Dense_0: Optional[eqx.nn.Linear]
    GroupNorm_1: eqx.nn.GroupNorm
    Dropout_0: eqx.nn.Dropout
    Conv_1: eqx.nn.Conv
    Conv_2: Optional[eqx.nn.Conv]

    def __init__(
        self,
        act: Callable,
        in_ch: int,
        out_ch: Optional[int] = None,
        temb_dim: Optional[int] = None,
        up: bool = False,
        down: bool = False,
        dropout: float = 0.1,
        fir: bool = False,
        fir_kernel: tuple = (1, 3, 3, 1),
        skip_rescale: bool = True,
        init_scale: float = 0.0,
        dimensions: int = 2,
        *,
        key: PRNGKeyArray,
    ):
        self.activation = act
        self.dimensions = dimensions
        out_ch = out_ch if out_ch is not None else in_ch
        self.up = up
        self.down = down
        self.fir = fir
        self.fir_kernel = fir_kernel
        self.skip_rescale = skip_rescale
        self.in_ch = in_ch
        self.out_ch = out_ch
        
        key_conv0, key_dense0, key_conv1, key_conv2 = jax.random.split(key, 4)
        self.GroupNorm_0 = eqx.nn.GroupNorm(groups=min(in_ch // 4, 32), channels=in_ch, eps=1e-6)

        self.Conv_0 = conv3x3(in_ch, out_ch, dimensions=dimensions, key=key_conv0)
        if temb_dim is not None:
            self.Dense_0 = eqx.nn.Linear(temb_dim, out_ch, key=key_dense0)
        else:
            self.Dense_0 = None
        self.GroupNorm_1 = eqx.nn.GroupNorm(groups=min(out_ch // 4, 32), channels=out_ch, eps=1e-6)
        self.Dropout_0 = eqx.nn.Dropout(dropout)
        self.Conv_1 = conv3x3(out_ch, out_ch, dimensions=dimensions, key=key_conv1)
        if in_ch != out_ch or up or down:
            self.Conv_2 = conv1x1(in_ch, out_ch, dimensions=dimensions, key=key_conv2)
        else: # Have to initialize it to None in jax...
            self.Conv_2 = None

    def __call__(self, x, temb: Optional[Array] = None, *, key: Optional[PRNGKeyArray] = None, inference: bool = True):
        if key is not None:
            key1, key2 = jax.random.split(key, 2)
        else:
            key1, key2 = None, None
        h = self.activation(self.GroupNorm_0(x, key=key1))

        if self.up:
            if self.fir:
                h = upsample(h, self.fir_kernel, factor=2, dimensions=self.dimensions)
                x = upsample(x, self.fir_kernel, factor=2, dimensions=self.dimensions)
            else:
                h = naive_upsample(h, factor=2, dimensions=self.dimensions)
                x = naive_upsample(x, factor=2, dimensions=self.dimensions)
        elif self.down:
            if self.fir:
                h = downsample(h, self.fir_kernel, factor=2, dimensions=self.dimensions)
                x = downsample(x, self.fir_kernel, factor=2, dimensions=self.dimensions)
            else:
                h = naive_downsample(h, factor=2, dimensions=self.dimensions)
                x = naive_downsample(x, factor=2, dimensions=self.dimensions)

        h = self.Conv_0(h)
        if temb is not None and self.Dense_0 is not None:
            h += self.Dense_0(relu(temb)).reshape(-1, *([1] * self.dimensions))
        h = self.activation(self.GroupNorm_1(h))
        h = self.Dropout_0(h, key=key2, inference=inference)
        h = self.Conv_1(h)

        if self.in_ch != self.out_ch or self.up or self.down:
            x = self.Conv_2(x) if self.Conv_2 is not None else x

        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / SQRT2

