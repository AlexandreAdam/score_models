import equinox as eqx
from jax.nn import relu
from typing import Optional, Callable
from score_models.definitions import default_init
from .conv_layers import conv1x1, conv3x3
from .up_or_downsampling import upsample, downsample, naive_upsample, naive_downsample

SQRT2 = 1.41421356237


class ResnetBlockBigGANpp(eqx.Module):
    GroupNorm_0: eqx.Module
    Conv_0: eqx.Module
    Dense_0: Optional[eqx.Module]
    GroupNorm_1: eqx.Module
    Dropout_0: eqx.Module
    Conv_1: eqx.Module
    Conv_2: Optional[eqx.Module]
    act: Callable[[eqx.Array], eqx.Array]
    dimensions: int
    up: bool
    down: bool
    fir: bool
    fir_kernel: tuple
    skip_rescale: bool

    def __init__(
        self,
        act: Callable[[eqx.Array], eqx.Array],
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
    ):
        self.act = act
        self.dimensions = dimensions
        out_ch = out_ch if out_ch is not None else in_ch
        self.up = up
        self.down = down
        self.fir = fir
        self.fir_kernel = fir_kernel
        self.skip_rescale = skip_rescale

        self.GroupNorm_0 = eqx.nn.GroupNorm(
            num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6
        )
        self.Conv_0 = conv3x3(
            in_ch, out_ch, dimensions=dimensions, kernel_init=default_init()
        )
        self.Dense_0 = (
            eqx.nn.Linear(
                temb_dim, out_ch, kernel_init=default_init(), bias_init=eqx.init.zeros
            )
            if temb_dim is not None
            else None
        )
        self.GroupNorm_1 = eqx.nn.GroupNorm(
            num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6
        )
        self.Dropout_0 = eqx.nn.Dropout(dropout)
        self.Conv_1 = conv3x3(
            out_ch,
            out_ch,
            dimensions=dimensions,
            init_scale=init_scale,
            kernel_init=default_init(),
        )
        if in_ch != out_ch or up or down:
            self.Conv_2 = conv1x1(
                in_ch, out_ch, dimensions=dimensions, kernel_init=default_init()
            )

    def __call__(self, x, temb: Optional[eqx.Array] = None):
        B, *_ = x.shape
        h = self.act(self.GroupNorm_0(x))

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
            h += self.Dense_0(relu(temb)).reshape(B, -1, *([1] * self.dimensions))
        h = self.act(self.GroupNorm_1(h))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)

        if self.in_ch != self.out_ch or self.up or self.down:
            x = self.Conv_2(x) if self.Conv_2 is not None else x

        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / SQRT2
