import jax.numpy as jnp
import equinox as eqx
from .conv_layers import conv3x3
from .up_or_downsampling import downsample
from .style_gan_conv import StyleGANConv
from ..definitions import default_init


def avg_pool(x, dimensions, kernel_size=2, stride=2):
    if dimensions == 1:
        return jnp.mean(x.reshape(x.shape[0], x.shape[1], -1, kernel_size), axis=-1)[
            :, :, ::stride
        ]
    elif dimensions == 2:
        return jnp.mean(
            x.reshape(x.shape[0], x.shape[1], -1, kernel_size, kernel_size),
            axis=(-1, -2),
        )[:, :, ::stride, ::stride]
    elif dimensions == 3:
        return jnp.mean(
            x.reshape(
                x.shape[0], x.shape[1], -1, kernel_size, kernel_size, kernel_size
            ),
            axis=(-1, -2, -3),
        )[:, :, ::stride, ::stride, ::stride]


class DownsampleLayer(eqx.Module):
    Conv_0: eqx.Module = None
    dimensions: int
    with_conv: bool
    out_ch: int
    fir: bool
    fir_kernel: tuple

    def __init__(
        self,
        in_ch: int = None,
        out_ch: int = None,
        with_conv: bool = False,
        fir: bool = False,
        fir_kernel: tuple = (1, 3, 3, 1),
        dimensions: int = 2,
    ):
        out_ch = out_ch if out_ch is not None else in_ch
        self.dimensions = dimensions
        self.with_conv = with_conv
        self.out_ch = out_ch
        self.fir = fir
        self.fir_kernel = fir_kernel

        if not fir and with_conv:
            self.Conv_0 = conv3x3(in_ch, out_ch, stride=2, dimensions=dimensions)
        elif fir and with_conv:
            self.Conv_0 = StyleGANConv(
                in_ch,
                out_ch,
                kernel=3,
                down=True,
                resample_kernel=fir_kernel,
                use_bias=True,
                kernel_init=default_init(),
                dimensions=dimensions,
            )

    def __call__(self, x):
        if not self.fir:
            if self.with_conv:
                pad_width = [(0, 0), (0, 0)] + [(0, 1) for _ in range(self.dimensions)]
                x = jnp.pad(x, pad_width)
                x = self.Conv_0(x)
            else:
                x = avg_pool(x, self.dimensions)
        else:
            if not self.with_conv:
                x = downsample(x, self.fir_kernel, factor=2, dimensions=self.dimensions)
            else:
                x = self.Conv_0(x)
        return x
