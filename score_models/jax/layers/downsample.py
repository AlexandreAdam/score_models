from typing import Optional
import jax.numpy as jnp
import equinox as eqx
from jax import lax
from .conv_layers import conv3x3
from .up_or_downsampling import downsample
from .style_gan_conv import StyleGANConv
from ..definitions import default_init
from jaxtyping import PRNGKeyArray


class DownsampleLayer(eqx.Module):
    dimensions: int
    with_conv: bool
    out_ch: Optional[int]
    fir: bool
    fir_kernel: tuple
    Conv_0: eqx.Module

    def __init__(
        self,
        in_ch: Optional[int] = None,
        out_ch: Optional[int] = None,
        with_conv: bool = False,
        fir: bool = False,
        fir_kernel: tuple = (1, 3, 3, 1),
        dimensions: int = 2,
        *,
        key: PRNGKeyArray,
    ):
        out_ch = out_ch if out_ch is not None else in_ch
        self.dimensions = dimensions
        self.with_conv = with_conv
        self.out_ch = out_ch
        self.fir = fir
        self.fir_kernel = fir_kernel
        
        if not fir and with_conv:
            self.Conv_0 = conv3x3(in_ch, out_ch, stride=2, dimensions=dimensions, key=key)
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
                key=key,
            )
        else:
            self.Conv_0 = None

    def __call__(self, x):
        if not self.fir:
            if self.with_conv:
                pad_width = [(0, 0), (0, 0)] + [(0, 1) for _ in range(self.dimensions)]
                x = jnp.pad(x, pad_width)
                x = self.Conv_0(x)
            else:
                # Average Pool
                x = x.reshape(1, *x.shape) # Add an artifical batch dimension to work with lax.conv
                w = jnp.ones((1, 1, *[2]*self.dimensions)) / 2 ** self.dimensions
                x = lax.conv(x, w, window_strides=[2]*self.dimensions, padding="VALID")
                x = x.reshape(*x.shape[1:]) # Remove batch dimension
        else:
            if not self.with_conv:
                x = downsample(x, self.fir_kernel, factor=2, dimensions=self.dimensions)
            else:
                x = self.Conv_0(x)
        return x

