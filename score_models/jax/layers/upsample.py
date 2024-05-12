import equinox as eqx
import jax
from typing import Optional
from jaxtyping import PRNGKeyArray
from .conv_layers import conv3x3
from .style_gan_conv import StyleGANConv
from .up_or_downsampling import upsample
from ..definitions import default_init


class UpsampleLayer(eqx.Module):
    fir: bool
    with_conv: bool
    fir_kernel: tuple
    out_ch: Optional[int]
    conv_0: eqx.Module
    conv1d_0: eqx.Module
    
    def __init__(
        self,
        in_ch: int,
        out_ch: Optional[int] = None,
        with_conv: bool = False,
        fir: bool = False,
        fir_kernel: tuple = (1, 3, 3, 1),
        dimensions: int = 2,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__()
        self.fir = fir
        self.with_conv = with_conv
        self.fir_kernel = fir_kernel
        self.out_ch = out_ch if out_ch else in_ch

        if self.out_ch != in_ch:
            assert with_conv
        if not fir:
            if with_conv:
                self.conv_0 = conv3x3(
                    in_ch, self.out_ch, dimensions=dimensions, key=key
                )
                self.conv1d_0 = None
        else:
            if with_conv:
                self.conv_0 = None
                self.conv1d_0 = StyleGANConv(
                    in_ch,
                    self.out_ch,
                    kernel=3,
                    up=True,
                    resample_kernel=fir_kernel,
                    use_bias=True,
                    kernel_init=default_init(),
                    dimensions=dimensions,
                    key=key,
                )
        if not with_conv:
            self.conv_0 = None
            self.conv1d_0 = None

    def __call__(self, x, key=jax.random.PRNGKey(0)):
        C, *D = x.shape
        if not self.fir:
            h = jax.image.resize(x, shape=(self.out_ch, *[d * 2 for d in D]), method="nearest")
            if self.with_conv:
                h = self.conv_0(h)
        else:
            if not self.with_conv:
                h = upsample(x, self.fir_kernel, factor=2, dimensions=len(D))
            else:
                h = self.conv1d_0(x)
        return h
