import jax.numpy as jnp
import jax.lax as lax
import equinox as eqx


class StyleGANConv(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray = None
    up: bool
    down: bool
    resample_kernel: tuple
    kernel: int
    use_bias: bool
    dimensions: int

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        up: bool = False,
        down: bool = False,
        resample_kernel: tuple = (1, 3, 3, 1),
        use_bias: bool = True,
        kernel_init: callable = None,
        dimensions: int = 2,
    ):
        super().__init__()
        assert not (up and down)
        assert kernel >= 1 and kernel % 2 == 1
        assert dimensions in [1, 2, 3]

        self.up = up
        self.down = down
        self.resample_kernel = resample_kernel
        self.kernel = kernel
        self.use_bias = use_bias
        self.dimensions = dimensions

        weight_shape = (out_ch, in_ch, *(kernel,) * dimensions)
        self.weight = (
            kernel_init(weight_shape)
            if kernel_init
            else eqx.nn.initializers.normal()(weight_shape)
        )

        if use_bias:
            self.bias = jnp.zeros(out_ch)

        if dimensions == 1:
            from .up_or_downsampling1d import upsample_conv_1d, conv_downsample_1d

            self.upsample_conv = upsample_conv_1d
            self.downsample_conv = conv_downsample_1d
        elif dimensions == 2:
            from .up_or_downsampling2d import upsample_conv_2d, conv_downsample_2d

            self.upsample_conv = upsample_conv_2d
            self.downsample_conv = conv_downsample_2d
        elif dimensions == 3:
            from .up_or_downsampling3d import upsample_conv_3d, conv_downsample_3d

            self.upsample_conv = upsample_conv_3d
            self.downsample_conv = conv_downsample_3d

    def __call__(self, x):
        if self.up:
            x = self.upsample_conv(x, self.weight, k=self.resample_kernel)
        elif self.down:
            x = self.downsample_conv(x, self.weight, k=self.resample_kernel)
        else:
            if self.dimensions == 1:
                dimension_numbers = ("NWC", "WIO", "NWC")
            elif self.dimensions == 2:
                dimension_numbers = ("NHWC", "HWIO", "NHWC")
            else:
                dimension_numbers = ("NDHWC", "DHWIO", "NDHWC")
            x = lax.conv_general_dilated(
                x,
                self.weight,
                window_strides=(1,) * self.dimensions,
                padding="SAME",
                dimension_numbers=dimension_numbers,
            )

        if self.use_bias:
            bias_shape = (1, -1) + (1,) * (self.dimensions)
            x += self.bias.reshape(bias_shape)
        return x
