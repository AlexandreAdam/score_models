import torch
import torch.nn.functional as F
from .conv_layers import conv3x3
from .up_or_downsampling import downsample
from .style_gan_conv import StyleGANConv
from ..definitions import default_init

AVGPOOL_FUNC = {1: F.avg_pool1d,
                2: F.avg_pool2d,
                3: F.avg_pool3d}

class DownsampleLayer(torch.nn.Module):
    def __init__(
            self, 
            in_ch=None, 
            out_ch=None, 
            with_conv=False, 
            fir=False, 
            fir_kernel=(1, 3, 3, 1),
            dimensions:int = 2,
            ):
        super().__init__()
        out_ch = out_ch if out_ch is not None else in_ch
        if out_ch != in_ch:
            assert with_conv
        self.dimensions = dimensions
        self.with_conv = with_conv
        self.out_ch = out_ch
        if not fir:
            if with_conv:
                self.Conv_0 = conv3x3(in_ch, out_ch, stride=2, dimensions=dimensions)
        else:
            if with_conv:
                self.Conv_0 = StyleGANConv(in_ch, out_ch,
                                        kernel=3, down=True,
                                        resample_kernel=fir_kernel,
                                        use_bias=True,
                                        kernel_init=default_init(),
                                        dimensions=dimensions
                                        )
        self.fir = fir
        self.fir_kernel = fir_kernel
    
    def forward(self, x):
        if not self.fir:
            if self.with_conv:
                pad = [0, 1]*self.dimensions
                x = F.pad(x, pad)
                x = self.Conv_0(x)
            else:
                x = AVGPOOL_FUNC[self.dimensions](x, 2, stride=2)
        else:
            if not self.with_conv:
                x = downsample(x, self.fir_kernel, factor=2, dimensions=self.dimensions)
            else:
                x = self.Conv_0(x)
        return x

