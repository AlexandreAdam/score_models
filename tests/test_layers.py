import torch
from score_models.layers import StyleGANConv, UpsampleLayer, DownsampleLayer, Combine, ResnetBlockBigGANpp
from score_models.layers.attention_block import SelfAttentionBlock, ScaledAttentionLayer
from score_models.definitions import default_init
from score_models.utils import get_activation
import numpy as np

def init_test_fn(shape, dtype=torch.float32, device="cpu"):
    return torch.ones(shape, dtype=dtype, device=device)

def test_attention():
    x = torch.randn([10, 4, 8, 8])
    print(x[0, 0, 0, 0], x[0, 0, 0, 1])
    att = SelfAttentionBlock(4)
    y = att(x)
    print(y[0, 0, 0, 0], y[0, 0, 0, 1])
    x = torch.randn([10, 4, 8, 8, 8])
    SelfAttentionBlock(4, dimensions=3)(x)
    x = torch.randn([10, 4, 8])
    SelfAttentionBlock(4, dimensions=1)(x)
    
    x = torch.randn(10, 5) * 100
    B, D = x.shape
    temb = torch.randn(B, D)
    context = torch.stack([x, temb], dim=1)
    print("context shape", context.shape)
    att = ScaledAttentionLayer(dimensions=5)
    out = att(x.view(B, 1, D), context)
    print("shape",out.shape)
    print("out", out)


def test_resnet_biggan():
    # out channels has to be at least 4
    act = get_activation("relu")
    layer = ResnetBlockBigGANpp(act=act, in_ch=8, out_ch=4, temb_dim=None, up=False, down=False, fir=False, skip_rescale=True, dimensions=2) 
    x = torch.randn(1, 8, 8, 8)
    out  = layer(x)
    assert list(out.shape) == [1, 4, 8, 8]

    layer = ResnetBlockBigGANpp(act=act, in_ch=8, out_ch=4, temb_dim=10, up=False, down=False, fir=False, skip_rescale=True, dimensions=3) 
    x = torch.randn(1, 8, 8, 8, 8)
    out  = layer(x)
    assert list(out.shape) == [1, 4, 8, 8, 8]

    layer = ResnetBlockBigGANpp(act=act, in_ch=8, out_ch=4, temb_dim=10, up=True, down=False, fir=False, skip_rescale=True, dimensions=3) 
    x = torch.randn(1, 8, 8, 8, 8)
    out  = layer(x)
    assert list(out.shape) == [1, 4, 16, 16, 16]

    layer = ResnetBlockBigGANpp(act=act, in_ch=8, out_ch=4, temb_dim=10, up=False, down=True, fir=True, skip_rescale=True, dimensions=3) 
    x = torch.randn(1, 8, 8, 8, 8)
    out  = layer(x)
    assert list(out.shape) == [1, 4, 4, 4, 4]

    layer = ResnetBlockBigGANpp(act=act, in_ch=8, out_ch=4, temb_dim=10, up=False, down=True, fir=True, skip_rescale=False, dimensions=1) 
    x = torch.randn(1, 8, 8)
    out  = layer(x)
    assert list(out.shape) == [1, 4, 4]

def test_combine():
    x = torch.randn(1, 1, 8, 8)
    y = torch.randn(1, 1, 8, 8)
    layer = Combine(in_ch=1, out_ch=4, method="cat", dimensions=2)
    out = layer(x, y)
    assert list(out.shape) == [1, 5, 8, 8]

    x = torch.randn(1, 1, 8, 8, 8)
    y = torch.randn(1, 1, 8, 8, 8)
    layer = Combine(in_ch=1, out_ch=4, method="cat", dimensions=3)
    out = layer(x, y)
    assert list(out.shape) == [1, 5, 8, 8, 8]

    x = torch.randn(1, 1, 8, 8, 8)
    y = torch.randn(1, 4, 8, 8, 8)
    layer = Combine(in_ch=1, out_ch=4, method="sum", dimensions=3)
    out = layer(x, y)
    assert list(out.shape) == [1, 4, 8, 8, 8]


def test_upsample_layer():
    x = torch.randn(1, 1, 8, 8)
    layer = UpsampleLayer(1, 3, with_conv=True, fir=True, dimensions=2)
    out = layer(x)
    assert list(out.shape) == [1, 3, 16, 16] 

    x = torch.randn(1, 1, 8)
    layer = UpsampleLayer(1, 3, with_conv=True, fir=True, dimensions=1)
    out = layer(x)
    assert list(out.shape) == [1, 3, 16] 

    x = torch.randn(1, 1, 8)
    layer = UpsampleLayer(1, 1, with_conv=False, fir=False, dimensions=1)
    out = layer(x)
    assert list(out.shape) == [1, 1, 16] 

    x = torch.randn(1, 1, 8, 8, 8)
    layer = UpsampleLayer(1, 1, with_conv=False, fir=False, dimensions=3)
    out = layer(x)
    assert list(out.shape) == [1, 1, 16, 16, 16] 
    

def test_downsample_layer():
    x = torch.randn(1, 1, 8, 8)
    layer = DownsampleLayer(1, 3, with_conv=True, fir=True, dimensions=2)
    out = layer(x)
    assert list(out.shape) == [1, 3, 4, 4] 

    x = torch.randn(1, 1, 8)
    layer = DownsampleLayer(1, 3, with_conv=True, fir=True, dimensions=1)
    out = layer(x)
    assert list(out.shape) == [1, 3, 4] 

    x = torch.randn(1, 1, 8)
    layer = DownsampleLayer(1, 1, with_conv=False, fir=False, dimensions=1)
    out = layer(x)
    assert list(out.shape) == [1, 1, 4] 

    x = torch.randn(1, 1, 8, 8, 8)
    layer = DownsampleLayer(1, 1, with_conv=False, fir=False, dimensions=3)
    out = layer(x)
    assert list(out.shape) == [1, 1, 4, 4, 4] 


def test_stylegan_conv_shape():
    x = torch.randn(1, 1, 8, 8)
    conv = StyleGANConv(in_ch=1, out_ch=3, kernel=3, up=False, down=False, use_bias=True, kernel_init=default_init(), dimensions=2)
    out = conv(x)
    assert list(out.shape) == [1, 3, 8, 8]
   
    conv = StyleGANConv(in_ch=1, out_ch=3, kernel=3, up=True, down=False, use_bias=True, kernel_init=default_init(), dimensions=2)
    out = conv(x)
    assert list(out.shape) == [1, 3, 16, 16]

    conv = StyleGANConv(in_ch=1, out_ch=3, kernel=3, up=False, down=True, use_bias=True, kernel_init=default_init(), dimensions=2)
    out = conv(x)
    assert list(out.shape) == [1, 3, 4, 4]
    
    x = torch.randn(1, 1, 8)
    conv = StyleGANConv(in_ch=1, out_ch=3, kernel=3, up=False, down=False, use_bias=True, kernel_init=default_init(), dimensions=1)
    out = conv(x)
    assert list(out.shape) == [1, 3, 8]

    conv = StyleGANConv(in_ch=1, out_ch=3, kernel=3, up=True, down=False, use_bias=True, kernel_init=default_init(), dimensions=1)
    out = conv(x)
    assert list(out.shape) == [1, 3, 16]

    conv = StyleGANConv(in_ch=1, out_ch=3, kernel=3, up=False, down=True, use_bias=True, kernel_init=default_init(), dimensions=1)
    out = conv(x)
    assert list(out.shape) == [1, 3, 4]

    x = torch.randn(1, 1, 8, 8, 8)
    conv = StyleGANConv(in_ch=1, out_ch=3, kernel=3, up=False, down=False, use_bias=True, kernel_init=default_init(), dimensions=3)
    out = conv(x)
    assert list(out.shape) == [1, 3, 8, 8, 8]

    conv = StyleGANConv(in_ch=1, out_ch=3, kernel=3, up=True, down=False, use_bias=True, kernel_init=default_init(), dimensions=3)
    out = conv(x)
    assert list(out.shape) == [1, 3, 16, 16, 16]

    conv = StyleGANConv(in_ch=1, out_ch=3, kernel=3, up=False, down=True, use_bias=True, kernel_init=default_init(), dimensions=3)
    out = conv(x)
    assert list(out.shape) == [1, 3, 4, 4, 4]


def test_stylegan_conv_resample_kernel():
    x = torch.ones(1, 1, 8, 8)
    conv = StyleGANConv(in_ch=1, out_ch=3, kernel=3, up=True, down=False, use_bias=True, kernel_init=init_test_fn, dimensions=2)
    out = conv(x)
    print(out)
    assert np.all(out.detach().numpy()[..., 2:-2, 2:-2] == 9.)

    conv = StyleGANConv(in_ch=1, out_ch=3, kernel=3, up=False, down=True, use_bias=True, kernel_init=init_test_fn, dimensions=2)
    out = conv(x)
    print(out)
    assert np.all(out.detach().numpy()[..., 1:-1, 1:-1] == 9.)

