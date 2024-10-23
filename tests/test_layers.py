import torch
from score_models.layers import StyleGANConv, UpsampleLayer, DownsampleLayer, Combine, ResnetBlockBigGANpp
from score_models.layers.attention_block import SelfAttentionBlock, ScaledAttentionLayer
from score_models.definitions import default_init
from score_models.utils import get_activation
from score_models.layers.up_or_downsampling import downsample, upsample
import numpy as np
import pytest

def init_test_fn(shape, dtype=torch.float32, device="cpu"):
    return torch.ones(shape, dtype=dtype, device=device)

@pytest.mark.parametrize("D", [1, 2, 3])
@pytest.mark.parametrize("P", [8])
@pytest.mark.parametrize("C", [4])
@pytest.mark.parametrize("B", [10])
def test_attention(B, D, C, P):
    x = torch.randn([B, C, *[P]*D])
    att = SelfAttentionBlock(C, dimensions=D)
    y = att(x)
    assert y.shape == x.shape
    

@pytest.mark.parametrize("C", [4])
@pytest.mark.parametrize("B", [10])
def test_scaled_attention_layer(C, B):
    x = torch.randn(B, C) * 100
    temb = torch.randn(B, C)
    context = torch.stack([x, temb], dim=1)
    att = ScaledAttentionLayer(channels=C)
    out = att(x.view(B, 1, C), context)
    assert out.squeeze().shape == x.shape
    
    print("context shape", context.shape)
    print("shape",out.shape)
    print("out", out)


@pytest.mark.parametrize("D", [1, 2, 3])
@pytest.mark.parametrize("P", [4, 8])
@pytest.mark.parametrize("Cin", [4])
@pytest.mark.parametrize("Cout", [2, 4])
@pytest.mark.parametrize("temb_dim", [None, 10])
@pytest.mark.parametrize("up_down", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("fir", [True, False])
@pytest.mark.parametrize("skip_rescale", [True, False])
def test_resnet_biggan(D, P, Cin, Cout, temb_dim, up_down, fir, skip_rescale):
    up = up_down[0]
    down = up_down[1]
    layer = ResnetBlockBigGANpp(
            act=get_activation("relu"), 
            in_ch=Cin, 
            out_ch=Cout, 
            temb_dim=temb_dim, 
            up=up, 
            down=down, 
            fir=fir, 
            skip_rescale=skip_rescale, 
            dimensions=D) 
    
    x = torch.randn(1, Cin, *[P]*D)
    out  = layer(x)
    Pout = P*2 if up else P//2 if down else P
    assert list(out.shape) == [1, Cout, *[Pout]*D]

@pytest.mark.parametrize("D", [1, 2, 3])
@pytest.mark.parametrize("P", [4, 8])
@pytest.mark.parametrize("Cin", [4])
@pytest.mark.parametrize("Cout", [2, 4])
@pytest.mark.parametrize("method", ["cat", "sum"])
def test_combine(D, P, Cin, Cout, method):
    if method == "sum":
        Cout = Cin # Sum requires the same number of channels
    layer = Combine(in_ch=Cin, out_ch=Cout, method=method, dimensions=D)
    x = torch.randn(1, Cin, *[P]*D)
    y = torch.randn(1, Cin, *[P]*D)
    out = layer(x, y)
    if method == "cat": # Concatenation will append the channels
        assert list(out.shape) == [1, Cin+Cout, *[P]*D]
    else:
        assert list(out.shape) == [1, Cout, *[P]*D]

def test_combine_errors():
    with pytest.raises(ValueError) as e:
        layer = Combine(in_ch=4, out_ch=6, method="not_a_method", dimensions=2)
        assert "Method not_a_method not recognized for the Combine layer." in str(e)

@pytest.mark.parametrize("D", [1, 2, 3])
@pytest.mark.parametrize("P", [4, 8])
@pytest.mark.parametrize("Cin", [1, 3])
@pytest.mark.parametrize("Cout", [3, 4])
@pytest.mark.parametrize("fir", [True, False])
@pytest.mark.parametrize("with_conv", [True, False])
def test_up_down_sampling_layer(D, P, Cin, Cout, fir, with_conv):
    x = torch.randn(1, Cin, *[P]*D)
    # Upsample layer
    if Cin != Cout: # If the number of channels is different, we need to use a convolutional layer
        with_conv = True
    layer = UpsampleLayer(Cin, Cout, with_conv=with_conv, fir=fir, dimensions=D)
    out = layer(x)
    assert list(out.shape) == [1, Cout, *[2*P]*D]
    # Downsample
    layer = DownsampleLayer(Cin, Cout, with_conv=with_conv, fir=fir, dimensions=D)
    out = layer(x)
    assert list(out.shape) == [1, Cout, *[P//2]*D]

@pytest.mark.parametrize("D", [1, 2, 3])
@pytest.mark.parametrize("P", [8, 16])
@pytest.mark.parametrize("factor", [[2, 1]])
def test_uneven_pooling(D, P, factor):
    # TODO: Generalize to 1d 3d
    Px, Py = 8, 16
    x = torch.ones(1, 1, Px, Py)
    k = (1, 3, 3, 1)
    y = downsample(x, k=k, factor=factor, dimensions=2)
    assert y.shape == torch.Size([1, 1, Px//factor[0], Py//factor[1]])
    
    y = upsample(x, factor=factor, dimensions=2)
    assert y.shape == torch.Size([1, 1, Px*factor[0], Py*factor[1]])
    
    assert 0 == 1

@pytest.mark.parametrize("D", [1, 2, 3])
@pytest.mark.parametrize("P", [4, 8])
@pytest.mark.parametrize("Cin", [1, 2])
@pytest.mark.parametrize("Cout", [3, 4])
@pytest.mark.parametrize("up_down", [(False, False), (True, False), (False, True)])
def test_stylegan_conv_shape(D, P, Cin, Cout, up_down):
    up = up_down[0]
    down = up_down[1]
    conv = StyleGANConv(
            in_ch=Cin,
            out_ch=Cout,
            kernel=3,
            up=up,
            down=down,
            use_bias=True,
            kernel_init=default_init(),
            dimensions=D
            )

    x = torch.randn(1, Cin, *[P]*D)
    out = conv(x)
    Pout = P*2 if up else P//2 if down else P
    assert list(out.shape) == [1, Cout, *[Pout]*D]

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

def test_transposed_conv():
    # Test that we can downsample and upsample odd numbered images with correct padding
    from score_models.layers import ConvTransposed1dSame, ConvTransposed2dSame, ConvTransposed3dSame
    from score_models.layers import Conv1dSame, Conv2dSame, Conv3dSame
    
    B = 10
    D = 15
    C = 16
    K = 3
    for dim in [1, 2, 3]:
        x = torch.randn(B, C, *[D]*dim)
        layer_ = [Conv1dSame, Conv2dSame, Conv3dSame][dim-1]
        layer = layer_(C, C, K, stride=2)
        x_down= layer(x)
        print("Down", x_down.shape)
        assert x_down.shape == torch.Size([B, C, *[D//2]*dim])
        
        layer_ = [ConvTransposed1dSame, ConvTransposed2dSame, ConvTransposed3dSame][dim-1]
        layer = layer_(C, C, K, stride=2)
        y = layer(x_down)
        print("Up", x.shape)
        assert x.shape == torch.Size([B, C, *[D]*dim])
        
