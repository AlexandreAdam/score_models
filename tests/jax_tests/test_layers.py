import jax.numpy as jnp
from jax import random
from score_models.layers import StyleGANConv, UpsampleLayer, DownsampleLayer, Combine, ResnetBlockBigGANpp
from score_models.layers.attention_block import SelfAttentionBlock, ScaledAttentionLayer
from score_models.definitions import default_init
from score_models.utils import get_activation


def init_test_fn(shape, dtype=jnp.float32):
    return jnp.ones(shape, dtype=dtype)


def test_attention():
    key = random.PRNGKey(0)
    x = random.normal(key, shape=[10, 4, 8, 8])
    print(x[0, 0, 0, 0], x[0, 0, 0, 1])
    att = SelfAttentionBlock(4)
    y = att(x)
    print(y[0, 0, 0, 0], y[0, 0, 0, 1])
    x = random.normal(key, shape=[10, 4, 8, 8, 8])
    SelfAttentionBlock(4, dimensions=3)(x)
    x = random.normal(key, shape=[10, 4, 8])
    SelfAttentionBlock(4, dimensions=1)(x)
    
    x = random.normal(key, shape=(10, 5)) * 100
    B, D = x.shape
    temb = random.normal(key, shape=(B, D))
    context = jnp.stack([x, temb], axis=1)
    print("context shape", context.shape)
    att = ScaledAttentionLayer(dimensions=5)
    out = att(x.reshape(B, 1, D), context)
    print("shape",out.shape)
    print("out", out)

def test_resnet_biggan():
    act = get_activation("relu")
    layer = ResnetBlockBigGANpp(act=act, in_ch=8, out_ch=4, temb_dim=None, up=False, down=False, fir=False, skip_rescale=True, dimensions=2) 
    x = random.normal(random.PRNGKey(1), (1, 8, 8, 8))
    out = layer(x)
    assert out.shape == (1, 4, 8, 8)

    layer = ResnetBlockBigGANpp(act=act, in_ch=8, out_ch=4, temb_dim=10, up=False, down=False, fir=False, skip_rescale=True, dimensions=3) 
    x = random.normal(random.PRNGKey(2), (1, 8, 8, 8, 8))
    out = layer(x)
    assert out.shape == (1, 4, 8, 8, 8)

    layer = ResnetBlockBigGANpp(act=act, in_ch=8, out_ch=4, temb_dim=10, up=True, down=False, fir=False, skip_rescale=True, dimensions=3) 
    x = random.normal(random.PRNGKey(3), (1, 8, 8, 8, 8))
    out = layer(x)
    assert out.shape == (1, 4, 16, 16, 16)

    layer = ResnetBlockBigGANpp(act=act, in_ch=8, out_ch=4, temb_dim=10, up=False, down=True, fir=True, skip_rescale=True, dimensions=3) 
    x = random.normal(random.PRNGKey(4), (1, 8, 8, 8, 8))
    out = layer(x)
    assert out.shape == (1, 4, 4, 4, 4)

    layer = ResnetBlockBigGANpp(act=act, in_ch=8, out_ch=4, temb_dim=10, up=False, down=True, fir=True, skip_rescale=False, dimensions=1) 
    x = random.normal(random.PRNGKey(5), (1, 8, 8))
    out = layer(x)
    assert out.shape == (1, 4, 4)

def test_combine():
    x = random.normal(random.PRNGKey(6), (1, 1, 8, 8))
    y = random.normal(random.PRNGKey(7), (1, 1, 8, 8))
    layer = Combine(in_ch=1, out_ch=4, method="cat", dimensions=2)
    out = layer(x, y)
    assert out.shape == (1, 5, 8, 8)

    x = random.normal(random.PRNGKey(8), (1, 1, 8, 8, 8))
    y = random.normal(random.PRNGKey(9), (1, 1, 8, 8, 8))
    layer = Combine(in_ch=1, out_ch=4, method="cat", dimensions=3)
    out = layer(x, y)
    assert out.shape == (1, 5, 8, 8, 8)

    x = random.normal(random.PRNGKey(10), (1, 1, 8, 8, 8))
    y = random.normal(random.PRNGKey(11), (1, 4, 8, 8, 8))
    layer = Combine(in_ch=1, out_ch=4, method="sum", dimensions=3)
    out = layer(x, y)
    assert out.shape == (1, 4, 8, 8, 8)

def test_upsample_layer():
    x = random.normal(random.PRNGKey(12), (1, 1, 8, 8))
    layer = UpsampleLayer(1, 3, with_conv=True, fir=True, dimensions=2)
    out = layer(x)
    assert out.shape == (1, 3, 16, 16)

    x = random.normal(random.PRNGKey(13), (1, 1, 8))
    layer = UpsampleLayer(1, 3, with_conv=True, fir=True, dimensions=1)
    out = layer(x)
    assert out.shape == (1, 3, 16)

    x = random.normal(random.PRNGKey(14), (1, 1, 8))
    layer = UpsampleLayer(1, 1, with_conv=False, fir=False, dimensions=1)
    out = layer(x)
    assert out.shape == (1, 1, 16)

    x = random.normal(random.PRNGKey(15), (1, 1, 8, 8, 8))
    layer = UpsampleLayer(1, 1, with_conv=False, fir=False, dimensions=3)
    out = layer(x)
    assert out.shape == (1, 1, 16, 16, 16)

def test_downsample_layer():
    x = random.normal(random.PRNGKey(16), (1, 1, 8, 8))
    layer = DownsampleLayer(1, 3, with_conv=True, fir=True, dimensions=2)
    out = layer(x)
    assert out.shape == (1, 3, 4, 4)

    x = random.normal(random.PRNGKey(17), (1, 1, 8))
    layer = DownsampleLayer(1, 3, with_conv=True, fir=True, dimensions=1)
    out = layer(x)
    assert out.shape == (1, 3, 4)

    x = random.normal(random.PRNGKey(18), (1, 1, 8))
    layer = DownsampleLayer(1, 1, with_conv=False, fir=False, dimensions=1)
    out = layer(x)
    assert out.shape == (1, 1, 4)

    x = random.normal(random.PRNGKey(19), (1, 1, 8, 8, 8))
    layer = DownsampleLayer(1, 1, with_conv=False, fir=False, dimensions=3)
    out = layer(x)
    assert out.shape == (1, 1, 4, 4, 4)

def test_stylegan_conv_shape():
    key = random.PRNGKey(20)
    x = random.normal(key, (1, 1, 8, 8))
    conv = StyleGANConv(in_ch=1, out_ch=3, kernel=3, up=False, down=False, use_bias=True, kernel_init=default_init(), dimensions=2)
    out = conv(x)
    assert out.shape == (1, 3, 8, 8)

    conv = StyleGANConv(in_ch=1, out_ch=3, kernel=3, up=True, down=False, use_bias=True, kernel_init=default_init(), dimensions=2)
    out = conv(x)
    assert out.shape == (1, 3, 16, 16)

    conv = StyleGANConv(in_ch=1, out_ch=3, kernel=3, up=False, down=True, use_bias=True, kernel_init=default_init(), dimensions=2)
    out = conv(x)
    assert out.shape == (1, 3, 4, 4)

    x = random.normal(key, (1, 1, 8))
    conv = StyleGANConv(in_ch=1, out_ch=3, kernel=3, up=False, down=False, use_bias=True, kernel_init=default_init(), dimensions=1)
    out = conv(x)
    assert out.shape == (1, 3, 8)

    conv = StyleGANConv(in_ch=1, out_ch=3, kernel=3, up=True, down=False, use_bias=True, kernel_init=default_init(), dimensions=1)
    out = conv(x)
    assert out.shape == (1, 3, 16)

    conv = StyleGANConv(in_ch=1, out_ch=3, kernel=3, up=False, down=True, use_bias=True, kernel_init=default_init(), dimensions=1)
    out = conv(x)
    assert out.shape == (1, 3, 4)

    x = random.normal(key, (1, 1, 8, 8, 8))
    conv = StyleGANConv(in_ch=1, out_ch=3, kernel=3, up=False, down=False, use_bias=True, kernel_init=default_init(), dimensions=3)
    out = conv(x)
    assert out.shape == (1, 3, 8, 8, 8)

    conv = StyleGANConv(in_ch=1, out_ch=3, kernel=3, up=True, down=False, use_bias=True, kernel_init=default_init(), dimensions=3)
    out = conv(x)
    assert out.shape == (1, 3, 16, 16, 16)

    conv = StyleGANConv(in_ch=1, out_ch=3, kernel=3, up=False, down=True, use_bias=True, kernel_init=default_init(), dimensions=3)
    out = conv(x)
    assert out.shape == (1, 3, 4, 4, 4)

def test_stylegan_conv_resample_kernel():
    x = jnp.ones((1, 1, 8, 8))
    conv = StyleGANConv(in_ch=1, out_ch=3, kernel=3, up=True, down=False, use_bias=True, kernel_init=init_test_fn, dimensions=2)
    out = conv(x)
    print(out)
    assert jnp.all(out[..., 2:-2, 2:-2] == 9.)

    conv = StyleGANConv(in_ch=1, out_ch=3, kernel=3, up=False, down=True, use_bias=True, kernel_init=init_test_fn, dimensions=2)
    out = conv(x)
    print(out)
    assert jnp.all(out[..., 1:-1, 1:-1] == 9.)

def test_transposed_conv():
    # Test that we can downsample and upsample odd numbered images with correct padding
    from score_models.layers import ConvTransposed1dSame, ConvTransposed2dSame, ConvTransposed3dSame
    from score_models.layers import Conv1dSame, Conv2dSame, Conv3dSame
    
    B = 10
    D = 15
    C = 16
    K = 3
    for dim in [1, 2, 3]:
        x = random.normal(random.PRNGKey(dim), (B, C, *[D]*dim))
        layer_ = [Conv1dSame, Conv2dSame, Conv3dSame][dim-1]
        layer = layer_(C, C, K, stride=2)
        x_down= layer(x)
        print("Down", x_down.shape)
        assert x_down.shape == (B, C, *[D//2]*dim)
        
        layer_ = [ConvTransposed1dSame, ConvTransposed2dSame, ConvTransposed3dSame][dim-1]
        layer = layer_(C, C, K, stride=2)
        y = layer(x_down)
        print("Up", y.shape)
        assert y.shape == (B, C, *[D]*dim)

