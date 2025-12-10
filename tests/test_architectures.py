from score_models.architectures import NCSNpp, DDPM, MLP, Encoder
import torch
import pytest

@pytest.mark.parametrize("D", [1, 2, 3])
@pytest.mark.parametrize("C", [1, 3])
@pytest.mark.parametrize("ch_mult", [(1, 1), (1, 2), (1, 1, 1)])
@pytest.mark.parametrize("nf", [2, 4]) # Number of filters needs to be power of 2 when small, or at least divisible by 4, need to debug this 
@pytest.mark.parametrize("num_res_blocks", [1, 3])
@pytest.mark.parametrize("attention", [True, False])
@pytest.mark.parametrize("Net", [NCSNpp, DDPM])
def test_unets(D, C, ch_mult, nf, num_res_blocks, attention, Net):
    B = 2
    P = 8
    x = torch.randn(B, C, *[P]*D) * 500
    t = torch.rand([B])
    model = Net(C, dimensions=D, nf=nf, ch_mult=ch_mult, num_res_blocks=num_res_blocks, attention=attention)
    out = model(t, x)
    assert out.shape == torch.Size([B, C, *[P]*D])
    assert torch.isfinite(out).all()

@pytest.mark.parametrize("width", [10, 50])
@pytest.mark.parametrize("layers", [1, 3])
@pytest.mark.parametrize("attention", [True, False])
def test_mlp(width, layers, attention):
    B = 2
    C = 10
    x = torch.randn(B, C)
    t = torch.randn([B])
    model = MLP(
            C, 
            layers=layers, 
            width=width,
            attention=attention)
    out = model(t, x)
    assert out.shape == torch.Size([B, C])
    assert torch.isfinite(out).all()


# @pytset.mark.parametrize("D", [1, 2, 3])
@pytest.mark.parametrize("D", [2])
@pytest.mark.parametrize("C", [1, 3])
@pytest.mark.parametrize("ch_mult", [(1, 1), (1, 2), (1, 1, 1)])
@pytest.mark.parametrize("latent_size", (1, 10, 100))
def test_encoder(D, C, ch_mult, latent_size):
    B = 2
    P = 16
    x = torch.randn(B, C, *[P]*D)
    t = torch.randn([B])
    model = Encoder(
            pixels=P,
            channels=C,
            dimensions=D,
            ch_mult=ch_mult,
            latent_size=latent_size
            )
    out = model(t, x)
    assert out.shape == torch.Size([B, latent_size])
    assert torch.isfinite(out).all()
