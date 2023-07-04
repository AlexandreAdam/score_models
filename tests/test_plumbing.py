from score_models.architectures import NCSNpp, DDPM
import torch


def test_ddpm():
    x = torch.randn(size=[1, 1, 128, 128]) * 230
    t = torch.randn([1])
    model = DDPM(1, nf=256)
    model(x, t)


def test_ncsnpp():
    x = torch.randn(size=[1, 1, 128, 128]) * 500
    t = torch.randn([1])
    model = NCSNpp(1, dimensions=2, nf=128, ch_mult=(1, 1, 2, 2, 2, 2, 2))
    model(x, t)


def test_ncsnpp1d():
    x = torch.randn(size=[1, 1, 2560]) * 500
    t = torch.randn([1])
    model = NCSNpp(1, dimensions=1, nf=128, ch_mult=(1, 1, 2, 2, 2, 2), attention=True)
    model(x, t)


def test_ncsnpp3d():
    x = torch.randn(size=[1, 1, 96, 96, 96]) * 500
    t = torch.randn([1])
    model = NCSNpp(1, dimensions=3, nf=16, ch_mult=(1, 1, 2, 2, 2), attention=True)
    model(x, t)

