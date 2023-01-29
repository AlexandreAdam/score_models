from score_models import DDPM, NCSNpp
import torch


def test_ddpm():
    x = torch.randn(size=[1, 1, 256, 256]) * 230
    t = torch.randn([1])
    model = DDPM(1, 256, sigma_max=230)
    model(x, t)


def test_ncsnpp():
    x = torch.randn(size=[1, 1, 128, 128]) * 500
    t = torch.randn([1])
    model = NCSNpp(1, 128, ch_mult=(1, 1, 2, 2, 2, 2, 2))
    model(x, t)
