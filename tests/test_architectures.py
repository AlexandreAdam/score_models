from score_models.architectures import NCSNpp, DDPM, MLP
import torch


def test_ddpm():
    x = torch.randn(size=[1, 1, 128, 128]) * 230
    t = torch.randn([1])
    model = DDPM(1, nf=256)
    model(x=x, t=t)


def test_ncsnpp():
    x = torch.randn(size=[1, 1, 128, 128]) * 500
    t = torch.randn([1])
    model = NCSNpp(1, dimensions=2, nf=128, ch_mult=(1, 1, 2, 2, 2, 2, 2))
    model(x=x, t=t)


def test_ncsnpp1d():
    x = torch.randn(size=[1, 1, 2560]) * 500
    t = torch.randn([1])
    model = NCSNpp(1, dimensions=1, nf=128, ch_mult=(1, 1, 2, 2, 2, 2), attention=True)
    model(x=x, t=t)


def test_ncsnpp3d():
    x = torch.randn(size=[1, 1, 96, 96, 96]) * 500
    t = torch.randn([1])
    model = NCSNpp(1, dimensions=3, nf=16, ch_mult=(1, 1, 2, 2, 2), attention=True)
    model(x=x, t=t)


def test_mlp():
    x = torch.randn(size=[10, 100]) * 100
    t = torch.randn([10])
    model = MLP(input_dimensions=100, units=100, layers=3, time_embedding_dimensions=32, time_branch_layers=2, bottleneck=10, attention=True)
    model(x=x, t=t)
    
