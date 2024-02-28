from score_models.architectures import NCSNpp, DDPM, MLP
import torch


def test_ddpm():
    x = torch.randn(size=[1, 1, 32, 32]) * 230
    t = torch.randn([1])
    model = DDPM(1, nf=64, ch_mult=(1, 1, 2, 2))
    model(t=t, x=x)


def test_ddpm_smallnf():
    x = torch.randn(size=[1, 1, 32, 32]) * 230
    t = torch.randn([1])
    model = DDPM(1, nf=8, ch_mult=(1, 1, 2, 2))
    model(t=t, x=x)


def test_ncsnpp():
    x = torch.randn(size=[1, 1, 32, 32]) * 500
    t = torch.randn([1])
    model = NCSNpp(1, dimensions=2, nf=8, ch_mult=(2, 2, 2, 2), num_res_blocks=3)
    model(t=t, x=x)


def test_ncsnpp1d():
    x = torch.randn(size=[1, 1, 256]) * 500
    t = torch.randn([1])
    model = NCSNpp(1, dimensions=1, nf=8, ch_mult=(1, 1, 2, 2), attention=True)
    model(t=t, x=x)


def test_ncsnpp3d():
    x = torch.randn(size=[1, 1, 32, 32, 32]) * 500
    t = torch.randn([1])
    model = NCSNpp(1, dimensions=3, nf=8, ch_mult=(1, 1, 2, 2), attention=True)
    model(t=t, x=x)


def test_mlp():
    x = torch.randn(size=[10, 10]) * 100
    t = torch.randn([10])
    model = MLP(
        dimensions=10,
        units=10,
        layers=3,
        time_embedding_dimensions=16,
        time_branch_layers=2,
        bottleneck=10,
        attention=True,
    )
    model(t=t, x=x)

    x = torch.randn(size=[1, 10]) * 100
    t = torch.randn([1])
    model = MLP(
        dimensions=10, units=10, layers=2, time_embedding_dimensions=16, time_branch_layers=1
    )
    model(t=t, x=x)
