from score_models import DDPM, NCSNpp, NCSNpp1d, NCSNpp3d, NCSNppLog
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
    model.score(x, t)
    model.sample(size=(1, 1, 128, 128), N=2)


def test_ncsnpp1d():
    x = torch.randn(size=[1, 1, 2560]) * 500
    t = torch.randn([1])
    model = NCSNpp1d(1, nf=128, ch_mult=(1, 1, 2, 2, 2, 2), attention=True)
    model(x, t)
    model.score(x, t)
    model.sample(size=(2, 1, 2560), N=2)


def test_ncsnpp3d():
    x = torch.randn(size=[1, 1, 96, 96, 96]) * 500
    t = torch.randn([1])
    model = NCSNpp3d(1, nf=16, ch_mult=(1, 1, 2, 2, 2, 2), attention=True)
    model(x, t)
    model.score(x, t)
    model.sample(size=(1, 1, 96, 96, 96), N=2)



def test_ncnsnpplog():
    hp = {
  "channels": 1,
  "nf": 128,
  "ch_mult": [2, 2, 2, 2],
  "image_size": 64,
  "activation_type": "swish",
  "resample_with_conv": True,
  "fir": True,
  "fir_kernel": [1, 3, 3, 1],
  "skip_rescale": True,
  "progressive": "output_skip",
  "progressive_input": "input_skip",
  "init_scale": 1e-2,
  "fourier_scale": 16.0,
  "resblock_type": "biggan",
  "combine_method": "sum",
  "num_res_blocks": 2,
  "dropout": 0,
  "attention": False,
  "beta0": 1e-2,
  "beta1": 1e5,
  "sigma_min": 1e-3,
  "sigma_max": 1e4
}
    x = torch.randn(size=[1, 1, 64, 64]) * 500
    t = torch.randn([1])
    model = NCSNppLog(**hp)
    model(x, t)
    model.score(x, t)
