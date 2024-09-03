import torch
import numpy as np
from score_models.sde import VESDE
from score_models import (
    GRFEnergyModel,
    MVGEnergyModel,
    JointScoreModel,
    SampleScoreModel,
    InterpolatedScoreModel,
    ConvolvedLikelihood,
    TweedieScoreModel,
)
import pytest


@pytest.mark.parametrize("psd_shape", [(32, 32), (32, 16), (25,), (64,)])
def test_grf(psd_shape):
    sde = VESDE(sigma_min=1e-2, sigma_max=10)

    if len(psd_shape) == 1:
        f = np.fft.fftfreq(psd_shape[0])
        f[0] = 1
        power_spectrum = 1.0 / f**2
    else:
        # Frequency indices
        u = np.fft.fftfreq(psd_shape[0])
        v = np.fft.fftfreq(psd_shape[1])

        # Create a grid of frequencies
        U, V = np.meshgrid(u, v, indexing="ij")

        # Compute the squared frequency magnitude
        freq_magnitude_squared = U**2 + V**2

        # Avoid division by zero for the zero frequency
        freq_magnitude_squared[0, 0] = 1

        # Inverse square of the frequency magnitude
        power_spectrum = 1.0 / freq_magnitude_squared

    psd = torch.tensor(power_spectrum, dtype=torch.float32)

    model = GRFEnergyModel(sde, power_spectrum=psd)

    samples = model.sample(shape=(2, *psd_shape), steps=25)

    assert torch.all(torch.isfinite(samples))


@pytest.mark.parametrize(
    "mean,cov",
    (
        ([0.0], [[1.0]]),  # 1D Gaussian
        ([0.0, 0.0], [[1.0, 0.1], [0.1, 1.0]]),  # 2D Gaussian
        (np.random.randn(5, 3), np.stack([np.eye(3)] * 5)),  # mixture of 5 3D Gaussians
    ),
)
def test_mvg(mean, cov):
    sde = VESDE(sigma_min=1e-2, sigma_max=10)
    mean = torch.tensor(mean, dtype=torch.float32)
    cov = torch.tensor(cov, dtype=torch.float32)
    model = MVGEnergyModel(
        sde,
        mean=mean,
        cov=cov,
    )

    samples = model.sample(shape=(2, mean.shape[-1]), steps=25)

    assert torch.all(torch.isfinite(samples))


@pytest.mark.parametrize("Nsamp,Ndim", ((10, 1), (1, 2), (5, 100)))
def test_joint_shared(Nsamp, Ndim):
    sde = VESDE(sigma_min=1e-2, sigma_max=10)
    model1 = SampleScoreModel(
        sde,
        samples=torch.randn(Nsamp, Ndim),
    )

    model2 = SampleScoreModel(
        sde,
        samples=torch.randn(Nsamp, Ndim),
    )

    model = JointScoreModel(
        sde,
        models=(model1, model2),
        x_shapes=[(Ndim,)],
        model_uses=[None, None],
    )

    samples = model.sample(shape=(2, Ndim), steps=25)

    assert torch.all(torch.isfinite(samples))


@pytest.mark.parametrize("Nsamp,Ndim1,Ndim2", ((10, 1, 3), (1, 2, 5), (5, 100, 1), (3, 1, 1)))
def test_joint_paired(Nsamp, Ndim1, Ndim2):
    sde = VESDE(sigma_min=1e-2, sigma_max=10)
    model1 = SampleScoreModel(
        sde,
        samples=torch.randn(Nsamp, Ndim1),
    )

    model2 = SampleScoreModel(
        sde,
        samples=torch.randn(Nsamp, Ndim2),
    )

    model = JointScoreModel(
        sde,
        models=(model1, model2),
        x_shapes=[(Ndim1,), (Ndim2,)],
        model_uses=[(0,), (1,)],
    )

    samples = model.sample(shape=(2, Ndim1 + Ndim2), steps=25)

    assert torch.all(torch.isfinite(samples))


@pytest.mark.parametrize("Nsamp,Ndim", ((10, 1), (1, 2), (5, 100)))
def test_sample_score(Nsamp, Ndim):
    sde = VESDE(sigma_min=1e-2, sigma_max=10)
    model = SampleScoreModel(
        sde,
        samples=torch.randn(Nsamp, Ndim),
    )

    samples = model.sample(shape=(2, Ndim), steps=25)

    assert torch.all(torch.isfinite(samples))


@pytest.mark.parametrize(
    "beta", ("linear", "square", "sqrt", "linear:2", "sqrt:2", "sin:2", lambda t: t**2)
)
@pytest.mark.parametrize("Nsamp,Ndim", ((10, 1), (1, 2)))
def test_interpolated(Nsamp, Ndim, beta):
    sde = VESDE(sigma_min=1e-2, sigma_max=10)
    model1 = SampleScoreModel(
        sde,
        samples=torch.randn(Nsamp, Ndim),
    )

    model2 = SampleScoreModel(
        sde,
        samples=torch.randn(Nsamp, Ndim),
    )

    model = InterpolatedScoreModel(
        sde,
        hight_model=model1,
        lowt_model=model2,
        beta_scheme=beta,
    )

    samples = model.sample(shape=(2, Ndim), steps=25)

    assert torch.all(torch.isfinite(samples))


def test_convolved_likelihood():
    sde = VESDE(sigma_min=1e-2, sigma_max=10)
    x_true = torch.randn(3)

    def fwd(x):
        return torch.arange(1, 4) * x + 2

    y = fwd(x_true) + torch.randn(3) * 0.1
    Sigma_y = torch.eye(3) * 0.1**2
    A = torch.func.jacrev(fwd)(x_true)

    priormodel = MVGEnergyModel(
        sde,
        mean=torch.zeros(3),
        cov=torch.eye(3),
    )

    likelihoodmodel = ConvolvedLikelihood(
        sde,
        y=y,
        Sigma_y=Sigma_y,
        x_shape=(3,),
        A=A,
    )

    model = JointScoreModel(
        sde,
        models=(priormodel, likelihoodmodel),
        x_shapes=[
            (3,),
        ],
        model_uses=[None, None],
    )

    samples = model.sample(shape=(2, 3), steps=25)

    assert torch.all(torch.isfinite(samples))


def test_tweedie():
    sde = VESDE(sigma_min=1e-2, sigma_max=10)
    x_true = torch.randn(3)

    def fwd(x):
        return torch.arange(1, 4) * x + 2

    y = fwd(x_true) + torch.randn(3) * 0.1
    Sigma_y = torch.eye(3) * 0.1**2
    A = torch.func.jacrev(fwd)(x_true)

    priormodel = MVGEnergyModel(
        sde,
        mean=torch.zeros(3),
        cov=torch.eye(3),
    )

    def log_likelihood(sigma_t, x):
        r = y - A @ x
        ret = (
            r.reshape(1, -1)
            @ torch.linalg.inv(
                Sigma_y
                + sigma_t**2
                * torch.eye(Sigma_y.shape[0], dtype=Sigma_y.dtype, device=Sigma_y.device)
            )
            @ r.reshape(-1, 1)
        )
        return ret.squeeze()

    likelihoodmodel = TweedieScoreModel(
        sde,
        prior_model=priormodel,
        log_likelihood=log_likelihood,
    )

    model = JointScoreModel(
        sde,
        models=(priormodel, likelihoodmodel),
        x_shapes=[
            (3,),
        ],
        model_uses=[None, None],
    )

    samples = model.sample(shape=(2, 3), steps=25)

    assert torch.all(torch.isfinite(samples))
