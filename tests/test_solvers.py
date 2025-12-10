import torch
import numpy as np
from score_models.sde import VESDE
from score_models import (
    MVGEnergyModel,
    Solver,
    ODESolver,
    EMSDESolver,
    HeunODESolver,
    MVGScoreModel,
)
import pytest


def test_solver_constructor():
    with pytest.raises(TypeError):  # abstract class cant be created
        Solver(None)
    assert isinstance(Solver(None, solver="EMSDESolver"), EMSDESolver), "EMSDESolver not created"
    assert isinstance(Solver(None, solver="HeunODESolver"), HeunODESolver), "HeunODESolver not created"
    assert isinstance(EMSDESolver(None), Solver), "EMSDESolver not created"
    with pytest.raises(ValueError):  # unknown solver
        Solver(None, solver="random_solver")
    with pytest.raises(ValueError):  # unknown ode solver
        ODESolver(None, solver="EMSDESolver")


@pytest.mark.parametrize(
    "mean,cov",
    (
        ([0.0], [[1.0]]),  # 1D Gaussian
        ([0.0, 0.0], [[1.0, 0.1], [0.1, 1.0]]),  # 2D Gaussian
    ),
)
@pytest.mark.parametrize(
    "solver", ["EMSDESolver", "HeunSDESolver", "RK4SDESolver", "EulerODESolver", "HeunODESolver", "RK4ODESolver"]
)
def test_solver_sample(solver, mean, cov):
    sde = VESDE(sigma_min=1e-2, sigma_max=10)
    mean = torch.tensor(mean, dtype=torch.float32)
    cov = torch.tensor(cov, dtype=torch.float32)
    model = MVGEnergyModel(
        sde,
        mean=mean,
        cov=cov,
    )
    samples = model.sample(
        shape=(100, mean.shape[-1]),
        steps=50,
        solver=solver,
        kill_on_nan=True,
    )
    assert torch.all(torch.isfinite(samples))
    assert torch.allclose(samples.mean(dim=0), mean, atol=1), "mean not close"
    assert torch.allclose(samples.std(dim=0), torch.tensor(1.0), atol=1), "std not close"


@pytest.mark.parametrize(
    "mean,cov",
    (
        ([0.0], [[1.0]]),  # 1D Gaussian
        ([0.0, 0.0], [[1.0, 0.1], [0.1, 1.0]]),  # 2D Gaussian
    ),
)
@pytest.mark.parametrize(
    "solver", ["EMSDESolver", "HeunSDESolver", "RK4SDESolver", "EulerODESolver", "HeunODESolver", "RK4ODESolver"]
)
def test_solver_forward(solver, mean, cov):
    sde = VESDE(sigma_min=1e-2, sigma_max=10)
    mean = torch.tensor(mean, dtype=torch.float32)
    cov = torch.tensor(cov, dtype=torch.float32)
    model = MVGEnergyModel(
        sde,
        mean=mean,
        cov=cov,
    )
    slvr = Solver(model, solver=solver)

    x0 = torch.tensor(np.random.multivariate_normal(mean, cov, 100), dtype=torch.float32)
    xT = slvr(x0, steps=50, forward=True, return_dlogp="ODE" in solver, progress_bar=False)
    if "ODE" in solver:  # check delta_logp calculation for ODE solvers
        xT, dlogp = xT
        assert torch.all(torch.isfinite(dlogp))
    assert torch.all(torch.isfinite(xT))


@pytest.mark.parametrize(
    "steps,time_steps",
    (
        (50, None),  # 50 steps normally
        (None, torch.linspace(1, 0, 50)),  # 50 steps set by user
        (None, torch.cat((torch.logspace(0, -2, 49), torch.zeros(1)))),  # 50 steps with log spacing
    ),
)
def test_solver_step(steps, time_steps):
    sde = VESDE(sigma_min=1e-2, sigma_max=10)
    mean = torch.zeros(2, dtype=torch.float32)
    cov = torch.ones(2, dtype=torch.float32)
    model = MVGScoreModel(
        sde,
        mean=mean,
        cov=cov,
    )
    samples = model.sample(shape=(100, mean.shape[-1]), steps=steps, time_steps=time_steps)
    assert torch.all(torch.isfinite(samples))
    assert torch.allclose(samples.mean(dim=0), mean, atol=1), "mean for MVG samples not close"
    assert torch.allclose(samples.std(dim=0), cov.sqrt(), atol=1), "std for MVG samples not close"


@pytest.mark.skip("Currently not passing, need to debug")
@pytest.mark.parametrize(
    "steps,time_steps",
    (
        (50, None),  # 50 steps normally
        (None, torch.linspace(1, 0, 50)),  # 50 steps set by user
        (None, torch.cat((torch.logspace(0, -2, 49), torch.zeros(1))),),  # 50 steps with log spacing
    ),
)
def test_solver_logprob(steps, time_steps):
    sde = VESDE(sigma_min=1e-2, sigma_max=100)
    mean = torch.zeros(2, dtype=torch.float32)
    # diagonal covariance matrix
    cov = torch.eye(2, dtype=torch.float32) * 0.1
    model = MVGScoreModel(
        sde,
        mean=mean,
        cov=cov,
    )

    x = torch.rand(100, 2, dtype=torch.float32)
    logp = model.log_prob(x, steps=steps, time_steps=time_steps)
    true_logp = torch.distributions.MultivariateNormal(mean, cov).log_prob(x)
    
    print(logp - true_logp)
    assert torch.all(torch.isfinite(logp))
    assert torch.allclose(logp, true_logp, atol=1e-3), "logp for MVG samples not close"
