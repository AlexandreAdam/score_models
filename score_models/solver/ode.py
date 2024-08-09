from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import torch
from torch import Tensor
from torch.func import vjp
from tqdm import tqdm
from ..utils import DEVICE

# TODO: maybe merge ODE and Solver into single class?


class ODE(ABC):
    def __init__(self, score, **kwargs):
        self.score = score

    @property
    def sde(self):
        return self.score.sde

    def forward(self, x0, N, **kwargs):
        """Call this to solve the ODE forward in time from x0 at time t_min to xT at time t_max"""
        return self._solve(x0, N, self.forward_dx, True, **kwargs)

    def reverse(self, xT, N, **kwargs):
        """Call this to solve the ODE backward in time from xT at time t_max to x0 at time t_min"""
        return self._solve(xT, N, self.reverse_dx, False, **kwargs)

    def forward_dx(self, t, x, dt):
        """Forward discretization of the ODE, this is the update for x"""
        return self.sde.drift(t, x) * dt

    def reverse_dx(self, t, x, dt, **kwargs):
        """Reverse discretization of the ODE, this is the update for x"""
        return (
            self.sde.drift(t, x) - 0.5 * self.sde.diffusion(t, x) ** 2 * self.score(t, x, **kwargs)
        ) * dt

    @torch.no_grad()
    def _solve(
        self,
        x: Tensor,
        N: int,
        dx: Callable,
        forward: bool,
        progress_bar: bool = False,
        **kwargs,
    ):
        B, *_ = x.shape
        h = 1 if forward else -1
        dt = h * self.stepsize(N, **kwargs)
        trace = kwargs.pop("trace", False)
        if trace:
            path = [x]
        T = self.time_steps(N, B, forward=forward, **kwargs)
        pbar = tqdm(T) if progress_bar else T
        for t in pbar:
            if progress_bar:
                pbar.set_description(
                    f"t = {t[0].item():.1g} | sigma = {self.sde.sigma(t)[0].item():.1g} | "
                    f"x = {x.mean().item():.1g} \u00B1 {x.std().item():.1g}"
                )
            if kwargs.get("kill_on_nan", False) and torch.any(torch.isnan(x)):
                raise ValueError("NaN encountered in SDE solver")
            x = self._step(t, x, dt, dx, **kwargs)
            if trace:
                path.append(x)
        if trace:
            return torch.stack(path)
        return x

    @abstractmethod
    def _step(self, t, x, dt, dx, **kwargs):
        """base ODE step"""
        ...

    @abstractmethod
    def _log_likelihood(self, x, N, forward=True, *args):
        """
        A basic implementation of Euler discretisation method of the ODE associated
        with the marginals of the learned SDE.

        ode_steps: Number of steps to perform in the ODE
        hutchinsons_samples: Number of samples to draw to compute the trace of the Jacobian (divergence)

        Note that this estimator only compute the likelihood for one trajectory.
        For more precise log likelihood estimation, tile x along the batch dimension
        and averge the results. You can also increase the number of ode steps and increase
        the number of cotangent vector for the Hutchinson estimator.

        Using the instantaneous change of variable formula
        (Chen et al. 2018,https://arxiv.org/abs/1806.07366)
        See also Song et al. 2020, https://arxiv.org/abs/2011.13456)
        """
        ...

    def log_likelihood(self, xT, N, *args):
        return self._log_likelihood(xT, N, self.reverse_dx, False, *args)

    def trace_jac_drift(
        self, t, x, n_cotangent_vectors: int = 1, noise_type="rademacher", **kwargs
    ):
        _, *D = x.shape
        # duplicate noisy samples for for the Hutchinson trace estimator
        samples = torch.tile(x, [n_cotangent_vectors, *[1] * len(D)])
        # TODO also duplicate args
        t = torch.tile(t, [n_cotangent_vectors])
        # sample cotangent vectors
        vectors = torch.randn_like(samples)
        if noise_type == "rademacher":
            vectors = vectors.sign()
        # Compute the trace of the Jacobian of the drift functions (Hessian if drift is just the score)
        f = lambda x: self.reverse_dx(t, x, 1, **kwargs)
        _, vjp_func = vjp(f, samples)
        divergence = (vectors * vjp_func(vectors)[0]).flatten(1).sum(dim=1)
        return divergence

    def time_steps(self, N, B=1, forward=True, device=DEVICE, **kwargs):
        t_min = kwargs.get("t_min", self.sde.t_min)
        t_max = kwargs.get("t_max", self.sde.t_max)
        if forward:
            return torch.linspace(t_min, t_max, N + 1, device=device)[:-1].repeat(B, 1).T
        else:
            return torch.linspace(t_max, t_min, N + 1, device=device)[:-1].repeat(B, 1).T

    def stepsize(self, N, device=DEVICE, **kwargs):
        t_min = kwargs.get("t_min", self.sde.t_min)
        t_max = kwargs.get("t_max", self.sde.t_max)
        return torch.as_tensor((t_max - t_min) / N, device=device)


class Euler_ODE(ODE):
    def _step(self, t, x, dt, dx, **kwargs):
        return x + dx(t, x, dt, **kwargs)

    def _log_likelihood(self, x, N, dx, forward=True, **kwargs):
        # TODO: this assumes user is going to call forward=False
        B, *D = x.shape
        h = 1 if forward else -1
        dt = h * self.stepsize(N, **kwargs)
        log_likelihood = 0.0

        for t in self.time_steps(N, B, forward):
            x = x + dx(t, x, dt, **kwargs)
            log_likelihood += self.trace_jac_drift(t, x, **kwargs) * dt
        log_likelihood += self.sde.prior(D).log_prob(x)
        return log_likelihood


class RK2_ODE(ODE):
    """
    Runge Kutta 2nd order ODE solver
    """

    def _step(self, t, x, dt, dx, **kwargs):
        k1 = dx(t, x, dt, **kwargs)
        k2 = dx(t + dt, x + k1, dt, **kwargs)
        return x + (k1 + k2) / 2

    def _log_likelihood(self, x, N, dx, forward=True, **kwargs):
        B, *D = x.shape
        h = 1 if forward else -1
        dt = h * self.stepsize(N, **kwargs)
        log_likelihood = 0.0

        for t in self.time_steps(N, B, forward):
            k1 = dx(t, x, dt, **kwargs)
            k2 = dx(t + dt, x + k1, dt, **kwargs)
            x = x + (k1 + k2) * dt / 2
            l1 = self.trace_jac_drift(t, x, **kwargs)
            l2 = self.trace_jac_drift(t + dt, x + k1, **kwargs)
            log_likelihood += (l1 + l2) * dt / 2
        log_likelihood += self.sde.prior(D).log_prob(x)
        return log_likelihood


class RK4_ODE(ODE):
    """
    Runge Kutta 4th order ODE solver
    """

    def _step(self, t, x, dt, dx, **kwargs):
        k1 = dx(t, x, dt, **kwargs)
        k2 = dx(t + dt / 2, x + k1 / 2, dt, **kwargs)
        k3 = dx(t + dt / 2, x + k2 / 2, dt, **kwargs)
        k4 = dx(t + dt, x + k3, dt, **kwargs)
        return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def _log_likelihood(self, x, N, dx, forward=True, **kwargs):
        B, *D = x.shape
        h = 1 if forward else -1
        dt = h * self.stepsize(N, **kwargs)
        log_likelihood = 0.0

        for t in self.time_steps(N, B, forward):
            k1 = dx(t, x, dt, **kwargs)
            k2 = dx(t + dt / 2, x + k1 / 2, dt, **kwargs)
            k3 = dx(t + dt / 2, x + k2 / 2, dt, **kwargs)
            k4 = dx(t + dt, x + k3, dt, **kwargs)
            x = x + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
            l1 = self.trace_jac_drift(t, x, **kwargs)
            l2 = self.trace_jac_drift(t + dt / 2, x + k1 / 2, **kwargs)
            l3 = self.trace_jac_drift(t + dt / 2, x + k2 / 2, **kwargs)
            l4 = self.trace_jac_drift(t + dt, x + k3, **kwargs)
            log_likelihood += (l1 + 2 * l2 + 2 * l3 + l4) * dt / 6
        log_likelihood += self.sde.prior(D).log_prob(x)
        return log_likelihood
