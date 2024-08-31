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
        return self._solve(x0, N, self.dx, True, **kwargs)

    def reverse(self, xT, N, **kwargs):
        """Call this to solve the ODE backward in time from xT at time t_max to x0 at time t_min"""
        return self._solve(xT, N, self.dx, False, **kwargs)

    def dx(self, t, x, dt, **kwargs):
        """Discretization of the ODE, this is the update for x"""
        f = self.sde.drift(t, x)
        g = self.sde.diffusion(t, x)
        s = self.score(t, x, **kwargs)
        return (f - 0.5 * g**2 * s) * dt

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
        dlogp = torch.zeros(B, device=x.device, dtype=x.dtype)
        ht = kwargs.get("hessian_trace", lambda *a, **k: 0.0)
        trace = kwargs.pop("trace", False)
        if trace:
            path = [x]
        T = self.time_steps(N, B, forward=forward, **kwargs)
        pbar = tqdm(T) if progress_bar else T
        for t in pbar:
            if progress_bar:
                pbar.set_description(
                    f"t={t[0].item():.1g} | sigma={self.sde.sigma(t)[0].item():.1g} | "
                    f"x={x.mean().item():.1g}\u00B1{x.std().item():.1g}"
                )
            if kwargs.get("kill_on_nan", False) and torch.any(torch.isnan(x)):
                raise ValueError("NaN encountered in SDE solver")
            x = x + self._step(t, x, dt, dx, **kwargs)
            dlogp = dlogp + self._step(t, x, dt, ht, **kwargs)
            if trace:
                path.append(x)
        if trace:
            return torch.stack(path)
        if "hessian_trace" in kwargs:
            return x, dlogp
        return x

    @abstractmethod
    def _step(self, t, x, dt, dx, **kwargs):
        """base ODE step"""
        ...

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
        return dx(t, x, dt, **kwargs)


class RK2_ODE(ODE):
    """
    Runge Kutta 2nd order ODE solver
    """

    def _step(self, t, x, dt, dx, **kwargs):
        k1 = dx(t, x, dt, **kwargs)
        k2 = dx(t + dt, x + k1, dt, **kwargs)
        return (k1 + k2) / 2


class RK4_ODE(ODE):
    """
    Runge Kutta 4th order ODE solver
    """

    def _step(self, t, x, dt, dx, **kwargs):
        k1 = dx(t, x, dt, **kwargs)
        k2 = dx(t + dt / 2, x + k1 / 2, dt, **kwargs)
        k3 = dx(t + dt / 2, x + k2 / 2, dt, **kwargs)
        k4 = dx(t + dt, x + k3, dt, **kwargs)
        return (k1 + 2 * k2 + 2 * k3 + k4) / 6
