from abc import ABC, abstractmethod

import torch
import numpy as np


class Solver(ABC):
    """
    Base solver for a stochastic differential equation (SDE).

    The SDE defines the differential element ``dx`` for the stochastic process
    ``x``. Both the forward and reverse processes are defined using the
    coefficients and score provided. This class represents essentially any SDE
    of the form: :math:`dx = f(x, t) dt + g(x, t) dw`.
    """

    def __init__(self, score):
        self.score = score

    @property
    def sde(self):
        return self.score.sde

    @abstractmethod
    def _solve(self, x, N, dx, **kwargs):
        """base SDE solver"""
        ...

    def forward(self, x0, N, **kwargs):
        """forward SDE solver"""

        return self._solve(x0, N, self.forward_dx, **kwargs)

    def reverse(self, xT, N, **kwargs):
        """reverse SDE solver"""

        return self._solve(xT, N, self.reverse_dx, **kwargs)

    def forward_f(self, t, x, **kwargs):
        """forward SDE coefficient a"""
        return self.sde.drift(t, x, **kwargs)

    def forward_dx(self, t, x, dt, dw, **kwargs):
        """forward SDE differential element dx"""
        return self.forward_f(x, t, **kwargs) * dt + self.sde.diffusion(t, **kwargs) * dw

    def reverse_f(self, x, t, **kwargs):
        """reverse SDE coefficient a"""
        return self.sde.drift(t, x, **kwargs) - self.sde.diffusion(t, **kwargs) ** 2 * self.score(
            x, t, **kwargs
        )

    def reverse_dx(self, x, t, dt, dw, **kwargs):
        """reverse SDE differential element dx"""
        return self.reverse_f(t, x, **kwargs) * dt + self.sde.diffusion(t, **kwargs) * dw


class EulerMaruyamaSDE(Solver):
    def _solve(self, x, N, dx, **kwargs):
        """base SDE solver"""
        dt = (self.sde.t_max - self.sde.t_min) / (N - 1)
        trace = kwargs.get("trace", False)
        if trace:
            path = [x]
        for t in np.linspace(self.sde.t_min, self.sde.t_max, N)[1:]:
            dw = torch.randn_like(x) * torch.sqrt(dt)
            x = x + dx(t, x, dt, dw, **kwargs)
            if trace:
                path.append(x)
        if trace:
            return torch.stack(path)
        return x


class RungeKuttaSDE_2(Solver):
    def _solve(self, x, N, dx, **kwargs):
        """Base SDE solver"""
        dt = (self.sde.t_max - self.sde.t_min) / (N - 1)
        trace = kwargs.get("trace", False)
        if trace:
            path = [x]
        for t in np.linspace(self.sde.t_min, self.sde.t_max, N)[1:]:
            dw = torch.randn_like(x) * torch.sqrt(dt)
            k1 = dx(t, x, dt, dw, **kwargs)
            k2 = dx(t + dt, x + k1, dt, dw, **kwargs)
            x = x + (k1 + k2) / 2
            if trace:
                path.append(x)
        if trace:
            return torch.stack(path)
        return x


class RungeKuttaSDE_4(Solver):
    def _solve(self, x, N, dx, **kwargs):
        """Base SDE solver"""
        dt = (self.sde.t_max - self.sde.t_min) / (N - 1)
        trace = kwargs.get("trace", False)
        if trace:
            path = [x]
        for t in np.linspace(self.sde.t_min, self.sde.t_max, N)[1:]:
            dw = torch.randn_like(x) * torch.sqrt(dt)
            k1 = dx(t, x, dt, dw, **kwargs)
            k2 = dx(t + dt / 2, x + k1 / 2, dt, dw, **kwargs)
            k3 = dx(t + dt / 2, x + k2 / 2, dt, dw, **kwargs)
            k4 = dx(t + dt, x + k3, dt, dw, **kwargs)
            x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            if trace:
                path.append(x)
        if trace:
            return torch.stack(path)
        return x
