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

    def __init__(self, sde, score):
        self.sde = sde
        self.score = score

    @abstractmethod
    def _solve(self, x, N, dx, *args):
        """base SDE solver"""
        ...

    def forward(self, x0, N, *args):
        """forward SDE solver"""

        return self._solve(x0, N, self.forward_dx, *args)

    def reverse(self, xT, N, *args):
        """reverse SDE solver"""

        return self._solve(xT, N, self.reverse_dx, *args)

    def forward_f(self, t, x, **kargs):
        """forward SDE coefficient a"""
        return self.sde.drift(t, x, **kargs)

    def forward_dx(self, t, x, dt, dw, **kargs):
        """forward SDE differential element dx"""
        return self.forward_f(x, t, **kargs) * dt + self.sde.diffusion(t, **kargs) * dw

    def reverse_f(self, x, t, **kargs):
        """reverse SDE coefficient a"""
        return self.sde.drift(t, x, **kargs) - self.sde.diffusion(t, **kargs) ** 2 * self.score(
            x, t, **kargs
        )

    def reverse_dx(self, x, t, dt, dw, **kargs):
        """reverse SDE differential element dx"""
        return self.reverse_f(t, x, **kargs) * dt + self.sde.diffusion(t, **kargs) * dw


class EulerMaruyamaSDE(Solver):
    def _solve(self, x, N, dx, *args):
        """base SDE solver"""
        dt = (self.sde.t_max - self.sde.t_min) / (N - 1)
        for t in np.linspace(self.sde.t_min, self.sde.t_max, N)[1:]:
            dw = torch.randn_like(x) * torch.sqrt(dt)
            x = x + dx(t, x, dt, dw, *args)

        return x


class RungeKuttaSDE_2(Solver):
    def _solve(self, x, N, dx, *args):
        """Base SDE solver"""
        dt = (self.sde.t_max - self.sde.t_min) / (N - 1)
        for t in np.linspace(self.sde.t_min, self.sde.t_max, N)[1:]:
            dw = torch.randn_like(x) * torch.sqrt(dt)
            k1 = dx(t, x, dt, dw, *args)
            k2 = dx(t + dt, x + k1, dt, dw, *args)
            x = x + (k1 + k2) / 2

        return x


class RungeKuttaSDE_4(Solver):
    def _solve(self, x, N, dx, *args):
        """Base SDE solver"""
        dt = (self.sde.t_max - self.sde.t_min) / (N - 1)
        for t in np.linspace(self.sde.t_min, self.sde.t_max, N)[1:]:
            dw = torch.randn_like(x) * torch.sqrt(dt)
            k1 = dx(t, x, dt, dw, *args)
            k2 = dx(t + dt / 2, x + k1 / 2, dt, dw, *args)
            k3 = dx(t + dt / 2, x + k2 / 2, dt, dw, *args)
            k4 = dx(t + dt, x + k3, dt, dw, *args)
            x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return x
