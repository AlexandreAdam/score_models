from abc import ABC, abstractmethod

import torch
import numpy as np


class Solver(ABC):
    """
    Base solver for a stochastic differential equation (SDE).

    The SDE defines the differential element ``dx`` for the stochastic process
    ``x``. Both the forward and reverse processes are defined using the
    coefficients and score provided. This class represents essentially any SDE
    of the form: :math:`dx = f(t, x) dt + g(t, x) dw`.
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

        return self._solve(x0, N, self.forward_dx, True, **kwargs)

    def reverse(self, xT, N, **kwargs):
        """reverse SDE solver"""

        return self._solve(xT, N, self.reverse_dx, False, **kwargs)

    def forward_f(self, t, x, **kwargs):
        """forward SDE coefficient a"""
        return self.sde.drift(t, x, **kwargs)

    def forward_dx(self, t, x, dt, dw=None, **kwargs):
        """forward SDE differential element dx"""
        if dw is None:
            dw = torch.randn_like(x) * torch.sqrt(dt)
        return self.forward_f(t, x, **kwargs) * dt + self.sde.diffusion(t, x, **kwargs) * dw

    def reverse_f(self, t, x, **kwargs):
        """reverse SDE coefficient a"""
        return self.sde.drift(t, x, **kwargs) - self.sde.diffusion(
            t, x, **kwargs
        ) ** 2 * self.score(t, x, **kwargs)

    def reverse_dx(self, t, x, dt, dw=None, **kwargs):
        """reverse SDE differential element dx"""
        if dw is None:
            dw = torch.randn_like(x) * torch.sqrt(dt)
        return self.reverse_f(t, x, **kwargs) * dt + self.sde.diffusion(t, x, **kwargs) * dw

    def time_steps(self, N, B=1, forward=True):
        if forward:
            return torch.linspace(self.sde.t_min, self.sde.t_max, N)[1:].repeat(B, 1).T
        else:
            return torch.linspace(self.sde.t_max, self.sde.t_min, N)[1:].repeat(B, 1).T

    def stepsize(self, N, device=None):
        return torch.tensor((self.sde.t_max - self.sde.t_min) / (N - 1), device=device)


class EulerMaruyamaSDE(Solver):
    def _solve(self, x, N, dx, forward, **kwargs):
        """base SDE solver"""
        B, *D = x.shape
        dt = self.stepsize(N, x.device)
        trace = kwargs.get("trace", False)
        if trace:
            path = [x]
        for t in self.time_steps(N, B, forward=forward):
            x = x + dx(t, x, dt, **kwargs)
            if trace:
                path.append(x)
        if trace:
            return torch.stack(path)
        return x


class RungeKuttaSDE_2(Solver):
    def _solve(self, x, N, dx, forward, **kwargs):
        """Base SDE solver"""
        B, *D = x.shape
        h = 1 if forward else -1
        dt = self.stepsize(N, x.device)
        trace = kwargs.get("trace", False)
        if trace:
            path = [x]
        sk = kwargs.get("sk", -1)
        for t in self.time_steps(N, B, forward=forward):
            dw = torch.randn_like(x) * torch.sqrt(dt)
            sk = -sk
            k1 = dx(t, x, dt, dw - sk * torch.sqrt(dt), **kwargs)
            k2 = dx(t + h * dt, x + k1, dt, dw + sk * torch.sqrt(dt), **kwargs)
            x = x + (k1 + k2) / 2
            if trace:
                path.append(x)
        if trace:
            return torch.stack(path)
        return x


class RungeKuttaSDE_4(Solver):
    def _solve(self, x, N, dx, forward, **kwargs):
        """Base SDE solver"""
        B, *D = x.shape
        h = 1 if forward else -1
        dt = self.stepsize(N, x.device)
        trace = kwargs.get("trace", False)
        if trace:
            path = [x]
        sk = kwargs.get("sk", -1)
        for t in self.time_steps(N, B, forward=forward):
            dw = torch.randn_like(x) * torch.sqrt(dt)
            sk = -sk
            k1 = dx(t, x, dt, dw - sk * torch.sqrt(dt), **kwargs)
            k2 = dx(t + h * dt / 2, x + k1 / 2, dt, dw + sk * torch.sqrt(dt), **kwargs)
            k3 = dx(t + h * dt / 2, x + k2 / 2, dt, dw - sk * torch.sqrt(dt), **kwargs)
            k4 = dx(t + h * dt, x + k3, dt, dw + sk * torch.sqrt(dt), **kwargs)
            x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            if trace:
                path.append(x)
        if trace:
            return torch.stack(path)
        return x
