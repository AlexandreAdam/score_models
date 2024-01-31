from abc import ABC, abstractmethod

import torch
import numpy as np
from tqdm import tqdm


class Solver(ABC):
    """
    Base solver for a stochastic differential equation (SDE).

    The SDE defines the differential element ``dx`` for the stochastic process
    ``x``. Both the forward and reverse processes are defined using the
    coefficients and score provided. This class represents essentially any SDE
    of the form: :math:`dx = f(t, x) dt + g(t, x) dw`.
    """

    def __init__(self, score, corrector=False, **kwargs):
        self.score = score
        self.corrector = corrector
        self.corrector_epsilon = kwargs.get("corrector_epsilon", lambda t: 0.1)
        self.corrector_steps = kwargs.get("corrector_steps", 1)

    @property
    def sde(self):
        return self.score.sde

    @property
    def corrector(self):
        return self._corrector

    @corrector.setter
    def corrector(self, corrector):
        if corrector:
            self._corrector = self.corrector_step
        else:
            self._corrector = self.null_corrector

    def null_corrector(self, t, x, **kwargs):
        return x

    def corrector_step(self, t, x, **kwargs):
        """Basic Langevin corrector step for the SDE."""
        z = torch.randn_like(x)
        return (
            x
            + self.corrector_epsilon(t) * self.score(t, x)
            + z * torch.sqrt(2 * self.corrector_epsilon(t))
        )

    def forward(self, x0, N, **kwargs):
        """Call this to solve the SDE forward in time from x0 at time t_min to xT at time t_max"""
        return self._solve(x0, N, self.forward_dx, True, **kwargs)

    def reverse(self, xT, N, **kwargs):
        """Call this to solve the SDE backward in time from xT at time t_max to x0 at time t_min"""
        return self._solve(xT, N, self.reverse_dx, False, **kwargs)

    def _solve(self, x, N, dx, forward, **kwargs):
        """base SDE solver"""
        B, *D = x.shape
        h = 1 if forward else -1
        dt = h * self.stepsize(N, x.device)
        trace = kwargs.get("trace", False)
        if trace:
            path = [x]
        sk = kwargs.get("sk", -1)
        T = self.time_steps(N, B, forward=forward)
        pbar = kwargs.get("progress_bar", False)
        if pbar:
            T = tqdm(T)
        for t in T:
            if pbar:
                T.set_description(
                    f"t = {t[0].item():.1e} | sigma = {self.sde.sigma(t)[0].item():.1e} | "
                    f"x = {x.mean().item():.1e} +- {x.std().item():.1e}"
                )
            x = self._step(t, x, dt, dx, sk=sk, **kwargs)
            for _ in range(self.corrector_steps):
                x = self.corrector(t, x, **kwargs)
            if trace:
                path.append(x)
        if trace:
            return torch.stack(path)
        return x

    @abstractmethod
    def _step(self, x, t, dt, dx, sk=0.0, **kwargs):
        """base SDE solver"""
        ...

    def forward_f(self, t, x, **kwargs):
        """forward SDE coefficient a"""
        return self.sde.drift(t, x)

    def forward_dx(self, t, x, dt, dw=None, **kwargs):
        """forward SDE differential element dx"""
        if dw is None:
            dw = torch.randn_like(x) * torch.sqrt(dt)
        return self.forward_f(t, x, **kwargs) * dt + self.sde.diffusion(t, x) * dw

    def reverse_f(self, t, x, **kwargs):
        """reverse SDE coefficient a"""
        return self.sde.drift(t, x) - self.sde.diffusion(t, x) ** 2 * self.score(t, x)

    def reverse_dx(self, t, x, dt, dw=None, **kwargs):
        """reverse SDE differential element dx"""
        if dw is None:
            dw = torch.randn_like(x) * torch.sqrt(torch.abs(dt))
        return self.reverse_f(t, x, **kwargs) * dt + self.sde.diffusion(t, x) * dw

    def time_steps(self, N, B=1, forward=True):
        if forward:
            return torch.linspace(self.sde.t_min, self.sde.t_max, N)[1:].repeat(B, 1).T
        else:
            return torch.linspace(self.sde.t_max, self.sde.t_min, N)[1:].repeat(B, 1).T

    def stepsize(self, N, device=None):
        return torch.tensor((self.sde.t_max - self.sde.t_min) / (N - 1), device=device)


class EulerMaruyamaSDE(Solver):
    def _step(self, t, x, dt, dx, **kwargs):
        """base SDE solver"""
        return x + dx(t, x, dt, **kwargs)


class RungeKuttaSDE_2(Solver):
    def _step(self, t, x, dt, dx, sk, **kwargs):
        """Base SDE solver"""
        dw = torch.randn_like(x) * torch.sqrt(dt.abs())
        skdt = sk * np.random.choice([-1, 1]) * torch.sqrt(dt.abs())
        k1 = dx(t, x, dt, dw - skdt, **kwargs)
        k2 = dx(t + dt, x + k1, dt, dw + skdt, **kwargs)
        return x + (k1 + k2) / 2


class RungeKuttaSDE_4(Solver):
    def _step(self, t, x, dt, dx, sk, **kwargs):
        """Base SDE solver"""
        dw = torch.randn_like(x) * torch.sqrt(dt.abs())
        skdt = sk * np.random.choice([-1, 1]) * torch.sqrt(dt.abs())
        k1 = dx(t, x, dt, dw - skdt, **kwargs)
        k2 = dx(t + dt / 2, x + k1 / 2, dt, dw + skdt, **kwargs)
        k3 = dx(t + dt / 2, x + k2 / 2, dt, dw - skdt, **kwargs)
        k4 = dx(t + dt, x + k3, dt, dw + skdt, **kwargs)
        return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
