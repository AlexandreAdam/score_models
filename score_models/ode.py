from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.func import vjp


class ODE(ABC):
    def __init__(self, sde, score):
        self.sde = sde
        self.score = score

    def dx_dt(self, t, x):
        return self.sde.drift(x, t) - 0.5 * self.sde.diffusion(t) ** 2 * self.score(t, x)

    @abstractmethod
    def solve(self, x, N, forward=True, *args):
        ...

    @abstractmethod
    def log_likelihood(self, x, N, forward=True, *args):
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

    def trace_jac_drift(self, t, x, n_cotangent_vectors: int = 1, noise_type="rademacher"):
        B, *D = x.shape
        # duplicate noisy samples for for the Hutchinson trace estimator
        samples = torch.tile(x, [n_cotangent_vectors, *[1] * len(D)])
        # TODO also duplicate args
        t = torch.tile(t, [n_cotangent_vectors])
        # sample cotangent vectors
        vectors = torch.randn_like(samples)
        if noise_type == "rademacher":
            vectors = vectors.sign()
        # Compute the trace of the Jacobian of the drift functions (Hessian if drift is just the score)
        f = lambda x: self.dx_dt(t, x)
        _, vjp_func = vjp(f, samples)
        divergence = (vectors * vjp_func(vectors)[0]).flatten(1).sum(dim=1)
        return divergence

    def time_steps(self, N, forward=True):
        if forward:
            return np.linspace(self.sde.t_min, self.sde.t_max, N)[1:]
        else:
            return np.linspace(self.sde.t_max, self.sde.t_min, N)[1:]

    def stepsize(self, N):
        return (self.sde.t_max - self.sde.t_min) / (N - 1)


class EulerODE(ODE):
    def solve(self, x, N, forward=True, *args):
        dt = self.stepsize(N)

        for t in self.time_steps(N, forward):
            x = x + self.dx_dt(t, x) * dt
        return x

    def log_likelihood(self, x, N, forward=True, *args):
        B, *D = x.shape
        dt = self.stepsize(N)
        log_likelihood = 0.0
        for t in self.time_steps(N, forward):
            x = x + self.dx_dt(t, x) * dt
            log_likelihood += self.sde.trace_jac_drift(t, x) * dt
        log_likelihood += self.sde.prior(D).log_prob(x)
        return log_likelihood


class RungeKuttaODE_2(ODE):
    def solve(self, x, N, forward=True, *args):
        dt = self.stepsize(N)

        for t in self.time_steps(N, forward):
            k1 = self.dx_dt(t, x)
            k2 = self.dx_dt(t + dt, x + k1)
            x = x + (k1 + k2) * dt / 2
        return x

    def log_likelihood(self, x, N, forward=True, *args):
        B, *D = x.shape
        dt = self.stepsize(N)
        log_likelihood = 0.0
        for t in self.time_steps(N, forward):
            k1 = self.dx_dt(t, x)
            k2 = self.dx_dt(t + dt, x + k1)
            x = x + (k1 + k2) * dt / 2
            l1 = self.sde.trace_jac_drift(t, x)
            l2 = self.sde.trace_jac_drift(t + dt, x + k1)
            log_likelihood += (l1 + l2) * dt / 2
        log_likelihood += self.sde.prior(D).log_prob(x)
        return log_likelihood


class RungeKuttaODE_4(ODE):
    def solve(self, x, N, forward=True, *args):
        dt = self.stepsize(N)

        for t in self.time_steps(N, forward):
            k1 = self.dx_dt(t, x)
            k2 = self.dx_dt(t + dt / 2, x + k1 / 2)
            k3 = self.dx_dt(t + dt / 2, x + k2 / 2)
            k4 = self.dx_dt(t + dt, x + k3)
            x = x + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
        return x

    def log_likelihood(self, x, N, forward=True, *args):
        B, *D = x.shape
        dt = self.stepsize(N)
        log_likelihood = 0.0
        for t in self.time_steps(N, forward):
            k1 = self.dx_dt(t, x)
            k2 = self.dx_dt(t + dt / 2, x + k1 / 2)
            k3 = self.dx_dt(t + dt / 2, x + k2 / 2)
            k4 = self.dx_dt(t + dt, x + k3)
            x = x + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
            l1 = self.trace_jac_drift(t, x)
            l2 = self.trace_jac_drift(t + dt / 2, x + k1 / 2)
            l3 = self.trace_jac_drift(t + dt / 2, x + k2 / 2)
            l4 = self.trace_jac_drift(t + dt, x + k3)
            log_likelihood += (l1 + 2 * l2 + 2 * l3 + l4) * dt / 6
        log_likelihood += self.sde.prior(D).log_prob(x)
        return log_likelihood
