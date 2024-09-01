from typing import Callable, Literal

import torch
from torch import Tensor
from torch.func import vjp
from tqdm import tqdm
from .solver import Solver

# TODO: maybe merge ODE and Solver into single class?


class ODESolver(Solver):

    @torch.no_grad()
    def solve(
        self,
        x: Tensor,
        steps: int,
        forward: bool,
        progress_bar: bool = True,
        trace=False,
        kill_on_nan: bool = False,
        get_logP: bool = False,
        **kwargs,
    ):
        B, *D = x.shape

        # Step
        dt = self.step_size(steps, forward=forward, **kwargs)
        T = self.time_steps(steps, B, forward=forward, **kwargs)

        # log P(xt) if requested
        logp = torch.zeros(B, device=x.device, dtype=x.dtype)
        if self.score.hessian_trace_model is None:
            ht = self.divergence_hutchinson_trick
        else:
            ht = lambda *args, **kwargs: self.score.hessian_trace_model(*args, **kwargs) * dt

        # Trace ODE path if requested
        if trace:
            path = [x]

        # Progress bar
        pbar = tqdm(T) if progress_bar else T
        for t in pbar:
            if progress_bar:
                pbar.set_description(
                    f"t={t[0].item():.1g} | sigma={self.sde.sigma(t)[0].item():.1g} | "
                    f"x={x.mean().item():.1g}\u00B1{x.std().item():.1g}"
                )

            # Check for NaNs
            if kill_on_nan and torch.any(torch.isnan(x)):
                raise ValueError("NaN encountered in ODE solver")

            # Update x
            x = x + self._step(t, x, dt, self.dx, **kwargs)

            # Update logP if requested
            if get_logP:
                logp = logp + self._step(t, x, dt, ht, **kwargs)

            # Trace path if requested
            if trace:
                path.append(x)

        # add boundary condition PDF probability
        if get_logP and forward:
            logp = self.sde.prior(D).log_prob(x) + logp

        # Return path or final x
        if trace:
            if get_logP:
                return torch.stack(path), logp
            return torch.stack(path)
        if get_logP:
            return x, logp
        return x

    def dx(self, t: Tensor, x: Tensor, dt: Tensor, **kwargs):
        """Discretization of the ODE, this is the update for x"""
        f = self.sde.drift(t, x)
        g = self.sde.diffusion(t, x)
        s = self.score(t, x, **kwargs)
        return (f - 0.5 * g**2 * s) * dt

    def divergence_hutchinson_trick(
        self,
        t: Tensor,
        x: Tensor,
        dt: Tensor,
        *args,
        n_cot_vec: int = 1,
        noise_type: Literal["rademacher", "gaussian"] = "rademacher",
        **kwargs,
    ):
        B, *D = x.shape
        # duplicate samples for for the Hutchinson trace estimator
        samples = torch.tile(x, [n_cot_vec, *[1] * len(D)])
        t = torch.tile(t, [n_cot_vec])
        _args = []
        for arg in args:
            _, *DA = arg.shape
            arg = torch.tile(arg, [n_cot_vec, *[1] * len(DA)])
            _args.append(arg)

        # sample cotangent vectors
        vectors = torch.randn_like(samples)
        if noise_type == "rademacher":
            vectors = vectors.sign()

        f = lambda x: self.dx(t, x, dt, *_args, **kwargs)
        _, vjp_func = vjp(f, samples)
        divergence = (vectors * vjp_func(vectors)[0]).flatten(1).sum(dim=1)
        return divergence


class Euler_ODE(ODESolver):
    def _step(self, t, x, dt, dx, **kwargs):
        return dx(t, x, dt, **kwargs)


class RK2_ODE(ODESolver):
    """
    Runge Kutta 2nd order ODE solver
    """

    def _step(self, t, x, dt, dx, **kwargs):
        k1 = dx(t, x, dt, **kwargs)
        k2 = dx(t + dt, x + k1, dt, **kwargs)
        return (k1 + k2) / 2


class RK4_ODE(ODESolver):
    """
    Runge Kutta 4th order ODE solver
    """

    def _step(self, t, x, dt, dx, **kwargs):
        k1 = dx(t, x, dt, **kwargs)
        k2 = dx(t + dt / 2, x + k1 / 2, dt, **kwargs)
        k3 = dx(t + dt / 2, x + k2 / 2, dt, **kwargs)
        k4 = dx(t + dt, x + k3, dt, **kwargs)
        return (k1 + 2 * k2 + 2 * k3 + k4) / 6
