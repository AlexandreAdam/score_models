from typing import Callable, Literal

import torch
from torch import Tensor
from torch.func import vjp
from tqdm import tqdm
from .solver import Solver


class ODESolver(Solver):

    @torch.no_grad()
    def solve(
        self,
        x: Tensor,
        steps: int,
        forward: bool,
        *args: tuple,
        progress_bar: bool = True,
        trace=False,
        kill_on_nan: bool = False,
        denoise_last_step: bool = False,
        get_logP: bool = False,
        **kwargs,
    ):
        """
        Integrate the diffusion ODE forward or backward in time.

        Discretizes the ODE using the given method and integrates the ODE with

        .. math::
            x_{i+1} = x_i + \\frac{dx}{dt}(t_i, x_i) dt

        where the :math:`\\frac{dx}{dt}` is the diffusion drift of

        .. math::
            \\frac{dx}{dt} = f(t, x) - \\frac{1}{2} g(t, x)^2 s(t, x)

        where :math:`f(t, x)` is the sde drift, :math:`g(t, x)` is the sde diffusion,
        and :math:`s(t, x)` is the score.

        Args:
            x: Initial condition.
            steps: integration discretization.
            forward: Direction of integration.
            *args: Additional arguments to pass to the score model.
            progress_bar: Whether to display a progress bar.
            trace: Whether to return the full path or just the last point.
            kill_on_nan: Whether to raise an error if NaNs are encountered.
            denoise_last_step: Whether to project to the boundary at the last step.
            get_logP: Whether to return the log probability of the input x (should be used with forward=True).
        """
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

        if get_logP:
            dp = ht
        else:
            dp = lambda *args, **kwargs: 0.0

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
            step = self.step(t, x, args, dt, self.dx, dp, **kwargs)
            x = x + step[0]
            logp = logp + step[1]

            if trace:
                path.append(x)

        # Project to boundary if denoising
        if denoise_last_step and not forward:
            x = self.tweedie(t, x, *args, **kwargs)
            if trace:
                path[-1] = x

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

    def dx(self, t: Tensor, x: Tensor, args: tuple, dt: Tensor, **kwargs):
        """Discretization of the ODE, this is the update for x"""
        f = self.sde.drift(t, x)
        g = self.sde.diffusion(t, x)
        s = self.score(t, x, *args, **kwargs)
        ret = (f - 0.5 * g**2 * s) * dt
        return ret

    def divergence_hutchinson_trick(
        self,
        t: Tensor,
        x: Tensor,
        args,
        dt: Tensor,
        n_cot_vec: int = 1,
        noise_type: Literal["rademacher", "gaussian"] = "rademacher",
        **kwargs,
    ):
        """
        Compute the divergence of the drift function using the Hutchinson trace estimator.

        Args:
            t: Time of the ODE.
            x: State of the ODE.
            args: Additional arguments to pass to the drift function.
            dt: Time step of the ODE.
            n_cot_vec: Number of cotangent vectors to sample for the Hutchinson trace estimator.
            noise_type: Type of noise to sample, either 'rademacher' or 'gaussian'.
        """
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

        f = lambda x: self.dx(t, x, _args, dt, **kwargs)
        _, vjp_func = vjp(f, samples)
        divergence = (vectors * vjp_func(vectors)[0]).flatten(1).sum(dim=1)
        return divergence


class Euler_ODE(ODESolver):
    """
    Euler method for solving an ODE
    """

    def step(self, t, x, args, dt, dx, dp, **kwargs):
        return dx(t, x, args, dt, **kwargs), dp(t, x, args, dt, **kwargs)


class RK2_ODE(ODESolver):
    """
    Runge Kutta 2nd order ODE solver
    """

    def step(self, t, x, args, dt, dx, dp, **kwargs):
        k1 = dx(t, x, args, dt, **kwargs)
        l1 = dp(t, x, args, dt, **kwargs)
        k2 = dx(t + dt, x + k1, args, dt, **kwargs)
        l2 = dp(t + dt, x + k1, args, dt, **kwargs)
        return (k1 + k2) / 2, (l1 + l2) / 2


class RK4_ODE(ODESolver):
    """
    Runge Kutta 4th order ODE solver
    """

    def step(self, t, x, args, dt, dx, dp, **kwargs):
        k1 = dx(t, x, args, dt, **kwargs)
        l1 = dp(t, x, args, dt, **kwargs)
        k2 = dx(t + dt / 2, x + k1 / 2, args, dt, **kwargs)
        l2 = dp(t + dt / 2, x + k1 / 2, args, dt, **kwargs)
        k3 = dx(t + dt / 2, x + k2 / 2, args, dt, **kwargs)
        l3 = dp(t + dt / 2, x + k2 / 2, args, dt, **kwargs)
        k4 = dx(t + dt, x + k3, args, dt, **kwargs)
        l4 = dp(t + dt, x + k3, args, dt, **kwargs)
        return (k1 + 2 * k2 + 2 * k3 + k4) / 6, (l1 + 2 * l2 + 2 * l3 + l4) / 6
