from typing import Callable, Literal, Optional

from torch import Tensor
from torch.func import vjp, vmap
from tqdm import tqdm
from .solver import Solver
import torch

__all__ = ["ODESolver", "EulerODESolver", "HeunODESolver", "RK4ODESolver"]


class ODESolver(Solver):

    @torch.no_grad()
    def solve(
        self,
        x: Tensor,
        *args,
        steps: int,
        forward: bool,
        progress_bar: bool = True,
        trace: bool = False,
        kill_on_nan: bool = False,
        time_steps: Optional[Tensor] = None,
        dlogp: Optional[Callable] = None,
        return_dlogp: bool = False,
        hook: Optional[Callable] = None,
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
            dlogp: Optional function to compute the divergence of the drift function.
            return_dlogp: Whether to return the log probability change.
            time_steps: Optional time steps to use for integration. Should be a 1D tensor containing the bin edges of the
                time steps. For example, if one wanted 50 steps from 0 to 1, the time steps would be ``torch.linspace(0, 1, 51)``.
            hook: Optional hook function to call after each step. Will be called with the signature ``hook(t, x, sde, score, solver)``.
        """
        B, *D = x.shape
        if return_dlogp:
            if dlogp:
                self.dlogp = dlogp
            else:
                self.dlogp = self.divergence_hutchinson_trick
        else:
            self.dlogp = lambda t, x, *args, dt, **kwargs: torch.zeros(B, device=x.device, dtype=x.dtype)

        T, dT = self.time_steps(steps, B, D, time_steps=time_steps, forward=forward, **kwargs)
        logp = torch.zeros(B, device=x.device, dtype=x.dtype)

        if trace:
            path = [x]
        pbar = tqdm(tuple(zip(T, dT))) if progress_bar else zip(T, dT)
        for t, dt in pbar:
            ######### Update #########
            dx, dlogp = self.step(t, x, *args, dt=dt, dx=self.dx, dp=self.dlogp, **kwargs)
            x = x + dx
            logp = logp + dlogp

            ######### Logging #########
            if progress_bar:
                pbar.set_description(
                    f"t={t[0].item():.1g} | sigma={self.sde.sigma(t)[0].item():.1g} | "
                    f"x={x.mean().item():.1g}\u00B1{x.std().item():.1g}"
                )
            if kill_on_nan and torch.any(torch.isnan(x)):
                raise ValueError("NaN encountered in ODE solver")
            if trace:
                path.append(x)
            if hook is not None:
                hook(t, x, self.sde, self.sbm.score, self)
        if trace:
            if return_dlogp:
                return torch.stack(path), logp
            return torch.stack(path)
        if return_dlogp:
            return x, logp
        return x

    def dx(self, t: Tensor, x: Tensor, *args, dt: Tensor, **kwargs):
        """Discretization of the ODE, this is the update for x"""
        f = self.sde.drift(t, x)
        g = self.sde.diffusion(t, x)
        s = self.sbm.score(t, x, *args, **kwargs)
        return (f - 0.5 * g**2 * s) * dt

    def divergence_hutchinson_trick(
        self,
        t: Tensor,
        x: Tensor,
        *args,
        dt: Tensor,
        cotangent_vectors: int = 1,
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
        # Sample cotangent vectors
        z = torch.randn([cotangent_vectors, B, *D], device=x.device, dtype=x.dtype)
        if noise_type == "rademacher":
            z = z.sign()

        # Compute the trace of the divergence of the drift function
        f = lambda x: self.dx(t, x, *args, dt=dt, **kwargs)
        _, vjp_func = vjp(f, x)
        divergence = (z * vmap(vjp_func)(z)[0]).mean(0).flatten(1).sum(1)
        return divergence


class EulerODESolver(ODESolver):
    """
    Euler method for solving an ODE
    """

    def step(self, t, x, *args, dt, dx, dp, **kwargs):
        return dx(t, x, *args, dt=dt, **kwargs), dp(t, x, *args, dt=dt, **kwargs)


class HeunODESolver(ODESolver):
    """
    Runge Kutta 2nd order ODE solver, also know as Heun method
    """
    def step(self, t, x, *args, dt, dx, dp, **kwargs):
        k1 = dx(t, x, *args, dt=dt, **kwargs)
        l1 = dp(t, x, *args, dt=dt, **kwargs)
        k2 = dx(t + dt.squeeze(), x + k1, *args, dt=dt, **kwargs)
        l2 = dp(t + dt.squeeze(), x + k1, *args, dt=dt, **kwargs)
        return (k1 + k2) / 2, (l1 + l2) / 2


class RK4ODESolver(ODESolver):
    """
    Runge Kutta 4th order ODE solver
    """
    def step(self, t, x, *args, dt, dx, dp, **kwargs):
        k1 = dx(t, x, *args, dt=dt, **kwargs)
        l1 = dp(t, x, *args, dt=dt, **kwargs)
        k2 = dx(t + dt.squeeze() / 2, x + k1 / 2, *args, dt=dt, **kwargs)
        l2 = dp(t + dt.squeeze() / 2, x + k1 / 2, *args, dt=dt, **kwargs)
        k3 = dx(t + dt.squeeze() / 2, x + k2 / 2, *args, dt=dt, **kwargs)
        l3 = dp(t + dt.squeeze() / 2, x + k2 / 2, *args, dt=dt, **kwargs)
        k4 = dx(t + dt.squeeze(), x + k3, *args, dt=dt, **kwargs)
        l4 = dp(t + dt.squeeze(), x + k3, *args, dt=dt, **kwargs)
        return (k1 + 2 * k2 + 2 * k3 + k4) / 6, (l1 + 2 * l2 + 2 * l3 + l4) / 6

