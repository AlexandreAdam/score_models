from typing import Callable, Optional
from contextlib import nullcontext

from torch import Tensor
from tqdm import tqdm
import torch
import numpy as np

from .solver import Solver

__all__ = ["SDESolverAdaptive", "HeunSDESolverAdaptive"]


class SDESolverAdaptive(Solver):

    @torch.no_grad()
    def solve(
        self,
        x: Tensor,
        *args: tuple,
        forward: bool,
        dt_init: Optional[float] = None,
        steps: Optional[int] = None,
        accuracy: float = 1e-1,
        progress_bar: bool = True,
        trace: bool = False,
        kill_on_nan: bool = False,
        corrector_steps: int = 0,
        corrector_snr: float = 0.1,
        hook: Optional[Callable] = None,
        **kwargs,
    ):
        """
        Integrate the diffusion SDE forward or backward in time.

        Discretizes the SDE using the given method and integrates with

        .. math::
            x_{i+1} = x_i + \\frac{dx}{dt}(t_i, x_i) * dt + g(t_i, x_i) * dw

        where the :math:`\\frac{dx}{dt}` is the diffusion drift of

        .. math::
            \\frac{dx}{dt} = f(t, x) - \\frac{1}{2} g(t, x)^2 s(t, x)

        where :math:`f(t, x)` is the sde drift, :math:`g(t, x)` is the sde diffusion,
        and :math:`s(t, x)` is the score.

        Args:
            x: Initial condition.
            dt_init: Initial time step.
            forward: Direction of integration.
            *args: Additional arguments to pass to the score model.
            progress_bar: Whether to display a progress bar.
            trace: Whether to return the full path or just the last point.
            kill_on_nan: Whether to raise an error if NaNs are encountered.
            time_steps: Optional time steps to use for integration. Should be a 1D tensor containing the bin edges of the
                time steps. For example, if one wanted 50 steps from 0 to 1, the time steps would be ``torch.linspace(0, 1, 51)``.
            corrector_steps: Number of corrector steps to add after each SDE step (0 for no corrector steps).
            corrector_snr: Signal-to-noise ratio for the corrector steps.
            hook: Optional hook function to call after each step. Will be called with the signature ``hook(t, x, sde, score, solver)``.
        """
        B, *D = x.shape

        # Step
        if dt_init is None and steps is not None:
            dt_init = 1.0 / steps
        t_min = kwargs.get("t_min", self.sde.t_min)
        t_max = kwargs.get("t_max", self.sde.t_max)
        if forward:
            dt = torch.tensor(dt_init, device=x.device, dtype=x.dtype)
            T = [t_min]
        else:
            dt = -torch.tensor(dt_init, device=x.device, dtype=x.dtype)
            T = [t_max]
        dt = dt.reshape(1, *[1] * len(D)).repeat(B, *[1] * len(D))

        if trace:
            path = [x]

        if progress_bar:
            ptotal = int((t_max - t_min) / dt_init)
            _pbar = tqdm(total=ptotal)
            pcurrent = 0
        else:
            _pbar = nullcontext()
        with _pbar as pbar:
            while (forward and T[-1] < t_max * 0.999999) or (
                not forward and T[-1] > t_min * 1.000001
            ):
                t = torch.tensor(T[-1], device=x.device, dtype=x.dtype).repeat(B)
                if forward:  # don't pass integration endpoint
                    max_dt = torch.tensor(
                        min(t_max - T[-1], kwargs.get("max_dt", 1.0)),
                        device=x.device,
                        dtype=x.dtype,
                    )
                else:
                    max_dt = -torch.tensor(
                        min(T[-1] - t_min, kwargs.get("max_dt", 1.0)),
                        device=x.device,
                        dtype=x.dtype,
                    )
                dx, dt = self.step(t, x, args, dt, forward, accuracy, max_dt, **kwargs)
                x = x + dx
                T.append(T[-1] + dt.flatten()[0].item())
                for _ in range(corrector_steps):
                    x = self.corrector_step(t, x, args, corrector_snr, **kwargs)

                # Logs
                if progress_bar:
                    if forward:
                        frac_progress = int((T[-1] - t_min) / (t_max - t_min) * ptotal)
                    else:
                        frac_progress = int((t_max - T[-1]) / (t_max - t_min) * ptotal)
                    if frac_progress > pcurrent:
                        pbar.update(frac_progress - pcurrent)
                        pcurrent = frac_progress
                    pbar.set_description(
                        f"t={T[-1]:.1g} | sigma={self.sde.sigma(t[0]).item():.1g} | "
                        f"x={x.mean().item():.1g}\u00b1{x.std().item():.1g}"
                    )
                if kill_on_nan and torch.any(torch.isnan(x)):
                    raise ValueError("NaN encountered in SDE solver")
                if trace:
                    path.append(x)
                if hook is not None:
                    hook(t, x, self.sde, self.sbm.score, self)
        if trace:
            return torch.stack(path), T
        return x

    def corrector_step(self, t, x, args, snr, **kwargs):
        """Basic Langevin corrector step for the SDE."""
        _, *D = x.shape
        z = torch.randn_like(x)
        epsilon = (snr * self.sde.sigma(t).view(-1, *[1] * len(D))) ** 2
        return x + epsilon * self.sbm.score(t, x, *args, **kwargs) + z * torch.sqrt(2 * epsilon)

    def drift(self, t: Tensor, x: Tensor, args: tuple, forward: bool, **kwargs):
        """SDE drift term"""
        f = self.sde.drift(t, x)
        if forward:
            return f
        g = self.sde.diffusion(t, x)
        s = self.sbm.score(t, x, *args, **kwargs)
        return f - g**2 * s

    def dx(self, t, x, args, dt, forward, dw=None, **kwargs):
        """SDE differential element dx"""
        if dw is None:
            dw = torch.randn_like(x) * torch.sqrt(dt.abs())
        return self.drift(t, x, args, forward, **kwargs) * dt + self.sde.diffusion(t, x) * dw


class HeunSDESolverAdaptive(SDESolverAdaptive):
    """
    Base SDE solver using a 2nd order Runge-Kutta method. For more
    details see Equation 2.5 in chapter 7.2 of the book "Introduction to
    Stochastic Differential Equations" by Thomas C. Gard.

    This solver adopts the Stratonovich interpretation of the SDE,
    though we note that the interpretation does not affect our package
    because our diffusion coefficient are homogeneous, i.e. they do not depend on x.
    The dependence of sde.diffusion on x is artificial in that it's only used
    to infer the shape of the state space.
    """

    def step(self, t, x, args, dt, forward, accuracy, max_dt, **kwargs):
        if "dw" in kwargs:
            dw = kwargs.pop("dw")
        else:
            dw = torch.randn_like(x) * torch.sqrt(dt.abs())

        k1 = self.dx(t, x, args, dt, forward, dw, **kwargs)
        k2 = self.dx(t + dt.squeeze(), x + k1, args, dt, forward, dw, **kwargs)
        dx = (k1 + k2) / 2

        # Check error is within noise level
        dw_norm = (
            self.sde.diffusion(t, x).flatten()[0]
            * torch.sqrt(dt.flatten()[0].abs())
            * np.sqrt(x.numel())
        )
        if torch.linalg.norm(k1 - dx).item() / dw_norm > accuracy:
            return self.step(
                t,
                x,
                args,
                dt * kwargs.get("stepsize_down", 0.5),
                forward,
                accuracy,
                max_dt,
                dw=dw * np.sqrt(kwargs.get("stepsize_down", 0.5)),
                **kwargs,
            )

        if forward:
            return dx, torch.min(dt * kwargs.get("stepsize_up", 1.4), max_dt)
        return dx, torch.max(dt * kwargs.get("stepsize_up", 1.4), max_dt)
