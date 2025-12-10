from typing import Callable, Optional

from torch import Tensor
from tqdm import tqdm
from .solver import Solver
import torch
import numpy as np

__all__ = ["SDESolver", "EMSDESolver", "HeunSDESolver", "RK4SDESolver"]


class SDESolver(Solver):

    @torch.no_grad()
    def solve(
        self,
        x: Tensor,
        *args: tuple,
        steps: int,
        forward: bool,
        progress_bar: bool = True,
        trace: bool = False,
        kill_on_nan: bool = False,
        time_steps: Optional[Tensor] = None,
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
            steps: integration discretization.
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
        T, dT = self.time_steps(steps, B, D, time_steps=time_steps, forward=forward, **kwargs)
        if trace:
            path = [x]

        pbar = tqdm(tuple(zip(T, dT))) if progress_bar else zip(T, dT)
        for t, dt in pbar:
            x = x + self.step(t, x, args, dt, forward, **kwargs)
            for _ in range(corrector_steps):
                x = self.corrector_step(t, x, args, corrector_snr, **kwargs)
            
            # Logs
            if progress_bar:
                pbar.set_description(
                    f"t={t[0].item():.1g} | sigma={self.sde.sigma(t)[0].item():.1g} | "
                    f"x={x.mean().item():.1g}\u00B1{x.std().item():.1g}"
                )
            if kill_on_nan and torch.any(torch.isnan(x)):
                raise ValueError("NaN encountered in SDE solver")
            if trace:
                path.append(x)
            if hook is not None:
                hook(t, x, self.sde, self.sbm.score, self)
        if trace:
            return torch.stack(path)
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


class EMSDESolver(SDESolver):
    """
    Base solver for a stochastic differential equation (SDE) using the Euler-Maruyama method.
    """
    def step(self, t, x, args, dt, forward, **kwargs):
        dw = torch.randn_like(x) * torch.sqrt(dt.abs())
        return self.dx(t, x, args, dt, forward, dw, **kwargs)


class HeunSDESolver(SDESolver):
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
    def step(self, t, x, args, dt, forward, **kwargs):
        z = torch.randn_like(x)
        dw = z * torch.sqrt(dt.abs())
        k1 = self.dx(t, x, args, dt, forward, dw, **kwargs)
        k2 = self.dx(t + dt.squeeze(), x + k1, args, dt, forward, dw, **kwargs)
        return (k1 + k2) / 2


class RK4SDESolver(SDESolver):
    """Base SDE solver using a 4th order Runge-Kutta method. For more
    details see Equation 3.6 in chapter 7.3 of the book "Introduction to
    Stochastic Differential Equations" by Thomas C. Gard. 
    
    This solver adopts the Stratonovich interpretation of the SDE, 
    though we note that the interpretation does not affect our package 
    because our diffusion coefficient are homogeneous, i.e. they do not depend on x. 
    The dependence of sde.diffusion on x is artificial in that it's only used 
    to infer the shape of the state space.
    """
    def step(self, t, x, args, dt, forward, **kwargs):
        z = torch.randn_like(x)
        dw = z * torch.sqrt(dt.abs())
        k1 = self.dx(t, x, args, dt, forward, dw, **kwargs)
        k2 = self.dx(t + dt.squeeze() / 2, x + k1 / 2, args, dt, forward, dw, **kwargs)
        k3 = self.dx(t + dt.squeeze() / 2, x + k2 / 2, args, dt, forward, dw, **kwargs)
        k4 = self.dx(t + dt.squeeze(), x + k3, args, dt, forward, dw, **kwargs)
        return (k1 + 2 * k2 + 2 * k3 + k4) / 6

