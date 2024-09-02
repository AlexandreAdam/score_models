from typing import Callable

import torch
from torch import Tensor
import numpy as np
from tqdm import tqdm
from .solver import Solver


class SDESolver(Solver):

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
        corrector_steps: int = 0,
        corrector_snr: float = 0.1,
        sk: float = 0,  # Set to -1 for Ito SDE, TODO: make sure this is right
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
            denoise_last_step: Whether to project to the boundary at the last step.
            corrector_steps: Number of corrector steps to add after each SDE step (0 for no corrector steps).
            corrector_snr: Signal-to-noise ratio for the corrector steps.
            sk: Stratonovich correction term (set to -1 for Ito SDE).
        """
        B, *D = x.shape

        # Step
        dt = self.step_size(steps, forward=forward, **kwargs)
        T = self.time_steps(steps, B, forward=forward, **kwargs)

        # Trace if requested
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
                raise ValueError("NaN encountered in SDE solver")

            # Update x
            x = x + self.step(t, x, args, dt, forward, sk=sk, **kwargs)

            # Add requested corrector steps
            for _ in range(corrector_steps):
                x = self.corrector_step(t, x, args, corrector_snr, **kwargs)

            if trace:
                path.append(x)

        # Project to boundary if denoising
        if denoise_last_step and not forward:
            x = self.tweedie(t, x, *args, **kwargs)
            if trace:
                path[-1] = x

        if trace:
            return torch.stack(path)
        return x

    def corrector_step(self, t, x, args, snr, **kwargs):
        """Basic Langevin corrector step for the SDE."""
        _, *D = x.shape
        z = torch.randn_like(x)
        epsilon = (snr * self.sde.sigma(t).view(-1, *[1] * len(D))) ** 2
        return x + epsilon * self.score(t, x, *args, **kwargs) + z * torch.sqrt(2 * epsilon)

    def drift(self, t: Tensor, x: Tensor, args: tuple, forward: bool, **kwargs):
        """SDE drift term"""
        f = self.sde.drift(t, x)
        if forward:
            return f
        g = self.sde.diffusion(t, x)
        s = self.score(t, x, *args, **kwargs)
        return f - g**2 * s

    def dx(self, t, x, args, dt, forward, dw=None, **kwargs):
        """SDE differential element dx"""
        if dw is None:
            dw = torch.randn_like(x) * torch.sqrt(dt.abs())
        return self.drift(t, x, args, forward, **kwargs) * dt + self.sde.diffusion(t, x) * dw


class EM_SDE(SDESolver):
    """
    Base solver for a stochastic differential equation (SDE) using the Euler-Maruyama method.
    """

    def step(self, t, x, args, dt, forward, sk=None, **kwargs):
        """base SDE solver"""
        dw = torch.randn_like(x) * torch.sqrt(dt.abs())
        return self.dx(t, x, args, dt, forward, dw, **kwargs)


class RK2_SDE(SDESolver):
    def step(self, t, x, args, dt, forward, sk, **kwargs):
        """Base SDE solver using a 2nd order Runge-Kutta method. For more
        details see Equation 2.5 in chapter 7.2 of the book "Introduction to
        Stochastic Differential Equations" by Thomas C. Gard. The equations have
        been adapted by including a term ``skdt`` which allows for solving the
        Ito SDE or the Stratonovich SDE."""
        dw = torch.randn_like(x) * torch.sqrt(dt.abs())
        skdt = sk * np.random.choice([-1, 1]) * torch.sqrt(dt.abs())
        k1 = self.dx(t, x, args, dt, forward, dw - skdt, **kwargs)
        k2 = self.dx(t + dt, x + k1, args, dt, forward, dw + skdt, **kwargs)
        return (k1 + k2) / 2


class RK4_SDE(SDESolver):
    def step(self, t, x, args, dt, forward, sk, **kwargs):
        """Base SDE solver using a 4th order Runge-Kutta method. For more
        details see Equation 3.6 in chapter 7.3 of the book "Introduction to
        Stochastic Differential Equations" by Thomas C. Gard. The equations have
        been adapted by including a term ``skdt`` which allows for solving the
        Ito SDE or the Stratonovich SDE."""
        dw = torch.randn_like(x) * torch.sqrt(dt.abs())
        skdt = sk * np.random.choice([-1, 1]) * torch.sqrt(dt.abs())
        k1 = self.dx(t, x, args, dt, forward, dw - skdt, **kwargs)
        k2 = self.dx(t + dt / 2, x + k1 / 2, args, dt, forward, dw + skdt, **kwargs)
        k3 = self.dx(t + dt / 2, x + k2 / 2, args, dt, forward, dw - skdt, **kwargs)
        k4 = self.dx(t + dt, x + k3, args, dt, forward, dw + skdt, **kwargs)
        return (k1 + 2 * k2 + 2 * k3 + k4) / 6
