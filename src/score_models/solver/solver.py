from abc import ABC, abstractmethod
from typing import Optional

from ..utils import DEVICE
from torch import Tensor
import torch


__all__ = ["Solver"]


def all_subclasses(cls):
    subclasses = {}
    for subcls in cls.__subclasses__():
        subclasses[subcls.__name__.lower()] = subcls
        subclasses.update(all_subclasses(subcls))
    return subclasses


class Solver(ABC):
    """
    Base class for a solver of a stochastic/ordinary differential equation
    (SDE/ODE).

    Defines the signatures for methods related to integrating a differential
    equation (stochastic or ordinary) in the context of diffusion models.

    The only requirement on init is a ScoreModel object, which is used to define
    the DE by providing the SDE object and the score.
    """
    def __new__(cls, *args, solver=None, **kwargs):
        """Create the correct Solver subclass given the solver name."""
        if solver is not None:
            SOLVERS = all_subclasses(cls)
            try:
                # Removed arguments from super to fix autoreload bug in jupyter notebooks
                return super().__new__(SOLVERS[solver.lower()])
            except KeyError:
                raise ValueError(
                    f'Unknown solver type: "{solver}". Must be one of {list(filter(lambda s: "_" in s, SOLVERS.keys()))}'
                )

        return super(Solver, cls).__new__(cls)

    def __init__(self, sbm, *args, **kwargs):
        self.sbm = sbm

    @abstractmethod
    def solve(
            self, x, *args, steps: int, forward: bool, progress_bar=True, trace=False, kill_on_nan=False, **kwargs
    ): ...

    @abstractmethod
    def dx(self, t, x, *args, dt, **kwargs): ...

    @abstractmethod
    def step(self, t, x, *args, dt, dx, **kwargs): ...

    def __call__(
        self,
        x,
        *args,
        steps: int,
        forward=False,
        progress_bar=True,
        trace=False,
        kill_on_nan=False,
        **kwargs,
    ):
        return self.solve(
            x,
            *args,
            steps=steps,
            forward=forward,
            progress_bar=progress_bar,
            trace=trace,
            kill_on_nan=kill_on_nan,
            **kwargs,
        )

    @property
    def sde(self):
        return self.sbm.sde

    def time_steps(
        self,
        steps: int,
        B: int,
        D: tuple,
        time_steps: Optional[Tensor] = None,
        forward: bool = True,
        device=DEVICE,
        **kwargs,
    ):
        """
        Generate a tensor of time steps for integration. Note that the last
        entry is removed because it is the endpoint and not a step. For example
        if going from 0 to 1 with 10 steps, the steps are [0, 0.1, 0.2, ...,
        0.9], thus the returned tensor has the time value for the beginning of
        each block of time.
        """
        if time_steps is None:
            t_min = torch.as_tensor(kwargs.get("t_min", self.sde.t_min), device=device)
            t_max = torch.as_tensor(kwargs.get("t_max", self.sde.t_max), device=device)
            delta_t = t_max - t_min
            assert torch.allclose(
                delta_t, delta_t.reshape(-1)[0]
            ), "All time steps must be the same"
            delta_t = delta_t.reshape(-1)[0]  # Get the scalar value
            t_min = t_min.reshape(-1)[0]
            if forward:
                T = torch.linspace(0, 1, steps + 1, device=device)
            else:
                T = torch.linspace(1, 0, steps + 1, device=device)
            T = delta_t * T + t_min
        else:
            T = time_steps
        dT = T[1:] - T[:-1]
        T = T[:-1]
        T = T.reshape(-1, 1).repeat(1, B)
        dT = dT.reshape(T.shape[0], 1, *[1] * len(D)).repeat(1, B, *[1] * len(D))
        return T, dT

    def step_size(self, steps: int, forward: bool, device=DEVICE, **kwargs):
        """Returns the step size for the integration. This is simply the time
        window divided by the number of steps. However, it is negative if going
        backwards in time."""

        h = 1 if forward else -1
        t_min = kwargs.get("t_min", self.sde.t_min)
        t_max = kwargs.get("t_max", self.sde.t_max)
        return torch.as_tensor(h * (t_max - t_min) / steps, device=device)

