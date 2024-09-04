from abc import ABC, abstractmethod

import torch
from torch import Tensor

from ..utils import DEVICE

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
                return super(Solver, cls).__new__(SOLVERS[solver.lower()])
            except KeyError:
                raise ValueError(
                    f'Unknown solver type: "{solver}". Must be one of {list(filter(lambda s: "_" in s, SOLVERS.keys()))}'
                )

        return super(Solver, cls).__new__(cls)

    def __init__(self, score, *args, **kwargs):
        self.score = score

    @abstractmethod
    def solve(
        self, x, steps, forward, *args, progress_bar=True, trace=False, kill_on_nan=False, **kwargs
    ): ...

    @abstractmethod
    def dx(self, t, x, args, dt, **kwargs): ...

    @abstractmethod
    def step(self, t, x, args, dt, dx, **kwargs): ...

    def __call__(
        self,
        x,
        steps,
        *args,
        forward=False,
        progress_bar=True,
        trace=False,
        kill_on_nan=False,
        **kwargs,
    ):
        """Calls the solve method with the given arguments."""
        return self.solve(
            x,
            steps,
            forward,
            *args,
            progress_bar=progress_bar,
            trace=trace,
            kill_on_nan=kill_on_nan,
            **kwargs,
        )

    @property
    def sde(self):
        return self.score.sde

    def time_steps(self, steps: int, B: int = 1, forward: bool = True, device=DEVICE, **kwargs):
        """
        Generate a tensor of time steps for integration. Note that the last
        entry is removed because it is the endpoint and not a step. For example
        if going from 0 to 1 with 10 steps, the steps are [0, 0.1, 0.2, ...,
        0.9], thus the returned tensor has the time value for the beginning of
        each block of time.
        """
        t_min = kwargs.get("t_min", self.sde.t_min)
        t_max = kwargs.get("t_max", self.sde.t_max)
        delta_t = torch.as_tensor(t_max - t_min, device=device)
        assert torch.allclose(delta_t, delta_t.reshape(-1)[0]), "All time steps must be the same"
        if forward:
            T = torch.linspace(0, 1, steps + 1, device=device)[:-1].repeat(B, 1).T
        else:
            T = torch.linspace(1, 0, steps + 1, device=device)[:-1].repeat(B, 1).T
        ret = delta_t * T + t_min
        return ret

    def step_size(self, steps: int, forward: bool, device=DEVICE, **kwargs):
        """Returns the step size for the integration. This is simply the time
        window divided by the number of steps. However, it is negative if going
        backwards in time."""

        h = 1 if forward else -1
        t_min = kwargs.get("t_min", self.sde.t_min)
        t_max = kwargs.get("t_max", self.sde.t_max)
        return torch.as_tensor(h * (t_max - t_min) / steps, device=device)

    def tweedie(self, t: Tensor, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Compute the Tweedie formula for the expectation E[x0 | xt]
        """
        B, *D = x.shape
        mu = self.sde.mu(t).view(-1, *[1] * len(D))
        sigma = self.sde.sigma(t).view(-1, *[1] * len(D))
        return (x + sigma**2 * self.score(t, x, *args, **kwargs)) / mu
