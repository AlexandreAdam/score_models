from abc import ABC, abstractmethod

import torch

from ..utils import DEVICE


class Solver(ABC):

    def __init__(self, score, *args, **kwargs):
        self.score = score

    @abstractmethod
    def solve(
        self, x, steps, forward, progress_bar=True, trace=False, kill_on_nan=False, **kwargs
    ): ...

    @abstractmethod
    def dx(self, t, x, **kwargs): ...

    @abstractmethod
    def _step(self, t, x, dt, dx, **kwargs): ...

    def __call__(
        self, x, steps, forward=False, progress_bar=True, trace=False, kill_on_nan=False, **kwargs
    ):
        return self.solve(x, steps, forward, progress_bar, trace, kill_on_nan, **kwargs)

    @property
    def sde(self):
        return self.score.sde

    def time_steps(self, steps: int, B: int = 1, forward: bool = True, device=DEVICE, **kwargs):
        t_min = kwargs.get("t_min", self.sde.t_min)
        t_max = kwargs.get("t_max", self.sde.t_max)
        if forward:
            return torch.linspace(t_min, t_max, steps + 1, device=device)[:-1].repeat(B, 1).T
        else:
            return torch.linspace(t_max, t_min, steps + 1, device=device)[:-1].repeat(B, 1).T

    def step_size(self, steps: int, forward: bool, device=DEVICE, **kwargs):
        h = 1 if forward else -1
        t_min = kwargs.get("t_min", self.sde.t_min)
        t_max = kwargs.get("t_max", self.sde.t_max)
        return torch.as_tensor(h * (t_max - t_min) / steps, device=device)
