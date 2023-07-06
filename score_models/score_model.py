from .base import ScoreModelBase, Union, Module
from torch.func import grad
from torch import vmap
import torch

class ScoreModel(ScoreModelBase):
    def __init__(self, model: Union[str, Module] = None, checkpoints_directory=None, **hyperparameters):
        super().__init__(model, checkpoints_directory, **hyperparameters)

    def score(self, t, x):
        _, *D = x.shape
        return self.model(t=t, x=x) / self.sde.sigma(t).view(-1, *[1]*len(D))


class EnergyModel(ScoreModelBase):
    def __init__(self, model: Union[str, Module] = None, checkpoints_directory=None, **hyperparameters):
        super().__init__(model, checkpoints_directory, **hyperparameters)
    
    def energy(self, t, x):
        _, *D = x.shape
        return 0.5 * torch.sum((x - self.model(t=t, x=x))**2, dim=list(range(1, 1+len(D))))
    
    def score(self, t, x):
        _, *D = x.shape
        # small wrapper to account for input without batch dim from vmap
        energy = lambda t, x: self.energy(t.unsqueeze(0), x.unsqueeze(0)).squeeze(0)
        return - vmap(grad(energy, argnums=1))(t, x) / self.sde.sigma(t).view(-1, *[1]*len(D))
    
