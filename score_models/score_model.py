from .base import ScoreModelBase, Union, Module
from .sde import SDE
from torch.func import grad
from torch import vmap
import torch

class ScoreModel(ScoreModelBase):
    def __init__(self, model: Union[str, Module] = None, sde: SDE=None, checkpoints_directory=None, **hyperparameters):
        super().__init__(model, sde=sde, checkpoints_directory=checkpoints_directory, **hyperparameters)

    def score(self, t, x):
        _, *D = x.shape
        return self.model(t=t, x=x) / self.sde.sigma(t).view(-1, *[1]*len(D))
    

class EnergyModel(ScoreModelBase):
    def __init__(self, model: Union[str, Module] = None, sde: SDE=None, checkpoints_directory=None, **hyperparameters):
        super().__init__(model, sde=sde, checkpoints_directory=checkpoints_directory, **hyperparameters)
        nn_is_energy = model.hyperparameters.get("nn_is_energy", False)
        i=self.nn_is_energy = nn_is_energy
    
    def energy(self, t, x):
        if self.nn_is_energy:
            return self._nn_energy(t, x)
        else:
            return self._unet_energy(t, x)

    def _unet_energy(self, t, x):
        _, *D = x.shape
        return 0.5 / self.sde.sigma(t) * torch.sum((x - self.model(t=t, x=x))**2, dim=list(range(1, 1+len(D))))
    
    def _nn_energy(self, t, x):
        return self.model(t, x).squeeze(1) / self.sde.sigma(t) 
    
    def score(self, t, x):
        _, *D = x.shape
        # small wrapper to account for input without batch dim from vmap
        energy = lambda t, x: self.energy(t.unsqueeze(0), x.unsqueeze(0)).squeeze(0)
        # Don't forget the minus sign!
        return -vmap(grad(energy, argnums=1))(t, x)
    
