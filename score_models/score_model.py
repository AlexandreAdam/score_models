from .base import ScoreModelBase, Union, Module
from .dsm import denoising_score_matching
from .sde import SDE
from torch.func import grad
from torch import vmap
import torch

class ScoreModel(ScoreModelBase):
    def __init__(self, model: Union[str, Module] = None, sde: SDE=None, checkpoints_directory=None, **hyperparameters):
        super().__init__(model, sde=sde, checkpoints_directory=checkpoints_directory, **hyperparameters)
    
    def loss_fn(self, x, *args):
        return denoising_score_matching(self, x, *args)

    def score(self, t, x, *args):
        _, *D = x.shape
        return self.model(t, x, *args) / self.sde.sigma(t).view(-1, *[1]*len(D))
    

class EnergyModel(ScoreModelBase):
    def __init__(self, model: Union[str, Module] = None, sde: SDE=None, checkpoints_directory=None, **hyperparameters):
        super().__init__(model, sde=sde, checkpoints_directory=checkpoints_directory, **hyperparameters)
        nn_is_energy = self.model.hyperparameters.get("nn_is_energy", False)
        self.nn_is_energy = nn_is_energy

    def loss_fn(self, x, *args):
        return denoising_score_matching(self, x, *args)
    
    def energy(self, t, x, *args):
        if self.nn_is_energy:
            return self._nn_energy(t, x, *args)
        else:
            return self._unet_energy(t, x, *args)

    def _unet_energy(self, t, x, *args):
        _, *D = x.shape
        return 0.5 / self.sde.sigma(t) * torch.sum((x - self.model(t, x, *args))**2, dim=list(range(1, 1+len(D))))
    
    def _nn_energy(self, t, x, *args):
        return self.model(t, x, *args).squeeze(1) / self.sde.sigma(t) 
    
    def score(self, t, x, *args):
        _, *D = x.shape
        # small wrapper to account for input without batch dim from vmap
        energy = lambda t, x: self.energy(t.unsqueeze(0), x.unsqueeze(0), *args).squeeze(0)
        return -vmap(grad(energy, argnums=1))(t, x, *args) # Don't forget the minus sign!
    
