from .base import ScoreModelBase, Union, Module
from .dsm import denoising_score_matching
from .sde import SDE
from torch.func import grad
from torch import vmap
import torch
from torch import Tensor

class ScoreModel(ScoreModelBase):
    def __init__(self, model: Union[str, Module] = None, sde: SDE=None, checkpoints_directory=None, **hyperparameters):
        super().__init__(model, sde=sde, checkpoints_directory=checkpoints_directory, **hyperparameters)
    
    def loss_fn(self, x: Tensor, *args):
        return denoising_score_matching(self, x, *args)

    def score(self, x: Tensor, t: Tensor, *args):
        _, *D = x.shape
        return self.model(x, t, *args) / self.sde.sigma(t).view(-1, *[1]*len(D))
    

class EnergyModel(ScoreModelBase):
    def __init__(self, model: Union[str, Module] = None, sde: SDE=None, checkpoints_directory=None, **hyperparameters):
        super().__init__(model, sde=sde, checkpoints_directory=checkpoints_directory, **hyperparameters)
        nn_is_energy = self.model.hyperparameters.get("nn_is_energy", False)
        self.nn_is_energy = nn_is_energy

    def loss_fn(self, x: Tensor, *args):
        return denoising_score_matching(self, x, *args)
    
    def energy(self, x: Tensor, t: Tensor, *args):
        if self.nn_is_energy:
            return self._nn_energy(x, t, *args)
        else:
            return self._unet_energy(x, t, *args)

    def _unet_energy(self, x: Tensor, t: Tensor, *args):
        _, *D = x.shape
        return 0.5 / self.sde.sigma(t) * torch.sum((x - self.model(x, t, *args))**2, dim=list(range(1, 1+len(D))))
    
    def _nn_energy(self, x: Tensor, t: Tensor, *args):
        return self.model(x, t, *args).squeeze(1) / self.sde.sigma(t) 
    
    def score(self, x: Tensor, t: Tensor, *args):
        _, *D = x.shape
        # small wrapper to account for input without batch dim from vmap
        energy = lambda x, t: self.energy(x.unsqueeze(0), t.unsqueeze(0), *args).squeeze(0)
        return -vmap(grad(energy, argnums=0))(x, t, *args) # Don't forget the minus sign!
    
