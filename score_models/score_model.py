from .base import ScoreModelBase, Union, Module
import numpy as np
from torch.nn import functional as F
from torch.func import grad
from torch import vmap
import torch

class ScoreModel(ScoreModelBase):
    def __init__(self, model: Union[str, Module] = None, checkpoints_directory=None, **hyperparameters):
        super().__init__(model, checkpoints_directory, **hyperparameters)

    def score(self, t, x):
        _, *D = x.shape
        return self.model(t=t, x=x) / self.sde.sigma(t).view(-1, *[1]*len(D))
    
    def score_and_controle_variate(self, t, x):
        # Method for training
        return self.score(t, x), 0.


class EnergyModel(ScoreModelBase):
    def __init__(self, model: Union[str, Module] = None, checkpoints_directory=None, **hyperparameters):
        super().__init__(model, checkpoints_directory, **hyperparameters)
        nn_is_energy = model.hyperparameters.get("nn_is_energy", False)
        if nn_is_energy:
            # TODO give MLP theoption of an output activation to make energy positive
            self.energy = lambda t, x: self.model(t, x).squeeze(1) # remove nn feature dim
        else:
            self.energy = self._unet_energy
    
    def _unet_energy(self, t, x):
        _, *D = x.shape
        return 0.5 * torch.sum((x - self.model(t=t, x=x))**2, dim=list(range(1, 1+len(D))))
    
    def score(self, t, x):
        _, *D = x.shape
        # small wrapper to account for input without batch dim from vmap
        energy = lambda t, x: self.energy(t.unsqueeze(0), x.unsqueeze(0)).squeeze(0)
        return - vmap(grad(energy, argnums=1))(t, x)
    
    def score_and_control_variate(self, t, x, z):
        """
        Method used for training 
        
        See Song & Kingma (2021): How to train your Energy-Based Model (https://arxiv.org/abs/2101.03288)
        and reference therein. This avoids loss explosion when sigma -> 0. 
        See also Wang et al. (2020): A Wasserstein Minimum Velocity Approach to Learning Unnormalized Models 
            (https://arxiv.org/abs/2002.07501)
        
        The control variate terms helps alleviate the variance of the loss as sigma -> 0.
        """
        _, *D = x.shape
        d = np.prod(D)
        score = self.score(t, x)
        sigma = self.sde.sigma(t)
        c = 2/sigma*(z * score).flatten(1).sum(dim=1) + (z**2).flatten(1).sum(dim=1)/sigma**2 - d/sigma**2
        return score, c
        
        
        
    
