import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from jax import grad, vmap
from .base import ScoreModelBase
from .dsm import denoising_score_matching
from .sde import SDE
from typing import Union, Optional
import equinox as eqx


__all__ = ["ScoreModel", "EnergyModel"]


class ScoreModel(ScoreModelBase):
    def __init__(
            self,
            model: Optional[Union[str, eqx.Module]] = None,
            sde: Optional[Union[SDE, str]] = None,
            checkpoints_directory: Optional[str] = None,
            **hyperparameters):
        super().__init__(model, sde=sde, checkpoints_directory=checkpoints_directory, **hyperparameters)
    
    def loss_fn(self, key: PRNGKeyArray, x: Array, *args):
        return denoising_score_matching(key, self, x, *args)

    def score(self, t, x, *args):
        _, *D = x.shape
        return self.model(t, x, *args) / self.sde.sigma(t).reshape(-1, *[1]*len(D))


class EnergyModel(ScoreModelBase):
    nn_is_energy: bool

    def __init__(
            self, 
            model: Optional[Union[str, eqx.Module]] = None, 
            sde: Optional[Union[str, SDE]] = None, 
            checkpoints_directory: Optional[str] = None, 
            **hyperparameters):
        super().__init__(model, sde=sde, checkpoints_directory=checkpoints_directory, **hyperparameters)
        nn_is_energy = self.hyperparameters.get("nn_is_energy", False)
        self.nn_is_energy = nn_is_energy

    def loss_fn(self, key: PRNGKeyArray, x: Array, *args):
        return denoising_score_matching(key, self, x, *args)
    
    def energy(self, t, x, *args):
        if self.nn_is_energy:
            return self._nn_energy(t, x, *args)
        else:
            return self._unet_energy(t, x, *args)

    def _unet_energy(self, t, x, *args):
        _, *D = x.shape
        return 0.5 / self.sde.sigma(t) * jnp.sum((x - self.model(t, x, *args))**2, axis=tuple(range(1, 1+len(D))))
    
    def _nn_energy(self, t, x, *args):
        return self.model(t, x, *args).squeeze(1) / self.sde.sigma(t)
    
    def score(self, t, x, *args):
        _, *D = x.shape
        # small wrapper to account for input without batch dim from vmap
        energy = lambda t, x: self.energy(t[None], x[None], *args).squeeze(0)
        return -vmap(grad(energy, argnums=1))(t, x, *args)

