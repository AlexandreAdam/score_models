from typing import Union, Optional

from torch.func import grad
from torch import vmap
from torch.nn import Module
import torch

from .score_model import ScoreModel
from ..utils import DEVICE
from ..sde import SDE

__all__ = ["EnergyModel"]

class EnergyModel(ScoreModel):
    def __init__(
            self, 
            net: Optional[Union[str, Module]] = None,
            sde: Optional[Union[str, SDE]] = None,
            path: Optional[str] = None,
            checkpoint: Optional[int] = None,
            device=DEVICE,
            **hyperparameters
            ):
        super().__init__(net, sde, path, checkpoint=checkpoint, device=device, **hyperparameters)
        nn_is_energy = self.net.hyperparameters.get("nn_is_energy", False)
        self.nn_is_energy = nn_is_energy
        if nn_is_energy:
            self._energy = self._nn_energy
        else:
            self._energy = self._unet_energy

    def forward(self, t, x, *args):
        """
        Overwrite the forward method to return the energy function instead of the model output.
        """
        return self.energy(t, x, *args)

    def reparametrized_score(self, t, x, *args):
        """
        Numerically stable reparametrization of the score function for the DSM loss.
        Score function uses this method so self.score(t, x, *args) will also work as expected.
        """
        def energy(t, x, *args):
            # wrapper to feed energy im vmap
            t = t.unsqueeze(0)
            x = x.unsqueeze(0)
            args = [a.unsqueeze(0) for a in args]
            return self.unnormalized_energy(t, x, *args).squeeze(0)
        return - vmap(grad(energy, argnums=1))(t, x, *args) # Don't forget the minus sign!

    def unnormalized_energy(self, t, x, *args):
        return self._energy(t, x, *args)

    def energy(self, t, x, *args):
        sigma_t = self.sde.sigma(t)
        energy = self.unnormalized_energy(t, x, *args)
        return energy / sigma_t

    def _unet_energy(self, t, x, *args):
        _, *D = x.shape
        return 0.5 * torch.sum((x - self.net(t, x, *args)).flatten(1)**2, dim=1)
    
    def _nn_energy(self, t, x, *args):
        return self.net(t, x, *args).squeeze(1)
