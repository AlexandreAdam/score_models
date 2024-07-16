from typing import Callable, Union

from torch.nn import Module
from torch.func import vjp

from ..sde import SDE
from .score_model import ScoreModel

__all__ = ["SLIC"]

class SLIC(ScoreModel):
    def __init__(
            self, 
            forward_model: Callable, # TODO inspect signature and check for differentiability
            net: Union[str, Module] = None, 
            sde: SDE=None, 
            path=None,
            **hyperparameters
            ):
        super().__init__(net, sde=sde, path=path, **hyperparameters)
        self.forward_model = forward_model
        
    def slic_score(self, t, x, y):
        """
        See Legin et al. (2023), https://iopscience.iop.org/article/10.3847/2041-8213/acd645
        """
        y_hat, vjp_func = vjp(self.forward_model, x)
        return - vjp_func(self.score(t, y - y_hat))[0]
