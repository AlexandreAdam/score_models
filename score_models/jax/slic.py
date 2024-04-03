from typing import Callable, Union, Optional

import equinox as eqx
from jax import vjp
from .sde import SDE
from .score_model import ScoreModel

class SLIC(ScoreModel):
    """
    Original implementation of SLIC
    """
    forward_model: Callable

    def __init__(
            self, 
            forward_model: Callable,
            model: Optional[Union[str, eqx.Module]] = None, 
            sde: Optional[SDE] = None, 
            checkpoints_directory: Optional[str] = None,
            **hyperparameters
            ):
        super().__init__(model, sde=sde, checkpoints_directory=checkpoints_directory, **hyperparameters)
        self.forward_model = forward_model
        
    def slic_score(self, t, x, y):
        """
        See Legin et al. (2023), https://iopscience.iop.org/article/10.3847/2041-8213/acd645
        """
        y_hat, vjp_func = vjp(self.forward_model, x)
        return - vjp_func(self.score(t, y - y_hat))[0]
