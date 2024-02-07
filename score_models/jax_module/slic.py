from typing import Callable, Union

from torch.nn import Module
from torch.func import vjp
from .sde import SDE
from .score_model import ScoreModel

class SLIC(ScoreModel):
    """
    Original implementation of SLIC
    """
    def __init__(
            self, 
            model: Union[str, Module] = None, 
            forward_model: Callable = None, # need to be differentiable
            sde: SDE=None, 
            checkpoints_directory=None,
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
