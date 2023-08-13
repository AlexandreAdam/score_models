from typing import Union

from torch import Tensor
from torch.nn import Module
from torch.func import vjp, jacrev
from .sde import SDE
from .score_model import ScoreModel

class KernelSLIC(ScoreModel):
    def __init__(
            self, 
            kernel:Tensor=None, # Kernel of the forward model
            model: Union[str, Module] = None, 
            sde: SDE=None, 
            checkpoints_directory=None, 
            **hyperparameters
            ):
        """
        This clas overwrites DSM loss to include correlation from the forward model. We use the approximation 
        that the forward model can be represented as a convolution, which will need to be provided. 

        This approximation is useful because it can be used in the case where the Jacobian is quite large 
        or does not fit into memory. It allows us to use the convolution theorem to simplify the 
        Gaussian transition score function used as target for DSM. 
        """
        super().__init__(model, sde=sde, checkpoints_directory=checkpoints_directory, **hyperparameters)
        self.forward_model = forward_model
    
    def loss_fn() -> Tensor:
        pass
        


def effective_kernel(forward_model, input_dimensions, output_dimensions):
    """
    For an image to image forward model, we can compute the effective kernel in the
    cotangent space of the forward model using automatic differentiation and choosing 
    one of the rows as our effective kernel. 

    This approximates the entire forward model as a convolution between a tangent vector and
    an 'effective' kernel that represent the forward model. This allows us to compute 
    the
    """
    assert int(input_dimensions**(1/2)) == input_dimensions**(1/2)
    assert int(output_dimensions**(1/2)) == output_dimensions**(1/2)
