from typing import Callable, Union

from torch import Tensor
from torch.nn import Module
from torch.func import vjp, jacrev
from .sde import SDE
from .score_model import SLIC
from .utils import DEVICE

class KernelSLIC(SLIC):
    def __init__(
            self, 
            kernel:Tensor, # Kernel of the forward model
            input_dimensions,
            forward_model:Callable,
            model: Union[str, Module] = None, 
            sde: SDE=None, 
            checkpoints_directory=None, 
            tikhonov_regularisation:float=1e-3,
            **hyperparameters
            ):
        """
        This clas overwrites DSM loss to include correlation from the forward model. We use the approximation 
        that the forward model can be represented as a convolution. The kerel of this convolution
        is provided by the user. 

        This approximation is useful because it can be used in the case where the Jacobian is quite large 
        or does not fit into memory. It allows us to use the convolution theorem to simplify the 
        Gaussian transition score function used as target for DSM. 
        
        In this class, I assume that kernel has the same shape as the output of the forward model. See 
        the method to construct an effective kernel from the forward model.
        """
        super().__init__(model, sde=sde, checkpoints_directory=checkpoints_directory, **hyperparameters)
        assert len(kernel.shape) == 3, "Kernel should be an image with channels first." 
        self.kernel = kernel
        self.forward_model = forward_model
    
    def loss_fn(self, samples:Tensor, *args:list[Tensor,...]) -> Tensor:
        ...
        
