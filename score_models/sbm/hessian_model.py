from typing import Optional, Literal

import torch
from torch import Tensor
import os

from .base import Base
from .score_model import ScoreModel
from ..sde import SDE
from ..utils import DEVICE
from ..losses import second_order_dsm

__all__ = ["HessianDiagonal"]

class HessianDiagonal(Base):
    def __init__(
            self,
            score_model: Optional[ScoreModel] = None,
            net: Optional[torch.nn.Module] = None,
            sde: Optional[torch.Tensor] = None,
            path: Optional[str] = None,
            checkpoint: Optional[int] = None,
            device: torch.device = DEVICE,
            **hyperparameters
            ):
        if isinstance(score_model, ScoreModel):
            sde = score_model.sde
            print(sde, "HessianDiagonal")
        super().__init__(net, sde, path, checkpoint=checkpoint, device=device, **hyperparameters)
        # Check if SBM has been loaded, otherwise use the user provided SBM
        if not hasattr(self, "score_model"):
            if not isinstance(score_model, ScoreModel):
                raise ValueError("Must provide a ScoreModel instance to instantiate the HessianDiagonal model.")
            self.score_model = score_model
   
    def forward(self, t: Tensor, x: Tensor, *args):
        return self.diagonal(t, x, *args)
    
    def loss(self, x: Tensor, *args):
        return second_order_dsm(self, x, *args)
    
    def reparametrized_diagonal(self, t: Tensor, x: Tensor, *args):
        """
        
        """
        return self.net(t, x, *args)
    
    def diagonal(self, t: Tensor, x: Tensor, *args):
        """
        
        """
        B, *D = x.shape
        sigma_t = self.sde.sigma(t).view(B, *[1]*len(D))
        return (self.net(t, x, *args) - 1) / sigma_t**2
    
    def trace(self, t: Tensor, x: Tensor, *args):
        return self.diagonal(t, x, *args).flatten(1).sum(1)
                
    def save(
            self, 
            path: Optional[str] = None, 
            optimizer: Optional[torch.optim.Optimizer] = None,
            create_path: bool = True,
            update_sbm_checkpoint: bool = False
            ):
        """
        We use the super method to save checkpoints of the hessian diagonal net
        We need to save a copy of the score model net and hyperparameters 
        in order to reload it correctly. We have this special logic for that
        """
        super().save(path, optimizer, create_path) # Save Hessian net
        # Create a sub directory for the SBM 
        path = path or self.path
        sbm_path = os.path.join(path, "score_model")
        if not os.path.exists(sbm_path):
            self.score_model.save(sbm_path, create_path=True)
        elif update_sbm_checkpoint:
            raise NotImplementedError("Updating the SBM checkpoint is not supported yet.")
            # In case the SBM is optimized jointly... But right now we don't support that
            # self.score_model.save(sbm_path, create_path=False)
    
    def load(
            self, 
            checkpoint: Optional[int] = None, 
            optimizer: Optional[torch.optim.Optimizer] = None,
            raise_error: bool = True
            ):
        super().load(checkpoint, optimizer, raise_error)
        sbm_path = os.path.join(self.path, "score_model")
        self.score_model = ScoreModel(path=sbm_path)
