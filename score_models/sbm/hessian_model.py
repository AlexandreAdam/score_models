from typing import Optional, Literal

from torch import Tensor
import torch
import os

from .base import Base
from .score_model import ScoreModel
from ..sde import SDE
from ..utils import DEVICE
from ..losses import second_order_dsm, second_order_dsm_meng_variation

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
            loss: Literal["canonical", "meng"] = "canonical",
            **hyperparameters
            ):
        if isinstance(score_model, ScoreModel):
            sde = score_model.sde
        super().__init__(net, sde, path, checkpoint=checkpoint, device=device, **hyperparameters)
        # Check if SBM has been loaded, otherwise use the user provided SBM
        if not hasattr(self, "score_model"):
            if not isinstance(score_model, ScoreModel):
                raise ValueError("Must provide a ScoreModel instance to instantiate the HessianDiagonal model.")
            self.score_model = score_model
        if loss == "canonical":
            self._loss = second_order_dsm
        elif loss == "meng":
            self._loss = second_order_dsm_meng_variation
        else:
            raise ValueError(f"Loss function {loss} is not recognized. Choose 'canonical' or 'meng'.")
        
        # Make sure ScoreModel weights are frozen (this class does not allow joint optimization for now)
        for p in self.score_model.net.parameters():
            p.requires_grad = False
        print("Score model weights are now frozen. This class does not currently support joint optimization.")
   
    def forward(self, t: Tensor, x: Tensor, *args):
        return self.diagonal(t, x, *args)
    
    def loss(self, x: Tensor, *args):
        return self._loss(self, x, *args)
    
    def reparametrized_diagonal(self, t: Tensor, x: Tensor, *args):
        return self.net(t, x, *args)
    
    def diagonal(self, t: Tensor, x: Tensor, *args):
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
            ):
        """
        We use the super method to save checkpoints of the hessian diagonal network.
        We need to save a copy of the score model net and hyperparameters 
        in order to reload it correctly. This method add special routines for that.
        """
        super().save(path, optimizer, create_path) # Save Hessian net
        # Create a sub directory for the SBM 
        path = path or self.path
        sbm_path = os.path.join(path, "score_model")
        if not os.path.exists(sbm_path):
            self.score_model.save(sbm_path, create_path=True)
    
    def load(
            self, 
            checkpoint: Optional[int] = None, 
            raise_error: bool = True
            ):
        """
        Super method reloads the HessianDiagonal net.
        Then we load the base score model from the score_model sub-directory.
        """
        super().load(checkpoint, raise_error)
        sbm_path = os.path.join(self.path, "score_model")
        self.score_model = ScoreModel(path=sbm_path)
