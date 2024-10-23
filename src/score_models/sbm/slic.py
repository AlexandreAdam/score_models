from typing import Callable, Union, Optional

import torch
from torch.nn import Module
from torch.func import vjp
from inspect import signature, Parameter

from ..sde import SDE
from .score_model import ScoreModel
from ..utils import DEVICE

__all__ = ["SLIC"]

class SLIC(ScoreModel):
    def __init__(
            self, 
            forward_model: Callable,
            net: Optional[Union[str, Module]] = None, 
            sde: Optional[SDE]=None, 
            path: Optional[str] = None,
            checkpoint: Optional[int] = None,
            anneal_residuals: bool = False, # Add noise to residuals
            device=DEVICE,
            **hyperparameters
            ):
        """
        Args:
            forward_model: Callable
                Function that takes the inputs of the forward model and returns the model output.
            net: Optional[Union[str, Module]]
                Neural network architecture or path to the model
            sde: Optional[SDE]
                Stochastic differential equation
            anneal_residuals: bool
                Add noise to residuals according to the SLIC model SDE following Yang Song's hijacking trick
                (arxiv.org/pdf/2111.08005)
        """
        super().__init__(net, sde, path, checkpoint=checkpoint, device=device, **hyperparameters)
        self.forward_model = forward_model
        self.anneal_residuals = anneal_residuals
        if not self._valid_forward_model_signature(forward_model):
            raise ValueError("The forward model must have the signature: forward(t, x), with extra argument being optional.")
    
    def reparametrized_score(self, t, eta, *args):
        """
        Note: SLIC models should be trained as SBM models. SLIC is a wrapper class used only 
        for inference when a forward model is provided.
        
        Though with this reparametrization, the network can technically be trained with this class.
        
        Args:
            t: torch.Tensor
                Time index of the SDE
            eta: torch.Tensor
                Residuals of the model in the observation space
            args: list
                Additional arguments to the score model
        """
        return self.net(t, eta, *args)
    
    def residual_score(self, t, eta, *args):
        """
        Args:
            t: torch.Tensor
                Time index of the SDE
            eta: torch.Tensor
                Residuals of the model in the observation space
            args: list
                Additional arguments to the score model
        """
        B, *D = eta.shape
        sigma = self.sde.sigma(t).view(-1, *[1]*len(D))
        return self.net(t, eta, *args) / sigma
    
    def forward(self, t, y, x, *args):
        return self.score(t, y, x, *args)
        
    def score(self, t, y, x, *args):
        """
        See Legin et al. (2023), https://iopscience.iop.org/article/10.3847/2041-8213/acd645
        
        Args:
            t: torch.Tensor
                Time index of the SDE
            x: torch.Tensor
                Input tensor of the forward model
            y: torch.Tensor
                Observed output tensor
        """
        B, *D = y.shape
        y_hat, vjp_func = vjp(lambda x: self.forward_model(t, x), x)
        if self.anneal_residuals:
            mu = self.sde.mu(t).view(-1, *[1]*len(D))
            sigma = self.sde.sigma(t).view(-1, *[1]*len(D))
            z = torch.randn_like(y)
            eta = mu * (y - y_hat) + sigma * z
        else:
            eta = y - y_hat
        score = self.residual_score(t, eta, *args)
        return - vjp_func(score)[0]
    
    @staticmethod
    def _valid_forward_model_signature(f: Callable):
        sig = signature(f)
        args = list(sig.parameters.values())
        arg_names = list(sig.parameters.keys())
        if len(args) < 2:
            return False
        else:
            # Check if the first two arguments are positional
            check = all(_is_positional(a) for a in args[:2])
            check = check and arg_names[0] == "t" and arg_names[1] == "x"
            if len(args) > 2:
                # Check if the rest are optional
                check = check and all(_is_optional(a) for a in args[2:])
            return check
    
def _is_positional(param: Parameter) -> bool:
    return param.kind in [
        Parameter.POSITIONAL_OR_KEYWORD,
        Parameter.POSITIONAL_ONLY,
        Parameter.VAR_POSITIONAL]

def _is_optional(param: Parameter) -> bool:
    return (param.kind in [
        Parameter.VAR_POSITIONAL,
        Parameter.VAR_KEYWORD] or
        param.default is not Parameter.empty)
