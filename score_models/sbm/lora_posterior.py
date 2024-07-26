from typing import Union, Optional, Callable

import torch
import dill
import os
from torch import Tensor

from .lora import LoRAScoreModel
from .score_model import ScoreModel
from ..sde import SDE
from ..utils import DEVICE

__all__ = ["LoRAPosteriorScoreModel"]


class LoRAPosteriorScoreModel(LoRAScoreModel):
    """
    Class designed to learn a posterior approximation with a prior SBM 
    using LoRA weights to fine-tune the prior drift and learning a 
    scalar function to anneal the likelihood score.
    """
    def __init__(
            self,
            prior: Optional[ScoreModel] = None,
            likelihood_score: Optional[Callable] = None,
            path: Optional[str] = None,
            checkpoint: Optional[int] = None,
            lora_rank: Optional[int] = None,
            target_modules: Optional[str] = None,
            device: torch.device = DEVICE,
            **hyperparameters
            ):
        super().__init__(
                base_sbm=prior, 
                path=path, 
                checkpoint=checkpoint, 
                lora_rank=lora_rank, 
                target_modules=target_modules, 
                device=device, 
                **hyperparameters)
        if not hasattr(self, "likelihood_score"):
            if likelihood_score is None:
                raise ValueError("Must provide a likelihood score function when initializing a LoRAPosteriorScoreModel.")
            self.likelihood_score = likelihood_score
        
    def reparametrized_score(self, t, x, *args) -> Tensor:
        return self.lora_net(t, x, *args)

    def save(
            self,
            path: Optional[str] = None,
            optimizer: Optional[torch.optim.Optimizer] = None,
            create_path: bool = True
            ):
        super().save(path=path, optimizer=optimizer, create_path=create_path)
        path = path or self.path
        # Serialize the likelihood score function
        if path:
            with open(os.path.join(path, "likelihood_score.pkl"), "wb") as f:
                dill.dump(self.likelihood_score, f)

    def load(
            self, 
            checkpoint: Optional[int] = None, 
            optimizer: Optional[torch.optim.Optimizer] = None,
            raise_error: bool = True
            ):
        super().load(checkpoint=checkpoint, optimizer=optimizer, raise_error=raise_error)
        # Load/deserialize the likelihood score function
        with open(os.path.join(self.path, "likelihood_score.pkl"), "rb") as f:
            self.likelihood_score = dill.load(f)
