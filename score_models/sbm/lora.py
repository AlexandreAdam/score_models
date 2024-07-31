from typing import Union, Optional

import torch
import copy
import glob
import os
from torch import Tensor
from peft import LoraConfig, get_peft_model

from .score_model import ScoreModel
from ..architectures import NCSNpp, MLP
from ..sde import SDE
from ..utils import DEVICE
from ..save_load_utils import save_checkpoint, load_checkpoint

__all__ = ["LoRAScoreModel"]


def get_specific_layer_names(model):
    layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv1d, torch.nn.Conv3d)):
            layer_names.append(name)
    return layer_names


class LoRAScoreModel(ScoreModel):
    """
    Class designed to fine-tune an existing SBM with LoRA weights.
    """
    def __init__(
            self,
            base_sbm: Optional[ScoreModel] = None,
            path: Optional[str] = None,
            checkpoint: Optional[int] = None,
            lora_rank: Optional[int] = None,
            target_modules: Optional[str] = None,
            device: torch.device = DEVICE,
            **hyperparameters
            ):
        if base_sbm:
            # Initialize from scratch
            net = base_sbm.net
            base_hyperparameters = base_sbm.hyperparameters
            super().__init__(net, device=device, **base_hyperparameters)
            
            # Freeze the base net
            for param in self.net.parameters():
                param.requires_grad = False

            # Construct the LoRA model around the base net
            if target_modules is None:
                if isinstance(self.net, NCSNpp):
                    target_modules = ["Dense_0", "conv"]
                else:
                    target_modules = list(set(get_specific_layer_names(self.net)))
                    print(f"Automatically detecting target modules {' '.join(target_modules)}")
            if lora_rank is None:
                raise ValueError("LoRA rank must be provided when initializing from a base SBM.")
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank,
                init_lora_weights="gaussian",
                target_modules=target_modules
            )
            self.lora_net = get_peft_model(copy.deepcopy(self.net), lora_config)
            self.lora_net.print_trainable_parameters()
            self.hyperparameters["lora_rank"] = lora_rank
            self.hyperparameters["target_modules"] = target_modules
            print(f"Initialized LoRA weights with rank {lora_rank}")
        else:
            # Base model and LoRA initialized with the self.load method 
            super().__init__(path=path, checkpoint=checkpoint, device=device, **hyperparameters)
        
    def reparametrized_score(self, t, x, *args) -> Tensor:
        """
        Modify forward method to return the LoRA score function instead of the base SBM score function.
        This method is also used in the DSM loss function, such that the LoRA weights are used in the loss computation 
        for backpropagation.
        """
        return self.lora_net(t, x, *args)

    def save(
            self,
            path: Optional[str] = None,
            optimizer: Optional[torch.optim.Optimizer] = None,
            create_path: bool = True
            ):
        """
        Update the save method to save only one copy of the base SBM alongside the LoRA checkpoints.
        """
        path = path or self.path
        if path:
            # Save the base ScoreModel only once
            base_sbm_path = os.path.join(path, "base_sbm")
            if not os.path.exists(base_sbm_path):
                super().save(base_sbm_path, create_path=True)
            
            # Save the LoRA weights and the optimizer associated with them
            if optimizer: # Save optimizer first since checkpoint number is inferred from number of checkpoint files 
                save_checkpoint(model=optimizer, path=path, key="optimizer", create_path=create_path)
            save_checkpoint(model=self.lora_net, path=path, key="lora_checkpoint", create_path=create_path)
            self.save_hyperparameters(path)
        else:
            raise ValueError("No path provided to save the model. Please provide a valid path or initialize the model with a path.")

    def load(
            self, 
            checkpoint: Optional[int] = None, 
            raise_error: bool = True
            ):
        if self.path is None:
            raise ValueError("A checkpoint can only be loaded if the model is instantiated with a path, e.g. model = ScoreModel(path='path/to/checkpoint').")
        # Load base SBM (and freeze it)
        base_path = os.path.join(self.path, "base_sbm")
        self.net = ScoreModel(path=base_path).net
        for param in self.net.parameters():
            param.requires_grad = False
        
        # Load LoRA weights
        self.loaded_checkpoint = load_checkpoint(model=self, checkpoint=checkpoint, path=self.path, key="lora_checkpoint", raise_error=raise_error)
        print(f"Loaded LoRA weights with rank {self.hyperparameters['lora_rank']}")
        self.lora_net.print_trainable_parameters()
