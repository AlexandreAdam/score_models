from typing import Union, Optional
from abc import ABC, abstractmethod

import torch
from torch.nn import Module
from torch import Tensor
from torch_ema import ExponentialMovingAverage

from ..save_load_utils import (
    save_checkpoint,
    save_hyperparameters,
    load_checkpoint,
    load_architecture,
    load_sde
    )
from ..utils import DEVICE
from ..sde import SDE

class Base(Module, ABC):
    def __init__(
            self, 
            net: Optional[Union[str, Module]] = None,
            sde: Optional[Union[str, SDE]] = None,
            path: Optional[str] = None,
            checkpoint: Optional[int] = None,
            device=DEVICE,
            **hyperparameters
            ):
        super().__init__()
        if net is None and path is None:
            raise ValueError("Must provide either 'net' or 'path' to instantiate the model.")
        
        if not isinstance(net, Module):
            self.net, self.hyperparameters = load_architecture(
                    path,
                    net=net,
                    device=device,
                    checkpoint=checkpoint,
                    **hyperparameters
                    )
        else:
            self.net = net
            self.hyperparameters = hyperparameters
        if path:
            self.load(checkpoint=checkpoint, path=path)

        if hasattr(self.net, "hyperparameters"):
            self.hyperparameters.update(self.net.hyperparameters)
            
        self.sde = load_sde(sde, **hyperparameters)
        self.path = path
        self.device = device
        self.net.to(device)
        self.to(device)

    def forward(self, t, x, *args) -> Tensor:
        return self.net(t, x, *args)
    
    @abstractmethod
    def loss(self, x, *args) -> Tensor:
        ...
    
    def save(
            self, 
            path: Optional[str] = None, 
            optimizer: Optional[torch.optim.Optimizer] = None,
            create_path: bool = True
            ):
        """
        Save the model checkpoint to the provided path or the path provided during initialization.
        
        Args:
            path (str, optional): The path to save the checkpoint. Default is path provided during initialization.
            optimizer (torch.optim.Optimizer, optional): Optimizer to save alongside the checkpoint. Default is None.
            create_path (bool, optional): Whether to create the path if it does not exist. Default is True.
        """
        path = path or self.path
        if optimizer: # Save optimizer first since checkpoint number is inferred from number of checkpoint files 
            save_checkpoint(model=optimizer, path=path, key="optimizer", create_path=create_path)
        save_checkpoint(model=self.net, path=path, key="checkpoint", create_path=create_path)
    
    def save_hyperparameters(self, path: Optional[str] = None):
        """
        Save the hyperparameters of the model to a json file in the checkpoint directory.
        """
        path = path or self.path
        if path:
            save_hyperparameters(self.hyperparameters, path)
    
    def load(
            self, 
            checkpoint: Optional[int] = None, 
            path: Optional[str] = None, 
            optimizer: Optional[torch.optim.Optimizer] = None,
            raise_error: bool = True
            ):
        """
        Load a specific checkpoint from the model.

        Args:
            checkpoint (int): The checkpoint number to load. If not provided, load the lastest checkpoint found.
            path (str, optional): The path to the checkpoint. Default is path provided during initialization. 
                If no path was provided during initialization, load_checkpoint will raise an error.
            optimizer (torch.optim.Optimizer, optional): The optimizer to load. Default is None.
            raise_error (bool, optional): Whether to raise an error if checkpoint is not found. Default is True.
        """
        path = path or self.path
        if optimizer:
            load_checkpoint(model=optimizer, checkpoint=checkpoint, path=path, key="optimizer", raise_error=raise_error)
        load_checkpoint(model=self.net, checkpoint=checkpoint, path=path, key="checkpoint", raise_error=raise_error)
    
    def fit(
            self,
            ) -> list:
        ...

