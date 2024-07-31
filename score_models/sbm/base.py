from typing import Union, Optional, Callable
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
from ..trainer import Trainer

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
        # Backward compatibility
        if "checkpoints_directory" in hyperparameters.keys() and path is None:
            path = hyperparameters["checkpoints_directory"]
            hyperparameters.pop("checkpoints_directory")
        if "model" in hyperparameters.keys() and net is None:
            net = hyperparameters["model"]
            hyperparameters.pop("model")
        if "model_checkpoint" in hyperparameters.keys() and checkpoint is None:
            checkpoint = hyperparameters["model_checkpoint"]
            hyperparameters.pop("model_checkpoint")
        
        if net is None and path is None:
            raise ValueError("Must provide either 'net' or 'path' to instantiate the model.")

        self.path = path
        if net is None or isinstance(net, str):
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

        # Important to set these attributes before any loading attempt (device is needed)
        if isinstance(sde, SDE):
            self.hyperparameters["sde"] = sde.__class__.__name__.lower()
            self.sde = sde
            sde_params = sde.hyperparameters
        else:
            if isinstance(sde, str):
                self.hyperparameters["sde"] = sde
            self.sde, sde_params = load_sde(**self.hyperparameters)
        self.hyperparameters.update(sde_params) # Save the SDE hyperparameters, including the defaults
        self.device = device
        self.net.to(device)
        self.to(device)
        if self.path:
            self.load(checkpoint, raise_error=False) # If no checkpoint is found, loaded_checkpoint will be None
        else:
            self.loaded_checkpoint = None

        if hasattr(self.net, "hyperparameters"):
            self.hyperparameters.update(self.net.hyperparameters)
        
        # Backward compatibility
        if "model_architecture" not in self.hyperparameters:
            self.hyperparameters["model_architecture"] = self.net.__class__.__name__.lower()
        self.model = self.net

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
        if path:
            if optimizer: # Save optimizer first since checkpoint number is inferred from number of checkpoint files 
                save_checkpoint(model=optimizer, path=path, key="optimizer", create_path=create_path)
            save_checkpoint(model=self.net, path=path, key="checkpoint", create_path=create_path)
            self.save_hyperparameters(path) # If already present in path, this does nothing
        else:
            raise ValueError("No path provided to save the model. Please provide a valid path or initialize the model with a path.")
    
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
            raise_error: bool = True
            ):
        """
        Load a specific checkpoint from the model.

        Args:
            checkpoint (int): The checkpoint number to load. If not provided, load the lastest checkpoint found.
            optimizer (torch.optim.Optimizer, optional): The optimizer to load. Default is None.
            raise_error (bool, optional): Whether to raise an error if checkpoint is not found. Default is True.
        """
        if self.path is None:
            raise ValueError("A checkpoint can only be loaded if the model is instantiated with a path, e.g. model = ScoreModel(path='path/to/checkpoint').")
        self.loaded_checkpoint = load_checkpoint(model=self, checkpoint=checkpoint, path=self.path, key="checkpoint", raise_error=raise_error)
    
    def fit(
            self,
            dataset: torch.utils.data.Dataset,
            preprocessing: Optional[Callable] = None,
            batch_size: int = 1,
            shuffle: bool = False,
            epochs: int = 100, 
            iterations_per_epoch: Optional[int] = None,
            max_time: float = float('inf'),
            optimizer: Optional[torch.optim.Optimizer] = None,
            learning_rate: float = 1e-3,
            ema_decay: float = 0.999,
            clip: float = 0.,
            warmup: int = 0,
            checkpoint_every: int = 10,
            models_to_keep: int = 3,
            path: Optional[str] = None,
            name_prefix: Optional[str] = None,
            seed: Optional[int] = None,
            **kwargs
            ) -> list:
        # Backward compatibility
        if "checkpoints_directory" in kwargs and path is None:
            path = kwargs["checkpoints_directory"]
        if "preprocessing_fn" in kwargs and preprocessing is None:
            preprocessing = kwargs["preprocessing_fn"]
        if "checkpoints" in kwargs and checkpoint_every is None:
            checkpoint_every = kwargs["checkpoints"]
        trainer = Trainer(
            model=self,
            dataset=dataset,
            preprocessing=preprocessing,
            batch_size=batch_size,
            shuffle=shuffle,
            epochs=epochs,
            iterations_per_epoch=iterations_per_epoch,
            max_time=max_time,
            optimizer=optimizer,
            learning_rate=learning_rate,
            ema_decay=ema_decay,
            clip=clip,
            warmup=warmup,
            checkpoint_every=checkpoint_every,
            models_to_keep=models_to_keep,
            path=path,
            name_prefix=name_prefix,
            seed=seed
            )
        losses = trainer.train()
        return losses
