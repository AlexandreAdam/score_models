from typing import Union, Optional, Callable
from abc import ABC, abstractmethod

import torch
from torch.nn import Module
from torch import Tensor

from ..save_load_utils import (
    save_checkpoint,
    save_hyperparameters,
    load_hyperparameters,
    load_checkpoint,
    initialize_architecture,
    initialize_sde,
)
from ..utils import DEVICE
from ..sde import SDE
from ..trainer import Trainer
from ..phema import PostHocEMA, ema_lengths_from_path


class Base(Module, ABC):
    def __init__(
        self,
        net: Optional[Union[str, Module]] = None,
        sde: Optional[Union[str, SDE]] = None,
        path: Optional[str] = None,
        checkpoint: Optional[int] = None,
        ema_length: Optional[float] = None,
        device=DEVICE,
        **hyperparameters
    ):
        super().__init__()
        self.device = device
        # Backward compatibility
        self.reload_optimizer = True
        if "checkpoints_directory" in hyperparameters.keys() and path is None:
            path = hyperparameters["checkpoints_directory"]
            hyperparameters.pop("checkpoints_directory")
        if "model" in hyperparameters.keys() and net is None:
            net = hyperparameters["model"]
            hyperparameters.pop("model")
        if "model_checkpoint" in hyperparameters.keys() and checkpoint is None:
            checkpoint = hyperparameters["model_checkpoint"]
            hyperparameters.pop("model_checkpoint")
        
        # Validate inputs
        if ema_length is not None and path is None:
            raise ValueError("Must provide a 'path' to use PostHocEMA.")
        if net is None and path is None:
            raise ValueError("Must provide either 'net' or 'path' to instantiate the model.")
        
        # First load architecture and hyperparameters
        self.path = path
        if net is None or isinstance(net, str):
            self.net, self.hyperparameters = initialize_architecture(
                path, net=net, device=device, checkpoint=checkpoint, verbose=0 if self.path else 0, **hyperparameters
            )
        else:
            self.net = net
            self.hyperparameters = hyperparameters

        # Load the SDE
        if isinstance(sde, SDE):
            self.hyperparameters["sde"] = sde.__class__.__name__.lower()
            self.sde = sde
            sde_params = sde.hyperparameters
        else:
            if isinstance(sde, str):
                self.hyperparameters["sde"] = sde
            self.sde, sde_params = initialize_sde(**self.hyperparameters)
        self.hyperparameters.update(sde_params)  
        
        # Load the checkpoint (or a combination of them with PostHoc EMA)
        if self.path:
            self.load(checkpoint, raise_error=False, ema_length=ema_length, verbose=1)
        else:
            self.loaded_checkpoint = None
        self.net.to(device)
        self.to(device)

        if hasattr(self.net, "hyperparameters"):
            self.hyperparameters.update(self.net.hyperparameters)

        # Backward compatibility
        if "model_architecture" not in self.hyperparameters:
            self.hyperparameters["model_architecture"] = self.net.__class__.__name__.lower()
        self.model = self.net

    @abstractmethod
    def forward(self, t, x, *args, **kwargs) -> Tensor: ...

    @abstractmethod
    def loss(self, x, *args, **kwargs) -> Tensor: ...

    def save(
        self,
        path: Optional[str] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        create_path: bool = True,
        step: Optional[int] = None, # Iteration number
        ema_length: Optional[float] = None, # Relative EMA length scale
        verbose: int = 0,
        **kwargs
    ):
        """
        Save the model checkpoint to the provided path or the path provided during initialization.

        Args:
            path (str, optional): The path to save the checkpoint. Default is path provided during initialization.
            optimizer (torch.optim.Optimizer, optional): Optimizer to save alongside the checkpoint. Default is None.
            create_path (bool, optional): Whether to create the path if it does not exist. Default is True.
            step (int, optional): The iteration number to save. Default is None.
            ema_length (float, optional): The relative EMA length scale to save. Default is None.
        """
        path = path or self.path
        if path is None:
            raise ValueError(
                "No path provided to save the model. Please provide a valid path or initialize the model with a path."
            )
        if optimizer:
            save_checkpoint(model=optimizer, path=path, key="optimizer", create_path=create_path)
        save_checkpoint(model=self.net, path=path, key="checkpoint", create_path=create_path, step=step, ema_length=ema_length, verbose=verbose)
        self.save_hyperparameters(path)

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
            ema_length: Optional[float] = None,
            raise_error: bool = True,
            verbose: int = 0,
            **kwargs
            ):
        """
        Load a specific checkpoint from the model.

        Args:
            checkpoint (int): The checkpoint number to load. If not provided, load the lastest checkpoint found.
            optimizer (torch.optim.Optimizer, optional): The optimizer to load. Default is None.
            raise_error (bool, optional): Whether to raise an error if checkpoint is not found. Default is True.
            ema_length (float, optional): The relative EMA length scale to save. Default is None.
        """

        if self.path is None:
            raise ValueError(
                "A checkpoint can only be loaded if the model is instantiated with a path, e.g. model = ScoreModel(path='path/to/checkpoint')."
            )
        ema_lengths = ema_lengths_from_path(self.path)
        if len(ema_lengths) > 1:
            if ema_length is None:
                raise ValueError(
                    f"Multiple EMA lengths found in {self.path}. Please provide a specific ema_length to be synthesized from these checkpoints."
                )
            # Synthesize the model PostHoc
            ema = PostHocEMA(self.path, device=self.device)
            self.net = ema.synthesize_ema(ema_length)
            self.loaded_checkpoint = None # We load a weighted average of all the checkpoints
            self.reload_optimizer = False # If we fit the model again, we don't want to reload the optimizer, since model has changed 
            print(f"Synthesized the neural network with EMA length {ema_length:.2f}.")
        else:
            self.loaded_checkpoint = load_checkpoint(
                model=self,
                checkpoint=checkpoint,
                path=self.path,
                key="checkpoint",
                raise_error=raise_error,
                verbose=verbose
            )
            self.reload_optimizer = True
        self.hyperparameters.update(self.net.hyperparameters)

    def fit(
        self,
        dataset: torch.utils.data.Dataset,
        epochs: int = 1,
        batch_size: Optional[int] = None,
        learning_rate: float = 1e-3,
        learning_rate_decay: Optional[int] = None, # Number of steps to decay the learning rate
        clip: float = 1.,                       # Gradient clipping
        force_finite: bool = True,              # Force finite gradients
        warmup: int = 0,                        # Number of steps before reaching the target learning rate (good to let Adam warm up)
        ema_decay: Optional[float] = None,      # Traditional EMA
        ema_lengths: Optional[Union[float, tuple]] = 0.13, # Karras EMA 
        start_ema_after_step: int = 100,        # Delay update of traditional EMA by this number of steps
        soft_reset_every: Optional[int] = None, # Number of epochs before resetting the online model (and optimizer) to EMA model (ala Hare and Tortoise)
        update_ema_every: int = 1,              # How often to update the EMA model (steps)
        iterations_per_epoch: Optional[int] = None, # Number of iterations per epoch, can be defined independently of the number of items in the dataset
        checkpoint_every: int = 10,             # Save a checkpoint every this number of epochs
        models_to_keep: int = 1,                # Number of models to keep (the rest will be deleted)
        total_checkpoints_to_save: Optional[int] = None,
        max_time: float = float("inf"),
        shuffle: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None,
        path: Optional[str] = None,
        name_prefix: Optional[str] = None,
        seed: Optional[int] = None,
        reload_optimizer: bool = True,
        verbose: int = 0,
        **kwargs
    ) -> list:
        # Backward compatibility
        if "checkpoints_directory" in kwargs and path is None:
            path = kwargs["checkpoints_directory"]
        if "preprocessing_fn" in kwargs or "preprocessing" in kwargs:
            raise NotImplementedError("The 'preprocessing' argument has been removed. The preprocessing must be performed in the dataset.")
        if "checkpoints" in kwargs and checkpoint_every is None:
            checkpoint_every = kwargs["checkpoints"]
        trainer = Trainer(
            model=self,
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_decay=learning_rate_decay,
            clip=clip,
            force_finite=force_finite,
            warmup=warmup,
            ema_lengths=ema_lengths,
            ema_decay=ema_decay,
            start_ema_after_step=start_ema_after_step,
            soft_reset_every=soft_reset_every,
            update_ema_every=update_ema_every,
            iterations_per_epoch=iterations_per_epoch,
            max_time=max_time,
            optimizer=optimizer,
            checkpoint_every=checkpoint_every,
            models_to_keep=models_to_keep,
            total_checkpoints_to_save=total_checkpoints_to_save,
            path=path,
            name_prefix=name_prefix,
            seed=seed,
            shuffle=shuffle,
            reload_optimizer=self.reload_optimizer and reload_optimizer,
            verbose=verbose,
        )
        losses = trainer.train()
        return losses
