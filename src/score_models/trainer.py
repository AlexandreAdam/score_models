from typing import Optional, Callable, TYPE_CHECKING, Union
if TYPE_CHECKING:
    from score_models import ScoreModel

import torch
import json, os
import time
import numpy as np
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
from tqdm import tqdm


from .ema import EMA
from .utils import DEVICE
from .save_load_utils import (
        remove_oldest_checkpoint, 
        last_checkpoint,
        load_checkpoint,
        save_checkpoint,
        load_global_step,
        update_loss_file
        )


__all__ = ["Trainer"]

def is_finite(number: Union[int, float]) -> bool:
    return not (np.isnan(number) or np.isinf(number))

def inverse_sqrt_schedule(step: int, learning_rate_decay: Optional[int] = None, warmup: int = 0):
    if learning_rate_decay is None:
        return 1.0
    return 1 / np.sqrt(max((step - warmup) / learning_rate_decay, 1))

def warmup_schedule(step: int, warmup: int = 0):
    if warmup == 0:
        return 1.0
    return np.minimum(step / warmup, 1.0)


class Trainer:
    def __init__(
        self,
        model: "ScoreModel",
        dataset: Dataset,
        epochs: int = 100,
        batch_size: Optional[int] = None,
        learning_rate: float = 1e-3,
        learning_rate_decay: Optional[int] = None, # Number of steps to decay the learning rate
        ema_decay: Optional[float] = None,         # Traditional EMA
        ema_lengths: Union[float, tuple] = 0.13,   # Karras EMA
        start_ema_after_step: int = 100,           # Parameter to delay update of traditional EMA
        soft_reset_every: Optional[int] = None,    # Number of epochs before resetting the online model (and optimizer) to EMA model (ala Hare and Tortoise)
        update_ema_every: int = 1,                 # Update for the EMA
        clip: float = 1.,
        force_finite: bool = False,
        warmup: int = 0,
        shuffle: bool = False,
        iterations_per_epoch: Optional[int] = None,
        max_time: float = float('inf'),
        checkpoint_every: int = 10,
        models_to_keep: int = 1,
        total_checkpoints_to_save: Optional[int] = None,
        path: Optional[str] = None,
        name_prefix: Optional[str] = None,
        seed: Optional[int] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        reload_optimizer: bool = True,
        verbose: int = 0,
        ): 
        self.verbose = verbose
        
        # Model
        self.model = model
        self.net = model.net # Neural network to train
        
        # Gradient clipping
        self.clip = clip
        self.force_finite = force_finite # Force NaN and inf values to 0 in the gradients to prevent loss spikes from ruining the training
        
        # Dataset
        if batch_size is not None:
            self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        else:
            self.dataloader = dataset
        self.data_iter = iter(self.dataloader)
        
        # Optimizer
        self.lr = learning_rate
        self.iterations_per_epoch = iterations_per_epoch or len(self.dataloader)
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Learning rate schedule
        self.global_step = 0
        if learning_rate_decay:
            assert learning_rate_decay > 0, "learning_rate_decay must be a positive integer."
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer,
            # Use self.global_step, so that we can reload training from a checkpoint and have the correct learning rate
            lr_lambda=lambda step: 
                warmup_schedule(self.global_step, warmup) * inverse_sqrt_schedule(self.global_step, learning_rate_decay, warmup)
        )
        self.warmup = warmup
        self.learning_rate_decay = learning_rate_decay
        
        # Exponential Moving Averages
        self.soft_reset_every = soft_reset_every
        if ema_lengths:
            if not isinstance(ema_lengths, (list, tuple)):
                ema_lengths = [ema_lengths]
            ema_lengths = sorted(ema_lengths)
            for sigma_rel in ema_lengths:
                assert sigma_rel > 0, "ema_length must be a positive float."
                assert sigma_rel < 0.28, "ema_length must be less than 0.28, see algorithm 2 from Karras et al. 2024."
            self.emas = [EMA(model=self.model, sigma_rel=ema_length, update_every=update_ema_every) for ema_length in ema_lengths]
            self.ema_lengths = ema_lengths
            print(f"Using Karras EMA with ema lengths [" + ",".join([f"{sigma_rel:.2f}" for sigma_rel in ema_lengths]) + "]")
        elif ema_decay:
            self.emas = [EMA(self.model, beta=ema_decay, start_update_after_step=start_ema_after_step, update_every=update_ema_every)]
            self.ema_lengths = [None]
            print(f"Using traditional EMA with decay {ema_decay}")
        else:
            raise ValueError("Either ema_decay or ema_lengths must be provided.")
        if soft_reset_every:
            print(f"Trainer will soft reset every {soft_reset_every} epochs using the EMA model with ema length {ema_lengths[-1]:.2f}.")
        
        if seed:
            torch.manual_seed(seed)
            
        # Logic to save checkpoints
        self.epochs = epochs
        self.checkpoint_every = checkpoint_every
        self.models_to_keep = models_to_keep
        self.max_time = max_time
        if total_checkpoints_to_save:
            # Implement some logic to recalculate the total number of checkpoints to save based on resources (epochs, max_time, etc.)
            # This is super useful for PostHocEMA, where we want to save a specific number of snapshots.
            # The logic here will override the models_to_keep parameter and checkpoint_every parameter.
            raise NotImplementedError("total_checkpoints_to_save is not implemented yet.")
        
        # Provided model already has a path to load a checkpoint from
        if path and self.model.path:
            print(f"Loading a checkpoint from the model path {self.model.path} and saving in new path {path}...")
        if self.model.path:
            if not os.path.isdir(self.model.path): # Double check the path is valid
                print(f"Provided path {self.model.path} is not a valid directory. Can't load checkpoint.")
            else:
                if reload_optimizer:
                    checkpoint = load_checkpoint(
                            model=self.optimizer, 
                            checkpoint=self.model.loaded_checkpoint, 
                            path=self.model.path, 
                            key="optimizer",
                            device=self.model.device,
                            raise_error=False
                            )
                # Resume global step
                self.global_step = load_global_step(self.model.path, self.model.loaded_checkpoint)
                print(f"Resumed training from checkpoint {checkpoint}.")
            
        # Create a new checkpoint and save checkpoint there
        if path:
            self.path = path
            if name_prefix: # Instantiate a new model, stamped with the current time
                model_name = name_prefix + "_" + datetime.now().strftime("%y%m%d%H%M%S")
                self.path = os.path.join(self.path, model_name)
            else:
                model_name = os.path.split(self.path)[-1]
            if not os.path.isdir(self.path):
                os.makedirs(self.path, exist_ok=True)

            # Save Training parameters
            file = os.path.join(self.path, "script_params.json")
            if not os.path.isfile(file):
                with open(file, "w") as f:
                    json.dump(
                        {
                            "dataset": dataset.__class__.__name__,
                            "optimizer": self.optimizer.__class__.__name__,
                            "learning_rate": self.lr,
                            "ema_decay": ema_decay,
                            "learning_rate_decay": learning_rate_decay,
                            "iterations_per_epoch": iterations_per_epoch,
                            "batch_size": batch_size,
                            "shuffle": shuffle,
                            "epochs": epochs,
                            "max_time": max_time,
                            "warmup": warmup,
                            "clip": clip,
                            "checkpoint_every": checkpoint_every,
                            "models_to_keep": models_to_keep,
                            "seed": seed,
                            "path": str(path),
                            "model_name": model_name,
                            "name_prefix": name_prefix,
                        },
                        f,
                        indent=4
                    )
            # Save model hyperparameters to reconstruct the model later
            self.model.save_hyperparameters(path)
            
            # Create the loss.txt file
            file = os.path.join(self.path, "loss.txt")
            if not os.path.isfile(file):
                with open(file, "w") as f:
                    f.write("checkpoint step loss time_per_step\n")
            else:
                # Grab the last checkpoint and step
                self.global_step = load_global_step(self.path)
        
        elif self.model.path:
            # Continue saving checkpoints in the model path
            self.path = self.model.path
        else:
            self.path = None
            if len(ema_lengths) > 1:
                raise ValueError("When training with more than one EMA length, a path must be provided to save the checkpoints and allow PostHocEMA.")
            print("No path provided. Training checkpoints will not be saved.")
    
    def reinitialize_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lambda step: 
                warmup_schedule(self.global_step, self.warmup) * inverse_sqrt_schedule(self.global_step, self.learning_rate_decay, self.warmup)
        )

    def save_checkpoint(self, loss: float, time_per_step):
        """
        Save the EMA models and the checkpoint Also, save the loss and time_per_step information in the loss.txt file.
        The global step is provided to the save method to be saved in the name of the file, alongside the ema_length. 
        This is needed for PostHocEMA to synthetize a model from the list of checkpoints.
        """
        if self.path:
            save_checkpoint(self.optimizer, self.path, key="optimizer", step=self.global_step, verbose=self.verbose)
            for i, (ema_length, ema) in enumerate(zip(self.ema_lengths, self.emas)):
                ema.ema_model.save(self.path, ema_length=ema_length, step=self.global_step, verbose=self.verbose)
            checkpoint = last_checkpoint(self.path)
            update_loss_file(self.path, checkpoint, self.global_step, loss, time_per_step)
                
            if self.models_to_keep:
                remove_oldest_checkpoint(self.path, self.models_to_keep)
    
    def train_epoch(self):
        time_per_step_avg = 0
        cost = 0
        for _ in range(self.iterations_per_epoch):
            start = time.time()
            # Load data
            try:
                X = next(self.data_iter)
            except StopIteration:
                self.data_iter = iter(self.dataloader) # Reset the iterator
                X = next(self.data_iter)
            if isinstance(X, (list, tuple)): # Handle conditional arguments
                x, *args = X
            else:
                x = X
                args = []
            # Training step
            self.optimizer.zero_grad()
            loss = self.model.loss(x, *args)
            loss.backward()
            # Regularization
            if self.clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip)
            if self.force_finite:
                for param in self.model.parameters():
                    if param.grad is not None:
                        torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
            # Update
            self.optimizer.step()
            for ema in self.emas:
                ema.update()
            self.lr_scheduler.step()
            # Logging
            time_per_step_avg += time.time() - start
            if not is_finite(loss.item()):
                if not self.force_finite: # If force_finite, prevent the NaN from ruining the training
                    cost += loss.item() # Make sure training stops when loss is NaN or inf
                    break
            else:
                cost += loss.item()
            self.global_step += 1
        cost /= self.iterations_per_epoch 
        time_per_step_avg /= self.iterations_per_epoch
        return cost, time_per_step_avg
        
    def train(self) -> list:
        losses = []
        global_start = time.time()
        estimated_time_for_epoch = 0
        out_of_time = False
        for epoch in (pbar := tqdm(range(self.epochs))):
            if (time.time() - global_start) > self.max_time * 3600 - estimated_time_for_epoch:
                break
            # Train
            epoch_start = time.time()
            cost, time_per_step_avg = self.train_epoch()
            
            # Soft Rest (ala Hare and Tortoise)
            if self.soft_reset_every and (epoch + 1) % self.soft_reset_every == 0:
                self.reinitialize_optimizer()
                self.emas[-1].soft_reset() # Use the longer average for the soft reset

            # Logging
            losses.append(cost)
            pbar.set_description(f"Epoch {epoch + 1:d} | Cost: {cost:.1e} | time per step {time_per_step_avg:.4f} s |")
            if np.isnan(cost):
                print("Model exploded and returns NaN")
                break
            if (time.time() - global_start) > self.max_time * 3600:
                out_of_time = True
            if (epoch + 1) % self.checkpoint_every == 0 or epoch == self.epochs - 1 or out_of_time:
                self.save_checkpoint(cost, time_per_step_avg)
            if out_of_time:
                print("Out of time")
                break
            if epoch > 0:
                estimated_time_for_epoch = time.time() - epoch_start

        print(f"Finished training after {(time.time() - global_start) / 3600:.3f} hours.")
        self.emas[-1].soft_reset() # Return the EMA model
        return losses
