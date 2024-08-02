from typing import Optional, Callable, TYPE_CHECKING
if TYPE_CHECKING:
    from score_models import ScoreModel

import torch
import json, os
import time
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch_ema import ExponentialMovingAverage
from datetime import datetime
from tqdm import tqdm

from .utils import DEVICE
from .save_load_utils import (
        remove_oldest_checkpoint, 
        last_checkpoint,
        load_checkpoint
        )


class Trainer:
    def __init__(
        self,
        model: "ScoreModel",
        dataset: Dataset,
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
    ):
        self.model = model
        self.net = model.net # Neural network to train
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        self .data_iter = iter(self.dataloader)
        self.preprocessing = preprocessing or (lambda x: x)
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        self.lr = learning_rate
        self.clip = clip
        self.warmup = warmup
        self.global_step = 0
        if seed:
            torch.manual_seed(seed)
        self.epochs = epochs
        self.checkpoint_every = checkpoint_every
        self.models_to_keep = models_to_keep
        self.iterations_per_epoch = iterations_per_epoch or len(self.dataloader)
        self.max_time = max_time
        
        # Provided model already has a path to load a checkpoint from
        if path and self.model.path:
            print(f"Loading a checkpoint from the model path {self.model.path} and saving in new path {path}...")
        if self.model.path:
            if not os.path.isdir(self.model.path): # Double check the path is valid
                print(f"Provided path {self.model.path} is not a valid directory. Can't load checkpoint.")
            else:
                checkpoint = load_checkpoint(
                        model=self.optimizer, 
                        checkpoint=self.model.loaded_checkpoint, 
                        path=self.model.path, 
                        key="optimizer",
                        device=self.model.device
                        )
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
                            "preprocessing": preprocessing.__name__ if preprocessing is not None else None,
                            "optimizer": self.optimizer.__class__.__name__,
                            "learning_rate": self.optimizer.param_groups[0]["lr"],
                            "ema_decay": ema_decay,
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
                            "iterations_per_epoch": iterations_per_epoch
                        },
                        f,
                        indent=4
                    )
            # Save model hyperparameters to reconstruct the model later
            self.model.save_hyperparameters(path)
        elif self.model.path:
            # Continue saving checkpoints in the model path
            self.path = self.model.path
        else:
            self.path = None
            print("No path provided. Training checkpoints will not be saved.")

    def save_checkpoint(self, loss: float):
        """
        Save model and optimizer if a path is provided. Then save loss and remove oldest checkpoints
        when the number of checkpoints exceeds models_to_keep.
        """
        if self.path:
            with self.ema.average_parameters():
                self.model.save(self.path, optimizer=self.optimizer)
        
            checkpoint = last_checkpoint(self.path)
            with open(os.path.join(self.path, "score_sheet.txt"), "a") as f:
                f.write(f"{checkpoint} {loss}\n")
                
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
            # Preprocessing
            x = self.preprocessing(x)
            # Training step
            self.optimizer.zero_grad()
            loss = self.model.loss(x, *args)
            loss.backward()
            if self.global_step < self.warmup:
                for g in self.optimizer.param_groups:
                    g['lr'] = self.lr * np.minimum(self.global_step / self.warmup, 1.0)
            if self.clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip)
            self.optimizer.step()
            self.ema.update()
            # Logging
            time_per_step_avg += time.time() - start
            cost += loss.item()
            self.global_step += 1
        cost /= self.iterations_per_epoch 
        time_per_step_avg /= self.iterations_per_epoch
        return cost, time_per_step_avg
        
    def train(self, verbose=0) -> list:
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
            # Logging
            losses.append(cost)
            pbar.set_description(f"Epoch {epoch + 1:d} | Cost: {cost:.1e} |")
            if verbose >= 2:
                print(f"epoch {epoch} | cost {cost:.2e} | time per step {time_per_step_avg:.2e} s", flush=True)
            elif verbose == 1:
                if (epoch + 1) % self.checkpoints == 0:
                    print(f"epoch {epoch} | cost {cost:.1e}", flush=True)
            if np.isnan(cost):
                print("Model exploded and returns NaN")
                break
            if (time.time() - global_start) > self.max_time * 3600:
                out_of_time = True
            if (epoch + 1) % self.checkpoint_every == 0 or epoch == self.epochs - 1 or out_of_time:
                self.save_checkpoint(cost)
            if out_of_time:
                print("Out of time")
                break
            if epoch > 0:
                estimated_time_for_epoch = time.time() - epoch_start

        print(f"Finished training after {(time.time() - global_start) / 3600:.3f} hours.")
        # Save EMA weights in the model for dynamic use (e.g. Jupyter notebooks)
        self.ema.copy_to(self.model.parameters())
        return losses
