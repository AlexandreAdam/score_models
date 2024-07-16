from typing import Optional, Callable

import torch
import json, os
from torch.utils.data import DataLoader, Dataset
from torch_ema import ExponentialMovingAverage
from datetime import datetime
from tqdm import tqdm

from .utils import DEVICE
from .save_load_utils import remove_oldest_checkpoint, last_checkpoint


class Trainer:
    def __init__(
        self,
        model: "Base",
        dataset: Dataset,
        epochs: int = 100,
        batch_size: int = 1,
        shuffle: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None,
        learning_rate: float = 1e-3,
        ema_decay: float = 0.999,
        preprocessing: Optional[Callable] = None,
        path: Optional[str] = None,
        path_prefix: Optional[str] = None,
        max_time: float = float('inf'),
        clip: float = 0.,
        warmup: int = 0,
        save_every: int = 1,
        models_to_keep: int = 3,
        iterations_per_epoch: Optional[int] = None,
    ):
        """
        Args:
            model: The model to train (e.g. a ScoreModel instance)
            optimizer: The optimizer (e.g. Adam)
            path: The path to save the checkpoints (a directory)
        
        """
        self.save_every = save_every
        self.models_to_keep = models_to_keep
        self.max_time = max_time
        self.epochs = epochs
        self.model = model
        self.net = model.net # Neural network to train
        self.ema = ExponentialMovingAverage(self.net.parameters(), decay=ema_decay)
        self.clip = clip
        self.warmup = warmup
        if optimizer is None:
            optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        else:
            self.optimizer = optimizer
        
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        self.preprocessing = preprocessing or (lambda x: x)
        
        self.iterations_per_epoch = iterations_per_epoch or len(self.dataloader)
        
        # Create a path if prefix is given or if path is given

        # if checkpoints_directory is not None:
            # if os.path.isdir(checkpoints_directory):
                # logname = os.path.split(checkpoints_directory)[-1]
        # elif logname is None:
            # logname = logname_prefix + "_" + datetime.now().strftime("%y%m%d%H%M%S")


        # if checkpoints_directory is not None or logdir is not None:
            # save_checkpoint = True
            # if checkpoints_directory is None:
                # checkpoints_directory = os.path.join(logdir, logname)
            # if not os.path.isdir(checkpoints_directory):
                # os.mkdir(checkpoints_directory)
        self.path = path
        path = self.path or self.model.path
        if path:
            # Save Training parameters
            file = os.path.join(path, "script_params.json")
            if not os.path.isfile(file):
                with open(file, "w") as f:
                    json.dump(
                        {
                            "preprocessing": preprocessing.__name__ if preprocessing is not None else None,
                            "learning_rate": self.optimizer.param_groups[0]["lr"],
                            "ema_decay": ema_decay,
                            "batch_size": batch_size,
                            "shuffle": shuffle,
                            "epochs": epochs,
                            "max_time": max_time,
                            "warmup": warmup,
                            "clip": clip,
                            "path": path,
                            "checkpoints": checkpoints,
                            "models_to_keep": models_to_keep,
                            "seed": seed,
                            "logname": logname,
                            "logname_prefix": logname_prefix,
                            "iterations_per_epoch": iterations_per_epoch,
                        },
                        f,
                        indent=4
                    )
            # Save model hyperparameters to reconstruct the model later
            self.model.save_hyperparameters(path)

    def save_checkpoint(self, loss: float):
        """
        Save model and optimizer if a path is provided. Then save loss and remove oldest checkpoints
        when the number of checkpoints exceeds models_to_keep.
        """
        path = self.path or self.model.path
        if path:
            with self.ema.average_parameters():
                self.model.save(self.path, optimizer=self.optimizer)
        
            checkpoint = last_checkpoint(self.path)
            with open(os.path.join(path, "score_sheet.txt"), "a") as f:
                f.write(f"{checkpoint} {loss}\n")
                
            remove_oldest_checkpoint(path, models_to_keep=self.models_to_keep)
    

    def train_epoch(self, data_loader: DataLoader, epoch: int) -> Tuple[float, Dict[str, float]]:
        total_loss = 0.0
        
        step = epoch * len(data_loader)
        for batch in data_loader:
            self.optimizer.zero_grad()
            
            loss = self.model.loss(x, *args)


        # if seed is not None:
            # torch.manual_seed(seed)
        # best_loss = float('inf')
        # losses = []
        # step = 0
        # global_start = time.time()
        # estimated_time_for_epoch = 0
        # out_of_time = False

        # data_iter = iter(dataloader)
        # for epoch in (pbar := tqdm(range(epochs))):
            # if (time.time() - global_start) > max_time * 3600 - estimated_time_for_epoch:
                # break
            # epoch_start = time.time()
            # time_per_step_epoch_mean = 0
            # cost = 0
            # for _ in range(n_iterations_in_epoch):
                # start = time.time()
                # try:
                    # X = next(data_iter)
                # except StopIteration:
                    # data_iter = iter(dataloader)
                    # X = next(data_iter)
                # if isinstance(X, (list, tuple)):
                    # x, *args = X
                # else:
                    # x = X
                    # args = []
                # if preprocessing_fn is not None:
                    # x = preprocessing_fn(x)
                # optimizer.zero_grad()
                # loss = self.loss_fn(x, *args)
                # loss.backward()

                # if step < warmup:
                    # for g in optimizer.param_groups:
                        # g['lr'] = learning_rate * np.minimum(step / warmup, 1.0)

                # if clip > 0:
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip)

                # optimizer.step()
                # ema.update()

                # _time = time.time() - start
                # time_per_step_epoch_mean += _time
                # cost += float(loss)
                # step += 1

            
            loss.backward()
            self.optimizer.step()
            self.ema.update()

            total_loss += loss.item()
        total_loss /= self.iterations_per_epoch
        return total_loss
        
    def train(
        self,
        train_loader: DataLoader,
        epochs: int = 10,
    ):
        for epoch in (pbar := tqdm(range(epochs))):
            train_loss, train_loss_dict = self.train_epoch(train_loader, epoch)
            pbar.set_description(f"Epoch {epoch+1}, Loss: {train_loss:.4f}")
 
            if epoch % self.save_every == 0:
                self.save_checkpoint(epoch, train_loss, train_loss_dict)


    # def fit(
        # self,
        # preprocessing_fn=None,
        # epochs=100,
        # learning_rate=1e-4,
        # ema_decay=0.9999,
        # batch_size=1,
        # shuffle=False,
        # max_time=float('inf'),
        # warmup=0,
        # clip=0.,
        # checkpoints_directory=None,
        # model_checkpoint=None,
        # checkpoints=10,
        # models_to_keep=2,
        # seed=None,
        # logname=None,
        # logdir=None,
        # n_iterations_in_epoch=None,
        # logname_prefix="score_model",
        # verbose=0
    # ):
        # """
        # Train the model on the provided dataset.

        # Parameters:
            # dataset (torch.utils.data.Dataset): The training dataset.
            # preprocessing_fn (function, optional): A function to preprocess the input data. Default is None.
            # learning_rate (float, optional): The learning rate for optimizer. Default is 1e-4.
            # ema_decay (float, optional): The decay rate for Exponential Moving Average. Default is 0.9999.
            # batch_size (int, optional): The batch size for training. Default is 1.
            # shuffle (bool, optional): Whether to shuffle the dataset during training. Default is False.
            # epochs (int, optional): The number of epochs for training. Default is 100.
            # patience (float, optional): The patience value for early stopping. Default is infinity.
            # tolerance (float, optional): The tolerance value for early stopping. Default is 0.
            # max_time (float, optional): The maximum training time in hours. Default is infinity.
            # warmup (int, optional): The number of warmup iterations for learning rate. Default is 0.
            # clip (float, optional): The gradient clipping value. Default is 0.
            # model_checkpoint (float, optional): If checkpoints_directory is provided, this can be used to restart training from checkpoint.
            # checkpoints_directory (str, optional): The directory to save model checkpoints. Default is None.
            # checkpoints (int, optional): The interval for saving model checkpoints. Default is 10 epochs.
            # models_to_keep (int, optional): The number of best models to keep. Default is 3.
            # seed (int, optional): The random seed for numpy and torch. Default is None.
            # logname (str, optional): The logname for saving checkpoints. Default is None.
            # logdir (str, optional): The path to the directory in which to create the new checkpoint_directory with logname.
            # logname_prefix (str, optional): The prefix for the logname. Default is "score_model".


        # if seed is not None:
            # torch.manual_seed(seed)
        # best_loss = float('inf')
        # losses = []
        # step = 0
        # global_start = time.time()
        # estimated_time_for_epoch = 0
        # out_of_time = False

        # data_iter = iter(dataloader)
        # for epoch in (pbar := tqdm(range(epochs))):
            # if (time.time() - global_start) > max_time * 3600 - estimated_time_for_epoch:
                # break
            # epoch_start = time.time()
            # time_per_step_epoch_mean = 0
            # cost = 0
            # for _ in range(n_iterations_in_epoch):
                # start = time.time()
                # try:
                    # X = next(data_iter)
                # except StopIteration:
                    # data_iter = iter(dataloader)
                    # X = next(data_iter)
                # if isinstance(X, (list, tuple)):
                    # x, *args = X
                # else:
                    # x = X
                    # args = []
                # if preprocessing_fn is not None:
                    # x = preprocessing_fn(x)
                # optimizer.zero_grad()
                # loss = self.loss_fn(x, *args)
                # loss.backward()

                # if step < warmup:
                    # for g in optimizer.param_groups:
                        # g['lr'] = learning_rate * np.minimum(step / warmup, 1.0)

                # if clip > 0:
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip)

                # optimizer.step()
                # ema.update()

                # _time = time.time() - start
                # time_per_step_epoch_mean += _time
                # cost += float(loss)
                # step += 1

            # time_per_step_epoch_mean /= len(dataloader)
            # cost /= len(dataloader)
            # pbar.set_description(f"Epoch {epoch + 1:d} | Cost: {cost:.1e} |")
            # losses.append(cost)
            # if verbose >= 2:
                # print(f"epoch {epoch} | cost {cost:.2e} | time per step {time_per_step_epoch_mean:.2e} s")
            # elif verbose == 1:
                # if (epoch + 1) % checkpoints == 0:
                    # print(f"epoch {epoch} | cost {cost:.1e}")

            # if np.isnan(cost):
                # print("Model exploded and returns NaN")
                # break

            # if cost < (1 - tolerance) * best_loss:
                # best_loss = cost
                # patience = patience
            # else:
                # patience -= 1

            # if (time.time() - global_start) > max_time * 3600:
                # out_of_time = True

            # if save_checkpoint:
                # if (epoch + 1) % checkpoints == 0 or patience == 0 or epoch == epochs - 1 or out_of_time:
                    # latest_checkpoint += 1
                    # with open(os.path.join(checkpoints_directory, "score_sheet.txt"), mode="a") as f:
                        # f.write(f"{latest_checkpoint} {cost}\n")
                    # with ema.average_parameters():
                        # torch.save(self.model.state_dict(), os.path.join(checkpoints_directory, f"checkpoint_{cost:.4e}_{latest_checkpoint:03d}.pt"))
                    # torch.save(optimizer.state_dict(), os.path.join(checkpoints_directory, f"optimizer_{cost:.4e}_{latest_checkpoint:03d}.pt"))
                    # paths = glob.glob(os.path.join(checkpoints_directory, "*.pt"))
                    # checkpoint_indices = [int(re.findall('[0-9]+', os.path.split(path)[-1])[-1]) for path in paths]
                    # scores = [float(re.findall('([0-9]{1}.[0-9]+e[+-][0-9]{2})', os.path.split(path)[-1])[-1]) for path in paths]
                    # if len(checkpoint_indices) > 2*models_to_keep: # has to be twice since we also save optimizer states
                        # index_to_delete = np.argmin(checkpoint_indices)
                        # os.remove(os.path.join(checkpoints_directory, f"checkpoint_{scores[index_to_delete]:.4e}_{checkpoint_indices[index_to_delete]:03d}.pt"))
                        # os.remove(os.path.join(checkpoints_directory, f"optimizer_{scores[index_to_delete]:.4e}_{checkpoint_indices[index_to_delete]:03d}.pt"))
                        # del scores[index_to_delete]
                        # del checkpoint_indices[index_to_delete]

            # if patience == 0:
                # print("Reached patience")
                # break

            # if out_of_time:
                # print("Out of time")
                # break

            # if epoch > 0:
                # estimated_time_for_epoch = time.time() - epoch_start

        # print(f"Finished training after {(time.time() - global_start) / 3600:.3f} hours.")
        # # Save EMA weights in the model
        # ema.copy_to(self.parameters())
        # return losses
