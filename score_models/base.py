from typing import Callable, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torch.func import vjp
from torch_ema import ExponentialMovingAverage
from .utils import DEVICE
from typing import Union
from abc import ABC, abstractmethod
from tqdm import tqdm
import time
import os, glob, re, json
import numpy as np
from datetime import datetime
from contextlib import nullcontext

from .sde import VESDE, VPSDE, TSVESDE, SDE, subVPSDE
from .utils import load_architecture
from .solver import EulerMaruyamaSDE, RungeKuttaSDE_2, RungeKuttaSDE_4
from .ode import EulerODE, RungeKuttaODE_2, RungeKuttaODE_4


class ScoreModelBase(Module, ABC):
    def __init__(
        self,
        model: Union[str, Module] = None,
        sde: Union[str, SDE] = None,
        checkpoints_directory=None,
        model_checkpoint: int = None,
        device=DEVICE,
        **hyperparameters,
    ):
        super().__init__()
        if model is None and checkpoints_directory is None:
            raise ValueError("Must provide one of 'model' or 'checkpoints_directory'")
        if model is None or isinstance(model, str):
            model, hyperparams, self.loaded_checkpoint = load_architecture(
                checkpoints_directory,
                model=model,
                device=device,
                hyperparameters=hyperparameters,
                model_checkpoint=model_checkpoint,
            )
            hyperparameters.update(hyperparams)
        elif hasattr(model, "hyperparameters"):
            hyperparameters.update(model.hyperparameters)
        if sde is None:
            # Some sane defaults for quick use
            if "t_star" in hyperparameters.keys():
                print("Using the Truncated Scaled Variance Exploding SDE")
                sde = "tsve"
            elif "sigma_min" in hyperparameters.keys():
                print("Using the Variance Exploding SDE")
                sde = "ve"
            elif "beta_min" in hyperparameters.keys():
                print("Using the Variance Preserving SDE")
                sde = "vp"
            else:
                raise KeyError("SDE parameters are missing, please specify which sde to use")
        if isinstance(sde, str):
            if sde.lower() not in ["ve", "vp", "tsve"]:
                raise ValueError(f"The SDE {sde} provided is no supported")
            hyperparameters["sde"] = sde.lower()

        if sde.lower() == "ve":
            sde = VESDE(**hyperparameters)
        elif sde.lower() == "vp":
            if "t_min" not in hyperparameters.keys():
                hyperparameters["t_min"] = 1e-5
            sde = VPSDE(**hyperparameters)
        elif sde.lower() == "subvp":
            if "t_min" not in hyperparameters.keys():
                hyperparameters["t_min"] = 1e-5
            sde = subVPSDE(**hyperparameters)
        elif sde.lower() == "tsve":
            sde = TSVESDE(**hyperparameters)
        else:
            raise ValueError(f"The SDE {sde} provided is no supported")

        hyperparameters["model_architecture"] = model.__class__.__name__
        self.hyperparameters = hyperparameters
        self.checkpoints_directory = checkpoints_directory
        self.model = model
        self.model.to(device)
        self.sde = sde
        self.device = device

    def forward(self, t, x, *args) -> Tensor:
        return self.score(t, x, *args)

    @abstractmethod
    def score(self, t, x, *args) -> Tensor:
        ...

    @abstractmethod
    def loss_fn(self, x, *args) -> Tensor:
        ...

    def log_likelihood(self, x, N, method="EulerODE"):
        if method == "EulerODE":
            ode = EulerODE(self)
        elif method == "RungeKuttaODE_2":
            ode = RungeKuttaODE_2(self)
        elif method == "RungeKuttaODE_4":
            ode = RungeKuttaODE_4(self)
        else:
            raise ValueError("Method not supported")
        return ode.log_likelihood(x, N)

    def sample(self, shape, N, method="EulerMaruyamaSDE"):
        if method == "EulerMaruyamaSDE":
            solver = EulerMaruyamaSDE(self)
        elif method == "RungeKuttaSDE_2":
            solver = RungeKuttaSDE_2(self)
        elif method == "RungeKuttaSDE_4":
            solver = RungeKuttaSDE_4(self)
        else:
            raise ValueError("Method not supported")

        B, *D = shape
        xT = self.sde.prior(D).sample([B])
        return solver.reverse(xT, N)

    # def score_at_zero_temperature(
    # self,
    # x,
    # *args,
    # ode_steps: int,
    # n_cotangent_vectors: int = 1,
    # noise_type="rademacher",
    # verbose=0,
    # return_ll=False
    # ) -> Tensor:
    # """
    # Takes a gradient through the log_likelihood solver to compute the score
    # at zero temperature.
    # """
    # kwargs = {"ode_steps": ode_steps,
    # "n_cotangent_vectors": n_cotangent_vectors,
    # "verbose": verbose,
    # "noise_type": noise_type
    # }
    # wrapped_ll = lambda x: self.log_likelihood(x, *args, **kwargs)
    # ll, vjp_func = vjp(wrapped_ll, x)
    # score = vjp_func(torch.ones_like(ll))
    # if return_ll:
    # return score, ll
    # return score

    def fit(
        self,
        dataset: Dataset,
        preprocessing_fn=None,
        epochs=100,
        optimizer_kwargs={"lr": 1e-4},
        ema_kwargs={"decay": 0.9999},
        batch_size=1,
        shuffle=False,
        tolerance=0,
        max_time=float("inf"),
        warmup=0,
        clip=0.0,
        model_checkpoint=None,
        checkpoints=10,
        models_to_keep=2,
        seed=None,
        logname=None,
        logdir=None,
        logname_prefix="score_model",
        verbose=0,
        **kwargs,
    ):
        """
        Train the model on the provided dataset.

        Parameters:
            dataset (torch.utils.data.Dataset): The training dataset.
            preprocessing_fn (function, optional): A function to preprocess the input data. Default is None.
            learning_rate (float, optional): The learning rate for optimizer. Default is 1e-4.
            ema_decay (float, optional): The decay rate for Exponential Moving Average. Default is 0.9999.
            batch_size (int, optional): The batch size for training. Default is 1.
            shuffle (bool, optional): Whether to shuffle the dataset during training. Default is False.
            epochs (int, optional): The number of epochs for training. Default is 100.
            patience (float, optional): The patience value for early stopping. Default is infinity.
            tolerance (float, optional): The tolerance value for early stopping. Default is 0.
            max_time (float, optional): The maximum training time in hours. Default is infinity.
            warmup (int, optional): The number of warmup iterations for learning rate. Default is 0.
            clip (float, optional): The gradient clipping value. Default is 0.
            model_checkpoint (float, optional): If checkpoints_directory is provided, this can be used to restart training from checkpoint.
            checkpoints_directory (str, optional): The directory to save model checkpoints. Default is None.
            checkpoints (int, optional): The interval for saving model checkpoints. Default is 10 epochs.
            models_to_keep (int, optional): The number of best models to keep. Default is 3.
            seed (int, optional): The random seed for numpy and torch. Default is None.
            logname (str, optional): The logname for saving checkpoints. Default is None.
            logdir (str, optional): The path to the directory in which to create the new checkpoint_directory with logname.
            logname_prefix (str, optional): The prefix for the logname. Default is "score_model".

        Returns:
            list: List of loss values during training.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), **optimizer_kwargs)
        ema = ExponentialMovingAverage(self.model.parameters(), **ema_kwargs)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
        n_iterations_in_epoch = kwargs.get("n_iterations_in_epoch", len(dataloader))
        checkpoints_directory = kwargs.get("checkpoints_directory", self.checkpoints_directory)

        if preprocessing_fn is not None:
            preprocessing_name = preprocessing_fn.__name__
        else:
            preprocessing_name = None
            preprocessing_fn = lambda x: x

        # ==== Take care of where to write checkpoints and stuff =================================================================
        if checkpoints_directory is not None:
            if os.path.isdir(checkpoints_directory):
                logname = os.path.split(checkpoints_directory)[-1]
        elif logname is None:
            logname = logname_prefix + "_" + datetime.now().strftime("%y%m%d%H%M%S")

        save_checkpoint = False
        latest_checkpoint = 0
        if checkpoints_directory is not None or logdir is not None:
            save_checkpoint = True
            if checkpoints_directory is None:
                checkpoints_directory = os.path.join(logdir, logname)
            if not os.path.isdir(checkpoints_directory):
                os.mkdir(checkpoints_directory)

            script_params_path = os.path.join(checkpoints_directory, "script_params.json")
            if not os.path.isfile(script_params_path):
                with open(script_params_path, "w") as f:
                    json.dump(
                        {
                            "preprocessing": preprocessing_name,
                            "learning_rate": optimizer_kwargs.get("learning_rate", 1e-4),
                            "ema_decay": ema_kwargs.get("decay", 0.9999),
                            "batch_size": batch_size,
                            "shuffle": shuffle,
                            "epochs": epochs,
                            "patience": kwargs.get("patience", float("inf")),
                            "tolerance": tolerance,
                            "max_time": max_time,
                            "warmup": warmup,
                            "clip": clip,
                            "checkpoint_directory": checkpoints_directory,
                            "checkpoints": checkpoints,
                            "models_to_keep": models_to_keep,
                            "seed": seed,
                            "logname": logname,
                            "logname_prefix": logname_prefix,
                        },
                        f,
                        indent=4,
                    )

            model_hparams_path = os.path.join(checkpoints_directory, "model_hparams.json")
            if not os.path.isfile(model_hparams_path):
                with open(model_hparams_path, "w") as f:
                    json.dump(self.hyperparameters, f, indent=4)

            # ======= Load model if model_id is provided ===============================================================
            paths = glob.glob(os.path.join(checkpoints_directory, "checkpoint*.pt"))
            opt_paths = glob.glob(os.path.join(checkpoints_directory, "optimizer*.pt"))
            checkpoint_indices = [
                int(re.findall("[0-9]+", os.path.split(path)[-1])[-1]) for path in paths
            ]
            scores = [
                float(re.findall("([0-9]{1}.[0-9]+e[+-][0-9]{2})", os.path.split(path)[-1])[-1])
                for path in paths
            ]
            if checkpoint_indices:
                if model_checkpoint is not None:
                    checkpoint_path = paths[checkpoint_indices.index(model_checkpoint)]
                    self.model.load_state_dict(
                        torch.load(checkpoint_path, map_location=self.model.device)
                    )
                    optimizer.load_state_dict(
                        torch.load(
                            opt_paths[checkpoints == model_checkpoint], map_location=self.device
                        )
                    )
                    print(f"Loaded checkpoint {model_checkpoint} of {logname}")
                    latest_checkpoint = model_checkpoint
                else:
                    max_checkpoint_index = np.argmax(checkpoint_indices)
                    checkpoint_path = paths[max_checkpoint_index]
                    opt_path = opt_paths[max_checkpoint_index]
                    self.model.load_state_dict(
                        torch.load(checkpoint_path, map_location=self.device)
                    )
                    optimizer.load_state_dict(torch.load(opt_path, map_location=self.device))
                    print(
                        f"Loaded checkpoint {checkpoint_indices[max_checkpoint_index]} of {logname}"
                    )
                    latest_checkpoint = checkpoint_indices[max_checkpoint_index]

        if seed is not None:
            torch.manual_seed(seed)
        best_loss = float("inf")
        losses = []
        step = 0
        global_start = time.time()
        estimated_time_for_epoch = 0
        out_of_time = False

        data_iter = iter(dataloader)
        for epoch in (pbar := tqdm(range(epochs))):
            if (time.time() - global_start) > max_time * 3600 - estimated_time_for_epoch:
                break
            epoch_start = time.time()
            time_per_step_epoch_mean = 0
            cost = 0
            for _ in range(n_iterations_in_epoch):
                start = time.time()
                try:
                    X = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    X = next(data_iter)
                if isinstance(X, (list, tuple)):
                    x, *args = X
                else:
                    x = X
                    args = []
                if preprocessing_fn is not None:
                    x = preprocessing_fn(x)
                optimizer.zero_grad()
                loss = self.loss_fn(x, *args)
                loss.backward()

                if step < warmup:
                    for g in optimizer.param_groups:
                        g["lr"] = optimizer_kwargs.get("learning_rate", 1e-4) * np.minimum(
                            step / warmup, 1.0
                        )

                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip)

                optimizer.step()
                ema.update()

                _time = time.time() - start
                time_per_step_epoch_mean += _time
                cost += float(loss)
                step += 1

            time_per_step_epoch_mean /= len(dataloader)
            cost /= len(dataloader)
            pbar.set_description(f"Epoch {epoch + 1:d} | Cost: {cost:.1e} |")
            losses.append(cost)
            if verbose >= 2:
                print(
                    f"epoch {epoch} | cost {cost:.2e} | time per step {time_per_step_epoch_mean:.2e} s"
                )
            elif verbose == 1:
                if (epoch + 1) % checkpoints == 0:
                    print(f"epoch {epoch} | cost {cost:.1e}")

            if np.isnan(cost):
                print("Model exploded and returns NaN")
                break

            if cost < (1 - tolerance) * best_loss:
                best_loss = cost
                patience = kwargs.get("patience", float("inf"))
            else:
                patience -= 1

            if (time.time() - global_start) > max_time * 3600:
                out_of_time = True

            if save_checkpoint:
                if (
                    (epoch + 1) % checkpoints == 0
                    or patience == 0
                    or epoch == epochs - 1
                    or out_of_time
                ):
                    latest_checkpoint += 1
                    with open(
                        os.path.join(checkpoints_directory, "score_sheet.txt"), mode="a"
                    ) as f:
                        f.write(f"{latest_checkpoint} {cost}\n")
                    with ema.average_parameters():
                        torch.save(
                            self.model.state_dict(),
                            os.path.join(
                                checkpoints_directory,
                                f"checkpoint_{cost:.4e}_{latest_checkpoint:03d}.pt",
                            ),
                        )
                    torch.save(
                        optimizer.state_dict(),
                        os.path.join(
                            checkpoints_directory,
                            f"optimizer_{cost:.4e}_{latest_checkpoint:03d}.pt",
                        ),
                    )
                    paths = glob.glob(os.path.join(checkpoints_directory, "*.pt"))
                    checkpoint_indices = [
                        int(re.findall("[0-9]+", os.path.split(path)[-1])[-1]) for path in paths
                    ]
                    scores = [
                        float(
                            re.findall("([0-9]{1}.[0-9]+e[+-][0-9]{2})", os.path.split(path)[-1])[
                                -1
                            ]
                        )
                        for path in paths
                    ]
                    if (
                        len(checkpoint_indices) > 2 * models_to_keep
                    ):  # has to be twice since we also save optimizer states
                        index_to_delete = np.argmin(checkpoint_indices)
                        os.remove(
                            os.path.join(
                                checkpoints_directory,
                                f"checkpoint_{scores[index_to_delete]:.4e}_{checkpoint_indices[index_to_delete]:03d}.pt",
                            )
                        )
                        os.remove(
                            os.path.join(
                                checkpoints_directory,
                                f"optimizer_{scores[index_to_delete]:.4e}_{checkpoint_indices[index_to_delete]:03d}.pt",
                            )
                        )
                        del scores[index_to_delete]
                        del checkpoint_indices[index_to_delete]

            if patience == 0:
                print("Reached patience")
                break

            if out_of_time:
                print("Out of time")
                break

            if epoch > 0:
                estimated_time_for_epoch = time.time() - epoch_start

        print(f"Finished training after {(time.time() - global_start) / 3600:.3f} hours.")
        # Save EMA weights in the model
        ema.copy_to(self.parameters())
        return losses
