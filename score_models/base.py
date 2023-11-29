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

from .sde import VESDE, VPSDE, TSVESDE, SDE
from .utils import load_architecture


class ScoreModelBase(Module, ABC):
    def __init__(
            self, 
            model: Union[str, Module]=None, 
            sde:Union[str, SDE]=None, 
            checkpoints_directory=None, 
            model_checkpoint:int=None, 
            device=DEVICE, 
            **hyperparameters
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
                    model_checkpoint=model_checkpoint
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
        if "T" not in hyperparameters.keys():
            hyperparameters["T"] = 1.
        if sde.lower() == "ve":
            sde = VESDE(sigma_min=hyperparameters["sigma_min"], sigma_max=hyperparameters["sigma_max"], T=hyperparameters["T"])
        elif sde.lower() == "vp":
            if "epsilon" not in hyperparameters.keys():
                hyperparameters["epsilon"] = 1e-5
            sde = VPSDE(
                    beta_min=hyperparameters["beta_min"], 
                    beta_max=hyperparameters["beta_max"], 
                    T=hyperparameters["T"],
                    epsilon=hyperparameters["epsilon"]
                    )
        elif sde.lower() == "tsve":
            if "epsilon" not in hyperparameters.keys():
                hyperparameters["epsilon"] = 0
            sde = TSVESDE(
                    sigma_min=hyperparameters["sigma_min"], 
                    sigma_max=hyperparameters["sigma_max"], 
                    t_star=hyperparameters["t_star"],
                    beta=hyperparameters["beta"],
                    T=hyperparameters["T"],
                    epsilon=hyperparameters["epsilon"]
                    )

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
    
    def ode_drift(self, t, x, *args):
        f = self.sde.drift(t, x)
        g = self.sde.diffusion(t, x)
        f_tilde = f - 0.5 * g**2 * self.score(t, x, *args)
        return f_tilde
    
    def hessian(self, t, x, *args, **kwargs):
        return self.divergence(self.drift_fn, t, x, *args, **kwargs)
    
    def divergence(self, drift_fn, t, x, *args, n_cotangent_vectors: int = 1, noise_type="rademacher") -> Tensor:
        B, *D = x.shape
        # duplicate noisy samples for for the Hutchinson trace estimator
        samples = torch.tile(x, [n_cotangent_vectors, *[1]*len(D)])
        # TODO also duplicate args
        t = torch.tile(t, [n_cotangent_vectors])
        # sample cotangent vectors
        vectors = torch.randn_like(samples)
        if noise_type == 'rademacher':
            vectors = vectors.sign()
        # Compute the trace of the Jacobian of the drift functions (Hessian if drift is just the score) 
        f = lambda x: drift_fn(t, x, *args)
        _, vjp_func = vjp(f, samples)
        divergence = (vectors * vjp_func(vectors)[0]).flatten(1).sum(dim=1)
        return divergence
    
    @torch.no_grad()
    def log_likelihood(
            self, 
            x,
            *args,
            ode_steps: int, 
            n_cotangent_vectors: int = 1, 
            noise_type="rademacher", 
            verbose=0,
            method="Euler",
            t0:int=0,
            t1:int=1
            ) -> Tensor:
        """
        A basic implementation of Euler discretisation method of the ODE associated 
        with the marginals of the learned SDE.
        
        ode_steps: Number of steps to perform in the ODE
        hutchinsons_samples: Number of samples to draw to compute the trace of the Jacobian (divergence)
        
        Note that this estimator only compute the likelihood for one trajectory. 
        For more precise log likelihood estimation, tile x along the batch dimension
        and averge the results. You can also increase the number of ode steps and increase
        the number of cotangent vector for the Hutchinson estimator.
        
        Using the instantaneous change of variable formula
        (Chen et al. 2018,https://arxiv.org/abs/1806.07366)
        See also Song et al. 2020, https://arxiv.org/abs/2011.13456)
        """
        kwargs = {"n_cotangent_vectors": n_cotangent_vectors, "noise_type": noise_type}
        disable = False if verbose else True  
        B, *D = x.shape
        log_p = 0.
        t = torch.ones([B]).to(self.device) * t0
        dt = (t1 - t0) / ode_steps
        # Small wrappers to make the notation a bit more readable
        f = lambda t, x: self.ode_drift(t, x, *args)
        div = lambda t, x: self.divergence(self.ode_drift, t, x, *args, **kwargs)
        for _ in tqdm(range(ode_steps), disable=disable):
            if method == "Euler":
                x = x + f(t, x) * dt
                log_p += div(t, x) * dt
                t = t + dt
            elif method == "Heun":
                previous_x = x.clone()
                drift = f(t, x)
                new_x = x + drift * dt
                x = x + 0.5 * (drift + f(t+dt, new_x)) * dt
                log_p += 0.5 * (div(t, previous_x) + div(t+dt, x)) * dt
                t = t + dt
            else:
                raise NotImplementedError("Invalid method, please select either Euler or Heun")
        log_p += self.sde.prior(D).log_prob(x)
        return log_p
    
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
    
    @torch.no_grad()
    def sample(
            self, 
            shape, # TODO change this so that specifying C, H, W is optional. Maybe save C, H, W in model hparams in the future
            steps, 
            condition:list=[],
            likelihood_score_fn:Callable=None,
            guidance_factor=1.
            ):
        """
        An Euler-Maruyama integration of the model SDE
        
        shape: Shape of the tensor to sample (including batch size)
        steps: Number of Euler-Maruyam steps to perform
        likelihood_score_fn: Add an additional drift to the sampling for posterior sampling. Must have the signature f(t, x)
        guidance_factor: Multiplicative factor for the likelihood drift
        """
        if not isinstance(condition, (list, tuple)):
            raise ValueError(f"condition must be a list or tuple or torch.Tensor, received {type(condition)}")
        B, *D = shape
        sampling_from = "prior" if likelihood_score_fn is None else "posterior"
        if likelihood_score_fn is None:
            likelihood_score_fn = lambda t, x: 0.
        x = self.sde.prior(D).sample([B]).to(self.device)
        dt = -(self.sde.T - self.sde.epsilon) / steps
        t = torch.ones(B).to(self.device) * self.sde.T
        for _ in (pbar := tqdm(range(steps))):
            pbar.set_description(f"Sampling from the {sampling_from} | t = {t[0].item():.1f} | sigma = {self.sde.sigma(t)[0].item():.1e}"
                                 f"| scale ~ {x.std().item():.1e}")
            t += dt
            if t[0] < self.sde.epsilon: # Accounts for numerical error in the way we discretize t.
                break
            g = self.sde.diffusion(t, x)
            f = self.sde.drift(t, x) - g**2 * (self.score(t, x, *condition) + guidance_factor * likelihood_score_fn(t, x))
            dw = torch.randn_like(x) * (-dt)**(1/2)
            x_mean = x + f * dt
            x = x_mean + g * dw 
            if torch.any(torch.isnan(x)):
                print("Diffusion is not stable, NaN were produced. Stopped sampling.")
                break
        return x_mean

    def fit(
        self,
        dataset: Dataset,
        preprocessing_fn=None,
        epochs=100,
        learning_rate=1e-4,
        ema_decay=0.9999,
        batch_size=1,
        shuffle=False,
        patience=float('inf'),
        tolerance=0,
        max_time=float('inf'),
        warmup=0,
        clip=0.,
        checkpoints_directory=None,
        model_checkpoint=None,
        checkpoints=10,
        models_to_keep=2,
        seed=None,
        logname=None,
        logdir=None,
        n_iterations_in_epoch=None,
        logname_prefix="score_model",
        verbose=0
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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
        if n_iterations_in_epoch is None:
            n_iterations_in_epoch = len(dataloader)
        if checkpoints_directory is None:
            checkpoints_directory = self.checkpoints_directory
        
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
                            "learning_rate": learning_rate,
                            "ema_decay": ema_decay,
                            "batch_size": batch_size,
                            "shuffle": shuffle,
                            "epochs": epochs,
                            "patience": patience,
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
                        indent=4
                    )
            
            model_hparams_path = os.path.join(checkpoints_directory, "model_hparams.json")
            if not os.path.isfile(model_hparams_path):
                with open(model_hparams_path, "w") as f:
                    json.dump(self.hyperparameters, f, indent=4)

            # ======= Load model if model_id is provided ===============================================================
            paths = glob.glob(os.path.join(checkpoints_directory, "checkpoint*.pt"))
            opt_paths = glob.glob(os.path.join(checkpoints_directory, "optimizer*.pt"))
            checkpoint_indices = [int(re.findall('[0-9]+', os.path.split(path)[-1])[-1]) for path in paths]
            scores = [float(re.findall('([0-9]{1}.[0-9]+e[+-][0-9]{2})', os.path.split(path)[-1])[-1]) for path in paths]
            if checkpoint_indices:
                if model_checkpoint is not None:
                    checkpoint_path = paths[checkpoint_indices.index(model_checkpoint)]
                    self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.model.device))
                    optimizer.load_state_dict(torch.load(opt_paths[checkpoints == model_checkpoint], map_location=self.device))
                    print(f"Loaded checkpoint {model_checkpoint} of {logname}")
                    latest_checkpoint = model_checkpoint
                else:
                    max_checkpoint_index = np.argmax(checkpoint_indices)
                    checkpoint_path = paths[max_checkpoint_index]
                    opt_path = opt_paths[max_checkpoint_index]
                    self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
                    optimizer.load_state_dict(torch.load(opt_path, map_location=self.device))
                    print(f"Loaded checkpoint {checkpoint_indices[max_checkpoint_index]} of {logname}")
                    latest_checkpoint = checkpoint_indices[max_checkpoint_index]

        if seed is not None:
            torch.manual_seed(seed)
        best_loss = float('inf')
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
                        g['lr'] = learning_rate * np.minimum(step / warmup, 1.0)

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
                print(f"epoch {epoch} | cost {cost:.2e} | time per step {time_per_step_epoch_mean:.2e} s")
            elif verbose == 1:
                if (epoch + 1) % checkpoints == 0:
                    print(f"epoch {epoch} | cost {cost:.1e}")

            if np.isnan(cost):
                print("Model exploded and returns NaN")
                break

            if cost < (1 - tolerance) * best_loss:
                best_loss = cost
                patience = patience
            else:
                patience -= 1

            if (time.time() - global_start) > max_time * 3600:
                out_of_time = True

            if save_checkpoint:
                if (epoch + 1) % checkpoints == 0 or patience == 0 or epoch == epochs - 1 or out_of_time:
                    latest_checkpoint += 1
                    with open(os.path.join(checkpoints_directory, "score_sheet.txt"), mode="a") as f:
                        f.write(f"{latest_checkpoint} {cost}\n")
                    with ema.average_parameters():
                        torch.save(self.model.state_dict(), os.path.join(checkpoints_directory, f"checkpoint_{cost:.4e}_{latest_checkpoint:03d}.pt"))
                    torch.save(optimizer.state_dict(), os.path.join(checkpoints_directory, f"optimizer_{cost:.4e}_{latest_checkpoint:03d}.pt"))
                    paths = glob.glob(os.path.join(checkpoints_directory, "*.pt"))
                    checkpoint_indices = [int(re.findall('[0-9]+', os.path.split(path)[-1])[-1]) for path in paths]
                    scores = [float(re.findall('([0-9]{1}.[0-9]+e[+-][0-9]{2})', os.path.split(path)[-1])[-1]) for path in paths]
                    if len(checkpoint_indices) > 2*models_to_keep: # has to be twice since we also save optimizer states
                        index_to_delete = np.argmin(checkpoint_indices)
                        os.remove(os.path.join(checkpoints_directory, f"checkpoint_{scores[index_to_delete]:.4e}_{checkpoint_indices[index_to_delete]:03d}.pt"))
                        os.remove(os.path.join(checkpoints_directory, f"optimizer_{scores[index_to_delete]:.4e}_{checkpoint_indices[index_to_delete]:03d}.pt"))
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
