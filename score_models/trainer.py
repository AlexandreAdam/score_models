from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm
import time
import json
import numpy as np
import torch
import os
from glob import glob
import re
from scope.definitions import DEVICE
from torch_ema import ExponentialMovingAverage


def train_score_model(
        model,
        dataset: DataLoader,
        epochs: int,
        max_time: float = np.inf,
        learning_rate: float = 2e-4,
        ema_decay: float = 0.9999,
        epsilon: float = 1e-5,
        warmup: int = 0,
        clip: float = 0,
        augment: bool = False,
        crop: bool = False,
        save_checkpoint_every: int = 10,
        checkpoints_directory: str = None,
        checkpoints_to_keep: int = 1,
        restart_from_checkpoint="last",
        seed=None
):
    """

    :param model: Either a DDPM model or NCSNpp.
    :param dataset: A DataLoader.
    :param epochs: The number of times to loop the dataset.
    :param max_time: The maximum amount of time (in hour) to train. Default is infinite.
    :param learning_rate: Adam learning rate.
    :param ema_decay: Exponential moving average decay for the saved weights.
    :param epsilon: SDE score can diverge when t vanishes, so we sample t from [epsilon, T] instead.
    :param warmup: Warmup the learning up to the target learning rate over this amount of iterations.
    :param clip: Gradient norm clipping. Default is no clipping.
    :param augment: Apply random rotations to images in the dataset. Default is no augmentation.
    :param crop: Apply cropping to the images in the dataset to match the hyperparameter "image size" of the model. Default is no cropping.
    :param save_checkpoint_every: Save a checkpoint of the models each {%} epoch.
    :param checkpoints_directory: Path to the directory where to save models checkpoints. If None, no checkpoints are saved.
    :param checkpoints_to_keep: Only keep {%} best models, on top of the last checkpoint.
    :param restart_from_checkpoint: If checkpoints_directory has some previous checkpoints, either restart from last checkpoint or {%} checkpoint.
    :param seed: Seed of the random generators of numpy and torch.
    :return: EMA context, history
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)
    augment = [T.RandomVerticalFlip(), T.RandomHorizontalFlip()] if augment else []
    crop = [T.CenterCrop(size=model.hyperameters["image_size"])] if crop else []
    data_augmentation = T.Compose(augment + crop)

    def loss_fn(x):
        B, *D = x.shape
        broadcast = [B, *[1] * len(D)]
        mu = torch.randn_like(x)
        t = torch.rand(B).to(DEVICE) * (model.sde.T - epsilon) + epsilon
        mean, sigma = model.sde.marginal_prob(x, t)
        sigma_ = sigma.view(*broadcast)
        return torch.sum((mu + model(mean + sigma_ * mu, t)) ** 2) / B

    # ===== Make sure directory and checkpoint manager are created to save model ===================================
    if checkpoints_directory is not None:
        if not os.path.isdir(checkpoints_directory):
            os.mkdir(checkpoints_directory)
        with open(os.path.join(checkpoints_directory, "model_hparams.json"), "w") as f:
            json.dump(model.hyperparameters, f, indent=4)
        save_checkpoint = True
        # ======= Load model if model_id is provided ===============================================================
        paths = glob(os.path.join(checkpoints_directory, "*.pt"))
        checkpoints = [int(re.findall('[0-9]+', os.path.split(path)[-1])[-1]) for path in paths]
        scores = [float(re.findall('([0-9]{1}.[0-9]+e[+-][0-9]{2})', os.path.split(path)[-1])[-1]) for path in paths]
        if checkpoints != []:
            if isinstance(restart_from_checkpoint, int):
                model.load_state_dict(torch.load(paths[checkpoints == restart_from_checkpoint], map_location=DEVICE))
                print(f"Loaded checkpoint {restart_from_checkpoint}")
                latest_checkpoint = restart_from_checkpoint
            elif restart_from_checkpoint == "last":
                model.load_state_dict(torch.load(paths[np.argmax(checkpoints)], map_location=DEVICE))
                print(f"Loaded last checkpoint ({max(checkpoints)})")
                latest_checkpoint = max(checkpoints)
        else:
            latest_checkpoint = 0
    else:
        save_checkpoint = False

    # ====== Training loop ============================================================================================
    history = {  # recorded at the end of an epoch only
        "cost": [],
        "learning_rate": [],
        "time_per_step": [],
        "step": [],
        "wall_time": []
    }
    step = 0
    global_start = time.time()
    estimated_time_for_epoch = 0
    out_of_time = False
    for epoch in tqdm(range(epochs)):
        if (time.time() - global_start) > max_time * 3600 - estimated_time_for_epoch:
            break
        epoch_start = time.time()
        time_per_step_epoch_mean = 0
        cost = 0
        for batch, x in enumerate(dataset):
            start = time.time()
            # preprocessing
            x = data_augmentation(x)
            # optimize network
            optimizer.zero_grad()
            loss = loss_fn(x)
            loss.backward()
            # warmup learning rate
            if step <= warmup:
                for g in optimizer.param_groups:
                    g['lr'] = learning_rate * np.minimum(step / warmup, 1.0)
            # gradient clipping
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
            optimizer.step()
            ema.update()
            # ========== Summary and logs ==============================================================================
            _time = time.time() - start
            time_per_step_epoch_mean += _time
            cost += float(loss)
            step += 1

        time_per_step_epoch_mean /= len(dataset)
        cost /= len(dataset)
        print(f"epoch {epoch} | cost {cost:.3e} | time per step {time_per_step_epoch_mean:.2e} s")
        history["cost"].append(cost)
        history["learning_rate"].append(optimizer.param_groups[0]['lr'])
        history["time_per_step"].append(time_per_step_epoch_mean)
        history["step"].append(step)
        history["wall_time"].append(time.time() - global_start)

        if np.isnan(cost):
            print("Training is unstable")
            break
        if (time.time() - global_start) > max_time * 3600:
            out_of_time = True
        if save_checkpoint:
            if epoch % save_checkpoint_every == 0 or epoch == epochs - 1 or out_of_time:
                latest_checkpoint += 1
                with open(os.path.join(checkpoints_directory, "score_sheet.txt"), mode="a") as f:
                    f.write(f"{latest_checkpoint} {cost}\n")
                with ema.average_parameters():  # save EMA parameters
                    torch.save(model.state_dict(), os.path.join(checkpoints_directory, f"checkpoint_{cost:.4e}_{latest_checkpoint:03d}.pt"))
                checkpoints.append(latest_checkpoint)
                scores.append(cost)
                print("Saved checkpoint for step {}".format(step))
            if len(checkpoints) >= checkpoints_to_keep + 1:
                # remove the worst score checkpoint (excluding the one we just saved)
                index_to_delete = np.argmax(scores[:-1])
                os.remove(os.path.join(checkpoints_directory, f"checkpoint_{scores[index_to_delete]:.4e}_{checkpoints[index_to_delete]:03d}.pt"))
                del scores[index_to_delete]
                del checkpoints[index_to_delete]
        if out_of_time:
            break
        if epoch > 0:  # First epoch is always very slow and not a good estimate of an epoch time.
            estimated_time_for_epoch = time.time() - epoch_start
    print(f"Finished training after {(time.time() - global_start) / 3600:.3f} hours.")
    return ema, history
