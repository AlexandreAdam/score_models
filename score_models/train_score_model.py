from score_models import ScoreModel, EnergyModel
from torch.utils.data import DataLoader
from datetime import datetime
from torch.nn import DataParallel
from tqdm import tqdm
import time
import json
import numpy as np
import torch
import os
from glob import glob
import re
from score_models.utils import DEVICE, DTYPE
from torch_ema import ExponentialMovingAverage
from argparse import ArgumentParser
import h5py

parser = ArgumentParser()
parser.add_argument("--model_architecture",     default="ncsnpp",               help="Either 'ddpm' or 'ncsnpp'")
parser.add_argument("--model_type",             default="score",                help="Either 'score' or 'energy', in both cases the model is trained with the DSM loss")
parser.add_argument("--dimensions",             default=2,      type=int,       help="Spatial dimensions (either 1, 2 or 3")
parser.add_argument("--dataset_path",           default=None)
parser.add_argument("--dataset_extension",      default=None,                   help="Type of dataset, either `h5` or `npy` file")
parser.add_argument("--dataset_key",            default=None,                   help="Key to the dataset in the h5 file if provided.")
parser.add_argument("--model_id",               default="none",                 help="The script will search in provided model_dir argument for model_id and load checkpoint if it exists.")
parser.add_argument("--model_checkpoint",       default=None,   type=int,       help="Index of the checkpoint to load.")

# Model parameters
parser.add_argument("--model_parameters",       default=None,                   help="Path to model parameter json file.")

# Optimization params
parser.add_argument("--epochs",                 default=100,    type=int,       help="Number of epochs for training.")
parser.add_argument("--learning_rate",          default=2e-4,   type=float,     help="Initial learning rate.")
parser.add_argument("--patience",               default=np.inf, type=int,       help="Number of step at which training is stopped if no improvement is recorder.")
parser.add_argument("--tolerance",              default=0,      type=float,     help="Current score <= (1 - tolerance) * best score => reset patience, else reduce patience.")
parser.add_argument("--max_time",               default=np.inf, type=float,     help="Time allowed for the training, in hours.")
parser.add_argument("--ema_decay",              default=0.9999, type=float)
parser.add_argument("--epsilon",                default=1e-5,   type=float,     help="VPSDE score diverges when t vanishes, so we sample t from [epsilon, T] instead")
parser.add_argument("--warmup",                 default=0,      type=int,       help="Warmup the learning up to the target learning rate over this amount of iterations")
parser.add_argument("--clip",                   default=0.,     type=float,     help="Gradient clipping")

# Training set params
parser.add_argument("--batch_size",             default=1,      type=int,       help="Number of images in a batch.")

# logs
parser.add_argument("--logname",                default=None,                   help="Overwrite name of the log with this argument")
parser.add_argument("--logname_prefixe",        default="score_model",          help="If name of the log is not provided, this prefix is prepended to the date")
parser.add_argument("--model_dir",              default="None",                 help="Path to the directory where to save models checkpoints.")
parser.add_argument("--checkpoints",            default=10,     type=int,       help="Save a checkpoint of the models each {%} epoch.")
parser.add_argument("--models_to_keep",         default=3,      type=int,       help="Only keep 3 best model, on top of the last checkpoint")

# Reproducibility params
parser.add_argument("--seed",                   default=None,   type=int,       help="Random seed for numpy and torch.")



class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, key, extension="h5", device=DEVICE):
        self.device = device
        self.key = key
        if extension in ["h5", "hdf5"]:
            self._data = h5py.File(self.filepath, "r")
            self.data = lambda index: self._data[self.key][index]
            self.size = self._data[self.key].shape[0]
        elif extension == "npy":
            self.data = np.load(path)
            self.size = self.data.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return torch.tensor(self.data[[index]], dtype=DTYPE).to(self.device)


def train_score_model(preprocessing_fn=None, **kwargs):
    args = parser.parse_args()
    args_dict = vars(args)
    args_dict.update(kwargs)
    if preprocessing_fn is not None:
        args.dict.update({"preprocessing": preprocessing_fn.__name__})
    else:
        preprocessing_fn = lambda x: x
        args.dict.update({"preprocessing": None})
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    if "hyperparameters" in kwargs.keys():
        args_dict.update({"hyperparameters": kwargs["hyperparameters"]})
    if isinstance(args.model_parameters, str):
        with open(args.model_parameters, "r") as f:
            hyperparameters = json.load(f)
        args_dict["hyperparameters"].update(hyperparameters)
    
    if args.model_type == "score":
        model = DataParallel(ScoreModel(checkpoint_directory=args.model_checkpoint, hyperparameters=args.hyperparameters).to(DEVICE),
                             device_ids=list(range(torch.cuda.device_count())))
    elif args.model_type == "energy":
        model = DataParallel(EnergyModel(checkpoint_directory=args.model_checkpoint, hyperparameters=args.hyperparameters).to(DEVICE)
                             device_ids=list(range(torch.cuda.device_count())))
    else:
        raise ValueError(f"{args.model_type} should be either 'score' or 'energy'")
    sde = model.module.sde
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay)
    if args.dataset_extension is None:
        raise ValueError("Specify dataset_extension")
    dataset = Dataset(path=args.dataset_path, key=args.dataset_key, extension=args.dataset_extension, device=DEVICE)
    dataset = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    def loss_fn(x):
        B, *D = x.shape
        broadcast = [B, *[1] * len(D)]
        z = torch.randn_like(x)
        t = torch.rand(B).to(DEVICE) * (sde.T - args.epsilon) + args.epsilon
        mean, sigma = sde.marginal_prob(t=t, x=x)
        sigma_ = sigma.view(*broadcast)
        return torch.sum((z + model(t=t, x=mean + sigma_ * z)) ** 2) / B

    # ==== Take care of where to write checkpoints and stuff =================================================================
    if args.model_id.lower() != "none":
        logname = args.model_id
    elif args.logname is not None:
        logname = args.logname
    else:
        logname = args.logname_prefixe + "_" + datetime.now().strftime("%y%m%d%H%M%S")
    # ===== Make sure directory and checkpoint manager are created to save model ===================================
    if args.model_dir.lower() != "none":
        checkpoints_dir = os.path.join(args.model_dir, logname)
        if not os.path.isdir(checkpoints_dir):
            os.mkdir(checkpoints_dir)
            with open(os.path.join(checkpoints_dir, "script_params.json"), "w") as f:
                json.dump(vars(args), f, indent=4)
            with open(os.path.join(checkpoints_dir, "model_hparams.json"), "w") as f:
                json.dump(hyperparameters, f, indent=4)
        save_checkpoint = True
        # ======= Load model if model_id is provided ===============================================================
        paths = glob(os.path.join(checkpoints_dir, "*.pt"))
        checkpoints = [int(re.findall('[0-9]+', os.path.split(path)[-1])[-1]) for path in paths]
        scores = [float(re.findall('([0-9]{1}.[0-9]+e[+-][0-9]{2})', os.path.split(path)[-1])[-1]) for path in paths]
        if args.model_id.lower() != "none" and checkpoints != []:
            if args.model_checkpoint is not None:
                model.module.load_state_dict(torch.load(paths[checkpoints == args.model_checkpoint], map_location=DEVICE))
                print(f"Loaded checkpoint {args.model_checkpoint} of {args.model_id}")
                lastest_checkpoint = args.model_checkpoint
            else:
                model.module.load_state_dict(torch.load(paths[np.argmax(checkpoints)], map_location=DEVICE))
                print(f"Loaded checkpoint {max(checkpoints)} of {args.model_id}")
                lastest_checkpoint = max(checkpoints)
        else:
            lastest_checkpoint = 0
    else:
        save_checkpoint = False

    # ====== Training loop ============================================================================================
    best_loss = np.inf
    patience = args.patience
    step = 0
    global_start = time.time()
    estimated_time_for_epoch = 0
    out_of_time = False
    for epoch in tqdm(range(args.epochs)):
        if (time.time() - global_start) > args.max_time * 3600 - estimated_time_for_epoch:
            break
        epoch_start = time.time()
        time_per_step_epoch_mean = 0
        cost = 0
        for _ in range(args.epoch_iterations):
            start = time.time()
            try:
                x = next(data_iter)
            except StopIteration:
                # End of dataset reached, reset the iterator
                data_iter = iter(dataloader)
                x = next(data_iter)
            # preprocessing
            x = preprocessing_fn(x)
            # optimize network
            optimizer.zero_grad()
            loss = loss_fn(x)
            loss.backward()
            # warmup learning rate
            if step <= args.warmup:
                for g in optimizer.param_groups:
                    g['lr'] = args.learning_rate * np.minimum(step / args.warmup, 1.0)
            # gradient clipping
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
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

        if np.isnan(cost):
            print("Training broke the Universe")
            break
        if cost < (1 - args.tolerance) * best_loss:
            best_loss = cost
            patience = args.patience
        else:
            patience -= 1
        if (time.time() - global_start) > args.max_time * 3600:
            out_of_time = True
        if save_checkpoint:
            if epoch % args.checkpoints == 0 or patience == 0 or epoch == args.epochs - 1 or out_of_time:
                lastest_checkpoint += 1
                with open(os.path.join(checkpoints_dir, "score_sheet.txt"), mode="a") as f:
                    f.write(f"{lastest_checkpoint} {cost}\n")
                with ema.average_parameters():  # save EMA parameters
                    # Use model.module to get the state dict from within the data parallel module
                    torch.save(model.module.state_dict(), os.path.join(checkpoints_dir, f"checkpoint_{cost:.4e}_{lastest_checkpoint:03d}.pt"))
                checkpoints.append(lastest_checkpoint)
                scores.append(cost)
                print("Saved checkpoint for step {}".format(step))
            if len(checkpoints) >= args.models_to_keep + 1:
                # remove the worst score checkpoint (excluding the one we just saved)
                index_to_delete = np.argmax(scores[:-1])
                os.remove(os.path.join(checkpoints_dir, f"checkpoint_{scores[index_to_delete]:.4e}_{checkpoints[index_to_delete]:03d}.pt"))
                del scores[index_to_delete]
                del checkpoints[index_to_delete]
        if patience == 0:
            print("Reached patience")
            break
        if out_of_time:
            break
        if epoch > 0:  # First epoch is always very slow and not a good estimate of an epoch time.
            estimated_time_for_epoch = time.time() - epoch_start
    print(f"Finished training after {(time.time() - global_start) / 3600:.3f} hours.")
    return model


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
