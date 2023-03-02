from score_models import DDPM, NCSNpp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from datetime import datetime
from tqdm import tqdm
import time
import json
import numpy as np
import torch
import os
from glob import glob
import re
from score_models.definitions import DEVICE, DTYPE
from torch_ema import ExponentialMovingAverage
import h5py
#TODO support VPSDE


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path_to_h5, key, device=DEVICE):
        self.filepath = path_to_h5
        self.device = device
        self.key = key
        with h5py.File(self.filepath, "r") as hf:
            self.size = hf[self.key].shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        with h5py.File(self.filepath, "r") as hf:
            return torch.tensor(hf[self.key][index][None], dtype=DTYPE).to(self.device)


def main(args):
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    if args.json_override is not None:
        if isinstance(args.json_override, list):
            files = args.json_override
        else:
            files = [args.json_override,]
        for file in files:
            with open(file, "r") as f:
                json_override = json.load(f)
            args_dict = vars(args)
            args_dict.update(json_override)

    with open(args.model_parameters, "r") as f:
        hyperparameters = json.load(f)
    if args.model_architecture.lower() == "ddpm":
        model = torch.nn.DataParallel(DDPM(**hyperparameters).to(DEVICE), device_ids=list(range(torch.cuda.device_count())))
    elif args.model_architecture.lower() == "ncsnpp":
        model = torch.nn.DataParallel(NCSNpp(**hyperparameters).to(DEVICE), device_ids=list(range(torch.cuda.device_count())))
    else:
        raise ValueError
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay)
    dataset = Dataset(args.dataset_path, args.dataset_key, device=DEVICE)
    dataset = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    augment = [T.RandomVerticalFlip(), T.RandomHorizontalFlip()] if args.flip else []
    data_augmentation = T.Compose(augment + [T.CenterCrop(size=hyperparameters["image_size"])])

    def loss_fn(x):
        B, *D = x.shape
        broadcast = [B, *[1] * len(D)]
        mu = torch.randn_like(x)
        t = torch.rand(B).to(DEVICE) * (model.sde.T - args.epsilon) + args.epsilon
        mean, sigma = model.sde.marginal_prob(x, t)
        sigma_ = sigma.view(*broadcast)
        return torch.sum((mu + model(mean + sigma_ * mu, t)) ** 2) / B

    # ==== Take care of where to write logs and stuff =================================================================
    if args.model_id.lower() != "none":
        logname = args.model_id
    elif args.logname is not None:
        logname = args.logname
    else:
        logname = args.logname_prefixe + "_" + datetime.now().strftime("%y%m%d%H%M%S")
    if args.logdir.lower() != "none":
        logdir = os.path.join(args.logdir, logname)
        if not os.path.isdir(logdir):
            os.mkdir(logdir)
        writer = SummaryWriter(log_dir=logdir)
    else:
        writer = SummaryWriter()
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
    history = {  # recorded at the end of an epoch only
        "cost": [],
        "learning_rate": [],
        "time_per_step": [],
        "step": [],
        "wall_time": []
    }
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
        for batch, x in enumerate(dataset):
            start = time.time()
            # preprocessing
            x = data_augmentation(x)
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
            if step % args.ema_log_freq == 0:
                with ema.average_parameters():
                    with torch.no_grad():
                        loss = loss_fn(x)
                        writer.add_scalar("EMA MSE", float(loss), step)
            step += 1

        time_per_step_epoch_mean /= len(dataset)
        cost /= len(dataset)
        writer.add_scalar("MSE", cost, step)
        print(f"epoch {epoch} | cost {cost:.3e} "
              f"| time per step {time_per_step_epoch_mean:.2e} s")
        history["cost"].append(cost)
        history["learning_rate"].append(optimizer.param_groups[0]['lr'])
        history["time_per_step"].append(time_per_step_epoch_mean)
        history["step"].append(step)
        history["wall_time"].append(time.time() - global_start)

        writer.flush()
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
    return history, best_loss


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model_architecture",     required=True,                  help="Either 'ddpm' or 'ncsnpp'")
    parser.add_argument("--dataset_path",           required=True)
    parser.add_argument("--dataset_key",            required=True)
    parser.add_argument("--model_id",               default="none",                     help="The script will search in provided model_dir argument for model_id and load checkpoint if it exists.")
    parser.add_argument("--model_checkpoint",       default=None,       type=int,       help="Index of the checkpoint to load.")

    # Model parameters
    parser.add_argument("--model_parameters",               required=True,                  help="Path to model parameter json file.")

    # Optimization params
    parser.add_argument("--epochs",                         default=10,     type=int,       help="Number of epochs for training.")
    parser.add_argument("--learning_rate",                  default=2e-4,   type=float,     help="Initial learning rate.")
    parser.add_argument("--patience",                       default=np.inf, type=int,       help="Number of step at which training is stopped if no improvement is recorder.")
    parser.add_argument("--tolerance",                      default=0,      type=float,     help="Current score <= (1 - tolerance) * best score => reset patience, else reduce patience.")
    parser.add_argument("--max_time",                       default=np.inf, type=float,     help="Time allowed for the training, in hours.")
    parser.add_argument("--ema_decay",                      default=0.9999, type=float)
    parser.add_argument("--epsilon",                        default=1e-5,   type=float,     help="SDE score diverges when t vanishes, so we sample t from [epsilon, T] instead")
    parser.add_argument("--warmup",                         default=5000,   type=int,       help="Warmup the learning up to the target learning rate over this amount of iterations")
    parser.add_argument("--clip",                           default=0.,     type=float,     help="Gradient clipping")

    # Training set params
    parser.add_argument("--batch_size",             default=1,      type=int,       help="Number of images in a batch.")
    parser.add_argument("--flip",                   action="store_true",            help="Data augmention, add horizontal and vertical flipx")

    # logs
    parser.add_argument("--logdir",             default="None",                     help="Path of logs directory. Default if None, no logs recorded.")
    parser.add_argument("--logname",            default=None,                       help="OEerwrite name of the log with this argument")
    parser.add_argument("--logname_prefixe",    default="DDPM",                     help="If name of the log is not provided, this prefix is prepended to the date")
    parser.add_argument("--model_dir",          default="None",                     help="Path to the directory where to save models checkpoints.")
    parser.add_argument("--checkpoints",        default=10, type=int,               help="Save a checkpoint of the models each {%} epoch.")
    parser.add_argument("--models_to_keep",     default=10,  type=int,               help="Only keep 3 best model, on top of the last checkpoint")
    parser.add_argument("--ema_log_freq",       default=1000,                        help="Log the ema loss function every x step")

    # Reproducibility params
    parser.add_argument("--seed",                   default=None,   type=int,       help="Random seed for numpy and tensorflow.")
    parser.add_argument("--json_override",          default=None,   nargs="+",      help="A json filepath that will override every command line parameters. Useful for reproducibility")

    args = parser.parse_args()
    main(args)
