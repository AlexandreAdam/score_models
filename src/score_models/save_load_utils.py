from typing import Optional, Literal, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from score_models import ScoreModel, LoRAScoreModel, SDE

import torch
import os, glob, re, json
import numpy as np
import warnings
import copy
import shutil
import dill
import gzip
import h5py
import hashlib
from datetime import datetime
from torch.nn import Module
from peft import PeftModel

from .utils import DEVICE


def checkpoint_number(path: str) -> int:
    return int(re.findall(r'[0-9]+', path)[-1])


def maybe_raise_error(message: str, throw_error: bool = True, error_type=FileNotFoundError):
    if throw_error:
        raise error_type(message)
    else:
        warnings.warn(message)


def last_checkpoint(path: str) -> int:
    if os.path.isdir(path):
        paths = sorted(glob.glob(os.path.join(path, "*checkpoint*")), key=checkpoint_number)
        if len(paths) > 0:
            return checkpoint_number(paths[-1])
        else:
            return 0
    else:
        return 0


def next_checkpoint(path: str) -> int:
    return last_checkpoint(path) + 1


def save_checkpoint(
        model: Module,
        path: str,
        create_path: bool = True,
        key: Literal["checkpoint", "optimizer", "lora_checkpoint"] = "checkpoint"
        ):
    """
    Utility function to save checkpoints of a model and its optimizer state. 
    This utility will save files in path with the following pattern
    ```
        Path
        ├── checkpoint_001.pt
        ├── checkpoint_002.pt
        ├── ...
        ├── optimizer_001.pt
        ├── optimizer_002.pt
        ├── ...
    ```
    
    Args:
        model: Model instance to save.
        path: Path to a directory where to save the checkpoint files. Defaults to the path in the ScoreModel instance.
        create_path: If True, create the directory if it does not exist.
        key: Key to save the checkpoint with. Defaults to "checkpoint". Alternative is "optimizer".
    """
    if not os.path.isdir(path):
        if create_path:
            os.makedirs(path, exist_ok=True)
        else:
            raise FileNotFoundError(f"Directory {os.path.dirname(path)} does not exist")

    checkpoint = next_checkpoint(path)
    if key == "lora_checkpoint":
        model.save_pretrained(os.path.join(path, f"{key}_{checkpoint:03d}"))
    else:
        torch.save(model.state_dict(), os.path.join(path, f"{key}_{checkpoint:03d}.pt"))
    print(f"Saved {key} {checkpoint} to {path}")
 

def save_hyperparameters(hyperparameters: dict, path: str, key: str = "model_hparams"):
    """
    Utility function to save the hyperparameters of a model to a standard file.
    """
    file = os.path.join(path, f"{key}.json")
    if not os.path.isfile(file):
        with open(file, "w") as f:
            json.dump(hyperparameters, f, indent=4)
        print(f"Saved {key} to {path}")


def load_hyperparameters(path: str, key: str = "model_hparams") -> dict:
    """
    Utility function to load the hyperparameters of a model from a standard file.
    """
    file = os.path.join(path, f"{key}.json")
    if os.path.isfile(file):
        with open(file, "r") as f:
            hparams = json.load(f)
        return hparams
    else:
        raise FileNotFoundError(f"Could not find hyperparameters in {path}.")


def remove_oldest_checkpoint(path: str, models_to_keep: int = 5):
    """
    Utility function to clean up old checkpoints in a directory.
    This utility will delete the oldest checkpoints and their optimizer states.
    """
    if models_to_keep:
        paths = sorted(glob.glob(os.path.join(path, "*checkpoint*")), key=checkpoint_number)
        checkpoints = [checkpoint_number(os.path.split(path)[-1]) for path in paths]
        if len(paths) > models_to_keep:
            # Clean up oldest models
            path_to_remove = paths[0]
            if os.path.isfile(path_to_remove):
                os.remove(path_to_remove)
            # Handle case (e.g. LoRA) where checkpoint is a directory
            elif os.path.isdir(path_to_remove):
                shutil.rmtree(path_to_remove)
            # remove associated optimizer
            opt_path = os.path.join(path, "optimizer_{:03d}.pt".format(checkpoints[0]))
            if os.path.exists(opt_path):
                os.remove(opt_path)
            # # remove associated scalar net
            # scalar_path = os.path.join(path, "scalar_net_{:03d}.pt".format(checkpoints[0]))
            # if os.path.exists(scalar_path):
                # os.remove(scalar_path)

def load_sbm_state(sbm: "ScoreModel", path: str, device=DEVICE):
    """
    Utility function to load the state dictionary of a model from a file.
    We use a try except to catch an old error in the model saving process.
    """
    try:
        sbm.net.load_state_dict(torch.load(path, map_location=sbm.device, weights_only=True))
    except (KeyError, RuntimeError) as e:
        # Maybe the ScoreModel instance was used when saving the weights... (mostly backward compatibility with old bugs)
        try:
            sbm.load_state_dict(torch.load(path, map_location=sbm.device, weights_only=True))
        except (KeyError, RuntimeError):
            print(e)
            raise KeyError(f"Could not load state of model from {path}. Make sure you are loading the correct model.")

def load_optimizer_state(optimizer: torch.optim.Optimizer, path: str, raise_error: bool = True, device=DEVICE):
    try:
        optimizer.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    except (KeyError, RuntimeError) as e:
        if raise_error:
            print(e)
        maybe_raise_error(f"Could not load state of the optimizer from {path}.", raise_error, error_type=KeyError)

def load_lora_state(lora_sbm: "LoRAScoreModel", path: str, device=DEVICE):
    lora_sbm.lora_net = PeftModel.from_pretrained(copy.deepcopy(lora_sbm.net), path, is_trainable=True)

# def load_scalar_net(posterior_sbm: "LoRAPosteriorScoreModel", path: str):
    # posterior_sbm.scalar_net.load_state_dict(torch.load(path, map_location=posterior_sbm.device))
        
def load_checkpoint(
        model: Module,
        path: str,
        checkpoint: Optional[int] = None,
        raise_error: bool = True,
        key: Literal["checkpoint", "optimizer", "lora_checkpoint"] = "checkpoint",
        device=DEVICE
        ):
    """
    Utility function to load the checkpoint of a model and its optimizer state. 
    This utility assumes the directory contains files with the following pattern:
    ```
        Path
        ├── *checkpoint_*_001.pt
        ├── *checkpoint_*_002.pt
        ├── ...
        ├── optimizer_*_001.pt
        ├── optimizer_*_002.pt
        ├── ...
    ```
    
    Args:
        checkpoint: Checkpoint number to load. If None, the last checkpoint is loaded.
        path: Path to load the checkpoint files from. Defaults to the path in the ScoreModelBase instance.
        raise_error: If True, raise an error if no checkpoints are found in the directory.
    """
    if not os.path.isdir(path):
        if raise_error:
            raise FileNotFoundError(f"Directory {path} does not exist.")
        else: # If no directory is found, don't do anything. This is useful for initialization of Base.
            return
    name = os.path.split(path)[-1]
    # Collect all checkpoint paths sorted by the checkpoint number (*_001.pt, *_002.pt, ...)
    paths = sorted(glob.glob(os.path.join(path, f"{key}*")), key=checkpoint_number)
    checkpoints = [checkpoint_number(os.path.split(path)[-1]) for path in paths]
    if checkpoint and checkpoint not in checkpoints:
        # Make sure the requested checkpoint exists
        maybe_raise_error(f"{key} {checkpoint} not found in directory {path}.", raise_error)
        checkpoint = None # Overwrite to load the last checkpoint
    
    # Refactor to use setattr for more generality or just returns the net.
    if key == "checkpoint":
        loading_mecanism = load_sbm_state
    elif key == "lora_checkpoint":
        loading_mecanism = load_lora_state
    elif key == "optimizer":
        loading_mecanism = load_optimizer_state
    # elif key == "scalar_net":
        # loading_mecanism = load_scalar_net
    else:
        raise ValueError(f"Key {key} not recognized.")
    
    if checkpoints:
        if checkpoint:
            # Load requested checkpoint
            index = checkpoints.index(checkpoint)
        else:
            # Load last checkpoint
            index = np.argmax(checkpoints)
            checkpoint = checkpoints[index]
        loading_mecanism(model, paths[index], device=device)
        print(f"Loaded {key} {checkpoint} from {name}.")
        return checkpoint
    else:
        maybe_raise_error(f"No {key} found in {path}")
        return None

def load_architecture(
        path: Optional[str] = None,
        net: Optional[str] = None,
        device=DEVICE,
        hparams_filename="model_hparams",
        **hyperparameters
        ) -> Tuple[Module, dict]:
    """
    Utility function to load a model architecture from a checkpoint directory or 
    a dictionary of hyperparameters. 
    
    Args:
        path (str): Path to the checkpoint directory. If None, the model is loaded from the hyperparameters.
        model (str): Model architecture to load. If provided, hyperparameters are used to instantiate the model.
        device (torch.device): Device to load the model to.
        hyperparameters: hyperparameters to instantiate the model.
    """
    if path:
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Directory {path} does not exist. "
                                     "Please make sure to provide a valid path to the checkpoint directory.")
        hparams = load_hyperparameters(path, key=hparams_filename)
        hyperparameters.update(hparams)
        net = hyperparameters.get("model_architecture", "ncsnpp")
    
    if isinstance(net, str):
        if net.lower() == "ncsnpp":
            from score_models.architectures import NCSNpp
            net = NCSNpp(**hyperparameters).to(device)
        elif net.lower() == "ddpm":
            from score_models.architectures import DDPM
            net = DDPM(**hyperparameters).to(device)
        elif net.lower() == "mlp":
            from score_models import MLP
            net = MLP(**hyperparameters).to(device)
        elif net.lower() == "encoder":
            from score_models import Encoder
            net = Encoder(**hyperparameters).to(device)
        else:
            raise ValueError(f"Architecture {net} not recognized.")
    else:
        raise ValueError(f"A model architecture or a path to a checkpoint directory must be provided.")
 
    # Backward compatibility
    if "model_architecture" not in hyperparameters.keys():
        hyperparameters["model_architecture"] = net.__class__.__name__

    if path:
        print(f"Loaded model architecture {net.__class__.__name__} from {os.path.split(path)[-1]}.")
    else:
        print(f"Loaded model architecture {net.__class__.__name__} from hyperparameters.")
    return net, hyperparameters


def load_sde(sde: Optional[Literal["ve", "vp", "tsve"]] = None, **kwargs) -> Tuple["SDE", dict]:
    if sde is None:
        if "sde" not in kwargs.keys():
            # Some sane defaults for quick use of VE or VP
            if "sigma_min" in kwargs.keys() or "sigma_max" in kwargs.keys():
                print("Using the Variance Exploding SDE")
                sde = "ve"
            elif "beta_max" in kwargs.keys() or "beta_min" in kwargs.keys():
                print("Using the Variance Preserving SDE")
                sde = "vp"
            else:
                raise KeyError("SDE parameters are missing, please specify which sde to use by using e.g. sde='ve' or sde='vp'")
        else:
            # Backward compatibility
            if kwargs["sde"] == "vpsde":
                kwargs["sde"] = "vp"
            elif kwargs["sde"] == "vesde":
                kwargs["sde"] = "ve"
            sde = kwargs["sde"]
    else:
        # Backward compatibility
        if sde == "vpsde":
            sde = "vp"
        elif sde == "vesde":
            sde = "ve"

    if sde.lower() not in ["ve", "vp", "tsve"]:
        raise ValueError(f"The SDE {sde} provided is not recognized. Please use 've', 'vp', or 'tsve'.")

    # Load the SDE from the keyword
    if sde.lower() == "ve":
        if "sigma_min" not in kwargs.keys() or "sigma_max" not in kwargs.keys():
            raise KeyError("Variance Exploding SDE requires sigma_min and sigma_max to be specified.")
        from score_models.sde import VESDE
        sde_hyperparameters = {
                "sigma_min": kwargs.get("sigma_min"),
                "sigma_max": kwargs.get("sigma_max"),
                "T": kwargs.get("T", VESDE.__init__.__defaults__[0])
                }
        sde = VESDE(**sde_hyperparameters)
        
    elif sde.lower() == "vp":
        from score_models.sde import VPSDE
        sde_hyperparameters = {
                "beta_min": kwargs.get("beta_min", VPSDE.__init__.__defaults__[0]),
                "beta_max": kwargs.get("beta_max", VPSDE.__init__.__defaults__[1]),
                "T": kwargs.get("T", VPSDE.__init__.__defaults__[2]),
                "epsilon": kwargs.get("epsilon", VPSDE.__init__.__defaults__[3]),
                "schedule": kwargs.get("schedule", VPSDE.__init__.__defaults__[4])
                }
        sde = VPSDE(**sde_hyperparameters)
    
    elif sde.lower() == "tsve":
        if "sigma_min" not in kwargs.keys() or "sigma_max" not in kwargs.keys():
            raise KeyError("Truncated Scaled Variance Exploding SDE requires sigma_min and sigma_max to be specified.")
        from score_models.sde import TSVESDE
        sde_hyperparameters = {
                "sigma_min": kwargs.get("sigma_min"),
                "sigma_max": kwargs.get("sigma_max"),
                "t_star": kwargs.get("t_star"),
                "beta": kwargs.get("beta"),
                "T": kwargs.get("T", TSVESDE.__init__.__defaults__[0]),
                "epsilon": kwargs.get("epsilon", TSVESDE.__init__.__defaults__[1])
                }
        sde = TSVESDE(**sde_hyperparameters)
    # Making sure the sde name is recorded
    sde_hyperparameters["sde"] = sde.__class__.__name__.lower()
    return sde, sde_hyperparameters


def serialize_object(obj, h5_path, object_name, date_str=None, metadata_path=None):
    # Serialize and compress the object
    serialized_obj = dill.dumps(obj)
    compressed_obj = gzip.compress(serialized_obj)

    # Save compressed object to h5
    with h5py.File(h5_path, 'a') as hf:
        hf.create_dataset(object_name, data=np.void(compressed_obj))

    # Save metadata with checksum of the compressed object
    if metadata_path is not None:
        checksum = hashlib.sha256(compressed_obj).hexdigest()
        metadata = {
            "filename": os.path.basename(h5_path),
            "object_name": object_name,
            "checksum": checksum,
            "creation_time": date_str if date_str is not None else datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
        with open(metadata_path, 'w') as meta_file:
            json.dump(metadata, meta_file)


def deserialize_object(h5_path, dataset_name, metadata_path=None, checksum=None, safe_mode=True):
    # Load compressed serialized object from H5 file
    with h5py.File(h5_path, 'r') as hf:
        compressed_obj = bytes(hf[dataset_name][()])

    # Decompress the object
    decompressed_obj = gzip.decompress(compressed_obj)

    # Verify checksum in safe mode
    if safe_mode:
        expected_checksum = checksum
        if metadata_path is not None:
            with open(metadata_path, 'r') as meta_file:
                metadata = json.load(meta_file)
            expected_checksum = metadata['checksum']

        loaded_checksum = hashlib.sha256(compressed_obj).hexdigest()
        if expected_checksum is None or loaded_checksum != expected_checksum:
            raise ValueError("Checksum does not match. Data may have been tampered with.")

    # Deserialize the object
    return dill.loads(decompressed_obj)

