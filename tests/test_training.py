from torch.utils.data import TensorDataset
from score_models import ScoreModel, EnergyModel, SLIC, HessianDiagonal, LoRAScoreModel, MLP, NCSNpp, DDPM, EDMv2Net
from score_models import PostHocEMA
from functools import partial
import pytest
import torch
import shutil, os
import numpy as np
import glob

    
class Dataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            size, 
            channels, 
            dimensions,
            time_branch_channels: int = 4,
            conditions=None, 
            condition_channels=None,
            condition_embeddings=None,
            batch_size=None,
            **kwargs
            ):
        self.size = size
        self.C = channels
        self.D = dimensions
        self.conditions = conditions
        self.condition_channels = condition_channels
        self.condition_embeddings = condition_embeddings
        self.B = batch_size or 1

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # We use a batch size to simulate the case when batch_size = None and we don't have a dataloader
        x = [torch.randn(self.B, self.C, *self.D),]
        if self.conditions:
            c_idx = 0
            e_idx = 0
            for condition in self.conditions:
                if condition == "time_continuous":
                    c = torch.randn(self.B, 1)
                    x.append(c)
                elif condition == "time_discrete":
                    tokens = self.condition_embeddings[e_idx]
                    c = torch.randint(tokens, (self.B, 1,))
                    x.append(c)
                    e_idx += 1
                elif condition == "time_vector":
                    c = torch.randn(self.B, self.condition_channels[c_idx])
                    x.append(c)
                    c_idx += 1
                elif condition == "input_tensor":
                    c = torch.randn(self.B, self.condition_channels[c_idx], *self.D)
                    x.append(c)
                    c_idx += 1
        return [x_.squeeze(0) for x_ in x] # Remove the batch dimension for dataloader

def assert_checkpoint_was_saved(path, checkpoint, key="checkpoint", number_expected=1):
    if key == "lora_checkpoint":
        pattern = f"{key}*_{checkpoint:03d}"
    else:
        pattern = f"{key}*_{checkpoint:03d}.pt"
    files = glob.glob(os.path.join(path, pattern))
    assert len(files) == number_expected, f"Expected to find {number_expected} checkpoint file(s), found {len(files)}"

def assert_checkpoint_was_cleanedup(path, checkpoint, key="checkpoint"):
    if key == "lora_checkpoint":
        pattern = f"{key}*_{checkpoint:03d}"
    else:
        pattern = f"{key}*_{checkpoint:03d}.pt"
    files = glob.glob(os.path.join(path, pattern))
    assert len(files) == 0, f"Expected to find no checkpoint files, found {len(files)}"


@pytest.mark.parametrize("soft_reset_every", [None, 1])
@pytest.mark.parametrize("ema_decay", [None, 0.999])
@pytest.mark.parametrize("models_to_keep", [1, 2])
@pytest.mark.parametrize("conditions", [
    (None, None, None), 
    (("input_tensor", "time_continuous", "time_vector", "time_discrete"), (15,), (15, 3)),
    ])
@pytest.mark.parametrize("sde", [
    {"sde": "vp"}, 
    {"sde": "ve", "sigma_min": 1e-2, "sigma_max": 1e2}, 
    {"sde": "vp", "schedule": "cosine", "beta_max": 100}
    ])
@pytest.mark.parametrize("Net", [MLP, NCSNpp, DDPM])
@pytest.mark.parametrize("B", [None, 2]) # Make sure we don't create a dataloader if batch_size is None
def test_training_score_model(B, conditions, sde, Net, models_to_keep, ema_decay, soft_reset_every, tmp_path, capsys):
    condition_type, embeddings, channels = conditions
    hp = { # Hyperparameters for the dataset
            "ch_mult": (1, 1),
            "nf": 2,
            "conditions": condition_type,
            "condition_channels": channels,
            "condition_embeddings": embeddings,
            "batch_size": B # If B is None, we use the Dataloader to handle the batch size (which we set to a default below)
            }
    E = 3 # epochs
    C = 3
    N = 4
    D = [] if Net == MLP else [4, 4]
    dataset = Dataset(N, C, dimensions=D, **hp)
    net = Net(C, **hp)
    model = ScoreModel(net, **sde)
    
    path = tmp_path / "test"
    # Fitting method's batch_size argument is basically the reverse of Dataset, since we turn off/on the dataloader
    if B is None:
        B = 2
    elif isinstance(B, int):
        B = None
    losses = model.fit(
            dataset, 
            batch_size=B, 
            epochs=E, 
            path=path, 
            checkpoint_every=1, 
            ema_decay=ema_decay,
            soft_reset_every=soft_reset_every,
            models_to_keep=models_to_keep)

    print(losses)
    assert len(losses) == E, f"Expected {E} losses, got {len(losses)}"
    # Check that some improvement happens
    assert os.path.isfile(os.path.join(path, "hyperparameters.json")), "hyperparameters.json not found"
    assert os.path.isfile(os.path.join(path, "script_params.json")), "script_params.json not found"
    for key in ["checkpoint", "optimizer"]:
        for i in range(E+1-models_to_keep, E+1):
            assert_checkpoint_was_saved(path, i, key)
        for i in range(0, E+1-models_to_keep): # Check that files are cleaned up
            assert_checkpoint_was_cleanedup(path, i, key)
    
    # Test resume from checkpoint
    new_model = ScoreModel(path=path)
    assert new_model.loaded_checkpoint == E, f"Expected loaded_checkpoint to be {E}, got {new_model.loaded_checkpoint}"
    losses = new_model.fit(
            dataset,
            batch_size=B,
            epochs=E,
            checkpoint_every=1,
            ema_decay=ema_decay,
            models_to_keep=models_to_keep
            )
    # Check stdout for the print statement declaring we resumed from a previous checkpoint for the optimizer
    captured = capsys.readouterr()
    print(captured.out)
    assert f"Resumed training from checkpoint {E}." in captured.out
    
    # Check that the new checkpoints are updated correctly
    for key in ["checkpoint", "optimizer"]:
        for i in range(2*E+1-models_to_keep, 2*E+1):
            assert_checkpoint_was_saved(path, i, key)
        for i in range(E, 2*E+1-models_to_keep): # Check that files are cleaned up
            assert_checkpoint_was_cleanedup(path, i, key)

@pytest.mark.parametrize("Net", [MLP, NCSNpp])
@pytest.mark.parametrize("sde", [
    {"sde": "vp"}, 
    {"sde": "ve", "sigma_min": 1e-2, "sigma_max": 1e2}, 
    {"sde": "vp", "schedule": "cosine", "beta_max": 100}
    ])
def test_training_energy_model(sde, Net, tmp_path, capsys):
    hp = {
            "ch_mult": (1, 1),
            "nf": 2,
            }
    E = 2 # epochs
    B = 2
    C = 3
    N = 4
    models_to_keep = 1
    D = [] if Net == MLP else [4, 4]
    dataset = Dataset(N, C, dimensions=D, **hp)
    net = Net(C, **hp)
    model = EnergyModel(net, **sde)
    
    path = tmp_path / "test"
    losses = model.fit(
            dataset, 
            batch_size=B, 
            epochs=E, 
            path=path, 
            checkpoint_every=1, 
            models_to_keep=models_to_keep)

    print(losses)
    assert len(losses) == E, f"Expected {E} losses, got {len(losses)}"
    assert os.path.isfile(os.path.join(path, "hyperparameters.json")), "hyperparameters.json not found"
    assert os.path.isfile(os.path.join(path, "script_params.json")), "script_params.json not found"
    for key in ["checkpoint", "optimizer"]:
        for i in range(E+1-models_to_keep, E+1):
            assert_checkpoint_was_saved(path, i, key)
        for i in range(0, E+1-models_to_keep): # Check that files are cleaned up
            assert_checkpoint_was_cleanedup(path, i, key)

    # Test resume from checkpoint
    new_model = EnergyModel(path=path)
    assert new_model.loaded_checkpoint == E, f"Expected loaded_checkpoint to be {E}, got {new_model.loaded_checkpoint}"
    losses = new_model.fit(
            dataset,
            batch_size=B,
            epochs=E,
            checkpoint_every=1,
            models_to_keep=models_to_keep
            )
    # Check stdout for the print statement declaring we resumed from a previous checkpoint for the optimizer
    captured = capsys.readouterr()
    print(captured.out)
    assert f"Resumed training from checkpoint {E}." in captured.out
    
    # Check that the new checkpoints are updated correctly
    for key in ["checkpoint", "optimizer"]:
        for i in range(2*E+1-models_to_keep, 2*E+1):
            assert_checkpoint_was_saved(path, i, key)
        for i in range(E, 2*E+1-models_to_keep): # Check that files are cleaned up
            assert_checkpoint_was_cleanedup(path, i, key)


@pytest.mark.parametrize("conditions", [
    (None, None, None), # conditions, embeddings, channels
    (("input_tensor", "time_continuous", "time_vector", "time_discrete"), (15,), (15, 3)),
    ])
@pytest.mark.parametrize("loss", ["lu", "meng"])
@pytest.mark.parametrize("sde", [
    {"sde": "vp"}, 
    {"sde": "ve", "sigma_min": 1e-2, "sigma_max": 1e2}, 
    {"sde": "vp", "schedule": "cosine", "beta_max": 100}
    ])
@pytest.mark.parametrize("Net", [MLP, NCSNpp])
def test_training_hessian_diagonal_model(conditions, loss, sde, Net, tmp_path, capsys):
    condition_type, embeddings, channels = conditions
    hp = {
            "ch_mult": (1, 1),
            "nf": 2,
            "conditions": condition_type,
            "condition_channels": channels,
            "condition_embeddings": embeddings,
            }
    E = 3 # epochs
    B = 2
    C = 3
    N = 4
    D = [] if Net == MLP else [4, 4]
    models_to_keep = 1
    dataset = Dataset(N, C, dimensions=D, **hp)
    net = Net(C, **hp)
    base_model = ScoreModel(net, **sde)
    derivative_net = Net(C, **hp)
    derivative_model = HessianDiagonal(base_model, net=derivative_net, loss=loss)
    
    path = tmp_path / "test"
    losses = derivative_model.fit(
            dataset, 
            batch_size=B, 
            epochs=E, 
            path=path, 
            checkpoint_every=1, 
            models_to_keep=models_to_keep)

    print(losses)
    assert len(losses) == E, f"Expected {E} losses, got {len(losses)}"
    # Check that some improvement happens
    assert os.path.isdir(os.path.join(path, "score_model")), "score_model directory not found, the base SBM has not been saved"
    assert os.path.isfile(os.path.join(path, "hyperparameters.json")), "hyperparameters.json not found"
    assert os.path.isfile(os.path.join(path, "script_params.json")), "script_params.json not found"
    for key in ["checkpoint", "optimizer"]:
        for i in range(E+1-models_to_keep, E+1):
            assert_checkpoint_was_saved(path, i, key)
        for i in range(0, E+1-models_to_keep): # Check that files are cleaned up
            assert_checkpoint_was_cleanedup(path, i, key)

    # Test resume from checkpoint
    new_model = HessianDiagonal(path=path)
    assert new_model.loaded_checkpoint == E, f"Expected loaded_checkpoint to be {E}, got {new_model.loaded_checkpoint}"
    losses = new_model.fit(
            dataset,
            batch_size=B,
            epochs=E,
            checkpoint_every=1,
            models_to_keep=models_to_keep
            )
    # Check stdout for the print statement declaring we resumed from a previous checkpoint for the optimizer
    captured = capsys.readouterr()
    print(captured.out)
    assert f"Resumed training from checkpoint {E}." in captured.out
    
    # Check that the new checkpoints are updated correctly
    for key in ["checkpoint", "optimizer"]:
        for i in range(2*E+1-models_to_keep, 2*E+1):
            assert_checkpoint_was_saved(path, i, key)
        for i in range(E, 2*E+1-models_to_keep): # Check that files are cleaned up
            assert_checkpoint_was_cleanedup(path, i, key)

@pytest.mark.parametrize("conditions", [
    (None, None, None), # conditions, embeddings, channels
    (("input_tensor", "time_continuous", "time_vector", "time_discrete"), (15,), (15, 3)),
    ])
@pytest.mark.parametrize("lora_rank", [1, 2])
@pytest.mark.parametrize("sde", [
    {"sde": "vp"}, 
    {"sde": "ve", "sigma_min": 1e-2, "sigma_max": 1e2}, 
    {"sde": "vp", "schedule": "cosine", "beta_max": 100}
    ])
@pytest.mark.parametrize("Net", [MLP, NCSNpp])
def test_training_lora_model(conditions, lora_rank, sde, Net, tmp_path, capsys):
    condition_type, embeddings, channels = conditions
    hp = {
            "ch_mult": (1, 1),
            "nf": 2,
            "conditions": condition_type,
            "condition_channels": channels,
            "condition_embeddings": embeddings,
            }
    E = 3 # epochs
    B = 2
    C = 3
    N = 4
    D = [] if Net == MLP else [4, 4]
    models_to_keep = 1
    dataset = Dataset(N, C, dimensions=D, **hp)
    net = Net(C, **hp)
    base_model = ScoreModel(net, **sde)
    lora_model = LoRAScoreModel(base_model, lora_rank=lora_rank)
    
    path = tmp_path / "test"
    losses = lora_model.fit(dataset, batch_size=B, epochs=E, path=path, checkpoint_every=1, models_to_keep=models_to_keep)

    print(losses)
    assert len(losses) == E, f"Expected {E} losses, got {len(losses)}"
    # Check that some improvement happens
    assert os.path.isdir(os.path.join(path, "base_sbm")), "base_sbm directory not found, the base SBM has not been saved"
    print(os.listdir(os.path.join(path, "base_sbm")))
    assert os.path.isfile(os.path.join(path, "base_sbm", "hyperparameters.json")), "hyperparameters.json not found in base_sbm directory"
    assert os.path.isfile(os.path.join(path, "base_sbm", "checkpoint_001.pt")), "checkpout_001.pt not found in base_sbm directory"
    assert os.path.isfile(os.path.join(path, "hyperparameters.json")), "hyperparameters.json not found"
    assert os.path.isfile(os.path.join(path, "script_params.json")), "script_params.json not found"
    print(os.listdir(path))
    for key in ["lora_checkpoint", "optimizer"]:
        for i in range(E+1-models_to_keep, E+1):
            assert_checkpoint_was_saved(path, i, key)
        for i in range(0, E+1-models_to_keep): # Check that files are cleaned up
            assert_checkpoint_was_cleanedup(path, i, key)

    # Check the network is reloaded correctly
    new_model = LoRAScoreModel(path=path)
    assert new_model.loaded_checkpoint == E, f"Expected loaded_checkpoint to be {E}, got {new_model.loaded_checkpoint}"
    losses = new_model.fit(
            dataset,
            batch_size=B,
            epochs=E,
            checkpoint_every=1,
            models_to_keep=models_to_keep
            )
    # Check stdout for the print statement declaring we resumed from a previous checkpoint for the optimizer
    captured = capsys.readouterr()
    print(captured.out)
    assert f"Resumed training from checkpoint {E}." in captured.out
    
    # Check that the new checkpoints are updated correctly
    print(os.listdir(path))
    for key in ["lora_checkpoint", "optimizer"]:
        for i in range(2*E+1-models_to_keep, 2*E+1):
            assert_checkpoint_was_saved(path, i, key)
        for i in range(E, 2*E+1-models_to_keep): # Check that files are cleaned up
            assert_checkpoint_was_cleanedup(path, i, key)


def test_backward_compatibility_optimizer_state(tmp_path, capsys):
    # First, train a model with custom optimizer target network
    E = 3 # epochs
    B = 2
    C = 3
    N = 4
    D = []
    dataset = Dataset(N, C, dimensions=D)
    net = MLP(C)
    model = ScoreModel(net, "vp")
    
    # Simulate case where optimizer targets the network
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)
    path = tmp_path / "test"
    losses = model.fit(
            dataset, 
            batch_size=B, 
            epochs=E, 
            path=path, 
            checkpoint_every=1, 
            optimizer=optim,
            models_to_keep=1)

    
    # Now we resume training, and check that we managed to load the checkpoint
    new_model = ScoreModel(path=path)
    assert new_model.loaded_checkpoint == E, f"Expected loaded_checkpoint to be {E}, got {new_model.loaded_checkpoint}"
    # Don't provide the optimizer here to simulate the backward compatibility component of loading the optimizer
    losses = new_model.fit(
            dataset,
            batch_size=B,
            epochs=E,
            checkpoint_every=1,
            models_to_keep=1,
            verbose=1
            )
    # Check stdout for the print statement declaring we resumed from a previous checkpoint for the optimizer
    captured = capsys.readouterr()
    print(captured.out)
    assert f"Resumed training from checkpoint {E}." in captured.out
    assert f"Loaded optimizer {E} from test." in captured.out


def test_lr_scheduler_with_edm(tmp_path):
    E = 10
    B = 2
    C = 3
    N = 4
    D = [8, 8]
    dataset = Dataset(N, C, dimensions=D)
    net = EDMv2Net(8, C, ch_mult=(2, 2), nf=8)
    model = ScoreModel(net, "vp", formulation="edm")
    
    losses = model.fit(
            dataset, 
            learning_rate=1e-3,
            learning_rate_decay=10,
            batch_size=B, 
            epochs=E, 
            path=tmp_path, 
            checkpoint_every=1, 
            models_to_keep=1,
            )

    new_model = ScoreModel(path=tmp_path)
    assert new_model.loaded_checkpoint == E, f"Expected loaded_checkpoint to be {E}, got {new_model.loaded_checkpoint}"
    
    from score_models import Trainer
    # # Do not provide the path, it is inferred from the model
    trainer = Trainer(
            model=new_model,
            dataset=dataset,
            learning_rate=1e-3,
            learning_rate_decay=10,
            batch_size=B,
            epochs=E,
            checkpoint_every=1,
            models_to_keep=1
            )
    assert trainer.global_step == E * len(dataset) // B, f"Expected global_step to be {E * len(dataset) // B}, got {trainer.global_step}"
   

def test_training_with_different_ema_lengths_and_posthoc_ema(tmp_path):
    E = 10
    B = 2
    C = 3
    N = 4
    D = [8, 8]
    models_to_keep = 4    

    dataset = Dataset(N, C, dimensions=D)
    net = EDMv2Net(8, C, ch_mult=(2, 2), nf=8)
    model = ScoreModel(net, "vp", formulation="edm")
    
    losses = model.fit(
            dataset,
            learning_rate=1e-3,
            learning_rate_decay=10,
            batch_size=B,
            ema_lengths=(0.05, 0.1),
            epochs=E,
            path=tmp_path,
            checkpoint_every=1,
            models_to_keep=models_to_keep,
            )

    path = tmp_path
    for file in os.listdir(path):
        print(file) 
    for i in range(E+1-models_to_keep, E+1):
        assert_checkpoint_was_saved(path, i, key="checkpoint", number_expected=2)
        assert_checkpoint_was_saved(path, i, key="optimizer", number_expected=1)
    for i in range(0, E+1-models_to_keep): # Check that files are cleaned up
        assert_checkpoint_was_cleanedup(path, i, key="checkpoint")
        assert_checkpoint_was_cleanedup(path, i, key="optimizer")
    
    # Now, some under the hood check that PostHocEMA works
    ema = PostHocEMA(path=tmp_path)
    
    # Make sure the zero model is created correctly
    zero_model = ema.zero_model()
    B = 5
    t = torch.rand(B)
    x = torch.randn(B, C, *D)
    out = zero_model(t, x)
    assert out.shape == x.shape, f"Expected output shape to be {x.shape}, got {out.shape}"
    assert torch.all(out == 0.), f"Expected output to be all zeros, got {out}"
    new_model = ema.synthesize_ema(0.075) # Check that model can be synthesized at a specific ema_length
    out = new_model(t, x)
    assert out.shape == x.shape, f"Expected output shape to be {x.shape}, got {out.shape}"
    assert out.sum() != 0., f"Expected output to be non-zero for synthesized model, got {out}"
    
    # Check that the score model loads properly with ema_length
    new_model = ScoreModel(path=tmp_path, ema_length=0.075)
    
    with pytest.raises(ValueError):
        new_model = ScoreModel(path=tmp_path) # Need to provide ema_length in that context, where more than one were used 
    
    # Does it make sense to keep training from there?? Need to reset the optimizer..
