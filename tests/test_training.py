from torch.utils.data import TensorDataset
from score_models import ScoreModel, EnergyModel, SLIC, HessianDiagonal, LoRAScoreModel, MLP, NCSNpp, DDPM
from functools import partial
import pytest
import torch
import shutil, os
import numpy as np

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
            **kwargs
            ):
        self.size = size
        self.C = channels
        self.D = dimensions
        self.conditions = conditions
        self.condition_channels = condition_channels
        self.condition_embeddings = condition_embeddings

    def __len__(self):
        return self.size

    def __getitem__(self, index):

        x = [torch.randn(self.C, *self.D),]
        if self.conditions:
            c_idx = 0
            e_idx = 0
            for condition in self.conditions:
                if condition == "time_continuous":
                    c = torch.randn(1)
                    x.append(c)
                elif condition == "time_discrete":
                    tokens = self.condition_embeddings[e_idx]
                    c = torch.randint(tokens, (1,))
                    x.append(c)
                    e_idx += 1
                elif condition == "time_vector":
                    c = torch.randn(self.condition_channels[c_idx])
                    x.append(c)
                    c_idx += 1
                elif condition == "input_tensor":
                    c = torch.randn(self.condition_channels[c_idx], *self.D)
                    x.append(c)
                    c_idx += 1
        return x

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
def test_training_score_model(conditions, sde, Net, models_to_keep, tmp_path, capsys):
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
    dataset = Dataset(N, C, dimensions=D, **hp)
    net = Net(C, **hp)
    model = ScoreModel(net, **sde)
    
    path = tmp_path / "test"
    losses = model.fit(dataset, batch_size=B, epochs=E, path=path, checkpoint_every=1, models_to_keep=models_to_keep)

    print(losses)
    assert len(losses) == E, f"Expected {E} losses, got {len(losses)}"
    # Check that some improvement happens
    assert os.path.isfile(os.path.join(path, "model_hparams.json")), "model_hparams.json not found"
    assert os.path.isfile(os.path.join(path, "script_params.json")), "script_params.json not found"
    for i in range(E+1-models_to_keep, E+1):
        assert os.path.isfile(os.path.join(path, f"checkpoint_{i:03}.pt")), f"checkpoint_{i:03}.pt not found"
        assert os.path.isfile(os.path.join(path, f"optimizer_{i:03}.pt")), f"optimizer_{i:03}.pt not found"
    for i in range(0, E+1-models_to_keep): # Check that files are cleaned up
        assert not os.path.exists(os.path.join(path, f"checkpoint_{i:03}.pt")), f"checkpoint_{i:03}.pt found, should not be there"
        assert not os.path.exists(os.path.join(path, f"optimizer_{i:03}.pt")), f"optimizer_{i:03}.pt found, should not be there"
    
    # Test resume from checkpoint
    new_model = ScoreModel(path=path)
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
    for i in range(2*E+1-models_to_keep, 2*E+1):
        assert os.path.isfile(os.path.join(path, f"checkpoint_{i:03}.pt")), f"checkpoint_{i:03}.pt not found"
        assert os.path.isfile(os.path.join(path, f"optimizer_{i:03}.pt")), f"optimizer_{i:03}.pt not found"
    for i in range(E, 2*E+1-models_to_keep): # Check that files are cleaned up
        assert not os.path.exists(os.path.join(path, f"checkpoint_{i:03}.pt")), f"checkpoint_{i:03}.pt found, should not be there"
        assert not os.path.exists(os.path.join(path, f"optimizer_{i:03}.pt")), f"optimizer_{i:03}.pt found, should not be there"


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
    losses = model.fit(dataset, batch_size=B, epochs=E, path=path, checkpoint_every=1, models_to_keep=models_to_keep)

    print(losses)
    assert len(losses) == E, f"Expected {E} losses, got {len(losses)}"
    assert os.path.isfile(os.path.join(path, "model_hparams.json")), "model_hparams.json not found"
    assert os.path.isfile(os.path.join(path, "script_params.json")), "script_params.json not found"
    for i in range(E+1-models_to_keep, E+1):
        assert os.path.isfile(os.path.join(path, f"checkpoint_{i:03}.pt")), f"checkpoint_{i:03}.pt not found"
        assert os.path.isfile(os.path.join(path, f"optimizer_{i:03}.pt")), f"optimizer_{i:03}.pt not found"
    for i in range(0, E+1-models_to_keep): # Check that files are cleaned up
        assert not os.path.exists(os.path.join(path, f"checkpoint_{i:03}.pt")), f"checkpoint_{i:03}.pt found, should not be there"
        assert not os.path.exists(os.path.join(path, f"optimizer_{i:03}.pt")), f"optimizer_{i:03}.pt found, should not be there"

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
    for i in range(2*E+1-models_to_keep, 2*E+1):
        assert os.path.isfile(os.path.join(path, f"checkpoint_{i:03}.pt")), f"checkpoint_{i:03}.pt not found"
        assert os.path.isfile(os.path.join(path, f"optimizer_{i:03}.pt")), f"optimizer_{i:03}.pt not found"
    for i in range(E, 2*E+1-models_to_keep): # Check that files are cleaned up
        assert not os.path.exists(os.path.join(path, f"checkpoint_{i:03}.pt")), f"checkpoint_{i:03}.pt found, should not be there"
        assert not os.path.exists(os.path.join(path, f"optimizer_{i:03}.pt")), f"optimizer_{i:03}.pt found, should not be there"


@pytest.mark.parametrize("conditions", [
    (None, None, None), # conditions, embeddings, channels
    (("input_tensor", "time_continuous", "time_vector", "time_discrete"), (15,), (15, 3)),
    ])
@pytest.mark.parametrize("loss", ["canonical", "meng"])
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
    losses = derivative_model.fit(dataset, batch_size=B, epochs=E, path=path, checkpoint_every=1, models_to_keep=models_to_keep)

    print(losses)
    assert len(losses) == E, f"Expected {E} losses, got {len(losses)}"
    # Check that some improvement happens
    assert os.path.isdir(os.path.join(path, "score_model")), "score_model directory not found, the base SBM has not been saved"
    assert os.path.isfile(os.path.join(path, "model_hparams.json")), "model_hparams.json not found"
    assert os.path.isfile(os.path.join(path, "script_params.json")), "script_params.json not found"
    for i in range(E+1-models_to_keep, E+1):
        assert os.path.isfile(os.path.join(path, f"checkpoint_{i:03}.pt")), f"checkpoint_{i:03}.pt not found"
        assert os.path.isfile(os.path.join(path, f"optimizer_{i:03}.pt")), f"optimizer_{i:03}.pt not found"
    for i in range(0, E+1-models_to_keep): # Check that files are cleaned up
        assert not os.path.exists(os.path.join(path, f"checkpoint_{i:03}.pt")), f"checkpoint_{i:03}.pt found, should not be there"
        assert not os.path.exists(os.path.join(path, f"optimizer_{i:03}.pt")), f"optimizer_{i:03}.pt found, should not be there"

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
    for i in range(2*E+1-models_to_keep, 2*E+1):
        assert os.path.isfile(os.path.join(path, f"checkpoint_{i:03}.pt")), f"checkpoint_{i:03}.pt not found"
        assert os.path.isfile(os.path.join(path, f"optimizer_{i:03}.pt")), f"optimizer_{i:03}.pt not found"
    for i in range(E, 2*E+1-models_to_keep): # Check that files are cleaned up
        assert not os.path.exists(os.path.join(path, f"checkpoint_{i:03}.pt")), f"checkpoint_{i:03}.pt found, should not be there"
        assert not os.path.exists(os.path.join(path, f"optimizer_{i:03}.pt")), f"optimizer_{i:03}.pt found, should not be there"


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
    assert os.path.isfile(os.path.join(path, "base_sbm", "model_hparams.json")), "model_hparams.json not found in base_sbm directory"
    assert os.path.isfile(os.path.join(path, "base_sbm", "checkpoint_001.pt")), "checkpout_001.pt not found in base_sbm directory"
    assert os.path.isfile(os.path.join(path, "model_hparams.json")), "model_hparams.json not found"
    assert os.path.isfile(os.path.join(path, "script_params.json")), "script_params.json not found"
    print(os.listdir(path))
    for i in range(E+1-models_to_keep, E+1):
        assert os.path.isdir(os.path.join(path, f"lora_checkpoint_{i:03}")), f"lora_checkpoint_{i:03} not found"
        assert os.path.isfile(os.path.join(path, f"optimizer_{i:03}.pt")), f"optimizer_{i:03}.pt not found"
    for i in range(0, E+1-models_to_keep): # Check that files are cleaned up
        assert not os.path.exists(os.path.join(path, f"lora_checkpoint_{i:03}")), f"lora_checkpoint_{i:03} found, should not be there"
        assert not os.path.exists(os.path.join(path, f"optimizer_{i:03}.pt")), f"optimizer_{i:03}.pt found, should not be there"
    

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
    for i in range(2*E+1-models_to_keep, 2*E+1):
        assert os.path.isdir(os.path.join(path, f"lora_checkpoint_{i:03}")), f"lora_checkpoint_{i:03} not found"
        assert os.path.isfile(os.path.join(path, f"optimizer_{i:03}.pt")), f"optimizer_{i:03}.pt not found"
    for i in range(E, 2*E+1-models_to_keep): # Check that files are cleaned up
        assert not os.path.exists(os.path.join(path, f"lora_checkpoint_{i:03}")), f"lora_checkpoint_{i:03} found, should not be there"
        assert not os.path.exists(os.path.join(path, f"optimizer_{i:03}.pt")), f"optimizer_{i:03}.pt found, should not be there"


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
            models_to_keep=1
            )
    # Check stdout for the print statement declaring we resumed from a previous checkpoint for the optimizer
    captured = capsys.readouterr()
    print(captured.out)
    assert f"Resumed training from checkpoint {E}." in captured.out
    assert f"Loaded optimizer {E} from test." in captured.out
