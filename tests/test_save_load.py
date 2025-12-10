from score_models import MLP, NCSNpp, ScoreModel, EnergyModel, SLIC
import torch
import pytest
import numpy as np
import os

@pytest.mark.parametrize("net", [MLP(10), NCSNpp(10, nf=8, ch_mult=[1, 1])])
@pytest.mark.parametrize("sde", [{"sde": "vp"}, {"sde": "ve", "sigma_min": 1e-2, "sigma_max": 1e2}])
@pytest.mark.parametrize("Model", [ScoreModel, EnergyModel, SLIC])
def test_save(net, sde, Model, tmp_path):
    if Model == SLIC:
        forward_model = lambda t, x: x
        model = Model(forward_model, net, **sde)
    else:
        model = Model(net, **sde)
    
    path = os.path.join(tmp_path, "test")
    model.save(path)
    
    assert os.path.exists(path)
    assert os.path.exists(os.path.join(path, "checkpoint_001.pt"))
    assert os.path.exists(os.path.join(path, "hyperparameters.json"))
    
    # Save again
    model.save(path)
    assert os.path.exists(os.path.join(path, "checkpoint_002.pt"))
    
    # Check if we can load and continue saving checkpoints
    if Model == SLIC:
        model = Model(forward_model, path=path, **sde)
    else:
        model = Model(path=path, **sde)
    model.save()
    assert os.path.exists(os.path.join(path, "checkpoint_003.pt"))
     

@pytest.mark.parametrize("net", [MLP(10), NCSNpp(1, nf=8, ch_mult=[1, 1],)])
@pytest.mark.parametrize("sde", [{"sde": "vp"}, {"sde": "ve", "sigma_min": 1e-2, "sigma_max": 1e2}])
@pytest.mark.parametrize("Model", [ScoreModel, EnergyModel, SLIC])
def test_load(net, sde, Model, tmp_path):
    path = os.path.join(tmp_path, "test")
    if Model == SLIC:
        forward_model = lambda t, x: x
        model = Model(forward_model, net, path=path, **sde)
    else:
        model = Model(net, path=path, **sde)
    
    for i in range(10):
        model.save()
    
    # Load checkpoint
    model.load(10)
    assert model.loaded_checkpoint == 10
    
    # reload last checkpoint to compare with load
    model.load()
    
    # Check that the architecture is reloaded correctly
    if Model == SLIC: # Currently we do not save the forward model, though we could serialize it in a custom save and load function.
        new_model = Model(forward_model, path=path)
    else:
        new_model = Model(path=path)
    B = 10
    if isinstance(net, MLP):
        D = net.hyperparameters["channels"]
        x = torch.randn(B, D)
    else:
        C = net.channels
        D = net.dimensions
        x = torch.randn(B, C, *[32]*D)

    t = torch.rand(B)
    if Model == SLIC:
        y = forward_model(t, x)
        print(model(t, y, x) - new_model(t, y, x))
        assert torch.allclose(model(t, y, x), new_model(t, y, x), atol=1e-3)
    else:
        print(model(t, x) - new_model(t, x))
        assert torch.allclose(model(t, x), new_model(t, x), atol=1e-3)

