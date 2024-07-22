from score_models import LoRAScoreModel, NCSNpp, MLP, ScoreModel
import os
import torch
import pytest

@pytest.mark.parametrize("lora_rank", [1, 10, 30])
@pytest.mark.parametrize("sde", [{"sde": "vp"}, {"sde": "ve", "sigma_min": 0.1, "sigma_max": 100.0}])
@pytest.mark.parametrize("net", [MLP(10), NCSNpp(1, ch_mult=[1, 1], nf=8)])
def test_lora_sbm(net, sde, lora_rank, tmp_path):
    base_sbm = ScoreModel(net, **sde)
    sbm = LoRAScoreModel(base_sbm, lora_rank=lora_rank)
    
    # Check that checkpoints are being saved correctly
    path = os.path.join(tmp_path, "test")
    for i in range(3):
        sbm.save(path)
    
    print(os.listdir(path))
    assert os.path.exists(path)
    assert os.path.exists(os.path.join(path, "model_hparams.json"))
    assert os.path.exists(os.path.join(path, "base_sbm"))
    assert os.path.isdir(os.path.join(path, "base_sbm"))
    for i in range(1, 4):
        assert os.path.exists(os.path.join(path, f"lora_checkpoint_{i:03d}"))
        assert os.path.isdir(os.path.join(path, f"lora_checkpoint_{i:03d}"))
    
    # Check that we can reload the whole setup just from path
    new_sbm = LoRAScoreModel(path=path)
    
    # Check that models are consistent with each other
    B = 10
    D = [10] if isinstance(net, MLP) else [1, 8, 8]
    t = torch.rand(B)
    x = torch.randn(B, *D)
    with torch.no_grad():
        print(sbm(t, x) - new_sbm(t, x))
        assert torch.allclose(sbm(t, x), new_sbm(t, x))
        # Sanity check that we are using the LoRA model
        assert torch.allclose(sbm.lora_net(t, x), new_sbm.lora_net(t, x))
        assert torch.allclose(sbm.lora_net(t, x), sbm.reparametrized_score(t, x))
        assert torch.allclose(new_sbm.lora_net(t, x), new_sbm.reparametrized_score(t, x))
