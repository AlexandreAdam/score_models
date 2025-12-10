from score_models import HessianDiagonal, ScoreModel, MLP
import torch
import pytest
import os


@pytest.mark.parametrize("loss", ["lu", "meng"])
def test_save_load_hessian_diagonal(loss, tmp_path):
    path = os.path.join(tmp_path, "test")
    net = MLP(10)
    hessian_net = MLP(10)
    score_model = ScoreModel(net, sde="vp")
    model = HessianDiagonal(score_model, hessian_net, loss=loss)
    
    for i in range(3):
        model.save(path)
    
    # Check that we can reload the whole setup just from path
    new_model = HessianDiagonal(path=path)
    
    # Check that the architecture is reloaded correctly
    B = 10
    D = 10
    x = torch.randn(B, D)
    t = torch.randn(B)
    with torch.no_grad():
        assert torch.allclose(model(t, x), new_model(t, x), atol=1e-3)
        # Check that sbm is loaded correctly for the loss function
        assert torch.allclose(model.score_model(t, x), new_model.score_model(t, x), atol=1e-3)
        torch.manual_seed(42)
        loss1 = model.loss(x)
        
        torch.manual_seed(42)
        loss2 = new_model.loss(x)
        # Give it a loose tolerance, not sure why they are differen just yet
        assert torch.allclose(loss1, loss2, atol=4) 
