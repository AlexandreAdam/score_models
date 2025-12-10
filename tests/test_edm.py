from score_models import ScoreModel, EDMScoreModel, MLP, NCSNpp, VPSDE, EDMv2Net, MLPv2
import torch
import pytest

def test_edm_init(tmp_path):
    D = 2
    B = 10
    sde = VPSDE()
    net = MLP(D)
    model = ScoreModel(net, sde, formulation="edm")
    print(model)
    
    x = torch.randn(B, D)
    t = torch.rand(B)
    out = model(t, x)
    assert tuple(out.shape) == (B, D)
    
    # Make sure that when we reload the model, it is an EDM model
    model.save(tmp_path)
    new_model = ScoreModel(path=tmp_path)
    assert isinstance(new_model, EDMScoreModel)


# TODO improve MLP with EDM framework as well
def test_edm_net(tmp_path):
    P = 16      # pixels
    D = [P]*2   # dimensions
    C = 3       # channels
    B = 10      # batch size
    
    # Parameters
    nf = 16
    ch_mult = [1, 2]
    
    sde = VPSDE()
    net = EDMv2Net(P, C, nf=nf, ch_mult=ch_mult)
    model = ScoreModel(net, sde, formulation="edm")
    
    x = torch.randn(B, C, *D)
    t = torch.rand(B)
    out = model(t, x)
    assert tuple(out.shape) == (B, C, *D)
    
    # Make sure that the model is reloaded correctly
    model.save(tmp_path)
    
    new_model = ScoreModel(path=tmp_path)
    out2 = new_model(t, x)
    assert torch.allclose(out, out2)


def test_adaptive_loss():
    D = 2
    B = 10
    net = MLPv2(D)
    model = ScoreModel(net, "vp", formulation="edm")
    
    # Check that we can use return_logvar correctly
    x = torch.randn(B, D)
    t = torch.rand(B)
    out, logvar = model.net(t, x, return_logvar=True)
    assert tuple(out.shape) == (B, D)
    assert tuple(logvar.shape) == (B, 1)
    
    # Check that this works also when calling the score
    out, logvar = model(t, x, return_logvar=True)
    assert tuple(out.shape) == (B, D)
    assert tuple(logvar.shape) == (B, 1)
    
    # Check that the loss works correctly when using the uncertainty
    loss = model.loss(x) # Baseline loss
    # Check that the loss changes when we use the uncertainty as a way to check if it is applied
    model.adaptive_loss = True # When calling fit, this varible triggers the use of the uncertainty layer
    loss2 = model.loss(x)
    assert torch.all(loss != loss2)
    
    # Check that a model instantiated with v1 neural nets yield an error
    net = MLP(D)
    model = ScoreModel(net, "vp", formulation="edm")
    with pytest.raises(ValueError):
        model(t, x, return_logvar=True)


def test_conditional_architecture():
    P = 8
    C = 1
    B = 10
    E = 5
    net = EDMv2Net(
            conditions=("time_discrete",), 
            condition_embeddings=(E,), 
            pixels=P, 
            channels=C, 
            nf=8, 
            ch_mult=(1, 1, 1, 1)
            )
    
    x = torch.randn(B, C, P, P)
    t = torch.rand(B)
    condition = torch.randint(0, E, (B,))
    out = net(t, x, condition)
    assert tuple(out.shape) == (B, C, P, P)
    
    # Make sure net has a conditional branch
    assert hasattr(net.unet, "conditioned")
    assert net.unet.conditioned
    assert hasattr(net.unet, "conditional_branch")
