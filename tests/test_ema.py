from score_models import EMA, MLP, ScoreModel
import torch
import pytest

@pytest.mark.parametrize("decay", [[None, 0.13], [0.99, None]])
def test_ema(decay):
    beta, sigma_rel = decay
    model = ScoreModel(MLP(10, width=4), "vp")
    ema = EMA(model=model, beta=beta, sigma_rel=sigma_rel)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    print("Before update:")
    for ema_p, (name, model_p) in zip(ema.ema_model.parameters(), model.named_parameters()):
        print(name)
        print(ema_p.data)
        print(model_p.data)
    
    # Simul Gradient descent
    for _ in range(20):
        x = torch.randn(2, 10)
        loss = model.loss(x)
        loss.backward()
        opt.step()
        ema.update()
        print(ema.beta)

    print("After update:")
    for ema_p, model_p in zip(ema.ema_model.parameters(), model.parameters()):
        print(ema_p.data)
        print(model_p.data)
    
    # Make sure ema model was updated and different from the online model
    for i, (model_param, ema_param, online_param) in enumerate(zip(model.parameters(), ema.ema_model.parameters(), ema.online_model.parameters())):
        if i == 0:
            continue # Skip the frozen Projection layer for the time conditioning
        assert torch.allclose(online_param, model_param) # Online model should track the model
        assert torch.all(ema_param != online_param) # EMA model should be different from the online model

