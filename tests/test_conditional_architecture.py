from score_models import NCSNpp, DDPM, MLP
from functools import partial
import torch
import pytest

@pytest.mark.parametrize("Net", [NCSNpp, MLP, DDPM])
@pytest.mark.parametrize("conditions", [
    (("time_continuous",), None, None),
    (("input_tensor",), None, (10,)),
    (("time_discrete",), (15,), None),
    (("time_vector",), None, (12,)),
    (("input_tensor", "time_continuous"), None, (15,)),
    (("time_continuous", "time_discrete"), (32,), None),
    (("time_continuous", "time_discrete", "time_discrete"), (32, 12), None),
    (("input_tensor", "time_continuous", "time_vector", "time_discrete"), (15,), (3, 12)),
    ])
def test_conditional_branch(Net, conditions):
    condition_type, condition_embeddings, condition_channels = conditions
    hp = {
            "ch_mult": (1, 1), 
            "nf": 8,
            "conditions": condition_type,
            "condition_channels": condition_channels,
            "condition_embeddings": condition_embeddings
            }
    B = 5
    C = 10
    D = [] if Net == MLP else [8, 8]
    net = Net(C, **hp)
    assert net.conditioned
    assert hasattr(net, "conditional_branch")
    
    x = torch.randn(B, C, *D)
    t = torch.randn(B)
    c = []
    c_idx = 0
    for condition in condition_type:
        if condition == "time_continuous":
            c.append(torch.randn(B))
        elif condition == "time_discrete":
            c.append(torch.randint(10, (B,)))
        elif condition == "time_vector":
            c.append(torch.randn(B, condition_channels[c_idx]))
            c_idx += 1
        elif condition == "input_tensor":
            c.append(torch.randn(B, condition_channels[c_idx], *D))
            c_idx += 1
    
    print([_c.shape for _c in c])
    out = net(t, x, *c)
    assert out.shape == x.shape


@pytest.mark.parametrize("Net", [NCSNpp, MLP, DDPM])
@pytest.mark.parametrize("conditions", [
    (("input_tensor",), None, None), # Channel not provided
    (("time_discrete",), None, None), # Embedding not provided
    (("time_vector",), None, None),
    (("input_tensor", "time_vector"), None, (15,)), # Not enough channels
    (("time_discrete", "time_discrete"), (32,), None), # Not enough embeddings
    ])
def test_validate_conditional_branch_errors(Net, conditions):
    condition_type, condition_embeddings, condition_channels = conditions
    hp = {
            "ch_mult": (1, 1), 
            "nf": 8,
            "conditions": condition_type,
            "condition_channels": condition_channels,
            "condition_embeddings": condition_embeddings
            }
    B = 5
    C = 10
    D = [] if Net == MLP else [8, 8]
    with pytest.raises(ValueError):
        net = Net(C, **hp)


@pytest.mark.parametrize("Net", [NCSNpp, MLP, DDPM])
def test_merging_errors_len_args(Net):
    condition_type = ("input_tensor", "time_vector")
    condition_embeddings = None
    condition_channels = (15, 15)
    hp = {
            "ch_mult": (1, 1), 
            "nf": 8,
            "conditions": condition_type,
            "condition_channels": condition_channels,
            "condition_embeddings": condition_embeddings
            }
    B = 5
    C = 10
    D = [] if Net == MLP else [8, 8]
    net = Net(C, **hp)
    t = torch.randn(B)
    x = torch.randn(B, C, *D)
    c = [torch.randn(B, 15),] # Not enough arguments provided 
    with pytest.raises(ValueError) as exc_info:
        net(t, x, *c)
    assert "The network requires 2 additional arguments, but 1 were provided." in str(exc_info.value)
        

@pytest.mark.parametrize("Net", [NCSNpp, MLP, DDPM])
@pytest.mark.parametrize("conditions", [
    (("time_discrete",), (3,), None),
    (("time_discrete", "time_discrete"), (3, 10), None),
    ])
def test_merging_errors_embedding_arg(conditions,  Net):
    condition_type, condition_embeddings, condition_channels = conditions
    hp = {
            "ch_mult": (1, 1), 
            "nf": 8,
            "conditions": condition_type,
            "condition_channels": condition_channels,
            "condition_embeddings": condition_embeddings
            }
    B = 5
    C = 10
    D = [] if Net == MLP else [8, 8]
    net = Net(C, **hp)
    t = torch.randn(B)
    x = torch.randn(B, C, *D)
    if len(condition_type) == 1:
        c = [torch.ones(B, 1).long() * 4,]
        with pytest.raises(ValueError) as exc_info:
            net(t, x, *c)
        max_int = condition_embeddings[0] - 1
        assert f"Additional argument 0 must be a long tensor with values between 0 and {max_int} inclusively." in str(exc_info.value)
    elif len(condition_type) == 2:
        c = [torch.ones(B, 1).long() * 2, torch.ones(B, 1).long() * 15]
        with pytest.raises(ValueError) as exc_info:
            net(t, x, *c)
        max_int = condition_embeddings[1] - 1
        assert f"Additional argument 1 must be a long tensor with values between 0 and {max_int} inclusively." in str(exc_info.value)
