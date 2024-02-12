from score_models import NCSNpp, DDPM, MLP
import torch
import pytest

def test_discrete_timelike_conditional():
    nf = 32
    net = NCSNpp(
            nf=nf, 
            ch_mult=(1, 1), 
            condition=["discrete_timelike"],
            condition_num_embedding=[10],
            )
   
    B = 10
    c = torch.randint(10, (B,))
    x = torch.randn(B, 1, 8, 8)
    t = torch.rand(B)
    
    out = net(t, x, c)
    assert out.shape == x.shape
    assert net.condition_embedding_layers[0](c).shape == torch.Size([B, nf])

def test_continuous_timelike_conditional():
    nf = 32
    net = NCSNpp(
            nf=nf, 
            ch_mult=(1, 1), 
            condition=["continuous_timelike"]
            )
   
    B = 10
    c = torch.randn(B)
    x = torch.randn(B, 1, 8, 8)
    t = torch.rand(B)
    
    out = net(t, x, c)
    assert out.shape == x.shape
    assert net.condition_embedding_layers[0](c).shape == torch.Size([B, nf])

def test_continuous_input_conditional():
    nf = 32
    C_cond = 3
    net = NCSNpp(
            nf=nf, 
            ch_mult=(1, 1), 
            condition=["input"],
            condition_input_channels=3
            )
   
    B = 10
    c = torch.randn(B, C_cond, 8, 8)
    x = torch.randn(B, 1, 8, 8)
    t = torch.rand(B)
    
    out = net(t, x, c)
    assert out.shape == x.shape

def test_vector_condition():
    nf = 32
    C_cond = 3
    net = NCSNpp(
            nf=nf, 
            ch_mult=(1, 1), 
            condition=["vector"],
            condition_vector_channels=3
            )
   
    B = 10
    c = torch.randn(B, C_cond)
    x = torch.randn(B, 1, 8, 8)
    t = torch.rand(B)
    
    out = net(t, x, c)
    assert out.shape == x.shape


def test_mix_condition_type():
    nf = 32
    C_cond = 3
    net = NCSNpp(
            nf=nf, 
            ch_mult=(1, 1), 
            condition=["input", "discrete_timelike", "continuous_timelike", "continuous_timelike"],
            condition_input_channels=3,
            condition_num_embedding=(15,),
            )
   
    B = 10
    c_input = torch.randn(B, C_cond, 8, 8)
    c_discrete = torch.randint(10, (B,))
    c_cont1 = torch.randn(B)
    c_cont2 = torch.randn(B)
    x = torch.randn(B, 1, 8, 8)
    t = torch.rand(B)
    
    out = net(t, x, c_input, c_discrete, c_cont1, c_cont2)
    assert out.shape == x.shape


def test_conditional_architecture_raising_errors():
    nf = 32
    with pytest.raises(ValueError):
        net = NCSNpp(
                nf=nf, 
                ch_mult=(1, 1), 
                condition=["discrete_timelike"],
                )

    with pytest.raises(ValueError):
        net = NCSNpp(
                nf=nf, 
                ch_mult=(1, 1), 
                condition=["discrete_timelike"],
                condition_num_embedding=15
                )

    with pytest.raises(ValueError):
        net = NCSNpp(
                nf=nf, 
                ch_mult=(1, 1), 
                condition=["input"],
                )

    with pytest.raises(ValueError):
        net = NCSNpp(
                nf=nf, 
                ch_mult=(1, 1), 
                condition="input",
                )
