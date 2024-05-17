from score_models.jax.layers import Conv2dSame as jax_Conv2dSame
from score_models.torch.layers import Conv2dSame as torch_Conv2dSame
from score_models.jax.layers.group_norm import GroupNorm
from jax import vmap
import jax.numpy as jnp
import equinox as eqx
import torch
import jax

def test_conv2dsame():
    x = jnp.ones((1, 3, 32, 32))
    torch_x = torch.ones((1, 3, 32, 32))

    torch_conv = torch_Conv2dSame(3, 1, 3, 1, 1)
    jax_conv = jax_Conv2dSame(3, 1, 3, 1, 1, key=jax.random.key(0))
    
    get_leaf = lambda t: t.conv.weight
    jax_conv = eqx.tree_at(get_leaf, jax_conv, torch_conv.state_dict()["conv.weight"].detach().numpy())

    get_leaf = lambda t: t.conv.bias
    jax_conv = eqx.tree_at(get_leaf, jax_conv, torch_conv.state_dict()["conv.bias"].detach().numpy())
    
    jax_out = vmap(jax_conv)(x)
    torch_out = torch_conv(torch_x).detach().numpy()

    assert jnp.allclose(jax_out, torch_out, atol=1e-6, rtol=1e-6)


def test_group_norm():
    torch_x = torch.randn((3, 32, 32))
    x = jnp.asarray(torch_x)
    
    torch_gn = torch.nn.GroupNorm(3, 3, eps=1e-6)
    jax_gn = GroupNorm(3, 3, eps=1e-6)
    
    get_leaf = lambda t: t.weight
    jax_gn = eqx.tree_at(get_leaf, jax_gn, torch_gn.state_dict()["weight"].detach().numpy())

    get_leaf = lambda t: t.bias
    jax_gn = eqx.tree_at(get_leaf, jax_gn, torch_gn.state_dict()["bias"].detach().numpy())

    jax_out = jax_gn(x)
    torch_out = torch_gn(torch_x.unsqueeze(0)).detach().numpy().squeeze(0)
    
    # import matplotlib.pyplot as plt 
    # plt.imshow(jax_out[0] - torch_out[0])
    # plt.colorbar()
    # plt.show()

    # Only works up to a floating point precision of 1e-5, no idea why, has to do with torch.var vs jax.var
    assert jnp.allclose(jax_out, torch_out, atol=1e-5, rtol=1e-5)
    
