import jax
import equinox as eqx
import jax.numpy as jnp
from score_models.jax.utils import model_state_dict, update_model_params


class TestModel(eqx.Module):
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, in_features, hidden_features, out_features, *, key):
        key1, key2 = jax.random.split(key)
        self.linear1 = eqx.nn.Linear(in_features, hidden_features, key=key1)
        self.linear2 = eqx.nn.Linear(hidden_features, out_features, key=key2)

    def __call__(self, x):
        x = self.linear1(x)
        x = jnp.tanh(x)
        x = self.linear2(x)
        return x


def test_model_state_update():
    key1 = jax.random.PRNGKey(0)
    model = TestModel(10, 20, 5, key=key1)
    original_state_dict = model_state_dict(model)

    # Modify the state dict to test the update
    modified_state_dict = {k: jnp.zeros_like(v) for k, v in original_state_dict.items()}
    updated_model = update_model_params(model, modified_state_dict)

    # Ensure the model parameters were updated correctly
    updated_state_dict = model_state_dict(updated_model)
    for key, value in updated_state_dict.items():
        assert jnp.allclose(value, jnp.zeros_like(value)), "Model parameters were not updated correctly"

    # Test reverting back to the original state
    reverted_model = update_model_params(updated_model, original_state_dict)
    reverted_state_dict = model_state_dict(reverted_model)
    for key, orig_value in original_state_dict.items():
        reverted_value = reverted_state_dict[key]
        assert jnp.allclose(reverted_value, orig_value), "Model parameters were not reverted correctly"

