import jax
import jax.numpy as jnp
import pytest

from trex.svm_tree.model import LearnableTreeModel


@pytest.fixture
def model_key():
    """Provides a JAX random key for model initialization."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def model(model_key):
    """Provides an instance of the LearnableTreeModel."""
    return LearnableTreeModel(in_features=10, num_classes=3, n_ancestors=2, key=model_key)


def test_learnable_tree_model_init(model):
    """Tests the initialization of the LearnableTreeModel."""
    assert model is not None
    assert model.num_classes == 3
    assert model.n_leaves == 3
    assert model.n_ancestors == 2
    assert model.svm_weights.shape == (2, 10)
    assert model.svm_biases.shape == (2,)
    assert model.topology.n_leaves == 3
    assert model.topology.n_ancestors == 2


def test_learnable_tree_model_forward_pass(model):
    """Tests the forward pass of the LearnableTreeModel."""
    key = jax.random.PRNGKey(0)
    # Use a non-zero input to avoid potential issues with all-zero inputs
    x = jnp.arange(10, dtype=jnp.float32)
    output = model(x, key=key)

    assert output.shape == (3,), "Output shape should match the number of classes."
    assert jnp.allclose(
        jnp.sum(output), 1.0,
    ), "Output should be a probability distribution summing to 1."


def test_learnable_tree_model_loss(model):
    """Tests the loss calculation of the LearnableTreeModel."""
    key = jax.random.PRNGKey(0)
    # The loss function requires an adjacency matrix as input
    adj = model.topology(key)
    loss = model.loss(adj)

    assert isinstance(loss, jax.Array), "Loss should be a JAX array."
    assert loss.shape == (), "Loss should be a scalar value."
    assert loss >= 0, "Loss should be non-negative."
