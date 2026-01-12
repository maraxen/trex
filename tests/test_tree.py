"""Unit tests for tree.py using pytest and chex."""

import chex
import jax
import jax.numpy as jnp

from trex.tree import (
  compute_cost,
  compute_loss,
  compute_surrogate_cost,
  discretize_tree_topology,
  enforce_graph_constraints,
  update_seq,
  update_tree,
)


def test_discretize_tree_topology():
  soft = jnp.array([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1], [0.3, 0.3, 0.4]])
  n_nodes = 3
  onehot = discretize_tree_topology(soft, n_nodes)
  chex.assert_shape(onehot, (3, 3))
  assert jnp.all((onehot == 0) | (onehot == 1))
  assert jnp.all(onehot.sum(axis=1) == 1)


def test_update_tree_shape():
  params = {"tree_params": jnp.ones((2, 1))}
  out = update_tree(jax.random.key(0), params, temperature=1.0)
  chex.assert_shape(out, (3, 3))
  assert jnp.allclose(out.sum(axis=1), 1.0)


def test_update_tree_no_ancestors():
  params = {"tree_params": jnp.ones((1, 0))}  # No ancestor nodes
  out = update_tree(jax.random.key(0), params, temperature=1.0)
  chex.assert_shape(out, (2, 2))  # Expecting shape to be valid for 1 leaf node


def test_update_seq_shape():
  # n_nodes=4, seq_len=5, n_states=4
  params = {"ancestors": jnp.ones((2, 5, 4))}
  seqs = jnp.zeros((4, 5, 4))
  out = update_seq(params, seqs, temperature=1.0)
  chex.assert_shape(out, (4, 5, 4))


def test_enforce_graph_constraints():
  soft = jnp.eye(5)
  loss = enforce_graph_constraints(soft, 10.0)
  assert loss >= 0


def test_compute_surrogate_cost():
  # n_nodes = 2, seq_len = 3, n_states = 4
  tree = jnp.eye(2)
  # Sequences must be 3D: (n_nodes, seq_len, n_states)
  seq_indices = jnp.array([[0, 1, 2], [3, 2, 1]])
  seqs = jax.nn.one_hot(seq_indices, num_classes=4)
  cost = compute_surrogate_cost(seqs, tree)
  assert cost >= 0


def test_compute_cost():
  # Input shape is (n_nodes=3, seq_len=2, n_states=2)
  seqs = jax.nn.one_hot(jnp.array([[0, 1], [1, 0], [0, 0]]), 2)
  tree = jnp.eye(3)
  subst = jnp.ones((2, 2)) - jnp.eye(2)
  cost = compute_cost(seqs, tree, subst)
  assert cost >= 0


def test_compute_loss_basic():
  # n_nodes=4, seq_len=5, n_states=4
  params = {
    # FIX: Shape changed from (2, 1) to (3, 2) to create a 4x4 adjacency matrix
    "tree_params": jnp.ones((3, 2)),
    "ancestors": jnp.ones((2, 5, 4)),  # (n_ancestors, seq_len, n_states)
  }
  seqs = jnp.zeros((4, 5, 4))  # (n_nodes, seq_len, n_states)
  metadata = {"n_all": 4, "n_states": 4}
  key = jax.random.PRNGKey(0)
  # Dummy adjacency matrix for the test
  adjacency = jnp.eye(4)
  loss = compute_loss(key, params, seqs, metadata, temperature=1.0, adjacency=adjacency)
  assert not jnp.isnan(loss)
  assert loss >= 0
