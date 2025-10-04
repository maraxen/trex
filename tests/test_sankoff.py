"""Unit tests for sankoff.py using pytest and chex."""

import chex
import jax.numpy as jnp

from trex.sankoff import run_dp, run_sankoff


def test_run_dp_basic():
  # Simple binary tree: 3 nodes (2 leaves, 1 ancestor)
  adj = jnp.array(
    [
      [0, 1, 0],
      [0, 1, 0],
      [0, 0, 0],
    ],
    dtype=jnp.float32,
  )
  n_states = 2
  seqs = jnp.array(
    [
      [0],
      [1],
      [0],
    ],
    dtype=jnp.float32,
  )
  cost_matrix = jnp.array([[0, 1], [1, 0]], dtype=jnp.float32)
  dp = jnp.full((3, n_states), 1e5, dtype=jnp.float32)
  back = jnp.zeros((3, n_states, 4), dtype=jnp.float32)
  dp_out, back_out = run_dp(adj, dp, back, seqs, cost_matrix)
  chex.assert_shape(dp_out, (3, n_states))
  chex.assert_shape(back_out, (3, n_states, 4))
  # Leaves should have 0 at their observed state
  assert dp_out[0, 0] == 0
  assert dp_out[1, 1] == 0


def test_run_sankoff_basic():
  # Tree: 3 leaves, 2 ancestors (n_all=5)
  adj = jnp.zeros((5, 5), dtype=jnp.int32)
  adj = adj.at[0, 3].set(1)
  adj = adj.at[1, 3].set(1)
  adj = adj.at[2, 4].set(1)
  adj = adj.at[3, 4].set(1)
  cost_matrix = jnp.ones((2, 2)) - jnp.eye(2)
  seqs = jnp.array(
    [
      [0, 1],
      [1, 0],
      [0, 0],
      [0, 0],
      [0, 0],
    ],
    dtype=jnp.float32,
  )
  leaf_seqs = seqs[:3]

  rec_seqs, dp, total_cost = run_sankoff(
    adj,
    cost_matrix,
    leaf_seqs,
    5,
    2,
    3,
    return_path=True,
  )
  chex.assert_shape(rec_seqs, (5, 2))
  chex.assert_shape(dp, (2, 5, 2))
  assert total_cost >= 0
  # Leaves should match input
  assert jnp.all(rec_seqs[:3] == seqs[:3])
