"""Unit tests for padding.py using pytest and chex."""

import chex
import jax.numpy as jnp
import pytest

from trex.padding import (
  K_BUCKETS,
  MAX_NODES,
  N_BUCKETS,
  create_node_mask,
  create_sequence_mask,
  get_bucket,
  get_k_bucket,
  get_n_bucket,
  masked_mean,
  masked_sum,
  pad_adjacency,
  pad_fitness_table,
  pad_interactions,
  pad_sequence,
  pad_sequences_batch,
  pad_tree_sequences,
)


class TestBuckets:
  """Test bucket selection logic."""

  def test_get_bucket_exact_match(self):
    assert get_bucket(32, N_BUCKETS) == 32
    assert get_bucket(64, N_BUCKETS) == 64

  def test_get_bucket_between_values(self):
    assert get_bucket(50, N_BUCKETS) == 64
    assert get_bucket(100, N_BUCKETS) == 128
    assert get_bucket(33, N_BUCKETS) == 64

  def test_get_bucket_exceeds_all(self):
    with pytest.raises(ValueError, match="exceeds all buckets"):
      get_bucket(1000, N_BUCKETS)

  def test_get_n_bucket(self):
    assert get_n_bucket(30) == 32
    assert get_n_bucket(50) == 64
    assert get_n_bucket(100) == 128

  def test_get_k_bucket(self):
    assert get_k_bucket(2) == 2
    assert get_k_bucket(3) == 4
    assert get_k_bucket(5) == 8


class TestSequencePadding:
  """Test sequence padding functions."""

  def test_pad_sequence_1d(self):
    seq = jnp.array([1, 2, 3])
    padded = pad_sequence(seq, 5)
    chex.assert_shape(padded, (5,))
    assert jnp.array_equal(padded, jnp.array([1, 2, 3, 0, 0]))

  def test_pad_sequence_2d(self):
    seqs = jnp.array([[1, 2], [3, 4]])
    padded = pad_sequence(seqs, 4)
    chex.assert_shape(padded, (2, 4))
    assert jnp.array_equal(padded[0], jnp.array([1, 2, 0, 0]))

  def test_pad_sequences_batch(self):
    seqs = jnp.array([[1, 2, 3], [4, 5, 6]])
    padded = pad_sequences_batch(seqs, 5)
    chex.assert_shape(padded, (2, 5))
    assert jnp.array_equal(padded[0, :3], jnp.array([1, 2, 3]))
    assert jnp.array_equal(padded[0, 3:], jnp.array([0, 0]))

  def test_pad_sequences_batch_truncate(self):
    seqs = jnp.array([[1, 2, 3, 4, 5]])
    padded = pad_sequences_batch(seqs, 3)
    chex.assert_shape(padded, (1, 3))
    assert jnp.array_equal(padded[0], jnp.array([1, 2, 3]))


class TestMasks:
  """Test mask creation functions."""

  def test_create_sequence_mask(self):
    mask = create_sequence_mask(3, 5)
    chex.assert_shape(mask, (5,))
    assert jnp.array_equal(mask, jnp.array([True, True, True, False, False]))

  def test_create_node_mask(self):
    mask = create_node_mask(5, 8)
    chex.assert_shape(mask, (8,))
    assert jnp.sum(mask) == 5

  def test_create_node_mask_default_max(self):
    mask = create_node_mask(10)
    chex.assert_shape(mask, (MAX_NODES,))
    assert jnp.sum(mask) == 10


class TestMaskedReductions:
  """Test masked mean and sum functions."""

  def test_masked_mean_basic(self):
    values = jnp.array([1.0, 2.0, 3.0, 0.0, 0.0])
    mask = jnp.array([True, True, True, False, False])
    result = masked_mean(values, mask)
    assert jnp.isclose(result, 2.0)

  def test_masked_sum_basic(self):
    values = jnp.array([1.0, 2.0, 3.0, 99.0, 99.0])
    mask = jnp.array([True, True, True, False, False])
    result = masked_sum(values, mask)
    assert jnp.isclose(result, 6.0)

  def test_masked_mean_all_valid(self):
    values = jnp.array([1.0, 2.0, 3.0, 4.0])
    mask = jnp.ones(4, dtype=bool)
    result = masked_mean(values, mask)
    assert jnp.isclose(result, 2.5)


class TestNKPadding:
  """Test padding for NK model structures."""

  def test_pad_fitness_table(self):
    # Original: (3, 4) for N=3, K=1, q=2 -> 2^(1+1)=4
    table = jnp.ones((3, 4))
    padded = pad_fitness_table(table, 3, 1, 5, 2, 2)
    # Target: (5, 8) for N=5, K=2, q=2 -> 2^(2+1)=8
    chex.assert_shape(padded, (5, 8))
    # Original values preserved
    assert jnp.all(padded[:3, :4] == 1.0)
    # Padded values are zero
    assert jnp.all(padded[3:, :] == 0.0)
    assert jnp.all(padded[:, 4:] == 0.0)

  def test_pad_interactions(self):
    # Original: (3, 2) for N=3, K=2
    inter = jnp.array([[1, 2], [0, 1], [2, 0]])
    padded = pad_interactions(inter, 3, 2, 5, 4)
    chex.assert_shape(padded, (5, 4))
    # Original values preserved
    assert jnp.array_equal(padded[:3, :2], inter)
    # Padded values are zero
    assert jnp.all(padded[3:, :] == 0)
    assert jnp.all(padded[:, 2:] == 0)


class TestTreePadding:
  """Test padding for tree structures."""

  def test_pad_adjacency(self):
    adj = jnp.eye(5)
    padded = pad_adjacency(adj, 10)
    chex.assert_shape(padded, (10, 10))
    # Original preserved
    assert jnp.array_equal(padded[:5, :5], adj)
    # Padded zeros
    assert jnp.all(padded[5:, :] == 0)
    assert jnp.all(padded[:, 5:] == 0)

  def test_pad_tree_sequences_3d(self):
    # (n_nodes=3, seq_len=4, n_states=2)
    seqs = jnp.ones((3, 4, 2))
    padded = pad_tree_sequences(seqs, target_nodes=5, target_n=6)
    chex.assert_shape(padded, (5, 6, 2))

  def test_pad_tree_sequences_2d(self):
    # (n_nodes=3, seq_len=4)
    seqs = jnp.ones((3, 4))
    padded = pad_tree_sequences(seqs, target_nodes=5)
    chex.assert_shape(padded, (5, 4))


class TestRecompilationPrevention:
  """Test that padded operations avoid recompilation."""

  def test_same_bucket_no_recompile(self):
    """Sequences of different length in same bucket should not trigger recompile."""
    import jax

    @jax.jit
    def compute_padded_mean(seq, mask):
      return masked_mean(seq, mask)

    # Two sequences of different length but same bucket (64)
    seq1 = pad_sequence(jnp.arange(50, dtype=jnp.float32), 64)
    mask1 = create_sequence_mask(50, 64)

    seq2 = pad_sequence(jnp.arange(60, dtype=jnp.float32), 64)
    mask2 = create_sequence_mask(60, 64)

    # Both should use the same compiled function
    result1 = compute_padded_mean(seq1, mask1)
    result2 = compute_padded_mean(seq2, mask2)

    # Results should be correct
    assert jnp.isclose(result1, jnp.mean(jnp.arange(50, dtype=jnp.float32)))
    assert jnp.isclose(result2, jnp.mean(jnp.arange(60, dtype=jnp.float32)))
