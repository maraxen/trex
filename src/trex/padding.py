"""Padding and masking utilities for static-shape JAX compilation.

Provides bucketed shapes for N (sequence length), K (epistasis), and tree nodes
to avoid XLA recompilation when parameters change during optimization.

Bucketing Strategy:
    - N (sequence length): buckets at [32, 64, 128, 256]
    - K (epistasis): buckets at [2, 4, 8]
    - MAX_NODES: fixed at 63 (2^6 - 1 for binary trees up to 32 leaves)

All input arrays are padded to bucket ceilings, with masks used to exclude
padded positions from computations.

References:
    - projects/asr/src/asr/oed/padding.py (original implementation)
    - .agents/codestyles/jax.md (static shape guidelines)
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

# Bucket definitions
N_BUCKETS: tuple[int, ...] = (32, 64, 128, 256)
K_BUCKETS: tuple[int, ...] = (2, 4, 8)
MAX_NODES: int = 63  # 2^6 - 1, supports binary trees up to 32 leaves


def get_bucket(value: int, buckets: tuple[int, ...]) -> int:
  """Find the smallest bucket that can fit the value.

  Args:
      value: The actual parameter value.
      buckets: Tuple of bucket sizes in ascending order.

  Returns:
      The bucket size to use.

  Raises:
      ValueError: If value exceeds all buckets.
  """
  for bucket in buckets:
    if value <= bucket:
      return bucket
  raise ValueError(f"Value {value} exceeds all buckets {buckets}")


def get_n_bucket(n: int) -> int:
  """Get the N bucket for sequence length.

  Args:
      n: Actual sequence length.

  Returns:
      The bucket size to use for sequence-length arrays.
  """
  return get_bucket(n, N_BUCKETS)


def get_k_bucket(k: int) -> int:
  """Get the K bucket for epistasis density.

  Args:
      k: Actual epistasis parameter.

  Returns:
      The bucket size to use for K-dimensioned arrays.
  """
  return get_bucket(k, K_BUCKETS)


def pad_sequence(sequence: Array, target_n: int) -> Array:
  """Pad a sequence to the target length.

  Args:
      sequence: Sequence array of shape (N,) or (batch, N).
      target_n: Target padded length.

  Returns:
      Padded sequence of shape (target_n,) or (batch, target_n).
  """
  if sequence.ndim == 1:
    current_n = sequence.shape[0]
    pad_width = target_n - current_n
    return jnp.pad(sequence, (0, pad_width), constant_values=0)
  # Batch case: (batch, N)
  current_n = sequence.shape[1]
  pad_width = target_n - current_n
  return jnp.pad(sequence, ((0, 0), (0, pad_width)), constant_values=0)


def pad_sequences_batch(sequences: Array, target_n: int) -> Array:
  """Pad a batch of sequences to the target length.

  Args:
      sequences: Sequences array of shape (batch, N).
      target_n: Target padded length.

  Returns:
      Padded sequences of shape (batch, target_n).
  """
  current_n = sequences.shape[1]
  if current_n >= target_n:
    return sequences[:, :target_n]
  pad_width = target_n - current_n
  return jnp.pad(sequences, ((0, 0), (0, pad_width)), constant_values=0)


def create_sequence_mask(real_n: int, padded_n: int) -> Bool[Array, " padded_n"]:
  """Create a mask for valid sequence positions.

  Args:
      real_n: Actual sequence length.
      padded_n: Padded sequence length.

  Returns:
      Boolean mask of shape (padded_n,), True for valid positions.
  """
  return jnp.arange(padded_n) < real_n


def create_node_mask(real_nodes: int, max_nodes: int = MAX_NODES) -> Bool[Array, " max_nodes"]:
  """Create a mask for valid tree nodes.

  Args:
      real_nodes: Actual number of nodes.
      max_nodes: Maximum padded node count (default: MAX_NODES).

  Returns:
      Boolean mask of shape (max_nodes,), True for valid nodes.
  """
  return jnp.arange(max_nodes) < real_nodes


def pad_fitness_table(
  fitness_tables: Array,
  real_n: int,
  real_k: int,
  target_n: int,
  target_k: int,
  q: int,
) -> Array:
  """Pad fitness tables for the NK model.

  The fitness table has shape (N, q^(K+1)). We pad the N dimension
  and potentially the table entries if K changes.

  Args:
      fitness_tables: Original fitness tables of shape (real_n, q^(real_k+1)).
      real_n: Actual sequence length.
      real_k: Actual epistasis parameter.
      target_n: Target padded sequence length.
      target_k: Target padded epistasis parameter.
      q: Alphabet size.

  Returns:
      Padded fitness tables of shape (target_n, q^(target_k+1)).
  """
  # Pad K dimension (if needed, expand table)
  real_table_size = q ** (real_k + 1)
  target_table_size = q ** (target_k + 1)
  k_pad = target_table_size - real_table_size

  if k_pad > 0:
    fitness_tables = jnp.pad(fitness_tables, ((0, 0), (0, k_pad)), constant_values=0.0)

  # Pad N dimension
  n_pad = target_n - real_n
  if n_pad > 0:
    fitness_tables = jnp.pad(fitness_tables, ((0, n_pad), (0, 0)), constant_values=0.0)

  return fitness_tables


def pad_interactions(
  interactions: Array, real_n: int, real_k: int, target_n: int, target_k: int
) -> Array:
  """Pad interaction indices for the NK model.

  The interactions array has shape (N, K). Padded indices point to
  the first position (index 0) which is safe/neutral.

  Args:
      interactions: Original interactions of shape (real_n, real_k).
      real_n: Actual sequence length.
      real_k: Actual epistasis parameter.
      target_n: Target padded sequence length.
      target_k: Target padded epistasis parameter.

  Returns:
      Padded interactions of shape (target_n, target_k).
  """
  # Pad K dimension first
  k_pad = target_k - real_k
  if k_pad > 0:
    interactions = jnp.pad(interactions, ((0, 0), (0, k_pad)), constant_values=0)

  # Pad N dimension
  n_pad = target_n - real_n
  if n_pad > 0:
    interactions = jnp.pad(interactions, ((0, n_pad), (0, 0)), constant_values=0)

  return interactions


def pad_adjacency(adjacency: Array, target_nodes: int = MAX_NODES) -> Array:
  """Pad an adjacency matrix to the target node count.

  Args:
      adjacency: Adjacency matrix of shape (n_nodes, n_nodes).
      target_nodes: Target padded node count.

  Returns:
      Padded adjacency matrix of shape (target_nodes, target_nodes).
  """
  real_nodes = adjacency.shape[0]
  n_pad = target_nodes - real_nodes
  if n_pad > 0:
    adjacency = jnp.pad(adjacency, ((0, n_pad), (0, n_pad)), constant_values=0)
  return adjacency


def pad_tree_sequences(
  sequences: Array, target_nodes: int = MAX_NODES, target_n: int | None = None
) -> Array:
  """Pad tree sequences to target dimensions.

  Args:
      sequences: Sequences of shape (n_nodes, seq_len, n_states) or (n_nodes, seq_len).
      target_nodes: Target padded node count.
      target_n: Target padded sequence length (optional).

  Returns:
      Padded sequences.
  """
  real_nodes = sequences.shape[0]
  real_n = sequences.shape[1]

  if sequences.ndim == 2:
    # Shape: (n_nodes, seq_len)
    pad_width = [(0, target_nodes - real_nodes), (0, 0)]
    if target_n is not None:
      pad_width[1] = (0, target_n - real_n)
  else:
    # Shape: (n_nodes, seq_len, n_states)
    pad_width = [(0, target_nodes - real_nodes), (0, 0), (0, 0)]
    if target_n is not None:
      pad_width[1] = (0, target_n - real_n)

  return jnp.pad(sequences, pad_width, constant_values=0)


def masked_mean(values: Array, mask: Array) -> Float[Array, ""]:
  """Compute mean over valid (masked) elements.

  Args:
      values: Array of values.
      mask: Boolean mask (True for valid elements).

  Returns:
      Mean over valid elements.
  """
  return jnp.sum(values * mask) / jnp.sum(mask)


def masked_sum(values: Array, mask: Array) -> Float[Array, ""]:
  """Compute sum over valid (masked) elements.

  Args:
      values: Array of values.
      mask: Boolean mask (True for valid elements).

  Returns:
      Sum over valid elements.
  """
  return jnp.sum(values * mask)
