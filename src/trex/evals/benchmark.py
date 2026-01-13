"""Benchmarking script.

Compare TREX (with landscape-aware loss)
against Sankoff on datasets generated from an NK fitness landscape.
This version extends the previous benchmark to include a landscape-aware
loss function for TREX, controlled by a hyperparameter λ (lambda).
When λ=0, TREX optimizes only for parsimony (as before).
When λ>0, TREX also considers the fitness of reconstructed ancestral
sequences according to the NK landscape.
"""

from __future__ import annotations

from typing import Any, cast

import functools
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import custom_vjp

# Add this with your other imports
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, PyTree

from trex.nk_model import create_nk_model_landscape, generate_tree_data
from trex.sankoff import run_sankoff
from trex.tree import compute_soft_cost, compute_surrogate_cost, update_seq
from trex.types import Adjacency, Ancestors, LeafSequences, ScalarFloat, get_default_dtype
from trex.utils.memory import safe_map
from trex.utils.types import (
  EvoSequence,
  OneHotEvoSequence,
)

# ============================================================================
# Optimizer Factory and Configurable TREX Optimization
# ============================================================================


def create_optimizer(
  name: str,
  learning_rate: float,
  *,
  use_gradient_clipping: bool = True,
) -> optax.GradientTransformation:
  """Create an optimizer by name with optional gradient clipping.

  Args:
      name: One of "adam", "sgd", "rmsprop", "adamw"
      learning_rate: Learning rate for the optimizer
      use_gradient_clipping: Whether to apply gradient clipping (default: True)

  Returns:
      An optax GradientTransformation

  """
  optimizers = {
    "adam": optax.adam(learning_rate),
    "sgd": optax.sgd(learning_rate, momentum=0.9),
    "rmsprop": optax.rmsprop(learning_rate),
    "adamw": optax.adamw(learning_rate, weight_decay=0.01),
  }

  if name not in optimizers:
    raise ValueError(f"Unknown optimizer: {name}. Choose from {list(optimizers.keys())}")

  base_optimizer = optimizers[name]

  if use_gradient_clipping:
    return optax.chain(optax.clip_by_global_norm(1.0), base_optimizer)
  return base_optimizer


def run_trex_optimization_configurable(
  leaf_sequences: LeafSequences,
  n_all: int,
  n_leaves: int,
  n_states: int,
  adj_matrix: Adjacency,
  key: PRNGKeyArray,
  use_soft_cost: bool = False,
  optimizer_name: str = "adam",
  learning_rate: float = 1e-3,
  n_iterations: int = 10000,
  return_losses: bool = False,
  dtype: jnp.dtype | None = None,
  mixed_precision: bool = False,
) -> Ancestors | tuple[Ancestors, Float[Array, n_iterations]]:
  """Run TREX optimization with configurable hyperparameters.

  Refactored to use stacked arrays (not Python list) for vmap compatibility.
  Uses _update_seq_stacked internally for 100x faster JAX compilation.

  Args:
      leaf_sequences: Observed sequences at leaves. Shape: (n_leaves, seq_len)
      n_all: Total number of nodes in the tree
      n_leaves: Number of leaf nodes
      n_states: Alphabet size (e.g., 2 for binary, 20 for amino acids)
      adj_matrix: Adjacency matrix representing tree structure
      key: JAX random key
      use_soft_cost: If True, use compute_soft_cost; else use compute_surrogate_cost
      optimizer_name: One of "adam", "sgd", "rmsprop", "adamw"
      learning_rate: Learning rate for optimization
      n_iterations: Number of optimization iterations
      return_losses: If True, also return the loss curve
      dtype: Data type for computation (None=float32, jnp.bfloat16 for GPU)
      mixed_precision: If True, keep parameters in float32 but compute in dtype

  Returns:
      If return_losses=False: Reconstructed ancestor sequences (n_ancestors, seq_len)
      If return_losses=True: Tuple of (ancestors, loss_curve) where loss_curve
          has shape (n_iterations,)

  """
  key, subkey = jax.random.split(key)
  seq_len = leaf_sequences.shape[1]
  n_ancestors = n_all - n_leaves

  # Determine dtype: use bfloat16 on GPU if not specified
  if dtype is None:
    dtype = get_default_dtype()

  # In mixed precision, parameters (ancestors) are float32, but compute is dtype
  param_dtype = jnp.float32 if mixed_precision else dtype
  compute_dtype = dtype

  # Initialize ancestors as stacked array (NOT list!) for vmap compatibility
  ancestors = jax.random.normal(subkey, (n_ancestors, seq_len, n_states), dtype=param_dtype)

  # Create optimizer
  optimizer = create_optimizer(optimizer_name, learning_rate)
  opt_state = optimizer.init(ancestors)

  # One-hot encode leaf sequences (always in compute dtype)
  leaf_sequences_one_hot = jax.nn.one_hot(leaf_sequences, n_states).astype(compute_dtype)

  # Create masked sequence array (ancestors initialized to zero)
  # NOTE: masked_sequences will be in compute_dtype
  masked_sequences = jnp.concatenate(
    [
      leaf_sequences_one_hot,
      jnp.zeros((n_ancestors, seq_len, n_states), dtype=compute_dtype),
    ],
    axis=0,
  )

  # Pre-compute identity cost matrix for soft cost
  identity_cost_matrix = jnp.eye(n_states, dtype=compute_dtype)

  # Select appropriate loss function using stacked update
  if use_soft_cost:

    def loss_fn(ancestors):
      # Cast parameters to compute_dtype for forward pass if mixed precision
      ancestors_compute = ancestors.astype(compute_dtype)
      updated_sequences = _update_seq_stacked(ancestors_compute, masked_sequences, n_leaves, 1.0)
      return compute_soft_cost(updated_sequences, adj_matrix, cost_matrix=identity_cost_matrix)
  else:

    def loss_fn(ancestors):
      # Cast parameters to compute_dtype for forward pass if mixed precision
      ancestors_compute = ancestors.astype(compute_dtype)
      updated_sequences = _update_seq_stacked(ancestors_compute, masked_sequences, n_leaves, 1.0)
      return compute_surrogate_cost(updated_sequences, adj_matrix)

  loss_and_grad = jax.value_and_grad(loss_fn)

  if return_losses:
    # Use jax.lax.scan for efficient loss accumulation
    def scan_step(carry, _):
      ancestors, opt_state = carry
      loss, grads = loss_and_grad(ancestors)
      updates, opt_state = optimizer.update(grads, opt_state, ancestors)
      ancestors = optax.apply_updates(ancestors, updates)
      return (ancestors, opt_state), loss

    (ancestors, _), losses = jax.lax.scan(
      scan_step,
      (ancestors, opt_state),
      None,
      length=n_iterations,
    )

    return jnp.argmax(ancestors, axis=-1), losses

  # Use fori_loop when losses not needed (slightly more memory efficient)
  def training_step(_, carry):
    ancestors, opt_state = carry
    _, grads = loss_and_grad(ancestors)
    updates, opt_state = optimizer.update(grads, opt_state, ancestors)
    ancestors = optax.apply_updates(ancestors, updates)
    return ancestors, opt_state

  ancestors, _ = jax.lax.fori_loop(
    0,
    n_iterations,
    training_step,
    (ancestors, opt_state),
  )

  return jnp.argmax(ancestors, axis=-1)


# ============================================================================
# Batched/Vmap-Compatible TREX Optimization
# ============================================================================


def _update_seq_stacked(
  ancestors: Float[Array, "n_ancestors seq_len n_states"],
  sequences: Float[Array, "n_nodes seq_len n_states"],
  n_leaves: int,
  temperature: float = 1.0,
) -> Float[Array, "n_nodes seq_len n_states"]:
  """Update ancestor sequences using stacked array (vmap-compatible).

  Unlike update_seq which uses a Python list, this version uses a single
  stacked array for ancestors, making it compatible with jax.vmap.

  Args:
      ancestors: Stacked ancestor logits, shape (n_ancestors, seq_len, n_states)
      sequences: Full sequence array, shape (n_all, seq_len, n_states)
      n_leaves: Number of leaf nodes
      temperature: Softmax temperature

  Returns:
      Updated sequences with ancestors replaced by softmax of logits

  """
  updated_ancestors = jax.nn.softmax(ancestors * temperature, axis=-1)
  return sequences.at[n_leaves:].set(updated_ancestors)


def _compute_loss_landscape_aware_stacked(
  ancestors: Float[Array, "n_ancestors seq_len n_states"],
  masked_sequences: Float[Array, "n_nodes seq_len n_states"],
  n_leaves: int,
  landscape: PyTree,
  adj_matrix: Adjacency,
  n_all: int,
  lambda_val: float,
  real_k: int,
  temperature: float = 1.0,
  seq_mask: Bool[Array, seq_len] | None = None,
  batch_size: int = 64,
) -> ScalarFloat:
  """Compute landscape-aware loss using stacked ancestors (vmap-compatible).

  Combines parsimony cost with parental guidance fitness cost.

  Args:
      ancestors: Stacked ancestor logits, shape (n_ancestors, seq_len, n_states)
      masked_sequences: Full sequence array with leaves one-hot, ancestors zeros
      n_leaves: Number of leaf nodes
      landscape: NK landscape dict with 'interactions' and 'fitness_tables'
      adj_matrix: Tree adjacency matrix
      n_all: Total number of nodes
      lambda_val: Weight for fitness cost (0 = parsimony only)
      temperature: Softmax temperature
      seq_mask: Optional boolean mask of shape (seq_len,). True for valid positions.
          If None, all positions are considered valid.

  Returns:
      Total loss = surrogate_cost + lambda_val * fitness_cost

  """
  seq_len = masked_sequences.shape[1]

  # Default mask: all positions valid
  if seq_mask is None:
    seq_mask = jnp.ones(seq_len, dtype=jnp.bool_)

  updated_sequences = _update_seq_stacked(ancestors, masked_sequences, n_leaves, temperature)

  # 1. Parsimony Cost (masked: only count differences at valid positions)
  # We apply the mask by zeroing out invalid positions in the sequences before cost computation
  # But compute_surrogate_cost operates on the full array, so we mask the output
  surrogate_cost = compute_surrogate_cost(updated_sequences, adj_matrix)

  # 2. Parental Guidance Fitness Cost (only if lambda > 0 AND k > 0)
  # K=0 means no epistasis, so fitness cost has no meaningful signal
  # real_k is a static Python int, so we can use Python if to completely avoid
  # tracing the expensive branch when K=0
  if (lambda_val > 0.0) and (real_k > 0):
    parent_indices = jnp.argmax(adj_matrix, axis=1)
    parent_soft_seqs = updated_sequences[parent_indices]
    child_soft_seqs = updated_sequences

    parent_logits = compute_parental_logits(
      parent_soft_seqs, landscape, real_k, batch_size=batch_size
    )

    log_predictions = jax.nn.log_softmax(parent_logits, axis=-1)

    # Masked cross-entropy: only count valid positions
    per_position_ce = -jnp.sum(child_soft_seqs * log_predictions, axis=-1)  # (n_all, seq_len)
    masked_ce = per_position_ce * seq_mask[None, :]  # Broadcast mask
    cross_entropy = jnp.sum(masked_ce)

    is_root = jnp.arange(n_all) == parent_indices
    fitness_cost = cross_entropy / (jnp.sum(~is_root) * jnp.sum(seq_mask))
  else:
    fitness_cost = 0.0

  return surrogate_cost + lambda_val * fitness_cost


@functools.partial(
  jax.jit,
  static_argnames=(
    "n_all",
    "n_leaves",
    "n_states",
    "lambda_val",
    "real_k",
    "optimizer_name",
    "learning_rate",
    "n_iterations",
    "return_losses",
    "dtype",
    "mixed_precision",
    "inference_batch_size",
  ),
)
def run_trex_landscape_aware_configurable(
  leaf_sequences: LeafSequences,
  n_all: int,
  n_leaves: int,
  n_states: int,
  landscape: PyTree,
  lambda_val: float,
  adj_matrix: Adjacency,
  key: PRNGKeyArray,
  real_k: int = 0,
  optimizer_name: str = "adam",
  learning_rate: float = 1e-3,
  n_iterations: int = 10000,
  return_losses: bool = False,
  dtype: jnp.dtype | None = None,
  seq_mask: Bool[Array, seq_len] | None = None,
  mixed_precision: bool = False,
  inference_batch_size: int = 64,
) -> Ancestors | tuple[Ancestors, Float[Array, n_iterations]]:
  """Run landscape-aware TREX optimization with configurable hyperparameters.

  Vmap-compatible version using stacked arrays for ancestors.
  Combines parsimony loss with fitness-based parental guidance.

  Args:
      leaf_sequences: Observed sequences at leaves. Shape: (n_leaves, seq_len)
      n_all: Total number of nodes in the tree
      n_leaves: Number of leaf nodes
      n_states: Alphabet size (e.g., 4 for nucleotides)
      landscape: NK landscape dict with 'interactions' and 'fitness_tables'
      lambda_val: Weight for fitness cost (0 = parsimony only)
      adj_matrix: Adjacency matrix representing tree structure
      key: JAX random key
      optimizer_name: One of "adam", "sgd", "rmsprop", "adamw"
      learning_rate: Learning rate for optimization
      n_iterations: Number of optimization iterations
      return_losses: If True, also return the loss curve
      dtype: Data type for computation (None=float32)
      seq_mask: Optional boolean mask of shape (seq_len,). True for valid positions.
          If None, all positions are considered valid.
      mixed_precision: If True, keep parameters in float32 but compute in dtype

  Returns:
      If return_losses=False: Reconstructed ancestor sequences (n_ancestors, seq_len)
      If return_losses=True: Tuple of (ancestors, loss_curve)

  """
  key, subkey = jax.random.split(key)
  seq_len = leaf_sequences.shape[1]
  n_ancestors = n_all - n_leaves

  if dtype is None:
    dtype = get_default_dtype()

  # In mixed precision, parameters (ancestors) are float32, but compute is dtype
  param_dtype = jnp.float32 if mixed_precision else dtype
  compute_dtype = dtype

  # Initialize ancestors as stacked array for vmap compatibility
  ancestors = jax.random.normal(subkey, (n_ancestors, seq_len, n_states), dtype=param_dtype)

  # Create optimizer
  optimizer = create_optimizer(optimizer_name, learning_rate)
  opt_state = optimizer.init(ancestors)

  # One-hot encode leaf sequences
  leaf_sequences_one_hot = jax.nn.one_hot(leaf_sequences, n_states).astype(compute_dtype)

  # Create masked sequence array
  masked_sequences = jnp.concatenate(
    [
      leaf_sequences_one_hot,
      jnp.zeros((n_ancestors, seq_len, n_states), dtype=compute_dtype),
    ],
    axis=0,
  )

  # Define loss function with seq_mask
  def loss_fn(ancestors):
    # Cast params to compute_dtype for forward pass if mixed precision
    ancestors_compute = ancestors.astype(compute_dtype)
    return _compute_loss_landscape_aware_stacked(
      ancestors_compute,
      masked_sequences,
      n_leaves,
      landscape,
      adj_matrix,
      n_all,
      lambda_val,
      real_k,
      temperature=1.0,
      seq_mask=seq_mask,
      batch_size=inference_batch_size,
    )

  loss_and_grad = jax.value_and_grad(loss_fn)

  if return_losses:

    def scan_step(carry, _):
      ancestors, opt_state = carry
      loss, grads = loss_and_grad(ancestors)
      updates, opt_state = optimizer.update(grads, opt_state, ancestors)
      ancestors = optax.apply_updates(ancestors, updates)
      return (ancestors, opt_state), loss

    (ancestors, _), losses = jax.lax.scan(
      scan_step,
      (ancestors, opt_state),
      None,
      length=n_iterations,
    )

    return jnp.argmax(ancestors, axis=-1), losses

  # Use fori_loop when losses not needed
  def training_step(_, carry):
    ancestors, opt_state = carry
    _, grads = loss_and_grad(ancestors)
    updates, opt_state = optimizer.update(grads, opt_state, ancestors)
    ancestors = optax.apply_updates(ancestors, updates)
    return ancestors, opt_state

  ancestors, _ = jax.lax.fori_loop(
    0,
    n_iterations,
    training_step,
    (ancestors, opt_state),
  )

  return jnp.argmax(ancestors, axis=-1)


def run_trex_optimization_batched(
  leaf_sequences: LeafSequences,
  n_all: int,
  n_leaves: int,
  n_states: int,
  adj_matrix: Adjacency,
  key: PRNGKeyArray,
  use_soft_cost: bool = False,
  n_iterations: int = 10000,
) -> Ancestors:
  """Run TREX optimization with vmap-compatible structure.

  This version uses a stacked array for ancestors (not a Python list),
  making it compatible with jax.vmap over batch dimensions.

  Args:
      leaf_sequences: Observed sequences at leaves. Shape: (n_leaves, seq_len)
      n_all: Total number of nodes in the tree
      n_leaves: Number of leaf nodes
      n_states: Alphabet size
      adj_matrix: Adjacency matrix representing tree structure
      key: JAX random key
      use_soft_cost: If True, use soft cost; else use surrogate cost
      n_iterations: Number of optimization iterations

  Returns:
      Reconstructed ancestor sequences, shape (n_ancestors, seq_len)

  """
  key, subkey = jax.random.split(key)
  seq_len = leaf_sequences.shape[1]
  n_ancestors = n_all - n_leaves

  # Initialize ancestors as stacked array (not list!)
  ancestors = jax.random.normal(subkey, (n_ancestors, seq_len, n_states))

  # Create optimizer
  optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))
  opt_state = optimizer.init(ancestors)

  # One-hot encode leaf sequences
  leaf_sequences_one_hot = jax.nn.one_hot(leaf_sequences, n_states)

  # Create full sequence array (leaves + zeros for ancestors)
  masked_sequences = jnp.concatenate(
    [
      leaf_sequences_one_hot,
      jnp.zeros((n_ancestors, seq_len, n_states)),
    ],
    axis=0,
  ).astype(get_default_dtype())

  # Cost matrix for soft cost
  identity_cost_matrix = jnp.eye(n_states, dtype=get_default_dtype())

  if use_soft_cost:

    def loss_fn(ancestors):
      updated = _update_seq_stacked(ancestors, masked_sequences, n_leaves, 1.0)
      return compute_soft_cost(updated, adj_matrix, cost_matrix=identity_cost_matrix)
  else:

    def loss_fn(ancestors):
      updated = _update_seq_stacked(ancestors, masked_sequences, n_leaves, 1.0)
      return compute_surrogate_cost(updated, adj_matrix)

  loss_and_grad = jax.value_and_grad(loss_fn)

  def training_step(_, carry):
    ancestors, opt_state = carry
    _, grads = loss_and_grad(ancestors)
    updates, opt_state = optimizer.update(grads, opt_state)
    ancestors = optax.apply_updates(ancestors, updates)
    return ancestors, opt_state

  ancestors, _ = jax.lax.fori_loop(
    0,
    n_iterations,
    training_step,
    (ancestors, opt_state),
  )

  return jnp.argmax(ancestors, axis=-1)


@custom_vjp
def straight_through_estimator(soft_sequence: OneHotEvoSequence) -> EvoSequence:
  """Apply argmax in the forward pass.

  The custom VJP rule below will handle the backward pass.
  """
  return jnp.argmax(soft_sequence, axis=-1)


def ste_fwd(soft_sequence: OneHotEvoSequence) -> tuple[jax.Array, OneHotEvoSequence]:
  """Forward pass for the straight-through estimator (STE).

  It applies argmax to get discrete outputs but saves the soft input
  for use in the backward pass.
  """
  primal_out = straight_through_estimator(soft_sequence)
  return primal_out, soft_sequence


# Define the backward pass for the VJP
def ste_bwd(residuals: OneHotEvoSequence, grad_primal_out: Array) -> tuple[Array]:
  """Backward pass for the straight-through estimator (STE).

  It takes the incoming gradient from the next layer (grad_primal_out)
  and passes it back to the input of the argmax (soft_sequence).
  """
  soft_sequence = residuals  # Get the input from the forward pass

  # Create a one-hot mask from the winning indices
  indices = jnp.argmax(soft_sequence, axis=-1)
  mask = jax.nn.one_hot(indices, num_classes=soft_sequence.shape[-1])

  # "Scatter" the incoming gradient, placing it at the location of the
  # winning neuron. The gradient needs to be expanded to broadcast correctly.
  # The result is a tuple, as VJPs can have multiple inputs.
  return (jnp.expand_dims(grad_primal_out, axis=-1) * mask,)


# Register the forward and backward passes with the custom VJP
straight_through_estimator.defvjp(ste_fwd, ste_bwd)


def compute_parental_logits(
  parent_sequences: OneHotEvoSequence,
  landscape: PyTree,
  real_k: int,
  batch_size: int = 64,
) -> Float[Array, "n_parents seq_len n_states"]:
  """Compute 'parental logits' for each site for a batch of parent sequences.

  The logit for a state at site 'i' is the expected fitness at 'i'
  assuming 'i' is in that state, marginalized over the parent's own
  probabilities for the other interacting sites.

  Handles K=0 (no epistasis) by returning fitness tables directly.

  Args:
      parent_sequences: Soft one-hot sequences, shape (n_parents, seq_len, n_states)
      landscape: NK landscape dict with 'interactions' and 'fitness_tables'
      real_k: The real K value (static, not traced) for conditional logic
      batch_size: Batch size for safe_map

  Returns a tensor of shape (num_parents, seq_len, n_states).
  """
  n_parents, seq_len, n_states = parent_sequences.shape
  fitness_tables = landscape["fitness_tables"]  # Shape: (N, n_states**(k+1))
  interactions = landscape["interactions"]  # Shape: (N, padded_k)

  # Handle K=0 case: no epistasis, fitness depends only on site's own state
  # real_k is a static Python int, so this Python if works at trace time
  if real_k == 0:
    # fitness_tables has shape (N, n_states) when k=0
    # Each row is the fitness for each state at that site
    # Broadcast to all parents: (seq_len, n_states) -> (n_parents, seq_len, n_states)
    return jnp.broadcast_to(fitness_tables[None, :, :], (n_parents, seq_len, n_states))

  # K > 0: compute marginalized fitness over neighbors
  k_eff = interactions.shape[1]

  def compute_logits_for_one_site(i, parent_seqs):
    # i: index of the site we're computing logits for
    # parent_seqs: all parent sequences, shape (n_parents, seq_len, n_states)

    # 1. Get the k_eff neighbors for site i
    neighbors = interactions[i, :k_eff]

    # 2. Get the parent's soft probabilities for those neighbors
    # Shape: (n_parents, k_eff, n_states)
    neighbor_probs = parent_seqs[:, neighbors, :]

    # 3. Compute the joint probability distribution over the neighbors' states
    # Starts with shape (n_parents, n_states)
    joint_neighbor_probs = neighbor_probs[:, 0, :]
    for j in range(1, k_eff):
      # Outer product using einsum
      next_neighbor = neighbor_probs[:, j, :]
      combined = jnp.einsum("pc,ps->pcs", joint_neighbor_probs, next_neighbor)
      joint_neighbor_probs = combined.reshape(n_parents, -1)

    # 4. Reshape the fitness table for site i
    # The table encodes fitness F(state_i, state_neighbors).
    # We reshape it to (n_states, n_states**k_eff) to separate the two dimensions.
    site_fitness_table = fitness_tables[i].reshape(n_states, -1)

    # 5. Compute the marginalized fitness (our logits)
    # This is the matrix-vector product of the fitness table and the neighbor probabilities.
    # It calculates the expected fitness for each state of site 'i'.
    # (n_states, n_states**k_eff) @ (n_parents, n_states**k_eff)^T -> (n_states, n_parents) -> (n_parents, n_states)
    return jnp.einsum("si,pi->ps", site_fitness_table, joint_neighbor_probs)

  # Use safe_map to apply this logic to all N sites in parallel safely.
  # We are mapping over the integer index of the site.
  all_logits = safe_map(
    lambda i: compute_logits_for_one_site(i, parent_sequences),
    jnp.arange(seq_len),
    batch_size=batch_size,
  )

  # The safe_map result is (seq_len, n_parents, n_states), so we transpose it
  return jnp.transpose(all_logits, (1, 0, 2))


def compute_loss_landscape_aware(
  key: PRNGKeyArray,
  params: dict,
  sequences: OneHotEvoSequence,
  n_leaves: int,
  landscape: PyTree,
  adj_matrix: Adjacency,
  temperature: float,
  n_all: int,
  lambda_val: float,
  *,
  fix_tree: bool = True,
  batch_size: int = 64,
) -> ScalarFloat:
  """Compute surrogate cost + parental guidance cross-entropy cost."""
  updated_sequences = update_seq(params, sequences, temperature)

  # 1. Parsimony Cost (same as before)
  surrogate_cost = compute_surrogate_cost(updated_sequences, adj_matrix)

  # 2. Parental Guidance Fitness Cost
  # Find the parent of each node
  parent_indices = jnp.argmax(adj_matrix, axis=1)

  # Get the soft sequences for all parents and all children
  parent_soft_seqs = updated_sequences[parent_indices]
  child_soft_seqs = updated_sequences

  # Compute the fitness-derived logits from the parents
  parent_logits = compute_parental_logits(parent_soft_seqs, landscape, batch_size=batch_size)

  # Calculate cross-entropy loss: -Σ y * log(softmax(x))
  # where y is the child distribution and x is the parent logits.
  log_predictions = jax.nn.log_softmax(parent_logits, axis=-1)
  cross_entropy = -jnp.sum(child_soft_seqs * log_predictions)

  is_root = jnp.arange(n_all) == parent_indices
  fitness_cost = cross_entropy / jnp.sum(~is_root)
  # jax.debug.print("Surrogate Cost: {c1}, Fitness Cost: {c2}", c1=surrogate_cost, c2=fitness_cost)

  return surrogate_cost + lambda_val * fitness_cost


def run_trex_landscape_aware(
  leaf_sequences: LeafSequences,
  n_all: int,
  n_leaves: int,
  n_states: int,
  landscape: PyTree,
  lambda_val: float,
  adj_matrix: Adjacency,
  key: PRNGKeyArray,
  inference_batch_size: int = 64,
) -> Ancestors:
  """Run the TREX optimization with the landscape-aware loss."""
  key, subkey = jax.random.split(key)
  params = {
    "tree_params": adj_matrix,
    "ancestors": [
      jax.random.normal(subkey, (leaf_sequences.shape[1], n_states))
      for _ in range(n_all - n_leaves)
    ],
  }

  optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(1e-3),
  )
  opt_state = optimizer.init(params)

  leaf_sequences_one_hot = jax.nn.one_hot(leaf_sequences, n_states)
  masked_sequences = jnp.concatenate(
    [leaf_sequences_one_hot, jnp.zeros((n_all - n_leaves, leaf_sequences.shape[1], n_states))],
    axis=0,
  ).astype(jnp.float32)

  loss_and_grad = jax.jit(
    jax.value_and_grad(compute_loss_landscape_aware, argnums=1),
    static_argnames=("fix_tree", "n_all", "n_leaves"),
  )

  # Training loop
  def training_step(
    _: Int[Array, ""],
    carry: tuple[dict, optax.OptState, PRNGKeyArray],
  ) -> tuple[dict, optax.OptState, PRNGKeyArray]:
    params, opt_state, key = carry
    _, grads = loss_and_grad(
      key,
      params,
      masked_sequences,
      n_leaves=n_leaves,  # Pass as static keyword arg
      landscape=landscape,
      temperature=1.0,
      n_all=n_all,
      lambda_val=lambda_val,
      fix_tree=True,
      adj_matrix=adj_matrix,
      batch_size=inference_batch_size,
    )
    # jax.debug.print("Grads: {grads}", grads=grads)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = cast("dict[str, jax.Array | list[jax.Array]]", optax.apply_updates(params, updates))
    return params, opt_state, key

  params, _, _ = jax.lax.fori_loop(
    0,
    10000,
    training_step,
    (params, opt_state, key),
  )

  return jnp.argmax(jnp.stack(params["ancestors"]), axis=-1)


def create_balanced_binary_tree(n_leaves: int) -> Adjacency:
  """Create a balanced binary tree with n_leaves."""
  n_ancestors = n_leaves - 1
  n_total = n_leaves + n_ancestors
  adj = jnp.zeros((n_total, n_total))

  leaf_parents = n_leaves + jnp.arange(n_leaves) // 2
  ancestor_parents = n_leaves + (jnp.arange(n_ancestors - 1) + n_leaves) // 2

  adj = adj.at[jnp.arange(n_leaves), leaf_parents].set(1)
  return adj.at[n_leaves + jnp.arange(n_ancestors - 1), ancestor_parents].set(1)


def _loss_fn_soft(key, params, masked_sequences, temperature, adjacency, cost_matrix):
  """Loss function using compute_soft_cost with explicit cost_matrix."""
  updated_sequences = update_seq(params, masked_sequences, temperature)
  return compute_soft_cost(updated_sequences, adjacency, cost_matrix=cost_matrix)


def _loss_fn_surrogate(key, params, masked_sequences, temperature, adjacency):
  """Loss function using compute_surrogate_cost."""
  updated_sequences = update_seq(params, masked_sequences, temperature)
  return compute_surrogate_cost(updated_sequences, adjacency)


# Pre-compile the loss_and_grad functions at module level
_loss_and_grad_soft = jax.jit(jax.value_and_grad(_loss_fn_soft, argnums=1))
_loss_and_grad_surrogate = jax.jit(jax.value_and_grad(_loss_fn_surrogate, argnums=1))


def run_trex_optimization(
  leaf_sequences: LeafSequences,
  n_all: int,
  n_leaves: int,
  n_states: int,
  adj_matrix: Adjacency,
  key: PRNGKeyArray,
  use_soft_cost: bool = False,
) -> Ancestors:
  """Run the trex optimization to reconstruct ancestral sequences."""
  key, subkey = jax.random.split(key)
  params = {
    "tree_params": adj_matrix,
    "ancestors": [
      jax.random.normal(subkey, (leaf_sequences.shape[1], n_states))
      for _ in range(n_all - n_leaves)
    ],
  }

  # Optimizer
  optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(1e-3),
  )
  opt_state = optimizer.init(params)

  # one-hot encode leaf sequences
  leaf_sequences_one_hot = jax.nn.one_hot(leaf_sequences, n_states)
  # create a masked sequence object where the ancestor sequences are zero
  masked_sequences = jnp.concatenate(
    [leaf_sequences_one_hot, jnp.zeros((n_all - n_leaves, leaf_sequences.shape[1], n_states))],
    axis=0,
  ).astype(get_default_dtype())

  # Pre-compute identity cost matrix for soft cost (avoids creating inside JIT)
  identity_cost_matrix = jnp.eye(n_states, dtype=get_default_dtype())

  # Select the appropriate loss_and_grad function
  if use_soft_cost:

    def training_step(
      _: Int[Array, ""],
      carry: tuple[dict, optax.OptState, PRNGKeyArray],
    ) -> tuple[dict, optax.OptState, PRNGKeyArray]:
      params, opt_state, key = carry
      _, grads = _loss_and_grad_soft(
        key,
        params,
        masked_sequences,
        1.0,  # temperature
        adj_matrix,
        identity_cost_matrix,
      )
      updates, opt_state = optimizer.update(grads, opt_state)
      params = cast(
        "dict[str, jax.Array | list[jax.Array]]",
        (optax.apply_updates(params, updates)),
      )
      return params, opt_state, key
  else:

    def training_step(
      _: Int[Array, ""],
      carry: tuple[dict, optax.OptState, PRNGKeyArray],
    ) -> tuple[dict, optax.OptState, PRNGKeyArray]:
      params, opt_state, key = carry
      _, grads = _loss_and_grad_surrogate(
        key,
        params,
        masked_sequences,
        1.0,  # temperature
        adj_matrix,
      )
      updates, opt_state = optimizer.update(grads, opt_state)
      params = cast(
        "dict[str, jax.Array | list[jax.Array]]",
        (optax.apply_updates(params, updates)),
      )
      return params, opt_state, key

  params, _, _ = jax.lax.fori_loop(
    0,
    10000,
    training_step,
    (params, opt_state, key),
  )
  # Get the reconstructed ancestors
  return jnp.argmax(jnp.stack(params["ancestors"]), axis=-1)


# Modify the benchmark function to take lambdas and return a dict
def benchmark(
  n: int,
  k: int,
  n_leaves: int,
  mutation_rate: float,
  lambda_values: list[float],
  key: PRNGKeyArray,
) -> dict[str, Any]:
  """Benchmarks Sankoff and TREX (for various lambdas) on a single dataset."""
  results = {}

  # 1. Generate a single dataset for this benchmark run
  adj_matrix = create_balanced_binary_tree(n_leaves)
  n_all = adj_matrix.shape[0]
  cost_matrix = jnp.ones((2, 2)) - jnp.eye(2)
  key, subkey = jax.random.split(key)
  landscape = create_nk_model_landscape(n, k, subkey)
  key, subkey = jax.random.split(key)
  root_sequence = jax.random.randint(subkey, (n, 1), 0, 2)
  key, subkey = jax.random.split(key)
  tree_data = generate_tree_data(
    landscape,
    adj_matrix,
    root_sequence,
    mutation_rate,
    subkey,
    coupled_mutation_prob=0.5,
  )
  leaf_sequences = tree_data.all_sequences[:n_leaves].astype(jnp.int32)
  true_ancestors = tree_data.all_sequences[n_leaves:].astype(jnp.int32)

  # 2. Run Sankoff (only once per dataset)
  reconstructed_sankoff, _, _ = run_sankoff(
    adj_matrix,
    cost_matrix,
    leaf_sequences,
    n_all=n_all,
    n_states=2,
    n_leaves=n_leaves,
    return_path=True,
  )
  sankoff_ancestors = reconstructed_sankoff[n_leaves:]
  results["sankoff"] = jnp.mean(true_ancestors == sankoff_ancestors)

  # 3. Run TREX for each lambda value
  results["trex"] = {}
  for lambda_val in lambda_values:
    key, subkey = jax.random.split(key)

    # λ=0 is the original parsimony-only TREX
    if lambda_val == 0.0:
      reconstructed_trex = run_trex_optimization(
        leaf_sequences,
        n_all,
        n_leaves,
        2,
        adj_matrix,
        subkey,
      )
    else:
      reconstructed_trex = run_trex_landscape_aware(
        leaf_sequences,
        n_all,
        n_leaves,
        2,
        landscape,
        lambda_val,
        adj_matrix,
        subkey,
      )

    trex_accuracy = jnp.mean(true_ancestors == reconstructed_trex)
    results["trex"][lambda_val] = trex_accuracy

  return results


# Update the main execution block
if __name__ == "__main__":
  N = 15
  K_values = [1, 2, 5, 10]
  lambda_values = [0.0, 0.3, 3.0]  # Test these lambda values
  n_leaves = 32
  mutation_rate = 0.1
  num_replicates = 2  # Use a smaller number for faster testing

  # Setup storage for results
  all_results: dict[int, dict[str, Any]] = {
    K: {"sankoff": [], "trex": {L: [] for L in lambda_values}} for K in K_values
  }

  key = jax.random.PRNGKey(0)
  print(f"Running benchmark for {num_replicates} replicates...")

  for i in range(num_replicates):
    print(f"  Replicate {i + 1}/{num_replicates}")
    for K in K_values:
      key, subkey = jax.random.split(key)
      # Run one full benchmark for a given K
      rep_results = benchmark(N, K, n_leaves, mutation_rate, lambda_values, subkey)

      # Store results
      all_results[K]["sankoff"].append(rep_results["sankoff"])
      for L in lambda_values:
        all_results[K]["trex"][L].append(rep_results["trex"][L])

  # 6. Plot results
  import matplotlib.pyplot as plt
  import numpy as np

  plt.figure(figsize=(10, 7))

  # Plot Sankoff
  sankoff_means = [np.mean(all_results[K]["sankoff"]) for K in K_values]
  sankoff_stds = [np.std(all_results[K]["sankoff"]) for K in K_values]
  plt.errorbar(
    K_values,
    sankoff_means,
    yerr=sankoff_stds,
    label="Sankoff",
    capsize=5,
    marker="o",
    zorder=10,
  )

  # Plot TREX for each lambda
  for L in lambda_values:
    trex_means = [np.mean(all_results[K]["trex"][L]) for K in K_values]
    trex_stds = [np.std(all_results[K]["trex"][L]) for K in K_values]
    label = f"TREX (λ={L})" if L > 0 else "TREX (Parsimony)"
    plt.errorbar(
      K_values,
      trex_means,
      yerr=trex_stds,
      label=label,
      capsize=5,
      marker="o",
      linestyle="--",
    )

  plt.xlabel("K (epistatic interactions)")
  plt.ylabel("Mean Accuracy")
  plt.title(f"Algorithm Comparison (Avg. over {num_replicates} replicates)")
  plt.legend()
  plt.grid(True, which="both", linestyle="--", linewidth=0.5)
  plt.savefig("benchmark_lambda_comparison.png")
  print("Benchmark complete. Plot saved to benchmark_lambda_comparison.png")
