"""JAX implementation of the Sankoff algorithm for phylogenetic reconstruction."""

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from trex.types import get_default_dtype
from trex.utils.types import (
  AdjacencyMatrix,
  BacktrackingTable,
  BatchEvoSequence,
  CostMatrix,
  DPTable,
  Node,
  ReconstructedSequence,
  State,
  TotalCost,
  VmappedDPTable,
)


@jax.jit
def run_dp(
  adjacency_matrix: AdjacencyMatrix,
  dynamic_programming_table: DPTable,
  backtracking_table: BacktrackingTable,
  sequences: BatchEvoSequence,
  cost_matrix: CostMatrix,
) -> tuple[DPTable, BacktrackingTable]:
  """Run Sankoff's algorithm using dynamic programming over a tree topology.

  Args:
      adjacency_matrix: The adjacency matrix representing the tree structure.
      dynamic_programming_table: The dynamic programming table, initialized with high costs.
      backtracking_table: A table to store backtracking information.
      sequences: The sequences of the leaf nodes.
      cost_matrix: The cost matrix for character substitutions.

  Returns:
      A tuple containing the filled DP table and the backtracking table.

  """
  n_all = adjacency_matrix.shape[0]
  n_leaves = (n_all + 1) // 2

  # Initialize DP table for leaf nodes
  for i in range(n_leaves):
    dynamic_programming_table = dynamic_programming_table.at[i, sequences[i].astype(jnp.int32)].set(
      0,
    )

  # Iterate through ancestor nodes in topological order
  def body_fun(
    node: int,
    val: tuple[jax.Array, jax.Array],
  ) -> tuple[jax.Array, jax.Array]:
    dp_val, dp_nodes_val = val
    children = jnp.where(adjacency_matrix[:, node] == 1, size=2, fill_value=-1)[0]

    def child_cost_fun(
      carry: jax.Array,
      child_idx: jax.Array,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
      child_node = children[child_idx]
      cost_array = cost_matrix + dp_val[child_node, :]
      cost = jnp.min(cost_array, axis=1)
      char = jnp.argmin(cost_array, axis=1)
      return carry + cost, (child_node, char)

    # Calculate costs for children
    total_cost, (child_node_arr, child_char_arr) = jax.lax.scan(
      child_cost_fun,
      jnp.zeros_like(dp_val[0]),
      jnp.arange(2, dtype=jnp.int32),
    )

    dp_val = dp_val.at[node, :].set(total_cost)
    dp_nodes_val = dp_nodes_val.at[node, :, 0].set(child_node_arr[0])  # Child 1 Node
    dp_nodes_val = dp_nodes_val.at[node, :, 1].set(child_char_arr[0])  # Child 1 States
    dp_nodes_val = dp_nodes_val.at[node, :, 2].set(child_node_arr[1])  # Child 2 Node
    dp_nodes_val = dp_nodes_val.at[node, :, 3].set(child_char_arr[1])  # Child 2 States

    return dp_val, dp_nodes_val

  dynamic_programming_table, backtracking_table = jax.lax.fori_loop(
    n_leaves,
    n_all,
    body_fun,
    (dynamic_programming_table, backtracking_table),
  )

  return dynamic_programming_table, backtracking_table


vectorized_dp = jax.vmap(run_dp, (None, 0, 0, 1, None), 0)


if TYPE_CHECKING:

  def run_sankoff(
    adjacency_matrix: AdjacencyMatrix,
    cost_matrix: CostMatrix,
    sequences: BatchEvoSequence,
    n_all: int,
    n_states: int,
    n_leaves: int,
    *,
    return_path: bool = False,
  ) -> tuple[BatchEvoSequence, VmappedDPTable, TotalCost]: ...
else:

  @partial(jax.jit, static_argnames=("return_path", "n_all", "n_states", "n_leaves"))
  def run_sankoff(
    adjacency_matrix: AdjacencyMatrix,
    cost_matrix: CostMatrix,
    sequences: BatchEvoSequence,
    n_all: int,
    n_states: int,
    n_leaves: int,
    *,
    return_path: bool = False,
  ) -> tuple[BatchEvoSequence, VmappedDPTable, TotalCost]:
    """Run the Sankoff algorithm over a batch of sequences.

    Args:
        adjacency_matrix: The adjacency matrix of the tree.
        cost_matrix: The substitution cost matrix.
        sequences: A batch of leaf sequences.
        n_all (int): The total number of nodes in the tree.
        n_states (int): The number of possible states (characters).
        n_leaves (int): The number of leaf nodes.
        return_path: If True, reconstructs and returns the ancestral sequences.

    Returns:
        A tuple containing the reconstructed sequences (if requested),
        the final DP table, and the total parsimony score.

    """
    adjacency_matrix = adjacency_matrix.at[-1, -1].set(0)  # Remove self-connection at the root

    # Ensure correct dtypes
    adjacency_matrix = adjacency_matrix.astype(get_default_dtype())
    sequences = sequences.astype(get_default_dtype())
    cost_matrix = cost_matrix.astype(get_default_dtype())

    sequence_length = sequences.shape[1]

    # Initialize DP tables
    dp_nodes = jnp.zeros((sequence_length, n_all, n_states, 4), dtype=get_default_dtype())
    dp = jnp.full((sequence_length, n_all, n_states), 1e5, dtype=get_default_dtype())

    dp, backtracking_connections = vectorized_dp(
      adjacency_matrix,
      dp,
      dp_nodes,
      sequences,
      cost_matrix,
    )
    reconstructed_sequences = jnp.zeros((n_all, sequence_length), dtype=get_default_dtype())
    reconstructed_sequences = reconstructed_sequences.at[:n_leaves, :].set(sequences[:n_leaves])
    if return_path:
      root_node = jnp.asarray(adjacency_matrix.shape[0] - 1, jnp.int32)

      vmapped_backtrack = jax.vmap(
        backtrack_sankoff_jit,
        in_axes=(None, 0, 0, None, None),
        out_axes=1,
      )

      all_root_states = jnp.argmin(dp[:, root_node, :], axis=1).astype(jnp.int32)

      all_reconstructed_chars = vmapped_backtrack(
        root_node,
        all_root_states,
        backtracking_connections,
        n_all,
        n_leaves,
      )

      ancestor_chars = all_reconstructed_chars[n_leaves:]
      reconstructed_sequences = reconstructed_sequences.at[n_leaves:, :].set(
        ancestor_chars,
      )

    total_cost = dp[:, -1].min(axis=1).sum()
    return reconstructed_sequences, dp, total_cost


@partial(jax.jit, static_argnames=("n_all", "n_leaves"))
def backtrack_sankoff_jit(
  root_node: Node,
  root_state: State,
  backtracking_table: BacktrackingTable,
  n_all: int,
  n_leaves: int,
) -> ReconstructedSequence:
  """Reconstruct the ancestral sequence for a single position using a JIT-compatible while_loop.

  Args:
      root_node: The index of the root node.
      root_state: The optimal state (character) for the root.
      backtracking_table: The table containing child nodes and states.
      n_all: The total number of nodes in the tree.
      n_leaves: The number of leaf nodes.

  Returns:
      The reconstructed characters for all nodes for one sequence position.

  """
  initial_stack = jnp.zeros((n_all, 2), dtype=jnp.int32)
  initial_stack = initial_stack.at[0].set(jnp.array([root_node, root_state]))
  initial_stack_ptr = 1
  reconstructed_chars = jnp.zeros(n_all, dtype=jnp.int32)

  loop_state = (initial_stack, initial_stack_ptr, reconstructed_chars)

  # 2. Define the condition function for the while_loop
  def cond_fun(val: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
    """Loop continues as long as the stack is not empty."""
    _, stack_ptr, _ = val
    return stack_ptr > 0

  def body_fun(
    val: tuple[jax.Array, jax.Array, jax.Array],
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Perform one step of the backtracking traversal."""
    stack, stack_ptr, reconstructed = val

    # Pop a (node, state) pair from the top of the stack
    current_stack_ptr = stack_ptr - 1
    node, state = stack[current_stack_ptr]

    # Use jax.lax.cond for conditional logic inside a JIT context
    def process_ancestor(recon: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
      new_recon = recon.at[node].set(state)

      # Get children from the backtracking table
      child_info = backtracking_table[node, state]
      child1_node, child1_state = child_info[0], child_info[1]
      child2_node, child2_state = child_info[2], child_info[3]

      # Push children onto the stack
      new_stack = stack.at[current_stack_ptr].set(jnp.array([child1_node, child1_state]))
      new_stack = new_stack.at[current_stack_ptr + 1].set(jnp.array([child2_node, child2_state]))

      # The stack grew by one (popped 1, pushed 2)
      new_ptr = current_stack_ptr + 2

      return new_stack, new_ptr, new_recon

    def process_leaf(recon: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
      return stack, current_stack_ptr, recon

    # The branch must be decided by a traced value (node)
    return jax.lax.cond(
      node >= n_leaves,
      process_ancestor,  # Function to run if true
      process_leaf,  # Function to run if false
      reconstructed,  # Argument to pass to the function
    )

  # 4. Run the while_loop
  _, _, final_chars = jax.lax.while_loop(cond_fun, body_fun, loop_state)  # pyright: ignore[reportArgumentType]

  return final_chars
