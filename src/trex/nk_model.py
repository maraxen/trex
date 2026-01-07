"""NK model implementation for simulating sequence evolution on a fitness landscape."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from .utils.types import (
  EvoSequence,
  PhylogeneticTree,
)


def create_nk_model_landscape(n: int, k: int, key: Array, n_states: int = 2) -> PyTree:
  """Create a random fitness landscape for an NK model.

  Args:
      n: The length of the sequence.
      k: The number of epistatic interactions.
      key: A JAX random key.
      n_states: The alphabet size (q).

  Returns:
      A PyTree representing the fitness landscape.
      Contains 'interactions' (N, K) and 'fitness_tables' (N, q, q, ..., q) flattened.

  """
  key, subkey = jax.random.split(key)
  interactions = jax.random.randint(subkey, (n, k), 0, n)

  key, subkey = jax.random.split(key)
  # The table size for each site is q^(k+1)
  fitness_tables = jax.random.uniform(subkey, (n, n_states ** (k + 1)))

  return {
    "interactions": interactions,
    "fitness_tables": fitness_tables,
    "n_states": n_states,
    "k": k,
  }


def get_fitness(sequence: EvoSequence, landscape: PyTree) -> Float:
  """Calculate the fitness of a sequence on a given landscape.

  Args:
      sequence: The sequence to evaluate.
      landscape: The fitness landscape.

  Returns:
      The fitness of the sequence.

  """
  n_states = landscape["n_states"]
  # k = landscape["k"] # Not strictly needed if we trust the table size, but good for clarity if needed

  def site_fitness_contribution(
    i: jax.Array,
    seq: EvoSequence,
    interactions: Array,
    fitness_tables: Array,
  ) -> Float:
    interacting_sites = jnp.append(jnp.array([i]), interactions[i])
    interacting_states = seq[interacting_sites]

    # Calculate index in base-q: sum(state[j] * q^j)
    powers = n_states ** jnp.arange(len(interacting_states))
    index = jnp.sum(interacting_states * powers)

    return fitness_tables[i, index]

  # Vectorize the site fitness contribution function over all sites
  vectorized_site_fitness = jax.vmap(
    site_fitness_contribution,
    in_axes=(0, None, None, None),
  )

  # Calculate the fitness contributions for all sites
  all_site_fitness = vectorized_site_fitness(
    jnp.arange(len(sequence)),
    sequence,
    landscape["interactions"],
    landscape["fitness_tables"],
  )

  # Return the mean fitness
  return jnp.mean(all_site_fitness)


batched_get_fitness = jax.vmap(get_fitness, in_axes=(0, None))


def generate_tree_data(
  landscape: PyTree,
  adjacency: Array,
  root_sequence: EvoSequence,
  mutation_rate: float,
  key: Array,
  coupled_mutation_prob: float = 0.5,
  n_states: int = 20,
) -> PhylogeneticTree:
  """Generate a tree of sequences using the NK model.

  Args:
      landscape: The NK model fitness landscape.
      adjacency: An adjacency matrix representing the tree structure.
      root_sequence: The sequence at the root of the tree.
      mutation_rate: The probability of a mutation at each site.
      key: A JAX random key.
      coupled_mutation_prob: The probability of performing a coupled mutation.
      n_states: The alphabet size for mutations. Should match the landscape's n_states.

  Returns:
      A PhylogeneticTree object containing the generated sequences.

  """
  n_nodes = adjacency.shape[0]
  seq_length = len(root_sequence)

  # Ensure input n_states matches landscape if present
  # (Fallback for backward compatibility or loose dicts)
  if "n_states" in landscape:
    # We cast to int to avoid JAX/numpy scalar issues if it's an array
    n_states = int(landscape["n_states"])

  # Get the topological sort of the nodes using a breadth-first search (BFS)
  parent_indices = jnp.argmax(adjacency, axis=1)
  root_node = jnp.where(parent_indices == jnp.arange(n_nodes), size=1)[0][0]

  def bfs_cond_fun(state: tuple[Array, Array, Array]) -> jax.Array:
    queue, _, _ = state
    return jnp.any(queue != -1)

  def bfs_body_fun(state: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
    queue, visited, sorted_nodes = state
    current_node = queue[0]
    queue = queue.at[0].set(-1)
    queue = jnp.roll(queue, -1)

    sorted_nodes = sorted_nodes.at[jnp.sum(visited)].set(current_node)
    visited = visited.at[current_node].set(True)

    children = jnp.where(adjacency[:, current_node] == 1, size=n_nodes)[0]

    def add_child_to_queue(i: jax.Array, q: jax.Array) -> jax.Array:
      child = children[i]
      q = jax.lax.cond(
        ~visited[child],
        lambda: q.at[jnp.sum(q != -1)].set(child),
        lambda: q,
      )
      return q

    queue = jax.lax.fori_loop(0, n_nodes, add_child_to_queue, queue)
    return queue, visited, sorted_nodes

  initial_queue = jnp.full((n_nodes,), -1).at[0].set(root_node)
  initial_visited = jnp.zeros((n_nodes,), dtype=bool)
  initial_sorted_nodes = jnp.full((n_nodes,), -1)

  _, _, sorted_nodes = jax.lax.while_loop(
    bfs_cond_fun,
    bfs_body_fun,
    (initial_queue, initial_visited, initial_sorted_nodes),
  )

  all_sequences = jnp.zeros((n_nodes, seq_length), dtype=jnp.int32)
  all_sequences = all_sequences.at[root_node].set(jnp.squeeze(root_sequence))

  def body_fun(i: jax.Array, val: tuple[Array, Array]) -> tuple[Array, Array]:
    sequences, current_key = val
    node_index = sorted_nodes[i]
    parent_index = parent_indices[node_index]

    def evolve(parent_sequence: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array]:
      def random_mutation(key: jax.Array, parent_sequence: jax.Array) -> jax.Array:
        # Standard random mutation
        key, subkey = jax.random.split(key)
        mutation_mask = jax.random.bernoulli(subkey, mutation_rate, (seq_length,))
        key, subkey = jax.random.split(key)
        new_values = jax.random.randint(subkey, (seq_length,), 0, n_states)
        return jnp.where(
          mutation_mask,
          new_values,
          parent_sequence,
        )

      def coupled_mutation(key: jax.Array, parent_sequence: jax.Array) -> jax.Array:
        # Coupled mutation on interacting sites
        key, subkey1, subkey2 = jax.random.split(key, 3)
        site_to_mutate = jax.random.randint(subkey1, (), 0, seq_length)
        interacting_sites = landscape["interactions"][site_to_mutate]
        sites_to_mutate = jnp.append(jnp.array([site_to_mutate]), interacting_sites)

        mutation_mask = jnp.zeros_like(parent_sequence, dtype=bool).at[sites_to_mutate].set(True)

        new_values = jax.random.randint(subkey2, (seq_length,), 0, n_states)
        return jnp.where(
          mutation_mask,
          new_values,
          parent_sequence,
        )

      key, subkey = jax.random.split(key)
      mutated_sequence = jax.lax.cond(
        jax.random.bernoulli(subkey, coupled_mutation_prob),
        coupled_mutation,
        random_mutation,
        key,
        parent_sequence,
      )

      # Metropolis-Hastings step
      parent_fitness = get_fitness(parent_sequence, landscape)
      mutated_fitness = get_fitness(mutated_sequence, landscape)
      acceptance_prob = jnp.exp(mutated_fitness - parent_fitness)

      key, subkey = jax.random.split(key)
      accepted = jax.random.bernoulli(subkey, jnp.minimum(1.0, acceptance_prob))

      return jnp.asarray(jnp.where(accepted, mutated_sequence, parent_sequence)), key

    evolved_sequence, new_key = jax.lax.cond(
      node_index != root_node,
      lambda: evolve(sequences[parent_index], current_key),
      lambda: (sequences[node_index], current_key),
    )

    return sequences.at[node_index].set(evolved_sequence), new_key

  all_sequences, _ = jax.lax.fori_loop(
    0,
    n_nodes,
    body_fun,
    (all_sequences, key),
  )

  # Masked sequences are not needed for this part of the task
  masked_sequences = jnp.zeros_like(all_sequences, dtype=jnp.bfloat16)

  return PhylogeneticTree(
    masked_sequences=masked_sequences,
    all_sequences=all_sequences.astype(jnp.bfloat16),
    adjacency=adjacency.astype(jnp.bfloat16),
  )
