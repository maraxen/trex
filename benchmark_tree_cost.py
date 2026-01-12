import time

import jax
import jax.numpy as jnp

from trex.tree import compute_soft_cost


# -- Naive Implementation (Previous Version) --
@jax.jit
def naive_compute_soft_cost(
  sequences: jnp.ndarray,  # (n_nodes, seq_len, n_states)
  adjacency: jnp.ndarray,  # (n_nodes, n_nodes)
  cost_matrix: jnp.ndarray | None = None,
) -> jnp.ndarray:
  n_states = sequences.shape[-1]
  if cost_matrix is None:
    cost_matrix = jnp.eye(n_states)

  # Compute pairwise differences: diff[i,j,l,q] = S_i[l,q] - S_j[l,q]
  # Shape: (N, N, L, Q) -- This is the memory bottleneck!
  diff = sequences[:, jnp.newaxis, :, :] - sequences[jnp.newaxis, :, :, :]

  # Apply cost matrix: diff @ C -> (N, N, L, Q)
  weighted_diff = diff @ cost_matrix

  # Compute quadratic form: sum over q of diff * weighted_diff
  # Then sum over positions l, weighted by adjacency
  cost_per_edge = jnp.sum(diff * weighted_diff, axis=(-1, -2))

  # Weight by adjacency and sum (divide by 2 for undirected edges)
  return jnp.sum(cost_per_edge * adjacency) / 2


def run_benchmark() -> None:
  """Run performance benchmark comparing optimized and naive soft cost."""
  print("Running JAX Performance Benchmark: Optimized vs Naive Soft Cost")
  print("-" * 60)

  # Setup Parameters
  seq_len = 1000  # Sequence Length
  alphabet_size = 20  # Alphabet Size
  num_nodes_values = [10, 50, 100, 200]  # Number of nodes to sweep

  naive_times = []
  opt_times = []

  key = jax.random.PRNGKey(42)

  for n_nodes in num_nodes_values:
    print(f"\nBenchmarking N={n_nodes}, L={seq_len}, Q={alphabet_size}...")

    # Data Generation
    key, subkey1, subkey2 = jax.random.split(key, 3)
    sequences = jax.random.normal(subkey1, (n_nodes, seq_len, alphabet_size))
    sequences = jax.nn.softmax(sequences, axis=-1)
    adjacency = jax.random.bernoulli(subkey2, 0.1, (n_nodes, n_nodes)).astype(jnp.float32)

    # Benchmark Optimized
    print("  Running Optimized...")
    try:
      # Warmup
      _ = compute_soft_cost(sequences, adjacency)
      jax.block_until_ready(_)

      start = time.time()
      for _ in range(10):  # Run 10 times to average
        res = compute_soft_cost(sequences, adjacency)
        jax.block_until_ready(res)
      end = time.time()
      avg_time = (end - start) / 10.0
      opt_times.append(avg_time)
      print(f"    Avg Time: {avg_time:.6f} s")

    except RuntimeError as e:
      print(f"    Optimized Failed: {e}")
      opt_times.append(None)

    # Benchmark Naive
    print("  Running Naive...")
    try:
      # Warmup
      _ = naive_compute_soft_cost(sequences, adjacency)
      jax.block_until_ready(_)

      start = time.time()
      for _ in range(10):
        res = naive_compute_soft_cost(sequences, adjacency)
        jax.block_until_ready(res)
      end = time.time()
      avg_time = (end - start) / 10.0
      naive_times.append(avg_time)
      print(f"    Avg Time: {avg_time:.6f} s")

    except RuntimeError as e:  # Catch OOM
      print(f"    Naive Failed (likely OOM): {e}")
      naive_times.append(None)

  # Reporting
  print_results(num_nodes_values, opt_times, naive_times)


def print_results(
  num_nodes_values: list[int], opt_times: list[float | None], naive_times: list[float | None]
) -> None:
  """Print benchmark results table."""
  print("\n" + "=" * 60)
  print("Benchmark Results Summary")
  print(f"{'N':<10} | {'Optimized (s)':<15} | {'Naive (s)':<15} | {'Speedup':<10}")
  print("-" * 60)
  for i, n_nodes in enumerate(num_nodes_values):
    opt = opt_times[i]
    naive = naive_times[i]

    opt_str = f"{opt:.6f}" if opt else "N/A"
    naive_str = f"{naive:.6f}" if naive else "OOM/Fail"

    speedup = f"{naive / opt:.2f}x" if (opt and naive) else "-"
    print(f"{n_nodes:<10} | {opt_str:<15} | {naive_str:<15} | {speedup:<10}")


if __name__ == "__main__":
  run_benchmark()
