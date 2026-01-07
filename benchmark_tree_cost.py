import jax
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt
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


def run_benchmark():
  print("Running JAX Performance Benchmark: Optimized vs Naive Soft Cost")
  print("-" * 60)

  # Setup Parameters
  L = 1000  # Sequence Length
  Q = 20  # Alphabet Size
  N_values = [10, 50, 100, 200]  # Number of nodes to sweep

  naive_times = []
  opt_times = []

  key = jax.random.PRNGKey(42)

  for N in N_values:
    print(f"\nBenchmarking N={N}, L={L}, Q={Q}...")

    # Data Generation
    key, subkey1, subkey2 = jax.random.split(key, 3)
    sequences = jax.random.normal(subkey1, (N, L, Q))
    sequences = jax.nn.softmax(sequences, axis=-1)
    adjacency = jax.random.bernoulli(subkey2, 0.1, (N, N)).astype(jnp.float32)

    # Warmup and Run Optimized
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

    except Exception as e:
      print(f"    Optimized Failed: {e}")
      opt_times.append(None)

    # Warmup and Run Naive
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

    except Exception as e:  # Catch OOM
      print(f"    Naive Failed (likely OOM): {e}")
      naive_times.append(None)

  # Reporting
  print("\n" + "=" * 60)
  print("Benchmark Results Summary")
  print(f"{'N':<10} | {'Optimized (s)':<15} | {'Naive (s)':<15} | {'Speedup':<10}")
  print("-" * 60)
  for i, N in enumerate(N_values):
    opt = opt_times[i]
    naive = naive_times[i]

    opt_str = f"{opt:.6f}" if opt else "N/A"
    naive_str = f"{naive:.6f}" if naive else "OOM/Fail"

    speedup = f"{naive / opt:.2f}x" if (opt and naive) else "-"
    print(f"{N:<10} | {opt_str:<15} | {naive_str:<15} | {speedup:<10}")


if __name__ == "__main__":
  try:
    run_benchmark()
  except Exception as e:
    print(f"Benchmark script crashed: {e}")
