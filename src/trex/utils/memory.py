"""Memory-safe utilities for JAX-based operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import lax

if TYPE_CHECKING:
  from collections.abc import Callable
  from typing import Any


def safe_map(fn: Callable, xs: Any, batch_size: int):
  """Memory-safe version of jax.vmap using jax.lax.map.

  Processes the input data in chunks of batch_size.

  Args:
      fn: The function to apply.
      xs: The input data (array or PyTree of arrays).
      batch_size: Number of elements to process in parallel.

  Returns:
      The result of applying fn to xs.
  """
  return lax.map(fn, xs, batch_size=batch_size)


def estimate_memory_usage(shapes: list[tuple], dtypes: list[jnp.dtype]) -> float:
  """Estimates memory usage in bytes.

  Args:
      shapes: List of array shapes.
      dtypes: List of array dtypes.

  Returns:
      Estimated memory usage in bytes.
  """
  total_bytes = 0
  for shape, dtype in zip(shapes, dtypes, strict=True):
    size = 1
    for dim in shape:
      size *= dim
    total_bytes += size * jnp.dtype(dtype).itemsize
  return total_bytes
