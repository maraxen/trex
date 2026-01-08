"""Type definitions for TREX project."""

from typing import Literal

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

# JAX Aliases
PRNGKey = PRNGKeyArray

# Configuration
DTypeStr = Literal["float32", "bfloat16", "float64"]
_DEFAULT_DTYPE: DTypeStr = "float32"


# Scalar Types
Scalar = Int[Array, ""]
ScalarFloat = Float[Array, ""]

# Sequence Types
Ancestors = Int[Array, "n_ancestors seq_len"]
LeafSequences = Int[Array, "n_leaves seq_len"]
FullSequences = Int[Array, "n_nodes seq_len"]
Sequence = Int[Array, "seq_len"]
Sequences = Int[Array, "batch seq_len"]
SoftSequences = Float[Array, "n_nodes seq_len n_states"]
SeqMask = Bool[Array, "seq_len"]

# Tree Structure
Adjacency = Int[Array, "n_nodes n_nodes"]
CostMatrix = Float[Array, "n_states n_states"]
CostVector = Float[Array, "n_states"]


def get_default_dtype() -> jnp.dtype:
  """Return the configured default floating-point dtype."""
  return jnp.dtype(_DEFAULT_DTYPE)


def set_default_dtype(dtype: DTypeStr) -> None:
  """Set the default floating-point dtype."""
  global _DEFAULT_DTYPE
  _DEFAULT_DTYPE = dtype
