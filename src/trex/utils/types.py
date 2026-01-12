"""Core types for the diff-evol-tree-search project.

This module uses jaxtyping for type annotations of JAX arrays, providing
shape and dtype information directly in the type hints. This is crucial for
maintaining clarity and correctness in a JAX-based scientific computing codebase.
"""

from typing import Any, NamedTuple

from jaxtyping import Array, BFloat16, Float, Int

# Re-export PRNGKeyArray for convenient access throughout the project.


EvoSequence = Int[Array, "seq_length"]
"""Represents a single biological sequence as a 1D array of integers."""

OneHotEvoSequence = Float[Array, "seq_length states"]
"""Represents a single biological sequence as a one-hot encoded 2D array."""

BatchEvoSequence = Int[Array, "batch seq_length"]
"""Represents a batch of biological sequences as a 2D array."""

BatchOneHotEvoSequence = Float[Array, "batch seq_length states"]
"""Represents a batch of one-hot encoded biological sequences as a 3D array."""

BFloat16TreeSequences = BFloat16[Array, "n_all seq_length"]
"""Represents all sequences in a tree as a single batch in BFloat16 format."""

AdjacencyMatrix = Float[Array, "num_nodes num_nodes"]
"""An adjacency matrix representing the structure of a phylogenetic tree.

Typically, `tree[i, j] = 1` indicates that node `i` is a child of node `j`.
Using a Float dtype allows for compatibility with differentiable operations.
"""

BFloat16AdjacencyMatrix = BFloat16[Array, "num_nodes num_nodes"]
"""An adjacency matrix in BFloat16 format, often used for memory efficiency."""

GroundTruthMetadata = dict[str, int]
"""A dictionary holding the metadata required to generate a ground truth tree.

Expected keys typically include "n_leaves", "seq_length", "n_states",
and "n_mutations".
"""

CostMatrix = Float[Array, "states states"]
"""A matrix defining the cost of substitution between any two characters."""

DPTable = Float[Array, "nodes states"]
"""The dynamic programming table used in Sankoff's algorithm."""

VmappedDPTable = Float[Array, "seq_length nodes states"]
"""A batch of DP tables, typically vmapped over the sequence length."""

BacktrackingTable = Int[Array, "nodes states children"]
"""A table storing backtracking information for reconstructing ancestral sequences."""

ReconstructedSequence = Float[Array, "nodes 1"]
"""A single reconstructed sequence for all nodes in the tree."""

TotalCost = Float[Array, ""]
"""A scalar representing the total parsimony cost."""

State = Int[Array, ""]
"""An integer representing a single character in the sequence alphabet."""

Node = Int[Array, ""]
"""An integer representing a single node in the tree."""

SubstitutionMatrix = Int[Array, "states states"]

Cost = Float[Array, ""]
"""A scalar cost value, typically the total cost of a tree traversal."""


EvoSequencePyTree = Any


class PhylogeneticTree(NamedTuple):
  """Represents a phylogenetic tree with its sequences and structure.

  This is a JAX-compatible PyTree, allowing it to be passed to and from
  JIT-compiled functions.
  """

  masked_sequences: BFloat16TreeSequences
  """The leaf sequences, with ancestor nodes masked (zeroed)."""
  all_sequences: BFloat16TreeSequences
  """The complete set of true sequences, including ancestors."""
  adjacency: BFloat16AdjacencyMatrix
  """The adjacency matrix representing the parent-child relationships."""
