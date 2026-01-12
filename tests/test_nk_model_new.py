
"""Unit tests for NK model tree generation with variable branch lengths and mutation noise."""

import jax
import jax.numpy as jnp
import pytest
import numpy as np
import sys
import os

# Ensure we test the local code
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from trex.nk_model import create_nk_model_landscape, generate_tree_data, get_fitness
from trex.evals.benchmark import create_balanced_binary_tree

class TestNKModelNewFeatures:
    """Tests for new NK model features: variable branch length and mutation rate noise."""

    @pytest.fixture
    def setup_landscape_and_tree(self):
        """Setup a simple landscape and tree for testing."""
        key = jax.random.PRNGKey(42)
        n = 20
        k = 2
        q = 4
        n_leaves = 16
        landscape = create_nk_model_landscape(n, k, key, n_states=q)
        adj_matrix = create_balanced_binary_tree(n_leaves)
        
        key, subkey = jax.random.split(key)
        root_sequence = jax.random.randint(subkey, (n,), 0, q)
        
        return landscape, adj_matrix, root_sequence, q

    def test_generate_tree_variable_branch_length(self, setup_landscape_and_tree):
        """Test that longer branch lengths produce more divergent sequences."""
        landscape, adj_matrix, root_sequence, q = setup_landscape_and_tree
        key = jax.random.PRNGKey(101)
        mutation_rate = 0.05
        
        # Helper to get mean distance from root for all nodes
        def get_mean_dist(branch_len, key):
            tree = generate_tree_data(
                landscape, adj_matrix, root_sequence[:, None], mutation_rate, 
                key, branch_length=branch_len
            )
            # Calculate Hamming distance from root for all nodes
            root_seq_bc = root_sequence[None, :]
            all_seqs = tree.all_sequences.astype(jnp.int32)
            node_dists = jnp.mean(all_seqs != root_seq_bc, axis=1)
            # Exclude root itself (index -1 in sorted, but easiest is to take mean of non-zero)
            # or just mean of all non-root nodes
            return jnp.mean(node_dists)

        # 1. Branch length = 1
        dist_1 = get_mean_dist(1, key)
        
        # 2. Branch length = 10
        dist_10 = get_mean_dist(10, key)
        
        # 3. Branch length = 10 must imply more mutations -> larger distance from root
        # (Given mutation_rate is small enough not to saturate)
        assert dist_10 > dist_1, f"Expected more divergence with length 10 vs 1. Got {dist_10} vs {dist_1}"
        
        print(f"Mean dist (L=1): {dist_1:.4f}")
        print(f"Mean dist (L=10): {dist_10:.4f}")


    def test_generate_tree_backward_compatible(self, setup_landscape_and_tree):
        """Verify default parameters match original behavior (noise=0, len=1)."""
        landscape, adj_matrix, root_sequence, q = setup_landscape_and_tree
        key = jax.random.PRNGKey(303)
        mutation_rate = 0.1
        
        # Run with defaults
        tree_default = generate_tree_data(
            landscape, adj_matrix, root_sequence[:, None], mutation_rate, key
        )
        
        # Run with explicit defaults
        tree_explicit = generate_tree_data(
            landscape, adj_matrix, root_sequence[:, None], mutation_rate, key,
            mutation_rate_noise_std=0.0, branch_length=1
        )
        
        assert jnp.array_equal(tree_default.all_sequences, tree_explicit.all_sequences)

