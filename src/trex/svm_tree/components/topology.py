import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from trex.tree import enforce_graph_constraints, update_tree


class DifferentiableTopology(eqx.Module):
    """A component that manages the learnable tree structure.
    This module wraps the logic for updating a tree's topology using the
    Gumbel-Softmax trick and applies a gating mechanism to prune connections.
    It also computes regularization losses to encourage sparsity and enforce
    graph constraints.

    Attributes:
        tree_params: Trainable parameters for the tree's connections.
        gate_logits: Trainable logits for the gating mechanism.
        n_leaves: The number of leaf nodes in the tree.
        n_ancestors: The number of ancestor nodes in the tree.
        sparsity_regularization_strength: Weight for the sparsity loss.
        graph_constraint_scale: Scaling factor for the graph constraint loss.

    """

    tree_params: Float[Array, " n_all_minus_1 n_ancestors"]
    gate_logits: Float[Array, " n_all_minus_1 n_ancestors"]

    n_leaves: int = eqx.field(static=True)
    n_ancestors: int = eqx.field(static=True)

    sparsity_regularization_strength: float = eqx.field(static=True)
    graph_constraint_scale: float = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        n_leaves: int,
        n_ancestors: int,
        sparsity_regularization_strength: float = 0.01,
        graph_constraint_scale: float = 10.0,
    ):
        """Initializes the DifferentiableTopology component.

        Args:
            key: A JAX random key.
            n_leaves: The number of leaf nodes.
            n_ancestors: The number of ancestor nodes.
            sparsity_regularization_strength: The strength of the L1 sparsity penalty.
            graph_constraint_scale: The scaling factor for the graph constraint loss.

        """
        tree_key, gate_key = jax.random.split(key)
        n_all_minus_1 = n_leaves + n_ancestors - 1

        self.tree_params = jax.random.normal(
            tree_key, (n_all_minus_1, n_ancestors),
        )
        self.gate_logits = jax.random.normal(
            gate_key, (n_all_minus_1, n_ancestors),
        )

        self.n_leaves = n_leaves
        self.n_ancestors = n_ancestors
        self.sparsity_regularization_strength = sparsity_regularization_strength
        self.graph_constraint_scale = graph_constraint_scale

    def __call__(
        self, key: PRNGKeyArray, temperature: float = 1.0,
    ) -> Float[Array, "n_nodes n_nodes"]:
        """Computes the soft adjacency matrix of the tree.

        Args:
            key: A JAX random key for the Gumbel-Softmax trick.
            temperature: The temperature for the Gumbel-Softmax.

        Returns:
            The soft adjacency matrix representing the tree topology.

        """
        params = {"tree_params": self.tree_params}
        gates = jax.nn.sigmoid(self.gate_logits)
        return update_tree(key, params, temperature=temperature, gates=gates)

    def loss(
        self, adjacency: Float[Array, "n_nodes n_nodes"],
    ) -> Float[Array, ""]:
        """Computes the regularization losses for the topology.

        Args:
            adjacency: The current soft adjacency matrix of the tree.

        Returns:
            The total regularization loss.

        """
        # 1. Sparsity loss on the gates (L1 regularization)
        sparsity_loss = jnp.mean(jnp.abs(self.gate_logits))

        # 2. Graph constraint loss
        graph_loss = enforce_graph_constraints(
            adjacency, self.graph_constraint_scale,
        )

        total_loss = (
            self.sparsity_regularization_strength * sparsity_loss
        ) + graph_loss
        return total_loss
