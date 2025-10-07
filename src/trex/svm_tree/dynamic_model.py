from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, Float

from trex.tree import update_tree
from .components.feature_encoder import CNNFeatureEncoder
from .components.svm import LinearSVM
from .components.topology_head import TopologyHead


class DynamicHierarchicalSVM(eqx.Module):
    """A hierarchical SVM with a topology that is dynamically generated per sample."""

    feature_encoder: CNNFeatureEncoder
    topology_head: TopologyHead
    leaves: list[LinearSVM]
    routers: list[LinearSVM]

    n_leaves: int = eqx.field(static=True)
    n_ancestors: int = eqx.field(static=True)
    num_classes: int = eqx.field(static=True)
    total_nodes: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        depth: int,
        in_features: int,
        num_classes: int,
        embedding_dim: int,
        key: PRNGKeyArray,
    ):
        """Initializes the DynamicHierarchicalSVM.

        Args:
            depth: The depth of the binary tree.
            in_features: The number of input features for the SVMs.
            num_classes: The number of output classes for the leaf classifiers.
            embedding_dim: The dimensionality of the feature embedding from the CNN.
            key: A JAX PRNG key for initializing parameters.
        """
        self.n_leaves = 2**depth
        self.n_ancestors = 2**depth - 1
        self.num_classes = num_classes
        self.total_nodes = self.n_leaves + self.n_ancestors

        encoder_key, head_key, leaves_key, routers_key = jax.random.split(key, 4)

        self.feature_encoder = CNNFeatureEncoder(key=encoder_key, embedding_dim=embedding_dim)

        # The topology head needs to generate parameters for two matrices (left/right connections)
        n_children_nodes = self.total_nodes - 1  # Root cannot be a child
        topology_output_dim = 2 * n_children_nodes * self.n_ancestors

        self.topology_head = TopologyHead(
            embedding_dim=embedding_dim,
            n_leaves=self.n_leaves,
            n_ancestors=self.n_ancestors,
            key=head_key,
        )

        # Adjust the final layer of the MLP to match the required output dimension
        head_mlp_layers = list(self.topology_head.mlp.layers)
        final_linear_layer = head_mlp_layers[-2]
        new_final_layer = eqx.nn.Linear(
            final_linear_layer.in_features,
            topology_output_dim,
            key=jax.random.PRNGKey(0), # Dummy key, will be replaced by head_key later
        )
        # Re-init the final layer with the correct key
        final_layer_key, _ = jax.random.split(head_key, 2)
        new_final_layer = eqx.tree_at(
            lambda l: l.weight, new_final_layer, jax.random.normal(final_layer_key, (topology_output_dim, final_linear_layer.in_features)) * 0.01
        )
        new_final_layer = eqx.tree_at(
            lambda l: l.bias, new_final_layer, jnp.zeros((topology_output_dim,))
        )
        head_mlp_layers[-2] = new_final_layer
        self.topology_head = eqx.tree_at(lambda m: m.mlp.layers, self.topology_head, head_mlp_layers)


        # Create leaf and router nodes
        leaf_keys = jax.random.split(leaves_key, self.n_leaves)
        self.leaves = [
            LinearSVM(in_features, num_classes, key=k) for k in leaf_keys
        ]

        router_keys = jax.random.split(routers_key, self.n_ancestors)
        self.routers = [LinearSVM(in_features, 1, key=k) for k in router_keys]

    def __call__(
        self, x: Float[Array, "1 28 28"], *, key: PRNGKeyArray, temperature: float = 1.0
    ) -> Float[Array, "num_classes"]:
        """Performs a forward pass with a dynamically generated tree.

        Args:
            x: A single input image.
            key: A JAX PRNG key.
            temperature: The temperature for the Gumbel-Softmax distribution.

        Returns:
            The final classification logits.
        """
        # 1. Get feature embedding from the image
        embedding = self.feature_encoder(x)

        # 2. Generate topology parameters from the embedding
        key, topo_key = jax.random.split(key)
        combined_params = self.topology_head(embedding)

        n_children_nodes = self.total_nodes - 1
        n_params_per_matrix = n_children_nodes * self.n_ancestors

        param_chunks = jnp.split(combined_params, 2)
        params_left = param_chunks[0].reshape((n_children_nodes, self.n_ancestors))
        params_right = param_chunks[1].reshape((n_children_nodes, self.n_ancestors))

        # 3. Generate soft adjacency matrices for left and right children
        key_left, key_right = jax.random.split(topo_key)
        # Pad params to match expected shape for update_tree if necessary
        # update_tree expects (n_all-1, n_ancestors), which matches n_children_nodes
        adj_left = update_tree(key_left, {"tree_params": params_left}, temperature=temperature)
        adj_right = update_tree(key_right, {"tree_params": params_right}, temperature=temperature)

        # Transpose to get child-to-parent probabilities
        adj_left_T = adj_left.T
        adj_right_T = adj_right.T

        # 4. Perform bottom-up classification pass
        x_flat = jnp.ravel(x)

        # Initialize node outputs array
        node_outputs = jnp.zeros((self.total_nodes, self.num_classes))

        # Compute outputs for all leaf nodes
        leaf_vmap = jax.vmap(lambda leaf: leaf(x_flat))
        all_leaf_outputs = leaf_vmap(self.leaves)
        node_outputs = node_outputs.at[:self.n_leaves].set(all_leaf_outputs)

        # Compute routing decisions for all internal nodes
        router_vmap = jax.vmap(lambda router: jax.nn.sigmoid(router(x_flat)))
        all_router_outputs = router_vmap(self.routers).flatten()

        # Propagate information up the tree from leaves to root
        def body_fun(i, current_node_outputs):
            # Process nodes from lowest level of ancestors up to the root
            ancestor_idx_rev = self.n_ancestors - 1 - i
            node_idx = self.n_leaves + ancestor_idx_rev

            # Get probabilities of each node being the left/right child
            left_child_probs = adj_left_T[node_idx, :]
            right_child_probs = adj_right_T[node_idx, :]

            # Compute expected output from left and right children
            left_child_output = jnp.dot(left_child_probs, current_node_outputs)
            right_child_output = jnp.dot(right_child_probs, current_node_outputs)

            # Get routing probability for the current internal node
            go_right_prob = all_router_outputs[ancestor_idx_rev]

            # Combine children outputs based on the router
            parent_output = (1 - go_right_prob) * left_child_output + go_right_prob * right_child_output

            return current_node_outputs.at[node_idx].set(parent_output)

        # Loop over ancestors, starting from those just above the leaves
        final_node_outputs = jax.lax.fori_loop(0, self.n_ancestors, body_fun, node_outputs)

        # The root is the last node in the array
        root_output = final_node_outputs[self.total_nodes - 1]

        # Return both prediction and auxiliary data for loss/visualization
        return root_output, (adj_left, adj_right)