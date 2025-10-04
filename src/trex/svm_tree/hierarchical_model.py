"""A hierarchical SVM model with a fixed topology."""

from __future__ import annotations

import equinox as eqx
import jax

from .components.ste import STEGating
from .components.svm import LinearSVM


class LeafNode(eqx.Module):
    """A leaf node in the hierarchical SVM, which makes a final prediction."""

    classifier: LinearSVM

    def __call__(self, x: jax.Array) -> jax.Array:
        """Return the prediction from the leaf's classifier."""
        return self.classifier(x)


class InternalNode(eqx.Module):
    """An internal node in the hierarchical SVM, which routes data."""

    router: LinearSVM
    ste: STEGating
    left: InternalNode | LeafNode
    right: InternalNode | LeafNode

    def __call__(self, x: jax.Array) -> jax.Array:
        """Route the input to the left or right child based on the router's output."""
        decision = self.router(x)
        go_right = self.ste(decision)

        # Use jax.lax.cond to handle the dynamic routing in a JIT-compatible way.
        return jax.lax.cond(
            go_right > 0,
            lambda: self.right(x),
            lambda: self.left(x),
        )


class HierarchicalSVM(eqx.Module):
    """A hierarchical SVM with a fixed binary tree structure.

    This model uses hard routing decisions at each internal node to guide the input
    to a specific leaf node, which then makes the final classification.
    """

    root: InternalNode | LeafNode
    depth: int = eqx.field(static=True)
    in_features: int = eqx.field(static=True)
    num_classes: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        depth: int,
        in_features: int,
        num_classes: int,
        key: jax.random.PRNGKey,
    ) -> None:
        """Initialize the HierarchicalSVM.

        Args:
            depth: The depth of the binary tree.
            in_features: The number of input features.
            num_classes: The number of output classes for the leaf classifiers.
            key: A JAX PRNG key for initializing the model's parameters.

        """
        self.depth = depth
        self.in_features = in_features
        self.num_classes = num_classes
        self.root = self._build_tree(depth, key)

    def _build_tree(
        self, depth: int, key: jax.random.PRNGKey,
    ) -> InternalNode | LeafNode:
        """Recursively build the binary tree of nodes.

        Args:
            depth: The current depth in the tree.
            key: A JAX PRNG key.

        Returns:
            The root of the constructed subtree.

        """
        if depth == 0:
            return LeafNode(
                LinearSVM(self.in_features, out_features=self.num_classes, key=key),
            )

        key, router_key, left_key, right_key = jax.random.split(key, 4)
        left_child = self._build_tree(depth - 1, left_key)
        right_child = self._build_tree(depth - 1, right_key)

        return InternalNode(
            router=LinearSVM(self.in_features, key=router_key),
            ste=STEGating(),
            left=left_child,
            right=right_child,
        )

    def __call__(self, x: jax.Array, *, key: jax.random.PRNGKey | None = None) -> jax.Array:
        """Perform a forward pass through the tree.

        Args:
            x: The input data.
            key: A JAX PRNG key (unused in this model, but kept for API consistency).

        Returns:
            The prediction from the selected leaf node.

        """
        del key  # Unused.
        return self.root(x)
