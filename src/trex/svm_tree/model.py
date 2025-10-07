import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .components.ste import hard_decision
from .components.svm import LinearSVM
from .components.topology import DifferentiableTopology


class LearnableHierarchicalSVM(eqx.Module):
  """A hierarchical SVM model with a dynamically learnable topology.
  This model uses a top-down traversal with hard routing decisions
  enabled by a Straight-Through Estimator (STE).
  """

  svm_weights: Float[Array, "n_ancestors in_features"]
  svm_biases: Float[Array, "n_ancestors"]
  topology: DifferentiableTopology
  num_classes: int
  n_leaves: int
  n_ancestors: int
  in_features: int
  n_total: int

  def __init__(
    self,
    in_features: int,
    num_classes: int,
    n_ancestors: int,
    *,
    key: "jax.random.PRNGKey",
    sparsity_regularization_strength: float = 0.01,
    graph_constraint_scale: float = 10.0,
  ):
    self.num_classes = num_classes
    self.n_leaves = num_classes
    self.n_ancestors = n_ancestors
    self.in_features = in_features
    self.n_total = self.n_leaves + self.n_ancestors

    svm_key, topo_key = jax.random.split(key)

    # Initialize SVM parameters for all potential internal nodes
    svm_keys = jax.random.split(svm_key, self.n_ancestors)
    svms = [LinearSVM(in_features, key=k) for k in svm_keys]
    self.svm_weights = jnp.stack([svm.weights for svm in svms])
    self.svm_biases = jnp.stack([svm.bias for svm in svms])

    # Initialize the differentiable topology learner
    self.topology = DifferentiableTopology(
      key=topo_key,
      n_leaves=self.n_leaves,
      n_ancestors=self.n_ancestors,
      sparsity_regularization_strength=sparsity_regularization_strength,
      graph_constraint_scale=graph_constraint_scale,
    )

  def __call__(
    self,
    x: Float[Array, "in_features"],
    *,
    key: "jax.random.PRNGKey",
  ) -> Float[Array, "num_classes"]:
    """Performs a top-down, differentiable traversal of the tree."""
    adj = self.topology(key)
    root_index = self.n_total - 1
    initial_path_probs = jnp.zeros(self.n_total).at[root_index].set(1.0)
    ancestor_indices_descending = jnp.arange(root_index, self.n_leaves - 1, -1)

    def body_fn(path_probs, node_index):
      prob_reaching_node = path_probs[node_index]
      svm_index = node_index - self.n_leaves
      w = self.svm_weights[svm_index]
      b = self.svm_biases[svm_index]
      decision = hard_decision(jnp.dot(w, x) + b)
      prob_left, prob_right = 1 - decision, decision

      indices = jnp.arange(self.n_total)
      child_mask = indices < node_index
      children_parent_probs = adj[:, node_index] * child_mask

      pair_probs = children_parent_probs[:, None] * children_parent_probs[None, :]
      i_indices = indices[:, None]
      j_indices = indices[None, :]
      pair_mask = i_indices < j_indices
      pair_probs *= pair_mask

      pair_probs_sum = jnp.sum(pair_probs)
      pair_probs = jax.lax.cond(
        pair_probs_sum > 0,
        lambda: pair_probs / pair_probs_sum,
        lambda: pair_probs,
      )

      prob_updates_i = jnp.sum(pair_probs, axis=1) * prob_left
      prob_updates_j = jnp.sum(pair_probs, axis=0) * prob_right
      path_probs_updates = (prob_updates_i + prob_updates_j) * prob_reaching_node

      new_path_probs = path_probs.at[node_index].set(0.0)
      new_path_probs = new_path_probs.at[indices].add(path_probs_updates)
      return new_path_probs, None

    final_path_probs, _ = jax.lax.scan(
      body_fn,
      initial_path_probs,
      ancestor_indices_descending,
    )
    return final_path_probs[: self.n_leaves]

  def loss(self, adjacency: Float[Array, "n_nodes n_nodes"]) -> Float[Array, ""]:
    """Computes the regularization loss for the topology."""
    return self.topology.loss(adjacency)


class LearnableTreeModel(eqx.Module):
  """A tree model with a learnable topology."""

  svm_weights: Float[Array, "n_ancestors in_features"]
  svm_biases: Float[Array, "n_ancestors"]
  topology: DifferentiableTopology
  num_classes: int
  n_leaves: int
  n_ancestors: int
  in_features: int

  def __init__(
    self,
    in_features: int,
    num_classes: int,
    n_ancestors: int,
    *,
    key: "jax.random.PRNGKey",
    sparsity_regularization_strength: float = 0.01,
    graph_constraint_scale: float = 10.0,
  ):
    """Initializes the LearnableTreeModel.

    Args:
        in_features: The number of input features.
        num_classes: The number of output classes.
        n_ancestors: The number of ancestor nodes.
        key: A JAX PRNG key for initializing the SVMs and topology.
        sparsity_regularization_strength: The strength of the L1 sparsity penalty.
        graph_constraint_scale: The scaling factor for the graph constraint loss.

    """
    self.num_classes = num_classes
    self.n_leaves = num_classes
    self.n_ancestors = n_ancestors
    self.in_features = in_features

    svm_key, topo_key = jax.random.split(key)
    svm_keys = jax.random.split(svm_key, self.n_ancestors)

    svms = [LinearSVM(in_features, key=k) for k in svm_keys]
    self.svm_weights = jnp.stack([svm.weights for svm in svms])
    self.svm_biases = jnp.stack([svm.bias for svm in svms])

    self.topology = DifferentiableTopology(
      key=topo_key,
      n_leaves=self.n_leaves,
      n_ancestors=self.n_ancestors,
      sparsity_regularization_strength=sparsity_regularization_strength,
      graph_constraint_scale=graph_constraint_scale,
    )

  def __call__(
    self,
    x: Float[Array, "in_features"],
    *,
    key: "jax.random.PRNGKey",
  ) -> Float[Array, "num_classes"]:
    """Computes the forward pass using a differentiable, bottom-up approach.

    Args:
        x: The input data.
        key: A JAX PRNG key for the topology calculation.

    Returns:
        The model's output (a probability distribution over classes).

    """
    adj = self.topology(key)

    # Initialize node distributions: one-hot for leaves, zeros for ancestors
    leaf_dists = jax.nn.one_hot(jnp.arange(self.n_leaves), self.num_classes)
    ancestor_dists = jnp.zeros((self.n_ancestors, self.num_classes))
    node_dists = jnp.concatenate([leaf_dists, ancestor_dists], axis=0)

    # Iterate bottom-up from the lowest ancestors to the root
    ancestor_indices = jnp.arange(self.n_leaves, self.n_leaves + self.n_ancestors)

    def body_fn(dists, k):
      svm_index = k - self.n_leaves
      w = self.svm_weights[svm_index]
      b = self.svm_biases[svm_index]
      prob_right = jax.nn.sigmoid(jnp.dot(w, x) + b)

      # Use masking to handle dynamic child selection instead of slicing.
      # This keeps array shapes static and JAX-compatible.
      indices = jnp.arange(self.n_leaves + self.n_ancestors)
      child_mask = indices < k

      # P(parent(i) = k) for each potential child i, masked for i < k
      children_parent_probs = adj[:, k] * child_mask

      # P(c1=i, c2=j | parent=k) ~= P(parent(i)=k) * P(parent(j)=k)
      pair_probs = children_parent_probs[:, None] * children_parent_probs[None, :]

      # Mask to avoid self-pairing (i=j) and double-counting (j, i)
      i_indices = indices[:, None]
      j_indices = indices[None, :]
      pair_mask = i_indices < j_indices
      pair_probs = pair_probs * pair_mask

      # Use the full dists array, not a slice
      dists_left = dists[:, None, :]
      dists_right = dists[None, :, :]

      # Mix distributions for all possible pairs
      mixed_dists = (1 - prob_right) * dists_left + prob_right * dists_right

      # Weight the mixed distributions by the pair probabilities
      weighted_dists = pair_probs[..., None] * mixed_dists

      # Sum over all pairs to get the final distribution for the current node
      dist_k = jnp.sum(weighted_dists, axis=(0, 1))

      # Normalize distribution to prevent vanishing/exploding probs
      dist_k_sum = jnp.sum(dist_k)
      dist_k = jax.lax.cond(dist_k_sum > 0, lambda: dist_k / dist_k_sum, lambda: dist_k)

      return dists.at[k].set(dist_k), None

    final_dists, _ = jax.lax.scan(body_fn, node_dists, ancestor_indices)
    return final_dists[-1]

  def loss(self, adjacency: Float[Array, "n_nodes n_nodes"]) -> Float[Array, ""]:
    """Computes the regularization loss for the topology."""
    return self.topology.loss(adjacency)


class Node(eqx.Module):
  """Abstract base class for nodes in the tree."""


class Leaf(Node):
  """A leaf node in the tree, representing a single class."""

  class_id: int
  num_classes: int

  def __call__(self, x: Float[Array, "in_features"]) -> Float[Array, "num_classes"]:
    """Returns a one-hot encoded vector for the class.

    Args:
        x: The input data (unused at leaf nodes).

    Returns:
        A one-hot encoded vector representing the leaf's class.

    """
    return jax.nn.one_hot(self.class_id, self.num_classes)


class InternalNode(Node):
  """An internal node in the tree, which splits the data."""

  svm: LinearSVM
  left: Node
  right: Node

  def __call__(self, x: Float[Array, "in_features"]) -> Float[Array, "num_classes"]:
    """Recursively traverses the tree with soft routing.

    Args:
        x: The input data.

    Returns:
        A probability distribution over the classes.

    """
    svm_output = self.svm(x)
    prob_right = jax.nn.sigmoid(svm_output)

    dist_left = self.left(x)
    dist_right = self.right(x)

    return (1 - prob_right) * dist_left + prob_right * dist_right


class BaseTreeModel(eqx.Module):
  """A tree model with a fixed, hard-coded topology."""

  root: Node
  in_features: int
  num_classes: int

  def __init__(
    self,
    in_features: int,
    num_classes: int,
    *,
    key: "jax.random.PRNGKey",
  ):
    """Initializes the BaseTreeModel.

    Args:
        in_features: The number of input features.
        num_classes: The number of output classes.
        key: A JAX PRNG key for initializing the SVMs.

    """
    self.in_features = in_features
    self.num_classes = num_classes

    # Need 9 keys for the 9 internal nodes
    keys = jax.random.split(key, 9)

    # Create all leaf nodes
    leaves = [Leaf(i, num_classes) for i in range(num_classes)]

    # Build the tree from the bottom up, following a balanced binary structure
    node_0_1 = InternalNode(
      LinearSVM(in_features, key=keys[0]),
      leaves[0],
      leaves[1],
    )
    node_3_4 = InternalNode(
      LinearSVM(in_features, key=keys[1]),
      leaves[3],
      leaves[4],
    )
    node_5_6 = InternalNode(
      LinearSVM(in_features, key=keys[2]),
      leaves[5],
      leaves[6],
    )
    node_8_9 = InternalNode(
      LinearSVM(in_features, key=keys[3]),
      leaves[8],
      leaves[9],
    )

    node_0_1_2 = InternalNode(
      LinearSVM(in_features, key=keys[4]),
      node_0_1,
      leaves[2],
    )
    node_5_6_7 = InternalNode(
      LinearSVM(in_features, key=keys[5]),
      node_5_6,
      leaves[7],
    )

    node_0_4 = InternalNode(
      LinearSVM(in_features, key=keys[6]),
      node_0_1_2,
      node_3_4,
    )
    node_5_9 = InternalNode(
      LinearSVM(in_features, key=keys[7]),
      node_5_6_7,
      node_8_9,
    )

    self.root = InternalNode(LinearSVM(in_features, key=keys[8]), node_0_4, node_5_9)

  def __call__(self, x: Float[Array, "in_features"]) -> Float[Array, "num_classes"]:
    """Computes the forward pass of the model.

    Args:
        x: The input data.

    Returns:
        The model's output (a probability distribution over classes).

    """
    return self.root(x)


class SingleSVMModel(eqx.Module):
  """A single, multi-class SVM model."""

  svm: LinearSVM
  in_features: int
  num_classes: int

  def __init__(
    self,
    in_features: int,
    num_classes: int,
    *,
    key: "jax.random.PRNGKey",
  ):
    """Initializes the SingleSVMModel."""
    self.in_features = in_features
    self.num_classes = num_classes
    self.svm = LinearSVM(in_features, out_features=num_classes, key=key)

  def __call__(self, x: Float[Array, "in_features"]) -> Float[Array, "num_classes"]:
    """Computes the decision function for the SVM."""
    return self.svm(x)


class OvR_SVM_Model(eqx.Module):
  """A One-vs-Rest model using multiple LinearSVMs."""

  svms: list[LinearSVM]

  def __init__(self, in_features: int, num_classes: int, *, key: "jax.random.PRNGKey"):
    """Initializes the OvR_SVM_Model.

    Args:
        in_features: The number of input features.
        num_classes: The number of output classes.
        key: A JAX PRNG key for initializing the SVMs.

    """
    keys = jax.random.split(key, num_classes)
    self.svms = [LinearSVM(in_features, key=k) for k in keys]

  def __call__(self, x: Float[Array, "in_features"]) -> Float[Array, "num_classes"]:
    """Computes the decision function for each class.

    Args:
        x: The input data.

    Returns:
        A vector of decision function outputs, one for each class.

    """
    # Apply each SVM to the input x
    return jnp.stack([svm(x) for svm in self.svms])
