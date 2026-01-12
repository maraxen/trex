"""End-to-end convergence tests for isolated and joint optimization.

1.  test_fixed_topology_learned_sequences:
    - Tree topology is FIXED to the ground truth.
    - ONLY ancestor sequences are learned.
    - This tests if the sequence optimization and surrogate cost can find the
      optimal reconstruction for a known-good tree.
    - EXPECTATION: Must converge to the Sankoff score.

2.  test_fixed_sequences_learned_topology:
    - Ancestor sequences are FIXED to the Sankoff-reconstructed ground truth.
    - ONLY the tree topology is learned.
    - This tests if the tree optimization can find the correct topology
      when given the optimal sequences.
    - EXPECTATION: Must converge to the Sankoff score.

3.  test_joint_optimization:
    - BOTH tree topology and ancestor sequences are learned.
    - This is the original, difficult problem.
    - This tests the ability of the system to escape local minima.
    - EXPECTATION: May get stuck in local minima, but provides a benchmark.
"""

import jax
import jax.numpy as jnp
import optax

from trex.ground_truth import generate_groundtruth
from trex.sankoff import run_sankoff
from trex.tree import (
  compute_cost,
  compute_loss,
  compute_surrogate_cost,
  discretize_tree_topology,
  enforce_graph_constraints,
  update_seq,
  update_tree,
)


# Helper function to set up the common ground truth data
def setup_test_case():
  n_leaves = 4
  seq_length = 20
  n_states = 4
  n_mutations = 3
  n_ancestors = n_leaves - 1
  n_all = n_leaves + n_ancestors

  gt = generate_groundtruth(n_leaves, n_states, n_mutations, seq_length, seed=42)
  cost_matrix = jnp.ones((n_states, n_states), dtype=jnp.float32) - jnp.eye(
    n_states,
    dtype=jnp.float32,
  )

  sankoff_reconstructed_seqs, _, true_parsimony_score = run_sankoff(
    adjacency_matrix=gt.adjacency.astype(jnp.int32),
    cost_matrix=cost_matrix,
    sequences=gt.all_sequences[:n_leaves],
    n_all=n_all,
    n_states=n_states,
    n_leaves=n_leaves,
    return_path=True,
  )

  # This is the full, optimal sequence set (leaves + reconstructed ancestors)

  # Sanity check that the Sankoff score and compute_cost score are consistent
  assert jnp.isclose(
    true_parsimony_score,
    compute_cost(jax.nn.one_hot(sankoff_reconstructed_seqs, n_states), gt.adjacency, cost_matrix),
    atol=1e-3,
  )

  return {
    "gt": gt,
    "sankoff_seqs": sankoff_reconstructed_seqs,
    "true_parsimony_score": true_parsimony_score,
    "cost_matrix": cost_matrix,
    "n_leaves": n_leaves,
    "n_ancestors": n_ancestors,
    "n_all": n_all,
    "seq_length": seq_length,
    "n_states": n_states,
  }


# Test 1: Fix the tree, learn the sequences
def test_fixed_topology_learned_sequences():
  """Tests if sequence optimization converges given the correct tree."""
  data = setup_test_case()
  key = jax.random.PRNGKey(42)

  key, tree_key, seq_key = jax.random.split(key, 3)
  params = {
    "tree_params": jax.random.normal(tree_key, (data["n_all"] - 1, data["n_ancestors"])),
    "ancestors": jax.random.normal(
      seq_key,
      (data["n_ancestors"], data["seq_length"], data["n_states"]),
    ),
  }

  optimizer = optax.adam(learning_rate=0.01)
  opt_state = optimizer.init(params)

  initial_sequences_for_loss = jnp.zeros(
    (data["n_all"], data["seq_length"], data["n_states"]),
  )
  initial_sequences_for_loss = initial_sequences_for_loss.at[: data["n_leaves"]].set(
    jax.nn.one_hot(data["gt"].all_sequences[: data["n_leaves"]], data["n_states"]),
  )
  gt_adjacency_one_hot = jax.nn.one_hot(
    jnp.argmax(data["gt"].adjacency, axis=1),
    num_classes=data["n_all"],
  )

  @jax.jit
  def train_step(current_params, current_opt_state):
    def loss_fn(p):
      updated_sequences = update_seq(p, initial_sequences_for_loss, temperature=1.0)
      return compute_surrogate_cost(updated_sequences, gt_adjacency_one_hot)

    loss_val, grads = jax.value_and_grad(loss_fn)(current_params)
    grad_mask = {
      "tree_params": jnp.zeros_like(current_params["tree_params"]),
      "ancestors": jnp.ones_like(current_params["ancestors"]),
    }
    active_grads = jax.tree_util.tree_map(lambda g, m: g * m, grads, grad_mask)
    updates, new_opt_state = optimizer.update(active_grads, current_opt_state, current_params)
    new_params = optax.apply_updates(current_params, updates)
    return new_params, new_opt_state, loss_val

  for _epoch in range(5000):
    params, opt_state, _loss = train_step(params, opt_state)

  # Evaluation
  learned_sequences_one_hot = update_seq(params, initial_sequences_for_loss, temperature=0.01)

  learned_parsimony_score = compute_cost(
    learned_sequences_one_hot,
    data["gt"].adjacency,
    data["cost_matrix"],
  )
  print(f"\n[Fixed Topology] True Sankoff Score: {data['true_parsimony_score']:.2f}")
  print(f"[Fixed Topology] Learned Parsimony Score: {learned_parsimony_score:.2f}")
  assert jnp.isclose(data["true_parsimony_score"], learned_parsimony_score, atol=2.0)


# Test 2: Fix the sequences, learn the tree
def test_fixed_sequences_learned_topology():
  """Tests if tree optimization converges given the correct sequences."""
  data = setup_test_case()
  key = jax.random.PRNGKey(42)

  key, tree_key = jax.random.split(key)
  params = {
    "tree_params": jax.random.normal(tree_key, (data["n_all"] - 1, data["n_ancestors"])),
    "ancestors": jnp.zeros((data["n_ancestors"], data["seq_length"], data["n_states"])),
  }

  optimizer = optax.adam(learning_rate=0.05)
  opt_state = optimizer.init(params)
  true_sequences_one_hot = jax.nn.one_hot(data["sankoff_seqs"], data["n_states"])

  @jax.jit
  def train_step(current_params, current_opt_state, step_key):
    def loss_fn(p):
      updated_tree = update_tree(step_key, p, temperature=1.0)
      surrogate_cost = compute_surrogate_cost(true_sequences_one_hot, updated_tree)
      constraint_loss = enforce_graph_constraints(updated_tree, 10.0)
      return surrogate_cost + constraint_loss

    loss_val, grads = jax.value_and_grad(loss_fn)(current_params)
    grad_mask = {
      "tree_params": jnp.ones_like(current_params["tree_params"]),
      "ancestors": jnp.zeros_like(current_params["ancestors"]),
    }
    active_grads = jax.tree_util.tree_map(lambda g, m: g * m, grads, grad_mask)
    updates, new_opt_state = optimizer.update(active_grads, current_opt_state, current_params)
    new_params = optax.apply_updates(current_params, updates)
    return new_params, new_opt_state, loss_val

  for _epoch in range(3000):
    key, step_key = jax.random.split(key)
    params, opt_state, _loss = train_step(params, opt_state, step_key)

  # Evaluation
  learned_tree_soft = update_tree(key, params, temperature=0.01)
  learned_parents = jnp.argmax(discretize_tree_topology(learned_tree_soft, data["n_all"]), axis=1)
  gt_parents = jnp.argmax(discretize_tree_topology(data["gt"].adjacency, data["n_all"]), axis=1)

  print("\n[Fixed Sequences] Learned Parent Assignments (non-root):")
  print(learned_parents[:-1])  # Exclude the root
  print("[Fixed Sequences] Ground Truth Parent Assignments (non-root):")
  print(gt_parents[:-1])  # Exclude the root

  learned_parsimony_score = compute_cost(
    true_sequences_one_hot,
    learned_tree_soft,
    data["cost_matrix"],
  )
  print(f"[Fixed Sequences] True Sankoff Score: {data['true_parsimony_score']:.2f}")
  print(f"[Fixed Sequences] Learned Parsimony Score: {learned_parsimony_score:.2f}")
  assert jnp.isclose(data["true_parsimony_score"], learned_parsimony_score, atol=1.0)


# Test 3: Learn both jointly
def test_joint_optimization():
  """Tests joint optimization of both topology and sequences."""
  data = setup_test_case()
  key = jax.random.PRNGKey(42)

  key, tree_key, seq_key = jax.random.split(key, 3)
  params = {
    "tree_params": jax.random.normal(
      tree_key,
      (data["n_all"] - 1, data["n_ancestors"]),
      dtype=jnp.float32,  # Change dtype to float32
    ),
    "ancestors": jax.random.normal(
      seq_key,
      (data["n_ancestors"], data["seq_length"], data["n_states"]),
      dtype=jnp.float32,  # Change dtype to float32
    ),
  }

  optimizer = optax.adam(learning_rate=0.01)
  opt_state = optimizer.init(params)

  initial_sequences_for_loss = jnp.zeros(
    (data["n_all"], data["seq_length"], data["n_states"]),
  )
  initial_sequences_for_loss = initial_sequences_for_loss.at[: data["n_leaves"]].set(
    jax.nn.one_hot(data["gt"].all_sequences[: data["n_leaves"]], data["n_states"]),
  )
  metadata = {"n_all": data["n_all"], "n_states": data["n_states"]}

  @jax.jit
  def train_step(current_params, current_opt_state, temperature, step_key):
    def loss_fn(p):
      # Dummy adjacency, not used when fix_tree=False
      dummy_adjacency = jnp.zeros((data["n_all"], data["n_all"]))
      return compute_loss(
        step_key,
        p,
        initial_sequences_for_loss,
        metadata,
        temperature=temperature,
        adjacency=dummy_adjacency,
        graph_constraint_scale=10.0,
      )

    loss, grads = jax.value_and_grad(loss_fn)(current_params)
    updates, new_opt_state = optimizer.update(grads, current_opt_state, current_params)
    new_params = optax.apply_updates(current_params, updates)
    return new_params, new_opt_state, loss

  for epoch in range(5000):
    key, step_key = jax.random.split(key)
    temp = jnp.maximum(0.1, 2.0 * (1.0 - epoch / 5000))
    params, opt_state, _loss = train_step(params, opt_state, temp, step_key)

  # Evaluation
  learned_seqs = update_seq(params, initial_sequences_for_loss, temperature=0.01)
  learned_seqs_one_hot = jax.nn.one_hot(
    jnp.argmax(learned_seqs, axis=-1),
    data["n_states"],
  )
  learned_tree_soft = update_tree(key, params, temperature=0.01)
  learned_parents = discretize_tree_topology(learned_tree_soft, data["n_all"])
  gt_parents = discretize_tree_topology(data["gt"].adjacency, data["n_all"])

  print("\n[Fixed Sequences] Learned Parent Assignments (non-root):")
  print(learned_parents[:-1])  # Exclude the root
  print("[Fixed Sequences] Ground Truth Parent Assignments (non-root):")
  print(gt_parents[:-1])  # Exclude the root

  # reorder learned tree and seqs based on learned parents
  # so that we can compare the parsimony scores directly
  learned_tree_soft = learned_tree_soft[
    jnp.ix_(jnp.argmax(learned_parents, axis=1), jnp.argmax(learned_parents, axis=1))
  ]
  learned_seqs_one_hot = learned_seqs_one_hot[
    jnp.ix_(jnp.argmax(learned_parents, axis=1), jnp.argmax(learned_parents, axis=1))
  ]

  print("\n[Joint Optimization] Reordered Learned Tree:")
  print(discretize_tree_topology(learned_tree_soft, data["n_all"])[:-1])  # Exclude the root
  print("[Joint Optimization] Ground Truth Tree:")
  print(gt_parents[:-1])  # Exclude the root

  learned_parsimony_score = compute_cost(
    learned_seqs_one_hot,
    learned_tree_soft,
    data["cost_matrix"],
  )
  print(f"[Fixed Sequences] True Sankoff Score: {data['true_parsimony_score']:.2f}")
  print(f"[Fixed Sequences] Learned Parsimony Score: {learned_parsimony_score:.2f}")
  assert learned_parsimony_score <= data["true_parsimony_score"]
