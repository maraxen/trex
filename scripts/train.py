import dataclasses
from collections.abc import Iterator
from functools import partial

from jax import config

config.update("jax_debug_nans", True)

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import tyro
import wandb
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from trex.svm_tree.configs import (
  LearnableHierarchicalSVMConfig,
  LearnableMNISTConfig,
  MNISTConfig,
  ModelType,
)
from trex.svm_tree.data_utils import get_mnist_dataloaders
from trex.svm_tree.model import (
  BaseTreeModel,
  LearnableHierarchicalSVM,
  LearnableTreeModel,
  OvR_SVM_Model,
)

# -------------------
# Model and Loss
# -------------------


def call_model_with_key(model, x, key):
  """Helper to call a model's __call__ method with a key."""
  return model(x, key=key)


def loss_fn(model, x, y, key, use_topo_loss: bool, topology_loss_weight: float):
  """Computes the total loss for a given model and batch.
  This function is pure and is called from within the JIT-compiled train_step.
  """
  # --- Differentiable model call ---
  # For models with stochastic components, we need a unique key per sample.
  if isinstance(model, (LearnableTreeModel, LearnableHierarchicalSVM)):
    keys = jax.random.split(key, x.shape[0])
    pred_y_batched = jax.vmap(call_model_with_key, in_axes=(None, 0, 0))(
      model,
      x,
      keys,
    )
    # --- Topology loss for learnable tree structures ---
    adj = model.topology(key)
    topo_loss = model.loss(adj)
  else:
    pred_y_batched = jax.vmap(model)(x)
    topo_loss = 0.0

  # --- Supervised classification loss ---
  ce_loss = optax.softmax_cross_entropy_with_integer_labels(
    pred_y_batched,
    y,
  ).mean()

  total_loss = ce_loss
  if use_topo_loss:
    total_loss += topology_loss_weight * topo_loss

  return total_loss


# -------------------
# JIT-compiled Steps
# -------------------


@eqx.filter_jit
def train_step(
  model: eqx.Module,
  x: jax.Array,
  y: jax.Array,
  optimizer: optax.GradientTransformation,
  opt_state: optax.OptState,
  key: jax.random.PRNGKey,
  # These arguments are static because they are not JAX arrays.
  # eqx.filter_jit handles this automatically.
  use_topo_loss: bool,
  topology_loss_weight: float,
) -> tuple[eqx.Module, optax.OptState, jax.Array]:
  """Performs a single JIT-compiled training step."""
  # Create a loss function partial that captures the static arguments.
  loss_fn_for_grad = partial(
    loss_fn,
    key=key,
    use_topo_loss=use_topo_loss,
    topology_loss_weight=topology_loss_weight,
  )

  # Calculate loss and gradients.
  loss, grads = eqx.filter_value_and_grad(loss_fn_for_grad)(model, x, y)

  # Apply updates.
  updates, opt_state = optimizer.update(grads, opt_state)
  model = eqx.apply_updates(model, updates)
  return model, opt_state, loss


@eqx.filter_jit
def eval_step(
  model: eqx.Module,
  x: jax.Array,
  y: jax.Array,
  key: jax.random.PRNGKey,
) -> tuple[jax.Array, jax.Array]:
  """Computes accuracy and predictions on a batch of data."""
  # For models with stochastic components (like LearnableTreeModel).
  if isinstance(model, (LearnableTreeModel, LearnableHierarchicalSVM)):
    keys = jax.random.split(key, x.shape[0])
    pred_y = jax.vmap(call_model_with_key, in_axes=(None, 0, 0))(model, x, keys)
  else:
    pred_y = jax.vmap(model)(x)

  pred_labels = jnp.argmax(pred_y, axis=1)
  accuracy = jnp.mean(pred_labels == y)
  return accuracy, pred_labels


# -------------------
# Data Handling
# -------------------


def data_stream(dataloader: DataLoader) -> Iterator[tuple[jax.Array, jax.Array]]:
  """An idiomatic JAX data loader. It wraps a PyTorch DataLoader to yield
  batches of JAX arrays, avoiding CPU-GPU synchronization within the training loop.
  """
  for x, y in dataloader:
    # Convert torch tensors to numpy arrays, then to JAX arrays.
    # This transfer happens once per batch, outside the JIT-compiled functions.
    yield jnp.asarray(x), jnp.asarray(y)


# -------------------
# Epoch Logic
# -------------------


def train_epoch(
  model: eqx.Module,
  optimizer: optax.GradientTransformation,
  opt_state: optax.OptState,
  train_loader: DataLoader,
  key: jax.random.PRNGKey,
  use_topo_loss: bool,
  topology_loss_weight: float,
) -> tuple[eqx.Module, optax.OptState, jax.Array, jax.random.PRNGKey]:
  """Handles the training logic for a single epoch."""
  total_loss = 0.0
  train_data = data_stream(train_loader)
  pbar = tqdm(
    train_data,
    total=len(train_loader),
    desc="Training",
  )

  for x, y in pbar:
    key, step_key = jax.random.split(key)
    model, opt_state, loss = train_step(
      model,
      x,
      y,
      optimizer,
      opt_state,
      step_key,
      use_topo_loss,
      topology_loss_weight,
    )
    total_loss += loss
    # .item() creates a sync point, but is acceptable for progress bars.
    pbar.set_postfix({"loss": f"{loss.item():.4f}"})

  avg_loss = total_loss / len(train_loader)
  return model, opt_state, avg_loss, key


def evaluate_model(
  model: eqx.Module,
  test_loader: DataLoader,
  key: jax.random.PRNGKey,
) -> tuple[jax.Array, np.ndarray, np.ndarray, jax.random.PRNGKey]:
  """Handles the evaluation logic for the model."""
  total_accuracy = 0.0
  all_preds, all_labels = [], []
  test_data = data_stream(test_loader)

  for x, y in test_data:
    key, step_key = jax.random.split(key)
    accuracy, pred_labels = eval_step(model, x, y, step_key)
    total_accuracy += accuracy
    # We collect predictions on the CPU for sklearn metrics later.
    all_preds.append(np.array(pred_labels))
    all_labels.append(np.array(y))

  avg_accuracy = total_accuracy / len(test_loader)
  # Concatenate all batch predictions into single numpy arrays.
  final_preds = np.concatenate(all_preds)
  final_labels = np.concatenate(all_labels)

  return avg_accuracy, final_preds, final_labels, key


# -------------------
# Main Execution
# -------------------


def main(
  cfg: MNISTConfig | LearnableMNISTConfig | LearnableHierarchicalSVMConfig,
):
  """Main training and evaluation loop."""
  if cfg.wandb.use_wandb:
    run_name = cfg.wandb.run_name
    if isinstance(cfg, LearnableHierarchicalSVMConfig):
      run_name = "learnable-hierarchical-svm"
    elif isinstance(cfg, LearnableMNISTConfig):
      run_name = "learnable-tree"
    elif cfg.model.model_type == ModelType.SINGLE_SVM:
      run_name = "single-svm"

    wandb.init(
      project=cfg.wandb.project,
      entity=cfg.wandb.entity,
      name=run_name,
      config=dataclasses.asdict(cfg),
    )

  # --- Setup ---
  key = jax.random.PRNGKey(cfg.train.seed)
  model_key, train_key, eval_key = jax.random.split(key, 3)

  train_loader, test_loader = get_mnist_dataloaders(
    cfg.data.batch_size,
    cfg.data.train_subset_size,
    cfg.data.test_subset_size,
  )

  # --- Model and Optimizer Initialization ---
  if isinstance(cfg, LearnableHierarchicalSVMConfig):
    model = LearnableHierarchicalSVM(
      in_features=cfg.model.in_features,
      num_classes=10,
      n_ancestors=cfg.model.n_ancestors,
      key=model_key,
      sparsity_regularization_strength=cfg.model.sparsity_regularization_strength,
      graph_constraint_scale=cfg.model.graph_constraint_scale,
    )
  elif isinstance(cfg, LearnableMNISTConfig):
    model = LearnableTreeModel(
      in_features=cfg.model.in_features,
      num_classes=10,
      n_ancestors=cfg.model.n_ancestors,
      key=model_key,
      sparsity_regularization_strength=cfg.model.sparsity_regularization_strength,
      graph_constraint_scale=cfg.model.graph_constraint_scale,
    )
  elif isinstance(cfg, MNISTConfig):
    if cfg.model.model_type == ModelType.BASE_TREE:
      model = BaseTreeModel(
        in_features=cfg.model.in_features,
        num_classes=10,
        key=model_key,
      )
    elif cfg.model.model_type == ModelType.SINGLE_SVM:
      model = OvR_SVM_Model(
        in_features=cfg.model.in_features,
        num_classes=10,
        key=model_key,
      )
    else:
      raise ValueError(f"Unknown model type: {cfg.model.model_type}")
  else:
    raise TypeError(f"Unknown config type: {type(cfg)}")

  use_topo_loss = isinstance(cfg, (LearnableMNISTConfig, LearnableHierarchicalSVMConfig))
  topology_loss_weight = cfg.train.topology_loss_weight

  optimizer = optax.adam(cfg.train.learning_rate)
  # Initialize optimizer state with the model's trainable parameters.
  opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

  # --- Training Loop ---
  print("Starting training...")
  for epoch in range(cfg.train.num_epochs):
    print(f"--- Epoch {epoch + 1}/{cfg.train.num_epochs} ---")

    # Train for one epoch
    model, opt_state, avg_loss, train_key = train_epoch(
      model,
      optimizer,
      opt_state,
      train_loader,
      train_key,
      use_topo_loss,
      topology_loss_weight,
    )

    # Evaluate the model
    avg_accuracy, all_preds, all_labels, eval_key = evaluate_model(
      model,
      test_loader,
      eval_key,
    )

    print(
      f"Epoch {epoch + 1}: Loss = {avg_loss.item():.4f}, Accuracy = {avg_accuracy.item():.4f}",
    )

    if cfg.wandb.use_wandb:
      log_data = {
        "epoch": epoch,
        "train_loss": avg_loss.item(),
        "test_accuracy": avg_accuracy.item(),
      }
      if isinstance(model, (LearnableHierarchicalSVM, LearnableTreeModel)):
        adj_key, _ = jax.random.split(eval_key)
        adj = model.topology(adj_key)
        plt.figure(figsize=(10, 10))
        plt.imshow(np.array(adj), cmap="hot", interpolation="nearest")
        plt.title(f"Learned Adjacency Matrix - Epoch {epoch + 1}")
        log_data["adjacency_matrix"] = wandb.Image(plt)
        plt.close()
      wandb.log(log_data)

  print("Training finished.")

  # --- Final Evaluation and Logging ---
  if cfg.wandb.use_wandb:
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax.set(
      xlabel="Predicted",
      ylabel="True",
      title="Confusion Matrix",
    )
    for i in range(cm.shape[0]):
      for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), va="center", ha="center")
    wandb.log({"confusion_matrix": wandb.Image(plt)})
    plt.close(fig)
    wandb.finish()


if __name__ == "__main__":
  COMMAND_DEFAULTS = {
    "base-tree": MNISTConfig(),
    "single-svm": MNISTConfig(
      model=dataclasses.replace(
        MNISTConfig().model,
        model_type=ModelType.SINGLE_SVM,
      ),
    ),
    "learnable-tree": LearnableMNISTConfig(),
    "learnable-hierarchical-svm": LearnableHierarchicalSVMConfig(),
  }
  Subcommands = tyro.extras.subcommand_type_from_defaults(COMMAND_DEFAULTS)

  cfg = tyro.cli(Subcommands, description="Train an SVM tree model on MNIST.")
  main(cfg)
