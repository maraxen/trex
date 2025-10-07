from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ModelType(Enum):
    """Enum for model types."""

    BASE_TREE = "base_tree"
    SINGLE_SVM = "single_svm"
    LEARNABLE_HIERARCHICAL_SVM = "learnable_hierarchical_svm"


@dataclass
class DataConfig:
    """Configuration for data loading."""

    batch_size: int = 128
    train_subset_size: Optional[int] = None
    test_subset_size: Optional[int] = None


@dataclass
class ModelConfig:
    """Configuration for the model."""

    in_features: int = 28 * 28
    model_type: ModelType = ModelType.BASE_TREE

@dataclass
class TrainConfig:
    """Configuration for training."""

    learning_rate: float = 1e-3
    num_epochs: int = 10
    seed: int = 42
    topology_loss_weight: float = 1.0


@dataclass
class WandbConfig:
    """Configuration for Weights & Biases."""

    use_wandb: bool = True
    project: str = "differentiable-svm-tree"
    entity: str = "trex"
    run_name: str = "mnist_baseline"


@dataclass
class MNISTConfig:
    """Top-level configuration for the MNIST experiment."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass
class LearnableModelConfig:
    """Configuration for the learnable tree model."""

    in_features: int = 28 * 28
    sparsity_regularization_strength: float = 0.01
    graph_constraint_scale: float = 10.0
    n_ancestors: int = 2


@dataclass
class LearnableMNISTConfig:
    """Top-level configuration for the MNIST experiment with a learnable tree."""

    data: DataConfig = field(default_factory=DataConfig)
    model: LearnableModelConfig = field(default_factory=LearnableModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass
class LearnableHierarchicalSVMConfig:
    """Top-level configuration for the learnable hierarchical SVM."""

    data: DataConfig = field(default_factory=DataConfig)
    model: LearnableModelConfig = field(default_factory=LearnableModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
