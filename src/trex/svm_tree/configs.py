"""Configurations for the SVM tree model and experiments."""

from dataclasses import dataclass, field
from enum import Enum


class ModelType(Enum):
    """Enum for model types."""

    BASE_TREE = "base_tree"
    SINGLE_SVM = "single_svm"
    LEARNABLE_HIERARCHICAL_SVM = "learnable_hierarchical_svm"
    HIERARCHICAL_SVM = "hierarchical_svm"
    DYNAMIC_HIERARCHICAL_SVM = "dynamic_hierarchical_svm"


@dataclass
class DataConfig:
    """Configuration for data loading."""

    batch_size: int = 128
    train_subset_size: int | None = None
    test_subset_size: int | None = None


@dataclass
class BaseModelConfig:
    """Base configuration for a model."""

    in_features: int = 28 * 28
    model_type: ModelType = ModelType.BASE_TREE


@dataclass
class HierarchicalSVMModelConfig(BaseModelConfig):
    """Configuration for the HierarchicalSVM model."""

    model_type: ModelType = ModelType.HIERARCHICAL_SVM
    depth: int = 2
    num_classes: int = 10


ModelConfig = BaseModelConfig | HierarchicalSVMModelConfig


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
    model: BaseModelConfig = field(default_factory=BaseModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass
class HierarchicalSVMConfig:
    """Top-level configuration for the HierarchicalSVM experiment."""

    data: DataConfig = field(default_factory=DataConfig)
    model: HierarchicalSVMModelConfig = field(default_factory=HierarchicalSVMModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass
class LearnableModelConfig:
    """Configuration for the learnable tree model."""

    in_features: int = 28 * 28
    sparsity_regularization_strength: float = 0.01
    graph_constraint_scale: float = 10.0


@dataclass
class LearnableMNISTConfig:
    """Top-level configuration for the MNIST experiment with a learnable tree."""

    data: DataConfig = field(default_factory=DataConfig)
    model: LearnableModelConfig = field(default_factory=LearnableModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass
class DynamicHierarchicalSVMModelConfig(BaseModelConfig):
    """Configuration for the DynamicHierarchicalSVM model."""

    model_type: ModelType = ModelType.DYNAMIC_HIERARCHICAL_SVM
    depth: int = 2
    num_classes: int = 10
    embedding_dim: int = 64


@dataclass
class DynamicHierarchicalSVMConfig:
    """Top-level configuration for the DynamicHierarchicalSVM experiment."""

    data: DataConfig = field(default_factory=DataConfig)
    model: DynamicHierarchicalSVMModelConfig = field(
        default_factory=DynamicHierarchicalSVMModelConfig
    )
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass
class LearnableHierarchicalSVMConfig:
    """Top-level configuration for the learnable hierarchical SVM."""

    data: DataConfig = field(default_factory=DataConfig)
    model: LearnableModelConfig = field(default_factory=LearnableModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
