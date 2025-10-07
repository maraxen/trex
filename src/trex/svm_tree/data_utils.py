import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets


def get_mnist_dataloaders(batch_size: int, train_subset_size: int = None, test_subset_size: int = None):
    """Returns MNIST dataloaders for training and testing."""
    # Download and load the training data
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=None,
    )
    train_images = train_dataset.data.numpy().reshape(-1, 28 * 28).astype(np.float32)
    train_labels = train_dataset.targets.numpy().astype(np.int32)

    # Download and load the test data
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=None,
    )
    test_images = test_dataset.data.numpy().reshape(-1, 28 * 28).astype(np.float32)
    test_labels = test_dataset.targets.numpy().astype(np.int32)

    # Normalize the data
    train_images /= 255.0
    test_images /= 255.0

    # Optionally subset the data
    if train_subset_size:
        train_images = train_images[:train_subset_size]
        train_labels = train_labels[:train_subset_size]
    if test_subset_size:
        test_images = test_images[:test_subset_size]
        test_labels = test_labels[:test_subset_size]

    # Create torch datasets and dataloaders
    train_dataset_torch = TensorDataset(
        torch.from_numpy(train_images), torch.from_numpy(train_labels),
    )
    test_dataset_torch = TensorDataset(
        torch.from_numpy(test_images), torch.from_numpy(test_labels),
    )

    train_loader = DataLoader(train_dataset_torch, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset_torch, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
