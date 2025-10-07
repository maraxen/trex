import equinox as eqx
import jax
import jax.numpy as jnp


class CNNFeatureEncoder(eqx.Module):
    """A simple CNN to extract features from images."""

    layers: list

    def __init__(self, key: jax.random.PRNGKey, embedding_dim: int = 64):
        """Initializes the CNNFeatureEncoder.

        Args:
            key: A JAX random key for initializing the parameters.
            embedding_dim: The dimension of the output feature embedding.
        """
        key1, key2, key3 = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, key=key1),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.MaxPool2d(kernel_size=2, stride=2),
            eqx.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, key=key2),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.MaxPool2d(kernel_size=2, stride=2),
            eqx.nn.Lambda(lambda x: x.flatten()),
            eqx.nn.Linear(in_features=32 * 5 * 5, out_features=embedding_dim, key=key3),
            eqx.nn.Lambda(jax.nn.relu),
        ]

    def __call__(self, x: jax.Array) -> jax.Array:
        """Computes the feature embedding for the input image.

        Args:
            x: The input image, expected to be of shape (1, 28, 28).

        Returns:
            The feature embedding of the specified dimension.
        """
        # Add a channel dimension if it's missing
        if x.ndim == 2:
            x = x[None, :, :]
        # Ensure the input has a channel dimension for the conv layers
        if x.ndim == 3:
            x = x[None, ...]

        for layer in self.layers:
            x = layer(x)
        return x