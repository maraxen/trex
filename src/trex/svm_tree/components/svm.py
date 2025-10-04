
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class LinearSVM(eqx.Module):
    """A simple linear Support Vector Machine."""

    weights: Array
    bias: Array

    def __init__(
        self,
        in_features: int,
        *,
        out_features: int | None = None,
        key: "jax.random.PRNGKey",
    ):
        """Initializes the LinearSVM module.

        Args:
            in_features: The number of input features.
            out_features: The number of output classes. If None, a single output is
                assumed.
            key: A JAX PRNG key used to initialize the weights.

        """
        wkey, bkey = jax.random.split(key)
        if out_features is None:
            self.weights = jax.random.normal(wkey, (in_features,))
            self.bias = jax.random.normal(bkey, ())
        else:
            self.weights = jax.random.normal(wkey, (out_features, in_features))
            self.bias = jax.random.normal(bkey, (out_features,))

    def __call__(self, x: Float[Array, "in_features"]) -> Array:
        """Computes the decision function for the SVM.

        Args:
            x: The input data.

        Returns:
            The decision function output.

        """
        return jnp.dot(self.weights, x) + self.bias
