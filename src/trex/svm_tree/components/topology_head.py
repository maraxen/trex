import equinox as eqx
import jax


class TopologyHead(eqx.Module):
    """An MLP that generates tree topology parameters from a feature embedding."""

    mlp: eqx.nn.MLP

    def __init__(
        self,
        *,
        embedding_dim: int,
        n_leaves: int,
        n_ancestors: int,
        key: jax.random.PRNGKey,
    ):
        """Initializes the TopologyHead.

        Args:
            embedding_dim: The dimension of the input feature embedding.
            n_leaves: The number of leaf nodes in the target tree.
            n_ancestors: The number of ancestor nodes in the target tree.
            key: A JAX random key for initializing the MLP.
        """
        n_all_minus_1 = n_leaves + n_ancestors - 1
        output_dim = n_all_minus_1 * n_ancestors
        self.mlp = eqx.nn.MLP(
            in_size=embedding_dim,
            out_size=output_dim,
            width_size=128,
            depth=2,
            activation=jax.nn.relu,
            key=key,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Generates tree parameters from the feature embedding.

        Args:
            x: The input feature embedding.

        Returns:
            The generated tree parameters.
        """
        return self.mlp(x)