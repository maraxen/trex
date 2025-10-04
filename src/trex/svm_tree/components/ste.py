"""Straight-through estimator for JAX."""


import equinox as eqx
import jax
import jax.numpy as jnp
from jax import custom_vjp
from jaxtyping import Array, Float


@custom_vjp
def straight_through_estimator(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Use identity function with a straight-through estimator for the gradient."""
    return x


def ste_fwd(x: Float[Array, "..."]) -> tuple[Float[Array, "..."], None]:
    """Define the forward pass for the STE."""
    return x, None


def ste_bwd(_: None, grad: Float[Array, "..."]) -> tuple[Float[Array, "..."]]:
    """Define the backward pass for the STE."""
    return (grad,)


straight_through_estimator.defvjp(ste_fwd, ste_bwd)


def hard_decision(
    x: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Apply a hard threshold at 0, with a straight-through gradient."""
    zero = x - jax.lax.stop_gradient(x)
    return straight_through_estimator(jnp.heaviside(x, 0.5) + zero)


class STEGating(eqx.Module):
    """A module that applies a hard decision with a straight-through estimator."""

    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        """Apply a hard threshold at 0, with a straight-through gradient.

        Args:
            x: The input data.

        Returns:
            The hard decision output.

        """
        return hard_decision(x)
