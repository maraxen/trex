"""Composable interaction losses for landscape-aware TREX (v4 P1).

Public trex must not import private packages (e.g. ``asr``). Callers inject
Potts / MPNN / custom losses as callables satisfying :class:`InteractionLoss`.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Protocol

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

SoftAncestors = Float[Array, "n_ancestors seq_len n_states"]
SoftAll = Float[Array, "n_nodes seq_len n_states"]
Adjacency = Array


class InteractionLoss(Protocol):
    """Interaction-aware regularizer on soft sequences + optional tree context."""

    def __call__(
        self,
        soft_ancestors: SoftAncestors,
        soft_all: SoftAll,
        adj: Adjacency,
        key: PRNGKeyArray,
    ) -> Float[Array, ""]:
        ...


class WeightedLoss(NamedTuple):
    """Named loss with scalar weight."""

    name: str
    weight: float
    fn: InteractionLoss


def build_total_loss(
    interaction_losses: tuple[WeightedLoss, ...],
    *,
    split_keys: bool = True,
) -> Callable[
    [SoftAncestors, SoftAll, Adjacency, PRNGKeyArray],
    tuple[Float[Array, ""], dict[str, Float[Array, ""]]],
]:
    """Combine weighted interaction losses; returns (total, per-name metrics).

    Args:
        interaction_losses: Tuple of :class:`WeightedLoss` entries (may be empty).
        split_keys: If True (default), each loss receives an independent subkey from
            ``jax.random.split``. If False, every loss receives the same ``key`` (use when
            composing losses that share one RNG stream, e.g. MPNN with the key passed into
            :func:`trex.evals.benchmark.run_trex_landscape_aware_configurable`).

    Returns:
        A function ``(soft_ancestors, soft_all, adj, key) -> (total, metrics)``.
        ``metrics`` maps each ``name`` to its weighted contribution ``w * L_i``.
    """

    def total_fn(
        soft_ancestors: SoftAncestors,
        soft_all: SoftAll,
        adj: Adjacency,
        key: PRNGKeyArray,
    ) -> tuple[Float[Array, ""], dict[str, Float[Array, ""]]]:
        if not interaction_losses:
            return jnp.array(0.0, dtype=jnp.float32), {}
        if split_keys:
            keys = jax.random.split(key, len(interaction_losses))
        else:
            keys = (key,) * len(interaction_losses)
        metrics: dict[str, Float[Array, ""]] = {}
        parts: list[Float[Array, ""]] = []
        for wl, k in zip(interaction_losses, keys, strict=True):
            li = wl.fn(soft_ancestors, soft_all, adj, k)
            weighted = wl.weight * li
            metrics[wl.name] = weighted
            parts.append(weighted)
        return jnp.sum(jnp.stack(parts)), metrics

    return total_fn
