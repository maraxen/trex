"""Tests for trex.interactions (Protocol + weighted composition)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float, PRNGKeyArray

from trex.interactions import InteractionLoss, WeightedLoss, build_total_loss

SoftAncestors = Float[Array, "n_ancestors seq_len n_states"]
SoftAll = Float[Array, "n_nodes seq_len n_states"]
Adjacency = Array


def _zeros_interaction() -> InteractionLoss:
    def fn(
        soft_ancestors: SoftAncestors,
        soft_all: SoftAll,
        adj: Adjacency,
        key: PRNGKeyArray,
    ) -> Float[Array, ""]:
        del soft_ancestors, soft_all, adj, key
        return jnp.array(0.0, dtype=jnp.float32)

    return fn


def _sum_ancestors_interaction() -> InteractionLoss:
    """Scalar = sum of soft_ancestors (deterministic, differentiable smoke)."""

    def fn(
        soft_ancestors: SoftAncestors,
        soft_all: SoftAll,
        adj: Adjacency,
        key: PRNGKeyArray,
    ) -> Float[Array, ""]:
        del soft_all, adj, key
        return jnp.sum(soft_ancestors)

    return fn


@pytest.fixture
def toy_shapes() -> tuple[SoftAncestors, SoftAll, Adjacency, PRNGKeyArray]:
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    soft_ancestors = jax.random.uniform(k1, (2, 4, 3))
    soft_all = jax.random.uniform(k2, (5, 4, 3))
    adj = jnp.zeros((5, 5), dtype=jnp.float32)
    return soft_ancestors, soft_all, adj, key


def test_build_total_loss_empty_returns_zero(toy_shapes):
    soft_ancestors, soft_all, adj, key = toy_shapes
    total_fn = build_total_loss(())
    total, metrics = total_fn(soft_ancestors, soft_all, adj, key)
    assert total.shape == ()
    assert float(total) == pytest.approx(0.0)
    assert metrics == {}


def test_build_total_loss_single_weighted(toy_shapes):
    soft_ancestors, soft_all, adj, key = toy_shapes
    wl = WeightedLoss(name="a", weight=2.0, fn=_sum_ancestors_interaction())
    total_fn = build_total_loss((wl,))
    total, metrics = total_fn(soft_ancestors, soft_all, adj, key)
    raw = float(jnp.sum(soft_ancestors))
    assert float(total) == pytest.approx(2.0 * raw)
    assert set(metrics) == {"a"}
    assert float(metrics["a"]) == pytest.approx(2.0 * raw)


def test_build_total_loss_two_stochastic_keys_split(toy_shapes):
    """Each loss gets a distinct subkey (split count matches tuple length)."""
    soft_ancestors, soft_all, adj, key = toy_shapes
    wl1 = WeightedLoss(name="z", weight=1.0, fn=_zeros_interaction())
    wl2 = WeightedLoss(name="s", weight=0.5, fn=_sum_ancestors_interaction())
    total_fn = build_total_loss((wl1, wl2))
    total, metrics = total_fn(soft_ancestors, soft_all, adj, key)
    raw = float(jnp.sum(soft_ancestors))
    assert float(total) == pytest.approx(0.5 * raw)
    assert float(metrics["z"]) == pytest.approx(0.0)
    assert float(metrics["s"]) == pytest.approx(0.5 * raw)


def test_build_total_loss_split_keys_false_shares_key(toy_shapes):
    """split_keys=False passes the same key to every loss (benchmark / MPNN contract)."""
    soft_ancestors, soft_all, adj, key = toy_shapes
    wl1 = WeightedLoss(name="z", weight=1.0, fn=_zeros_interaction())
    wl2 = WeightedLoss(name="s", weight=0.5, fn=_sum_ancestors_interaction())
    total_fn = build_total_loss((wl1, wl2), split_keys=False)
    total, metrics = total_fn(soft_ancestors, soft_all, adj, key)
    raw = float(jnp.sum(soft_ancestors))
    assert float(total) == pytest.approx(0.5 * raw)
    assert float(metrics["z"]) == pytest.approx(0.0)
    assert float(metrics["s"]) == pytest.approx(0.5 * raw)
