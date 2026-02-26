"""Shared pytest fixtures for sheaf-transformers tests."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Seeded random number generator for reproducible tests."""
    return np.random.RandomState(42)


@pytest.fixture
def small_heads(rng):
    """Small synthetic head outputs (H=4, n=8, d=4)."""
    H, n, d = 4, 8, 4
    return rng.randn(H, n, d)


@pytest.fixture
def small_attention(rng):
    """Small synthetic attention matrices (H=4, n=8)."""
    H, n = 4, 8
    return rng.dirichlet(np.ones(n), size=(H, n))


@pytest.fixture
def coherent_heads(rng):
    """Coherent heads: all identical (should give H_coh ~ 0)."""
    n, d, H = 8, 4, 4
    base = rng.randn(n, d)
    return np.stack([base] * H)


@pytest.fixture
def large_heads(rng):
    """Larger synthetic data for stress tests (H=6, n=20, d=8)."""
    H, n, d = 6, 20, 8
    return rng.randn(H, n, d)


@pytest.fixture
def large_attention(rng):
    """Larger synthetic attention matrices (H=6, n=20)."""
    H, n = 6, 20
    return rng.dirichlet(np.ones(n), size=(H, n))
