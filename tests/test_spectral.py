"""Tests for sheaf Laplacian and spectral analysis."""

import numpy as np
import pytest

from sheaf_transformers.metrics.spectral import SheafLaplacian
from sheaf_transformers.metrics.eigenvalue_stats import EigenvalueStatistics


class TestSheafLaplacian:

    def test_laplacian_shape(self, rng):
        """L_F has shape (n*d, n*d)."""
        n, d, H = 6, 3, 2
        A = rng.rand(H, n, n)
        A /= A.sum(axis=-1, keepdims=True)
        W = rng.randn(H, d, d) * 0.1
        sl = SheafLaplacian(n, d)
        L = sl.construct_laplacian(A, W)
        assert L.shape == (n * d, n * d)

    def test_laplacian_symmetric(self, rng):
        """L_F = B^T B must be symmetric."""
        n, d, H = 8, 4, 3
        A = rng.rand(H, n, n)
        A /= A.sum(axis=-1, keepdims=True)
        W = rng.randn(H, d, d) * 0.1
        sl = SheafLaplacian(n, d)
        L = sl.construct_laplacian(A, W)
        assert np.allclose(L, L.T, atol=1e-10)

    def test_laplacian_psd(self, rng):
        """L_F must be positive semi-definite."""
        n, d, H = 8, 4, 3
        A = rng.rand(H, n, n)
        A /= A.sum(axis=-1, keepdims=True)
        W = rng.randn(H, d, d) * 0.1
        sl = SheafLaplacian(n, d)
        L = sl.construct_laplacian(A, W)
        eigs = np.linalg.eigvalsh(L)
        assert np.all(eigs >= -1e-10), f"Min eigenvalue = {eigs.min()}"

    def test_spectral_gap_nonnegative(self, rng):
        """Spectral gap gamma >= 0."""
        n, d, H = 6, 3, 2
        A = rng.rand(H, n, n)
        A /= A.sum(axis=-1, keepdims=True)
        W = rng.randn(H, d, d) * 0.1
        sl = SheafLaplacian(n, d)
        L = sl.construct_laplacian(A, W)
        _, gap = sl.compute_spectral_gap(L)
        assert gap >= -1e-10

    def test_sheaf_energy_equals_quadratic_form(self, rng):
        """E = ||B_F s||^2 = s^T L_F s."""
        n, d, H = 6, 3, 2
        A = rng.rand(H, n, n)
        A /= A.sum(axis=-1, keepdims=True)
        W = rng.randn(H, d, d) * 0.1
        sl = SheafLaplacian(n, d)
        L = sl.construct_laplacian(A, W)
        s = rng.randn(n * d)
        energy_B = sl.compute_sheaf_energy(s, A, W)
        energy_L = float(s @ L @ s)
        assert abs(energy_B - energy_L) < 1e-8

    def test_constant_sheaf_identity_maps(self, rng):
        """With identity restriction maps, L_F reduces to graph Laplacian."""
        n, d, H = 5, 3, 2
        A = rng.rand(H, n, n)
        A /= A.sum(axis=-1, keepdims=True)
        W = np.stack([np.eye(d)] * H)
        sl = SheafLaplacian(n, d)
        L = sl.construct_laplacian(A, W)
        assert L.shape == (n * d, n * d)
        assert np.allclose(L, L.T, atol=1e-10)

    def test_zero_attention_no_edges(self):
        """Zero attention below threshold => no edges => L = 0."""
        n, d, H = 4, 2, 2
        A = np.zeros((H, n, n))
        W = np.eye(d)[np.newaxis].repeat(H, axis=0)
        sl = SheafLaplacian(n, d, threshold=0.5)
        L = sl.construct_laplacian(A, W)
        assert np.allclose(L, 0)

    def test_coboundary_dimensions(self, rng):
        """B has shape (n_edges * d, n * d)."""
        n, d, H = 6, 3, 2
        A = rng.rand(H, n, n)
        A /= A.sum(axis=-1, keepdims=True)
        W = rng.randn(H, d, d) * 0.1
        sl = SheafLaplacian(n, d)
        B, edges = sl.construct_coboundary(A, W)
        assert B.shape[0] == len(edges) * d
        assert B.shape[1] == n * d

    def test_spectral_gap_returns_eigenvalues(self, rng):
        """compute_spectral_gap returns k eigenvalues."""
        n, d, H = 6, 3, 2
        A = rng.rand(H, n, n)
        A /= A.sum(axis=-1, keepdims=True)
        W = rng.randn(H, d, d) * 0.1
        sl = SheafLaplacian(n, d)
        L = sl.construct_laplacian(A, W)
        eigs, gap = sl.compute_spectral_gap(L, k=5)
        assert len(eigs) <= 5
        assert isinstance(gap, float)


class TestEigenvalueStatistics:

    def test_unfolded_spacings_nonneg(self):
        """Unfolded spacings should be non-negative."""
        eigs = np.array([0.1, 0.3, 0.5, 0.8, 1.2, 1.8, 2.5])
        spacings = EigenvalueStatistics.unfold_eigenvalues(eigs)
        assert np.all(spacings >= 0)

    def test_goe_pdf_normalized(self):
        """GOE Wigner surmise integrates to ~1."""
        s = np.linspace(0, 6, 1000)
        ds = s[1] - s[0]
        integral = np.sum(EigenvalueStatistics.goe_pdf(s)) * ds
        assert abs(integral - 1.0) < 0.05

    def test_poisson_pdf_normalized(self):
        """Poisson PDF integrates to ~1."""
        s = np.linspace(0, 10, 1000)
        ds = s[1] - s[0]
        integral = np.sum(EigenvalueStatistics.poisson_pdf(s)) * ds
        assert abs(integral - 1.0) < 0.05

    def test_analyze_returns_finite(self, rng):
        """Analysis should return finite values."""
        A = rng.randn(20, 20)
        analyzer = EigenvalueStatistics()
        stats = analyzer.analyze(A)
        assert np.isfinite(stats["kl_goe"])
        assert np.isfinite(stats["kl_poisson"])

    def test_analyze_returns_spectral_gap(self, rng):
        """Analysis returns spectral_gap key."""
        A = rng.randn(20, 20)
        analyzer = EigenvalueStatistics()
        stats = analyzer.analyze(A)
        assert "spectral_gap" in stats
        assert np.isfinite(stats["spectral_gap"])

    def test_goe_matrix_closer_to_goe(self, rng):
        """A GOE-like random symmetric matrix should be closer to GOE distribution."""
        n = 100
        A = rng.randn(n, n)
        A = (A + A.T) / 2  # symmetric
        analyzer = EigenvalueStatistics()
        stats = analyzer.analyze(A)
        # GOE matrices have smaller KL to GOE than to Poisson
        assert stats["kl_goe"] < stats["kl_poisson"]

    def test_small_matrix_returns_inf(self):
        """Very small matrices (< 10 spacings) return inf."""
        A = np.array([[1, 0.5], [0.5, 1]])
        analyzer = EigenvalueStatistics()
        stats = analyzer.analyze(A)
        assert stats["kl_goe"] == np.inf

    def test_unfold_constant_spacings(self):
        """Uniform spacings should unfold to all 1.0."""
        eigs = np.arange(10, dtype=float)
        spacings = EigenvalueStatistics.unfold_eigenvalues(eigs)
        assert np.allclose(spacings, 1.0)
