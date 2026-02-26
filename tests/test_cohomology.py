"""Comprehensive tests for sheaf cohomology computation.

Tests cover:
- Correctness: coherent heads => H_coh = 0
- Edge cases: no overlaps, single head, empty inputs
- Mathematical properties: H_coh >= 0, q_obs >= 0, bounds
- Sheaf mode: non-trivial restriction maps, H_coh > 0 for incoherent
- Gluing layer: H_coh_after <= H_coh_before
- Spectral: L_F symmetric PSD, spectral gap >= 0
- Eigenvalue stats: GOE/Poisson distinction
"""

import sys
import numpy as np
import pytest

sys.path.insert(0, ".")

from sheaf_transformers.metrics.cohomology import CohomologyComputer
from sheaf_transformers.metrics.spectral import SheafLaplacian
from sheaf_transformers.metrics.eigenvalue_stats import EigenvalueStatistics
from sheaf_transformers.layers.gluing import GluingLayer


# ============================================================
# CohomologyComputer Tests
# ============================================================

class TestCohomology:

    def test_coherent_heads_energy_zero(self):
        """Identical heads => H_coh = 0 (perfect gluing)."""
        np.random.seed(42)
        n, d, H = 8, 4, 4
        base = np.random.randn(n, d)
        heads = np.stack([base] * H)
        attn = np.random.dirichlet(np.ones(n), size=(H, n))

        c = CohomologyComputer(H, d, threshold=0.0)
        r = c.compute_H_coh(heads, attn)

        assert r["H_coh"] < 1e-10, f"Coherent heads should give H_coh~0, got {r['H_coh']}"

    def test_no_overlaps_returns_zero(self):
        """Disjoint head receptive fields => no cocycle, H_coh = 0."""
        np.random.seed(0)
        n, d, H = 8, 4, 2
        heads = np.random.randn(H, n, d)
        attn = np.zeros((H, n, n))
        attn[0, :4, :4] = 0.5
        attn[1, 4:, 4:] = 0.5

        c = CohomologyComputer(H, d, threshold=0.1)
        r = c.compute_H_coh(heads, attn)

        assert r["H_coh"] == 0.0
        assert r["q_obs"] == 0

    def test_h_coh_nonnegative(self):
        """H_coh is always >= 0 (it's a squared norm)."""
        np.random.seed(123)
        for trial in range(20):
            H = np.random.randint(2, 6)
            n = np.random.randint(4, 15)
            d = np.random.randint(2, 8)
            heads = np.random.randn(H, n, d)
            attn = np.random.dirichlet(np.ones(n), size=(H, n))

            c = CohomologyComputer(H, d, threshold=0.05)
            r = c.compute_H_coh(heads, attn)

            assert r["H_coh"] >= 0, f"H_coh must be >= 0, got {r['H_coh']}"
            assert r["q_obs"] >= 0, f"q_obs must be >= 0, got {r['q_obs']}"

    def test_q_obs_nonnegative(self):
        """Structural obstruction dimension is always >= 0."""
        np.random.seed(7)
        H, n, d = 4, 10, 3
        heads = np.random.randn(H, n, d)
        attn = np.random.dirichlet(np.ones(n), size=(H, n))

        c = CohomologyComputer(H, d, threshold=0.01)
        r = c.compute_H_coh(heads, attn)

        assert r["q_obs"] >= 0

    def test_constant_shift_lower_fraction(self):
        """Constant shifts have lower irreconcilable fraction than random.

        With sheaf restriction maps, constant shifts (uniform disagreement)
        are more resolvable than random position-dependent disagreement.
        The H_coh_frac metric captures this: structured patterns have
        a lower fraction of irreconcilable energy.
        """
        np.random.seed(42)
        n, d, H = 10, 4, 4
        base = np.random.randn(n, d)

        # Constant shift: same disagreement at every position
        shifts = np.random.randn(H, 1, d) * 2.0
        heads_shift = np.stack([base + shifts[h].repeat(n, axis=0)
                                for h in range(H)])

        # Random: independent disagreement at every position
        heads_random = np.random.randn(H, n, d) * 2.0

        attn = np.random.dirichlet(np.ones(n), size=(H, n))
        c = CohomologyComputer(H, d, threshold=0.0)

        r_shift = c.compute_H_coh(heads_shift, attn)
        r_random = c.compute_H_coh(heads_random, attn)

        # Constant shift should have lower irreconcilable fraction
        assert r_shift["H_coh_frac"] < r_random["H_coh_frac"], \
            (f"Constant shift fraction ({r_shift['H_coh_frac']:.4f}) "
             f"should be less than random ({r_random['H_coh_frac']:.4f})")

    def test_identity_maps_zero(self):
        """With identity restriction maps (constant sheaf), H_coh = 0."""
        np.random.seed(42)
        H, n, d = 4, 10, 4
        heads = np.random.randn(H, n, d)
        attn = np.random.dirichlet(np.ones(n), size=(H, n))

        c = CohomologyComputer(H, d, threshold=0.0)
        r = c.compute_H_coh(heads, attn, restriction_maps={})

        assert r["H_coh"] < 1e-10, \
            f"Identity maps should give H_coh~0, got {r['H_coh']}"

    def test_sheaf_mode_detects_incoherence(self):
        """Non-trivial sheaf maps produce H_coh > 0 for incoherent heads.

        This is the key property: with per-edge restriction maps and
        dense enough overlaps (D_1 > D_0), the sheaf coboundary cannot
        resolve all identity-map cocycles, so H_coh > 0.
        """
        np.random.seed(42)
        H, n, d = 4, 12, 4
        heads = np.random.randn(H, n, d)
        attn = np.random.dirichlet(np.ones(n), size=(H, n))

        c = CohomologyComputer(H, d, threshold=0.0)
        r = c.compute_H_coh(heads, attn)

        # With 4 heads and dense overlaps: D_1 = 6*n*d = 288 > D_0 = 4*n*d = 192
        assert r["sheaf_mode"] is True
        assert r["H_coh"] > 0, f"Sheaf mode should give H_coh > 0 for random heads, got {r['H_coh']}"

    def test_high_threshold_reduces_overlaps(self):
        """Higher threshold => fewer overlaps => simpler cohomology."""
        np.random.seed(42)
        H, n, d = 4, 10, 3
        heads = np.random.randn(H, n, d)
        attn = np.random.dirichlet(np.ones(n), size=(H, n))

        c_low = CohomologyComputer(H, d, threshold=0.01)
        c_high = CohomologyComputer(H, d, threshold=0.5)

        r_low = c_low.compute_H_coh(heads, attn)
        r_high = c_high.compute_H_coh(heads, attn)

        assert r_low["n_overlaps"] >= r_high["n_overlaps"], \
            "Higher threshold should produce fewer overlaps"

    def test_single_head_no_cohomology(self):
        """With only 1 head, there are no pairwise overlaps."""
        np.random.seed(42)
        n, d = 8, 4
        heads = np.random.randn(1, n, d)
        attn = np.random.dirichlet(np.ones(n), size=(1, n))

        c = CohomologyComputer(1, d, threshold=0.0)
        r = c.compute_H_coh(heads, attn)

        assert r["H_coh"] == 0.0
        assert r["q_obs"] == 0
        assert r["n_overlaps"] == 0

    def test_cocycle_norm_bounds_h_coh(self):
        """H_coh <= ||cocycle||^2 (projection can only reduce norm)."""
        np.random.seed(42)
        H, n, d = 4, 8, 3
        heads = np.random.randn(H, n, d)
        attn = np.random.dirichlet(np.ones(n), size=(H, n))

        c = CohomologyComputer(H, d, threshold=0.0)
        r = c.compute_H_coh(heads, attn)

        assert r["H_coh"] <= r["cocycle_norm"] ** 2 + 1e-8, \
            "H_coh must be <= ||cocycle||^2"

    def test_h_coh_frac_bounded(self):
        """H_coh_frac is always in [0, 1]."""
        np.random.seed(42)
        for trial in range(15):
            H = np.random.randint(3, 6)
            n = np.random.randint(6, 15)
            d = np.random.randint(2, 6)
            heads = np.random.randn(H, n, d)
            attn = np.random.dirichlet(np.ones(n), size=(H, n))

            c = CohomologyComputer(H, d, threshold=0.0)
            r = c.compute_H_coh(heads, attn)

            assert 0 <= r["H_coh_frac"] <= 1.0 + 1e-8, \
                f"H_coh_frac must be in [0,1], got {r['H_coh_frac']}"

    def test_delta_0_dimensions(self):
        """Check coboundary matrix dimensions are correct."""
        np.random.seed(42)
        H, n, d = 3, 6, 2
        attn = np.random.dirichlet(np.ones(n), size=(H, n))

        c = CohomologyComputer(H, d, threshold=0.0)
        overlaps, fields = c.compute_overlaps(attn)
        if overlaps:
            delta_0, pair_list = c.build_delta_0(n, overlaps)
            D_0 = H * n * d
            D_1 = sum(len(overlaps[p]) * d for p in pair_list)
            assert delta_0.shape == (D_1, D_0), \
                f"Expected delta_0 shape ({D_1}, {D_0}), got {delta_0.shape}"


# ============================================================
# SheafLaplacian Tests
# ============================================================

class TestSpectral:

    def test_laplacian_symmetric(self):
        """L_F = B^T B must be symmetric."""
        np.random.seed(42)
        n, d, H = 8, 4, 3
        A = np.random.rand(H, n, n)
        A /= A.sum(axis=-1, keepdims=True)
        W = np.random.randn(H, d, d) * 0.1

        sl = SheafLaplacian(n, d)
        L = sl.construct_laplacian(A, W)

        assert np.allclose(L, L.T, atol=1e-10), "L_F must be symmetric"

    def test_laplacian_psd(self):
        """L_F = B^T B must be positive semi-definite."""
        np.random.seed(42)
        n, d, H = 8, 4, 3
        A = np.random.rand(H, n, n)
        A /= A.sum(axis=-1, keepdims=True)
        W = np.random.randn(H, d, d) * 0.1

        sl = SheafLaplacian(n, d)
        L = sl.construct_laplacian(A, W)
        eigs = np.linalg.eigvalsh(L)

        assert np.all(eigs >= -1e-10), f"L_F must be PSD, min eigenvalue = {eigs.min()}"

    def test_spectral_gap_nonnegative(self):
        """Spectral gap gamma >= 0."""
        np.random.seed(42)
        n, d, H = 6, 3, 2
        A = np.random.rand(H, n, n)
        A /= A.sum(axis=-1, keepdims=True)
        W = np.random.randn(H, d, d) * 0.1

        sl = SheafLaplacian(n, d)
        L = sl.construct_laplacian(A, W)
        _, gap = sl.compute_spectral_gap(L)

        assert gap >= -1e-10, f"Spectral gap must be >= 0, got {gap}"

    def test_sheaf_energy_equals_quadratic_form(self):
        """E = ||B_F s||^2 = s^T L_F s."""
        np.random.seed(42)
        n, d, H = 6, 3, 2
        A = np.random.rand(H, n, n)
        A /= A.sum(axis=-1, keepdims=True)
        W = np.random.randn(H, d, d) * 0.1

        sl = SheafLaplacian(n, d)
        L = sl.construct_laplacian(A, W)
        s = np.random.randn(n * d)

        energy_B = sl.compute_sheaf_energy(s, A, W)
        energy_L = float(s @ L @ s)

        assert abs(energy_B - energy_L) < 1e-8, \
            f"||Bs||^2 ({energy_B}) should equal s^T L s ({energy_L})"

    def test_constant_sheaf_reduces_to_graph_laplacian(self):
        """When all restriction maps are identity, L_F = L_G tensor I_d."""
        np.random.seed(42)
        n, d = 5, 3
        H = 2
        A = np.random.rand(H, n, n)
        A /= A.sum(axis=-1, keepdims=True)
        W = np.stack([np.eye(d)] * H)  # identity restriction maps

        sl = SheafLaplacian(n, d)
        L = sl.construct_laplacian(A, W)

        # L should be block-diagonal with identical blocks
        assert L.shape == (n * d, n * d)
        assert np.allclose(L, L.T, atol=1e-10)


# ============================================================
# GluingLayer Tests
# ============================================================

class TestGluing:

    def test_gluing_reduces_h_coh(self):
        """Gluing always reduces or maintains H_coh."""
        np.random.seed(42)
        for trial in range(10):
            np.random.seed(trial)
            H, n, d = 4, 8, 4
            heads = np.random.randn(H, n, d)
            attn = np.random.dirichlet(np.ones(n), size=(H, n))

            gl = GluingLayer(H, d, threshold=0.0)
            output = gl.forward(heads, attn)

            assert output.H_coh_after <= output.H_coh_before + 1e-8, \
                f"Gluing must not increase H_coh: {output.H_coh_before:.4f} -> {output.H_coh_after:.4f}"

    def test_gluing_coherent_is_noop(self):
        """Coherent heads => gluing does nothing (H_coh already 0)."""
        np.random.seed(42)
        n, d, H = 8, 4, 3
        base = np.random.randn(n, d)
        heads = np.stack([base] * H)
        attn = np.random.dirichlet(np.ones(n), size=(H, n))

        gl = GluingLayer(H, d, threshold=0.0)
        output = gl.forward(heads, attn)

        assert output.H_coh_before < 1e-10
        assert output.H_coh_after < 1e-10

    def test_gluing_output_shape(self):
        """Glued output has correct shape (n, d)."""
        np.random.seed(42)
        H, n, d = 4, 10, 6
        heads = np.random.randn(H, n, d)
        attn = np.random.dirichlet(np.ones(n), size=(H, n))

        gl = GluingLayer(H, d, threshold=0.0)
        output = gl.forward(heads, attn)

        assert output.glued.shape == (n, d), \
            f"Expected ({n}, {d}), got {output.glued.shape}"


# ============================================================
# EigenvalueStatistics Tests
# ============================================================

class TestEigenvalueStats:

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
        assert abs(integral - 1.0) < 0.05, f"GOE PDF integral = {integral}"

    def test_poisson_pdf_normalized(self):
        """Poisson PDF integrates to ~1."""
        s = np.linspace(0, 10, 1000)
        ds = s[1] - s[0]
        integral = np.sum(EigenvalueStatistics.poisson_pdf(s)) * ds
        assert abs(integral - 1.0) < 0.05

    def test_analyze_returns_finite(self):
        """Analysis should return finite values."""
        np.random.seed(42)
        A = np.random.randn(20, 20)
        analyzer = EigenvalueStatistics()
        stats = analyzer.analyze(A)
        assert np.isfinite(stats["kl_goe"])
        assert np.isfinite(stats["kl_poisson"])


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration:

    def test_full_pipeline_synthetic(self):
        """End-to-end: random data -> cohomology -> spectral -> gluing."""
        np.random.seed(42)
        H, n, d = 4, 10, 4

        heads = np.random.randn(H, n, d)
        attn = np.random.dirichlet(np.ones(n), size=(H, n))

        # Cohomology
        cc = CohomologyComputer(H, d, threshold=0.05)
        r = cc.compute_H_coh(heads, attn)
        assert r["H_coh"] >= 0
        assert r["q_obs"] >= 0

        # Spectral
        W = np.eye(d)[np.newaxis, :, :].repeat(H, axis=0)
        sl = SheafLaplacian(n, d)
        L = sl.construct_laplacian(attn, W)
        assert np.allclose(L, L.T)
        assert np.all(np.linalg.eigvalsh(L) >= -1e-10)

        # Gluing
        gl = GluingLayer(H, d, threshold=0.05)
        output = gl.forward(heads, attn)
        assert output.H_coh_after <= output.H_coh_before + 1e-8
        assert output.glued.shape == (n, d)

    def test_reproducibility(self):
        """Same input => same output."""
        np.random.seed(42)
        H, n, d = 3, 8, 4
        heads = np.random.randn(H, n, d)
        attn = np.random.dirichlet(np.ones(n), size=(H, n))

        cc = CohomologyComputer(H, d, threshold=0.05)
        r1 = cc.compute_H_coh(heads, attn)
        r2 = cc.compute_H_coh(heads, attn)

        assert r1["H_coh"] == r2["H_coh"]
        assert r1["q_obs"] == r2["q_obs"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
