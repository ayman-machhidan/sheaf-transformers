"""Tests for sheaf layers: GluingLayer and holomorphic attention."""

import numpy as np
import pytest

from sheaf_transformers.layers.gluing import GluingLayer, GluingOutput
from sheaf_transformers.layers.holomorphic_attention import (
    barycentric_weights,
    holomorphic_attention,
    meromorphic_kernel,
)


class TestGluingLayer:

    def test_gluing_reduces_h_coh(self, rng):
        """Gluing always reduces or maintains H_coh."""
        for trial in range(10):
            rng2 = np.random.RandomState(trial)
            H, n, d = 4, 8, 4
            heads = rng2.randn(H, n, d)
            attn = rng2.dirichlet(np.ones(n), size=(H, n))
            gl = GluingLayer(H, d, threshold=0.0)
            output = gl.forward(heads, attn)
            assert output.H_coh_after <= output.H_coh_before + 1e-8

    def test_gluing_coherent_is_noop(self, rng):
        """Coherent heads => gluing does nothing."""
        n, d, H = 8, 4, 3
        base = rng.randn(n, d)
        heads = np.stack([base] * H)
        attn = rng.dirichlet(np.ones(n), size=(H, n))
        gl = GluingLayer(H, d, threshold=0.0)
        output = gl.forward(heads, attn)
        assert output.H_coh_before < 1e-10
        assert output.H_coh_after < 1e-10

    def test_gluing_output_shape(self, rng):
        """Glued output has correct shape (n, d)."""
        H, n, d = 4, 10, 6
        heads = rng.randn(H, n, d)
        attn = rng.dirichlet(np.ones(n), size=(H, n))
        gl = GluingLayer(H, d, threshold=0.0)
        output = gl.forward(heads, attn)
        assert output.glued.shape == (n, d)

    def test_gluing_output_type(self, rng):
        """forward() returns a GluingOutput dataclass."""
        H, n, d = 3, 6, 3
        heads = rng.randn(H, n, d)
        attn = rng.dirichlet(np.ones(n), size=(H, n))
        gl = GluingLayer(H, d, threshold=0.0)
        output = gl.forward(heads, attn)
        assert isinstance(output, GluingOutput)
        assert isinstance(output.H_coh_before, float)
        assert isinstance(output.H_coh_after, float)

    def test_gluing_no_overlaps(self, rng):
        """Disjoint heads => gluing returns mean, H_coh=0."""
        H, n, d = 2, 8, 4
        heads = rng.randn(H, n, d)
        attn = np.zeros((H, n, n))
        attn[0, :4, :4] = 0.5
        attn[1, 4:, 4:] = 0.5
        gl = GluingLayer(H, d, threshold=0.1)
        output = gl.forward(heads, attn)
        assert output.H_coh_before == 0.0
        assert output.H_coh_after == 0.0

    def test_gluing_shape_mismatch_raises(self, rng):
        """Shape mismatch between config and input raises ValueError."""
        gl = GluingLayer(n_heads=4, d_model=8, threshold=0.0)
        heads = rng.randn(3, 6, 8)  # Wrong n_heads
        attn = rng.dirichlet(np.ones(6), size=(3, 6))
        with pytest.raises(ValueError, match="Shape mismatch"):
            gl.forward(heads, attn)

    def test_gluing_single_head(self, rng):
        """Single head => no overlaps => returns mean (which is the head)."""
        n, d = 8, 4
        heads = rng.randn(1, n, d)
        attn = rng.dirichlet(np.ones(n), size=(1, n))
        gl = GluingLayer(1, d, threshold=0.0)
        output = gl.forward(heads, attn)
        assert np.allclose(output.glued, heads[0])


class TestHolomorphicAttention:

    def test_output_shape(self, rng):
        """holomorphic_attention returns (n, d_k) array."""
        n, d, dk = 6, 8, 4
        z = np.exp(2j * np.pi * np.arange(n) / n)
        h = rng.randn(n, d)
        WQ = rng.randn(d, dk) * 0.1
        WK = rng.randn(d, dk) * 0.1
        WV = rng.randn(d, dk) * 0.1
        out = holomorphic_attention(z, h, WQ, WK, WV, lam=0.1)
        assert out.shape == (n, dk)

    def test_no_softmax_normalization(self, rng):
        """Kernel weights are NOT row-stochastic (no softmax)."""
        n = 6
        z = np.exp(2j * np.pi * np.arange(n) / n)
        kernel = np.array([
            [np.real(meromorphic_kernel(z[i], z[j], 0.1))
             for j in range(n)] for i in range(n)
        ])
        row_sums = kernel.sum(axis=1)
        assert not np.allclose(row_sums, 1.0), "Should NOT be row-stochastic"

    def test_meromorphic_kernel_symmetric(self):
        """K_lambda(z_i, z_j) = K_lambda(z_j, z_i) for real parts."""
        z1, z2 = 1 + 0.5j, 0.3 + 0.8j
        k12 = meromorphic_kernel(z1, z2, 0.1)
        k21 = meromorphic_kernel(z2, z1, 0.1)
        assert abs(k12 - k21) < 1e-10

    def test_meromorphic_kernel_positive_for_real(self):
        """K_lambda > 0 for real-valued inputs and lambda > 0."""
        val = meromorphic_kernel(1.0, 2.0, 0.1)
        assert np.real(val) > 0

    def test_barycentric_weights_shape(self):
        """Barycentric weights have the same length as nodes."""
        n = 5
        z = np.exp(2j * np.pi * np.arange(n) / n)
        w = barycentric_weights(z)
        assert w.shape == (n,)

    def test_deterministic_output(self, rng):
        """Same input => same output."""
        n, d, dk = 6, 8, 4
        z = np.exp(2j * np.pi * np.arange(n) / n)
        h = rng.randn(n, d)
        WQ = rng.randn(d, dk) * 0.1
        WK = rng.randn(d, dk) * 0.1
        WV = rng.randn(d, dk) * 0.1
        out1 = holomorphic_attention(z, h, WQ, WK, WV, lam=0.1)
        out2 = holomorphic_attention(z, h, WQ, WK, WV, lam=0.1)
        assert np.allclose(out1, out2)

    def test_lambda_controls_kernel_width(self):
        """Larger lambda => smoother kernel (more uniform across distances).

        A wider lambda reduces the ratio K(near)/K(far), making the
        kernel less peaked and more uniform.
        """
        z_self, z_near, z_far = 0.0 + 0j, 0.1 + 0j, 1.0 + 0j
        # Narrow kernel: big difference between near and far
        k_near_narrow = np.real(meromorphic_kernel(z_self, z_near, 0.01))
        k_far_narrow = np.real(meromorphic_kernel(z_self, z_far, 0.01))
        ratio_narrow = k_near_narrow / k_far_narrow

        # Wide kernel: smaller difference between near and far
        k_near_wide = np.real(meromorphic_kernel(z_self, z_near, 1.0))
        k_far_wide = np.real(meromorphic_kernel(z_self, z_far, 1.0))
        ratio_wide = k_near_wide / k_far_wide

        # Wider lambda => flatter kernel => lower near/far ratio
        assert ratio_wide < ratio_narrow
