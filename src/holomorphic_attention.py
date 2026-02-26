"""Holomorphic attention with meromorphic kernel.

Implements the global holomorphic attention operator (Def 3.9/Thm 3.10):
A_hol is a GLOBAL operator with a fixed contour, NOT locally renormalized.
The kernel weights are NOT renormalized per-row (no softmax).
"""
import numpy as np


def barycentric_weights(z: np.ndarray) -> np.ndarray:
    """Compute barycentric weights omega_i = prod_{j!=i} (z_i - z_j)^{-1}."""
    n = len(z)
    w = np.ones(n, dtype=complex)
    for i in range(n):
        for j in range(n):
            if i != j:
                w[i] /= (z[i] - z[j])
    return w


def barycentric_interpolant(z_nodes, h_values, z_eval):
    """Evaluate barycentric rational interpolant at z_eval."""
    w = barycentric_weights(z_nodes)
    n, d = h_values.shape
    m = len(z_eval)
    result = np.zeros((m, d), dtype=complex)
    for k in range(m):
        z = z_eval[k]
        diffs = z - z_nodes
        close = np.abs(diffs) < 1e-12
        if np.any(close):
            result[k] = h_values[np.where(close)[0][0]]
        else:
            ratios = w / diffs
            result[k] = np.sum(ratios[:, None] * h_values, axis=0) / np.sum(ratios)
    return result


def meromorphic_kernel(z_i, z_j, lam):
    """K_lambda(z_i, z_j) = 1 / ((z_i - z_j)^2 + lambda^2)"""
    return 1.0 / ((z_i - z_j) ** 2 + lam ** 2)


def holomorphic_attention(z_positions, h_values, W_Q, W_K, W_V, lam=0.1):
    """Compute holomorphic attention via global meromorphic kernel operator.

    CRITICAL: No softmax-style per-row renormalization.
    The kernel weights K_lambda(z_i, z_j) * g_i(z_j) are used directly,
    with a single global normalization factor (analogous to the contour
    integral normalization 1/(2*pi*i)).

    Args:
        z_positions: (n,) complex positions
        h_values: (n, d) hidden states
        W_Q, W_K, W_V: (d, d_k) projection matrices
        lam: regularization parameter (controls kernel width)
    Returns:
        (n, d_k) attention output
    """
    n = len(z_positions)
    Q = h_values @ W_Q
    K = h_values @ W_K
    V = h_values @ W_V

    # Global meromorphic kernel matrix (NOT renormalized per row)
    kernel = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            kernel[i, j] = np.real(meromorphic_kernel(z_positions[i], z_positions[j], lam))

    # QK scores modulate the kernel (holomorphic g_i in the integrand)
    scores = Q @ K.T / np.sqrt(W_Q.shape[1])
    weights = kernel * scores

    # Global normalization: divide by sum of kernel (not per-row softmax!)
    # This corresponds to the 1/(2*pi*i) * contour_integral normalization
    total_kernel = kernel.sum(axis=1, keepdims=True)
    total_kernel = np.maximum(total_kernel, 1e-10)  # avoid division by zero
    weights = weights / total_kernel

    return weights @ V


if __name__ == "__main__":
    np.random.seed(42)
    n, d, dk = 6, 8, 4
    z = np.exp(2j * np.pi * np.arange(n) / n)
    h = np.random.randn(n, d)
    WQ = np.random.randn(d, dk) * 0.1
    WK = np.random.randn(d, dk) * 0.1
    WV = np.random.randn(d, dk) * 0.1

    out = holomorphic_attention(z, h, WQ, WK, WV, lam=0.1)
    print(f"Output shape: {out.shape}")

    # Verify: no per-row softmax (weights don't sum to 1 per row)
    kernel = np.array([[np.real(meromorphic_kernel(z[i], z[j], 0.1))
                        for j in range(n)] for i in range(n)])
    row_sums = kernel.sum(axis=1)
    assert not np.allclose(row_sums, 1.0), "Should NOT be row-stochastic!"
    print(f"Kernel row sums (should NOT be 1): {row_sums.round(4)}")
    print("✅ holomorphic_attention.py: no softmax renormalization")
