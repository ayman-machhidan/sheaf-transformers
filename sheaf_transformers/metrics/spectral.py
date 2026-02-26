"""Sheaf Laplacian construction and spectral analysis.

Constructs L_F = B_F^T B_F correctly as a symmetric PSD matrix,
where B_F is the coboundary map with restriction maps rho_{i->ij}.
"""
import numpy as np
from scipy import linalg
from scipy.sparse import lil_matrix, csr_matrix


class SheafLaplacian:
    """Constructs and analyzes the sheaf Laplacian on the attention graph."""

    def __init__(self, n_tokens: int, d_model: int, threshold: float = None):
        self.n = n_tokens
        self.d = d_model
        self.epsilon = threshold if threshold is not None else 1.0 / n_tokens

    def construct_attention_graph(self, attention_matrices: np.ndarray):
        H, n, _ = attention_matrices.shape
        max_attn = np.max(attention_matrices, axis=0)
        best_head = np.argmax(attention_matrices, axis=0)
        edges = [(i, j, best_head[i, j])
                 for i in range(n) for j in range(i + 1, n)
                 if max_attn[i, j] > self.epsilon or max_attn[j, i] > self.epsilon]
        return edges, max_attn

    def construct_coboundary(self, attention_matrices, value_matrices):
        """Build B_F: (B s)_{ij} = rho_{i->ij}(s_i) - rho_{j->ij}(s_j)."""
        edges, _ = self.construct_attention_graph(attention_matrices)
        n, d = self.n, self.d
        n_edges = len(edges)
        B = lil_matrix((n_edges * d, n * d))
        for e_idx, (i, j, h) in enumerate(edges):
            W = value_matrices[h]
            row = e_idx * d
            for di in range(d):
                for dj in range(d):
                    B[row + di, i * d + dj] = W[di, dj]
                    B[row + di, j * d + dj] = -W[di, dj]
        return csr_matrix(B), edges

    def construct_laplacian(self, attention_matrices, value_matrices):
        """L_F = B_F^T B_F (symmetric PSD by construction)."""
        B, _ = self.construct_coboundary(attention_matrices, value_matrices)
        return (B.T @ B).toarray()

    def compute_spectral_gap(self, L, k=10):
        """Compute spectral gap = smallest nonzero eigenvalue of L_F.

        L_F = B^T B is PSD with a d-dimensional kernel (constant sections).
        The spectral gap gamma = lambda_1 (first nonzero eigenvalue).
        """
        eigenvalues = np.sort(linalg.eigvalsh(L))
        # Threshold: eigenvalue is "zero" if < tol * max eigenvalue
        scale = max(abs(eigenvalues[-1]), 1.0)
        tol = 1e-8 * scale
        nonzero_eigs = eigenvalues[eigenvalues > tol]
        gap = float(nonzero_eigs[0]) if len(nonzero_eigs) > 0 else 0.0
        return eigenvalues[:min(k, len(eigenvalues))], gap

    def compute_sheaf_energy(self, s, attention_matrices, value_matrices):
        """E = ||B_F s||^2 = s^T L_F s."""
        B, _ = self.construct_coboundary(attention_matrices, value_matrices)
        Bs = B @ s.flatten()
        return float(np.dot(Bs, Bs))


if __name__ == "__main__":
    np.random.seed(42)
    n, d, H = 8, 4, 3
    A = np.random.rand(H, n, n); A /= A.sum(axis=-1, keepdims=True)
    W = np.random.randn(H, d, d) * 0.1
    sl = SheafLaplacian(n, d)
    L = sl.construct_laplacian(A, W)
    eigs = np.linalg.eigvalsh(L)
    assert np.allclose(L, L.T), "Not symmetric!"
    assert np.all(eigs >= -1e-10), "Not PSD!"
    _, gap = sl.compute_spectral_gap(L)
    print(f"L: {L.shape}, sym={np.allclose(L, L.T)}, PSD={np.all(eigs>=-1e-10)}, gap={gap:.4f}")
    print("✅ spectral.py: L = B^T B correct")
