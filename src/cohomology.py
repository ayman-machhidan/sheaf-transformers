"""
Cech Cohomology computation for multi-head attention.

Implements:
- Structural obstruction dimension q_obs = dim H^1
- Instance hallucination score H_coh = ||c_perp||^2 (sheaf energy)
- Sparse coboundary matrices delta_0 and delta_1
"""
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import svds, lsqr


class CohomologyComputer:
    """Computes Cech cohomology for multi-head attention."""

    def __init__(self, n_heads: int, d_model: int, threshold: float = 0.0):
        self.H = n_heads
        self.d = d_model
        self.threshold = threshold

    def compute_overlaps(self, attention_matrices: np.ndarray) -> dict:
        """Compute pairwise overlaps between head receptive fields."""
        H, n, _ = attention_matrices.shape
        fields = []
        for h in range(H):
            field = set()
            for j in range(n):
                if np.any(attention_matrices[h, :, j] > self.threshold):
                    field.add(j)
            fields.append(field)
        overlaps = {}
        for h in range(H):
            for k in range(h + 1, H):
                o = sorted(fields[h] & fields[k])
                if o:
                    overlaps[(h, k)] = o
        return overlaps, fields

    def build_delta_0(self, n: int, overlaps: dict):
        """Build sparse coboundary delta_0: C^0 -> C^1."""
        H, d = self.H, self.d
        D_0 = H * n * d
        pair_list = sorted(overlaps.keys())
        D_1 = sum(len(overlaps[p]) * d for p in pair_list)

        delta_0 = lil_matrix((D_1, D_0))
        row = 0
        for (h, k) in pair_list:
            idx = overlaps[(h, k)]
            for pos, ti in enumerate(idx):
                for dd in range(d):
                    r = row + pos * d + dd
                    delta_0[r, h * n * d + ti * d + dd] = 1.0
                    delta_0[r, k * n * d + ti * d + dd] = -1.0
            row += len(idx) * d

        return csr_matrix(delta_0), pair_list

    def build_delta_1(self, n: int, overlaps: dict, pair_list: list):
        """Build sparse coboundary delta_1: C^1 -> C^2 from triple overlaps."""
        d = self.d
        D_1 = sum(len(overlaps[p]) * d for p in pair_list)

        # Find triples (h, k, l) where all three pairwise overlaps exist
        pair_set = set(pair_list)
        triples = []
        for i, (h, k) in enumerate(pair_list):
            for j, (k2, l) in enumerate(pair_list):
                if k == k2 and (h, l) in pair_set:
                    triple_overlap = sorted(
                        set(overlaps[(h, k)]) &
                        set(overlaps[(k, l)]) &
                        set(overlaps.get((h, l), overlaps.get((l, h), [])))
                    )
                    if triple_overlap:
                        triples.append((h, k, l, triple_overlap))

        D_2 = sum(len(t[3]) * d for t in triples) if triples else 0
        if D_2 == 0:
            return csr_matrix((D_2, D_1)), triples

        delta_1 = lil_matrix((D_2, D_1))

        # Map from pair -> column offset in C^1
        pair_offset = {}
        offset = 0
        for p in pair_list:
            pair_offset[p] = offset
            offset += len(overlaps[p]) * d

        row = 0
        for (h, k, l, triple_idx) in triples:
            # delta_1 maps (c_{hk}, c_{kl}, c_{hl}) -> c_{hk} + c_{kl} - c_{hl}
            for pos, ti in enumerate(triple_idx):
                for dd in range(d):
                    r = row + pos * d + dd

                    # +c_{hk} contribution
                    if (h, k) in pair_offset:
                        hk_idx = overlaps[(h, k)]
                        if ti in hk_idx:
                            local_pos = hk_idx.index(ti)
                            delta_1[r, pair_offset[(h, k)] + local_pos * d + dd] = 1.0

                    # +c_{kl} contribution
                    if (k, l) in pair_offset:
                        kl_idx = overlaps[(k, l)]
                        if ti in kl_idx:
                            local_pos = kl_idx.index(ti)
                            delta_1[r, pair_offset[(k, l)] + local_pos * d + dd] = 1.0

                    # -c_{hl} contribution
                    hl_key = (h, l) if (h, l) in pair_offset else (l, h)
                    if hl_key in pair_offset:
                        hl_idx = overlaps[hl_key]
                        if ti in hl_idx:
                            local_pos = hl_idx.index(ti)
                            sign = 1.0 if hl_key == (h, l) else -1.0
                            delta_1[r, pair_offset[hl_key] + local_pos * d + dd] = -sign

            row += len(triple_idx) * d

        return csr_matrix(delta_1), triples

    def compute_H_coh(self, head_outputs: np.ndarray,
                      attention_matrices: np.ndarray) -> dict:
        """Compute hallucination metrics.

        Returns:
            q_obs: structural obstruction dimension = dim(ker delta_1) - rank(delta_0)
            H_coh: instance hallucination score = ||c_perp||^2 (sheaf energy)
        """
        overlaps, fields = self.compute_overlaps(attention_matrices)
        if not overlaps:
            return {'q_obs': 0, 'H_coh': 0.0, 'n_overlaps': 0}

        H, n, d = self.H, head_outputs.shape[1], self.d

        # Build sparse coboundary matrices
        delta_0, pair_list = self.build_delta_0(n, overlaps)
        delta_1, triples = self.build_delta_1(n, overlaps, pair_list)

        # Cocycle vector
        cocycle_parts = []
        for (h, k) in pair_list:
            idx = overlaps[(h, k)]
            c_hk = (head_outputs[h, idx, :] - head_outputs[k, idx, :]).flatten()
            cocycle_parts.append(c_hk)
        cocycle_vec = np.concatenate(cocycle_parts)

        # Rank computations via sparse SVD
        D_1, D_0 = delta_0.shape
        D_2 = delta_1.shape[0]
        tol = 1e-8

        # rank(delta_0) via sparse SVD
        k_svd_0 = min(min(D_1, D_0) - 1, 50)
        if k_svd_0 > 0 and D_1 > 0 and D_0 > 0:
            try:
                s0 = svds(delta_0.astype(float), k=k_svd_0, return_singular_vectors=False)
                r0 = int(np.sum(s0 > tol * s0.max()))
            except Exception:
                r0 = int(np.linalg.matrix_rank(delta_0.toarray(), tol=tol))
        else:
            r0 = 0

        # rank(delta_1) via sparse SVD
        if D_2 > 0 and D_1 > 0:
            k_svd_1 = min(min(D_2, D_1) - 1, 50)
            if k_svd_1 > 0:
                try:
                    s1 = svds(delta_1.astype(float), k=k_svd_1, return_singular_vectors=False)
                    r1 = int(np.sum(s1 > tol * s1.max()))
                except Exception:
                    r1 = int(np.linalg.matrix_rank(delta_1.toarray(), tol=tol))
            else:
                r1 = 0
        else:
            r1 = 0

        # Structural obstruction dimension
        ker_delta1_dim = D_1 - r1
        q_obs = max(0, ker_delta1_dim - r0)

        # Instance hallucination score via sparse least-squares
        # Project c onto im(delta_0) using LSQR (iterative, sparse)
        if r0 > 0:
            result = lsqr(delta_0, cocycle_vec, atol=1e-10, btol=1e-10)
            x_ls = result[0]
            c_parallel = delta_0 @ x_ls
            c_perp = cocycle_vec - c_parallel
        else:
            c_perp = cocycle_vec

        H_coh = float(np.dot(c_perp, c_perp))

        return {
            'q_obs': q_obs,
            'H_coh': H_coh,
            'rank_delta_0': r0,
            'rank_delta_1': r1,
            'dim_C1': D_1,
            'dim_C2': D_2,
            'n_overlaps': len(overlaps),
            'n_triples': len(triples),
            'cocycle_norm': float(np.linalg.norm(cocycle_vec)),
        }


if __name__ == "__main__":
    np.random.seed(42)
    H, n, d = 4, 10, 3
    A = np.random.rand(H, n, n)
    A /= A.sum(axis=-1, keepdims=True)
    heads = np.random.randn(H, n, d)

    cc = CohomologyComputer(H, d, threshold=0.05)
    result = cc.compute_H_coh(heads, A)
    print(f"q_obs (structural): {result['q_obs']}")
    print(f"H_coh (energy):     {result['H_coh']:.4f}")
    print(f"rank(delta_0): {result['rank_delta_0']}, rank(delta_1): {result['rank_delta_1']}")
    print(f"dim C^1: {result['dim_C1']}, dim C^2: {result['dim_C2']}")
    print(f"Overlaps: {result['n_overlaps']}, Triples: {result['n_triples']}")
    print("✅ cohomology.py: sparse, delta_1 filled, energy-based H_coh")
