"""
Cech Cohomology computation for multi-head attention.

Implements:
- Instance hallucination score H_coh = ||c_perp||^2 (sheaf energy)
- Structural obstruction dimension q_obs = dim H^1
- Sparse coboundary matrices delta_0 and delta_1
- Non-trivial restriction maps for genuine sheaf structure

KEY INSIGHT (Remark 3.2): With identity restriction maps (constant sheaf),
every cocycle c = s_h - s_k is a coboundary (set x_h = s_h), giving H_coh = 0
always. The correct formulation uses a MIXED approach:

  cocycle:    c_{hk}(t) = s_h(t) - s_k(t)            [raw disagreement, identity maps]
  coboundary: (d0 x)_{hk,t} = rho_h x_h(t) - rho_k x_k(t)  [resolvable part, sheaf maps]
  H_coh = ||c - proj_{im d0}(c)||^2                  [irreconcilable part]

Different restriction maps rho_h, rho_k mean im(d0) doesn't contain all
identity-map cocycles, producing H_coh > 0 for disagreeing heads.

For coherent heads (s_h = s_k): c = 0, so H_coh = 0 regardless. For
incoherent heads with H >= 3 and non-trivial maps: the over-determined
system (same x_h for multiple edges) has no exact solution, so H_coh > 0.
"""
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import svds, lsqr
from typing import Optional, Dict


class CohomologyComputer:
    """Computes Cech cohomology for multi-head attention.

    The key metric H_coh measures irreconcilable inter-head disagreement.
    With non-trivial restriction maps (default), H_coh > 0 indicates that
    heads disagree in ways that cannot be resolved by per-head adjustments
    within the sheaf structure -- a signature of hallucination.

    Args:
        n_heads: number of attention heads
        d_model: dimension per head
        threshold: attention weight threshold for overlap detection
    """

    def __init__(self, n_heads: int, d_model: int, threshold: float = 0.0):
        self.H = n_heads
        self.d = d_model
        self.threshold = threshold

    def compute_overlaps(self, attention_matrices: np.ndarray) -> tuple:
        """Compute pairwise overlaps between head receptive fields.

        Token j is in head h's field if any query attends to j above threshold.
        """
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
        """Build sparse coboundary d0: C^0 -> C^1 with identity restriction maps.

        (d0 x)_{(h,k),t} = x_h(t) - x_k(t)
        """
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

    def build_delta_0_sheaf(self, n: int, overlaps: dict,
                            restriction_maps: Dict[tuple, np.ndarray]):
        """Build coboundary d0 with NON-TRIVIAL restriction maps.

        (d0 x)_{(h,k),t} = rho_{h->hk} x_h(t) - rho_{k->hk} x_k(t)

        With non-trivial rho, im(d0) is a different subspace of C^1 than
        with identity maps. The identity-map cocycle c = s_h - s_k may NOT
        lie in im(d0), producing H_coh > 0.

        Args:
            n: number of tokens
            overlaps: dict mapping (h,k) -> list of overlap token indices
            restriction_maps: dict mapping (h, (h,k)) -> d x d matrix.
                              Missing entries default to identity.

        Returns:
            delta_0: sparse (D_1, D_0) coboundary matrix
            pair_list: list of edge pairs
        """
        H, d = self.H, self.d
        pair_list = sorted(overlaps.keys())
        D_0 = H * n * d
        D_1 = sum(len(overlaps[p]) * d for p in pair_list)

        delta_0 = lil_matrix((D_1, D_0))
        row = 0
        for (h, k) in pair_list:
            idx = overlaps[(h, k)]
            rho_h = restriction_maps.get((h, (h, k)), np.eye(d))
            rho_k = restriction_maps.get((k, (h, k)), np.eye(d))

            for pos, ti in enumerate(idx):
                for di in range(d):
                    r = row + pos * d + di
                    for dj in range(d):
                        if abs(rho_h[di, dj]) > 1e-15:
                            delta_0[r, h * n * d + ti * d + dj] = rho_h[di, dj]
                        if abs(rho_k[di, dj]) > 1e-15:
                            delta_0[r, k * n * d + ti * d + dj] = -rho_k[di, dj]
            row += len(idx) * d

        return csr_matrix(delta_0), pair_list

    def build_delta_1(self, n: int, overlaps: dict, pair_list: list):
        """Build sparse coboundary d1: C^1 -> C^2 from triple overlaps."""
        d = self.d
        D_1 = sum(len(overlaps[p]) * d for p in pair_list)

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

        pair_offset = {}
        offset = 0
        for p in pair_list:
            pair_offset[p] = offset
            offset += len(overlaps[p]) * d

        row = 0
        for (h, k, l, triple_idx) in triples:
            for pos, ti in enumerate(triple_idx):
                for dd in range(d):
                    r = row + pos * d + dd

                    if (h, k) in pair_offset:
                        hk_idx = overlaps[(h, k)]
                        if ti in hk_idx:
                            local_pos = hk_idx.index(ti)
                            delta_1[r, pair_offset[(h, k)] + local_pos * d + dd] = 1.0

                    if (k, l) in pair_offset:
                        kl_idx = overlaps[(k, l)]
                        if ti in kl_idx:
                            local_pos = kl_idx.index(ti)
                            delta_1[r, pair_offset[(k, l)] + local_pos * d + dd] = 1.0

                    hl_key = (h, l) if (h, l) in pair_offset else (l, h)
                    if hl_key in pair_offset:
                        hl_idx = overlaps[hl_key]
                        if ti in hl_idx:
                            local_pos = hl_idx.index(ti)
                            sign = 1.0 if hl_key == (h, l) else -1.0
                            delta_1[r, pair_offset[hl_key] + local_pos * d + dd] = -sign

            row += len(triple_idx) * d

        return csr_matrix(delta_1), triples

    @staticmethod
    def generate_restriction_maps(n_heads: int, d_model: int,
                                  overlaps: dict,
                                  seed: Optional[int] = None,
                                  perturbation: float = 0.3) -> Dict:
        """Generate restriction maps as perturbed per-head rotations.

        Each head h gets a base rotation Q_h. For each edge (h,k), the
        restriction map is Q_h + perturbation * R_{h,edge}, re-orthogonalized.

        This preserves structural relationships: constant shifts (which are
        coboundaries in the per-head case) remain NEARLY resolvable, while
        position-dependent disagreements are not. The perturbation controls
        how much the sheaf deviates from the constant sheaf.

        Args:
            perturbation: strength of per-edge deviation from per-head base.
                0.0 = per-head maps (H_coh = 0 always).
                1.0 = fully random per-edge (loses structural info).
        """
        rng = np.random.RandomState(seed)

        # Per-head base rotations
        head_bases = {}
        for h in range(n_heads):
            Q, _ = np.linalg.qr(rng.randn(d_model, d_model))
            head_bases[h] = Q

        # Per-edge: perturbed versions of the base
        maps = {}
        for (h, k) in overlaps:
            # Head h on edge (h,k)
            base_h = head_bases[h]
            perturb_h = rng.randn(d_model, d_model) * perturbation
            M_h = base_h + perturb_h
            Q_h, _ = np.linalg.qr(M_h)
            maps[(h, (h, k))] = Q_h

            # Head k on edge (h,k)
            base_k = head_bases[k]
            perturb_k = rng.randn(d_model, d_model) * perturbation
            M_k = base_k + perturb_k
            Q_k, _ = np.linalg.qr(M_k)
            maps[(k, (h, k))] = Q_k

        return maps

    def compute_H_coh(self, head_outputs: np.ndarray,
                      attention_matrices: np.ndarray,
                      restriction_maps: Optional[Dict] = None) -> dict:
        """Compute hallucination metrics using the mixed formulation.

        The cocycle always uses identity maps (raw disagreement):
            c_{hk}(t) = s_h(t) - s_k(t)

        The coboundary d0 uses restriction maps (resolvable part):
            (d0 x)_{hk,t} = rho_h x_h(t) - rho_k x_k(t)

        H_coh = ||c - proj_{im d0}(c)||^2 = irreconcilable energy.

        Args:
            head_outputs: (H, n, d) per-head output embeddings
            attention_matrices: (H, n, n) attention weights
            restriction_maps: Optional dict of d x d restriction maps.
                None (default): generate random orthogonal maps (seed=42).
                {}: identity maps (constant sheaf, H_coh always 0).

        Returns:
            dict with q_obs, H_coh, and diagnostic info
        """
        overlaps, fields = self.compute_overlaps(attention_matrices)
        if not overlaps:
            return {'q_obs': 0, 'H_coh': 0.0, 'n_overlaps': 0,
                    'cocycle_norm': 0.0, 'sheaf_mode': False}

        H, n, d = self.H, head_outputs.shape[1], self.d

        # Restriction maps: default to random orthogonal maps
        if restriction_maps is None:
            restriction_maps = self.generate_restriction_maps(
                H, d, overlaps, seed=42)

        use_sheaf = len(restriction_maps) > 0

        # Build coboundary operator d0
        if use_sheaf:
            delta_0, pair_list = self.build_delta_0_sheaf(
                n, overlaps, restriction_maps)
        else:
            delta_0, pair_list = self.build_delta_0(n, overlaps)

        # Build d1 for q_obs (uses identity maps as approximation)
        delta_1, triples = self.build_delta_1(n, overlaps, pair_list)

        # Cocycle: ALWAYS identity maps (raw head disagreement)
        cocycle_parts = []
        for (h, k) in pair_list:
            idx = overlaps[(h, k)]
            c_hk = (head_outputs[h, idx, :] - head_outputs[k, idx, :]).flatten()
            cocycle_parts.append(c_hk)
        cocycle_vec = np.concatenate(cocycle_parts)

        if len(cocycle_vec) == 0:
            return {'q_obs': 0, 'H_coh': 0.0, 'n_overlaps': len(overlaps),
                    'cocycle_norm': 0.0, 'sheaf_mode': use_sheaf}

        # Dimensions
        D_1, D_0 = delta_0.shape
        D_2 = delta_1.shape[0]
        tol = 1e-8

        # rank(d0) via sparse SVD
        k_svd_0 = min(min(D_1, D_0) - 1, 50)
        if k_svd_0 > 0 and D_1 > 0 and D_0 > 0:
            try:
                s0 = svds(delta_0.astype(float), k=k_svd_0,
                          return_singular_vectors=False)
                r0 = int(np.sum(s0 > tol * s0.max()))
            except Exception:
                r0 = int(np.linalg.matrix_rank(delta_0.toarray(), tol=tol))
        else:
            r0 = 0

        # rank(d1) via sparse SVD
        if D_2 > 0 and D_1 > 0:
            k_svd_1 = min(min(D_2, D_1) - 1, 50)
            if k_svd_1 > 0:
                try:
                    s1 = svds(delta_1.astype(float), k=k_svd_1,
                              return_singular_vectors=False)
                    r1 = int(np.sum(s1 > tol * s1.max()))
                except Exception:
                    r1 = int(np.linalg.matrix_rank(delta_1.toarray(), tol=tol))
            else:
                r1 = 0
        else:
            r1 = 0

        # q_obs = dim(ker d1) - rank(d0)
        ker_delta1_dim = D_1 - r1
        q_obs = max(0, ker_delta1_dim - r0)

        # H_coh = ||c - proj_{im d0}(c)||^2 via least-squares
        if r0 > 0:
            result = lsqr(delta_0, cocycle_vec, atol=1e-10, btol=1e-10)
            x_ls = result[0]
            c_parallel = delta_0 @ x_ls
            c_perp = cocycle_vec - c_parallel
        else:
            c_perp = cocycle_vec

        H_coh = float(np.dot(c_perp, c_perp))
        cocycle_norm = float(np.linalg.norm(cocycle_vec))
        cocycle_energy = cocycle_norm ** 2

        return {
            'q_obs': q_obs,
            'H_coh': H_coh,
            'H_coh_frac': H_coh / max(cocycle_energy, 1e-15),
            'rank_delta_0': r0,
            'rank_delta_1': r1,
            'dim_C0': D_0,
            'dim_C1': D_1,
            'dim_C2': D_2,
            'n_overlaps': len(overlaps),
            'n_triples': len(triples),
            'cocycle_norm': cocycle_norm,
            'sheaf_mode': use_sheaf,
        }


if __name__ == "__main__":
    np.random.seed(42)
    H, n, d = 4, 10, 3
    A = np.random.rand(H, n, n)
    A /= A.sum(axis=-1, keepdims=True)
    heads = np.random.randn(H, n, d)

    # With default sheaf restriction maps
    cc = CohomologyComputer(H, d, threshold=0.05)
    result = cc.compute_H_coh(heads, A)
    print(f"q_obs (structural): {result['q_obs']}")
    print(f"H_coh (energy):     {result['H_coh']:.4f}")
    print(f"sheaf_mode:         {result['sheaf_mode']}")
    print(f"rank(d0): {result['rank_delta_0']}, rank(d1): {result['rank_delta_1']}")
    print(f"dim C^0: {result['dim_C0']}, dim C^1: {result['dim_C1']}")
    print(f"Overlaps: {result['n_overlaps']}, Triples: {result['n_triples']}")

    # With identity maps (constant sheaf => H_coh always 0)
    result_id = cc.compute_H_coh(heads, A, restriction_maps={})
    print(f"\nIdentity maps: H_coh = {result_id['H_coh']:.6f} (should be ~0)")

    # Coherent heads => H_coh = 0 regardless
    base = np.random.randn(n, d)
    heads_same = np.stack([base] * H)
    result_coh = cc.compute_H_coh(heads_same, A)
    print(f"Coherent heads: H_coh = {result_coh['H_coh']:.6f} (should be ~0)")
