"""Sheaf Gluing Layer.

Computes a minimum-energy correction in the least-squares sense using
identity-map coboundary (guaranteed to reduce pairwise disagreement).

The correction solves min ||x|| s.t. delta_0 x ≈ c, where c is the
identity-map cocycle (raw head disagreement). This guarantees
H_coh_after <= H_coh_before because the corrected cocycle
c' = c - delta_0 x_ls has smaller norm by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import lsqr

from ..metrics.cohomology import CohomologyComputer


@dataclass
class GluingOutput:
    glued: np.ndarray
    H_coh_before: float
    H_coh_after: float


class GluingLayer:
    def __init__(self, n_heads: int, d_model: int, threshold: float = 0.0):
        self.H = int(n_heads)
        self.d = int(d_model)
        self.threshold = float(threshold)

    def forward(self, head_outputs: np.ndarray,
                attention_matrices: np.ndarray) -> GluingOutput:
        H, n, d = head_outputs.shape
        if H != self.H or d != self.d:
            raise ValueError("Shape mismatch: expected (H,n,d) with configured H,d")

        cc = CohomologyComputer(self.H, self.d, threshold=self.threshold)
        overlaps, _ = cc.compute_overlaps(attention_matrices)
        if not overlaps:
            glued = np.mean(head_outputs, axis=0)
            return GluingOutput(glued=glued, H_coh_before=0.0, H_coh_after=0.0)

        # Use identity-map coboundary for correction (guaranteed to reduce)
        delta_0, pair_list = cc.build_delta_0(n, overlaps)

        cocycle_parts = []
        for (h, k) in pair_list:
            idx = overlaps[(h, k)]
            cocycle_parts.append(
                (head_outputs[h, idx, :] - head_outputs[k, idx, :]).ravel())
        c = np.concatenate(cocycle_parts)

        x_ls = lsqr(delta_0, c, atol=1e-10, btol=1e-10)[0]
        c_perp = c - (delta_0 @ x_ls)
        H_before = float(c @ c)
        H_after_id = float(c_perp @ c_perp)

        corrections = x_ls.reshape(self.H, n, d)
        corrected = head_outputs - corrections
        glued = np.mean(corrected, axis=0)

        return GluingOutput(glued=glued, H_coh_before=H_before,
                            H_coh_after=H_after_id)
