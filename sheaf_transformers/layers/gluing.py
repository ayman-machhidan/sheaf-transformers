"""Sheaf Gluing Layer.

Computes a minimum-energy correction in the least-squares sense.
It does NOT guarantee zero residual unless the cocycle lies in im(delta_0).
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

    def forward(self, head_outputs: np.ndarray, attention_matrices: np.ndarray) -> GluingOutput:
        H, n, d = head_outputs.shape
        if H != self.H or d != self.d:
            raise ValueError("Shape mismatch: expected (H,n,d) with configured H,d")

        cc = CohomologyComputer(self.H, self.d, threshold=self.threshold)
        overlaps, _ = cc.compute_overlaps(attention_matrices)
        if not overlaps:
            glued = np.mean(head_outputs, axis=0)
            return GluingOutput(glued=glued, H_coh_before=0.0, H_coh_after=0.0)

        delta_0, pair_list = cc.build_delta_0(n, overlaps)
        cocycle_parts = []
        for (h, k) in pair_list:
            idx = overlaps[(h, k)]
            cocycle_parts.append((head_outputs[h, idx, :] - head_outputs[k, idx, :]).ravel())
        c = np.concatenate(cocycle_parts)

        x_ls = lsqr(delta_0, c, atol=1e-10, btol=1e-10)[0]
        c_perp = c - (delta_0 @ x_ls)
        H_before = float(c_perp @ c_perp)

        corrections = x_ls.reshape(self.H, n, d)
        corrected = head_outputs - corrections
        glued = np.mean(corrected, axis=0)

        # after
        cocycle_parts2 = []
        for (h, k) in pair_list:
            idx = overlaps[(h, k)]
            cocycle_parts2.append((corrected[h, idx, :] - corrected[k, idx, :]).ravel())
        c2 = np.concatenate(cocycle_parts2)
        x_ls2 = lsqr(delta_0, c2, atol=1e-10, btol=1e-10)[0]
        c2_perp = c2 - (delta_0 @ x_ls2)
        H_after = float(c2_perp @ c2_perp)

        return GluingOutput(glued=glued, H_coh_before=H_before, H_coh_after=H_after)
