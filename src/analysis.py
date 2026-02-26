"""Legacy analysis pipeline (kept for compatibility).

Prefer `sheaf_transformers.pipeline.SheafExperiment`.
"""

from __future__ import annotations

import numpy as np

from .cohomology import CohomologyComputer


class SheafExperiment:
    def __init__(self, model_data: dict, threshold: float | None = None):
        self.data = model_data
        self.n_layers = int(model_data["n_layers"])
        self.n_heads = int(model_data["n_heads"])
        self.d_head = int(model_data["d_head"])
        self.n_tokens = int(model_data["attention"][0].shape[1])
        self.threshold = float(threshold) if threshold is not None else 1.0 / max(self.n_tokens, 1)

    def run_cohomology(self, layers=None):
        layers = layers or range(self.n_layers)
        results = []
        for ell in layers:
            attn = np.asarray(self.data["attention"][ell])
            hidden = np.asarray(self.data["hidden_states"][ell + 1])
            ho = hidden.reshape(self.n_tokens, self.n_heads, self.d_head).transpose(1, 0, 2)
            c = CohomologyComputer(self.n_heads, self.d_head, self.threshold)
            r = c.compute_H_coh(ho, attn)
            r["layer"] = int(ell)
            results.append(r)
        return results

    def full_analysis(self):
        cohom = self.run_cohomology()
        for r in cohom:
            print(f"Layer {r[layer]:2d}: H_coh={r[H_coh]:.6g} | q_obs={r[q_obs]}")
        return {"cohomology": cohom}
