"""High-level analysis pipeline."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .metrics.cohomology import CohomologyComputer


class SheafExperiment:
    """Run sheaf-theoretic diagnostics over multiple layers."""

    def __init__(self, model_data: Dict[str, Any], threshold: Optional[float] = None):
        self.data = model_data
        self.n_layers = int(model_data["n_layers"])
        self.n_heads = int(model_data["n_heads"])
        self.d_head = int(model_data["d_head"])
        self.tokens = model_data.get("tokens", [])
        self.n_tokens = int(model_data["attention"][0].shape[1])
        self.threshold = float(threshold) if threshold is not None else 1.0 / max(self.n_tokens, 1)

    def run_cohomology(self, layers: Optional[Iterable[int]] = None) -> List[Dict[str, Any]]:
        layers = list(layers) if layers is not None else list(range(self.n_layers))
        results: List[Dict[str, Any]] = []
        for ell in layers:
            attn = np.asarray(self.data["attention"][ell])
            hidden = np.asarray(self.data["hidden_states"][ell + 1])
            ho = hidden.reshape(self.n_tokens, self.n_heads, self.d_head).transpose(1, 0, 2)
            comp = CohomologyComputer(self.n_heads, self.d_head, threshold=self.threshold)
            r = comp.compute_H_coh(ho, attn)
            r["layer"] = int(ell)
            results.append(r)
        return results

    def summary(self, layers: Optional[Iterable[int]] = None) -> None:
        res = self.run_cohomology(layers)
        print("=" * 72)
        print("SHEAF COHERENCE SUMMARY")
        print("=" * 72)
        for r in res:
            print(f"Layer {r['layer']:>2}: H_coh={r['H_coh']:.6g} | q_obs={r['q_obs']} | overlaps={r.get('n_overlaps', 0)}")
