"""Experiment 2: Eigenvalue statistics vs Random Matrix Theory (toy).

Usage:
  python experiments/exp2_rmt_statistics.py --model gpt2
"""

import argparse
import numpy as np

from sheaf_transformers.extract.hf import extract_attention_from_gpt2
from sheaf_transformers.metrics.eigenvalue_stats import EigenvalueStatistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--text", default=("The quick brown fox jumps over the lazy dog. " * 5))
    args = parser.parse_args()

    data = extract_attention_from_gpt2(args.model, args.text)
    analyzer = EigenvalueStatistics()

    for ell in range(data["n_layers"]):
        goe_scores, poi_scores = [], []
        for h in range(data["n_heads"]):
            stats = analyzer.analyze(data["attention"][ell][h])
            goe_scores.append(stats["kl_goe"])
            poi_scores.append(stats["kl_poisson"])
        pref = "GOE" if float(np.mean(goe_scores)) < float(np.mean(poi_scores)) else "Poisson"
        print(
            f"Layer {ell:2d}: KL(GOE)={np.mean(goe_scores):.4f} KL(Poi)={np.mean(poi_scores):.4f} -> {pref}"
        )


if __name__ == "__main__":
    main()
