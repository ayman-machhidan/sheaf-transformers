"""Experiment 1: compute H_coh on TruthfulQA questions (prompt-only).

This is a lightweight smoke-test experiment for the metric pipeline.
TruthfulQA does not ship binary hallucination labels by default; for an AUROC-style
evaluation you must define a scoring protocol (e.g., generate answers and label factuality).

Usage:
  python experiments/exp1_truthfulqa.py --model gpt2 --threshold 0.01
"""

import argparse
import numpy as np

from sheaf_transformers.extract.hf import extract_attention_from_gpt2
from sheaf_transformers.metrics.cohomology import CohomologyComputer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--max_samples", type=int, default=50)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
        ds = load_dataset("truthful_qa", "generation", split="validation")
    except Exception as e:
        print(f"Could not load TruthfulQA: {e}")
        print("Install: pip install datasets")
        return

    scores = []

    for i, sample in enumerate(ds):
        if i >= args.max_samples:
            break
        q = sample["question"]

        try:
            data = extract_attention_from_gpt2(args.model, q)
        except Exception:
            continue

        total = 0.0
        for ell in range(data["n_layers"]):
            attn = data["attention"][ell]
            n_tok = attn.shape[1]
            hidden = data["hidden_states"][ell + 1]
            ho = hidden.reshape(n_tok, data["n_heads"], data["d_head"]).transpose(1, 0, 2)

            c = CohomologyComputer(data["n_heads"], data["d_head"], args.threshold)
            r = c.compute_H_coh(ho, attn)
            total += r["H_coh"]

        scores.append(total)
        print(f"[{i+1:>3}/{args.max_samples}] H_coh_total={total:.6g}")

    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        print("No samples processed.")
        return

    print("\nSummary")
    print(f"  mean   : {scores.mean():.6g}")
    print(f"  median : {np.median(scores):.6g}")
    print(f"  p90    : {np.quantile(scores, 0.90):.6g}")


if __name__ == "__main__":
    main()
