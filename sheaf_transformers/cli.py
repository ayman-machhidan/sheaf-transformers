"""Command-line interface."""

from __future__ import annotations

import argparse


def _cmd_coherence(args: argparse.Namespace) -> int:
    from .extract.hf import extract_attention_from_gpt2, bundle_layer_heads
    from .metrics.cohomology import CohomologyComputer

    data = extract_attention_from_gpt2(model_name=args.model, text=args.text, device=args.device)
    bundle = bundle_layer_heads(data, layer=args.layer)

    comp = CohomologyComputer(n_heads=data["n_heads"], d_model=data["d_head"], threshold=args.threshold)
    r = comp.compute_H_coh(bundle.head_outputs, bundle.attention)

    if bundle.tokens:
        print("Tokens:")
        print(" ".join(bundle.tokens))
        print()
    print("Metrics:")
    print(f"  H_coh  (energy)     : {r['H_coh']:.6g}")
    print(f"  q_obs  (structural) : {r['q_obs']}")
    print(f"  overlaps            : {r.get('n_overlaps', 0)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sheaf-transformers", description="Sheaf coherence diagnostics for Transformers")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("coherence", help="Compute H_coh and q_obs for a model+prompt")
    c.add_argument("--model", default="gpt2")
    c.add_argument("--text", default="The capital of France is Paris.")
    c.add_argument("--layer", type=int, default=-1)
    c.add_argument("--threshold", type=float, default=0.01)
    c.add_argument("--device", default="cpu")
    c.set_defaults(func=_cmd_coherence)

    return p


def main() -> int:
    p = build_parser()
    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
