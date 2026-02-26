from sheaf_transformers.extract.hf import extract_attention_from_gpt2, bundle_layer_heads
from sheaf_transformers.metrics.cohomology import CohomologyComputer


if __name__ == "__main__":
    data = extract_attention_from_gpt2("gpt2", "The capital of France is Paris.")
    bundle = bundle_layer_heads(data, layer=-1)

    cc = CohomologyComputer(n_heads=data["n_heads"], d_model=data["d_head"], threshold=0.01)
    res = cc.compute_H_coh(bundle.head_outputs, bundle.attention)

    print("Tokens:", " ".join(bundle.tokens))
    print("H_coh:", res["H_coh"])
    print("q_obs:", res["q_obs"])
