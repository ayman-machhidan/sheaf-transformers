# API Reference

## Core Metrics

### `CohomologyComputer`

Main class for computing sheaf cohomology metrics on multi-head attention outputs.

```python
from sheaf_transformers.metrics.cohomology import CohomologyComputer

cc = CohomologyComputer(n_heads=H, d_model=d, threshold=0.01)
result = cc.compute_H_coh(head_outputs, attention_matrices)
```

**Constructor parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_heads` | `int` | Number of attention heads |
| `d_model` | `int` | Dimension per head |
| `threshold` | `float` | Attention weight threshold for overlap detection (default: `0.0`) |

**`compute_H_coh(head_outputs, attention_matrices, restriction_maps=None)`**

Compute hallucination metrics using the mixed formulation.

| Parameter | Type | Description |
|-----------|------|-------------|
| `head_outputs` | `np.ndarray (H, n, d)` | Per-head output embeddings |
| `attention_matrices` | `np.ndarray (H, n, n)` | Attention weight matrices |
| `restriction_maps` | `dict \| None` | Optional restriction maps. `None` = auto-generate, `{}` = identity (constant sheaf) |

**Returns** `dict` with:

| Key | Type | Description |
|-----|------|-------------|
| `H_coh` | `float` | Irreconcilable inter-head disagreement energy |
| `H_coh_frac` | `float` | Fraction of irreconcilable energy in `[0, 1]` |
| `q_obs` | `int` | Structural obstruction dimension (dim H^1) |
| `cocycle_norm` | `float` | Norm of the full cocycle vector |
| `n_overlaps` | `int` | Number of pairwise head overlaps |
| `n_triples` | `int` | Number of triple overlaps |
| `sheaf_mode` | `bool` | Whether non-trivial restriction maps were used |
| `rank_delta_0` | `int` | Rank of the coboundary operator delta_0 |
| `rank_delta_1` | `int` | Rank of the coboundary operator delta_1 |
| `dim_C0` | `int` | Dimension of the 0-cochain space |
| `dim_C1` | `int` | Dimension of the 1-cochain space |
| `dim_C2` | `int` | Dimension of the 2-cochain space |

---

### `SheafLaplacian`

Constructs and analyzes the sheaf Laplacian on the attention graph.

```python
from sheaf_transformers.metrics.spectral import SheafLaplacian

sl = SheafLaplacian(n_tokens=n, d_model=d)
L = sl.construct_laplacian(attention_matrices, value_matrices)
eigenvalues, gap = sl.compute_spectral_gap(L)
```

**Constructor parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_tokens` | `int` | Number of tokens in the sequence |
| `d_model` | `int` | Dimension per head |
| `threshold` | `float \| None` | Edge threshold (default: `1/n_tokens`) |

**Key methods:**

- `construct_laplacian(A, W)` -- Build `L_F = B^T B` from attention `A (H, n, n)` and value matrices `W (H, d, d)`
- `compute_spectral_gap(L, k=10)` -- Returns `(eigenvalues[:k], gap)` where gap = first nonzero eigenvalue
- `compute_sheaf_energy(s, A, W)` -- Compute `||B_F s||^2 = s^T L_F s`

---

### `EigenvalueStatistics`

Random Matrix Theory analysis: compares eigenvalue spacing distributions against GOE (Wigner) and Poisson.

```python
from sheaf_transformers.metrics.eigenvalue_stats import EigenvalueStatistics

analyzer = EigenvalueStatistics()
stats = analyzer.analyze(symmetric_matrix)
print(stats["kl_goe"], stats["kl_poisson"])
```

**Key methods:**

- `analyze(A)` -- Symmetrize `A`, compute eigenvalues, return KL divergences to GOE/Poisson
- `unfold_eigenvalues(eigs)` -- Normalize eigenvalue spacings by mean spacing
- `goe_pdf(s)` / `poisson_pdf(s)` -- Reference distribution PDFs

---

## Layers

### `GluingLayer`

Minimum-energy coherence correction via pseudoinverse (Theorem 6.3).

```python
from sheaf_transformers.layers.gluing import GluingLayer

gl = GluingLayer(n_heads=H, d_model=d, threshold=0.0)
output = gl.forward(head_outputs, attention_matrices)
print(f"H_coh: {output.H_coh_before:.4f} -> {output.H_coh_after:.4f}")
print(f"Glued shape: {output.glued.shape}")  # (n, d)
```

**`GluingOutput` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `glued` | `np.ndarray (n, d)` | Corrected output (mean of corrected heads) |
| `H_coh_before` | `float` | H_coh before correction |
| `H_coh_after` | `float` | H_coh after correction (always <= before) |

---

### PyTorch Layers (requires `torch >= 2.0`)

```python
from sheaf_transformers.layers.torch_layers import (
    HolomorphicAttention,
    SheafGluingLayer,
    SheafCoherenceLoss,
    STransformerBlock,
)
```

- **`HolomorphicAttention(d_model, n_heads, lam, dropout)`** -- Holomorphic attention with meromorphic kernel (no softmax)
- **`SheafGluingLayer(n_heads, d_head, threshold)`** -- Differentiable gluing layer
- **`SheafCoherenceLoss(alpha, beta, threshold)`** -- Coherence regularization loss
- **`STransformerBlock(d_model, n_heads, d_ff, lam, dropout)`** -- Complete S-Transformer block

---

## Pipeline

### `SheafExperiment`

High-level analysis pipeline for multi-layer diagnostics.

```python
from sheaf_transformers.pipeline import SheafExperiment

exp = SheafExperiment(model_data, threshold=0.01)
results = exp.run_cohomology(layers=[0, 5, 11])
exp.summary()
```

**`model_data` dict format:**

| Key | Type | Description |
|-----|------|-------------|
| `attention` | `list[np.ndarray]` | Per-layer attention matrices `(H, n, n)` |
| `hidden_states` | `list[np.ndarray]` | Per-layer hidden states `(n, d_model)` |
| `n_layers` | `int` | Number of layers |
| `n_heads` | `int` | Number of attention heads |
| `d_head` | `int` | Dimension per head |
| `tokens` | `list[str]` | Optional token strings |

---

## Extraction (requires `transformers + torch`)

```python
from sheaf_transformers.extract.hf import extract_attention_from_gpt2, bundle_layer_heads

data = extract_attention_from_gpt2("gpt2", "The capital of France is Paris.")
bundle = bundle_layer_heads(data, layer=-1)

# bundle.head_outputs: (H, n, d_head)
# bundle.attention: (H, n, n)
# bundle.tokens: list[str]
```

---

## CLI

```bash
# Basic usage
sheaf-transformers coherence --model gpt2 --text "The capital of France is Paris."

# Custom parameters
sheaf-transformers coherence --model gpt2-medium --threshold 0.05 --layer 8 --device cuda
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `gpt2` | HuggingFace model name |
| `--text` | `"The capital of France is Paris."` | Input text |
| `--layer` | `-1` | Layer index (`-1` = last) |
| `--threshold` | `0.01` | Attention overlap threshold |
| `--device` | `cpu` | Compute device |
