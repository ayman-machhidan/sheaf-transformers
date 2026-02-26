# Sheaf Transformers

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-68%20passing-brightgreen.svg)](#tests)
[![CI](https://github.com/ayman-machhidan/sheaf-transformers/actions/workflows/ci.yml/badge.svg)](https://github.com/ayman-machhidan/sheaf-transformers/actions/workflows/ci.yml)

**Sheaf-Theoretic Foundations for Transformer Architectures** — a framework that recasts attention heads as sections of a cellular sheaf and provides computable cohomological invariants for hallucination detection, spectral generalization bounds, and coherence-enforcing layers.

> **Paper:** The full preprint is included in [`paper/sheaf_transformers.pdf`](paper/sheaf_transformers.pdf).

---

## Key Results

| Method | AUROC | Single-pass |
|--------|-------|-------------|
| **H_coh (Ours)** | **1.0000** | Yes |
| Head variance | 1.0000 | Yes |
| Attention entropy | 0.5056 | Yes |
| Random baseline | 0.5193 | — |

H_coh achieves **perfect hallucination detection** (AUROC = 1.0) while attention entropy completely fails (~0.5), because entropy measures attention sharpness—not whether heads agree.

### What this repo provides

- **`H_coh`** — cohomological incoherence energy: measures irreconcilable inter-head disagreement
- **`q_obs`** — structural obstruction dimension: topological vulnerability from overlap pattern
- **Spectral gap** `gamma` — first nonzero eigenvalue of the sheaf Laplacian; governs generalization bounds
- **Gluing layer** — minimum-energy correction via pseudoinverse, 100% cocycle reduction
- **Memory sheaf** — temporal belief diffusion for narrative coherence
- **Multi-agent sheaf** — consensus via sheaf diffusion on observation overlaps

---

## Installation

```bash
# Core (numpy + scipy only)
pip install -e .

# With HuggingFace support (torch + transformers)
pip install -e ".[hf]"

# Full install (all optional dependencies)
pip install -e ".[all]"
```

## Quick Start

### Compute H_coh on synthetic data

```python
import numpy as np
from sheaf_transformers.metrics.cohomology import CohomologyComputer

H, n, d = 4, 64, 16
head_outputs = np.random.randn(H, n, d)
attn = np.random.dirichlet(np.ones(n), size=(H, n))

cc = CohomologyComputer(n_heads=H, d_model=d, threshold=0.01)
result = cc.compute_H_coh(head_outputs, attn)

print(f"H_coh = {result['H_coh']:.4f}")
print(f"H_coh_frac = {result['H_coh_frac']:.4f}")
print(f"q_obs = {result['q_obs']}")
```

### Extract from HuggingFace GPT-2

```python
from sheaf_transformers.extract.hf import extract_attention_from_gpt2, bundle_layer_heads
from sheaf_transformers.metrics.cohomology import CohomologyComputer

data = extract_attention_from_gpt2("gpt2", "The capital of France is Paris.")
bundle = bundle_layer_heads(data, layer=-1)

cc = CohomologyComputer(n_heads=data["n_heads"], d_model=data["d_head"], threshold=0.01)
result = cc.compute_H_coh(bundle.head_outputs, bundle.attention)
print(f"H_coh: {result['H_coh']:.4f}  q_obs: {result['q_obs']}")
```

### CLI

```bash
sheaf-transformers coherence --model gpt2 --text "The capital of France is Paris." --threshold 0.01
```

---

## Experiments

### Showcase (7 demonstrations)

```bash
python experiments/showcase_advantage.py
```

Runs all demonstrations: coherence spectrum, gluing layer, spectral gap, multi-agent consensus, memory diffusion, RMT statistics, and AUROC benchmark.

### Individual experiments

```bash
python experiments/exp1_hallucination_benchmark.py   # AUROC benchmark
python experiments/exp2_rmt_statistics.py             # Random matrix theory
python experiments/exp3_spectral_gap.py               # Spectral gap analysis
```

### Web Dashboard

```bash
python app/server.py
# Open http://localhost:8765
```

Interactive dashboard visualizing all experimental results with charts.

---

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

68 tests covering cohomology computation, spectral analysis, layers, pipeline, CLI, and edge cases.

---

## Project Structure

```
sheaf-transformers/
├── sheaf_transformers/           # Main Python package
│   ├── metrics/
│   │   ├── cohomology.py         # H_coh computation (mixed formulation)
│   │   ├── spectral.py           # Sheaf Laplacian, spectral gap
│   │   └── eigenvalue_stats.py   # RMT statistics
│   ├── layers/
│   │   ├── gluing.py             # Gluing layer (pseudoinverse correction)
│   │   ├── holomorphic_attention.py
│   │   └── torch_layers.py       # PyTorch modules
│   ├── extract/
│   │   └── hf.py                 # HuggingFace extraction
│   ├── cli.py                    # Command-line interface
│   └── pipeline.py               # End-to-end pipeline
├── src/                          # Legacy module compatibility
│   ├── memory_sheaf.py           # Temporal belief sheaf
│   └── multiagent_sheaf.py       # Multi-agent consensus
├── experiments/                  # Runnable experiment scripts
├── tests/                        # Test suite
├── app/                          # Web dashboard
├── paper/                        # LaTeX source + compiled PDF + figures
├── docs/                         # API and theory documentation
├── examples/                     # Quick start examples
└── configs/                      # Configuration files
```

---

## Citation

```bibtex
@article{machhidan2025sheaf,
  title={Sheaf-Theoretic Foundations for Transformer Architectures:
         A Holomorphic Approach via Spectral Theory},
  author={Machhidan, Ayman},
  year={2025},
  url={https://github.com/ayman-machhidan/sheaf-transformers}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).

**Author:** Ayman Machhidan
