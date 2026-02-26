# API

## Core metric

```python
from sheaf_transformers.metrics.cohomology import CohomologyComputer

cc = CohomologyComputer(n_heads=H, d_model=d, threshold=0.01)
res = cc.compute_H_coh(head_outputs, attention_matrices)
print(res["H_coh"], res["q_obs"])
```

### Inputs
- `head_outputs`: numpy array `(H, n, d)`
- `attention_matrices`: numpy array `(H, n, n)`

### Outputs
- `H_coh`: float (energy residual)
- `q_obs`: int (structural obstruction)

## CLI

```bash
sheaf-transformers coherence --model gpt2 --text "The capital of France is Paris." --threshold 0.01
```
