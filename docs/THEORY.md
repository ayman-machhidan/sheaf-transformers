# Theory (short)

This project provides **sheaf-theoretic diagnostics** for Transformer attention.

## What we measure

Given a multi-head attention layer, each head provides a local section over tokens.
Overlaps between heads induce compatibility constraints. Disagreement on overlaps forms a cocycle.

We expose two quantities:

- **Structural obstruction** `q_obs`
  A topological quantity (dimension of a cohomology space) determined by the overlap pattern.

- **Instance coherence energy** `H_coh`
  A per-input score: squared norm of the residual after projecting the cocycle onto `im(delta_0)`
  (computed by sparse least squares / LSQR).

In practice, `H_coh` is the metric you use to flag risky outputs.

## Why both matter

- `q_obs > 0` means the overlap pattern allows inconsistent cycles (structural vulnerability).
- `H_coh` measures the actual disagreement for the given forward pass.
