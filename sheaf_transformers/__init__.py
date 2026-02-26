"""Sheaf Transformers: sheaf-theoretic diagnostics & layers for Transformer attention.

Core components:
  - CohomologyComputer: Cech cohomology H_coh computation
  - SheafLaplacian: Spectral analysis via L_F = B^T B
  - EigenvalueStatistics: RMT comparison (GOE vs Poisson)
  - GluingLayer: Minimum-energy coherence correction
  - SheafExperiment: High-level analysis pipeline

Optional (requires torch>=2.0):
  - layers.torch_layers.HolomorphicAttention
  - layers.torch_layers.SheafGluingLayer
  - layers.torch_layers.SheafCoherenceLoss
  - layers.torch_layers.STransformerBlock
"""

from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version('sheaf-transformers')
except Exception:
    __version__ = '0.2.0'

from .metrics.cohomology import CohomologyComputer
from .metrics.spectral import SheafLaplacian
from .metrics.eigenvalue_stats import EigenvalueStatistics
from .layers.gluing import GluingLayer, GluingOutput
from .pipeline import SheafExperiment

__all__ = [
    "CohomologyComputer",
    "SheafLaplacian",
    "EigenvalueStatistics",
    "GluingLayer",
    "GluingOutput",
    "SheafExperiment",
]
