"""Eigenvalue statistics analysis for RMT comparison (Conjecture 5.1)."""
import numpy as np
from scipy import linalg
from scipy.stats import ks_2samp


class EigenvalueStatistics:

    @staticmethod
    def unfold_eigenvalues(eigs: np.ndarray) -> np.ndarray:
        eigs = np.sort(eigs)
        spacings = np.diff(eigs)
        m = np.mean(spacings)
        return spacings / m if m > 0 else spacings

    @staticmethod
    def goe_pdf(s): return (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4)

    @staticmethod
    def poisson_pdf(s): return np.exp(-s)

    def analyze(self, A: np.ndarray) -> dict:
        A_sym = (A + A.T) / 2
        eigs = linalg.eigvalsh(A_sym)
        spacings = self.unfold_eigenvalues(eigs)
        if len(spacings) < 10:
            return {'kl_goe': np.inf, 'kl_poisson': np.inf}

        bins = np.linspace(0, 4, 50)
        hist, _ = np.histogram(spacings, bins=bins, density=True)
        bc = (bins[:-1] + bins[1:]) / 2
        eps = 1e-10
        p = hist + eps; p /= p.sum()
        q_g = self.goe_pdf(bc) + eps; q_g /= q_g.sum()
        q_p = self.poisson_pdf(bc) + eps; q_p /= q_p.sum()
        return {
            'kl_goe': float(np.sum(p * np.log(p / q_g))),
            'kl_poisson': float(np.sum(p * np.log(p / q_p))),
            'spectral_gap': float(eigs[1] - eigs[0]) if len(eigs) > 1 else 0
        }
