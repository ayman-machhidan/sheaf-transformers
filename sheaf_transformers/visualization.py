"""Visualization utilities for Sheaf Transformer analysis.

Generates publication-quality plots demonstrating the advantage of
sheaf-theoretic diagnostics over baselines.

All functions return matplotlib Figure objects for flexible display/saving.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _get_plt():
    """Lazy import matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError("matplotlib required: pip install matplotlib")


# ============================================================
# 1. H_coh Distribution Comparison
# ============================================================

def plot_hcoh_distribution(
    coherent_scores: np.ndarray,
    incoherent_scores: np.ndarray,
    title: str = "H_coh Distribution: Coherent vs Hallucinated",
    save_path: Optional[str] = None,
):
    """Plot overlapping histograms of H_coh for coherent vs incoherent inputs.

    This is the key figure showing that H_coh separates the two classes.
    """
    plt = _get_plt()
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    bins = np.linspace(
        min(coherent_scores.min(), incoherent_scores.min()),
        max(coherent_scores.max(), incoherent_scores.max()),
        40,
    )

    ax.hist(coherent_scores, bins=bins, alpha=0.6, color="#2ecc71", label="Coherent (truthful)",
            density=True, edgecolor="white", linewidth=0.5)
    ax.hist(incoherent_scores, bins=bins, alpha=0.6, color="#e74c3c", label="Hallucinated",
            density=True, edgecolor="white", linewidth=0.5)

    # Add vertical lines for means
    ax.axvline(np.mean(coherent_scores), color="#27ae60", linestyle="--", linewidth=2,
               label=f"Mean coherent: {np.mean(coherent_scores):.3f}")
    ax.axvline(np.mean(incoherent_scores), color="#c0392b", linestyle="--", linewidth=2,
               label=f"Mean hallucinated: {np.mean(incoherent_scores):.3f}")

    ax.set_xlabel("$\\mathcal{H}_{coh}$ (Sheaf Energy)", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ============================================================
# 2. Per-Layer H_coh Heatmap
# ============================================================

def plot_layer_hcoh_heatmap(
    per_layer_data: List[List[Dict]],
    labels: np.ndarray,
    save_path: Optional[str] = None,
):
    """Heatmap of H_coh per layer, grouped by coherent/hallucinated.

    Shows which layers are most discriminative.
    """
    plt = _get_plt()

    if not per_layer_data or not per_layer_data[0]:
        return None

    n_layers = len(per_layer_data[0])
    n_samples = len(per_layer_data)

    matrix = np.zeros((n_samples, n_layers))
    for i, pld in enumerate(per_layer_data):
        for ell, layer_data in enumerate(pld):
            matrix[i, ell] = layer_data.get("H_coh", 0.0)

    # Sort by label then by total H_coh
    sort_idx = np.lexsort((matrix.sum(axis=1), labels))
    matrix = matrix[sort_idx]
    sorted_labels = labels[sort_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [4, 1]})

    im = ax1.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax1.set_xlabel("Layer", fontsize=12)
    ax1.set_ylabel("Sample (sorted by label)", fontsize=12)
    ax1.set_title("$\\mathcal{H}_{coh}$ per Layer per Sample", fontsize=14, fontweight="bold")

    # Mark boundary between classes
    boundary = np.searchsorted(sorted_labels, 1)
    ax1.axhline(boundary - 0.5, color="white", linewidth=2, linestyle="--")
    ax1.text(n_layers / 2, boundary / 2, "Coherent", color="white", fontsize=12,
             ha="center", va="center", fontweight="bold")
    ax1.text(n_layers / 2, (boundary + n_samples) / 2, "Hallucinated", color="white",
             fontsize=12, ha="center", va="center", fontweight="bold")

    plt.colorbar(im, ax=ax1, label="$\\mathcal{H}_{coh}$")

    # Per-layer AUROC
    aurocs = []
    for ell in range(n_layers):
        col = matrix[:, ell]
        from experiments.exp1_hallucination_benchmark import compute_auroc
        aurocs.append(compute_auroc(sorted_labels, col))

    ax2.barh(range(n_layers), aurocs, color="#3498db", edgecolor="white")
    ax2.set_xlabel("AUROC", fontsize=12)
    ax2.set_ylabel("Layer", fontsize=12)
    ax2.set_title("Per-Layer AUROC", fontsize=14, fontweight="bold")
    ax2.set_xlim(0.4, 1.0)
    ax2.axvline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax2.invert_yaxis()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ============================================================
# 3. AUROC Comparison Bar Chart
# ============================================================

def plot_auroc_comparison(
    results: Dict[str, float],
    save_path: Optional[str] = None,
):
    """Bar chart comparing AUROC across methods.

    Highlights the advantage of sheaf-theoretic approach.
    """
    plt = _get_plt()
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    names = list(results.keys())
    aurocs = list(results.values())

    colors = []
    for name in names:
        if "H_coh" in name or "Ours" in name:
            colors.append("#e74c3c")  # Red for ours
        elif "Random" in name:
            colors.append("#95a5a6")
        else:
            colors.append("#3498db")

    bars = ax.barh(range(len(names)), aurocs, color=colors, edgecolor="white", height=0.6)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("AUROC", fontsize=13)
    ax.set_title("Hallucination Detection: AUROC Comparison", fontsize=14, fontweight="bold")
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.5, label="Random baseline")
    ax.set_xlim(0.3, 1.05)

    # Add value labels
    for bar, val in zip(bars, aurocs):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10, fontweight="bold")

    ax.legend(fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ============================================================
# 4. Spectral Gap vs Model Quality
# ============================================================

def plot_spectral_gap_correlation(
    spectral_gaps: np.ndarray,
    quality_metric: np.ndarray,
    quality_name: str = "Perplexity",
    save_path: Optional[str] = None,
):
    """Scatter plot of spectral gap vs model quality."""
    plt = _get_plt()
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    ax.scatter(spectral_gaps, quality_metric, c="#3498db", alpha=0.6, s=50, edgecolors="white")

    # Fit line
    if len(spectral_gaps) > 2:
        z = np.polyfit(spectral_gaps, quality_metric, 1)
        p = np.poly1d(z)
        x_fit = np.linspace(spectral_gaps.min(), spectral_gaps.max(), 100)
        ax.plot(x_fit, p(x_fit), "r--", alpha=0.8, linewidth=2, label=f"Linear fit (slope={z[0]:.3f})")

        # Correlation
        corr = np.corrcoef(spectral_gaps, quality_metric)[0, 1]
        ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
                fontsize=12, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Spectral Gap $\\gamma = \\lambda_2(L_\\mathcal{F})$", fontsize=13)
    ax.set_ylabel(quality_name, fontsize=13)
    ax.set_title(f"Spectral Gap vs {quality_name}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ============================================================
# 5. Consensus Convergence Plot
# ============================================================

def plot_consensus_convergence(
    history: List[Dict],
    theoretical_rate: Optional[float] = None,
    title: str = "Sheaf Consensus Protocol Convergence",
    save_path: Optional[str] = None,
):
    """Plot energy decay during sheaf consensus protocol.

    Shows exponential convergence matching Theorem 9.4.
    """
    plt = _get_plt()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    rounds = [h["round"] for h in history]
    energies = [h["energy"] for h in history]

    # Linear scale
    ax1.plot(rounds, energies, "b-o", markersize=3, linewidth=1.5, label="Disagreement Energy $E$")
    ax1.set_xlabel("Communication Round", fontsize=12)
    ax1.set_ylabel("Disagreement Energy", fontsize=12)
    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Log scale
    log_energies = [max(e, 1e-15) for e in energies]
    ax2.semilogy(rounds, log_energies, "b-o", markersize=3, linewidth=1.5, label="$E$ (log scale)")

    if theoretical_rate is not None and len(rounds) > 1:
        # Plot theoretical bound
        E0 = energies[0]
        theoretical = [E0 * theoretical_rate ** (2 * k) for k in rounds]
        ax2.semilogy(rounds, theoretical, "r--", linewidth=2, alpha=0.7,
                      label=f"Theoretical: $E_0 \\cdot \\rho^{{2k}}$, $\\rho$={theoretical_rate:.3f}")

    ax2.set_xlabel("Communication Round", fontsize=12)
    ax2.set_ylabel("Disagreement Energy (log)", fontsize=12)
    ax2.set_title("Convergence Rate (Theorem 9.4)", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ============================================================
# 6. Memory Sheaf Diffusion Plot
# ============================================================

def plot_memory_diffusion(
    history: List[Dict],
    title: str = "Belief Propagation via Sheaf Diffusion",
    save_path: Optional[str] = None,
):
    """Plot H_coh^mem during sheaf diffusion (Theorem 8.4)."""
    plt = _get_plt()
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    iterations = [h["iteration"] for h in history]
    h_coh_mem = [h["H_coh_mem"] for h in history]

    ax.semilogy(iterations, [max(v, 1e-15) for v in h_coh_mem],
                "g-o", markersize=4, linewidth=2, label="$E_{mem}$ (memory incoherence)")
    ax.axhline(0, color="black", linestyle=":", alpha=0.3)

    ax.set_xlabel("Diffusion Iteration", fontsize=13)
    ax.set_ylabel("Memory Incoherence $E_{mem}$ (log)", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ============================================================
# 7. RMT Eigenvalue Statistics
# ============================================================

def plot_eigenvalue_statistics(
    spacings: np.ndarray,
    title: str = "Eigenvalue Spacing Distribution",
    save_path: Optional[str] = None,
):
    """Plot eigenvalue spacing distribution against GOE and Poisson."""
    plt = _get_plt()
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    s = np.linspace(0, 4, 200)
    goe = (np.pi / 2) * s * np.exp(-np.pi * s ** 2 / 4)
    poisson = np.exp(-s)

    ax.hist(spacings, bins=30, density=True, alpha=0.6, color="#3498db",
            edgecolor="white", label="Observed spacings")
    ax.plot(s, goe, "r-", linewidth=2, label="GOE (Wigner surmise)")
    ax.plot(s, poisson, "g--", linewidth=2, label="Poisson")

    ax.set_xlabel("Normalized spacing $s$", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ============================================================
# 8. Gluing Layer Before/After
# ============================================================

def plot_gluing_effect(
    h_coh_before: List[float],
    h_coh_after: List[float],
    save_path: Optional[str] = None,
):
    """Show H_coh reduction from gluing layer (Theorem 6.3)."""
    plt = _get_plt()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter: before vs after
    ax1.scatter(h_coh_before, h_coh_after, c="#3498db", alpha=0.6, s=40, edgecolors="white")
    lim = max(max(h_coh_before), max(h_coh_after)) * 1.1
    ax1.plot([0, lim], [0, lim], "k--", alpha=0.3, label="No improvement")
    ax1.set_xlabel("$\\mathcal{H}_{coh}$ Before Gluing", fontsize=12)
    ax1.set_ylabel("$\\mathcal{H}_{coh}$ After Gluing", fontsize=12)
    ax1.set_title("Gluing Layer: Coherence Improvement", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Bar: reduction ratio
    reductions = [1 - (a / max(b, 1e-10)) for b, a in zip(h_coh_before, h_coh_after)]
    ax2.hist(reductions, bins=20, color="#2ecc71", edgecolor="white", alpha=0.7)
    ax2.axvline(np.mean(reductions), color="red", linestyle="--", linewidth=2,
                label=f"Mean reduction: {np.mean(reductions)*100:.1f}%")
    ax2.set_xlabel("Reduction Ratio", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Distribution of $\\mathcal{H}_{coh}$ Reduction", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
