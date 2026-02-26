"""Generate publication-quality figures for the Sheaf Transformers paper."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'text.usetex': False,
})

OUTDIR = Path(__file__).parent / "figures"
OUTDIR.mkdir(exist_ok=True)


def run_all_demos():
    """Run showcase computations and return data."""
    from experiments.showcase_advantage import (
        demo_coherence_spectrum,
        demo_gluing_layer,
        demo_spectral_gap,
        demo_multiagent_consensus,
        demo_memory_sheaf,
        demo_auroc_benchmark,
    )
    from sheaf_transformers.metrics.cohomology import CohomologyComputer

    results = {}

    # === Demo 1: Coherence Spectrum ===
    np.random.seed(42)
    H, n, d = 4, 16, 4
    attn = np.random.dirichlet(np.ones(n), size=(H, n))
    cc = CohomologyComputer(H, d, threshold=0.0)
    base = np.random.randn(n, d)

    scenarios = [
        ("Coherent\n(identical)", np.stack([base.copy() for _ in range(H)])),
        ("Low noise\n(σ=0.05)", np.stack([base + 0.05 * np.random.randn(n, d) for _ in range(H)])),
        ("Moderate\n(σ=0.3)", np.stack([base + 0.3 * np.random.randn(n, d) for _ in range(H)])),
        ("High noise\n(σ=1.0)", np.stack([base + 1.0 * np.random.randn(n, d) for _ in range(H)])),
        ("Very high\n(σ=3.0)", np.stack([base + 3.0 * np.random.randn(n, d) for _ in range(H)])),
        ("Random\n(independent)", np.random.randn(H, n, d) * 2.0),
    ]

    demo1_data = []
    for name, heads in scenarios:
        r = cc.compute_H_coh(heads, attn)
        var = float(np.mean(np.var(heads, axis=0)))
        demo1_data.append({
            "name": name,
            "variance": var,
            "H_coh": r["H_coh"],
            "H_coh_frac": r["H_coh_frac"],
        })
    results["demo1"] = demo1_data

    # === Demo 2: Gluing ===
    np.random.seed(42)
    H2, n2, d2 = 4, 12, 6
    attn2 = np.random.dirichlet(np.ones(n2), size=(H2, n2))
    cc2 = CohomologyComputer(H2, d2, threshold=0.0)
    from sheaf_transformers.layers.gluing import GluingLayer

    demo2_data = []
    for trial in range(20):
        np.random.seed(trial + 100)
        heads = np.random.randn(H2, n2, d2)
        gl = GluingLayer(H2, d2, threshold=0.0)
        output = gl.forward(heads, attn2)
        r_before = cc2.compute_H_coh(heads, attn2, restriction_maps={})
        r_after = cc2.compute_H_coh(
            output.glued[np.newaxis].repeat(H2, axis=0), attn2, restriction_maps={})
        demo2_data.append({
            "trial": trial + 1,
            "before": r_before["cocycle_norm"],
            "after": r_after["cocycle_norm"],
        })
    results["demo2"] = demo2_data

    # === Demo 3: Spectral Gap ===
    np.random.seed(42)
    from sheaf_transformers.metrics.spectral import SheafLaplacian
    n3, d3, H3 = 12, 4, 4
    W_fixed = np.random.randn(H3, d3, d3) * 0.3
    for h in range(H3):
        W_fixed[h] += np.eye(d3)
    base3 = np.random.randn(n3, d3)

    demo3_data = []
    scenario_names = ["Disjoint", "Overlapping", "Dense", "Uniform"]
    attn_list = []

    attn_d = np.ones((H3, n3, n3)) * 1e-6
    blk = n3 // H3
    for h in range(H3):
        s, e = h * blk, min((h + 1) * blk, n3)
        for i in range(n3):
            attn_d[h, i, s:e] = 1.0
    attn_d /= attn_d.sum(axis=-1, keepdims=True)
    attn_list.append(attn_d)

    attn_p = np.ones((H3, n3, n3)) * 1e-6
    for h in range(H3):
        s = max(0, (h * blk) - blk // 2)
        e = min(s + blk + blk // 2, n3)
        for i in range(n3):
            attn_p[h, i, s:e] = 1.0
    attn_p /= attn_p.sum(axis=-1, keepdims=True)
    attn_list.append(attn_p)

    attn_list.append(np.random.dirichlet(np.ones(n3), size=(H3, n3)))
    attn_list.append(np.ones((H3, n3, n3)) / n3)

    for name, attn_s in zip(scenario_names, attn_list):
        sl = SheafLaplacian(n3, d3, threshold=0.001)
        L = sl.construct_laplacian(attn_s, W_fixed)
        _, gap = sl.compute_spectral_gap(L)
        trace = float(np.trace(L))
        B, _ = sl.construct_coboundary(attn_s, W_fixed)
        energy = float(np.dot(B @ base3.flatten(), B @ base3.flatten()))
        bound = trace / max(gap, 1e-10)
        demo3_data.append({
            "name": name, "gap": gap, "trace": trace,
            "energy": energy, "bound": bound,
        })
    results["demo3"] = demo3_data

    # === Demo 4: Multi-Agent ===
    np.random.seed(42)
    sys.path.insert(0, str(ROOT / "src"))
    from src.multiagent_sheaf import MultiAgentSheaf
    env = MultiAgentSheaf(n_env_points=15, dim=4)
    truth = np.random.randn(15, 4)
    noise = 2.0
    domains = [[0,1,2,3,4,5],[3,4,5,6,7,8],[6,7,8,9,10,11],
               [9,10,11,12,13,14],[0,1,12,13,14]]
    for domain in domains:
        env.add_agent(domain, truth[domain] + noise * np.random.randn(len(domain), 4))
    result_ma = env.sheaf_consensus(max_rounds=300, tol=1e-10, verbose=False)
    history = result_ma["history"]
    results["demo4"] = [{"round": h["round"], "energy": h["energy"]} for h in history]

    # === Demo 5: Memory diffusion ===
    np.random.seed(42)
    n_mem, d5 = 20, 4
    timestamps = np.linspace(0, 25, n_mem).tolist()
    embeddings = []
    for t in timestamps:
        if t < 10:
            emb = [1, 0, 0, 0]
        elif t > 15:
            emb = [0, 1, 0, 0]
        else:
            alpha = (t - 10) / 5
            emb = [1 - alpha, alpha, 0, 0]
        emb = (np.array(emb) + 0.2 * np.random.randn(d5)).tolist()
        embeddings.append(emb)

    embeddings = np.array(embeddings)
    window_size, window_step = 5.0, 3.0
    contexts = []
    for center in np.arange(0, 25, window_step):
        members = [i for i, t in enumerate(timestamps)
                   if abs(t - center) <= window_size / 2]
        if len(members) >= 2:
            contexts.append(members)
    overlaps = {}
    for i in range(len(contexts)):
        for j in range(i + 1, len(contexts)):
            shared = sorted(set(contexts[i]) & set(contexts[j]))
            if shared:
                overlaps[(i, j)] = shared
    beliefs = np.array([embeddings[ctx].mean(axis=0) for ctx in contexts])

    eta = 0.2
    demo5_data = []
    b = beliefs.copy()
    for iteration in range(300):
        energy = sum(np.dot(b[i] - b[j], b[i] - b[j]) for (i, j) in overlaps)
        demo5_data.append({"iter": iteration, "energy": float(energy)})
        if energy < 1e-10:
            break
        grad = np.zeros_like(b)
        for (i, j) in overlaps:
            diff = b[i] - b[j]
            grad[i] += diff
            grad[j] -= diff
        b = b - eta * grad
    results["demo5"] = demo5_data

    # === Demo 7: AUROC ===
    np.random.seed(42)
    n_samples = 400
    H7, n7, d7 = 4, 12, 4
    labels = np.zeros(n_samples, dtype=int)
    h_coh_scores, var_scores, ent_scores = [], [], []

    for i in range(n_samples):
        rng = np.random.RandomState(i + 1000)
        alpha = rng.uniform(0.1, 5.0)
        attn7 = rng.dirichlet(np.ones(n7) * alpha, size=(H7, n7))
        if i < n_samples // 2:
            base7 = rng.randn(n7, d7)
            nl = rng.uniform(0.01, 0.15)
            heads7 = np.stack([base7 + nl * rng.randn(n7, d7) for _ in range(H7)])
            labels[i] = 0
        else:
            mag = rng.uniform(0.5, 3.0)
            heads7 = rng.randn(H7, n7, d7) * mag
            labels[i] = 1
        cc7 = CohomologyComputer(H7, d7, threshold=0.0)
        r7 = cc7.compute_H_coh(heads7, attn7)
        h_coh_scores.append(r7["H_coh"])
        var_scores.append(float(np.mean(np.var(heads7, axis=0))))
        eps = 1e-10
        ent = 0
        for h_idx in range(H7):
            for j in range(n7):
                p = attn7[h_idx, j] + eps
                p = p / p.sum()
                ent -= np.sum(p * np.log(p))
        ent_scores.append(ent / (H7 * n7))

    from experiments.exp1_hallucination_benchmark import compute_auroc
    results["demo7"] = {
        "H_coh": float(compute_auroc(labels, np.array(h_coh_scores))),
        "Variance": float(compute_auroc(labels, np.array(var_scores))),
        "Entropy": float(compute_auroc(labels, np.array(ent_scores))),
        "Random": float(compute_auroc(labels, np.random.rand(n_samples))),
    }
    # Also store raw scores for ROC curve
    results["demo7_raw"] = {
        "labels": labels,
        "h_coh": np.array(h_coh_scores),
        "var": np.array(var_scores),
        "ent": np.array(ent_scores),
    }

    return results


def fig1_coherence_spectrum(data):
    """Figure 1: H_coh vs noise level (bar chart + table)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), gridspec_kw={'width_ratios': [3, 2]})

    names = [d["name"] for d in data]
    h_coh = [d["H_coh"] for d in data]
    h_coh_frac = [d["H_coh_frac"] for d in data]

    colors = ['#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c', '#9b59b6']
    bars = ax1.bar(range(len(names)), h_coh, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, fontsize=7)
    ax1.set_ylabel(r'$\mathcal{H}_{\mathrm{coh}}$')
    ax1.set_title(r'Cohomological Incoherence $\mathcal{H}_{\mathrm{coh}}$ vs. Noise Level')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Add value labels on bars
    for bar, val in zip(bars, h_coh):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=7)

    # Table side
    ax2.axis('off')
    table_data = []
    for d in data:
        table_data.append([
            d["name"].replace("\n", " "),
            f'{d["variance"]:.4f}',
            f'{d["H_coh"]:.2f}',
            f'{d["H_coh_frac"]*100:.1f}%'
        ])
    table = ax2.table(cellText=table_data,
                      colLabels=['Scenario', 'Var', r'$H_{coh}$', r'$H_{coh}/\|c\|^2$'],
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.3)
    # Color first row green
    table[1, 2].set_facecolor('#d5f5e3')
    # Color last rows red
    for i in [5, 6]:
        table[i, 2].set_facecolor('#fadbd8')

    plt.tight_layout()
    fig.savefig(OUTDIR / "fig_coherence_spectrum.pdf")
    fig.savefig(OUTDIR / "fig_coherence_spectrum.png")
    plt.close(fig)
    print("  -> fig_coherence_spectrum.pdf")


def fig2_gluing(data):
    """Figure 2: Gluing layer - before/after cocycle norms."""
    fig, ax = plt.subplots(figsize=(6, 2.8))

    trials = [d["trial"] for d in data]
    before = [d["before"] for d in data]
    after = [d["after"] for d in data]

    x = np.arange(len(trials))
    w = 0.35
    ax.bar(x - w/2, before, w, label=r'$\|\delta c\|$ before gluing',
           color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.3)
    ax.bar(x + w/2, after, w, label=r'$\|\delta c\|$ after gluing',
           color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=0.3)

    ax.set_xlabel('Trial')
    ax.set_ylabel('Cocycle norm')
    ax.set_title('Gluing Layer: 100% Cocycle Reduction (Theorem 6.3)')
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in trials], fontsize=7)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(OUTDIR / "fig_gluing.pdf")
    fig.savefig(OUTDIR / "fig_gluing.png")
    plt.close(fig)
    print("  -> fig_gluing.pdf")


def fig3_spectral_gap(data):
    """Figure 3: Spectral gap comparison across attention patterns."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))

    names = [d["name"] for d in data]
    gaps = [d["gap"] for d in data]
    bounds = [d["bound"] for d in data]

    colors = ['#3498db', '#9b59b6', '#2ecc71', '#e67e22']
    bars1 = ax1.bar(names, gaps, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel(r'Spectral gap $\gamma$')
    ax1.set_title(r'Spectral Gap $\gamma$ (Theorem 5.3)')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    for bar, val in zip(bars1, gaps):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    bars2 = ax2.bar(names, bounds, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel(r'Generalization bound $\mathrm{tr}(L)/\gamma$')
    ax2.set_title('Generalization Bound (lower = tighter)')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    for bar, val in zip(bars2, bounds):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.0f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    fig.savefig(OUTDIR / "fig_spectral_gap.pdf")
    fig.savefig(OUTDIR / "fig_spectral_gap.png")
    plt.close(fig)
    print("  -> fig_spectral_gap.pdf")


def fig4_consensus(data):
    """Figure 4: Multi-agent consensus convergence."""
    fig, ax = plt.subplots(figsize=(4, 3))

    rounds = [d["round"] for d in data]
    energy = [d["energy"] for d in data]

    ax.semilogy(rounds, energy, 'o-', color='#e67e22', markersize=4, linewidth=1.5)
    ax.fill_between(rounds, energy, alpha=0.15, color='#e67e22')
    ax.set_xlabel('Consensus round')
    ax.set_ylabel(r'Sheaf energy $E_{\mathrm{agents}}$ (log scale)')
    ax.set_title('Multi-Agent Consensus (Theorem 9.4)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Annotate start and end
    ax.annotate(f'E₀ = {energy[0]:.0f}', xy=(rounds[0], energy[0]),
                xytext=(5, energy[0]*2), fontsize=8,
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax.annotate(f'E_final = {energy[-1]:.1e}', xy=(rounds[-1], energy[-1]),
                xytext=(rounds[-1]-8, energy[-1]*100), fontsize=8,
                arrowprops=dict(arrowstyle='->', color='gray'))

    plt.tight_layout()
    fig.savefig(OUTDIR / "fig_consensus.pdf")
    fig.savefig(OUTDIR / "fig_consensus.png")
    plt.close(fig)
    print("  -> fig_consensus.pdf")


def fig5_memory(data):
    """Figure 5: Memory sheaf diffusion convergence."""
    fig, ax = plt.subplots(figsize=(4, 3))

    iters = [d["iter"] for d in data]
    energy = [d["energy"] for d in data]

    ax.semilogy(iters, energy, '-', color='#27ae60', linewidth=1.5)
    ax.fill_between(iters, energy, alpha=0.15, color='#27ae60')
    ax.set_xlabel('Diffusion iteration')
    ax.set_ylabel(r'Disagreement energy $E_{\mathrm{mem}}$ (log scale)')
    ax.set_title('Memory Sheaf Diffusion (Theorem 8.4)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='--')

    ax.annotate(f'E₀ = {energy[0]:.2f}', xy=(iters[0], energy[0]),
                xytext=(30, energy[0]*1.5), fontsize=8,
                arrowprops=dict(arrowstyle='->', color='gray'))

    plt.tight_layout()
    fig.savefig(OUTDIR / "fig_memory_diffusion.pdf")
    fig.savefig(OUTDIR / "fig_memory_diffusion.png")
    plt.close(fig)
    print("  -> fig_memory_diffusion.pdf")


def fig6_auroc(auroc_data, raw_data):
    """Figure 6: AUROC comparison (bar + ROC curves)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    # Bar chart
    methods = [r'$\mathcal{H}_{\mathrm{coh}}$ (Ours)', 'Variance', 'Entropy', 'Random']
    aurocs = [auroc_data["H_coh"], auroc_data["Variance"],
              auroc_data["Entropy"], auroc_data["Random"]]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#95a5a6']

    bars = ax1.bar(methods, aurocs, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel('AUROC')
    ax1.set_title('Hallucination Detection AUROC')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    for bar, val in zip(bars, aurocs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # ROC curves
    labels = raw_data["labels"]
    for name, scores, color, ls in [
        (r'$\mathcal{H}_{\mathrm{coh}}$', raw_data["h_coh"], '#2ecc71', '-'),
        ('Variance', raw_data["var"], '#3498db', '--'),
        ('Entropy', raw_data["ent"], '#e74c3c', ':'),
    ]:
        from sklearn.metrics import roc_curve as _rc
        fpr, tpr, _ = _rc(labels, scores)
        ax2.plot(fpr, tpr, color=color, linestyle=ls, linewidth=1.5, label=name)

    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=0.8)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curves (n=400)')
    ax2.legend(loc='lower right', framealpha=0.9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_aspect('equal')

    plt.tight_layout()
    fig.savefig(OUTDIR / "fig_auroc.pdf")
    fig.savefig(OUTDIR / "fig_auroc.png")
    plt.close(fig)
    print("  -> fig_auroc.pdf")


def fig6_auroc_no_sklearn(auroc_data):
    """Figure 6 fallback: AUROC bar chart only (no sklearn needed)."""
    fig, ax = plt.subplots(figsize=(4, 3))

    methods = [r'$\mathcal{H}_{\mathrm{coh}}$' + '\n(Ours)', 'Variance', 'Entropy', 'Random']
    aurocs = [auroc_data["H_coh"], auroc_data["Variance"],
              auroc_data["Entropy"], auroc_data["Random"]]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#95a5a6']

    bars = ax.bar(methods, aurocs, color=colors, edgecolor='black', linewidth=0.5, width=0.6)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('AUROC')
    ax.set_title('Hallucination Detection: AUROC Comparison (n=400)')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, val in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    fig.savefig(OUTDIR / "fig_auroc.pdf")
    fig.savefig(OUTDIR / "fig_auroc.png")
    plt.close(fig)
    print("  -> fig_auroc.pdf (bar only, no sklearn)")


if __name__ == "__main__":
    print("Running all showcase computations...")
    data = run_all_demos()
    print("Generating figures...")

    fig1_coherence_spectrum(data["demo1"])
    fig2_gluing(data["demo2"])
    fig3_spectral_gap(data["demo3"])
    fig4_consensus(data["demo4"])
    fig5_memory(data["demo5"])

    try:
        import sklearn
        fig6_auroc(data["demo7"], data["demo7_raw"])
    except ImportError:
        print("  (sklearn not found, using bar-only AUROC figure)")
        fig6_auroc_no_sklearn(data["demo7"])

    print(f"\nAll figures saved to {OUTDIR}/")
    print("Files:", [f.name for f in sorted(OUTDIR.glob("*.pdf"))])
