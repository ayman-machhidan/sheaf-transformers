"""Simple web server to display Sheaf Transformers showcase results."""
import http.server
import json
import os
import sys
import io
import traceback
from pathlib import Path

# Add project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

PORT = 8765


def run_showcase():
    """Run the showcase and capture output."""
    from experiments.showcase_advantage import (
        demo_coherence_spectrum,
        demo_gluing_layer,
        demo_spectral_gap,
        demo_multiagent_consensus,
        demo_memory_sheaf,
        demo_rmt,
        demo_auroc_benchmark,
    )
    import numpy as np

    results = {}

    # Demo 1: Coherence Spectrum
    np.random.seed(42)
    H, n, d = 4, 16, 4
    attn = np.random.dirichlet(np.ones(n), size=(H, n))
    from sheaf_transformers.metrics.cohomology import CohomologyComputer
    cc = CohomologyComputer(H, d, threshold=0.0)
    base = np.random.randn(n, d)

    scenarios = [
        ("Coherent (identical)", np.stack([base.copy() for _ in range(H)])),
        ("Low noise (0.05)", np.stack([base + 0.05 * np.random.randn(n, d) for _ in range(H)])),
        ("Moderate noise (0.3)", np.stack([base + 0.3 * np.random.randn(n, d) for _ in range(H)])),
        ("High noise (1.0)", np.stack([base + 1.0 * np.random.randn(n, d) for _ in range(H)])),
        ("Very high noise (3.0)", np.stack([base + 3.0 * np.random.randn(n, d) for _ in range(H)])),
        ("Random (independent)", np.random.randn(H, n, d) * 2.0),
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

    # Demo 2: Gluing
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

    # Demo 3: Spectral Gap
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
            "name": name,
            "gap": gap,
            "trace": trace,
            "energy": energy,
            "bound": bound,
        })
    results["demo3"] = demo3_data

    # Demo 4: Multi-Agent
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
    demo4_data = [{"round": h["round"], "energy": h["energy"]} for h in history]
    results["demo4"] = demo4_data

    # Demo 5: Memory diffusion
    np.random.seed(42)
    n_mem = 20
    d5 = 4
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

    # Demo 7: AUROC
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

    return results


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "app"), **kwargs)

    def do_GET(self):
        if self.path == "/api/data":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                data = run_showcase()
                self.wfile.write(json.dumps(data).encode())
            except Exception as e:
                tb = traceback.format_exc()
                self.wfile.write(json.dumps({"error": str(e), "traceback": tb}).encode())
        else:
            if self.path == "/":
                self.path = "/index.html"
            super().do_GET()

    def log_message(self, format, *args):
        print(f"[server] {format % args}")


if __name__ == "__main__":
    os.chdir(str(ROOT / "app"))
    print(f"Starting Sheaf Transformers Dashboard on http://localhost:{PORT}")
    print(f"Open your browser to http://localhost:{PORT}")
    server = http.server.HTTPServer(("0.0.0.0", PORT), Handler)
    server.serve_forever()
