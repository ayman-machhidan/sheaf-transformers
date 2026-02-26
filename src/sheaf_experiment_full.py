"""
Sheaf-Theoretic Analysis of Transformer Attention
===================================================
Experimental Protocol for Validating the Hallucination Cohomology Index

This module implements:
1. Extraction of attention matrices from pre-trained Transformers (GPT-2, LLaMA)
2. Construction of the sheaf Laplacian on the attention graph
3. Computation of the hallucination cohomology index H_coh
4. Eigenvalue statistics analysis (GOE/Poisson comparison)
5. Spectral gap analysis and correlation with model quality
6. Benchmark evaluation on TruthfulQA

Requirements:
    pip install torch transformers numpy scipy matplotlib seaborn tqdm datasets

Author: Ayman Machhidan
"""

import numpy as np
from scipy import linalg
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.stats import ks_2samp, spearmanr
from typing import Optional
import warnings


# =============================================================================
# PART 1: SHEAF LAPLACIAN CONSTRUCTION
# =============================================================================

class SheafLaplacian:
    """
    Constructs and analyzes the sheaf Laplacian on the attention graph.
    
    Given:
    - Attention matrices A^{(ℓ,h)} ∈ R^{n×n} for each layer ℓ and head h
    - Hidden states H ∈ R^{n×d} at each layer
    - Value projection matrices W_V^{(h)} ∈ R^{d×d_v}
    
    Constructs:
    - Attention graph G = (X_n, E) with edges where max_h a_{ij}^{(h)} > ε
    - Sheaf restriction maps ρ_{ij} = W_V^{(h*)} where h* = argmax_h a_{ij}^{(h)}
    - Sheaf Laplacian L_F = B_F^T B_F
    """
    
    def __init__(self, n_tokens: int, d_model: int, threshold: float = None):
        """
        Args:
            n_tokens: Number of tokens in the sequence
            d_model: Dimension of hidden states (per head)
            threshold: Attention threshold ε for edge construction.
                       Default: 1/n_tokens
        """
        self.n = n_tokens
        self.d = d_model
        self.epsilon = threshold if threshold is not None else 1.0 / n_tokens
        
    def construct_attention_graph(self, attention_matrices: np.ndarray) -> tuple:
        """
        Construct the attention graph from multi-head attention matrices.
        
        Args:
            attention_matrices: shape (H, n, n) — attention weights per head
            
        Returns:
            edges: list of (i, j, h*) tuples
            max_attention: (n, n) matrix of max attention across heads
        """
        H, n, _ = attention_matrices.shape
        assert n == self.n, f"Expected {self.n} tokens, got {n}"
        
        # Max attention across heads
        max_attn = np.max(attention_matrices, axis=0)  # (n, n)
        best_head = np.argmax(attention_matrices, axis=0)  # (n, n)
        
        # Construct edge set
        edges = []
        for i in range(n):
            for j in range(n):
                if i != j and max_attn[i, j] > self.epsilon:
                    edges.append((i, j, best_head[i, j]))
        
        return edges, max_attn
    
    def construct_laplacian(self, 
                           attention_matrices: np.ndarray,
                           value_matrices: np.ndarray) -> np.ndarray:
        """
        Construct the sheaf Laplacian L_F.
        
        Args:
            attention_matrices: shape (H, n, n)
            value_matrices: shape (H, d, d) — W_V matrices per head
            
        Returns:
            L_F: shape (n*d, n*d) — the sheaf Laplacian
        """
        edges, _ = self.construct_attention_graph(attention_matrices)
        n, d = self.n, self.d
        
        # Construct L_F as dense matrix (for moderate n)
        # For large n, use sparse construction below
        L = np.zeros((n * d, n * d))
        
        for (i, j, h_star) in edges:
            W_V = value_matrices[h_star]  # (d, d) restriction map
            
            # L_F[i*d:(i+1)*d, i*d:(i+1)*d] += W_V^T @ W_V (diagonal block)
            WtW = W_V.T @ W_V
            L[i*d:(i+1)*d, i*d:(i+1)*d] += WtW
            
            # L_F[i*d:(i+1)*d, j*d:(j+1)*d] -= W_V^T @ W_V (off-diagonal)
            L[i*d:(i+1)*d, j*d:(j+1)*d] -= WtW
        
        return L
    
    def construct_laplacian_sparse(self,
                                   attention_matrices: np.ndarray,
                                   value_matrices: np.ndarray) -> csr_matrix:
        """Sparse version for large sequences."""
        edges, _ = self.construct_attention_graph(attention_matrices)
        n, d = self.n, self.d
        
        L = lil_matrix((n * d, n * d))
        
        for (i, j, h_star) in edges:
            W_V = value_matrices[h_star]
            WtW = W_V.T @ W_V
            
            for a in range(d):
                for b in range(d):
                    L[i*d + a, i*d + b] += WtW[a, b]
                    L[i*d + a, j*d + b] -= WtW[a, b]
        
        return L.tocsr()
    
    def compute_spectral_gap(self, L: np.ndarray, k: int = 5) -> tuple:
        """
        Compute the spectral gap λ_2 of the sheaf Laplacian.
        
        Args:
            L: Sheaf Laplacian (dense or sparse)
            k: Number of smallest eigenvalues to compute
            
        Returns:
            eigenvalues: k smallest eigenvalues
            spectral_gap: λ_2 - λ_1
        """
        if isinstance(L, np.ndarray):
            eigenvalues = np.sort(linalg.eigvalsh(L))[:k]
        else:
            # Sparse: use Lanczos
            eigenvalues = eigsh(L, k=k, which='SM', return_eigenvectors=False)
            eigenvalues = np.sort(eigenvalues)
        
        spectral_gap = eigenvalues[1] - eigenvalues[0] if len(eigenvalues) > 1 else 0.0
        return eigenvalues, spectral_gap


# =============================================================================
# PART 2: HALLUCINATION COHOMOLOGY INDEX
# =============================================================================

class CohomologyComputer:
    """
    Computes the Čech cohomology of multi-head attention outputs.
    
    Given H attention heads with overlapping receptive fields,
    computes H_coh = dim(ker δ^1) - rank(δ^0)
    which measures inter-head inconsistency.
    """
    
    def __init__(self, n_heads: int, d_model: int, threshold: float = 0.0):
        """
        Args:
            n_heads: Number of attention heads H
            d_model: Representation dimension d (per head)
            threshold: Minimum attention weight to be in receptive field
        """
        self.H = n_heads
        self.d = d_model
        self.threshold = threshold
    
    def compute_overlaps(self, attention_matrices: np.ndarray) -> dict:
        """
        Compute receptive field overlaps between attention heads.
        
        Args:
            attention_matrices: shape (H, n, n) — per-head attention
            
        Returns:
            overlaps: dict mapping (h, k) -> list of token indices in U_h ∩ U_k
        """
        H, n, _ = attention_matrices.shape
        
        # Receptive field of head h = set of tokens with above-threshold
        # attention FROM any token
        # U_h = {j : exists i such that a_{ij}^{(h)} > threshold}
        receptive_fields = []
        for h in range(H):
            field = set()
            for j in range(n):
                if np.any(attention_matrices[h, :, j] > self.threshold):
                    field.add(j)
            receptive_fields.append(field)
        
        # Compute pairwise overlaps
        overlaps = {}
        for h in range(H):
            for k in range(h + 1, H):
                overlap = sorted(receptive_fields[h] & receptive_fields[k])
                if len(overlap) > 0:
                    overlaps[(h, k)] = overlap
        
        return overlaps
    
    def compute_cocycles(self, 
                         head_outputs: np.ndarray,
                         overlaps: dict) -> dict:
        """
        Compute the Čech 1-cocycles c_{hk} = s_h|_overlap - s_k|_overlap.
        
        Args:
            head_outputs: shape (H, n, d) — output of each attention head
            overlaps: dict from compute_overlaps
            
        Returns:
            cocycles: dict mapping (h, k) -> c_{hk} ∈ R^{d * |overlap|}
        """
        cocycles = {}
        for (h, k), indices in overlaps.items():
            s_h = head_outputs[h, indices, :]  # (|overlap|, d)
            s_k = head_outputs[k, indices, :]
            c_hk = (s_h - s_k).flatten()  # R^{d * |overlap|}
            cocycles[(h, k)] = c_hk
        
        return cocycles
    
    def construct_coboundary_matrices(self, 
                                       head_outputs: np.ndarray,
                                       attention_matrices: np.ndarray) -> tuple:
        """
        Construct the Čech coboundary matrices δ^0 and δ^1.
        
        Returns:
            delta_0: matrix R^{D_1 × D_0}
            delta_1: matrix R^{D_2 × D_1}
            cocycle_vector: the actual 1-cochain c ∈ R^{D_1}
            overlaps: the overlap structure
        """
        H = self.H
        d = self.d
        n = head_outputs.shape[1]
        
        overlaps = self.compute_overlaps(attention_matrices)
        
        if not overlaps:
            # No overlaps: H_coh = 0 trivially
            return None, None, None, overlaps
        
        # Index the cochains
        # C^0 = prod_h F(U_h) — for simplicity, use full F(X_n) for each head
        D_0 = H * n * d
        
        # C^1 = prod_{h<k} F(U_h ∩ U_k)
        pair_dims = {}
        D_1 = 0
        for (h, k), indices in overlaps.items():
            pair_dims[(h, k)] = len(indices) * d
            D_1 += len(indices) * d
        
        # Construct δ^0: C^0 → C^1
        delta_0 = np.zeros((D_1, D_0))
        
        row_offset = 0
        for (h, k), indices in overlaps.items():
            m = len(indices)
            for idx_pos, token_idx in enumerate(indices):
                for dd in range(d):
                    row = row_offset + idx_pos * d + dd
                    # Contribution from head h (positive)
                    col_h = h * n * d + token_idx * d + dd
                    delta_0[row, col_h] = 1.0
                    # Contribution from head k (negative)
                    col_k = k * n * d + token_idx * d + dd
                    delta_0[row, col_k] = -1.0
            row_offset += m * d
        
        # Construct the actual cocycle vector
        cocycles = self.compute_cocycles(head_outputs, overlaps)
        cocycle_vector = np.concatenate([cocycles[(h, k)] for (h, k) in sorted(overlaps.keys())])
        
        # Construct δ^1: C^1 → C^2 (triple overlaps)
        # For efficiency, compute triple overlaps
        triples = []
        overlap_keys = sorted(overlaps.keys())
        for i, (h, k) in enumerate(overlap_keys):
            for j, (k2, l) in enumerate(overlap_keys):
                if k == k2 and (h, l) in overlaps:
                    # Triple overlap: h, k, l
                    triple_idx = sorted(set(overlaps[(h, k)]) & 
                                       set(overlaps[(k, l)]) & 
                                       set(overlaps[(h, l)]))
                    if triple_idx:
                        triples.append((h, k, l, triple_idx))
        
        if not triples:
            D_2 = 0
            delta_1 = np.zeros((0, D_1))
        else:
            D_2 = sum(len(t[3]) * d for t in triples)
            delta_1 = np.zeros((D_2, D_1))
            
            # Fill δ^1: (c_{hk})_{hkl} = c_{kl}|_triple - c_{hl}|_triple + c_{hk}|_triple
            row_off = 0
            for (h, k, l, triple_idx) in triples:
                m = len(triple_idx)
                for idx_pos, token_idx in enumerate(triple_idx):
                    for dd in range(d):
                        row = row_off + idx_pos * d + dd
                        
                        # Find column offsets in C^1
                        def find_col_offset(pair):
                            off = 0
                            for key in sorted(overlaps.keys()):
                                if key == pair:
                                    idx_in_overlap = overlaps[pair].index(token_idx)
                                    return off + idx_in_overlap * d + dd
                                off += len(overlaps[key]) * d
                            return None
                        
                        col_hk = find_col_offset((h, k))
                        col_kl = find_col_offset((k, l))
                        col_hl = find_col_offset((h, l))
                        
                        if col_hk is not None:
                            delta_1[row, col_hk] = 1.0
                        if col_kl is not None:
                            delta_1[row, col_kl] = 1.0
                        if col_hl is not None:
                            delta_1[row, col_hl] = -1.0
                
                row_off += m * d
        
        return delta_0, delta_1, cocycle_vector, overlaps
    
    def compute_H_coh(self,
                      head_outputs: np.ndarray,
                      attention_matrices: np.ndarray,
                      return_details: bool = False) -> dict:
        """
        Compute the hallucination cohomology index.
        
        H_coh = dim(ker δ^1) - rank(δ^0) = D_1 - rank(δ^0) - rank(δ^1)
        
        Args:
            head_outputs: (H, n, d) per-head outputs
            attention_matrices: (H, n, n) per-head attention
            return_details: if True, return intermediate computations
            
        Returns:
            dict with keys:
                'H_coh': integer hallucination index
                'H_coh_continuous': continuous relaxation
                'details': (optional) intermediate matrices and ranks
        """
        delta_0, delta_1, cocycle_vec, overlaps = \
            self.construct_coboundary_matrices(head_outputs, attention_matrices)
        
        if delta_0 is None:
            return {'H_coh': 0, 'H_coh_continuous': 0.0}
        
        D_1 = delta_0.shape[0]
        
        # Compute ranks via SVD (numerically stable)
        tol = 1e-8
        
        # rank(δ^0)
        _, s0, _ = linalg.svd(delta_0, full_matrices=False)
        rank_delta_0 = np.sum(s0 > tol * s0[0]) if len(s0) > 0 else 0
        
        # rank(δ^1)
        if delta_1.shape[0] > 0:
            _, s1, _ = linalg.svd(delta_1, full_matrices=False)
            rank_delta_1 = np.sum(s1 > tol * s1[0]) if len(s1) > 0 else 0
        else:
            rank_delta_1 = 0
        
        # H_coh = dim(ker δ^1) - rank(δ^0) = (D_1 - rank_delta_1) - rank_delta_0
        H_coh = max(0, (D_1 - rank_delta_1) - rank_delta_0)
        
        # Continuous relaxation: ||c||² - ||π_{im δ^0}(c)||²
        if cocycle_vec is not None and len(cocycle_vec) > 0:
            # Project cocycle onto image of δ^0
            # π_{im δ^0}(c) = δ^0 @ (δ^0)^† @ c
            proj = delta_0 @ linalg.pinv(delta_0) @ cocycle_vec
            residual = cocycle_vec - proj
            H_coh_continuous = float(np.dot(residual, residual))
        else:
            H_coh_continuous = 0.0
        
        result = {
            'H_coh': int(H_coh),
            'H_coh_continuous': H_coh_continuous,
            'rank_delta_0': rank_delta_0,
            'rank_delta_1': rank_delta_1,
            'D_1': D_1,
            'n_overlaps': len(overlaps)
        }
        
        if return_details:
            result['delta_0'] = delta_0
            result['delta_1'] = delta_1
            result['cocycle_vector'] = cocycle_vec
        
        return result


# =============================================================================
# PART 3: EIGENVALUE STATISTICS ANALYSIS
# =============================================================================

class EigenvalueStatistics:
    """
    Analyzes eigenvalue statistics of attention matrices and
    compares with Random Matrix Theory predictions.
    """
    
    @staticmethod
    def unfold_eigenvalues(eigenvalues: np.ndarray) -> np.ndarray:
        """
        Unfold eigenvalues to unit mean spacing.
        Uses the standard procedure: sort, compute spacings, normalize.
        """
        eigs = np.sort(eigenvalues)
        spacings = np.diff(eigs)
        mean_spacing = np.mean(spacings)
        if mean_spacing > 0:
            return spacings / mean_spacing
        return spacings
    
    @staticmethod
    def goe_distribution(s: np.ndarray) -> np.ndarray:
        """
        Wigner surmise for GOE (real symmetric matrices):
        p(s) = (π/2) s exp(-πs²/4)
        """
        return (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4)
    
    @staticmethod
    def gue_distribution(s: np.ndarray) -> np.ndarray:
        """
        Wigner surmise for GUE (complex Hermitian matrices):
        p(s) = (32/π²) s² exp(-4s²/π)
        """
        return (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)
    
    @staticmethod
    def poisson_distribution(s: np.ndarray) -> np.ndarray:
        """Poisson: p(s) = exp(-s)"""
        return np.exp(-s)
    
    def analyze_attention_matrix(self, A: np.ndarray) -> dict:
        """
        Full eigenvalue analysis of a single attention matrix.
        
        Args:
            A: (n, n) attention matrix
            
        Returns:
            dict with eigenvalue statistics
        """
        # Symmetrize
        A_sym = (A + A.T) / 2
        
        # Compute eigenvalues
        eigenvalues = linalg.eigvalsh(A_sym)
        
        # Unfold
        spacings = self.unfold_eigenvalues(eigenvalues)
        
        if len(spacings) < 10:
            return {'eigenvalues': eigenvalues, 'spacings': spacings,
                    'kl_goe': np.inf, 'kl_poisson': np.inf, 'ks_goe': 1.0}
        
        # Compare with RMT predictions
        # Use histogram-based KL divergence
        bins = np.linspace(0, 4, 50)
        hist_empirical, _ = np.histogram(spacings, bins=bins, density=True)
        
        bin_centers = (bins[:-1] + bins[1:]) / 2
        hist_goe = self.goe_distribution(bin_centers)
        hist_poisson = self.poisson_distribution(bin_centers)
        
        # Normalize
        hist_goe /= np.sum(hist_goe) * (bins[1] - bins[0])
        hist_poisson /= np.sum(hist_poisson) * (bins[1] - bins[0])
        
        # KL divergence (with smoothing)
        eps = 1e-10
        p = hist_empirical + eps
        p /= p.sum()
        
        q_goe = hist_goe + eps
        q_goe /= q_goe.sum()
        
        q_poisson = hist_poisson + eps
        q_poisson /= q_poisson.sum()
        
        kl_goe = float(np.sum(p * np.log(p / q_goe)))
        kl_poisson = float(np.sum(p * np.log(p / q_poisson)))
        
        # KS test
        # Generate GOE samples for comparison
        n = len(eigenvalues)
        goe_matrix = np.random.randn(n, n)
        goe_matrix = (goe_matrix + goe_matrix.T) / 2
        goe_eigs = linalg.eigvalsh(goe_matrix)
        goe_spacings = self.unfold_eigenvalues(goe_eigs)
        
        ks_stat, ks_pvalue = ks_2samp(spacings, goe_spacings)
        
        return {
            'eigenvalues': eigenvalues,
            'spacings': spacings,
            'kl_goe': kl_goe,
            'kl_poisson': kl_poisson,
            'ks_goe_stat': float(ks_stat),
            'ks_goe_pvalue': float(ks_pvalue),
            'spectral_gap': float(eigenvalues[1] - eigenvalues[0]) if len(eigenvalues) > 1 else 0,
            'trace': float(np.sum(eigenvalues)),
            'frobenius_norm': float(np.sqrt(np.sum(eigenvalues**2)))
        }


# =============================================================================
# PART 4: TRANSFORMER EXTRACTION (requires torch & transformers)
# =============================================================================

def extract_attention_from_gpt2(
    model_name: str = "gpt2",
    text: str = "The quick brown fox jumps over the lazy dog.",
    device: str = "cpu"
) -> dict:
    """
    Extract attention matrices and hidden states from GPT-2.
    
    Returns:
        dict with:
            'attention': list of (H, n, n) arrays per layer
            'hidden_states': list of (n, d) arrays per layer
            'value_matrices': list of (H, d_head, d_head) arrays per layer
            'tokens': list of token strings
    """
    try:
        import torch
        from transformers import GPT2Tokenizer, GPT2Model
    except ImportError:
        raise ImportError("Install: pip install torch transformers")
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2Model.from_pretrained(model_name, output_attentions=True)
    model.eval()
    model.to(device)
    
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
    
    attentions = [a.squeeze(0).cpu().numpy() for a in outputs.attentions]
    hidden_states = [h.squeeze(0).cpu().numpy() for h in outputs.hidden_states]
    
    # Extract value projection matrices
    value_matrices = []
    n_heads = model.config.n_head
    d_head = model.config.n_embd // n_heads
    
    for layer_idx in range(model.config.n_layer):
        W_V = model.h[layer_idx].attn.c_attn.weight.data.cpu().numpy()
        # GPT-2 c_attn packs Q, K, V: shape (d_model, 3*d_model)
        d_model = model.config.n_embd
        W_V_full = W_V[:, 2*d_model:]  # (d_model, d_model) — V projection
        
        # Reshape to per-head: (H, d_model, d_head)
        W_V_heads = W_V_full.reshape(d_model, n_heads, d_head)
        W_V_heads = W_V_heads.transpose(1, 0, 2)  # (H, d_model, d_head)
        
        # For sheaf Laplacian, we need square restriction maps
        # Use d_head × d_head approximation: W_V_heads[:, :d_head, :]
        value_matrices.append(W_V_heads[:, :d_head, :])
    
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    return {
        'attention': attentions,
        'hidden_states': hidden_states,
        'value_matrices': value_matrices,
        'tokens': tokens,
        'n_layers': model.config.n_layer,
        'n_heads': n_heads,
        'd_head': d_head,
        'd_model': model.config.n_embd
    }


# =============================================================================
# PART 5: FULL EXPERIMENTAL PIPELINE
# =============================================================================

class SheafExperiment:
    """
    Full experimental pipeline for sheaf-theoretic analysis.
    """
    
    def __init__(self, model_data: dict):
        """
        Args:
            model_data: output of extract_attention_from_gpt2
        """
        self.data = model_data
        self.n_layers = model_data['n_layers']
        self.n_heads = model_data['n_heads']
        self.d_head = model_data['d_head']
        self.n_tokens = model_data['attention'][0].shape[1]
        
    def run_cohomology_analysis(self, layers: Optional[list] = None) -> list:
        """
        Compute H_coh for specified layers.
        
        Returns:
            list of H_coh results per layer
        """
        if layers is None:
            layers = range(self.n_layers)
        
        results = []
        for ell in layers:
            attn = self.data['attention'][ell]  # (H, n, n)
            
            # Use hidden states reshaped per head as head outputs
            hidden = self.data['hidden_states'][ell + 1]  # (n, d_model)
            # Reshape to per-head: (H, n, d_head)
            head_outputs = hidden.reshape(
                self.n_tokens, self.n_heads, self.d_head
            ).transpose(1, 0, 2)
            
            cohom = CohomologyComputer(
                n_heads=self.n_heads,
                d_model=self.d_head,
                threshold=1.0 / self.n_tokens
            )
            
            result = cohom.compute_H_coh(head_outputs, attn)
            result['layer'] = ell
            results.append(result)
        
        return results
    
    def run_spectral_analysis(self, layers: Optional[list] = None) -> list:
        """
        Compute sheaf Laplacian spectral gap for specified layers.
        """
        if layers is None:
            layers = range(self.n_layers)
        
        results = []
        for ell in layers:
            attn = self.data['attention'][ell]
            W_V = self.data['value_matrices'][ell]
            
            sheaf_lap = SheafLaplacian(
                n_tokens=self.n_tokens,
                d_model=self.d_head,
                threshold=1.0 / self.n_tokens
            )
            
            L = sheaf_lap.construct_laplacian(attn, W_V)
            eigenvalues, gap = sheaf_lap.compute_spectral_gap(L, k=10)
            
            results.append({
                'layer': ell,
                'spectral_gap': float(gap),
                'top_eigenvalues': eigenvalues.tolist(),
                'trace': float(np.trace(L)),
                'sheaf_complexity': float(np.trace(L) / gap) if gap > 1e-10 else float('inf')
            })
        
        return results
    
    def run_eigenvalue_statistics(self, layers: Optional[list] = None) -> list:
        """
        Analyze eigenvalue statistics of attention matrices vs RMT.
        """
        if layers is None:
            layers = range(self.n_layers)
        
        analyzer = EigenvalueStatistics()
        results = []
        
        for ell in layers:
            layer_results = {'layer': ell, 'heads': []}
            attn = self.data['attention'][ell]
            
            for h in range(self.n_heads):
                stats = analyzer.analyze_attention_matrix(attn[h])
                # Remove large arrays for summary
                summary = {k: v for k, v in stats.items() 
                          if k not in ['eigenvalues', 'spacings']}
                summary['head'] = h
                layer_results['heads'].append(summary)
            
            # Aggregate across heads
            kl_goe_vals = [r['kl_goe'] for r in layer_results['heads'] 
                          if np.isfinite(r['kl_goe'])]
            kl_poisson_vals = [r['kl_poisson'] for r in layer_results['heads']
                              if np.isfinite(r['kl_poisson'])]
            
            layer_results['mean_kl_goe'] = float(np.mean(kl_goe_vals)) if kl_goe_vals else float('inf')
            layer_results['mean_kl_poisson'] = float(np.mean(kl_poisson_vals)) if kl_poisson_vals else float('inf')
            layer_results['rmt_preference'] = 'GOE' if layer_results['mean_kl_goe'] < layer_results['mean_kl_poisson'] else 'Poisson'
            
            results.append(layer_results)
        
        return results
    
    def full_analysis(self) -> dict:
        """Run all analyses and return comprehensive results."""
        print("=" * 60)
        print("SHEAF-THEORETIC ANALYSIS OF TRANSFORMER ATTENTION")
        print("=" * 60)
        print(f"Model: {self.n_layers} layers, {self.n_heads} heads, "
              f"d_head={self.d_head}, n_tokens={self.n_tokens}")
        print()
        
        # 1. Cohomology analysis
        print("1. Computing hallucination cohomology index...")
        cohom_results = self.run_cohomology_analysis()
        for r in cohom_results:
            print(f"   Layer {r['layer']:2d}: H_coh = {r['H_coh']:3d}, "
                  f"H̃_coh = {r['H_coh_continuous']:.4f}, "
                  f"overlaps = {r['n_overlaps']}")
        
        total_H_coh = sum(r['H_coh'] for r in cohom_results)
        total_H_cont = sum(r['H_coh_continuous'] for r in cohom_results)
        print(f"   TOTAL: H_coh = {total_H_coh}, H̃_coh = {total_H_cont:.4f}")
        print()
        
        # 2. Spectral analysis
        print("2. Computing sheaf Laplacian spectral gaps...")
        spectral_results = self.run_spectral_analysis()
        for r in spectral_results:
            print(f"   Layer {r['layer']:2d}: γ = {r['spectral_gap']:.6f}, "
                  f"C_F = {r['sheaf_complexity']:.2f}")
        print()
        
        # 3. Eigenvalue statistics
        print("3. Analyzing eigenvalue statistics (RMT comparison)...")
        eigen_results = self.run_eigenvalue_statistics()
        for r in eigen_results:
            print(f"   Layer {r['layer']:2d}: KL(GOE) = {r['mean_kl_goe']:.4f}, "
                  f"KL(Poisson) = {r['mean_kl_poisson']:.4f}, "
                  f"preference = {r['rmt_preference']}")
        print()
        
        return {
            'cohomology': cohom_results,
            'spectral': spectral_results,
            'eigenvalue_statistics': eigen_results,
            'summary': {
                'total_H_coh': total_H_coh,
                'total_H_coh_continuous': total_H_cont,
                'mean_spectral_gap': float(np.mean([r['spectral_gap'] for r in spectral_results])),
                'n_layers_goe': sum(1 for r in eigen_results if r['rmt_preference'] == 'GOE'),
                'n_layers_poisson': sum(1 for r in eigen_results if r['rmt_preference'] == 'Poisson'),
            }
        }


# =============================================================================
# PART 6: TRUTHFULQA BENCHMARK EVALUATION
# =============================================================================

def evaluate_truthfulqa_protocol():
    """
    Protocol for evaluating H_coh as hallucination detector on TruthfulQA.
    
    This function outlines the complete experimental protocol.
    Actual execution requires downloading TruthfulQA and running
    generation + cohomology analysis.
    """
    protocol = """
    =========================================================
    EXPERIMENTAL PROTOCOL: H_coh on TruthfulQA
    =========================================================
    
    SETUP:
    1. Models: gpt2, gpt2-medium, gpt2-large
    2. Dataset: TruthfulQA (817 questions)
       pip install datasets
       from datasets import load_dataset
       ds = load_dataset("truthful_qa", "generation")
    
    PROTOCOL:
    For each question q in TruthfulQA:
      1. Generate answer a = model.generate(q)
      2. During generation, extract at each token step:
         - Attention matrices A^{(ℓ,h)} for all layers ℓ, heads h
         - Hidden states H^{(ℓ)} for all layers
      3. Compute H_coh^{(ℓ)} for each layer
      4. Aggregate: H_total = Σ_ℓ w_ℓ H_coh^{(ℓ)}
         (weights w_ℓ learned on validation split)
      5. Record: (q, a, H_total, truthful_label)
    
    EVALUATION:
    1. AUROC: Use H_total as score, truthful_label as ground truth
    2. Spearman correlation: H_total vs human truthfulness ratings
    3. Per-layer analysis: which layers contribute most?
    4. Comparison with baselines:
       - SelfCheckGPT (sampling-based)
       - P(True) (Kadavath et al.)
       - Entropy-based detection
    
    ABLATIONS:
    a) Threshold sensitivity: vary ε ∈ {0.01, 0.05, 0.1, 0.2}
    b) Continuous vs discrete H_coh
    c) Layer weighting: uniform vs learned vs last-only
    d) Head subset: all heads vs top-k attention heads
    
    EXPECTED RESULTS:
    - H_coh should be significantly higher for hallucinated outputs
    - AUROC target: ≥ 0.70 (competitive with sampling-based methods)
    - Per-layer: later layers should show stronger signal
    - Advantage: single forward pass (vs 5-20 samples for SelfCheckGPT)
    =========================================================
    """
    print(protocol)


# =============================================================================
# MAIN: Demo with synthetic data
# =============================================================================

def demo_synthetic():
    """
    Demonstrate the full pipeline with synthetic attention data.
    Useful for testing without GPU/model access.
    """
    print("=" * 60)
    print("DEMO: Sheaf Analysis with Synthetic Attention Data")
    print("=" * 60)
    
    np.random.seed(42)
    
    n = 16       # tokens
    d = 8        # dimension per head
    H = 4        # heads
    n_layers = 3
    
    for scenario in ['coherent', 'hallucinating']:
        print(f"\n--- Scenario: {scenario} ---")
        
        for ell in range(n_layers):
            # Generate attention matrices
            attn = np.random.dirichlet(np.ones(n), size=(H, n))  # (H, n, n)
            
            if scenario == 'coherent':
                # Coherent: all heads attend to similar tokens
                base_attn = np.random.dirichlet(np.ones(n) * 0.5, size=n)
                attn = np.stack([base_attn + np.random.randn(n, n) * 0.01 
                                for _ in range(H)])
                attn = np.abs(attn)
                attn /= attn.sum(axis=-1, keepdims=True)
                
                # Head outputs: similar across heads
                base_output = np.random.randn(n, d)
                head_outputs = np.stack([base_output + np.random.randn(n, d) * 0.1
                                        for _ in range(H)])
            else:
                # Hallucinating: heads disagree significantly
                head_outputs = np.stack([np.random.randn(n, d) * (1 + h * 0.5)
                                        for h in range(H)])
            
            # Compute H_coh
            cohom = CohomologyComputer(n_heads=H, d_model=d, threshold=1.0/n)
            result = cohom.compute_H_coh(head_outputs, attn)
            
            print(f"  Layer {ell}: H_coh = {result['H_coh']}, "
                  f"H̃_coh = {result['H_coh_continuous']:.4f}, "
                  f"overlaps = {result['n_overlaps']}")
        
        # Eigenvalue analysis on last layer
        analyzer = EigenvalueStatistics()
        for h in range(min(2, H)):
            stats = analyzer.analyze_attention_matrix(attn[h])
            print(f"  Head {h} eigenvalue stats: "
                  f"KL(GOE)={stats['kl_goe']:.3f}, "
                  f"KL(Poisson)={stats['kl_poisson']:.3f}")
    
    print("\n" + "=" * 60)
    print("Demo complete. For real model analysis:")
    print("  data = extract_attention_from_gpt2('gpt2', 'Your text here')")
    print("  exp = SheafExperiment(data)")
    print("  results = exp.full_analysis()")
    print("=" * 60)


if __name__ == "__main__":
    demo_synthetic()
    print()
    evaluate_truthfulqa_protocol()
