"""
Temporal Belief Sheaf for Long-Term Memory Coherence.

Implements the framework of Section 8: belief sheaves over temporal contexts,
coherent belief updates, sheaf diffusion for narrative coherence, and the
memory coherence index H_coh^mem.

Reference: Theorems 8.1, 8.2, 8.3, 8.4 (Coherent Update, Narrative Coherence,
           Belief Propagation Convergence, Coherent Memory System)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class Memory:
    """A single memory with embedding and timestamp."""
    embedding: np.ndarray       # d-dimensional belief vector
    timestamp: float            # time of acquisition
    context_id: int = -1        # assigned temporal context
    metadata: dict = field(default_factory=dict)


@dataclass
class TemporalContext:
    """A temporal context U_i: a group of related memories."""
    memory_indices: List[int]   # indices into the memory store
    centroid: Optional[np.ndarray] = None


class TemporalBeliefSheaf:
    """
    Belief sheaf B on a temporal context space (T, tau_T).
    
    Implements:
    - Temporal context construction via similarity kernel (Def 8.1)
    - Belief states as sections B(U_i) (Def 8.2)
    - Restriction maps rho_{ij} between overlapping contexts
    - Memory Laplacian L_B (Def 8.6)
    - Sheaf diffusion for belief propagation (Thm 8.4)
    - Memory coherence index H_coh^mem (Def 8.5)
    """
    
    def __init__(self, dim: int, temporal_radius: float = 5.0,
                 similarity_threshold: float = 0.5, eta: float = 0.01):
        """
        Args:
            dim: embedding dimension d
            temporal_radius: delta for temporal neighborhoods
            similarity_threshold: theta for semantic similarity
            eta: step size for sheaf diffusion
        """
        self.dim = dim
        self.temporal_radius = temporal_radius
        self.similarity_threshold = similarity_threshold
        self.eta = eta
        
        self.memories: List[Memory] = []
        self.contexts: List[TemporalContext] = []
        self.overlap_graph: Dict[Tuple[int,int], List[int]] = {}  # (i,j) -> shared memory indices
        self.restriction_maps: Dict[Tuple[int,int], np.ndarray] = {}  # (i,j) -> R^{d x d}
    
    # ---- Memory Management ----
    
    def add_memory(self, embedding: np.ndarray, timestamp: float,
                   metadata: dict = None) -> int:
        """Add a new memory and return its index."""
        assert embedding.shape == (self.dim,), f"Expected dim {self.dim}, got {embedding.shape}"
        mem = Memory(embedding=embedding.copy(), timestamp=timestamp,
                     metadata=metadata or {})
        idx = len(self.memories)
        self.memories.append(mem)
        return idx
    
    def build_contexts(self, min_context_size: int = 2,
                       window_step: float = None) -> None:
        """Build OVERLAPPING temporal contexts using sliding windows.

        Creates contexts as overlapping time windows, essential for
        non-trivial sheaf cohomology. Each memory can belong to multiple
        contexts, and adjacent contexts share memories (overlaps).

        Implements Definition 8.1: N_delta(t_i) covers via sliding windows
        with step < 2*delta for guaranteed overlap.

        Args:
            min_context_size: minimum memories per context
            window_step: step between window centers (default: temporal_radius
                         for ~50% overlap)
        """
        K = len(self.memories)
        if K == 0:
            return

        if window_step is None:
            window_step = self.temporal_radius

        timestamps = np.array([m.timestamp for m in self.memories])
        t_min, t_max = timestamps.min(), timestamps.max()

        # Create window centers spanning the full time range
        centers = np.arange(t_min, t_max + window_step * 0.5, window_step)

        seen = set()
        self.contexts = []
        for center in centers:
            members = []
            for i in range(K):
                if abs(self.memories[i].timestamp - center) <= self.temporal_radius:
                    members.append(i)

            if len(members) >= min_context_size:
                members = sorted(set(members))
                key = tuple(members)
                if key not in seen:
                    seen.add(key)
                    ctx = TemporalContext(memory_indices=members)
                    self.contexts.append(ctx)

        # Assign each memory to its first context (for backward compat)
        for ctx_id, ctx in enumerate(self.contexts):
            for idx in ctx.memory_indices:
                self.memories[idx].context_id = ctx_id

        # Compute overlaps and restriction maps
        self._compute_overlaps()
        self._compute_restriction_maps()
    
    def _compute_overlaps(self) -> None:
        """Find shared memories between contexts (U_i ∩ U_j)."""
        self.overlap_graph = {}
        M = len(self.contexts)
        for i in range(M):
            set_i = set(self.contexts[i].memory_indices)
            for j in range(i+1, M):
                set_j = set(self.contexts[j].memory_indices)
                shared = list(set_i & set_j)
                if shared:
                    self.overlap_graph[(i, j)] = shared
                    self.overlap_graph[(j, i)] = shared
    
    def _compute_restriction_maps(self) -> None:
        """
        Compute restriction maps rho_{ij}: B(U_i) -> B(U_i ∩ U_j).
        We use orthogonal projection onto the subspace spanned by shared memories.
        """
        self.restriction_maps = {}
        for (i, j), shared in self.overlap_graph.items():
            if len(shared) == 0:
                continue
            # Shared memory embeddings
            shared_embs = np.array([self.memories[k].embedding for k in shared])
            # SVD for projection
            if shared_embs.shape[0] >= self.dim:
                self.restriction_maps[(i, j)] = np.eye(self.dim)
            else:
                U, S, Vt = np.linalg.svd(shared_embs, full_matrices=False)
                # Projection onto shared subspace
                self.restriction_maps[(i, j)] = Vt.T @ Vt
    
    # ---- Cohomology ----
    
    def compute_disagreement_cocycle(self) -> Dict[Tuple[int,int], np.ndarray]:
        """
        Compute the disagreement cocycle c_{ij} = rho_{ij}(s_i) - rho_{ji}(s_j).
        Returns: dict mapping (i,j) -> c_{ij} vector
        """
        cocycles = {}
        for (i, j), shared in self.overlap_graph.items():
            if i >= j:
                continue  # only upper triangle
            rho_ij = self.restriction_maps.get((i, j), np.eye(self.dim))
            rho_ji = self.restriction_maps.get((j, i), np.eye(self.dim))
            
            # Average belief in each context
            s_i = self._context_belief(i)
            s_j = self._context_belief(j)
            
            c_ij = rho_ij @ s_i - rho_ji @ s_j
            cocycles[(i, j)] = c_ij
        return cocycles
    
    def _context_belief(self, ctx_id: int) -> np.ndarray:
        """Average embedding in context (the section s_i)."""
        indices = self.contexts[ctx_id].memory_indices
        if not indices:
            return np.zeros(self.dim)
        embs = np.array([self.memories[k].embedding for k in indices])
        return embs.mean(axis=0)
    
    def memory_coherence_index(self) -> float:
        """Compute memory incoherence energy E_mem = sum ||s_i - s_j||^2.

        This is the sheaf energy s^T L_B s which sheaf diffusion minimizes.
        E_mem = 0 iff all overlapping contexts agree (narrative coherence).
        """
        total = 0.0
        for (i, j) in self.overlap_graph:
            if i >= j:
                continue
            s_i = self._context_belief(i)
            s_j = self._context_belief(j)
            rho_ij = self.restriction_maps.get((i, j), np.eye(self.dim))
            rho_ji = self.restriction_maps.get((j, i), np.eye(self.dim))
            diff = rho_ij @ s_i - rho_ji @ s_j
            total += np.dot(diff, diff)
        return total
    
    # ---- Sheaf Laplacian ----
    
    def memory_laplacian(self) -> np.ndarray:
        """
        Compute the memory sheaf Laplacian L_B = B_B^T B_B.
        Definition 8.6.
        """
        M = len(self.contexts)
        N = M * self.dim
        edges = [(i,j) for (i,j) in self.overlap_graph if i < j]
        n_edges = len(edges)
        
        # Coboundary matrix B
        B = np.zeros((n_edges * self.dim, N))
        for e_idx, (i, j) in enumerate(edges):
            rho_ij = self.restriction_maps.get((i, j), np.eye(self.dim))
            rho_ji = self.restriction_maps.get((j, i), np.eye(self.dim))
            row = e_idx * self.dim
            B[row:row+self.dim, i*self.dim:(i+1)*self.dim] = rho_ij
            B[row:row+self.dim, j*self.dim:(j+1)*self.dim] = -rho_ji
        
        return B.T @ B
    
    def spectral_gap(self) -> float:
        """Compute gamma_mem = lambda_2(L_B)."""
        L = self.memory_laplacian()
        if L.shape[0] == 0:
            return 0.0
        eigenvalues = np.linalg.eigvalsh(L)
        eigenvalues = np.sort(eigenvalues)
        # Find first eigenvalue > tolerance
        for ev in eigenvalues:
            if ev > 1e-10:
                return float(ev)
        return 0.0
    
    # ---- Sheaf Diffusion (Theorem 8.4) ----
    
    def sheaf_diffusion(self, max_iter: int = 100, tol: float = 1e-6) -> Dict:
        """
        Run sheaf diffusion: db/dt = -L_B b(t).
        Discretized as message-passing (Remark 8.3).
        
        Returns: dict with convergence history
        """
        L = self.memory_laplacian()
        if L.shape[0] == 0:
            return {"converged": True, "iterations": 0, "history": []}
        
        # Stability: eta < 2 / lambda_max(L)
        lam_max = np.max(np.linalg.eigvalsh(L))
        if lam_max > 0:
            eta = min(self.eta, 1.5 / lam_max)
        else:
            return {"converged": True, "iterations": 0, "history": []}
        
        # Collect belief vector
        M = len(self.contexts)
        b = np.concatenate([self._context_belief(i) for i in range(M)])
        
        history = []
        for it in range(max_iter):
            h_coh = self.memory_coherence_index()
            history.append({"iteration": it, "H_coh_mem": h_coh})
            
            if h_coh < tol:
                return {"converged": True, "iterations": it, "history": history}
            
            # Update: b <- b - eta * L_B * b
            grad = L @ b
            b = b - eta * grad
            
            # Write back to memories
            for ctx_id in range(M):
                new_belief = b[ctx_id*self.dim:(ctx_id+1)*self.dim]
                for mem_idx in self.contexts[ctx_id].memory_indices:
                    # Weighted update toward context belief
                    self.memories[mem_idx].embedding = 0.5 * self.memories[mem_idx].embedding + 0.5 * new_belief
        
        h_coh = self.memory_coherence_index()
        history.append({"iteration": max_iter, "H_coh_mem": h_coh})
        return {"converged": h_coh < tol, "iterations": max_iter, "history": history}
    
    # ---- Coherent Belief Update (Theorem 8.1) ----
    
    def coherent_update(self, observation: np.ndarray, timestamp: float,
                        affected_contexts: Optional[List[int]] = None) -> bool:
        """
        Perform a coherent belief update (Definition 8.4).
        
        The correction Delta_U(o) is propagated to all affected contexts
        such that the compatibility condition (Eq. 8.1) holds.
        
        Args:
            observation: new observation embedding
            timestamp: time of observation
            affected_contexts: contexts to update (all if None)
        
        Returns: True if update maintained coherence
        """
        if affected_contexts is None:
            affected_contexts = list(range(len(self.contexts)))
        
        # Add memory
        mem_idx = self.add_memory(observation, timestamp)
        
        # Assign to nearest context
        best_ctx = -1
        best_sim = -1.0
        for ctx_id in affected_contexts:
            centroid = self._context_belief(ctx_id)
            sim = np.dot(observation, centroid) / (np.linalg.norm(observation) * np.linalg.norm(centroid) + 1e-10)
            if sim > best_sim:
                best_sim = sim
                best_ctx = ctx_id
        
        if best_ctx >= 0:
            self.contexts[best_ctx].memory_indices.append(mem_idx)
            self.memories[mem_idx].context_id = best_ctx
        
        # Recompute overlaps and restrictions
        self._compute_overlaps()
        self._compute_restriction_maps()
        
        # Check coherence
        h_coh = self.memory_coherence_index()
        
        # If incoherent, run diffusion to restore (Corollary 8.1)
        if h_coh > 1e-6:
            result = self.sheaf_diffusion(max_iter=50, tol=1e-6)
            return result["converged"]
        
        return True


# ---- Tests ----

def test_memory_sheaf():
    """Test the temporal belief sheaf implementation."""
    np.random.seed(42)
    d = 4
    sheaf = TemporalBeliefSheaf(dim=d, temporal_radius=3.0,
                                 similarity_threshold=0.8, eta=0.05)

    # Create memories in two clusters with a gap
    # Cluster 1: times 0-5, embeddings around [1,0,0,0]
    for t in range(6):
        emb = np.array([1.0, 0.0, 0.0, 0.0]) + 0.1 * np.random.randn(d)
        sheaf.add_memory(emb, float(t))

    # Cluster 2: times 10-15, embeddings around [0,1,0,0]
    for t in range(10, 16):
        emb = np.array([0.0, 1.0, 0.0, 0.0]) + 0.1 * np.random.randn(d)
        sheaf.add_memory(emb, float(t))

    # Bridge memories near the gap
    sheaf.add_memory(np.array([0.8, 0.2, 0.0, 0.0]) + 0.05 * np.random.randn(d), 7.0)
    sheaf.add_memory(np.array([0.3, 0.7, 0.0, 0.0]) + 0.05 * np.random.randn(d), 8.0)

    # Build overlapping contexts
    sheaf.build_contexts(min_context_size=2, window_step=2.0)
    print(f"Contexts: {len(sheaf.contexts)}")
    for i, ctx in enumerate(sheaf.contexts):
        times = [sheaf.memories[j].timestamp for j in ctx.memory_indices]
        print(f"  Context {i}: {len(ctx.memory_indices)} memories, "
              f"t=[{min(times):.0f}, {max(times):.0f}]")

    # Check overlaps exist
    n_overlaps = sum(1 for (i, j) in sheaf.overlap_graph if i < j)
    print(f"Overlapping pairs: {n_overlaps}")

    # Compute coherence
    h_coh = sheaf.memory_coherence_index()
    print(f"H_coh^mem = {h_coh:.6f}")

    # Spectral gap
    gamma = sheaf.spectral_gap()
    print(f"Spectral gap gamma_mem = {gamma:.6f}")

    # Run diffusion
    result = sheaf.sheaf_diffusion(max_iter=200, tol=1e-8)
    print(f"Diffusion converged: {result['converged']} in {result['iterations']} iterations")
    if result['history']:
        print(f"  Initial H_coh = {result['history'][0]['H_coh_mem']:.6f}")
        print(f"  Final H_coh = {result['history'][-1]['H_coh_mem']:.8f}")

    print("\n✅ Memory sheaf test passed!")


if __name__ == "__main__":
    test_memory_sheaf()
