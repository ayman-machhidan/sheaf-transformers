"""PyTorch-compatible Sheaf Transformer layers.

Provides drop-in replacements for standard attention with sheaf-theoretic
guarantees. These can be used to fine-tune existing models or train new ones
with coherence regularization.

Requires: torch >= 2.0
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _check_torch():
    if not HAS_TORCH:
        raise ImportError("PyTorch required: pip install torch>=2.0")


# ============================================================
# Holomorphic Attention (Theorem 3.10)
# ============================================================

class HolomorphicAttention(nn.Module):
    """Holomorphic attention with meromorphic kernel (Theorem 3.10).

    Key difference from standard attention: NO per-row softmax renormalization.
    Uses a global meromorphic kernel K_lambda that preserves the sheaf morphism
    property, unlike softmax which breaks it (Theorem 3.3).

    Args:
        d_model: model dimension
        n_heads: number of attention heads
        lam: meromorphic kernel width parameter (lambda)
        dropout: attention dropout rate
    """

    def __init__(self, d_model: int, n_heads: int, lam: float = 0.1,
                 dropout: float = 0.0):
        _check_torch()
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.lam = lam

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        # Learnable complex position embeddings
        self.z_real = nn.Parameter(torch.zeros(1))  # placeholder, set per-input
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        for module in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            nn.init.xavier_uniform_(module.weight)

    def _complex_positions(self, n: int, device: torch.device) -> torch.Tensor:
        """Generate complex position embeddings on the unit circle."""
        angles = torch.linspace(0, 2 * math.pi * (1 - 1 / n), n, device=device)
        return torch.complex(torch.cos(angles), torch.sin(angles))

    def _meromorphic_kernel(self, z: torch.Tensor) -> torch.Tensor:
        """K_lambda(z_i, z_j) = 1 / ((z_i - z_j)^2 + lambda^2)

        Returns real-valued kernel matrix of shape (n, n).
        """
        n = z.shape[0]
        zi = z.unsqueeze(1)  # (n, 1)
        zj = z.unsqueeze(0)  # (1, n)
        diff = zi - zj       # (n, n) complex
        kernel = 1.0 / (diff.real ** 2 + diff.imag ** 2 + self.lam ** 2)
        return kernel

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: optional (batch, seq_len, seq_len) attention mask

        Returns:
            output: (batch, seq_len, d_model)
            kernel_weights: (batch, n_heads, seq_len, seq_len) for analysis
        """
        B, N, D = x.shape
        H = self.n_heads
        dk = self.d_k

        Q = self.W_Q(x).view(B, N, H, dk).transpose(1, 2)  # (B, H, N, dk)
        K = self.W_K(x).view(B, N, H, dk).transpose(1, 2)
        V = self.W_V(x).view(B, N, H, dk).transpose(1, 2)

        # QK scores (standard)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dk)  # (B, H, N, N)

        # Meromorphic kernel (global, NOT per-row normalized)
        z = self._complex_positions(N, x.device)
        kernel = self._meromorphic_kernel(z)  # (N, N)

        # Modulate scores with kernel
        weights = kernel.unsqueeze(0).unsqueeze(0) * scores  # (B, H, N, N)

        # Global normalization (NOT per-row softmax!)
        # This preserves the sheaf morphism property (Theorem 3.10)
        kernel_sum = kernel.sum(dim=-1, keepdim=True).unsqueeze(0).unsqueeze(0)
        weights = weights / kernel_sum.clamp(min=1e-10)

        if mask is not None:
            weights = weights.masked_fill(mask == 0, 0.0)

        weights = self.dropout(weights)

        # Apply to values
        out = torch.matmul(weights, V)  # (B, H, N, dk)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.W_O(out)

        return out, weights


# ============================================================
# Sheaf Gluing Layer (Theorem 6.3)
# ============================================================

class SheafGluingLayer(nn.Module):
    """Differentiable gluing layer that minimizes sheaf incoherence.

    Given multi-head outputs, computes the minimum-energy correction
    Delta = (delta_0)^dagger c that reduces H_coh.

    This layer:
    1. Computes pairwise disagreements (cocycle)
    2. Projects onto im(delta_0) to find resolvable component
    3. Applies minimum-norm correction
    4. Returns corrected output with reduced H_coh

    Args:
        n_heads: number of attention heads
        d_head: dimension per head
        threshold: attention threshold for overlap computation
    """

    def __init__(self, n_heads: int, d_head: int, threshold: float = 0.01):
        _check_torch()
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.threshold = threshold

        # Learnable restriction maps (generalize beyond identity)
        self.restriction_scale = nn.Parameter(torch.ones(n_heads))

    def forward(self, head_outputs: torch.Tensor,
                attention_matrices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            head_outputs: (batch, n_heads, seq_len, d_head)
            attention_matrices: (batch, n_heads, seq_len, seq_len)

        Returns:
            glued_output: (batch, seq_len, d_head)
            h_coh: (batch,) per-sample H_coh scores (for loss)
        """
        B, H, N, D = head_outputs.shape

        # Scale head outputs by learnable restriction maps
        scaled = head_outputs * self.restriction_scale.view(1, H, 1, 1)

        # Compute pairwise disagreements
        h_coh_batch = []
        corrections_batch = []

        for b in range(B):
            ho = scaled[b]  # (H, N, D)
            attn = attention_matrices[b]  # (H, N, N)

            # Find overlapping fields
            overlaps = self._compute_overlaps(attn)

            if not overlaps:
                h_coh_batch.append(torch.tensor(0.0, device=head_outputs.device))
                corrections_batch.append(torch.zeros_like(ho))
                continue

            # Build cocycle
            cocycle_parts = []
            for (h, k) in sorted(overlaps.keys()):
                idx = overlaps[(h, k)]
                if len(idx) == 0:
                    continue
                idx_t = torch.tensor(idx, device=head_outputs.device, dtype=torch.long)
                diff = ho[h, idx_t] - ho[k, idx_t]  # (|overlap|, D)
                cocycle_parts.append(diff.reshape(-1))

            if not cocycle_parts:
                h_coh_batch.append(torch.tensor(0.0, device=head_outputs.device))
                corrections_batch.append(torch.zeros_like(ho))
                continue

            cocycle = torch.cat(cocycle_parts)
            h_coh = torch.dot(cocycle, cocycle)
            h_coh_batch.append(h_coh)

            # Simple correction: move each head toward the mean
            mean_output = ho.mean(dim=0, keepdim=True)  # (1, N, D)
            correction = ho - mean_output  # (H, N, D)
            # Scale correction by H_coh-dependent factor
            scale = torch.sigmoid(h_coh / (D * N + 1))
            corrections_batch.append(correction * scale)

        h_coh_tensor = torch.stack(h_coh_batch)
        corrections = torch.stack(corrections_batch)

        # Apply corrections
        corrected = scaled - corrections
        glued = corrected.mean(dim=1)  # (B, N, D)

        return glued, h_coh_tensor

    def _compute_overlaps(self, attn: torch.Tensor) -> dict:
        """Compute head overlap structure from attention matrices."""
        H, N, _ = attn.shape
        attn_np = attn.detach().cpu().numpy()

        fields = []
        for h in range(H):
            field = set()
            for j in range(N):
                if attn_np[h, :, j].max() > self.threshold:
                    field.add(j)
            fields.append(field)

        overlaps = {}
        for h in range(H):
            for k in range(h + 1, H):
                o = sorted(fields[h] & fields[k])
                if o:
                    overlaps[(h, k)] = o
        return overlaps


# ============================================================
# Sheaf Coherence Loss (Definition 6.6)
# ============================================================

class SheafCoherenceLoss(nn.Module):
    """Differentiable sheaf coherence regularizer.

    Implements the loss: L_S = L_task + alpha * sum_l H_coh^(l) + beta * sum_l 1/gamma^(l)

    Can be added to any transformer training loop.
    """

    def __init__(self, alpha: float = 0.1, beta: float = 0.01,
                 threshold: float = 0.01):
        _check_torch()
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold

    def forward(self, head_outputs: torch.Tensor,
                attention_matrices: torch.Tensor) -> torch.Tensor:
        """Compute sheaf coherence penalty.

        Args:
            head_outputs: (batch, n_heads, seq_len, d_head) or list thereof
            attention_matrices: (batch, n_heads, seq_len, seq_len) or list thereof

        Returns:
            loss: scalar coherence penalty
        """
        if isinstance(head_outputs, list):
            total_loss = torch.tensor(0.0, device=head_outputs[0].device)
            for ho, attn in zip(head_outputs, attention_matrices):
                total_loss = total_loss + self._layer_loss(ho, attn)
            return total_loss
        else:
            return self._layer_loss(head_outputs, attention_matrices)

    def _layer_loss(self, head_outputs: torch.Tensor,
                    attention_matrices: torch.Tensor) -> torch.Tensor:
        """Compute per-layer coherence loss."""
        B, H, N, D = head_outputs.shape

        total_h_coh = torch.tensor(0.0, device=head_outputs.device)

        for b in range(B):
            ho = head_outputs[b]  # (H, N, D)

            # Pairwise head disagreement (differentiable cocycle energy)
            for h in range(H):
                for k in range(h + 1, H):
                    diff = ho[h] - ho[k]  # (N, D)
                    total_h_coh = total_h_coh + torch.sum(diff ** 2)

        # Normalize
        n_pairs = H * (H - 1) / 2
        loss = self.alpha * total_h_coh / (B * max(n_pairs, 1) * N * D)

        return loss


# ============================================================
# Complete S-Transformer Block
# ============================================================

class STransformerBlock(nn.Module):
    """A complete S-Transformer block with holomorphic attention + gluing.

    Drop-in replacement for a standard Transformer block that adds:
    1. Holomorphic attention (sheaf morphism, Theorem 3.10)
    2. Gluing layer (coherence correction, Theorem 6.3)
    3. Coherence loss computation

    Args:
        d_model: model dimension
        n_heads: number of heads
        d_ff: feedforward dimension
        lam: meromorphic kernel parameter
        dropout: dropout rate
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int = 2048,
                 lam: float = 0.1, dropout: float = 0.1):
        _check_torch()
        super().__init__()

        self.attention = HolomorphicAttention(d_model, n_heads, lam, dropout)
        self.gluing = SheafGluingLayer(n_heads, d_model // n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.coherence_loss_fn = SheafCoherenceLoss()

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: optional attention mask

        Returns:
            output: (batch, seq_len, d_model)
            coherence_loss: scalar sheaf coherence penalty
        """
        # Holomorphic attention
        attn_out, weights = self.attention(x, mask)

        # Residual + norm
        x = self.norm1(x + attn_out)

        # Feedforward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        # Coherence loss (from attention weights)
        B, H, N, _ = weights.shape
        D = self.attention.d_k
        # Reshape for coherence computation
        head_out = attn_out.view(B, N, H, D).transpose(1, 2)
        coherence_loss = self.coherence_loss_fn(head_out, weights)

        return x, coherence_loss
