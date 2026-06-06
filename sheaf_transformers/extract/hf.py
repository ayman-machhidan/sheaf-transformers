"""HuggingFace extraction helpers.

These functions are optional at runtime (depend on transformers + torch).
They are kept minimal so the core metrics can be used without HF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ExtractedBundle:
    """A thin container for attention analysis inputs."""

    attention: Any
    head_outputs: Any
    tokens: List[str]
    meta: Dict[str, Any]
    restriction_maps: Optional[Dict] = field(default=None)


def extract_attention_from_gpt2(model_name: str = "gpt2", text: str = "Hello world.",
                                device: str = "cpu") -> Dict[str, Any]:
    """Extract attention matrices, hidden states, and W_V from GPT-2.

    Returns a dict compatible with sheaf_transformers.pipeline.SheafExperiment,
    including per-layer per-head value projection matrices (W_V) for use as
    learned restriction maps in sheaf cohomology.
    """
    import torch
    from transformers import GPT2Tokenizer, GPT2Model

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2Model.from_pretrained(model_name, output_attentions=True,
                                      attn_implementation="eager")
    model.eval().to(device)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model(**inputs, output_attentions=True, output_hidden_states=True)

    attentions = [a.squeeze(0).cpu().numpy() for a in out.attentions]
    hidden = [h.squeeze(0).cpu().numpy() for h in out.hidden_states]

    n_heads = model.config.n_head
    d_model = model.config.n_embd
    d_head = d_model // n_heads

    w_v_per_layer = _extract_wv_matrices(model, n_heads, d_model, d_head)

    return {
        "attention": attentions,
        "hidden_states": hidden,
        "tokens": tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]),
        "n_layers": model.config.n_layer,
        "n_heads": n_heads,
        "d_head": d_head,
        "d_model": d_model,
        "model_name": model_name,
        "w_v": w_v_per_layer,
    }


def extract_from_model(model, tokenizer, text: str,
                       device: str = "cpu") -> Dict[str, Any]:
    """Extract from an already-loaded model (avoids repeated loading)."""
    import torch

    model.eval().to(device)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model(**inputs, output_attentions=True, output_hidden_states=True)

    attentions = [a.squeeze(0).cpu().numpy() for a in out.attentions]
    hidden = [h.squeeze(0).cpu().numpy() for h in out.hidden_states]

    n_heads = model.config.n_head
    d_model = model.config.n_embd
    d_head = d_model // n_heads

    w_v_per_layer = _extract_wv_matrices(model, n_heads, d_model, d_head)

    return {
        "attention": attentions,
        "hidden_states": hidden,
        "tokens": tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]),
        "n_layers": model.config.n_layer,
        "n_heads": n_heads,
        "d_head": d_head,
        "d_model": d_model,
        "model_name": getattr(model.config, '_name_or_path', 'unknown'),
        "w_v": w_v_per_layer,
    }


def _extract_wv_matrices(model, n_heads, d_model, d_head):
    """Extract per-layer, per-head W_V matrices from GPT-2.

    GPT-2 concatenates Q, K, V projections into a single c_attn weight.
    We extract the V portion and reshape it per head to get (d_head, d_head)
    restriction maps suitable for sheaf cohomology.

    Each head's restriction map is its d_head x d_head sub-block of W_V,
    preserving the learned structure rather than orthogonalizing it away.

    Returns: list of dicts, one per layer, mapping head index -> (d_head, d_head) array
    """
    w_v_per_layer = []
    blocks = model.h if hasattr(model, 'h') else model.transformer.h

    for block in blocks:
        w = block.attn.c_attn.weight.detach().cpu().numpy()
        W_V_all = w[:, 2 * d_model:]
        W_V_heads = W_V_all.reshape(d_model, n_heads, d_head)

        head_maps = {}
        for h in range(n_heads):
            wv_h = W_V_heads[h * d_head:(h + 1) * d_head, h, :]
            head_maps[h] = wv_h

        w_v_per_layer.append(head_maps)
    return w_v_per_layer


def build_restriction_maps_from_wv(w_v_layer: Dict[int, np.ndarray],
                                   overlaps: Dict) -> Dict:
    """Build sheaf restriction maps from learned W_V matrices.

    For each edge (h, k) in the overlap graph, the restriction maps are:
        rho_{h -> (h,k)} = QR-orthogonalized W_V^(h)
        rho_{k -> (h,k)} = QR-orthogonalized W_V^(k)

    This gives the sheaf non-trivial structure derived from the model's
    learned value projections, rather than random maps.
    """
    maps = {}
    for (h, k) in overlaps:
        maps[(h, (h, k))] = w_v_layer[h]
        maps[(k, (h, k))] = w_v_layer[k]
    return maps


def bundle_layer_heads(model_data: Dict[str, Any], layer: int = -1) -> ExtractedBundle:
    """Build (head_outputs, attention) for a single layer from extracted GPT-2 data."""
    n_layers = int(model_data["n_layers"])
    if layer < 0:
        layer = n_layers + layer

    attn = np.asarray(model_data["attention"][layer])
    hidden = np.asarray(model_data["hidden_states"][layer + 1])

    H = int(model_data["n_heads"])
    d_head = int(model_data["d_head"])
    n = int(hidden.shape[0])

    head_outputs = hidden.reshape(n, H, d_head).transpose(1, 0, 2)

    restriction_maps = None
    if "w_v" in model_data and model_data["w_v"] is not None:
        from sheaf_transformers.metrics.cohomology import CohomologyComputer
        cc = CohomologyComputer(H, d_head, threshold=0.01)
        overlaps, _ = cc.compute_overlaps(attn)
        if overlaps:
            restriction_maps = build_restriction_maps_from_wv(
                model_data["w_v"][layer], overlaps)

    return ExtractedBundle(
        attention=attn,
        head_outputs=head_outputs,
        tokens=list(model_data.get("tokens", [])),
        meta={"layer": layer, "model_name": model_data.get("model_name", "")},
        restriction_maps=restriction_maps,
    )
