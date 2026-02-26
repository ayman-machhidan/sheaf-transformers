"""HuggingFace extraction helpers.

These functions are optional at runtime (depend on transformers + torch).
They are kept minimal so the core metrics can be used without HF.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ExtractedBundle:
    """A thin container for attention analysis inputs."""

    attention: Any
    head_outputs: Any
    tokens: List[str]
    meta: Dict[str, Any]


def extract_attention_from_gpt2(model_name: str = "gpt2", text: str = "Hello world.", device: str = "cpu") -> Dict[str, Any]:
    """Extract attention matrices and hidden states from GPT-2.

    Returns a dict compatible with sheaf_transformers.pipeline.SheafExperiment.
    """
    import numpy as np
    import torch
    from transformers import GPT2Tokenizer, GPT2Model

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2Model.from_pretrained(model_name, output_attentions=True)
    model.eval().to(device)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model(**inputs, output_attentions=True, output_hidden_states=True)

    attentions = [a.squeeze(0).cpu().numpy() for a in out.attentions]  # list[(H,n,n)]
    hidden = [h.squeeze(0).cpu().numpy() for h in out.hidden_states]   # list[(n,d_model)]

    n_heads = model.config.n_head
    d_model = model.config.n_embd
    d_head = d_model // n_heads

    return {
        "attention": attentions,
        "hidden_states": hidden,
        "tokens": tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]),
        "n_layers": model.config.n_layer,
        "n_heads": n_heads,
        "d_head": d_head,
        "d_model": d_model,
        "model_name": model_name,
    }


def bundle_layer_heads(model_data: Dict[str, Any], layer: int = -1) -> ExtractedBundle:
    """Build (head_outputs, attention) for a single layer from extracted GPT-2 data."""
    import numpy as np

    n_layers = int(model_data["n_layers"])
    if layer < 0:
        layer = n_layers + layer

    attn = np.asarray(model_data["attention"][layer])  # (H,n,n)
    hidden = np.asarray(model_data["hidden_states"][layer + 1])  # (n, d_model)

    H = int(model_data["n_heads"])
    d_head = int(model_data["d_head"])
    n = int(hidden.shape[0])

    head_outputs = hidden.reshape(n, H, d_head).transpose(1, 0, 2)

    return ExtractedBundle(
        attention=attn,
        head_outputs=head_outputs,
        tokens=list(model_data.get("tokens", [])),
        meta={"layer": layer, "model_name": model_data.get("model_name", "")},
    )
