"""Tests for the pipeline and CLI modules."""

import numpy as np
import pytest

from sheaf_transformers.pipeline import SheafExperiment
from sheaf_transformers.cli import build_parser


def _make_model_data(rng, n_layers=3, n_heads=4, d_head=4, n_tokens=8):
    """Create synthetic model data compatible with SheafExperiment."""
    d_model = n_heads * d_head
    attention = [
        rng.dirichlet(np.ones(n_tokens), size=(n_heads, n_tokens))
        for _ in range(n_layers)
    ]
    hidden_states = [
        rng.randn(n_tokens, d_model)
        for _ in range(n_layers + 1)
    ]
    return {
        "attention": attention,
        "hidden_states": hidden_states,
        "tokens": [f"tok_{i}" for i in range(n_tokens)],
        "n_layers": n_layers,
        "n_heads": n_heads,
        "d_head": d_head,
        "d_model": d_model,
    }


class TestSheafExperiment:

    def test_init(self, rng):
        """SheafExperiment initializes correctly."""
        data = _make_model_data(rng)
        exp = SheafExperiment(data)
        assert exp.n_layers == 3
        assert exp.n_heads == 4
        assert exp.d_head == 4

    def test_run_cohomology_all_layers(self, rng):
        """run_cohomology returns results for all layers."""
        data = _make_model_data(rng)
        exp = SheafExperiment(data)
        results = exp.run_cohomology()
        assert len(results) == 3
        for r in results:
            assert "H_coh" in r
            assert "q_obs" in r
            assert "layer" in r
            assert r["H_coh"] >= 0

    def test_run_cohomology_specific_layers(self, rng):
        """run_cohomology with specific layers."""
        data = _make_model_data(rng)
        exp = SheafExperiment(data)
        results = exp.run_cohomology(layers=[0, 2])
        assert len(results) == 2
        assert results[0]["layer"] == 0
        assert results[1]["layer"] == 2

    def test_run_cohomology_single_layer(self, rng):
        """run_cohomology with a single layer."""
        data = _make_model_data(rng)
        exp = SheafExperiment(data)
        results = exp.run_cohomology(layers=[1])
        assert len(results) == 1
        assert results[0]["layer"] == 1

    def test_summary_output(self, rng, capsys):
        """summary() prints formatted output."""
        data = _make_model_data(rng)
        exp = SheafExperiment(data)
        exp.summary(layers=[0])
        captured = capsys.readouterr()
        assert "SHEAF COHERENCE SUMMARY" in captured.out
        assert "Layer" in captured.out
        assert "H_coh" in captured.out

    def test_custom_threshold(self, rng):
        """Custom threshold affects overlap computation."""
        data = _make_model_data(rng)
        exp_low = SheafExperiment(data, threshold=0.01)
        exp_high = SheafExperiment(data, threshold=0.5)
        r_low = exp_low.run_cohomology(layers=[0])
        r_high = exp_high.run_cohomology(layers=[0])
        assert r_low[0]["n_overlaps"] >= r_high[0]["n_overlaps"]

    def test_reproducibility(self, rng):
        """Same data => same results."""
        data = _make_model_data(rng)
        exp = SheafExperiment(data)
        r1 = exp.run_cohomology(layers=[0])
        r2 = exp.run_cohomology(layers=[0])
        assert r1[0]["H_coh"] == r2[0]["H_coh"]


class TestCLI:

    def test_parser_builds(self):
        """CLI parser builds without errors."""
        p = build_parser()
        assert p is not None

    def test_coherence_subcommand_defaults(self):
        """coherence subcommand parses default args."""
        p = build_parser()
        args = p.parse_args(["coherence"])
        assert args.cmd == "coherence"
        assert args.model == "gpt2"
        assert args.threshold == 0.01
        assert args.device == "cpu"

    def test_coherence_custom_args(self):
        """coherence subcommand parses custom args."""
        p = build_parser()
        args = p.parse_args([
            "coherence",
            "--model", "gpt2-medium",
            "--text", "test input",
            "--threshold", "0.05",
            "--layer", "3",
        ])
        assert args.model == "gpt2-medium"
        assert args.text == "test input"
        assert args.threshold == 0.05
        assert args.layer == 3

    def test_missing_subcommand_raises(self):
        """Missing subcommand raises SystemExit."""
        p = build_parser()
        with pytest.raises(SystemExit):
            p.parse_args([])
