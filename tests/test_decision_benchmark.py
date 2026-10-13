"""recipes/decision/benchmark.py: the typed-decisions scorer held to the dataset card's own reference row."""

import importlib.util
import sys
from pathlib import Path

import pytest

BENCHMARK = Path(__file__).parents[1] / "recipes" / "decision" / "benchmark.py"


@pytest.fixture(scope="module")
def benchmark():
    spec = importlib.util.spec_from_file_location("decision_benchmark", BENCHMARK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.network
def test_the_scorer_reproduces_the_cards_uniform_row(benchmark):
    """The card's Uniform reference row reports KL from gold 0.444 and Brier
    0.238 over the 2,000 test decisions; the scorer gives the same for the
    same answers, so it measures what the card's rows measure."""
    scores = benchmark.typed_decisions(benchmark.uniform)["all"]
    assert scores["decisions"] == 2000
    assert round(scores["kl_from_gold"], 3) == 0.444
    assert round(scores["brier"], 3) == 0.238
