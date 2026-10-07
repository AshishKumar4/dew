"""recipes/decision/benchmark.py: the typed-decisions scorer held to the dataset card's own reference row."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
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


def test_an_answer_reads_as_its_distribution_in_the_options_order(benchmark):
    choice = {"type": "choice", "choice": "b", "probabilities": {"a": 0.25, "b": 0.75}}
    np.testing.assert_array_equal(benchmark.distribution(choice, ["b", "a"]), [0.75, 0.25])
    np.testing.assert_array_equal(benchmark.distribution({"type": "noul", "noul": 0.9}, ["false", "true"]),
                                  [pytest.approx(0.1), 0.9])
    # Half the answers right at confidence 0.5, all right at 1.0: no calibration error.
    assert benchmark.ece(np.array([0.5, 0.5, 1.0]), np.array([1.0, 0.0, 1.0])) == 0.0


@pytest.mark.network
def test_the_scorer_reproduces_the_cards_uniform_row(benchmark):
    """The card's Uniform reference row reports KL from gold 0.444 and Brier
    0.238 over the 2,000 test decisions; the scorer gives the same for the
    same answers, so it measures what the card's rows measure."""
    scores = benchmark.typed_decisions(benchmark.uniform)["all"]
    assert scores["decisions"] == 2000
    assert round(scores["kl_from_gold"], 3) == 0.444
    assert round(scores["brier"], 3) == 0.238
