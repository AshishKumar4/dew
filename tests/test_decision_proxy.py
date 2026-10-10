"""The DI proxy allocator and group bootstrap, without importing the kit or calling a server."""

import importlib.util
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

PROXY = Path(__file__).parents[1] / "recipes" / "decision" / "proxy.py"


@pytest.fixture(scope="module")
def proxy():
    spec = importlib.util.spec_from_file_location("decision_proxy", PROXY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _spec() -> dict:
    return {"areas": [{"id": "knowledge", "benchmarks": [1, 2, 3, 4]},
                      {"id": "language", "benchmarks": [5, 6, 7, 8]},
                      {"id": "retrieval", "benchmarks": [9, 10]},
                      {"id": "tools", "benchmarks": [11]},
                      {"id": "arts", "benchmarks": [12, 13]}],
            "area_sizing": {"rule": "sqrt", "fixed": {"arts": 0.1}}, "gold": {"1": 1.2},
            "chance": {str(n): {"name": f"benchmark-{n}"} for n in range(1, 14)}}


def _rows(count: int = 1000, size: int = 1) -> list[dict]:
    return [{"_evaluation": {"catalog_id": number, "group_id": f"group-{group}",
                              "run_id": f"{number}:{group}:{variant}", "track": f"track-{variant}"},
             "questions": {"answer": {"type": "choice", "criteria": {"no": "No", "yes": "Yes"}}},
             "expected": {"answer": "yes" if group % 2 else "no"}}
            for number in range(1, 14) for group in range(count) for variant in range(size)]


def test_proxy_apportions_requests_using_the_specs_area_and_gold_weights(proxy):
    spec = _spec()
    selected = proxy.allocate(spec, _rows(), requests=600)
    counts = Counter(row["_evaluation"]["catalog_id"] for row in selected)
    weights = proxy.area_weights(spec)
    assert len(selected) == 600
    assert weights["arts"] == 0.1
    assert weights["knowledge"] / weights["tools"] == 2
    for area in spec["areas"]:
        assert sum(counts[number] for number in area["benchmarks"]) == pytest.approx(
            600 * weights[area["id"]], abs=1)
    assert counts[1] / counts[2] == pytest.approx(1.2, abs=0.03)
    assert counts[2] == pytest.approx(counts[3], abs=1)


def test_proxy_keeps_every_track_and_variant_in_a_whole_group(proxy):
    rows = _rows(count=100, size=3)
    selected = proxy.allocate(_spec(), rows, requests=257)
    full, subset = defaultdict(set), defaultdict(set)
    for entries, groups in ((rows, full), (selected, subset)):
        for row in entries:
            evaluation = row["_evaluation"]
            groups[(evaluation["catalog_id"], evaluation["group_id"])].add(evaluation["run_id"])
    assert len(selected) == 255
    assert subset and all(members == full[key] for key, members in subset.items())
    assert len(subset) * 3 == len(selected)


def test_proxy_floors_cover_every_banking_and_clinc_gold_class_including_oos(proxy):
    spec, rows = _spec(), _rows(count=48, size=2)
    spec["chance"]["9"]["name"], spec["chance"]["10"]["name"] = "BANKING77", "CLINC150"
    classes = {9: [f"bank-{index}" for index in range(8)],
               10: [f"intent-{index}" for index in range(11)] + ["oos"]}
    for row in rows:
        evaluation = row["_evaluation"]
        number = evaluation["catalog_id"]
        if number in classes:
            index = int(evaluation["group_id"].split("-")[-1])
            row["expected"]["answer"] = classes[number][index % len(classes[number])]
            row["questions"]["answer"]["criteria"] = dict.fromkeys(classes[number], "intent")
    selected = proxy.allocate(spec, rows, requests=80)
    for number, golds in classes.items():
        assert {row["expected"]["answer"] for row in selected
                if row["_evaluation"]["catalog_id"] == number} == set(golds)
    assert len(selected) == 80
    with pytest.raises(ValueError, match="mandatory requests"):
        proxy.allocate(spec, rows, requests=30)


def test_proxy_sampling_is_seeded_and_independent_of_file_order(proxy):
    rows = _rows(count=50)
    first = proxy.allocate(_spec(), rows, requests=100, seed=17)
    assert first == proxy.allocate(_spec(), list(reversed(rows)), requests=100, seed=17)
    assert first != proxy.allocate(_spec(), rows, requests=100, seed=18)
    with pytest.raises(ValueError, match="missing index benchmarks"):
        proxy.allocate(_spec(), rows[:50], requests=100)
    with pytest.raises(ValueError, match="duplicate request"):
        proxy.allocate(_spec(), [*rows, rows[0]])
    with pytest.raises(ValueError, match="positive"):
        proxy.allocate(_spec(), rows, requests=0)


def test_bootstrap_draws_whole_groups_and_keeps_repeated_draws_distinct(proxy):
    rows = _rows(count=2, size=2)[:4]
    extra = {**rows[-1], "_evaluation": {**rows[-1]["_evaluation"], "run_id": "1:1:extra"}}
    rows.append(extra)
    results = {row["_evaluation"]["run_id"]: {"status": "ok", "score": index}
               for index, row in enumerate(rows)}
    sampled, responses = proxy.resample(rows, results, random.Random(0))
    assert len(sampled) == len(responses) == 6
    assert len({row["_evaluation"]["group_id"] for row in sampled}) == 2
    for members in proxy.groups(sampled)[1]:
        assert len(members) == 3
        assert {responses[row["_evaluation"]["run_id"]]["score"] for row in members} == {2, 3, 4}


def test_bootstrap_reports_the_central_eighty_percent_interval(proxy):
    rows = _rows(count=2)[:2]
    results = {row["_evaluation"]["run_id"]: {"status": "ok"} for row in rows}
    assert proxy.interval(rows, results, lambda rows, responses: 1.0, samples=10) == [1.0, 1.0]
    sequence = iter(range(10))
    assert proxy.interval(rows, results, lambda rows, responses: float(next(sequence)),
                          samples=10) == [0.9, 8.1]
    with pytest.raises(ValueError, match="positive"):
        proxy.interval(rows, results, lambda rows, responses: 1.0, samples=0)
