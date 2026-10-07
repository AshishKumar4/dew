"""Score a decision model served at `/v1/systemone` on typed-decisions and on Laya's application battery.

    python examples/serve_decisions.py --run runs/encoder --whole --port 8000 &
    python recipes/decision/benchmark.py typed-decisions --url http://127.0.0.1:8000 --out td.json
    python recipes/decision/benchmark.py battery --url http://127.0.0.1:8000 --laya laya-src \\
        --out battery.json

Both talk to the model through its wire format, as their published rows were
scored, so any `/v1/systemone` server can be measured the same way.
Decision Index has its own kit and engine (`python -m decision_index pipeline
--engine http`), which the same server answers.

typed-decisions (LocalLLaMA/typed-decisions, test split, 400 cases): each
case is one request with its five questions, as the dataset's card scores
them. Accuracy is the most likely option against the gold label, as the
published Jev-and-Laya harness counts it (pavanjava/jev_and_laya_benchmarking
@ebeddf4); KL is KL(gold ‖ model) and Brier the squared distance summed over
the options, both per decision, which reproduce the card's Uniform row
exactly (0.444 and 0.238; `uniform`); ECE is Dew's, 15 bins over the top
probability, which the card says submitters compute differently.

Open-Jev's test and OOD splits (ZefanCai/Open-Jev, release-v2): the same
metrics against each question's target distribution.

Laya's battery (NandhaKishorM/laya @a4a8921, research/scripts/bench_apps.py):
its ten tasks are built by its own `build()`, 400 cases each from seed 13,
and scored by its own `metrics`, from the probabilities the server answers.
"""

import argparse
import itertools
import json
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

TYPED_DECISIONS = ("LocalLLaMA/typed-decisions", "d0e2f0c42fef86cc15d1688d25a19f5ba7c85b18")
LAYA_COMMIT = "a4a8921afebfd852bba0000475cfb6ab737a124c"

type Engine = Callable[[Mapping[str, object]], Mapping[str, object]]
"""A Jev request body to its response body."""


def http(url: str, *, timeout: float = 600.0) -> Engine:
    """An engine posting to `url`'s `/v1/systemone`.

    A request the server refuses for its capacity (a 422, as a server that
    answers only whole requests gives) comes back with no answers, which the
    scorers count as refused and score as an even spread, as Decision Index
    counts an unanswered request as wrong.
    """
    import httpx

    client = httpx.Client(base_url=url, timeout=timeout)

    def answer(request: Mapping[str, object]) -> Mapping[str, object]:
        response = client.post("/v1/systemone", json=dict(request))
        if response.status_code == 422:
            return {"answers": {}, "refused": response.json().get("error")}
        response.raise_for_status()
        return response.json()

    return answer


def _found(answers: Mapping[str, object], name: str, options: list[str]) -> np.ndarray:
    """An answer's distribution, or an even spread over `options` for a question left unanswered."""
    return (distribution(answers[name], options) if name in answers
            else np.full(len(options), 1.0 / len(options)))


def distribution(answer: Mapping[str, object], options: list[str]) -> np.ndarray:
    """An answer's probabilities in `options`' order; a noul's are false, then true."""
    if answer["type"] == "noul":
        true = float(answer["noul"])
        return np.array([1.0 - true, true])
    probabilities = answer["probabilities"]
    return np.array([float(probabilities[option]) for option in options])


def ece(confidence: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error over `bins` equal bins of the top probability."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for low, high in itertools.pairwise(edges):
        inside = (confidence > low) & (confidence <= high)
        if inside.any():
            error += inside.mean() * abs(confidence[inside].mean() - correct[inside].mean())
    return float(error)


def typed_decisions(engine: Engine) -> dict:
    """The test split's accuracy, KL from gold, Brier and ECE, overall and per question type."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(TYPED_DECISIONS[0], "all/test-00000-of-00001.parquet", repo_type="dataset",
                           revision=TYPED_DECISIONS[1])
    rows = defaultdict(list)
    started, refused = time.time(), 0
    for case in pq.read_table(path).to_pylist():
        questions, gold = json.loads(case["questions"]), json.loads(case["gold"])
        answers = engine({"state": json.loads(case["state"]), "questions": questions})["answers"]
        refused += not answers
        for name, expected in gold.items():
            options = list(expected["probabilities"])
            truth = np.array([float(expected["probabilities"][option]) for option in options])
            found = _found(answers, name, options)
            rows[expected["type"]].append((truth, found, options.index(expected["label"])))
    scores = {kind: _scored(entries) for kind, entries in rows.items()}
    scores["all"] = _scored([entry for entries in rows.values() for entry in entries])
    scores["refused_requests"] = refused
    scores["seconds"] = time.time() - started
    return scores


def _scored(entries: list[tuple[np.ndarray, np.ndarray, int]]) -> dict:
    top = np.array([int(np.argmax(found)) for _, found, _ in entries])
    label = np.array([right for _, _, right in entries])
    confidence = np.array([float(np.max(found)) for _, found, _ in entries])
    kl = [float(np.sum(truth * (np.log(np.clip(truth, 1e-12, 1)) - np.log(np.clip(found, 1e-12, 1)))))
          for truth, found, _ in entries]
    brier = [float(np.sum((found - truth) ** 2)) for truth, found, _ in entries]
    correct = (top == label).astype(float)
    return {"decisions": len(entries), "accuracy": float(correct.mean()), "kl_from_gold": float(np.mean(kl)),
            "brier": float(np.mean(brier)), "ece": ece(confidence, correct)}


def open_jev(engine: Engine, split: str, limit: int | None = None) -> dict:
    """Open-Jev's `test` or `ood` split, each example's questions asked in one request, against its targets.

    The examples are grouped as the recipes train on them (`sources.OpenJev`);
    the same metrics as typed-decisions, the gold label the target's most
    likely option.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from sources import OpenJev

    rows = defaultdict(list)
    started, refused = time.time(), 0
    for example in OpenJev().read(split)[:limit]:
        request = {"state": example.state,
                   "questions": {name: question.wire() for name, question in example.questions.items()}}
        answers = engine(request)["answers"]
        refused += not answers
        for name, question in example.questions.items():
            truth = example.distribution(name)
            found = _found(answers, name, list(question.options))
            rows[question.kind].append((truth, found, int(np.argmax(truth))))
    scores = {kind: _scored(entries) for kind, entries in rows.items()}
    scores["all"] = _scored([entry for entries in rows.values() for entry in entries])
    scores["refused_requests"] = refused
    scores["seconds"] = time.time() - started
    return scores


def uniform(request: Mapping[str, object]) -> Mapping[str, object]:
    """The card's Uniform reference: the same probability on every option."""
    answers = {}
    for name, question in dict(request["questions"]).items():
        if question["type"] == "noul":
            answers[name] = {"type": "noul", "noul": 0.5}
            continue
        criteria = question["criteria"]
        options = (list(criteria) if isinstance(criteria, dict)
                   else [str(level) for level in range(len(criteria))])
        answers[name] = {"type": question["type"],
                         "probabilities": dict.fromkeys(options, 1.0 / len(options))}
    return {"answers": answers}


def battery(engine: Engine, laya: Path) -> dict:
    """Laya's ten application tasks, built and scored by Laya's own code.

    `laya` is a checkout of NandhaKishorM/laya at LAYA_COMMIT.
    """
    scripts = laya / "research" / "scripts"
    sys.path[:0] = [str(laya), str(scripts)]
    import bench_apps
    import bench_local

    bench_apps.build()
    scores = {}
    for name, suite in bench_apps.SUITES.items():
        started, rows = time.time(), []
        refused = 0
        for (state, questions), gold in zip(suite["cases"], suite["gold"], strict=True):
            answers = engine({"state": state, "questions": questions})["answers"]
            refused += not answers
            (question_name, question), = questions.items()
            noul = question["type"] == "noul"
            options = ["false", "true"] if noul else list(question["criteria"])
            rows.append((gold, list(_found(answers, question_name, options))))
        scores[name] = {**bench_local.metrics(rows), "refused_requests": refused,
                        "seconds": time.time() - started}
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    suites = ("typed-decisions", "battery", "open-jev-test", "open-jev-ood", "uniform")
    parser.add_argument("suite", choices=suites)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--laya", type=Path, help="a checkout of NandhaKishorM/laya at LAYA_COMMIT")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="ask only the first this many Open-Jev examples")
    options = parser.parse_args()
    if options.suite == "uniform":
        scores = typed_decisions(uniform)
    elif options.suite == "typed-decisions":
        scores = typed_decisions(http(options.url))
    elif options.suite.startswith("open-jev"):
        scores = open_jev(http(options.url), options.suite.removeprefix("open-jev-"), options.limit)
    else:
        scores = battery(http(options.url), options.laya)
    options.out.write_text(json.dumps(scores, indent=1) + "\n")
    print(json.dumps(scores))


if __name__ == "__main__":
    main()
