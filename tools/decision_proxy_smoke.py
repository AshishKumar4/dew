"""Exercise the DI proxy CLI on a pinned kit and synthetic perfect and random full-run results.

Run on CI, with the separately cloned kit at 9eb2dbe. The suite uses the
kit's admitted request ids where 0.2.1 has frozen subsets, and its native
retrieval, F1, forecast and multi-track scoring shapes. No model is run.
"""

import argparse
import gzip
import hashlib
import json
import math
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _suite(kit: Path, directory: Path) -> list[dict]:
    sys.path.insert(0, str(kit))
    from decision_index import constants, editions
    from decision_index.scoring import added, index02, report
    from decision_index.suite.build.layout import NAMES

    definition = index02.spec("0.2.1")
    admitted = editions.scoring_subsets("0.2.1")
    rows = []
    for area in definition["areas"]:
        for number in area["benchmarks"]:
            ids = sorted(admitted[number]) if number in admitted else []
            tracks = NAMES[number] if number == 30 else [constants.HEADLINE.get(number, "synthetic")]
            for group in range(100):
                for variant, track in enumerate(tracks):
                    gold = "yes" if group % 2 else "no"
                    question = {"type": "choice", "instructions": "Synthetic choice", "criteria": {
                        "no": "No", "yes": "Yes"}}
                    key = next(iter(added.FIELDS.get(number, ("answer",))))
                    questions, expected, scoring = {key: question}, {key: gold}, {"accepted": [gold]}
                    if number in report.RANKING_BENCHMARKS:
                        questions = {"relevant": question, "irrelevant": question}
                        expected = {"relevant": "yes", "irrelevant": "no"}
                        scoring = {"type": "retrieval_ranking", "field_to_document": {
                            "relevant": "doc-1", "irrelevant": "doc-0"}, "qrels": {"doc-1": 1, "doc-0": 0},
                            "scorable_ids": ["doc-0", "doc-1"], "retrieved_ids": ["doc-0", "doc-1"],
                            "candidate_recall": 1.0}
                    elif str(number) in definition["loss_rules"]:
                        scoring = {"type": "forecast_probability"}
                    elif number in added.F1_POSITIVE:
                        questions = {key: {"type": "noul", "instructions": "Synthetic binary choice"}}
                        expected = {key: bool(group % 2)}
                    run_id = (ids[group * len(tracks) + variant] if ids
                              else f"synthetic:{number}:{group}:{variant}")
                    rows.append({"state": f"synthetic item {group}", "questions": questions,
                                 "expected": expected, "scoring": scoring,
                                 "metadata": {"variant": str(variant), "song_id": group % 3,
                                              "user_id": group % 3, "gold_class": gold},
                                 "_evaluation": {"catalog_id": number,
                                                  "dataset": definition["chance"][str(number)]["name"],
                                                  "group_id": f"synthetic:{number}:{group}",
                                                  "run_id": run_id, "track": track}})
    directory.mkdir(parents=True)
    added_ids = {int(number) for number in definition["added"]}
    for filename, extra in ((editions.ROWS_FILE, False), (editions.ADDED_FILE, True)):
        with gzip.open(directory / filename, "wt") as file:
            for row in rows:
                if (row["_evaluation"]["catalog_id"] in added_ids) == extra:
                    file.write(json.dumps(row) + "\n")
    (directory / editions.MANIFEST_FILE).write_text(json.dumps({"edition": "0.2.1"}))
    return rows


def _results(rows: list[dict], path: Path, *, perfect: bool) -> None:
    rng = random.Random(73)
    with path.open("w") as file:
        for row in rows:
            answers = {}
            for key, question in row["questions"].items():
                gold = row["expected"][key]
                correct = perfect or rng.random() < 0.65
                if question["type"] == "noul":
                    answers[key] = {"type": "noul", "noul": float(gold if correct else not gold)}
                else:
                    choice = (gold if correct else
                              next(option for option in question["criteria"] if option != gold))
                    answers[key] = {"type": "choice", "choice": choice,
                                    "probabilities": {option: float(option == choice)
                                                      for option in question["criteria"]}}
            file.write(json.dumps({**row["_evaluation"],
                                   "status": "ok" if perfect or rng.random() > 0.1 else "unsupported",
                                   "response": {"answers": answers}, "total_wall_ms": 1.0}) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    options = parser.parse_args()
    suite_dir = options.out / "suite"
    rows = _suite(options.kit, suite_dir)
    from decision_index.pipeline import score_run
    from decision_index.suite.io import Suite

    suite = Suite(suite_dir, edition="0.2.1")
    report = {}
    for name in ("perfect", "random"):
        results = options.out / f"{name}.jsonl"
        _results(rows, results, perfect=name == "perfect")
        output = options.out / f"{name}-proxy.json"
        subprocess.run([sys.executable, str(ROOT / "recipes/decision/benchmark.py"), "di-proxy", "--kit",
                        str(options.kit), "--suite-dir", str(suite_dir), "--results", str(results), "--out",
                        str(output), "--bootstrap", "1000"], check=True)
        proxy = json.loads(output.read_text())
        full = score_run(suite, results, "synthetic", options.out / f"{name}-full")["decision_index"]
        assert set(proxy) == {"di", "interval", "areas", "benchmarks", "requests",
                              "subset_sha256", "seed", "kit"}
        assert proxy["requests"] == 2500 and math.isfinite(proxy["di"])
        digest = hashlib.sha256(output.with_suffix(".rows.jsonl").read_bytes()).hexdigest()
        assert proxy["subset_sha256"] == digest
        if name == "perfect":
            assert abs(proxy["di"] - full) < 0.005 and full == 100.0
        report[name] = {"proxy_di": proxy["di"], "full_di": full, "interval": proxy["interval"],
                        "full_in_interval": proxy["interval"][0] <= full <= proxy["interval"][1],
                        "requests": proxy["requests"], "subset_sha256": proxy["subset_sha256"],
                        "kit": proxy["kit"]}
    print("SMOKE " + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
