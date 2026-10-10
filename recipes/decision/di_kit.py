"""Thin glue to the separately checked-out Decision Index 0.2.1 kit.

No kit is installed as a Dew dependency. The caller supplies its pinned
checkout and a built suite; every benchmark's native scorer, chance
correction, unsupported-request coverage and aggregation remain the kit's.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from proxy import allocate, interval

REVISION = "9eb2dbe"


class Subset:
    """A suite view exposing only selected rows to the kit's panel scorer."""

    def __init__(self, rows: list[dict]):
        self.selected = rows

    def rows(self, apply_exclusions: bool = True):
        yield from self.selected


def score(rows: list[dict], results: dict, spec: dict) -> dict:
    """Score only these rows using the kit's native benchmark metrics and its own aggregate."""
    from decision_index.scoring import added, index02
    from decision_index.scoring.index import score_panel
    from decision_index.scoring.report import benchmark_summary

    subset = Subset(rows)
    added_ids = {int(number) for number in spec["added"]}
    base = [row for row in rows if row["_evaluation"]["catalog_id"] not in added_ids]
    metrics = {int(number): (metric["name"], metric["key"])
               for number, metric in spec.get("metrics", {}).items()}
    summary = benchmark_summary(subset, results, "di-proxy", rows=base, metrics=metrics)
    native = {entry["catalog_id"]: entry for entry in summary["benchmarks"]}
    for number in added_ids:
        extra = [row for row in rows if row["_evaluation"]["catalog_id"] == number]
        if extra:
            native[number] = added.report(number, extra, results)
    tracks = score_panel(subset, results)
    values = {number: index02.benchmark_value(number, spec, tracks.get(number), native.get(number))
              for area in spec["areas"] for number in area["benchmarks"]}
    scores, areas = index02.aggregate(values, spec)
    return {"di": scores[spec["headline"]], "areas": areas,
            "benchmarks": {str(number): value for number, value in sorted(values.items())}}


def run(kit: Path, suite_dir: Path, out: Path, *, url: str | None = None, results: Path | None = None,
        requests: int = 2500, seed: int = 0, bootstrap: int = 1000) -> dict:
    """Write and score a weighted whole-group DI proxy, with a group-bootstrap 80% interval.

    Supply either an HTTP server URL, which the kit runs on the subset, or
    a full run's results file, from which only selected request ids are
    read. The adjacent `<out stem>.rows.jsonl` records the exact subset; its
    bytes' SHA-256 and the kit's full commit identify the measurement.
    """
    if (url is None) == (results is None):
        raise ValueError("di-proxy needs exactly one of --url and --results")
    if bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    kit = kit.resolve()
    revision = subprocess.check_output(["git", "-C", str(kit), "rev-parse", "HEAD"], text=True).strip()
    if not revision.startswith(REVISION):
        raise ValueError(f"di-proxy needs kit {REVISION}, got {revision}")
    sys.path.insert(0, str(kit))
    from decision_index.scoring.index02 import spec
    from decision_index.scoring.report import load_results
    from decision_index.suite.io import Suite

    definition = spec("0.2.1")
    subset = allocate(definition, Suite(suite_dir, edition="0.2.1").rows(apply_exclusions=True),
                      requests=requests, seed=seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows_path = out.with_suffix(".rows.jsonl")
    rows_path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                                 for row in subset), encoding="utf-8")
    if url is not None:
        directory = out.with_suffix(".run")
        environment = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(kit),
                                                                               os.environ.get("PYTHONPATH")]))}
        subprocess.run([sys.executable, "-m", "decision_index", "run", "--engine", "http", "--option",
                        f"base_url={url}", "--rows", str(rows_path), "--out", str(directory), "--fresh"],
                       env=environment, check=True)
        results = directory / "results.jsonl"
    found = load_results(results)
    run_ids = {row["_evaluation"]["run_id"] for row in subset}
    if missing := run_ids - found.keys():
        raise ValueError(f"results are missing {len(missing)} subset requests; supply a completed full run")
    selected = {run_id: found[run_id] for run_id in run_ids}
    scores = score(subset, selected, definition)
    bounds = interval(subset, selected, lambda rows, responses: score(rows, responses, definition)["di"],
                      samples=bootstrap, seed=seed)
    scores.update(interval=bounds, requests=len(subset),
                  subset_sha256=hashlib.sha256(rows_path.read_bytes()).hexdigest(), seed=seed, kit=revision)
    return scores
