"""Run a shard of Cosmic Ray mutations, retaining every outcome and diff."""

from __future__ import annotations

import argparse
import dataclasses
import json
import shlex
import sys
from collections import Counter
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class Target:
    path: str
    tests: tuple[str, ...]


# Tests execute in fresh interpreters: a JAX backend is never forked. Naming
# files rather than an expression also makes an empty selection a failure.
TARGETS = {
    "transaction": Target("src/dew/training/transaction.py", (
        "tests/test_training_transactions.py", "tests/test_support_replay.py",
        "tests/test_optional_ema.py", "tests/test_training_dtypes.py")),
    "checkpoints": Target("src/dew/checkpoints", (
        "tests/test_training_transactions.py", "tests/test_optional_ema.py",
        "tests/test_posthoc.py", "tests/test_frozen_reference.py")),
    "process": Target("src/dew/diffusion/process.py", (
        "tests/test_guidance.py", "tests/test_shortcut.py", "tests/test_mean_flow.py")),
    "schedules": Target("src/dew/diffusion/schedules", (
        "tests/test_schedulers.py", "tests/test_flow_matching.py",
        "tests/test_native_diffusion_edges.py")),
    "attention": Target("src/dew/nn/attention.py", (
        "tests/test_causal_transformer.py", "tests/test_local_attention.py",
        "tests/test_sequence_parallel.py", "tests/test_attention_sinks.py")),
    "sources": Target("src/dew/data/sources", (
        "tests/test_hf_data.py", "tests/test_online_loader.py",
        "tests/test_text_data.py", "tests/test_tfds_read.py", "tests/test_data_av.py")),
}


def classify(result) -> str:
    """Count test failures as kills; report timeouts separately without failing CI.

    A timeout may be a mutated infinite loop, but it is not an observed test
    failure. Worker errors and missing runs cannot produce a valid score.
    """
    if result is None:
        return "pending"
    if result.worker_outcome != "normal":
        return "worker-error"
    if result.test_outcome == "incompetent":
        return "incompetent"
    if result.output == "timeout":
        return "timeout"
    if result.test_outcome == "killed":
        return "killed"
    if result.test_outcome == "survived":
        return "survived"
    return "worker-error"


def report(database) -> dict:
    """Keep untested, invalid and surviving mutations visible in the report."""
    mutations = []
    for work, result in (*database.completed_work_items,
                         *((work, None) for work in database.pending_work_items)):
        spec, = work.mutations
        mutations.append({
            "path": str(spec.module_path), "line": spec.start_pos[0],
            "operator": spec.operator_name, "occurrence": spec.occurrence,
            "outcome": classify(result),
            "diff": None if result is None else result.diff,
            "output": None if result is None else result.output,
        })
    counts = Counter(mutation["outcome"] for mutation in mutations)
    tested = counts["killed"] + counts["survived"]
    return {"counts": dict(counts), "total": len(mutations),
            "score": None if tested == 0 else counts["killed"] / tested,
            "mutations": sorted(mutations, key=lambda mutation: (
                mutation["path"], mutation["line"], mutation["operator"], mutation["occurrence"]))}


def run(target: Target, directory: Path, shard: int, shards: int, timeout: float) -> dict:
    """Partition deterministically; no operator or source line is excluded."""
    from cosmic_ray.commands import execute, init
    from cosmic_ray.config import ConfigDict
    from cosmic_ray.modules import find_modules
    from cosmic_ray.work_db import use_db
    from cosmic_ray.work_item import WorkItem

    command = shlex.join([sys.executable, "-m", "pytest", "-x", "-q", "--tb=short",
                          "-m", "not network", *target.tests])
    config = ConfigDict({"module-path": target.path, "timeout": timeout,
                         "test-command": command, "excluded-modules": [],
                         "distributor": {"name": "local"}})
    directory.mkdir(parents=True)
    with use_db(directory / "baseline.sqlite") as baseline:
        baseline.add_work_item(WorkItem(job_id="baseline", mutations=()))
        execute(baseline, config)
        _, result = next(baseline.results)
        if classify(result) != "survived":
            raise RuntimeError(f"unmutated tests failed: {classify(result)}\n{result.output}")
    with use_db(directory / "population.sqlite") as population:
        init(find_modules([Path(target.path)]), population, config.operators_config)
        work = sorted(population.work_items, key=lambda work: (
            str(work.mutations[0].module_path), work.mutations[0].start_pos,
            work.mutations[0].operator_name, work.mutations[0].occurrence))
    with use_db(directory / "session.sqlite") as session:
        session.add_work_items(work[shard::shards])
        try:
            execute(session, config)
        finally:
            results = report(session)
            results["population"] = len(work)
            results["shard"] = shard
            results["shards"] = shards
            (directory / "report.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def summarize(directory: Path) -> dict:
    """Combine disjoint shards; never average their percentages."""
    reports = [json.loads(path.read_text()) for path in sorted(directory.glob("**/report.json"))]
    if not reports:
        raise ValueError("no mutation reports were found")
    population, shards = reports[0]["population"], reports[0]["shards"]
    if any(report["population"] != population or report["shards"] != shards for report in reports):
        raise ValueError("reports describe different mutation populations")
    if sorted(report["shard"] for report in reports) != list(range(shards)):
        raise ValueError("the report is missing a shard or contains a duplicate shard")
    counts = Counter()
    for report in reports:
        counts.update(report["counts"])
    if sum(counts.values()) != population:
        raise ValueError("the shard totals do not cover the full mutation population")
    tested = counts["killed"] + counts["survived"]
    return {"counts": dict(counts), "total": population,
            "score": None if tested == 0 else counts["killed"] / tested}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=TARGETS, nargs="?")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summarize", type=Path)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180)
    arguments = parser.parse_args()
    if arguments.summarize is not None:
        print(json.dumps(summarize(arguments.summarize), indent=2))
        return
    if arguments.target is None or arguments.output is None:
        parser.error("target and --output are required when running mutations")
    if arguments.shards < 1 or not 0 <= arguments.shard < arguments.shards:
        parser.error("shard must be in [0, shards), and shards must be positive")
    if arguments.timeout <= 0:
        parser.error("timeout must be positive")
    results = run(TARGETS[arguments.target], arguments.output,
                  arguments.shard, arguments.shards, arguments.timeout)
    print(json.dumps({key: value for key, value in results.items() if key != "mutations"}, indent=2))
    if results["population"] == 0:
        raise SystemExit("no mutations were generated")
    if any(results["counts"].get(status, 0)
           for status in ("pending", "worker-error", "incompetent")):
        raise SystemExit("the mutation run was incomplete; inspect report.json")


if __name__ == "__main__":
    main()
