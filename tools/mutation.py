"""Run a shard of Cosmic Ray mutations, retaining every outcome and diff."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shlex
import sys
import time
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
        "tests/test_posthoc.py", "tests/test_frozen_reference.py",
        "tests/test_trainer.py::test_restore_preserves_the_optimizer_state_the_ema_and_the_key",
        "tests/test_trainer.py::test_the_ema_comes_back_bit_for_bit_however_it_is_read",
        "tests/test_trainer.py::test_an_ema_held_in_lists_comes_back_bit_for_bit",
        "tests/test_trainer.py::test_a_checkpoint_that_stores_the_ema_as_itself_still_restores",
        "tests/test_trainer.py::test_a_global_position_is_read_by_any_process_count")),
    "process": Target("src/dew/diffusion/process.py", (
        "tests/test_diffusion_process.py", "tests/test_guidance.py",
        "tests/test_shortcut.py", "tests/test_mean_flow.py")),
    "schedules": Target("src/dew/diffusion/schedules", (
        "tests/test_schedule_contracts.py", "tests/test_schedulers.py", "tests/test_flow_matching.py",
        "tests/test_native_diffusion_edges.py")),
    "attention": Target("src/dew/nn/attention.py", (
        "tests/test_causal_transformer.py", "tests/test_local_attention.py",
        "tests/test_sequence_parallel.py", "tests/test_attention_sinks.py")),
    "sources": Target("src/dew/data/sources", (
        "tests/test_hf_options_contract.py", "tests/test_hf_data.py", "tests/test_online_loader.py",
        "tests/test_text_data.py", "tests/test_tfds_read.py", "tests/test_data_av.py")),
}

# A file batch runs the proofs of that file's behavior, not unrelated I/O or
# reference-model suites from neighboring source modules. Directory batches
# retain the complete target suite above.
FILE_TESTS = {
    "src/dew/data/sources/hf.py": ("tests/test_hf_options_contract.py", "tests/test_hf_data.py"),
    "src/dew/diffusion/schedules/common.py": ("tests/test_schedule_contracts.py", "tests/test_schedulers.py"),
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


def score_counts(counts) -> dict:
    """Distinguish an observed partial fraction from a completed batch's score."""
    tested = counts.get("killed", 0) + counts.get("survived", 0)
    observed = None if tested == 0 else counts.get("killed", 0) / tested
    complete = not any(counts.get(status, 0) for status in ("pending", "worker-error", "incompetent"))
    return {"counts": dict(counts), "total": sum(counts.values()),
            "score": observed if complete else None, "observed_score": observed, "complete": complete}


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
    paths = sorted({mutation["path"] for mutation in mutations})
    files = {path: score_counts(Counter(mutation["outcome"] for mutation in mutations
                                       if mutation["path"] == path)) for path in paths}
    return {**score_counts(counts), "files": files,
            "mutations": sorted(mutations, key=lambda mutation: (
                mutation["path"], mutation["line"], mutation["operator"], mutation["occurrence"]))}


def source_digest(paths: list[Path]) -> str:
    """Bind a resumed session to exactly the source population it mutated."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path).encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def selected_target(target: Target, filename: Path | None) -> Target:
    """Choose one source file within a target, without excluding any of its lines."""
    if filename is None:
        return target
    path, boundary = filename.resolve(), Path(target.path).resolve()
    within = path == boundary if boundary.is_file() else path.is_relative_to(boundary)
    if not within or path.suffix != ".py" or not path.is_file():
        raise ValueError(f"{filename} is not a Python source file within {target.path}")
    relative = str(path.relative_to(Path.cwd()))
    return dataclasses.replace(target, path=relative, tests=FILE_TESTS.get(relative, target.tests))


# A mutant runs the same suite as the unmutated baseline, so it gets twice the baseline's time before it
# counts as a timeout: a fixed per-mutant limit the suite has outgrown times out the baseline itself.
BASELINE_MULTIPLE = 2.0


def mutant_timeout(timeout: float, baseline_seconds: float) -> float:
    """The limit each mutant runs under: `timeout`, or twice the unmutated suite's time if longer."""
    return max(timeout, BASELINE_MULTIPLE * baseline_seconds)


def run(target: Target, directory: Path, shard: int, shards: int, timeout: float, *,
        resume: bool = False, seconds: float | None = None) -> dict:
    """Partition deterministically; retain a bounded run's pending work for resume.

    The unmutated suite runs once under the whole budget (`seconds`, else ten
    times `timeout`) and sets each mutant's limit (`mutant_timeout`)."""
    from cosmic_ray.commands import init
    from cosmic_ray.modules import find_modules
    from cosmic_ray.mutating import mutate_and_test
    from cosmic_ray.work_db import use_db
    from cosmic_ray.work_item import WorkItem

    command = shlex.join([sys.executable, "-m", "pytest", "-x", "-q", "--tb=short",
                          "-m", "not network", *target.tests])
    modules = list(find_modules([Path(target.path)]))
    identity = {"source_digest": source_digest(modules), "path": target.path,
                "tests_digest": source_digest([Path(test.split("::", 1)[0]) for test in target.tests]
                                               + [Path("tests/conftest.py")]),
                "tests": target.tests, "shard": shard, "shards": shards}
    manifest = directory / "manifest.json"
    if resume:
        if json.loads(manifest.read_text()) != json.loads(json.dumps(identity)):
            raise ValueError("the session's source, tests or shard selection changed; start a new session")
    else:
        directory.mkdir(parents=True)
        manifest.write_text(json.dumps(identity, indent=2) + "\n")
    started = time.monotonic()
    with use_db(directory / "baseline.sqlite") as baseline:
        baseline.clear()
        baseline.add_work_item(WorkItem(job_id="baseline", mutations=()))
        result = mutate_and_test((), command, seconds if seconds is not None else 10 * timeout)
        baseline.set_result("baseline", result)
        if classify(result) != "survived":
            raise RuntimeError(f"unmutated tests failed: {classify(result)}\n{result.output}")
    baseline_seconds = time.monotonic() - started
    timeout = mutant_timeout(timeout, baseline_seconds)
    with use_db(directory / "population.sqlite") as population:
        if not resume:
            init(modules, population, {})
        work = sorted(population.work_items, key=lambda work: (
            str(work.mutations[0].module_path), work.mutations[0].start_pos,
            work.mutations[0].operator_name, work.mutations[0].occurrence))
    with use_db(directory / "session.sqlite") as session:
        if not resume:
            session.add_work_items(work[shard::shards])
        try:
            for pending in session.pending_work_items:
                if seconds is not None and time.monotonic() - started + timeout >= seconds:
                    break
                result = mutate_and_test(pending.mutations, command, timeout)
                session.set_result(pending.job_id, result)
        finally:
            results = report(session)
            results.update(identity)
            results.update(population=len(work), baseline_seconds=baseline_seconds, mutant_timeout=timeout)
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
    for field in ("path", "source_digest", "tests_digest"):
        if any(report.get(field) != reports[0].get(field) for report in reports):
            raise ValueError(f"reports disagree about {field}")
    counts = Counter()
    for report in reports:
        counts.update(report["counts"])
    if sum(counts.values()) != population:
        raise ValueError("the shard totals do not cover the full mutation population")
    paths = sorted({path for report in reports for path in report.get("files", {})})
    files = {}
    for path in paths:
        outcomes = Counter()
        for report in reports:
            outcomes.update(report.get("files", {}).get(path, {}).get("counts", {}))
        files[path] = score_counts(outcomes)
    return {**score_counts(counts), "files": files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=TARGETS, nargs="?")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summarize", type=Path)
    parser.add_argument("--file", type=Path, help="one file within the chosen module's source directory")
    parser.add_argument("--resume", action="store_true", help="continue the existing SQLite session")
    parser.add_argument("--seconds", type=float, help="stop admitting work before this budget expires")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180,
                        help="each mutant's least limit; twice the unmutated suite's time where longer")
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
    if arguments.seconds is not None and arguments.seconds <= arguments.timeout:
        parser.error("seconds must exceed the per-mutation timeout")
    target = selected_target(TARGETS[arguments.target], arguments.file)
    results = run(target, arguments.output, arguments.shard, arguments.shards, arguments.timeout,
                  resume=arguments.resume, seconds=arguments.seconds)
    print(json.dumps({key: value for key, value in results.items() if key != "mutations"}, indent=2))
    if results["population"] == 0:
        raise SystemExit("no mutations were generated")
    if any(results["counts"].get(status, 0)
           for status in ("pending", "worker-error", "incompetent")):
        raise SystemExit("the mutation run was incomplete; inspect report.json")


if __name__ == "__main__":
    main()
