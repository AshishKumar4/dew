"""Recording a reference over seeds, for check 1's `indexed_recording` (tests/reference_error.py).

A reference that cannot run in the test (a CUDA-only library, a fetched source) is recorded once, over
the seeds check 1 draws: `record_case` plans the count on a proxy that can run (Dew's own XLA path),
records (1 + REDRAW) times it, confirms the count on the recorded reference itself with `seeds_for`,
records the confirmed count's whole redraw and returns the `<site>.input.*`, `<site>.reference.*`,
`<site>.truth.*` and `<site>.seeds` arrays to save beside its `provenance` with `save`, which commits a
recording of at most 1 MiB and sends a larger one to the fetched store (tools/fetched_fixtures.py). The
proxy only plans: the declared count is the recorded reference's own. (tools/recorded_draws.py is
another thing: it records the random draws a reference generator's library makes, so a second
precision reads the same samples.)
"""

import json
import sys
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetched_fixtures import save
from reference_error import REDRAW, as_given, input_digest, leaf_summary, seeds_for, truth_sketch

__all__ = ["provenance", "record_case", "record_reference", "save"]


def provenance(reference: str, revision: str, device: str, command: str, **details) -> np.ndarray:
    """A recording's reference, its revision, the device and the public command that reproduces it,
    as a JSON string array to save beside the recorded arrays."""
    return np.asarray(json.dumps({"reference": reference, "revision": revision, "device": device,
                                  "command": command, **details}))


def record_reference(label: str, draw, evaluate, proxy, *, package: str, repo: str, tool: str, out: Path,
                     description: str | None = None, argv: list[str] | None = None, **constants) -> Path:
    """A parity tool's whole command line: `label`'s recording over its seeds (`record_case`), the release
    of `package` (`repo`'s) pinned (`reference_releases.require`) and run on one CPU thread, saved to
    `out` or the `--out` given, with `constants` beside it and the provenance `tool` reproduces it by
    (`save`). Returns the file written."""
    import argparse
    import importlib

    import torch
    from reference_releases import require

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--out", type=Path, default=out)
    out = parser.parse_args(argv).out
    require(package)
    torch.set_num_threads(1)
    version = importlib.import_module(package).__version__
    recorded = record_case(label, draw, evaluate, proxy)
    written = save(out, {**recorded, **{name: np.asarray(value) for name, value in constants.items()},
                         "provenance": provenance(package, version, "cpu", f"python {tool}",
                                                  torch=torch.__version__)},
                   reference=f"{repo} {version}", license="Apache-2.0")
    print(f"{written}: {int(recorded[f'{label}.seeds'])} seeds, {package} {version}, "
          f"torch {torch.__version__}")
    return written


def record_case(label: str, draw, evaluate, proxy, *, truth_proxy=None, store_inputs: bool = True,
                summary: bool = False) -> dict[str, np.ndarray]:
    """`label`'s recording: `draw(seed)` gives a seed's inputs (a dict of arrays), `evaluate(inputs)` the
    reference's outputs and their float64 truth (dicts), and `proxy(inputs)` (with `truth_proxy`) a
    function that can run here, which plans the count. The recorded reference then confirms it,
    recording further seeds as `seeds_for` asks, and the recording keeps exactly the confirmed count's
    draws and their redraw, (1 + REDRAW) times it. Where `draw` is a seeded function the test can call
    again, `store_inputs=False` records the outputs alone and each draw's `input_digest`, and the test
    hands `draw` to `indexed_recording` as `inputs_of`. With `summary`, the recording keeps no outputs
    at all: per seed and leaf, the reference's evidence check 1 reads (`leaf_summary`: its mean squared
    error from its truth, its worst sized row, its errors on the truth's rows of zero, finiteness) and
    its truth's sketch
    (`truth_sketch`), which the test's own float64 path is held to (`indexed_recording`'s `truth_fn`):
    kilobytes where outputs over (1 + REDRAW) times the seeds are gigabytes."""
    planned = seeds_for(jax.jit(proxy), lambda seed: (draw(seed),), label,
                        truth_fn=None if truth_proxy is None else jax.jit(truth_proxy))
    print(f"{label}: planned {planned} seeds on the proxy", flush=True)
    records = []

    def ensure(count):
        for seed in range(len(records), count):
            inputs = draw(seed)
            reference, truth = evaluate(inputs)
            records.append((inputs, jax.tree.map(np.asarray, reference), jax.tree.map(np.asarray, truth)))

    def recorded_draw(seed):
        ensure(seed + 1)
        return np.int32(seed), records[seed][0]

    ensure((1 + REDRAW) * planned)
    # The recorded reference is read by index from the host's records, as given.
    confirmed = seeds_for(as_given(lambda index, inputs: records[int(index)][1]), recorded_draw, label,
                          truth_fn=lambda index, inputs: records[int(index)][2])
    # The recording is the confirmed count's draws and their redraw, no more: a proxy that planned more
    # than the recorded reference confirms leaves draws the recording does not keep.
    kept = (1 + REDRAW) * confirmed
    ensure(kept)
    result = {}
    if summary:
        kept_records = records[:kept]
        for name in records[0][1]:
            evidence = [leaf_summary(record[1][name], record[2][name]) for record in records[:kept]]
            result.update({f"{label}.mse.{name}": np.asarray([leaf.mse for leaf in evidence]),
                           f"{label}.worst.{name}": np.asarray([leaf.worst for leaf in evidence]),
                           f"{label}.finite.{name}": np.asarray([leaf.finite for leaf in evidence]),
                           f"{label}.zero_rows.{name}": np.concatenate(
                               [np.zeros(0, np.int64), *(leaf.zero_rows for leaf in evidence)]),
                           f"{label}.zero_errors.{name}": np.concatenate(
                               [np.zeros(0), *(leaf.zero_errors for leaf in evidence)]),
                           f"{label}.zero_offsets.{name}": np.cumsum(
                               [0, *(len(leaf.zero_rows) for leaf in evidence)]),
                           f"{label}.sketch.{name}": np.stack([truth_sketch(record[2][name], seed)
                                                               for seed, record in enumerate(kept_records)]),
                           f"{label}.shape.{name}": np.asarray(np.shape(records[0][2][name]), np.int64),
                           f"{label}.dtype.{name}": np.asarray(evidence[0].dtype)})
        result[f"{label}.input_digest"] = np.asarray([input_digest(record[0]) for record in records[:kept]])
        result.update({f"{label}.seeds": np.int32(confirmed), f"{label}.redraw": np.int32(REDRAW)})
        print(f"{label}: declared {confirmed} seeds; recorded {kept} draws' evidence", flush=True)
        return result
    for index, kind in enumerate(("input", "reference", "truth")):
        if kind == "input" and not store_inputs:
            continue
        for name in records[0][index]:
            result[f"{label}.{kind}.{name}"] = np.stack([record[index][name] for record in records[:kept]])
    if not store_inputs:
        result[f"{label}.input_digest"] = np.asarray([input_digest(record[0]) for record in records[:kept]])
    result.update({f"{label}.seeds": np.int32(confirmed), f"{label}.redraw": np.int32(REDRAW)})
    print(f"{label}: declared {confirmed} seeds; recorded {kept} draws", flush=True)
    return result
