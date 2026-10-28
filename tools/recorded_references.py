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

A summarized recording too slow for one process is made in shards: `record_shard` keeps the evidence of
a run of seeds, each drawn as one run draws it, and `merge_shards` joins the shards in seed order and
declares the count their joined evidence resolves (`seeds_resolving`), writing what `record_case`'s
summary writes.
"""

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetched_fixtures import save
from reference_error import (
    REDRAW,
    SIZING_FROM,
    as_given,
    input_digest,
    leaf_summary,
    seeds_for,
    seeds_resolving,
    truth_sketch,
)

__all__ = ["merge_shards", "provenance", "record_case", "record_reference", "record_shard", "save",
           "with_pilots"]


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


def record_case(label: str, draw, evaluate, proxy=None, *, truth_proxy=None, store_inputs: bool = True,
                summary: bool = False, dew=None) -> dict[str, np.ndarray]:
    """`label`'s recording: `draw(seed)` gives a seed's inputs (a dict of arrays), `evaluate(inputs)` the
    reference's outputs and their float64 truth (dicts), and `proxy(inputs)` (with `truth_proxy`) a
    function that can run here, which plans the count. With no proxy, the recorded reference alone
    sizes the count, recording the draws its doublings take. The recorded reference then confirms it,
    recording further seeds as `seeds_for` asks, and the recording keeps exactly the confirmed count's
    draws and their redraw, (1 + REDRAW) times it. Where `draw` is a seeded function the test can call
    again, `store_inputs=False` records the outputs alone and each draw's `input_digest`, and the test
    hands `draw` to `indexed_recording` as `inputs_of`. With `summary`, the recording keeps no outputs
    at all: per seed and leaf, the reference's evidence check 1 reads (`leaf_summary`: its mean squared
    error from its truth, its worst sized row, its errors on the truth's rows of zero, finiteness) and
    its truth's sketch
    (`truth_sketch`), which the test's own float64 path is held to (`indexed_recording`'s `truth_fn`):
    kilobytes where outputs over (1 + REDRAW) times the seeds are gigabytes. With `dew(inputs)`, Dew's
    own outputs on a seed's inputs, the recorded reference confirms the count paired, as check 1's
    verdict pairs it (`seeds_for`'s `dew_fn`), for a site whose errors are heavy-tailed across draws
    alike in both arms (a lognormal training sigma), where the reference compared with itself on other
    draws never resolves."""
    if proxy is None and truth_proxy is not None:
        raise ValueError("a planning truth requires its proxy")
    # Without a proxy nothing is drawn ahead: the recorded reference's own doublings draw what they read.
    planned = (0 if proxy is None else
               seeds_for(jax.jit(proxy), lambda seed: (draw(seed),), label,
                         truth_fn=None if truth_proxy is None else jax.jit(truth_proxy)))
    print(f"{label}: starting with {planned} seeds", flush=True)
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
                          truth_fn=lambda index, inputs: records[int(index)][2],
                          dew_fn=None if dew is None else as_given(lambda index, inputs: dew(inputs)))
    # The recording is the confirmed count's draws and their redraw, no more: a proxy that planned more
    # than the recorded reference confirms leaves draws the recording does not keep.
    kept = (1 + REDRAW) * confirmed
    ensure(kept)
    if summary:
        result = _finished(label, _evidence(label, range(kept), records[:kept]), confirmed)
        print(f"{label}: declared {confirmed} seeds; recorded {kept} draws' evidence", flush=True)
        return result
    result = {}
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


def record_shard(label: str, draw, evaluate, seeds: range) -> dict[str, np.ndarray]:
    """The evidence `record_case`'s summary keeps of `seeds`, a run of a recording's seeds, each drawn as
    one run draws it (`draw(seed)`) and evaluated (`evaluate(inputs)`, the reference's outputs and their
    float64 truth): one shard of a recording `merge_shards` joins. Each seed's outputs are summarized
    as they are evaluated and dropped, so a shard holds its evidence alone."""
    def records():
        for seed in seeds:
            inputs = draw(seed)
            reference, truth = evaluate(inputs)
            yield inputs, jax.tree.map(np.asarray, reference), jax.tree.map(np.asarray, truth)
    return _evidence(label, seeds, records())


def merge_shards(label: str, shards: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """`label`'s summarized recording from its shards (`record_shard`), as `record_case`'s summary writes
    it: their seeds joined in order from seed 0, the count the joined evidence resolves
    (`seeds_resolving`, the reference against itself), and that count's (1 + REDRAW) times kept. Shards
    that leave a seed out or repeat one are refused, and so are too few seeds for the count's redraw.
    Shards of seeds from SIZING_FROM on are the recording's pilots, the draws a paired sizing of it takes
    and no verdict does (`seeds_for`): joined in order from SIZING_FROM, every one kept, under
    `<label>.pilot.`, their leaves the verdict's in shape and dtype, as a run from seed 0 is."""
    straddling = [_runs(shard[f"{label}.drawn"]) for shard in shards
                  if (np.asarray(shard[f"{label}.drawn"]) < SIZING_FROM).any()
                  and (np.asarray(shard[f"{label}.drawn"]) >= SIZING_FROM).any()]
    if straddling:
        raise ValueError(f"{label}: shards of seeds {straddling} hold verdict seeds and pilots from "
                         f"SIZING_FROM alike")
    verdict = [shard for shard in shards if int(shard[f"{label}.drawn"][0]) < SIZING_FROM]
    pilots = [shard for shard in shards if int(shard[f"{label}.drawn"][0]) >= SIZING_FROM]
    joined = _joined(label, verdict, 0)
    drawn = joined[f"{label}.drawn"]
    names = [key.removeprefix(f"{label}.mse.") for key in joined if key.startswith(f"{label}.mse.")]
    confirmed = seeds_resolving(np.stack([joined[f"{label}.mse.{name}"] for name in names], axis=1),
                                names, label)
    kept = (1 + REDRAW) * confirmed
    if len(drawn) < kept:
        raise ValueError(f"{label}: {confirmed} seeds resolve, and their redraw needs {kept} seeds; the "
                         f"shards hold {len(drawn)}")
    result = _finished(label, joined, confirmed)
    print(f"{label}: declared {confirmed} seeds; merged {kept} draws' evidence", flush=True)
    return with_pilots(label, result, pilots) if pilots else result


def with_pilots(label: str, recording: Mapping[str, np.ndarray],
                shards: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """`recording`, a summarized one (`record_case`'s summary or `merge_shards`'), with the pilots of
    `shards` (`record_shard` over seeds from SIZING_FROM on): joined in order from SIZING_FROM, every one
    kept, under `<label>.pilot.`, their leaves the recording's in name, shape and dtype. A recording that
    holds pilots already is refused, as are pilots that leave a seed out of their run."""
    if f"{label}.pilot.input_digest" in recording:
        raise ValueError(f"{label}: the recording holds pilots already")
    held = _joined(label, shards, SIZING_FROM)
    pilot = _kept(label, held, len(held[f"{label}.drawn"]))
    whole = {key for key in recording if _whole(label, key)}
    if {key for key in pilot if _whole(label, key)} != whole or any(
            not np.array_equal(pilot[key], recording[key]) for key in whole):
        raise ValueError(f"{label}: the pilots' leaves are not the verdict's in name, shape and dtype")
    print(f"{label}: with {len(held[f'{label}.drawn'])} pilots", flush=True)
    return {**recording, **{f"{label}.pilot.{key.removeprefix(f'{label}.')}": value
                            for key, value in pilot.items() if not _whole(label, key)}}


def _joined(label: str, shards: Sequence[Mapping[str, np.ndarray]], first: int) -> dict[str, np.ndarray]:
    """`shards`' evidence joined in order, refused unless their seeds are one run from `first`."""
    if not shards:
        raise ValueError(f"{label}: no shard holds seeds from {first}")
    ordered = sorted(shards, key=lambda shard: int(shard[f"{label}.drawn"][0]))
    joined = {key: np.concatenate([shard[key] for shard in ordered])
              for key in ordered[0] if not _whole(label, key)}
    joined.update({key: ordered[0][key] for key in ordered[0] if _whole(label, key)})
    for shard in ordered[1:]:
        for key in (key for key in shard if _whole(label, key)):
            if not np.array_equal(shard[key], joined.get(key)):
                raise ValueError(f"{label}: one shard's {key} is {shard[key]}, another's {joined.get(key)}")
    drawn = joined[f"{label}.drawn"]
    if not np.array_equal(drawn, np.arange(first, first + len(drawn))):
        raise ValueError(f"{label}: the shards hold seeds {_runs(drawn)}, not one run from seed {first}")
    return joined


def _evidence(label: str, seeds: range, records) -> dict[str, np.ndarray]:
    """`records`' evidence, one row a seed of `seeds`, read one record at a time (an iterable of
    (inputs, reference, truth)): each leaf's `leaf_summary` (its rows of float64 zero joined, with their
    count a seed) and its truth's sketch, and each seed's input digest."""
    digests, leaves, sketches, shapes = [], {}, {}, {}
    for seed, (inputs, reference, truth) in zip(seeds, records, strict=True):
        digests.append(input_digest(inputs))
        for name in reference:
            leaves.setdefault(name, []).append(leaf_summary(reference[name], truth[name]))
            sketches.setdefault(name, []).append(truth_sketch(truth[name], seed))
            shapes.setdefault(name, np.asarray(np.shape(truth[name]), np.int64))
    evidence = {f"{label}.drawn": np.asarray(seeds, np.int64), f"{label}.input_digest": np.asarray(digests)}
    for name, summaries in leaves.items():
        evidence.update({
            f"{label}.mse.{name}": np.asarray([leaf.mse for leaf in summaries]),
            f"{label}.worst.{name}": np.asarray([leaf.worst for leaf in summaries]),
            f"{label}.finite.{name}": np.asarray([leaf.finite for leaf in summaries]),
            f"{label}.zero_counts.{name}": np.asarray([len(leaf.zero_rows) for leaf in summaries], np.int64),
            f"{label}.zero_rows.{name}": np.concatenate([np.zeros(0, np.int64),
                                                         *(leaf.zero_rows for leaf in summaries)]),
            f"{label}.zero_errors.{name}": np.concatenate(
                [np.zeros(0), *(leaf.zero_errors for leaf in summaries)]),
            f"{label}.sketch.{name}": np.stack(sketches[name]),
            f"{label}.shape.{name}": shapes[name],
            f"{label}.dtype.{name}": np.asarray(summaries[0].dtype)})
    return evidence


def _finished(label: str, evidence: Mapping[str, np.ndarray], confirmed: int) -> dict[str, np.ndarray]:
    """The recording `evidence` (`_evidence`) makes at `confirmed` declared seeds: its first (1 + REDRAW)
    times that many seeds, each leaf's rows of float64 zero by their span a seed (`zero_offsets`)."""
    result = _kept(label, evidence, (1 + REDRAW) * confirmed)
    result.update({f"{label}.seeds": np.int32(confirmed), f"{label}.redraw": np.int32(REDRAW)})
    return result


def _kept(label: str, evidence: Mapping[str, np.ndarray], kept: int) -> dict[str, np.ndarray]:
    """`evidence`'s (`_evidence`) first `kept` seeds, each leaf's rows of float64 zero by their span a seed
    (`zero_offsets`)."""
    result = {f"{label}.input_digest": evidence[f"{label}.input_digest"][:kept]}
    names = [key.removeprefix(f"{label}.mse.") for key in evidence if key.startswith(f"{label}.mse.")]
    for name in names:
        offsets = np.cumsum([0, *evidence[f"{label}.zero_counts.{name}"][:kept]])
        result.update({f"{label}.{field}.{name}": evidence[f"{label}.{field}.{name}"][:kept]
                       for field in ("mse", "worst", "finite", "sketch")})
        result.update({f"{label}.zero_rows.{name}": evidence[f"{label}.zero_rows.{name}"][:offsets[-1]],
                       f"{label}.zero_errors.{name}": evidence[f"{label}.zero_errors.{name}"][:offsets[-1]],
                       f"{label}.zero_offsets.{name}": offsets,
                       f"{label}.shape.{name}": evidence[f"{label}.shape.{name}"],
                       f"{label}.dtype.{name}": evidence[f"{label}.dtype.{name}"]})
    return result


def _whole(label: str, key: str) -> bool:
    """Whether `key` holds one value for the whole recording (a leaf's shape and dtype), not one a seed."""
    return key.startswith((f"{label}.shape.", f"{label}.dtype."))


def _runs(seeds: np.ndarray) -> str:
    """`seeds`, sorted, as their runs: 0-15, 32-47."""
    ordered = np.unique(seeds)
    breaks = np.flatnonzero(np.diff(ordered) != 1) + 1
    return ", ".join(f"{run[0]}-{run[-1]}" for run in np.split(ordered, breaks))
