"""Recording a reference over seeds, for check 1's `indexed_recording` (tests/reference_error.py).

A reference that cannot run in the test (a CUDA-only library, a fetched source) is recorded once, over
the seeds check 1 draws: `record_case` sizes the count on the recorded reference itself with `seeds_for`,
records the confirmed count's whole redraw and returns the `<site>.input.*`, `<site>.reference.*`,
`<site>.truth.*` and `<site>.seeds` arrays to save beside its `provenance` with `save`, which commits a
recording of at most 1 MiB and sends a larger one to the fetched store (tools/fetched_fixtures.py).
(tools/recorded_draws.py is
another thing: it records the random draws a reference generator's library makes, so a second
precision reads the same samples.)

A summarized recording too slow for one process is made in shards: `record_shard` keeps the evidence of
a run of seeds, each drawn as one run draws it, and `merge_shards` joins the shards in seed order and
declares the count their joined evidence resolves (`seeds_resolving`), writing what `record_case`'s
summary writes. A reference whose recorder keeps each row's distance from its truth alone, its outputs
too large to keep over the seeds, is summarized from those by `record_distances`.
"""

import dataclasses
import sys
import urllib.parse
from collections.abc import Mapping, Sequence
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetched_fixtures import provenance, save
from reference_error import FEWEST_SEEDS, REDRAW, as_given, input_digest, seeds_for
from truth_sketch import truth_sketch

from dew.parity import (
    MOST_SEEDS,
    RECORDED_KINDS,
    SIZING_FROM,
    Conditioning,
    Inconclusive,
    RecordedLeaf,
    example_mse,
    example_worst,
    heavy_masks,
    leaf_summaries,
    recorded_kinds,
    rows_summary,
    seeds_resolving,
)
from dew.parity_stages import staged_units

__all__ = ["merge_shards", "provenance", "record_case", "record_distances", "record_logits",
           "record_reference", "record_shard", "record_truths", "save", "with_pilots"]


def record_reference(label: str, draw, evaluate, *, package: str, repo: str, tool: str, out: Path,
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
    recorded = record_case(label, draw, evaluate)
    written = save(out, {**recorded, **{name: np.asarray(value) for name, value in constants.items()},
                         "provenance": provenance(package, version, "cpu", f"python {tool}",
                                                  torch=torch.__version__)},
                   reference=f"{repo} {version}", license="Apache-2.0")
    print(f"{written}: {int(recorded[f'{label}.seeds'])} seeds, {package} {version}, "
          f"torch {torch.__version__}")
    return written


class _Kept:
    """A recording's evaluations kept as they are made and found again, so a long recording survives its
    task's recall (armada hands a recalled attempt's artifacts to the next): each one's trees (dicts of
    arrays) saved under `kept` by its recording's `label`, its kind and its seed, with its inputs' digest,
    and one an interrupted run left under `resumed` or `kept` read in place of making it, and kept on.
    A record found is this recording's where its reference makes it again: the first found of each kind is
    made again and held to it bit for bit, so another reference's records (other weights, another site
    drawing the same inputs) are refused, and so is a record of other inputs. A reference whose evaluation
    does not repeat to the bit therefore does not resume. With neither directory nothing is kept."""

    def __init__(self, kept: Path | None, resumed: Path | None, label: str):
        self.kept, self.resumed, self.label = kept, resumed, urllib.parse.quote(label, safe="")
        self.proven: set[str] = set()
        if kept is not None:
            kept.mkdir(parents=True, exist_ok=True)

    def made(self, kind: str, seed: int, inputs, make) -> tuple[dict, ...]:
        """`make()`'s trees for `kind` at `seed` over `inputs`, found where kept, else made and kept."""
        digest, name = input_digest(inputs), f"{self.label}.{kind}.{seed}.npz"
        for directory in filter(None, (self.resumed, self.kept)):
            if (path := directory / name).exists():
                with np.load(path, allow_pickle=False) as held:
                    if int(held["digest"]) != int(digest):
                        raise ValueError(f"{path} is a record of other inputs than seed {seed}'s here: "
                                         f"another recording's")
                    trees = tuple({key.split("/", 1)[1]: held[key] for key in held.files
                                   if key.startswith(f"{index}/")} for index in range(int(held["trees"])))
                if kind not in self.proven:
                    again = tuple(jax.tree.map(np.asarray, tree) for tree in make())
                    if _bytes(again) != _bytes(trees):
                        raise ValueError(f"{path} is not what this recording's reference makes at seed "
                                         f"{seed}: another reference's record, or one that does not repeat")
                    self.proven.add(kind)
                if directory != self.kept:
                    self._write(name, digest, trees)
                return trees
        trees = tuple(jax.tree.map(np.asarray, tree) for tree in make())
        self._write(name, digest, trees)
        return trees

    def _write(self, name: str, digest: np.uint64, trees: tuple[dict, ...]) -> None:
        if self.kept is None:
            return
        # Whole or absent: a task recalled mid-write leaves no half record for the next to read.
        partial = self.kept / f"{name}.partial"
        leaves = {f"{index}/{leaf}": value
                  for index, tree in enumerate(trees) for leaf, value in tree.items()}
        with open(partial, "wb") as file:
            np.savez(file, digest=np.asarray(digest), trees=np.int64(len(trees)), **leaves)
        partial.replace(self.kept / name)


def _bytes(trees: tuple[dict, ...]) -> list:
    """`trees`' leaves as their names, dtypes, shapes and bytes: equal where the trees are, to the bit."""
    return [[(name, str(np.asarray(value).dtype), np.shape(value), np.asarray(value).tobytes())
             for name, value in sorted(tree.items())] for tree in trees]


class _Records:
    """A recording's draws (`record_case`), each seed drawn once, as the sizing reads it: a paired sizing
    reads its pilots from SIZING_FROM, which a run of every seed below it would draw too. A pilot is read as
    it is drawn and never again (`_drawn` reads each draw through before the next), so only the one being read
    is held. With `reduced`, a verdict seed is held the same way, and of each the row `reduced(seed, record)`
    alone is kept (`rows`): a summarized recording's evidence, where its outputs over (1 + REDRAW) times the
    seeds are gigabytes. With `followed`, an evaluation's third value is the reference's decisions, which its
    truth followed, each verdict seed's kept (`decisions`) whatever else is dropped."""

    def __init__(self, draw, evaluate, reduced=None, kept: _Kept | None = None, followed: bool = False):
        self.draw, self.evaluate, self.reduced = draw, evaluate, reduced
        self.kept = _Kept(None, None, "") if kept is None else kept
        self.records, self.reading, self.rows, self.replayed = {}, {}, {}, set()
        self.followed, self.decisions = followed, {}

    def evaluated(self, seed: int):
        inputs = self.draw(seed)
        return inputs, self.kept.made("draw", seed, inputs, lambda: self.evaluate(inputs))

    def record(self, seed: int):
        """Seed `seed`'s inputs (with the leaves the reference made, `replayed`), reference and truth."""
        verdict = seed < SIZING_FROM
        held = self.records if verdict and self.reduced is None else self.reading
        if seed not in held:
            self.reading.clear()
            inputs, (reference, truth, *made) = self.evaluated(seed)
            if self.followed:
                (decisions,), made = made[:1], made[1:]
                if verdict:
                    self.decisions[seed] = jax.tree.map(np.asarray, decisions)
            made = jax.tree.map(np.asarray, made[0] if made else {})
            self.replayed.update(made)
            held[seed] = ({**inputs, **made}, jax.tree.map(np.asarray, reference),
                          jax.tree.map(np.asarray, truth))
            if verdict and self.reduced is not None:
                self.rows[seed] = self.reduced(seed, held[seed])
        return held[seed]

    def ensure(self, count: int) -> None:
        """Seeds 0 to `count` drawn, none of them again: a seed whose row is kept was."""
        for seed in range(count):
            if seed not in self.rows:
                self.record(seed)


def record_case(label: str, draw, evaluate, *, store_inputs: bool = True,
                summary: bool = False, dew=None, start: int = FEWEST_SEEDS,
                examples: bool = False, scale=None, judged: bool = False,
                kept_in: Path | None = None, resumed_from: Path | None = None,
                followed: bool = False) -> dict[str, np.ndarray]:
    """`label`'s recording: `draw(seed)` gives a seed's inputs (a dict of arrays), `evaluate(inputs)` the
    reference's outputs and their float64 truth (dicts). The recorded reference alone sizes the count
    (`seeds_for`), recording the draws its doublings take, and the recording keeps exactly the confirmed
    count's draws and their redraw, (1 + REDRAW) times it. Where `draw` is a seeded function the test can
    call again, `store_inputs=False` records the outputs alone and each draw's `input_digest`, and the test
    hands `draw` to `indexed_recording` as `inputs_of`. Where `evaluate` gives a third dict, the leaves the
    reference made that Dew replays (an
    engine's sampled ids, its kept support, its routes), the recording keeps them whole as
    `<label>.replayed.<leaf>`, and Dew's run and the test's take them beside the draw's inputs: the comparison
    is of the replay. With `summary`, the recording keeps no outputs at all: per seed and leaf, the
    reference's evidence check 1 reads (`leaf_summary`: its mean squared error from its truth, its worst sized
    row, its errors on the truth's rows of zero, finiteness) and its truth's sketch (`truth_sketch`), which
    the test's own float64 path is held to (`indexed_recording`'s `truth_fn`): kilobytes where outputs over (1
    + REDRAW) times the seeds are gigabytes. Each draw's evidence is read as the draw is made and its
    outputs dropped (`_Records`), so a summary holds one draw's outputs at a time, unless a weighed check
    reads them (below). With `dew(inputs)`, Dew's own outputs on a seed's inputs, the
    recorded reference confirms the count paired, as check 1's verdict pairs it (`seeds_for`'s `dew_fn`), for
    a site whose errors are heavy-tailed across draws alike in both arms (a lognormal training sigma), where
    the reference compared with itself on other draws never resolves. With `judged`, the recording is the
    arm the site's check 1 judges and `dew` its reference (another library running what Dew exports for it),
    and the count is confirmed with the two in those places. A site whose GPU paired sizing needs
    more seeds than the recording CPU's declares that count through `start`: the recording confirms it here
    and keeps its whole redraw. A site whose leaves stack independent examples (`examples`) is sized as its
    verdict judges it. A site that declares the weighed check (`assert_as_exact`'s `conditioning`) gives
    `scale(inputs, seed)`, its reference's own rounding scale at a draw (`_dew_torch.rounding.rounding_scale`
    over its float64 twin, every operation tapped): each draw of the weighed check's stages has its own
    recorded as `<label>.scale.<leaf>`, which the test's weighing reads (`conditioning_scale`); the weighed
    check's seeds are sized on the recorded reference as that check judges them (`seeds_for(...,
    conditioning=True)`, paired where the unweighted check's are), recorded as far as its stages draw (from
    the first seed, as the unweighted check's do), and declared as `<label>.conditioning`. Its outputs are
    kept whole, which the weighing reads; or, with `summary`, none are: the scales stay, which weigh Dew's
    own arm in the test, and beside them the reference's side of the weighed check as read here, with its
    outputs in hand (`_weighed`): each leaf's heavy elements over the weighed check's draws
    (`<label>.heavy.<leaf>`) and the reference's weighed evidence a draw, a recording of its own under
    `<label>.weighed` whose leaves are the classes' (`dew.parity.Conditioning.recorded_evidence`). Its
    unweighted sizing that never resolves declares the cap, MOST_SEEDS (docs/design/testing.md).
    With `kept_in`, a directory, each seed's evaluation and scale is saved there as
    it is made, and with `resumed_from`, an interrupted run's `kept_in`, found there in place of being made
    again (`_Kept`): the recording a run resumed makes is the one an uninterrupted run makes. With `followed`,
    a quantized computation's summary keeps the reference's decisions too (`_decided`)."""
    _followed_case(label, followed, dew, scale, summary)
    # A summary that weighs nothing keeps each draw's evidence as it is drawn and drops its outputs; a weighed
    # one reads each of its check's draws whole, twice (its sizing, then its heavy elements over them all).
    streamed = summary and scale is None

    def row(seed: int, record) -> _Row:
        return _row(seed, _summarized(record, examples))

    keeping = _Kept(kept_in, resumed_from, label)
    held = _Records(draw, evaluate, row if streamed else None, keeping, followed)
    record, ensure = held.record, held.ensure
    records, replayed = held.records, held.replayed

    def recorded_draw(seed):
        return np.int32(seed), record(int(seed))[0]

    recorded, live = _arms_of(label, record, dew, judged=judged, weighed=scale is not None)
    reference, arm = (live, recorded) if judged else (recorded, live)
    try:
        confirmed = seeds_for(reference, recorded_draw, label, start=start,
                              truth_fn=as_given(lambda index, inputs: record(int(index))[2]),
                              dew_fn=arm, examples=examples)
    except Inconclusive:
        # A weighed site whose unweighted sizing never resolves declares the cap (docs/design/testing.md).
        if scale is None:
            raise
        confirmed = MOST_SEEDS
    # The recording is the confirmed count's draws and their redraw, no more.
    kept = staged_units(confirmed)
    weighed, scales = 0, {}

    def scaled(seed: int):
        if seed not in scales:
            inputs = record(seed)[0]
            (made,) = held.kept.made("scale", seed, inputs, lambda: (scale(inputs, seed),))
            # As the recording holds it, float32, for the sizing, the reference's weighed evidence and
            # the test's weighing of Dew's arm alike: one denominator for both arms.
            scales[seed] = {name: np.asarray(value, np.float32) for name, value in made.items()}
        return scales[seed]

    if scale is not None:
        weighed = seeds_for(recorded, recorded_draw, label, start=start,
                            truth_fn=as_given(lambda index, inputs: record(int(index))[2]),
                            dew_fn=live, conditioning=True,
                            conditioning_scale=lambda inputs, seed: scaled(int(inputs[0])))
        # The weighed check draws its own stages' seeds from the first, as the unweighted check does.
        kept = max(kept, staged_units(weighed))
    ensure(kept)
    kept_records = [] if streamed else [records[seed] for seed in range(kept)]
    decided = _decided(label, held.decisions, kept) if followed else {}
    # The weighed check's own entries, the same whether the outputs are kept or summarized.
    weighing = {} if scale is None else {
        f"{label}.conditioning": np.int32(weighed),
        **{f"{label}.scale.{name}": np.stack([scaled(seed)[name]
                                             for seed in range(staged_units(weighed))])
           for name in scaled(0)}}
    if summary:
        # A weighed check that draws past the unweighted one's stages reads the reference and its truth's
        # sketch at every draw of its own, so the evidence covers the further of the two.
        rows = (held.rows[seed] if streamed else row(seed, kept_records[seed]) for seed in range(kept))
        result = _finished(label, _rows_evidence(label, range(kept), rows), confirmed, kept)
        result.update(decided)
        if scale is not None:
            result.update(weighing)
            result.update(_weighed(label, kept_records[:staged_units(weighed)], scaled))
        print(f"{label}: declared {confirmed} seeds; recorded {kept} draws' evidence", flush=True)
        return result
    result = _stacked(label, kept_records, replayed, store_inputs)
    result.update({f"{label}.seeds": np.int32(confirmed), f"{label}.redraw": np.int32(REDRAW)})
    result.update(weighing)
    print(f"{label}: declared {confirmed} seeds; recorded {kept} draws", flush=True)
    return result


def _followed_case(label: str, followed: bool, dew, scale, summary: bool) -> None:
    """A followed recording's case (`record_case`'s `followed`, docs/design/testing.md: a quantized
    computation, each arm held to the float64 computation that rounds where it rounded) refused unless it is a
    summary sized on the recorded reference alone, whose truth is the one that followed its own decisions."""
    if followed and (dew is not None or scale is not None or not summary):
        raise ValueError(f"{label}: a followed recording is a summary sized on the recorded reference alone, "
                         f"its truth the one that followed its own decisions")


def _decided(label: str, decisions: Mapping[int, Mapping[str, np.ndarray]],
             kept: int) -> dict[str, np.ndarray]:
    """A followed recording's decisions (`record_case`'s `followed`): an evaluation's third value, the
    reference's roundings' outputs, which its truth followed, each kept seed's stacked by leaf as
    `<label>.decisions.<name>`; the test's float64 path follows them to be held to the recorded truth
    (`indexed_recording`)."""
    return {f"{label}.decisions.{name}": np.stack([decisions[seed][name] for seed in range(kept)])
            for name in decisions[0]}


def _arms_of(label: str, record, dew, *, judged: bool, weighed: bool) -> tuple:
    """A recording's two arms as its sizing reads them (`record_case`): the recorded one, read by index from
    the host's records, as given, and Dew's run beside it (None where no `dew` is given). The recorded arm is
    the reference and Dew's the arm judged, or with `judged` the other way round, as the site's verdict has
    them: a recording judged takes `dew`, the reference it is judged against, and no scale, which is a
    recorded reference's own."""
    recorded = as_given(lambda index, inputs: record(int(index))[1])
    live = None if dew is None else as_given(lambda index, inputs: dew(inputs))
    if judged and (live is None or weighed):
        raise ValueError(f"{label}: a recording judged takes `dew`, the reference it is judged against, "
                         f"and no `scale`, which is a recorded reference's own")
    return recorded, live


def _stacked(label: str, records: Sequence, replayed: set, store_inputs: bool) -> dict[str, np.ndarray]:
    """`records`' outputs kept whole (`record_case` without a summary): each seed's reference and truth
    stacked by leaf, its inputs too or, not `store_inputs`, their digest, and the leaves the reference made
    (`replayed`) as `<label>.replayed.<leaf>` either way."""
    result = {}
    for index, kind in enumerate(("input", "reference", "truth")):
        if kind == "input" and not store_inputs:
            continue
        for name in records[0][index]:
            stored = "replayed" if kind == "input" and name in replayed else kind
            result[f"{label}.{stored}.{name}"] = np.stack([entry[index][name] for entry in records])
    for name in replayed if not store_inputs else ():
        result[f"{label}.replayed.{name}"] = np.stack([entry[0][name] for entry in records])
    if not store_inputs:
        result[f"{label}.input_digest"] = np.asarray([
            input_digest({name: value for name, value in entry[0].items() if name not in replayed})
            for entry in records])
    return result


def _weighed(label: str, records, scaled) -> dict[str, np.ndarray]:
    """The reference's side of a summarized recording's weighed check, over `records`, its draws from the
    first, each (inputs, reference, truth), their rounding scales `scaled(seed)`: each leaf's heavy
    elements over those draws (`heavy_masks`, as the test's own classing reads them of a reference it
    can run), `<label>.heavy.<leaf>` where a leaf has any; and the reference's weighed evidence a draw
    (`Conditioning.recorded_evidence`), kept as a recording's evidence is (`_evidence`) under
    `<label>.weighed`, its leaves the classes' paths and its truths their zeros."""
    heavy = heavy_masks((reference, truth) for _, reference, truth in records)
    conditioning = Conditioning(lambda inputs, seed: scaled(seed), heavy)

    def evidence():
        for seed, (inputs, reference, truth) in enumerate(records):
            summaries, zeros = conditioning.recorded_evidence(reference, truth, inputs, seed)
            yield inputs, summaries, zeros

    nested = f"{label}.weighed"
    result = _kept(nested, _evidence(nested, range(len(records)), evidence()), len(records))
    # A truth's leaves in their tree's order, the masks' (`heavy_masks`).
    names = [jax.tree_util.keystr(path, simple=True, separator="/")
             for path, _ in jax.tree_util.tree_leaves_with_path(records[0][2])]
    result.update({f"{label}.heavy.{name}": mask for name, mask in zip(names, heavy, strict=True)
                   if mask is not None})
    return result


def record_truths(label: str, draw, evaluate, count: int, *, followed: bool = False) -> dict[str, np.ndarray]:
    """A recording of the reference's float64 truth alone at its first `count` draws (`draw(seed)`, the
    inputs, `evaluate(inputs)` as `record_case`'s, its float32 outputs unread): each draw's inputs
    (`<label>.input.<name>`), each leaf's truth's sketch (`truth_sketch`) and shape, and with `followed` the
    reference's decisions, an evaluation's third value, which its truth followed (`_decided`); the test holds
    Dew's float64 path to it per leaf at every draw (`reference_error.assert_truths_as_recorded`). No count is
    sized and no float32 evidence kept: the site holds its function, not its rounding (docs/design/testing.md,
    a site whose float32 errors no count of draws resolves)."""
    inputs, sketches, shapes, decisions = [], {}, {}, {}
    for seed in range(count):
        drawn = {name: np.asarray(value) for name, value in draw(seed).items()}
        made = evaluate(drawn)
        if followed:
            decisions[seed] = jax.tree.map(np.asarray, made[2])
        for name, value in made[1].items():
            sketches.setdefault(name, []).append(truth_sketch(value, seed))
            shapes.setdefault(name, np.asarray(np.shape(value), np.int64))
        inputs.append(drawn)
    return {f"{label}.truths": np.int32(count),
            **{f"{label}.input.{name}": np.stack([each[name] for each in inputs]) for name in inputs[0]},
            **{f"{label}.sketch.{name}": np.stack(rows) for name, rows in sketches.items()},
            **{f"{label}.shape.{name}": shape for name, shape in shapes.items()},
            **(_decided(label, decisions, count) if followed else {})}


def record_logits(directory: Path, model, wide, command: str, *, reference: str, revision: str, license: str,
                  draw=None, padded: Mapping[str, np.ndarray] | None = None,
                  extra=None, stepped: int | None = None, **details) -> Path:
    """The tiny decoder at `directory`'s reference recorded over check 1's seeds, which its test reads in
    the reference's place (a release's remote code or a release apart from the suite's, which the test
    cannot run; hf_reference.py `write_logits_recording`'s): each seed's ids, `draw(seed)` (`{"input_ids":
    ids}`, by default tests/interop_support.py `fixture_ids`' in the fixture's shape), the torch causal LM
    `model`'s fp32 eager logits on the CPU and `wide`'s, its float64 twin's
    (`_dew_torch.precision.float64_twin`), inside `_dew_torch.precision.float64`, as the leaf `logits`; and
    with
    `padded`, the fields the reference's padded rows take
    (`attention_mask` and its `position_ids`), the leaf `padded_logits` too: its logits over the ids
    padded with 0 where the mask is false, at the mask's valid places; and with `extra`, the leaves
    `extra(network, tokens)` gives over the unpadded ids (a looped model's exit gates), the reference's
    alone; and with `stepped`, the leaf `stepped_logits`: the logits at
    the shapes a decode of a `stepped`-token prefill computes them, the prefill's from one forward over
    it and each later id's from one over the ids up to it (ledger 1543), which a test's cached steps
    are judged against. The count is the reference's own (`record_case`); the
    reference's evidence alone is kept with its truth's sketch (`record_case`'s summary), which the
    test's float64 run of Dew's model is held to, saved as recorded_logits.npz beside the fixture with
    its provenance, `reference` at `revision` under `license`, and `details`. Returns the file written."""
    import torch
    from interop_support import fixture_ids

    from _dew_torch.precision import float64

    mask = None if padded is None else np.asarray(padded["attention_mask"], bool)
    fields = {name: torch.from_numpy(np.asarray(value, np.int64 if name == "position_ids" else bool))
              for name, value in (padded or {}).items()}
    if draw is None:
        ids_of = fixture_ids(directory)

        def draw(seed):
            return {"input_ids": ids_of(seed)[0]}

    def logits(network, ids, *, prefixes: bool = True):
        tokens = torch.from_numpy(ids.astype(np.int64))
        with torch.no_grad():
            out = {"logits": network(input_ids=tokens, use_cache=False).logits.numpy()}
            if mask is not None:
                rows = torch.from_numpy(np.where(mask, ids, 0).astype(np.int64))
                out["padded_logits"] = network(input_ids=rows, **fields, use_cache=False).logits.numpy()[mask]
            if extra is not None:
                out.update({name: value.numpy() for name, value in extra(network, tokens).items()})
            if stepped is not None and not prefixes:
                # A truth's products round alike at every shape: the one forward's.
                out["stepped_logits"] = out["logits"]
            elif stepped is not None:
                out["stepped_logits"] = np.concatenate([
                    network(input_ids=tokens[:, :stepped], use_cache=False).logits.numpy(),
                    *(network(input_ids=tokens[:, :length], use_cache=False).logits.numpy()[:, -1:]
                      for length in range(stepped + 1, tokens.shape[1] + 1))], axis=1)
        return out

    def evaluate(inputs):
        with float64():
            truth = logits(wide, inputs["input_ids"], prefixes=False)
        return logits(model, inputs["input_ids"]), truth

    with jax.enable_x64(new_val=True):
        recording = record_case("logits", draw, evaluate, store_inputs=False, summary=True)
    record = provenance(reference, revision, "cpu", command, torch=torch.__version__, dtype="float32",
                        attention="eager", **details)
    return save(directory / "recorded_logits.npz", {**recording, "logits.provenance": record},
                reference=reference, license=license)


def record_shard(label: str, draw, evaluate, seeds: range) -> dict[str, np.ndarray]:
    """The evidence `record_case`'s summary keeps of `seeds`, a run of a recording's seeds, each drawn as
    one run draws it (`draw(seed)`) and evaluated (`evaluate(inputs)`, the reference's outputs and their
    float64 truth): one shard of a recording `merge_shards` joins. Each seed's outputs are summarized
    as they are evaluated and dropped, so a shard holds its evidence alone."""
    def records():
        for seed in seeds:
            inputs = draw(seed)
            reference, truth = evaluate(inputs)
            yield _summarized((inputs, jax.tree.map(np.asarray, reference), jax.tree.map(np.asarray, truth)))
    return _evidence(label, seeds, records())


def record_distances(label: str, draw, distances: Mapping[str, np.ndarray], truth_of,
                     dtypes: Mapping[str, str]) -> dict[str, np.ndarray]:
    """`label`'s summarized recording, as `merge_shards` writes it, of a reference that ran apart and kept
    each row's RMS distance from its float64 truth alone (a recorder whose outputs over the seeds are too
    large to keep): `distances[name]` `[seeds, rows]` from seed 0 for each leaf, `truth_of(seed)` the seed's
    float64 truth by leaf, `draw(seed)` its inputs, and `dtypes[name]` the reference's leaf's dtype. The
    count the evidence resolves is declared (`seeds_resolving`, the reference against itself), and that
    count's (1 + REDRAW) times kept, each kept seed's evidence read from its rows (`rows_summary`) beside
    its truth's sketch."""
    names = list(distances)
    # A leaf of no rows is exact, its error zero.
    errors = np.stack([np.mean(np.square(rows), axis=1) if rows.shape[1] else np.zeros(rows.shape[0])
                       for rows in (np.asarray(distances[name], np.float64) for name in names)], axis=1)
    confirmed = seeds_resolving(errors, names, label)
    kept = staged_units(confirmed)

    def records():
        for seed in range(kept):
            truth = truth_of(seed)
            summaries = {name: rows_summary(distances[name][seed], truth[name],
                                            finite=bool(np.isfinite(distances[name][seed]).all()),
                                            dtype=dtypes[name]) for name in names}
            yield draw(seed), summaries, truth
    result = _finished(label, _evidence(label, range(kept), records()), confirmed)
    print(f"{label}: declared {confirmed} seeds; recorded {kept} draws' evidence", flush=True)
    return result


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
    kept = staged_units(confirmed)
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


def _summarized(record, examples: bool = False):
    """A record of (inputs, reference, truth) as `_evidence` reads it: each leaf's `leaf_summary`, the
    record's leaves read at once (`leaf_summaries`), and at a site whose leaves stack independent
    `examples`, each example's evidence (`example_mse`, `example_worst`)."""
    inputs, reference, truth = record
    names = list(reference)
    summaries = leaf_summaries([reference[name] for name in names], [truth[name] for name in names])
    if examples:
        summaries = [dataclasses.replace(summary, example_mse=example_mse(reference[name], truth[name]),
                                         example_worst=example_worst(reference[name], truth[name]))
                     for name, summary in zip(names, summaries, strict=True)]
    return inputs, dict(zip(names, summaries, strict=True)), truth


@dataclasses.dataclass(frozen=True)
class _Row:
    """One seed's evidence as a summarized recording keeps it (`_rows_evidence`): its inputs' digest and, by
    leaf, the reference's summary, its truth's sketch and its truth's shape."""

    digest: str
    leaves: Mapping[str, RecordedLeaf]
    sketches: Mapping[str, np.ndarray]
    shapes: Mapping[str, np.ndarray]


def _row(seed: int, record) -> _Row:
    """`seed`'s row of a record (inputs, each leaf's `RecordedLeaf`, truth), which holds none of its
    arrays."""
    inputs, summaries, truth = record
    return _Row(input_digest(inputs), dict(summaries),
                {name: truth_sketch(truth[name], seed) for name in summaries},
                {name: np.asarray(np.shape(truth[name]), np.int64) for name in summaries})


def _evidence(label: str, seeds: range, records) -> dict[str, np.ndarray]:
    """`records`' evidence, one row a seed of `seeds`, read one record at a time (an iterable of
    (inputs, each leaf's `RecordedLeaf`, truth), `_row`), joined (`_rows_evidence`)."""
    return _rows_evidence(label, seeds,
                          (_row(seed, record) for seed, record in zip(seeds, records, strict=True)))


def _rows_evidence(label: str, seeds: range, rows) -> dict[str, np.ndarray]:
    """`rows`' evidence (`_Row`), one a seed of `seeds`: each leaf's summary (its rows of float64 zero
    joined, with their count a seed) and its truth's sketch, and each seed's input digest."""
    digests, leaves, sketches, shapes = [], {}, {}, {}
    for _, held in zip(seeds, rows, strict=True):
        digests.append(held.digest)
        for name, leaf in held.leaves.items():
            leaves.setdefault(name, []).append(leaf)
            sketches.setdefault(name, []).append(held.sketches[name])
            shapes.setdefault(name, held.shapes[name])
    evidence = {f"{label}.drawn": np.asarray(seeds, np.int64), f"{label}.input_digest": np.asarray(digests)}
    for name, summaries in leaves.items():
        evidence.update({
            f"{label}.mse.{name}": np.asarray([leaf.mse for leaf in summaries]),
            f"{label}.worst.{name}": np.asarray([leaf.worst for leaf in summaries]),
            # Each worst row's largest element's share of it, which the gate's row factor reads; a leaf read
            # from its rows' distances alone has none (`rows_summary`), and its recording keeps none.
            **({f"{label}.share.{name}": np.asarray([leaf.share for leaf in summaries], np.float64)}
               if all(leaf.share is not None for leaf in summaries) else {}),
            f"{label}.finite.{name}": np.asarray([leaf.finite for leaf in summaries]),
            f"{label}.zero_counts.{name}": np.asarray([len(leaf.zero_rows) for leaf in summaries], np.int64),
            f"{label}.zero_rows.{name}": np.concatenate([np.zeros(0, np.int64),
                                                         *(leaf.zero_rows for leaf in summaries)]),
            f"{label}.zero_errors.{name}": np.concatenate(
                [np.zeros(0), *(leaf.zero_errors for leaf in summaries)]),
            f"{label}.sketch.{name}": np.stack(sketches[name]),
            f"{label}.shape.{name}": shapes[name],
            f"{label}.dtype.{name}": np.asarray(summaries[0].dtype)})
        if summaries[0].example_mse is not None:
            # An example of no sized row leaves its leaf's worst rows unread (NaN), as `example_worst` does.
            unread = np.full(len(summaries[0].example_mse), np.nan)
            evidence[f"{label}.example_mse.{name}"] = np.stack([leaf.example_mse for leaf in summaries])
            evidence[f"{label}.example_worst.{name}"] = np.stack(
                [unread if leaf.example_worst is None else leaf.example_worst for leaf in summaries])
    return evidence


def _finished(label: str, evidence: Mapping[str, np.ndarray], confirmed: int,
              kept: int | None = None) -> dict[str, np.ndarray]:
    """The recording `evidence` (`_evidence`) makes at `confirmed` declared seeds: its first (1 + REDRAW)
    times that many seeds, or `kept` of them where a weighed check draws further, each leaf's rows of
    float64 zero by their span a seed (`zero_offsets`)."""
    result = _kept(label, evidence, staged_units(confirmed) if kept is None else kept)
    result.update({f"{label}.seeds": np.int32(confirmed), f"{label}.redraw": np.int32(REDRAW)})
    return result


def _kept(label: str, evidence: Mapping[str, np.ndarray], kept: int) -> dict[str, np.ndarray]:
    """`evidence`'s (`_evidence`) first `kept` seeds, each of a leaf's entries cut as the recording's layout
    holds its kind (`dew.parity.RECORDED_KINDS`): a row a seed, its rows of float64 zero by their span a seed
    with the spans' offsets, or whole."""
    result = {f"{label}.input_digest": evidence[f"{label}.input_digest"][:kept]}
    names = [key.removeprefix(f"{label}.mse.") for key in evidence if key.startswith(f"{label}.mse.")]
    (offsets_kind,) = recorded_kinds("offsets")
    for name in names:
        offsets = np.cumsum([0, *evidence[f"{label}.zero_counts.{name}"][:kept]])
        cuts = {"seed": slice(kept), "span": slice(offsets[-1]), "whole": ...}
        result.update({f"{label}.{kind}.{name}": evidence[f"{label}.{kind}.{name}"][cuts[held]]
                       for kind, held in RECORDED_KINDS.items()
                       if held in cuts and f"{label}.{kind}.{name}" in evidence})
        result[f"{label}.{offsets_kind}.{name}"] = offsets
    return result


def _whole(label: str, key: str) -> bool:
    """Whether `key` holds one value for the whole recording (a leaf's shape and dtype), not one a seed
    (`dew.parity.RECORDED_KINDS`' whole kinds)."""
    return key.startswith(tuple(f"{label}.{kind}." for kind in recorded_kinds("whole")))


def _runs(seeds: np.ndarray) -> str:
    """`seeds`, sorted, as their runs: 0-15, 32-47."""
    ordered = np.unique(seeds)
    breaks = np.flatnonzero(np.diff(ordered) != 1) + 1
    return ", ".join(f"{run[0]}-{run[-1]}" for run in np.split(ordered, breaks))


def held_to_the_oracle(exact_run, arrays, truth: dict, roundings: int, label: str) -> None:
    """Dew's float64 path `exact_run` on a seed's `arrays` held to the oracle's `truth`, leaf by leaf, within
    float64's rounding of `roundings` operations (`assert_computes_the_oracle`): a recorder's cross-check
    that its truth and the test's own float64 path compute the same function."""
    from reference_error import assert_computes_the_oracle, widened

    with jax.enable_x64(new_val=True):
        twin = jax.tree.map(np.asarray, exact_run(widened(arrays)))
    for name in truth:
        assert_computes_the_oracle(twin[name], truth[name], f"{label} float64 {name}", roundings=roundings)

