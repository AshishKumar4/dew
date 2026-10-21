"""What a token mixture weighs the windows a windows or pack stage makes by: the rule their
documents' targets were counted by, in every read and layout, after a resume and a publication."""

import dataclasses

import numpy as np
import pytest
from test_pipeline_core import READ, Scored, delivered, roled

from dew.data.cache import Cache
from dew.data.compile import compile, publish
from dew.data.dataset import DataPartition
from dew.data.pipeline import Pipeline, mix
from dew.data.sources import Memory


@dataclasses.dataclass(frozen=True)
class Sourced:
    """Labels a window with its corpus, which a windows or pack stage leaves out of its windows."""

    source: int

    def __call__(self, row):
        return {**row, "source": np.int32(self.source)}


ROLED = ("text", "text_roles")
LAYOUTS = {
    # Each layout's documents of seven tokens, and the most targets one of its windows holds.
    "windows": (lambda counted: counted.windows(8, fields=ROLED), 8),
    "first-fit": (lambda counted: counted.pack(16, fields=ROLED), 12),
    "concat": (lambda counted: counted.pack(8, fields=ROLED, policy="concat"), 7),
    "windows-then-pack": (lambda counted: counted.windows(4, fields=ROLED).pack(10, fields=ROLED), 8),
}


@pytest.mark.parametrize("role", [None, 1])
@pytest.mark.parametrize("layout", list(LAYOUTS))
@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_a_token_mixture_weighs_laid_out_windows_by_the_rule_their_documents_were_counted_by(
        read, layout, role):
    """Targets counted on documents by a rule (`CountTargets(role=)`) describe none of the windows a
    windows or pack stage makes of them: each window is counted again by that rule, in every read,
    and a second layout counts its own windows by it too. Two corpora whose documents score one
    and five role-1 targets deliver equal targets by the rule, to within one window of each."""
    from dew.data.steps.text import CountTargets

    laid_out, most = LAYOUTS[layout]

    def corpus(source, scored):
        counted = Pipeline(roled(source, 7, scored), name=str(source)).filter_map(CountTargets(role=role))
        return laid_out(counted).map(Sourced(source))

    left, right = delivered([corpus(0, 1), corpus(1, 5)], role, read)
    assert abs(int(left) - int(right)) <= most + most


@pytest.mark.parametrize("route", ["cached prefix", "generation source"])
@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_a_token_mixture_weighs_a_published_generations_windows_by_its_documents_rule(tmp_path, read, route):
    """A generation published from documents counted by a role's rule (`CountTargets(role=1)`)
    stores a plan of its documents' lengths, which counts every target of a window: its pack lays
    the documents out as their rows arrive, each window counted by that rule, whether the chain
    names the steps that counted them or reads the generation's source, so two such corpora
    deliver equal role-1 targets, to within one window of each."""
    from dew.data.steps.text import CountTargets
    from dew.data.text import Chunks

    def corpus(source, scored):
        # A chunk step states each document's length, which a published plan needs; none is cut.
        prefix = (Pipeline(roled(source, 7, scored), name=str(source)).filter_map(CountTargets(role=1))
                  .flat_map(Chunks(16, fields=ROLED)))
        cache = Cache(str(tmp_path / f"cache-{source}"), spool_bytes=4 * 1024**2, shard_bytes=256 * 1024)
        generation = publish(prefix.pack(16, fields=ROLED), cache, read="indexed", loading=READ)
        documents = prefix.cache(generation) if route == "cached prefix" else Pipeline(generation.source())
        return documents.pack(16, fields=ROLED).map(Sourced(source))

    left, right = delivered([corpus(0, 1), corpus(1, 5)], 1, read)
    assert abs(int(left) - int(right)) <= 12 + 12


@pytest.mark.parametrize("counted", [True, False], ids=["counted", "uncounted"])
@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_a_crop_after_a_layout_is_counted_as_the_windows_it_leaves(read, counted):
    """Documents, counted by every target or never, whose roles a projection leaves out, laid out in
    windows of four that a crop cuts to three, beside a corpus whose windows of four are left
    whole: each cropped window delivers two targets and each whole one four, and a mixture by
    tokens weighs them so, so the two deliver equal targets to within a window of each."""
    from dew.data.steps.text import CountTargets, LengthPolicy

    def corpus(source, *, cropped):
        documents = Pipeline(roled(source, 7, 3), name=str(source))
        documents = documents.filter_map(CountTargets()) if counted else documents
        windows = documents.select_columns("text", "source").windows(4)
        return (windows.map(LengthPolicy(3)) if cropped else windows).map(Sourced(source))

    left, right = delivered([corpus(0, cropped=True), corpus(1, cropped=False)], None, read)
    assert abs(int(left) - int(right)) <= 2 + 4


@dataclasses.dataclass(frozen=True)
class HalfScored:
    """Counts the targets of a record with an odd id in role 1, and of the rest every target."""

    def __call__(self, row):
        from dew.data.steps.text import CountTargets

        return (CountTargets(role=1) if int(row["id"]) % 2 else CountTargets()).map(row)


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_a_generation_of_documents_counted_by_two_rules_is_packed_as_its_rows_arrive(tmp_path, read):
    """A generation whose rows were counted by two rules, every target's and role 1's, states the
    one it cannot plan by, so a pack of its source does not plan by the documents' lengths: it lays
    the rows out as they arrive, where a layout holding two rules is refused."""
    from dew.data.text import Chunks

    documents = (Pipeline(roled(0, 7, 3), name="half").filter_map(HalfScored(), identity="half-scored@1")
                 .flat_map(Chunks(16, fields=ROLED)))
    cache = Cache(str(tmp_path / "cache"), spool_bytes=4 * 1024**2, shard_bytes=256 * 1024)
    generation = publish(documents, cache, read="indexed", loading=READ)
    assert generation.source().metadata()["count_rules"] == [[1, None]]
    chain = Pipeline(generation.source()).pack(16, fields=ROLED).repeat().batch(2)
    with pytest.raises(ValueError):
        stream = compile(chain, read=read, loading=READ).train(DataPartition())
        try:
            next(stream)
        finally:
            stream.close()


@pytest.mark.parametrize("store", ["generations concatenated", "an export", "memory"])
@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_a_store_of_role_counted_documents_states_its_rule_so_its_pack_counts_by_it(tmp_path, read, store):
    """Documents counted in role 1, held by stores that state their documents' lengths: two
    generations concatenated, an export read back, or a store in memory. Each states the rule its
    rows were counted by, so its pack lays them out as they arrive and counts each window by it,
    and a token mixture of two such stores, scoring one and five role-1 targets a document,
    delivers equal role-1 targets to within a window of each."""
    from dew.data.sinks import export
    from dew.data.sources import Files
    from dew.data.sources.concatenate import Concatenate
    from dew.data.steps.text import CountTargets
    from dew.data.text import Chunks

    def documents(source, scored):
        counted = Pipeline(roled(source, 7, scored), name=str(source)).filter_map(CountTargets(role=1))
        return counted.flat_map(Chunks(16, fields=ROLED))

    def stored(source, scored):
        if store == "generations concatenated":
            caches = [Cache(str(tmp_path / f"cache-{source}-{part}"), spool_bytes=4 * 1024**2,
                            shard_bytes=256 * 1024) for part in range(2)]
            return Concatenate(tuple(publish(documents(source, scored), cache, read="indexed", loading=READ)
                                     .source() for cache in caches))
        # Rows a chain delivers carry no count of Dew's; these are the step's own outputs, called
        # on each record, which do.
        rows = [CountTargets(role=1).map(row) for row in roled(source, 7, scored).records("memory")]
        if store == "memory":
            # Columns with each document's length, which a store in memory states.
            return Memory({**{name: [row[name] for row in rows] for name in rows[0]},
                           "length": [len(row["text"]) for row in rows]})
        export(rows, str(tmp_path / f"export-{source}"), format="array_record", length_field="text")
        return Files(str(tmp_path / f"export-{source}"))

    def corpus(source, scored):
        held = stored(source, scored)
        assert held.metadata()["count_rules"] == [[1, None]]
        return Pipeline(held, name=str(source)).pack(16, fields=ROLED).map(Sourced(source))

    left, right = delivered([corpus(0, 1), corpus(1, 5)], 1, read)
    assert abs(int(left) - int(right)) <= 12 + 12


def test_a_parquet_store_of_token_documents_with_no_counts_is_packed_by_its_lengths(tmp_path):
    """Parquet token documents with their lengths and no counted targets column hold no count of
    Dew's, so their pack plans by the lengths, as a global order: a shuffle of its windows reads."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from dew.data.sources import Files

    documents = [np.arange(3 + index % 5, dtype=np.int32) for index in range(24)]
    table = pa.table({"text": [document.tolist() for document in documents],
                      "length": [len(document) for document in documents]})
    pq.write_table(table, tmp_path / "documents.parquet")
    store = Files(str(tmp_path / "documents.parquet"))
    assert store.metadata()["count_rules"] == []
    chain = Pipeline(store).pack(16).shuffle(0).repeat().batch(2)
    stream = compile(chain, read="indexed", loading=READ).train(DataPartition())
    try:
        assert np.asarray(next(stream)["text"]).shape == (2, 16)
    finally:
        stream.close()


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_documents_counted_by_every_target_lay_out_beside_uncounted_ones(read):
    """A layout of documents some of which were counted by every target and some never counted:
    both are counted alike, by every target, so one layout holds them."""
    from dew.data.steps.text import CountTargets

    counted = Pipeline(roled(0, 7, 3), name="counted").filter_map(CountTargets())
    plain = Pipeline(roled(1, 7, 3), name="plain")
    mixed = mix([counted, plain]).windows(8, fields=ROLED).repeat().batch(2)
    stream = compile(mixed, read=read, loading=READ).train(DataPartition())
    try:
        assert all(np.asarray(next(stream)["text"]).shape == (2, 9) for _ in range(3))
    finally:
        stream.close()


@pytest.mark.parametrize("layout", ["windows", "first-fit"])
@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_windows_that_drop_the_roles_their_documents_were_counted_in_are_refused(read, layout):
    """A layout that leaves out the roles a count was taken in (`CountTargets(role=1)`) makes
    windows no rule can count again, which a token mixture would weigh by another rule."""
    from dew.data.steps.text import CountTargets

    counted = Pipeline(roled(0, 7, 3), name="dropped").filter_map(CountTargets(role=1))
    laid_out = counted.windows(8) if layout == "windows" else counted.pack(16)
    data = mix([laid_out.map(Sourced(0)).repeat()], by="tokens").batch(2)
    with pytest.raises(ValueError):
        stream = compile(data, read=read, loading=READ).train(DataPartition())
        try:
            next(stream)
        finally:
            stream.close()


@dataclasses.dataclass(frozen=True)
class Delivered:
    """A delivered row as its tokens and the role-1 targets the loss scores of it."""

    def __call__(self, row):
        return {"text": row["text"], **Scored(1)(row)}


@pytest.mark.parametrize("read", ["indexed", "streamed"])
@pytest.mark.parametrize(("layout", "workers"), [("windows", 0), ("first-fit", 2), ("concat", 2)])
def test_a_laid_out_corpus_resumed_reads_its_windows_counted_as_the_uninterrupted_read(
        read, layout, workers):
    """A read of windows counted by their documents' rule saves the rule with its position, so a
    resume whose next windows come from documents already read counts them by it too: windows
    cut from a carried document, and a first-fit pack's open bins or a concatenating pack's
    remainder held in loading workers."""
    from dew.data.steps.text import CountTargets

    def corpus(source, scored):
        # Documents longer than a window, of lengths a window doesn't divide, leave windows and
        # partial bins held when the read stops, so a resume reads next from documents read before;
        # each of its own tokens, so a lost, repeated or replayed row shows.
        length = 40 if layout == "windows" else 11
        documents = Memory([{"text": np.arange(length, dtype=np.int32) + 1000 * index + 100_000 * source,
                             # The first `scored` targets in role 1, as `roled` scores them: a
                             # token's role scores the one after it.
                             "text_roles": (np.arange(length) < scored + 1).astype(np.int32),
                             "source": np.int32(source)} for index in range(8)], shard_rows=4)
        # A stream read by two loading workers holds a shard for each of them.
        counted = Pipeline(documents, name=str(source)).filter_map(CountTargets(role=1))
        laid_out = (counted.windows(8, fields=ROLED) if layout == "windows"
                    else counted.pack(16, fields=ROLED, policy=layout, bins=3) if layout == "first-fit"
                    else counted.pack(16, fields=ROLED, policy=layout))
        return laid_out.map(Sourced(source)).repeat()

    loading = dataclasses.replace(READ, workers=workers)

    def stream():
        mixed = mix([corpus(0, 1), corpus(1, 5)], by="tokens")
        return compile(mixed.map(Delivered()).batch(2), read=read, loading=loading).train(DataPartition())

    uninterrupted = stream()
    try:
        expected = [{name: np.asarray(value).tolist() for name, value in next(uninterrupted).items()}
                    for _ in range(6)]
    finally:
        uninterrupted.close()
    stopped = stream()
    try:
        for _ in range(3):
            next(stopped)
        saved = stopped.get_state()
    finally:
        stopped.close()
    resumed = stream()
    try:
        resumed.set_state(saved)
        rest = [{name: np.asarray(value).tolist() for name, value in next(resumed).items()}
                for _ in range(3)]
    finally:
        resumed.close()
    assert rest == expected[3:]
