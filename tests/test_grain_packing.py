"""Packed windows against grain's own packers: a first-fit plan reads the
windows `grain.experimental.FirstFitPackIterDataset` packs, and a memory read
packs by the stage's policy as an indexed read does."""

import grain
import jax
import numpy as np
import pytest
from grain.experimental import FirstFitPackIterDataset

from dew.data import Loading, Memory
from dew.data.compile import compile, validation
from dew.data.dataset import DataPartition
from dew.data.pipeline import Pipeline
from dew.data.plan import fingerprint
from dew.data.text import Chunks

READ = Loading(workers=0, threads=1, read_buffer=0, worker_buffer=1)


FIELDS = ("text", "text_roles")


@pytest.mark.parametrize("window, bins, segments", [(8, 1, None), (8, 3, None), (16, 4, 3), (5, 2, 2)])
def test_planned_windows_are_grains_first_fit_windows(window, bins, segments):
    """Documents from one token to well past the window, a few exactly the
    window, with ids and roles, cut by a chunk step before a first-fit pack:
    grain's packer, given the chunks in file order and its bins emitted in
    order, writes window for window what a memory read and an indexed read of
    documents that state their lengths read by the plan (`plan_windows`), and
    what an indexed read of documents that state none packs by grain's packer
    as they stream: the ids, the roles and each token's document and place in
    it. Every read leaves out a window with no target; the memory read counts
    it as zero_targets, and its order facts name grain's plan and targets."""
    rng = np.random.default_rng(window * 10 + bins)
    lengths = np.concatenate([rng.integers(1, 3 * window, 60), [window, window, 1]])
    ids = [(np.arange(length) + 1000 * index).astype(np.int32) for index, length in enumerate(lengths)]
    roles = [np.full(length, index % 4, np.int32) for index, length in enumerate(lengths)]
    chunks = [{"text": tokens[start:start + window], "text_roles": role[start:start + window]}
              for tokens, role in zip(ids, roles, strict=True) for start in range(0, len(tokens), window)]
    reference = list(FirstFitPackIterDataset(
        grain.MapDataset.source(chunks).to_iter_dataset(), length_struct=dict.fromkeys(FIELDS, window),
        num_packing_bins=bins, shuffle_bins=False, max_sequences_per_bin=segments))
    targets = [int(np.count_nonzero((segment[1:] == segment[:-1]) & (segment[1:] != 0)))
               for segment in (packed["text_segment_ids"] for packed in reference)]
    targeted = [packed for packed, count in zip(reference, targets, strict=True) if count]

    def pipeline(source):
        return Pipeline(source).flat_map(Chunks(window, fields=FIELDS)).pack(
            window, fields=FIELDS, bins=bins, segments=segments)

    def windows(source, read):
        reader = validation(pipeline(source), batch=1, read=read, loading=READ)(DataPartition())
        try:
            return list(reader)
        finally:
            reader.close()

    documents = Memory([{"text": tokens, "text_roles": role}
                        for tokens, role in zip(ids, roles, strict=True)])
    stated = Memory({"text": ids, "text_roles": roles, "length": lengths.tolist()})
    for source, read in ((documents, "memory"), (stated, "indexed"), (documents, "indexed")):
        read_windows = windows(source, read)
        assert len(read_windows) == len(targeted)
        for index, (got, expected) in enumerate(zip(read_windows, targeted, strict=True)):
            laid = got["text_coordinates"]
            for value, key in ((got["text"], "text"), (got["text_roles"], "text_roles"),
                               (laid.segments, "text_segment_ids"), (laid.positions, "text_positions")):
                np.testing.assert_array_equal(value[0], expected[key], err_msg=f"{read} window {index} {key}")
    data = compile(pipeline(documents).batch(1), read="memory", loading=READ)
    (counted,) = data.counts.values()
    assert counted.accepted == len(targeted)
    assert counted.rejected["zero_targets"] == len(reference) - len(targeted)
    # The plan as grain followed it: the chunks in the order it added them, by their first token,
    # and where each window starts in that order.
    first = {int(piece["text"][0]): index for index, piece in enumerate(chunks)}
    order = [first[int(packed["text"][at])] for packed in reference
             for at in np.flatnonzero((packed["text_positions"] == 0) & (packed["text_segment_ids"] != 0))]
    starts = np.cumsum([0, *(int(packed["text_segment_ids"].max()) for packed in reference)]).tolist()
    assert data.plan.derived["0/packing"] == fingerprint([order, starts])
    assert data.plan.derived["0/valid_targets"] == fingerprint(sum(targets))


@pytest.mark.parametrize("policy", ["first-fit", "best-fit", "concat"])
def test_a_memory_read_packs_by_the_policy_an_indexed_read_packs_by(policy):
    """read='memory' packs the whole corpus before the order and an indexed
    read packs it as it streams. Both follow the stage's policy, so a pass
    over either is the same windows, segment ids and positions in the same
    order; first-fit, which has its own plan, is the control."""
    lengths = (3, 7, 2, 5, 8, 4, 6, 2, 8, 3)
    documents = [{"text": (np.arange(length) + 100 * index).astype(np.int32)}
                 for index, length in enumerate(lengths)]
    pipeline = Pipeline(Memory(documents)).pack(8, bins=3, policy=policy)

    def windows(read):
        reader = validation(pipeline, batch=2, read=read, loading=READ)(DataPartition())
        try:
            # The windows' arrays: each read declares its own bound on the documents a window holds
            # (`Coordinates.max_segments`), the memory read the most it packed into one and the
            # indexed read, packing as it streams, its window's length.
            return [[np.asarray(value).tolist() for value in jax.tree.leaves(batch)] for batch in reader]
        finally:
            reader.close()

    memory = windows("memory")
    assert memory and memory == windows("indexed")


@pytest.mark.parametrize("read", ["indexed", "streamed"])
def test_a_window_with_no_target_is_left_out_whatever_the_read(read):
    """A window that holds one token alone has nothing to predict; a memory read leaves it out,
    and a read that packs as it streams leaves it out too, so the steps a run takes do not change
    with how it reads."""
    documents = Memory([{"text": np.asarray([1], np.int32)}, {"text": np.asarray([2, 3], np.int32)}])
    pipeline = Pipeline(documents).pack(4, segments=1)

    def windows(read):
        reader = validation(pipeline, batch=1, read=read, loading=READ)(DataPartition())
        try:
            return [np.asarray(batch["text"]).tolist() for batch in reader]
        finally:
            reader.close()

    memory = windows("memory")
    assert memory == [[[2, 3, 0, 0]]]
    assert windows(read) == memory
