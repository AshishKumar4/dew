"""Packed windows against grain's own packers: `first_fit` plans as
`grain.experimental.FirstFitPackIterDataset` packs, and a memory read packs by
the stage's policy as an indexed read does."""

import grain
import numpy as np
import pytest
from grain.experimental import FirstFitPackIterDataset

from dew.data import Loading, Memory, text
from dew.data.compile import validation
from dew.data.dataset import DataPartition
from dew.data.pipeline import Pipeline
from dew.data.tokens import PackedWindows

READ = Loading(workers=0, threads=1, read_buffer=0, worker_buffer=1)


@pytest.mark.parametrize("window, bins", [(8, 1), (8, 3), (16, 4), (5, 2)])
def test_packed_windows_are_grains_first_fit_windows(window, bins):
    """Documents from one token to well past the window, a few exactly the
    window and one empty, with ids and roles: grain's packer, given the
    chunks `text.chunk` cuts explicitly in file order and its bins emitted in
    order, writes window for window what `PackedWindows` reads by index,
    the ids, the roles and the segment ids and positions. Grain writes a
    segment-id and position pair per feature; the roles' pair is the ids'
    pair, which is why Dew writes the one."""
    rng = np.random.default_rng(window * 10 + bins)
    lengths = np.concatenate([rng.integers(1, 3 * window, 60), [window, window, 0, 1]])
    documents = [{"text": (np.arange(length) + 1000 * index).astype(np.int32),
                  "text_roles": np.full(length, index % 4, np.int32)} for index, length in enumerate(lengths)]
    chunking = text.chunk(window, fields=("text", "text_roles"))
    chunks = [{name: part[name] for name in ("text", "text_roles")}
              for document in documents for part in chunking.flat_map(document) if len(part["text"])]
    source = grain.MapDataset.source(chunks)
    packed = PackedWindows(source, [len(chunk["text"]) for chunk in chunks], window, bins, "corpus")
    reference = list(FirstFitPackIterDataset(
        source.to_iter_dataset(),
        length_struct={"text": window, "text_roles": window}, num_packing_bins=bins, shuffle_bins=False))
    assert len(packed) == len(reference)
    for index, expected in enumerate(reference):
        got = packed[index]
        for key in ("text", "text_roles", "text_segment_ids", "text_positions"):
            np.testing.assert_array_equal(got[key], expected[key], err_msg=f"window {index} {key}")
        for key in ("segment_ids", "positions"):
            np.testing.assert_array_equal(expected[f"text_roles_{key}"], expected[f"text_{key}"])


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
            return [{key: np.asarray(value).tolist() for key, value in batch.items()} for batch in reader]
        finally:
            reader.close()

    memory = windows("memory")
    assert memory and memory == windows("indexed")
