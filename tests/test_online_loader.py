"""URL fetches use the ordinary pipeline worker and rejection paths."""

import itertools
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dew.data import DataPartition, Loading, Memory, Pipeline, Rejected, compile
from dew.data.compile import iterate
from dew.data.mix import SearchBudget
from dew.data.sources.urls import Fetch
from dew.data.steps.image import DecodeImage, Resize
from dew.objectives.base import VALID_ROWS

READ = Loading(workers=0, threads=1, read_buffer=2, worker_buffer=1)


def catalog(root: Path, count: int = 16) -> Pipeline:
    rows = []
    for index in range(count):
        path = root / f"image-{index}.png"
        Image.fromarray(np.full((8, 8, 3), index + 1, np.uint8)).save(path)
        rows.append({"url": path.as_uri(), "caption": f"sample {index}", "label": index})
    return Pipeline(Memory(rows)).map(Fetch(output="image", retries=0), DecodeImage(), Resize(8))


def test_fetch_preserves_the_catalog_and_reads_exact_bytes(tmp_path):
    path = tmp_path / "payload"
    path.write_bytes(b"source bytes")
    row = Fetch(retries=0)({"url": path.as_uri(), "caption": "the record"})
    assert not isinstance(row, Rejected)
    assert row["bytes"] == b"source bytes" and row["caption"] == "the record"


def test_failed_fetches_are_counted_before_batches(tmp_path):
    prefix = catalog(tmp_path)
    rows = [{"url": (tmp_path / "missing").as_uri(), "label": -1},
            *prefix.source.records("indexed")]
    pipeline = Pipeline(Memory(rows)).shuffle(7).repeat().map(
        Fetch(output="image", retries=0), DecodeImage(), Resize(8)).batch(4)
    data = compile(pipeline, read="indexed", loading=READ)
    stream = data.train(DataPartition())
    try:
        batches = list(itertools.islice(stream, 8))
    finally:
        stream.close()
    assert all(np.all(batch["label"] >= 0) for batch in batches)
    assert all(np.array_equal(batch["image"][:, 0, 0, 0], batch["label"] + 1) for batch in batches)


def test_a_transient_failure_is_not_a_permanent_tombstone(tmp_path):
    result = Fetch(retries=0)({"url": (tmp_path / "missing").as_uri()})
    assert isinstance(result, Rejected)
    assert result.reason == "transient" and not result.permanent


def test_an_oversized_object_is_rejected_before_decode(tmp_path):
    path = tmp_path / "oversized"
    path.write_bytes(b"x" * 17)
    result = Fetch(retries=0, max_bytes=16)({"url": path.as_uri()})
    assert isinstance(result, Rejected) and result.reason == "too_large"


def test_an_all_rejected_catalog_exhausts_its_search_budget(tmp_path):
    pipeline = Pipeline(Memory([{"url": (tmp_path / "missing").as_uri()}])).repeat().map(
        Fetch(output="image", retries=0), DecodeImage()).batch(4)
    data = compile(pipeline, read="indexed", loading=READ,
                   budget=SearchBudget(consecutive=8, seconds=2))
    stream = data.train(DataPartition())
    try:
        with pytest.raises(ValueError, match="accepted|rejection|budget"):
            next(stream)
    finally:
        stream.close()


@pytest.mark.parametrize("workers", [0, 2])
def test_real_workers_fetch_pixels_without_losing_validation_rows(tmp_path, workers):
    prefix = catalog(tmp_path, 13)
    data = compile(prefix.shuffle(3).repeat().batch(4), validation=prefix.batch(4),
                   read="indexed", loading=Loading(workers=workers, threads=1, read_buffer=2,
                                                   worker_buffer=1))
    assert data.val is not None
    stream = data.val(DataPartition())
    try:
        batches = list(stream)
    finally:
        stream.close()
    labels = np.concatenate([batch["label"][batch.get(VALID_ROWS, np.ones(4, bool))] for batch in batches])
    assert labels.tolist() == list(range(13))
    for batch in batches:
        np.testing.assert_array_equal(batch["image"][:, 0, 0, 0], batch["label"] + 1)


def test_a_finite_fetch_pass_ends_after_its_catalog(tmp_path):
    rows = list(iterate(catalog(tmp_path, 3), read="indexed", loading=READ))
    assert [row["label"] for row in rows] == [0, 1, 2]
    assert all(np.all(row["image"] == index + 1) for index, row in enumerate(rows))
