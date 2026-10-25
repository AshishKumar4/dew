"""The data layer's distributed and resume contract, one cell at a time.

Each cell is a read mode (memory, indexed, streamed), with or without a cache
the pool fills, a filter that rejects rows and online packing
(`multiprocess_worker.matrix_data`). A pool of one, two or four processes
trains it for three steps and saves; a pool of the same size resumes that
checkpoint for three more and lands where a pool of its size that nobody
stopped lands: every process reads, after the resume, the batches that
pool's process read after its third step, and ends at the same data
position with the same parameters, bit for bit.
A cell compiled to one global order resumes on any process count, so the
checkpoint also resumes on another count, there on that count's run's
parameters; a per-share cell keeps each share's rows, so it resumes on its
own count and refuses every other.

The rows are a covering set, not the whole product: every pair of settings
(read mode, cache, filter, packing, the count that saves and the other count
the save is resumed on or refused) meets in some row, and the boundary rows
hold the indexed read beside each single step that turns its global order
into a per-share one. A pool that nobody stopped is run once per cell and
count and read by every row that needs it.

Run the matrix alone with `pytest tests/test_data_matrix.py`; `-k` picks rows.
"""

import functools
import shutil

import pytest
from matrix_support import published_matrix
from process_support import run_pool
from test_multiprocess import assert_identical_parameters, assert_same_parameters, dumped_params

from dew.data.dataset import DataPartition
from dew.training import Checkpoints

pytestmark = [pytest.mark.mesh, pytest.mark.distributed]

STEPS = 3
ROWS = [
    # Every pair of settings meets in one of these ten: read mode, cache, filter, packing, the count
    # that saves and the other count the save resumes on or is refused.
    ("memory,cache,all,rows", 2, 1),
    ("memory,cache,rejecting,rows", 1, 4),
    ("memory,nocache,all,packed", 4, 1),
    ("memory,nocache,rejecting,rows", 1, 2),
    ("indexed,cache,all,rows", 1, 4),
    ("indexed,cache,rejecting,packed", 4, 2),
    ("indexed,nocache,rejecting,packed", 2, 1),
    ("streamed,cache,all,packed", 1, 2),
    ("streamed,nocache,all,packed", 2, 4),
    ("streamed,nocache,rejecting,rows", 4, 1),
    # The indexed read's global order, elastic, and each single step that makes it per-share.
    ("indexed,nocache,all,rows", 2, 4),
    ("indexed,nocache,all,rows", 4, 1),
    ("indexed,nocache,rejecting,rows", 2, 1),
    ("indexed,nocache,all,packed", 1, 2),
    # The filtered and the packed read made global by preparation: their rows, and the plan of their
    # windows, published once and read as a generation.
    ("indexed,published,rejecting,rows", 2, 4),
    ("indexed,published,rejecting,packed", 4, 1),
]


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    """The flags a pool that trains `cell` takes: for a published cell, the cache its generation was
    published to once here, in the test, as a preparation job would, which every pool training the cell
    reads (saving, resuming and uninterrupted alike)."""
    @functools.cache
    def flags(cell: str) -> dict[str, str]:
        _, cached, rejecting, packed = cell.split(",")
        if cached != "published":
            return {}
        directory = tmp_path_factory.mktemp("prepared")
        published_matrix(rejecting == "rejecting", packed == "packed", directory)
        return {"prepared": str(directory)}

    return flags


@pytest.fixture(scope="module")
def uninterrupted(tmp_path_factory, prepared):
    """A pool of `processes` that trains `cell` for six steps and nobody stops: its reports and
    where it wrote them, run once a cell and count."""
    @functools.cache
    def run(cell: str, processes: int):
        directory = tmp_path_factory.mktemp("whole")
        reports = run_pool("matrix", directory / "out", processes, run_dir=directory / "run", cell=cell,
                           name="matrix", steps=2 * STEPS, **prepared(cell))
        return reports, directory / "out"

    return run


@pytest.mark.parametrize("cell, saved_by, other", ROWS,
                         ids=[f"{cell}-saved-by-{saved}-then-{other}" for cell, saved, other in ROWS])
def test_a_saved_position_resumes_where_an_uninterrupted_pool_lands(tmp_path, uninterrupted, prepared, cell,
                                                                    saved_by, other):
    saved = tmp_path / "saved" / "run"
    first = run_pool("matrix", tmp_path / "first", saved_by, run_dir=saved, cell=cell, name="matrix",
                     steps=STEPS, save_every=STEPS, **prepared(cell))
    elastic = first[0]["lowering"] == "global"
    # A published generation is the preparation that makes a filtered or packed read global.
    assert elastic or ",published," not in cell
    for processes in (saved_by, other):
        directory = tmp_path / f"resumed{processes}" / "run"
        shutil.copytree(saved, directory)
        if not elastic and processes != saved_by:
            # The checkpoint resumes the shares that saved it and refuses another count's.
            checkpoints = Checkpoints(str(saved / "matrix"))
            checkpoints.restore(share=DataPartition(0, saved_by))
            with pytest.raises(ValueError):
                checkpoints.restore(share=DataPartition(0, processes))
            with pytest.raises(AssertionError):
                run_pool("matrix", tmp_path / f"resumed{processes}-out", processes, run_dir=directory,
                         cell=cell, name="matrix", steps=2 * STEPS, **prepared(cell))
            continue
        resumed = run_pool("matrix", tmp_path / f"resumed{processes}-out", processes, run_dir=directory,
                           cell=cell, name="matrix", steps=2 * STEPS, **prepared(cell))
        whole, written = uninterrupted(cell, processes)
        # On the count that saved it a resume reduces in the order the run it continues did, so it lands
        # on that run's parameters bit for bit; on another count it reduces otherwise.
        same = assert_identical_parameters if processes == saved_by else assert_same_parameters
        for index, report in enumerate(resumed):
            assert report["restored_step"] == STEPS
            assert report["step"] == 2 * STEPS
            assert report["batches"][:STEPS] == whole[index]["batches"][STEPS:2 * STEPS]
            assert report["dataset_state"] == whole[index]["dataset_state"]
            name = f"process{index}.json"
            same(dumped_params(tmp_path / f"resumed{processes}-out" / name), dumped_params(written / name))
