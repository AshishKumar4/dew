"""The data layer's distributed and resume contract, one cell at a time.

Each cell is a read mode (memory, indexed, streamed), with or without a cache
the pool fills, a filter that rejects rows and online packing
(`multiprocess_worker.matrix_data`). A pool of two trains it for three steps
and saves; pools of one, two and four processes resume that checkpoint for
three more, and each lands where a pool of its own size that nobody stopped
lands: every process reads, after the resume, the batches that pool's process
read after its third step, and ends at the same data position with the same
parameters.
A cell compiled to one global order resumes on any process count; a per-share
cell keeps each share's rows, so it resumes on its own count and refuses
another.

Run the matrix alone with `pytest tests/test_data_matrix.py`; `-k` picks cells.
"""

import shutil

import pytest
from process_support import run_pool
from test_multiprocess import assert_same_parameters, dumped_params

pytestmark = [pytest.mark.mesh, pytest.mark.distributed]

STEPS = 3
CELLS = [
    f"{read},{cache},{rejecting},{packed}"
    for read in ("memory", "indexed", "streamed")
    for cache in ("nocache", "cache")
    for rejecting in ("all", "rejecting")
    for packed in ("rows", "packed")
]


@pytest.mark.parametrize("cell", CELLS)
def test_a_saved_position_resumes_where_an_uninterrupted_pool_lands(tmp_path, cell):
    saved = tmp_path / "saved" / "run"
    first = run_pool("matrix", tmp_path / "first", 2, run_dir=saved, cell=cell, name="matrix",
                     steps=STEPS, save_every=STEPS)
    elastic = first[0]["lowering"] == "global"
    for processes in (1, 2, 4):
        directory = tmp_path / f"resumed{processes}" / "run"
        shutil.copytree(saved, directory)
        if not elastic and processes != 2:
            with pytest.raises(AssertionError, match="cannot be translated to another share"):
                run_pool("matrix", tmp_path / f"resumed{processes}-out", processes, run_dir=directory,
                         cell=cell, name="matrix", steps=2 * STEPS)
            continue
        resumed = run_pool("matrix", tmp_path / f"resumed{processes}-out", processes, run_dir=directory,
                           cell=cell, name="matrix", steps=2 * STEPS)
        whole = run_pool("matrix", tmp_path / f"whole{processes}-out", processes,
                         run_dir=tmp_path / f"whole{processes}" / "run", cell=cell, name="matrix",
                         steps=2 * STEPS)
        for index, report in enumerate(resumed):
            assert report["restored_step"] == STEPS
            assert report["step"] == 2 * STEPS
            assert report["batches"][:STEPS] == whole[index]["batches"][STEPS:2 * STEPS]
            assert report["dataset_state"] == whole[index]["dataset_state"]
            name = f"process{index}.json"
            assert_same_parameters(dumped_params(tmp_path / f"resumed{processes}-out" / name),
                                   dumped_params(tmp_path / f"whole{processes}-out" / name))
