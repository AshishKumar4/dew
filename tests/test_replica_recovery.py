"""Which replica gives a lost local checkpoint each of its shards (`dew.checkpoints._sources`), read
from where a local save recorded its arrays (`_grid`): the pool's lifecycle is
tests/test_multiprocess.py's (`test_a_process_that_lost_its_local_copy_recovers_it_from_a_replica`)."""

import numpy as np
import pytest

from dew.checkpoints import _covered, _sources

GRID = {"axes": ["data", "fsdp"], "shape": [2, 4], "devices": list(range(8)),
        "split": {"['params']['w']": ["fsdp"], "['step']": []}}
"""Eight devices, a data axis of two over two processes (devices 0 to 3 on process 0, 4 to 7 on
process 1), fsdp of four within each."""
PROCESS = {device: device // 4 for device in range(8)}


@pytest.mark.parametrize(("split", "holders", "sources"), [
    (["fsdp"], {0}, [[True] * 4, [False] * 4]),
    (["fsdp"], {1}, [[False] * 4, [True] * 4]),
    ([], {1}, [[False] * 4, [True, False, False, False]]),
    (["fsdp"], {0, 1}, [[True] * 4, [False] * 4]),
    (["data", "fsdp"], {0}, None),
], ids=["fsdp-from-process0", "fsdp-from-process1", "replicated-from-process1", "both-hold", "unreplicated"])
def test_each_shard_is_given_by_its_first_copy_on_a_process_that_holds_the_step(split, holders, sources):
    """An array split over fsdp has a copy of each shard on each data row: the holder's row gives
    them. A replicated array's one source is the first device of the holders. An array split over
    both axes has no copy of the lost process's shards, and none is chosen."""
    found = _sources(GRID, split, frozenset(holders), PROCESS)
    if sources is None:
        assert found is None
    else:
        assert found is not None and found.tolist() == sources and np.all(found.sum(axis=0) <= 1)


def test_a_save_whose_shards_have_no_copy_off_the_lost_process_is_not_covered():
    """The grid's arrays split over fsdp have copies across the data axis, so a process holding the
    step covers the other; one array split over both axes has none, and neither does a save that
    recorded no grid (an array placed otherwise than by name), so the pool resumes elsewhere."""
    assert _covered(GRID, frozenset({0}), PROCESS)
    unreplicated = {**GRID, "split": {**GRID["split"], "['params']['x']": ["data", "fsdp"]}}
    assert not _covered(unreplicated, frozenset({0}), PROCESS)
    assert not _covered(None, frozenset({0}), PROCESS)
