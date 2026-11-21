"""A snapshot (`Pipeline.snapshot`): one pass over a chain's records sealed as an immutable generation that
`Files(url)` reads back, by the read the chain asks for, provisioning it, and with its random steps' draws."""
import threading

import numpy as np
import pytest

from dew.data.sources import Files, Memory


@pytest.mark.parametrize("read", [None, "memory", "indexed"])
def test_snapshot_uses_requested_or_preferred_read_and_loading(tmp_path, read):
    import time

    from torch.utils.data import Dataset

    from dew.data import from_torch

    lock = threading.Lock()
    active = maximum = 0

    class Counted(Dataset):
        def __len__(self):
            return 32

        def __getitem__(self, index):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            try:
                time.sleep(0.01)
                return {"id": index}
            finally:
                with lock:
                    active -= 1

    chain = from_torch(Counted(), identity="snapshot@1").map(lambda row: {"id": row["id"] * 2}).threads(2)
    if read is not None:
        chain = chain.read(read)
    destination = tmp_path / "snapshot"
    manifest = chain.snapshot(str(destination), rows_per_shard=7)
    stored = Files(str(destination)).records("indexed")
    assert [int(stored[index]["id"]) for index in range(len(stored))] == list(range(0, 64, 2))
    assert manifest["rows"] == 32
    assert maximum <= 2


def test_snapshot_provisions_the_requested_indexed_read(tmp_path):
    input = tmp_path / "rows.jsonl"
    input.write_text('{"id":0}\n{"id":1}\n')
    files = Files(str(input))
    assert not files.supports("indexed")
    destination = tmp_path / "snapshot"
    files.read("indexed").threads(1).snapshot(str(destination))
    assert Files(str(input)).supports("indexed")
    stored = Files(str(destination)).records("indexed")
    assert [stored[index] for index in range(len(stored))] == [{"id": 0}, {"id": 1}]


def test_a_snapshot_seals_a_random_steps_draws_and_a_publication_holds_none(tmp_path):
    """A snapshot of a chain with a random step (a video's clip drawn per record, say) draws once under its
    seed and seals the rows it drew, which read back as drawn, its seed in the manifest's metadata; the
    same seed seals the same rows, another seed others. A publication, whose cache a later pass writes
    through, still refuses the draw (row 9056)."""
    from dew.data.compile import iterate
    from dew.data.pipeline import Pipeline

    chain = Pipeline(Memory({"id": np.arange(16)}, identity="clips")).random_map(
        lambda row, rng: {**row, "clip": int(rng.integers(0, 1000))}, identity="clip@1")

    def sealed(name: str, seed: int) -> tuple[list[tuple[int, int]], dict]:
        manifest = chain.snapshot(str(tmp_path / name), seed=seed)
        stored = Files(str(tmp_path / name)).records("indexed")
        return [(int(stored[at]["id"]), int(stored[at]["clip"])) for at in range(len(stored))], manifest

    drawn, manifest = sealed("first", 7)
    assert [row for row, _ in drawn] == list(range(16)) and manifest["metadata"]["seed"] == 7
    assert sealed("again", 7)[0] == drawn
    assert sealed("other", 8)[0] != drawn
    with pytest.raises(ValueError):
        iterate(chain)
