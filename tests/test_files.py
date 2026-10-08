"""A file written through dew.files is the old one or the new one, never part of one."""

import threading

import pytest

from dew.files import replacing, staged, write_atomically


def test_a_write_that_fails_leaves_the_old_file_and_no_temporary(tmp_path):
    target = tmp_path / "run.json"
    target.write_text("old")
    with pytest.raises(RuntimeError, match="crash"), replacing(target) as temporary:
        temporary.write_text("half of the new")
        raise RuntimeError("crash")
    assert target.read_text() == "old"
    assert list(tmp_path.iterdir()) == [target]


def test_concurrent_writers_of_one_file_each_publish_it_whole(tmp_path):
    """Each thread writes its own megabyte; the file that remains is one of
    them entire, and no writer's temporary is left beside it."""
    target = tmp_path / "ledger.json"
    contents = [bytes([index]) * 2**20 for index in range(8)]
    threads = [threading.Thread(target=write_atomically, args=(target, content)) for content in contents]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert target.read_bytes() in contents
    assert list(tmp_path.iterdir()) == [target]


def test_a_directory_is_published_by_the_writer_that_finishes_first_and_kept_whole(tmp_path):
    """Two writers stage one cache entry; the one whose block ends first
    publishes it, the later one's is dropped, and no staging is left."""
    target = tmp_path / "entry"
    with staged(target) as later, staged(target) as sooner:
        (later / "weights").write_text("later")
        (sooner / "weights").write_text("sooner")
    assert (target / "weights").read_text() == "sooner"
    assert list(tmp_path.iterdir()) == [target]


def test_a_directory_whose_block_fails_is_not_published(tmp_path):
    target = tmp_path / "entry"
    with pytest.raises(RuntimeError, match="crash"), staged(target) as staging:
        (staging / "weights").write_text("half")
        raise RuntimeError("crash")
    assert list(tmp_path.iterdir()) == []
