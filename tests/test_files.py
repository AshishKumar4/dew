"""A file written through dew._files is the old one or the new one, never part of one."""

import concurrent.futures
import inspect
import os
import stat
import sys

import pytest

from dew._files import published, replacing, staged, swapped, write_atomically


def test_a_write_that_fails_leaves_the_old_file_and_no_temporary(tmp_path):
    target = tmp_path / "run.json"
    target.write_text("old")
    with pytest.raises(RuntimeError), replacing(target) as temporary:
        temporary.write_text("half of the new")
        raise RuntimeError("crash")
    assert target.read_text() == "old"
    assert list(tmp_path.iterdir()) == [target]


def test_concurrent_writers_of_one_file_each_publish_it_whole(tmp_path):
    """Each thread writes its own megabyte and every write succeeds; the file
    that remains is one of them entire, and no writer's temporary is left
    beside it."""
    target = tmp_path / "ledger.json"
    contents = [bytes([index]) * 2**20 for index in range(8)]
    with concurrent.futures.ThreadPoolExecutor(len(contents)) as pool:
        writes = [pool.submit(write_atomically, target, content) for content in contents]
    for write in writes:
        write.result()
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
    with pytest.raises(RuntimeError), staged(target) as staging:
        (staging / "weights").write_text("half")
        raise RuntimeError("crash")
    assert list(tmp_path.iterdir()) == []


def test_what_is_published_has_the_mode_a_plain_write_gives(tmp_path):
    """Under the usual umask a new file is 0644 and a directory 0755, as
    open and mkdir make them, and a rewrite keeps the mode the file had."""
    previous = os.umask(0o022)
    try:
        write_atomically(tmp_path / "run.json", "{}")
        write_atomically(tmp_path / "ledger.json", "{}")
        with staged(tmp_path / "entry") as staging:
            (staging / "weights").write_text("w")
        os.chmod(tmp_path / "run.json", 0o640)
        write_atomically(tmp_path / "run.json", "{}")
        (tmp_path / "plain").write_text("{}")
    finally:
        os.umask(previous)
    modes = {path.name: stat.S_IMODE(path.stat().st_mode) for path in tmp_path.iterdir()}
    assert modes == {"run.json": 0o640, "ledger.json": 0o644, "entry": 0o755, "plain": 0o644}


def test_a_file_whose_name_is_the_longest_the_filesystem_allows_is_published(tmp_path):
    """The temporary beside a target is named apart from the target's own name, so a target whose name
    is as long as the directory allows (NAME_MAX) is still written, and so is such a directory."""
    longest = os.pathconf(tmp_path, "PC_NAME_MAX")
    write_atomically(tmp_path / ("f" * longest), "whole")
    assert (tmp_path / ("f" * longest)).read_text() == "whole"
    with staged(tmp_path / ("d" * longest)) as staging:
        (staging / "inside").write_text("whole")
    assert (tmp_path / ("d" * longest) / "inside").read_text() == "whole"


def test_a_built_directory_published_second_leaves_the_first_whole(tmp_path):
    """Two writers build the same directory apart and publish it by one rename each: the first's
    stands whole, and the second's is removed, not merged into it or moved inside it."""
    directory = tmp_path / "published"
    for writer in ("first", "second"):
        built = tmp_path / f"built-{writer}"
        built.mkdir()
        (built / "who").write_text(writer)
        assert published(built, directory) == (writer == "first")
        assert not built.exists()
    assert sorted(path.name for path in directory.iterdir()) == ["who"]
    assert (directory / "who").read_text() == "first"


@pytest.mark.parametrize("path", ["../escape.json", "/etc/passwd", "a/../../b", "a/./b", "a//b", "a\\..\\b",
                                  "C:/windows/file", "."])
def test_every_reader_of_outside_paths_refuses_one_out_of_its_directory_alike(path):
    """A path out of the directory it is read against, or written in a form that could reach out of it (a
    `.` or empty part, a backslash, a drive), is refused alike by every reader of outside input: a saved
    run's tokenizer files, a DDUF archive's entries, a source's subfolder and a sandboxed program's files,
    and a store's child, each through the one test of it (`dew._files.inside`). Their rules differed: only
    a tokenizer's refused `.`, empty parts, backslashes and drives, so a fix to one left the others
    weaker."""
    from dew._files import inside
    from dew.data.storage import Store
    from dew.data.text import confined_files
    from dew.interop.dduf import _contained
    from dew.interop.sources import SourceReference
    from dew.rl.sandbox import Program

    assert not inside(path)
    for read in (lambda: confined_files({path: "0" * 64}), lambda: _contained({path: (0, 1)}, "archive"),
                 lambda: SourceReference("org/name", subfolder=path),
                 lambda: Program(command=("python",), files={path: ""}),
                 lambda: Store("/tmp").child(path)):
        with pytest.raises(ValueError):
            read()
    assert inside("transformer/config.json") and inside("unet/") and not inside("unet/", directory=False)


def _nothing_foreign(written, held, directory):
    return set()


def test_a_relative_destination_is_swapped_into_its_real_parent(tmp_path, monkeypatch):
    """A destination named relative to the working directory is published as its absolute name would be:
    its sibling made in its real parent (train 1383's review: `.` made a sibling of an empty name); the
    working directory itself, or one holding it, is refused."""
    monkeypatch.chdir(tmp_path)
    with swapped("release", _nothing_foreign) as staging:
        (staging / "model.safetensors").write_text("its own")
    assert (tmp_path / "release" / "model.safetensors").read_text() == "its own"
    # The working directory itself, or one holding it, is refused before anything is written: its
    # replacement would leave the process in a folder removed (train 1399's review).
    (tmp_path / "release" / "inner").mkdir()
    for working, destination in ((tmp_path / "release", "."), (tmp_path / "release" / "inner", "..")):
        monkeypatch.chdir(working)
        with pytest.raises(ValueError), swapped(destination, _nothing_foreign) as staging:
            (staging / "model.safetensors").write_text("new")
        assert (tmp_path / "release" / "model.safetensors").read_text() == "its own"
        assert os.getcwd() == str(working)


def test_a_private_destination_stays_private_through_its_staging(tmp_path):
    """A tree swapped over a `0700` directory is written in a staging no other user reads and takes the
    destination's mode; one into a directory not yet made takes the mode a new directory takes under the
    umask (train 1383's review: staging at 0755 exposed weights, and replaced 0700 with 0755)."""
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    (private / "model.safetensors").write_text("old")
    with swapped(private, _nothing_foreign) as staging:
        assert stat.S_IMODE(staging.stat().st_mode) == 0o700
        (staging / "model.safetensors").write_text("new")
    assert stat.S_IMODE(private.stat().st_mode) == 0o700
    probe = tmp_path / "probe"
    probe.mkdir()
    with swapped(tmp_path / "fresh", _nothing_foreign) as staging:
        (staging / "model.safetensors").write_text("new")
    assert stat.S_IMODE((tmp_path / "fresh").stat().st_mode) == stat.S_IMODE(probe.stat().st_mode)


def test_an_empty_destination_is_replaced_without_an_exchange(tmp_path):
    """A destination made empty beforehand (a staging `push_to_hub` or a pooled export makes) is replaced by
    one rename where the filesystem exchanges no directories; one holding a tree is refused there, kept
    (train 1383's review)."""
    def no_exchange(staging, directory):
        raise OSError(38, "this system exchanges no directories in one call")

    empty = tmp_path / "empty"
    empty.mkdir()
    with swapped(empty, _nothing_foreign, exchange=no_exchange) as staging:
        (staging / "model.safetensors").write_text("new")
    assert (empty / "model.safetensors").read_text() == "new"
    with pytest.raises(OSError), swapped(empty, _nothing_foreign, exchange=no_exchange) as staging:
        (staging / "model.safetensors").write_text("newer")
    assert (empty / "model.safetensors").read_text() == "new"


def test_publishers_of_one_destination_validate_and_replace_it_in_turn(tmp_path):
    """Two Dew publishers of one destination take its validation and replacement one at a time: one held
    between its validation and its exchange keeps the other from validating the tree it is about to
    replace (train 1383's review: a publisher's exchange displaced a sidecar another had published after
    its validation)."""
    import threading

    from dew._files import exchange_directories

    destination = tmp_path / "release"
    destination.mkdir()
    (destination / "model.safetensors").write_text("first")
    events: list[str] = []
    paused, resume = threading.Event(), threading.Event()

    def validated(written, held, directory):
        events.append(f"{threading.current_thread().name} validates")
        return set()

    def exchanged(staging, directory):
        if threading.current_thread().name == "first":
            paused.set()
            assert resume.wait(30)
        exchange_directories(staging, directory)
        events.append(f"{threading.current_thread().name} exchanged")

    def publish(text):
        with swapped(destination, validated, exchange=exchanged) as staging:
            (staging / "model.safetensors").write_text(text)

    first = threading.Thread(target=publish, args=("one",), name="first")
    second = threading.Thread(target=publish, args=("two",), name="second")
    first.start()
    assert paused.wait(30)
    second.start()
    second.join(1)
    # The second publisher waits on the first's publication, not validating the tree it replaces.
    assert second.is_alive() and "second validates" not in events
    resume.set()
    first.join(30)
    second.join(30)
    assert events.index("first exchanged") < events.index("second validates")
    assert (destination / "model.safetensors").read_text() == "two"


def test_a_release_whose_name_is_the_longest_the_filesystem_allows_is_swapped(tmp_path):
    """A release named as long as the filesystem allows is staged, locked and published: its publishers'
    lock is named by a digest of its name, not its name (train 1383's review)."""
    name = "r" * os.pathconf(tmp_path, "PC_NAME_MAX")
    with swapped(tmp_path / name, _nothing_foreign) as staging:
        (staging / "model.safetensors").write_text("new")
    assert (tmp_path / name / "model.safetensors").read_text() == "new"


def test_a_staged_directory_is_private_until_its_publication(tmp_path):
    """A directory written through `staged` is no other user's to read while it is written (mode 0700) and
    takes the mode a folder made beside it takes when it is published (train 1399's review: a cache's
    shards were readable under the staging name as they were written)."""
    probe = tmp_path / "probe"
    probe.mkdir()
    with staged(tmp_path / "release") as staging:
        assert stat.S_IMODE(staging.stat().st_mode) == 0o700
        (staging / "model.safetensors").write_text("new")
    assert stat.S_IMODE((tmp_path / "release").stat().st_mode) == stat.S_IMODE(probe.stat().st_mode)


def test_a_private_file_stays_private_through_its_replacement(tmp_path):
    """A release re-exported over one whose weights a user made private (`0600` in a `0755` folder) keeps
    them private: each file the new tree writes by an earlier one's name takes its mode, a new file its
    umask's (train 1406's review: the replacement exposed them `0644`)."""
    release = tmp_path / "release"
    release.mkdir(mode=0o755)
    (release / "model.safetensors").write_text("old")
    (release / "model.safetensors").chmod(0o600)
    probe = tmp_path / "probe"
    probe.write_text("")
    with swapped(release, _nothing_foreign) as staging:
        (staging / "model.safetensors").write_text("new")
        (staging / "config.json").write_text("{}")
    assert stat.S_IMODE((release / "model.safetensors").stat().st_mode) == 0o600
    assert stat.S_IMODE((release / "config.json").stat().st_mode) == stat.S_IMODE(probe.stat().st_mode)


def test_a_lock_that_is_no_regular_file_is_refused_without_waiting(tmp_path):
    """A FIFO where a destination's publication lock is, which an open for writing would wait on a reader
    for, is refused at once, the destination as it was (train 1406's review)."""
    import hashlib

    from dew._files import PUBLISHING_LOCK

    release = tmp_path / "release"
    os.mkfifo(tmp_path / (PUBLISHING_LOCK + hashlib.sha256(b"release").hexdigest()[:16]))
    with pytest.raises(FileExistsError), swapped(release, _nothing_foreign) as staging:
        (staging / "model.safetensors").write_text("new")
    assert not release.exists()


def test_a_read_only_folder_replaced_leaves_no_tree_behind(tmp_path):
    """A release whose folder was read-only (`0555`), replaced by one that writes that folder: the new
    folder takes its mode and the earlier tree, put aside by the exchange, is removed whole, every folder
    made writable first (train 1407's review: the earlier tree was left as a `.staging` sibling)."""
    release = tmp_path / "release"
    (release / "text_encoder").mkdir(parents=True)
    (release / "text_encoder" / "model.safetensors").write_text("old")
    (release / "text_encoder").chmod(0o555)
    with swapped(release, _nothing_foreign) as staging:
        (staging / "text_encoder").mkdir()
        (staging / "text_encoder" / "model.safetensors").write_text("new")
    assert stat.S_IMODE((release / "text_encoder").stat().st_mode) == 0o555
    assert (release / "text_encoder" / "model.safetensors").read_text() == "new"
    # The release and its publication lock alone: nothing of the earlier tree or the staging.
    left = sorted(path.name for path in tmp_path.iterdir() if not path.name.startswith(".dew-lock-"))
    assert left == ["release"], left


def test_a_build_published_second_is_removed_whole_where_its_owner_cannot_search_it(tmp_path):
    """A build that loses its publication to another writer, each folder of it readable to its owner but
    not searchable (`0400`), a file and a folder in each, is removed whole: each folder is opened to its
    owner before its entries are looked up (train 1430's review: the first lookup failed, and the build
    was left beside the release)."""
    directory = tmp_path / "published"
    directory.mkdir()
    (directory / "who").write_text("first")
    built = tmp_path / "built"
    (built / "inner").mkdir(parents=True)
    (built / "who").write_text("second")
    (built / "inner" / "weights").write_text("second")
    (built / "inner").chmod(0o400)
    built.chmod(0o400)
    assert not published(built, directory)
    assert not built.exists()
    assert (directory / "who").read_text() == "first"


def test_a_build_deeper_than_pythons_stack_is_removed_whole_when_published_second(tmp_path):
    """A build that loses its publication to another writer, nested deeper than this process's call stack
    allows (its recursion limit lowered to 150 frames past the test's own), is removed whole: its folders
    are walked on a stack of descriptors, not Python's (theme 74's review: a recursive walk raised
    RecursionError out of the cleanup)."""
    directory = tmp_path / "published"
    directory.mkdir()
    (directory / "who").write_text("first")
    built = tmp_path / "built"
    built.mkdir()
    folder = os.open(built, os.O_RDONLY)
    try:
        for _ in range(300):
            os.mkdir("d", dir_fd=folder)
            inner = os.open("d", os.O_RDONLY, dir_fd=folder)
            os.close(folder)
            folder = inner
    finally:
        os.close(folder)
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(len(inspect.stack()) + 150)
    try:
        assert not published(built, directory)
    finally:
        sys.setrecursionlimit(limit)
    assert not built.exists()


def test_a_release_takes_the_group_of_the_tree_it_replaces_as_the_lock_finds_it(tmp_path):
    """A publisher that began against no tree, which another published while it wrote, publishes in the
    group and setgid inheritance of the tree it replaces as its publication lock finds it, every entry of
    it (audit 2026-11-27 P05: the group was chosen from the view at its start)."""
    groups = sorted({*os.getgroups(), os.getgid()} - {os.getgid()})
    other = groups[0] if groups else (os.getgid() + 1 if os.geteuid() == 0 else None)
    if other is None:
        pytest.skip("this process can give no group but its own")
    destination = tmp_path / "release"
    with swapped(destination, _nothing_foreign) as staging:
        (staging / "nested").mkdir()
        (staging / "nested" / "model.safetensors").write_text("ours")
        # Another publisher's tree, in another group its folders pass on (setgid), published while this one
        # is written.
        destination.mkdir()
        (destination / "model.safetensors").write_text("theirs")
        os.chown(destination, -1, other)
        destination.chmod(0o2755)
    published = [destination, *destination.rglob("*")]
    assert {path.lstat().st_gid for path in published} == {other}
    assert destination.stat().st_mode & (destination / "nested").stat().st_mode & stat.S_ISGID


def test_a_component_in_a_group_of_its_own_keeps_it_and_its_access_when_replaced(tmp_path):
    """A release whose component folder and weights are in a group of their own, the folder setgid and the
    weights private to that group, replaced whole: the component's folder and weights keep that group and
    that access, not the release root's group (train 1414's review: only the modes were kept, so the
    root's group could read the weights)."""
    groups = sorted({*os.getgroups(), os.getgid()} - {os.getgid()})
    other = groups[0] if groups else (os.getgid() + 1 if os.geteuid() == 0 else None)
    if other is None:
        pytest.skip("this process can give no group but its own")
    destination = tmp_path / "release"
    component = destination / "text_encoder"
    component.mkdir(parents=True)
    (component / "model.safetensors").write_text("before")
    for path in (component, component / "model.safetensors"):
        os.chown(path, -1, other)
    component.chmod(0o2750)
    (component / "model.safetensors").chmod(0o640)
    with swapped(destination, _nothing_foreign) as staging:
        (staging / "text_encoder").mkdir()
        (staging / "text_encoder" / "model.safetensors").write_text("after")
        (staging / "text_encoder" / "config.json").write_text("{}")
    published = [component, component / "model.safetensors", component / "config.json"]
    assert {path.lstat().st_gid for path in published} == {other}
    assert stat.S_IMODE(component.stat().st_mode) == 0o2750
    assert stat.S_IMODE((component / "model.safetensors").stat().st_mode) == 0o640


def test_a_new_release_takes_the_setgid_its_parent_holds_as_the_lock_finds_it(tmp_path):
    """A new release whose parent became setgid in another group while it was written publishes as a
    folder made there then would be, setgid and in that group, every entry of it (Theme 70's review: its
    setgid bit was the parent's before the write)."""
    groups = sorted({*os.getgroups(), os.getgid()} - {os.getgid()})
    other = groups[0] if groups else (os.getgid() + 1 if os.geteuid() == 0 else None)
    if other is None:
        pytest.skip("this process can give no group but its own")
    parent = tmp_path / "shared"
    parent.mkdir()
    destination = parent / "release"
    with swapped(destination, _nothing_foreign) as staging:
        (staging / "nested").mkdir()
        (staging / "nested" / "model.safetensors").write_text("ours")
        os.chown(parent, -1, other)
        parent.chmod(0o2775)
    published = [destination, *destination.rglob("*")]
    assert {path.lstat().st_gid for path in published} == {other}
    assert destination.stat().st_mode & (destination / "nested").stat().st_mode & stat.S_ISGID
