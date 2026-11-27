"""Writing a file so that a reader finds the old one or the new one, never part of one.

A run's record, a sweep's ledger, a rung, exported weights and their index
are all read by other processes or a later run, some while the writer is
still running. `replacing` hands out a temporary path beside the target,
unique to this write, and renames it over the target once the block ends with
its bytes flushed to disk, so a crash at any point leaves either file whole.
`write_atomically` is that for data in memory, and `staged` is it for a
directory, which a cache publishes once. A location with a scheme (`gs://`,
`s3://`) is written in place: its store makes an object visible whole or not
at all, and has no rename to stage one with. `stat_identity` is what such a
cache keys a source file by.
"""

from __future__ import annotations

import contextlib
import ctypes
import errno
import fcntl
import functools
import hashlib
import operator
import os
import secrets
import shutil
import stat
from collections.abc import Callable, Collection, Iterator, Mapping
from pathlib import Path

from etils import epath


def is_uri(path: str) -> bool:
    """Whether a path names a `<scheme>://` location, such as a gs:// bucket, which a store makes visible
    whole and which has no rename to stage a write with."""
    return "://" in path


def inside(path: str, *, directory: bool = True) -> bool:
    """Whether `path` names a place within the directory it is read against: relative, in forward slashes,
    no part `.`, `..` or empty and no drive; a trailing slash, a directory's, admitted where `directory`,
    else refused, `path` a file's. The one test of it every reader of outside input applies (a saved run's
    tokenizer files, `dew.data.text.confined_files`; a DDUF archive's entries; a source's subfolder and its
    code's files; a sandboxed program's files; a store's child, `dew.data.storage.Store.child`), so a path
    one admits none refuses."""
    parts = path.split("/")
    if directory and parts[-1:] == [""] and len(parts) > 1:
        parts = parts[:-1]
    return not ("\\" in path or ":" in parts[0] or any(part in ("", ".", "..") for part in parts))


def stat_identity(path: str | os.PathLike) -> tuple[str, int, int]:
    """A file's resolved path, size and modification time in ns, which a
    conversion cache keys it by in place of a hash of its bytes.

    Hashing the contents would read the whole file on every cache hit, the
    cost the cache exists to avoid. A Hub file resolves to its
    content-addressed blob, so for a published file the key is a content
    key. A local file relies on its mtime: rewriting it in place at the
    same size within one mtime tick, or `cp -p` of another file of the
    same size over it, reuses the stale conversion; deleting the cache's
    directory under `dew._builds.dew_cache_dir()` clears it."""
    status = os.stat(path)
    return str(Path(path).resolve()), status.st_size, status.st_mtime_ns


@contextlib.contextmanager
def replacing(path: str | os.PathLike) -> Iterator[epath.Path]:
    """A path to write the new contents of `path` at, published over it when
    the block ends and removed if the block raises.

    The temporary sits in `path`'s directory, so the rename never crosses a
    filesystem, and is created for this write alone, so concurrent writers of
    one target (threads, processes) never share it. The file published has
    the mode a plain write gives: the target's own, or a new file's under the
    umask. A location with a scheme yields `path` itself.
    """
    if is_uri(str(path)):
        yield epath.Path(path)
        return
    target = Path(path)
    temporary = epath.Path(_beside(target, ".tmp"))
    os.close(os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666))
    try:
        yield temporary
        with open(temporary, "rb+") as written:
            os.fsync(written.fileno())
        with contextlib.suppress(FileNotFoundError):
            os.chmod(temporary, stat.S_IMODE(os.stat(target).st_mode))
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def write_atomically(path: str | os.PathLike, contents: str | bytes) -> None:
    """Write `contents`, text or bytes, as the whole of `path` (`replacing`)."""
    with replacing(path) as temporary:
        if isinstance(contents, str):
            temporary.write_text(contents)
        else:
            temporary.write_bytes(contents)


@contextlib.contextmanager
def staged(directory: Path) -> Iterator[Path]:
    """A directory to write the contents of `directory` into, renamed to it
    when the block ends and removed if the block raises.

    A `directory` another writer published first is kept, whole, and this
    write dropped: every writer publishes by the one rename. The directory
    is private (mode 0700) while it is written, so no other user reads a
    file of it before its publication, and takes the mode a folder made
    beside it takes under the umask when it is published (`_made_mode`).
    """
    directory.parent.mkdir(parents=True, exist_ok=True)
    # Read before anything is written: a probe made once the write is done could find no inode left.
    mode = _made_mode(directory.parent)
    staging = _beside(directory, "")
    os.mkdir(staging, 0o700)
    try:
        yield staging
        staging.chmod(mode)
        published(staging, directory)
    finally:
        _removed(staging)


def published(built: Path, directory: Path) -> bool:
    """Publish the directory `built`, written on `directory`'s filesystem, as `directory` by one
    rename, whether this call published it. A `directory` another writer published first is kept,
    whole, and `built` removed: every writer of it publishes by the one rename (`staged`)."""
    try:
        os.replace(built, directory)
    except OSError:
        if not directory.is_dir():
            raise
        _removed(built)
        return False
    return True


_RENAME_EXCHANGE = 2
_AT_FDCWD = -100


def exchange_directories(staging: Path, directory: Path) -> None:
    """Exchange the directories `staging` and `directory` in one call, `staging` then holding the earlier
    one; refused, both as they were, where the kernel or the filesystem has no such exchange."""
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        exchange = libc.renameat2
    except AttributeError:
        raise OSError(errno.ENOSYS, f"this system exchanges no directories in one call: {directory} is "
                                    "replaced only so; export into a directory not yet made") from None
    if exchange(_AT_FDCWD, os.fsencode(staging), _AT_FDCWD, os.fsencode(directory), _RENAME_EXCHANGE):
        code = ctypes.get_errno()
        raise OSError(code, f"{directory} could not be exchanged with its new export in one call: "
                            f"{os.strerror(code)}")


Entries = Mapping[Path, str]
"""A tree's entries (`_entries`): each by its path relative to the tree's root, with its kind, `"file"` (a
regular file), `"folder"`, `"link"` or `"special"` (a FIFO, a socket, a device)."""


def same_entries(written: Entries, held: Entries) -> dict[Path, set[Path]]:
    """Each file and folder `written` where `held` holds the same entry, of the same kind: the earlier entry
    it replaces (`swapped`'s `replaces`)."""
    return {entry: {entry} for entry, kind in written.items()
            if kind in ("file", "folder") and held.get(entry) == kind}


@contextlib.contextmanager
def swapped(directory: Path, foreign: Callable[[Entries, Entries, Path], set[Path]], *,
            exchange: Callable[[Path, Path], None] = exchange_directories,
            replaces: Callable[[Entries, Entries], Mapping[Path, Collection[Path]]] = same_entries
            ) -> Iterator[Path]:
    """A sibling directory to write the whole of `directory` into, which replaces it when the block ends
    in one call, so a write that fails at any file leaves the one `directory` held whole and the one
    that succeeds is put in place whole; removed if the block raises, or if its own setup fails.

    The publication is failure-safe, not a reader's snapshot: a path names the new tree from the exchange
    on, so a reader that opens a release's files one by one across a save may read files of both; a
    reader synchronizes with the saves it reads. `directory` is found by its real parent and its own name,
    a relative one as an absolute one, a final link left the link it is. A `directory` that does not exist
    yet, or is empty, is put in place by one rename; one that holds anything is exchanged with the new one
    in one call (Linux's `renameat2(RENAME_EXCHANGE)`), and refused, before anything moves, where the
    filesystem has no such exchange: two renames would leave the path missing between them. It is
    replaced only where `foreign` of the new tree's entries, its own (every file, folder and link, at any
    depth, a link not followed, each with its kind, `_entries`) and its path names none: what it holds that
    the new one neither writes nor owns, which is named. Refused before anything is written: one that is
    no directory of its own (a file, a link), one that is the process's working directory or holds it,
    whose replacement would leave the process in a folder removed, and one named as Dew's publishers'
    locks are (`PUBLISHING_LOCK`); and one whose tree cannot be read whole, when it is replaced.

    The new tree is made private from its first call (mode 0700), in the group and with the setgid bit
    of the one it replaces, so what is written in it takes that group, and published with the mode of the
    one it replaces, or the mode a new folder takes beside it under the umask (`_made_mode`), each file
    and folder it writes where the earlier tree held one by that name with that one's mode, so a private
    destination's weights, or a private file in it, are never readable by others. Dew's publishers of one
    destination take its validation and replacement in turn (`_publishing`, a lock file of its own beside
    it), so one cannot validate a tree another then replaces; a writer outside Dew is not excluded.
    The earlier one is removed once it is out of place. `exchange` is the one call that exchanges the two
    (`exchange_directories`, Linux's `renameat2`), a filesystem's own, which a test of a filesystem that
    has none passes in."""
    directory = _located(Path(directory))

    def existing() -> os.stat_result | None:
        """`directory`'s own stat where it exists, a directory of its own, else None: anything else
        there refused."""
        if not os.path.lexists(directory):
            return None
        if directory.is_symlink() or not directory.is_dir():
            raise FileExistsError(f"{directory} is no directory of its own, which an export replaces")
        return directory.stat()

    if directory.name.startswith(PUBLISHING_LOCK):
        raise ValueError(f"{directory} is named as Dew's publishers' locks are: export under another name")
    working = Path(os.getcwd())
    if working == directory or directory in working.parents:
        raise ValueError(f"{directory} is or holds the working directory, which its replacement would "
                         "leave removed: export from another")
    # Refused before the block writes anything, and again before the replacement, as the path is then.
    before = existing()
    directory.parent.mkdir(parents=True, exist_ok=True)
    # A new tree's mode read before anything is written: a probe once the write is done could find no
    # inode left.
    made = _made_mode(directory.parent)
    staging = _beside(directory, ".staging")
    os.mkdir(staging, 0o700)
    try:
        if before is not None:
            # The group and setgid inheritance of the tree it replaces, before anything is written in it.
            if staging.stat().st_gid != before.st_gid:
                os.chown(staging, -1, before.st_gid)
            staging.chmod(0o700 | (stat.S_IMODE(before.st_mode) & stat.S_ISGID))
        yield staging
        with _publishing(directory):
            held_at = existing()
            if held_at is None:
                staging.chmod(made)
                os.rename(staging, directory)
                return
            written, held = _entries(staging), _entries(directory)
            refused = foreign(written, held, directory)
            if refused:
                raise FileExistsError(f"{directory} holds {sorted(map(str, refused))}, which the export "
                                      "would not write: export into an empty directory, or one holding only "
                                      "an export")
            # Each entry the new tree writes in place of earlier ones (`replaces`: the same entry, or a
            # release's reshard of its weights) takes the access every one of them allowed, a private file
            # kept private, a new one its umask's (trains 1406 and 1407's reviews), while the staging is
            # still private; its own mode last, just before it is put in place.
            for entry, earlier in replaces(written, held).items():
                if earlier:
                    modes = (stat.S_IMODE((directory / each).lstat().st_mode) for each in earlier)
                    allowed = functools.reduce(operator.and_, modes)
                    (staging / entry).chmod(allowed)
            staging.chmod(stat.S_IMODE(held_at.st_mode))
            if held:
                exchange(staging, directory)
            else:
                # An empty directory is replaced by a rename, which every filesystem makes in one step.
                os.rename(staging, directory)
    finally:
        _removed(staging)


def _removed(tree: Path) -> None:
    """Remove `tree`, a tree of this process's own (a staging, the earlier tree an exchange put there, or a
    build another writer published before), every folder in it first made writable, as one whose mode an
    earlier tree's restored read-only is not: no folder of a save left behind (train 1407's review). The
    live tree is never walked."""
    if not os.path.lexists(tree) or os.path.islink(tree):
        return
    for here, _, _ in os.walk(tree, followlinks=False):
        Path(here).chmod(0o700)
    shutil.rmtree(tree, ignore_errors=True)


def _made_mode(parent: Path) -> int:
    """The mode a folder made in `parent` takes under the process's umask, the parent's setgid bit among
    it: read off one made and removed, the umask read nowhere else, so no other thread's is changed."""
    probe = _beside(parent / "probe", ".mode")
    probe.mkdir()
    try:
        return stat.S_IMODE(probe.stat().st_mode)
    finally:
        probe.rmdir()


def _located(directory: Path) -> Path:
    """`directory` by its real parent and its own name: a relative path made absolute, its parent's links
    resolved, and a final link left the link it is, which `swapped` refuses; a path ending in `.` or `..`,
    which names no entry of its own, resolved whole."""
    absolute = directory.absolute()
    if absolute.name in ("", ".", ".."):
        return absolute.resolve()
    return absolute.parent.resolve() / absolute.name


PUBLISHING_LOCK = ".dew-lock-"
"""The prefix of the file beside a release whose exclusive lock Dew's publishers of it take in turn
(`_publishing`): `.dew-lock-` and a digest of the release's name, short whatever the name's length, a name
no release takes (`swapped` refuses one)."""


@contextlib.contextmanager
def _publishing(directory: Path) -> Iterator[None]:
    """Hold Dew's lock of `directory`'s publication, an exclusive `flock` of its own file beside it (named
    by a digest of its name, `PUBLISHING_LOCK`), so its publishers validate and replace it one at a time and
    one destination's publication holds no other's. The file is opened for writing, which an exclusive lock
    takes on every filesystem (NFS's among them), no link followed to it: an existing one as it is (a
    sticky folder refuses creating over another user's), else made exclusively, as every file Dew writes is,
    under the umask, and left in place, the one inode every publisher locks. Whoever the umask lets write
    beside the release takes its lock; one it does not is refused opening it, not left waiting. The lock is
    keyed by the destination's name as given, so publishers of one release name it alike: two spellings a
    case-folding filesystem takes for one entry take two locks."""
    # Keyed by the name as given: publishers of one release name it alike; two spellings a case-folding
    # filesystem takes for one entry take two locks (row 13744).
    name = PUBLISHING_LOCK + hashlib.sha256(os.fsencode(directory.name)).hexdigest()[:16]
    lock = _lock_file(directory.parent / name)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield
    finally:
        os.close(lock)


def _lock_file(path: Path) -> int:
    """The lock file at `path` opened for writing alone, no link followed: made by this call exclusively,
    with write permission alone (under the umask), so only who may write beside the release locks it; else
    the one an earlier publication made, which is never removed, opened as it is, never created over (a
    sticky folder refuses that of another user's file)."""
    flags = os.O_WRONLY | os.O_NOFOLLOW
    try:
        return os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o222)
    except FileExistsError:
        # Opened without blocking and taken only as a regular file: a FIFO another made there, which an
        # open for writing would wait on a reader for, is refused (train 1406's review).
        try:
            lock = os.open(path, flags | os.O_NONBLOCK)
        except OSError as refused:
            if refused.errno != errno.ENXIO:
                raise
            raise FileExistsError(f"{path} is no regular file, which Dew's publishers lock") from refused
    try:
        if not stat.S_ISREG(os.fstat(lock).st_mode):
            raise FileExistsError(f"{path} is no regular file, which Dew's publishers lock")
    except BaseException:
        os.close(lock)
        raise
    return lock


def _entries(root: Path) -> dict[Path, str]:
    """Every entry of the tree at `root` (files, folders, and links of any kind, none followed), each by its
    path relative to `root`, with its kind (`Entries`); a tree read in part is no inventory of it, so a
    folder it cannot list fails the walk."""
    def unread(error: OSError) -> None:
        raise error

    def kind(path: Path) -> str:
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            return "link"
        if stat.S_ISDIR(mode):
            return "folder"
        return "file" if stat.S_ISREG(mode) else "special"

    entries = {}
    for here, directories, files in os.walk(root, onerror=unread, followlinks=False):
        base = Path(here).relative_to(root)
        for name in (*directories, *files):
            entries[base / name] = kind(Path(here) / name)
    return entries


def _beside(target: Path, suffix: str) -> Path:
    """A name in `target`'s directory that no other write uses, named apart from `target`'s own, so a
    target whose name is as long as the filesystem allows has a temporary too."""
    return target.with_name(f".dew-{secrets.token_hex(8)}{suffix}")
