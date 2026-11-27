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
import contextvars
import ctypes
import errno
import fcntl
import functools
import hashlib
import logging
import operator
import os
import secrets
import stat
from collections.abc import Callable, Iterator, Mapping
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


Replacements = list[tuple[list[Path], list[Path]]]
"""Groups of a new tree's entries, each with the earlier tree's entries they replace together (`swapped`'s
`replaces`)."""


def same_entries(written: Entries, held: Entries) -> Replacements:
    """Each file and folder `written` where `held` holds the same entry, of the same kind: the earlier entry
    it replaces (`swapped`'s `replaces`)."""
    return [([entry], [entry]) for entry, kind in written.items()
            if kind in ("file", "folder") and held.get(entry) == kind]


_STAGED: contextvars.ContextVar[tuple[Path, Path] | None] = contextvars.ContextVar("_STAGED", default=None)
"""The staging `swapped` is writing and the destination it replaces, while its block runs."""


def published_counterpart(path: Path) -> Path | None:
    """Where `path`, inside the staging `swapped` is writing, lies in the tree the staging replaces; None
    outside one. A writer that names its files apart from the earlier tree's reads it there
    (`safetensors_io.save_sharded`)."""
    staged = _STAGED.get()
    if staged is None:
        return None
    staging, directory = staged
    absolute = Path(os.path.abspath(path))
    return directory / absolute.relative_to(staging) if absolute.is_relative_to(staging) else None


@contextlib.contextmanager
def swapped(directory: Path, foreign: Callable[[Entries, Entries, Path], set[Path]], *,
            exchange: Callable[[Path, Path], None] = exchange_directories,
            replaces: Callable[[Entries, Entries], Replacements] = same_entries
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
    of the one it replaces, so what is written in it takes that group, and published, under the lock, as
    the destination is then (`_granted`): with the group and mode of the tree it replaces, or the mode a
    new folder takes beside it under the umask (`_made_mode`) with the group and setgid bit its parent
    gives one, each file and folder it writes where the earlier tree held one by that name with that one's
    group and mode, so a private destination's weights, or a private component's in a group of its own,
    are never readable by others. Dew's publishers of one
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
    # Whether this filesystem gives an entry its folder's group where the folder is not setgid (`grpid`),
    # read before anything is written, as the mode is.
    grpid = _keeps_folder_group(directory.parent)
    staging = _beside(directory, ".staging")
    os.mkdir(staging, 0o700)
    try:
        if before is not None:
            # The group and setgid inheritance of the tree it replaces, before anything is written in it.
            if staging.stat().st_gid != before.st_gid:
                os.chown(staging, -1, before.st_gid)
            staging.chmod(0o700 | (stat.S_IMODE(before.st_mode) & stat.S_ISGID))
        staged = _STAGED.set((Path(os.path.abspath(staging)), directory))
        try:
            yield staging
        finally:
            _STAGED.reset(staged)
        with _publishing(directory):
            held_at = existing()
            written = _entries(staging)
            held = {} if held_at is None else _entries(directory)
            refused = set() if held_at is None else foreign(written, held, directory)
            if refused:
                raise FileExistsError(f"{directory} holds {sorted(map(str, refused))}, which the export "
                                      "would not write: export into an empty directory, or one holding only "
                                      "an export")
            # The groups and access the published tree takes, bound under the lock to what the destination
            # is now (audit 2026-11-27 P05: a publisher that began before another's commit, or against no
            # tree, chose them from another view), while the staging is still private.
            mode = _granted(staging, directory, held_at, made, grpid, written, replaces(written, held))
            if held_at is None:
                staging.chmod(mode)
                os.rename(staging, directory)
                return
            staging.chmod(mode)
            if held:
                exchange(staging, directory)
            else:
                # An empty directory is replaced by a rename, which every filesystem makes in one step.
                os.rename(staging, directory)
    finally:
        _removed(staging)


def _removed(tree: Path) -> None:
    """Remove `tree`, a tree of a save's own (a staging, the earlier tree an exchange put there, or a build
    another writer published before), as far as this process may: a folder whose read-only mode keeps an
    entry in it (one an earlier tree's mode restored) made writable to remove it, where this process owns
    it, and no other touched, nor any failure raised, so a cleanup neither fails a save already published
    nor hides the error it unwinds from (train 1407's review). The live tree is never walked.

    Walked bottom-up through descriptors (`os.fwalk`, no link followed), each folder's entries removed, and
    a removal its mode refuses retried, through the very descriptor the walk holds of that folder: a folder
    or a link put in the tree's place meanwhile is neither changed nor followed (trains 1414 and 1423's
    reviews). The root's own parent, the release's, is never changed."""
    if not os.path.lexists(tree) or os.path.islink(tree):
        return
    log = logging.getLogger(__name__)

    def unread(error: OSError) -> None:
        log.warning("could not read %s of a save's discarded tree: %s", error.filename, error)

    try:
        for _, folders, files, held in os.fwalk(tree, topdown=False, onerror=unread, follow_symlinks=False):
            for name in (*files, *folders):
                _removed_in(held, name, log)
    except OSError as unwalked:
        # Its root opened by none of this process's rights (a mode an earlier tree's restored on it).
        unread(unwalked)
    try:
        os.rmdir(tree)
    except OSError as left:
        # A save's own leftover, named for whoever clears it; the save's outcome is its own.
        log.warning("could not remove a save's discarded tree %s: %s", tree, left)


def _removed_in(folder: int, name: str, log: logging.Logger) -> None:
    """Remove the entry `name` of the folder the descriptor `folder` holds (a folder, emptied already, or
    any other entry, a link not followed), its folder made writable to retry where its mode refuses the
    removal and this process owns it; a failure logged, not raised."""
    try:
        kind = os.stat(name, dir_fd=folder, follow_symlinks=False).st_mode
        remove = os.rmdir if stat.S_ISDIR(kind) else os.unlink
        try:
            remove(name, dir_fd=folder)
        except PermissionError:
            held = os.fstat(folder)
            # Only a folder of this process's own whose mode keeps the entry: no other touched.
            if held.st_uid != os.geteuid() or held.st_mode & 0o300 == 0o300:
                raise
            os.fchmod(folder, 0o700)
            remove(name, dir_fd=folder)
    except OSError as left:
        log.warning("could not remove %s from a save's discarded tree: %s", name, left)


def _granted(staging: Path, directory: Path, held_at: os.stat_result | None, made: int, grpid: bool | None,
             written: Entries, replacements: Replacements) -> int:
    """Give each entry of `staging` the group and access the tree published from it takes at `directory`,
    as the destination is now, and return the mode its root takes, which its caller sets last.

    The root takes the group, mode and setgid bit of the tree it replaces (`held_at`), or, new, a folder's
    made there (`made`, its mode under the umask) with the setgid bit and group its parent gives one now.
    An entry that replaces earlier ones (`replacements`) takes their group, which they share (one they do
    not, or one this process cannot give, refused, as `chown` refuses it, before anything moves), and the
    access every one of them allowed, so a component's private weights in a group of their own stay so
    (trains 1406, 1407 and 1414's reviews); any other the group an entry made in its folder as it publishes
    takes (its folder's where it is setgid or the filesystem is `grpid`, the publisher's otherwise), and a
    new folder the setgid bit of its own. Each folder's mode is set after its entries'."""
    earlier: dict[Path, tuple[int, int]] = {}
    for entries, replaced in replacements:
        if not replaced:
            continue
        found = [(directory / each).lstat() for each in replaced]
        groups = {each.st_gid for each in found}
        if len(groups) > 1:
            raise PermissionError(f"{directory}'s {sorted(map(str, replaced))}, which "
                                  f"{list(map(str, entries))} replace, are in the groups {sorted(groups)}, "
                                  "which no one entry keeps")
        group = next(iter(groups))
        allowed = functools.reduce(operator.and_, (stat.S_IMODE(each.st_mode) for each in found))
        earlier.update((entry, (group, allowed)) for entry in entries)
    egid = os.getegid()
    if held_at is None:
        parent = directory.parent.stat()
        setgid = parent.st_mode & stat.S_ISGID
        # The group a folder made in the parent takes now: its own where it is setgid or the filesystem
        # gives an entry its folder's (`grpid`), the publisher's otherwise.
        if not setgid and grpid is None and parent.st_gid != egid:
            raise PermissionError(f"{directory} is new in a folder of another group than the publisher's, "
                                  "which this filesystem gives it or not by a rule no probe of this process "
                                  "reads: publish it as a user of another group, or set the folder's setgid "
                                  "bit")
        root = (parent.st_gid if setgid or grpid else egid, (made & ~stat.S_ISGID) | setgid)
    else:
        root = (held_at.st_gid, stat.S_IMODE(held_at.st_mode))
    _grouped(staging, root[0])
    # Each folder's group and setgid bit as published, its entries following it (a folder before them).
    placed = {Path(): (staging.lstat().st_gid, root[1] & stat.S_ISGID)}
    # Each folder's mode, set once every entry below it is: a mode its earlier folder held that keeps even
    # its owner out (`0055`) set first would refuse its entries' (train 1423's review).
    folders: list[tuple[Path, int]] = []
    for entry in sorted(written, key=lambda entry: len(entry.parts)):
        path, kind = staging / entry, written[entry]
        group, setgid = placed[entry.parent]
        if entry in earlier:
            group, mode = earlier[entry]
        else:
            # A new entry the group an entry made in its folder as it publishes takes: the folder's where
            # the folder is setgid or the filesystem gives its folder's (`grpid`), the publisher's otherwise,
            # so no folder's group reads a new file its owner did not open to it (trains 1423 and theme 72's
            # reviews); stated, not the one the kernel gave it under the staging's own folders.
            if not setgid and grpid is None and group != egid:
                raise PermissionError(f"{entry} is new in a folder of another group than the publisher's, "
                                      "which this filesystem gives it or not by a rule no probe of this "
                                      "process reads: publish it as a user of another group, or set the "
                                      "folder's setgid bit")
            group = group if setgid or grpid else egid
            mode = (stat.S_IMODE(path.lstat().st_mode) & ~stat.S_ISGID) | setgid if kind == "folder" else None
        _grouped(path, group)
        if kind == "folder":
            held = path.lstat()
            mode = stat.S_IMODE(held.st_mode) if mode is None else mode
            folders.append((path, mode))
            placed[entry] = (held.st_gid, mode & stat.S_ISGID)
        elif kind != "link" and mode is not None and stat.S_IMODE(path.lstat().st_mode) != mode:
            path.chmod(mode)
    for path, mode in reversed(folders):
        if stat.S_IMODE(path.lstat().st_mode) != mode:
            path.chmod(mode)
    return root[1]


def _grouped(path: Path, group: int | None) -> None:
    """`path` (a link not followed) in `group` where it is in another, as it is where None (the kernel's)."""
    if group is not None and path.lstat().st_gid != group:
        os.chown(path, -1, group, follow_symlinks=False)


def _keeps_folder_group(parent: Path) -> bool | None:
    """Whether the filesystem at `parent` gives an entry made in a folder that is not setgid that folder's
    group (`grpid`, BSD's rule) rather than its maker's: read off a probe folder made there and removed, its
    setgid bit cleared and, where it is in the maker's group, given another group the maker may give (one
    of its own, or any where it is root), and a folder made in it. None where no other group is the
    maker's to give, whose answer no probe reads (`_granted` refuses only where it decides a group)."""
    egid = os.getegid()
    others = sorted(set(os.getgroups()) - {egid}) or ([egid + 1] if os.geteuid() == 0 else [])
    probe = _beside(parent / "probe", ".group")
    probe.mkdir(0o700)
    try:
        probe.chmod(0o700)
        if probe.stat().st_gid == egid:
            if not others:
                return None
            os.chown(probe, -1, others[0])
        inner = probe / "inner"
        inner.mkdir()
        return inner.stat().st_gid == probe.stat().st_gid
    finally:
        _removed(probe)


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
