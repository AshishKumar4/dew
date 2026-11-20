"""What keys a program in JAX's persistent cache, as two processes compile it: a build where no GPU is visible
and the run on the GPU it built for (rows 8372, 8299). Each process records every compile's key parts, as
JAX's cache logs them, with the options Dew compiled it under (`report.xla_compiled`) and the file JAX dumped
its text to (`JAX_DUMP_IR_TO`), each read on the thread that compiled it; a program the run missed the cache
for is named with the parts no compile of its name in the build shares, and where it is the text or the
options, how they differ."""

import logging
import re
from pathlib import Path


class Keyed(logging.Handler):
    """Each compile JAX keys its persistent cache by, as it logs it: its module's name, whether the cache held
    it, the hash of each part of its key (the computation, the backend's version, the compile options, the
    accelerator's config, ...), the options Dew compiled it under, and the file its text was dumped to,
    gathered by the thread that compiled it (each logs its dump, then its key, then its lookup). Its records
    are taken one at a time under the handler's own lock (`logging.Handler.handle`)."""

    PART = re.compile(r"get_cache_key hash of serialized (.+): ([0-9a-f]+)$")
    READ = re.compile(r"(Persistent compilation cache hit|PERSISTENT COMPILATION CACHE MISS) for '(.+)' "
                      r"with key")
    OPTIONS = re.compile(r"a compile's options: (.*)$")
    DUMPED = re.compile(r"Dumped the module to (.+)\.$")

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.programs: list[list] = []
        self.parts: dict[int, dict[str, str]] = {}
        self.options: dict[int, str] = {}
        self.dumped: dict[int, str] = {}

    def emit(self, record):
        message, thread = record.getMessage(), record.thread
        if options := self.OPTIONS.match(message):
            self.options[thread] = options[1]
        elif dumped := self.DUMPED.match(message):
            self.dumped[thread] = dumped[1]
        elif part := self.PART.match(message):
            self.parts.setdefault(thread, {})[part[1]] = part[2]
        elif read := self.READ.match(message):
            self.programs.append([read[2], read[1].startswith("Persistent"), self.parts.pop(thread, {}),
                                  self.options.pop(thread, None), self.dumped.pop(thread, None)])


def keyed() -> Keyed:
    """A `Keyed` listening to JAX's cache and Dew's compiles from here on."""
    handler = Keyed()
    for name in ("jax._src.cache_key", "jax._src.compiler", "dew.kernels"):
        logging.getLogger(name).setLevel(logging.DEBUG)
        logging.getLogger(name).addHandler(handler)
    return handler


def text(directory: Path, name: str | None) -> list[str]:
    """The lines of the text JAX dumped for a compile to `name` in `directory` (the process's
    `JAX_DUMP_IR_TO`, its log naming the file alone), its locations dropped: the computation its key hashes;
    none where it dumped none."""
    if name is None:
        return []
    dumped = (directory / name).read_text()
    return re.sub(r"\s*loc\([^)]*\)|^#loc.*$", "", dumped, flags=re.M).splitlines()


def unread(ran: list, ran_dumps: Path, prepared: list, prepared_dumps: Path) -> dict[str, dict]:
    """Each compile the run missed the cache for, by its module's name: the parts of its key no compile of
    that name the build made shares (all of them where the build compiled none of that name), and where
    the computation is among them, the first line its text and the build's nearest differ at (each read from
    its side's dumps, `text`); where the compile options are, the options each compiled under."""
    built: dict[str, list] = {}
    for module, _, parts, options, dumped in prepared:
        built.setdefault(module, []).append((parts, options, dumped))
    missed = {}
    for module, held, parts, options, dumped in ran:
        if held:
            continue
        others = built.get(module, [])
        differing = sorted(part for part, value in parts.items()
                           if all(other.get(part) != value for other, _, _ in others))
        said: dict = {"parts": differing}
        if "computation" in differing:
            said["text"] = first_difference(text(ran_dumps, dumped),
                                            [text(prepared_dumps, theirs) for _, _, theirs in others])
        if "compile_options" in differing:
            said["options"] = {"run": options, "build": sorted({held for _, held, _ in others}, key=str)}
        missed[module] = said
    return missed


def first_difference(mine: list[str], built: list[list[str]]) -> list[str] | None:
    """The first line at which a run's text and the build's text nearest it (the one agreeing longest)
    differ, each side's; None where either holds none."""
    if not mine or not any(built):
        return None

    def agreed(theirs: list[str]) -> int:
        return next((index for index, (a, b) in enumerate(zip(mine, theirs, strict=False)) if a != b),
                    min(len(mine), len(theirs)))

    theirs = max(built, key=agreed)
    at = agreed(theirs)
    return [line[:300] for line in (mine[at:at + 1] + theirs[at:at + 1])]
