"""A kernel's machine code as nvdisasm lists it (`report.binaries` reads it where it decodes a binary): the
disassembler, a cubin's listing, the roles a spill stands in, the registers its roles reconfigure to, whether
it issues tensor-memory ops, and what it breaks of the SASS rules."""

import importlib.util
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


def disassembler() -> Path | None:
    """NVIDIA's nvdisasm, which the SASS rules read a kernel's machine code through (`sass_listing`): the
    nvdisasm wheel's (`nvidia/cu13/bin`), which the `ci` extra installs in every lock the suite runs on, or
    the PATH's; None where neither is installed, a user's install among them."""
    spec = importlib.util.find_spec("nvidia")
    for root in (spec.submodule_search_locations or ()) if spec else ():
        if (found := Path(root) / "cu13" / "bin" / "nvdisasm").is_file():
            return found
    found = shutil.which("nvdisasm")
    return None if found is None else Path(found)


def sass_listing(elf: bytes) -> str | None:
    """nvdisasm's listing of the machine code of the cubin `elf`; None where no disassembler is installed
    (`disassembler`)."""
    tool = disassembler()
    if tool is None:
        return None
    with tempfile.NamedTemporaryFile(suffix=".cubin") as cubin:
        cubin.write(elf)
        cubin.flush()
        listing = subprocess.run([str(tool), "-c", cubin.name], capture_output=True, text=True, check=True)
    return listing.stdout


PRODUCER, CONSUMERS = "producer", "consumers"
"""A warpgroup-specialized kernel's roles, as its SASS places them (`spilled_roles`)."""


def spilled_roles(listing: str) -> tuple[str, ...]:
    """The roles whose code a kernel's spill stands in, from nvdisasm's `listing`: each local-memory store
    and load (STL, LDL) the role of the last setmaxnreg before it, ptxas laying each role's branch out after
    its own, the producer's drop (`USETMAXREG.DEALLOC`) or the consumers' raise (`USETMAXREG.TRY_ALLOC`); one
    before either, a prologue's, neither role's (row 6744: a B200's decayed keys spilled after its producer's
    drop to the least, an H100's output walk between the consumers' raise and the producer's drop)."""
    roles, role = set(), None
    for line in listing.splitlines():
        if not (found := _INSTRUCTION.search(line)):
            continue
        opcode = _parsed(found.group(2))[0]
        if opcode.startswith("USETMAXREG"):
            role = PRODUCER if ".DEALLOC" in opcode else CONSUMERS
        elif role is not None and opcode.split(".")[0] in ("STL", "LDL"):
            roles.add(role)
    return tuple(sorted(roles))


_RECONFIGURED = re.compile(r"\bUSETMAXREG\.(DEALLOC|ALLOC|TRY_ALLOC)\S*\s+(?:UP\d+,\s*)?(0x[0-9a-f]+|\d+)")


def uses_tmem(listing: str) -> bool:
    """Whether nvdisasm's `listing` issues Blackwell tensor-memory ops (`LDTM`, `STTM`, `UTC*`)."""
    return any(_parsed(found.group(2))[0].split(".")[0].startswith(("LDTM", "STTM", "UTC"))
               for line in listing.splitlines() if (found := _INSTRUCTION.search(line)))


def reconfigured(listing: str) -> tuple[int | None, int | None]:
    """The registers a kernel's machine code (nvdisasm's `listing`) lowers its producer to and raises its
    consumers to: its first USETMAXREG deallocation's and allocation's (a try's too) counts, as its PTX's
    `setmaxnreg.dec` and `.inc` state them; None for one it makes none of."""
    found: dict[str, int] = {}
    for kind, count in _RECONFIGURED.findall(listing):
        found.setdefault("dec" if kind == "DEALLOC" else "inc", int(count, 0))
    return found.get("dec"), found.get("inc")


_INSTRUCTION = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;")
WARPGROUP_MMAS = frozenset({"HGMMA", "QGMMA", "IGMMA"})
"""Hopper's warpgroup MMA opcodes, by their operands: 16-bit floats, 8-bit floats (fp8) and integers."""


def _dead_mma(instruction: str) -> bool:
    """Whether a SASS instruction is a warpgroup MMA writing no register (its destination RZ): what ptxas
    leaves of a kernel's MMAs whose results it found dead, the products never read (row 4346: ptxas 12.9 folds
    an integer op on an accumulator through its value before the MMA, so a negated product read stores
    -0.0). Read from the instruction's opcode and first operand, a predicate guard aside."""
    words = instruction.split()
    if words and words[0].startswith("@"):
        words = words[1:]
    return len(words) > 1 and words[0].split(".")[0] in WARPGROUP_MMAS and words[1].rstrip(",") == "RZ"
_LABEL = re.compile(r"^\s*(\.L_\w+):")
_WATERFALL = re.compile(r"\bBRA\.U\.ANY\s+`?\(?(\.L_\w+)\)?`?")
_BRANCH = re.compile(r"\bBRA(?:\.\S+)?\s+`?\(?(\.L_\w+)\)?`?")
_REGISTER = re.compile(r"R(\d+)")
REGISTER_A = 4
"""The registers a thread holds a warpgroup MMA's register operand A in (its 64-row fragment's: eight 16-bit
values, or sixteen 8-bit)."""


def _parsed(instruction: str) -> tuple[str, list[str]]:
    """A SASS instruction's opcode and its operands, its predicate (`@P0`) apart."""
    words = instruction.split()
    if words and words[0].startswith("@"):
        words = words[1:]
    return (words[0] if words else ""), " ".join(words[1:]).split(", ")


def _written(opcode: str, operands: list[str]) -> set[int]:
    """The registers a SASS instruction writes: its destination, the first operand, and the ones after it
    that its width takes (a 128-bit load's or a four-matrix `LDSM`'s four, a 64-bit one's, a two-matrix
    `LDSM`'s or a `.WIDE` product's two)."""
    if not (target := _REGISTER.fullmatch(operands[0])):
        return set()
    matrices = opcode.startswith("LDSM")
    pair = ".64" in opcode or ".WIDE" in opcode or opcode.startswith("CS2R") or (matrices and ".2" in opcode)
    count = 4 if ".128" in opcode or (matrices and ".4" in opcode) else 2 if pair else 1
    return set(range(int(target.group(1)), int(target.group(1)) + count))


def _predicated(instruction: str) -> bool:
    """Whether a SASS instruction runs under a predicate other than the always-true one (`@P0`, `@!P1`;
    not `@PT`)."""
    first = instruction.split(maxsplit=1)[0] if instruction.strip() else ""
    return first.startswith("@") and first.lstrip("@!") != "PT"


def _reused_register_a(instructions: list[str], labels: dict[str, int]) -> int:
    """The writes in a loop (a backward branch's body, walked over two trips) to the registers of a warpgroup
    MMA's register operand A whose group is still pending: PTX leaves an operand A read until a
    `wgmma.wait_group` retires its group (PTX ISA, "wgmma.mma_async"), so a write before it is undefined
    behaviour, the product reading whichever value its warp reached (row 5416: a prologue's next step
    rewrote A under `wait_group 1`, its forward differing run to run on a GH200). A predicated wait
    (`@P0 WARPGROUP.DEPBAR.LE`) retires nothing on the walk, which takes no predicate as known."""
    reused = 0
    for at, text in enumerate(instructions):
        head = labels.get(branch.group(1)) if (branch := _BRANCH.search(text)) else None
        if head is None or head > at:
            continue
        body = [(*_parsed(instruction), instruction) for instruction in instructions[head:at + 1]]
        if not any(opcode.split(".")[0] in WARPGROUP_MMAS for opcode, _, _ in body):
            continue
        committed: list[set[int]] = []
        issuing: set[int] = set()
        for _ in range(2):
            for opcode, operands, instruction in body:
                if opcode.split(".")[0] in WARPGROUP_MMAS:
                    if len(operands) > 1 and (left := _REGISTER.fullmatch(operands[1])):
                        issuing |= set(range(int(left.group(1)), int(left.group(1)) + REGISTER_A))
                    if "gsb0" in instruction:
                        committed, issuing = [*committed, issuing], set()
                elif opcode == "WARPGROUP.DEPBAR.LE" and not _predicated(instruction):
                    pending = int(operands[-1], 0)
                    committed = committed[len(committed) - pending:] if pending else []
                else:
                    reused += bool(_written(opcode, operands) & set().union(issuing, *committed))
    return reused


def sass_findings_of(listing: str) -> tuple[str, ...]:
    """What nvdisasm's `listing` of a kernel breaks of the kernel challenge's SASS rules (row 4071, its
    harness/ptxas checks; notes/ptxas-re.md §2): an elect waterfall (a backward BRA.U.ANY) holding an R2UR, a
    copy issued lane by lane because ptxas could not prove its operand uniform; a warpgroup MMA writing
    no register (`_dead_mma`), a product ptxas dropped as never read (row 4346); and a loop writing a
    pending warpgroup MMA's register operand A (`_reused_register_a`, row 5416)."""
    instructions, labels, pending = [], {}, []
    for line in listing.splitlines():
        if label := _LABEL.match(line):
            pending.append(label.group(1))
        elif found := _INSTRUCTION.search(line):
            labels.update(dict.fromkeys(pending, len(instructions)))
            pending = []
            instructions.append(" ".join(found.group(2).split()))
    serialized = 0
    for at, text in enumerate(instructions):
        head = labels.get(branch.group(1)) if (branch := _WATERFALL.search(text)) else None
        if head is not None and head <= at:
            serialized += any("R2UR" in held for held in instructions[head:at + 1])
    dead = sum(1 for text in instructions if _dead_mma(text))
    reused = _reused_register_a(instructions, labels)
    found = []
    if serialized:
        found.append(f"{serialized} elect waterfall holding an R2UR, a copy issued lane by lane")
    if dead:
        found.append(f"{dead} warpgroup MMA writing no register, its products dropped as never read")
    if reused:
        found.append(f"{reused} write to a warpgroup MMA's register operand A before its group retired")
    return tuple(found)
