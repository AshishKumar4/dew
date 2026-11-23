"""What a compiled program's kernels assembled to, read from the executable XLA serializes: each Mosaic GPU
and Pallas-Triton kernel's binary (`binaries`), its registers, stack frame, spills, shared memory and
ptxas's notes (`Binary`), and the thunks and custom kernels a run launches. A kernel's name carries the
launch that traced it (`LAUNCH_SUFFIX`, `report.named`), so a binary names its launch and its config.
Certification and the compile gate read them (`report.certified`, tests/kernel_compile.py)."""

import dataclasses
import math
import re
import struct
from collections.abc import Collection

import jax
from google.protobuf import empty_pb2
from google.protobuf.message import DecodeError
from google.protobuf.unknown_fields import UnknownFieldSet
from jax.extend import mosaic_gpu as mosaic_gpu_records
from jax.extend.mlir import ir

from dew.nn.kernels.templates.sass import (
    reconfigured,
    sass_findings_of,
    sass_listing,
    spilled_roles,
    uses_tmem,
)

LAUNCH_SUFFIX = re.compile(r"__l([0-9a-f]{10})(?:__(\d+))?$")
"""The suffix a kernel's name carries within the launch that traced it (`named`): the digest of its identity
there, its launch's digest where the launch states no kernel's part (`tuning.KernelOp.emitted`)."""
KERNEL_FUNCTION = re.compile(r"(?P<kernel>.*)_mosaic_gpu_kernel")
"""A Mosaic GPU kernel's function in its binary: its kernel's name, the host function's `_mosaic_gpu`
(jax/experimental/mosaic/gpu/core.py) and GPU outlining's `_kernel`."""


def kernel_of(name: str) -> tuple[str, str | None, str | None]:
    """`(kernel, identity, index)` of a kernel's `name` as it traced (`named`), its function in its binary
    or its PTX entry (`KERNEL_FUNCTION`) or its call's location: its name less its suffix, the suffix's
    digest (`LAUNCH_SUFFIX`), and any XLA uniquification index (`__<digits>`), None where absent."""
    function = KERNEL_FUNCTION.fullmatch(name)
    kernel = function.group("kernel") if function else name
    if not (launch := LAUNCH_SUFFIX.search(kernel)):
        return kernel, None, None
    return kernel[:launch.start()], launch.group(1), launch.group(2)


def launch_name(kernel: str, launch: str | None, index: str | None = None) -> str:
    """`kernel`'s name carrying the digest `launch` (`LAUNCH_SUFFIX`) and optional XLA uniquification
    `index`, as is where None, `kernel_of`'s inverse: the lossless key a kernel's evidence goes by beside its
    launch (row 12014: written in report.py, read here)."""
    named = kernel if launch is None else f"{kernel}__l{launch}"
    return named if index is None else f"{named}__{index}"


MOSAIC_TARGET = b"mosaic_gpu_v2"
"""The custom call target of a Mosaic GPU kernel, which its thunk names (`_kernels`)."""
SERIALIZED = frozenset({*(f"C751{digit}" for digit in range(6)), "C7519"})
"""The codes of ptxas's notes (`AssemblerNote`) of a kernel's warpgroup MMAs serialized, each product waited
for before the next issues (C7510 to C7515; C7519, a dependence in a divergent path): a run refuses a kernel
so noted as a spilling one (`Binary.refused`), and the compile gate fails it (tests/kernel_compile.py): the
ranking's sm90 attention backward held its registers and serialized its products (C7512)."""
IGNORED_RAISE = "C7507"
"""ptxas's note of a kernel's setmaxnreg ignored, its warp roles' paths rejoining after the split and its
threads left at the entry's registers (row 4071): the compile gate fails a derived kernel for it, a template's
structure that leaves its roles their registers; a run does not refuse it, as a role short of registers there
spills, which `Binary.refused` reads."""
CUSTOM_KERNEL_THUNK = 36
"""The `ThunkProto` field of a custom kernel's thunk, a Pallas-Triton launch's (`CustomKernelThunkProto`): its
kernel (3, `CustomKernelProto`), whose loader spec (2) holds its cubin (2, `CudaCubinProto`; 1) and which
states the dynamic shared memory its launch asks for (6, `shared_memory_bytes`)."""
ELF = b"\x7fELF"
"""A cubin's first bytes."""
REGISTER_COUNT, FRAME_SIZE, MAX_THREADS = 0x2F, 0x11, 0x05
"""A cubin's `.nv.info` attributes for a function's registers a thread and its stack frame's bytes
(EIATTR_REGCOUNT, EIATTR_FRAME_SIZE), as ptxas states them: its frame holds its spills and its locals."""


@dataclasses.dataclass(frozen=True)
class AssemblerNote:
    """A note ptxas made of a loss of performance in a kernel's assembly, as it states it: its code (`C7512`,
    a warpgroup MMA serialized for want of registers) and its message."""
    code: str
    message: str


def spilled_bytes(frame: int, spilled: int | None) -> int:
    """A kernel's spilled bytes: ptxas's measure (`spilled`), else its `frame`, which bounds it (row 9649)."""
    return frame if spilled is None else spilled


def refuses(spilled: int, codes: Collection[str]) -> bool:
    """Whether a kernel spilling `spilled` bytes (`Binary.spills`), noting `codes` (its record's
    `assembler_notes`'),
    is refused for its config: it spills, or its warpgroup MMAs were serialized (`SERIALIZED`).
    Certification's verdict (`Binary.refused`) and the compile gate's alike."""
    return bool(spilled) or bool(SERIALIZED & set(codes))

@dataclasses.dataclass(frozen=True)
class Binary:
    """A Mosaic GPU or Pallas-Triton kernel's binary as its executable carries it (`binaries`): its name less
    the launch it carries (`LAUNCH_SUFFIX`) and that launch's digest (`named`), its registers a thread and its
    stack frame's bytes a thread, which hold its spills and its locals (`REGISTER_COUNT`, `FRAME_SIZE`), the
    dynamic shared memory its launch asks for (a Mosaic kernel proto's `smem_bytes`, a custom kernel's
    `shared_memory_bytes`), None where its proto states none, and, read `decoded`, what its machine code
    breaks of the SASS rules (`sass_findings_of`), None where no disassembler read it."""
    kernel: str
    launch: str | None
    registers: int
    frame: int
    kind: str
    shared: int | None = None
    sass: tuple[str, ...] | None = ()
    caller: str | None = None
    """The launch that traced it, as its call's caller stated it (its record's `caller_metadata`, a Mosaic GPU
    kernel's); None where its call stated none."""
    notes: tuple["AssemblerNote", ...] = ()
    """What ptxas noted of a loss of performance as it assembled it, in its order (its record's
    `assembler_notes`, a Mosaic GPU kernel's); none where its call states none (a Pallas-Triton kernel's)."""
    spilled_in: tuple[str, ...] | None = ()
    """The roles whose code its spill stands in where it spills (`spilled_roles`: `PRODUCER`, `CONSUMERS`),
    read `decoded` or for its `spills` (`binaries`, certification's), none where it spills none; None where
    it was not read, or no disassembler listed it (`disassembler`)."""
    threads: int | None = None
    """Its threads a block, as its `.nv.info` states them (`MAX_THREADS`, its PTX's `.maxntid`); None where
    it states none."""
    decreased: int | None = None
    raised: int | None = None
    """The registers its producer lowers to and its consumers raise to, as its machine code reconfigures
    them (`reconfigured`: USETMAXREG's deallocation and allocation, its PTX's `setmaxnreg.dec` and `.inc`);
    None where it reconfigures none that way, or no disassembler read it."""
    uses_tmem: bool | None = None
    """Whether its decoded SASS issues Blackwell tensor-memory ops (`sass.uses_tmem`); None undecoded."""
    index: str | None = None
    """XLA uniquification suffix (`__1`) where a second cubin shares `(kernel, launch)`; None otherwise."""
    spilled: int | None = None
    """The bytes ptxas spilled for it (its record's `spilled_bytes`); None where its record states none."""

    spills = property(lambda self: spilled_bytes(self.frame, self.spilled))
    """The bytes it spills a thread (`spilled_bytes`)."""

    @property
    def launched(self) -> str | None:
        """The digest of the launch that traced it: its caller's, else its name's, which is its launch's where
        the launch states no kernel's part (`named`)."""
        return self.caller or self.launch

    @property
    def refused(self) -> bool:
        """Whether its config is refused for it (`refuses`)."""
        return refuses(self.spills, [note.code for note in self.notes])


def binaries(executable: jax.stages.Compiled, *, decoded: bool = False, spills: bool = False) -> list[Binary]:
    """Every Mosaic GPU and Pallas-Triton kernel binary `executable` carries, read from its own serialized
    form: XLA keeps a Mosaic kernel's cubin in its custom call's state (`MosaicGpuKernelProto`), whether this
    compile ran ptxas for it, another thread's program compiled it first (Mosaic compiles a module once a
    process), or the executable was loaded from the persistent cache, and a Pallas-Triton template
    launch's in its custom kernel's (`custom_kernel`), its kernel naming its launch: the thunk's own
    `binary` is its module's, which in a program holding a constant is that constant's module alone, no
    kernel's code in it. Nothing is read from or written to the process's environment or disk: a compile
    that set Mosaic's dump settings around itself handed its directory to every other thread's compile and
    removed it under them ('Directory not empty', row 2790), and the settings turn on every dump Mosaic has
    (row 2816). `decoded`, each binary's SASS findings too (`sass_findings_of`, the compile gate's), each
    cubin disassembled from a file of its own; `spills`, each spilling binary's roles its spill stands in
    (`Binary.spilled_in`, certification's: its ladder steps by them), its cubin disassembled alike."""
    runtime = executable.runtime_executable()
    if runtime is None:
        raise ValueError("a compiled program with no runtime executable carries no kernel binaries to read")
    return carried(runtime.serialize(), decoded, spills)


def carried(serialized: bytes, decoded: bool = False, spills: bool = False) -> list[Binary]:
    """`binaries`' reading of an executable as XLA serializes it (a program's, or a kernel's compiled alone,
    `report.compiled_alone`)."""
    found: list[Binary] = []
    for name, proto in _thunk_protos(serialized):
        found.extend(thunk_kernels(proto, decoded=decoded, spills=spills))
        # XLA's own kernels (its fusions, its Triton GEMMs) carry their cubins too: a template launch's
        # names its launch (`named`).
        kernel = custom_kernel(proto) if LAUNCH_SUFFIX.search(name) else None
        if kernel is not None:
            elf, shared = kernel
            found.append(dataclasses.replace(binary_of(name, elf, decoded, spills, kind="triton"),
                                             shared=shared))
    return found


def custom_kernels(executable: jax.stages.Compiled) -> list[tuple[str, bytes, int]]:
    """Every custom kernel `executable` launches, by the name its thunk gives it, with its cubin and the
    shared bytes its launch asks for (`custom_kernel`): a Pallas-Triton call's whatever its name (`binaries`
    keeps a template launch's), and XLA's own custom kernels, read from its serialized form as `binaries`."""
    runtime = executable.runtime_executable()
    if runtime is None:
        raise ValueError("a compiled program with no runtime executable launches no kernel to read")
    return [(name, *kernel) for name, proto in _thunk_protos(runtime.serialize())
            if (kernel := custom_kernel(proto)) is not None]


def launched_thunks(executable: jax.stages.Compiled) -> list[str]:
    """Every thunk `executable` launches, in order, by its kernel's name (`_thunk_protos`): XLA's own (a
    fusion of one operation's `wrapped_<operation>`) beside the custom kernels, a run's every launch."""
    runtime = executable.runtime_executable()
    if runtime is None:
        raise ValueError("a compiled program with no runtime executable launches no kernel to read")
    return [name for name, _ in _thunk_protos(runtime.serialize())]


def launched_kernels(executable: jax.stages.Compiled) -> list[tuple[str, int]]:
    """Every custom kernel `executable` launches (`custom_kernels`), by its name, with the shared bytes its
    launch asks for."""
    return [(name, shared) for name, _, shared in custom_kernels(executable)]


def _thunk_protos(serialized: bytes) -> list[tuple[str, bytes]]:
    """Each thunk an executable XLA serialized as `serialized` calls (its thunk sequence, read by thunky's
    reader of the serialized form, which its public `jax_executable_to_mlir` reads from a
    `jax.stages.Compiled`, where a kernel's compiled alone is a backend's `LoadedExecutable`), by its
    kernel's name, with its serialized proto."""
    # Imported where it is read: the lowering cache keys a lowering by the rules it can reach alone (JAX's
    # build from 0.12.0.1.post122, row 5919), so thunky's discharge rule splits no lowering's key, where its
    # import at a first reading keyed a warm start's lowerings apart from a cold one's (FS-CERTREG).
    from jax._src.thunky import thunky

    found: list[tuple[str, bytes]] = []
    thunky._check_thunky_available()
    context = thunky.make_ir_context()
    with context:
        module = thunky._thunky.jax_executable_to_mlir(serialized, context)

        def visit(operation: ir.Operation) -> ir.WalkResult:
            if operation.name == "thunky.call_thunk_proto":
                attributes = operation.attributes
                found.append((ir.StringAttr(attributes["kernel_name"]).value,
                              ir.StringAttr(attributes["thunk_proto"]).value_bytes))
            return ir.WalkResult.ADVANCE

        module.operation.walk(visit)
    return found


def thunk_kernels(proto: bytes, *, decoded: bool = False, spills: bool = False) -> list[Binary]:
    """The kernel binaries of a thunk's serialized proto (`_kernels`), one for each Mosaic GPU call it names:
    a call whose binary is missing would read as a kernel that does not spill, so it raises."""
    kernels, calls = _kernels(proto, decoded, spills)
    if len(kernels) != calls:
        raise ValueError(f"a thunk names {calls} Mosaic GPU calls but carries {len(kernels)} kernel binaries")
    return kernels


def custom_kernel(proto: bytes) -> tuple[bytes, int] | None:
    """The cubin of the custom kernel a thunk's serialized proto launches and the shared bytes its launch
    asks for (`CUSTOM_KERNEL_THUNK`), None where the thunk launches no custom kernel's cubin."""
    kernels = [kernel for thunk in proto_fields(proto).get(CUSTOM_KERNEL_THUNK, [])
               for kernel in proto_fields(thunk).get(3, [])]
    if len(kernels) != 1:
        return None
    kernel = proto_fields(kernels[0])
    cubins = [elf for spec in kernel.get(2, []) for cubin in proto_fields(spec).get(2, [])
              for elf in proto_fields(cubin).get(1, [])]
    if len(cubins) != 1 or cubins[0][:4] != ELF:
        return None
    # proto3 omits a zero.
    return cubins[0], kernel.get(6, [0])[0]


def proto_fields(message: bytes) -> dict[int, list]:
    """A serialized proto's fields by number, each its values (a varint's int, a length-delimited field's
    bytes), read without its schema; none where `message` parses as no proto. Dew's one reader of XLA's
    serialized protos: a thunk's (`custom_kernel`, `_kernels`) and a GPU topology's
    (`dew.training.ahead.topology_memory`)."""
    parsed = empty_pb2.Empty()
    try:
        parsed.ParseFromString(message)
    except DecodeError:
        return {}
    fields: dict[int, list] = {}
    for field in UnknownFieldSet(parsed):
        fields.setdefault(field.field_number, []).append(field.data)
    return fields


def _kernels(message: bytes, decoded: bool = False, spills: bool = False) -> tuple[list[Binary], int]:
    """The kernels of each Mosaic GPU kernel's record within a thunk's serialized proto (a
    `MosaicGpuKernelProto` read through the plugin's own schema, `jax.extend.mosaic_gpu.kernel_record`, whose
    binary is an ELF), and how many Mosaic GPU custom calls it names (`MOSAIC_TARGET`, a custom call thunk's
    target). A message holding neither the target's bytes nor an ELF's holds neither, and is not parsed: most
    of a program's thunks and their messages (a 4B decoder's step read 223,561 messages, 1.8 of its compile
    point's 3.5 seconds)."""
    if MOSAIC_TARGET not in message and ELF not in message:
        return [], 0
    record = mosaic_gpu_records.kernel_record(message)
    if record is not None and record.gpu_binary[:4] == ELF:
        notes = tuple(AssemblerNote(note.code, note.message) for note in record.assembler_notes)
        spilled = record.spilled_bytes if record.HasField("spilled_bytes") else None
        kernel = binary_of(record.kernel_name, record.gpu_binary, decoded, spills, spilled=spilled)
        return [dataclasses.replace(kernel, shared=record.smem_bytes,
                                    caller=record.caller_metadata.decode() or None, notes=notes)], 0
    fields = proto_fields(message)
    found, calls = [], 0
    for values in fields.values():
        for value in values:
            if value == MOSAIC_TARGET:
                calls += 1
            elif isinstance(value, bytes):
                kernels, named = _kernels(value, decoded, spills)
                found += kernels
                calls += named
    return found, calls


def binary_of(function: str, elf: bytes, decoded: bool = False, spills: bool = False, *,
            kind: str = "mosaic_gpu", spilled: int | None = None) -> Binary:
    """The `Binary` of the kernel `function` from its cubin `elf` and ptxas's `spilled` bytes: its `.nv.info`
    attributes' registers, frame and threads (`REGISTER_COUNT`, `FRAME_SIZE`, `MAX_THREADS`; a Mosaic module
    compiles one kernel function), `decoded` its SASS findings (`sass_findings_of`), and its register
    reconfiguration (`reconfigured`) and, where it spills, the roles its spill stands in (`spilled_roles`)
    wherever its one listing is read (`sass_listing`: `decoded`, or `spills` where it spills)."""
    (sections_at,) = struct.unpack_from("<Q", elf, 0x28)
    entry, count, names = struct.unpack_from("<HHH", elf, 0x3A)
    headers = [struct.unpack_from("<IIQQQQIIQQ", elf, sections_at + index * entry) for index in range(count)]
    names_at = headers[names][4]
    stated: dict[int, int] = {}
    for name, _, _, _, offset, size, *_ in headers:
        title = elf[names_at + name:elf.index(b"\0", names_at + name)]
        if title != b".nv.info" and not title.startswith(b".nv.info."):
            continue
        at = offset
        while at + 4 <= offset + size:
            form, attribute, length = struct.unpack_from("<BBH", elf, at)
            # A sized attribute (EIFMT_SVAL) holds its symbol and value after its header; the others are
            # their header alone.
            if form == 0x04:
                if attribute in (REGISTER_COUNT, FRAME_SIZE):
                    (value,) = struct.unpack_from("<I", elf, at + 8)
                    stated[attribute] = max(stated.get(attribute, 0), value)
                elif attribute == MAX_THREADS:
                    # A function's own section's: its block's three extents, no symbol before them.
                    stated[attribute] = math.prod(struct.unpack_from("<III", elf, at + 4))
                at += 4 + length
            else:
                at += 4
    if REGISTER_COUNT not in stated:
        raise ValueError(f"the binary of {function} states no registers")
    frame = stated.get(FRAME_SIZE, 0)
    spilling = spilled_bytes(frame, spilled)
    listing = sass_listing(elf) if decoded or (spills and spilling) else None
    decreased, raised = (None, None) if listing is None else reconfigured(listing)
    sass = (None if listing is None else sass_findings_of(listing)) if decoded else ()
    tensor_memory = None if listing is None else uses_tmem(listing)
    kernel, launch, index = kernel_of(function)
    return Binary(kernel, launch, stated[REGISTER_COUNT], frame, kind=kind, sass=sass,
                  spilled_in=(None if listing is None else spilled_roles(listing)) if spilling else (),
                  threads=stated.get(MAX_THREADS), decreased=decreased, raised=raised,
                  uses_tmem=tensor_memory, index=index, spilled=spilled)
