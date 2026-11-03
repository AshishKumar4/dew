"""Every compile of a step, an evaluation or a serving program reads the kernel templates' reports
(`dew.nn.kernels.templates.report.compiled`, row 2384): where a launch's kernel spills, its config is
refused and the program lowered afresh, the launch's derivation taking its next config, and what the
run keeps, its plan, its step compiled ahead and the executable it runs, is the program it lowered
then. A program with no template launch compiles as jit compiles it."""

import dataclasses
import gc
import os
import time
import weakref

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from generation_support import lowers
from jax.experimental.layout import Format, Layout
from kernel_compile import Uncounted
from process_support import fresh

from dew.inputs import Field, InputSpec
from dew.nn.attention import scaled_dot_product_attention
from dew.nn.kernels import generation, grouped_matmul
from dew.nn.kernels.options import Fetched, compiled_with, owned_program, program
from dew.nn.kernels.templates import report
from dew.nn.precision import Numerics
from dew.objectives.supervised import Supervised
from dew.telemetry.compiles import Compiles
from dew.training import Trainer

ROWS, FEATURES, WIDTH, EXPERTS = 4096, 2048, 4096, 8


class Experts(nn.Module):
    """An MoE layer's paired SwiGLU up projection, each expert taking an equal share of the rows,
    through a launch whose register model misses (`kernel_compile.Uncounted`), and a trained scale over its
    outputs: the launch is the forward kernel alone, so the step differentiates the scale, not the launch. Its
    initialization, traced for the host, draws the weights without launching it. `nonce`, added to the
    rows, makes its step a program no compilation cache holds, whose compile runs ptxas and reports
    what it finds: a cached program reports nothing."""

    nonce: float = 0.0

    @nn.compact
    def __call__(self, x):
        drawn = nn.initializers.normal(0.02)
        gate, up = (jax.lax.stop_gradient(self.param(name, drawn, (EXPERTS, FEATURES, WIDTH), jnp.bfloat16))
                    for name in ("gate", "up"))
        scale = self.param("scale", nn.initializers.ones, (WIDTH,), jnp.float32)
        if self.is_initializing():
            return jnp.zeros((x.shape[0], WIDTH), jnp.bfloat16) * scale
        sizes = jnp.full((EXPERTS,), x.shape[0] // EXPERTS, jnp.int32)
        projected = Uncounted()(
            jax.lax.stop_gradient((x + self.nonce).astype(jnp.bfloat16)), gate, sizes, paired=up,
            hooks=grouped_matmul.Hooks(epilogue=lambda g, u: jax.nn.silu(g) * u), out_dtype="bfloat16",
            trans_rhs=False, interpret=False)
        return projected * scale


def squared(outputs, batch):
    return jnp.square(outputs.astype(jnp.float32)).mean(axis=-1)


def experts(mesh, nonce: float, ahead=None) -> Trainer:
    return Trainer(Supervised(Experts(nonce), squared, inputs=InputSpec(Field("x", (FEATURES,)))),
                   optax.adam(1e-3), key=0, mesh=mesh, ahead=ahead)


BATCH = {"x": np.zeros((ROWS, FEATURES), np.float32)}


@pytest.fixture(name="nonce")
def fresh_compilation_nonce() -> float:
    """A value no earlier run's step was compiled with (`Experts.nonce`)."""
    return float(time.time_ns() % 10**9) / 10**12


def spilled(tmp_path, script: str, *arguments: str) -> dict:
    """`script`'s last printed JSON, run by a fresh interpreter over a compilation cache of its own,
    apart from every other process's: a process keeps each Mosaic kernel it compiled (jaxlib's
    `KernelHash` cache), and a cache each executable it compiled, either of which runs ptxas again for
    none, so a spill is reported at a kernel's first compile in a process with an empty cache alone."""
    cache = tmp_path / f"cache-{len(list(tmp_path.glob('cache-*')))}"
    return fresh(script, *arguments, cache_dir=cache, flags=os.environ.get("XLA_FLAGS", ""))


AHEAD = """
import json, sys
from test_kernel_spills import BATCH, experts
from dew.nn.kernels.templates import report
from dew.training import MeshSpec
from dew.training.ahead import compile_ahead, compile_only_gpus

nonce, refused = float(sys.argv[1]), sys.argv[3] == "refused"
with report.refusals():
    if refused:
        report.refuse(pickle_load(sys.argv[2]))
    ahead = compile_ahead(experts(MeshSpec().build(compile_only_gpus("sm90", 1)), nonce), BATCH)
    ahead.save(sys.argv[2] + (".again" if refused else ""))
    print(json.dumps({"program": ahead.program, "refusals": len(ahead.refusals),
                      "stepped": [miss.stepped for miss in report.misses()],
                      "missed": [repr(miss.config) for miss in report.misses()],
                      "planned": {name: ahead.plan[name] for name in ("spills", "refused")}}))
"""
"""A step compiled ahead for a compile-only H100, in a registry of its own: the one at argv[2], or, its
argv[3] `refused`, another from the refusals that one carries."""


@lowers("sm90")
def test_a_step_whose_launch_spills_is_compiled_ahead_at_its_next_config(tmp_path, nonce):
    """A step compiled ahead for a compile-only H100 on this CPU, its launch's derived tile spilling at
    ptxas, is refused that tile and lowered once more at the next, the miss recorded (`report.misses`)
    and its plan holding the spill and the refusal, as its telemetry reports them; the step it saves
    carries the refusal (`AheadOfTime.refusals`), and another process that starts from it derives past the
    refused tile, which it never compiles, its plan recording its own refusals alone (train 329's review)."""
    step = str(tmp_path / "step.dew-aot")
    first = spilled(tmp_path, PRELUDE + AHEAD, str(nonce), step, "fresh")
    assert first["refusals"] and first["stepped"][:1] == [True]
    spills, refused = first["planned"]["spills"], first["planned"]["refused"]
    assert [spill["config"] for spill in spills] == first["missed"] and spills[0]["stepped"]
    assert len(refused) == first["refusals"] and refused[0]["config"] == first["missed"][0]
    again = spilled(tmp_path, PRELUDE + AHEAD, str(nonce), step, "refused")
    # Its plan records the refusals its own compiles made, not those it was seeded with.
    assert first["missed"][0] not in again["missed"]
    assert first["missed"][0] not in [refusal["config"] for refusal in again["planned"]["refused"]]


HANDED = """
import json, sys
import jax
from test_kernel_spills import BATCH, experts
from dew._records import listed, record
from dew.nn.kernels.templates import report
from dew.training import MeshSpec
from dew.training.ahead import AheadOfTime

nonce, handed = float(sys.argv[1]), sys.argv[2]
here = MeshSpec().build(jax.devices()[:1])
with report.refusals():
    run = experts(here, nonce, ahead=AheadOfTime.load(handed) if handed else None)
    state = run.fit(BATCH, steps=1, log_every=1)
    first, *_ = listed(record(run.plan, "plan")["probes"], "probes", read=record, what="probes")
    print(json.dumps({"step": int(state.step), "ahead": first["ahead"],
                      "stepped": [miss.stepped for miss in report.misses()]}))
"""
"""A run on this host's GPU, in a registry of its own, handed the step compiled ahead at argv[2], or
none where it is empty."""


def a_cached_spilling_run_trains_its_stepped_program(tmp_path, nonce):
    """A restart over the same cache reads the cached first program's kernel binaries, finds the spill
    again and steps down to the stepped program, not the refused one. Different compiler flags compile and
    judge their own program, without inherited refusals."""
    shared, flags = tmp_path / "cache", os.environ.get("XLA_FLAGS", "")
    first = fresh(PRELUDE + HANDED, str(nonce), "", cache_dir=shared, flags=flags)
    again = fresh(PRELUDE + HANDED, str(nonce), "", cache_dir=shared, flags=flags)
    assert first["step"] == again["step"] == 1
    assert first["stepped"][:1] == again["stepped"][:1] == [True]
    other = fresh(PRELUDE + HANDED, str(nonce), "", cache_dir=shared,
                  flags=f"{flags} --xla_gpu_enable_latency_hiding_scheduler=true")
    assert other["step"] == 1 and other["stepped"][:1] == [True]


SCOPED = """
import json, sys
from dew.nn.kernels.generation import assumed
from dew.nn.kernels.templates import report
from dew.training.ahead import AheadOfTime, handed

step = AheadOfTime.load(sys.argv[1])
identity, config, names = step.refusals[0]
foreign = (identity, config, frozenset({"another kernel"}))
report.refuse([foreign])
with assumed(step.target), handed(step):
    within = report.refused_launches()
print(json.dumps({"within": sorted(map(repr, within)), "step": sorted(map(repr, step.refusals)),
                  "after": repr(foreign) in map(repr, report.refused_launches())}))
"""
"""A process on the step's GPU (`AheadOfTime.target`) that refused a launch of another program, handed the
step compiled ahead at argv[1]."""


@lowers("sm90")
def test_a_run_handed_a_step_compiled_ahead_derives_within_the_steps_refusals_alone(tmp_path, nonce):
    """A process handed a step compiled ahead, whatever launches it or its cache refused before, lowers
    it within the launch configs the step's compiles refused alone (`dew.training.ahead.handed`): one it
    refused of another program is no refusal there, and is the process's own again after (ledger 2453,
    where a refusal shared through a cache led a handed run to derive another program)."""
    step = str(tmp_path / "step.dew-aot")
    spilled(tmp_path, PRELUDE + AHEAD, str(nonce), step, "fresh")
    scoped = spilled(tmp_path, SCOPED, step)
    assert scoped["within"] == scoped["step"] and scoped["after"]


@pytest.mark.gpu(reason="trains a step whose launch spills on this host's H100")
@lowers("sm90")
def test_a_run_steps_its_spilling_launch_down_and_trains(tmp_path, nonce):
    """On an H100, a run whose launch's derived tile spills steps it down once, the miss recorded, and
    trains (row 2384's proof)."""
    trained = spilled(tmp_path, PRELUDE + HANDED, str(nonce), "")
    assert trained["step"] == 1 and trained["stepped"][:1] == [True]


PRELUDE = """
import pickle
def pickle_load(path):
    from dew.training.ahead import AheadOfTime
    return AheadOfTime.load(path).refusals
"""
"""What the scripts share: the refusals a saved step carries."""


@pytest.mark.gpu(reason="hands a step compiled ahead for the canonical H100 to a run on this host's H100")
@lowers("sm90")
def test_a_run_handed_a_step_compiled_ahead_for_the_canonical_h100_takes_it_on_one_of_its_multiprocessors(
        tmp_path, nonce):
    """A run in a fresh process handed a step compiled ahead for the canonical H100, whose launch's first
    tile spilled, on this host's H100 lowers the saved program where this H100 counts its 132
    multiprocessors: it derives its launches for the target the step was compiled for
    (`AheadOfTime.target`), within the step's refusals, and trains from it, its plan's first compile the
    step compiled ahead (ledger 2453). Where it counts others (an H100 PCIe's 114) the run is refused, as a
    balanced tail's grid derived past a GPU's residency waits on a program that never starts (row 3606)."""
    step = str(tmp_path / "step.dew-aot")
    spilled(tmp_path, PRELUDE + AHEAD, str(nonce), step, "fresh")
    canonical = generation.gpu_target("sm90").multiprocessors
    if generation.device_target(jax.devices()[0]).multiprocessors != canonical:
        from dew.training.ahead import AheadOfTime, handed

        with pytest.raises(ValueError), handed(AheadOfTime.load(step)):
            pass
        return
    handed = spilled(tmp_path, PRELUDE + HANDED, str(nonce), step)
    assert handed["step"] == 1 and handed["ahead"] is True


def test_a_handed_step_derives_for_its_target_on_a_gpu_that_runs_it_alone():
    """While a step compiled ahead is reproduced (`handed`), its launches derive for the GPU it was
    compiled for (`AheadOfTime.target`), where this process's GPU is of its generation, counts as many
    multiprocessors and holds a block's shared memory its launches take, whatever its name or memory;
    outside, the process's own facts. A GPU of another generation, of other multiprocessors (an H100
    PCIe's 114: a balanced tail's grid derived past a GPU's residency waits on a program that never
    starts, row 3606), holding less shared memory a block, or whose shared memory is unknown, is refused
    (ledger 2453)."""
    import dataclasses

    from dew.nn.kernels.generation import assumed, gpu_target
    from dew.training.ahead import AheadOfTime, Executable, handed

    canonical = gpu_target("sm90")
    step = AheadOfTime("sm90", 1, {}, "", Executable(b"", ()), None, 0, 0, 0, 0, 0, {}, canonical)
    h200 = dataclasses.replace(canonical, device_kind="NVIDIA H200")
    with assumed(h200):
        with handed(step):
            assert gpu_target() == canonical
        assert gpu_target() == h200
    # A step compiled ahead before steps recorded their GPU, unpickled with no target, is refused so too.
    earlier = AheadOfTime("sm90", 1, {}, "", Executable(b"", ()), None, 0, 0, 0, 0, 0, {})
    vars(earlier).pop("target")
    with assumed(h200), pytest.raises(ValueError), handed(earlier):
        pass
    pcie = dataclasses.replace(canonical, device_kind="NVIDIA H100 PCIe", multiprocessors=114)
    smaller = dataclasses.replace(h200, shared_limit=canonical.shared_limit - 1)
    unknown = dataclasses.replace(h200, shared_limit=None)
    for other in (gpu_target("sm80"), pcie, smaller, unknown):
        with assumed(other), pytest.raises(ValueError), handed(step):
            pass


def test_a_program_runs_the_executable_its_calls_kernel_choice_traces():
    """A program (`dew.nn.kernels.options.program`) compiled through the reports keys its executables
    as jit does: called under a run's fast attention forms and its accurate ones in turn
    (`generation.run_precision`), it compiles each form once and each call computes what jit computes
    under that form."""
    query = jax.random.normal(jax.random.key(0), (1, 16, 2, 8))

    def attend(query):
        return scaled_dot_product_attention(query, query, query, causal=True)

    run = program(attend)
    ran: dict[str, list[jax.Array]] = {"fast": [], "accurate": []}
    with Compiles("attend") as compiles:
        for form in ("fast", "accurate", "fast", "accurate"):
            with generation.run_precision(Numerics(forms=form)):
                ran[form].append(run(query))
    assert sum(found.kind == "compiled" for found in compiles.programs) == 2
    for form, outputs in ran.items():
        with generation.run_precision(Numerics(forms=form)):
            expected = jax.jit(attend)(query)
        for output in outputs:
            np.testing.assert_array_equal(output, expected)


def test_a_released_program_drops_its_executables_and_compiles_again():
    """`options.release` drops every live program's executables, which jit's cache clearing never reaches
    (XLA:CPU maps each one's code into the process, and a program keeps them for as long as it lives): the
    executable a call compiled is freed once nothing else holds it, and the next call compiles again and
    computes what the first did."""
    from dew.nn.kernels import options

    def doubled_sine(x):
        return jnp.sin(x) * 2.0

    run = program(doubled_sine)
    x = jnp.arange(8.0)
    first = run(x)
    held = weakref.ref(run.executable(x))
    options.release()
    gc.collect()
    assert held() is None
    with Compiles("doubled_sine") as compiles:
        np.testing.assert_array_equal(run(x), first)
    assert sum(found.kind == "compiled" for found in compiles.programs) == 1


def test_a_programs_calls_of_two_shapes_compile_at_once_from_two_threads():
    """Two threads calling one program at two shapes each lower their own executable while the other
    does, as jit's calls do: each tracing waits for the other's, which a call that held the program for
    its whole compile would never reach (train 329's review: `Decide.compile_scoring`'s threads)."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    met = threading.Barrier(2, timeout=30)

    def doubled(x):
        met.wait()
        return x * 2

    run = program(doubled)
    with ThreadPoolExecutor(2) as pool:
        outputs = list(pool.map(run, (jnp.ones(2), jnp.ones(3))))
    np.testing.assert_array_equal(outputs[1], jnp.full(3, 2.0))


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Slots:
    """A state of two arrays, as a server's resident slots are, placed by a tree of its own shape."""

    rows: jax.Array
    columns: jax.Array


def test_a_program_donates_its_state_and_places_it_by_a_tree_of_formats():
    """A program a server steps its state with (`program`, row 3327) donates the state it is handed
    (`donate_argnums`, `donate_argnames`), the state's arrays deleted once it ran, and places its arguments
    and results by a tree of formats of the state's own structure, each array in the layout its format
    names."""
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    formats = Slots(Format(Layout((0, 1)), sharding), Format(Layout((1, 0)), sharding))

    def stepped(slots: Slots, scale):
        return Slots(slots.rows + scale, slots.columns * scale)

    run = program(stepped, donate_argnums=0, in_shardings=(formats, None), out_shardings=formats)
    slots = Slots(jax.device_put(jnp.ones((2, 3)), formats.rows),
                  jax.device_put(jnp.ones((2, 3)), formats.columns))
    stepped_slots = run(slots, 2.0)
    assert slots.rows.is_deleted() and slots.columns.is_deleted()
    assert [leaf.format.layout.major_to_minor for leaf in (stepped_slots.rows, stepped_slots.columns)] == [
        (0, 1), (1, 0)]
    np.testing.assert_array_equal(np.asarray(stepped_slots.rows), np.full((2, 3), 3.0))
    named = program(lambda scale, slots: stepped(slots, scale), donate_argnames="slots",
                    out_shardings=formats)
    again = named(2.0, slots=stepped_slots)
    assert stepped_slots.rows.is_deleted() and stepped_slots.columns.is_deleted()
    np.testing.assert_array_equal(np.asarray(again.columns), np.full((2, 3), 4.0))


def test_a_program_is_its_jit_within_another_trace_and_keys_its_statics_as_jit_does():
    """Within `jax.grad`, `jax.vmap` or an enclosing jit a program is its jit, traced into theirs, its
    tracers its arguments or ones its function closes over, and under `jax.disable_jit` Python; called
    alone, it keys a static argument by its type as its value, as jit does (True and 1 are two programs),
    and keeps dynamic an argument its jit, given both its static positions and names, takes as one."""
    squared = program(lambda x: x * x)
    assert jax.grad(squared)(3.0) == 6.0
    np.testing.assert_array_equal(jax.vmap(squared)(jnp.arange(3.0)), jnp.arange(3.0) ** 2)
    assert jax.jit(lambda x: squared(x) + 1)(2.0) == 5.0
    assert jax.jit(lambda x: program(lambda: x * x)())(2.0) == 4.0
    with jax.disable_jit():
        assert program(lambda x: x if x > 0 else -x)(-1.0) == 1.0

    def scaled(x, flag):
        return x * 2 if type(flag) is bool else x * 3

    keyed = program(scaled, static_argnums=1)
    named = program(scaled, static_argnums=(), static_argnames="flag")
    x, flag = jnp.ones(2), True
    np.testing.assert_array_equal(keyed(x, flag), x * 2)
    np.testing.assert_array_equal(keyed(x, 1), x * 3)
    np.testing.assert_array_equal(named(x, 2.0), x * 3)
    # An array in another layout is another program's argument, as jit keys it.
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    rows, columns = (jax.device_put(jnp.arange(6.0).reshape(2, 3), Format(Layout(order), sharding))
                     for order in ((0, 1), (1, 0)))
    doubled = program(lambda x: x * 2)
    np.testing.assert_array_equal(doubled(rows), doubled(columns))
    # A typed key array, which states no layout, keys by its sharding.
    drawn = program(lambda key: jax.random.normal(key, (2,)))
    np.testing.assert_array_equal(drawn(jax.random.key(0)), jax.random.normal(jax.random.key(0), (2,)))


def _field(number: int, value: bytes | int) -> bytes:
    """One protobuf field: a varint, or bytes length-delimited."""
    def varint(n: int) -> bytes:
        out = b""
        while True:
            out += bytes([(n & 0x7F) | (0x80 if n > 0x7F else 0)])
            n >>= 7
            if not n:
                return out

    if isinstance(value, int):
        return varint(number << 3) + varint(value)
    return varint(number << 3 | 2) + varint(len(value)) + value


def _cubin(registers: int, frame: int) -> bytes:
    """An ELF holding one `.nv.info` section stating a kernel's registers and stack frame, as a cubin does."""
    import struct

    names = b"\0.shstrtab\0.nv.info\0"
    info = b"".join(struct.pack("<BBHII", 0x04, attribute, 8, 0, value)
                    for attribute, value in ((0x2F, registers), (0x11, frame)))
    sections_at = 64 + len(names) + len(info)
    header = (b"\x7fELF\x02\x01\x01" + b"\0" * 9
              + struct.pack("<HHIQQQIHHHHHH", 2, 190, 1, 0, 0, sections_at, 0, 64, 0, 0, 64, 3, 1))
    sections = [struct.pack("<IIQQQQIIQQ", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                struct.pack("<IIQQQQIIQQ", 1, 3, 0, 0, 64, len(names), 0, 0, 1, 0),
                struct.pack("<IIQQQQIIQQ", 11, 0x70000000, 0, 0, 64 + len(names), len(info), 0, 0, 4, 0)]
    return header + names + info + b"".join(sections)


def _kernel_proto(name: str, registers: int, frame: int) -> bytes:
    """A `MosaicGpuKernelProto`: its version, binary, kernel name and launch's shared memory."""
    return (_field(1, 4) + _field(8, _cubin(registers, frame))
            + _field(9, f"{name}_mosaic_gpu_kernel".encode()) + _field(10, 1024))


def _thunk(calls: int, kernels: list[bytes]) -> bytes:
    """A thunk's proto naming `calls` Mosaic GPU custom calls, each kernel proto in a message of its own."""
    return b"".join(_field(2, _field(1, b"mosaic_gpu_v2")) for _ in range(calls)) + b"".join(
        _field(3, _field(5, kernel)) for kernel in kernels)


def test_a_thunk_whose_kernel_binary_is_missing_refuses_rather_than_reading_as_no_spill():
    """Each Mosaic GPU call a thunk names has its binary (`report.thunk_kernels`): two calls and two kernel
    protos read as two kernels, their registers and frames their cubins'; two calls and one is refused, as a
    missing binary would read as a kernel that does not spill (row 2851)."""

    both = report.thunk_kernels(_thunk(2, [_kernel_proto("dew_gmm__l0123456789", 168, 0),
                                           _kernel_proto("dew_tgmm", 255, 344)]))
    read = [(kernel.kernel, kernel.launch, kernel.registers, kernel.frame, kernel.shared) for kernel in both]
    assert read == [("dew_gmm", "0123456789", 168, 0, 1024), ("dew_tgmm", None, 255, 344, 1024)]
    with pytest.raises(ValueError):
        report.thunk_kernels(_thunk(2, [_kernel_proto("dew_gmm", 168, 0)]))


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("statics", [False, True], ids=["dynamic", "static"])
def test_a_program_called_under_another_mesh_of_its_shape_runs_on_that_meshs_devices(statics):
    """A program called under two meshes of one shape over other devices runs on each mesh's own devices,
    as jit does: its executable is keyed by the mesh in place, its dynamic and its static calls alike
    (ledger 3452), where under the second mesh it ran the first's executable on the first's devices."""
    from jax.sharding import AxisType, Mesh, PartitionSpec

    def function(x, scale=1):
        return jax.lax.with_sharding_constraint(x, PartitionSpec("x")) * scale + 1

    devices = jax.devices()
    meshes = [Mesh(np.asarray(devices[start:start + 2]), ("x",), axis_types=(AxisType.Auto,))
              for start in (0, 2)]
    compiled = program(function, **({"static_argnames": "scale"} if statics else {}))
    given, scale = np.arange(8, dtype=np.float32), 2 if statics else 1
    for mesh in meshes:
        with jax.set_mesh(mesh):
            ran = compiled(given, **({"scale": scale} if statics else {}))
        assert ran.sharding.device_set == set(mesh.devices.flat)
        np.testing.assert_array_equal(np.asarray(ran), given * scale + 1)


@lowers("sm90")
def test_an_executable_carries_each_mosaic_and_triton_kernels_binary_once():
    """A program of a Pallas-Triton launch feeding a Mosaic GPU kernel and an XLA fusion, compiled for an
    H100: `report.binaries` reads the launch once, its cubin from its thunk's own binary, and the Mosaic
    kernel once, from its kernel proto (a Mosaic thunk's binary is empty), each stating its registers; the
    fusion, whose thunk carries its cubin too, names no launch and is no template's."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as pltriton
    from jax.sharding import NamedSharding, PartitionSpec
    from kernel_compile import staged_copy

    from dew.nn.kernels import generation
    from dew.training.ahead import compile_only_gpus

    def added(x_ref, y_ref, o_ref):
        o_ref[...] = x_ref[...] + y_ref[...]

    copy = staged_copy((128, 128), jnp.float32)
    shape = jax.ShapeDtypeStruct((128, 128), jnp.float32)

    def program(x, y):
        launched = pl.pallas_call(added, out_shape=shape, grid=(1,), name="dew_added__l0123456789",
                                  compiler_params=pltriton.CompilerParams(num_warps=4, num_stages=1))(x, y)
        return jnp.concatenate([copy(launched), jnp.sin(x)]).sum(axis=1)

    devices = jax.sharding.Mesh(np.array(compile_only_gpus("sm90")), ("data",))
    spec = jax.ShapeDtypeStruct((128, 128), jnp.float32, sharding=NamedSharding(devices, PartitionSpec()))
    with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(generation.gpu_mesh(devices, "sm90")):
        compiled = jax.jit(program).trace(spec, spec).lower().compile()
    found = report.binaries(compiled)
    assert len(found) == 2 and all(binary.registers > 0 and not binary.frame for binary in found)
    launched = [(binary.kernel, binary.launch) for binary in found if binary.launch]
    assert launched == [("dew_added", "0123456789")]

def test_a_program_its_caller_owns_is_released_with_its_caller():
    """A program over a closure or a bound method its caller owns (`owned_program`), as a run's placement
    and its host phases compile, keeps no reference of Dew's: once the caller lets it go, the owner the
    closure holds (a trainer, its objective and their arrays) is released, as no process-wide cache holds
    the closure. A program over a module's function (`program`) is kept for every caller alike."""

    class Owner:
        """A run's stand-in, holding an array its closure reads."""

        def __init__(self):
            self.held = jnp.arange(4.0)

        def scaled(self, x):
            return x * self.held

    def reading(held: Owner):
        return lambda x: x + held.held

    owner = Owner()
    released = weakref.ref(owner)
    closure = owned_program(reading(owner))
    bound = owned_program(owner.scaled)
    np.testing.assert_array_equal(np.asarray(closure(jnp.ones(4))), np.arange(4.0) + 1)
    np.testing.assert_array_equal(np.asarray(bound(jnp.ones(4))), np.arange(4.0))
    del owner, closure, bound
    gc.collect()
    assert released() is None
    assert program(jnp.sum) is program(jnp.sum)


def test_a_fetched_program_runs_the_executable_of_its_first_call():
    """A program a host loop calls every step (`Fetched`) answers as the program does for arguments of its
    first call's shapes and dtypes, however their values change; arguments of another shape are refused
    where the program itself would compile for them; and a program with a static argument, whose
    executable takes the dynamic arguments alone, is refused at once."""
    doubled = owned_program(lambda x: x * 2.0)
    fetched = Fetched(doubled)
    for value in (1.0, 3.0):
        np.testing.assert_array_equal(np.asarray(fetched(jnp.full(3, value))), np.full(3, 2 * value))
    with pytest.raises(TypeError):
        fetched(jnp.ones(4))
    np.testing.assert_array_equal(np.asarray(doubled(jnp.ones(4))), np.full(4, 2.0))
    with pytest.raises(ValueError):
        Fetched(owned_program(lambda scale, x: scale * x, static_argnums=0))


def test_a_program_within_another_takes_the_enclosing_programs_options():
    """A program called within another trace (a kernel's own program inside a step) is the enclosing
    program's part and compiles under its options, as JAX takes compiler options on the outermost jit
    alone: a program opened with options is called inside another without refusal."""
    with compiled_with({"xla_cpu_enable_fast_math": False}):
        inner = owned_program(lambda x: x * 2.0)
        outer = owned_program(lambda x: inner(x) + 1.0)
        np.testing.assert_array_equal(np.asarray(outer(jnp.ones(3))), np.full(3, 3.0))

def test_a_program_under_an_eager_transformation_compiles_under_its_own_options():
    """A program called under an eager `jax.vmap` or `jax.grad`, which no enclosing program compiles,
    compiles under its own options as a direct call does, so an option XLA does not know is refused
    there too; within another program's trace it is that program's part and takes its options."""
    with compiled_with({"xla_dew_no_such_option": True}):
        unknown = owned_program(lambda x: x * 2.0)
    rows = jnp.ones((2, 3))
    for called in (lambda: unknown(rows), lambda: jax.vmap(unknown)(rows),
                   lambda: jax.grad(lambda x: jnp.sum(unknown(x)))(rows)):
        with pytest.raises(jax.errors.JaxRuntimeError):
            called()
    enclosing = owned_program(lambda x: unknown(x) + 1.0)
    np.testing.assert_array_equal(np.asarray(enclosing(rows)), np.full((2, 3), 3.0))
