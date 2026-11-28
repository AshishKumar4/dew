"""The compile gate's reading of ptxas's notes: a kernel whose assembly ptxas notes a performance loss of
fails it."""
import dataclasses
import functools
from typing import ClassVar

import jax
import jax.numpy as jnp
import kernel_compile as kc
import pytest
from fla_gdn_support import KernelCase
from generation_support import lowers
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels.templates import assembled, swizzling

SERIALIZING = ("ptxas info    : (C7512) Potential Performance Loss: wgmma.mma_async instructions are "
               "serialized due to insufficient register resources for the function "
               "'recurrence_output_mosaic_gpu_kernel'\n"
               "ptxas info    : Compiling entry function 'recurrence_output_mosaic_gpu_kernel' for 'sm_90a'\n"
               "ptxas info    : Used 152 registers, used 1 barriers, 0 bytes smem\n"
               "    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads")
"""ptxas's report of the recurrence output kernel at a consumer cap of 152 (FS-LT-OUTPUT-CAPS1): spill-free,
its warpgroup MMAs serialized."""


IGNORING = ("ptxas info    : (C7507) Potential Performance Loss: 'setmaxnreg' ignored to maintain minimum "
            "register requirements\n"
            "ptxas info    : Compiling entry function 'gemm_mosaic_gpu_kernel' for 'sm_90a'\n"
            "ptxas info    : Used 168 registers, used 1 barriers, 0 bytes smem\n"
            "    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads")
"""ptxas's report of a warp-specialized kernel whose roles' paths rejoin after their setmaxnreg: the kernel
challenge's C7507 (notes/ptxas-re.md §4, 'setmaxnreg' ignored), its threads left at the entry's registers."""


DIVERGED = ("ptxas info    : (C7520) Potential Performance Loss: wgmma.mma_async instructions are serialized "
            "due to program dependence on compiler-inserted WG.AR in divergent path in the function "
            "'dew_gmm__la5e5fcdee4_mosaic_gpu_kernel'\n"
            "ptxas info    : Compiling entry function 'dew_gmm__la5e5fcdee4_mosaic_gpu_kernel' for 'sm_90a'\n"
            "ptxas info    : Used 168 registers, used 16 barriers, 1 bytes smem\n"
            "    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads")
"""ptxas's report of M-FS-74's Qwen3.5-4B q input gradient (CUDA 12.9 and 13.0 alike, FS-QDX-SASS3):
spill-free, every HGMMA of its walk waited, its merge of a shared tile's partials branching on a verdict held
lane by lane (row 10151)."""


@pytest.mark.parametrize(("notes", "registers", "code", "named"),
                         [(SERIALIZING, 152, "C7512", "serialize"), (DIVERGED, 168, "C7520", "serialize"),
                          (IGNORING, 168, "C7507", "setmaxnreg")])
def test_a_kernel_ptxas_notes_a_performance_loss_of_fails_the_gate(notes, registers, code, named):
    """A Mosaic GPU kernel ptxas notes a performance loss of fails the compile gate (`mosaic_failures`), its
    registers and spills clean: one whose warpgroup MMAs it serializes (C7510 to C7515 and C7520: each HGMMA
    waited, as a spill-free 152-register recurrence output was, and a GEMM whose merge branched on a verdict
    held lane by lane) runs its MMAs one at a time (rows 3615, 10151; the kernel challenge's C7511 at 112
    registers, C7512 at 96), and one whose setmaxnreg it ignores (C7507: code after
    the warp roles' split reachable from both budgets, every raise and decrease dropped) keeps its consumers
    at the entry's registers, never the ones its tile was sized for (row 4071)."""
    read = kc.ptxas_read(notes)
    assert (read.registers, read.spilled, read.losses) == (registers, 0, (code,))
    (failure,) = kc.mosaic_failures([_assembled_record("recurrence_output_mosaic_gpu_kernel", read)])
    assert named in failure


def _assembled_record(kernel: str, read) -> dict:
    """A compile record of one spill-free H100 kernel as ptxas `read` it."""
    return {"name": kernel, "generation": "sm90", "limit": 232448, "tmem_limit": 0,
            "kernels": [{"kernel": kernel, "shared": 1024, "tmem": 0, "registers": read.registers,
                         "spilled": read.spilled, "losses": read.losses, "unread": []}]}


@pytest.fixture
def judging():
    """judged_builds' cache empty around a cell that fakes its host: no later cell reads the fake's answer."""
    kc.judged_builds.cache_clear()
    yield
    kc.judged_builds.cache_clear()


@pytest.mark.parametrize(("gpus", "installed", "generation", "judged"), [
    (None, ["cuda12"], "sm90", ("cuda13", "cuda12")),
    (None, ["cuda12"], "sm100", ("cuda13",)),
    (("580.65", (8, 0)), ["cuda12"], "sm80", ("cuda12",)),
    (("580.65", (8, 0)), ["cuda13"], "sm90", ("cuda13",)),
    (("580.65", (8, 0)), [], "sm90", ("cuda13", "cuda12")),
])
def test_the_gate_judges_every_build_on_a_host_with_no_gpu_and_a_gpu_environments_own_on_one(
        monkeypatch, judging, gpus, installed, generation, judged):
    """The kernel-compile environment, on a host with no GPU, judges a generation's kernels under every CUDA
    build its GPUs run, whatever plugin it holds, as does an environment with none on any host; a GPU node's
    environment under the build its CUDA plugin is alone, as it holds no other build's ptxas (row 8769: an
    A100's cuda12 environment asked CUDA 13's)."""
    monkeypatch.setattr(kc, "nvidia_gpus", lambda: gpus)
    monkeypatch.setattr(kc, "cuda_builds", lambda: installed)
    assert kc.judged_builds(generation) == judged


def test_a_gpu_environment_none_of_whose_builds_a_generation_runs_is_refused(monkeypatch, judging):
    """A generation whose GPUs run no build this GPU environment installs (a B200's cuda13 on a cuda12
    node) is refused, never judged under no assembler."""
    monkeypatch.setattr(kc, "nvidia_gpus", lambda: ("580.65", (8, 0)))
    monkeypatch.setattr(kc, "cuda_builds", lambda: ["cuda12"])
    with pytest.raises(RuntimeError):
        kc.judged_builds("sm100")
    # The same environment judges a generation its build runs: the refusal is the generation's alone.
    assert kc.judged_builds("sm80") == ("cuda12",)


def _register_bound_product(depth: int):
    """A warpgroup's 64 x 128 product of a depth-`depth` operand it holds in registers by one staged in
    shared memory, five blocks a multiprocessor (`min_blocks_per_sm`, a launch's register cap of 96), which
    leaves its MMA pipeline short of registers: ptxas serializes it (FS-6456-SCAN5: C7511 spill-free at
    depth 128, C7512 at 64)."""
    from jax._src.pallas.mosaic_gpu import pallas_call as mosaic_pallas_call

    swizzled = (plgpu.TilingTransform((8, 64)), plgpu.SwizzleTransform(128))

    def kernel(a_ref, b_ref, o_ref):
        def scope(acc_ref):
            plgpu.wgmma(acc_ref, a_ref[...], b_ref)
            return acc_ref[...]

        o_ref[...] = swizzling.run_scoped(scope, plgpu.ACC((64, 128), jnp.float32))

    return mosaic_pallas_call.pallas_call(
        kernel,
        in_specs=[plgpu.BlockSpec((64, depth), lambda: (0, 0), transforms=swizzled),
                  plgpu.BlockSpec((depth, 128), lambda: (0, 0), transforms=swizzled)],
        out_specs=plgpu.BlockSpec((64, 128), lambda: (0, 0)),
        out_shape=jax.ShapeDtypeStruct((64, 128), jnp.float32),
        compiler_params=plgpu.CompilerParams(min_blocks_per_sm=5))


@lowers("sm90")
@pytest.mark.parametrize("depth", [128, 64])
def test_a_kernel_ptxas_serializes_carries_ptxas_own_notes_in_its_binary(depth):
    """A warpgroup product ptxas serializes, compiled by Dew's JAX build for a compile-only H100: the notes
    its binary carries (`assembled.Binary.notes`, the native compile's own) are ptxas's of its PTX, the gate's
    read of it, a serialization among them, so a run's certification refuses it on its binary's word (row
    6456: the notes once read none, a serialized kernel passing `Binary.refused`)."""
    shapes = (jax.ShapeDtypeStruct((64, depth), jnp.float16), jax.ShapeDtypeStruct((depth, 128), jnp.float16))
    (record,) = kc.compile_mosaic_kernels(
        [KernelCase("register_bound", lambda: (_register_bound_product(depth), shapes))], "sm90")
    (kernel,) = record["kernels"]
    assert set(kernel["losses"]) & assembled.SERIALIZED, kernel
    assert tuple(kernel["binary_losses"]) == tuple(kernel["losses"]), kernel





def test_a_triton_template_binary_in_compile_mosaic_kernels_reports_shared_registers_and_spills():
    """Row 7263: when a public route compiled through `compile_mosaic_kernels` falls back to a
    Pallas-Triton template launch, `reports` + `inherited` + `mosaic_failures` read its cubin's shared
    bytes, registers and stack frame (zero unread, tmem 0 when `uses_tmem is False`), refuse spills,
    preserve XLA-uniquified Triton cubins losslessly apart from independent stems and Mosaic inventories,
    and mark TMEM unread when `uses_tmem` is True or None."""
    import dataclasses

    assert assembled.kernel_of("probe") == ("probe", None, None)
    assert assembled.kernel_of("probe__l0123456789") == ("probe", "0123456789", None)
    assert assembled.kernel_of("probe__l0123456789__1") == ("probe", "0123456789", "1")
    assert assembled.kernel_of("probe__1__l0123456789") == ("probe__1", "0123456789", None)
    for name in ("probe", "probe__l0123456789", "probe__l0123456789__1", "probe__1__l0123456789"):
        assert assembled.launch_name(*assembled.kernel_of(name)) == name

    clean = assembled.Binary("attention_forward", "0123456789", 64, 0, "triton",
                          shared=16384, sass=(), uses_tmem=False)
    uniquified_spill = assembled.Binary("attention_forward", "0123456789", 96, 24, "triton",
                                     shared=16384, sass=(), uses_tmem=False, index="1")
    distinct_stem = assembled.Binary("attention_forward__1", "0123456789", 48, 0, "triton",
                                  shared=8192, sass=(), uses_tmem=False)
    tmem_using = assembled.Binary("tmem_triton", "0123456789", 64, 0, "triton",
                               shared=16384, sass=(), uses_tmem=True)
    undecoded = assembled.Binary("undecoded_triton", "0123456789", 64, 0, "triton",
                              shared=16384, sass=(), uses_tmem=None)

    found = [kc.inherited(dataclasses.asdict(r), {}, "sm80")
             for r in kc.reports({}, [clean, uniquified_spill, distinct_stem, tmem_using, undecoded], {})]
    by_key = {assembled.launch_name(item["kernel"], item["launch"], item["index"]): item for item in found}
    assert len(by_key) == 5, by_key
    clean_item = by_key["attention_forward__l0123456789"]
    assert clean_item["unread"] == ()
    assert clean_item["shared"] == 16384
    assert clean_item["tmem"] == 0
    assert clean_item["registers"] == 64
    assert clean_item["spilled"] == 0

    clean_rec = {"name": "decode", "limit": 166912, "tmem_limit": 0, "kernels": [clean_item]}
    assert kc.mosaic_failures([clean_rec]) == []

    spill_rec = {"name": "decode", "limit": 166912, "tmem_limit": 0,
                 "kernels": [clean_item, by_key["attention_forward__l0123456789__1"]]}
    failures = kc.mosaic_failures([spill_rec])
    assert len(failures) == 1 and "24 bytes spilled" in failures[0], failures

    for key in ("tmem_triton__l0123456789", "undecoded_triton__l0123456789"):
        assert "tensor memory columns" in by_key[key]["unread"], by_key[key]
        assert by_key[key]["tmem"] is None
        tmem_rec = {"name": "decode", "limit": 166912, "tmem_limit": 512, "kernels": [by_key[key]]}
        assert kc.mosaic_failures([tmem_rec]), key

    # Mosaic estimates join only Mosaic binaries: a same-named Triton binary never masks a missing Mosaic
    # binary and never collides with a same-named Mosaic binary.
    mosaic_est = {
        "attention_forward__l0123456789": (
            "Resources(smem_scratch_bytes=1024, tmem_scratch_cols=0, tmem_collective_scratch_cols=0)"
        )
    }
    mixed = kc.reports(mosaic_est, [clean], {})
    assert len(mixed) == 2, mixed
    assert any(r.kind == "mosaic_gpu" and "binary" in r.unread for r in mixed), mixed
    assert any(r.kind == "triton" and not r.unread for r in mixed), mixed
    mosaic_bin = assembled.Binary("attention_forward", "0123456789", 64, 0, "mosaic_gpu", shared=1024,
                                  sass=())
    both = kc.reports(mosaic_est, [clean, mosaic_bin], {})
    assert {r.kind for r in both} == {"mosaic_gpu", "triton"}, both
    # A Mosaic model with a moving producer applies only to the Mosaic kernel, not its clean Triton sibling.
    mosaic_item = {**dataclasses.asdict(next(r for r in both if r.kind == "mosaic_gpu")),
                   "registers": 64, "spilled": 0, "decreased": 40, "cap": 255}
    triton_item = dataclasses.asdict(next(r for r in both if r.kind == "triton"))
    mixed_rec = {"name": "decode", "limit": 166912, "tmem_limit": 0,
                 "kernels": [mosaic_item, triton_item],
                 "models": {"attention_forward": {"shared": 1024, "producer": 40, "moves": True}},
                 "stated_models": {"attention_forward__l0123456789": {"shared": 1024, "producer": 40,
                                                                       "moves": True}}}
    assert kc.mosaic_failures([mixed_rec]) == []


@lowers("sm80")
def test_compile_mosaic_kernels_reports_both_xla_uniquified_triton_cubins():
    """Row 7263: two Pallas-Triton calls sharing one launch-tagged kernel name with distinct shapes force
    XLA to emit a uniquified second cubin (`__1`); `compile_mosaic_kernels` reports both distinct cubins
    with zero unread fields and zero spills."""
    import jax.numpy as jnp
    from fla_gdn_support import KernelCase
    from jax.experimental import pallas as pl

    from dew.nn.kernels import generation

    def add_one(x_ref, o_ref):
        o_ref[...] = x_ref[...] + 1.0

    def program(a, b):
        def call(x):
            return pl.pallas_call(add_one, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
                                  name="probe__l0123456789")(x)
        return call(a), call(b)

    a = jax.ShapeDtypeStruct((128,), jnp.float32)
    b = jax.ShapeDtypeStruct((256,), jnp.float32)
    with generation.assumed("sm80"):
        found = kc.compile_mosaic_kernels([KernelCase("uniquified", lambda: (program, (a, b)))], "sm80")
    kernels = [k for k in found[0]["kernels"] if k["kind"] == "triton"]
    assert len(kernels) == 2, kernels
    assert {k["index"] for k in kernels} == {None, "1"}, kernels
    assert all(k["kernel"] == "probe" and k["launch"] == "0123456789" for k in kernels), kernels
    assert all(not k["unread"] for k in kernels), kernels
    assert kc.mosaic_failures(found) == []


@lowers("sm100")
def test_report_binaries_distinguishes_sm100_tmem_mosaic_cubins_from_non_tmem_triton_cubins():
    """Row 7263: `assembled.binaries(..., decoded=True)` sets `uses_tmem=True` on an sm100 Mosaic GPU TMEM
    cubin (tcgen05's persistent walk, its accumulators in tensor memory), `uses_tmem=False` on an sm100
    Triton cubin that allocates no TMEM (an fp32 row doubled, named as a template launch), and
    `uses_tmem=None` when `decoded=False`; each the least kernel of its kind, where an attention forward
    and an fp32 decode's four Triton kernels took 122 s (row 12717; the decode alone 66 s, FS-12717-TRITON1;
    these two 21 s, FS-12717-TMEM3)."""

    import numpy as np
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as pltriton
    from test_mosaic_toolkit import _persisted_operands, _persisted_products

    from dew.nn.kernels import generation
    from dew.nn.kernels.templates import mosaic_sm100

    family = mosaic_sm100.RINGS
    operands = tuple(jax.ShapeDtypeStruct(np.shape(each), each.dtype)
                     for each in _persisted_operands(family.rows))
    with generation.assumed("sm100"):
        exe_m = kc.lowered_for(functools.partial(_persisted_products, family=family, interpret=False),
                               operands, "sm100").compile()
    mosaic_decoded = [b for b in assembled.binaries(exe_m, decoded=True) if b.kind == "mosaic_gpu"]
    mosaic_raw = [b for b in assembled.binaries(exe_m, decoded=False) if b.kind == "mosaic_gpu"]
    assert mosaic_decoded and any(b.uses_tmem is True for b in mosaic_decoded), mosaic_decoded
    assert mosaic_raw and all(b.uses_tmem is None for b in mosaic_raw), mosaic_raw

    def doubled(x_ref, o_ref):
        o_ref[...] = x_ref[...] * 2

    row = jax.ShapeDtypeStruct((1024,), jnp.float32)
    with generation.assumed("sm100"):
        # Named as a template launch is (`assembled.LAUNCH_SUFFIX`): `binaries` reads a Triton launch's
        # cubin by its name.
        doubling = pl.pallas_call(doubled, out_shape=row, compiler_params=pltriton.CompilerParams(),
                                  name="doubled__l0123456789")
        exe_t = kc.lowered_for(doubling, (row,), "sm100").compile()
    triton_decoded = [b for b in assembled.binaries(exe_t, decoded=True) if b.kind == "triton"]
    triton_raw = [b for b in assembled.binaries(exe_t, decoded=False) if b.kind == "triton"]
    assert triton_decoded and all(b.uses_tmem is False for b in triton_decoded), triton_decoded
    assert triton_raw and all(b.uses_tmem is None for b in triton_raw), triton_raw


def _run(name: str, spilled: int = 0, **fields) -> dict:
    """A compile record of one H100 kernel ptxas assembled spilling `spilled` bytes at its launch's register
    cap, its PTX dumped here."""
    record = _assembled_record("k", kc.Assembled(160, spilled, ()))
    kernel = {**record["kernels"][0], "dumped": True, "cap": 168}
    return {**record, "name": name, "kernels": [kernel], **fields}


def test_each_builds_run_of_a_case_is_held_to_the_gates_checks():
    """Row 10766: each build's run of a case (`record['builds']`, the program that build's derivations select,
    as it assembles it) is held to the gate's checks as the case's first run is (`mosaic_failures`,
    `failing_runs`), one of the first run's program (`same`) as much as one of its own: a build's evidence
    failing them fails the case; a run whose kernel its build could not judge for want of its PTX
    (`unjudged`) fails and runs the case again in a process of its own (`_undumped`)."""
    clean, spilling = _run("k"), _run("k under cuda13", spilled=24, same="k")
    assert kc.mosaic_failures([clean]) == [] and not kc._undumped([clean])
    case = {**clean, "builds": {"cuda13": spilling}}
    (failure,) = kc.mosaic_failures([case])
    assert failure.startswith("k under cuda13")
    assert [run["name"] for run in kc.failing_runs(case)] == ["k under cuda13"]
    assert kc.mosaic_failures([{**clean, "builds": {"cuda13": _run("k under cuda13", same="k")}}]) == []
    unjudged = {**clean, "builds": {"cuda13": _run("k under cuda13", same="k", unjudged=["k"])}}
    assert kc._undumped([unjudged])
    assert [run["name"] for run in kc.failing_runs(unjudged)] == ["k under cuda13"]


def test_a_certifications_ptx_is_the_one_its_own_compile_printed(tmp_path, monkeypatch):
    """PTX modules Mosaic dumped of one kernel's name and launch, each under a name of its compile's own
    numbering: a certification's bound to its own kernels' module, of the dumps its own compile wrote alone
    (`kernel_compile._Bound`, `report.Judge.compiling`): a failed compile's dump none of a later one's, a
    compile that printed none (Mosaic compiles a module once a process) binding none, a later certification's
    written over an earlier one's file its own, two of one name in one compile neither's; a program's read
    after its compile (`_read_printed`), apart from a certification's of its name."""
    lowering, compiling = tmp_path / "lowering", tmp_path / "compiling"
    for directory in (lowering, compiling):
        directory.mkdir()
    monkeypatch.setenv("MOSAIC_GPU_DUMP_TO", str(lowering))

    def dumped(directory, number, body):
        (directory / f"k_{number}.ptx").write_text(
            f"//\n{kc.PTX_MODULE}\n.visible .entry k__l0123456789_mosaic_gpu_kernel(\n)\n{{\n{body}\n}}\n")

    def certified(judge, module, *bodies):
        with judge.compiling():
            for number, body in bodies:
                dumped(lowering, number, body)
        judge({"k__l0123456789": (module, None)})

    judge = kc._Bound()
    with judge.compiling():
        dumped(lowering, 1, "failed")  # a compile that failed: its judge never called
    certified(judge, "a" * 16)  # its module compiled before: printed none
    certified(judge, "b" * 16, (2, "certified"))
    certified(judge, "c" * 16, (2, "later rung"))
    certified(judge, "d" * 16, (3, "one"), (4, "other"))
    dumped(compiling, 7, "compiled")
    kc._read_printed(compiling.glob("*.ptx"), {"k__l0123456789": "e" * 16})
    assert "a" * 16 not in kc.PTX and "d" * 16 not in kc.PTX
    assert ("certified" in kc.PTX["b" * 16] and "later rung" in kc.PTX["c" * 16]
            and "compiled" in kc.PTX["e" * 16])


@lowers("sm90")
def test_a_builds_rebuilt_kernel_is_that_builds_assembly_of_its_own_modules_ptx():
    """A kernel remade as a build assembles it (`kernel_compile._Rebuilt`, the judge a build's run certifies
    and reads its program by): its registers and notes of a loss that build's ptxas's of the PTX printed of
    its module (`_assembly`), its launch's facts (its name, launch and shared bytes) its compile's; one of a
    module no PTX of which printed here is left as compiled and named unjudged."""
    shapes = (jax.ShapeDtypeStruct((64, 128), jnp.float16), jax.ShapeDtypeStruct((128, 128), jnp.float16))
    (record,) = kc.compile_mosaic_kernels(
        [KernelCase("register_bound", lambda: (_register_bound_product(128), shapes))], "sm90")
    (found,) = record["binaries"]
    binary = assembled.Binary(**{**found, "notes": tuple(assembled.AssemblerNote(**note)
                                                         for note in found["notes"])})
    (module,) = record["modules"].values()
    for build in kc.judged_builds("sm90"):
        judge = kc._Rebuilt(build, "sm90")
        ((_, rebuilt),) = judge({"k": (module, binary)}).values()
        read = kc._assembly(module, "sm90", build).read
        codes = tuple(note.code for note in rebuilt.notes)
        assert (rebuilt.registers, codes) == (read.registers, read.losses)
        assert (rebuilt.kernel, rebuilt.launch, rebuilt.shared) == (
            binary.kernel, binary.launch, binary.shared)
        assert judge.unjudged == []
    judge = kc._Rebuilt("cuda12", "sm90")
    assert judge({"k": ("0" * 16, binary)}) == {"k": ("0" * 16, binary)}
    assert judge.unjudged == [assembled.launch_name(binary.kernel, binary.launch)]


@lowers("sm90")
def test_a_case_runs_as_another_builds_verdicts_select_its_program():
    """Row 10766: a case run as another build certifies its launches (`kernel_compile._run` under a judge,
    `report.judging`) runs the program that build's derivations select, compiled and read as the case's own
    run is: a judge refusing the derived config's kernels steps its launch past it to a program of its own,
    the derived config certified refused there alone; one refusing nothing selects the case's own program,
    compiling nothing (`same`)."""
    from dew.nn.kernels import gemm, generation

    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    shapes = (bf16((2048, 4096)), bf16((4096, 4096)))
    case = KernelCase("dense", lambda: (lambda x, w: gemm.gemm(x, w, None, out_dtype=jnp.bfloat16), shapes))
    class Refusing(kc._Bound):
        """The first launch certified (the derived config's) spilling, every other as compiled."""

        first: ClassVar[set[str]] = set()

        def __call__(self, kernels):
            kernels = super().__call__(kernels)
            if not Refusing.first:
                Refusing.first.update(module for module, _ in kernels.values())
            return {name: (module, dataclasses.replace(binary, spilled=64) if module in Refusing.first
                           else binary) for name, (module, binary) in kernels.items()}

    with generation.chosen(grouped_matmul="mosaic"):
        own = kc._header(case, "sm90")
        kc._run(case, "sm90", own, "cuda12")
        runs = {}
        for name, judge in (("refusing", Refusing()), ("holding", kc._Bound())):
            runs[name] = kc._header(case, "sm90")
            kc._run(case, "sm90", runs[name], "cuda12", judge=judge, earlier={own["lowered"]: own})
    assert "error" not in own and own["kernels"], own
    refusing, holding = runs["refusing"], runs["holding"]
    assert holding["same"] == own["name"] and holding["kernels"] == own["kernels"]
    assert "same" not in refusing and refusing["lowered"] != own["lowered"]
    assert refusing["kernels"] and kc.failing_runs(refusing) == [], kc.mosaic_failures([refusing])
    verdicts = [any(refused for *_, refused in cert["kernels"].values())
                for cert in refusing["certifications"]]
    assert verdicts[0] and not verdicts[-1], refusing["certifications"]
