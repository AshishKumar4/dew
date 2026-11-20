"""The compile gate's reading of ptxas's notes: a kernel whose assembly ptxas notes a performance loss of
fails it."""
import jax
import jax.numpy as jnp
import kernel_compile as kc
import pytest
from fla_gdn_support import KernelCase
from generation_support import lowers
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels.templates import report

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


@pytest.mark.parametrize(("notes", "registers", "code", "named"),
                         [(SERIALIZING, 152, "C7512", "serialize"), (IGNORING, 168, "C7507", "setmaxnreg")])
def test_a_kernel_ptxas_notes_a_performance_loss_of_fails_the_gate(notes, registers, code, named):
    """A Mosaic GPU kernel ptxas notes a performance loss of fails the compile gate (`mosaic_failures`), its
    registers and spills clean: one whose warpgroup MMAs it serializes (C7510 to C7515: each HGMMA waited, as
    a spill-free 152-register recurrence output was) runs its MMAs one at a time (row 3615; the kernel
    challenge's C7511 at 112 registers, C7512 at 96), and one whose setmaxnreg it ignores (C7507: code after
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
    from jax.experimental import pallas as pl

    swizzled = (plgpu.TilingTransform((8, 64)), plgpu.SwizzleTransform(128))

    def kernel(a_ref, b_ref, o_ref):
        def scope(acc_ref):
            plgpu.wgmma(acc_ref, a_ref[...], b_ref)
            return acc_ref[...]

        o_ref[...] = pl.run_scoped(scope, plgpu.ACC((64, 128), jnp.float32))

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
    its binary carries (`report.Binary.notes`, the native compile's own) are ptxas's of its PTX, the gate's
    read of it, a serialization among them, so a run's certification refuses it on its binary's word (row
    6456: the notes once read none, a serialized kernel passing `Binary.refused`)."""
    shapes = (jax.ShapeDtypeStruct((64, depth), jnp.float16), jax.ShapeDtypeStruct((depth, 128), jnp.float16))
    (record,) = kc.compile_mosaic_kernels(
        [KernelCase("register_bound", lambda: (_register_bound_product(depth), shapes))], "sm90")
    (kernel,) = record["kernels"]
    assert set(kernel["losses"]) & report.SERIALIZED, kernel
    assert tuple(kernel["binary_losses"]) == tuple(kernel["losses"]), kernel





def test_a_triton_template_binary_in_compile_mosaic_kernels_reports_shared_registers_and_spills():
    """Row 7263: when a public route compiled through `compile_mosaic_kernels` falls back to a
    Pallas-Triton template launch, `reports` + `inherited` + `mosaic_failures` read its cubin's shared
    bytes, registers and stack frame (zero unread, tmem 0 when `uses_tmem is False`), refuse spills,
    preserve XLA-uniquified Triton cubins losslessly apart from independent stems and Mosaic inventories,
    and mark TMEM unread when `uses_tmem` is True or None."""
    import dataclasses

    assert report.kernel_of("probe") == ("probe", None, None)
    assert report.kernel_of("probe__l0123456789") == ("probe", "0123456789", None)
    assert report.kernel_of("probe__l0123456789__1") == ("probe", "0123456789", "1")
    assert report.kernel_of("probe__1__l0123456789") == ("probe__1", "0123456789", None)
    for name in ("probe", "probe__l0123456789", "probe__l0123456789__1", "probe__1__l0123456789"):
        assert report.launch_name(*report.kernel_of(name)) == name

    clean = report.Binary("attention_forward", "0123456789", 64, 0, "triton",
                          shared=16384, sass=(), uses_tmem=False)
    uniquified_spill = report.Binary("attention_forward", "0123456789", 96, 24, "triton",
                                     shared=16384, sass=(), uses_tmem=False, index="1")
    distinct_stem = report.Binary("attention_forward__1", "0123456789", 48, 0, "triton",
                                  shared=8192, sass=(), uses_tmem=False)
    tmem_using = report.Binary("tmem_triton", "0123456789", 64, 0, "triton",
                               shared=16384, sass=(), uses_tmem=True)
    undecoded = report.Binary("undecoded_triton", "0123456789", 64, 0, "triton",
                              shared=16384, sass=(), uses_tmem=None)

    found = [kc.inherited(dataclasses.asdict(r), {}, "sm80")
             for r in kc.reports({}, [clean, uniquified_spill, distinct_stem, tmem_using, undecoded], {})]
    by_key = {report.launch_name(item["kernel"], item["launch"], item["index"]): item for item in found}
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
    mosaic_bin = report.Binary("attention_forward", "0123456789", 64, 0, "mosaic_gpu", shared=1024, sass=())
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
    """Row 7263: `report.binaries(..., decoded=True)` sets `uses_tmem=True` on an sm100 Mosaic GPU TMEM
    cubin, `uses_tmem=False` on an sm100 fp32 Triton cubin that allocates no TMEM, and `uses_tmem=None` when
    `decoded=False`."""
    import jax.numpy as jnp

    from dew.nn.kernels import generation
    from dew.nn.kernels.templates import attention as template, attention_mosaic, attention_route

    q_m = jax.ShapeDtypeStruct((1, 256, 2, 64), jnp.bfloat16)
    spans_m = template.spans(256, 256, 1, causal=True)
    with generation.assumed("sm100"):
        exe_m = kc.lowered_for(lambda q, k, v: attention_mosaic.forward(q, k, v, spans_m),
                               (q_m, q_m, q_m), "sm100").compile()
    mosaic_decoded = [b for b in report.binaries(exe_m, decoded=True) if b.kind == "mosaic_gpu"]
    mosaic_raw = [b for b in report.binaries(exe_m, decoded=False) if b.kind == "mosaic_gpu"]
    assert mosaic_decoded and any(b.uses_tmem is True for b in mosaic_decoded), mosaic_decoded
    assert mosaic_raw and all(b.uses_tmem is None for b in mosaic_raw), mosaic_raw

    q_t = jax.ShapeDtypeStruct((2, 1, 32, 256), jnp.float32)
    k_t = jax.ShapeDtypeStruct((2, 4096, 8, 256), jnp.float32)
    spans_t = template.spans(1, 4096, 2, causal=False)
    with generation.assumed("sm100"):
        exe_t = kc.lowered_for(lambda q, k, v: attention_route.attend(q, k, v, spans_t, natural=False),
                               (q_t, k_t, k_t), "sm100").compile()
    triton_decoded = [b for b in report.binaries(exe_t, decoded=True) if b.kind == "triton"]
    triton_raw = [b for b in report.binaries(exe_t, decoded=False) if b.kind == "triton"]
    assert triton_decoded and all(b.uses_tmem is False for b in triton_decoded), triton_decoded
    assert triton_raw and all(b.uses_tmem is None for b in triton_raw), triton_raw
