"""CPU first filters for the kernels: each of the fla port's declared configs, and the delta rule's and
the short conv's kernels at their training shapes, compiles for its GPU within the block's shared
memory, and the port's dots refuse mixed dtypes. None runs a kernel, so the GPU tier, whose kernel tests
are the verdict, runs none of them: the file carries no `kernel` mark."""

import dataclasses
import functools
import os
import re
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

import fla_gdn_support as support
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fla_gdn_support import CONV_SHAPES, conv_training_cases, kda_training_cases
from generation_support import COMPILES_FOR_TPU, generations, lowers
from jax.sharding import PartitionSpec
from kernel_compile import Uncounted, compile_kernels, failures

from dew.nn.kernels import chunked_launch, fla_gdn, fla_kda, generation, grouped_matmul, recurrence, tuning
from dew.nn.kernels.templates import bundles, report, resources
from dew.nn.kernels.templates.gemm import mosaic_gpu


def test_the_compiled_shapes_reach_the_widest_keys_and_values_the_port_takes():
    shapes = support.COMPILED_SHAPES.values()
    assert max(shape["width"] for shape in shapes) == fla_gdn.WIDEST_KEYS == 256
    assert max(shape["columns"] for shape in shapes) == fla_gdn.WIDEST == 128


def rule_launches(module) -> list:
    """The launches the rule of `module` (`fla_gdn`, `fla_kda`) makes forward and backward, as its trace
    records them (`tuning.seen`)."""
    per_key = module is fla_kda
    q = jax.ShapeDtypeStruct((1, 64, 1, 64), jnp.bfloat16)
    v = jax.ShapeDtypeStruct((1, 64, 2, 64), jnp.bfloat16)
    g = jax.ShapeDtypeStruct((1, 64, 2, *(64,) * per_key), jnp.float32)
    h0 = jax.ShapeDtypeStruct((1, 2, 64, 64), jnp.float32)
    port = fla_kda.chunk_kda if per_key else fla_gdn.chunk_gated_delta_rule

    def rule(*args):
        return port(*args, None, 0.125, 64, interpret=True)

    def both(q, k, v, g, beta, h0, do):
        # The output returned, so the forward's every launch is live: a dead one is DCE'd (jax 0.12).
        (out, final), pull = jax.vjp(rule, q, k, v, g, beta, h0)
        return out, pull((do, jnp.zeros_like(final)))

    beta = g.update(shape=g.shape[:3], dtype=jnp.bfloat16)
    with tuning.seen() as found:
        jax.make_jaxpr(both)(q, q, v, g, beta, h0, v)
    return [ba for _, ba in found]


@pytest.mark.parametrize("module", [fla_gdn, fla_kda])
def test_a_trace_finds_each_of_the_rules_launches(module):
    """The rule's forward and backward, traced (`tuning.seen`): each of its kernels is a launch; the
    gated delta rule's eight and the recurrence template's decay prefix, and Kimi delta attention's
    seven with the prefix and the memory kernels it shares."""
    shared = {recurrence.ChunkLocalCumsum, *((fla_gdn.ChunkFwdH, fla_gdn.ChunkBwdDhu) if module is fla_kda
                                             else ())}
    assert {type(ba.op) for ba in rule_launches(module)} == shared | {
        type(op) for op in vars(module).values() if isinstance(op, chunked_launch._Launch)}


def test_a_launch_clips_its_blocks_to_the_operands():
    """Blocks wider than a call's keys or values shrink to them: on an A100, whose picks block 64 values
    or more, the heuristic fits a call's 64-wide keys and 32-wide values."""
    q = jax.ShapeDtypeStruct((1, 64, 1, 64), jnp.bfloat16)
    v = jax.ShapeDtypeStruct((1, 64, 1, 32), jnp.bfloat16)
    h = jax.ShapeDtypeStruct((1, 1, 64, 32), jnp.bfloat16)
    g = jax.ShapeDtypeStruct((1, 64, 1), jnp.float32)
    spans = chunked_launch.sequences(1, 64)
    ba = fla_gdn._CHUNK_FWD_O.bind(q, q, v, h, g, None, *spans, scale=1.0, chunk=64, interpret=True)
    with generation.assumed("sm80"):
        tiles = fla_gdn._CHUNK_FWD_O._get_heuristics_config(ba)
    assert tiles.bk <= q.shape[-1] and tiles.bv <= v.shape[-1], tiles


@pytest.mark.parametrize("target", ["sm80", "sm120"])
def test_the_kernel_cases_lower_each_kernel_on_abstract_operands(target):
    """The compile gate's cases (`kernel_cases`) hold each kernel the gated delta rule launches forward
    and backward, at both dtypes at every compiled shape the dtype takes (fp32 keys at most
    `fla_gdn.WIDEST` wide), the decay prefix (`recurrence.cumulated`) reversed too, each once, on
    abstract operands of that shape lowered as on the GPU."""
    cases = list(support.kernel_cases(target))
    names = [case.name.split("/") for case in cases]
    assert len({tuple(name) for name in names}) == len(names)
    assert all(name[0] == target for name in names)
    covered = {(kernel, dtype, shape, *reverse) for _, kernel, dtype, _, shape, *reverse in names}
    launched = {ba.op.NAME for ba in rule_launches(fla_gdn)}
    reversals = {kernel: [(), ("reverse",)] if kernel == "cumulated" else [()]
                 for kernel in launched}
    assert covered == {(kernel, dtype, shape, *reverse)
                       for kernel in launched for dtype in ("bfloat16", "float32")
                       for shape, geometry in support.COMPILED_SHAPES.items()
                       if dtype == "bfloat16" or geometry["width"] <= fla_gdn.WIDEST
                       for reverse in reversals[kernel]}
    by_name = dict(zip((case.name for case in cases), cases, strict=True))
    for name, case in by_name.items():
        fn, args = case.call()
        geometry = support.COMPILED_SHAPES[name.split("/")[4]]
        assert fn.keywords["interpret"] is False
        assert args and all(isinstance(arg, jax.ShapeDtypeStruct) for arg in args)
        assert args[0].shape[:2] == (geometry["batch"], geometry["length"])


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("left,right", [(jnp.bfloat16, jnp.float32), (jnp.float32, jnp.bfloat16),
                                       (jnp.float16, jnp.bfloat16)])
def test_the_shared_dot_rejects_mismatches_before_interpret_widening(exact, left, right):
    a, b = jnp.ones((2, 2), left), jnp.ones((2, 2), right)
    with pytest.raises(TypeError):
        fla_gdn._dot(a, b, exact=exact)


def test_the_shared_dot_preserves_narrow_products_with_fp32_accumulation():
    a = jnp.asarray([[1.0, 0.5], [-0.25, 2.0]], jnp.bfloat16)
    got = fla_gdn._dot(a, a.T, exact=True)
    assert got.dtype == jnp.float32
    expected = a.astype(jnp.float32) @ a.T.astype(jnp.float32)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


@functools.cache
def _config_cases(target: str) -> dict:
    """Each declared config's case at each declared shape `target` runs it at (`support.kernel_cases`), by
    name."""
    return {case.name: case for case in support.kernel_cases(target)}


CONFIG_TESTS = [(target, name) for target in ("sm80", "sm89", "sm120") for name in _config_cases(target)]
"""A test each declared config at each declared shape and dtype it compiles at (`support.kernel_cases`, an
fp32 call's keys no wider than `fla_gdn.WIDEST`)."""


@pytest.mark.parametrize(("target", "name"), CONFIG_TESTS)
def test_every_declared_kernel_config_compiles_on_tritons_compiler_within_its_block(target, name):
    """Each declared config at each declared shape and dtype, a test each, lowered as on the GPU and compiled
    by XLA's CUDA backend for a compile-only device on this CPU (tests/kernel_compile.py): it compiles, to
    no more shared memory than its GPU's block holds. A kernel that does not compile (a two-result inline
    asm, a mixed-dtype dot) fails here and never reaches a GPU."""
    found = compile_kernels([_config_cases(target)[name]], target)
    assert found, "the compile gate must not be vacuous"
    assert not failures(found), "\n".join(failures(found))


@pytest.mark.parametrize("target", ["sm80", "sm89", "sm120"])
def test_kimi_delta_attentions_kernels_compile_on_tritons_compiler_within_each_gpus_block(target):
    """Every kernel of the port's forward and backward at the training shape, lowered as on the GPU and
    compiled by XLA's CUDA backend for a compile-only device on this CPU (tests/kernel_compile.py), within
    the compile bound and the block's shared memory."""
    found = compile_kernels(kda_training_cases(), target)
    assert len(found) == 2
    assert not failures(found), "\n".join(failures(found))


@pytest.mark.parametrize("target", ["sm80", "sm89", "sm120"])
def test_the_short_convs_kernels_compile_on_tritons_compiler_within_each_gpus_block(target):
    """Each of the short conv's training cases lowered as on the GPU and compiled by XLA's CUDA backend for a
    compile-only device on this CPU (tests/kernel_compile.py), within the compile bound and the block's
    shared memory."""
    found = compile_kernels(conv_training_cases(), target)
    assert len(found) == 2 * len(CONV_SHAPES) + 2
    assert not failures(found), "\n".join(failures(found))



@lowers("sm90")
def test_the_cooperative_grouped_matmul_compiles_for_hopper_within_its_resources():
    """The GEMM template's Mosaic GPU grouped matmul, lone and cooperative, compiled by Mosaic for a
    compile-only H100 on this CPU through ptxas (tests/kernel_compile.py): within the multiprocessor's
    shared memory, no register spilled, its register model deciding as ptxas does (`model_failures`, row
    1971). The resource gate every Mosaic GPU mainloop takes."""
    from kernel_compile import compile_mosaic_kernels, model_failures, mosaic_failures

    from dew.nn.kernels.templates.gemm import mosaic_gpu

    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    args = (bf16((1024, 512)), bf16((4, 512, 1024)), jax.ShapeDtypeStruct((4,), jnp.int32))
    launch = grouped_matmul.MosaicGmm()
    with generation.assumed("sm90"):
        ba = launch.bind(*args, out_dtype="bfloat16", trans_rhs=False, interpret=False)

    def case(tiles):
        with generation.assumed("sm90"):
            model = _model(launch, ba, tiles)
        return support.KernelCase(f"gmm {tiles}", lambda: launch.lowering(ba, tiles), {"dew_gmm": model})

    found = compile_mosaic_kernels([case(mosaic_gpu.Tiles(128, 128, 64, 3, 1, family='wgmma')),
                                    case(mosaic_gpu.Tiles(128, 256, 64, 3, 1, warpgroups=2,
                                                          family='wgmma'))], "sm90")
    assert len(found) == 2 and all(record.get("kernels") for record in found), found
    failed = mosaic_failures(found) + model_failures(found)
    assert not failed, "\n".join(failed)


@lowers("sm90")
def test_a_dense_gemm_mainloop_keeps_each_steps_products_running_past_it():
    """The dense GEMM's mainloop at a cluster pair's 128 x 256 tile, compiled for a compile-only H100 as the
    GPU compiles it, has its HGMMAs left running while the next step's operands are awaited: ptxas
    serializes none (its note C7514, a wait for each HGMMA). A guard on every step of a rotation, a cond
    between a warpgroup's products, had it serialize the walk (K40's SASS: 16 waits for 16 HGMMAs), a loop
    the gate's registers and spills called clean."""
    from kernel_compile import _compiled, ptxas_notes

    from dew.nn.kernels.templates.gemm import mosaic_gpu

    def call():
        def gmm(lhs, rhs, sizes):
            return mosaic_gpu.gmm(lhs, rhs, sizes, tiles=mosaic_gpu.Tiles(128, 256, 64, 4, 8, warpgroups=2,
                                                                          cluster=2, family='wgmma'),
                                  transpose_rhs=False, out_dtype=jnp.bfloat16, dense=True,
                                  multiprocessors=generation.gpu_target("sm90").multiprocessors)

        return gmm, (jax.ShapeDtypeStruct((4096, 4096), jnp.bfloat16),
                     jax.ShapeDtypeStruct((1, 4096, 4096), jnp.bfloat16),
                     jax.ShapeDtypeStruct((1,), jnp.int32))

    record = {"name": "dense gmm", "generation": "sm90", "limit": generation.gpu_target("sm90").shared_limit}
    with tempfile.TemporaryDirectory(prefix="dew-dense-gemm-") as directory:
        _compiled(support.KernelCase("dense gmm", call), "sm90", record, directory)
        assert "error" not in record, record["error"]
        (ptx,) = Path(directory).glob("*.ptx")
        notes = ptxas_notes(ptx.read_text(), "sm90")
    assert "bytes spill stores" in notes, notes
    assert "serialized" not in notes, notes


class _OnlyTile(Uncounted):
    """A launch with one tile, the wide one on a producer's layout, so refusing it exhausts its derivation
    (a tile refused at the planner's first stage count, two, gives way to the next layout, which takes the
    same shape: `wgmma.fed`)."""

    def candidate(self, ba, config, target):
        return (config.block_m == 128 and config.block_n == 256 and config.warpgroups == 2
                and config.family == "wgmma" and super().candidate(ba, config, target))


def _spilling_forward(launch, rows: int, devices):
    """A paired SwiGLU forward of `launch` at `rows` rows, its abstract operands placed on `devices`."""
    placed = jax.sharding.NamedSharding(devices, PartitionSpec())
    shapes = [jax.ShapeDtypeStruct(shape, dtype, sharding=placed)
              for shape, dtype in (((rows, 2048), jnp.bfloat16), ((8, 2048, 4096), jnp.bfloat16),
                                   ((8, 2048, 4096), jnp.bfloat16), ((8,), jnp.int32))]

    def forward(x, gate, up, sizes):
        return launch(x, gate, sizes, paired=up, hooks=grouped_matmul.Hooks(
            epilogue=lambda g, u: jax.nn.silu(g) * u), out_dtype="bfloat16", trans_rhs=False, interpret=False)

    return forward, shapes


def _lowering(forward, shapes, traced: list):
    """`forward` lowered at `shapes` afresh at each call, the config its launch takes added to `traced`."""
    def lowered():
        with tuning.seen() as seen:
            program = jax.jit(forward).trace(*shapes).lower()
        traced.append(seen.taken[0][0])
        return program

    return lowered


@lowers("sm90")
@pytest.mark.parametrize("exhausted", [False, True])
def test_a_spilling_launch_is_refused_and_its_derivation_steps_down_once(exhausted):
    """A compile that finds a template launch's kernel spilling (`report.compiled`, owner 10-29) refuses that
    launch's config and lowers once more, where the derivation takes its next tile by its own rule; a
    kernel spilling again is recorded and not stepped again. A paired SwiGLU forward whose model counts no
    registers, compiled for a compile-only H100 on this CPU through ptxas. A derivation with no replacement
    keeps its first program; the refusals saved with that program must derive it again."""
    from dew.nn.kernels.templates import report
    from dew.training.ahead import compile_only_gpus

    devices = jax.sharding.Mesh(np.array(compile_only_gpus("sm90")), ("data",))
    traced = []
    lowered = _lowering(*_spilling_forward(_OnlyTile() if exhausted else Uncounted(), 4096, devices), traced)

    with (report.refusals(), generation.assumed("sm90"),
          jax.sharding.use_abstract_mesh(generation.gpu_mesh(devices, "sm90"))):
        original = lowered()
        report.compiled(original, relower=lowered)
        misses, refused = report.misses(), report.refused_launches()
    first = traced[0]
    if exhausted:
        assert len(traced) == 1
        assert [(miss.config, miss.stepped) for miss in misses] == [(first, False)]
        assert not refused
        kept = first
    else:
        first, kept = traced
        assert kept != first
        assert [(miss.config, miss.stepped) for miss in misses][:1] == [(first, True)]
        assert all(miss.config == kept and not miss.stepped for miss in misses[1:])
        assert [config for _, config, _ in refused] == [first]
    # Another process seeded with the run's refusals (its plan's) derives the tile this compile kept.
    with (report.refusals(), generation.assumed("sm90"),
          jax.sharding.use_abstract_mesh(generation.gpu_mesh(devices, "sm90"))):
        report.refuse(refused)
        lowered()
    assert traced[-1] == kept
    # An already compiled artifact emits no new ptxas report. A fresh decision scope still steps past
    # its observed spill, while an exhausted derivation retains the artifact without a false refusal.
    with (report.refusals(), generation.assumed("sm90"),
          jax.sharding.use_abstract_mesh(generation.gpu_mesh(devices, "sm90"))):
        report.compiled(original, relower=lowered)
        assert [(miss.config, miss.stepped) for miss in report.misses()[:1]] == [
            (miss.config, miss.stepped) for miss in misses[:1]]
        assert report.refused_launches() == refused


@lowers("sm90")
def test_a_compile_beside_other_threads_mosaic_compiles_reads_only_its_own_kernels():
    """A run compiles its programs from several threads at once (scoring's pass shapes, row 2790). A compile
    through `report.compiled` beside a thread compiling other spilling Mosaic programs throughout finishes,
    refuses its own launch alone and steps it down: a scoring run crashed here, the compile's dump directory
    removed under the other thread's writes, and the other thread's kernels read as the compile's own."""
    from concurrent.futures import ThreadPoolExecutor

    from dew.nn.kernels.templates import report
    from dew.training.ahead import compile_only_gpus

    devices = jax.sharding.Mesh(np.array(compile_only_gpus("sm90")), ("data",))
    mesh = generation.gpu_mesh(devices, "sm90")
    started, stop = threading.Event(), threading.Event()

    def others():
        # Launches of other rows, so other kernels, spilling as the reading compile's does; a failure
        # releases the wait for it too, and its result raises it.
        rows = 4096
        try:
            with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh):
                while not stop.is_set():
                    rows += 64
                    forward, shapes = _spilling_forward(Uncounted(), rows, devices)
                    lowered = jax.jit(forward).trace(*shapes).lower()
                    started.set()
                    lowered.compile()
        finally:
            started.set()

    traced = []
    lowered = _lowering(*_spilling_forward(Uncounted(), 4096, devices), traced)

    with ThreadPoolExecutor(1) as pool:
        other = pool.submit(others)
        started.wait()
        try:
            with (report.refusals(), generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh)):
                report.compiled(lowered(), relower=lowered)
                refused = report.refused_launches()
        finally:
            stop.set()
        other.result()
    first, kept = traced
    assert kept != first
    assert [config for _, config, _ in refused] == [first]


GEMM_SHAPES = {"qwen3 gate": (4096, 2048, 768, 128), "mixtral up": (4096, 4096, 14336, 8),
               "mixtral down": (4096, 14336, 4096, 8), "deepseek-v3 up": (4096, 7168, 2048, 256),
               "deepseek-v3 decode down": (64, 2048, 7168, 256),
               "llama up": (4096, 4096, 14336, 1), "qwen3.5-4b up": (24576, 2560, 9216, 1),
               "qwen3.5-4b down": (24576, 9216, 2560, 1), "qwen3.5-4b key": (24576, 2560, 1024, 1),
               "qwen3.5-4b gates": (24576, 2560, 64, 1), "narrow kernel gradient": (4096, 64, 256, 1)}
"""Real projections' extents `(rows, depth, width, groups)` for the GEMM mainloops' compile gate, over which
the derived tiles move: MoE expert products (Qwen3-30B-A3B's, Mixtral's and DeepSeek-V3's: rows a few tiles a
group, or less than one, a wide and a deep product, and a decode step's, 8 tokens' top-8 rows over 256
experts), dense ones (Llama 3 8B's up, and Qwen3.5-4B's training step at its 24,576 rows a device: its widest,
deepest, narrow key and narrowest gate products, its HLO 10-28), and a kernel gradient of 64 output rows,
which one consumer computes."""

PLAIN_FORMS = ("forward", "forward bf16", "forward fp16", "input gradient", "input gradient fp16",
               "kernel gradient", "kernel gradient fp32", "kernel gradient fp16")
"""The forms every family's GEMM launches emit: each product and output dtype they take."""

FP8_FORMS = ("forward fp8",)
"""The FP8 forms a family whose products take FP8 operands emits (`Family.takes_fp8`)."""

HOOKED_FORMS = ("kernel gradient exact fp32", "normed biased forward", "normed biased input gradient",
                "normed kernel gradient", "exact normed kernel gradient", "fp16 normed biased forward",
                "cast forward", "gated forward", "gated kernel gradient", "residual forward",
                "two bf16 outputs forward", "scaled forward", "row scaled forward")
"""The hooked and exactly summed forms a family whose model counts what they hold emits
(`Family.counts_hooks`): prologues (a norm, a cast, SwiGLU's gate tile), epilogues over side vectors and a
side tile, a second output, and the running sum, each priced by what it materializes
(`grouped_matmul._Mosaic.registers`)."""

PAIRED_FORMS = ("paired swiglu forward", "paired forward")
"""The paired products a family whose consumers issue no ring's loads emits beside those counting hooks,
their only hook the epilogue over their two accumulators (`grouped_matmul._pair_alone`, row 2100)."""


def _gemm_forms(rows: int, depth: int, width: int, groups: int) -> list:
    """Each GEMM Mosaic launch form (`PLAIN_FORMS`, `PAIRED_FORMS`, `HOOKED_FORMS`) at a projection's
    extents: its name, launch, abstract operands and options."""
    from test_gemm_template import (
        biased,
        casted,
        doubled,
        gated,
        normed,
        scaled,
        scaled_rows,
        silu_residual,
        swiglu,
    )

    shaped = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    vector = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.float32)
    sizes = jax.ShapeDtypeStruct((groups,), jnp.int32)
    x, w, dy = shaped((rows, depth)), shaped((groups, depth, width)), shaped((rows, width))
    x16, w16, dy16 = (jax.ShapeDtypeStruct(array.shape, jnp.float16) for array in (x, w, dy))
    # An FP8 product's operands (E4M3), its matrix K-major as Hopper's FP8 tensor cores read it (row 3620).
    x8, w8 = (jax.ShapeDtypeStruct(shape, jnp.float8_e4m3fn)
              for shape in ((rows, depth), (groups, width, depth)))
    gmm, tgmm, hooks = grouped_matmul.MosaicGmm(), grouped_matmul.MosaicTgmm(), grouped_matmul.Hooks
    norm = hooks(normed, (("rows",), ("depth",)))
    forward = {"interpret": False, "out_dtype": "bfloat16", "trans_rhs": False}
    backward = {"interpret": False, "out_dtype": "float32"}
    return [
        ("forward", gmm, (x, w, sizes), {**forward, "out_dtype": "float32"}),
        ("forward bf16", gmm, (x, w, sizes), forward),
        ("forward fp16", gmm, (x16, w16, sizes), {**forward, "out_dtype": "float16"}),
        ("forward fp8", gmm, (x8, w8, sizes),
         {**forward, "out_dtype": "float32", "trans_rhs": True, "exact_sums": True}),
        ("input gradient", gmm, (dy, w, sizes), {**backward, "trans_rhs": True}),
        ("input gradient fp16", gmm, (dy16, w16, sizes),
         {**forward, "out_dtype": "float16", "trans_rhs": True}),
        ("kernel gradient", tgmm, (x, dy, sizes), {**backward, "out_dtype": "bfloat16"}),
        ("kernel gradient fp32", tgmm, (x, dy, sizes), backward),
        ("kernel gradient fp16", tgmm, (x16, dy16, sizes), backward),
        ("kernel gradient exact fp32", tgmm, (x, dy, sizes), {**backward, "exact_sums": True}),
        ("normed biased forward", gmm,
         (x, w, sizes, (vector((rows,)), vector((depth,))), (vector((width,)),)),
         {**forward, "hooks": hooks(normed, (("rows",), ("depth",)), biased, (("columns",),))}),
        ("normed biased input gradient", gmm,
         (dy, w, sizes, (vector((rows,)), vector((width,))), (vector((depth,)),)),
         {**backward, "trans_rhs": True,
          "hooks": hooks(normed, (("rows",), ("depth",)), biased, (("columns",),), ("float32",))}),
        ("normed kernel gradient", tgmm, (x, dy, sizes, (vector((rows,)), vector((depth,)))),
         {**backward, "hooks": norm}),
        ("exact normed kernel gradient", tgmm, (x, dy, sizes, (vector((rows,)), vector((depth,)))),
         {**backward, "exact_sums": True, "hooks": norm}),
        ("fp16 normed biased forward", gmm,
         (x16, w16, sizes, (vector((rows,)), vector((depth,))), (vector((width,)),)),
         {**forward, "out_dtype": "float16",
          "hooks": hooks(normed, (("rows",), ("depth",)), biased, (("columns",),))}),
        ("cast forward", gmm, (x, w, sizes, (vector((depth,)),)),
         {**forward, "hooks": hooks(casted, (("depth",),))}),
        ("gated forward", gmm, (x, w, sizes, (shaped((rows, depth)),)),
         {**forward, "hooks": hooks(gated, (("rows", "depth"),))}),
        ("gated kernel gradient", tgmm, (x, dy, sizes, (shaped((rows, depth)),)),
         {**backward, "hooks": hooks(gated, (("rows", "depth"),))}),
        ("residual forward", gmm, (x, w, sizes, (), (vector((width,)), shaped((rows, width)))),
         {**forward,
          "hooks": hooks(epilogue=silu_residual, epilogue_axes=(("columns",), ("rows", "columns")))}),
        ("two bf16 outputs forward", gmm, (x, w, sizes),
         {**forward, "hooks": hooks(epilogue=doubled, out_dtypes=("bfloat16", "bfloat16"))}),
        ("scaled forward", gmm, (x, w, sizes, (), (vector((width,)),)),
         {**forward, "out_dtype": "float32",
          "hooks": hooks(epilogue=scaled, epilogue_axes=(("columns",),), out_dtypes=("float32",))}),
        ("row scaled forward", gmm, (x, w, sizes, (), (vector((rows,)),)),
         {**forward, "hooks": hooks(epilogue=scaled_rows, epilogue_axes=(("rows",),))}),
        ("paired swiglu forward", gmm, (x, w, sizes, (), (), w),
         {**forward, "hooks": hooks(epilogue=swiglu)}),
        ("paired forward", gmm, (x, w, sizes, (), (), w),
         {**forward, "hooks": hooks(out_dtypes=("bfloat16", "bfloat16"))}),
    ]


def _gemm_mainloop_cases(generation_name: str, shape: str, *, every_consumer_count: bool = False) -> list:
    """Each GEMM Mosaic launch form a `generation_name` GPU runs at the projection `shape`
    (`GEMM_SHAPES`, `_gemm_forms`), at the one tile its launch derives from the generation's facts
    (`Template.derived`) and at each tile the derivation refused for its registers alone
    (`_derived_and_refused`); each a family's mainloop only where it takes the call
    (`Template.eligible`). With `every_consumer_count`, each form at one consumer and at the family's most
    (`Family.consumers`) where that tile is legal instead, each case named with its count (`x1`, `x2`)."""
    calls = _gemm_forms(*GEMM_SHAPES[shape])
    cases = []
    with generation.assumed(generation_name):
        target = generation.gpu_target()
        for name, launch, args, options in calls:
            ba = launch.bind(*args, **options)
            if not launch.eligible(ba, target):
                continue
            config, refused = _derived_and_refused(launch, ba, target)
            configs = {f"{name} {config}": config, **{f"{name} refused {tile}": tile for tile in refused}}
            if every_consumer_count:
                family = mosaic_gpu.family_of(config, jnp.bfloat16)
                configs = {}
                for consumers in sorted({1, family.consumers}):
                    # Each consumer keeps its share of the derived tile's rows, at as many of its stages as
                    # the block holds (a one-consumer tile's deeper ring past two consumers' rows); one
                    # consumer runs alone, a cluster being two consumers' (`mosaic_gpu.Tiles`).
                    rows_ = config.block_m // config.warpgroups * consumers
                    staged = (dataclasses.replace(config, warpgroups=consumers, block_m=rows_, stages=stages,
                                                  cluster=config.cluster if consumers > 1 else 1)
                              for stages in range(config.stages, 1, -1))
                    tile = next((tile for tile in staged if launch.legal(ba, tile, target)), None)
                    if tile is not None:
                        configs[f"{name} x{consumers} {tile}"] = tile
            kernel = "dew_tgmm" if isinstance(launch, grouped_matmul.MosaicTgmm) else "dew_gmm"
            for label, config in configs.items():
                cases.append(support.KernelCase(label,
                                                lambda launch=launch, ba=ba, config=config:
                                                launch.lowering(ba, config),
                                                {kernel: _model(launch, ba, config)},
                                                required=launch.legal(ba, config, target)))
    return cases


def _model(launch, ba, config) -> resources.RegisterModel:
    """What a launch's model decides at `config` for the call `ba`, as the gate holds it to the compiled
    kernel (`resources.register_model`, `kernel_compile.decisions`), and the shared memory the planner
    sized it by (`resources.shared_bytes` of `launch_of`'s allocations, including barriers), against the
    kernel's own."""
    family = mosaic_gpu.family_of(config, ba.arguments['tokens'].dtype)
    shared = resources.shared_bytes(family, launch.launch_of(ba, config).scope(config.stages))
    return resources.register_model(family, config.warpgroups, launch.registers(ba, config), shared=shared)


def _derived_and_refused(launch, ba, target) -> tuple:
    """A launch's derived tile, and the tiles its derivation asked the planner about (`launch_of`) that
    it refused for their registers alone (`resources.plan`: a ring its program's share holds, the registers
    not), whose need the gate checks refuses them too: a model over the need throws tiles away
    (`kernel_compile.decisions`). Each is the largest such of its rows, columns and layout, at the deepest
    ring the planner holds without the registers, the one a derivation would take had they held (the
    plans it asked, `resources.recording`)."""
    with resources.recording() as asked:
        config = launch.derived(ba, target)
    refused: dict = {}
    for each in asked:
        (tile,) = each.tiles
        if each.plans is not None:
            continue
        held = resources.plan([dataclasses.replace(launch.launch_of(ba, tile), registers=0)], each.family,
                              target, programs=each.programs)
        if held is not None:
            tile = dataclasses.replace(tile, stages=held[0].stages)
            shape = (tile.block_m, tile.block_n, tile.block_k, tile.warpgroups, tile.family)
            refused[shape] = max(refused.get(shape, tile), tile, key=lambda each: each.stages)
    return config, list(refused.values())


def _emitted_forms(family, shape: str | None = None) -> set[str]:
    """The forms a family's GEMM launches emit (`PLAIN_FORMS`, `HOOKED_FORMS`, `FP8_FORMS`): none without a
    bundle, the hooked and exactly summed ones where its register model counts what they hold
    (`Family.counts_hooks`), the FP8 ones where its products take FP8 (`Family.takes_fp8`);
    at a projection `shape`, each whose output's width steps by the family's column unit, whose stores take
    it (`Family.stores_columns`: a grouped product's rows ragged to one, a kernel gradient's whole role's)
    and whose contraction steps by its narrowest depth (`Family.column_unit`, `Family.depths`)."""
    if family is None:
        return set()
    forms = {*PLAIN_FORMS, *(PAIRED_FORMS if family.counts_hooks or not family.feeds_itself else ()),
             *(HOOKED_FORMS if family.counts_hooks else ()),
             *(FP8_FORMS if family.takes_fp8 else ())}
    if shape is None:
        return forms
    _, depth, width, _ = GEMM_SHAPES[shape]

    def takes(form: str) -> bool:
        # Each form at its operands' units: an FP8 product's column unit and narrowest step are FP8's (128
        # elements a 128-byte swizzle), the others' bf16's.
        dtype = jnp.float8_e4m3fn if form in FP8_FORMS else jnp.bfloat16
        unit, step = family.column_unit(dtype), min(family.depths(dtype))
        # An input gradient's output is the depth wide, contracting the width; every other form's output
        # is the width wide, a kernel gradient's contracting the rows (padded to its steps).
        columns, contracted = (depth, width) if "input gradient" in form else (width, depth)
        stored = family.role_rows if "kernel gradient" in form else 1
        return (columns % unit == 0 and family.stores_columns(columns, stored)
                and ("kernel gradient" in form or contracted % step == 0))

    return {form for form in forms if takes(form)}


@pytest.mark.parametrize("shape", GEMM_SHAPES)
@pytest.mark.parametrize("generation_name", generations("sm80", "sm86", "sm89", "sm90", "sm100", "sm120"))
def test_every_gemm_form_derives_a_tile_at_every_shape(generation_name, shape):
    """Every GEMM Mosaic GPU launch form a generation's family emits (`_emitted_forms`) derives a tile at
    every real projection (`_gemm_mainloop_cases`): an eligibility regression fails here rather than leaving
    the compile gate fewer cases. A GPU whose family has no bundle (`bundles.of`: a B200's tcgen05, consumer
    Blackwell's TMA-fed mma.sync) emits none."""
    cases = _gemm_mainloop_cases(generation_name, shape)
    forms = _emitted_forms(bundles.of(generation_name, jnp.bfloat16), shape)
    assert {case.name.split(" Tiles(")[0].removesuffix(" refused") for case in cases} == forms


def _kernel_of(name: str) -> str:
    """A case's kernel: its form at its tile, its band (`Tiles.grid_block_n`, the order its tiles are
    walked) apart, its shared memory and registers the tile's; a count the form's name carries (` x2`, the
    family contract's) apart too, the tile's consumers stating it."""
    form, _, tile = name.partition(" Tiles(")
    return f"{re.sub(r' x\d+$', '', form)} Tiles({re.sub(r'grid_block_n=\d+, ', '', tile)}"


@functools.cache
def _gate_cases(generation_name: str) -> list:
    """Every projection's cases (`_gemm_mainloop_cases`), each distinct kernel once (`_kernel_of`)."""
    found = {}
    for shape in GEMM_SHAPES:
        for case in _gemm_mainloop_cases(generation_name, shape):
            found.setdefault(_kernel_of(case.name), case)
    return list(found.values())


def _gated(names: tuple[str, ...], cases: Callable[[str], list]) -> list:
    """A test each case `cases(generation)` gives for each generation of `names`, marked as `generations`
    marks it: `(generation, case's index)`, each id the generation and the case's name."""
    return [pytest.param(name, index, marks=param.marks, id=f"{name}-{case.name}")
            for param in generations(*names) for name in param.values
            for index, case in enumerate(cases(name))]


@pytest.mark.parametrize(("generation_name", "index"),
                         _gated(("sm80", "sm86", "sm89", "sm90", "sm100", "sm120"), _gate_cases))
def test_each_gemm_mainloop_compiles_at_its_picks_within_each_gpus_resources(generation_name, index):
    """Every distinct GEMM Mosaic GPU kernel a generation runs at a real projection (`_gate_cases`), a test
    each, at the tile its launch derives from the generation's facts and at each tile the derivation refused
    for its registers, compiled by Mosaic for a compile-only device of the generation on this CPU through
    ptxas. Required launches must spill nothing, and their register model decide as ptxas does
    (`model_failures`: the GEMM's models match the compiler on every case, row 1971). Rejected
    configurations supply model evidence; their spills are refusals, never qualified launches. Compiler
    errors and missing reports fail either."""
    from kernel_compile import compile_mosaic_kernels, model_failures, mosaic_failures

    cases = _gate_cases(generation_name)[index:index + 1]
    found = compile_mosaic_kernels(cases, generation_name)
    failed = failures(found)
    required = [record for case, record in zip(cases, found, strict=True) if case.required]
    failed += mosaic_failures(required) + model_failures(required)
    assert not failed, "\n".join(failed)
    assert all(record["kernels"] and all(not kernel["unread"] for kernel in record["kernels"])
               for record in found)


@functools.cache
def _family_cases(generation_name: str) -> list:
    """The family contract's cases (`_gemm_mainloop_cases(every_consumer_count=True)` at the first
    projection) the compile gate does not already hold (`_gate_cases`)."""
    held = {_kernel_of(case.name) for case in _gate_cases(generation_name)}
    return [case for case in _gemm_mainloop_cases(generation_name, next(iter(GEMM_SHAPES)),
                                                  every_consumer_count=True)
            if _kernel_of(case.name) not in held]


@pytest.mark.parametrize("generation_name", generations("sm80", "sm90"))
def test_each_family_emits_every_launch_form_at_each_consumer_count(generation_name):
    """The whole contract a Mosaic GPU Family bundle offers its templates is present: every launch form it
    emits (`_emitted_forms`: plain and transposed right operands, 16-bit and fp32 stores, fixed and streamed
    side vectors and tiles, several outputs, the running sum) at one consumer and at the family's most
    (`_gemm_mainloop_cases(every_consumer_count=True)`)."""
    family = bundles.of(generation_name, jnp.bfloat16)
    assert family is not None
    cases = _gemm_mainloop_cases(generation_name, next(iter(GEMM_SHAPES)), every_consumer_count=True)
    counts = sorted({1, family.consumers})
    expected = {f"{form} x{count}" for form in _emitted_forms(family) for count in counts}
    got = {case.name.split(" Tiles(")[0] for case in cases}
    assert got == expected, (sorted(expected - got), sorted(got - expected))


@pytest.mark.parametrize(("generation_name", "index"), _gated(("sm80", "sm90"), _family_cases))
def test_each_family_holds_its_contract_on_the_pinned_jax(generation_name, index):
    """Each launch of the contract a Mosaic GPU Family bundle offers its templates that the compile gate
    does not hold (`_family_cases`), a test each, compiled through ptxas for a compile-only GPU of the
    generation on this CPU with the JAX this checkout pins, before any consumer adopts a Family head: it
    lowers, within the block, spilling nothing. No version guard skips it: a JAX that breaks the contract
    fails here."""
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    cases = _family_cases(generation_name)[index:index + 1]
    found = compile_mosaic_kernels(cases, generation_name)
    failed = mosaic_failures(found)
    assert len(found) == len(cases) and not failed, "\n".join(failed)


@pytest.fixture
def libtpu_shared(monkeypatch):
    """libtpu loaded in this process beside another's: its lock file holds one process's libtpu at a time,
    so two test workers compiling for a TPU at once failed the second's load (ABORTED, /tmp/libtpu_lockfile).
    A compile-only TPU needs no device a lock guards, so each loads with libtpu's own allowance and takes no
    lock file."""
    monkeypatch.setenv("ALLOW_MULTIPLE_LIBTPU_LOAD", "1")


@COMPILES_FOR_TPU
@pytest.mark.usefixtures("libtpu_shared")
@pytest.mark.parametrize("tpu", ["v5e", "v6e"])
def test_each_gemm_tpu_mainloop_compiles_at_its_picks_with_libtpu(tpu):
    """The GEMM template's TPU mainloops (`MosaicTpuGmm` forward and input gradient, `MosaicTpuTgmm`),
    at the block their launch derives from the TPU's facts, compiled by libtpu for a compile-only TPU of
    the generation on this CPU; a generation whose scoped VMEM no measurement states takes none of them
    (`MosaicTpuGmm.eligible`)."""
    from kernel_compile import compile_tpu_kernels

    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    # A Qwen3 MoE gate-up's widths, rows a few tiles a group.
    rows, depth, width, groups = 4096, 2048, 1536, 8
    sizes = jax.ShapeDtypeStruct((groups,), jnp.int32)
    calls = [("forward", grouped_matmul.MosaicTpuGmm(),
              (bf16((rows, depth)), bf16((groups, depth, width)), sizes),
              {"out_dtype": "bfloat16", "trans_rhs": False}),
             ("input gradient", grouped_matmul.MosaicTpuGmm(),
              (bf16((rows, width)), bf16((groups, depth, width)), sizes),
              {"out_dtype": "bfloat16", "trans_rhs": True}),
             ("kernel gradient", grouped_matmul.MosaicTpuTgmm(),
              (bf16((rows, depth)), bf16((rows, width)), sizes), {"out_dtype": "bfloat16"})]
    cases = []
    with generation.assumed(tpu):
        target = generation.gpu_target()
        for name, launch, args, options in calls:
            ba = launch.bind(*args, interpret=False, **options)
            if not launch.eligible(ba, target):
                continue
            config = launch.derived(ba, target)
            configured = functools.partial(launch.replace(config=config), interpret=False, **options)
            cases.append(support.KernelCase(f"{name} {config}",
                                            lambda configured=configured, args=args: (configured, args)))
    if not cases:
        assert generation.gpu_target(tpu).scoped_vmem is None
        pytest.skip(f"no measurement states {tpu}'s scoped VMEM, which the TPU launches fit their blocks to")
    forms = {"forward", "input gradient", "kernel gradient"}
    assert {case.name.split(" Blocks(")[0] for case in cases} == forms
    found = compile_tpu_kernels(cases, tpu)
    failed = [f"{record['name']}: {record['error']}" for record in found if "error" in record]
    assert not failed, "\n".join(failed)


@pytest.mark.parametrize("generation_name", generations("sm80"))
def test_a_program_that_launches_no_triton_kernel_gives_no_evidence(generation_name):
    """A case whose program XLA compiles without a Triton kernel (an elementwise op, XLA's own fusion)
    states no kernel's shared bytes: the gate fails it, never reading the missing launch as none used
    (row 1381)."""
    operand = jax.ShapeDtypeStruct((8,), jnp.float32)
    case = support.KernelCase("no Triton kernel", lambda: (lambda x: x * 2, (operand,)))
    assert failures(compile_kernels([case], generation_name))


def test_each_triton_call_needs_a_launch_of_its_own():
    """A program's Triton calls, one named as another begins and one made twice, compiled for a compile-only
    A100: with every launch line XLA writes, the gate reads their shared bytes; without one of the two
    `probe` launches, or without `probe_headroom`'s, it has no evidence (row 1381), where matching a launch
    by a name's prefix took one launch for both calls."""
    from jax.experimental import pallas as pl
    from kernel_compile import _compiled, triton_evidence

    def double(x_ref, o_ref):
        o_ref[...] = x_ref[...] * 2

    def program(x):
        shape = jax.ShapeDtypeStruct(x.shape, x.dtype)

        def call(name, operand):
            return pl.pallas_call(double, out_shape=shape, name=name)(operand)
        return call("probe", call("probe_headroom", call("probe", x)) + 1)

    operand = jax.ShapeDtypeStruct((16, 128), jnp.float32)
    compiled = _compiled(support.KernelCase("probe", lambda: (program, (operand,))), "sm80", {})
    assert compiled is not None
    thunks, calls = compiled
    assert calls == {"probe": 2, "probe_headroom": 1}
    assert "shared" in triton_evidence(calls, thunks)
    for missing, said in ((" probe grid", "probe"), (" probe_headroom grid", "probe_headroom")):
        lines = "\n".join(thunks).splitlines()
        dropped = next(index for index, line in enumerate(lines) if missing in line)
        without = ["\n".join(lines[:dropped] + lines[dropped + 1:])]
        assert triton_evidence(calls, without) == {
            "error": f"no launch of its own states the shared bytes of {said}"}


def test_records_a_failed_process_of_their_own_wrote_are_refused():
    """A test that compiles Mosaic GPU kernels after another capability's runs again in a pytest process of
    its own (`kernel_compile.compiled_apart`), whose records are written before its own checks and teardown:
    a process that wrote them and then failed gives none (row 1877)."""
    from kernel_compile import OWN_PROCESS, compiled_apart

    if OWN_PROCESS in os.environ:
        Path(os.environ[OWN_PROCESS]).write_text("[]")
        pytest.fail("this process failed after it wrote its records, as a failed teardown does")
    with pytest.raises(RuntimeError):
        compiled_apart()


def test_the_resource_gate_refuses_spills_and_invalid_allocations():
    """A record the gate reads (`mosaic_failures`): a kernel past its shared memory, its tensor memory or
    spilling fails; a spill-free model disagreement is reported."""
    from kernel_compile import inherited, mosaic_failures, reports

    within = {"kernel": "k", "unread": [], "shared": 1024, "tmem": 256, "registers": 168, "spilled": 0}
    record = {"name": "case", "limit": 2048, "tmem_limit": 512, "kernels": [within]}
    assert not mosaic_failures([record])
    for past in ({"shared": 4096}, {"tmem": 513}, {"spilled": 40}):
        assert mosaic_failures([{**record, "kernels": [{**within, **past}]}])
    overestimated = resources.RegisterModel(200, 168, 2, 1, 8, 168, shared=1024)
    disagreement = {**record, "models": {"k": overestimated.record()},
                    "kernels": [{**within, "need": 160}]}
    assert not mosaic_failures([disagreement])
    assert any(not row["agrees"] for row in disagreement["evidence"])
    # A kernel whose reports state no resources, spilling nothing, is no evidence: it fails, as a record
    # with no kernel does (row 1877: a contract gate read only its shared count and positive spills).
    unstated = {**within, "unread": ["shared bytes", "tensor memory columns"], "shared": None, "tmem": None}
    assert mosaic_failures([{**record, "kernels": [unstated]}])
    assert mosaic_failures([{**record, "kernels": []}])
    # The program's own refusal of a block past shared memory fails it whatever its Mosaic kernels say.
    assert mosaic_failures([{**record, "shared": 4096}])
    # A binary without an estimate, an estimate without a field, or a kernel whose PTX no compile covered,
    # is read as missing, not as within.
    complete = "Resources(smem_scratch_bytes=1024, tmem_scratch_cols=0, tmem_collective_scratch_cols=0)"
    compiled = {"a": {"registers": 168, "spilled": 0}, "b": {"registers": 255, "spilled": 40}}
    a, b, c = (inherited(dataclasses.asdict(found), compiled) for found in reports(
        {"a": complete, "c": "Resources(smem_scratch_bytes=1024)"},
        [report.Binary("a", None, 168, 0, shared=1024), report.Binary("b", None, 255, 40, shared=1024)]))
    assert not a["unread"]
    assert set(b["unread"]) == {"shared bytes", "tensor memory columns"}
    assert set(c["unread"]) == {"shared bytes", "tensor memory columns", "binary", "registers", "spills"}
    assert len(mosaic_failures([{**record, "kernels": [a, b, c]}])) == 2
    # An estimate past the shared memory its binary's launch asks for is another kernel's: it raises.
    with pytest.raises(ValueError):
        reports({"a": complete}, [report.Binary("a", None, 168, 0, shared=512)])
    # A complete estimate and its PTX's compile without a binary: no compiled kernel, read as missing.
    (unbound,) = (inherited(dataclasses.asdict(found), compiled) for found in reports({"a": complete}, []))
    assert "binary" in unbound["unread"]
    assert mosaic_failures([{**record, "kernels": [unbound]}])


def test_a_gate_holding_its_models_fails_a_spill_free_disagreement_and_an_unmodelled_launch():
    """A family's gate that holds its register model to ptxas (`model_failures`, row 1971) fails a kernel
    whose model decides otherwise than ptxas (here four programs where ptxas holds two) and a launch with
    no model, as the resource gate fails a spill; a kernel whose model agrees passes."""
    from kernel_compile import model_failures, mosaic_failures

    need = {"admitted": True, "programs": 2, "budget": 255, "refusal_holds": True, "refusal_programs": 2,
            "caps": [{"cap": 255, "registers": 96, "spilled": 0, "compiled": True, "holds": True}]}
    kernel = {"kernel": "dew_gmm", "launch": None, "unread": [], "shared": 1024, "tmem": 0, "registers": 96,
              "spilled": 0, "need": need}

    def judged(most: int) -> list[str]:
        model = resources.RegisterModel(96, 255, 1, most, 8, 0, shared=1024)
        record = {"name": "forward", "generation": "sm120", "limit": 101376, "tmem_limit": 0,
                  "models": {}, "lowered_models": {"dew_gmm": model.record()}, "kernels": [kernel]}
        assert not mosaic_failures([record])
        return model_failures([record])

    assert judged(most=2) == []
    assert judged(most=4)
    unmodelled = {"name": "forward", "generation": "sm90", "limit": 232448, "tmem_limit": 0, "models": {},
                  "lowered": {"dew_gmm": ["launch"]}, "kernels": [kernel]}
    mosaic_failures([unmodelled])
    assert model_failures([unmodelled])


def test_a_kernel_is_judged_by_the_model_its_launch_lowered_not_the_cases_declared_one():
    """A kernel whose launch lowered its own register model (`lowered_models`: its plan's two programs)
    is judged by that model, whose residency ptxas's need was searched at: the case's declared model for
    its name (a family's four programs by registers alone) states no decision this launch made, so it
    adds no disagreement (row 1971: 381 of the gate's 511 recorded misses were the declared model's
    residency against the lowered launch's need)."""
    from kernel_compile import _register_evidence

    declared = resources.RegisterModel(112, 255, 1, 4, 8, 0, shared=49232)
    lowered = resources.RegisterModel(112, 255, 1, 2, 8, 0, shared=49232)
    need = {"admitted": True, "programs": 2, "budget": 255, "refusal_holds": True, "refusal_programs": 2,
            "caps": [{"cap": 255, "registers": 96, "spilled": 0, "compiled": True, "holds": True}]}
    kernel = {"kernel": "dew_tgmm", "launch": None, "unread": [], "shared": 49232, "tmem": 0,
              "registers": 96, "spilled": 0, "need": need}
    record = {"name": "kernel gradient", "generation": "sm120", "limit": 101376, "tmem_limit": 0,
              "models": {"dew_tgmm": declared.record()}, "lowered_models": {"dew_tgmm": lowered.record()},
              "kernels": [kernel]}
    rows = _register_evidence(record)
    assert rows and all(row["agrees"] for row in rows), rows


def test_the_resource_gate_refuses_two_kernels_of_one_name_and_launch():
    """Evidence of one kernel's name and launch from two kernels is one kernel's at most: two modules
    (`kernel_modules`), two binaries (`reports`) or two PTX modules (`write_modules`) of one raise."""
    from kernel_compile import PTX_MODULE, kernel_modules, reports, write_modules

    from dew.nn.kernels.lowered import KernelCall

    calls = [KernelCall("mosaic_gpu", "k", b"", module, None, "0123456789") for module in ("aa", "bb")]
    assert kernel_modules(calls[:1]) == {"k__l0123456789": "aa"}
    with pytest.raises(ValueError):
        kernel_modules(calls)
    binaries = [report.Binary("k", "0123456789", registers, 0) for registers in (168, 200)]
    assert len(reports({}, binaries[:1])) == 1
    with pytest.raises(ValueError):
        reports({}, binaries)
    module = f"//\n{PTX_MODULE}\n.visible .entry k__l0123456789_mosaic_gpu_kernel(\n)\n{{\n}}\n"
    with tempfile.TemporaryDirectory() as directory:
        write_modules(module, directory)
        assert [path.name for path in Path(directory).iterdir()] == ["k__l0123456789.ptx"]
    with tempfile.TemporaryDirectory() as directory, pytest.raises(ValueError):
        write_modules(module * 2, directory)


@COMPILES_FOR_TPU
@pytest.mark.usefixtures("libtpu_shared")
def test_splash_compiles_for_a_v5e_with_libtpu():
    """Dew's splash path, forward and backward, compiled by libtpu for a compile-only v5e on this CPU
    (tests/kernel_compile.py): within the v5e's scoped VMEM, as the blocks splash takes are sized
    (`dew.nn.kernels.splash.block_sizes`)."""
    from kernel_compile import compile_tpu_kernels

    from dew.nn.attention import scaled_dot_product_attention

    def call():
        def step(query, key, value):
            def loss(q, k, v):
                return jnp.sum(scaled_dot_product_attention(q, k, v, implementation="tpu", causal=True)
                               .astype(jnp.float32))
            return jax.grad(loss, argnums=(0, 1, 2))(query, key, value)

        shape = jax.ShapeDtypeStruct((1, 2048, 4, 128), jnp.float32)
        return step, (shape, shape, shape)

    found = compile_tpu_kernels([support.KernelCase("splash fp32", call)], "v5e")
    assert not [record for record in found if "error" in record], found


def test_a_refusal_applies_only_to_the_target_capacities_that_generated_its_kernel():
    """The same operand shapes and tile can lower for different device capacities in one process. A
    refusal belongs to the target that generated it; rate-only differences do not change that tile's code."""
    import dataclasses

    from dew.nn.kernels.templates import report

    launch = grouped_matmul.MosaicGmm()
    x = jax.ShapeDtypeStruct((4096, 2048), jnp.bfloat16)
    weights = jax.ShapeDtypeStruct((8, 2048, 4096), jnp.bfloat16)
    sizes = jax.ShapeDtypeStruct((8,), jnp.int32)
    arguments = launch.bind(x, weights, sizes, out_dtype="bfloat16", trans_rhs=False, interpret=False)
    target = generation.TARGETS["sm90"]
    with report.refusals(), generation.assumed(target):
        tile = launch.derived(arguments, target)
        report.refuse([(report._identity(launch, arguments), tile, {"dew_gmm"})])
        assert report.refused(launch, arguments, tile)
        with generation.assumed(dataclasses.replace(target, clock_hz=1.98e9)):
            assert report.refused(launch, arguments, tile)
        with generation.assumed(dataclasses.replace(target, multiprocessors=64)):
            assert not report.refused(launch, arguments, tile)
        with generation.assumed(dataclasses.replace(target, shared_limit=128 * 1024)):
            assert not report.refused(launch, arguments, tile)


def test_refusals_cross_processes_only_through_an_explicit_handoff(tmp_path):
    """A shared compilation cache is not a shared derivation registry. A fresh process sees no foreign
    refusals; a handed plan can seed exactly the refusals it exported."""
    import json
    import subprocess
    import sys

    tile = "mosaic_gpu.Tiles(128, 256, 64, 4, 8, warpgroups=2, family='wgmma')"
    preamble = ("import json, sys\n"
                "from dew.nn.kernels import tuning\n"
                "from dew.nn.kernels.templates import report\n"
                "from dew.nn.kernels.templates.gemm import mosaic_gpu\n")
    snapshot = ("print(json.dumps([[identity, tuning.encoded(config), sorted(kernels)] "
                "for identity, config, kernels in report.refused_launches()]))")
    environment = {**os.environ, "JAX_COMPILATION_CACHE_DIR": str(tmp_path / "cache")}

    seed = preamble + f"report.refuse([('a launch', {tile}, {{'dew_gmm'}})])\n" + snapshot
    exported = subprocess.run([sys.executable, "-c", seed], check=True, env=environment,
                              capture_output=True, text=True).stdout
    assert len(json.loads(exported)) == 1
    fresh = subprocess.run([sys.executable, "-c", preamble + snapshot], check=True, env=environment,
                           capture_output=True, text=True)
    assert json.loads(fresh.stdout) == []
    handoff = (preamble + "report.refuse((identity, tuning.decoded(config), kernels) "
               "for identity, config, kernels in json.load(sys.stdin))\n" + snapshot)
    handed = subprocess.run([sys.executable, "-c", handoff], check=True, env=environment,
                            input=exported, capture_output=True, text=True)
    assert json.loads(handed.stdout) == json.loads(exported)
