"""CPU first filters for the kernels: each of the fla port's declared configs, and the delta rule's and
the short conv's kernels at their training shapes, compiles for its GPU within the block's shared
memory, and the port's dots refuse mixed dtypes. None runs a kernel, so the GPU tier, whose kernel tests
are the verdict, runs none of them: the file carries no `kernel` mark."""

import dataclasses
import functools
import os
import re
import tempfile
from pathlib import Path

import fla_gdn_support as support
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fla_gdn_support import CONV_SHAPES, conv_training_cases, kda_training_cases
from generation_support import COMPILES_FOR_TPU, generations, lowers
from jax.sharding import PartitionSpec
from kernel_compile import compile_kernels, failures

from dew.nn.kernels import chunked_launch, fla_gdn, fla_kda, generation, grouped_matmul, recurrence, tuning
from dew.nn.kernels.templates import bundles, resources
from dew.nn.kernels.templates.gemm import mosaic_gpu


def test_the_compiled_shapes_reach_the_widest_keys_and_values_the_port_takes():
    shapes = support.COMPILED_SHAPES.values()
    assert max(shape["width"] for shape in shapes) == fla_gdn.WIDEST_KEYS == 256
    assert max(shape["columns"] for shape in shapes) == fla_gdn.WIDEST == 128


@pytest.mark.parametrize("target, wide", [("sm80", True), ("sm90", True), ("sm86", False), ("sm89", False),
                                          ("sm120", False), ("sm100", True), ("sm110", False),
                                          ("cpu", False)])
def test_a_gpu_takes_an_a100s_picks_where_its_block_holds_as_much_shared_memory(target, wide):
    """An A100's picks on every GPU whose block holds as much shared memory, the 99 KB ones elsewhere: on a
    GPU the targets do not list (sm110), and on a CPU, which has no GPU's facts unless a test states the GPU
    it emulates (`generation.assumed`)."""
    with generation.assumed(target):
        assert chunked_launch._wide() is wide


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


CONFIG_CELLS = [(dtype, shape) for dtype in ("bfloat16", "float32")
                for shape, case in support.COMPILED_SHAPES.items()
                if not (dtype == "float32" and case["width"] > support.fla_gdn.WIDEST)]
"""Each dtype beside each declared shape it compiles at: an fp32 call takes no keys past `fla_gdn.WIDEST`
(`support.kernel_cases`)."""


@pytest.mark.parametrize(("dtype", "shape"), CONFIG_CELLS)
@pytest.mark.parametrize("target", ["sm80", "sm89", "sm120"])
def test_every_declared_kernel_config_compiles_on_tritons_compiler_within_its_block(target, dtype, shape):
    """Each declared config at each declared shape, of each dtype and shape apart (one test of a dtype's
    every shape outlasted the suite's 240 s target, row 1849), lowered as on the GPU and compiled by XLA's
    CUDA backend for a compile-only device on this CPU (tests/kernel_compile.py): it compiles, to no more
    shared memory than its GPU's block holds. A kernel that does not compile (a two-result inline asm, a
    mixed-dtype dot) fails here and never reaches a GPU."""
    cases = [case for case in support.kernel_cases(target, (dtype,)) if case.name.split("/")[4] == shape]
    found = compile_kernels(cases, target)
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
    shared memory, no register spilled. The resource gate every Mosaic GPU mainloop takes."""
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    from dew.nn.kernels.templates.gemm import mosaic_gpu

    def case(tiles):
        def call():
            def gmm(lhs, rhs, sizes):
                hopper = generation.gpu_target("sm90")
                return mosaic_gpu.gmm(lhs, rhs, sizes, tiles=tiles, transpose_rhs=False,
                                      out_dtype=jnp.bfloat16, multiprocessors=hopper.multiprocessors)

            return gmm, (jax.ShapeDtypeStruct((1024, 512), jnp.bfloat16),
                         jax.ShapeDtypeStruct((4, 512, 1024), jnp.bfloat16),
                         jax.ShapeDtypeStruct((4,), jnp.int32))

        return support.KernelCase(f"gmm {tiles}", call)

    found = compile_mosaic_kernels([case(mosaic_gpu.Tiles(128, 128, 64, 3, 1)),
                                    case(mosaic_gpu.Tiles(128, 256, 64, 3, 1, warpgroups=2))], "sm90")
    assert len(found) == 2 and all(record.get("kernels") for record in found), found
    assert not mosaic_failures(found), "\n".join(mosaic_failures(found))


class _Uncounted(grouped_matmul.MosaicGmm):
    """The grouped forward with a register model that counts nothing, so its derivation takes a tile its
    registers do not hold: a launch whose model misses."""

    def registers(self, ba, config):
        return 0


class _OnlyTile(_Uncounted):
    """A launch with one tile shape, so refusing that shape exhausts its derivation."""

    def candidate(self, ba, config, target):
        return (config.block_m == 128 and config.block_n == 128 and config.warpgroups == 2
                and super().candidate(ba, config, target))


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

    launch = _OnlyTile() if exhausted else _Uncounted()
    devices = jax.sharding.Mesh(np.array(compile_only_gpus("sm90")), ("data",))
    placed = jax.sharding.NamedSharding(devices, PartitionSpec())
    shapes = [jax.ShapeDtypeStruct(shape, dtype, sharding=placed)
              for shape, dtype in (((4096, 2048), jnp.bfloat16), ((8, 2048, 4096), jnp.bfloat16),
                                   ((8, 2048, 4096), jnp.bfloat16), ((8,), jnp.int32))]
    traced = []

    def lowered():
        def forward(x, gate, up, sizes):
            return launch(x, gate, sizes, paired=up, hooks=grouped_matmul.Hooks(
                epilogue=lambda g, u: jax.nn.silu(g) * u), out_dtype="bfloat16", trans_rhs=False,
                          interpret=False)
        with tuning.seen() as seen:
            program = jax.jit(forward).trace(*shapes).lower()
        traced.append(seen.taken[0][0])
        return program

    with (report.refusals(), generation.assumed("sm90"),
          jax.sharding.use_abstract_mesh(generation.gpu_mesh(devices, "sm90"))):
        report.compiled(lowered(), relower=lowered)
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


GEMM_SHAPES = {"qwen3 gate": (4096, 2048, 768, 128), "mixtral up": (4096, 4096, 14336, 8),
               "mixtral down": (4096, 14336, 4096, 8), "deepseek-v3 up": (4096, 7168, 2048, 256),
               "llama up": (4096, 4096, 14336, 1), "qwen3.5-4b up": (24576, 2560, 9216, 1),
               "qwen3.5-4b down": (24576, 9216, 2560, 1), "qwen3.5-4b key": (24576, 2560, 1024, 1),
               "qwen3.5-4b gates": (24576, 2560, 64, 1), "narrow kernel gradient": (4096, 64, 256, 1)}
"""Real projections' extents `(rows, depth, width, groups)` for the GEMM mainloops' compile gate, over which
the derived tiles move: MoE expert products (Qwen3-30B-A3B's, Mixtral's and DeepSeek-V3's: rows a few tiles a
group, or less than one, a wide and a deep product), dense ones (Llama 3 8B's up, and Qwen3.5-4B's training
step at its 24,576 rows a device: its widest, deepest, narrow key and narrowest gate products, its HLO
10-28), and a kernel gradient of 64 output rows, which one consumer computes."""

PLAIN_FORMS = ("forward", "forward bf16", "forward fp16", "input gradient", "input gradient fp16",
               "kernel gradient", "kernel gradient fp32", "kernel gradient fp16")
"""The forms every family's GEMM launches emit: each product and output dtype they take."""

HOOKED_FORMS = ("kernel gradient exact fp32", "scaled forward", "row scaled forward", "paired swiglu forward")
"""The hooked and exactly summed forms whose register need the family's model prices (`Family.counts_hooks`,
`grouped_matmul._priced`): epilogues over side vectors (a column or row scale), a paired product's SwiGLU
over its two accumulators, and the running sum. A prologue, an epilogue's side tile and a second output are
not yet priced; their calls take another launch."""


def _gemm_forms(rows: int, depth: int, width: int, groups: int) -> list:
    """Each GEMM Mosaic launch form (`PLAIN_FORMS`, `HOOKED_FORMS`) at a projection's extents: its name,
    launch, abstract operands and options."""
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
    gmm, tgmm, hooks = grouped_matmul.MosaicGmm(), grouped_matmul.MosaicTgmm(), grouped_matmul.Hooks
    norm = hooks(normed, (("rows",), ("depth",)))
    forward = {"interpret": False, "out_dtype": "bfloat16", "trans_rhs": False}
    backward = {"interpret": False, "out_dtype": "float32"}
    return [
        ("forward", gmm, (x, w, sizes), {**forward, "out_dtype": "float32"}),
        ("forward bf16", gmm, (x, w, sizes), forward),
        ("forward fp16", gmm, (x16, w16, sizes), {**forward, "out_dtype": "float16"}),
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
                    # Each consumer keeps its share of the derived tile's rows; one consumer runs alone,
                    # a cluster being two consumers' (`mosaic_gpu.Tiles`).
                    rows_ = config.block_m // config.warpgroups * consumers
                    tile = dataclasses.replace(config, warpgroups=consumers, block_m=rows_,
                                               cluster=config.cluster if consumers > 1 else 1)
                    if launch.legal(ba, tile, target):
                        configs[f"{name} x{consumers} {tile}"] = tile
            kernel = "dew_tgmm" if isinstance(launch, grouped_matmul.MosaicTgmm) else "dew_gmm"
            for label, config in configs.items():
                configured = (functools.partial(launch.replace(config=config), **options)
                              if launch.legal(ba, config, target) else
                              # A tile the launch refuses runs its kernel unpinned (`KernelOp.pinnable`).
                              lambda *arrays, config=config, options=options, launch=launch:
                              launch._fwd(*arrays, **options, config=config)[0])
                cases.append(support.KernelCase(label,
                                                lambda configured=configured, args=args: (configured, args),
                                                {kernel: _model(launch, ba, config)}))
    return cases


def _model(launch, ba, config) -> resources.RegisterModel:
    """What a launch's model decides at `config` for the call `ba`, as the gate holds it to the compiled
    kernel (`resources.register_model`, `kernel_compile.decisions`), and the scratch the planner sized it by
    (`resources.scratch_bytes` of `launch_of`'s allocations, its barriers apart), against the kernel's
    own."""
    family = mosaic_gpu.family_of(config, ba.arguments['tokens'].dtype)
    shared = resources.scratch_bytes(family, launch.launch_of(ba, config).scope(config.stages))
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
    """The forms a family's GEMM launches emit (`PLAIN_FORMS`, `HOOKED_FORMS`): none without a bundle, the
    hooked and exactly summed ones where its register model counts what they hold (`Family.counts_hooks`);
    at a projection `shape`, each whose output's width steps by the family's column unit, whose stores take
    it (`Family.stores_columns`: a grouped product's rows ragged to one, a kernel gradient's whole role's)
    and whose contraction steps by its narrowest depth (`Family.column_unit`, `Family.depths`)."""
    if family is None:
        return set()
    forms = {*PLAIN_FORMS, *(HOOKED_FORMS if family.counts_hooks else ())}
    if shape is None:
        return forms
    _, depth, width, _ = GEMM_SHAPES[shape]
    unit, step = family.column_unit(jnp.bfloat16), min(family.depths(jnp.bfloat16))

    def takes(form: str) -> bool:
        # An input gradient's output is the depth wide, contracting the width; every other form's output
        # is the width wide, a kernel gradient's contracting the rows (padded to its steps).
        columns, contracted = (depth, width) if "input gradient" in form else (width, depth)
        stored = family.role_rows if "kernel gradient" in form else 1
        return (columns % unit == 0 and family.stores_columns(columns, stored)
                and ("kernel gradient" in form or contracted % step == 0))

    return {form for form in forms if takes(form)}


GATE_PARTS = 6
"""The parts each generation's compile gate runs in apart (`test_each_gemm_mainloop_compiles_at_its_picks`),
each within a CI group's time: sm90's 85 distinct kernels at about 4.7 s each (GATEPROF1, GATECOUNT1)."""


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


def _gate_cases(generation_name: str) -> list:
    """Every projection's cases (`_gemm_mainloop_cases`), each distinct kernel once (`_kernel_of`)."""
    found = {}
    for shape in GEMM_SHAPES:
        for case in _gemm_mainloop_cases(generation_name, shape):
            found.setdefault(_kernel_of(case.name), case)
    return list(found.values())


@pytest.mark.parametrize("part", range(GATE_PARTS))
@pytest.mark.parametrize("generation_name", generations("sm80", "sm86", "sm89", "sm90", "sm100", "sm120"))
def test_each_gemm_mainloop_compiles_at_its_picks_within_each_gpus_resources(generation_name, part):
    """Every distinct GEMM Mosaic GPU kernel a generation runs at a real projection (`_gate_cases`), at the
    tile its launch derives from the generation's facts and at each tile the derivation refused for its
    registers, compiled by Mosaic for a compile-only device of the generation on this CPU through ptxas, in
    `GATE_PARTS` parts: within the block's shared memory, spilling nothing, and its register model deciding
    as ptxas does (`kernel_compile.mosaic_failures`)."""
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    cases = _gate_cases(generation_name)[part::GATE_PARTS]
    found = compile_mosaic_kernels(cases, generation_name)
    failed = mosaic_failures(found)
    assert len(found) == len(cases) and not failed, "\n".join(failed)


FAMILY_PARTS = 3
"""The parts the family contract compiles in apart, each within a CI group's time (sm90's whole, 294 s,
train 270's CI)."""


@pytest.mark.parametrize("part", range(FAMILY_PARTS))
@pytest.mark.parametrize("generation_name", generations("sm80", "sm90"))
def test_each_family_holds_its_contract_on_the_pinned_jax(generation_name, part):
    """The whole contract a Mosaic GPU Family bundle offers its templates, compiled through ptxas for a
    compile-only GPU of the generation on this CPU with the JAX this checkout pins, before any consumer
    adopts a Family head: every launch form it emits (`_emitted_forms`: plain and transposed right
    operands, 16-bit and fp32 stores, fixed and streamed side vectors and tiles, several outputs, the
    running sum) at one consumer and at the family's most (`_gemm_mainloop_cases(every_consumer_count=
    True)`). Every form and count must be present, and each the compile gate does not hold
    (`test_each_gemm_mainloop_compiles_at_its_picks_within_each_gpus_resources`) lower, within the block,
    spilling nothing, in `FAMILY_PARTS` parts. No version guard skips it: a JAX that breaks the contract
    fails here."""
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    family = bundles.of(generation_name, jnp.bfloat16)
    assert family is not None
    cases = _gemm_mainloop_cases(generation_name, next(iter(GEMM_SHAPES)), every_consumer_count=True)
    counts = sorted({1, family.consumers})
    expected = {f"{form} x{count}" for form in _emitted_forms(family) for count in counts}
    got = {case.name.split(" Tiles(")[0] for case in cases}
    assert got == expected, (sorted(expected - got), sorted(got - expected))
    # A kernel the compile gate already holds (`_gate_cases`) is compiled there, not again.
    held = {_kernel_of(case.name) for case in _gate_cases(generation_name)}
    cases = [case for case in cases if _kernel_of(case.name) not in held][part::FAMILY_PARTS]
    found = compile_mosaic_kernels(cases, generation_name)
    failed = mosaic_failures(found)
    assert len(found) == len(cases) and not failed, "\n".join(failed)


@COMPILES_FOR_TPU
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
    thunks, programs = compiled
    assert "shared" in triton_evidence(programs, thunks)
    for missing, said in ((" probe grid", "probe"), (" probe_headroom grid", "probe_headroom")):
        lines = "\n".join(thunks).splitlines()
        dropped = next(index for index, line in enumerate(lines) if missing in line)
        without = ["\n".join(lines[:dropped] + lines[dropped + 1:])]
        assert triton_evidence(programs, without) == {
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


def test_the_resource_gate_refuses_a_spilling_mosaic_kernel():
    """A record the gate reads (`mosaic_failures`): a kernel past its shared memory, its tensor memory or
    spilling fails; within all three, none."""
    from kernel_compile import mosaic_failures, mosaic_kernels

    within = {"kernel": "k", "unread": [], "shared": 1024, "tmem": 256, "registers": 168, "spilled": 0}
    record = {"name": "case", "limit": 2048, "tmem_limit": 512, "kernels": [within]}
    assert not mosaic_failures([record])
    for past in ({"shared": 4096}, {"tmem": 513}, {"spilled": 40}):
        assert mosaic_failures([{**record, "kernels": [{**within, **past}]}])
    # A kernel whose reports state no resources, spilling nothing, is no evidence: it fails, as a record
    # with no kernel does (row 1877: a contract gate read only its shared count and positive spills).
    unstated = {**within, "unread": ["shared bytes", "tensor memory columns"], "shared": None, "tmem": None}
    assert mosaic_failures([{**record, "kernels": [unstated]}])
    assert mosaic_failures([{**record, "kernels": []}])
    # The program's own refusal of a block past shared memory fails it whatever its Mosaic kernels say.
    assert mosaic_failures([{**record, "shared": 4096}])
    # Either report without the other, or a report without a field, is read as missing, not as within.
    complete = ("smem_scratch_bytes=1024 tmem_scratch_cols=0, tmem_collective_scratch_cols=0",
                "Used 168 registers\n0 bytes spill stores, 0 bytes spill loads")
    with tempfile.TemporaryDirectory() as directory:
        Path(directory, "a.resources.txt").write_text(complete[0])
        Path(directory, "a.ptxas").write_text(complete[1])
        spilling = "Used 255 registers\n40 bytes spill stores, 40 bytes spill loads"
        Path(directory, "b.ptxas").write_text(spilling)
        Path(directory, "c.resources.txt").write_text("smem_scratch_bytes=1024")
        a, b, c = mosaic_kernels(directory)
        assert not a["unread"]
        assert set(b["unread"]) == {"shared bytes", "tensor memory columns"}
        assert set(c["unread"]) == {"tensor memory columns", "registers", "spills"}
        assert len(mosaic_failures([{**record, "kernels": [a, b, c]}])) == 2


@COMPILES_FOR_TPU
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


def test_a_refusal_holds_in_every_process_sharing_its_records(tmp_path):
    """A process's refusal (`report.refuse`, as a compile's spill records it) kept beside the compilation
    cache (`report.records`) holds in the next process reading them, which so derives the stepped-down
    config without compiling the spilling one."""
    import subprocess
    import sys

    tile = "mosaic_gpu.Tiles(128, 256, 64, 4, 8, warpgroups=2)"
    preamble = ("from dew.nn.kernels.templates import report\n"
                "from dew.nn.kernels.templates.gemm import mosaic_gpu\n")
    environment = {**os.environ, "DEW_REFUSAL_RECORDS": str(tmp_path / "refusals.jsonl")}
    seeded = preamble + f"report.refuse([('a launch', {tile}, {{'dew_gmm'}})])"
    subprocess.run([sys.executable, "-c", seeded], check=True, env=environment)
    expected = f"(('a launch', {tile}, frozenset({{'dew_gmm'}})),)"
    reading = preamble + f"print(report.refused_launches() == {expected})"
    read = subprocess.run([sys.executable, "-c", reading], check=True, env=environment, capture_output=True,
                          text=True)
    assert read.stdout.strip() == "True"
