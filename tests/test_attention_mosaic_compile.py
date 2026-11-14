"""The attention template's Mosaic GPU mainloop compiled at the tiles its models decide
(dew.nn.kernels.templates.attention_mosaic): each launch, forward, backward and decode step, compiled by XLA's
CUDA backend for a compile-only A100 and H100 on the CPU (tests/kernel_compile.py), within its block and
spill-free; and an A100's launches counting the shared memory their lowering asks for. The mainloop's values
are tests/test_attention_mosaic.py's."""


import dataclasses
from unittest import mock

import jax
import jax.numpy as jnp
import pytest
from generation_support import COMPILES_ON_CPU, INTERPRETS
from kernel_compile import gate_generations
from test_attention_mosaic import BACKWARD_HOOKS, SHAPES, _check_backward, _residuals
from test_attention_triton import SCORE_HOOKS, score_hooks

from dew.nn.kernels import generation, tuning
from dew.nn.kernels.templates import (
    attention as template,
    attention_mosaic,
    attention_mosaic_program,
    bundles,
)

pytestmark = pytest.mark.kernel


GATED = gate_generations(scoped=True, tpu=True,
                         needs=lambda family, architecture: attention_mosaic_program.qualified(family))
"""The GPUs whose launches the resource gates compile: one of each architecture whose bundle the mainloop
runs on (`kernel_compile.gate_generations`, its launches' flags, the bundle qualified for it,
`attention_mosaic_program.qualified`: a family whose consumers issue their products)."""

BACKWARD_GATED = gate_generations(tpu=True, needs=lambda family, architecture: (
    attention_mosaic_program.runs_backward(family)))
"""The GPUs whose backward launches the gate compiles: one of each architecture with a bundle the backward
runs on (its launches scope no product scratch: `runs_backward`), a B200's and consumer Blackwell's the
mma.sync rings their consumers feed, the one launch's alone."""

FP32_GATED = gate_generations(scoped=True, tpu=True, needs=lambda family, architecture: (
    attention_mosaic_program.qualified(family) and attention_mosaic_program.loads_parts(family)))
"""The GPUs whose fp32 forwards the gate compiles: one of each architecture whose bundle the mainloop runs on
and loads a forward's fp32 parts (`loads_parts`); every other's fp32 call takes another mainloop."""


def _takes(generation_name: str, accurate: bool, hooks: str) -> bool:
    """Whether the backward takes a gate call of `accurate`'s form under `hooks` on `generation_name`: every
    form where its bundle's dQ launch has a producer, the one launch's forms (`fuses`) where it has none."""
    family = attention_mosaic_program.backward_bundle(generation_name, jnp.bfloat16)
    parametrized = hooks != "none" and bool(score_hooks(SCORE_HOOKS[hooks], 8, 8, 64, rows=1, heads=4)[1])
    return not attention_mosaic_program.split_queries(family) or attention_mosaic_program.fuses(accurate,
                                                                                                 parametrized)


COMPILED_HOOKS = [hooks for hooks in BACKWARD_HOOKS if hooks not in ("none", "resumed")]
"""The score hooks whose launches the gates compile: every one but a resumed partial, an initial state as
sinks are (`template.sinks`: a numerator of zeros), which a launch reads as arrays whatever their values, so
its every kernel is the sinks' (the same module hash at each gate cell, 2026-11-12)."""

BACKWARD_GATE = [(generation_name, width, accurate, "none") for generation_name in BACKWARD_GATED
                 for width in (64, 128, 256) for accurate in (False, True)]
BACKWARD_GATE += [(generation_name, width, accurate, hooks) for generation_name in BACKWARD_GATED
                  for width in (64, 128) for accurate in (False, True) for hooks in COMPILED_HOOKS]
BACKWARD_GATE += [("sm90", 256, False, "softcap")]
BACKWARD_GATE = [case for case in BACKWARD_GATE if _takes(case[0], case[2], case[3])]
SPILLING = {("sm86", 64, False, "temperature"), ("sm89", 64, False, "temperature")}
"""The gate's cases that spill, each an open row's: an sm86's and an sm89's dK/dV at width 64 under a
temperature, 64 bytes, the operand pipeline's per-thread shared offsets formed at the launch's entry and
held through its loop beside the statistics' reads (row 5483, the owner's exception). Each fails here until
it compiles spill-free, and is deleted with the fix."""
SPILLS = pytest.mark.xfail(strict=True, reason="row 5483: 64 bytes spilled")
BACKWARD_GATE = [pytest.param(*case, marks=SPILLS) if case in SPILLING else case for case in BACKWARD_GATE]
"""The backward resource gate's cases, each a form its GPU's bundles take (`_takes`; a form of two
launches refused by its typed fact where the bundle's dQ launch has no producer): each GPU whose family the
backward runs on (`BACKWARD_GATED`), a causal
call's head widths, each form, hookless; each score hook at the widths a hooked call's derivation takes
whole, in each form (row 2281: its live set beside the step's; FS-A100CHECK1d: the accurate form under a
row-indexed hook failed an A100's layout inference, a case no fast-form gate lowered); and Gemma 2's, a
softcap 256 wide on an H100, whose one launch's consumers share a key block within its block where its hook
reserves no reduction scratch (row 3704, `hook_reduces`)."""


def _gated_hooks(hooks: str, queries: int, width: int):
    """The gate's score hook `SCORE_HOOKS[hooks]` over `queries` queries and keys, four query heads, its
    parameters' first draw (`score_hooks`), as `template.Hooks`; None for none."""
    if hooks == "none":
        return None
    score_mod, kinds, draw_hooks = score_hooks(SCORE_HOOKS[hooks], queries, queries, width, rows=1, heads=4)
    params, initial = draw_hooks(0)
    return template.Hooks(score_mod, tuple(template.Param(value, kind) for value, kind in zip(params, kinds,
                                                                                               strict=True)),
                          initial)


def _gated_backward(generation_name: str, width: int, accurate: bool, hooks: str = "none",
                    dtype=jnp.bfloat16):
    """The resource gate's backward op, bound arguments and derived config: a causal call of 512 queries,
    four query heads over two key heads `width` wide, in `dtype` (bf16 by default), in the form `accurate`
    names, under the score hook `hooks` names (`_gated_hooks`), for `generation_name`."""
    queries = 512
    spans = template.spans(queries, queries, 1, causal=True)
    query = jnp.zeros((1, queries, 4, width), dtype)
    key = jnp.zeros((1, queries, 2, width), dtype)
    hooked = _gated_hooks(hooks, queries, width)
    with generation.assumed(generation_name):
        op, args = attention_mosaic._backward_call(query, key, key, spans, hooked, query,
                                                   _residuals(query.shape), accurate=accurate, scale=None,
                                                   natural=attention_mosaic.NATURAL)
        ba = op.bind(*args, interpret=False)
        return op, ba, op.derived(ba, generation.gpu_target())


def _least_beyond(op, ba, generation_name: str):
    """The tile `op`'s derivation takes for `ba` on `generation_name` grown by the least factor that holds one
    (its block's shared memory and its consumers' registers each that many times their own, bisected up to
    four), or None where none at four holds one: the least the model says its launches stage and hold."""
    with generation.assumed(generation_name):
        target = generation.gpu_target()
        own = generation.shared_budget(target)
    budget = type(op).register_budget

    def derived(factor: float):
        grown = dataclasses.replace(target, shared_limit=int(own * factor))
        with (generation.assumed(generation_name),
              mock.patch.object(type(op), "register_budget",
                                lambda self, config: int(budget(self, config) * factor))):
            return op.derived(ba, grown)

    low, high = 1.0, 4.0
    if derived(high) is None:
        return None
    while high - low > 1 / 256:
        middle = (low + high) / 2
        low, high = (low, middle) if derived(middle) is not None else (middle, high)
    return derived(high)


FORWARD_GATE = [(generation_name, width, training, "none") for generation_name in GATED
                for width in (64, 128, 256)
                for training in (template.Kept.NONE, template.Kept.OUTPUT, template.Kept.SPLIT)]
FORWARD_GATE += [(generation_name, width, training, hooks) for generation_name in GATED for width in (64, 128)
                 for training in (template.Kept.NONE, template.Kept.OUTPUT, template.Kept.SPLIT)
                 for hooks in COMPILED_HOOKS]
FORWARD_GATE += [(generation_name, 256, template.Kept.NONE, "softcap") for generation_name in GATED]
FORWARD_GATE = [(*case, jnp.bfloat16) for case in FORWARD_GATE]
FORWARD_GATE += [(generation_name, width, training, "none", jnp.float16) for generation_name in GATED
                 for width in (64, 128) for training in (template.Kept.NONE, template.Kept.SPLIT)]
FORWARD_GATE += [("sm100", width, training, "none", jnp.bfloat16) for width in (64, 128)
                 for training in (template.Kept.NONE, template.Kept.SPLIT)]
"""The forward resource gate's cases: each GPU whose family the mainloop qualifies (`GATED`), a causal call's
head widths, inference and both training residual forms (the 16-bit fast path's output and the anchor's
split sums), hookless, each score hook at the widths a hooked call's derivation takes (row 2281: its live set
beside the walk's), a softcap's inference forward 256 wide (Gemma's), all in bf16, and fp16's inference and
training forwards (row 3514: an A100's fp16 training forward at width 64 spilled 24 bytes); and a B200's
bf16 forwards on tcgen05's chunks (products into scoped tensor memory, read back into registers, the
weights staged for the values' product: the Blackwell fold's H1)."""


def _template_step(query, key, value):
    """A training step's attention on the public call's template route, causal: its output's sum's gradients
    in the query, key and value."""
    from dew.nn.attention import scaled_dot_product_attention

    def loss(query, key, value):
        out = scaled_dot_product_attention(query, key, value, implementation="template", causal=True)
        return out.astype(jnp.float32).sum()
    return jax.grad(loss, argnums=(0, 1, 2))(query, key, value)


@COMPILES_ON_CPU
def test_the_public_template_route_runs_each_shards_rows_and_heads_with_no_collective():
    """The public call on the template (`scaled_dot_product_attention`), causal, forward and backward, its
    batch split over a compile-only H100 mesh's data axis and its heads over its tensor axis, compiled for
    them: the forward launches, reading its shard's rows and heads (four query heads of eight, two key heads
    of four, two rows of four), and the program moves and sums nothing, its backward's launches whichever
    the derivation takes (`kernel_compile.Expected`, its route evidence): a head's softmax and its gradients
    are its own."""
    import fla_gdn_support as support
    from jax.sharding import PartitionSpec
    from kernel_compile import Expected, compile_mosaic_kernels, route_failures

    rows, tokens, heads, kv_heads, width = 4, 256, 8, 4, 64

    def call():
        def shaped(kv):
            return jax.ShapeDtypeStruct((rows, tokens, kv, width), jnp.bfloat16)
        return _template_step, (shaped(heads), shaped(kv_heads), shaped(kv_heads))

    spec = PartitionSpec("data", None, "tensor", None)
    expected = Expected(axes=(("data", 2), ("tensor", 2)), specs=(spec, spec, spec),
                        launches=frozenset({"attention_forward"}),
                        shards={"attention_forward": (rows // 2, tokens, heads // 2, width)})
    cases = [support.KernelCase("template route", call, expected=expected)]
    assert route_failures(cases, compile_mosaic_kernels(cases, "sm90")) == []


@COMPILES_ON_CPU
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16], ids=["bfloat16", "float16"])
@pytest.mark.parametrize("kept", [template.Kept.NONE, template.Kept.OUTPUT], ids=["inference", "training"])
def test_an_a100_forward_counts_the_shared_memory_its_lowering_asks_for(dtype, kept):
    """An A100's D256 forward (the receipt's: 1024 causal queries, 8 heads over 4) at 64 by 64 two stages
    deep on mma.sync, inference and the 16-bit training path's: the model counts exactly the shared bytes
    its lowering asks for, the reductions' scratch and both barriers included, past the A100's limit, so no
    derivation that reads the model takes it."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels

    queries, width = 1024, 256
    spans = template.spans(queries, queries, 1, causal=True)
    with generation.assumed("sm80"):
        query = jnp.zeros((1, queries, 8, width), dtype)
        key = jnp.zeros((1, queries, 4, width), dtype)
        op, args = attention_mosaic._forward_call(query, key, key, spans, None, scale=None,
                                                  natural=attention_mosaic.NATURAL)
        ba = op.bind(*args, residuals=kept, interpret=False)
        config = attention_mosaic.Tiles(64, 64, 2, 1, family="mma.sync", narrow=True, relaunch_k=64,
                                        relaunch_consumers=1, relaunch_columns=128)
        target = generation.gpu_target()
        shared = op.staged_bytes(ba, config, target)
        (found,) = compile_mosaic_kernels([KernelCase("forward", lambda: op.lowering(ba, config))], "sm80")
        assert found["shared"] == shared
        assert shared > target.shared_limit


@COMPILES_ON_CPU
def test_an_a100_fused_backward_counts_the_shared_memory_its_lowering_asks_for():
    """An A100's fused backward at 64 by 64 two stages deep on mma.sync, whose adds read no stage
    (`Transport.adds_through_shared`), so its dK/dV output stage is scoped after its walks: the model counts
    exactly the shared bytes its binary takes, the stage within the walks' slots, not beside them, and no
    reduction scratch (the launch reduces across no warps)."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels

    op, ba, _ = _gated_backward("sm80", 128, accurate=False)
    config = attention_mosaic.BackwardTiles(64, 64, 2, 1, fused=True, family="mma.sync")
    with generation.assumed("sm80"):
        shared = op.staged_bytes(ba, config, generation.gpu_target())
        (found,) = compile_mosaic_kernels([KernelCase("backward", lambda: op.lowering(ba, config))], "sm80")
    (binary,) = found["binaries"]
    assert binary["shared"] == shared


@COMPILES_ON_CPU
@pytest.mark.parametrize(("accurate", "config"), [
    (False, attention_mosaic.BackwardTiles(32, 32, 2, 1, family="mma.sync", staged_logits=True)),
    (True, attention_mosaic.BackwardTiles(32, 32, 2, 1, family="mma.sync", columns=2))],
    ids=["default", "accurate"])
def test_an_a100_backward_256_wide_compiles_on_32_row_tiles_within_its_block(accurate, config):
    """An A100's backward of heads 256 wide (Gemma's) on 32 x 32 tiles, two launches on mma.sync arranged by
    their 32-row shares (`backward_family`), each register operand crossing its warps relayed through its
    consumer's tile (`_relay_tiles`): the fast form's logits' queries staged a step, the accurate form's sums
    in two column halves, within the A100's block as its model sizes it, spilling nothing, lowered and
    compiled for an sm80 on this CPU."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    op, ba, _ = _gated_backward("sm80", 256, accurate)
    with generation.assumed("sm80"):
        target = generation.gpu_target()
        assert op.staged_bytes(ba, config, target) <= target.shared_limit
        found = compile_mosaic_kernels([KernelCase("backward", lambda: op.lowering(ba, config))], "sm80")
    failed = mosaic_failures(found)
    assert not failed, "\n".join(failed)


def staged_widths(generation_name: str, dtype) -> tuple[int, ...]:
    """The head widths the mainloop stages for operands of `dtype` on a GPU of `generation_name`, every
    bundle of its taking them (`bundles.layouts`, `attention_mosaic_program.refused`): each power of two a
    call's width rounds up to, in whole swizzled rows (`attention_mosaic_program._width`), up to the widest
    a bundle copies; the widths the route takes and stages, each a distinct launch."""
    from dew.nn.kernels.templates.attention_mosaic_program import _width, refused

    with generation.assumed(generation_name):
        families = bundles.layouts(generation_name, dtype, 1, scoped=True)
        return tuple(sorted({_width(2 ** power, family, dtype) for family in families for power in range(3, 9)
                             if refused(dtype, 2 ** power, 2 ** power, None, backward=False, family=family)
                             is None}))


WORK_LISTS = (("decode", 1, 32, 8), ("prefill", 100, 4, 2))
"""The work list's calls the gate lowers: a decode step's one position over 32 query heads of 8 key heads,
and an inference call of 100 causal positions over 4 of 2, which the template route serves as the work list
where its packed rows stream (`attention_decode.serves`; IG-FINAL3-h100's 32-wide call, no case of a
decode-only gate)."""


def _served(generation_name: str, positions: int, heads: int, kv_heads: int, width: int, stored: str) -> bool:
    """Whether the template route runs a call of `positions` causal positions over `heads` query heads of
    `kv_heads`, `width` wide, its cache `stored`, as the work list on a GPU of `generation_name`: the Mosaic
    forward takes it (`attention_mosaic.forward_declined`), the work list serves it
    (`attention_decode.serves`), the route's own declarations (`attention_route`), whose walks derive a plan
    wherever it serves a 16-bit or fp32 cache; an int8 cache's codes and scales served where its walks
    derive the plan they launch, as its selector asks (`attention_decode.planned`,
    `dew.nn.kv_cache._template_tiles`)."""
    from dew.nn.kernels.templates import attention_decode

    rows, keys = 2, 4096 if positions == 1 else positions
    dtype = jnp.float32 if stored == "float32" else jnp.bfloat16
    query = jnp.zeros((rows, positions, heads, width), dtype)
    key = jnp.zeros((rows, keys, kv_heads, width), dtype)
    cache = jnp.zeros(key.shape, jnp.dtype(stored))
    scale = jnp.zeros((rows, kv_heads, keys), jnp.float32)
    scales = (scale, scale) if stored == "int8" else None
    spans = template.spans(positions, keys, rows, causal=positions > 1)
    with generation.assumed(generation_name):
        return (attention_mosaic.forward_declined(query, key, key, spans) is None
                and attention_decode.serves(query, key, None, natural=False, target=generation.gpu_target())
                and (scales is None or attention_decode.planned(query, cache, cache, spans, template.Hooks(),
                                                                scales=scales)[0] is not None))


WORK_LIST_GPUS = tuple(name for name in generation.TARGETS if staged_widths(name, jnp.bfloat16))
"""Every GPU generation whose bundles stage a head for the Mosaic mainloop (`generation.TARGETS`,
`staged_widths`), each the work list's gate compiles for."""


DECODE_GATE = [(generation_name, width, stored, call) for generation_name in WORK_LIST_GPUS
               for width in staged_widths(generation_name, jnp.bfloat16)
               for stored in ("bfloat16", "int8", "float32")
               for call, positions, heads, kv_heads in WORK_LISTS
               if (stored == "bfloat16" or call == "decode")
               and _served(generation_name, positions, heads, kv_heads, width, stored)]
"""The work list gate's cases: each GPU the route serves the work list on (`WORK_LIST_GPUS`), each width its
bundles stage (`staged_widths`, the declarations the route's legality reads: a 32-wide call's mma.sync launch
was no case of the gate's 64 and 128, and its magnitudes' transposed load failed on an H100, IG-FINAL3), each
cache of a decode step (bf16, int8 codes, an fp32 query's parts) and a bf16 inference call, wherever the route
serves the call as the work list (`_served`)."""


@COMPILES_ON_CPU
@pytest.mark.parametrize(("generation_name", "width", "stored", "call"), DECODE_GATE)
def test_a_decode_step_compiles_spill_free_at_its_plans_tiles(generation_name, width, stored, call):
    """A decode step (`attention_decode.attend`: its plain launch at the plan's streaming key tile and
    slices, `MosaicForward.streaming`, and its masked rewalk's launches), 32 query heads over 8 key heads
    `width` wide, two rows of 4096 keys, its cache bf16, int8 codes with their scales (each slot converted
    in registers, `_Codes`) or float32 under a float32 query (its parts, `MosaicForward.parts`), lowered
    and compiled by ptxas for `generation_name` on this CPU (tests/kernel_compile.py): every launch within
    its shared memory and spilling nothing (`mosaic_failures`), where the forward gate's prefill calls
    derive their own tiles."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    from dew.nn.kernels.templates import attention_decode

    positions, heads, kv_heads = next(shape for name, *shape in WORK_LISTS if name == call)
    rows, keys = 2, 4096 if positions == 1 else positions
    query = jax.ShapeDtypeStruct((rows, positions, heads, width),
                                 jnp.float32 if stored == "float32" else jnp.bfloat16)
    key = jax.ShapeDtypeStruct((rows, keys, kv_heads, width), jnp.dtype(stored))
    scale = jax.ShapeDtypeStruct((rows, kv_heads, keys), jnp.float32)
    quantized = stored == "int8"
    spans = (template.spans(1, keys, rows, lengths=jnp.asarray([keys, keys - 77], jnp.int32))
             if positions == 1 else template.spans(positions, keys, rows, causal=True))

    def decode(query, key, value, *scales):
        return attention_decode.attend(query, key, value, spans, scales=scales or None)

    operands = (query, key, key, *((scale, scale) if quantized else ()))
    with generation.assumed(generation_name):
        found = compile_mosaic_kernels([KernelCase("decode", lambda: (decode, operands))], generation_name)
    failed = mosaic_failures(found)
    assert not failed, "\n".join(failed)


@COMPILES_ON_CPU
@pytest.mark.parametrize("page", [16, 4, 1])
@pytest.mark.parametrize("generation_name", GATED)
def test_a_paged_decode_step_compiles_spill_free_at_its_plans_tiles(generation_name, page):
    """A decode step over a paged cache of `page`-key pages (`attention_decode.attend` over pools `[KV, P,
    page, D]` and a table), 16 rows of 4096 keys, 32 query heads over 8 key heads 64 wide, its key tile of
    several pages, its rows gathered through the table (`attention_mosaic_forward.paged_reads`): lowered and
    compiled by ptxas for `generation_name` on this CPU, every launch within its shared memory and spilling
    nothing (`mosaic_failures`). An A100's rows its one warpgroup's cp.async copies a row a copy in its
    pipeline's groups (`mosaic_sm80._gathered_walk`; its 16-key pages were each a copy of their own,
    M-DD-10's W64-a100: 'Number of tiles is not a multiple of 4'; its 4- and 1-key pages refused, row
    4772)."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    from dew.nn.kernels.templates import attention_decode

    rows, keys, width = 16, 4096, 64
    query = jax.ShapeDtypeStruct((rows, 1, 32, width), jnp.bfloat16)
    pool = jax.ShapeDtypeStruct((8, rows * keys // page, page, width), jnp.bfloat16)
    table = jax.ShapeDtypeStruct((rows, keys // page), jnp.int32)
    spans = template.spans(1, keys, rows, lengths=jnp.full((rows,), keys - 77, jnp.int32))

    def decode(query, key, value, table):
        return attention_decode.attend(query, key, value, spans, table=table)

    with generation.assumed(generation_name):
        case = KernelCase("paged decode", lambda: (decode, (query, pool, pool, table)))
        found = compile_mosaic_kernels([case], generation_name)
    failed = mosaic_failures(found)
    assert not failed, "\n".join(failed)


@COMPILES_ON_CPU
@pytest.mark.parametrize(("generation_name", "width", "training", "hooks", "dtype"), FORWARD_GATE)
def test_the_forward_compiles_spill_free_as_its_model_decides(generation_name, width, training, hooks,
                                                              dtype):
    """The forward, the training forward where `training`, at its derived tiles of a causal call `width`
    wide under the score hook `hooks` names, in `dtype`, lowered and compiled by ptxas for `generation_name`
    on this CPU (tests/kernel_compile.py): every launch within its shared memory, spilling nothing
    (`mosaic_failures`)."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    queries = 512
    spans = template.spans(queries, queries, 1, causal=True)
    query = jax.ShapeDtypeStruct((1, queries, 4, width), dtype)
    key = jax.ShapeDtypeStruct((1, queries, 2, width), dtype)
    hooked = _gated_hooks(hooks, queries, width)

    def forward(query, key, value):
        return attention_mosaic.forward(query, key, value, spans, hooked, residuals=training)

    with generation.assumed(generation_name):
        zeros = [jnp.zeros(array.shape, array.dtype) for array in (query, key, key)]
        op, args = attention_mosaic._forward_call(*zeros, spans, hooked, scale=None,
                                                  natural=attention_mosaic.NATURAL)
        ba = op.bind(*args, residuals=training, interpret=False)
        config = op.derived(ba, generation.gpu_target())
        assert config is not None, f"the Mosaic forward derives no config {width} wide"
        found = compile_mosaic_kernels([KernelCase("forward", lambda: (forward, (query, key, key)))],
                                       generation_name)
    failed = mosaic_failures(found)
    assert not failed, "\n".join(failed)


@COMPILES_ON_CPU
@pytest.mark.parametrize("width", [64, 128])
@pytest.mark.parametrize("generation_name", FP32_GATED)
def test_an_fp32_forward_compiles_spill_free_as_its_model_decides(generation_name, width):
    """An fp32 call's inference forward (`MosaicForward.parts`: its operands bf16 parts) at its derived tiles
    of a causal call `width` wide, lowered and compiled by ptxas for `generation_name` on this CPU: every
    launch within its shared memory, spilling nothing."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    queries = 512
    spans = template.spans(queries, queries, 1, causal=True)
    query = jax.ShapeDtypeStruct((1, queries, 4, width), jnp.float32)
    key = jax.ShapeDtypeStruct((1, queries, 2, width), jnp.float32)

    def forward(query, key, value):
        return attention_mosaic.forward(query, key, value, spans, None)

    with generation.assumed(generation_name):
        zeros = [jnp.zeros(array.shape, array.dtype) for array in (query, key, key)]
        op, args = attention_mosaic._forward_call(*zeros, spans, None, scale=None,
                                                  natural=attention_mosaic.NATURAL)
        ba = op.bind(*args, residuals=template.Kept.NONE, interpret=False)
        config = op.derived(ba, generation.gpu_target())
        assert config is not None, f"the Mosaic forward derives no fp32 config {width} wide"
        found = compile_mosaic_kernels([KernelCase("forward", lambda: (forward, (query, key, key)))],
                                       generation_name)
    failed = mosaic_failures(found)
    assert not failed, "\n".join(failed)


@INTERPRETS
@pytest.mark.parametrize("call", ["full", "causal"])
def test_a_deterministic_backward_holds_check_1_interpreted(call, with_deterministic_ops):
    """The backward under --xla_gpu_deterministic_ops on a full or causal call, at the tiles it derives
    there: no launch's programs add across programs (row 4732: none waits on another), a key head's query
    heads' dK and dV summed in their order; its gradients the reference's (check 1), under the
    interpreter."""
    _check_backward(call, SHAPES["interpreted"], form=None, interpret=True)


@COMPILES_ON_CPU
@pytest.mark.parametrize("width", [64, 128])
def test_a_deterministic_backward_compiles_spill_free_as_its_model_decides(width, with_deterministic_ops):
    """The backward under --xla_gpu_deterministic_ops at its derived tiles of a causal call `width` wide,
    lowered and compiled by ptxas for an H100 on this CPU: within its shared memory, spilling nothing."""
    test_the_backward_compiles_spill_free_as_its_model_decides(generation_name="sm90", width=width,
                                                               accurate=False, hooks="none")


@COMPILES_ON_CPU
@pytest.mark.parametrize("deterministic", [False, True], ids=["one-launch", "deterministic"])
@pytest.mark.parametrize("tokens", [16, 32])
def test_a_short_training_calls_backward_compiles_spill_free_for_an_a100(tokens, deterministic, request):
    """A bf16 training step's template attention over fewer tokens than a role's 64 rows, four query heads
    over two 32 wide, its backward one launch or, under --xla_gpu_deterministic_ops, two, lowered and
    compiled by ptxas for an A100 on this CPU, within its shared memory and spilling nothing: a backward's
    tiles hold a role's rows of keys however few the call's (`attention_mosaic._role_rows`), where a
    forward's few-query tile (`query_rows`) under 64 query rows lowered in neither launch ("Cannot cast to
    layout" of 16 key rows to a consumer's 64; at 32 the one launch's "cannot relayout"; row 5326)."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    if deterministic:
        request.getfixturevalue("with_deterministic_ops")
    query = jax.ShapeDtypeStruct((8, tokens, 4, 32), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((8, tokens, 2, 32), jnp.bfloat16)
    case = KernelCase("short step", lambda: (_template_step, (query, key, key)), {})
    found = compile_mosaic_kernels([case], "sm80")
    assert not mosaic_failures(found), "\n".join(mosaic_failures(found))


@COMPILES_ON_CPU
@pytest.mark.parametrize(("generation_name", "width", "accurate", "hooks"), BACKWARD_GATE)
def test_the_backward_compiles_spill_free_as_its_model_decides(generation_name, width, accurate, hooks):
    """The backward at its derived tiles of a causal call `width` wide in the form `accurate` names under the
    score hook `hooks` names, lowered and compiled by ptxas for `generation_name` on this CPU
    (tests/kernel_compile.py): every launch within its shared memory, spilling nothing, and its register
    model's decision ptxas's (`mosaic_failures`, each launch's model `resources.register_model` at its own
    layout and consumers, `MosaicBackward._launch_registers`)."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    queries = 512
    spans = template.spans(queries, queries, 1, causal=True)
    query = jax.ShapeDtypeStruct((1, queries, 4, width), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((1, queries, 2, width), jnp.bfloat16)
    residuals = _residuals(query.shape)

    hooked = _gated_hooks(hooks, queries, width)

    def backward(query, key, value, dout, residuals):
        grads = attention_mosaic.backward(query, key, value, spans, hooked, dout, residuals=residuals,
                                          accurate=accurate)
        return grads.query, grads.key, grads.value

    op, ba, config = _gated_backward(generation_name, width, accurate, hooks)
    if config is None and accurate and hooks != "none":
        pytest.skip(f"the accurate form under {hooks} derives no tile {width} wide on {generation_name}: "
                    "its registers hold none, and the Pallas-Triton mainloop takes the call")
    if config is None:
        # No tile its block and registers hold: the least the model takes past them (`_least_beyond`) fails
        # the compile for them too, past the block or spilling, the refusal true (an sm86's 64 x 64 dQ launch
        # 101,392 bytes at width 128 under a score hook, its dK/dV 131,088, of 101,376; the Pallas-Triton
        # backward takes the call).
        least = _least_beyond(op, ba, generation_name)
        assert least is not None, f"the Mosaic backward derives no config {width} wide on {generation_name}"
        with generation.assumed(generation_name):
            found = compile_mosaic_kernels([KernelCase("backward", lambda: op.lowering(ba, least))],
                                           generation_name)
        # Past the block (its lowering's or XLA's asked bytes) or spilling, each record: an error of another
        # kind is no proof of either.
        assert found and all(record.get("shared", 0) > record["limit"]
                             or any(kernel.get("spilled") for kernel in record.get("kernels", ()))
                             for record in found), found
        return
    with generation.assumed(generation_name):
        shapes = jax.tree.map(lambda array: jax.ShapeDtypeStruct(array.shape, array.dtype), residuals)
        found = compile_mosaic_kernels(
            [KernelCase("backward", lambda: (backward, (query, key, key, query, shapes)))], generation_name)
    failed = mosaic_failures(found)
    assert not failed, "\n".join(failed)


@COMPILES_ON_CPU
@pytest.mark.parametrize("width", [64, 128])
def test_an_fp16_backward_compiles_spill_free_as_its_model_decides(width):
    """An fp16 call's backward at its derived tiles of a causal call `width` wide (the one launch's two parts
    at running scales, `attention_mosaic_fused._sixteen_step`, where its registers hold it), lowered and
    compiled by ptxas for an H100 on this CPU: within its shared memory, spilling nothing, its register
    model's decision ptxas's."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    op, ba, config = _gated_backward("sm90", width, accurate=False, dtype=jnp.float16)
    with generation.assumed("sm90"):
        found = compile_mosaic_kernels([KernelCase("backward", lambda: op.lowering(ba, config))], "sm90")
    failed = mosaic_failures(found)
    assert not failed, "\n".join(failed)


@COMPILES_ON_CPU
@pytest.mark.parametrize("kept", [template.Kept.SPLIT, template.Kept.NONE], ids=["training", "inference"])
@pytest.mark.parametrize("rows", [2, 1], ids=["step", "decision"])
def test_each_tcgen05_forward_the_4b_steps_refusals_derive_lowers_for_a_b200(rows, kept):
    """Qwen3.5-4B's attention (16 query heads over 4, 256 wide, 12,288 causal tokens: its step's two rows and
    the planner's one-row decision) on a compile-only B200: each config its derivation offers on tcgen05's
    chunks as a compile refuses the one before (`report.refuse_launch`, a spill's refusal) lowers, so a
    refusal never steps the step onto a forward it cannot build (the B200 4B step, its spill refused onto 64 x
    128 keys at 16 value columns, raised at its values' padded operand)."""
    from dew.nn.kernels.templates import report

    tokens, heads, kv_heads, width = 12288, 16, 4, 256
    spans = template.spans(tokens, tokens, rows, causal=True)
    query = jax.ShapeDtypeStruct((rows, tokens, heads, width), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((rows, tokens, kv_heads, width), jnp.bfloat16)

    def forward(query, key, value):
        return attention_mosaic.forward(query, key, value, spans, None, residuals=kept)

    with generation.assumed("sm100"), report.refusals():
        target = generation.gpu_target()
        op, args = attention_mosaic._forward_call(query, key, key, spans, None, scale=None,
                                                  natural=attention_mosaic.NATURAL)
        ba = op.bind(*args, residuals=kept, interpret=False)
        offered = 0
        while (config := op.derived(ba, target)) is not None and config.family == "tcgen05.chunks":
            with tuning.pinned({attention_mosaic.MosaicForward: config}):
                jax.jit(forward).trace(query, key, key).lower(lowering_platforms=("cuda",))
            report.refuse_launch(op, ba, config, {"attention_forward"})
            offered += 1
        assert offered


@COMPILES_ON_CPU
@pytest.mark.parametrize("kept", [template.Kept.SPLIT, template.Kept.NONE], ids=["training", "inference"])
@pytest.mark.parametrize("rows", [2, 1], ids=["step", "decision"])
def test_the_4b_steps_forward_compiles_spill_free_on_tensor_memory_for_a_b200(rows, kept):
    """Qwen3.5-4B's attention forward (16 query heads over 4, 256 wide, 12,288 causal tokens: the step's two
    rows and the planner's one-row decision) at the config its derivation gives on a compile-only B200: on a
    tensor-memory family, every launch compiled by ptxas spilling nothing (`mosaic_failures`). A B200's 4B
    step stepped through 19 spilling chunks forwards for 42 minutes and landed on the Ampere family's."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    tokens, heads, kv_heads, width = 12288, 16, 4, 256
    spans = template.spans(tokens, tokens, rows, causal=True)
    query = jax.ShapeDtypeStruct((rows, tokens, heads, width), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((rows, tokens, kv_heads, width), jnp.bfloat16)

    def forward(query, key, value):
        return attention_mosaic.forward(query, key, value, spans, None, residuals=kept)

    with generation.assumed("sm100"):
        op, args = attention_mosaic._forward_call(query, key, key, spans, None, scale=None,
                                                  natural=attention_mosaic.NATURAL)
        config = op.derived(op.bind(*args, residuals=kept, interpret=False), generation.gpu_target())
        assert config is not None and config.family.startswith("tcgen05"), config
        case = KernelCase("4B forward", lambda: (forward, (query, key, key)))
        found = compile_mosaic_kernels([case], "sm100")
    failed = mosaic_failures(found)
    assert not failed, "\n".join(failed)


CERTIFIED_FORWARD = """
import json
import jax
import jax.numpy as jnp
from kernel_compile import lowered_for
from dew.nn.kernels.templates import attention as template, attention_mosaic, report
from dew.training.ahead import compile_only_gpus

tokens, heads, kv_heads, width = 640, 4, 2, 128
spans = template.spans(tokens, tokens, 1, causal=True)
query = jax.ShapeDtypeStruct((1, tokens, heads, width), jnp.bfloat16)
key = jax.ShapeDtypeStruct((1, tokens, kv_heads, width), jnp.bfloat16)
with report.certifying(compile_only_gpus("sm90")[0]):
    lowered_for(lambda q, k, v: attention_mosaic.forward(q, k, v, spans, None), (query, key, key), "sm90")
print(json.dumps([[found.op, found.stored, {name: module for name, (module, _) in found.kernels.items()}]
                  for found in report.certifications()]))
"""
"""A start's lowering of an H100's attention forward in a process of its own, each launch certified on a
compile-only H100: each certification's op, whether its verdict was read back, and its kernels' modules."""


@COMPILES_ON_CPU
def test_a_launch_is_certified_once_a_machine(tmp_path):
    """A launch certified by one process (`report.certified`: compiled alone, its verdict stored with the
    kernel picks) is certified in the next on the same machine by its stored verdict, nothing compiled, the
    same kernels by module: a warm start pays nothing to certify."""
    from process_support import fresh, keeping_picks

    def certified(found: list) -> list:
        return sorted((op, sorted(modules.items())) for op, _, modules in found)

    cold = fresh(CERTIFIED_FORWARD, cache_dir=tmp_path, flags=keeping_picks())
    warm = fresh(CERTIFIED_FORWARD, cache_dir=tmp_path, flags=keeping_picks())
    assert cold and not any(stored for _, stored, _ in cold), cold
    assert all(stored for _, stored, _ in warm), warm
    assert certified(warm) == certified(cold)


@COMPILES_ON_CPU
def test_a_start_runs_the_kernels_its_launches_certified():
    """A launch's derived config certified as it traces (`report.certified`: its kernels compiled alone as
    the GPU's compiler assembles them) on a compile-only B200, and then compiled within its program: every
    Mosaic GPU kernel the program runs is one certification compiled, module for module (its kernel hash),
    which Mosaic compiles once a process, so certification adds a start no second assembly."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    tokens, heads, kv_heads, width = 1600, 4, 2, 128
    spans = template.spans(tokens, tokens, 1, causal=True)
    query = jax.ShapeDtypeStruct((1, tokens, heads, width), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((1, tokens, kv_heads, width), jnp.bfloat16)

    def forward(query, key, value):
        return attention_mosaic.forward(query, key, value, spans, None)

    found = compile_mosaic_kernels([KernelCase("a start's forward", lambda: (forward, (query, key, key)))],
                                   "sm100")
    assert mosaic_failures(found) == []
    record, = found
    run = record["modules"]
    assert run and {name: record["certified"].get(name) for name in run} == run
