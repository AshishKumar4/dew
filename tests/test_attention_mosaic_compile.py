"""The attention template's Mosaic GPU mainloop compiled at the tiles its models decide
(dew.nn.kernels.templates.attention_mosaic): each launch, forward, backward and decode step, compiled by XLA's
CUDA backend for a compile-only A100 and H100 on the CPU (tests/kernel_compile.py), within its block and
spill-free; an A100's launches counting the shared memory their lowering asks for; and the launches lowered
for an H100 and a v5e: each stating its outputs' dimensions, a hook's approximated operations exact, every
score hook's lowering. The mainloop's values are tests/test_attention_mosaic.py's."""


import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fla_gdn_support import KernelCase
from generation_support import COMPILES_FOR_TPU, COMPILES_ON_CPU, INTERPRETS, lowers
from jax.sharding import AbstractDevice, AbstractMesh, AxisType, Mesh
from kernel_compile import compile_mosaic_kernels, gate_generations, lowered_for, mosaic_failures
from test_attention_mosaic import (
    BACKWARD_HOOKS,
    CALLS,
    FORWARD_HOOKS,
    SHAPES,
    _backward_call,
    _bound,
    _check_backward,
    _forward_call,
    _kept,
    _pinned,
    _residuals,
)
from test_attention_template import CASES, ROWS, RUN_CASES, arguments, geometry as core_geometry
from test_attention_triton import HOOKS, SCORE_HOOKS, gate_of, hooks_for, score_hooks

from dew.nn.attention import scaled_dot_product_attention
from dew.nn.kernels import generation, tuning
from dew.nn.kernels.templates import (
    attention as template,
    attention_mosaic,
    attention_mosaic_derivation,
    attention_mosaic_program,
    attention_mosaic_registers,
    attention_route,
    bundles,
)

pytestmark = pytest.mark.kernel


GATED = gate_generations(scoped=True, tpu=True,
                         needs=lambda family, architecture: attention_mosaic_program.qualified(family))
"""The GPUs whose launches the resource gates compile: one of each architecture whose bundle the mainloop
runs on (`kernel_compile.gate_generations`, its launches' flags, the bundle qualified for it,
`attention_mosaic_program.qualified`: a family whose consumers issue their products)."""

FAMILY_GPUS = ("sm80", "sm86", "sm90", "sm100", "sm120")
"""One GPU of each family's distinct facts, the hooked and cached gate cells' (row 12438): an A100's mma.sync
at 164 KiB of shared memory, an A10's at 99 KiB, an H100's wgmma, a B200's tcgen05, consumer Blackwell's;
an L4 (sm89) is an A10's family and shared memory for every 16-bit call the gates compile
(`generation.GpuTarget`), so its hookless cells alone are its own."""

BACKWARD_GATED = gate_generations(tpu=True, needs=lambda family, architecture: (
    attention_mosaic_program.runs_backward(family)))
"""The GPUs whose backward launches the gate compiles: one of each architecture with a bundle the backward
runs on (its launches scope no product scratch: `runs_backward`), a B200's and consumer Blackwell's the
mma.sync rings their consumers feed, the one launch's alone."""

FP32_GATED = gate_generations(scoped=True, tpu=True, needs=lambda family, architecture: (
    attention_mosaic_program.qualified(family) and attention_mosaic_program.loads_parts(family)))
"""The GPUs whose fp32 forwards the gate compiles: one of each architecture whose bundle the mainloop runs on
and loads a forward's fp32 parts (`loads_parts`); every other's fp32 call takes another mainloop."""


@functools.cache
def _hook_has_params(hooks: str) -> bool:
    return hooks != "none" and bool(score_hooks(SCORE_HOOKS[hooks], 8, 8, 64, rows=1, heads=4)[1])


def _takes(generation_name: str, accurate: bool, hooks: str) -> bool:
    """Whether the backward takes a gate call of `accurate`'s form under `hooks` on `generation_name`: every
    form where its bundle's dQ launch has a producer, the one launch's forms (`fuses`) where it has none."""
    family = attention_mosaic_program.backward_bundle(generation_name, jnp.bfloat16)
    parametrized = _hook_has_params(hooks)
    return not attention_mosaic_program.split_queries(family) or attention_mosaic_program.fuses(accurate,
                                                                                                 parametrized)


COMPILED_HOOKS = [hooks for hooks in BACKWARD_HOOKS if hooks not in ("none", "resumed")]
"""The score hooks whose launches the gates compile: every one but a resumed partial, an initial state as
sinks are (`template.sinks`: a numerator of zeros), which a launch reads as arrays whatever their values, so
its every kernel is the sinks' (the same module hash at each gate cell, 2026-11-12)."""

BACKWARD_GATE = [(generation_name, width, accurate, "none") for generation_name in BACKWARD_GATED
                 for width in (64, 128, 256) for accurate in (False, True)]
BACKWARD_GATE += [(generation_name, 128, accurate, hooks) for generation_name in BACKWARD_GATED
                  if generation_name in FAMILY_GPUS for accurate in (False, True) for hooks in COMPILED_HOOKS]
BACKWARD_GATE += [("sm90", 256, False, "softcap")]
BACKWARD_GATE = [case for case in BACKWARD_GATE if _takes(case[0], case[2], case[3])]
"""The backward resource gate's cases, each a form its GPU's bundles take (`_takes`; a form of two launches
refused by its typed fact where the bundle's dQ launch has no producer): each GPU whose family the backward
runs on (`BACKWARD_GATED`), a causal call's head widths, each form, hookless; each score hook 128 wide on
each family's GPU (`FAMILY_GPUS`, row 12438: the 64-wide cell held what the 128-wide one did), in each form
(row 2281: its live set beside the step's; FS-A100CHECK1d: the accurate form under a row-indexed hook
failed an A100's layout inference, a case no fast-form gate lowered); and Gemma 2's, a softcap 256 wide on
an H100, whose one launch's consumers share a key block within its block where its hook reserves no
reduction scratch (row 3704, `hook_reduces`)."""


def _gated_hooks(hooks: str, queries: int, width: int):
    """The gate's score hook `SCORE_HOOKS[hooks]` over `queries` queries and keys, four query heads, its
    parameters' first draw (`score_hooks`), as `template.Hooks`; None for none; `'dropout'` for fp16/bf16
    probability dropout at rate 0.1 (row 9526)."""
    if hooks == "none":
        return None
    if hooks == "dropout":
        return template.Hooks(drop=template.Dropout(
            jax.random.key_data(jax.random.key(0)),
            jnp.arange(1, dtype=jnp.int32), jnp.arange(4, dtype=jnp.int32), 0.1))
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
    (its block's shared memory that many times its own, bisected up to four), or None where none at four
    holds one: the least the model says its launches stage."""
    with generation.assumed(generation_name):
        target = generation.gpu_target()
        own = generation.shared_budget(target)

    def derived(factor: float):
        grown = dataclasses.replace(target, shared_limit=int(own * factor))
        with generation.assumed(generation_name):
            return op.derived(ba, grown)

    low, high = 1.0, 4.0
    if derived(high) is None:
        return None
    while high - low > 1 / 256:
        middle = (low + high) / 2
        low, high = (low, middle) if derived(middle) is not None else (middle, high)
    return derived(high)

DROPPING = tuple(generation_name for generation_name in GATED
                 if (family := bundles.of(generation_name, jnp.float16, scoped=True, tpu=True)) is not None
                 and not family.places_on_role)
"""`GATED`'s GPUs whose family's mainloop drops: every one but those whose products a role places, whose walk
takes no dropout (`MosaicForward._placed_refusal`)."""

FORWARD_GATE = [(generation_name, width, training, "none") for generation_name in GATED
                for width in (64, 128, 256)
                for training in (template.Kept.NONE, template.Kept.OUTPUT, template.Kept.SPLIT)]
FORWARD_GATE += [(generation_name, 128, training, hooks) for generation_name in GATED
                 if generation_name in FAMILY_GPUS for training in (template.Kept.NONE, template.Kept.SPLIT)
                 for hooks in COMPILED_HOOKS]
FORWARD_GATE += [(generation_name, 256, template.Kept.NONE, "softcap") for generation_name in GATED]
FORWARD_GATE = [(*case, jnp.bfloat16) for case in FORWARD_GATE]
FORWARD_GATE += [(generation_name, width, training, "none", jnp.float16) for generation_name in GATED
                 for width in (64, 128) for training in (template.Kept.NONE, template.Kept.SPLIT)]
FORWARD_GATE += [(generation_name, width, training, "dropout", jnp.float16) for generation_name in DROPPING
                 for width in (64, 128) for training in (template.Kept.NONE, template.Kept.SPLIT)]
"""The forward resource gate's cases: each GPU whose family the mainloop qualifies (`GATED`), a causal call's
head widths, inference and both training residual forms (the 16-bit fast path's output and the anchor's
split sums), hookless, each score hook 128 wide on each family's GPU (`FAMILY_GPUS`) in inference and the
split sums, the largest live set of training's (row 2281: its live set beside the walk's; row 12438), a
softcap's inference forward 256 wide (Gemma's), all in bf16, and fp16's inference and training forwards
(row 3514: an A100's fp16 training forward at width 64 spilled 24 bytes), hookless and dropping where the
mainloop drops (`DROPPING`: its kept weights' two parts beside the step, `weight_parts`; row 9526: an
sm100's split training forward at width 128 spilled 20 bytes, its warpgroup selector and walk bound held
across the register move, none with each consumer a static branch, its draw unpriced); and a B200's bf16
forwards on tcgen05's chunks (products into scoped tensor memory, read back into registers, the weights
staged for the values' product: the Blackwell fold's H1)."""


def _template_step(query, key, value):
    """A training step's attention on the public call's template route, causal: its output's sum's gradients
    in the query, key and value."""

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
        config = attention_mosaic.Tiles(64, 64, 3, 1, family="mma.sync", narrow=True, relaunch_k=64,
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


def _int8_derives(query, cache, scales, positions: int, keys: int, rows: int,
                  target: generation.GpuTarget) -> bool:
    """Fast model-only derivation check for int8 decode at pytest collection time (rows 11724, 11883,
    11898-11900, 11923-11932): avoids invoking ptxas certification (`attention_decode.planned`) on every CI
    shard during module import; full certified planning runs inside the test itself."""
    from dew.nn.kernels.templates import attention_decode

    def derived(q, c, sc):
        spans = template.spans(positions, keys, rows, causal=positions > 1)
        packed_q, row_spans, initial = attention_decode.packed(q, spans, c.shape[2], None)
        hooks = template.Hooks(initial=initial)

        def walk_derives(walk_name: str) -> tuple[bool, bool]:
            chosen = attention_decode.WALKS[walk_name]
            op, args = attention_mosaic._forward_call(
                packed_q, c, c, row_spans, hooks, scale=None, natural=False, out_dtype=None, masked=True,
                splits=1, subtracted=True, scales=sc, scaled=chosen.scaled, bounded=chosen.bounded)
            ba = op.bind(*args, residuals=template.Kept.NONE, interpret=False)
            cfg = op._tiles(ba, target, priced=False)
            return cfg is not None, (cfg is not None and op.deciding(ba, cfg))

        ok, decides = walk_derives("plain")
        if ok and decides:
            return True
        return all(walk_derives(name)[0] for name in attention_decode.launched(q, c, sc))

    result: list[bool] = []
    jax.eval_shape(lambda q, c, sc: (result.append(derived(q, c, sc)), q)[1], query, cache, scales)
    return result[0]


def _served(generation_name: str, positions: int, heads: int, kv_heads: int, width: int, stored: str) -> bool:
    """Whether the template route runs a call of `positions` causal positions over `heads` query heads of
    `kv_heads`, `width` wide, its cache `stored`, as the work list on a GPU of `generation_name`: the Mosaic
    forward takes it (`attention_mosaic.forward_declined`), the work list serves it
    (`attention_decode.serves`), the route's own declarations (`attention_route`), whose walks derive a plan
    wherever it serves a 16-bit or fp32 cache; an int8 cache's codes and scales served where its walks
    derive a config (`_int8_derives`, model-only at collection; certification runs in the cell itself)."""
    from dew.nn.kernels.templates import attention_decode

    rows, keys = 2, 4096 if positions == 1 else positions
    dtype = jnp.float32 if stored == "float32" else jnp.bfloat16
    query = jax.ShapeDtypeStruct((rows, positions, heads, width), dtype)
    key = jax.ShapeDtypeStruct((rows, keys, kv_heads, width), dtype)
    cache = jax.ShapeDtypeStruct(key.shape, jnp.dtype(stored))
    scale = jax.ShapeDtypeStruct((rows, kv_heads, keys), jnp.float32)
    scales = (scale, scale) if stored == "int8" else None
    with generation.assumed(generation_name):
        target = generation.gpu_target()
        if not attention_decode.serves(query, key, None, natural=False, target=target):
            return False
        declined: list[object] = []
        jax.eval_shape(
            lambda q, k: (declined.append(attention_mosaic.forward_declined(
                q, k, k, template.spans(positions, keys, rows, causal=positions > 1))), q)[1],
            query, key)
        if declined[0] is not None:
            return False
        return scales is None or _int8_derives(query, cache, scales, positions, keys, rows, target)


WORK_LIST_GPUS = tuple(name for name in GATED if staged_widths(name, jnp.bfloat16))
"""One GPU of each architecture whose bundles stage a head for the Mosaic mainloop (`GATED`,
`staged_widths`), each the work list's gate compiles for."""


DECODE_GATE = [(generation_name, width, stored, call) for generation_name in WORK_LIST_GPUS
               if generation_name in FAMILY_GPUS
               for width in staged_widths(generation_name, jnp.bfloat16)
               if generation_name == "sm90" or width == max(staged_widths(generation_name, jnp.bfloat16))
               for stored in ("bfloat16", "int8", "float32")
               for call, positions, heads, kv_heads in WORK_LISTS
               if stored == "bfloat16" or call == "decode"]
"""The work list gate's cells: each family's GPU the route serves the work list on (`WORK_LIST_GPUS`,
`FAMILY_GPUS`), at its widest staged head and on an H100 each width its bundles stage (row 12438;
`staged_widths`, the declarations the route's legality reads: a 32-wide call's mma.sync launch was no case
of the gate's 64 and 128, and its magnitudes' transposed load failed on an H100, IG-FINAL3), each cache of
a decode step (bf16, int8 codes, an fp32 query's parts) and a bf16 inference call; a cell decides as it
runs whether the route serves it as the work list (`_served`), so collecting the gate derives no tiles (its
96 derivations took 59 of the module's 67 s import, every CI shard's)."""



@COMPILES_ON_CPU
@pytest.mark.parametrize(("generation_name", "width", "stored", "call"), DECODE_GATE,
                         ids=[f"{g}-{w}-{s}-{c}" for g, w, s, c in DECODE_GATE])
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
    served = _served(generation_name, positions, heads, kv_heads, width, stored)
    rows, keys = 2, 4096 if positions == 1 else positions
    query = jax.ShapeDtypeStruct((rows, positions, heads, width),
                                 jnp.float32 if stored == "float32" else jnp.bfloat16)
    key = jax.ShapeDtypeStruct((rows, keys, kv_heads, width), jnp.dtype(stored))
    scale = jax.ShapeDtypeStruct((rows, kv_heads, keys), jnp.float32)
    quantized = stored == "int8"
    spans = (template.spans(1, keys, rows, lengths=jnp.asarray([keys, keys - 77], jnp.int32))
             if positions == 1 else template.spans(positions, keys, rows, causal=True))

    def decode(query, key, value, *scales):
        # A 16-bit or fp32 cache the work list does not serve runs as the route runs it (`_served`), the
        # cell compiling that; an int8 cache's refusal is certified below.
        if served or quantized:
            return attention_decode.attend(query, key, value, spans, scales=scales or None)
        return attention_route.attend(query, key, value, spans, natural=False)

    operands = (query, key, key, *((scale, scale) if quantized else ()))
    with generation.assumed(generation_name):
        found = compile_mosaic_kernels([KernelCase(f"decode-{width}-{call}", lambda: (decode, operands))],
                                       generation_name)
    failed = mosaic_failures(found)
    if failed and stored == "int8":
        record = found[0]
        offered = [cert for cert in record.get("certifications", ()) if cert["op"] == "MosaicForward"]
        assert "derives no config" in record.get("error", ""), "\n".join(failed)
        assert not record.get("uncertified"), record
        assert offered and all(any(refused for _, _, refused in cert["kernels"].values())
                               for cert in offered), "\n".join(failed)
    else:
        assert not failed, "\n".join(failed)


@COMPILES_ON_CPU
@pytest.mark.parametrize(("generation_name", "width"), [("sm80", 64), ("sm100", 256)])
def test_a_public_fp32_decode_fallback_reports_every_plain_forward_kernels_resources(generation_name, width):
    """Row 7263: a public fp32 decode whose head width or GPU falls back to Pallas-Triton plain-forward
    kernels (`attention_route.attend`: sm80 64-wide, sm90/sm100 256-wide) reports every Triton kernel's
    shared bytes, registers, spills and TMEM usage through `compile_mosaic_kernels` / `mosaic_failures` with
    zero unread fields."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    from dew.nn.kernels.templates import attention_route

    rows, keys = 2, 4096
    query = jax.ShapeDtypeStruct((rows, 1, 32, width), jnp.float32)
    key = jax.ShapeDtypeStruct((rows, keys, 8, width), jnp.float32)
    spans = template.spans(1, keys, rows, causal=False)

    def decode(query, key, value):
        return attention_route.attend(query, key, value, spans, natural=False)

    with generation.assumed(generation_name):
        found = compile_mosaic_kernels([KernelCase("decode", lambda: (decode, (query, key, key)))],
                                       generation_name)
    kernels = found[0]["kernels"]
    assert kernels and all(k["kind"] == "triton" for k in kernels), kernels
    names = {k["kernel"] for k in kernels}
    assert "attention_forward" in names and "attention_forward_scaled" in names, names
    assert all(not k["unread"] for k in kernels), kernels
    failed = mosaic_failures(found)
    assert not failed, "\n".join(failed)


@COMPILES_ON_CPU
@pytest.mark.parametrize("generation_name", GATED)
@pytest.mark.parametrize("page", [16, 1], ids=["p16", "p1"])
def test_a_paged_decode_step_compiles_spill_free_at_its_plans_tiles(generation_name, page):
    """A decode step over a paged cache of 16- or 1-key pages (`attention_decode.attend` over pools
    `[KV, P, page, D]` and a table), 16 rows of 4096 keys, 32 query heads over 8 key heads 64 wide, its key
    tile of several pages, its rows gathered through the table (`attention_mosaic_forward.paged_reads`):
    lowered and compiled by ptxas for `generation_name` on this CPU, every launch within its shared memory
    and spilling nothing (`mosaic_failures`)."""
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

    cases = [KernelCase(f"paged-decode-p{page}", lambda: (decode, (query, pool, pool, table)))]
    with generation.assumed(generation_name):
        found = compile_mosaic_kernels(cases, generation_name)
    failed = mosaic_failures(found)
    assert not failed, "\n".join(failed)


@COMPILES_ON_CPU
@pytest.mark.parametrize(("generation_name", "width", "training", "hooks", "dtype"), FORWARD_GATE,
                         ids=[f"{g}-{w}-{t.name}-{jnp.dtype(d).name}-{h}"
                              for g, w, t, h, d in FORWARD_GATE])
def test_the_forward_compiles_spill_free_as_its_model_decides(generation_name, width, training, hooks,
                                                              dtype):
    """The forward, the training forward where `training`, at its derived tiles of a causal call `width`
    wide under `hooks`, in `dtype`, lowered and compiled by ptxas for `generation_name` on this CPU
    (tests/kernel_compile.py): every launch within its shared memory, spilling nothing (`mosaic_failures`)."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    queries = 512
    spans = template.spans(queries, queries, 1, causal=True)
    query = jax.ShapeDtypeStruct((1, queries, 4, width), dtype)
    key = jax.ShapeDtypeStruct((1, queries, 2, width), dtype)
    zeros = [jnp.zeros(array.shape, array.dtype) for array in (query, key, key)]
    with generation.assumed(generation_name):
        hooked = _gated_hooks(hooks, queries, width)
        op, args = attention_mosaic._forward_call(*zeros, spans, hooked, scale=None,
                                                  natural=attention_mosaic.NATURAL)
        ba = op.bind(*args, residuals=training, interpret=False)
        config = op.derived(ba, generation.gpu_target())
        assert config is not None, f"the Mosaic forward derives no config {width} wide ({hooks})"

        def forward(query, key, value):
            return attention_mosaic.forward(query, key, value, spans, hooked, residuals=training)

        found = compile_mosaic_kernels([KernelCase(f"forward-{hooks}", lambda: (forward, (query, key, key)))],
                                       generation_name)
    failed = mosaic_failures(found)
    assert not failed, "\n".join(failed)


FP32_GATED_WIDTHS = [(gen, w) for gen in FP32_GATED for w in ((64, 128, 256) if gen == "sm90" else (64, 128))]


@COMPILES_ON_CPU
@pytest.mark.parametrize(("generation_name", "width"), FP32_GATED_WIDTHS,
                         ids=[f"{g}-{w}" for g, w in FP32_GATED_WIDTHS])
def test_an_fp32_forward_compiles_spill_free_as_its_model_decides(generation_name, width):
    """An fp32 call's inference forward (`MosaicForward.parts`: its operands bf16 parts) at its derived tiles
    of a causal call 64, 128 (and 256 on Hopper, row 6508) wide, lowered and compiled by ptxas for
    `generation_name` on this CPU: every launch within its shared memory, spilling nothing."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    queries = 512
    spans = template.spans(queries, queries, 1, causal=True)
    query = jax.ShapeDtypeStruct((1, queries, 4, width), jnp.float32)
    key = jax.ShapeDtypeStruct((1, queries, 2, width), jnp.float32)
    zeros = [jnp.zeros(array.shape, array.dtype) for array in (query, key, key)]
    with generation.assumed(generation_name):
        op, args = attention_mosaic._forward_call(*zeros, spans, None, scale=None,
                                                  natural=attention_mosaic.NATURAL)
        ba = op.bind(*args, residuals=template.Kept.NONE, interpret=False)
        config = op.derived(ba, generation.gpu_target())
        assert config is not None, f"the Mosaic forward derives no fp32 config {width} wide"

        def forward(query, key, value):
            return attention_mosaic.forward(query, key, value, spans, None)

        found = compile_mosaic_kernels([KernelCase(f"fp32-forward-{width}",
                                                   lambda: (forward, (query, key, key)))], generation_name)
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
    _check_backward_case("sm90", width, accurate=False, hooks="none")


@COMPILES_ON_CPU
@pytest.mark.parametrize("deterministic", [False, True], ids=["one-launch", "deterministic"])
@pytest.mark.parametrize("tokens", [16, 32])
def test_a_short_training_calls_backward_compiles_spill_free_for_an_a100(tokens, deterministic, request):
    """A bf16 training step's template attention over fewer tokens than a role's 64 rows, four query heads
    over two 32 wide, its backward one launch or, under --xla_gpu_deterministic_ops, two, lowered and
    compiled by ptxas for an A100 on this CPU, within its shared memory and spilling nothing: a backward's
    tiles hold a role's rows of keys however few the call's (`attention_mosaic_derivation._role_rows`), where
    a forward's few-query tile (`query_rows`) under 64 query rows lowered in neither launch ("Cannot cast to
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


def _check_backward_case(generation_name: str, width: int, accurate: bool, hooks: str) -> None:
    """Compile and check one backward gate cell `(generation_name, width, accurate, hooks)`."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    queries = 512
    spans = template.spans(queries, queries, 1, causal=True)
    query = jax.ShapeDtypeStruct((1, queries, 4, width), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((1, queries, 2, width), jnp.bfloat16)
    residuals = _residuals(query.shape)
    shapes = jax.tree.map(lambda array: jax.ShapeDtypeStruct(array.shape, array.dtype), residuals)

    mandatory = (generation_name in ("sm86", "sm89") and width == 64 and not accurate
                 and hooks == "temperature")
    op, ba, config = _gated_backward(generation_name, width, accurate, hooks)
    if config is None:
        assert not mandatory, (f"mandatory Mosaic backward cell derived no config: "
                               f"{generation_name} {width} {hooks}")
        least = _least_beyond(op, ba, generation_name)
        assert least is not None, (f"the Mosaic backward derives no config {width} wide on "
                                   f"{generation_name} ({hooks})")
        with generation.assumed(generation_name):
            beyond_found = compile_mosaic_kernels(
                [KernelCase(f"beyond-{hooks}", lambda: op.lowering(ba, least))], generation_name)
        assert beyond_found and all(record.get("shared", 0) > record["limit"]
                                    or any(kernel.get("spilled") for kernel in record.get("kernels", ()))
                                    for record in beyond_found), beyond_found
        return
    hooked = _gated_hooks(hooks, queries, width)

    def backward(query, key, value, dout, residuals):
        grads = attention_mosaic.backward(query, key, value, spans, hooked, dout,
                                          residuals=residuals, accurate=accurate)
        return grads.query, grads.key, grads.value

    with generation.assumed(generation_name):
        found = compile_mosaic_kernels(
            [KernelCase(f"backward-{hooks}", lambda: (backward, (query, key, key, query, shapes)))],
            generation_name)

    def kernels(cert: dict) -> str:
        return ", ".join(f"{kernel} {registers} registers {frame} bytes"
                         f"{' serialized' if refused and not frame else ''}"
                         for kernel, (registers, frame, refused) in cert["kernels"].items())

    record = found[0]
    failed = mosaic_failures([record])
    offered = [cert for cert in record.get("certifications", ()) if cert["op"] == type(op).__name__]
    print(f"GATE backward {generation_name} {width} {'accurate' if accurate else 'fast'} {hooks}: "
          + ("claimed" if not failed else "declined " + "; ".join(f"{cert['config']}: {kernels(cert)}"
                                                                   for cert in offered)))
    if not failed:
        return
    if mandatory:
        assert not failed, "\n".join(failed)
    assert record.get("exhausted"), "\n".join(failed)
    assert not record.get("uncertified"), record
    assert offered and all(any(refused for _, _, refused in cert["kernels"].values())
                           for cert in offered), "\n".join(failed)


@COMPILES_ON_CPU
@pytest.mark.parametrize(("generation_name", "width", "accurate", "hooks"), BACKWARD_GATE,
                         ids=[f"{g}-{w}-{'acc' if a else 'fast'}-{h}" for g, w, a, h in BACKWARD_GATE])
def test_the_backward_compiles_spill_free_as_its_model_decides(generation_name, width, accurate, hooks):
    """The backward at its derived tiles of a causal call `width` wide in the form `accurate` names under
    `hooks`, lowered and compiled by ptxas for `generation_name` on this CPU (tests/kernel_compile.py):
    every launch within its shared memory, spilling nothing (`mosaic_failures`)."""
    _check_backward_case(generation_name, width, accurate, hooks)


@COMPILES_ON_CPU
@pytest.mark.parametrize("width", [64, 128])
def test_an_fp16_backward_compiles_spill_free_as_its_model_decides(width):
    """An fp16 call's backward of a causal call `width` wide (the one launch's two parts at running scales,
    `attention_mosaic_fused._sixteen_step`, where its registers hold it), certified as the route certifies it
    (`MosaicBackward.uncertified`: a config the GPU's assembler refuses, spilling or serializing its
    warpgroup MMAs, stepped past) and lowered and compiled by ptxas for an H100 on this CPU: within its
    shared memory, spilling nothing, its warpgroup MMAs unserialized."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    queries = 512
    spans = template.spans(queries, queries, 1, causal=True)
    query = jax.ShapeDtypeStruct((1, queries, 4, width), jnp.float16)
    key = jax.ShapeDtypeStruct((1, queries, 2, width), jnp.float16)
    shapes = jax.tree.map(lambda array: jax.ShapeDtypeStruct(array.shape, array.dtype),
                          _residuals(query.shape))

    def backward(query, key, value, dout, residuals):
        grads = attention_mosaic.backward(query, key, value, spans, None, dout, residuals=residuals)
        return grads.query, grads.key, grads.value

    with generation.assumed("sm90"):
        found = compile_mosaic_kernels(
            [KernelCase("backward", lambda: (backward, (query, key, key, query, shapes)))], "sm90")
    failed = mosaic_failures(found)
    assert not failed, "\n".join(failed)


@COMPILES_ON_CPU
@pytest.mark.parametrize("split_cols", [False, True], ids=["whole", "split"])
@pytest.mark.parametrize("kept", [template.Kept.SPLIT, template.Kept.NONE], ids=["training", "inference"])
@pytest.mark.parametrize("rows", [2, 1], ids=["step", "decision"])
def test_each_tcgen05_forward_legal_for_the_4b_step_lowers_for_a_b200(rows, kept, split_cols):
    """Qwen3.5-4B's attention (16 query heads over 4, 256 wide, 12,288 causal tokens: its step's two rows and
    the planner's one-row decision) on a compile-only B200: every config of each tcgen05 family legal for the
    call (`MosaicForward.offered`: the family's configs its derivation searches, each as it launches), pinned,
    lowers, so a compile's refusal never steps the call onto a forward it cannot build (the B200 4B step, its
    spill refused onto 64 x 128 keys at 16 value columns, raised at its values' padded operand; tcgen05's rows
    at 16 value columns, its rescale's block of 32 past them, et-rows-spill). The call has such configs."""
    tokens, heads, kv_heads, width = 12288, 16, 4, 256
    spans = template.spans(tokens, tokens, rows, causal=True)
    query = jax.ShapeDtypeStruct((rows, tokens, heads, width), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((rows, tokens, kv_heads, width), jnp.bfloat16)

    def forward(query, key, value):
        return attention_mosaic.forward(query, key, value, spans, None, residuals=kept)

    with generation.assumed("sm100"):
        target = generation.gpu_target()
        op, args = attention_mosaic._forward_call(query, key, key, spans, None, scale=None,
                                                  natural=attention_mosaic.NATURAL)
        ba = op.bind(*args, residuals=kept, interpret=False)
        all_legal = [config for family in attention_mosaic_program.layouts("sm100", jnp.bfloat16, tokens)
                     if family.name.startswith("tcgen05") for config in op.offered(ba, family, target)]
        assert all_legal
        legal = [config for config in all_legal if (config.columns is not None) == split_cols]
        for config in legal:
            with tuning.pinned({attention_mosaic.MosaicForward: config}):
                jax.jit(forward).trace(query, key, key).lower(lowering_platforms=("cuda",))


@COMPILES_ON_CPU
@pytest.mark.parametrize("kept", [template.Kept.SPLIT, template.Kept.NONE], ids=["training", "inference"])
@pytest.mark.parametrize("rows", [2, 1], ids=["step", "decision"])
def test_the_4b_steps_forward_compiles_spill_free_on_tensor_memory_for_a_b200(rows, kept):
    """Qwen3.5-4B's attention forward (16 query heads over 4, 256 wide, 12,288 causal tokens: the step's two
    rows and the planner's one-row decision) at the config its derivation gives on a compile-only B200: every
    launch compiled by ptxas spilling nothing (`mosaic_failures`). A B200's 4B step stepped through 19
    spilling chunks forwards for 42 minutes and landed on the Ampere family's."""
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
        assert config is not None
        case = KernelCase("4B forward", lambda: (forward, (query, key, key)))
        found = compile_mosaic_kernels([case], "sm100")
    failed = mosaic_failures(found)
    assert not failed, "\n".join(failed)


@COMPILES_ON_CPU
@pytest.mark.parametrize("width", [128, 256])
def test_the_walked_backward_of_the_4b_heads_compiles_spill_free_on_a_b200(width):
    """Qwen3.5-4B's heads (16 query heads over 4, 12,288 causal tokens) at widths 128 and 256 on a
    compile-only B200: the call has a backward whose launches tcgen05's walked placements take
    (`MosaicBackward.walked`, row 4839: the derivation's search of them by what fits, the one launch keys
    first or the two), and pinned there, its training step's every launch compiles by ptxas spilling nothing
    (`mosaic_failures`). Today's dQ on mma.sync held 340 registers at width 256, past a thread's 248, and the
    4B backward ran on Triton."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    from dew.nn.kernels import tuning

    tokens, heads, kv_heads = 12288, 16, 4
    spans = template.spans(tokens, tokens, 1, causal=True)
    query = jax.ShapeDtypeStruct((1, tokens, heads, width), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((1, tokens, kv_heads, width), jnp.bfloat16)

    def trained(query, key, value):
        return attention_mosaic.forward(query, key, value, spans, None, residuals=template.Kept.SPLIT)

    def step(query, key, value, dout):
        _, *residuals = trained(query, key, value)
        return attention_mosaic.backward(query, key, value, spans, None, dout, residuals=tuple(residuals))

    with generation.assumed("sm100"):
        _, *residuals = jax.eval_shape(trained, query, key, key)
        op, args = attention_mosaic._backward_call(query, key, key, spans, None, query, tuple(residuals),
                                                   scale=None, natural=attention_mosaic.NATURAL,
                                                   accurate=attention_mosaic.ACCURATE)
        config = op.walked(op.bind(*args, interpret=False), generation.gpu_target())
        assert config is not None
        with tuning.pinned({attention_mosaic.MosaicBackward: config}):
            found = compile_mosaic_kernels([KernelCase("4B step", lambda: (step, (query, key, key, query)))],
                                           "sm100")
    failed = mosaic_failures(found)
    assert not failed, "\n".join(failed)


STACK = (1, 4096, 16, 4, 256, 8192)
"""A checkpointed layer's batch, tokens, query heads, key heads, head width and MLP width: the 4B's
grouped 256-wide heads, whose backward on an H100 adds its dQ and dK/dV into accumulators in place."""


def _stack_temp(layers: int) -> int:
    """The temporaries of a training step of `layers` checkpointed layers (each a grouped causal attention
    through the template, then a wide MLP), compiled for a compile-only H100 stating 128 MiB, below the step's
    need, so XLA's scheduler holds the least it can and the temporaries are its buffers' lifetimes alone (an
    H100's 80 GB lets it overlap the layers, 244 MB at two; a compile-only GPU stated 0 bytes before row
    8768)."""
    batch, tokens, heads, groups, width, hidden = STACK

    def layer(x, weights):
        keys, up, down = weights
        k = jnp.einsum("bshd,hg->bsgd", x, keys).astype(jnp.bfloat16)
        attended = scaled_dot_product_attention(x, k, k, causal=True, implementation="template")
        flat = attended.reshape(batch, tokens, heads * width)
        return (jnp.tanh(flat @ up) @ down).reshape(x.shape).astype(jnp.bfloat16)

    def loss(x, stack):
        for weights in stack:
            x = jax.checkpoint(layer)(x, weights)
        return jnp.sum(x.astype(jnp.float32))

    x = jax.ShapeDtypeStruct((batch, tokens, heads, width), jnp.bfloat16)
    one = (jax.ShapeDtypeStruct((heads, groups), jnp.bfloat16),
           jax.ShapeDtypeStruct((heads * width, hidden), jnp.bfloat16),
           jax.ShapeDtypeStruct((hidden, heads * width), jnp.bfloat16))
    with generation.assumed("sm90"):
        compiled = lowered_for(jax.grad(loss, argnums=(0, 1)), (x, (one,) * layers), "sm90",
                               memory=2**27).compile()
    return compiled.memory_analysis().temp_size_in_bytes


@COMPILES_ON_CPU
def test_a_stack_of_layers_holds_no_launchs_accumulator_past_its_own_layer():
    """Two checkpointed layers' training step on a compile-only H100 holds no more temporaries than one
    layer's step and the second layer's input it saves for its backward: each backward launch's in-place
    accumulators its own (`launches.filled`), made before it and freed after it. Where every layer's equal
    zeros were one value to XLA, the step held the shared buffer across both layers' backwards (269 MB
    against one layer's 179 MB; the 4B's split-1 step, 805 MB across 28 layers)."""
    batch, tokens, heads, _, width, _ = STACK
    saved = batch * tokens * heads * width * jnp.dtype(jnp.bfloat16).itemsize
    assert _stack_temp(2) <= _stack_temp(1) + saved


@pytest.mark.parametrize("accurate", [False, True], ids=["fast", "accurate"])
def test_each_attention_launch_states_its_outputs_dimensions_for_an_h100(accurate):
    """A training step's attention on the Mosaic mainloop, lowered for an H100, and its forward alone: every
    kernel call that defines arrays of its own (not refs it writes in place) states the operand axes each
    output descends from (`outputs.Output`, its call's `outputs.OUTPUT_DIMS` frontend attribute), an output
    an entry a dimension, so a compile's live buffers name the call's axes they scale with; the fast form's
    one launch and the accurate form's two (a split of the columns among them included)."""
    from kernel_compile import stated_outputs

    tokens, heads, kv_heads, width = 1024, 4, 2, 128
    spans = template.spans(tokens, tokens, 1, causal=True)
    query = jax.ShapeDtypeStruct((1, tokens, heads, width), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((1, tokens, kv_heads, width), jnp.bfloat16)

    def step(query, key, value):
        out, *residuals = attention_mosaic.forward(query, key, value, spans, None,
                                                   residuals=template.Kept.SPLIT)
        return attention_mosaic.backward(query, key, value, spans, None, out, residuals=tuple(residuals),
                                         accurate=accurate)

    def forward(query, key, value):
        return attention_mosaic.forward(query, key, value, spans, None)

    for fn in (step, forward):
        with generation.assumed("sm90"):
            lowered = jax.jit(fn).trace(query, key, key).lower(lowering_platforms=("cuda",))
        # Each Mosaic GPU call: the outputs it defines and the dimensions it states (`stated_outputs`).
        returning = [stated for count, stated in stated_outputs(lowered, "mosaic_gpu_v2") if count]
        assert returning, "no Mosaic launch defines an output"
        for stated in returning:
            assert stated is not None, "a launch returning arrays states no dimensions"
            assert all(dims is not None for dims in stated), stated


def _lowered_call(call: str) -> tuple:
    """A lowering test's call form `call` as abstract operands: its spans, its query and key `[B, S, H, 128]`
    bf16 and a training forward's residuals' shapes, the compiled shape's causal call or a template core
    case's (`CASES`, `RUN_CASES`) over four query heads and two key heads."""
    if call == "causal":
        rows, tokens, heads, kv_heads = SHAPES["compiled"]
        spans = template.spans(tokens, tokens, rows, causal=True)
    else:
        case = CASES[call] if call in CASES else RUN_CASES[call]
        (tokens, _), rows, heads, kv_heads = core_geometry(case), ROWS, 4, 2
        spans = template.spans(tokens, tokens, rows, **arguments(case))
    q = jax.ShapeDtypeStruct((rows, tokens, heads, 128), jnp.bfloat16)
    kv = jax.ShapeDtypeStruct((rows, tokens, kv_heads, 128), jnp.bfloat16)
    rows_shape = jax.ShapeDtypeStruct((rows, heads, tokens), jnp.float32)
    others = jax.ShapeDtypeStruct((rows, heads, tokens, 128), jnp.float32)
    split = template.Split(others, rows_shape, rows_shape, rows_shape)
    anchors = jax.ShapeDtypeStruct((rows, heads, tokens), jnp.int32)
    return spans, q, kv, (rows_shape, rows_shape, anchors, split)


@lowers("sm90")
@pytest.mark.parametrize("accurate", [False, True], ids=["default", "accurate"])
@pytest.mark.parametrize("call", ["causal", "causal-documents-window", "runs-documents-valid"])
def test_the_launches_lower_for_an_h100(call, accurate):
    """The forward and the backward's launches lower for an H100 on any host, as the GPU gate runs them: a
    causal call, a spans one of documents, whose programs read its block map and ids, and one over a
    decision row's runs, whose programs read each run's bounds; the forward under every hook too, whose
    programs read each kind of parameter and the initial state."""
    spans, q, kv, residuals = _lowered_call(call)
    tokens = q.shape[1]
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90")
    score_mod, kinds, draw_hooks = score_hooks(FORWARD_HOOKS["all"][0], tokens, tokens, 128)
    params, initial = draw_hooks(0)
    gate = hooks_for(HOOKS["gate"], tokens, 128, gate_of(tokens, 128))
    hooks = template.Hooks(score_mod, tuple(template.Param(value, kind)
                                            for value, kind in zip(params, kinds, strict=True)), initial,
                           gate.output_mod, gate.output_params)
    with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh):
        for forward_hooks in (None, hooks):
            for training in (template.Kept.NONE, template.Kept.SPLIT):
                entry = functools.partial(attention_mosaic.forward, spans=spans, hooks=forward_hooks,
                                          residuals=training)
                jax.jit(entry).trace(q, kv, kv).lower(lowering_platforms=("cuda",))
        jax.jit(lambda q, k, v, dout, residuals: attention_mosaic.backward(
            q, k, v, spans, None, dout, residuals=residuals, accurate=accurate)).trace(
            q, kv, kv, q, residuals).lower(lowering_platforms=("cuda",))


@pytest.mark.parametrize("call", ["causal", "causal-documents-window"])
def test_the_launches_lower_for_a_v5e(call):
    """The forward, the training forward and the backward on a v5e's MXU (`mosaic_tpu.MXU`, the route's Mosaic
    mainloop on a TPU) lower for a TPU on this CPU, Pallas TPU's Mosaic lowering, as a TPU compiles them: a
    causal call and a spans one of documents under a window, whose programs read its block map and ids."""
    spans, q, kv, residuals = _lowered_call(call)
    # An abstract mesh of one v5e device, which Pallas TPU's lowering reads its target from on any host.
    device = AbstractDevice(device_kind=generation.gpu_target("v5e").device_kind, num_cores=1, platform="tpu")
    mesh = AbstractMesh((1,), ("data",), axis_types=(AxisType.Auto,), abstract_device=device)
    with generation.assumed("v5e"), jax.sharding.use_abstract_mesh(mesh):
        for training in (template.Kept.NONE, template.Kept.SPLIT):
            entry = functools.partial(attention_mosaic.forward, spans=spans, hooks=None, residuals=training)
            jax.jit(entry).trace(q, kv, kv).lower(lowering_platforms=("tpu",))
        jax.jit(lambda q, k, v, dout, residuals: attention_mosaic.backward(
            q, k, v, spans, None, dout, residuals=residuals)).trace(q, kv, kv, q, residuals).lower(
            lowering_platforms=("tpu",))


@COMPILES_FOR_TPU
@pytest.mark.usefixtures("libtpu_shared")
@pytest.mark.parametrize("case_name", ["v5e-fwd", "v5e-step", "v5e-fwd-d64", "v5e-fwd-8192", "v5e-step-8192",
                                       "v6e-fwd-512", "v6e-step"])
def test_bf16_template_compiles_for_a_v5e(case_name):
    """Sub-32-bit TPU arrays whose head width divides the TPU lane unit (`column_unit`, 128 lanes) bitcast
    `[B, S, H, D]` to `[B, S, H * D]` (`head_flat`) with zero XLA transposes on both v5e and v6e; narrower
    widths fall back to `head_first`, fp32 lowers in place, and every case compiles cleanly through `libtpu`
    (rows 6993, 10152, 10683)."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_tpu_kernels, failures

    query = jax.ShapeDtypeStruct((1, 256, 4, 128), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((1, 256, 2, 128), jnp.bfloat16)
    query_512 = jax.ShapeDtypeStruct((1, 512, 4, 128), jnp.bfloat16)
    key_512 = jax.ShapeDtypeStruct((1, 512, 2, 128), jnp.bfloat16)
    query_d64 = jax.ShapeDtypeStruct((1, 256, 4, 64), jnp.bfloat16)
    key_d64 = jax.ShapeDtypeStruct((1, 256, 2, 64), jnp.bfloat16)
    long_q = jax.ShapeDtypeStruct((1, 8192, 32, 128), jnp.bfloat16)
    long_k = jax.ShapeDtypeStruct((1, 8192, 8, 128), jnp.bfloat16)

    def fwd(q, k, v):
        with generation.chosen(attention_mainloop="mosaic"):
            return scaled_dot_product_attention(q, k, v, causal=True, implementation="template")

    def step(q, k, v, dout):
        with generation.chosen(attention_mainloop="mosaic"):
            out, pull = jax.vjp(fwd, q, k, v)
            return out, *pull(dout)

    if case_name == "v5e-fwd":
        tpu_family = attention_mosaic_program.bundle("v5e", jnp.bfloat16)
        assert tpu_family is not None
        assert attention_mosaic_program.head_flat(tpu_family, jnp.bfloat16, 128)
        assert attention_mosaic_program.head_flat(tpu_family, jnp.bfloat16, 256)
        assert not attention_mosaic_program.head_flat(tpu_family, jnp.bfloat16, 64)
        assert not attention_mosaic_program.head_flat(tpu_family, jnp.float32, 128)
        assert not attention_mosaic_program.head_first(tpu_family, jnp.bfloat16, 128)
        assert attention_mosaic_program.head_first(tpu_family, jnp.bfloat16, 64)
        assert not attention_mosaic_program.head_first(tpu_family, jnp.float32, 64)

    cases = {
        "v5e-fwd": ("v5e", KernelCase("v5e-bf16-fwd", lambda: (fwd, (query, key, key)))),
        "v5e-step": ("v5e", KernelCase("v5e-bf16-step", lambda: (step, (query, key, key, query)))),
        "v5e-fwd-d64": ("v5e", KernelCase("v5e-bf16-fwd-d64", lambda: (fwd, (query_d64, key_d64, key_d64)))),
        "v5e-fwd-8192": ("v5e", KernelCase("v5e-bf16-fwd-8192", lambda: (fwd, (long_q, long_k, long_k)))),
        "v5e-step-8192": ("v5e", KernelCase("v5e-bf16-step-8192",
                                            lambda: (step, (long_q, long_k, long_k, long_q)))),
        "v6e-fwd-512": ("v6e", KernelCase("v6e-bf16-fwd-512", lambda: (fwd, (query_512, key_512, key_512)))),
        "v6e-step": ("v6e", KernelCase("v6e-bf16-step", lambda: (step, (query, key, key, query)))),
    }
    gen, case = cases[case_name]
    found = compile_tpu_kernels([case], gen)
    assert not failures(found), "\n".join(failures(found))


APPROXIMATED_BY_MOSAIC = ("exp", "exp2", "log", "log2", "log1p", "tanh", "logistic", "sin", "cos", "sqrt",
                          "rsqrt")
"""The unary operations whose Mosaic GPU lowering takes `approx_math` (jax's Pallas Mosaic GPU lowering rules
for exp, exp2, log, log2, tanh, sin, cos, sqrt and rsqrt, which pass it to the lane implementation and set
the fast-math flag; logistic and log1p lower through them), read from the lowering, not from the
mainloop's own list."""


@pytest.mark.parametrize("operation", APPROXIMATED_BY_MOSAIC)
def test_every_approximated_operation_in_a_hook_takes_exact_math(operation):
    """A score hook, an output hook and a query prologue each computing one operation Mosaic GPU approximates
    under `approx_math` (`APPROXIMATED_BY_MOSAIC`) take exact math, in the forward and (the score hook)
    differentiated: no such operation of a hook passes unseen."""
    def applied(x):
        return getattr(jax.lax, operation)(jnp.abs(x) + 1.0)

    def score_mod(s, q_pos, k_pos):
        del q_pos, k_pos
        return s + applied(s)

    def output_mod(out):
        return applied(out)

    def query_mod(q, q_pos):
        del q_pos
        return applied(q.astype(jnp.float32)).astype(q.dtype)

    for hooks in (template.Hooks(score_mod), template.Hooks(output_mod=output_mod)):
        for differentiated in (False, True):
            assert not attention_mosaic_program.approximate(hooks, (64, 64), 64,
                                                            differentiated=differentiated)
    # A query prologue a launch takes on its query tile (`attention_mosaic_forward._prologued`).
    assert not attention_mosaic_program.approximate(
        template.Hooks(query_mod=query_mod), (64, 64), 64, differentiated=False,
        query=(64, 64, jnp.bfloat16))


def _fullest(op, ba, target):
    """The fitting config of `op`'s launch of `ba` on `target` that stages the most shared memory."""
    return max(op.fitting(ba, target), key=lambda config: op.staged_bytes(ba, config, target))


@lowers("sm90")
@pytest.mark.parametrize("launch", ["forward", "training", "default", "accurate"])
def test_the_fullest_fitting_tiles_lower_within_a_hoppers_block(launch):
    """Mosaic GPU's own lowering, which refuses a kernel staging past a block's shared memory, takes each
    launch at the fitting config that stages the most (`staged_bytes`), the one nearest the H100's block: the
    model counts every array and barrier the launch allocates (`_staged`), an independent witness of it."""
    rows, tokens, heads, kv_heads = SHAPES["compiled"]
    width = 128
    spans = template.spans(tokens, tokens, rows, causal=True)
    q = jax.ShapeDtypeStruct((rows, tokens, heads, width), jnp.bfloat16)
    kv = jax.ShapeDtypeStruct((rows, tokens, kv_heads, width), jnp.bfloat16)
    stats = jnp.zeros((rows, heads, tokens), jnp.float32)
    residuals = (stats, stats, jnp.zeros((rows, heads, tokens), jnp.int32),
                 template.Split(jnp.zeros((rows, heads, tokens, width), jnp.float32), stats, stats, stats))
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90")
    with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh):
        target = generation.gpu_target()
        if launch in ("forward", "training"):
            training = _kept(launch)
            op, ba = _forward_call((rows, tokens, heads, kv_heads), width, width, training)
            fullest = _fullest(op, ba, target)
            assert op.staged_bytes(ba, fullest, target) <= generation.shared_budget(target)
            with _pinned(fullest):
                entry = functools.partial(attention_mosaic.forward, spans=spans, residuals=training)
                jax.jit(entry).trace(q, kv, kv).lower(lowering_platforms=("cuda",))
            return
        accurate = launch == "accurate"
        op = attention_mosaic.MosaicBackward(form="causal", keys=tokens, scale=width ** -0.5,
                                             accurate=accurate)
        ba = _bound(op, q, kv, kv, q, spans.start, spans.end, None, None, (), None, residuals)
        fullest = _fullest(op, ba, target)
        assert op.staged_bytes(ba, fullest, target) <= generation.shared_budget(target)
        with _pinned(backward=fullest):
            jax.jit(lambda q, k, v, dout, residuals: attention_mosaic.backward(
                q, k, v, spans, None, dout, residuals=residuals, accurate=accurate)).trace(
                q, kv, kv, q, residuals).lower(lowering_platforms=("cuda",))


@lowers("sm90")
@pytest.mark.parametrize("accurate", [False, True], ids=["default", "accurate"])
@pytest.mark.parametrize("hooks", BACKWARD_HOOKS)
def test_the_backward_under_every_score_hook_lowers_for_an_h100(hooks, accurate):
    """The backward under each score hook lowers for an H100 on any host, as the GPU gate runs it: each hook
    parameter and the initial state read per tile by the dQ and dK/dV programs."""
    case = CASES["causal-documents-window"]
    queries, keys = core_geometry(case)
    spans = template.spans(queries, keys, ROWS, **arguments(case))
    q = jax.ShapeDtypeStruct((ROWS, queries, 4, 128), jnp.bfloat16)
    kv = jax.ShapeDtypeStruct((ROWS, keys, 2, 128), jnp.bfloat16)
    rows_shape = jnp.zeros((ROWS, 4, queries), jnp.float32)
    split = template.Split(jnp.zeros((ROWS, 4, queries, 128), jnp.float32), rows_shape, rows_shape,
                           rows_shape)
    residuals = (rows_shape, rows_shape, jnp.zeros((ROWS, 4, queries), jnp.int32), split)
    score_mod, kinds, draw_hooks = score_hooks(SCORE_HOOKS[hooks], queries, keys, 128)
    params, initial = draw_hooks(0)

    def backward_of(q, k, v, dout, residuals, params, initial):
        hooked = template.Hooks(score_mod, tuple(template.Param(value, kind)
                                                 for value, kind in zip(params, kinds, strict=True)),
                                initial)
        return attention_route.backward(q, k, v, spans, hooked, dout, residuals=residuals, accurate=accurate)

    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90")
    with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh):
        jax.jit(backward_of).trace(q, kv, kv, q, residuals, params, initial).lower(
            lowering_platforms=("cuda",))


@lowers("sm90")
@pytest.mark.parametrize("width", [64, 128])
@pytest.mark.parametrize("natural", [False, True], ids=["folded", "natural"])
@pytest.mark.parametrize("hooks", BACKWARD_HOOKS)
def test_the_training_forward_under_every_score_hook_lowers_for_an_h100(hooks, natural, width):
    """The training forward that supplies each hooked backward's residuals lowers for an H100 at the GPU
    cells' head widths, in either form: a row-indexed hook's unit column of its query places found no layout
    beside the walk's split on mma.sync (row 5279)."""
    case = CASES["causal-documents-window"]
    queries, keys = core_geometry(case)
    spans = template.spans(queries, keys, ROWS, **arguments(case))
    q = jax.ShapeDtypeStruct((ROWS, queries, 4, width), jnp.bfloat16)
    kv = jax.ShapeDtypeStruct((ROWS, keys, 2, width), jnp.bfloat16)
    score_mod, kinds, draw_hooks = score_hooks(SCORE_HOOKS[hooks], queries, keys, width)
    params, initial = draw_hooks(0)

    def trained(q, k, v, params, initial):
        hooked = template.Hooks(score_mod, tuple(template.Param(value, kind)
                                                 for value, kind in zip(params, kinds, strict=True)),
                                initial)
        return attention_mosaic.forward(q, k, v, spans, hooked, natural=natural,
                                        residuals=template.Kept.SPLIT)

    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90")
    with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh):
        jax.jit(trained).trace(q, kv, kv, params, initial).lower(lowering_platforms=("cuda",))


@COMPILES_ON_CPU
@pytest.mark.parametrize("axis", [0, 1])
def test_a_hook_that_forms_a_unit_axis_compiles_for_an_h100(axis):
    """A score hook that forms a unit column (`q_pos[:, :1]`, `axis` 0) or a unit row (`k_pos[:1, :]`, 1,
    which Mosaic GPU selects by a reduction across warps, its scratch the launch's warpgroups reserve,
    ledger 1428), the training forward and the backward lowered and compiled by ptxas for an H100 on this
    CPU (tests/kernel_compile.py), within its shared memory (`failures`: no lowering error, no block past
    it). Its registers at the derived tiles are the template's register model's (ledger 1617), hooked or
    not."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, failures

    queries = 256
    spans = template.spans(queries, queries, 1, causal=True)
    def score_mod(s, q_pos, k_pos):
        return s + 0.01 * (q_pos[:, :1] if axis == 0 else k_pos[:1, :]).astype(s.dtype)

    hooks = template.Hooks(score_mod)
    query = jax.ShapeDtypeStruct((1, queries, 4, 128), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((1, queries, 2, 128), jnp.bfloat16)

    def trained(query, key, value):
        return attention_mosaic.forward(query, key, value, spans, hooks, residuals=template.Kept.SPLIT)

    def backward(query, key, value, dout):
        residuals = trained(query, key, value)[1:]
        grads = attention_mosaic.backward(query, key, value, spans, hooks, dout, residuals=residuals)
        return grads.query, grads.key, grads.value

    found = compile_mosaic_kernels([KernelCase("forward", lambda: (trained, (query, key, key))),
                                    KernelCase("backward", lambda: (backward, (query, key, key, query)))],
                                   "sm90")
    assert not failures(found), "\n".join(failures(found))


@COMPILES_ON_CPU
@pytest.mark.parametrize("generation_name", ["sm90", "sm80"])
def test_an_int8_cache_decode_at_width_256_derives_and_compiles_spill_free(generation_name):
    """Row 5449: an int8 KV-cache decode at width 256 derives a legal config and compiles spill-free once
    `Family.converted` converts at most `CONVERT_COLUMNS` (128) columns at a time."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    from dew.nn.kernels.templates import attention_decode

    rows, keys, heads, kv_heads, width = 1, 512, 8, 2, 256
    spans = template.spans(1, keys, rows)
    query = jax.ShapeDtypeStruct((rows, 1, heads, width), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((rows, keys, kv_heads, width), jnp.int8)
    scales = (jax.ShapeDtypeStruct((rows, keys, kv_heads), jnp.float32),
              jax.ShapeDtypeStruct((rows, keys, kv_heads), jnp.float32))

    def decoded(query, key, value, k_scale, v_scale):
        return attention_decode.attend(query, key, value, spans, scales=(k_scale, v_scale))

    with generation.assumed(generation_name):
        found = compile_mosaic_kernels(
            [KernelCase("int8-w256-decode", lambda: (decoded, (query, key, key, *scales)))],
            generation_name)
    assert not mosaic_failures(found), "\n".join(mosaic_failures(found))


@COMPILES_ON_CPU
@pytest.mark.parametrize("stored", [jnp.float8_e4m3fn, jnp.float8_e5m2], ids=["e4m3fn", "e5m2"])
def test_an_fp8_cache_decode_on_sm80_derives_and_compiles_spill_free(stored):
    """Row 8549: an FP8 KV-cache decode on sm80 decodes via `bits.decoded` integer arithmetic (no sm_89
    `cvt.e4m3x2`/`e5m2x2` instruction) and compiles spill-free with ptxas."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    from dew.nn.kernels.templates import attention_decode

    rows, keys, heads, kv_heads, width = 1, 512, 8, 2, 128
    spans = template.spans(1, keys, rows)
    query = jax.ShapeDtypeStruct((rows, 1, heads, width), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((rows, keys, kv_heads, width), stored)
    scales = (jax.ShapeDtypeStruct((rows, keys, kv_heads), jnp.float32),
              jax.ShapeDtypeStruct((rows, keys, kv_heads), jnp.float32))

    def decoded(query, key, value, k_scale, v_scale):
        return attention_decode.attend(query, key, value, spans, scales=(k_scale, v_scale))

    with generation.assumed("sm80"):
        found = compile_mosaic_kernels(
            [KernelCase("fp8-sm80-decode", lambda: (decoded, (query, key, key, *scales)))], "sm80")
    assert not mosaic_failures(found), "\n".join(mosaic_failures(found))


def test_fused_keys_first_register_model_counts_dq_product():
    """`_keys_first_registers` includes the fused `dQ = dS^T @ K` product fragments beside `step[0]` and
    the dQ accumulator (row 6944): on `sm80` (`mma.sync`, `operands_in_registers=True`, `in_flight=0`),
    at `128x64` width 128 (`split=True`), `fused=True` counts strictly more registers than `fused=False`."""
    with generation.assumed("sm80"):
        family = attention_mosaic_program.bundle("sm80", jnp.bfloat16)
        fused = attention_mosaic.BackwardTiles(
            128, 64, 2, 1, fused=True, family=family.name, dkv_family=family.name, dkv_consumers=2,
            narrow=True)
        unfused = dataclasses.replace(fused, fused=False)
        dkv_family = attention_mosaic_program.bundle_of(fused.dkv, jnp.bfloat16)
        reg_fused = attention_mosaic_registers.backward_registers(
            dkv_family, fused.dkv, 128, 128, jnp.bfloat16, accurate=False, dq=False, split=True)
        reg_unfused = attention_mosaic_registers.backward_registers(
            dkv_family, unfused.dkv, 128, 128, jnp.bfloat16, accurate=False, dq=False, split=True)
        assert reg_fused > reg_unfused, (reg_fused, reg_unfused)


@COMPILES_ON_CPU
def test_an_a100_d128_and_d256_forward_use_whole_row_warps_and_high_residency():
    """An A100's 64x64 forward at width 128 keeps four warps along rows (no relay or cross-warp softmax)
    and omits unused vector staging (`CpAsync.stages_vectors`), fitting two resident programs per SM
    spill-free; width 256 at columns=128 caps its natural rewalk at columns=128 and compiles spill-free
    (rows 5272, 5273)."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    with generation.assumed("sm80"):
        target = generation.gpu_target()
        spans = template.spans(512, 512, 1, causal=True)
        q128 = jnp.zeros((1, 512, 4, 128), jnp.bfloat16)
        k128 = jnp.zeros((1, 512, 2, 128), jnp.bfloat16)
        op128, args128 = attention_mosaic._forward_call(q128, k128, k128, spans, None, scale=None,
                                                        natural=attention_mosaic.NATURAL)
        ba128 = op128.bind(*args128, residuals=template.Kept.NONE, interpret=False)
        cfg128 = op128._launchable(ba128, attention_mosaic.Tiles(64, 64, 2, 1, family="mma.sync"), target)
        assert cfg128 is not None
        assert attention_mosaic_program.bundle_of(cfg128, jnp.bfloat16).rows_whole
        assert op128.resident(ba128, cfg128, target) >= 2

        q256 = jnp.zeros((1, 512, 4, 256), jnp.bfloat16)
        k256 = jnp.zeros((1, 512, 2, 256), jnp.bfloat16)
        op256, args256 = attention_mosaic._forward_call(q256, k256, k256, spans, None, scale=None,
                                                        natural=attention_mosaic.NATURAL)
        ba256 = op256.bind(*args256, residuals=template.Kept.NONE, interpret=False)
        cfg256 = op256._launchable(
            ba256, attention_mosaic.Tiles(64, 64, 2, 1, family="mma.sync", columns=128), target)
        assert cfg256 is not None
        assert cfg256.relaunch_columns == 128

        found = compile_mosaic_kernels([
            KernelCase("sm80-d128-64x64", lambda: op128.lowering(ba128, cfg128)),
            KernelCase("sm80-d256-64x64-c128", lambda: op256.lowering(ba256, cfg256)),
        ], "sm80")
        assert not mosaic_failures(found), "\n".join(mosaic_failures(found))


@COMPILES_ON_CPU
def test_fp32_forward_derives_parted_wgmma_and_consumer_choice_uses_launch_seconds():
    """An accurate fp32 forward on sm90 derives its 3-part bf16 wgmma plan from
    `dew.algebra.error.product_precision` (row 5473) and compiles without spills, while a split bf16
    forward derives its consumer count via `cost.launch_seconds` (row 10811)."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    spans = template.spans(512, 512, 1, causal=True)
    with generation.assumed("sm90"):
        target = generation.gpu_target()
        q32 = jnp.zeros((1, 512, 8, 64), jnp.float32)
        op32, args32 = attention_mosaic._forward_call(q32, q32, q32, spans, None, scale=None,
                                                      natural=attention_mosaic.NATURAL)
        ba32 = op32.bind(*args32, residuals=template.Kept.NONE, interpret=False)
        cfg32 = op32.derived(ba32, target)
        assert cfg32 is not None and op32.parts == 3, (op32.parts, cfg32)

        q16 = jax.ShapeDtypeStruct((2, 512, 32, 128), jnp.bfloat16)
        k16 = jax.ShapeDtypeStruct((2, 512, 8, 128), jnp.bfloat16)
        op16, args16 = attention_mosaic._forward_call(q16, k16, k16, spans, None, scale=None,
                                                      natural=attention_mosaic.NATURAL)
        ba16 = op16.bind(*args16, residuals=template.Kept.SPLIT, interpret=False)
        cfg16 = op16.derived(ba16, target)
        assert cfg16 is not None and op16.takes(ba16, cfg16, target), cfg16
        # Row 12074: compare explicit 1- and 2-consumer schedule variants via pricing without prescribing
        # which consumer count the unpinned planner picks.
        variants = [v for c in (1, 2) if (v := op16._launchable(
            ba16, dataclasses.replace(cfg16, consumers=c, relaunch_k=None, relaunch_consumers=None,
                                      relaunch_columns=None), target)) is not None
            and op16.takes(ba16, v, target)]
        assert len(variants) == 2, variants
        prices = [op16.price_config(ba16, v, target) for v in variants]
        assert all(p is not None and p > 0 for p in prices), prices
        chosen16 = op16._choose_consumers(ba16, variants, target=target)
        assert isinstance(chosen16, attention_mosaic.Tiles) and chosen16 in variants
        chosen_price = op16.price_config(ba16, chosen16, target)
        assert chosen_price is not None and chosen_price == min(p for p in prices if p is not None)
        found = compile_mosaic_kernels([
            KernelCase("sm90-fp32-3part", lambda: op32.lowering(ba32, cfg32)),
        ], "sm90")
        assert not mosaic_failures(found), "\n".join(mosaic_failures(found))


@COMPILES_ON_CPU
def test_sm90_fused_and_coop_backward_pipeline_dq_dkv():
    """Row 5272: sm90 fused (d64, d128) and cooperative (d256) backward kernels stage P/dS under one
    proxy fence and pipeline dQ/dV/dK WGMMAs (dQ read past dV and dK's groups, `product_with`) with zero
    register spills."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    from dew.nn.kernels.templates import attention_mosaic_coop, attention_mosaic_fused

    spans = template.spans(512, 512, 1, causal=True)
    with generation.assumed("sm90"):
        target = generation.gpu_target()
        cases = []
        for label, width in (("sm90-fused-d64", 64), ("sm90-fused-d128", 128), ("sm90-coop-d256", 256)):
            q = jnp.zeros((1, 512, 8, width), jnp.bfloat16)
            k = jnp.zeros((1, 512, 2, width), jnp.bfloat16)
            dout = jnp.zeros_like(q)
            residuals = (jnp.zeros((1, 8, 512), jnp.float32), jnp.ones((1, 8, 512), jnp.float32),
                         jnp.zeros((1, 8, 512), jnp.int32),
                         attention_mosaic.Split(jnp.zeros((1, 8, 512, width), jnp.float32),
                                                jnp.ones((1, 8, 512), jnp.float32),
                                                jnp.zeros((1, 8, 512), jnp.float32),
                                                jnp.zeros((1, 8, 512), jnp.float32)))
            op, args = attention_mosaic._backward_call(q, k, k, spans, None, dout, residuals, scale=None,
                                                       natural=attention_mosaic.NATURAL, accurate=False)
            ba = op.bind(*args, interpret=False)
            cfg = op.derived(ba, target)
            assert cfg is not None and cfg.fused, (label, cfg)
            family = attention_mosaic.bundle_of(cfg, jnp.bfloat16)
            if cfg.shared_keys:
                assert attention_mosaic_fused.streamed_statistics(
                    family, cfg, width, jnp.bfloat16, cfg.stages, reduced=0,
                    allocations_of=attention_mosaic_coop._coop_allocations) >= 2, (label, cfg)
            else:
                assert attention_mosaic_fused.streamed_statistics(
                    family, cfg, width, jnp.bfloat16, cfg.stages, reduced=0) >= 3, (label, cfg)
            cases.append(KernelCase(label, lambda op=op, ba=ba, cfg=cfg: op.lowering(ba, cfg)))
        found = compile_mosaic_kernels(cases, "sm90")
        assert not mosaic_failures(found), "\n".join(mosaic_failures(found))


@COMPILES_ON_CPU
@pytest.mark.parametrize("gen", ["sm90", "sm80"])
@pytest.mark.parametrize(("label", "residuals"),
                         [("infer", template.Kept.NONE), ("train", template.Kept.OUTPUT)],
                         ids=["infer", "train"])
def test_sm90_d256_forward_fits_single_launch_without_spills(gen, label, residuals):
    """Rows 5272, 11965, 11990: narrow 32-col TMA epilogue staging lets d256 bf16 causal forward
    (inference and training) compile at whole-width `columns=None` (including 2-consumer FA3 ping-pong
    on sm90) with 1 primary launch, 0 spills, and physical scope verified from typed emitted/certification
    records."""
    from fla_gdn_support import KernelCase
    from kernel_compile import Report, compile_mosaic_kernels, mosaic_failures

    from dew.nn.kernels.lowered import kernel_calls

    spans = template.spans(4096, 4096, 1, causal=True)
    q = jax.ShapeDtypeStruct((4, 4096, 8, 256), jnp.bfloat16)
    k = jax.ShapeDtypeStruct((4, 4096, 2, 256), jnp.bfloat16)
    with generation.assumed(gen):
        target = generation.gpu_target()
        op, args = attention_mosaic._forward_call(q, k, k, spans, None, scale=None,
                                                  natural=attention_mosaic.NATURAL)
        ba = op.bind(*args, residuals=residuals, interpret=False)
        cfg = op.derived(ba, target)
        assert cfg is not None and op.takes(ba, cfg, target), (gen, cfg)
        whole_cfg = (op.completed(ba, dataclasses.replace(
            cfg, consumers=2, columns=None, relaunch_k=None, relaunch_consumers=None,
            relaunch_columns=None, relaunch_family=None, relaunch_q=None))
            if gen == "sm90" else None)
        if whole_cfg is not None:
            assert op.takes(ba, whole_cfg, target), (gen, whole_cfg)

        def forward(query, key, value):
            if whole_cfg is not None:
                with tuning.pinned({attention_mosaic.MosaicForward: whole_cfg}):
                    return attention_mosaic.forward(query, key, value, spans, None, residuals=residuals)
            return attention_mosaic.forward(query, key, value, spans, None, residuals=residuals)

        found = compile_mosaic_kernels([KernelCase(f"{gen}-d256-{label}", lambda: (forward, (q, k, k)))],
                                       gen)
        assert not mosaic_failures(found), "\n".join(mosaic_failures(found))
        if whole_cfg is not None:
            lowered = lowered_for(forward, (q, k, k), gen)
            primary_calls = [call for call in kernel_calls(lowered, unrun=True)
                             if call.kind == "mosaic_gpu" and call.name == "attention_forward"]
            assert len(primary_calls) == 1, primary_calls
            assert q.shape in primary_calls[0].operands and k.shape in primary_calls[0].operands, (
                primary_calls[0].operands)
        primary_name = ("attention_forward" if whole_cfg is not None or cfg.columns is None
                        else f"attention_forward_columns{cfg.columns}")
        for record in found:
            reports = [Report(**{k_name: v for k_name, v in item.items()
                                 if k_name in {f.name for f in dataclasses.fields(Report)}})
                       for item in record.get("kernels", ())]
            assert reports and all(r.spilled == 0 and r.registers is not None and r.registers <= 255
                                   for r in reports), record
            primary = [r for r in reports if r.kernel == primary_name]
            assert len(primary) == 1, (primary_name, reports)
            main = primary[0]
            assert main.failure(record["limit"], record["tmem_limit"]) is None, main
            assert main.shared is not None and 0 < main.shared <= record["limit"], main
            if isinstance(whole_cfg, attention_mosaic.Tiles):
                assert main.threads == (whole_cfg.consumers + 1) * 128, main

@COMPILES_ON_CPU
@pytest.mark.parametrize("gen", ["sm80", "sm100"])
def test_fused_dkv_width64_derives_and_compiles_spill_free_on_sm80_and_sm100(gen):
    """Row 10682: fused dK/dV at width 64 derives and compiles spill-free on sm80 and sm100."""
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    queries = 512
    spans = template.spans(queries, queries, 1, causal=True)
    query = jax.ShapeDtypeStruct((1, queries, 4, 64), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((1, queries, 2, 64), jnp.bfloat16)
    shapes = jax.tree.map(lambda array: jax.ShapeDtypeStruct(array.shape, array.dtype),
                          _residuals(query.shape))
    op_b, ba_b, derived = _gated_backward(gen, 64, accurate=False)
    assert isinstance(derived, attention_mosaic.BackwardTiles), derived
    with generation.assumed(gen):
        # The derived launch where fused (a B200's walk), else the fused one its derivation priced beside the
        # two it took (`MosaicBackward.candidates`: an A100's, row 12752).
        cfg = derived if derived.fused else next(
            config for config in op_b.candidates(ba_b, generation.gpu_target(gen)) if config.fused)
        if gen == "sm100":
            t_sm100 = generation.gpu_target(gen)
            shallow = dataclasses.replace(cfg, stages=2)
            deep = op_b._deepened(ba_b, shallow, t_sm100)
            assert deep.stages > shallow.stages and cfg.stages == deep.stages, (shallow, deep, cfg)
            assert op_b.resident(ba_b, deep, t_sm100) == op_b.resident(ba_b, shallow, t_sm100)
        case = _fast_backward_compile_case(f"{gen}-fused-d64", query, key, spans, shapes, cfg)
        found = compile_mosaic_kernels([case], gen)
        assert not mosaic_failures(found), "\n".join(mosaic_failures(found))


@COMPILES_ON_CPU
@pytest.mark.parametrize(("label", "heads", "kv_heads", "width"),
                         [("sm90-mha-d64", 16, 16, 64), ("sm90-qwen-d128", 32, 8, 128),
                          ("sm90-qwen-d256", 8, 2, 256)],
                         ids=["d64", "d128", "d256"])
def test_sm90_forward_deepens_pipeline_stages_without_losing_residency_and_compiles_spill_free(
        label, heads, kv_heads, width):
    """Row 5272: compute-bound multi-consumer sm90 forward deepens its TMA ring to `consumers + 1` stages
    when SMEM holds it without reducing SM residency, and compiles spill-free at d64, d128 and d256."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    with generation.assumed("sm90"):
        target = generation.gpu_target()
        spans = template.spans(4096, 4096, 1, causal=True)
        q = jax.ShapeDtypeStruct((2, 4096, heads, width), jnp.bfloat16)
        k = jax.ShapeDtypeStruct((2, 4096, kv_heads, width), jnp.bfloat16)
        op, args = attention_mosaic._forward_call(q, k, k, spans, None, scale=None,
                                                  natural=attention_mosaic.NATURAL)
        ba = op.bind(*args, residuals=template.Kept.NONE, interpret=False)
        cfg = op.derived(ba, target)
        assert cfg is not None and op.takes(ba, cfg, target), (label, cfg)
        # Row 12074: check explicit shallow/deepened schedule legality and residency without pinning
        # the unpinned planner's stage count.
        shallow = dataclasses.replace(cfg, stages=2)
        assert op.takes(ba, shallow, target), (label, shallow)
        deep = op._deepened(ba, shallow, target)
        assert op.takes(ba, deep, target), (label, deep)
        assert op.resident(ba, deep, target) == op.resident(ba, shallow, target), (label, shallow, deep)
        next_stage = dataclasses.replace(shallow, stages=3)
        if op.takes(ba, next_stage, target) and (
                op.resident(ba, next_stage, target) == op.resident(ba, shallow, target)):
            assert deep.stages >= next_stage.stages, (label, shallow, deep)

        def forward(query, key, value):
            return attention_mosaic.forward(query, key, value, spans, None, residuals=template.Kept.NONE)

        found = compile_mosaic_kernels([KernelCase(label, lambda: (forward, (q, k, k)))], "sm90")
        assert not mosaic_failures(found), "\n".join(mosaic_failures(found))

def test_attention_derives_product_precision_and_bf16_parts_from_algebra():
    """Unified `product.bf16_part_count` derives fp32 bf16 parts through `error.product_precision` on both
    GPU and TPU (rows 5473, 11943); `programs.triton_dot_precision` keeps IEEE fp32 off bf16 splits (row
    11878); and 16-bit presets come from `programs.own_preset` (row 10808)."""
    from dew.nn import programs
    from dew.nn.kernels.templates import product

    with generation.assumed("sm90"):
        assert product.bf16_part_count(jnp.float32, accurate=True) == 3
        assert product.bf16_part_count(jnp.float32, accurate=False) == 1
        assert product.bf16_part_count(jnp.bfloat16) == 1
        assert product.bf16_part_count(jnp.float16) == 2
        assert programs.own_preset(jnp.bfloat16) == jax.lax.DotAlgorithmPreset.BF16_BF16_F32
        assert programs.own_preset(jnp.float16) == jax.lax.DotAlgorithmPreset.F16_F16_F32
        assert (programs.triton_dot_precision(16, 16, 16, jnp.float32, ieee=True)
                == jax.lax.DotAlgorithmPreset.F32_F32_F32)
    with generation.assumed("v5e"):
        assert product.bf16_part_count(jnp.float32, accurate=True) == 3
        assert product.bf16_part_count(jnp.float32, accurate=False) == 1
        assert (programs.mosaic_tpu_dot_precision(jnp.float32, jnp.float32, 64, 64, 64)
                == jax.lax.Precision.HIGHEST)
        assert (programs.mosaic_tpu_dot_precision(jnp.bfloat16, jnp.bfloat16, 64, 64, 64)
                == jax.lax.Precision.DEFAULT)
        assert (programs.mosaic_tpu_dot_precision(jnp.float32, jnp.bfloat16, 64, 64, 64)
                == jax.lax.Precision.DEFAULT)


def test_derived_one_consumer_heights_are_compared_by_choose():
    """Row 6216: `_derived` with `choose` evaluates one-consumer `heights` (e.g. 128x32 and 64x64 on
    mma.sync) together so `cost.launch_seconds` can pick 64x64 over 128x32 instead of returning the first
    fitting height uncompared (M-FS-15 A100 hooked forwards)."""
    from dew.nn.kernels.templates import attention_mosaic_program as program

    seen: list[list[attention_mosaic.Tiles]] = []

    def choose(candidates):
        seen.append(list(candidates))
        return min(candidates, key=lambda c: (c.block_q, -c.block_k))

    family = program.layouts("sm80", jnp.bfloat16, 512)[0]

    def tiles(rows, walked, consumers):
        return attention_mosaic.Tiles(rows, walked, 2, consumers, family=family.name)

    def fits(cfg):
        return (cfg.block_q == 128 and cfg.block_k == 32) or (cfg.block_q == 64 and cfg.block_k == 64)

    derive = attention_mosaic_derivation._derived
    picked = derive(family, jnp.bfloat16, 512, 512, tiles, fits, overlapped=True,
                    walks=program.walked_extents, choose=choose)
    assert picked is not None and (picked.block_q, picked.block_k) == (64, 64), (picked, seen)
    assert any({(c.block_q, c.block_k) for c in group} >= {(128, 32), (64, 64)} for group in seen), seen


def _fast_backward_compile_case(name: str, q, k, spans, shapes, config=None):
    """Build a `KernelCase` for fast-form Mosaic backward compilation, at `config` where given (pinned), else
    the derived one."""
    from fla_gdn_support import KernelCase

    def backward(query, key, value, dout, residuals):
        with tuning.pinned({attention_mosaic.MosaicBackward: config} if config is not None else {}):
            grads = attention_mosaic.backward(query, key, value, spans, None, dout,
                                              residuals=residuals, accurate=False)
        return grads.query, grads.key, grads.value

    return KernelCase(name, lambda: (backward, (q, k, k, q, shapes)))


@COMPILES_ON_CPU
@pytest.mark.parametrize("gen", ["sm90", "sm80"])
def test_d128_fused_backward_deepens_to_3_stages_without_losing_residency(gen):
    """Row 5272: qwen3-8b d128 fused backward deepens its ring to 3 stages without losing SM residency
    on both sm90 (`wgmma.fed`) and sm80 (`mma.sync` cp.async) and compiles spill-free."""
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    with generation.assumed(gen):
        target = generation.gpu_target()
        spans = template.spans(4096, 4096, 1, causal=True)
        q = jax.ShapeDtypeStruct((2, 4096, 32, 128), jnp.bfloat16)
        k = jax.ShapeDtypeStruct((2, 4096, 8, 128), jnp.bfloat16)
        shapes = jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), _residuals(q.shape))
        op, args = attention_mosaic._backward_call(q, k, k, spans, None, q, residuals=shapes,
                                                   scale=None, natural=attention_mosaic.NATURAL,
                                                   accurate=False)
        ba = op.bind(*args, interpret=False)
        # The fused launch the derivation prices beside the two (`MosaicBackward.candidates`).
        cfg = next(config for config in op.candidates(ba, target) if config.fused)
        assert op.takes(ba, cfg, target), (gen, cfg)
        shallow = dataclasses.replace(cfg, stages=2)
        deep = op._deepened(ba, shallow, target)
        assert deep.stages > shallow.stages and cfg.stages == deep.stages, (gen, shallow, deep, cfg)
        assert op.resident(ba, deep, target) == op.resident(ba, shallow, target)
        case = _fast_backward_compile_case(f"{gen}-fused-d128-3stage", q, k, spans, shapes, cfg)
        found = compile_mosaic_kernels([case], gen)
        assert not mosaic_failures(found), "\n".join(mosaic_failures(found))

@pytest.mark.parametrize("ambient", ["sm80", "sm90"])
def test_an_h100s_fast_backward_derives_the_same_under_any_gpu_in_effect(ambient):
    """An H100's fast backward of a plain bf16 causal call 64 wide (Llama 3.2 1B's heads, 32 over 8, 4,096
    tokens) derived for an explicit H100 target under another GPU in effect (an A100's, which raises no
    registers) is the one derived under the H100 itself, and legal there: its layout order reads the
    target's multiprocessor (`attention_mosaic.producer_raised`), not the ambient one's."""
    h100 = generation.gpu_target("sm90")
    spans = template.spans(4096, 4096, 1, causal=True)
    q = jax.ShapeDtypeStruct((1, 4096, 32, 64), jnp.bfloat16)
    k = jax.ShapeDtypeStruct((1, 4096, 8, 64), jnp.bfloat16)
    shapes = jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), _residuals(q.shape))

    def derived():
        op, args = attention_mosaic._backward_call(q, k, k, spans, None, q, residuals=shapes, scale=None,
                                                   natural=attention_mosaic.NATURAL, accurate=False)
        ba = op.bind(*args, interpret=False)
        return op, ba, op.derived(ba, h100)

    # The ambient GPU's derivation first: a derivation the H100's own made first is read back.
    with generation.assumed(ambient):
        op, ba, config = derived()
    with generation.assumed("sm90"):
        _, _, own = derived()
    assert config is not None and config == own and op.legal(ba, config, h100), (config, own)


def test_overlapped_derived_weights_and_from_weights_match_derived_cotangents():
    """Row 5272: split `derived_weights` + `derived_from_weights` match `derived_cotangents` bit-for-bit
    (masked and unmasked), and `streamed_statistics(..., anchored=False)` excludes dead `chosen`."""
    from dew.nn.kernels.templates import (
        attention_backward_algebra as bwd_alg,
        attention_mosaic_fused as fused,
        attention_mosaic_program as program,
    )

    rng = jax.random.key(5272)
    k1, k2, k3, k4, k5 = jax.random.split(rng, 5)
    logits = jax.random.normal(k1, (64, 64), jnp.float32)
    products = jax.random.normal(k2, (64, 64), jnp.float32)
    peak = jnp.max(logits, axis=-1)
    total = jnp.sum(jnp.exp(logits - peak[:, None]), axis=-1)
    anchor = jax.random.normal(k3, (64,), jnp.float32)
    delta = jax.random.normal(k4, (64,), jnp.float32)
    seen = jax.random.bernoulli(k5, 0.7, (64, 64))

    for masked in (False, True):
        full_in = {"logits": logits, "products": products, "peak": peak, "total": total,
                   "anchor": anchor, "delta": delta, **({"seen": seen} if masked else {})}
        e_ref, w_ref, ds_ref = bwd_alg.derived_cotangents(
            keys_first=False, dropped=False, masked=masked).mapped(full_in)
        e_got, w_got = bwd_alg.derived_weights(keys_first=False, accurate=False).mapped(
            {"logits": logits, "peak": peak, "total": total})
        ds_in = {"weights": w_got, "products": products, "anchor": anchor, "delta": delta,
                 **({"seen": seen} if masked else {})}
        ds_got = bwd_alg.derived_from_weights(keys_first=False, masked=masked).mapped(ds_in)
        assert jnp.array_equal(e_got, e_ref)
        assert jnp.array_equal(w_got, w_ref)
        assert jnp.array_equal(ds_got, ds_ref)

    with generation.assumed("sm90"):
        family = program.bundle("sm90", jnp.bfloat16)
        cfg = program.BackwardTiles(64, 64, 2, 2, fused=True, family=family.name)
        unanchored = fused.streamed_statistics(family, cfg, 64, jnp.bfloat16, 2, reduced=0, anchored=False)
        anchored = fused.streamed_statistics(family, cfg, 64, jnp.bfloat16, 2, reduced=0, anchored=True)
        assert unanchored == 4 and anchored == 5, (unanchored, anchored)
        assert "chosen" not in program.STATISTICS_STREAM_ORDER[:unanchored]

@COMPILES_ON_CPU
@pytest.mark.parametrize("width", [64, 128])
def test_sm90_overlapped_fused_backward_compiles_spill_free(width):
    """Row 5272: sm90 overlapped fused backward (`_overlapped_fused_step`: S softmax overlapped with dP,
    dS overlapped with dV, add_dq overlapped with dK) compiles spill-free at d64 and d128."""
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    with generation.assumed("sm90"):
        spans = template.spans(4096, 4096, 1, causal=True)
        q = jax.ShapeDtypeStruct((2, 4096, 32, width), jnp.bfloat16)
        k = jax.ShapeDtypeStruct((2, 4096, 8, width), jnp.bfloat16)
        shapes = jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), _residuals(q.shape))
        case = _fast_backward_compile_case(f"sm90-overlapped-fused-d{width}", q, k, spans, shapes)
        found = compile_mosaic_kernels([case], "sm90")
        assert not mosaic_failures(found), "\n".join(mosaic_failures(found))


@COMPILES_ON_CPU
def test_sm90_overlapped_coop_backward_compiles_spill_free_for_d256():
    """Row 5272: sm90 overlapped shared-keys backward (`attention_dkv_shared`, qwen3.5-0.8b d256) compiles
    spill-free with `_tile_scores` S/dP overlap and dead `chosen` excluded."""
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    from dew.nn.kernels.templates import (
        attention_mosaic_coop as coop,
        attention_mosaic_fused as fused,
        attention_mosaic_program as program,
    )

    with generation.assumed("sm90"):
        target = generation.gpu_target()
        spans = template.spans(4096, 4096, 1, causal=True)
        q = jax.ShapeDtypeStruct((4, 4096, 8, 256), jnp.bfloat16)
        k = jax.ShapeDtypeStruct((4, 4096, 2, 256), jnp.bfloat16)
        shapes = jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), _residuals(q.shape))
        op, args = attention_mosaic._backward_call(q, k, k, spans, None, q, shapes, scale=1.0,
                                                   natural=attention_mosaic.NATURAL, accurate=False)
        cfg = program.BackwardTiles(64, 64, 2, 2, fused=True, family="wgmma.fed", shared_keys=True)
        assert op.takes(op.bind(*args, interpret=False), cfg, target), cfg
        family = program.bundle_of(cfg, jnp.bfloat16)
        streamed = fused.streamed_statistics(family, cfg, 256, jnp.bfloat16, cfg.stages, reduced=0,
                                             allocations_of=coop._coop_allocations, anchored=False)
        assert streamed > 0 and "chosen" not in program.streamed_statistics_order(streamed, anchored=False)
        def backward(query, key, value, dout, residuals):
            with tuning.pinned({attention_mosaic.MosaicBackward: cfg}):
                grads = attention_mosaic.backward(query, key, value, spans, None, dout,
                                                  residuals=residuals, scale=1.0, accurate=False)
            return grads.query, grads.key, grads.value
        from fla_gdn_support import KernelCase
        found = compile_mosaic_kernels([KernelCase("sm90-overlapped-coop-d256",
                                                   lambda: (backward, (q, k, k, q, shapes)))], "sm90")
        assert not mosaic_failures(found), "\n".join(mosaic_failures(found))
        assert any(kernel["kernel"] == "attention_dkv_shared"
                   for kernel in found[0].get("kernels", ())), found


@COMPILES_ON_CPU
def test_sm90_overlapped_keys_first_dkv_compiles_spill_free_deterministic(with_deterministic_ops):
    """Row 5272: sm90 deterministic two-launch keys-first dK/dV (`_splits_scores`) compiles spill-free at
    d128 with unit prescale."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    with generation.assumed("sm90"):
        spans = template.spans(4096, 4096, 1, causal=True)
        q = jax.ShapeDtypeStruct((2, 4096, 32, 128), jnp.bfloat16)
        k = jax.ShapeDtypeStruct((2, 4096, 8, 128), jnp.bfloat16)
        shapes = jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), _residuals(q.shape))
        def backward(query, key, value, dout, residuals):
            grads = attention_mosaic.backward(query, key, value, spans, None, dout,
                                              residuals=residuals, scale=None, accurate=False)
            return grads.query, grads.key, grads.value
        found = compile_mosaic_kernels([KernelCase("sm90-overlapped-dkv-det-d128",
                                                   lambda: (backward, (q, k, k, q, shapes)))], "sm90")
        assert not mosaic_failures(found), "\n".join(mosaic_failures(found))
        assert any(kernel["kernel"] == "attention_dkv"
                   for kernel in found[0].get("kernels", ())), found


@COMPILES_ON_CPU
def test_decode_deepens_to_3_stages_without_losing_residency():
    """Row 5274: long-context decode (B8 K4096 d128) compiles spill-free at pinned 2- and 3-stage configs
    with preserved SM residency on sm90."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    from dew.nn.kernels.templates import attention_decode

    with generation.assumed("sm90"):
        target = generation.gpu_target()
        spans = template.spans(1, 4096, 8, causal=False)
        qd = jax.ShapeDtypeStruct((8, 1, 32, 128), jnp.bfloat16)
        kd = jax.ShapeDtypeStruct((8, 4096, 8, 128), jnp.bfloat16)
        work, rows, row_spans, whole = attention_decode.planned(
            jnp.zeros(qd.shape, qd.dtype), kd, kd, spans, template.Hooks())
        assert work is not None, work
        base = work["plain"].config
        cfg2 = dataclasses.replace(base, stages=2)
        cfg3 = dataclasses.replace(base, stages=3)
        op, args = attention_mosaic._forward_call(
            rows, kd, kd, row_spans, whole, scale=None, natural=False,
            out_dtype=jnp.float32 if work["plain"].slices > 1 else None, masked=True,
            splits=work["plain"].slices, subtracted=True)
        kept = template.Kept.OUTPUT if work["plain"].slices > 1 else template.Kept.NONE
        ba = op.bind(*args, residuals=kept, interpret=False)
        assert op.takes(ba, base, target) and op.takes(ba, cfg2, target) and op.takes(ba, cfg3, target)
        assert op.resident(ba, cfg3, target) == op.resident(ba, cfg2, target)
        assert base.stages == cfg3.stages, base
        def forward_pinned(cfg):
            def run(*operands):
                with tuning.pinned({attention_mosaic.MosaicForward: cfg}):
                    return op(*operands, residuals=kept, interpret=False)
            return run
        cases = [
            KernelCase("sm90-dec-b8-d128-2stage", lambda: (forward_pinned(cfg2), args)),
            KernelCase("sm90-dec-b8-d128-3stage", lambda: (forward_pinned(cfg3), args)),
        ]
        found = compile_mosaic_kernels(cases, "sm90")
        assert not mosaic_failures(found), "\n".join(mosaic_failures(found))


@COMPILES_ON_CPU
@pytest.mark.parametrize("gen", ["sm90", "sm100"])
def test_forward_registers_charges_prescale_only_when_non_unit(gen):
    """Rows 5272, 11789: `ForwardRegisters.at` prices the pre-walk prescale fragment only when
    `spec.prescale != 1.0`, and wide non-unit prescale forward compiles spill-free on sm90 and sm100."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    from dew.nn.kernels.templates import attention_mosaic_program as program

    with generation.assumed(gen):
        spans = template.spans(512, 512, 1, causal=True)
        q = jax.ShapeDtypeStruct((1, 512, 4, 256), jnp.bfloat16)
        k = jax.ShapeDtypeStruct((1, 512, 2, 256), jnp.bfloat16)
        op_folded, _ = attention_mosaic._forward_call(
            q, k, k, spans, None, scale=None, natural=False)
        op_scaled, _ = attention_mosaic._forward_call(
            q, k, k, spans, None, scale=None, natural=True)
        op_unit, _ = attention_mosaic._forward_call(
            q, k, k, spans, None, scale=1.0, natural=True)
        reg_folded = op_folded.registered(q, k, (), template.Kept.NONE)
        reg_scaled = op_scaled.registered(q, k, (), template.Kept.NONE)
        reg_unit = op_unit.registered(q, k, (), template.Kept.NONE)
        assert not reg_folded.prescaled and reg_folded.rewalk_prescaled
        assert reg_scaled.prescaled and not reg_unit.prescaled
        family = program.bundle(gen, jnp.bfloat16)
        wide_reg_folded = dataclasses.replace(reg_folded, width=576, value_width=128)
        wide_reg_scaled = dataclasses.replace(reg_scaled, width=576, value_width=128)
        wide_reg_unit = dataclasses.replace(reg_unit, width=576, value_width=128)
        tiles = program.Tiles(64, 64, 2, 1, family=family.name)
        if family.budgets_registers and program.query_swizzle(family) is None:
            scaled_regs = wide_reg_scaled.at(family, tiles, peaks=True)
            folded_regs = wide_reg_folded.at(family, tiles, peaks=True)
            unit_regs = wide_reg_unit.at(family, tiles, peaks=True)
            assert scaled_regs is not None and folded_regs is not None and unit_regs is not None
            assert scaled_regs > unit_regs and folded_regs == unit_regs
        def forward_scaled(query, key, value):
            return attention_mosaic.forward(query, key, value, spans, None, residuals=template.Kept.NONE,
                                            natural=True)
        found = compile_mosaic_kernels(
            [KernelCase(f"{gen}-prescaled-d256", lambda: (forward_scaled, (q, k, k)))], gen)
        assert not mosaic_failures(found), "\n".join(mosaic_failures(found))


def test_paged_forward_emitted_work_counts_table_bytes():
    """Row 11737: a public paged forward's emitted `Work.device_bytes` equals `stored` of Q/K/V/table."""
    from dew.nn import programs
    from dew.nn.kernels.templates import attention_decode, cost, emission

    rows, keys, page, width = 16, 4096, 16, 64
    q = jax.ShapeDtypeStruct((rows, 1, 32, width), jnp.bfloat16)
    pool = jax.ShapeDtypeStruct((8, rows * keys // page, page, width), jnp.bfloat16)
    table = jax.ShapeDtypeStruct((rows, keys // page), jnp.int32)
    spans = template.spans(1, keys, rows)
    with generation.assumed("sm90"):
        traced = jax.make_jaxpr(lambda q, k, v, t: attention_decode.attend(q, k, v, spans, table=t))(
            q, pool, pool, table).jaxpr
    eq, work = next((eq, w) for eq in programs.equations(traced)
                    if eq.primitive in programs.NATIVE_KERNELS and (w := emission.work_of(eq)) is not None)
    q_a, k_a, v_a = (var.aval for var in eq.invars[:3])
    t_a = next(var.aval for var in eq.invars[3:] if var.aval.dtype == jnp.int32 and var.aval.ndim == 2)
    assert work.device_bytes == cost.stored(q_a, k_a, v_a, t_a)


def test_step_live_tiles_tracks_hooks_dropout_accuracy_and_nested_subjaxprs():
    """`step_live_tiles` traces the emitted step at its spec (score hook, dropout, accurate form, dtype)
    and counts nested sub-jaxpr transients via `programs.live` (rows 11652, 11993, 11999)."""
    from dew.nn.kernels.templates import attention_algebra as algebra
    from dew.nn.kernels.templates.attention_mosaic_scores import step_live_tiles

    causal_spec = algebra.Forward("causal", 128, ids=False, scale=1.0 / 8.0)

    def nested_hook(logits, _q, _k, bias):
        def branch(x):
            a = jnp.sin(x)
            b = jnp.cos(x)
            c = jnp.tanh(x + bias)
            return (a + b) * c + bias
        return jax.lax.cond(jnp.any(logits > 0), branch, lambda x: x + bias, logits)

    plain_fwd = len(step_live_tiles(64, 64, spec=causal_spec))
    hooked_fwd = len(step_live_tiles(64, 64, spec=causal_spec, score_mod=nested_hook,
                                     score_types=((template.Kind.TILE, "float32"),)))
    dropped_fwd = len(step_live_tiles(64, 64, spec=causal_spec, dropped=True))
    assert hooked_fwd > plain_fwd, (hooked_fwd, plain_fwd)
    assert dropped_fwd >= plain_fwd, (dropped_fwd, plain_fwd)

    plain_bwd = len(step_live_tiles(64, 64, spec=causal_spec, backward=True))
    acc_bwd = len(step_live_tiles(64, 64, spec=causal_spec, backward=True, accurate=True))
    hooked_bwd = len(step_live_tiles(64, 64, spec=causal_spec, backward=True, score_mod=nested_hook,
                                     score_types=((template.Kind.TILE, "float32"),)))
    assert acc_bwd >= plain_bwd, (acc_bwd, plain_bwd)
    assert hooked_bwd > plain_bwd, (hooked_bwd, plain_bwd)

def test_wide_epilogue_chunks_tcgen05_chunks_and_prices_masked_rewalks():
    """Rows 11852, 11853: `BlockPlan.epilogue_chunk` chunks wide (>128) unmasked hookless `tcgen05.chunks`
    array outputs while leaving `tcgen05.rows` (`places_on_role`) unchunked, and `_epilogue_registers` /
    `MosaicForward._relaunched` price wide in-place masked WGMMA rewalks at whole-width finish + preserve."""
    from dew.nn.kernels.templates import attention_mosaic_program as program

    plan = program.BlockPlan(split=False, out_dtype=jnp.bfloat16, output_hook=False)
    with generation.assumed("sm100"):
        chunks_family = program.bundle_of(attention_mosaic.Tiles(128, 64, 2, 1, family="tcgen05.chunks"),
                                          jnp.bfloat16)
        rows_family = program.bundle_of(attention_mosaic.Tiles(128, 64, 2, 1, family="tcgen05.rows"),
                                        jnp.bfloat16)
        assert plan.epilogue_chunk(chunks_family, 256, jnp.bfloat16) > 0
        assert plan.epilogue_chunk(rows_family, 256, jnp.bfloat16) == 0

    with generation.assumed("sm90"):
        wgmma = program.bundle_of(attention_mosaic.Tiles(64, 64, 2, 1, family="wgmma"), jnp.bfloat16)
        unmasked = attention_mosaic_registers._epilogue_registers(
            wgmma, [(64, 256, jnp.float32)], 64, 256, jnp.bfloat16, residuals=False, plan=plan, masked=False)
        masked = attention_mosaic_registers._epilogue_registers(
            wgmma, [(64, 256, jnp.float32)], 64, 256, jnp.bfloat16, residuals=False, plan=plan, masked=True)
        assert masked > unmasked > 0, (masked, unmasked)
        from dew.nn.kernels.templates import attention_mosaic_forward
        q = jax.ShapeDtypeStruct((4, 4096, 8, 256), jnp.bfloat16)
        assert attention_mosaic_forward.rewalks_in_place(wgmma, q, q)
        spans = template.spans(4096, 4096, 4, causal=True)
        op, args = attention_mosaic._forward_call(q, q, q, spans, None, scale=None,
                                                  natural=attention_mosaic.NATURAL)
        ba = op.bind(*args, residuals=template.Kept.NONE, interpret=False)
        target = generation.gpu_target()
        cfg = op.derived(ba, target)
        assert cfg is not None and op.takes(ba, cfg, target), cfg
        # Wide (>128) in-place masked WGMMA rewalk exceeds 255 regs; column-blocked rewalk fits.
        wide_rewalk = dataclasses.replace(cfg, block_k=64, consumers=1, natural_form=True, columns=None)
        blocked_rewalk = dataclasses.replace(wide_rewalk, columns=128)
        assert not op.within(op.registers(ba, wide_rewalk, masked=True), wide_rewalk)
        assert op.within(op.registers(ba, blocked_rewalk, masked=True), blocked_rewalk)
        # Row 11881 / 11906: two-launch backward `_deepened` keeps residency on explicit stage pin.
        qb = jax.ShapeDtypeStruct((2, 4096, 8, 64), jnp.bfloat16)
        vb = jax.ShapeDtypeStruct((2, 4096, 8, 128), jnp.bfloat16)
        res_b = jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), _residuals(vb.shape))
        bwd_op, bwd_args = attention_mosaic._backward_call(
            qb, qb, vb, spans, None, vb, residuals=res_b, scale=None,
            natural=attention_mosaic.NATURAL, accurate=False)
        bwd_ba = bwd_op.bind(*bwd_args, interpret=False)
        bwd_cfg = bwd_op.derived(bwd_ba, target)
        assert bwd_cfg is not None and not bwd_cfg.fused and bwd_op.takes(bwd_ba, bwd_cfg, target), bwd_cfg
        # Row 11991: pin explicit shallow/deep configs for residency comparison instead of pinning depth rule.
        shallow_bwd = dataclasses.replace(bwd_cfg, stages=2)
        deep_bwd = dataclasses.replace(bwd_cfg, stages=3)
        assert bwd_op.takes(bwd_ba, shallow_bwd, target), shallow_bwd
        deepened = bwd_op._deepened(bwd_ba, shallow_bwd, target)
        assert bwd_op.takes(bwd_ba, deepened, target), deepened
        assert bwd_op.resident(bwd_ba, deepened, target) == bwd_op.resident(bwd_ba, shallow_bwd, target)
        if bwd_op.takes(bwd_ba, deep_bwd, target) and (
                bwd_op.resident(bwd_ba, deep_bwd, target) == bwd_op.resident(bwd_ba, shallow_bwd, target)):
            assert deepened.stages >= deep_bwd.stages, (deepened, deep_bwd)

@COMPILES_ON_CPU
@pytest.mark.usefixtures("without_deterministic_ops")
@pytest.mark.parametrize("gen", ["sm80", "sm86", "sm120"])
def test_mma_sync_fused_32x32_backward_derives_and_compiles_spill_free(gen):
    """Row 5272: mma.sync (sm80/sm86/sm120) derives and compiles one-launch 32x32 (`m_warps=2`) fused
    backward for d128 bf16 causal spill-free via `MmaIssuer.rows_loaded`, `_store_staged`, and `left_of`."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    spans = template.spans(512, 512, 1, causal=True)
    q = jax.ShapeDtypeStruct((1, 512, 4, 128), jnp.bfloat16)
    residuals = _residuals(q.shape)
    shapes = jax.tree.map(lambda array: jax.ShapeDtypeStruct(array.shape, array.dtype), residuals)
    with generation.assumed(gen):
        op, args = attention_mosaic._backward_call(
            q, q, q, spans, None, q, residuals=shapes, scale=None,
            natural=attention_mosaic.NATURAL, accurate=False)
        ba = op.bind(*args, interpret=False)
        target = generation.gpu_target()
        cfg = next(config for config in op.candidates(ba, target) if config.fused)
        pin = attention_mosaic.BackwardTiles(32, 32, 2, 1, fused=True, family=cfg.family)
        assert op.takes(ba, pin, target), (gen, pin)
        bad = attention_mosaic.BackwardTiles(96, 64, 2, 1, fused=True, family=cfg.family)
        assert not op.legal(ba, bad, target), (gen, bad)
        assert not op.legal(ba, dataclasses.replace(bad, block_q=64, block_k=96), target), (gen, bad)

    def backward_default(query, key, value, dout, res):
        grads = attention_mosaic.backward(query, key, value, spans, None, dout, residuals=res)
        return grads.query, grads.key, grads.value

    def backward_32x32(query, key, value, dout, res):
        with tuning.pinned({attention_mosaic.MosaicBackward: pin}):
            grads = attention_mosaic.backward(query, key, value, spans, None, dout, residuals=res)
        return grads.query, grads.key, grads.value

    found = compile_mosaic_kernels([
        KernelCase(f"{gen}-fused-default-d128", lambda: (backward_default, (q, q, q, q, shapes))),
        KernelCase(f"{gen}-fused-32x32-d128", lambda: (backward_32x32, (q, q, q, q, shapes))),
    ], gen)
    assert not mosaic_failures(found), "\n".join(mosaic_failures(found))
    for record in found:
        bwd = [k for k in record.get("kernels", ()) if "backward" in k["kernel"] or "dkv" in k["kernel"]]
        assert len(bwd) == 1, record


@COMPILES_ON_CPU
@pytest.mark.usefixtures("without_deterministic_ops")
def test_sm90_d256_gqa_forward_and_shared_keys_backward_compile_spill_free():
    """sm90 d256 GQA (16:4) causal forward and shared-keys cooperative backward derive and compile
    without spills or wasted step-downs."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    spans = template.spans(4096, 4096, 1, causal=True)
    q = jax.ShapeDtypeStruct((2, 4096, 16, 256), jnp.bfloat16)
    k = jax.ShapeDtypeStruct((2, 4096, 4, 256), jnp.bfloat16)
    res = jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), _residuals(q.shape))

    def fwd(query, key, value):
        return attention_mosaic.forward(query, key, value, spans, None, residuals=template.Kept.OUTPUT)

    found = compile_mosaic_kernels([
        KernelCase("sm90-d256-gqa-fwd", lambda: (fwd, (q, k, k))),
        _fast_backward_compile_case("sm90-d256-gqa-bwd", q, k, spans, res),
    ], "sm90")
    assert not mosaic_failures(found), "\n".join(mosaic_failures(found))


PAIRED = attention_mosaic.BackwardTiles(64, 64, 2, 2, fused=True, family="mma.sync")
"""An A100's fast backward in one launch on two consumers sharing one cp.async pipeline (`mosaic_sm80`'s
`shared_consumers`), each its own 64 keys of a 128-key block, a query tile's dQ summed between them before
its one add (`attention_mosaic_fused.dq_summed`)."""


@INTERPRETS
@pytest.mark.parametrize("call", CALLS)
def test_an_a100s_one_launch_on_two_paired_consumers_holds_check_1_interpreted(call,
                                                                                 without_deterministic_ops):
    """`PAIRED` over every query tile the block's readers read (a tile none of a consumer's keys sees
    masked whole, the consumers' segments the same, `attention_mosaic_fused._walked_run`): within check 1 of
    the reference from float64 over grouped heads, causal and full, under the interpreter."""
    _check_backward(call, SHAPES["interpreted"], form=None, config=PAIRED, interpret=True, width=64,
                    target=generation.gpu_target("sm80"))


def test_an_a100s_two_consumers_are_legal_on_its_paired_launch_alone():
    """An A100's two consumers past its derivations' one (`mosaic_sm80`'s `shared_consumers`) are legal on
    the paired launch (`PAIRED`: queries first, each consumer its own keys, their dQ summed) and on no launch
    whose consumers share a key block (`shared_keys`, `_coop_launch`, which loads its block by one)."""
    with generation.assumed("sm80"):
        target = generation.gpu_target()
        op, ba = _backward_call((1, 512, 8, 2), 64, accurate=False)
        assert op.legal(ba, PAIRED, target)
        assert not op.legal(ba, dataclasses.replace(PAIRED, shared_keys=True), target)


@COMPILES_ON_CPU
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "full"])
def test_an_a100s_fast_backward_on_two_paired_consumers_compiles(causal, without_deterministic_ops):
    """An A100's fast backward at Llama 3.2 1B's head width (8 query heads over 2, 512 tokens, causal and
    full) at its derived tiles on two paired consumers (`PAIRED`'s) is legal and lowers and compiles on a
    compile-only A100 within its resources: the collective pipeline's two warpgroups walking the same
    segments, their staged scores and the sum's parts and barriers in one block."""
    with generation.assumed("sm80"):
        target = generation.gpu_target()
        op, ba = _backward_call((1, 512, 8, 2), 64, accurate=False, causal=causal)
        fused = next(config for config in op.candidates(ba, target) if config.fused)
        config = dataclasses.replace(fused, consumers=PAIRED.consumers)
        assert op.legal(ba, config, target), config
        case = KernelCase(f"paired backward {config}", lambda: op.lowering(ba, config))
        found = compile_mosaic_kernels([case], "sm80")
    assert not (failed := mosaic_failures(found)), "\n".join(failed)
