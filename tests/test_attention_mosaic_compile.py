"""The attention template's Mosaic GPU mainloop compiled at the tiles its models decide
(dew.nn.kernels.templates.attention_mosaic): each launch, forward, backward and decode step, compiled by XLA's
CUDA backend for a compile-only A100 and H100 on the CPU (tests/kernel_compile.py), within its block, its
registers as its model states them and spill-free; and an A100's launches counting the shared memory their
lowering asks for. The mainloop's values are tests/test_attention_mosaic.py's."""

import dataclasses

import jax
import jax.numpy as jnp
import pytest
from generation_support import COMPILES_ON_CPU, INTERPRETS
from test_attention_mosaic import BACKWARD_HOOKS, SHAPES, _check_backward, _residuals
from test_attention_triton import SCORE_HOOKS, score_hooks

from dew.nn.kernels import generation
from dew.nn.kernels.templates import (
    attention as template,
    attention_mosaic,
    attention_mosaic_program,
    bundles,
    resources,
)

pytestmark = pytest.mark.kernel


GATED = ("sm90", "sm80")
"""The GPUs whose launches the resource gates compile (`attention_mosaic_program.qualified`, a family whose
consumers issue their products): an H100's wgmma and an A100's mma.sync staged by cp.async. Consumer
Blackwell's mma.sync fed by TMA (`bundles.MmaFed`) runs the mainloop too, but derives no tile within its
99 KB block yet (every sm120 case None at 3569816c26, lean1b)."""

BACKWARD_GATE = [(generation_name, width, accurate, "none") for generation_name in GATED
                 for width in (64, 128, 256) for accurate in (False, True)]
BACKWARD_GATE += [(generation_name, width, accurate, hooks) for generation_name in GATED
                  for width in (64, 128) for accurate in (False, True)
                  for hooks in BACKWARD_HOOKS if hooks != "none"]
BACKWARD_GATE += [("sm90", 256, False, "softcap")]
"""The backward resource gate's cases: each GPU whose family the mainloop qualifies (`GATED`), a causal
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


FORWARD_GATE = [(generation_name, width, training, "none") for generation_name in GATED
                for width in (64, 128, 256)
                for training in (template.Kept.NONE, template.Kept.OUTPUT, template.Kept.SPLIT)]
FORWARD_GATE += [(generation_name, width, training, hooks) for generation_name in GATED for width in (64, 128)
                 for training in (template.Kept.NONE, template.Kept.OUTPUT, template.Kept.SPLIT)
                 for hooks in BACKWARD_HOOKS if hooks != "none"]
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
    in two column halves, within the A100's block as its model sizes it, spilling nothing, every launch's
    model ptxas's, lowered and compiled for an sm80 on this CPU."""
    import kernel_compile
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    op, ba, _ = _gated_backward("sm80", 256, accurate)
    with generation.assumed("sm80"):
        target = generation.gpu_target()
        assert op.staged_bytes(ba, config, target) <= target.shared_limit
        models = _backward_models(op, ba, config)
        found = compile_mosaic_kernels([KernelCase("backward", lambda: op.lowering(ba, config), models)],
                                       "sm80")
    assert not mosaic_failures(found), "\n".join(mosaic_failures(found))
    assert set(models) <= kernel_compile.compiled_kernels(found[0]), (set(models), found[0].get("kernels"))


def _forward_kernels(op, ba, config) -> dict:
    """The forward's register model by each name a launch of it takes (`attention_mosaic_forward._launch`:
    `attention_{forward, training, peaks}`, then `_natural`, `_scaled`, `_kept`, `_columns{width}` and
    `_split` where it is so), each the op's (`MosaicForward.registers`) at its config: a scaled launch's the
    scaled op's at the config it derives (`MosaicForward.scaled`, its own registers and shared memory), a
    cache's walk's its own (`MosaicForward.cached`, scaled in its kernel), a peak walk's its own, a natural
    launch's at its own tiles (`Tiles.relaunch`), a launch over a block of the values' columns at that
    block's width."""
    if op.scales_in_kernel(ba):
        # A cache's walk scales in its own kernel at its own config (`MosaicForward.cached`).
        variants = (("_scaled", op, config),)
    else:
        scaled_op = dataclasses.replace(op, scaled=True)
        variants = (("", op, config), ("_scaled", scaled_op, scaled_op.derived(ba, generation.gpu_target())))
    found = {}
    for scaled, of, at in variants:
        if at is None:
            continue
        family = attention_mosaic_program.bundle_of(at, jnp.bfloat16)
        for natural, tiles in (("", at), ("_natural", at.relaunch)):
            for kind in ("forward", "training", "peaks"):
                for columns in (None, 16, 32, 64, 128, 256):
                    registers = of.registers(ba, tiles, peaks=kind == "peaks", columns=columns)
                    model = resources.register_model(family, tiles.consumers, registers)
                    named = "" if columns is None else f"_columns{columns}"
                    found.update({f"attention_{kind}{natural}{scaled}{kept}{named}{split}": model
                                  for kept in ("", "_kept") for split in ("", "_split")})
    return found


DECODE_GATE = [(generation_name, width, stored) for generation_name in GATED for width in (64, 128)
               for stored in ("bfloat16", "int8", "float32")
               if stored != "float32" or bundles.of(generation_name, jnp.bfloat16).copies_in_pieces]
"""The decode gate's cases: each GPU whose family the mainloop qualifies, both widths, each cache; a float32
query's parts where its family loads a block in pieces (`mosaic.Transport.copies_in_pieces`: not cp.async's,
where an fp32 decode takes XLA by legality)."""


@COMPILES_ON_CPU
@pytest.mark.parametrize(("generation_name", "width", "stored"), DECODE_GATE)
def test_a_decode_step_compiles_spill_free_at_its_plans_tiles(generation_name, width, stored):
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

    rows, keys = 2, 4096
    query = jax.ShapeDtypeStruct((rows, 1, 32, width), jnp.float32 if stored == "float32" else jnp.bfloat16)
    key = jax.ShapeDtypeStruct((rows, keys, 8, width), jnp.dtype(stored))
    scale = jax.ShapeDtypeStruct((rows, 8, keys), jnp.float32)
    quantized = stored == "int8"
    spans = template.spans(1, keys, rows, lengths=jnp.asarray([keys, keys - 77], jnp.int32))

    def decode(query, key, value, *scales):
        return attention_decode.attend(query, key, value, spans, scales=scales or None)

    operands = (query, key, key, *((scale, scale) if quantized else ()))
    with generation.assumed(generation_name):
        # The plan's op over the packed rows, its launches' models (`_forward_kernels`).
        zeros = [jnp.zeros(array.shape, array.dtype) for array in operands]
        scales = tuple(zeros[3:]) or None
        rows, row_spans, _ = attention_decode.packed(zeros[0], spans, key.shape[2])
        op, args = attention_mosaic._forward_call(rows, *zeros[1:3], row_spans, None, scale=None,
                                                  natural=False, out_dtype=jnp.float32, masked=True,
                                                  scales=scales)
        ba = op.bind(*args, residuals=template.Kept.OUTPUT, interpret=False)
        config = attention_decode.plan(rows, *zeros[1:3], row_spans, None, scale=None,
                                       target=generation.gpu_target(), scales=scales).config
        found = compile_mosaic_kernels(
            [KernelCase("decode", lambda: (decode, operands), _forward_kernels(op, ba, config))],
            generation_name)
    assert not mosaic_failures(found), "\n".join(mosaic_failures(found))


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
        # The plan's op over the packed rows, its launches' models (`_forward_kernels`).
        zeros = [jnp.zeros(array.shape, array.dtype) for array in (query, pool, table)]
        rows_of, row_spans, _ = attention_decode.packed(zeros[0], spans, pool.shape[0])
        op, args = attention_mosaic._forward_call(rows_of, zeros[1], zeros[1], row_spans, None, scale=None,
                                                  natural=False, out_dtype=jnp.float32, masked=True,
                                                  table=zeros[2])
        ba = op.bind(*args, residuals=template.Kept.OUTPUT, interpret=False)
        config = attention_decode.plan(rows_of, zeros[1], zeros[1], row_spans, None, scale=None,
                                       target=generation.gpu_target(), table=zeros[2]).config
        case = KernelCase("paged decode", lambda: (decode, (query, pool, pool, table)),
                          _forward_kernels(op, ba, config))
        found = compile_mosaic_kernels([case], generation_name)
    assert not mosaic_failures(found), "\n".join(mosaic_failures(found))


@COMPILES_ON_CPU
@pytest.mark.parametrize(("generation_name", "width", "training", "hooks", "dtype"), FORWARD_GATE)
def test_the_forward_compiles_spill_free_as_its_model_decides(generation_name, width, training, hooks,
                                                              dtype):
    """The forward, the training forward where `training`, at its derived tiles of a causal call `width`
    wide under the score hook `hooks` names, in `dtype`, lowered and compiled by ptxas for `generation_name`
    on this CPU (tests/kernel_compile.py): every launch within its shared memory, spilling nothing, and its
    register model's decision ptxas's (`mosaic_failures`)."""
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
        found = compile_mosaic_kernels(
            [KernelCase("forward", lambda: (forward, (query, key, key)), _forward_kernels(op, ba, config))],
            generation_name)
    assert not mosaic_failures(found), "\n".join(mosaic_failures(found))


@COMPILES_ON_CPU
@pytest.mark.parametrize("width", [64, 128])
@pytest.mark.parametrize("generation_name", ["sm90"])
def test_an_fp32_forward_compiles_spill_free_as_its_model_decides(generation_name, width):
    """An fp32 call's inference forward (`MosaicForward.parts`: its operands bf16 parts) at its derived tiles
    of a causal call `width` wide, lowered and compiled by ptxas for `generation_name` on this CPU: every
    launch within its shared memory, spilling nothing, its register model's decision ptxas's."""
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
        found = compile_mosaic_kernels(
            [KernelCase("forward", lambda: (forward, (query, key, key)), _forward_kernels(op, ba, config))],
            generation_name)
    assert not mosaic_failures(found), "\n".join(mosaic_failures(found))


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
    lowered and compiled by ptxas for an H100 on this CPU: within its shared memory, spilling nothing, its
    register model's decision ptxas's."""
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
    import kernel_compile
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
    assert config is not None, f"the Mosaic backward derives no config {width} wide on {generation_name}"
    with generation.assumed(generation_name):
        models = _backward_models(op, ba, config)
        shapes = jax.tree.map(lambda array: jax.ShapeDtypeStruct(array.shape, array.dtype), residuals)
        found = compile_mosaic_kernels(
            [KernelCase("backward", lambda: (backward, (query, key, key, query, shapes)), models)],
            generation_name)
    assert not mosaic_failures(found), "\n".join(mosaic_failures(found))
    # Every launch's model names a kernel it compiled: a launch renamed past its model is compared with none.
    assert set(models) <= kernel_compile.compiled_kernels(found[0]), (set(models), found[0].get("kernels"))


def _backward_models(op, ba, config) -> dict:
    """Each backward launch's register model by its kernel's name (`attention_dq`, `attention_dkv`, a width
    share's `_columnsN`, the one launch's `attention_dkv_fused`, or `attention_dkv_shared` over shared
    keys), at its own layout and consumers (`BackwardTiles.dkv`)."""
    dq, dkv = op._launch_registers(ba, config)
    if config.fused:
        family = attention_mosaic_program.bundle_of(config, jnp.bfloat16)
        name = "attention_dkv_shared" if config.shared_keys else "attention_dkv_fused"
        return {name: resources.register_model(family, config.consumers, dkv)}
    models = {}
    for kind, tiles, registers in (("dq", config, dq), ("dkv", config.dkv, dkv)):
        family = attention_mosaic_program.bundle_of(tiles, jnp.bfloat16)
        model = resources.register_model(family, tiles.consumers, registers)
        names = ([f"attention_{kind}"] if config.columns == 1
                 else [f"attention_{kind}_columns{share}" for share in range(config.columns)])
        models.update(dict.fromkeys(names, model))
    return models


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
        found = compile_mosaic_kernels([KernelCase("backward", lambda: op.lowering(ba, config),
                                                   _backward_models(op, ba, config))], "sm90")
    assert not mosaic_failures(found), "\n".join(mosaic_failures(found))
