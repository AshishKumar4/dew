"""The Mosaic attention mainloop's walks placed on tcgen05 (`placed`): a walk whose declared products a role
of their own issues, a step ahead of the warpgroups that read them, their accumulators in tensor memory
(FlashAttention 4's structure): the forward's rows (`placed.ROWS`, row 5368) and the backward's launches
keys first and queries first (`placed.KEYS`, `placed.QUERIES`, row 4839), each pinned, under the interpreter
for a B200 or lowered for one."""

import dataclasses

import jax
import jax.numpy as jnp
import pytest
from generation_support import INTERPRETS
from reference_error import assert_as_exact
from test_attention_mosaic import (
    CALLS,
    SEEDS,
    _check_forward,
    _check_spans,
    _counted,
    _device,
    _drawn,
    _pinned,
)
from test_attention_template import CASES, arguments, geometry as core_geometry

from dew.nn.kernels import generation
from dew.nn.kernels.templates import attention as template, attention_mosaic, mosaic, placed

pytestmark = pytest.mark.kernel


ROWS_TILES = attention_mosaic.Tiles(128, 64, 2, 1, family="tcgen05.rows")
"""tcgen05's rows' one tile (`placed.ROWS`): a product's 128 rows, a row a consumer thread, 64 keys
a step."""


@INTERPRETS
@pytest.mark.parametrize("width", [64, 128])
@pytest.mark.parametrize("natural", [True, False], ids=["natural", "folded"])
@pytest.mark.parametrize("call", ["full", "causal"])
def test_the_forward_on_tcgen05s_rows_holds_check_1_interpreted(call, natural, width):
    """The forward on a B200's tcgen05 rows (`placed.ROWS`, FlashAttention 4's structure: its walk's
    declared products issued by a warp of their own a step ahead of the softmax warpgroup, a row a thread in
    tensor memory's own layout, the weights written there as the weighted values' left operand, the sums
    kept there across the walk, rescaled in place and finished a block of columns at a time), its logits
    natural or folded (its fold's guard rewalking on the chunks, `Tiles.relaunch_family`), over 192 tokens
    (its query and key tiles padded), under the interpreter, which models tensor memory: on par with the
    reference (check 1)."""
    _check_forward(call, jnp.bfloat16, (1, 192, 2, 1), natural=natural, interpret=True, width=width,
                   value_width=width, tiles=ROWS_TILES, target=generation.gpu_target("sm100"))


@INTERPRETS
def test_the_forward_on_tcgen05s_rows_over_narrow_value_blocks_holds_check_1_interpreted():
    """The forward on a B200's tcgen05 rows with its values walked 16 columns a launch (`Tiles.columns`),
    a block narrower than its sums' rescale takes at once (`placed.RESCALED_COLUMNS`), causal and
    folded over 192 tokens, under the interpreter: on par with the reference (check 1)."""
    _check_forward("causal", jnp.bfloat16, (1, 192, 2, 1), natural=False, interpret=True, width=64,
                   value_width=64, tiles=dataclasses.replace(ROWS_TILES, columns=16),
                   target=generation.gpu_target("sm100"))


@INTERPRETS
def test_the_forward_on_tcgen05s_rows_holds_check_1_beside_a_parameter_no_score_hook_reads_interpreted():
    """The forward on a B200's tcgen05 rows (`ROWS_TILES`), causal over 192 tokens, given a key bias and no
    score hook to read it: the launch takes no stream of it, which no role of the rows would release (its
    issuer streams the keys and values alone), and it computes the plain call, on par with the reference
    (check 1)."""
    shape = (1, 192, 2, 1)
    spans = template.spans(shape[1], shape[1], shape[0], **CALLS["causal"])
    draw_operands = _drawn(shape, jnp.bfloat16, width=64)
    target = generation.gpu_target("sm100")

    def draw(seed):
        drawn = jax.random.fold_in(jax.random.PRNGKey(5), seed)
        return *draw_operands(seed), jax.random.normal(drawn, (1, 2, shape[1]), jnp.float32)

    def unread(bias):
        return template.Hooks(score_params=(template.Param(bias, template.Kind.KEY),))

    def ours(query, key, value, dout, bias):
        del dout
        with _device(interpret=True, target=target), _pinned(ROWS_TILES):
            return attention_mosaic.forward(query, key, value, spans, unread(bias), interpret=True)

    def reference(query, key, value, dout, bias):
        del dout
        return template.reference(query, key, value, spans, unread(bias))

    assert_as_exact(jax.jit(ours), jax.jit(reference), draw, "mosaic forward rows unread key bias",
                    seeds=SEEDS)


KEYS_TILES = {
    "two launches": attention_mosaic.BackwardTiles(64, 32, 2, 1, family="mma.sync.produced",
                                                   dkv_family="tcgen05.keys", dkv_consumers=2,
                                                   dkv_blocks=(64, 128)),
    "fused": attention_mosaic.BackwardTiles(64, 64, 2, 1, fused=True, family="mma.sync.produced",
                                            dkv_family="tcgen05.keys", dkv_consumers=2,
                                            dkv_blocks=(64, 128)),
    "walked dq": attention_mosaic.BackwardTiles(128, 64, 2, 1, family="tcgen05.queries",
                                                dkv_family="tcgen05.keys", dkv_consumers=2,
                                                dkv_blocks=(64, 128))}
"""A B200's backward with its dK/dV keys first on tcgen05 (`placed.KEYS`, row 4839): 128 keys a program, a
step of 64 queries, its two compute warpgroups in turn; its dQ launch on mma.sync's producer rings at 64 x 32
(at 64 x 64 its threads hold 254 registers, past the 248 they have), or its dQ its own (`fused`,
FlashAttention 4's: each step's `dS K` over its freed turns, added into dQ), or its dQ launch queries first
on tcgen05 too (`walked dq`, `placed.QUERIES`: 128 queries a step of 64 keys, dQ kept in tensor memory)."""


@INTERPRETS
@pytest.mark.parametrize("width", [64, 128])
@pytest.mark.parametrize("call", ["full", "causal"])
@pytest.mark.parametrize("form", KEYS_TILES)
def test_the_backward_keys_first_on_tcgen05_holds_check_1_interpreted(form, call, width):
    """The backward pinned with its dK/dV keys first on a B200's tcgen05 (`KEYS_TILES`: an issuer warp's
    scores and their cotangents a step ahead, dV's and dK's products onto their sums in tensor memory, two
    compute warpgroups forming each step's weights and score cotangents over the scores' own columns; its dQ
    a launch of its own, or the walk's own), over 256 tokens (two key blocks, four query steps each, the
    rings' slots read twice), under the interpreter: every gradient on par with the reference (check 1)."""
    spans = template.spans(256, 256, 1, **CALLS[call])
    _check_spans(call, spans, (1, 256, 2, 1), interpret=True, backward=True, width=width,
                 pins=(None, KEYS_TILES[form]), target=generation.gpu_target("sm100"))


@INTERPRETS
def test_a_refusal_of_the_rows_forwards_rewalk_alone_steps_its_rewalk_alone_interpreted():
    """The forward pinned on a B200's tcgen05 rows (`ROWS_TILES`), causal and folded over 192 tokens, its
    fold's rewalk on another family at the tiles that family's derivation gives it (`MosaicForward.completed`,
    `Tiles.relaunch_family`): a compile's refusal of the rewalk's kernel alone (its spill,
    `report.refuse_launch`) keeps the rows' own tiles and steps the rewalk to others
    (`MosaicForward._rewalked_apart`), at which the launch lowers for a B200 and holds check 1 under the
    interpreter. Refused as one config, the 4B forward's rows stepped down to a block of 16 value columns
    where its rewalk had spilled (et-rows-spill)."""
    from kernel_compile import lowered_for

    from dew.nn.kernels.templates import report
    from dew.nn.kernels.templates.attention_mosaic_forward import NATURAL_NAME

    shape, width = (1, 192, 2, 1), 64
    spans = template.spans(shape[1], shape[1], shape[0], **CALLS["causal"])
    query, key, value, _ = _drawn(shape, jnp.bfloat16, width=width, value_width=width)(0)
    target = generation.gpu_target("sm100")
    fields = ("relaunch_family", "relaunch_q", "relaunch_k", "relaunch_consumers", "relaunch_columns")

    def refused(interpret: bool) -> None:
        # The pinned config as the launch of `forward` completes it, then its rewalk's launch refused: the
        # rows' own tiles kept, the rewalk's stepped.
        op, args = attention_mosaic._forward_call(query, key, value, spans, template.Hooks(), scale=None,
                                                  natural=False, residuals=template.Kept.NONE)
        ba = op.bind(*args, residuals=template.Kept.NONE, interpret=interpret)
        completed = op.completed(ba, ROWS_TILES)
        assert completed.relaunch_family is not None, completed
        columns = "" if completed.relaunch_columns is None else f"_columns{completed.relaunch_columns}"
        report.refuse_launch(op, ba, completed, {f"attention_forward{NATURAL_NAME}{columns}"})
        stepped = op.completed(ba, ROWS_TILES)
        assert stepped.relaunch_family is not None, stepped
        assert dataclasses.replace(stepped, **dict.fromkeys(fields)) == ROWS_TILES, stepped
        assert [getattr(stepped, name) for name in fields] != [getattr(completed, name) for name in fields]

    with report.refusals():
        with _device(interpret=True, target=target):
            refused(interpret=True)
        with generation.assumed("sm100"):
            refused(interpret=False)
        abstract = tuple(jax.ShapeDtypeStruct(array.shape, array.dtype) for array in (query, key, value))
        with _pinned(ROWS_TILES):
            lowered_for(lambda q, k, v: attention_mosaic.forward(q, k, v, spans, None, natural=False),
                        abstract, "sm100")
        _check_forward("causal", jnp.bfloat16, shape, natural=False, interpret=True, width=width,
                       value_width=width, tiles=ROWS_TILES, target=target)


@pytest.mark.parametrize("count", [0, 1, 2, 3, 4, 5])
def test_the_rows_issuer_issues_each_tiles_logits_before_the_tile_befores_weighted_values(count):
    """tcgen05's rows' issuer warp over a walk of `count` key tiles (`placed.issued`, its loops run
    eagerly): each tile's logits and its weighted values issued once, at its turn (its place among the two
    tiles of each, its parity), the next tile's logits before a tile's weighted values (FlashAttention 4's
    S in flight: the tensor core takes them while the softmax weighs the tile), short walks odd and even."""
    issued = []

    def score(index, turn: int) -> None:
        issued.append(("S", int(index), turn))

    def weigh(index, turn: int) -> None:
        issued.append(("PV", int(index), turn))

    with jax.disable_jit():
        mosaic.looped(placed.issued(score, weigh, 0, count), None, ())
    walked = [("S", 0, 0)] if count else []
    for index in range(count):
        walked += [("S", index + 1, (index + 1) % 2)] if index + 1 < count else []
        walked.append(("PV", index, index % 2))
    assert issued == walked


@pytest.mark.parametrize("tiles", [2, 3, 4, 5])
def test_the_rows_forward_states_each_product_its_issuer_walks(tiles):
    """A bf16 inference forward on tcgen05's rows (`ROWS_TILES`) over 128 queries and `tiles` key tiles of
    64, lowered for a B200: its launch states the products its issuer warp issues (`placed.issued`,
    counted from its kernel's emission, `emission.costed`), each tile's logits and weighted values once, a
    multiply-add a reached pair, head and product column, the model's share the whole; an odd walk's and an
    even's alike, its first step's two logits and its last's none its steady step's two
    (`emission.Role.step`)."""
    rows, heads, kv_heads, width, queries = 1, 4, 2, 64, 128
    keys = 64 * tiles
    query = jax.ShapeDtypeStruct((rows, queries, heads, width), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((rows, keys, kv_heads, width), jnp.bfloat16)
    spans = template.spans(queries, keys, rows)
    issued = 2 * rows * heads * queries * keys * (width + width)
    with _pinned(ROWS_TILES):
        assert _counted(lambda q, k, v: attention_mosaic.forward(q, k, v, spans), query, key, key,
                        generation_name="sm100") == (issued, issued)


@INTERPRETS
def test_the_forward_on_tcgen05s_rows_holds_check_1_over_documents_interpreted():
    """The forward on a B200's tcgen05 rows over the template core's causal packed documents under a window
    (`CASES["causal-documents-window"]`: its rows' key intervals, block maps at its 128 x 64 tiles, rows
    whose keys a tile hides and padding), under the interpreter: on par with the reference (check 1)."""
    case = CASES["causal-documents-window"]
    queries, keys = core_geometry(case)
    called = {name: value[:1] if hasattr(value, "shape") else value
              for name, value in arguments(case).items()}
    spans = template.spans(queries, keys, 1, **called)
    draw = _drawn((1, queries, 2, 1), jnp.bfloat16, width=64, keys=spans.keys)
    target = generation.gpu_target("sm100")

    def ours(query, key, value, dout):
        del dout
        with _device(interpret=True, target=target), _pinned(ROWS_TILES):
            return attention_mosaic.forward(query, key, value, spans, None, interpret=True)

    def reference(query, key, value, dout):
        del dout
        return template.reference(query, key, value, spans)

    assert_as_exact(jax.jit(ours), jax.jit(reference), draw, "mosaic forward rows documents", seeds=SEEDS)


@INTERPRETS
@pytest.mark.parametrize("call", ["full", "causal"])
def test_the_walked_backward_at_width_256_holds_check_1_interpreted(call):
    """The backward at width 256 pinned on a B200's walked placements as the 4B heads' search finds them
    (`MosaicBackward.walked`: the one launch's resident keys and values and its query streams past the
    block there, the two launches, dQ queries first a step of 32 keys, dK/dV keys first a step of 32
    queries, dK's and dV's columns two shares of 128), over 256 tokens, under the interpreter: every
    gradient on par with the reference (check 1)."""
    tiles = attention_mosaic.BackwardTiles(128, 32, 2, 1, family="tcgen05.queries", columns=2,
                                           dkv_family="tcgen05.keys", dkv_consumers=2, dkv_blocks=(32, 128))
    spans = template.spans(256, 256, 1, **CALLS[call])
    _check_spans(call, spans, (1, 256, 2, 1), interpret=True, backward=True, width=256,
                 pins=(None, tiles), target=generation.gpu_target("sm100"))


@pytest.mark.parametrize("form", ["two launches", "fused", "walked dq"])
def test_the_walked_backward_states_each_product_its_issuers_walk(form):
    """A bf16 call's backward over 256 queries and keys, every pair reached, with its dK/dV keys first on a
    B200's tcgen05 (`KEYS_TILES`), lowered for a B200: its launches state the products their issuers walk,
    counted from their kernels' emission (`emission.costed`), each tile's once, a multiply-add a pair, head
    and product column: keys first `Sᵀ`, `dPᵀ`, dV's and dK's, a fused walk's dQ too, a dQ launch's `S`,
    `dP` and dQ's beside them (on mma.sync or queries first); the model's share the backward's four."""
    rows, heads, kv_heads, width, tokens = 1, 4, 2, 64, 256
    query = jax.ShapeDtypeStruct((rows, tokens, heads, width), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((rows, tokens, kv_heads, width), jnp.bfloat16)
    spans = template.spans(tokens, tokens, rows)
    with generation.assumed("sm100"):
        residuals = jax.eval_shape(lambda q, k, v: attention_mosaic.forward(
            q, k, v, spans, residuals=template.Kept.SPLIT), query, key, key)[1:]

    def backward(query, key, value, dout, residuals):
        return attention_mosaic.backward(query, key, value, spans, None, dout, residuals=residuals)

    product = 2 * rows * heads * tokens * tokens * width
    with _pinned(backward=KEYS_TILES[form]):
        assert _counted(backward, query, key, key, query, residuals, generation_name="sm100") == (
            4 * product, (5 if form == "fused" else 7) * product)


@pytest.mark.parametrize("pin", [
    pytest.param(dataclasses.replace(KEYS_TILES["two launches"], dkv_family="tcgen05.queries",
                                     dkv_consumers=1, dkv_blocks=(128, 64)), id="dq-placement-as-dkv"),
    pytest.param(attention_mosaic.BackwardTiles(64, 128, 2, 2, fused=True, family="tcgen05.keys"),
                 id="keys-as-the-one-launch"),
    pytest.param(dataclasses.replace(KEYS_TILES["fused"], dkv_blocks=(32, 128)), id="fused-dq-of-32-rows"),
])
def test_a_walked_placement_takes_only_the_launch_it_is_written_for(pin):
    """A backward config naming a walked placement on a launch it is not written for, or a fused walk whose
    dQ products would be 32 rows (tcgen05 issues 64 and 128): the B200's legality refuses it (`legal`), where
    the two launches' placements as written are taken."""
    shape, spans = (1, 256, 4, 64), template.spans(256, 256, 1, causal=True)
    query = jnp.zeros(shape, jnp.bfloat16)
    key = jnp.zeros((1, 256, 2, 64), jnp.bfloat16)
    with generation.assumed("sm100"):
        _, *residuals = jax.eval_shape(lambda q, k, v: attention_mosaic.forward(
            q, k, v, spans, None, residuals=template.Kept.SPLIT), query, key, key)
        op, args = attention_mosaic._backward_call(query, key, key, spans, None, query, tuple(residuals),
                                                   scale=None, natural=attention_mosaic.NATURAL,
                                                   accurate=False)
        ba, target = op.bind(*args, interpret=False), generation.gpu_target()
        assert op.legal(ba, KEYS_TILES["two launches"], target)
        assert not op.legal(ba, pin, target)
