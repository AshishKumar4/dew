"""The Mosaic attention mainloop's walks placed on tcgen05 (`placed`): a walk whose declared products a role
of their own issues, a step ahead of the warpgroups that read them, their accumulators in tensor memory
(FlashAttention 4's structure): the forward's rows (`placed.ROWS`, row 5368) and the backward's launches
keys first and queries first (`placed.KEYS`, `placed.QUERIES`, row 4839), each pinned, under the interpreter
for a B200 or lowered for one."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import COMPILES_ON_CPU, INTERPRETS
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

from dew.nn import programs
from dew.nn.kernels import generation, tuning
from dew.nn.kernels.templates import attention as template, attention_mosaic

pytestmark = pytest.mark.kernel


ROWS_TILES = attention_mosaic.Tiles(128, 64, 2, 1, family="tcgen05.rows")
"""tcgen05's rows' one tile (`placed.ROWS`): a product's 128 rows, a row a consumer thread, 64 keys
a step."""


@INTERPRETS
@pytest.mark.parametrize("width", [64, 128, 256])
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


ROWS_PAIR_TILES = attention_mosaic.Tiles(128, 64, 2, 2, family="tcgen05.rows")
"""tcgen05's rows at two consumers: two query tiles of 128 rows a program over its one key walk, their
products issued in turn (FlashAttention 4's two softmax warpgroups), each turn's weights over its logits."""


@INTERPRETS
@pytest.mark.parametrize("tokens", [192, 320])
@pytest.mark.parametrize("call", ["full", "causal"])
def test_the_forward_on_tcgen05s_rows_at_two_consumers_holds_check_1_interpreted(call, tokens):
    """The forward on a B200's tcgen05 rows at two consumers (`ROWS_PAIR_TILES`: the issuer serving each
    consumer's logits and weighted values in turn on one key and value slot a step, a causal walk's later
    query tile walking more of the program's keys than its partner, `placed.issuing`'s `reaches`), its
    query tiles padded, under the interpreter, which models tensor memory and its turns' weights over their
    logits: on par with the reference (check 1)."""
    _check_forward(call, jnp.bfloat16, (1, tokens, 2, 1), natural=False, interpret=True, width=64,
                   value_width=64, tiles=ROWS_PAIR_TILES, target=generation.gpu_target("sm100"))


def _placed_forward(tiles: attention_mosaic.Tiles, spans: template.KeySpans):
    """A traced sm100 forward call pinned at `tiles` over `spans` for `ring_protocol.programs`."""
    def forward(query, key):
        with generation.assumed("sm100"), _pinned(tiles):
            return attention_mosaic.forward(query, key, key, spans, interpret=False)

    return forward


@pytest.mark.parametrize("moves", [False, True], ids=["kept", "moved"])
@pytest.mark.parametrize("call", ["full", "causal"])
@pytest.mark.parametrize("tiles", [ROWS_TILES, ROWS_PAIR_TILES], ids=["one", "two"])
def test_the_forward_on_tcgen05s_rows_holds_every_property_in_every_interleaving(tiles, call, moves):
    """The forward's launch on a B200's tcgen05 rows at one consumer and two, traced as a B200 builds it over
    256 queries and 256 keys, the full call's and the causal's (each consumer's own key walk): every
    interleaving of its consumers and its producer's and issuer's warps explored (`ring_protocol.explore`),
    no deadlock and every barrier's phase whole and observed, every step's factor keeping its sums or moving
    them (`moves`, a step whose maximum moves, the rare span). A step waits on the weighted values of the
    step before only where its factor moves some row's sums (`placed.Tcgen05Placed.accumulating`), its
    weights' turn freed by the product two steps back."""
    from ring_protocol import explore, programs

    query, key = jnp.ones((1, 256, 1, 128), jnp.bfloat16), jnp.ones((1, 256, 1, 128), jnp.bfloat16)
    spans = template.spans(256, 256, 1, **CALLS[call])
    threads, arrivals = programs(_placed_forward(tiles, spans), query, key,
                                 threads=tiles.consumers + 1, rare=moves)
    assert explore(threads, arrivals) > 0


def _tmem_q_copies(issuer) -> list[tuple[int, object, object]]:
    """Each `(position, tmem_dst, smem_src)` pair emitted by `async_copy_smem_to_tmem` in `issuer`."""
    copies = []
    for i, event in enumerate(issuer):
        if (event.kind == "write" and i + 1 < len(issuer)
                and issuer[i + 1].kind == "product" and len(issuer[i + 1].argument) == 1):
            copies.append((i, event.at, issuer[i + 1].argument[0]))
    return copies


def _consumer_q_smem(threads, consumers: int) -> dict[int, object]:
    """Each consumer `c`'s Q SMEM ref from its `copy_gmem_to_smem` load event."""
    found = {}
    for c in range(consumers):
        loads = [event.at for event in threads[c] if event.kind == "load"]
        assert len(loads) == 1, f"consumer {c} expected one Q load, got {loads}"
        found[c] = loads[0]
    return found


@pytest.mark.parametrize(("tiles", "width"), [(ROWS_TILES, 256), (ROWS_PAIR_TILES, 64)],
                         ids=["one-d256", "two-d64"])
@pytest.mark.parametrize("call", ["full", "causal"])
def test_the_forward_on_tcgen05s_rows_at_d256_holds_every_property_in_every_interleaving(tiles, width, call):
    """The forward on a B200's tcgen05 rows where Q fits in TMEM (1-consumer d256 and 2-consumer d64, rows
    5368, 9103-9105, 9474): every interleaving explored (`ring_protocol.explore`), each consumer's own SMEM Q
    copied to a TMEM ref before any score MMA for that consumer reads it, and every score MMA signaling a
    consumer reading the TMEM tile filled from that consumer's own SMEM Q."""
    from ring_protocol import explore, programs

    from dew.nn.kernels.templates import attention_mosaic_forward as forward_mod, placed

    tmem = generation.gpu_target("sm100").tmem_columns
    for candidate, cand_width, expected in ((ROWS_TILES, 256, True), (ROWS_PAIR_TILES, 64, True),
                                            (ROWS_PAIR_TILES, 128, False)):
        products = forward_mod.forward_products(placed.ROWS, candidate, cand_width, jnp.bfloat16, ahead=True,
                                                query_width=cand_width, tmem_columns=tmem)
        assert (placed.QUERY in placed.ROWS.product_scratch(*products)) is expected
        assert placed.ROWS.product_columns(products) <= tmem

    query = jnp.ones((1, 256, 1, width), jnp.bfloat16)
    key = jnp.ones((1, 256, 1, width), jnp.bfloat16)
    spans = template.spans(256, 256, 1, **CALLS[call])
    threads, arrivals = programs(
        _placed_forward(tiles, spans), query, key,
        threads=tiles.consumers + 1 + len(placed.correction_roles(placed.ROWS, tiles.consumers, width)),
        rare=False)
    (issuer,) = [thread for thread in threads if any(event.kind == "product" for event in thread)]
    q_smem = _consumer_q_smem(threads, tiles.consumers)
    # Map each consumer-exclusive barrier name to its consumer index so each score MMA's `scored` arrival
    # identifies which consumer receives that MMA's logits (row 9474).
    waiters: dict[object, set[int]] = {}
    for c in range(tiles.consumers):
        for event in threads[c]:
            if event.kind == "wait":
                waiters.setdefault(event.at[0], set()).add(c)
    owner_of = {barrier: next(iter(owners)) for barrier, owners in waiters.items() if len(owners) == 1}
    committed_bars = {event.at[0] for event in issuer if event.kind == "committed"}
    # Each consumer's first exclusive waited barrier that the issuer commits to is its `scored` barrier
    # (`placed.py:321`; excluding the Q-load barrier which the issuer waits on but never commits to).
    scored_of = {
        next(event.at[0] for event in threads[c]
             if event.kind == "wait" and owner_of.get(event.at[0]) == c and event.at[0] in committed_bars): c
        for c in range(tiles.consumers)}

    copies = _tmem_q_copies(issuer)
    assert len(copies) == tiles.consumers
    written_at = {dst: pos for pos, dst, _ in copies}
    tmem_source = {dst: src for _, dst, src in copies}
    assert set(tmem_source.values()) == set(q_smem.values())

    # Identify every score MMA independently by its following `scored` commit (so a score MMA reading Q
    # directly from SMEM cannot be filtered out), and verify it reads the TMEM tile copied from its
    # consumer's own Q SMEM ref (row 9474).
    score_mmas = []
    for i, event in enumerate(issuer):
        if event.kind == "product" and len(event.argument) == 2:
            next_commit = next((later for later in issuer[i + 1:] if later.kind == "committed"), None)
            if next_commit is not None and next_commit.at[0] in scored_of:
                score_mmas.append((i, event, scored_of[next_commit.at[0]]))
    assert len(score_mmas) >= tiles.consumers
    seen_consumers = set()
    for i, event, consumer in score_mmas:
        q_srcs = [src for src in event.argument if src in written_at]
        assert len(q_srcs) == 1 and written_at[q_srcs[0]] < i
        assert tmem_source[q_srcs[0]] == q_smem[consumer]
        seen_consumers.add(consumer)
    assert seen_consumers == set(range(tiles.consumers))
    assert explore(threads, arrivals) > 0


@INTERPRETS
def test_two_consumers_with_distinct_queries_on_tcgen05s_rows_match_reference_interpreted():
    """Two consumers on `tcgen05.rows` width 64 (Q in TMEM) given distinct query tiles produce each
    consumer's own output: swapping `lefts` in `placed.py:350` fails numerically (row 9474)."""
    def draw(seed: int):
        rng = np.random.default_rng(9474 + seed)
        q0 = rng.standard_normal((1, 128, 1, 64)) + 1.0
        q1 = rng.standard_normal((1, 128, 1, 64)) - 1.0
        query = jnp.asarray(np.concatenate([q0, q1], axis=1), jnp.bfloat16)
        key = jnp.asarray(rng.standard_normal((1, 128, 1, 64)), jnp.bfloat16)
        value = jnp.asarray(rng.standard_normal((1, 128, 1, 64)), jnp.bfloat16)
        return query, key, value

    def ours(query, key, value):
        spans = template.spans(256, 128, 1)
        with generation.assumed("sm100"), _pinned(ROWS_PAIR_TILES):
            return attention_mosaic.forward(query, key, value, spans, interpret=True)

    def theirs(query, key, value):
        return template.reference(query, key, value, template.spans(256, 128, 1))

    assert_as_exact(jax.jit(ours), jax.jit(theirs), draw,
                    "two-consumer tcgen05.rows distinct Q in TMEM", seeds=SEEDS)


def test_an_empty_walk_on_tcgen05s_rows_skips_the_q_tmem_copy():
    """Concrete empty and live reaches traced with `ring_protocol.programs`: only the live consumer issues
    `async_copy_smem_to_tmem`, while an empty consumer issues none (changing `placed.py:349` `<` to `<=`
    fails, row 9473)."""
    from ring_protocol import explore, programs

    query = jnp.ones((1, 256, 1, 64), jnp.bfloat16)
    key = jnp.ones((1, 256, 1, 64), jnp.bfloat16)
    # Consumer 0 empty (0..0), consumer 1 live (0..2 tiles of 64 keys).
    spans_mixed = template.KeySpans(
        start=jnp.zeros((1, 256), jnp.int32),
        end=jnp.concatenate([jnp.zeros((1, 128), jnp.int32), jnp.full((1, 128), 128, jnp.int32)], axis=1),
        query_ids=None, key_ids=None, keys=256)
    mixed_maps = {5: np.array([[0, 0]], np.int32), 6: np.array([[0, 2]], np.int32),
                  7: np.array([[0, 0]], np.int32), 8: np.array([[0, 2]], np.int32)}
    threads, arrivals = programs(_placed_forward(ROWS_PAIR_TILES, spans_mixed), query, key,
                                 threads=ROWS_PAIR_TILES.consumers + 1, rare=False,
                                 known=lambda idx, _aval: mixed_maps.get(idx))
    (issuer,) = [thread for thread in threads if any(event.kind == "product" for event in thread)]
    q_smem = _consumer_q_smem(threads, ROWS_PAIR_TILES.consumers)
    copies = _tmem_q_copies(issuer)
    assert len(copies) == 1 and copies[0][2] == q_smem[1] and copies[0][2] != q_smem[0]
    assert explore(threads, arrivals) > 0

    # All-empty 1-consumer walk: zero Q copies issued.
    q1 = jnp.ones((1, 128, 1, 64), jnp.bfloat16)
    k1 = jnp.ones((1, 128, 1, 64), jnp.bfloat16)
    spans_empty = template.KeySpans(start=jnp.zeros((1, 128), jnp.int32), end=jnp.zeros((1, 128), jnp.int32),
                                    query_ids=None, key_ids=None, keys=128)
    empty_maps = {5: np.array([[0]], np.int32), 6: np.array([[0]], np.int32),
                  7: np.array([[0]], np.int32), 8: np.array([[0]], np.int32)}
    threads_e, arrivals_e = programs(_placed_forward(ROWS_TILES, spans_empty), q1, k1,
                                     threads=ROWS_TILES.consumers + 1, rare=False,
                                     known=lambda idx, _aval: empty_maps.get(idx))
    for thread in threads_e:
        assert _tmem_q_copies(thread) == []
    assert explore(threads_e, arrivals_e) > 0


@pytest.mark.parametrize("moves", [False, True], ids=["kept", "moved"])
@pytest.mark.parametrize("call", ["full", "causal"])
def test_the_correction_warpgroup_at_d256_holds_protocol_in_every_interleaving(call, moves):
    """Row 5368: at d256 on 1-consumer `tcgen05.rows`, the dedicated `CORRECTION` warpgroup rescales kept
    sums in TMEM off the softmax warpgroup's critical path: every interleaving of consumer, correction, and
    producer/issuer holds without deadlock or race over 5 steps (reusing both factor slots) and at 0- and
    1-step reaches, both when rescales are skipped (`kept`) and taken (`moved`), and dropping a `factor_ready`
    wait before reading `factors` is caught as a race."""
    from ring_protocol import Violation, explore, programs

    query, key = jnp.ones((1, 256, 1, 256), jnp.bfloat16), jnp.ones((1, 320, 1, 256), jnp.bfloat16)
    spans = template.spans(256, 320, 1, **CALLS[call])
    threads, arrivals = programs(_placed_forward(ROWS_TILES, spans), query, key, threads=3, rare=moves)
    consumer_writes = {e.at[0] for e in threads[0] if e.kind == "write" and e.at[0].startswith("smem")}
    factor_reads = [e for e in threads[1] if e.kind == "read" and e.at[0] in consumer_writes]
    assert factor_reads, "correction factor load must be traced"
    assert explore(threads, arrivals) > 0
    # Negative control: dropping the correction warpgroup's factor_ready waits lets the factor read race
    # with the consumer's factor write.
    factor_ready = next(e.at[0] for e in threads[1] if e.kind == "wait")
    mutated = list(threads)
    mutated[1] = [e for e in threads[1] if not (e.kind == "wait" and e.at[0] == factor_ready)]
    with pytest.raises(Violation):
        explore(mutated, arrivals)
    if call == "full" and not moves:
        q1, k1 = jnp.ones((1, 128, 1, 256), jnp.bfloat16), jnp.ones((1, 128, 1, 256), jnp.bfloat16)
        for steps in (0, 1):
            short = template.KeySpans(start=jnp.zeros((1, 128), jnp.int32),
                                      end=jnp.full((1, 128), steps * 64, jnp.int32),
                                      query_ids=None, key_ids=None, keys=128)
            short_maps = {5: np.array([[0]], np.int32), 6: np.array([[steps]], np.int32),
                          7: np.array([[0]], np.int32), 8: np.array([[steps]], np.int32)}
            short_threads, short_arrivals = programs(_placed_forward(ROWS_TILES, short), q1, k1, threads=3,
                                                     rare=False,
                                                     known=lambda idx, _aval, m=short_maps: m.get(idx))
            assert explore(short_threads, short_arrivals) > 0


@INTERPRETS
@pytest.mark.parametrize("call", ["full", "causal"])
def test_the_correction_warpgroup_at_d256_holds_check_1_interpreted(call):
    """Row 5368: 1-consumer d256 `tcgen05.rows` with the dedicated `CORRECTION` warpgroup holds check 1
    against the reference under the interpreter (full and causal)."""
    _check_forward(call, jnp.bfloat16, (1, 256, 2, 1), natural=False, interpret=True, width=256,
                   value_width=256, tiles=ROWS_TILES, target=generation.gpu_target("sm100"))


def test_placed_forward_uses_rounded_tmem_residency_for_query_placement():
    """Placed-forward derivation uses rounded TMEM and SMEM residency (`resources.register_model`) when
    deciding whether to place Q in TMEM, proved through the public `attention_mosaic.forward` route and its
    emitted `RegisterModel` (rows 9105, 9475, 9479, 9527, 9530):
    - `q64/v32` (`columns=32`, 48 KiB SMEM): 160 base cols -> 256 rounded, 192 with Q -> 256 rounded; raw
      `512 // 160 = 3` vs rounded `512 // 256 = 2`; keeps Q in TMEM and emits `RegisterModel.most == 2`.
    - `q128/v128` (256 base cols -> 256 rounded = 2 CTAs/SM, 320 with Q -> 512 rounded = 1 CTA/SM): drops
      residency 2 -> 1, so Q stays in SMEM and emits `RegisterModel.most == 2`.
    - `q128/v64` (`stages=5`, >120 KiB K/V rings): 192 base cols -> 256 with Q (2 CTAs/SM by TMEM and
      registers) while SMEM caps residency at 1; keeps Q in TMEM and emits `RegisterModel.most == 1`."""
    from jax._src.pallas.mosaic_gpu import primitives as gpu_primitives

    from dew.nn.kernels.templates import report

    def has_tmem_copy(jaxpr) -> bool:
        return any(eq.primitive is gpu_primitives.async_copy_smem_to_tmem_p
                   for eq in programs.equations(jaxpr))

    def traced_models(jaxpr) -> list:
        found = []
        for eq in programs.equations(jaxpr):
            name = eq.params.get("name")
            if isinstance(name, str):
                kernel, launch, _ = report.kernel_of(name)
                if (launch is not None and (record := report.traced(launch)) is not None
                        and getattr(record.config, "family", None) == "tcgen05.rows"
                        and kernel in record.kernels):
                    found.append(record.kernels[kernel])
        return found

    target = generation.gpu_target("sm100")
    spans = template.spans(128, 256, 1)
    v32_tiles = dataclasses.replace(ROWS_TILES, columns=32)
    smem_tiles = dataclasses.replace(ROWS_TILES, stages=5)
    # q64/v32 (48 KiB SMEM, 4 CTAs/SM SMEM cap): 160 base cols -> 256 rounded (2 CTAs) vs raw 512//160=3;
    # 192 with Q -> 256 rounded (2 CTAs) vs raw 512//192=2; distinguishes rounded from raw residency.
    # q128/v128: 256 base cols (2 CTAs) vs 320 with Q -> 512 rounded (1 CTA); rejects Q in TMEM.
    # q128/v64 (stages=5, >120 KiB rings): TMEM (192->256) and registers allow 2 CTAs, SMEM caps to 1.
    for q_width, v_width, pinned, expect_cols, expect_copy, expect_most in (
            (64, 32, v32_tiles, 192, True, 2), (128, 128, ROWS_TILES, 256, False, 2),
            (128, 64, smem_tiles, 256, True, 1)):
        q = jnp.ones((1, 128, 1, q_width), jnp.bfloat16)
        k = jnp.ones((1, 256, 1, q_width), jnp.bfloat16)
        v = jnp.ones((1, 256, 1, v_width), jnp.bfloat16)
        op = attention_mosaic.MosaicForward(scale=q_width ** -0.5, keys=256, form="full", natural=True)
        ba = op.bind(q, k, v, None, None, None, None, (), (), None,
                     residuals=template.Kept.NONE, interpret=False)
        with generation.assumed("sm100"), _pinned(pinned):
            assert op.tmem_columns(ba, pinned, target) == expect_cols
            assert op.resident(ba, pinned, target) == expect_most
            traced = jax.make_jaxpr(lambda q, k, v: attention_mosaic.forward(
                q, k, v, spans, natural=True, interpret=False))(q, k, v).jaxpr
        assert has_tmem_copy(traced) is expect_copy
        models = traced_models(traced)
        assert models and all(model.most == expect_most for model in models), models

    def block_copies(jaxpr) -> dict[int, bool]:
        """Map each tcgen05.rows kernel's V-block width (operand 2's last dim) to whether it copies Q."""
        return {
            eq.invars[2].aval.shape[-1]: any(
                has_tmem_copy(sub) for sub in jax.extend.core.jaxprs_in_params(eq.params))
            for eq in programs.equations(jaxpr)
            if (isinstance(name := eq.params.get("name"), str)
                and (launch := report.kernel_of(name)[1]) is not None
                and (record := report.traced(launch)) is not None
                and getattr(record.config, "family", None) == "tcgen05.rows")}

    # Nonuniform legal column pins:
    # - q64/v192 (padded 256, columns=192 -> blocks 192 and 64): both 192 and 64 tail place Q in TMEM.
    # - q64/v320 (padded 320, columns=192 -> blocks 192 and 128): 192 copies Q (1 CTA/SM), 128 tail omits Q
    #   (96 KiB SMEM -> 2 CTAs/SM, 256 -> 288 TMEM cols would drop residency 2 -> 1).
    q64 = jnp.ones((1, 128, 1, 64), jnp.float16)
    k64 = jnp.ones((1, 256, 1, 64), jnp.float16)
    for v_width, expected_blocks in ((192, {192: True, 64: True}), (320, {192: True, 128: False})):
        v_arr = jnp.ones((1, 256, 1, v_width), jnp.float16)
        with generation.assumed("sm100"), _pinned(dataclasses.replace(ROWS_TILES, columns=192)):
            tail_traced = jax.make_jaxpr(lambda q, k, v: attention_mosaic.forward(
                q, k, v, spans, natural=True, interpret=False))(q64, k64, v_arr).jaxpr
        assert block_copies(tail_traced) == expected_blocks

    # Rank-only `proposes` must short-circuit before calling `held` (row 9477).
    bwd = attention_mosaic.MosaicBackward(scale=128 ** -0.5, keys=256, form="full")
    called = False

    def boom() -> bool:
        nonlocal called
        called = True
        return False

    assert bwd.proposes(ba, KEYS_TILES["fused"], target, held=boom) is True
    assert not called


@INTERPRETS
@pytest.mark.parametrize("tokens", [64, 128, 256, 320])
def test_the_forward_on_tcgen05s_rows_holds_check_1_over_short_walks_interpreted(tokens):
    """The forward on a B200's tcgen05 rows over walks of one to five key tiles of 64 (a full call of
    `tokens`, every query block walking them all, 192's three the check above), odd and even, its issuer
    warp a tile's logits ahead of the softmax and its last tile's weighted values after the walk's turns
    (`placed.issued`), under the interpreter: on par with the reference (check 1)."""
    _check_forward("full", jnp.bfloat16, (1, tokens, 2, 1), natural=False, interpret=True, width=64,
                   value_width=64, tiles=ROWS_TILES, target=generation.gpu_target("sm100"))


@INTERPRETS
def test_the_forward_on_tcgen05s_rows_streams_values_its_keys_columns_as_handed_interpreted():
    """A forward whose values are its keys' leading columns (`in_key`) on a B200's tcgen05 rows, whose
    issuing role declares the values' stream (`MosaicForward.reads_key_slot`), holds check 1 against the
    reference over seeded draws under the interpreter (rows 6643, 6695)."""
    shape = (1, 128, 1, 64)
    spans = template.spans(shape[1], shape[1], shape[0], causal=True)

    def draw(seed: int):
        query, key = (jax.random.normal(stream, shape, jnp.float32).astype(jnp.bfloat16)
                      for stream in jax.random.split(jax.random.key(seed), 2))
        return query, key

    def ours(query, key):
        with generation.assumed("sm100"), tuning.pinned({attention_mosaic.MosaicForward: ROWS_TILES}):
            return attention_mosaic.forward(query, key, key, spans, None, interpret=True, in_key=True)

    def reference(query, key):
        return template.reference(query, key, key, spans)

    assert_as_exact(jax.jit(ours), jax.jit(reference), draw, "tcgen05.rows in_key forward", seeds=SEEDS)


@INTERPRETS
def test_the_forward_on_tcgen05s_rows_keeps_a_subnormal_output_interpreted():
    """The forward on a B200's tcgen05 rows, its sums kept in tensor memory and taken by their rows' ratio
    as stored (`mosaic.Kept.scale`), under the interpreter on XLA's CPU, which flushes a subnormal product:
    a full bf16 call of 128 keys, every logit 0, one key's values near 2^-120 and the rest 0, holds check 1
    against the reference over seeded draws (subnormal outputs near 2^-127, rows 6302, 6711)."""
    spans = template.spans(128, 128, 1)

    def draw(seed: int):
        streams = jax.random.split(jax.random.key(seed), 2)
        signs = jnp.where(jax.random.bernoulli(streams[0], shape=(1, 1, 1, 64)), 1.0, -1.0)
        mags = jax.random.uniform(streams[1], (1, 1, 1, 64), jnp.float32, minval=1.0, maxval=2.0)
        query = jnp.zeros((1, 128, 1, 64), jnp.bfloat16)
        scaled = (signs * mags * jnp.float32(2.0 ** -120)).astype(jnp.bfloat16)
        value = jnp.zeros_like(query).at[:, :1].set(scaled)
        return query, query, value
    from dew.nn.precision import power2

    def ours(query, key, value):
        with generation.assumed("sm100"), tuning.pinned({attention_mosaic.MosaicForward: ROWS_TILES}):
            return attention_mosaic.forward(query, key, value, spans, None, interpret=True)

    def exact_ref(query, key, value):
        del key
        return jnp.broadcast_to(power2(value[:, :1].astype(jnp.float32), -7).astype(jnp.bfloat16),
                                query.shape)

    def truth(query, key, value):
        return template.reference(query, key, value, spans)

    assert_as_exact(jax.jit(ours), jax.jit(exact_ref), draw, "tcgen05.rows subnormal output", seeds=SEEDS,
                    truth_fn=truth)


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


@pytest.mark.parametrize("name", ["sm100", "sm90"])
def test_a_tcgen05_devices_backward_derives_its_walked_launches(name):
    """Qwen3-8B's heads (32 query heads over 8, width 128, 2 x 4096 causal bf16) derive their backward on a
    B200 as the walked placements take it (`MosaicBackward.derived` is `walked`'s config, its dK/dV keys
    first on tcgen05: M-ET-74's 0.18 of the mma.sync two launches' time, row 6918), where an H100, whose
    walks place on no role, derives its own and has no walked config."""
    tokens, heads, kv_heads, width = 4096, 32, 8, 128
    spans = template.spans(tokens, tokens, 2, causal=True)
    query = jax.ShapeDtypeStruct((2, tokens, heads, width), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((2, tokens, kv_heads, width), jnp.bfloat16)
    with generation.assumed(name):
        residuals = jax.eval_shape(lambda q, k, v: attention_mosaic.forward(
            q, k, v, spans, residuals=template.Kept.SPLIT), query, key, key)[1:]
        op, args = attention_mosaic._backward_call(query, key, key, spans, None, query, tuple(residuals),
                                                   scale=None, natural=attention_mosaic.NATURAL,
                                                   accurate=False)
        ba, target = op.bind(*args, interpret=False), generation.gpu_target()
        walked, derived = op.walked(ba, target), op.derived(ba, target)
    if name == "sm100":
        assert walked is not None and derived == walked and derived.dkv_family == "tcgen05.keys", derived
    else:
        assert walked is None and derived is not None, (walked, derived)


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


@pytest.mark.parametrize("call", ["full", "causal"])
def test_a_fused_walks_dq_ring_and_its_reducer_hold_every_property_in_every_interleaving(call):
    """The fused keys-first dK/dV launch on a B200 (`KEYS_TILES["fused"]` at width 64: each compute
    warpgroup's step's dQ filled a slot at a time into its ring, which its reducer warp's TMA adds read, a
    slot's fill and free a barrier's phase each, `attention_mosaic_dkv._PlacedDq`), traced as a B200 builds
    it over 128 keys and two query steps, the full call's and the causal's (whose program walks fewer):
    every interleaving of its compute warpgroups, its load, issuer and two reducer warps and its copies'
    landings explored (`ring_protocol.explore`), no deadlock, every barrier's phase whole and observed, and
    no slot written while an add still reads it. Two query steps, each its dQ's chunks through the two slots
    (three's and four's interleavings passed the checker's bound once its stores' groups and fills were
    events)."""
    from ring_protocol import explore, programs

    tokens, width = 128, 64
    query = jnp.ones((1, tokens, 1, width), jnp.bfloat16)
    key = jnp.ones((1, 128, 1, width), jnp.bfloat16)
    spans = template.spans(tokens, 128, 1, **CALLS[call])
    with generation.assumed("sm100"):
        _, *residuals = jax.eval_shape(lambda q, k, v: attention_mosaic.forward(
            q, k, v, spans, None, residuals=template.Kept.SPLIT), query, key, key)

    def backward(query, key, dout, residuals):
        with generation.assumed("sm100"), _pinned(backward=KEYS_TILES["fused"]):
            return attention_mosaic.backward(query, key, key, spans, None, dout, residuals=residuals,
                                             accurate=False, interpret=False)

    residuals = jax.tree.map(lambda shaped: jnp.zeros(shaped.shape, shaped.dtype), tuple(residuals))
    threads, arrivals = programs(backward, query, key, query, residuals, threads=3)
    assert explore(threads, arrivals) > 0


def test_a_dq_ring_producer_and_reducer_hold_across_multiple_ordinals_on_one_role():
    """Row 8979: a focused producer/reducer trace of `attention_mosaic_dkv._PlacedDq` covering ordinals 0
    and 1 on the same compute role (`lo=0, hi=3, role=0`): explores every interleaving with zero violations,
    and fails with `Violation` when `_PlacedDq._turn` drops its `ordinal` term."""
    import types

    from jax.experimental import pallas as pl
    from jax.experimental.pallas import mosaic_gpu as plgpu
    from ring_protocol import Violation, explore, programs

    from dew.nn.kernels.templates import attention_mosaic_dkv as dkv, placed, rings

    block_q, width, slot_cols = 64, 64, 64
    dq_out = jnp.zeros((1, 1, 3 * block_q, 1, width), jnp.float32)
    walk = types.SimpleNamespace(
        interpret=False,
        program=types.SimpleNamespace(family=placed.KEYS, tiles=types.SimpleNamespace(block_q=block_q)))

    class _NoOrdinal(dkv._PlacedDq):
        def _turn(self, ordinal, piece: int):
            del ordinal
            return rings.Turn.of(piece, dkv.REDUCED)

    def traced(ring_cls, name: str):
        def launch(dq_arr):
            def body(dq_ref):
                def scope(stage, filled, emptied):
                    ring = ring_cls(walk, dq_ref, (None, stage, filled, emptied), (0, 0))
                    half = jnp.zeros((block_q, width), jnp.float32)
                    role = jax.lax.axis_index("wg")

                    @pl.when(role == 0)
                    def _():
                        ring.added((half,), 0, types.SimpleNamespace(turn=0, n=0))
                        ring.added((half,), 2 * block_q, types.SimpleNamespace(turn=0, n=placed.TURNS))
                        ring.drained(0, 3, role=0)

                    @pl.when(role == 1)
                    def _():
                        ring.reduced(0, 3, role=0)

                pl.run_scoped(
                    scope,
                    plgpu.SMEM((placed.TURNS, dkv.REDUCED, block_q, slot_cols), jnp.float32),
                    tuple(plgpu.Barrier(num_barriers=dkv.REDUCED) for _ in range(placed.TURNS)),
                    tuple(plgpu.Barrier(num_barriers=dkv.REDUCED) for _ in range(placed.TURNS)),
                )

            return plgpu.kernel(body, out_type=jax.ShapeDtypeStruct(dq_arr.shape, dq_arr.dtype),
                                grid=(1,), grid_names=("p",), num_threads=2, thread_name="wg",
                                kernel_name=name)()

        return programs(launch, dq_out, threads=2)

    threads, arrivals = traced(dkv._PlacedDq, "dq_ring_ordinals")
    assert explore(threads, arrivals) > 0

    # Dropping `ordinal` in `_PlacedDq._turn` must be caught by `explore`.
    bad_threads, bad_arrivals = traced(_NoOrdinal, "dq_ring_ordinals_mutated")
    with pytest.raises(Violation):
        explore(bad_threads, bad_arrivals)


@COMPILES_ON_CPU
@pytest.mark.parametrize("width", [64, 128])
def test_a_fused_walk_and_its_dq_reducer_compile_spill_free_on_a_b200(width):
    """The fused keys-first backward (`KEYS_TILES["fused"]`: its dQ rings drained by their reducer warps' TMA
    adds, `attention_mosaic_dkv._PlacedDq`) at widths 64 and 128 over 512 causal tokens, four query heads
    over two, compiled by ptxas for a compile-only B200: its launch lowers, the reducer's barrier waits,
    adds and arrivals among it, and spills nothing (`mosaic_failures`)."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    tokens = 512
    spans = template.spans(tokens, tokens, 1, causal=True)
    query = jax.ShapeDtypeStruct((1, tokens, 4, width), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((1, tokens, 2, width), jnp.bfloat16)

    def trained(query, key, value):
        return attention_mosaic.forward(query, key, value, spans, None, residuals=template.Kept.SPLIT)

    def step(query, key, value, dout):
        _, *residuals = trained(query, key, value)
        return attention_mosaic.backward(query, key, value, spans, None, dout, residuals=tuple(residuals),
                                         accurate=False)

    with generation.assumed("sm100"), _pinned(backward=KEYS_TILES["fused"]):
        case = KernelCase("fused", lambda: (step, (query, key, key, query)))
        found = compile_mosaic_kernels([case], "sm100")
    failed = mosaic_failures(found)
    assert not failed, "\n".join(failed)
