"""The template's decode (`attention_decode`): a decode step's attention through the template's forward over a
built work list, each query head of a key head's group a packed row, each row's keys cut into slices, merged.

Each case is held to the reference path's attention over the same keys (check 1): a step of one query position
and a verify block of four, grouped heads, rows of their own lengths and starts, sinks and a softcap, a row
past the fold's reach, and the builders' inverse (`unpacked` of `packed`) and the work list's slices exactly
covering each row's keys.
"""

import dataclasses
import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import COMPILES_ON_CPU, INTERPRETS
from reference_error import assert_as_exact, equations

from dew.nn import programs
from dew.nn.kernels import generation, tuning
from dew.nn.kernels.templates import (
    assembled,
    attention as template,
    attention_algebra as algebra,
    attention_decode,
    attention_mosaic,
    attention_mosaic_forward,
    bundles,
    cost,
    emission,
    report,
)
from dew.nn.kernels.templates.hooks import Hook, Side

KEYS = 256
SEEDS = 8
CALLS = {
    "one-query": (1, 8, 2),
    "verify-four": (4, 8, 2),
    "mha": (1, 4, 4),
}
"""Decode steps by query positions, query heads and key heads (grouped 4 over 1 and 2, and not)."""
HOOKS = ("plain", "sinks", "softcap")
LENGTHS, STARTS = (200, 37), (0, 16)
"""Two rows' keys: the first from 0 to 200, the second from 16 to 37, a window's."""


def _spans(positions: int) -> template.KeySpans:
    return template.spans(positions, KEYS, len(LENGTHS), lengths=jnp.asarray(LENGTHS, jnp.int32),
                          starts=jnp.asarray(STARTS, jnp.int32))


def _hooks(hook: str, heads: int, positions: int, sinks: jax.Array) -> template.Hooks | None:
    if hook == "sinks":
        return template.Hooks(initial=template.sinks(sinks, len(LENGTHS), positions, 64))
    if hook == "softcap":
        return template.Hooks(score=Hook(template.Capped(2.0)))
    return None


def _drawn(positions: int, heads: int, kv_heads: int, dtype, width: int = 64, value_width: int | None = None):
    def draw(seed: int):
        streams = jax.random.split(jax.random.key(seed), 4)
        query = jax.random.normal(streams[0], (len(LENGTHS), positions, heads, width), dtype)
        key, value = (jax.random.normal(stream, (len(LENGTHS), KEYS, kv_heads, wide), dtype)
                      for stream, wide in zip(streams[1:3], (width, value_width or width), strict=True))
        sinks = jax.random.normal(streams[3], (heads,), jnp.float32)
        return query, key, value, sinks

    return draw


def _operands(query_shape: tuple[int, ...], keys: int):
    """A draw of bf16 queries `query_shape` `[rows, T, 8, 64]` and two rows' keys and values over `keys`, two
    heads 64 wide."""
    def draw(seed: int):
        streams = jax.random.split(jax.random.key(seed), 3)
        query = jax.random.normal(streams[0], query_shape, jnp.bfloat16)
        key, value = (jax.random.normal(stream, (2, keys, 2, 64), jnp.bfloat16) for stream in streams[1:])
        return query, key, value

    return draw


@INTERPRETS
@pytest.mark.parametrize("device", ["sm90", "sm80"])
def test_a_decode_steps_rows_whose_sums_pass_fp32s_largest_walk_again_scaled_interpreted(device):
    """A bf16 decode step (`attention_decode.attend`) whose first row's values are positive and near 2^124, so
    its walk's unscaled sums over its 200 keys pass fp32's largest, each value within bf16's, beside a row of
    ordinary values, interpreted as an H100 or an A100 runs it: on par with the reference path at a scale its
    sums hold (check 1), the first row's programs walking their keys again scaled in their launch
    (`after_walk.Decided`) and the second's not, with no pass over the values before the launch to choose its
    walk."""
    positions, heads, kv_heads = CALLS["one-query"]
    spans = _spans(positions)
    drawn = _drawn(positions, heads, kv_heads, jnp.bfloat16)

    def draw(seed: int):
        query, key, value, _ = drawn(seed)
        return query, key, value.at[0].set(jnp.abs(value[0]) * jnp.bfloat16(2.0 ** 124))

    def ours(query, key, value):
        with generation.assumed(device):
            return attention_decode.attend(query, key, value, spans, interpret=True)

    def reference(query, key, value):
        # The reference path's first row's sums pass fp32's largest too: that row run at its values' scale
        # (2^-124 each, exact) and its output taken back by it, the same path at a scale its sums hold.
        scaled = value.at[0].multiply(jnp.asarray(2.0 ** -124, value.dtype))
        out = template.reference(query, key, scaled, spans)
        return out.at[0].multiply(jnp.asarray(2.0 ** 124, out.dtype))

    assert_as_exact(jax.jit(ours), jax.jit(reference), draw, f"template decode {device} past fp32's largest",
                    seeds=SEEDS)


@INTERPRETS
@pytest.mark.parametrize("device", ["sm90", "sm80"])
def test_a_decode_step_over_a_value_it_reads_not_a_number_is_the_references_interpreted(device):
    """A bf16 decode step whose first row reads a value of NaN at one key: its rows' sums are no number, so
    its programs walk their keys again scaled, and the step's output is NaN where the reference path's is,
    never made finite; the second row, its values ordinary, bit for bit its own without the NaN beside it."""
    positions, heads, kv_heads = CALLS["one-query"]
    spans = _spans(positions)
    query, key, value, _ = _drawn(positions, heads, kv_heads, jnp.bfloat16)(0)
    poisoned = value.at[0, 5, 0, 3].set(jnp.nan)
    ours = np.asarray(_assumed_attend(device, query, key, poisoned, spans), np.float32)
    theirs = np.asarray(template.reference(query, key, poisoned, spans), np.float32)
    np.testing.assert_array_equal(np.isnan(ours), np.isnan(theirs))
    assert np.isnan(theirs[0]).any() and not np.isnan(theirs[1]).any()
    alone = np.asarray(_assumed_attend(device, query, key, value, spans), np.float32)
    np.testing.assert_array_equal(ours[1].view(np.uint32), alone[1].view(np.uint32))


def _assumed_attend(device: str, query, key, value, spans):
    with generation.assumed(device):
        return attention_decode.attend(query, key, value, spans, interpret=True)


@INTERPRETS
@pytest.mark.parametrize("hook", HOOKS)
@pytest.mark.parametrize("call", CALLS)
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16], ids=["bfloat16", "float16"])
@pytest.mark.parametrize("device", ["sm90", "sm80"])
def test_a_decode_step_through_the_template_is_on_par_interpreted(call, hook, dtype, device):
    """A decode step through the template (`attention_decode.attend`) under the interpreter as an H100 or an
    A100 runs it, its few packed rows a 16-row tile of mma.sync's (row 4042), on par with the reference path
    over the same keys (check 1): packed, sliced and merged."""
    positions, heads, kv_heads = CALLS[call]
    spans = _spans(positions)

    def ours(query, key, value, sinks):
        with generation.assumed(device):
            return attention_decode.attend(query, key, value, spans, _hooks(hook, heads, positions, sinks),
                                           interpret=True)

    def reference(query, key, value, sinks):
        return template.reference(query, key, value, spans, _hooks(hook, heads, positions, sinks))

    assert_as_exact(jax.jit(ours), jax.jit(reference), _drawn(positions, heads, kv_heads, dtype),
                    f"template decode {device} {call} {hook} {jnp.dtype(dtype).name}", seeds=SEEDS)



@INTERPRETS
def test_an_absorbed_mla_decode_step_of_576_wide_keys_and_512_wide_values_is_on_par_interpreted():
    """An absorbed MLA decode step (DeepSeek's: 16 query heads over one latent key head, its key the latent
    512 and rotary 64, its value the latent 512; kernel-templates.md §A row 25) through the template as an
    H100 runs it, its heads padded to whole groups of 64 columns past 256 (`attention.head_width`), under the
    interpreter: on par with the reference path over the same keys (check 1)."""
    positions, heads, kv_heads = 1, 16, 1
    spans = _spans(positions)

    def ours(query, key, value, sinks):
        del sinks
        with generation.assumed("sm90"):
            return attention_decode.attend(query, key, value, spans, interpret=True)

    def reference(query, key, value, sinks):
        del sinks
        return template.reference(query, key, value, spans)

    draw = _drawn(positions, heads, kv_heads, jnp.bfloat16, 576, 512)
    assert_as_exact(jax.jit(ours), jax.jit(reference), draw, "template decode absorbed MLA", seeds=SEEDS)


@INTERPRETS
def test_an_absorbed_mla_decode_step_whose_values_are_its_keys_leading_columns_is_on_par_interpreted():
    """An absorbed MLA decode step whose values are its keys' leading 512 columns (`in_key`: the key tile
    loaded once, the value tile a view of its columns, `MosaicForward.in_key`) through the template as an
    H100 runs it, under the interpreter: on par with the reference path over the keys' columns sliced
    (check 1)."""
    positions, heads, kv_heads = 1, 16, 1
    spans = _spans(positions)

    def ours(query, key, value, sinks):
        del value, sinks
        with generation.assumed("sm90"):
            return attention_decode.attend(query, key, key[..., :512], spans, interpret=True, in_key=True)

    def reference(query, key, value, sinks):
        del value, sinks
        return template.reference(query, key, key[..., :512], spans)

    draw = _drawn(positions, heads, kv_heads, jnp.bfloat16, 576)
    assert_as_exact(jax.jit(ours), jax.jit(reference), draw, "template decode absorbed MLA in key",
                    seeds=SEEDS)


@INTERPRETS
def test_a_split_absorbed_mla_decode_step_whose_values_are_its_keys_leading_columns_is_on_par_interpreted():
    """An absorbed MLA decode step over 512 keys whose plan cuts its row into slices (`plan`: a split launch
    keeps its rows, `Kept.OUTPUT`), its values its keys' leading 512 columns read from the key's slot
    (`reads_in_key`), merged by their masses: on par with the reference path over the keys' columns sliced
    (check 1), as an H100 runs it under the interpreter."""
    keys = 512
    spans = template.spans(1, keys, 1, lengths=jnp.asarray([keys], jnp.int32))

    def ours(query, key):
        with generation.assumed("sm90"):
            return attention_decode.attend(query, key, key[..., :512], spans, interpret=True, in_key=True)

    def draw(seed: int):
        streams = jax.random.split(jax.random.key(seed), 2)
        return (jax.random.normal(streams[0], (1, 1, 16, 576), jnp.bfloat16),
                jax.random.normal(streams[1], (1, keys, 1, 576), jnp.bfloat16))

    def reference(query, key):
        return template.reference(query, key, key[..., :512], spans)

    assert_as_exact(jax.jit(ours), jax.jit(reference), draw, "template decode absorbed MLA in key split",
                    seeds=SEEDS)


@INTERPRETS
@pytest.mark.parametrize("width", [32, 192])
@pytest.mark.parametrize("device", ["sm90", "sm80"])
def test_a_decode_step_is_on_par_at_head_widths_interpreted(device, width):
    """A one-query decode step through the template at heads 32 wide and 192, which ride in zero-padded to
    256 (kernel-templates.md §A row 23; 64 the step's own, 256 the split launch's below), under the
    interpreter as an H100 or an A100 runs it: on par with the reference path over the same keys (check 1)."""
    positions, heads, kv_heads = CALLS["one-query"]
    spans = _spans(positions)

    def ours(query, key, value, sinks):
        del sinks
        with generation.assumed(device):
            return attention_decode.attend(query, key, value, spans, interpret=True)

    def reference(query, key, value, sinks):
        del sinks
        return template.reference(query, key, value, spans)

    draw = _drawn(positions, heads, kv_heads, jnp.bfloat16, width)
    assert_as_exact(jax.jit(ours), jax.jit(reference), draw, f"template decode {device} {width} wide",
                    seeds=SEEDS)


@INTERPRETS
def test_a_256_wide_decode_step_whose_split_launch_does_not_fit_is_on_par_interpreted():
    """A 256-wide bf16 decode step (Qwen3.5-0.8B's heads: 8 query heads over 2 key heads) on an H100, whose
    split launch's fp32 slices and the query's stage pass its shared memory, runs a slice a row through the
    template (`attention_decode.plan`), on par with the reference path (check 1)."""
    spans = _spans(1)

    def ours(query, key, value, sinks):
        del sinks
        with generation.assumed("sm90"):
            return attention_decode.attend(query, key, value, spans, None, interpret=True)

    def reference(query, key, value, sinks):
        del sinks
        return template.reference(query, key, value, spans, None)

    assert_as_exact(jax.jit(ours), jax.jit(reference), _drawn(1, 8, 2, jnp.bfloat16, 256),
                    "template decode 256 wide bfloat16", seeds=SEEDS)

@INTERPRETS
@pytest.mark.parametrize("page", [None, 4])
@pytest.mark.parametrize("call", ["one-query", "verify-four"])
def test_a_decode_rows_logits_past_the_folds_reach_walk_again_natural_interpreted(call, page):
    """A decode step whose first row's query is 2^16 times a drawn one (its folded logits past
    `algebra.FOLD_REACH`, where the fold's rounding moves its weights) and whose second row's is drawn: the
    first row walked again in the natural form by the masked launch (`algebra.walked_launches`), the second
    kept folded, each on par with the reference path (check 1); over a dense cache, and a paged pool of
    4-key pages, whose rewalk at its own tiles reads them through the one-key table its gather does
    (`attention_mosaic_forward.paged_reads`)."""
    positions, heads, kv_heads = CALLS[call]
    spans = _spans(positions)
    drawn = _drawn(positions, heads, kv_heads, jnp.bfloat16)
    table = None if page is None else jnp.asarray(
        np.random.default_rng(page).permutation(2 * KEYS // page).reshape(2, KEYS // page), jnp.int32)

    def pooled(cache):
        # [B, K, KV, D] as pools [KV, P, page, D] holding each row's keys at the pages its table names.
        pages = jnp.moveaxis(cache.reshape(2 * KEYS // page, page, *cache.shape[2:]), 2, 0)
        return pages[:, jnp.argsort(table.reshape(-1))]

    def draw(seed: int):
        query, key, value, _ = drawn(seed)
        return query.at[0].multiply(2.0 ** 16), key, value

    def ours(query, key, value):
        with generation.assumed("sm90"):
            if table is None:
                return attention_decode.attend(query, key, value, spans, interpret=True)
            return attention_decode.attend(query, pooled(key), pooled(value), spans, interpret=True,
                                           table=table)

    assert_as_exact(jax.jit(ours), jax.jit(lambda *operands: template.reference(*operands, spans)), draw,
                    f"template decode past the fold's reach {call} {'dense' if page is None else page}",
                    seeds=SEEDS)


@INTERPRETS
@pytest.mark.parametrize("far", [False, True], ids=["no-rewalk", "a-row-rewalked"])
@pytest.mark.parametrize("call", ["one-query", "verify-four"])
def test_every_output_a_decode_steps_launches_declare_is_written_interpreted(call, far):
    """Each launch of a decode step writes every output it declares on every path (NSNAN-2): its masked
    rewalk, whose programs that walk nothing write nothing, rewalks into the plain launch's outputs in place
    and declares none of its own, a merging walk's unsplit. Under JAX's NaN check (`jax.debug_nans`), which
    the Mosaic interpreter's unwritten outputs (NaN) trip, a step whose rows need no rewalk and one whose
    first row's query is 2^16 times a drawn one (past the fold's reach, rewalked natural) run op by op, so
    the check reads each launch's outputs, their result bit for bit the unchecked run's."""
    positions, heads, kv_heads = CALLS[call]
    spans = _spans(positions)
    query, key, value, _ = _drawn(positions, heads, kv_heads, jnp.bfloat16)(0)
    if far:
        query = query.at[0].multiply(2.0 ** 16)

    def ours(query, key, value):
        with generation.assumed("sm90"):
            return attention_decode.attend(query, key, value, spans, interpret=True)

    unchecked = np.asarray(ours(query, key, value))
    with jax.debug_nans(new_val=True):
        checked = np.asarray(ours(query, key, value))
    assert checked.tobytes() == unchecked.tobytes()


@INTERPRETS
@pytest.mark.parametrize("hook", ["plain", "sinks", "far"])
def test_a_decode_step_whose_programs_fill_the_gpu_finishes_each_row_in_its_walk_interpreted(hook):
    """A decode step of 17 rows over 8 key heads (136 programs, past an H100's 132 multiprocessors), so its
    work list takes one slice a row (`plan`) and the walk finishes each row itself from the initial state (a
    sink's), in the query's dtype; `far` its first row's query 2^16 times a drawn one, past the fold's reach,
    so the masked rewalk walks that row's query block alone: on par with the reference path
    (check 1)."""
    batch, heads, keys = 17, 8, 128
    lengths = jnp.asarray([keys - 7 * row for row in range(batch)], jnp.int32)
    spans = template.spans(1, keys, batch, lengths=lengths)

    def hooked(sinks):
        return template.Hooks(initial=template.sinks(sinks, batch, 1, 64)) if hook == "sinks" else None

    def ours(query, key, value, sinks):
        with generation.assumed("sm90"):
            return attention_decode.attend(query, key, value, spans, hooked(sinks), interpret=True)

    def draw(seed: int):
        streams = jax.random.split(jax.random.key(seed), 4)
        query = jax.random.normal(streams[0], (batch, 1, heads, 64), jnp.bfloat16)
        if hook == "far":
            query = query.at[0].multiply(2.0 ** 16)
        key, value = (jax.random.normal(stream, (batch, keys, heads, 64), jnp.bfloat16)
                      for stream in streams[1:3])
        return query, key, value, jax.random.normal(streams[3], (heads,), jnp.float32)

    assert_as_exact(jax.jit(ours), jax.jit(lambda q, k, v, s: template.reference(q, k, v, spans, hooked(s))),
                    draw, f"template decode of one slice a row {hook}", seeds=SEEDS)


@INTERPRETS
def test_a_split_decode_step_at_aligned_keys_beside_an_empty_row_is_on_par_interpreted():
    """A decode step on an H100 of two rows over 4096 keys, its four programs' keys cut into slices (`plan`)
    whose states merge by their masses (`algebra.merged_state`): the first row reads every key, whole key
    tiles and whole slices, the second none. On par with the reference path (check 1), the empty row zero.
    The merge at fp32's edges is the stability suite's (tests/test_attention_stability.py's template
    decode at fp32's edges)."""
    keys = 4096
    spans = template.spans(1, keys, 2, lengths=jnp.asarray([keys, 0], jnp.int32))
    draw = _operands((2, 1, 8, 64), keys)

    def ours(query, key, value):
        with generation.assumed("sm90"):
            return attention_decode.attend(query, key, value, spans, interpret=True)

    assert_as_exact(jax.jit(ours), jax.jit(lambda *operands: template.reference(*operands, spans)), draw,
                    "template decode split at aligned keys", seeds=SEEDS)


@INTERPRETS
@pytest.mark.parametrize("hook", ["plain", "sinks"])
@pytest.mark.parametrize("keys", [KEYS, 4096])
def test_a_decode_steps_state_resumed_over_the_rest_of_its_keys_is_its_whole_softmax_interpreted(keys, hook):
    """A decode step's keys cut in two (kernel-templates.md §A rows 19 and 63): the first part's state
    (`attend`'s `finished=False`, its softmax left unnormalized, an `Initial`), under its sinks where it has
    them, taken by the call over the rest as its initial state gives the step over every key: on par with
    the reference path over them all (check 1), at 256 keys and at 4096."""
    positions, heads, _ = CALLS["one-query"]
    lengths = jnp.asarray([keys - 56, keys // 3], jnp.int32)
    cut = lengths // 2
    whole, first = (template.spans(positions, keys, 2, lengths=ends) for ends in (lengths, cut))
    rest = template.spans(positions, keys, 2, lengths=lengths, starts=cut)
    operands = _operands((2, positions, heads, 64), keys)

    def draw(seed: int):
        return *operands(seed), jax.random.normal(jax.random.key(seed + 7), (heads,), jnp.float32)

    def ours(query, key, value, sinks):
        with generation.assumed("sm90"):
            state = attention_decode.attend(query, key, value, first, _hooks(hook, heads, positions, sinks),
                                            interpret=True, finished=False)
            return attention_decode.attend(query, key, value, rest, template.Hooks(initial=state),
                                           interpret=True)

    def reference(query, key, value, sinks):
        return template.reference(query, key, value, whole, _hooks(hook, heads, positions, sinks))

    assert_as_exact(jax.jit(ours), jax.jit(reference), draw,
                    f"template decode resumed over {keys} keys {hook}", seeds=SEEDS)


@INTERPRETS
def test_a_decode_steps_state_near_fp32s_largest_resumes_each_rows_own_output_interpreted():
    """A decode step's state (`attend`'s `finished=False`) of two rows over four keys of eight that weigh
    alike at a logit of 2^27, each row's values of one magnitude near fp32's largest (2^127 and 1.5 x 2^127,
    bf16), resumed over the other four, whose values are zeros: each row's output is half its values', its
    first four keys' mass against its last four's. The state's numerator, the output times a total of four,
    passes fp32's largest: held so, both rows resumed as +inf and their outputs were NaN; held a power of
    two down with its maximum as far up, the maximum rounded the power away at 2^27 and the first keys
    weighed an eighth of the rest (`attention_decode.resumable`)."""
    magnitudes = jnp.asarray([2.0**127, 1.5 * 2.0**127], jnp.float32)
    query = jnp.zeros((2, 1, 8, 64), jnp.bfloat16).at[..., 0].set(1.0)
    key = jnp.zeros((2, KEYS, 2, 64), jnp.bfloat16).at[..., 0].set(2.0**30)
    first_values = jnp.broadcast_to(magnitudes[:, None, None, None], (2, KEYS, 2, 64))
    value = jnp.where(jnp.arange(KEYS)[None, :, None, None] < 4, first_values, 0.0).astype(jnp.bfloat16)
    lengths, cut = jnp.asarray([8, 8], jnp.int32), jnp.asarray([4, 4], jnp.int32)
    first = template.spans(1, KEYS, 2, lengths=cut)
    rest = template.spans(1, KEYS, 2, lengths=lengths, starts=cut)
    with generation.assumed("sm90"):
        state = attention_decode.attend(query, key, value, first, None, interpret=True, finished=False)
        out = attention_decode.attend(query, key, value, rest, template.Hooks(initial=state), interpret=True)
    np.testing.assert_array_equal(np.asarray(out, np.float32),
                                  np.broadcast_to(np.asarray(magnitudes / 2)[:, None, None, None], out.shape))


@INTERPRETS
def test_a_resumed_decode_state_takes_its_steps_output_hook_at_one_slice_interpreted():
    """A decode step of 256 rows (one slice: its programs fill the device) resumed from a state over the
    first four of eight keys that weigh alike, values of one, under an output hook doubling its output:
    the step's output is two, the hook applied past the merge of the resumed state (`Initial.normalized`),
    where a hook of no parameters was dropped there and the output was one."""
    rows = 256
    query = jnp.zeros((rows, 1, 4, 64), jnp.bfloat16)
    key = jnp.zeros((rows, KEYS, 4, 64), jnp.bfloat16)
    value = jnp.ones((rows, KEYS, 4, 64), jnp.bfloat16)
    lengths, cut = jnp.full((rows,), 8, jnp.int32), jnp.full((rows,), 4, jnp.int32)
    first = template.spans(1, KEYS, rows, lengths=cut)
    rest = template.spans(1, KEYS, rows, lengths=lengths, starts=cut)
    with generation.assumed("sm90"):
        state = attention_decode.attend(query, key, value, first, None, interpret=True, finished=False)
        out = attention_decode.attend(query, key, value, rest, template.Hooks(
            initial=state, output=Hook(lambda o: o * 2.0)), interpret=True)
    np.testing.assert_array_equal(np.asarray(out, np.float32), 2.0)


def test_a_resumed_decode_state_is_no_walks_initial_state():
    """A resumed decode state holds its output (`Initial.normalized`), which no walk resumes from (each
    reads its numerator): the template's route, its reference, the Mosaic forward's certify form and the
    Triton forward's fitting refuse it, as the decode step alone merges it past its own walk; a numerator
    held read four keys of value two as one quarter of their output."""
    import functools

    from dew.nn.kernels.templates import attention_mosaic, attention_route, attention_triton

    query = jnp.zeros((1, 4, 2, 64), jnp.float32)
    key = value = jnp.zeros((1, 4, 2, 64), jnp.float32)
    spans = template.spans(4, 4, 1)
    state = template.Initial(jnp.zeros((1, 2, 4)), jnp.full((1, 2, 4), 4.0), jnp.full((1, 2, 4, 64), 2.0),
                             normalized=True)
    walks = (attention_route.attend, template.reference, attention_mosaic.certified,
             functools.partial(attention_triton.unfitted, certified=True))
    with generation.assumed("sm90"):
        for walk in walks:
            with pytest.raises(ValueError):
                walk(query, key, value, spans, template.Hooks(initial=state))


def test_a_resumed_decode_state_is_no_backwards_initial_state():
    """A resumed decode state (`Initial.normalized`) is refused by every standalone backward given the
    forward's residuals, as by every walk: the route's, the Triton mainloop's and its certified fitting, the
    portable one's and Mosaic GPU's binding. Taken, the state's gradient divided its output by its total a
    second time (an output of 6 over a total of 3 weighed as 2, train 1318's review)."""
    from dew.nn.kernels.templates import (
        attention_mosaic,
        attention_portable,
        attention_route,
        attention_triton,
    )

    query = key = value = dout = jnp.zeros((1, 4, 2, 64), jnp.float32)
    spans = template.spans(4, 4, 1)
    state = template.Initial(jnp.zeros((1, 2, 4)), jnp.full((1, 2, 4), 3.0), jnp.full((1, 2, 4, 64), 6.0),
                             normalized=True)
    residuals = (jnp.zeros((1, 2, 4)), jnp.ones((1, 2, 4)), None, None)
    hooks = template.Hooks(initial=state)
    tiles = attention_portable.Tiles(16, 16, 16)
    backwards = (
        lambda: attention_route.backward(query, key, value, spans, hooks, dout, residuals=residuals),
        lambda: attention_triton.backward(query, key, value, spans, hooks, dout, residuals=residuals),
        lambda: attention_portable.backward(tiles, query, key, value, spans, hooks, dout,
                                            residuals=residuals),
        lambda: attention_triton.backward_unfitted(query, key, value, spans, hooks, dout, residuals=residuals,
                                                   certified=True),
        lambda: attention_mosaic._backward_call(query, key, value, spans, hooks, dout, residuals, scale=None,
                                                natural=False, accurate=False))
    with generation.assumed("sm90"):
        for backward in backwards:
            with pytest.raises(ValueError):
                backward()


def test_a_resumed_states_output_spans_fp32s_range_as_its_finish_held_it():
    """A finished state whose output holds fp32's largest and least magnitudes in one row (2^127 and
    2^-149) over a total of four, resumed (`attention_decode.resumable`) and taken back as a merge takes an
    initial state (`algebra.initial_state`), gives its output back bit for bit: its numerator would pass
    fp32 at the large channel (2^129) and a power of two taken off it flushed the small one (2^-150)."""
    from dew.nn.kernels.templates import attention_algebra as algebra

    spec = algebra.finished_form(algebra.Forward(form="full", ids=False, keys=8, scale=1.0))
    output = jnp.asarray([[[2.0**127, 2.0**-149]]], jnp.float32)
    state = algebra.Partial(output, jnp.zeros((1, 1), jnp.float32), jnp.full((1, 1), 4.0, jnp.float32))
    resumed = attention_decode.resumable(spec, state)
    back = algebra.initial_state(spec, resumed, normalized=True)
    np.testing.assert_array_equal(np.asarray(back.acc), np.asarray(output))


@INTERPRETS
@pytest.mark.parametrize("page", [None, 16], ids=["rows", "pages"])
def test_a_work_lists_rows_read_their_cache_rows_in_place_interpreted(page):
    """A serving step's work list over a cache of two rows (kernel-templates.md §A row 33): three items, a
    decode row's one query and two pieces of four prompt queries each, reading cache rows 1, 0 and 1 where
    they lie: a dense cache's through `attend`'s `sources`, a paged pool's through each item's own page table
    (its row's, `table`, as the cache's work list gathers them: pages of 16 keys, each row's in an order of
    its own), each query over its own span (a piece's causal over its row's earlier keys, past its prefix):
    on par with the reference path over the rows gathered (check 1)."""
    sources = jnp.asarray([1, 0, 1], jnp.int32)
    starts = jnp.zeros((3, 4), jnp.int32)
    ends = jnp.asarray([[200, 0, 0, 0], [97, 98, 99, 100], [181, 182, 183, 184]], jnp.int32)
    spans = template.KeySpans(starts, ends, None, None, keys=KEYS)
    pages = None if page is None else KEYS // page
    tables = None if page is None else jnp.asarray(
        np.random.default_rng(page).permutation(2 * pages).reshape(2, pages), jnp.int32)

    def paged(rows: jax.Array) -> jax.Array:
        # The cache's rows `[2, K, KV, D]` laid in a pool `[KV, P, page, D]` at each row's pages.
        laid = rows.reshape(2 * pages, page, *rows.shape[2:]).transpose(2, 0, 1, 3)
        return jnp.zeros_like(laid).at[:, tables.reshape(-1)].set(laid)

    def ours(query, key, value):
        with generation.assumed("sm90"):
            if page is None:
                return attention_decode.attend(query, key, value, spans, interpret=True, sources=sources)
            return attention_decode.attend(query, paged(key), paged(value), spans, interpret=True,
                                           table=tables[sources])

    def reference(query, key, value):
        return template.reference(query, key[sources], value[sources], spans)

    assert_as_exact(jax.jit(ours), jax.jit(reference), _operands((3, 4, 8, 64), KEYS),
                    f"template decode over a work list's {'pages' if page else 'rows'}", seeds=SEEDS)


@INTERPRETS
@pytest.mark.parametrize("drafts", [4, 8, 16])
def test_a_verify_blocks_drafts_read_their_prefix_causally_interpreted(drafts):
    """A speculative verify block (`drafts` tokens a row written after the row's cache, `slots` theirs), each
    draft over its row's prefix and the drafts before it (causal over the block's slots) through the template
    decode, packed over 32 query heads on 8 key heads: on par with the reference path (check 1)."""
    held = jnp.asarray([200 - drafts, 37 - drafts], jnp.int32)
    slots = held[:, None] + jnp.arange(drafts, dtype=jnp.int32)[None]
    spans = template.spans(drafts, KEYS, 2, causal=True, lengths=held + drafts, slots=slots)
    drawn = _drawn(drafts, 32, 8, jnp.bfloat16)

    def ours(query, key, value):
        with generation.assumed("sm90"):
            return attention_decode.attend(query, key, value, spans, interpret=True)

    assert_as_exact(jax.jit(ours), jax.jit(lambda q, k, v: template.reference(q, k, v, spans)),
                    lambda seed: drawn(seed)[:3], f"template verify block of {drafts}", seeds=SEEDS)


def a_verify_block_through_the_route_is_on_par(drafts: int) -> None:
    """A GPU cell (tests/test_gpu_attention.py): a speculative verify block as the layer's cached call states
    it (`CrossDocuments` of its drafts over each row's held keys, causal), through the attention template's
    route, which runs a memory-bound inference call as the decode's work list (`attention_decode.serves`), 32
    query heads over 8 key heads, rows of their own lengths over 4096 keys: on par with the reference path
    over the same documents (check 1)."""
    from dew.nn.attention import scaled_dot_product_attention
    from dew.nn.documents import CrossDocuments

    rows, keys = 2, 4096
    held = jnp.asarray([keys - drafts, 1000], jnp.int32)
    documents = CrossDocuments(jnp.ones((rows, drafts), jnp.int32),
                               (jnp.arange(keys)[None] < (held + drafts)[:, None]).astype(jnp.int32))

    def attended(implementation):
        def attend(query, key, value):
            return scaled_dot_product_attention(query, key, value, causal=True, segment_ids=documents,
                                                implementation=implementation)
        return attend

    def draw(seed: int):
        streams = jax.random.split(jax.random.key(seed), 3)
        query = jax.random.normal(streams[0], (rows, drafts, 32, 128), jnp.bfloat16)
        key, value = (jax.random.normal(stream, (rows, keys, 8, 128), jnp.bfloat16) for stream in streams[1:])
        return query, key, value

    assert_as_exact(jax.jit(attended("template")), jax.jit(attended("reference")), draw,
                    f"verify block of {drafts} through the route", seeds=SEEDS)


def served_small_pages() -> tuple[int, ...]:
    """The pages fewer than a key tile holds whose decode this GPU's family serves
    (`attention_mosaic._pages_hold`): 16, 4 and 1 key where its loading warpgroup gathers a tile's rows
    through the table (`Transport.gathers_rows`), none where it does not."""
    family = bundles.of(generation.device_generation(), jnp.bfloat16)
    return (16, 4, 1) if family.gathers_rows else ()


def a_paged_decode_step_of_small_pages_is_on_par(page: int) -> None:
    """A GPU cell (tests/test_gpu_attention.py): a decode step over a paged pool of `page` keys a page, fewer
    than its key tile holds, compiled on this GPU, its tiles' rows gathered through the table by the loading
    warpgroup's cp.async copies (`gathered.Ring`, row 4772), 32 query heads over 8 key heads over 4096 keys
    in a shuffled table: on par with the reference path over the pool gathered dense (check 1)."""
    keys, kv_heads = 4096, 8
    spans = template.spans(1, keys, 2, lengths=jnp.asarray([keys, keys - 77], jnp.int32))
    per_row = keys // page
    table = jnp.asarray(np.random.default_rng(page).permutation(2 * per_row).reshape(2, per_row), jnp.int32)

    def gathered(pool):
        rows = pool[:, table]
        return jnp.transpose(rows, (1, 2, 3, 0, 4)).reshape(2, keys, kv_heads, pool.shape[-1])

    def draw(seed: int):
        streams = jax.random.split(jax.random.key(seed), 3)
        query = jax.random.normal(streams[0], (2, 1, 32, 128), jnp.bfloat16)
        pools = (jax.random.normal(stream, (kv_heads, 2 * per_row, page, 128), jnp.bfloat16)
                 for stream in streams[1:])
        return (query, *pools)

    def ours(query, key_pool, value_pool):
        return attention_decode.attend(query, key_pool, value_pool, spans, table=table)

    def reference(query, key_pool, value_pool):
        return template.reference(query, gathered(key_pool), gathered(value_pool), spans)

    assert_as_exact(jax.jit(ours), jax.jit(reference), draw, f"paged decode of {page}-key pages on a GPU",
                    seeds=SEEDS)


PAGE = 128
"""A paged cache's keys a page: whole key tiles of the derived decode launch's."""


@INTERPRETS
@pytest.mark.parametrize(("device", "page"),
                         [("sm90", PAGE), ("sm90", 16), ("sm90", 1), ("sm80", 16), ("sm80", 4), ("sm80", 1)])
@pytest.mark.parametrize("call", ["one-query", "verify-four"])
def test_a_paged_decode_step_through_the_template_is_on_par_interpreted(call, page, device):
    """A decode step over a paged cache (pools `[KV, P, page, D]` and their page table, each row's pages in an
    order of its own, some the other row's neighbours), read by the template where its table maps each key
    tile, on par with the reference path over the cache gathered dense (check 1): pages of whole key tiles,
    and pages of 16, 4 and 1 keys, the docs' (docs/concepts/inference.md), several to a key tile, its rows
    gathered through the table a key at a time (`attention_mosaic_forward.paged_reads`, row 4772): an
    H100's by its loading warpgroup's ring, an A100's by its one warpgroup's cp.async copies in its
    pipeline's groups (`mosaic_sm80._gathered_walk`)."""
    positions, heads, kv_heads = CALLS[call]
    spans = _spans(positions)
    per_row = KEYS // page
    table = jnp.asarray(np.random.default_rng(page).permutation(2 * per_row).reshape(2, per_row), jnp.int32)

    def gathered(pool):
        # [KV, P, page, D] at each row's pages: [B, K, KV, D].
        rows = pool[:, table]
        return jnp.transpose(rows, (1, 2, 3, 0, 4)).reshape(len(LENGTHS), KEYS, kv_heads, pool.shape[-1])

    def ours(query, key_pool, value_pool):
        with generation.assumed(device):
            return attention_decode.attend(query, key_pool, value_pool, spans, interpret=True, table=table)

    def reference(query, key_pool, value_pool):
        return template.reference(query, gathered(key_pool), gathered(value_pool), spans)

    def draw(seed: int):
        streams = jax.random.split(jax.random.key(seed), 3)
        query = jax.random.normal(streams[0], (len(LENGTHS), positions, heads, 64), jnp.bfloat16)
        pools = (jax.random.normal(stream, (kv_heads, 2 * per_row, page, 64), jnp.bfloat16)
                 for stream in streams[1:])
        return query, *pools

    assert_as_exact(jax.jit(ours), jax.jit(reference), draw, f"template paged decode {call} {page} {device}",
                    seeds=SEEDS)



QUANTIZED = [(device, page, stored) for stored in ("int8", "float8_e4m3fn")
             for device, page in (("sm90", "dense"), ("sm90", 16), ("sm80", "dense"), ("sm80", 16),
                                  ("sm80", 128))]
"""The quantized decode's devices, layouts and codes: dense, pages of 16 keys several to a key tile, their
rows gathered (`Transport.gathers_rows`), and on an A100 of whole key tiles; both int8 and FP8 on every GPU
(`mosaic.converts`, row 8549)."""


@INTERPRETS
@pytest.mark.parametrize(("device", "page", "stored"), QUANTIZED)
def test_a_decode_step_over_a_quantized_cache_is_on_par_interpreted(device, page, stored):
    """A decode step over a quantized cache (`dew.nn.kv_cache.quantize`: codes a token and head, each with its
    fp32 scale), its codes read and dequantized in the walk (`attention_decode.attend`'s `scales`, ledger
    3624), dense or paged (`QUANTIZED`), on par with the reference path over the keys and values the cache
    dequantizes them to (check 1), as an H100 and an A100 run it. The cache's codes and scales are drawn, the
    format's own (`quantize` of drawn keys and values): the truth reads them as they lie, each value its
    code times its scale."""
    from dew.nn.kv_cache import dequantize, quantize

    positions, heads, kv_heads = CALLS["one-query"]
    spans = _spans(positions)
    layout = "dense" if page == "dense" else "paged"
    page = KEYS if page == "dense" else page
    per_row = KEYS // page
    table = jnp.asarray(np.random.default_rng(7).permutation(2 * per_row).reshape(2, per_row), jnp.int32)

    def pooled(rows):
        # `[B, K, KV, ...]` as a pool `[KV, P, page, ...]` holding row `b`'s page `j` at `table[b, j]`.
        held = jnp.moveaxis(rows.reshape(len(LENGTHS), per_row, page, *rows.shape[2:]), 3, 0)
        pool = jnp.zeros((kv_heads, 2 * per_row, page, *rows.shape[3:]), rows.dtype)
        return pool.at[:, table].set(held)

    def ours(query, k_codes, k_scale, v_codes, v_scale):
        scales = tuple(jnp.swapaxes(scale, 1, 2) for scale in (k_scale, v_scale))
        with generation.assumed(device):
            if layout == "dense":
                return attention_decode.attend(query, k_codes, v_codes, spans, interpret=True, scales=scales)
            return attention_decode.attend(query, pooled(k_codes), pooled(v_codes), spans, interpret=True,
                                           table=table, scales=scales)

    def reference(query, k_codes, k_scale, v_codes, v_scale):
        return template.reference(query, dequantize(k_codes, k_scale, query.dtype),
                                  dequantize(v_codes, v_scale, query.dtype), spans)

    def draw(seed: int):
        query, key, value, _ = _drawn(positions, heads, kv_heads, jnp.bfloat16)(seed)
        return query, *quantize(key, stored), *quantize(value, stored)

    assert_as_exact(jax.jit(ours), jax.jit(reference), draw,
                    f"quantized decode {layout} {page} {stored} {device}", seeds=SEEDS)


def test_packing_a_calls_heads_into_its_rows_is_undone_by_unpacking():
    """A key head's group of query heads as its rows (`packed`) and back (`template.ungrouped_rows`): the
    query as it was, each packed row's keys its query's."""
    query = jax.random.normal(jax.random.key(0), (2, 3, 8, 16))
    spans = _spans(3)
    rows, row_spans, _ = attention_decode.packed(query, spans, 2)
    np.testing.assert_array_equal(template.ungrouped_rows(rows, 8), query)
    np.testing.assert_array_equal(row_spans.per_query().start,
                                  np.repeat(np.asarray(spans.per_query().start), 4, axis=1))



def test_a_packed_steps_spans_mean_its_rows_own_to_every_reader():
    """A decode step's packed rows keep one interval a position (`KeySpans.per_span`); every mainloop and the
    reference read them as each row's own: the reference over the packed rows and spans is the reference over
    the same rows with an interval a row, bit for bit (row 4487)."""
    query, key, value, _ = _drawn(4, 8, 2, jnp.float32)(0)
    rows, packed_spans, _ = attention_decode.packed(query, _spans(4), 2)
    np.testing.assert_array_equal(template.reference(rows, key, value, packed_spans),
                                  template.reference(rows, key, value, packed_spans.per_query()))

def _states(logits: jax.Array, values: jax.Array, cuts) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Each slice's softmax state over its keys `cuts[s]` to `cuts[s + 1]`, and one more of no key: its output
    normalized `[B, S, Q, KV, Dv]`, its maximum and denominator `[B, KV, S, Q]`."""
    outs, maxima, totals = [], [], []
    for lo, hi in itertools.pairwise(cuts):
        part = logits[..., lo:hi]
        peak = jnp.max(part, axis=-1)
        weights = jnp.exp(part - peak[..., None])
        total = jnp.sum(weights, axis=-1)
        outs.append(jnp.einsum("bqhk,bqhkd->bqhd", weights, values[..., lo:hi, :]) / total[..., None])
        maxima.append(peak)
        totals.append(total)
    outs.append(jnp.zeros_like(outs[0]))
    maxima.append(jnp.full_like(maxima[0], -jnp.inf))
    totals.append(jnp.zeros_like(totals[0]))
    maximum, total = (jnp.transpose(jnp.stack(x, axis=1), (0, 3, 1, 2)) for x in (maxima, totals))
    return jnp.stack(outs, axis=1), maximum, total


@COMPILES_ON_CPU
@pytest.mark.parametrize("page", [16, 4, 1])
def test_a_paged_decode_step_of_pages_smaller_than_its_tile_compiles_for_an_h100(page):
    """A decode step over a paged pool of `page` keys a page, fewer than its derived key tile holds, lowered
    and compiled for an H100 (compile-only, `kernel_compile.lowered_for`): its key tiles' rows gathered
    through the table by the loading warpgroup's cp.async copies (`gathered.Ring`, row 4772), whatever the
    page, where pages under 8 keys derived no launch before."""
    from kernel_compile import lowered_for

    keys, kv_heads, width = 4096, 8, 128
    spans = template.spans(1, keys, 2, lengths=jnp.asarray([keys, keys - 77], jnp.int32))
    query = jax.ShapeDtypeStruct((2, 1, 32, width), jnp.bfloat16)
    pool = jax.ShapeDtypeStruct((kv_heads, 2 * keys // page, page, width), jnp.bfloat16)
    table = jax.ShapeDtypeStruct((2, keys // page), jnp.int32)

    def decode(query, key_pool, value_pool, table):
        return attention_decode.attend(query, key_pool, value_pool, spans, table=table)

    # The Mosaic kernels the executable itself carries (`assembled.binaries`): a launch compiled, none
    # spilling.
    found = assembled.binaries(lowered_for(decode, (query, pool, pool, table), "sm90").compile())
    assert found and all(binary.spills == 0 for binary in found), found


@COMPILES_ON_CPU
def test_a_decode_step_launches_only_kernels_its_certification_held():
    """A decode step through the public route (`attention_route.attend`: fp16, one query row of four heads
    over two key heads 128 wide and 1024 keys) compiled for an H100 (`kernel_compile.compile_mosaic_kernels`):
    each Mosaic kernel it launches is one certification compiled alone and held (`report.certified`), its
    plan's config the one certified at the key tile its walk streams at, where the plan pinned its derived
    config uncertified and its kernels met the assembler first in the program's compile (row 7259)."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    from dew.nn.kernels.templates import attention_route

    query = jax.ShapeDtypeStruct((1, 1, 4, 128), jnp.float16)
    key = jax.ShapeDtypeStruct((1, 1024, 2, 128), jnp.float16)
    spans = template.spans(1, 1024, 1, causal=False)

    def decode(query, key, value):
        return attention_route.attend(query, key, value, spans, natural=False)

    with generation.assumed("sm90"):
        assert attention_decode.serves(query, key, None, natural=False, target=generation.gpu_target())
        case = KernelCase("decode", lambda: (decode, (query, key, key)))
        found = compile_mosaic_kernels([case], "sm90")
    assert not (failed := mosaic_failures(found)), "\n".join(failed)
    (record,) = found
    assert record["modules"] and record["modules"].items() <= record["certified"].items(), record


@COMPILES_ON_CPU
def test_a_decode_step_prepared_for_devices_no_run_tuned_is_certified():
    """The public decode step (`attention_route.attend`, the cell above's) lowered in a prepare for devices no
    run has tuned (`report.for_devices`, a compile ahead's) on a compile-only H100: its launch certified there
    as anywhere, its launch alone compiled to be read (`report.read_offline`), where XLA's want of a tuned
    result for the launch's own fusions ended its certification (`report.Untuned`)."""
    from kernel_compile import lowered_for

    from dew.nn.kernels.generation import compile_only_gpus
    from dew.nn.kernels.templates import attention_route

    query = jax.ShapeDtypeStruct((1, 1, 4, 128), jnp.float16)
    key = jax.ShapeDtypeStruct((1, 1024, 2, 128), jnp.float16)
    spans = template.spans(1, 1024, 1, causal=False)
    before = report.uncertified()
    with generation.assumed("sm90"), report.certifying(compile_only_gpus("sm90")[0]), report.for_devices():
        lowered_for(lambda q, k, v: attention_route.attend(q, k, v, spans, natural=False), (query, key, key),
                    "sm90")
    assert not set(report.uncertified()) - set(before), report.uncertified()
    assert any(found.op == "MosaicForward" for found in report.certifications()), report.certifications()


@pytest.mark.parametrize("slices", [1, 3, 7])
def test_merging_a_rows_slices_and_its_initial_state_is_its_whole_softmax(slices):
    """Slices of a row's keys, each its output normalized, its maximum and its denominator, a slice of no
    key, and a sink's initial state, merged (`merged`): on par with the row's softmax over every key and the
    sink at once (check 1)."""
    cuts = np.linspace(0, 40, slices + 1).astype(int)

    def ours(logits, values, sink):
        initial = template.Initial(sink, jnp.ones_like(sink), jnp.zeros((*sink.shape, values.shape[-1])))
        spec = algebra.Forward("spans", 40, ids=False, scale=1.0, natural=True, initial=True)
        out, maximum, total = _states(logits, values, cuts)
        state = algebra.merged_state(spec, out, maximum, total, initial)
        return algebra.finish(spec, state, initial=True, normalized=True)[0]

    def whole(logits, values, sink):
        scores = jnp.concatenate([logits, jnp.transpose(sink, (0, 2, 1))[..., None]], axis=-1)
        weights = jax.nn.softmax(scores, axis=-1)[..., :-1]
        return jnp.einsum("bqhk,bqhkd->bqhd", weights, values)

    def draw(seed: int):
        streams = jax.random.split(jax.random.key(seed), 3)
        return (4.0 * jax.random.normal(streams[0], (2, 3, 1, 40)),
                jax.random.normal(streams[1], (2, 3, 1, 40, 8)), jax.random.normal(streams[2], (2, 1, 3)))

    assert_as_exact(jax.jit(ours), jax.jit(whole), draw, f"decode merge over {slices} slices", seeds=SEEDS)



RESUMED = 1e38
"""A resumed state's maximum, finite, past fp32's largest in a raw fold's units at a scale of 1/8 (fp16's
decode 64 wide leaves its scale whole, `template.prescale`): audit A6's, infinite there unsaturated."""


def _resumed(maximum: jax.Array, acc: jax.Array) -> template.Initial:
    """A resumed state of `maximum` `[..., Q]` whose first row's is `RESUMED`, a denominator of 1 and the
    sums `acc`."""
    return template.Initial(maximum.at[0].set(RESUMED), jnp.ones_like(maximum), acc)


@pytest.mark.parametrize("slices", [1, 3])
def test_merging_slices_with_a_resumed_maximum_past_fp32s_largest_in_raw_units_keeps_its_share(slices):
    """A row's slices in the raw fold's units fp16's decode takes (`attention_decode._spec`: a scale of 1/8),
    merged in XLA with a resumed state whose first row's maximum is `RESUMED` (`algebra.merged_state`): the
    state's maximum saturates there as the walk's does (`algebra.initial_state`, `raw_maximum`), so that row
    is the state's own sums over its denominator, every row on par with the whole softmax (check 1)."""
    cuts = np.linspace(0, 40, slices + 1).astype(int)
    spec = attention_decode._spec(jnp.zeros((2, 3, 1, 64), jnp.float16), template.spans(3, 40, 2),
                                  template.Hooks(), scale=None)

    def ours(logits, values, maximum, acc):
        out, peak, total = _states(logits, values, cuts)
        state = algebra.merged_state(spec, out, peak / spec.scale, total, _resumed(maximum, acc))
        return algebra.finish(spec, state, initial=True, normalized=True)[0]

    def whole(logits, values, maximum, acc):
        initial = _resumed(maximum, acc)
        m = jnp.transpose(initial.m, (0, 2, 1))
        peak = jnp.maximum(jnp.max(logits, axis=-1), m)
        weights, own = jnp.exp(logits - peak[..., None]), jnp.exp(m - peak)
        state = own[..., None] * jnp.swapaxes(initial.acc, 1, 2)
        sums = jnp.einsum("bqhk,bqhkd->bqhd", weights, values) + state
        return sums / (jnp.sum(weights, axis=-1) + own)[..., None]

    def draw(seed: int):
        streams = jax.random.split(jax.random.key(seed), 4)
        return (4.0 * jax.random.normal(streams[0], (2, 3, 1, 40)),
                jax.random.normal(streams[1], (2, 3, 1, 40, 8)), jax.random.normal(streams[2], (2, 1, 3)),
                jax.random.normal(streams[3], (2, 1, 3, 8)))

    assert_as_exact(jax.jit(ours), jax.jit(whole), draw,
                    f"decode merge resumed past raw fp32 over {slices} slices", seeds=SEEDS)


@INTERPRETS
def test_a_split_decode_step_resumed_past_fp32s_largest_in_raw_units_keeps_its_share_interpreted():
    """An fp16 decode step on an H100 of two rows over 4096 keys, 8 query heads over 2 and 64 wide, its keys
    cut into slices merged with a resumed state whose first row's maximum is `RESUMED`
    (`Hooks.initial`): the state's maximum saturates in the raw fold's units (`algebra.initial_state`), so
    that row is the state's own sums over its denominator, on par with the reference path (check 1)."""
    keys = 4096
    spans = template.spans(1, keys, 2, lengths=jnp.asarray([keys, keys // 2], jnp.int32))

    def draw(seed: int):
        streams = jax.random.split(jax.random.key(seed), 5)
        query = jax.random.normal(streams[0], (2, 1, 8, 64), jnp.float16)
        key, value = (jax.random.normal(stream, (2, keys, 2, 64), jnp.float16) for stream in streams[1:3])
        return (query, key, value, jax.random.normal(streams[3], (2, 8, 1)),
                jax.random.normal(streams[4], (2, 8, 1, 64)))

    def ours(query, key, value, maximum, acc):
        with generation.assumed("sm90"):
            return attention_decode.attend(query, key, value, spans,
                                           template.Hooks(initial=_resumed(maximum, acc)), interpret=True)

    def reference(query, key, value, maximum, acc):
        return template.reference(query, key, value, spans, template.Hooks(initial=_resumed(maximum, acc)))

    assert_as_exact(jax.jit(ours), jax.jit(reference), draw, "template decode split resumed past raw fp32",
                    seeds=SEEDS)


@INTERPRETS
@pytest.mark.parametrize("held", ["dense", "sources", "paged"])
@pytest.mark.parametrize("large", [False, True], ids=["ordinary", "near-bf16s-largest"])
def test_a_decode_step_given_its_caches_value_bounds_walks_as_one_given_none_interpreted(held, large):
    """A bf16 decode step over a cache that states each key's largest value magnitude (`attend`'s `bounds`,
    head-major as a cache's scales: dense, a work list's rows by `sources`, a paged pool's by its table): its
    output is bit for bit the step's given no bounds, which takes them from the values it reads and so
    launches the same walk: the unscaled one on ordinary values, the scaled one on values near bf16's
    largest."""
    positions, heads, kv_heads = CALLS["one-query"]
    spans = _spans(positions)
    query, key, value, _ = _drawn(positions, heads, kv_heads, jnp.bfloat16)(0)
    if large:
        value = value.at[0, :8].set(jnp.finfo(jnp.bfloat16).max)
    bound = jnp.moveaxis(jnp.max(jnp.abs(value.astype(jnp.float32)), axis=-1), 1, 2)  # [B, KV, K]
    page = 4
    table = jnp.asarray(np.random.default_rng(page).permutation(2 * KEYS // page).reshape(2, KEYS // page),
                        jnp.int32)
    order = jnp.argsort(table.reshape(-1))

    def pooled(cache):
        # [B, K, ...] as pools [KV, P, page, ...] holding each row's keys at the pages its table names.
        pages = jnp.moveaxis(cache.reshape(2 * KEYS // page, page, *cache.shape[2:]), 2, 0)
        return pages[:, order]

    def attended(bounds):
        with generation.assumed("sm90"):
            if held == "paged":
                pooled_bounds = None if bounds is None else pooled(jnp.moveaxis(bounds, 1, 2))
                return attention_decode.attend(query, pooled(key), pooled(value), spans, interpret=True,
                                               table=table, bounds=pooled_bounds)
            if held == "sources":
                # Batch row b reads cache row sources[b]: the cache holds the rows in the other order.
                sources = jnp.asarray([1, 0], jnp.int32)
                rows = jnp.argsort(sources)
                held_bounds = None if bounds is None else bounds[rows]
                return attention_decode.attend(query, key[rows], value[rows], spans, interpret=True,
                                               sources=sources, bounds=held_bounds)
            return attention_decode.attend(query, key, value, spans, interpret=True, bounds=bounds)

    np.testing.assert_array_equal(np.asarray(jax.jit(attended)(bound)), np.asarray(jax.jit(attended)(None)))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["float32", "bfloat16"])
def test_a_walk_its_compile_refused_plans_a_config_its_launch_pins(dtype):
    """A decode step's walk whose launch a compile refused (its kernel spilled, `report.refuse_launch`) plans
    that walk again at a config its own launch pins: each walk's plan is its launched op's (`WALKS`), so the
    refusal of the scaled walk's launch steps it down, never pinning it to the config refused (row 5420)."""
    # 24 rows of 8 key heads, a step's 192 programs (each a key head's packed query rows), past an H100's
    # 132: one slice whatever its dtype.
    query = jnp.zeros((24, 1, 16, 64), dtype)
    key = value = jnp.zeros((24, KEYS, 8, 64), dtype)
    spans = template.spans(1, KEYS, 24, lengths=jnp.full((24,), KEYS, jnp.int32))
    with generation.assumed("sm90"):
        work, rows, row_spans, whole = attention_decode.planned(query, key, value, spans, template.Hooks())
        walk = "scaled" if "scaled" in work else "plain"
        assert work[walk].slices == 1
        op, args = attention_mosaic._forward_call(rows, key, value, row_spans, whole, scale=None,
                                                  natural=False, masked=True, subtracted=True,
                                                  **attention_decode.WALKS[walk]._asdict())
        launch = op.bind(*args, residuals=template.Kept.NONE, interpret=False)
        refused = op.completed(launch, work[walk].config)
        with report.refusals():
            report.refuse_launch(op, launch, refused, {"attention_forward"})
            again = attention_decode.planned(query, key, value, spans, template.Hooks())[0][walk]
            with tuning.pinned({attention_mosaic.MosaicForward: again.config}):
                chosen = op.taken(launch)
    assert chosen != refused


@pytest.mark.parametrize("lengths", [False, True], ids=["full", "lengths"])
def test_a_split_walk_its_compile_refused_plans_and_traces_another_config(lengths):
    """A split decode step whose priced walk a compile refused (its kernel spilled, `report.refuse_launch`)
    plans another config of its slices and its step traces, a cache's lengths or none: the priced tile is
    completed and legal at the call's own spans before it is certified, a refused one falling to its
    streaming walk's own config at those slices, never pinned as refused (the reviews of b11a701f4f0 and
    97072e2e343: a refused priced tile, priced over a cache's whole extent, raised `Unpinnable`)."""
    keys = 4096
    query = jnp.zeros((1, 1, 32, 64), jnp.bfloat16)
    key = value = jnp.zeros((1, keys, 8, 64), jnp.bfloat16)
    spans = template.spans(1, keys, 1, lengths=jnp.full((1,), keys, jnp.int32) if lengths else None)
    with generation.assumed("sm90"):
        work, rows, row_spans, _ = attention_decode.planned(query, key, value, spans, template.Hooks())
        plan = work["plain"]
        assert plan.slices > 1, plan
        op, args = attention_mosaic._forward_call(rows, key, value, row_spans, template.Hooks(), scale=None,
                                                  natural=False, out_dtype=jnp.float32, masked=True,
                                                  splits=plan.slices, subtracted=True)
        launch = op.bind(*args, residuals=template.Kept.OUTPUT, interpret=False)
        refused = op.completed(launch, plan.config)
        with report.refusals():
            report.refuse_launch(op, launch, refused, {"attention_forward_split"})
            again = attention_decode.planned(query, key, value, spans, template.Hooks())[0]["plain"]
            jax.make_jaxpr(lambda q, k, v: attention_decode.attend(q, k, v, spans))(query, key, value)
    assert op.completed(launch, again.config) != refused or again.slices != plan.slices, (plan, again)


@COMPILES_ON_CPU
def test_a_decode_steps_price_reads_its_keys_and_a_paged_pool_walks_no_fewer_steps():
    """A decode step's price (`attention_decode.priced_step`: its launches' work as their emission counts it,
    `cost.seconds` its seconds) for one row of 8 query heads over 2 key heads 128 wide in bf16 on an H100:
    twice the keys read more device bytes, walk at least as many steps and take longer, and the same keys
    from a paged pool of 64-key pages walk no fewer steps than from a dense cache, every step's tile within a
    page (row 7420: a server's choice between the two reads what each costs)."""
    def priced(keys, page_size=None):
        return attention_decode.priced_step(generation.gpu_target("sm90"), rows=1, queries=1, keys=keys,
                                            heads=8, kv_heads=2, width=128, dtype=jnp.bfloat16,
                                            page_size=page_size)

    dense, longer, paged = priced(4096), priced(8192), priced(4096, page_size=64)
    assert dense is not None and longer is not None and paged is not None
    assert dense.kernels >= 1 and dense.device_bytes > 0
    assert longer.device_bytes > dense.device_bytes and longer.span >= dense.span
    assert paged.span >= dense.span
    # Its seconds on the device (`cost.seconds`), the server's comparison, grow with the keys it reads.
    target = generation.gpu_target("sm90")
    assert cost.seconds(longer, target) > cost.seconds(dense, target) > 0


SPLIT_SPANS = {
    "ten tiles in three": ("full", 64, 640, 1, "wgmma", 3, 4, (10, 10)),
    "a causal pair in two": ("causal", 256, 256, 2, "wgmma", 2, 3, (10, 10)),
}
"""Each split launch's spans' form, queries, keys, consumers a program and family, slices, the span its
longest program walks, and its consumers' steps a head beside its unsplit launch's: ten 64-key tiles a query
tile, split three ways, slices of 3, 3 and 4 tiles, ten steps either way; a causal 256 by 256 call at 64 by
64 tiles, two consumers a program, two slices, whose last pair's consumers reach 3 and 4 tiles, their
second slices [1, 3) and [2, 4), which its producer streams as one, [1, 4) (the base review of 71686face,
F3), each consumer walking its own slices, 10 steps either way. A forward on a layout whose consumers feed
their own rings (`TileWalk.common`) takes one consumer (`MosaicForward.legal`, row 2599), so no split
launch's holders walk a common envelope."""


def _split_work(form: str, queries: int, keys: int, consumers: int, family: str, splits: int):
    """The work of a launch over `form`'s spans of `queries` and `keys` at 64 by 64 tiles, `consumers` a
    program on `family`'s walk, in `splits` slices, traced for an H100 as its costed launch counts it
    (`emission.work_of`): its walk's launch, beside the guard's natural rewalk
    (`attention_mosaic_forward.NATURAL_NAME`)."""
    query = jnp.zeros((1, queries, 2, 64), jnp.bfloat16)
    key = jnp.zeros((1, keys, 2, 64), jnp.bfloat16)
    spans = template.spans(queries, keys, 1, causal=form == "causal")
    with generation.assumed("sm90"):
        op, args = attention_mosaic._forward_call(query, key, key, spans, None, scale=None, natural=False,
                                                  splits=splits, out_dtype=jnp.float32)
        tiles = attention_mosaic.Tiles(64, 64, 2, consumers, family=family)
        with tuning.pinned({attention_mosaic.MosaicForward: tiles}):
            # A split launch's slices keep their rows, which the step's merge reads (`attention_decode`).
            kept = template.Kept.OUTPUT if splits > 1 else template.Kept.NONE
            traced = jax.make_jaxpr(lambda *operands: op(*operands, residuals=kept, interpret=False))(*args)
    (work,) = [emission.work_of(equation) for equation in programs.equations(traced.jaxpr)
               if equation.primitive in programs.NATIVE_KERNELS
               and attention_mosaic_forward.NATURAL_NAME not in equation.params["name"]]
    assert work is not None
    return work


@pytest.mark.parametrize(("form", "queries", "keys", "consumers", "family", "splits", "span", "steps"),
                         SPLIT_SPANS.values(), ids=SPLIT_SPANS.keys())
def test_a_split_launchs_span_is_its_longest_slice(form, queries, keys, consumers, family, splits, span,
                                                   steps):
    """A split launch (`MosaicForward.splits`, `_Program._sliced`) traced for an H100 at 64 by 64 tiles: its
    work as its costed launch counts it (`_split_work`) spans its longest program's walk
    (`cost.Stated.longest`), where its programs' mean was 10/3 for ten tiles in three (row 8844; the base
    review of 31b5fe85b, F2), and issues its consumers' steps as the kernel slices and streams them: its
    walk's tensor products its unsplit launch's by their ratio to its steps
    (`attention_mosaic_program.walks`), and every other unit its body issues (`cost.ISSUED_UNITS`, its
    integer work among them, row 10793) no less than that share, each slice its own epilogue, staging and
    index arithmetic beside its walk: its exponentials too, its epilogue's normalizing division a MUFU each
    (`emission.MOSAIC_DIVISION`); a causal launch's integer work, its diagonal's masks, counted at all (row
    12008: a share of none holds of none)."""
    split = _split_work(form, queries, keys, consumers, family, splits)
    unsplit = _split_work(form, queries, keys, consumers, family, 1)
    assert split.span == span
    # Each side of a share counted at all: a launch whose work dropped a unit held 0 >= 0 (row 12012).
    assert unsplit.flops > 0 and unsplit.exponentials > 0
    assert form != "causal" or min(split.integer_operations, unsplit.integer_operations) > 0
    for unit in cost.ISSUED_UNITS:
        shares = getattr(split, unit) * steps[1], getattr(unsplit, unit) * steps[0]
        assert shares[0] == shares[1] if unit == "flops" else shares[0] >= shares[1], unit


def test_a_decode_step_of_no_split_is_priced_at_its_keys_reads():
    """A decode step of 256 rows of one query, 8 heads over 2 key heads 128 wide in bf16, on an H100 over
    4096 keys (`attention_decode.priced_step`): its programs, 512, pass the multiprocessors' count, so no
    split merge's reads hide its walk's, which price at least its keys and values read once, where its
    launch, packed into spans of every key, stated none and the step's price was Work() (the base review of
    56a716c95, F3)."""
    rows, keys, kv_heads, width = 256, 4096, 2, 128
    priced = attention_decode.priced_step(generation.gpu_target("sm90"), rows=rows, queries=1, keys=keys,
                                          heads=8, kv_heads=kv_heads, width=width, dtype=jnp.bfloat16)
    assert priced is not None
    assert priced.device_bytes >= 2 * rows * keys * kv_heads * width * jnp.dtype(jnp.bfloat16).itemsize
    # Non-deciding fallback (`natural_form=True`, row 8813 F2) prices only the common bounded arm, not both.
    sm90 = generation.gpu_target("sm90")
    with generation.assumed(sm90):
        q90 = jnp.zeros((1, 1, 8, width), jnp.bfloat16)
        k90 = jnp.zeros((1, keys, kv_heads, width), jnp.bfloat16)
        base_plan, *_ = attention_decode.planned(q90, k90, k90, attention_decode.key_spans(1, keys, 1),
                                                 template.Hooks())
        assert base_plan is not None
        non_deciding = dataclasses.replace(base_plan["plain"].config, natural_form=True)
        with tuning.pinned({attention_mosaic.MosaicForward: non_deciding}):
            plan90, *_ = attention_decode.planned(q90, k90, k90, attention_decode.key_spans(1, keys, 1),
                                                  template.Hooks())
            fallback = attention_decode.priced_step(sm90, rows=1, queries=1, keys=keys, heads=8,
                                                    kv_heads=kv_heads, width=width, dtype=jnp.bfloat16)
    assert plan90 is not None and "bounded" in plan90 and "scaled" in plan90
    expected_kernels = 1 + int(plan90["bounded"].slices > 1)
    assert fallback is not None and fallback.kernels == expected_kernels
    assert fallback.device_bytes >= 2 * keys * kv_heads * width * jnp.dtype(jnp.bfloat16).itemsize


def test_a_walk_on_a_gpu_of_no_issue_interval_has_no_issued_price():
    """A one-row decode walk over 32768 keys, 32 heads over 8 key heads 64 wide in bf16, on an RTX PRO 6000,
    whose multiprocessor states no issue interval: its issued price (`MosaicForward.price_config`'s
    `issued`, the decode planner's) is unknown, where priced at its bytes alone it was level over every slice
    count and the planner took two of them, 16 programs on 188 multiprocessors (M-NA-35-decodec-g4: 281 of
    a step's 288 us); its unissued price still its own."""
    target = generation.gpu_target("NVIDIA RTX PRO 6000 Blackwell Server Edition")
    query = jax.ShapeDtypeStruct((1, 1, 32, 64), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((1, 32768, 8, 64), jnp.bfloat16)
    spans = attention_decode.key_spans(1, 32768, 1)
    op, ba = attention_decode._bound(query, key, key, spans, None, scale=None, table=None, sources=None,
                                     scales=None, splits=4, chosen=attention_decode.WALKS["plain"],
                                     in_key=False)
    with generation.assumed(target):
        tiles = op.derived(ba, target)
        assert tiles is not None
        assert op.price_config(ba, tiles, target, issued=True) is None
        assert op.price_config(ba, tiles, target) is not None


@COMPILES_ON_CPU
@pytest.mark.parametrize("width", [128, 256])
@pytest.mark.parametrize("generation_name", ["sm80", "sm90", "sm100", "sm120"])
def test_an_inference_forward_compiles_spill_free_whether_or_not_it_decides(generation_name, width):
    """A bf16 inference forward compiled for a compile-only A100, H100, B200 and RTX PRO 6000
    (`kernel_compile.compile_mosaic_kernels`): every kernel spills nothing, whether its launch decides its
    scaled walk itself (`MosaicForward.decided`: the scaled walk's registers held at the unscaled walk's
    config, priced at them) or takes the fallback, the scaled op its own launch (row 5274)."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    query = jax.ShapeDtypeStruct((1, 256, 8, width), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((1, 2048, 2, width), jnp.bfloat16)
    spans = template.spans(256, 2048, 1, causal=False)

    def forward(q, k, v):
        return attention_mosaic.forward(q, k, v, spans, natural=False)

    with generation.assumed(generation_name):
        found = compile_mosaic_kernels([KernelCase("inference", lambda: (forward, (query, key, key)))],
                                       generation_name)
    assert not (failed := mosaic_failures(found)), "\n".join(failed)


@COMPILES_ON_CPU
def test_a_split_decode_steps_price_includes_the_native_merges_bytes_and_span():
    """A split decode step on an H100 (`attention_decode.priced_step`: 1 row, 1 query, 8 heads over 2 key
    heads 128 wide over 8192 keys) prices its native `merged_padded` launch independently of
    `merged.launch_work` (rows 8845, 10533): its kernel count, span (`3 * chunks`) and device bytes equal the
    split walk's plus the emitted `get`/`swap` traffic of `merged._launch`'s traced Pallas body."""
    from test_merged import _traced_merge_bytes

    from dew.algebra import windows
    from dew.nn.kernels.templates import merged

    target = generation.gpu_target("sm90")
    priced = attention_decode.priced_step(target, rows=1, queries=1, keys=8192, heads=8, kv_heads=2,
                                          width=128, dtype=jnp.bfloat16)
    assert priced is not None and priced.kernels >= 2.0
    query = jax.ShapeDtypeStruct((1, 1, 8, 128), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((1, 8192, 2, 128), jnp.bfloat16)
    spans = template.spans(1, 8192, 1, causal=False)
    with generation.assumed(target):
        traced = jax.make_jaxpr(lambda q, k, v: attention_decode.attend(
            q, k, v, spans, None, scale=None, interpret=False, table=None, sources=None,
            scales=None, bounds=None, finished=True, in_key=False))(query, key, key)
    (merge_eq,) = [eq for eq in programs.equations(traced.jaxpr) if eq.primitive is merged.merged_padded_p]
    padded_s = jax.ShapeDtypeStruct(merge_eq.invars[0].aval.shape, merge_eq.invars[0].aval.dtype)
    stacked_s = jax.ShapeDtypeStruct(merge_eq.invars[1].aval.shape, merge_eq.invars[1].aval.dtype)
    queries, width = merge_eq.params["queries"], merge_eq.params["width"]
    out_dtype, out_heads = merge_eq.params.get("out_dtype"), merge_eq.params.get("out_heads")
    block = merged._channel_block(merge_eq.params["units"], padded_s, stacked_s, queries, width, target,
                                  out_dtype=out_dtype, out_heads=out_heads)
    emitted_bytes = _traced_merge_bytes(merge_eq.params["units"], padded_s, stacked_s, queries, width, block,
                                        out_dtype=out_dtype, out_heads=out_heads)
    slices = padded_s.shape[1]
    merge_span = float(3 * -(-slices // windows.chunk_width(slices)))
    walk_launches = [w for eq in programs.equations(traced.jaxpr)
                     if eq.primitive in programs.NATIVE_KERNELS
                     and (w := emission.work_of(eq)) is not None]
    assert merge_span >= 3.0 and emitted_bytes > 0
    assert priced.kernels == len(walk_launches) + 1.0
    assert priced.span == sum(w.span for w in walk_launches) + merge_span
    assert priced.device_bytes == sum(w.device_bytes for w in walk_launches) + emitted_bytes


@INTERPRETS
def test_a_decode_step_resuming_an_initial_state_near_fp32s_largest_keeps_its_mass():
    """A decode step through the public entry (`attention_decode.attend`) resuming an `Initial` state whose
    total is `1.5 * 2^126` (rows 6904, 6928, 6984): `initial_state` divides through `means.finished` without
    XLA:CPU reciprocal underflow so the resumed state's value is preserved instead of flushing to zero."""
    positions, heads, kv_heads = CALLS["one-query"]
    spans = _spans(positions)
    query, _, value, _ = _drawn(positions, heads, kv_heads, jnp.bfloat16)(0)
    total = jnp.full((query.shape[0], heads, positions), np.float32(1.5 * 2.0 ** 126), jnp.float32)
    m = jnp.full((query.shape[0], heads, positions), np.float32(20.0), jnp.float32)
    acc = total[..., None] * jnp.ones((query.shape[0], heads, positions, value.shape[-1]), jnp.float32)
    def ours(q, k, v, im, it, ia):
        with generation.assumed("sm90"):
            hooks = template.Hooks(initial=template.Initial(im, it, ia))
            return attention_decode.attend(q, k, v, spans, hooks, interpret=True)

    def reference(q, k, v, im, it, ia):
        return template.reference(q, k, v, spans, template.Hooks(initial=template.Initial(im, it, ia)))

    def draw(seed: int):
        q, k, v, _ = _drawn(positions, heads, kv_heads, jnp.bfloat16)(seed)
        return q, k, v, m, total, acc
    assert_as_exact(jax.jit(ours), jax.jit(reference), draw,
                    "decode step resuming initial state near fp32s largest", seeds=SEEDS)


def test_one_row_decode_split_k_keeps_sm_utilization_and_per_slice_stage_depth():
    """Row 5274: one-row decode (`B=1 K=4096 d64` on sm90) computes split-K `longest` from the base 2-stage
    prologue rather than the un-split 3-stage depth, keeping SM utilization high (`slices >= 12`), and
    `MosaicForward.streaming` deepens stages from each slice's own tile count (`2` for short slices, `3` for
    long `B=8 K=4096 d128` slices)."""
    with generation.assumed("sm90"):
        q1 = jax.ShapeDtypeStruct((1, 1, 32, 64), jnp.bfloat16)
        k1 = jax.ShapeDtypeStruct((1, 4096, 8, 64), jnp.bfloat16)
        work1, *_ = attention_decode.planned(
            jnp.zeros(q1.shape, q1.dtype), k1, k1, template.spans(1, 4096, 1, causal=False),
            template.Hooks())
        assert work1 is not None
        plan1 = work1["plain"]
        import math
        three_stage_cap = math.ceil(4096 / plan1.config.block_k) // 6
        assert plan1.slices > three_stage_cap, (plan1, three_stage_cap)
        tiles_per_slice = math.ceil(math.ceil(4096 / plan1.slices) / plan1.config.block_k)
        assert plan1.config.stages <= max(2, tiles_per_slice // 2), (plan1, tiles_per_slice)

        q8 = jax.ShapeDtypeStruct((8, 1, 32, 128), jnp.bfloat16)
        k8 = jax.ShapeDtypeStruct((8, 4096, 8, 128), jnp.bfloat16)
        work8, *_ = attention_decode.planned(
            jnp.zeros(q8.shape, q8.dtype), k8, k8, template.spans(1, 4096, 8, causal=False),
            template.Hooks())
        assert work8 is not None and work8["plain"].config.stages > plan1.config.stages, (work8, plan1)


@INTERPRETS
def test_an_fp32_decode_steps_programs_are_its_walks_grid_and_its_step_is_on_par_interpreted():
    """An fp32 decode step of 8 rows of 16 query heads over 8 key heads and 4096 keys on an H100: at an
    explicit tile its planner's programs a slice (`attention_decode.launch_programs`, a row's key head's
    packed query rows each, its parts stacked within their rows) times its slices are its walk's launch
    grid, 64 a slice, short of the device's 132, where counted with its parts the step filled it on paper
    (2026-11-26 audit); and the step as it plans itself is on par with the reference path (check 1)."""
    keys, heads, kv_heads = 4096, 16, 8
    spans = template.spans(1, keys, 8, lengths=jnp.full((8,), keys, jnp.int32))
    query = jnp.zeros((8, 1, heads, 64), jnp.float32)
    key = jnp.zeros((8, keys, kv_heads, 64), jnp.float32)
    # The walk's columns its launch splits the value's 64 into (an fp32 step's parts' launch's).
    tiles = attention_mosaic.Tiles(64, 64, 2, 1, family="wgmma", columns=32)
    with generation.assumed("sm90"), tuning.pinned({attention_mosaic.MosaicForward: tiles}):
        work, rows, *_ = attention_decode.planned(query, key, key, spans, template.Hooks())
        traced = jax.make_jaxpr(lambda q, k, v: attention_decode.attend(q, k, v, spans, interpret=True))(
            query, key, key).jaxpr
    # Each walk launch's programs, its mesh's grid (a Mosaic GPU launch's `mpmd_map`, `mesh.shape`'s grid
    # axes, its threads apart).
    walks = [math.prod(size for axis, size in equation.params["meshes"][0].shape.items()
                       if axis not in ("wg", "warp", "lane"))
             for equation in equations(traced) if "attention_forward" in str(equation.params.get("name"))]
    # One launch a column block of the values (`Tiles.columns`), each the same grid.
    assert walks and set(walks) == {attention_decode.launch_programs(rows, tiles) * work["plain"].slices}, (
        walks, work)
    assert attention_decode.launch_programs(rows, tiles) == 64

    def draw(seed: int):
        streams = jax.random.split(jax.random.key(seed), 3)
        return (jax.random.normal(streams[0], (8, 1, heads, 64), jnp.float32),
                *(jax.random.normal(stream, (8, keys, kv_heads, 64), jnp.float32) for stream in streams[1:]))

    def ours(query, key, value):
        with generation.assumed("sm90"):
            return attention_decode.attend(query, key, value, spans, interpret=True)

    def reference(query, key, value):
        return template.reference(query, key, value, spans)

    assert_as_exact(jax.jit(ours), jax.jit(reference), draw, "template fp32 decode of 4096 keys", seeds=SEEDS)


def test_an_a100_decode_plan_prices_the_count_of_slices_filling_one_wave():
    """The slice counts an A100's B1 decode step over 32768 keys at width 128, 32/8 heads, prices
    (`attention_decode._slice_counts`: 108 multiprocessors, 8 key heads' programs a slice, at most 54): the
    one filling its one wave, 13, not a power of two (its fastest walk, M-NA-27c-a100), and the powers of
    two beside it, none past the most it takes."""
    counts = attention_decode._slice_counts(108, 8, 54)
    assert {13, 27, 2, 4, 8, 16, 32} <= set(counts) and max(counts) <= 54 and min(counts) >= 2, counts


def test_a_decode_plan_takes_the_larger_key_tile_among_plans_priced_alike():
    """Row 5274: among plans priced alike, a decode step takes the fewer slices, then the larger key tile
    (`attention_decode.least_plan`), and a cheaper plan whatever its tile: a GH200's B8 over 4096 keys at
    width 64 on 4 slices walked 30.9 us at K128 and 37.2 at K64 (M-NA-27c-gh200), which its walk prices
    alike."""
    small, large = (attention_mosaic.Tiles(16, block_k, 3, 1, family="mma.sync.fed") for block_k in (64, 128))
    assert attention_decode.least_plan({(4, small): 1.0, (4, large): 1.0, (8, large): 1.0}) == (4, large)
    assert attention_decode.least_plan({(4, small): 1.0, (4, large): 1.0 + 2.0**-20}) == (4, small)
    assert attention_decode.least_plan({(2, small): 1.0, (4, large): 1.0}) == (2, small)


def _query_gated(o, q_p):
    """An output hook of a `Kind.QUERY` parameter, its tile's `[rows, 1]` query values (row 5385's shape)."""
    assert o.ndim == 2 and q_p.ndim == 2 and q_p.shape == (o.shape[0], 1)
    return o * jax.nn.sigmoid(q_p[:, :1])


@INTERPRETS
@pytest.mark.parametrize("kind", ["stateless", "head", "query", "output"])
@pytest.mark.parametrize("mode", ["single", "split"])
def test_a_decode_step_fuses_output_hooks_single_slice_and_split_interpreted(kind, mode):
    """Row 5385: `attention_decode.attend` admits and fuses an output hook (`stateless`, `Kind.HEAD`,
    `Kind.QUERY`, and `Kind.OUTPUT` bf16 gate) on both single-slice (`at.slices == 1`, fused in
    `MosaicForward`) and split-KV (`at.slices > 1`, fused in `merged.merged_finished`) GQA decode steps,
    matching `template.reference` under check 1, and rejects `finished=False` or unsupported kinds."""
    batch, positions, heads, kv_heads, width = (2, 2, 8, 2, 64) if mode == "single" else (1, 1, 8, 2, 64)
    keys = 256 if mode == "single" else 2048
    lengths = jnp.asarray([200, 37] if batch == 2 else [1700], jnp.int32)
    starts = jnp.asarray([0, 16] if batch == 2 else [0], jnp.int32)
    spans = template.spans(positions, keys, batch, lengths=lengths, starts=starts)
    sinks = jnp.linspace(-0.5, 0.5, heads, dtype=jnp.float32)
    initial = template.sinks(sinks, batch, positions, width)

    def draw(seed: int):
        rng = np.random.default_rng([5385, seed])
        q = jnp.asarray(rng.normal(size=(batch, positions, heads, width)), jnp.bfloat16)
        k = jnp.asarray(rng.normal(size=(batch, keys, kv_heads, width)) * 0.5, jnp.bfloat16)
        v = jnp.asarray(rng.normal(size=(batch, keys, kv_heads, width)), jnp.bfloat16)
        if kind == "stateless":
            return q, k, v
        if kind == "head":
            return q, k, v, jnp.asarray(rng.normal(size=(heads,)), jnp.float32)
        if kind == "query":
            return q, k, v, jnp.asarray(rng.normal(size=(batch, heads, positions)), jnp.float32)
        return q, k, v, jnp.asarray(rng.normal(size=(batch, positions, heads, width)), jnp.bfloat16)

    def hooks_of(*extra):
        if kind == "stateless":
            return template.Hooks(initial=initial, output=Hook(lambda o: jnp.tanh(o * 0.5)))
        if kind == "head":
            return template.Hooks(initial=initial, output=Hook(lambda o, h: o * jax.nn.sigmoid(h.reshape(())),
                                                               (Side(extra[0], template.Kind.HEAD),)))
        if kind == "query":
            return template.Hooks(initial=initial,
                                  output=Hook(_query_gated, (Side(extra[0], template.Kind.QUERY),)))
        return template.Hooks(initial=initial, output=Hook(lambda o, g: o * jax.nn.sigmoid(g),
                                                           (Side(extra[0], template.Kind.OUTPUT),)))

    def ours(q, k, v, *extra):
        with generation.assumed("sm90"):
            return attention_decode.attend(q, k, v, spans, hooks_of(*extra), interpret=True)

    def reference(q, k, v, *extra):
        hooks = hooks_of(*extra)
        wide = jnp.promote_types(q.dtype, jnp.float32)
        wide_params = tuple(
            Side(p.value.astype(jnp.promote_types(p.value.dtype, wide))
                           if jnp.issubdtype(p.value.dtype, jnp.floating) else p.value, p.axes)
            for p in hooks.output.sides)
        return template.reference(q, k, v, spans, dataclasses.replace(
            hooks, output=dataclasses.replace(hooks.output, sides=wide_params)))

    sample = draw(0)
    assert attention_decode.packable(hooks_of(*sample[3:]))
    with generation.assumed("sm90"):
        work, *_ = attention_decode.planned(sample[0], sample[1], sample[2], spans, hooks_of(*sample[3:]))
    assert work is not None
    assert (work["plain"].slices == 1 if mode == "single" else work["plain"].slices > 1), work
    assert_as_exact(jax.jit(ours), jax.jit(reference), draw,
                    f"template decode output hook {mode} {kind}", seeds=SEEDS)

    if kind == "output" and mode == "single":
        assert not attention_decode.packable(
            template.Hooks(output=Hook(lambda o, x: o + x, (Side(jnp.zeros((batch, heads, keys)),
                                                                 template.Kind.KEY),))))
        with generation.assumed("sm90"), pytest.raises(ValueError):
            attention_decode.attend(sample[0], sample[1], sample[2], spans, hooks_of(*sample[3:]),
                                    interpret=True, finished=False)


HOOKED_STEPS = [(generation_name, label) for generation_name in ("sm80", "sm90", "sm100", "sm120")
                for label in ("single", "split")] + [("sm90", "single-head"), ("sm90", "single-query")]
"""The hooked decode steps' compile cells, a step a cell (row 12435: an H100's four in one took 864 s)."""


def _gated(o, g):
    """A bf16 output gate (`Kind.OUTPUT`), row 5385's."""
    return o * jax.nn.sigmoid(g)


def _head_gated(o, h_p):
    """An output hook of a `Kind.HEAD` parameter, one value a head."""
    assert h_p.shape == ()
    return o * jax.nn.sigmoid(h_p)


def _hooked_step(step: str) -> tuple:
    """The hooked decode step `step` (`HOOKED_STEPS`): its call, its operands and the output hook its
    merge's primitive carries where it splits (None where it walks whole)."""
    if step in ("single", "split"):
        batch, keys = (32, 256) if step == "single" else (1, 4096)
        q = jax.ShapeDtypeStruct((batch, 1, 8, 128), jnp.bfloat16)
        k = jax.ShapeDtypeStruct((batch, keys, 2, 128), jnp.bfloat16)
        spans = template.spans(1, keys, batch, causal=False)
        sinks = jnp.zeros((8,), jnp.float32)

        def gated(q_in, k_in, v_in, g_in):
            hooks = template.Hooks(initial=template.sinks(sinks, batch, 1, 128),
                                   output=Hook(_gated, (Side(g_in, template.Kind.OUTPUT),)))
            return attention_decode.attend(q_in, k_in, v_in, spans, hooks, interpret=False)

        return gated, (q, k, k, q), _gated if step == "split" else None
    param, kind, mod = {"single-head": (jax.ShapeDtypeStruct((8,), jnp.float32), template.Kind.HEAD,
                                        _head_gated),
                        "single-query": (jax.ShapeDtypeStruct((32, 8, 1), jnp.bfloat16), template.Kind.QUERY,
                                         _query_gated)}[step]
    q = jax.ShapeDtypeStruct((32, 1, 8, 128), jnp.bfloat16)
    k = jax.ShapeDtypeStruct((32, 256, 2, 128), jnp.bfloat16)
    spans = template.spans(1, 256, 32, causal=False)

    def deferred(q_in, k_in, v_in, p_in):
        hooks = template.Hooks(output=Hook(mod, (Side(p_in, kind),)))
        return attention_decode.attend(q_in, k_in, v_in, spans, hooks, interpret=False)

    return deferred, (q, k, k, param), mod


@COMPILES_ON_CPU
@pytest.mark.parametrize(("generation_name", "step"), HOOKED_STEPS,
                         ids=[f"{name}-{step}" for name, step in HOOKED_STEPS])
def test_a_decode_step_with_output_hooks_compiles_spill_free(generation_name, step):
    """Row 5385: bf16 gated decode steps (`Kind.OUTPUT` bf16 gate + sinks) compile spill-free on
    sm80/sm90/sm100/sm120 for both batched single-slice and one-row split-KV, with split-KV carrying
    `output_mod` directly on the `merged_padded` primitive; an H100's head and query gates too (whose one
    query a merge of one slice finishes). Each cell traces and compiles its own step alone (row 12697: a
    cell traced every step and compiled the last loop's, the batched step never), a split step certifying
    no config of the one-slice launch it never runs."""
    from fla_gdn_support import KernelCase
    from jax.core import ShapedArray
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    from dew.nn import programs
    from dew.nn.kernels.templates import merged

    call, operands, merged_mod = _hooked_step(step)
    with generation.assumed(generation_name):
        traced = jax.make_jaxpr(call)(*operands)
        merges = [eq for eq in programs.equations(traced.jaxpr) if eq.primitive is merged.merged_padded_p]
        if merged_mod is not None:
            assert len(merges) == 1 and merges[0].params.get("output_mod") is merged_mod, merges
        if step.startswith("single-"):
            padded = merges[0].invars[0].aval
            assert isinstance(padded, ShapedArray) and padded.shape[1] == 1
        found = compile_mosaic_kernels([KernelCase(step, lambda: (call, operands))], generation_name)
    assert found and not (failed := mosaic_failures(found)), "\n".join(failed)
    if step == "split":
        # A split plan certifies its split walks alone (`_split` the last of a walk kernel's suffixes, a
        # scaled or natural one's too), no config of the one-slice launch it never runs.
        walks = [name.split("__")[0] for record in found for certified in record.get("certifications", [])
                 if certified["op"] == "MosaicForward" for name in certified["kernels"]]
        assert walks and all(name.endswith("_split") for name in walks), walks


def test_softcap_folds_into_base2_and_decode_segments_interior_tiles():
    """Row 4045: bounded `Capped(softcap)` folds `scale/softcap` and `softcap * LOG2E` into base-2 log2 units
    (`spec.base2=True`, `spec.reaches=False`, using 1-op `exp2` instead of 10-op `exp`), while `natural=True`
    or degenerate caps stay natural; `_Program.segmented` clips empty interior runs (`raw_start > raw_end`)
    without revisiting edge tiles and splits windowed spans into `(masked, unmasked, masked)`."""
    from dew.nn.kernels.templates import attention_mosaic_program as program

    class _ShiftedCapped(template.Capped):
        def __call__(self, s, q_pos, k_pos):
            return super().__call__(s, q_pos, k_pos) + 1.0

    cap = template.Capped(30.0)
    spec_fold = algebra.Forward("spans", 32768, ids=False, scale=64 ** -0.5, score_mod=cap, natural=False)
    spec_nat = algebra.Forward("spans", 32768, ids=False, scale=64 ** -0.5, score_mod=cap, natural=True)
    spec_zero = algebra.Forward("spans", 32768, ids=False, scale=64 ** -0.5,
                                score_mod=template.Capped(0.0), natural=False)
    spec_sub = algebra.Forward("spans", 32768, ids=False, scale=64 ** -0.5,
                               score_mod=_ShiftedCapped(30.0), natural=False)
    assert spec_fold.base2 and not spec_fold.reaches and not spec_fold.raw
    assert not spec_nat.base2 and not spec_zero.base2 and not spec_sub.base2
    # Subclass of `Capped` keeps its overridden `__call__` instead of folding.
    q_pos = jnp.zeros((2, 2), jnp.int32)
    coords = algebra.Coords(q_pos, q_pos, (), None, None)
    raw_s = jnp.ones((2, 2), jnp.float32)
    sub_logits = algebra.logits_of(spec_sub, raw_s, coords, masked=False)[0]
    ref_sub = _ShiftedCapped(30.0)(raw_s * float(np.float32(64 ** -0.5)), q_pos, q_pos)
    np.testing.assert_array_equal(np.asarray(sub_logits), np.asarray(ref_sub))
    # Native `MosaicForward` wraps `score_mod` in `tuning.Prepared`; folding must unwrap it.
    q_s = jax.ShapeDtypeStruct((1, 1, 8, 64), jnp.bfloat16)
    k_s = jax.ShapeDtypeStruct((1, 4096, 2, 64), jnp.bfloat16)
    sp_s = template.spans(1, 4096, 1, lengths=jnp.array([4096], jnp.int32))
    op_cap, _ = attention_mosaic._forward_call(
        q_s, k_s, k_s, sp_s, template.Hooks(score=Hook(cap)), scale=64 ** -0.5, natural=False)
    prepared_spec = algebra.Forward("spans", 4096, ids=False, scale=64 ** -0.5,
                                    score_mod=op_cap.score_mod, natural=False)
    assert prepared_spec.base2 and not prepared_spec.reaches and not prepared_spec.raw
    assert attention_mosaic.MosaicForward.revision == 2

    with generation.assumed("sm90"):
        family = program.layouts("sm90", jnp.bfloat16, 4)[0]
        tiles = attention_mosaic.Tiles(16, 64, 2, 1, family=family.name)
        prog = program._Program(family, tiles, 64, 2, 4096, causal=False, queries=4, per_span=4)
        # Empty interior (start=16, end=37 inside tile 0 -> full_lo=1 > full_hi=0) visits tile 0 once.
        segs_empty = prog.segmented(lambda *a, **kw: None, (jnp.int32(1), jnp.int32(0)),
                                    jnp.int32(0), jnp.int32(1))
        assert [(int(s), int(e)) for _, s, e in segs_empty] == [(0, 1), (1, 1), (1, 1)]
        # Windowed span [1000, 4096) at block_k=64: tile 15 masked, tiles [16, 64) unmasked, tail empty.
        segs_win = prog.segmented(lambda *a, **kw: None, (jnp.int32(16), jnp.int32(64)),
                                  jnp.int32(15), jnp.int32(64))
        assert [(int(s), int(e)) for _, s, e in segs_win] == [(15, 16), (16, 64), (64, 64)]


def test_a_folded_cap_and_a_state_resumed_at_it_hold_one_logit():
    """A saturated key's folded logit under `Capped(c)` (`logits_of`, base 2) is the logit a state resumed
    at a natural maximum `c` takes (`resumed`, `base2_maximum`), bit for bit: `Capped(32768 + 2^-10)`, whose
    cap is 32768 in fp32, folded its base-2 cap from float64 an ulp above the state's, so a key of value 1
    over a state of mass 1 and sum -1 at that maximum gave 0.00135 where the two cancel to 0 (the head
    review of 7188133e84a, F3)."""
    cap = template.Capped(32768 + 2.0 ** -10)
    spec = algebra.Forward("spans", 128, ids=False, scale=1.0, score_mod=cap, natural=False)
    assert spec.base2
    places = jnp.zeros((1, 1), jnp.int32)
    saturated = algebra.logits_of(spec, jnp.full((1, 1), 2.0 ** 30, jnp.float32),
                                  algebra.Coords(places, places, (), None, None), masked=False)[0]
    state = algebra.resumed(spec, jnp.full((1,), 32768.0), jnp.ones((1,)), -jnp.ones((1, 1)))
    np.testing.assert_array_equal(np.asarray(saturated)[0], np.asarray(state.m))


def _softcap_or_windowed(case: str) -> tuple:
    """`test_softcap_and_windowed_decode_compile_spill_free`'s call `case` and its operands."""
    capped = template.Hooks(score=Hook(template.Capped(30.0)))
    if case == "softcap-d64-decode":
        spans = template.spans(1, 32768, 32, lengths=jnp.full((32,), 32768, jnp.int32))
        operands = (jax.ShapeDtypeStruct((32, 1, 32, 64), jnp.bfloat16),
                    *(jax.ShapeDtypeStruct((32, 32768, 8, 64), jnp.bfloat16),) * 2)
        return (lambda q, k, v: attention_decode.attend(q, k, v, spans, capped, interpret=False)), operands
    if case == "windowed-d128-decode":
        spans = template.spans(1, 4096, 128, lengths=jnp.full((128,), 4096, jnp.int32),
                               starts=jnp.full((128,), 3072, jnp.int32))
        operands = (jax.ShapeDtypeStruct((128, 1, 32, 128), jnp.bfloat16),
                    *(jax.ShapeDtypeStruct((128, 4096, 8, 128), jnp.bfloat16),) * 2)
        return (lambda q, k, v: attention_decode.attend(q, k, v, spans, None, interpret=False)), operands
    spans = template.spans(1024, 1024, 1, causal=True)
    operands = (jax.ShapeDtypeStruct((1, 1024, 16, 256), jnp.bfloat16),
                *(jax.ShapeDtypeStruct((1, 1024, 4, 256), jnp.bfloat16),) * 2)
    return (lambda q, k, v: attention_mosaic.forward(q, k, v, spans, capped, natural=False)), operands


SOFTCAP_OR_WINDOWED = [(name, case) for name in ("sm80", "sm90", "sm100", "sm120")
                       for case in ("softcap-d64-decode", "windowed-d128-decode", "softcap-d256-prefill")]
"""Its compile cells, a call a cell (row 12697: an A100's three in one took 125 s)."""


@COMPILES_ON_CPU
@pytest.mark.parametrize(("generation_name", "case"), SOFTCAP_OR_WINDOWED,
                         ids=[f"{name}-{case}" for name, case in SOFTCAP_OR_WINDOWED])
def test_softcap_and_windowed_decode_compile_spill_free(generation_name, case):
    """Row 4045: base-2 folded `Capped(30.0)` and statically unmasked interior decode tiles compile
    spill-free across sm80/sm90/sm100/sm120 on the softcap d64 (`32 x 32768`), windowed (`128 x 4096`,
    `starts=3072`), and d256 softcap prefill (`ET-PINSOFT`) shapes."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    call, operands = _softcap_or_windowed(case)
    with generation.assumed(generation_name):
        found = compile_mosaic_kernels([KernelCase(case, lambda: (call, operands))], generation_name)
    assert not (failed := mosaic_failures(found)), "\n".join(failed)
