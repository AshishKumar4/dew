"""Gated DeltaNet against the transformers 5.16.1 reference, at fp32.

Dew's forms of the rule (chunked and recurrent, from a zero memory or a given one, a sequence cut in two),
its causal conv and one whole layer are held by check 1 to transformers' own code run live
(`torch_chunk_gated_delta_rule`, `torch_recurrent_gated_delta_rule`, `F.conv1d`, `Qwen3_5GatedDeltaNet`,
modeling_qwen3_5.py) on drawn operands of the fixture's geometry, the truth each computation in float64.
tests/fixtures/linear_attention (tools/qwen_linear_reference.py) holds that geometry and the layer's weights.

The two bugs the port hit on the way, an inclusive instead of a strictly lower triangle in the in-chunk
operator (the reference's masked_fill zeroes the diagonal) and the memory read as `S q` instead of `q^T S` in
the recurrent form, moved the output by 1.5e-1 and 4.7e-1, far past the reference's own error (about 1e-7).
"""

import functools
import itertools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from delta_rule_support import fused_step, recurrence, through
from flax import linen as nn
from forks_support import LAYOUTS, assert_forks_as_own_rows
from kernel_compile import gate_generations
from model_support import as_stepped, decode_through, float64_twin
from reference_error import FEWEST_SEEDS, as_given, assert_as_exact, widened, with_gradients, without_ynn

from dew.algebra.scan import _decays_of
from dew.batch import coordinates
from dew.nn.inputs import AttentionMetadata, MixerCall
from dew.nn.kernels import generation, recurrence as kernel
from dew.nn.kernels.recurrence_chunk import CHUNK_SIZE, Tiles, chunk_gated_delta_rule, fla_rule, xla_mainloop
from dew.nn.linear import (
    GatedDeltaNet,
    _masked_conv1d,
    causal_conv1d,
    held_conv1d,
    l2norm,
    recurrent_gated_delta_rule,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "linear_attention"

SHARDED_SEEDS = 128
"""The per-shard contract's declared draws (check 1, `sharded.assert_runs_per_shard`), sized by
`seeds_for`: the fewest for which one device compared with itself resolves the factor. Run as map tasks of
16 (`check1_shards`, whose counts the plan reads as literals): the interpreted step took over 180 s a case
on CI's CPU, past its group's deadline."""


@pytest.fixture(scope="module")
def reference():
    return dict(np.load(FIXTURES / "gated_delta_net.npz"))


@pytest.fixture(scope="module")
def geometry():
    return json.loads((FIXTURES / "config.json").read_text())


def decode_operands(rng, rows: int, heads: int, width: int, held=jnp.float32):
    """A decode token's operands from `rng`: unit queries and keys, values, a log decay in (-3, 0], a
    write gate in [0, 1), each fp32, and a memory of scale 0.1 held in `held`."""
    unit = [rng.normal(size=(rows, 1, heads, width)) for _ in range(2)]
    query, key = (jnp.asarray(x / np.linalg.norm(x, axis=-1, keepdims=True), jnp.float32) for x in unit)
    value = jnp.asarray(rng.normal(size=(rows, 1, heads, width)), jnp.float32)
    g = jnp.asarray(-rng.random((rows, 1, heads)) * 3, jnp.float32)
    beta = jnp.asarray(rng.random((rows, 1, heads)), jnp.float32)
    state = jnp.asarray(rng.normal(size=(rows, heads, width, width)) * 0.1, held)
    return query, key, value, g, beta, state


def portable(query, key, value, g, beta, state=None, starts=None, chunk: int = CHUNK_SIZE):
    """The gated rule on the portable math (`recurrence_chunk.xla_mainloop`), the port of
    `torch_chunk_gated_delta_rule` a GPU does not run: it runs fla's kernels, held to fla's numbers
    (tests/test_fla_gdn.py). A decay a key dimension's `[..., K]`, a head's as given."""
    if g.ndim == 4 and g.shape[-1] == 1:
        g = g[..., 0]
    # At its operands' width: fp32 over 16-bit or fp32 operands, float64 over a truth's.
    work = jnp.promote_types(jnp.result_type(query, key, value), jnp.float32)
    return xla_mainloop(Tiles(chunk), query, key, value, g, beta, state, starts, scale=key.shape[-1] ** -0.5,
                        delta=True, dtype=work)


RULE_SEEDS = {("chunked", False): 8, ("chunked", True): 8, ("portable", False): 8, ("portable", True): 8,
              ("recurrent", False): 8, ("recurrent", True): 8}
"""Each form's declared draws against its transformers form (check 1), from a zero memory or a drawn one,
sized by `seeds_for` paired with Dew's form on armada."""


def rule_draw(reference, seed: int, *, carried: bool = False):
    """The fixture's operands of the rule drawn anew (tools/qwen_linear_reference.py's rule fixture, its
    geometry and laws): normal queries, keys and values, a log decay `-softplus(N(0, 1))` and a write gate
    `sigmoid(N(0, 1))`, fp32, the queries and keys as the reference takes them, unnormalised; `carried`, a
    memory to start from after them, `0.5 N(0, 1)` as the fixture's initial state."""
    rng = np.random.default_rng([755, seed])
    shape = reference["rule.g"].shape
    query, key, value = (rng.normal(size=reference[f"rule.{name}"].shape)
                         for name in ("query", "key", "value"))
    g, beta = -np.logaddexp(0, rng.normal(size=shape)), 1 / (1 + np.exp(-rng.normal(size=shape)))
    memory = (0.5 * rng.normal(size=reference["rule.initial_state"].shape),) if carried else ()
    return tuple(jnp.asarray(x, jnp.float32) for x in (query, key, value, g, beta, *memory))


def _torch_rule(rule, query, key, value, g, beta, state):
    """transformers' `rule` from `state` (a zero one where None), normalising the queries and keys in the
    rule as Qwen3_5GatedDeltaNet calls it: its output and final state, at the operands' width."""
    import torch

    return rule(*(torch.as_tensor(np.asarray(x)) for x in (query, key, value, g, beta)),
                initial_state=None if state is None else torch.as_tensor(np.asarray(state)),
                output_final_state=True, use_qk_l2norm_in_kernel=True)


def torch_chunked(query, key, value, g, beta, state=None):
    """`torch_chunk_gated_delta_rule` (`_torch_rule`)."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

    return _torch_rule(torch_chunk_gated_delta_rule, query, key, value, g, beta, state)


def torch_recurrent(query, key, value, g, beta, state=None):
    """`torch_recurrent_gated_delta_rule` (`_torch_rule`)."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule

    return _torch_rule(torch_recurrent_gated_delta_rule, query, key, value, g, beta, state)


def normalised_chunked(query, key, value, g, beta, state=None):
    """The public chunked rule (`chunk_gated_delta_rule`, which picks its implementation and its chunk) of
    the queries and keys normalised before it, as the layer hands them over, at fp32 products (the
    reference's)."""
    with jax.default_matmul_precision("highest"):
        return chunk_gated_delta_rule(l2norm(query), l2norm(key), value, g, beta, state)


def normalised_portable(query, key, value, g, beta, state=None):
    """The portable rule (`portable`) of the queries and keys normalised before it, as the layer hands them
    over, at fp32 products (the reference's)."""
    with jax.default_matmul_precision("highest"):
        return portable(l2norm(query), l2norm(key), value, g, beta, state)


def normalised_recurrent(query, key, value, g, beta, state=None):
    """The recurrent rule (`recurrent_gated_delta_rule`, a token at a time) of the queries and keys normalised
    before it, at fp32 products (the reference's)."""
    with jax.default_matmul_precision("highest"):
        return recurrent_gated_delta_rule(l2norm(query), l2norm(key), value, g, beta, state)


FORMS = {"chunked": (normalised_chunked, torch_chunked), "portable": (normalised_portable, torch_chunked),
         "recurrent": (normalised_recurrent, torch_recurrent)}
"""Each of Dew's forms of the rule and the transformers form it ports: the public chunked rule, the portable
math it runs where no kernel takes it (at the upstream's chunk of 64), and the recurrent rule."""


@pytest.mark.parametrize("carried", [False, True], ids=["from zero", "from a memory"])
@pytest.mark.parametrize("form", FORMS)
def test_each_form_of_the_rule_rounds_as_close_to_float64_as_the_reference(reference, form, carried):
    """The fixture's 70 tokens (a full chunk and a padded one) drawn anew, from a zero memory or a drawn
    one, as every decode step past the first starts: Dew's fp32 output and final state of each form on par
    with its transformers form's own fp32 ones (check 1), the truth that transformers form in float64
    (`interop_support.torch_arms`), so a form that computes another function than the release's fails."""
    from interop_support import torch_arms

    dew, theirs = FORMS[form]
    reference_arm, truth = torch_arms(theirs, theirs)
    assert_as_exact(dew, reference_arm, functools.partial(rule_draw, reference, carried=carried),
                    f"the {form} rule {'from a memory' if carried else 'from zero'}",
                    seeds=RULE_SEEDS[form, carried], truth_fn=truth)


AGREEING_SEEDS = 8
"""The chunked form's declared draws against the recurrent one (check 1), sized by `seeds_for` on armada."""


def test_the_chunked_and_recurrent_forms_agree(reference):
    """The two forms are one computation: a model trains with the first and decodes with the second, so
    from the same memory they leave the same output and memory, the chunked form on par with the recurrent
    one from the rule in float64 (check 1, two programs of Dew's)."""
    assert_as_exact(normalised_chunked, normalised_recurrent,
                    functools.partial(rule_draw, reference, carried=True),
                    "the chunked rule against the recurrent one", seeds=AGREEING_SEEDS,
                    truth_fn=normalised_recurrent)


PACKED_RULE_SEEDS = {(1, False): 32, (1, True): 128, (8, False): 16, (8, True): 128}
"""Declared seeds for check 1, sized by `seeds_for` on armada, 2026-10-20: the separate documents'
chunked rule against the float64 recurrence, for each decay width and strength."""


@pytest.mark.parametrize("strong", [False, True], ids=["drawn decay", "strong decay"])
@pytest.mark.parametrize("decay_width", [1, 8], ids=["per head", "per key dimension"])
def test_the_chunked_rule_runs_packed_documents_as_each_runs_alone(decay_width, strong):
    """`the_chunked_rule_runs_packed_documents_as_each_runs_alone` at each decay width and strength: 18 to 20
    s a case cold, its reference's documents traced once (`delta_rule_support.recurrence`), where four
    documents traced apart took 115 to 153 s (cuts-whole at aad1c89d7, row 3772)."""
    the_chunked_rule_runs_packed_documents_as_each_runs_alone(decay_width, strong)


def the_chunked_rule_runs_packed_documents_as_each_runs_alone(decay_width, strong):
    """Packed documents in the chunked rule, one starting on a chunk's first token, one four tokens before a
    chunk's end and one spanning chunks: each document's output, the final memory and every gradient
    on par with the chunked rule on each document alone, both measured from the float64 recurrence.
    The first document reads the row's memory and every other starts from zero. Each draw has fresh
    operands and cotangents, with the key and value dimensions consistently permuted, an exact
    symmetry of the rule. Decay is per head (the gated delta net) or per key dimension (Kimi delta
    attention); at a strong decay (-4 a token) an earlier document sits 96 below the last one's,
    past exp's fp32 range."""
    batch, length, heads, width, chunk = 1, 160, 2, 8, 32
    begins = (0, 32, 60, 110)
    starts = np.zeros((batch, length), bool)
    starts[0, list(begins[1:])] = True

    def draw(seed):
        keys = jax.random.split(jax.random.key(23 + seed), 8)
        q = l2norm(jax.random.normal(keys[0], (batch, length, heads, width)))
        k = l2norm(jax.random.normal(keys[1], (batch, length, heads, width)))
        v = jax.random.normal(keys[2], (batch, length, heads, width))
        g = (jnp.full((batch, length, heads, decay_width), -4.0) if strong else
             -jnp.exp(jax.random.normal(keys[3], (batch, length, heads, decay_width)) - 2.0))
        beta = jax.nn.sigmoid(jax.random.normal(keys[4], (batch, length, heads)))
        h0 = 0.1 * jax.random.normal(keys[5], (batch, heads, width, width))
        do = jax.random.normal(keys[6], v.shape)
        dht = 0.1 * jax.random.normal(keys[7], h0.shape)
        rng = np.random.default_rng(seed)
        key_order, value_order = rng.permutation(width), rng.permutation(width)
        q, k = (x[..., key_order] for x in (q, k))
        v, do = (x[..., value_order] for x in (v, do))
        h0, dht = (x[..., key_order, :][..., value_order] for x in (h0, dht))
        if decay_width > 1:
            g = g[..., key_order]
        return q, k, v, g, beta, h0, do, dht

    def packed(*args):
        return portable(*args, starts=jnp.asarray(starts), chunk=chunk)

    def chunked(*args):
        return portable(*args, chunk=chunk)

    def evaluated(rule, *inputs):
        with jax.default_matmul_precision("highest"):
            return through(rule, *inputs[6:])(*inputs[:6])

    dew = jax.jit(functools.partial(evaluated, packed))
    reference = jax.jit(functools.partial(
        evaluated, functools.partial(recurrence, cuts=(begins,), rule=chunked)))
    truth = jax.jit(functools.partial(evaluated, functools.partial(recurrence, cuts=(begins,))))
    label = f"packed delta rule {decay_width=} {strong=}"
    assert_as_exact(dew, reference, draw, label, seeds=PACKED_RULE_SEEDS[decay_width, strong], truth_fn=truth)


@pytest.mark.parametrize("per_key", [False, True], ids=["gated delta net", "Kimi delta attention"])
def test_the_port_takes_each_packed_tokens_output_and_gradients_as_its_document_alone_does(per_key):
    """The port (`fla_rule`, interpreted off a GPU) runs each packed document as a sequence of
    its own, so each packed token's output and its gradients of the token's query, key, value, decay
    and beta are bitwise those of the same row with every other document's tokens zero: no token's
    value or gradient carries another document's, the decay's rounding included. The same program
    on the same shapes, so any difference is information crossing a document boundary. Documents of
    40, 60, 1, 49, 40, 10 and 56 tokens, one of a single token, resets two apart in a chunk, and on
    and off the chunk grid. The port's arithmetic is held against the token recurrence in float64 on its
    packed documents (test_fla_gdn), and trained natively on a GPU (tests/test_gpu_recurrence.py's port
    cells); a row of 32 two-token documents once lost the decays of its later ones to an fp32
    prefix that every earlier document's reset had shifted."""
    length, heads, width = 256, 2, 64
    begins = (0, 40, 100, 101, 150, 190, 200, length)
    keys = jax.random.split(jax.random.key(29), 6)
    q = 0.3 * jax.random.normal(keys[0], (1, length, heads, width))
    k = l2norm(jax.random.normal(keys[1], (1, length, heads, width)))
    v = jax.random.normal(keys[2], (1, length, heads, width))
    g = -0.1 * jax.nn.softplus(jax.random.normal(keys[3], (1, length, heads, *(width,) * per_key)))
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (1, length, heads)))
    do = jax.random.normal(keys[5], v.shape)

    def taken(starts):
        def run(*operands):
            *rule, do = operands

            def loss(*rule):
                out = fla_rule(*rule, starts=starts)[0]
                return jnp.sum(out * do), out
            (_, out), grads = jax.value_and_grad(loss, argnums=tuple(range(5)), has_aux=True)(*rule)
            return {"out": out, **dict(zip(("q", "k", "v", "g", "beta"), grads, strict=True))}
        return jax.jit(run)

    run = taken(jnp.asarray(np.isin(np.arange(length), begins[1:-1])[None]))
    packed = run(q, k, v, g, beta, do)
    for begin, end in itertools.pairwise(begins):
        mine = (begin <= np.arange(length)) & (np.arange(length) < end)

        def alone(x, mine=mine):
            return jnp.where(mine.reshape(1, length, *(1,) * (x.ndim - 2)), x, 0)

        found = run(*(alone(x) for x in (q, k, v, g, beta, do)))
        for name, value in packed.items():
            np.testing.assert_array_equal(np.asarray(value)[:, mine], np.asarray(found[name])[:, mine],
                                          err_msg=f"{name} of the document at {begin}")


def test_the_port_keeps_each_row_to_itself_at_its_rows_tightest_document_bound():
    """Two right-padded rows of two packed documents each (segment 0 after them) at the tightest
    bound their coordinates give (`Coordinates.max_segments`, two): each document's output on the
    port (`fla_rule`) on par with the document run alone, a row's first from the row's memory, both
    measured from the float64 recurrence (check 1). A row's padding runs as a sequence of its own, so a
    bound without it ran the second row's last document on from its first."""
    from dew.batch import coordinates

    segments = jnp.asarray([[1] * 70 + [2] * 50 + [0] * 8, [1] * 50 + [2] * 50 + [0] * 28])
    tokens = coordinates(segments=segments, max_segments=2, right_padded=True)
    documents = [(row, begin, end) for row, cuts in ((0, (0, 70, 120)), (1, (0, 50, 100)))
                 for begin, end in itertools.pairwise(cuts)]

    def draw(seed):
        keys = jax.random.split(jax.random.key(31 + seed), 5)
        q, k, v = (0.3 * jax.random.normal(key, (2, 128, 1, 64)) for key in keys[:3])
        g = -0.1 * jax.nn.softplus(jax.random.normal(keys[3], (2, 128, 1)))
        beta = jnp.full((2, 128, 1), 0.5)
        return q, k, v, g, beta, 0.1 * jax.random.normal(keys[4], (2, 1, 64, 64))

    def document(operands, row, begin, end):
        return [x[row:row + 1, begin:end] for x in operands[:5]]

    def packed(*operands):
        out, _ = fla_rule(*operands, starts=tokens.starts, max_segments=tokens.max_segments)
        return {f"row {row} from {begin}": out[row:row + 1, begin:end] for row, begin, end in documents}

    def alone(*operands):
        return {f"row {row} from {begin}": fla_rule(*document(operands, row, begin, end),
                                                    operands[5][row:row + 1] if begin == 0 else None)[0]
                for row, begin, end in documents}

    def truth(*operands):
        return {f"row {row} from {begin}": recurrence(*document(operands, row, begin, end),
                                                      operands[5][row:row + 1] if begin == 0
                                                      else jnp.zeros_like(operands[5][row:row + 1]))[0]
                for row, begin, end in documents}

    assert_as_exact(packed, alone, draw, "packed rows on the port", seeds=8, truth_fn=truth)


STRONG_DECAY_SEEDS = 32
"""The strong decay's declared draws of the chunked rule against the recurrent one (check 1), sized by
`seeds_for` on armada."""


def test_strong_decay_has_the_same_finite_gradient_in_both_forms():
    """Masked positive log differences must never reach exp in the chunk rule: at g = -4 a 64-token chunk has
    finite outputs but exp of its unused upper triangle overflows, and masking afterwards would leave a
    0 * inf NaN in the backward. Over drawn queries (keys and values their multiples, a write gate of 0.2),
    the chunked rule's output and its gradient into the decay are finite and on par with the recurrent
    form's, which stays finite, from it in float64 (check 1, two programs of Dew's)."""
    beta, decay = jnp.full((1, 64, 1), .2), jnp.full((1, 64, 1), -4.)

    def draw(seed):
        rng = np.random.default_rng([0x5DEC, seed])
        return (jnp.asarray(rng.normal(size=(1, 64, 1, 4)) * .1, jnp.float32),
                jnp.asarray(rng.normal(size=(1, 64, 1, 4)), jnp.float32))

    def differentiated(rule):
        def run(query, cotangent):
            out, pull = jax.vjp(lambda g: rule(query, query * .7, query * .3, g, beta.astype(query.dtype))[0],
                                decay.astype(query.dtype))
            return out, *pull(cotangent)
        return run

    out, gradient = differentiated(chunk_gated_delta_rule)(*draw(0))
    assert np.isfinite(np.asarray(out)).all() and np.isfinite(np.asarray(gradient)).all()
    assert_as_exact(differentiated(chunk_gated_delta_rule), differentiated(recurrent_gated_delta_rule), draw,
                    "the strong decay's output and decay gradient", seeds=STRONG_DECAY_SEEDS)



CUT = 30
"""Where a sequence is cut: inside the first chunk, so the carried state and the padding are what is read,
not a chunk boundary."""
CUT_SEEDS = 8
"""The cut sequence's declared draws against the whole one (check 1), sized by `seeds_for` on armada."""


def cut_in_two(query, key, value, g, beta):
    """The public chunked rule (`normalised_chunked`) over the first `CUT` tokens, then over the rest from
    the memory they leave: the outputs joined and the last memory, as a prefill and its continuation run."""
    head, memory = normalised_chunked(*(x[:, :CUT] for x in (query, key, value, g, beta)))
    tail, final = normalised_chunked(*(x[:, CUT:] for x in (query, key, value, g, beta)), memory)
    return jnp.concatenate([head, tail], axis=1), final


def test_a_sequence_split_in_two_carries_its_memory_across_the_cut(reference):
    """The property a prefill followed by a continuation depends on: the memory the first `CUT` tokens
    leave, fed to the rest, gives what the whole sequence gives at once, the cut on par with the whole from
    the rule in float64 (check 1, two programs of Dew's)."""
    assert_as_exact(cut_in_two, normalised_chunked, functools.partial(rule_draw, reference),
                    f"the rule cut at {CUT}", seeds=CUT_SEEDS)


CONV_SEEDS = 8
"""The causal conv's declared draws against `F.conv1d` (check 1), sized by `seeds_for` on armada."""


def conv_draw(reference, seed: int):
    """The fixture's conv drawn anew (tools/qwen_linear_reference.py's conv fixture, its geometry and
    laws): a normal input `[B, D, S]`, channel-major as the reference takes it, and taps `[D, 1, K]` of
    `0.3 N(0, 1)`, fp32."""
    rng = np.random.default_rng([757, seed])
    return tuple(jnp.asarray(scale * rng.normal(size=reference[f"conv.{name}"].shape), jnp.float32)
                 for name, scale in (("input", 1.0), ("weight", 0.3)))


def dew_conv(x, weight):
    """`causal_conv1d` of the reference's channel-major input and depthwise taps, Dew's being time-major."""
    return jnp.swapaxes(causal_conv1d(jnp.swapaxes(x, 1, 2), weight[:, 0, :]), 1, 2)


def torch_conv(x, weight):
    """`F.conv1d(padding=K - 1, groups=D)[..., :S]` then silu, the reference's short mixer
    (modeling_qwen3_next.py:345-365): position s reads s-K+1..s."""
    import torch
    import torch.nn.functional as F

    x, weight = torch.as_tensor(np.asarray(x)), torch.as_tensor(np.asarray(weight))
    return F.silu(F.conv1d(x, weight, padding=weight.shape[-1] - 1, groups=x.shape[1])[..., :x.shape[-1]])


def test_the_causal_conv_rounds_as_close_to_float64_as_f_conv1d(reference):
    """Dew's causal conv on par with `F.conv1d`'s from float64 (check 1), and causal: moving the last
    position's input leaves every earlier output's bits as they were, the same program on the same values."""
    from interop_support import torch_arms

    reference_arm, truth = torch_arms(torch_conv, torch_conv)
    assert_as_exact(dew_conv, reference_arm, functools.partial(conv_draw, reference), "the causal conv",
                    seeds=CONV_SEEDS, truth_fn=truth)
    x, weight = conv_draw(reference, 0)
    out = dew_conv(x, weight)
    changed = dew_conv(x.at[..., -1].add(3.0), weight)
    assert jnp.array_equal(out[..., :-1], changed[..., :-1])
    assert not jnp.array_equal(out[..., -1], changed[..., -1])


def layer_params(reference):
    """The reference layer's state_dict under dew's names: Linear weights
    transposed to [in, out], everything else as it stands."""
    def leaf(name):
        return jnp.asarray(reference[f"layer.{name}"])
    return {"params": {
        "in_proj_qkv": {"kernel": leaf("in_proj_qkv.weight").T},
        "in_proj_z": {"kernel": leaf("in_proj_z.weight").T},
        "in_proj_b": {"kernel": leaf("in_proj_b.weight").T},
        "in_proj_a": {"kernel": leaf("in_proj_a.weight").T},
        "out_proj": {"kernel": leaf("out_proj.weight").T},
        "conv1d": {"weight": leaf("conv1d.weight")},
        "norm": {"weight": leaf("norm.weight")},
        "A_log": leaf("A_log"),
        "dt_bias": leaf("dt_bias"),
    }}


def layer(geometry) -> GatedDeltaNet:
    return GatedDeltaNet(
        emb_features=geometry["hidden_size"],
        num_k_heads=geometry["linear_num_key_heads"],
        num_v_heads=geometry["linear_num_value_heads"],
        head_k_dim=geometry["linear_key_head_dim"],
        head_v_dim=geometry["linear_value_head_dim"],
        conv_kernel=geometry["linear_conv_kernel_dim"],
        norm_eps=geometry["rms_norm_eps"])


LAYER_SEEDS = 32
"""The layer's declared draws against `Qwen3_5GatedDeltaNet` (check 1), sized by `seeds_for` on armada."""


def torch_layer(reference, geometry):
    """Check 1's reference and float64 truth over transformers' `Qwen3_5GatedDeltaNet` built from the
    fixture's geometry and holding its weights, as tools/qwen_linear_reference.py runs it, its twin built
    at float64 (`interop_support.transformers_arms`, `_dew_torch.precision.float64_twin`)."""
    import torch
    from interop_support import transformers_arms
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    from _dew_torch.precision import float64_twin as torch_twin

    def built():
        return Qwen3_5GatedDeltaNet(Qwen3_5TextConfig(**geometry), layer_idx=0)

    module = built()
    module.load_state_dict({name.removeprefix("layer."): torch.as_tensor(value)
                            for name, value in reference.items()
                            if name.startswith("layer.") and name != "layer.hidden"})
    module.eval()
    return transformers_arms(module, lambda layer, hidden: layer(hidden, attention_mask=None),
                             twin=torch_twin(module, build=built))


def test_the_layer_rounds_as_close_to_float64_as_qwen3_5_gated_delta_net(reference, geometry):
    """The whole mixer on the reference's own weights: projections, the conv, the gates, the rule with value
    heads outnumbering key heads, the gated norm and the output projection, over drawn hidden states of the
    fixture's 70 tokens, on par with transformers' layer from the float64 layer's forward (check 1)."""
    module = layer(geometry)
    variables = layer_params(reference)
    sample = jnp.zeros((1, 4, geometry["hidden_size"]))
    template = nn.unbox(jax.eval_shape(module.init, jax.random.key(0), sample))
    assert jax.tree.map(jnp.shape, variables) == jax.tree.map(jnp.shape, template)
    reference_arm, truth = torch_layer(reference, geometry)
    assert_as_exact(lambda hidden: module.apply(variables, hidden), reference_arm,
                    decoded_draw(reference["layer.hidden"], positions=reference["layer.hidden"].shape[1]),
                    "the gated delta net layer", seeds=LAYER_SEEDS, truth_fn=truth)


DECODE_SEEDS = 8
"""The reference layer's cached decode's declared check-1 seeds against its parallel forward at each
step's width, sized by `seeds_for` on armada."""
DECODED = 32
"""A decode check's drawn positions: a prefill of 5 and 27 steps past it, which the conv tail and the
recurrent memory cross, the parallel forwards from 25 positions on past a chunk (24 tokens at these
keys' width, `held_chunk`). Each step's parallel forward is a program of its own width in the check's,
whose compile sets the test's time: over the recording's 70 positions, 66 of them took 89 s a test, and
in map tasks of 4 seeds 140 to 330 s a task under CI's load."""


def decoded_draw(recorded: np.ndarray, positions: int = DECODED):
    """A check's draw of hidden states: inputs `positions` long (a decode check's `DECODED`), of `recorded`'s
    rows, width and scale."""
    def draw(seed):
        rng = np.random.default_rng([0x2739, seed])
        shape = (*recorded.shape[:1], positions, *recorded.shape[2:])
        return (jnp.asarray(rng.standard_normal(shape) * recorded.std(), jnp.float32),)
    return draw


def test_the_layer_decodes_as_it_prefills(reference, geometry):
    """A prefill against the cache and single-token steps after it, on the reference's own weights over
    drawn inputs of the recording's width and scale (`DECODED` positions), on par with the parallel
    forward at each step's width (`as_stepped`) from the float64 layer's forward (check 1, ledger
    2739): the conv tail and the recurrent memory both have to cross the step boundary."""
    module = layer(geometry)
    variables = layer_params(reference)
    recorded = reference["layer.hidden"]

    draw = decoded_draw(recorded)

    wide = float64_twin(module)
    assert_as_exact(lambda hidden: decode_through(module, variables, hidden, prefill=5)[0],
                    lambda hidden: as_stepped(lambda held: module.apply(variables, held), hidden, 5),
                    draw, "gated delta net decode", seeds=DECODE_SEEDS,
                    truth_fn=lambda hidden: wide.apply(widened(variables), hidden))


BF16_DECODE_SEEDS = 8
"""The bf16 layer's cached decode's declared check-1 seeds against its bf16 parallel forward at each
step's width, sized by `seeds_for` on armada."""


def test_a_bf16_layer_decodes_its_state_in_bf16_on_par_with_its_prefill(reference, geometry):
    """A bf16 layer at the default matmul precision holds its decode state in
    bf16, as vLLM does, and runs the rule in fp32: over drawn inputs, a prefill
    and single-token steps after it on par with the bf16 parallel forward at
    each step's width (`as_stepped`, its chunk state in fp32) from the float64
    layer's forward (check 1, ledger 2739); the conv state, the projections'
    own bf16 values, is held as they are. Asked for the highest precision, as
    this lane runs, it holds fp32."""
    module = layer(geometry)
    variables = layer_params(reference)
    recorded = reference["layer.hidden"]
    narrow = module.clone(dtype=jnp.bfloat16)
    exact = narrow.apply(variables, jnp.asarray(recorded)[:, :1], MixerCall(decode=True),
                         mutable=["cache"])[1]["cache"]
    assert {leaf.dtype for leaf in jax.tree.leaves(exact)} == {jnp.dtype(jnp.float32)}
    with jax.default_matmul_precision("default"):
        _, cache, _ = decode_through(narrow, variables, jnp.asarray(recorded), prefill=5)
    assert {leaf.dtype for leaf in jax.tree.leaves(cache)} == {jnp.dtype(jnp.bfloat16)}

    draw = decoded_draw(recorded)

    def at_default(run):
        """`run` compiled as the gate compiles two Dew arms (`without_ynn`) at the default matmul
        precision, as a bf16 layer's users run it, and run as given."""
        compiled = without_ynn(run)

        @as_given
        def ran(hidden):
            with jax.default_matmul_precision("default"):
                return compiled(hidden)
        return ran

    wide = float64_twin(module)
    assert_as_exact(at_default(lambda hidden: decode_through(narrow, variables, hidden, prefill=5)[0]),
                    at_default(lambda hidden: as_stepped(lambda held: narrow.apply(variables, held), hidden,
                                                         5)),
                    draw, "bf16 gated delta net decode", seeds=BF16_DECODE_SEEDS,
                    truth_fn=lambda hidden: wide.apply(widened(variables), hidden))


def scan_masked_conv1d(x, kernel, valid, state=None):
    """The token-by-token form the compacted convolution replaced.

    One window per physical slot, the history advancing only on a real
    token. It stays here as the reference the compacted form reproduces.
    """
    batch, _, channels = x.shape
    width = kernel.shape[1] - 1
    if state is None:
        state = jnp.zeros((batch, width, channels), x.dtype)

    def step(history, inputs):
        token, active = inputs
        window = jnp.concatenate([history, token[:, None, :]], axis=1)
        output = nn.silu(jnp.sum(window * kernel.T[None], axis=1))
        history = jnp.where(active[:, None, None], window[:, 1:], history)
        return history, jnp.where(active[:, None], output, 0.0)

    state, output = jax.lax.scan(step, state, (jnp.moveaxis(x, 1, 0), valid.T))
    return jnp.moveaxis(output, 0, 1), state


MASKED_CHANNELS = 32
"""A masked conv check's channels: each channel convolves apart, so every leaf of a case holds this many
times its slots' values, over which a draw's error pools (at 3, a kernel's gradient over a few real tokens
needed up to 512 draws to resolve)."""


def masked_draw(rows, kernel_width: int, *, history: bool, differentiated: bool = True,
                channels: int = MASKED_CHANNELS):
    """A masked conv check's draw for a case's `rows` (each slot real or not): a normal input `[B, S, D]`
    and taps `[D, W]`, with `history` an incoming history `[B, W - 1, D]`, and where `differentiated` a
    cotangent of the outputs and of the history carried, fp32."""
    batch, length = np.asarray(rows).shape

    def draw(seed):
        rng = np.random.default_rng([0x5CA9, seed])
        held = (batch, kernel_width - 1, channels)
        drawn = [rng.normal(size=(batch, length, channels)), rng.normal(size=(channels, kernel_width)),
                 *([rng.normal(size=held)] if history else [])]
        cotangent = (rng.normal(size=(batch, length, channels)), rng.normal(size=held))
        arrays = tuple(jnp.asarray(x, jnp.float32) for x in drawn)
        return (*arrays, tuple(jnp.asarray(x, jnp.float32) for x in cotangent)) if differentiated else arrays
    return draw


def masked_arm(implementation, rows, *, history: bool):
    """`implementation(x, kernel, valid, state)` over a case's `rows`: its outputs and the history it
    carries, and their gradients into the input, the taps and, with `history`, the incoming history under
    the draw's cotangent (`with_gradients`)."""
    valid = jnp.asarray(np.asarray(rows, bool))
    if history:
        return with_gradients(lambda x, kernel, state: implementation(x, kernel, valid, state))
    return with_gradients(lambda x, kernel: implementation(x, kernel, valid))


MASKED_ROWS = {
    "right padding": ([[1] * 5 + [0] * 3, [1] * 3 + [0] * 5], 4),
    "left padding": ([[0] * 3 + [1] * 5, [0] * 1 + [1] * 7], 4),
    "interior holes": ([[1, 1, 0, 1, 1, 0, 0, 1, 1], [1, 1, 1, 0, 1, 1, 1, 0, 1]], 4),
    "alternating slots": ([[1, 0] * 6, [0, 1] * 6], 4),
    "a paused row beside a full one": ([[0] * 7, [1] * 7], 4),
    "kernel 2": ([[1, 0, 1, 1, 0, 1, 1]], 2),
    "kernel 8": ([[1, 1, 0, 1, 1, 1, 0, 1, 1, 1]], 8),
    "the kernel's own length": ([[1, 0, 1, 1]], 4),
    "one token past a chunk": ([[1] * 60 + [0] * 3 + [1] * 2], 4),
}


MASKED_SEEDS = {
    ("right padding", False): 8, ("right padding", True): 8,
    ("left padding", False): 8, ("left padding", True): 8,
    ("interior holes", False): 16, ("interior holes", True): 16,
    ("alternating slots", False): 8, ("alternating slots", True): 8,
    ("a paused row beside a full one", False): 8, ("a paused row beside a full one", True): 8,
    ("kernel 2", False): 16, ("kernel 2", True): 32,
    ("kernel 8", False): 8, ("kernel 8", True): 8,
    ("the kernel's own length", False): 16, ("the kernel's own length", True): 16,
    ("one token past a chunk", False): 8, ("one token past a chunk", True): 8,
}
"""Each masked conv case's declared draws against the token scan (check 1), with its history and without,
sized by `seeds_for` on armada."""


@pytest.mark.parametrize("case", list(MASKED_ROWS))
@pytest.mark.parametrize("history", [False, True])
def test_the_masked_conv_matches_the_token_scan(case, history):
    """The compacted convolution is the scan it replaced: its outputs, the history it carries and their
    gradients into the input, the taps and the incoming history on par with the scan's from the scan in
    float64 (check 1, two programs of Dew's); and on every draw a slot that holds no token outputs and passes
    back exactly zero, and a row of none carries its history unchanged, bit for bit."""
    rows, width = MASKED_ROWS[case]
    draw = masked_draw(rows, width, history=history)
    assert_as_exact(masked_arm(_masked_conv1d, rows, history=history),
                    masked_arm(scan_masked_conv1d, rows, history=history), draw,
                    f"the masked conv, {case}{', from a history' if history else ''}",
                    seeds=MASKED_SEEDS[case, history])
    *inputs, cotangent = draw(0)
    valid = jnp.asarray(np.asarray(rows, bool))
    (out, carried), dx, *_ = masked_arm(_masked_conv1d, rows, history=history)(*inputs, cotangent)
    assert jnp.array_equal(jnp.where(valid[:, :, None], 0.0, out), jnp.zeros_like(out))
    assert jnp.array_equal(jnp.where(valid[:, :, None], 0.0, dx), jnp.zeros_like(dx))
    if history:
        paused = ~jnp.any(valid, axis=1)
        assert jnp.array_equal(carried[paused], inputs[2][paused])


HOLED_SEEDS = 8
"""A holed row's declared draws against its real tokens alone (check 1), sized by `seeds_for` on armada."""


def test_a_holed_rows_masked_conv_is_the_conv_of_its_real_tokens_alone():
    """Each row of interior holes convolves its real tokens as one stream: at those tokens, the masked
    conv's outputs on par with `causal_conv1d` of the real tokens alone from it in float64 (check 1, two
    programs of Dew's). A conv over the row with its holes zero-filled reads the holes, an error near the
    outputs' own size, which no rounding reaches."""
    rows, width = MASKED_ROWS["interior holes"]
    kept = [np.flatnonzero(row).tolist() for row in np.asarray(rows, bool)]

    def holed(x, kernel):
        out, _ = _masked_conv1d(x, kernel, jnp.asarray(np.asarray(rows, bool)))
        return [out[row, real] for row, real in enumerate(kept)]

    def alone(x, kernel):
        return [causal_conv1d(x[row:row + 1, real], kernel)[0] for row, real in enumerate(kept)]

    assert_as_exact(holed, alone, masked_draw(rows, width, history=False, differentiated=False),
                    "a holed row's real tokens", seeds=HOLED_SEEDS)


CUT_CONV_SEEDS = 8
"""A masked conv cut at a call boundary's declared draws against one call (check 1), sized by `seeds_for`
on armada."""


def test_the_masked_conv_splits_at_a_call_boundary():
    """A prefill and a continuation see one stream of real tokens: the history the first call leaves makes
    the second one's outputs and its history on par with a single call over the whole row from that call in
    float64 (check 1, two programs of Dew's), the cut falling inside a gap."""
    rows, width = MASKED_ROWS["interior holes"]
    valid, cut = jnp.asarray(np.asarray(rows, bool)), 6

    def whole(x, kernel, state):
        return _masked_conv1d(x, kernel, valid, state)

    def split(x, kernel, state):
        head, history = _masked_conv1d(x[:, :cut], kernel, valid[:, :cut], state)
        tail, final = _masked_conv1d(x[:, cut:], kernel, valid[:, cut:], history)
        return jnp.concatenate([head, tail], axis=1), final

    assert_as_exact(split, whole, masked_draw(rows, width, history=True, differentiated=False),
                    f"the masked conv cut at {cut}", seeds=CUT_CONV_SEEDS)


PACKED_HISTORY_SEEDS = 8
"""A packed prefill's continuation's declared draws against its last document run alone (check 1), sized
by `seeds_for` on armada."""


def test_a_packed_prefill_leaves_only_its_last_documents_conv_history():
    """A row packing two documents, the last two tokens long, leaves the history its continuation reads: the
    last document's two tokens, moved bit for bit, and zeros where the first document's would be; so the next
    call's tokens convolve as that document's would on its own, the continuation on par with the last
    document and the next tokens in one call from it in float64 (check 1, two programs of Dew's)."""
    segments = jnp.asarray([[1, 1, 1, 1, 1, 2, 2]])

    def draw(seed):
        rng = np.random.default_rng([0x9AC4, seed])
        return tuple(jnp.asarray(rng.normal(size=shape), jnp.float32)
                     for shape in ((6, 4), (1, 7, 6), (1, 3, 6)))

    def continued(kernel, x, following):
        _, history = _masked_conv1d(x, kernel, jnp.ones((1, 7), bool), segments=segments)
        return _masked_conv1d(following, kernel, jnp.ones((1, 3), bool), history)[0]

    def alone(kernel, x, following):
        joined = jnp.concatenate([x[:, 5:], following], axis=1)
        return _masked_conv1d(joined, kernel, jnp.ones((1, 5), bool))[0][:, 2:]

    kernel, x, _ = draw(0)
    _, history = _masked_conv1d(x, kernel, jnp.ones((1, 7), bool), segments=segments)
    np.testing.assert_array_equal(np.asarray(history[:, 0]), 0)
    np.testing.assert_array_equal(np.asarray(history[:, 1:]), np.asarray(x[:, 5:]))
    assert_as_exact(continued, alone, draw, "a packed prefill's continuation", seeds=PACKED_HISTORY_SEEDS)


HOLED_LAYER_SEEDS = 8
"""The layer's holed row's declared draws against its real tokens alone (check 1), sized by `seeds_for` on
armada."""


def test_the_layer_reads_a_holed_row_as_the_row_without_its_holes(reference, geometry):
    """The mixer's own contract, through its public call: the states of the real tokens of a row with padded
    slots between them, whatever the slots hold, on par with those tokens' states on their own from them in
    float64 (check 1, two programs of Dew's), and the padded slots themselves exactly zero."""
    module = layer(geometry)
    variables = layer_params(reference)
    rows, width = reference["layer.hidden"].shape[0], reference["layer.hidden"].shape[2]
    kept, holes = jnp.asarray([0, 1, 4, 5, 7, 8]), jnp.asarray([2, 3, 6])
    valid = jnp.zeros((rows, 9), bool).at[:, kept].set(True)
    holed_call = MixerCall(AttentionMetadata(coordinates=coordinates(valid=valid)))

    def draw(seed):
        rng = np.random.default_rng([0x401E, seed])
        return tuple(jnp.asarray(rng.standard_normal((rows, length, width)), jnp.float32)
                     for length in (6, 3))

    def holed(hidden, filling):
        spread = jnp.zeros((rows, 9, width), hidden.dtype).at[:, kept].set(hidden).at[:, holes].set(filling)
        return module.apply(variables, spread, holed_call)

    wide = float64_twin(module)
    out = holed(*draw(0))
    assert jnp.array_equal(out[:, holes], jnp.zeros_like(out[:, holes]))
    assert_as_exact(lambda hidden, filling: holed(hidden, filling)[:, kept],
                    lambda hidden, filling: module.apply(variables, hidden), draw,
                    "a holed row's real tokens through the layer", seeds=HOLED_LAYER_SEEDS,
                    truth_fn=lambda hidden, filling: wide.apply(widened(variables), hidden))


RIGHT_PADDED = {
    "rows of five and three": [[1] * 5 + [0] * 3, [1] * 3 + [0] * 5],
    "a full row beside an empty one": [[1] * 8, [0] * 8],
    "fewer real tokens than the kernel's history": [[1] * 2 + [0] * 6, [1] + [0] * 7],
}


RIGHT_PADDED_CONV_SEEDS = 8
"""Each right-padded case's declared draws against the masked conv (check 1), sized by `seeds_for` on
armada."""


@pytest.mark.parametrize("case", list(RIGHT_PADDED))
@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_a_right_padded_row_convolves_as_the_masked_conv(case, history, packed):
    """Declared right-padded, a row convolves as it stands, with no gather: its outputs, padding zeroed, the
    history it carries and their gradients into the input, the taps and the incoming history on par with the
    masked conv's from it in float64 (check 1, two programs of Dew's), with its real tokens one document or
    packed as two, the second from the third token on, the padding a document of its own."""
    rows = RIGHT_PADDED[case]
    valid = jnp.asarray(np.asarray(rows, bool))
    segments = (jnp.where(valid, 1 + (jnp.arange(valid.shape[1]) >= 2), 0).astype(jnp.int32) if packed
                else None)

    def held(*, right_padded):
        def implementation(x, kernel, valid, state=None):
            return held_conv1d(x, kernel, valid, state, segments=segments, right_padded=right_padded)
        return masked_arm(implementation, rows, history=history)

    label = f"a right-padded conv, {case}{', packed' if packed else ''}{', from a history' * history}"
    assert_as_exact(held(right_padded=True), held(right_padded=False), masked_draw(rows, 4, history=history),
                    label, seeds=RIGHT_PADDED_CONV_SEEDS)


RIGHT_PADDED_SEEDS = 8
"""The right-padded layer's declared draws against the masked call (check 1), sized by `seeds_for`
paired with the declared call on the CPU."""


def test_a_right_padded_layer_keeps_its_real_tokens_states(reference, geometry):
    """The mixer told its rows are right-padded gives the masked call's
    outputs, zero at the padding: on par with the masked call from its
    float64 run over drawn hidden states (check 1), the two being separate
    programs, which a GPU rounds apart (1.7e-6 on an H100)."""
    module = layer(geometry)
    variables = layer_params(reference)
    shape = jnp.asarray(reference["layer.hidden"])[:, :9].shape
    valid = jnp.arange(9)[None, :] < jnp.asarray([[6], [9]])

    def call(metadata, variables=variables):
        return lambda hidden: module.apply(variables, hidden, MixerCall(metadata))

    masked = AttentionMetadata(coordinates=coordinates(valid=valid))
    declared = AttentionMetadata(coordinates=coordinates(valid=valid, right_padded=True))

    def draw(seed):
        return (jnp.asarray(np.random.default_rng(seed).standard_normal(shape), jnp.float32),)

    def truth(hidden):
        # On the host, as a GPU's float64 is not every one's.
        cpu = jax.devices("cpu")[0]
        with jax.default_device(cpu):
            wide = jax.device_put(widened(variables), cpu)
            on_host = jax.device_put(np.asarray(valid), cpu)
            metadata = AttentionMetadata(coordinates=coordinates(valid=on_host))
            return call(metadata, wide)(jax.device_put(hidden, cpu))

    (hidden,) = draw(0)
    assert jnp.array_equal(jax.jit(call(declared))(hidden)[0, 6:], jnp.zeros_like(hidden[0, 6:]))
    assert_as_exact(call(declared), call(masked), draw, "right-padded output", seeds=RIGHT_PADDED_SEEDS,
                    truth_fn=truth)


def between(g):
    """Each chunk's pairwise decays `[..., C, C, K]` of its log decays `g` `[..., C, K]`, `exp(G_s - G_t)` on
    and below the diagonal: the derived forms' (`dew.algebra.scan`'s `_Decays.between`), a chunk at a time."""
    return jax.vmap(lambda chunk: _decays_of(chunk, base2=False).between)(g)


def test_a_decay_deep_in_a_chunk_keeps_its_own_precision():
    """Each pairwise decay's exponent is a difference of two compensated
    cumulative sums, rounded twice: within 2u of its own range's magnitude,
    plus the compensated sums' own error (u^2 per term of the chunk's
    magnitude) and exp's rounding, however far into the chunk the pair
    sits. A difference of plain cumulative sums carries the rounding of the
    whole accumulated magnitude instead: at g near -2 over 64 positions,
    ~128 u on a decay between neighbours, which the KDA gradient amplified
    to twice the reference's distance from float64 (tests/test_kimi_linear.py)."""
    g = np.random.default_rng(7).uniform(-3, -1, (2, 64, 4)).astype(np.float32)
    decay = between(jnp.asarray(g))
    wide = np.asarray(g, np.float64)
    rows, cols = np.tril_indices(64)
    ranges = [wide[:, c + 1:r + 1] for r, c in zip(rows, cols, strict=True)]
    exponent = np.array([part.sum(1) for part in ranges])
    magnitude = np.array([np.abs(part).sum(1) for part in ranges])
    u = np.finfo(np.float32).eps / 2
    got = np.asarray(decay, np.float64)[:, rows, cols].transpose(1, 0, 2)
    kept = got > np.finfo(np.float32).tiny
    error = np.abs(np.log(got[kept]) - exponent[kept])
    chunk = np.broadcast_to(np.abs(wide).sum(1), magnitude.shape)[kept]
    bound = 2 * u * magnitude[kept] + 64 * u * u * chunk + 2 * u
    assert np.all(error <= bound), float(np.max(error / bound))
    above = ~np.tril(np.ones((64, 64), bool))
    assert not np.asarray(decay)[:, above].any()


def test_the_decay_gradient_sums_only_the_pairs_spanning_each_position():
    """The decay's gradient, `sum_{s >= i > t} cotangent[s, t] decay[s, t]` at position i, as exact
    as the sum of those pairs alone: within two sequential sums' rounding of their magnitudes
    against float64, where the prefix difference's gradient summed every pair after i into it
    with opposite signs and cancelled."""
    rng = np.random.default_rng(11)
    g = rng.uniform(-3, -1, (2, 64, 4)).astype(np.float32)
    cotangent = rng.standard_normal((2, 64, 64, 4)).astype(np.float32)

    def gradient(g, cotangent):
        _, pull = jax.vjp(between, g)
        return pull(cotangent)[0]

    got = np.asarray(gradient(jnp.asarray(g), jnp.asarray(cotangent)), np.float64)
    with jax.enable_x64(new_val=True):
        wide = widened((g, cotangent))
        want = np.asarray(gradient(*wide))
        weighed = np.abs(np.asarray(wide[1] * between(wide[0])))
    spans = np.array([weighed[:, i:, :i].sum((1, 2)) for i in range(64)]).transpose(1, 0, 2)
    bound = 2 * 64 * (np.finfo(np.float32).eps / 2) * spans
    assert np.all(np.abs(got - want) <= bound), float(np.max(np.abs(got - want) / np.maximum(bound, 1e-30)))


PAIRING_SEEDS = 8
"""The decays' pairings' declared draws against the plain decays' (check 1), sized by `seeds_for` on
armada."""


def plain_between(g):
    """`between` as its declaration states it, with no compensation: `exp(G_s - G_t)` of the plain prefix
    sums `G` of each chunk's log decays `g` `[..., C, K]`, on and below the diagonal."""
    prefix = jnp.cumsum(g, axis=1)
    below = jnp.tril(jnp.ones((g.shape[1], g.shape[1]), bool))[None, :, :, None]
    return jnp.where(below, jnp.exp(jnp.where(below, prefix[:, :, None] - prefix[:, None, :], 0.0)), 0.0)


def test_the_decay_takes_the_same_tangent_forward_as_backward():
    """Forward mode takes the decay gradient's tangent: a drawn tangent's forward-mode decays paired with a
    drawn cotangent, and the cotangent's gradient paired with the tangent, each on par with JAX's own
    derivatives of the plain decays (`plain_between`) from them in float64 (check 1), each batch row's and key
    dimension's pairing a value of its own (each walks its own decays). The two pairings are not held to
    each other: the forward one sums 64 x 64 products in float32 where the reverse one sums 64, and measured
    2.4 times its error for that alone; the tangent itself is on par with each span's own sums
    (tests/test_algebra_decays.py, row 5858)."""
    rows, keys = 4, 16

    def draw(seed):
        rng = np.random.default_rng([0x7A9E, seed])
        return tuple(jnp.asarray(x, jnp.float32) for x in (
            rng.uniform(-3, -1, (rows, 64, keys)), rng.standard_normal((rows, 64, keys)),
            rng.standard_normal((rows, 64, 64, keys))))

    def paired(decays):
        def run(g, tangent, cotangent):
            forward = jnp.sum(jax.jvp(decays, (g,), (tangent,))[1] * cotangent, axis=(1, 2))
            backward = jnp.sum(tangent * jax.vjp(decays, g)[1](cotangent)[0], axis=1)
            return {"forward": forward, "backward": backward}
        return run

    assert_as_exact(paired(between), paired(plain_between), draw, "the decays' tangent paired",
                    seeds=PAIRING_SEEDS)


ALIGNED_SEEDS = {"chunked": 8, "portable": 8}
"""The aligned keys' declared draws of each chunked form against `torch_chunk_gated_delta_rule` (check 1),
sized by `seeds_for` on armada."""


def aligned_draw(seed: int):
    """Keys nearly aligned within a chunk (a shared direction plus 0.05 of noise, normalised), unit queries,
    normal values, a slow decay (`-0.01 U(0, 1)`) and a write gate near one (`0.99 + 0.01 U(0, 1)`), fp32: the
    case real checkpoints reach."""
    rng = np.random.default_rng([0xA119, seed])
    batch, length, heads, width = 1, 128, 2, 16
    key = rng.normal(size=(1, 1, heads, width)) + 0.05 * rng.normal(size=(batch, length, heads, width))
    query = rng.normal(size=(batch, length, heads, width))
    value = rng.normal(size=(batch, length, heads, width))
    g = -0.01 * rng.random((batch, length, heads))
    beta = 0.99 + 0.01 * rng.random((batch, length, heads))
    query, key = (x / np.linalg.norm(x, axis=-1, keepdims=True) for x in (query, key))
    return tuple(jnp.asarray(x, jnp.float32) for x in (query, key, value, g, beta))


@pytest.mark.parametrize("form", ALIGNED_SEEDS)
def test_the_chunked_rule_holds_with_aligned_keys(form):
    """Keys nearly aligned within a chunk, the write gate near one and a slow decay: each chunked form's
    output and memory (the public rule, at its own chunk, and the portable math at the upstream's 64) on
    par with `torch_chunk_gated_delta_rule`'s, the chunked form they port, from that form in float64 (check
    1). Held to the token-by-token recurrence instead, the portable form measured 2.4 times the recurrent
    form's error at these draws (40 seeds), and it is on par with the upstream chunked form's there."""
    from interop_support import torch_arms

    reference_arm, truth = torch_arms(torch_chunked, torch_chunked)
    dew, _ = FORMS[form]
    assert_as_exact(dew, reference_arm, aligned_draw, f"the {form} rule at aligned keys",
                    seeds=ALIGNED_SEEDS[form], truth_fn=truth)


@pytest.mark.check1_shards(seeds=128, per_task=16)
@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("axis, split", [("data", 0), ("tensor", 2)])
def test_the_decode_kernel_keeps_the_kernel_contract_on_a_mesh(axis, split):
    """One decode token through `dew.nn.kernels.recurrence.step` over rows split two
    ways, or heads split two ways, idle rows among the drawing ones, runs per
    shard (`sharded.assert_runs_per_shard`): its output and state within the
    float64 gate of one device's, from the recurrence's in float64 (an idle
    row's output zeros and its state as it was), and no operand or state moved
    between devices. Off a GPU the kernel interprets."""
    from jax.sharding import Mesh, PartitionSpec
    from sharded import assert_runs_per_shard

    from dew.nn.linear import decode_gated_delta_rule

    # A model's 128 keys and values a head, as the template's step takes them (a thread a value column).
    rows, H, D = 4, 4, 128
    active = jnp.asarray([True, False, True, False])

    def draw(seed):
        rng = np.random.default_rng(2 + seed)
        query, key, value, g, beta, state = decode_operands(rng, rows, H, D)
        return query, key, value, g, beta, state, active

    @as_given
    def exact(*operands):
        """The recurrence in float64, an idle row's output zero and its state as it was."""
        out, final = recurrent_gated_delta_rule(*operands[:6])
        drawing = np.asarray(operands[6])[:, None, None, None]
        return np.where(drawing, np.asarray(out), 0.0), np.where(drawing, np.asarray(final), operands[5])

    def spec(rank: int, at: int):
        return PartitionSpec(*(axis if dim == at else None for dim in range(rank)))

    heads_at = {0: 0, 2: 1}[split]
    specs = [spec(4, split)] * 3 + [spec(3, split)] * 2 + [spec(4, heads_at), spec(1, split)]
    mesh = Mesh(np.asarray(jax.devices()[:2]), (axis,))
    # The kernel aliases its state into its output, and serving donates it (`donate`).
    assert_runs_per_shard(decode_gated_delta_rule, draw, specs, mesh, f"delta decode split over {axis}",
                          seeds=SHARDED_SEEDS, truth_fn=exact, donate=(5,))


KEYED = kernel.Hooks(decay='vector')
RATED = kernel.Hooks(decay='scalar', rated=True, delta=False, skip=True)
"""The decode step's hooks beside the gated delta net's: Kimi delta attention's decay a key dimension
(`linear.KEY_STEP`) and Mamba-2's rated scan with its skip (`mamba2.SSD`)."""


@pytest.mark.check1_shards(seeds=128, per_task=16)
@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("axis, split", [("data", 0), ("tensor", 2)])
@pytest.mark.parametrize("form", ["a decay a key dimension", "a rated decay"])
def test_the_decode_step_keeps_the_kernel_contract_on_a_mesh_for_each_decay(form, axis, split):
    """One decode token of Kimi delta attention's decay a key dimension, or of Mamba-2's rated scan with
    its skip (its queries and keys a group's, its rate and skip a head's), through the step kernel per
    shard (`linear.stepped_per_shard`, which `kda.py` and `mamba2.py` decode through) over rows split two
    ways or heads split two ways, idle rows among the drawing ones: its output and state within the float64
    gate of one device's, from the portable step's in float64 (`recurrence.step_reference`), no operand or
    state moved between devices (`sharded.assert_runs_per_shard`). Off a GPU the kernel interprets."""
    from jax.sharding import Mesh, PartitionSpec
    from sharded import assert_runs_per_shard

    from dew.nn.linear import stepped_per_shard

    rows, heads, groups, width = 4, 4, 2, 128
    keyed = form == "a decay a key dimension"
    hooks = KEYED if keyed else RATED
    active = jnp.asarray([True, False, True, False])

    def draw(seed):
        rng = np.random.default_rng(3 + seed)
        reads = heads if keyed else groups
        unit = [rng.normal(size=(rows, reads, width)) for _ in range(2)]
        query, key = (jnp.asarray(x / np.linalg.norm(x, axis=-1, keepdims=True), jnp.float32) for x in unit)
        value = jnp.asarray(rng.normal(size=(rows, heads, width)), jnp.float32)
        beta = jnp.asarray(rng.random((rows, heads)), jnp.float32)
        state = jnp.asarray(rng.normal(size=(rows, heads, width, width)) * 0.1, jnp.float32)
        if keyed:
            decay = jnp.asarray(-rng.random((rows, heads, width)) * 3, jnp.float32)
            return query, key, value, decay, beta, state, active
        rate = jnp.asarray(-rng.random(heads) * 3, jnp.float32)
        skip = jnp.asarray(rng.normal(size=heads), jnp.float32)
        return query, key, value, rate, beta, state, active, skip

    def stepped(query, key, value, decay_or_rate, beta, state, active, skip=None):
        if keyed:
            return stepped_per_shard(hooks, state, query, key, value, decay_or_rate, beta, active)
        return stepped_per_shard(hooks, state, query, key, value, None, beta, active, rate=decay_or_rate,
                                 skip=skip)

    @as_given
    def exact(query, key, value, decay_or_rate, beta, state, active, skip=None):
        with jax.enable_x64(new_val=True):
            given = {"log_decay": decay_or_rate} if keyed else {"rate": decay_or_rate, "skip": skip}
            out, _, final = kernel.step_reference(hooks, kernel.Step(
                jnp.asarray(state, jnp.float64), query=query, key=key, value=value, beta=beta, active=active,
                **given), dtype=jnp.float64)
            return np.asarray(out), np.asarray(final)

    def spec(rank: int, at: int):
        return PartitionSpec(*(axis if dim == at else None for dim in range(rank)))

    # A row-shaped operand split on its rows or its heads; a head's rate and skip on the heads alone.
    at = {0: 0, 2: 1}[split]
    per_head = spec(1, 0 if split == 2 else None)
    specs = [spec(3, at)] * 3 + [spec(3, at) if keyed else per_head, spec(2, at), spec(4, at),
                                 spec(1, 0 if split == 0 else None), *([] if keyed else [per_head])]
    mesh = Mesh(np.asarray(jax.devices()[:2]), (axis,))
    assert_runs_per_shard(stepped, draw, specs, mesh, f"{form} decode split over {axis}",
                          seeds=SHARDED_SEEDS, truth_fn=exact, donate=(5,))


def fused_case(heads, width, rows, dtype, held, seed: int = 0):
    """A gated delta net of `heads` value heads over half as many key heads, `width` wide, and its
    cache after a 5-token prefill of `rows` rows in `dtype` with its recurrent state `held`, and the
    next token: the layer's own conv, gates, update and gated norm unfused, and the decode token's
    inputs as the fused step takes them (`fused_step`); the hidden states and the weights `seed`'s."""
    layer = GatedDeltaNet(emb_features=64, num_k_heads=heads // 2, num_v_heads=heads, head_k_dim=width,
                          head_v_dim=width, dtype=dtype)
    hidden = jax.random.normal(jax.random.key(7 + 2 * seed), (rows, 6, 64), jnp.float32).astype(dtype)
    variables = nn.unbox(layer.init(jax.random.key(8 + 2 * seed), hidden[:, :1]))
    variables = {**variables, "params": {**variables["params"], "A_log": jnp.linspace(-1.0, 1.0, heads),
                                         "dt_bias": jnp.linspace(0.5, 1.5, heads)}}
    step = MixerCall(decode=True)
    allocated = layer.apply(variables, hidden[:, :1], step, mutable=["cache"])[1]
    cache = layer.apply({**variables, **allocated}, hidden[:, :5], step, mutable=["cache"])[1]["cache"]
    cache = {**cache, "recurrent_state": cache["recurrent_state"].astype(held)}
    return layer, variables, cache, hidden[:, 5:6]


def fused_inputs(layer, variables, cache, token):
    """The fused step's operands for `token` over `cache`, from the layer's own projections, as the
    layer hands them over on one device (`_fused_token`)."""
    mixed, z, b, a = layer.apply(variables, token, method=GatedDeltaNet._project)
    params = variables["params"]
    return (mixed[:, 0], cache["conv_state"], params["conv1d"]["weight"][:, 0, :], a[:, 0], b[:, 0],
            params["A_log"], params["dt_bias"], z[:, 0].reshape(z.shape[0], layer.num_v_heads, -1),
            params["norm"]["weight"], cache["recurrent_state"])


@pytest.mark.parametrize("staged", ["plain", "remat"])
def test_a_cached_layer_traced_for_a_jvp_decodes_its_token(staged):
    """A cached gated delta net on an A100, whose row's step kernel decodes its tokens, traced for
    `jax.jvp` as MeanFlow and consistency training trace it, plainly or under a remat that hides the
    tangent from the layer: its token decodes with the portable step's tangent (`recurrence.step`'s
    `generation.linear_tangent`), which the in-place launch does not give (row 2395)."""
    from dew.nn.kernels.generation import assumed

    layer, variables, cache, token = fused_case(4, 128, 2, jnp.float32, jnp.float32)

    def step(token):
        return layer.apply({**variables, "cache": cache}, token, MixerCall(decode=True), mutable=["cache"])[0]

    staged_step = jax.checkpoint(step) if staged == "remat" else step
    tangent = jax.random.normal(jax.random.key(9), token.shape)
    with assumed("sm80"):
        jax.make_jaxpr(lambda token, tangent: jax.jvp(staged_step, (token,), (tangent,)))(token, tangent)


def test_the_fused_decode_step_leaves_an_idle_row_alone():
    """A row the fused step marks idle: its history and state as they were, bitwise, and its output
    zeros, beside an active row that moves."""

    layer, variables, cache, token = fused_case(4, 128, 2, jnp.float32, jnp.float32)
    operands = fused_inputs(layer, variables, cache, token)
    out, history, state = jax.jit(functools.partial(
        fused_step, epsilon=layer.norm_eps, activation=layer.gate_activation,
        dtype=jnp.float32, active=jnp.asarray([False, True])))(*operands)
    np.testing.assert_array_equal(np.asarray(out[0]), 0)
    np.testing.assert_array_equal(np.asarray(history[0]), np.asarray(cache["conv_state"][0]))
    np.testing.assert_array_equal(np.asarray(state[0]), np.asarray(cache["recurrent_state"][0]))
    assert np.any(np.asarray(state[1]) != np.asarray(cache["recurrent_state"][1]))
    assert np.any(np.asarray(history[1]) != np.asarray(cache["conv_state"][1]))


FUSED_SEEDS = {jnp.float32: 8, jnp.bfloat16: 8}
"""The fused decode step's declared draws against the unfused token (check 1), by the dtype its state is held
in: sized by `seeds_for` paired with the fused step."""


@pytest.mark.parametrize("held", [jnp.float32, jnp.bfloat16])
def test_the_fused_decode_step_computes_the_layers_token(held):
    """A gated delta net's decode token in one launch (`fused_step`: the conv over the held history, the
    gates, the update and the gated norm), interpreted on a CPU at a small layer: its output through the
    layer's projection and its state, fp32 or bf16, on par (check 1) with the layer's own unfused token,
    holding the same state, from the float64 layer's: its output against the unfused token's from the state
    read in fp32, its state against the unfused token's written back `held`; its history bit for bit the
    unfused token's, and its state's dtype the one it holds. A GPU's widths are its flow's
    (tests/test_gpu_recurrence.py)."""
    heads, width, rows = 4, 128, 3
    layer = fused_case(heads, width, rows, jnp.float32, held)[0]
    wide = layer.clone(dtype=jnp.float64)
    step = MixerCall(decode=True)

    def draw(seed):
        return fused_case(heads, width, rows, jnp.float32, held, seed)[1:]

    def unfused(module, variables, cache, token):
        read = {**cache, "recurrent_state": cache["recurrent_state"].astype(
            jnp.promote_types(cache["recurrent_state"].dtype, jnp.float32))}
        want = module.apply({**variables, "cache": read}, token, step, mutable=["cache"])[0]
        written = module.apply({**variables, "cache": cache}, token, step, mutable=["cache"])[1]["cache"]
        return want, written["recurrent_state"]

    def fused(variables, cache, token):
        out, history, state = fused_step(*fused_inputs(layer, variables, cache, token),
                                         epsilon=layer.norm_eps, activation=layer.gate_activation,
                                         dtype=jnp.float32)
        flat = out.reshape(out.shape[0], 1, -1)
        return layer.apply(variables, flat, method=lambda module, out: module.out_proj(out)), history, state

    assert_as_exact(lambda *drawn: fused(*drawn)[::2], functools.partial(unfused, layer), draw,
                    f"the fused decode step, {heads} heads {width} wide held in {jnp.dtype(held).name}",
                    seeds=FUSED_SEEDS[held], truth_fn=functools.partial(unfused, wide))
    variables, cache, token = draw(0)
    _, history, state = jax.jit(fused)(variables, cache, token)
    after = layer.apply({**variables, "cache": cache}, token, step, mutable=["cache"])[1]["cache"]
    np.testing.assert_array_equal(np.asarray(history), np.asarray(after["conv_state"]))
    assert state.dtype == held


def test_a_layer_choosing_the_fused_step_allocates_its_cache_as_the_unfused_one():
    """A gated delta net choosing the fused step, as on an A100, initialized straight into decode mode:
    the allocation-only call allocates the caches the unfused layer does, reading nothing the fused
    step needs before the first token."""
    from dew.nn.kernels.generation import assumed, chosen

    module = GatedDeltaNet(emb_features=64, num_k_heads=2, num_v_heads=4, head_k_dim=128, head_v_dim=128,
                           dtype=jnp.bfloat16)
    x = jnp.zeros((2, 1, 64), jnp.bfloat16)
    unfused = jax.eval_shape(lambda: nn.unbox(module.init(jax.random.key(0), x, MixerCall(decode=True))))
    with assumed("sm80"), chosen(delta_decode="fused"):
        fused = jax.eval_shape(lambda: nn.unbox(module.init(jax.random.key(0), x, MixerCall(decode=True))))
    def described(tree):
        return jax.tree.map(lambda leaf: (leaf.shape, leaf.dtype), tree)

    assert described(fused) == described(unfused)


def _widths_draw(rows: int, heads: int, key_width: int, value_width: int):
    """`decode_operands` at `key_width` keys and `value_width` values a head, by seed."""
    def draw(seed):
        rng = np.random.default_rng(5 + seed)
        query, key, value, g, beta, state = decode_operands(rng, rows, heads, max(key_width, value_width))
        return (query[..., :key_width] / jnp.linalg.norm(query[..., :key_width], axis=-1, keepdims=True),
                key[..., :key_width] / jnp.linalg.norm(key[..., :key_width], axis=-1, keepdims=True),
                value[..., :value_width], g, beta, state[..., :key_width, :value_width])
    return draw


def _stepped(kernel: str):
    """A decode token `[B, 1, H, D]` through `linear.decode_step`, which asks the decode selector for its
    kernel (`decode_kernel`, a caller with nothing to fuse), `kernel` named (`generation.chosen`)."""
    from dew.nn.linear import STEP, decode_step

    def decoded(query, key, value, g, beta, state):
        query, key, value = (x[:, 0] for x in (query, key, value))
        with generation.chosen(delta_decode=kernel):
            out, final = decode_step(STEP, state, query * key.shape[-1] ** -0.5, key, value, g[:, 0],
                                     beta[:, 0], None)
        return out[:, None], final
    return decoded


@as_given
def _widths_exact(query, key, value, g, beta, state):
    """The recurrence one token on in float64 (`recurrent_gated_delta_rule`)."""
    out, final = recurrent_gated_delta_rule(query, key, value, g, beta, state)
    return np.asarray(out), np.asarray(final)


@pytest.mark.check1_shards(seeds=8, per_task=1)
@pytest.mark.parametrize("value_width", [512, 64])
def test_a_gated_delta_nets_decode_token_runs_the_step_at_fla_value_widths(value_width):
    """A gated delta net's decode token, its conv, gates and gated norm the fused step's to take, on an
    H100's route with the step named (`generation.chosen`, interpreted), at fla's default value heads of 512
    and of 64 over keys of 128 (two value heads over one key head): on par with the portable token (check 1,
    the layer in float64 the truth), the step's own widths holding where the fused step's (a value head a
    warpgroup, its norm reading a head whole) do not (`delta_decode_matrix.on_par`)."""
    import delta_decode_matrix

    layer = delta_decode_matrix.initialized("gated delta net", num_k_heads=1, num_v_heads=2,
                                            head_v_dim=value_width)
    with generation.assumed("sm90", interpreted=True):
        delta_decode_matrix.on_par(("sm90", "gated delta net", "step", "bfloat16"), layer)


@pytest.mark.parametrize("value_width", [512, 64])
def test_a_token_with_nothing_to_fuse_naming_the_fused_step_runs_the_step(value_width):
    """A decode token with no conv, gates or gated norm to fuse (`_stepped`, `linear.decode_step`) naming
    the fused step, on an H100's route (interpreted), at value heads of `value_width` over keys of 128: the
    fused step hands it to the step, which runs it on par with the portable token (check 1, the recurrence
    in float64 the truth), refusing it by no width the fused step's norm alone asks."""
    with generation.assumed("sm90", interpreted=True):
        assert_as_exact(_stepped("fused"), _stepped("xla"), _widths_draw(2, 2, 128, value_width),
                        "nothing to fuse", seeds=FEWEST_SEEDS, truth_fn=_widths_exact)


@pytest.mark.check1_shards(seeds=128, per_task=16)
@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("value_heads", [4, 2])
def test_a_decode_token_over_a_tensor_mesh_runs_the_step_where_each_shards_heads_fit_it(value_heads):
    """A decode token of 64-wide value heads over keys of 128 on an H100's route (interpreted), its heads
    split over a two-way tensor mesh (`stepped_per_shard`), the step named: the step's programs take two
    such heads each (`step.fits` on each shard's state), so a shard of two runs it per shard on par with the
    recurrence (`sharded.assert_runs_per_shard`), and a shard of one, no launch of the step's, refuses it."""
    from jax.sharding import Mesh, PartitionSpec
    from sharded import assert_runs_per_shard

    from dew.nn.kernels.selection import KernelRefused

    stepped = _stepped("step")
    heads = PartitionSpec(None, None, "tensor", None)
    specs = [heads] * 3 + [PartitionSpec(None, None, "tensor")] * 2 + [
        PartitionSpec(None, "tensor", None, None)]
    mesh = Mesh(np.asarray(jax.devices()[:2]), ("tensor",))
    draw = _widths_draw(2, value_heads, 128, 64)
    with generation.assumed("sm90", interpreted=True):
        if value_heads == 2:
            placed = jax.device_put(draw(0), tuple(jax.NamedSharding(mesh, spec) for spec in specs))
            with pytest.raises(KernelRefused), jax.set_mesh(mesh):
                jax.jit(stepped)(*placed)
            return
        assert_runs_per_shard(stepped, draw, specs, mesh, "64-wide heads split over tensor",
                              seeds=SHARDED_SEEDS, truth_fn=_widths_exact, donate=(5,))


@pytest.mark.parametrize("target", gate_generations(scoped=True))
def test_the_fused_step_compiles_within_each_gpus_resources(target):
    """The fused decode step, the recurrence template's step on Mosaic GPU, at Qwen3.5-0.8B's widths (16
    value heads over 16 key heads of 128, a conv of 4 taps over an fp32 history, a bf16 state) and at
    Qwen3.5-9B's value heads over half as many key heads, lowered for each GPU on this CPU and compiled by
    its compiler spill-free within its block (`kernel_compile.mosaic_failures`)."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    def case(key_heads, value_heads):
        rows, width = 8, 128
        channels = (2 * key_heads + value_heads) * width
        specs = (jax.ShapeDtypeStruct((rows, channels), jnp.bfloat16),
                 jax.ShapeDtypeStruct((rows, 3, channels), jnp.float32),
                 jax.ShapeDtypeStruct((channels, 4), jnp.bfloat16),
                 *(jax.ShapeDtypeStruct((rows, value_heads), jnp.bfloat16),) * 2,
                 *(jax.ShapeDtypeStruct((value_heads,), jnp.float32),) * 2,
                 jax.ShapeDtypeStruct((rows, value_heads, width), jnp.bfloat16),
                 jax.ShapeDtypeStruct((width,), jnp.float32),
                 jax.ShapeDtypeStruct((rows, value_heads, width, width), jnp.bfloat16))

        def run(*operands):
            return fused_step(*operands, epsilon=1e-6, activation="silu", dtype=jnp.bfloat16)
        return run, specs

    cases = [KernelCase(f"fused/{key}x{value}", lambda key=key, value=value: case(key, value))
             for key, value in ((16, 16), (16, 32))]
    found = compile_mosaic_kernels(cases, target)
    assert found and not mosaic_failures(found), "\n".join(mosaic_failures(found))


@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_documents_that_continue_another_read_its_memory_and_history_as_their_own_rows_do(reference,
                                                                                           geometry, layout):
    """A row of documents that continue another (`dew.batch.Forks`: the questions of rows over one state, a
    tree's candidates) gives each document's tokens, and the gradients through them, what its own row of its
    ancestors and itself gives, over drawn inputs (check 1): the conv reads its chain's last tokens
    (`forked_conv`) and the recurrence starts from its parent's final memory (`forked_scan`), documents
    numbered out of their order in the row, one token each among them, and a chain three deep."""
    assert_forks_as_own_rows(layer(geometry), layer_params(reference), geometry["hidden_size"], layout,
                               "gated delta net")
