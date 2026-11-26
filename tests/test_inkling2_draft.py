"""Inkling 2's multi-token-prediction draft (the release's `mtp_config` and `mtp.safetensors`, row 12945) as
Dew's Markov block drafter (`dew.nn.block_draft`), against vLLM's draft ported to torch
(tools/inkling2_draft_reference.py), which the suite cannot run, over transformers' own Inkling 2 decoder's
taps: tools/inkling2_fixture.py --draft writes the tiny draft beside tests/fixtures/hf/inkling2-tiny's decoder
and records both there (recorded_draft.npz)."""

import dataclasses
import json
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from model_support import attending
from reference_error import assert_as_exact, declared_seeds, indexed_recording, recording, widened
from speculative_support import assert_replay_drafts_as_a_prefill, block_draw

from dew.batch import coordinates as laid_out
from dew.interop import Pretrained
from dew.interop.hf_decoders import translate_config
from dew.nn.block_draft import BlockDrafter, Context, Draft, DraftAttention, MarkovBlockDraft, draft
from dew.nn.inputs import ModelInputs
from dew.nn.kv_cache import KVCache
from dew.sampling import Sample, Sampling, Speculative, generate

TINY = Path(__file__).parent / 'fixtures' / 'hf' / 'inkling2-tiny'
RECORDED = TINY / 'recorded_draft.npz'
RELEASE = json.loads((TINY / 'config.json').read_text())
SHAPE = (2, 40)
"""The ids' shape the recorder drew: the context, the anchor drawn after it."""
SPLIT = 24
"""Where the cached draft's context is cut: written in two calls, the second appending to the first."""
ACCEPTED = (5, 1)
"""Each row's tokens of the second call a verification accepted, the rest padding (a block's replay)."""


def draw(seed):
    """A recorded seed's inputs, as the recorder drew them (tools/inkling2_fixture.py `draft_draw`): ids, each
    row's anchor and the token fed back at each drafted place, uniform over the head's rows."""
    rng = np.random.default_rng(seed)
    vocabulary, block = RELEASE['text_config']['unpadded_vocab_size'], RELEASE['mtp_config']['block_size']
    return {'input_ids': rng.integers(0, vocabulary, SHAPE).astype(np.int32),
            'anchor': rng.integers(0, vocabulary, (SHAPE[0],)).astype(np.int32),
            'forced': rng.integers(0, vocabulary, (SHAPE[0], block - 1)).astype(np.int32)}


def context_of(model, variables, ids):
    """The draft's context over `ids`: the forward's taps (`draft_context`)."""
    _, state = model.apply(variables, ids, mutable=['prediction_inputs'])
    return model.apply(variables, state['prediction_inputs'], method='draft_context')


def forcing(forced):
    """A choice of each drafted place's token: the drawn one, fed back at the next place."""
    return lambda index, logits: forced[:, index]


def drafted(model, variables, inputs, valid=None):
    """The taps over the drawn ids and the draft's logits after them, uncached, over the context's real
    tokens (`valid`, every one where None)."""
    context = context_of(model, variables, inputs['input_ids'])
    coordinates = None if valid is None else laid_out(valid.shape, valid=valid)
    draft = model.apply(variables, context, inputs['anchor'], decode=False, choose=forcing(inputs['forced']),
                        coordinates=coordinates, method='draft')
    return {'context': context, 'logits': draft.logits}


def accepted():
    """The real tokens of a context whose `SPLIT` first are a prefill's and whose next are a replayed
    block's, each row's `ACCEPTED` of them, the rest padding."""
    kept = SPLIT + jnp.asarray(ACCEPTED)[:, None]
    return jnp.arange(SHAPE[1])[None, :] < kept


def cached(model, variables, inputs):
    """The draft's logits after its context written to its cache in two calls, the prefill's (`SPLIT`) then
    a block's whose rows a verification accepted apart (`accepted`), drafted from it."""
    context = context_of(model, variables, inputs['input_ids'])
    _, opened = model.apply(variables, SHAPE[0], method='init_draft_cache', mutable=['cache'])
    cache = opened['cache']
    valid = accepted()
    for part in (slice(0, SPLIT), slice(SPLIT, SHAPE[1])):
        _, written = model.apply({**variables, 'cache': cache}, context[:, part], None,
                                 coordinates=laid_out(valid[:, part].shape, valid=valid[:, part]),
                                 method='draft', mutable=['cache'])
        cache = written['cache']
    draft, _ = model.apply({**variables, 'cache': cache}, None, inputs['anchor'],
                           choose=forcing(inputs['forced']), method='draft', mutable=['cache'])
    return draft.logits


@pytest.fixture(scope='module')
@attending('reference')
def source():
    return Pretrained.load(TINY, dtype='float32')


@attending('reference')
@pytest.mark.parametrize('leaf', ['context', 'logits'])
@pytest.mark.parametrize(('site', 'dtype'), [('draft', 'float32'), ('draft_bf16', 'bfloat16')])
def test_its_taps_and_draft_are_on_par_with_the_releases(source, site, dtype, leaf):
    """The release-layout load's taps (each tapped visit's partial block, a looped layer's at its pass,
    and the final states: `context`) and its draft's logits after them, each place fed the drawn token
    before it (`logits`), in fp32 and in bf16 (the release's own dtype, where every rounding point of the
    draft's arithmetic shows: its fused adds and norms, its attention's scores and weights, its taps'
    weights, its logits' sum before their cap), on par with transformers' taps and vLLM's draft over them
    at that dtype from their float64 run's (check 1, the recording's), Dew's own float64 run the truth."""
    model, variables = source.model, source.variables
    narrow = source if dtype == 'float32' else Pretrained.load(TINY, dtype=dtype)
    wide, held = model.clone(dtype=jnp.float64), widened(variables)
    truth_of = jax.jit(lambda inputs: drafted(wide, held, inputs)[leaf])
    seeds = declared_seeds(RECORDED)[site]
    recorded, reference, truth = indexed_recording(
        recording(RECORDED), site, seeds, truth_fn=lambda inputs: {leaf: truth_of(inputs)},
        inputs_of=draw, leaves=(leaf,))
    run = jax.jit(lambda inputs: drafted(narrow.model, narrow.variables, inputs)[leaf])
    assert_as_exact(lambda index, inputs: {leaf: run(inputs)}, reference, recorded,
                    f'Inkling 2 draft ({dtype})', seeds=seeds, truth_fn=truth)


@attending('reference')
def test_a_cached_draft_after_a_partly_accepted_block_is_on_par_with_its_whole_context(source):
    """The draft after its context written to its cache in two calls, the second a replayed block each row
    of which a verification accepted apart (five tokens and one, the rest padding), each row's places
    counted on and its Markov head's taps its last accepted token's, on par with the draft over that whole
    context uncached (check 1, the uncached in float64 the truth)."""
    model, variables = source.model, source.variables
    wide, held = model.clone(dtype=jnp.float64), widened(variables)
    whole = jax.jit(lambda variables, inputs: drafted(model, variables, inputs, accepted())['logits'])
    truth = jax.jit(lambda _, inputs: drafted(wide, held, inputs, accepted())['logits'])
    assert_as_exact(jax.jit(lambda variables, inputs: cached(model, variables, inputs)), whole,
                    lambda seed: (variables, draw(seed)), 'Inkling 2 cached draft', seeds=8, truth_fn=truth)


@attending('reference')
def test_a_replayed_block_drafts_as_a_prefill_of_its_history(source):
    """A speculative block's verified stretch replayed into the cache saved before it and recorded into the
    Markov drafter's windows, its rows keeping four tokens and fewer, drafts its next block as a fresh
    prefill of each row's history does (row 13317; `speculative_support`)."""
    def drawn(seed):
        return block_draw(draw(seed)['input_ids'], 20, 4, np.random.default_rng([0x1331, seed]))

    assert_replay_drafts_as_a_prefill(source.model, source.variables, drawn,
                                      "Inkling 2's replayed block's draft", seeds=8)


def test_a_count_of_passes_is_refused_where_the_taps_are_recorded(source):
    """A forward recording the draft's taps over fewer passes than the loop's (`loop_passes`) is refused:
    its taps name visits in the loop's order, which a skipped pass would leave zeros in place of."""
    ids = jnp.asarray(draw(0)['input_ids'])
    source.model.apply(source.variables, ids, loop_passes=jnp.int32(2))
    with pytest.raises(ValueError):
        source.model.apply(source.variables, ids, loop_passes=jnp.int32(2), mutable=['prediction_inputs'])


def test_a_load_keeping_no_prediction_depths_reads_no_draft(tmp_path):
    """`prediction_depths=0` loads the decoder alone: no draft is built nor its `mtp.safetensors` read, so
    a copy of the release without that file loads, where the whole release builds the draft."""
    for name in ('config.json', 'model.safetensors'):
        shutil.copy(TINY / name, tmp_path / name)
    assert Pretrained.load(TINY, dtype='float32', weights=False).model.block_draft is not None
    assert Pretrained.load(tmp_path, dtype='float32', prediction_depths=0).model.block_draft is None


@attending('reference')
def test_speculative_decoding_with_its_draft_emits_the_greedy_walk(source):
    """At zero temperature every rejected draft is replaced by the target's own token, so Speculative
    decoding with the draft's blocks emits the greedy walk, a left-padded row included."""
    ids = jnp.asarray(draw(0)['input_ids'])[:, :SPLIT]
    valid = jnp.ones(ids.shape, bool).at[1, :3].set(False)
    inputs = ModelInputs(jnp.where(valid, ids, 0), {'attention_mask': valid})
    walked, speculated = (generate(source.model, source.variables, inputs, 6, key=jax.random.key(0),
                                   sampling=Sampling(temperature=0, strategy=strategy))
                          for strategy in (Sample(), Speculative(block=4)))
    np.testing.assert_array_equal(np.asarray(speculated.tokens), np.asarray(walked.tokens))
    np.testing.assert_array_equal(np.asarray(speculated.lengths), 6)


def test_a_draft_layer_norms_its_stream_before_rounding_it():
    """A draft layer's add and norm (`added_norm`, FlashInfer's `fused_add_rmsnorm`, as vLLM's draft runs
    it) norms the stream's sum before it rounds: a bf16 stream `[1, 1]` plus `[2^-8, -2^-9]` rounds to `[1,
    1]`, whose norm is `[1, 1]`, where the unrounded sum's second part normalizes below one by more than a
    bf16 step's half."""
    from dew.nn.block_draft import added_norm
    from dew.nn.norms import RMSNorm

    norm = RMSNorm(epsilon=1e-6, dtype=jnp.bfloat16)
    residual = jnp.ones((1, 2), jnp.bfloat16)
    x = jnp.asarray([[2.0 ** -8, -2.0 ** -9]], jnp.bfloat16)
    params = norm.init(jax.random.key(0), residual)
    stream, normed = norm.apply(params, x, method=lambda module, x: added_norm(module, x, residual))
    np.testing.assert_array_equal(np.asarray(stream, np.float32), [[1.0, 1.0]])
    assert float(normed[0, 1]) == 1.0 - 2.0 ** -8 and float(normed[0, 0]) == 1.0, normed


TINY_DRAFT = MarkovBlockDraft(layers=1, num_heads=1, num_kv_heads=1, head_dim=2, mlp_features=2, block_size=3,
                              taps=(0, 1, 2), markov_rank=1, rope_theta=10000.0, norm_eps=1e-6)
"""A draft of one layer over two-wide states and three taps, small enough to set its weights by hand."""


def tiny_drafter() -> BlockDrafter:
    return BlockDrafter(TINY_DRAFT, emb_features=2, vocab_rows=4, kv_cache=KVCache(), max_seq_len=8,
                        dtype=jnp.bfloat16)


def test_a_drafts_taps_mix_by_their_weights_rounded():
    """The taps' mixing weights round to the draft's dtype after their softmax, as vLLM's
    (`refresh_derived_weights`): weights `softmax([0, 0, log 3])` over taps whose first coordinates are `[3,
    0, -1]` mix it to `3 (0.2) - 0.6`, zero in fp32 and `-2^-10` with the bf16 weights `0.2002` and
    `0.6016`, which the Markov head's carried taps keep, normed, clearly below zero."""
    drafter = tiny_drafter()
    taps = jnp.asarray([[[3.0, 1.0, 0.0, 1.0, -1.0, 1.0]]], jnp.bfloat16)
    params = drafter.init(jax.random.key(0), taps, None, 1, decode=False, method='contexts')['params']
    params = {**params, 'fuse_weight': jnp.zeros((2, 3)).at[:, 2].set(np.log(3.0))}
    _, carried = drafter.apply({'params': params}, taps, None, 1, decode=False, method='contexts')
    assert float(carried[0, 0]) < -1e-3, carried


def test_a_drafts_attention_weighs_its_values_unrounded():
    """A draft layer's attention weighs its values by its probabilities unrounded and rounds its output once,
    as vLLM's: three keys of equal scores (its queries zero) over values `1, 1, 1 + 2^-7` weigh to `1 +
    2^-7 / 3`, which rounds to 1 in bf16, where probabilities rounded to `0.334` weigh them to `1 + 2^-7`."""
    attention = DraftAttention(TINY_DRAFT, emb_features=2, local=False, kv_cache=KVCache(), max_seq_len=8,
                               dtype=jnp.bfloat16)
    one = 1.0 + 2.0 ** -7
    context = Context(jnp.zeros((1, 2, 1, 2), jnp.bfloat16),
                      jnp.asarray([[[[1.0, 1.0]], [[one, one]]]], jnp.bfloat16),
                      jnp.asarray([[0, 1]]), jnp.ones((1, 2), bool), jnp.asarray([2]))
    x = jnp.asarray([[[1.0, 0.0]]], jnp.bfloat16)
    params = attention.init(jax.random.key(0), x, context)['params']
    params = {**params, 'q_proj': {'kernel': jnp.zeros((2, 2))},
              'v_proj': {'kernel': jnp.asarray([[1.0, 1.0], [0.0, 0.0]])},
              'o_proj': {'kernel': jnp.eye(2)}}
    out = attention.apply({'params': params}, x, context)
    np.testing.assert_array_equal(np.asarray(out, np.float32), [[[1.0, 1.0]]])


def test_a_drafts_head_scores_its_states_and_its_correction_apart():
    """A drafted place's logits are the head's scores of its states and of the Markov head's correction,
    each rounded to the draft's dtype, summed and rounded again (vLLM's `markov_logits`), never the scores
    of their sum: every score the head gives (`scores_of`) is of a place's states or a correction alone,
    and each place's logits are its two scores' rounded sum."""
    drafter = tiny_drafter()
    taps = jnp.ones((1, 4, 6), jnp.bfloat16)
    anchor = jnp.zeros((1,), jnp.int32)
    embedded = jnp.ones((1, 2), jnp.bfloat16)
    row = jnp.asarray([[1.0], [1.0]], jnp.bfloat16)
    scored = []

    def scores_of(x):
        out = (x @ row).astype(jnp.float32)
        scored.append((x, out))
        return out

    def drafting(module):
        return draft(module, taps, None, anchor, lambda tokens: embedded, scores_of, None, decode=False)

    variables = drafter.init(jax.random.key(0), method=drafting)
    scored.clear()
    drafted_block = drafter.apply(variables, method=drafting)
    assert isinstance(drafted_block, Draft)
    logits = drafted_block.logits
    (states, base), *corrections = scored
    places = TINY_DRAFT.block_size - 1
    assert states.shape == (1, places, 2) and len(corrections) == places

    def rounded(x):
        return x.astype(jnp.bfloat16).astype(jnp.float32)

    for index, (_, correction) in enumerate(corrections):
        np.testing.assert_array_equal(np.asarray(logits[:, index]),
                                      np.asarray(rounded(rounded(base[:, index]) + rounded(correction))))


def test_a_markov_draft_through_a_head_not_linear_in_its_states_is_refused():
    """A Markov block drafter sums the head's scores of a place's states and of its correction, which a head
    bias would count twice: a decoder with one is refused as it is built; the fixture's builds."""
    model = translate_config(RELEASE).value
    tokens = jnp.zeros((1, 4), jnp.int32)
    jax.eval_shape(lambda: model.init(jax.random.key(0), tokens))
    with pytest.raises(ValueError):
        jax.eval_shape(lambda: dataclasses.replace(model, head_bias=True).init(jax.random.key(0), tokens))


def built(taps):
    """The fixture's decoder with its draft tapping `taps`, built abstractly (its setup's checks run)."""
    draft = {**RELEASE['mtp_config'], 'target_layer_ids': taps}
    model = translate_config({**RELEASE, 'mtp_config': draft}).value
    return jax.eval_shape(lambda: model.init(jax.random.key(0), jnp.zeros((1, 4), jnp.int32)))


@pytest.mark.parametrize('taps', [[1, 7, 13, 15], [1, 7, 13, 17]], ids=['short', 'past'])
def test_taps_past_the_targets_visits_are_refused(taps):
    """A draft whose last tap is not its target's count of visits (a looped layer's each pass: 16 for the
    fixture's 2 + 4 x 3 + 2), which stands for its final states, is refused as the model is built; the
    fixture's own taps build."""
    built(RELEASE['mtp_config']['target_layer_ids'])
    with pytest.raises(ValueError):
        built(taps)
