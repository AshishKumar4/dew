"""Inkling 2's multi-token-prediction draft (the release's `mtp_config` and `mtp.safetensors`, row 12945) as
Dew's Markov block drafter (`dew.nn.block_draft`), against vLLM's draft ported to torch
(tools/inkling2_draft_reference.py), which the suite cannot run, over transformers' own Inkling 2 decoder's
taps: tools/inkling2_fixture.py --draft writes the tiny draft beside tests/fixtures/hf/inkling2-tiny's decoder
and records both there (recorded_draft.npz)."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from model_support import attending
from reference_error import assert_as_exact, declared_seeds, indexed_recording, recording, widened

from dew.interop import Pretrained
from dew.interop.hf_decoders import translate_config
from dew.nn.inputs import ModelInputs
from dew.sampling import Sample, Sampling, Speculative, generate

TINY = Path(__file__).parent / 'fixtures' / 'hf' / 'inkling2-tiny'
RECORDED = TINY / 'recorded_draft.npz'
RELEASE = json.loads((TINY / 'config.json').read_text())
SHAPE = (2, 40)
"""The ids' shape the recorder drew: the context, the anchor drawn after it."""
SPLIT = 24
"""Where the cached draft's context is cut: written in two calls, the second appending to the first."""


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


def drafted(model, variables, inputs):
    """The taps over the drawn ids and the draft's logits after them, uncached."""
    context = context_of(model, variables, inputs['input_ids'])
    draft = model.apply(variables, context, inputs['anchor'], decode=False, choose=forcing(inputs['forced']),
                        method='draft')
    return {'context': context, 'logits': draft.logits}


def cached(model, variables, inputs):
    """The draft's logits after its context written to its cache in two calls (`SPLIT`), drafted from it."""
    context = context_of(model, variables, inputs['input_ids'])
    _, opened = model.apply(variables, SHAPE[0], method='init_draft_cache', mutable=['cache'])
    cache = opened['cache']
    for part in (context[:, :SPLIT], context[:, SPLIT:]):
        _, written = model.apply({**variables, 'cache': cache}, part, None, method='draft', mutable=['cache'])
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
def test_its_taps_and_draft_are_on_par_with_the_releases(source, leaf):
    """The release-layout load's fp32 taps (each tapped visit's partial block, a looped layer's at its
    pass, and the final states: `context`) and its draft's logits after them, each place fed the drawn
    token before it (`logits`), on par with transformers' taps and vLLM's draft over them from their
    float64 run's (check 1, the recording's), Dew's own float64 run the truth."""
    model, variables = source.model, source.variables
    wide, held = model.clone(dtype=jnp.float64), widened(variables)
    truth_of = jax.jit(lambda inputs: drafted(wide, held, inputs)[leaf])
    seeds = declared_seeds(RECORDED)['draft']
    recorded, reference, truth = indexed_recording(
        recording(RECORDED), 'draft', seeds, truth_fn=lambda inputs: {leaf: truth_of(inputs)},
        inputs_of=draw, leaves=(leaf,))
    run = jax.jit(lambda inputs: drafted(model, variables, inputs)[leaf])
    assert_as_exact(lambda index, inputs: {leaf: run(inputs)}, reference, recorded, 'Inkling 2 draft',
                    seeds=seeds, truth_fn=truth)


@attending('reference')
def test_a_cached_draft_is_on_par_with_its_whole_context(source):
    """The draft after its context written to its cache in two calls, each row's places counted on, on par
    with the draft over the whole context uncached (check 1, the uncached in float64 the truth)."""
    model, variables = source.model, source.variables
    wide, held = model.clone(dtype=jnp.float64), widened(variables)
    whole = jax.jit(lambda variables, inputs: drafted(model, variables, inputs)['logits'])
    truth = jax.jit(lambda _, inputs: drafted(wide, held, inputs)['logits'])
    assert_as_exact(jax.jit(lambda variables, inputs: cached(model, variables, inputs)), whole,
                    lambda seed: (variables, draw(seed)), 'Inkling 2 cached draft', seeds=8, truth_fn=truth)


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


@pytest.mark.parametrize('change', [
    {'markov_head_type': 'bigram'}, {'q_bias': True}, {'hidden_size': 32}, {'dflare_mode': 'mean'},
    {'max_position_embeddings': 64},
], ids=['bigram-head', 'query-bias', 'narrower', 'unknown-field', 'short-reach'])
def test_a_draft_it_cannot_compute_is_refused(change):
    """An `mtp_config` whose head, biases, width or reach the release's draft states otherwise, or stating
    a field it states none of, is refused rather than read as some other draft; the fixture's own reads."""
    translate_config(RELEASE)
    with pytest.raises(ValueError):
        translate_config({**RELEASE, 'mtp_config': {**RELEASE['mtp_config'], **change}})


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
