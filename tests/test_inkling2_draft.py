"""Inkling 2's multi-token-prediction draft (the release's `mtp_config` and `mtp.safetensors`, row 12945) as
Dew's Markov block drafter (`dew.nn.block_draft`), against vLLM's draft ported to torch
(tools/inkling2_draft_reference.py), which the suite cannot run, over transformers' own Inkling 2 decoder's
taps: tools/inkling2_fixture.py --draft writes the tiny draft beside tests/fixtures/hf/inkling2-tiny's decoder
and records both there (recorded_draft.npz)."""

import json
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from model_support import attending
from reference_error import assert_as_exact, declared_seeds, indexed_recording, recording, widened

from dew.batch import coordinates as laid_out
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
