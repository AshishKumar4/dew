"""Inkling 2's text decoder against transformers' own (huggingface/transformers at 3a96649,
models/inkling2_loop/modeling_inkling2_loop.py), which the suite's transformers predates:
tools/inkling2_fixture.py writes tests/fixtures/hf/inkling2-tiny in the release's names and layout, and
inkling2-tiny-nvfp4 with its experts stored as the release stores them (NVFP4 and MXFP8), and records their
logits there, whole and at the shapes a cached decode computes them."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from model_support import allocated_cache, attending, steps_from
from reference_error import assert_as_exact, declared_seeds, indexed_recording, recording, widened

from dew.interop import Pretrained
from dew.interop.hf_decoders import translate_config

FIXTURES = Path(__file__).parent / 'fixtures' / 'hf'
TINY = FIXTURES / 'inkling2-tiny'
SHAPE = (2, 40)
"""The ids' shape check 1 draws in, as the recorder drew them (tools/inkling2_fixture.py)."""
STEPPED = 24
"""The tokens the recorder's cached decode prefills before it steps a token at a time."""
RELEASE = json.loads((TINY / 'config.json').read_text())


def load(fixture):
    return Pretrained.load(FIXTURES / fixture, dtype='float32')


def draw(seed):
    """Check 1's ids, uniform over the embedding's rows, as the recorder drew them."""
    vocabulary = RELEASE['text_config']['vocab_size']
    return np.random.default_rng(seed).integers(0, vocabulary, SHAPE).astype(np.int32)


def decoded(model, variables, ids):
    """`ids`' logits from a cached decode: the first `STEPPED` prefilled, then a token a call."""
    cache = allocated_cache(model, variables, ids.shape[0])
    prefilled, state = model.apply({**variables, 'cache': cache}, ids[:, :STEPPED], decode=True,
                                   mutable=['cache'])
    steps, _ = steps_from(model, variables, state['cache'], ids[:, STEPPED:])
    return jnp.concatenate([prefilled, steps], axis=1)


@attending("reference")
@pytest.mark.parametrize('leaf', ['logits', 'stepped_logits'])
@pytest.mark.parametrize('fixture', ['inkling2-tiny', 'inkling2-tiny-nvfp4'])
def test_its_logits_are_on_par_with_transformers(fixture, leaf):
    """The release-layout load's fp32 logits, whole (`logits`) and through its cache (`stepped_logits`: 24
    ids prefilled, then one a call through the loop's per-pass caches, the depth reads' blocks and the
    short convolutions' histories), on par with transformers' fp32 run from its float64 twin (check 1),
    over its recording's seeds: Dew's own float64 whole forward is the truth, held at every seed to the
    recording's sketch. The head scores the release's unpadded rows alone; the NVFP4 and MXFP8 experts
    decode as transformers dequantizes them (`codecs.inkling2_experts`)."""
    loaded = load(fixture)
    wide = loaded.model.clone(dtype=jnp.float64)
    path = FIXTURES / fixture / 'recorded_logits.npz'
    seeds = declared_seeds(path)['logits']
    truth_of = jax.jit(lambda ids: {leaf: wide.apply(widened(loaded.variables), ids)})
    recorded, reference, truth = indexed_recording(
        recording(path), 'logits', seeds, truth_fn=lambda inputs: truth_of(inputs['input_ids']),
        inputs_of=lambda seed: {'input_ids': draw(seed)}, leaves=(leaf,))
    model, variables = loaded.model, loaded.variables
    run = (jax.jit(lambda ids: model.apply(variables, ids)) if leaf == 'logits'
           else jax.jit(lambda ids: decoded(model, variables, ids)))
    assert_as_exact(lambda index, inputs: {leaf: run(inputs['input_ids'])}, reference, recorded, 'Inkling 2',
                    seeds=seeds, truth_fn=truth)


@pytest.mark.parametrize('change', [
    {'local_layer_ids': [0, 1, 2, 3, 4, 5]}, {'rope_theta': 10000.0},
    {'loop_config': {**RELEASE['text_config']['loop_config'], 'core_step': 1}}, {'scoring_func': 'softmax'},
], ids=['kinds-disagreeing', 'unknown-field', 'unknown-loop-field', 'softmax-router'])
def test_a_config_it_cannot_compute_is_refused(change):
    """A text config whose layers' kinds transformers and the release's own fields state apart, stating a
    field the release states none of, or scoring its router otherwise, is refused rather than read as some
    other model; the fixture's own config, which each changes one field of, reads (the logits' test)."""
    translate_config(RELEASE)
    with pytest.raises(ValueError):
        translate_config({**RELEASE, 'text_config': {**RELEASE['text_config'], **change}})


@attending("reference")
def test_a_cached_decode_at_given_places_scores_as_its_whole_call():
    """The release-layout load decoding at places the caller gives from 100 on (24 prefilled, then one a
    call), on par with its whole call at those places (check 1, the whole call in float64 the truth): its
    relative bias reads each token's slot in the stream against each key's, as transformers' does
    (`get_relative_distance`), and its temperatures the places themselves."""
    loaded = load('inkling2-tiny')
    model, variables = loaded.model, loaded.variables
    places = jnp.broadcast_to(jnp.arange(SHAPE[1]) + 100, SHAPE)

    def stepped(variables, ids):
        cache = allocated_cache(model, variables, ids.shape[0])
        prefilled, state = model.apply({**variables, 'cache': cache}, ids[:, :STEPPED], decode=True,
                                       positions=places[:, :STEPPED], mutable=['cache'])

        def step(cache, column):
            token, place = column
            logits, state = model.apply({**variables, 'cache': cache}, token[:, None], decode=True,
                                        positions=place[:, None], mutable=['cache'])
            return state['cache'], logits[:, 0]

        # The steps one program under a scan (`steps_from`'s), where unrolled they compiled 16 decodes.
        _, steps = jax.lax.scan(step, state['cache'], (jnp.moveaxis(ids[:, STEPPED:], 1, 0),
                                                       jnp.moveaxis(places[:, STEPPED:], 1, 0)))
        return jnp.concatenate([prefilled, jnp.moveaxis(steps, 0, 1)], axis=1)

    wide = model.clone(dtype=jnp.float64)
    whole = jax.jit(lambda variables, ids: model.apply(variables, ids, positions=places))
    held = widened(variables)
    truth = jax.jit(lambda _, ids: wide.apply(held, ids, positions=places))
    assert_as_exact(jax.jit(stepped), whole, lambda seed: (variables, draw(seed)),
                    'Inkling 2 at given places', seeds=8, truth_fn=truth)
