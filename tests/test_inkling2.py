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


@pytest.mark.parametrize(('section', 'change'), [
    ('text_config', {'local_layer_ids': [0, 1, 2, 3, 4, 5]}), ('text_config', {'rope_theta': 10000.0}),
    ('text_config', {'loop_config': {**RELEASE['text_config']['loop_config'], 'core_step': 1}}),
    ('text_config', {'scoring_func': 'softmax'}), ('mtp_config', {'markov_head_type': 'bigram'}),
    ('mtp_config', {'q_bias': True}), ('mtp_config', {'hidden_size': 32}),
    ('mtp_config', {'dflare_mode': 'mean'}), ('mtp_config', {'max_position_embeddings': 64}),
], ids=['kinds-disagreeing', 'unknown-field', 'unknown-loop-field', 'softmax-router', 'draft-bigram-head',
        'draft-query-bias', 'draft-narrower', 'draft-unknown-field', 'draft-short-reach'])
def test_a_config_it_cannot_compute_is_refused(section, change):
    """A text config whose layers' kinds transformers and the release's own fields state apart, stating a
    field the release states none of, or scoring its router otherwise, or a draft (`mtp_config`, row 12945)
    whose head, biases, width or reach the release's states otherwise, or stating a field it states none
    of, is refused rather than read as some other model; the fixture's own config, which each changes one
    field of, reads (the logits' test)."""
    translate_config(RELEASE)
    with pytest.raises(ValueError):
        translate_config({**RELEASE, section: {**RELEASE[section], **change}})


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


def release_tensors(fixture: str = 'inkling2-tiny') -> dict[str, np.ndarray]:
    """A tiny release's every tensor, its draft's under the `mtp.` its load reads them by, its float8 codes
    in their own dtype (Dew's reader: safetensors' numpy one names a dtype NumPy lacks)."""
    from dew.interop.safetensors_io import read_file

    def load_file(path):
        return read_file(path)[0]

    directory = FIXTURES / fixture
    sidecar = directory / 'mtp.safetensors'
    draft = {f'mtp.{name}': value for name, value in load_file(sidecar).items()} if sidecar.exists() else {}
    return {**{name: value for file in sorted(directory.glob('model*.safetensors'))
               for name, value in load_file(file).items()}, **draft}


def test_its_prepares_inverse_gives_back_every_tensor_of_the_release():
    """`unprepare` is `prepare`'s exact inverse over the release's every tensor, trunk and draft (row
    13561): its queries restacked, its message queries, controllers, dynamic taps and fused experts as
    stored, each back under its own name, bit for bit; and a prepared tensor gone (one of a layer's query
    sites, its MLP's query, a loop layer's every message query, a controller's or an expert's half, a
    layer's last expert whole, a loop layer's whole controller, every dynamic convolution's taps'
    projections), or
    layer 0's query holding a value no read reaches, is refused, as is a static_w its projection is not
    beside."""
    from dew.interop.families.inkling import INKLING2

    record = Pretrained.load(TINY, dtype='float32', weights=False).model_config
    tensors = release_tensors()
    prepared = dict(INKLING2.prepare(tensors, record))
    back = INKLING2.unprepare(prepared, record)
    assert sorted(back) == sorted(tensors)
    for name, stored in tensors.items():
        written = np.asarray(back[name])
        assert (written.dtype, written.shape, written.tobytes()) == (stored.dtype, stored.shape,
                                                                     stored.tobytes()), name
    first = record['loop']['layers'][0]
    messages = {f'model.llm.layers.{first}.controllers.{site}.query' for site in ('attn', 'mlp', 'k', 'v')}
    last = record['mixture']['experts'] - 1
    expert = {f'model.llm.layers.2.mlp.experts.{last}.{part}.weight' for part in ('gate_proj', 'up_proj',
                                                                                    'down_proj')}
    controller = {name for name in prepared if name.startswith(f'model.llm.layers.{first}.controllers.mlp.')}
    projections = {name for name in prepared if name.endswith('.k_sconv.weight_proj')}
    for gone in ({'model.llm.layers.2.controllers.k.query'}, {'model.llm.layers.3.k_query'}, controller,
                 projections,
                 {'model.llm.layers.2.mlp.experts.3.up_proj.weight'}, {'model.llm.layers.2.mlp_query'},
                 messages, expert):
        with pytest.raises(ValueError):
            INKLING2.unprepare({name: value for name, value in prepared.items() if name not in gone},
                               record)
    # A static_w beside no projection of its taps is half a dynamic convolution: no spelling of the release.
    projection = next(name for name in tensors if name.endswith('.k_sconv.weight_proj'))
    with pytest.raises(ValueError):
        INKLING2.prepare({name: value for name, value in tensors.items() if name != projection}, record)
    query = 'model.llm.layers.0.attn_query'
    touched = {**prepared, query: np.ones_like(prepared[query])}
    with pytest.raises(ValueError):
        INKLING2.unprepare(touched, record)


@pytest.mark.parametrize('fixture', ['inkling2-tiny', 'inkling2-tiny-nvfp4'])
def test_a_saved_source_loads_back_bitwise_its_changed_controller_and_taps_included(tmp_path, fixture):
    """A save of the release with one controller's forget projection and one dynamic convolution's taps'
    projection changed (a ramp, so a reversal of its taps would move it) loads back as the variables it
    was saved from, bit for bit, every tensor of the release written under its own name, the draft's in
    its own file (row 13561); the quantized release's expert banks, which its training left as loaded,
    written back as it stored them, codes and scales bit for bit (train 1366's review)."""
    from flax.traverse_util import flatten_dict, unflatten_dict

    from dew.interop.safetensors_io import read_file

    def load_file(path):
        return read_file(path)[0]

    loaded = load(fixture)
    paths = {layout.name: layout.paths[0] for layout in loaded.weight_layouts}
    flat = flatten_dict(loaded.variables)
    changed = dict(flat)
    for suffix in ('.controllers.attn.forget', '.k_sconv.weight_proj'):
        path = next(path for name, path in sorted(paths.items()) if name.endswith(suffix))
        leaf = np.asarray(flat[path])
        changed[path] = leaf + np.arange(leaf.size, dtype=leaf.dtype).reshape(leaf.shape) / leaf.size
    variables = unflatten_dict(changed)
    loaded.save(tmp_path, variables=variables)
    saved = {name for file in tmp_path.glob('*.safetensors') if file.name != 'mtp.safetensors'
             for name in load_file(file)}
    sidecar = tmp_path / 'mtp.safetensors'
    drafted = {f'mtp.{name}' for name in load_file(sidecar)} if sidecar.exists() else set()
    source = release_tensors(fixture)
    assert saved | drafted == set(source)
    written = {name: value for file in tmp_path.glob('model*.safetensors')
               for name, value in load_file(file).items()}
    for name in (name for name in source if '.experts.' in name):
        held = (written[name].dtype, written[name].tobytes())
        assert held == (source[name].dtype, source[name].tobytes()), name
    reloaded = flatten_dict(Pretrained.load(tmp_path, dtype='float32').variables)
    assert sorted(reloaded) == sorted(changed)
    for path, value in changed.items():
        np.testing.assert_array_equal(np.asarray(reloaded[path]), np.asarray(value), err_msg=str(path))


def test_a_source_whose_loop_changed_is_refused_its_save():
    """A save writes the release's config as published, so a loaded source whose loop computes otherwise
    than its config states is refused before a tensor is written (row 13561): one whose
    messages mix their summaries as they are, where the release's normalize them. One run at another pass
    count is refused before that, as a model: its draft's taps name the loop's visits."""
    import dataclasses

    from dew.interop.decoder_parts import ExportRefused

    loaded = load('inkling2-tiny')
    loop = loaded.model.loop
    plain = dataclasses.replace(loop, messages=dataclasses.replace(
        loop.messages, normalize_values=not loop.messages.normalize_values))
    with pytest.raises(ExportRefused):
        dataclasses.replace(loaded, model=loaded.model.clone(loop=plain)).export()
    passes = dataclasses.replace(loaded.model.loop, steps=loaded.model.loop.steps + 1)
    with pytest.raises(ValueError):
        dataclasses.replace(loaded, model=loaded.model.clone(loop=passes)).export()


def test_an_nvfp4_bank_writes_back_every_code_its_scales_decode_unchanged():
    """Inkling 2's NVFP4 writer keeps each stored code whose decode the bank still holds bit for bit, where
    no quotient recovers it: a zero group scale, under which every code decodes to zero, and an expert
    factor of 2^-149 that a group scale of 0.5 times underflows, under which code 7 (6.0) decodes to
    3 x 2^-149
    (train 1366's review of theme 59); and a changed value takes its nearest code."""
    import ml_dtypes

    from dew.interop.codecs import inkling2_experts

    name = 'model.llm.layers.2.mlp.experts.gate_up_proj'
    codes = np.full((2, 1, 16), 0x77, np.uint8)
    scale = np.asarray([[[0.0, 0.5]], [[0.5, 0.5]]], ml_dtypes.float8_e4m3fn)
    amax = np.asarray([2.0 ** -149, 1.0], np.float32)
    denominator = np.ones(2, np.float32)
    stored = {name: codes, name + '_scale': scale, name + '_amax': amax,
              name + '_outer_scale_denominator': denominator}
    codec = inkling2_experts(stored, {name: 'float32'})
    values = np.asarray(codec.decode(stored, name), np.float32)
    assert values[0, 0, 0] == 0 and values[0, 0, 16] == np.float32(3 * 2.0 ** -149)
    written = codec.encode(name, values)
    np.testing.assert_array_equal(written[name], codes)
    changed = values.copy()
    changed[1, 0, 0] = 1.0
    retouched = codec.encode(name, changed)[name]
    # 1.0 over its group's 0.5 is E2M1's 2.0, code 4, in its byte's low nibble.
    assert retouched[1, 0, 0] == 0x74 and (retouched != codes).sum() == 1
    # A quotient just below E2M1's 0.75 midpoint, which a float32 quotient rounds onto it (and its tie to
    # the even code 2): 1.125 + 2^-23 over a group scale of 1.5 and an expert factor of 1 + 2^-23 takes the
    # nearer code 1 (0.5).
    tiny = 2.0 ** -23
    near = {name: np.zeros((1, 1, 8), np.uint8),
            name + '_scale': np.asarray([[[1.5]]], ml_dtypes.float8_e4m3fn),
            name + '_amax': np.asarray([1 + tiny], np.float32),
            name + '_outer_scale_denominator': np.ones(1, np.float32)}
    nearby = np.zeros((1, 1, 16), np.float32)
    nearby[0, 0, 0] = np.float32(1.125 + tiny)
    assert inkling2_experts(near, {name: 'float32'}).encode(name, nearby)[name][0, 0, 0] == 0x01


def test_an_mxfp8_bank_held_in_bfloat16_writes_back_the_codes_it_was_loaded_from():
    """Inkling 2's MXFP8 writer compares each value against its code's decode as the load kept it, in the
    dtype the load decoded the bank into, whatever dtype the export hands it in: a code whose decode
    `2^-136` (E4M3's `2^-9` under E8M0's `2^-127`) a bfloat16 load holds as 0 writes back as the code it
    was loaded from, not as a fresh 0, from the bfloat16 tree and from that tree widened to float32 (train
    1370's review F2, train 1379's P-F3), and a value its training changed takes its nearest code."""
    import ml_dtypes

    from dew.interop.codecs import inkling2_experts

    name = 'model.llm.layers.2.mlp.experts.down_proj'
    codes = np.full((1, 2, 32), 1.0, ml_dtypes.float8_e4m3fn)
    codes[0, 0] = ml_dtypes.float8_e4m3fn(2.0 ** -9)
    exponents = np.asarray([[[0], [127]]], np.uint8)
    stored = {name: codes, name + '_scale': exponents}
    codec = inkling2_experts(stored, {name: 'bfloat16'})
    decoded = np.asarray(codec.decode(stored, name), np.float32)
    assert decoded[0, 0, 0] == np.float32(2.0 ** -136)
    held = decoded.astype(ml_dtypes.bfloat16)
    assert held[0, 0, 0] == 0
    for tree in (held, held.astype(np.float32)):
        np.testing.assert_array_equal(codec.encode(name, tree)[name].view(np.uint8), codes.view(np.uint8))
    changed = held.copy()
    changed[0, 1, 0] = ml_dtypes.bfloat16(2.0)
    written = codec.encode(name, changed)[name]
    assert written[0, 1, 0] == ml_dtypes.float8_e4m3fn(2.0)
    assert (written.view(np.uint8) != codes.view(np.uint8)).sum() == 1


@pytest.mark.parametrize('widened', [False, True], ids=['bfloat16', 'widened to float32'])
def test_a_bfloat16_load_of_quantized_banks_saves_them_as_stored(tmp_path, widened):
    """The quantized release loaded in bfloat16 and saved from its tree as loaded, or from that tree
    widened to float32 (an export writes each leaf in the dtype it is handed), writes its expert banks back
    as it stored them, codes and scales bit for bit: each value told apart from its load's in the dtype the
    load decoded it into (`Pretrained.quantized_dtypes`, train 1379's P-F3)."""
    from dew.interop.safetensors_io import read_file

    loaded = Pretrained.load(FIXTURES / 'inkling2-tiny-nvfp4', dtype='float32', param_dtype='bfloat16')
    assert loaded.quantized_dtypes and set(loaded.quantized_dtypes.values()) == {'bfloat16'}
    def wide(leaf):
        return np.asarray(leaf, np.float32) if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf

    variables = jax.tree.map(wide, loaded.variables) if widened else loaded.variables
    loaded.save(tmp_path, variables=variables)
    written = {name: value for file in tmp_path.glob('model*.safetensors')
               for name, value in read_file(file)[0].items()}
    source = release_tensors('inkling2-tiny-nvfp4')
    banks = [name for name in source if '.experts.' in name]
    assert banks
    for name in banks:
        held = (written[name].dtype, written[name].tobytes())
        assert held == (source[name].dtype, source[name].tobytes()), name
