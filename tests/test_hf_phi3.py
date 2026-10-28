"""Phi-3's fused GQA and LongRoPE against transformers 5.16.1."""

import dataclasses
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference, float64_forward

from dew.interop import Pretrained
from dew.interop.sources import load_shards

ROOT = Path(__file__).parent / 'fixtures' / 'hf'
DIRECTORY = ROOT / 'phi3-tiny'


@pytest.mark.parametrize('padded', [False, True])
def test_phi3_short_and_long_logits_match_transformers(padded):
    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    ids = np.load(DIRECTORY / ('padded_ids.npy' if padded else 'input_ids.npy'))
    mask = np.load(DIRECTORY / 'attention_mask.npy') if padded else np.ones_like(ids, bool)
    prefix = 'padded_' if padded else ''
    actual = np.asarray(loaded.model.apply(loaded.variables, ids, attention_mask=mask))
    _, terms = float64_forward(loaded.model, loaded.variables, ids, attention_mask=mask)
    expected, truth = (np.load(DIRECTORY / f'{prefix}logits{suffix}.npy') for suffix in ('', '_f64'))
    assert_as_exact_as_the_reference(actual[mask], expected[mask], truth[mask], 'Phi-3 long logits',
                                     terms=np.asarray(terms)[mask])
    np.testing.assert_array_equal(actual[mask].argmax(-1), expected[mask].argmax(-1))
    if not padded:
        short = loaded.model.apply(loaded.variables, ids[:, :4])
        _, terms = float64_forward(loaded.model, loaded.variables, ids[:, :4])
        assert_as_exact_as_the_reference(short, np.load(DIRECTORY / 'short_logits.npy'),
                                         np.load(DIRECTORY / 'short_logits_f64.npy'), 'Phi-3 short logits',
                                         terms=terms)


def test_phi3_cached_greedy_crosses_the_longrope_boundary():
    from dew.inference.tasks import TextGeneration
    from dew.sampling.text import Sampling

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference', max_seq_len=48)
    ids = np.load(DIRECTORY / 'input_ids.npy')
    task = TextGeneration(loaded.model, loaded.variables, sampling=Sampling(temperature=0))
    generated = task(ids[:, :4], max_new_tokens=6, key=0)
    np.testing.assert_array_equal(generated.tokens, np.load(DIRECTORY / 'generated.npy'))


def test_phi3_cache_rebuild_logits_match_uncached_transformers_on_both_sides_of_the_crossing():
    from dew.nn.inputs import ModelInputs
    from dew.sampling.text import decode_ops, prefill_state

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference', max_seq_len=48)
    generated = np.load(DIRECTORY / 'generated.npy')
    ops = decode_ops(loaded.model, loaded.variables, 0, 0)
    state, _ = prefill_state(loaded.model, loaded.variables,
                        ModelInputs(jnp.asarray(generated[:, :4], jnp.int32)), ops)
    outputs = []
    for step in range(6):
        outputs.append(np.asarray(state.logits))
        state = ops.advance(state, jnp.asarray(generated[:, 4 + step]), jnp.ones(2, bool))
    actual = np.stack(outputs, axis=1)
    reference = np.load(DIRECTORY / 'generation_logits.npy')
    truth = np.load(DIRECTORY / 'generation_logits_f64.npy')
    for step in range(6):
        # Each step's logits are the uncached forward's last over the tokens so far, LongRoPE's
        # factor chosen by their length, as transformers' were.
        _, terms = float64_forward(loaded.model, loaded.variables, generated[:, :4 + step])
        assert_as_exact_as_the_reference(actual[:, step], reference[:, step], truth[:, step],
                                         f'Phi-3 crossing step {step}', terms=np.asarray(terms)[:, -1])
    np.testing.assert_array_equal(actual.argmax(-1), generated[:, 4:])


def test_a_kinds_longrope_rebuilds_the_cache_at_its_crossing_as_the_models_does():
    """LongRoPE given on the layers' kind rather than on the model switches its factors at the same
    crossing, which the cache rebuilds at (`cache_rebuild_position`, over every layer's kind): the
    cached walk across it computes, bit for bit, what the model-level ramp's does, where a rebuild read
    from the model alone never came and the cache kept keys rotated by the short factors (audit
    NN-04)."""
    from dew.nn.backbones.layer_plan import LayerKind
    from dew.nn.inputs import ModelInputs
    from dew.sampling.text import decode_ops, prefill_state

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference', max_seq_len=48)
    kinds = dict(loaded.model.kinds or {})
    kinded = loaded.model.clone(rope_scaling=None, kinds={**kinds, **{
        kind: dataclasses.replace(kinds.get(kind, LayerKind()), rope_scaling=loaded.model.rope_scaling)
        for kind in set(loaded.model.per_layer_types)}})
    generated = jnp.asarray(np.load(DIRECTORY / 'generated.npy'), jnp.int32)

    def walk(model):
        ops = decode_ops(model, loaded.variables, 0, 0)
        state, _ = prefill_state(model, loaded.variables, ModelInputs(generated[:, :4]), ops)
        outputs = []
        for step in range(6):
            outputs.append(np.asarray(state.logits))
            state = ops.advance(state, generated[:, 4 + step], jnp.ones(2, bool))
        return np.stack(outputs, axis=1)

    np.testing.assert_array_equal(walk(kinded), walk(loaded.model))


def test_phi3_mixed_tables_match_each_rows_batch_one_reference_under_the_float64_rule():
    import torch
    from transformers import AutoModelForCausalLM

    from tools.torch_precision import float64

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference', max_seq_len=64)
    ids = np.load(DIRECTORY / 'input_ids.npy')[:, :4]
    positions = np.stack((np.arange(12, 16), np.arange(4)))
    actual = np.asarray(loaded.model.apply(loaded.variables, ids, positions=positions))
    reference = AutoModelForCausalLM.from_pretrained(DIRECTORY, attn_implementation='eager').float().eval()

    def rows(model):
        return np.concatenate([
            model(torch.from_numpy(ids[row:row + 1]).long(),
                  position_ids=torch.from_numpy(positions[row:row + 1]).long(),
                  use_cache=False).logits.detach().numpy() for row in range(2)], axis=0)

    with torch.no_grad():
        expected = rows(reference)
    with float64(), torch.no_grad():
        truth_model = AutoModelForCausalLM.from_config(
            reference.config, dtype=torch.float64, attn_implementation='eager').eval()
        truth_model.load_state_dict(reference.state_dict())
        truth = rows(truth_model)
    _, terms = float64_forward(loaded.model, loaded.variables, ids, positions=positions)
    for row in range(2):
        assert_as_exact_as_the_reference(actual[row], expected[row], truth[row], f'Phi-3 mixed row {row}',
                                         terms=np.asarray(terms)[row])
    np.testing.assert_array_equal(actual.argmax(-1), expected.argmax(-1))

    uniform = loaded.model.apply(loaded.variables, ids, positions=np.stack((np.arange(4), np.arange(4))))
    np.testing.assert_array_equal(actual[1], uniform[1])


def test_phi3_packed_documents_each_take_their_own_longrope_table():
    """A packed row selects LongRoPE's short or long factors per document, by how far the
    document reaches past the 8-token original context, as the document alone does: a 3-token
    document beside a 9-token one reads the short table, the 9-token one the long table. A slice
    of a packed row (`ModelInputs.slice_tokens`, as the LM objective cuts its inputs) keeps its
    documents' ids: the last two tokens of a fourth document, at places 7 and 8, are a two-token
    row holding id 4, which reads the long table. Each document's logits are as exact as its
    logits alone."""
    from dew.nn.inputs import ModelInputs

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference', max_seq_len=48)
    ids = np.load(DIRECTORY / 'input_ids.npy')[0].astype(np.int32)
    for lengths, start in (((3, 9), None), ((6,), None), ((1, 1, 1, 9), -2)):
        fill = np.zeros(12 - sum(lengths), np.int32)
        segments = np.concatenate([np.full(n, i, np.int32) for i, n in enumerate(lengths, 1)] + [fill])
        places = np.concatenate([np.arange(n, dtype=np.int32) for n in lengths] + [fill])
        tokens = np.concatenate([ids[:sum(lengths)], fill])
        inputs = ModelInputs(jnp.asarray(tokens[None]), {'positions': jnp.asarray(places[None]),
                                                         'segment_ids': jnp.asarray(segments[None])})
        inputs = inputs.slice_tokens(start)
        logits = np.asarray(loaded.model.apply(loaded.variables, inputs.tokens, **inputs.kwargs()))[0]
        held = {name: np.asarray(value)[0] for name, value in inputs.token_fields.items()}
        for document in np.unique(held['segment_ids'][held['segment_ids'] > 0]):
            span = held['segment_ids'] == document
            alone_ids, alone_places = np.asarray(inputs.tokens)[0][span][None], held['positions'][span][None]
            alone = np.asarray(loaded.model.apply(loaded.variables, alone_ids, positions=alone_places))[0]
            truth, terms = float64_forward(loaded.model, loaded.variables, alone_ids, positions=alone_places)
            assert_as_exact_as_the_reference(logits[span], alone, np.asarray(truth)[0],
                                             f'document {document} of {lengths} from {start}',
                                             terms=np.asarray(terms)[0])


def test_phi3_dense_server_and_generate_share_the_per_row_crossing_rebuild():
    from dew.inference import TextGeneration
    from dew.inference.serving import Server
    from dew.sampling import Sampling

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference', max_seq_len=64)
    task = TextGeneration(loaded.model, loaded.variables,
                          sampling=Sampling(temperature=0, eos_token_ids=None))
    ids = np.load(DIRECTORY / 'input_ids.npy').astype(np.int32)
    prompts = (ids[0, :4], ids[1, :2])
    server = Server.from_task(task, slots=2, capacity=64, admission=2)
    assert server.mixed_refusal is not None and 'LongRoPE crossing position 8' in server.mixed_refusal
    tickets = [server.submit(prompt, 8, key=index) for index, prompt in enumerate(prompts)]
    server.run()
    for index, (ticket, prompt) in enumerate(zip(tickets, prompts, strict=True)):
        served = ticket.result().host()
        alone = task(prompt[None], 8, key=index).host()
        np.testing.assert_array_equal(served.tokens, alone.tokens)
        np.testing.assert_array_equal(served.lengths, alone.lengths)


def test_a_fused_projection_and_its_part_stored_apart_must_agree(tmp_path):
    """Phi-3 fuses gate and up as gate_up_proj, so a checkpoint that also
    stores the gate under its own name holds that part twice. An equal copy
    loads as the fused tensor alone does; an unequal one is refused rather
    than one of the two silently winning."""
    from dew.interop.safetensors_io import save_hf_layout

    tensors = load_shards(DIRECTORY)
    config = json.loads((DIRECTORY / 'config.json').read_text())
    fused = tensors['model.layers.0.mlp.gate_up_proj.weight']
    gate = fused[: fused.shape[0] // 2]
    ids = np.load(DIRECTORY / 'input_ids.npy')
    alone = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')

    part = 'model.layers.0.mlp.gate_proj.weight'
    save_hf_layout({**tensors, part: gate.copy()}, config, tmp_path / 'equal')
    twice = Pretrained.load(tmp_path / 'equal', dtype='float32', attention_impl='reference')
    np.testing.assert_array_equal(twice.model.apply(twice.variables, ids),
                                  alone.model.apply(alone.variables, ids))

    save_hf_layout({**tensors, part: gate + 1}, config, tmp_path / 'unequal')
    with pytest.raises(ValueError):
        Pretrained.load(tmp_path / 'unequal', dtype='float32', attention_impl='reference')
