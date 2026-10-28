"""Phi-3's fused GQA and LongRoPE against transformers 5.17.0."""

import contextlib
import dataclasses
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from interop_support import fixture_ids, transformers_arms, transformers_model
from reference_error import as_given, assert_as_exact, widened

from dew.interop import Pretrained
from dew.interop.sources import load_shards

ROOT = Path(__file__).parent / 'fixtures' / 'hf'
DIRECTORY = ROOT / 'phi3-tiny'


LOGITS_SEEDS = {False: 0, True: 0}
"""The long and short logits' declared draws of ids against transformers' (check 1), unpadded and
left-padded, sized by `seeds_for` paired with Dew's logits on the CPU."""


@pytest.mark.parametrize('padded', [False, True])
def test_phi3_short_and_long_logits_match_transformers(padded):
    """Rows past LongRoPE's 8-token original context read the long factors and their first four ids
    alone the short ones: Dew's logits for each are on par with transformers', from its float64 twin's,
    over drawn ids (check 1), with the same argmax; left-padded rows under their mask, at the positions
    the mask counts, too."""
    import torch

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    mask = np.load(DIRECTORY / 'attention_mask.npy') if padded else None
    valid = np.ones(np.load(DIRECTORY / 'input_ids.npy').shape, bool) if mask is None else mask
    # The fields its recorder (tools/hf_reference.py `write_classic_tiny`) gave each row.
    positions = None if mask is None else np.maximum(np.cumsum(mask, -1) - 1, 0)
    fields = {} if mask is None else {'attention_mask': torch.from_numpy(mask),
                                      'position_ids': torch.from_numpy(positions)}
    draw = fixture_ids(DIRECTORY, mask)

    def ours(ids):
        logits = {'long': loaded.model.apply(loaded.variables, ids, attention_mask=valid)[valid]}
        return logits if padded else {**logits, 'short': loaded.model.apply(loaded.variables, ids[:, :4])}

    def read(model, ids):
        logits = {'long': model(ids, **fields, use_cache=False).logits[torch.from_numpy(valid)]}
        return logits if padded else {**logits, 'short': model(ids[:, :4], use_cache=False).logits}

    reference, truth = transformers_arms(transformers_model(DIRECTORY), read)
    assert_as_exact(ours, reference, draw, f'Phi-3 {"padded " if padded else ""}logits',
                    seeds=LOGITS_SEEDS[padded], truth_fn=truth)
    (ids,) = draw(0)
    np.testing.assert_array_equal(np.asarray(ours(jnp.asarray(ids))['long']).argmax(-1),
                                  reference(ids)['long'].argmax(-1))


def test_phi3_cached_greedy_crosses_the_longrope_boundary():
    from dew.inference.tasks import TextGeneration
    from dew.sampling.text import Sampling

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference', max_seq_len=48)
    ids = np.load(DIRECTORY / 'input_ids.npy')
    task = TextGeneration(loaded.model, loaded.variables, sampling=Sampling(temperature=0))
    generated = task(ids[:, :4], max_new_tokens=6, key=0)
    np.testing.assert_array_equal(generated.tokens, np.load(DIRECTORY / 'generated.npy'))


CROSSING_SEEDS = 0
"""The cached walk's declared draws of ids against transformers' uncached forward at each step (check 1),
sized by `seeds_for` paired with Dew's walk on the CPU."""


def test_phi3_cache_rebuild_logits_match_uncached_transformers_on_both_sides_of_the_crossing():
    """Four ids prefilled and six more fed one at a time through the cache, which rebuilds where the
    rows grow past LongRoPE's original context: each step's logits are on par with transformers'
    uncached forward's last over the ids so far, its factor chosen by their length, from its float64
    twin's, over drawn ids (check 1), with the same argmax."""
    from dew.nn.inputs import ModelInputs
    from dew.sampling.text import decode_ops, prefill_state

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference', max_seq_len=48)
    ops = decode_ops(loaded.model, loaded.variables, 0, 0)

    def ours(ids):
        ids = jnp.asarray(ids)
        state, _ = prefill_state(loaded.model, loaded.variables, ModelInputs(ids[:, :4]), ops)
        steps = {}
        for step in range(6):
            steps[f'step {step}'] = state.logits
            state = ops.advance(state, ids[:, 4 + step], jnp.ones(2, bool))
        return steps

    reference, truth = transformers_arms(transformers_model(DIRECTORY), lambda model, ids: {
        f'step {step}': model(ids[:, :4 + step], use_cache=False).logits[:, -1] for step in range(6)})
    draw = fixture_ids(DIRECTORY)
    assert_as_exact(ours, reference, draw, 'Phi-3 crossing', seeds=CROSSING_SEEDS, truth_fn=truth)
    (ids,) = draw(0)
    for step, logits in ours(jnp.asarray(ids)).items():
        np.testing.assert_array_equal(np.asarray(logits).argmax(-1), reference(ids)[step].argmax(-1))


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


MIXED_SEEDS = 32
"""The mixed-table rows' declared draws of ids against each row's batch-one reference (check 1), sized by
`seeds_for` paired with Dew's rows on the CPU."""


def test_phi3_mixed_tables_match_each_rows_batch_one_reference_under_the_float64_rule():
    import torch
    from transformers import AutoModelForCausalLM

    from tools.torch_precision import float64

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference', max_seq_len=64)
    shape = np.load(DIRECTORY / 'input_ids.npy')[:, :4].shape
    positions = np.stack((np.arange(12, 16), np.arange(4)))
    reference = AutoModelForCausalLM.from_pretrained(DIRECTORY, attn_implementation='eager').float().eval()
    with float64():
        truth_model = AutoModelForCausalLM.from_config(
            reference.config, dtype=torch.float64, attn_implementation='eager').eval()
    truth_model.load_state_dict(reference.state_dict())
    vocabulary = json.loads((DIRECTORY / 'config.json').read_text())['vocab_size']

    def draw(seed):
        return (np.random.default_rng(seed).integers(0, vocabulary, shape).astype(np.int32),)

    def rows(model, precision):
        """Each row's logits at batch one, the reference's own table per row."""
        def run(ids):
            ids = np.asarray(ids)
            with torch.no_grad(), precision():
                return {f'row {row}': model(torch.from_numpy(ids[row:row + 1]).long(),
                                            position_ids=torch.from_numpy(positions[row:row + 1]).long(),
                                            use_cache=False).logits.double().numpy()[0] for row in range(2)}
        return as_given(run)

    def ours(ids):
        logits = loaded.model.apply(loaded.variables, ids, positions=positions)
        return {f'row {row}': logits[row] for row in range(2)}

    assert_as_exact(ours, rows(reference, contextlib.nullcontext), draw, 'Phi-3 mixed rows',
                    seeds=MIXED_SEEDS, truth_fn=rows(truth_model, float64))
    (ids,) = draw(0)
    actual = np.asarray(loaded.model.apply(loaded.variables, ids, positions=positions))
    expected = np.stack([rows(reference, contextlib.nullcontext)(ids)[f'row {row}'] for row in range(2)])
    np.testing.assert_array_equal(actual.argmax(-1), expected.argmax(-1))

    uniform = loaded.model.apply(loaded.variables, ids, positions=np.stack((np.arange(4), np.arange(4))))
    np.testing.assert_array_equal(actual[1], uniform[1])


PACKED_SEEDS = 8
"""Each packing's declared draws of ids against each document alone (check 1), sized by `seeds_for` paired
with the packed row on the CPU."""


def test_phi3_packed_documents_each_take_their_own_longrope_table():
    """A packed row selects LongRoPE's short or long factors per document, by how far the
    document reaches past the 8-token original context, as the document alone does: a 3-token
    document beside a 9-token one reads the short table, the 9-token one the long table. A slice
    of a packed row (`ModelInputs.slice_tokens`, as the LM objective cuts its inputs) keeps its
    documents' ids: the last two tokens of a fourth document, at places 7 and 8, are a two-token
    row holding id 4, which reads the long table. Each document's logits are on par with its
    logits alone from their float64 run over drawn ids (check 1)."""
    from dew.nn.inputs import ModelInputs

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference', max_seq_len=48)
    vocabulary = json.loads((DIRECTORY / 'config.json').read_text())['vocab_size']
    wide = loaded.model.clone(dtype=jnp.float64)
    for lengths, start in (((3, 9), None), ((6,), None), ((1, 1, 1, 9), -2)):
        fill = np.zeros(12 - sum(lengths), np.int32)
        segments = np.concatenate([np.full(n, i, np.int32) for i, n in enumerate(lengths, 1)] + [fill])
        places = np.concatenate([np.arange(n, dtype=np.int32) for n in lengths] + [fill])

        def packed(tokens, start=start, places=places, segments=segments):
            """The packed row of `tokens`, sliced as the objective cuts it."""
            return ModelInputs(jnp.asarray(tokens[None]), {'positions': jnp.asarray(places[None]),
                                                           'segment_ids': jnp.asarray(segments[None])}
                               ).slice_tokens(start)

        fields = packed(np.zeros(12, np.int32)).token_fields
        held = {name: np.asarray(value)[0] for name, value in fields.items()}
        documents = np.unique(held['segment_ids'][held['segment_ids'] > 0])

        def draw(seed, lengths=lengths, fill=fill):
            drawn = np.random.default_rng(seed).integers(0, vocabulary, sum(lengths)).astype(np.int32)
            return (np.concatenate([drawn, fill]),)

        def in_place(tokens, packed=packed, held=held, documents=documents):
            inputs = packed(tokens)
            logits = loaded.model.apply(loaded.variables, inputs.tokens, **inputs.kwargs())[0]
            return {f'document {document}': logits[held['segment_ids'] == document] for document in documents}

        def by_itself(model, variables, packed=packed, held=held, documents=documents):
            def run(tokens):
                inputs = packed(tokens)
                return {f'document {document}': model.apply(
                    variables, inputs.tokens[0][held['segment_ids'] == document][None],
                    positions=held['positions'][held['segment_ids'] == document][None])[0]
                    for document in documents}
            return run

        assert_as_exact(in_place, by_itself(loaded.model, loaded.variables), draw,
                        f'documents of {lengths} from {start}', seeds=PACKED_SEEDS,
                        truth_fn=lambda tokens, by=by_itself: by(wide, widened(loaded.variables))(tokens))


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
