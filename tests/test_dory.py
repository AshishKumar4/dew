"""Dory, NVIDIA's looped nGPT, against vLLM's forward (vllm-project/vllm#60149, dory.py at
55076950): tools/dory_reference.py transcribes it and writes tests/fixtures/hf/dory-tiny, and
tests/fixtures/hf/dory-1.0-4b holds the released config."""

import dataclasses
import itertools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from interop_support import assert_reloads, assert_same_stored_tensors
from reference_error import assert_as_exact_as_the_reference, assert_rounds_where_the_reference_does
from tool_support import load as load_tool

from dew.checkpoints import Checkpoints
from dew.config.data import loader
from dew.data.dataset import Loader, replayed
from dew.decision import Choice, DecisionObjective, Example, JointLayout, JointSchemaHead
from dew.decision.data import Decisions
from dew.inference import TextGeneration
from dew.inference.serving import Server
from dew.interop import Pretrained, PretrainedDecoder, hf_decoders
from dew.interop.safetensors_io import read_file
from dew.interop.sources import load_shards
from dew.lora import LoRA
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import Mixture
from dew.nn.inputs import ModelInputs
from dew.nn.kv_cache import KVCache
from dew.nn.mixers.gated_delta_net import GatedDeltaNetMixer
from dew.nn.mixers.mlp import MLPMixer
from dew.objectives.lm import LMObjective
from dew.registry import models
from dew.sampling import Sampling
from dew.training import Trainer

FIXTURES = Path(__file__).parent / 'fixtures' / 'hf'
TINY = FIXTURES / 'dory-tiny'
RELEASED = FIXTURES / 'dory-1.0-4b'
REFERENCE = np.load(TINY / 'reference.npz')
TOOL = load_tool('dory_reference')


def load(**fields):
    return Pretrained.load(TINY, dtype='float32', attention_impl='xla', **fields)


def with_cache(model: CausalTransformer, mode: str) -> CausalTransformer:
    """`model` caching its looped layers as `recurrent_kv_cache_mode` `mode` says."""
    assert model.loop is not None
    return model.clone(loop=dataclasses.replace(model.loop, shared_cache=mode == 'last_loop'))


def changed(directory: Path, change: dict) -> Path:
    """The tiny Dory under `directory` with `change` over its config, its other files linked."""
    directory.mkdir()
    for source in TINY.iterdir():
        (directory / source.name).symlink_to(source)
    (directory / 'config.json').unlink()
    stated = json.loads((TINY / 'config.json').read_text())
    (directory / 'config.json').write_text(json.dumps({**stated, **change}))
    return directory


LOGIT_TOLERANCE = 2e-5
"""The largest absolute difference from the fp32 reference's logits a forward may show. The logits
reach 20, where an fp32 unit is 1.9e-6; the observed differences are 5.7e-6 at one loop, 6.7e-6 at
two, 7.6e-6 at four, and 5.7e-6 for the packed rows (xla and reference kernels), on CPU, fp32."""


def assert_parity(logits, name: str, label: str) -> None:
    """CONTRIBUTING's statement of logits parity against the reference's array `name`: the same
    argmax, within `LOGIT_TOLERANCE` of the fp32 reference, and within tests/reference_error.py's
    rule against float64."""
    reference = REFERENCE[name]
    np.testing.assert_array_equal(logits.argmax(-1), reference.argmax(-1))
    assert float(np.max(np.abs(logits - reference))) < LOGIT_TOLERANCE, label
    assert_as_exact_as_the_reference(logits, reference, REFERENCE[f'{name}_f64'], label)


@pytest.mark.parametrize('loops', [1, 2, 4])
def test_the_looped_stack_and_its_output_table_compute_vllms_logits_at_each_loop_count(loops):
    """The tiny Dory's logits with its recurrent block run once, twice (its own count) and four
    times, as vLLM's forward computes them, within tests/reference_error.py's rule of its fp32 run
    against float64; the same of its final states contracted with its output table, the head's
    matrix and its per-token gain past the product, as a tiled loss scores them."""
    loaded = load()
    model = loaded.model.with_passes(loops)
    ids = REFERENCE['input_ids']
    name = f'logits_{loops}'
    assert_parity(np.asarray(model.apply(loaded.variables, ids)), name, f'Dory logits at {loops} loops')
    table = model.apply(loaded.variables, method='output_table')
    hidden = model.apply(loaded.variables, ids, method='hidden_states')
    assert_parity(np.asarray(table.logits(hidden)), name, f"Dory's output table at {loops} loops")


def test_bf16_keys_rotate_and_normalize_where_vllm_rounds():
    """In bf16 the first layer's keys, the projections of the same embeddings, rotate as vLLM's
    CUDA rotary does (cos and sin rounded to bf16, the turn in fp32, one rounding) and unit-norm in
    fp32, so the keys a prefill caches sit within a quarter of the reference's own bf16 error of
    its keys (`assert_rounds_where_the_reference_does`); fp32 tables, which round only the turn,
    sit a whole bf16 rounding apart."""
    loaded = Pretrained.load(TINY, dtype='bfloat16', attention_impl='xla')
    ids = REFERENCE['input_ids']
    cache = loaded.model.apply(loaded.variables, ids.shape[0], method=CausalTransformer.init_cache,
                               mutable=['cache'])[1]['cache']
    _, written = loaded.model.apply({**loaded.variables, 'cache': cache}, ids, decode=True, mutable=['cache'])
    keys = np.asarray(written['cache']['layers_0']['self_attn']['cached_key'][:, :ids.shape[1]], np.float64)
    assert_rounds_where_the_reference_does(keys, REFERENCE['keys_bf16'], REFERENCE['keys_f64'], 'bf16 keys')


@pytest.mark.parametrize('mode', ['per_loop', 'last_loop'])
def test_a_cached_greedy_continuation_reads_each_cache_as_vllm_keeps_it(mode):
    """Greedy decoding over the cache vLLM keeps: one entry a recurrent layer and pass
    ('per_loop'), or one a recurrent layer that each pass overwrites ('last_loop', a pass reading
    its own keys for the step's tokens and the last pass's for the earlier ones). The tokens are
    the reference's and each step's logits within the rule of its fp32 decode against float64.
    A per-pass cache is the full forward's: the uncached logits of the whole continuation hold to
    the same rule at the positions each step decoded."""
    loaded = load()
    model = with_cache(loaded.model, mode)
    expected = REFERENCE[f'generated_{mode}']
    prompt = jnp.asarray(expected[None, :6])
    variables = loaded.variables
    cache = model.apply(variables, 1, method=CausalTransformer.init_cache, mutable=['cache'])[1]['cache']
    logits, mutated = model.apply({**variables, 'cache': cache}, prompt, decode=True, mutable=['cache'])
    steps = [np.asarray(logits[0, -1])]
    for token in expected[6:-1]:
        logits, mutated = model.apply({**variables, 'cache': mutated['cache']}, jnp.asarray([[token]]),
                                      decode=True, mutable=['cache'])
        steps.append(np.asarray(logits[0, -1]))
    steps = np.stack(steps)
    np.testing.assert_array_equal(steps.argmax(-1), expected[6:])
    reference, truth = REFERENCE[f'step_logits_{mode}'], REFERENCE[f'step_logits_{mode}_f64']
    assert_as_exact_as_the_reference(steps, reference, truth, f"Dory's {mode} decoding")
    if mode == 'per_loop':
        whole = np.asarray(model.apply(variables, jnp.asarray(expected[None, :-1])))[0, 5:]
        assert_as_exact_as_the_reference(whole, reference, truth, "Dory's forward over its continuation")


@pytest.mark.parametrize('attention_impl', ['xla', 'reference'])
def test_packed_documents_read_only_themselves_through_every_pass(attention_impl):
    """Rows of two and three documents, each restarting its positions, give each document the
    logits vLLM computes of it alone, within the rule: no pass of the loop, nor the input its
    passes reinject, reads another document, and the window of 5 crosses a document's start in
    every row."""
    loaded = Pretrained.load(TINY, dtype='float32', attention_impl=attention_impl)
    logits = loaded.model.apply(loaded.variables, REFERENCE['packed_ids'],
                                positions=REFERENCE['packed_positions'],
                                segment_ids=REFERENCE['packed_segments'])
    assert_parity(np.asarray(logits), 'packed_logits', 'Dory packed documents')


def greedy(mode: str) -> TextGeneration:
    loaded = load()
    return TextGeneration(with_cache(loaded.model, mode), loaded.variables,
                          sampling=Sampling(temperature=0.0))


def ragged_prompts() -> list[list[int]]:
    rows = zip(REFERENCE['ragged_prompts'], TOOL.RAGGED, strict=True)
    return [row[:length].tolist() for row, length in rows]


@pytest.mark.parametrize('mode', ['per_loop', 'last_loop'])
def test_prompts_of_two_lengths_decode_side_by_side_as_each_does_alone(mode):
    """Two prompts of different lengths, the shorter padded on the left, prefilled and decoded in
    one batch through either cache, each continue greedily as the reference continues that prompt
    alone: every row's passes rewind and advance its own cursor."""
    prompts = ragged_prompts()
    width = max(map(len, prompts))
    tokens = np.stack([np.pad(prompt, (width - len(prompt), 0)) for prompt in prompts]).astype(np.int32)
    valid = np.arange(width)[None] >= np.asarray([[width - len(prompt)] for prompt in prompts])
    generated = greedy(mode)(ModelInputs(jnp.asarray(tokens), {'attention_mask': jnp.asarray(valid)}), 6,
                             key=0).host()
    np.testing.assert_array_equal(generated.tokens[:, generated.prompt_width:], REFERENCE[f'ragged_{mode}'])


@pytest.mark.parametrize(('mode', 'options'), [
    ('per_loop', {}),
    ('per_loop', {'kv_cache': KVCache(page_size=4, pages=40), 'chunk': 2, 'prefix_cache': True}),
    ('last_loop', {}), ('last_loop', {'kv_cache': KVCache(page_size=4, pages=40)})],
    ids=['per_loop-dense', 'per_loop-paged-chunked-prefix', 'last_loop-dense', 'last_loop-paged'])
def test_a_server_draws_each_prompt_as_the_reference_continues_it(mode, options):
    """Served side by side, over a dense cache or a pool of pages, each prompt continues as the
    reference continues it alone, and so does the longer prompt asked again once a slot frees. A
    per-pass cache also takes its prompts in pieces of two, the second ask reusing the first's
    pages; a shared one prefills each prompt whole, as the reference's is."""
    task = greedy(mode)
    server = Server.from_task(task, slots=2, capacity=32, **options)
    prompts = [*ragged_prompts(), ragged_prompts()[1]]
    tickets = [server.submit(prompt, 6, key=index) for index, prompt in enumerate(prompts)]
    server.run()
    expected = [*REFERENCE[f'ragged_{mode}'], REFERENCE[f'ragged_{mode}'][1]]
    for ticket, continuation in zip(tickets, expected, strict=True):
        served = ticket.result().host()
        np.testing.assert_array_equal(served.tokens[0, served.prompt_width:], continuation)
    assert (server.prefix_hits > 0) == bool(options.get('prefix_cache'))


@pytest.mark.parametrize('options', [
    {'chunk': 2}, {'kv_cache': KVCache(page_size=4, pages=40), 'prefix_cache': True}],
    ids=['chunked', 'prefix'])
def test_a_shared_loop_cache_takes_each_prompt_whole(options):
    """A last-loop cache holds the last pass's keys of every token before a call, so a prompt's
    second piece, or a prompt after a reused prefix, would read those where the whole prompt's
    passes read their own: a server refuses chunked prefill and prefix reuse over it, naming the
    per-pass cache that serves both."""
    with pytest.raises(ValueError, match='per_loop'):
        Server.from_task(greedy('last_loop'), slots=2, capacity=32, **options)


@pytest.mark.parametrize(('name', 'loops', 'backprop_steps'), [('trained', 2, None), ('truncated', 3, 1)])
def test_the_gradient_of_each_tensor_is_the_references(name, loops, backprop_steps):
    """The gradient of a loss over the logits in the embeddings, an input layer's output-gate
    projection, a looped feed-forward and attention gain, an output layer's step and the logit
    gain, as the reference's autograd takes it, within the rule: through every pass, and with the
    loop cut before its last pass, where the input layers still train through what that pass
    reinjects (Huginn's truncation). Each is named by the export, as the checkpoint stores it."""
    loaded = load()
    model = loaded.model.with_passes(loops, backprop_steps=backprop_steps)
    ids, cotangent = REFERENCE['input_ids'], REFERENCE['cotangent']

    def loss(params):
        return jnp.sum(model.apply({**loaded.variables, 'params': params}, ids) * cotangent)

    gradients = loaded.export({'params': jax.grad(loss)(loaded.variables['params'])})
    for tensor in TOOL.GRADIENTS:
        assert_as_exact_as_the_reference(gradients[tensor], REFERENCE[f'gradient_{name}/{tensor}'],
                                         REFERENCE[f'gradient_{name}_f64/{tensor}'], f'{name} {tensor}')


def test_a_cut_loop_trains_the_layers_before_it_through_the_input_every_pass_reads():
    """Cutting the loop before its last pass moves no forward value and no gradient past the
    loop, sends the layers before it a gradient through the input the last pass reinjects, and
    drops the earlier passes' share of every gradient before the loop."""
    loaded = load()
    ids, cotangent = REFERENCE['input_ids'], REFERENCE['cotangent']

    def run(backprop_steps):
        model = loaded.model.with_passes(3, backprop_steps=backprop_steps)

        def loss(params):
            return jnp.sum(model.apply({**loaded.variables, 'params': params}, ids) * cotangent)

        return jax.value_and_grad(loss)(loaded.variables['params'])

    (whole, uncut), (cut, truncated) = run(None), run(1)
    assert whole == cut
    np.testing.assert_array_equal(truncated['layers_7']['self_attn']['down_proj']['kernel'],
                                  uncut['layers_7']['self_attn']['down_proj']['kernel'])
    assert np.any(truncated['embed_tokens']['embedding'])
    assert np.any(truncated['layers_1']['self_attn']['down_proj']['kernel'])
    assert not np.array_equal(truncated['layers_1']['self_attn']['down_proj']['kernel'],
                              uncut['layers_1']['self_attn']['down_proj']['kernel'])


PROJECTIONS = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'g_proj', 'gate_proj', 'up_proj', 'down_proj')
"""Every projection of a Dory layer by its checkpoint name: an attention's four and its output
gate's (`g_proj`), and a feed-forward's three."""


def train(objective, directory: Path, steps: int):
    """`objective` trained `steps` steps on one batch of ids, checkpointing each step under `directory`,
    resuming from what it holds."""
    batch = {'text': np.asarray(np.concatenate([REFERENCE['input_ids']] * 4)[:, :9], np.int32)}
    data = Loader(replayed(lambda partition: itertools.repeat(batch)), None, batch=8, records=None)
    checkpoints = Checkpoints(str(directory))
    state = Trainer(objective, optax.adam(1e-2), key=0, checkpoints=checkpoints).fit(
        data, steps=steps, log_every=1, checkpoint_every=1)
    checkpoints.wait()
    return jax.device_get(state.variables)


def test_a_lora_on_every_projection_trains_resumes_saves_and_merges(tmp_path):
    """A rank-2 adapter on all eight projections of every layer, the attention's output gate
    (`g_proj`) apart from the feed-forward's `gate_proj`: one step moves every B factor, so the
    loss reaches each projection; a run resumed after its first step ends where two uninterrupted
    steps end; the factors save as PEFT names them and load back; and the merged checkpoint
    reloads to the adapted model's logits."""
    loaded = load()
    adapted = loaded.adapt(LoRA(rank=2, modules=PROJECTIONS), key=0)
    targets = {'/'.join(module[1:]) for module in adapted.adapter.targets}
    sites = {'mlp': ('gate_proj', 'up_proj', 'down_proj'),
             'attention': ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj')}
    assert targets == {f'layers_{layer}/self_attn/{site}'
                       for layer, kind in enumerate(loaded.model.per_layer_types)
                       for site in sites['mlp' if kind == 'mlp' else 'attention']}

    def objective():
        return LMObjective(adapted.model, 8, variables=adapted.variables)

    once = train(objective(), tmp_path / 'resumed', 1)
    for module in adapted.adapter.targets:
        factor = jax.tree_util.keystr(module)
        before = adapted.variables['params']
        after = once['params']
        for name in module[1:]:
            before, after = before[name], after[name]
        assert np.any(after['lora_B'] != before['lora_B']), factor
    resumed = train(objective(), tmp_path / 'resumed', 2)
    whole = train(objective(), tmp_path / 'whole', 2)
    jax.tree.map(np.testing.assert_array_equal, resumed['params'], whole['params'])

    adapted.adapter.save(resumed, tmp_path / 'adapter')
    saved, _ = read_file(tmp_path / 'adapter' / 'adapter_model.safetensors')
    assert 'base_model.model.backbone.layers.0.mixer.g_proj.lora_A.weight' in saved
    assert 'base_model.model.backbone.layers.1.mixer.gate_proj.lora_B.weight' in saved
    again = LoRA.load(loaded.model, loaded.variables, tmp_path / 'adapter', layouts=loaded.layouts)
    ids = REFERENCE['input_ids']
    logits = np.asarray(adapted.model.apply(resumed, ids))
    np.testing.assert_array_equal(np.asarray(again.model.apply(again.variables, ids)), logits)
    adapted.save(tmp_path / 'merged', variables=resumed)
    merged = Pretrained.load(tmp_path / 'merged', dtype='float32', attention_impl='xla')
    np.testing.assert_allclose(np.asarray(merged.model.apply(merged.variables, ids)), logits, atol=1e-5)


INTENT = Choice('which team?', {'billing': 'payments', 'technical': 'outages', 'sales': 'pricing'})
EXAMPLES = [Example(state, {'team': INTENT}, {'team': label})
            for state, label in (('charged twice', 'billing'), ('site is down', 'technical'),
                                 ('how much is pro', 'sales'), ('refund please', 'billing'),
                                 ('error 500', 'technical'), ('upgrade cost', 'sales'),
                                 ('double charge', 'billing'), ('cannot log in', 'technical'))]


def test_a_joint_decision_head_reads_dorys_vocabulary_rows_and_trains(tmp_path):
    """The joint schema head's lexical prior gathers the output table's rows: the head's matrix
    rows times each token's logit gain, which contracted with the final states give Dory's logits.
    Over the release's own tokenizer it trains a step that moves the head and the backbone, and
    the saved run answers as the live one does."""
    loaded = load(max_seq_len=2048)
    ids = REFERENCE['input_ids']
    table = loaded.model.apply(loaded.variables, method='output_table')
    hidden = loaded.model.apply(loaded.variables, ids, method='hidden_states')
    assert_as_exact_as_the_reference(np.asarray(hidden @ table.rows().T), REFERENCE['logits_2'],
                                     REFERENCE['logits_2_f64'], "Dory's vocabulary rows")
    head = JointSchemaHead(32, width=16, routing_layers=1, layers=1, heads=2, feedforward=32)
    objective = DecisionObjective(loaded, layout=JointLayout(max_len=2048), head=head)
    data = loader(Decisions(source=EXAMPLES, validation=EXAMPLES), batch=8, read='memory',
                  objective=objective.data())
    trainer = Trainer(objective, optax.adam(1e-3), key=0, checkpoints=Checkpoints(str(tmp_path)))
    initial = trainer.initial_state()
    state = trainer.fit(data, steps=1, log_every=1, state=jax.tree.map(lambda leaf: leaf, initial))
    Checkpoints(str(tmp_path)).wait()
    for part in ('head', 'backbone'):
        before, after = initial.variables['params'][part], state.variables['params'][part]
        moved = jax.tree.map(lambda old, new: not np.array_equal(old, new), before, after)
        assert any(jax.tree.leaves(moved)), part
    asked = {'team': INTENT}
    live = objective.pipeline(state)('charged twice', asked)
    reloaded = objective.saved_task.from_run(str(tmp_path))('charged twice', asked)
    for answer, again in zip(live.values(), reloaded.values(), strict=True):
        np.testing.assert_array_equal(answer.probabilities, again.probabilities)


def test_a_dory_config_loads_as_its_looped_ngpt():
    """The config's fields land where Dory computes them: a block a layer, an attention alone or
    an MLP in the mixer's slot, the loop over the recurrent ones, reinjecting, and the cache mode."""
    model = load().model
    assert model.num_layers == 8 and model.ngpt is not None and not model.pre_norms and not model.mlp_features
    assert model.loop is not None and model.loop.layers == (2, 6) and model.loop.steps == 2
    assert model.loop.reinject and not model.loop.shared_cache
    assert model.per_layer_types == ('sliding_attention', 'mlp', 'full_attention', 'mlp') * 2
    assert isinstance(model.kind_of('mlp').mixer, MLPMixer)


DORY_FIELDS = ('vocab_size', 'hidden_size', 'num_hidden_layers', 'intermediate_size', 'num_attention_heads',
               'num_key_value_heads', 'head_dim', 'max_position_embeddings', 'tie_word_embeddings',
               'hidden_act', 'initializer_range', 'sliding_window', 'ngpt_alpha_init',
               'attention_output_gate_logit_scale_init', 'mlp_bias', 'attention_bias', 'n_input_layers',
               'n_recurrent_layers', 'n_output_layers', 'n_recurrent_loops', 'layer_types', 'rope_parameters')
"""Every field DoryConfig declares (configs/dory.py:14-38)."""


def test_the_released_config_round_trips_through_dews_fields():
    """NVIDIA-Nemotron-Labs-Dory1.0-4B's config: 16 input, 40 looped and 16 output layers, the
    looped ones run twice.
    It translates into Dew's fields, the model built from them writes back every field
    DoryConfig reads as the release states it (a null by leaving it out, which DoryConfig reads as
    its default null), and that config translates to the same fields."""
    released = json.loads((RELEASED / 'config.json').read_text())
    fields = hf_decoders.translate_config(released)
    model = models.build('causal_transformer', fields)
    assert model.num_layers == 72 and model.loop is not None and model.loop.layers == (16, 56)
    assert model.loop.steps == 2 and model.ngpt is not None
    written = hf_decoders.export_config(model)
    for field in DORY_FIELDS:
        assert written.get(field) == released[field], field
    assert hf_decoders.translate_config(written) == fields


@pytest.mark.parametrize('change', [
    {'tie_word_embeddings': True}, {'mlp_bias': True}, {'n_input_layers': 1},
    {'recurrent_kv_cache_mode': 'each'}, {'initializer_range': 0.0},
    {'n_input_layers': 4, 'n_recurrent_layers': 0}, {'n_recurrent_loops': 0},
    {'layer_types': ['sliding_attention', 'mlp'] * 3 + ['linear_attention', 'mlp']},
    {'rope_parameters': {'sliding_attention': {'rope_type': 'default', 'rope_theta': 10000}}},
    {'rope_parameters': {kind: {'rope_type': 'linear', 'rope_theta': 10000, 'factor': 2.0}
                         for kind in ('sliding_attention', 'full_attention')}}])
def test_a_config_dory_refuses_is_refused(tmp_path, change):
    """What DoryConfig refuses (configs/dory.py:39-133) is refused: a partition that does not
    cover the layers, a layer kind it does not run, no loop, a tied head, biases, an unknown cache
    mode, a zero initializer range and a rope it does not compute."""
    with pytest.raises(ValueError):
        Pretrained.load(changed(tmp_path / 'changed', change), dtype='float32', attention_impl='xla')


@pytest.mark.parametrize('variant', sorted(TOOL.VARIANTS))
def test_a_config_the_release_does_not_take_computes_vllms_logits(tmp_path, variant):
    """Configs DoryConfig accepts over the tiny tensors that the release does not take: a partial
    rotary per kind, which vLLM's get_rope reads as Dew's 'default' partial rotary, an nGPT step
    or gate logit scale of zero, whose gains start without dividing by it, a loop whose block
    starts at an MLP and ends at an attention, and a loop over no layer."""
    loaded = Pretrained.load(changed(tmp_path / variant, TOOL.VARIANTS[variant]), dtype='float32',
                             attention_impl='xla')
    logits = loaded.model.apply(loaded.variables, REFERENCE['input_ids'])
    assert_as_exact_as_the_reference(np.asarray(logits), REFERENCE[f'logits_{variant}'],
                                     REFERENCE[f'logits_{variant}_f64'], f'Dory logits, {variant}')


def test_a_dory_written_from_dew_alone_reloads_as_itself(tmp_path):
    """A Dory written from the model and its variables alone (`PretrainedDecoder.from_model`), not
    through the source's bindings, states DoryConfig's own fields, one rope record per attention
    kind and its partial rotary included, under the released tensor names, and reloads to the same
    logits. A model Dory does not compute, nGPT with no reinjecting loop, is refused."""
    loaded = Pretrained.load(changed(tmp_path / 'source', TOOL.VARIANTS['partial_rotary']), dtype='float32',
                             attention_impl='xla')
    PretrainedDecoder.from_model(loaded.model, loaded.variables).save(tmp_path / 'written')
    config = json.loads((tmp_path / 'written' / 'config.json').read_text())
    assert config['rope_parameters'] == TOOL.VARIANTS['partial_rotary']['rope_parameters']
    assert 'rope_theta' not in config and 'rope_local_base_freq' not in config
    assert sorted(load_shards(tmp_path / 'written')) == sorted(load_shards(TINY))
    again = Pretrained.load(tmp_path / 'written', dtype='float32', attention_impl='xla')
    np.testing.assert_array_equal(again.model.apply(again.variables, REFERENCE['input_ids']),
                                  loaded.model.apply(loaded.variables, REFERENCE['input_ids']))
    with pytest.raises(ValueError):
        PretrainedDecoder.from_model(loaded.model.clone(loop=None), loaded.variables)


def test_an_mlp_mixer_takes_ngpts_gains_and_what_ngpt_does_not_reach_is_refused():
    """nGPT's arithmetic lives in the attention and the dense gated feed-forward. An MLP in the
    mixer's slot takes the feed-forward's gains; routed experts, a gated delta net, a prediction
    depth and an MLP mixer's own experts, which would compute without them, are refused."""
    model = load().model
    ids = jnp.asarray(REFERENCE['input_ids'])
    mixed = model.clone(mixer=MLPMixer(intermediate_size=48, activation='swiglu'))
    tree = mixed.init(jax.random.key(0), ids)['params']
    assert set(tree['layers_0']['self_attn']) >= {'suv_gate', 'suv_up'}
    delta = GatedDeltaNetMixer(linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8,
                               linear_value_head_dim=8)
    routed = MLPMixer(intermediate_size=48, activation='swiglu', mixture=Mixture(experts=4))
    for refused in ({'mixture': Mixture(experts=4)}, {'num_nextn_predict_layers': 1}, {'mixer': delta},
                    {'mixer': routed}):
        with pytest.raises(ValueError):
            model.clone(**refused).init(jax.random.key(0), ids)


def test_the_releases_tokenizer_loads_with_the_checkpoint():
    """The tiny Dory ships a tokenizer as the release does, under a model type transformers does
    not register, whose 'mlp' layer types its untyped config would refuse; it loads with the
    checkpoint and encodes as its own tokenizer.json does."""
    from tokenizers import Tokenizer

    loaded = load()
    assert loaded.processor is not None
    text = 'charged twice, site 500!'
    assert loaded.processor.reference(text)['input_ids'] == Tokenizer.from_file(
        str(TINY / 'tokenizer.json')).encode(text).ids


def test_a_saved_checkpoint_reloads_bit_for_bit(tmp_path):
    """A loaded Dory saves its config and tensors as the released checkpoint names them, and the
    saved checkpoint rebuilds the same model and tensors."""
    loaded = Pretrained.load(TINY, dtype='float32', attention_impl='reference')
    loaded.save(tmp_path)
    assert_same_stored_tensors(load_shards(tmp_path), load_shards(TINY))
    assert json.loads((tmp_path / 'config.json').read_text())['model_type'] == 'dory'
    assert_reloads(loaded, tmp_path, REFERENCE['input_ids'])


MARGIN = 1e-3
"""The float64 top-two margin above which a position's argmax must be the reference's: far above
fp32's rounding of logits near 10, and below it a tie that any rounding may break."""


@pytest.mark.network
@pytest.mark.parametrize('dtype', ['float32', 'bfloat16'])
def test_the_release_computes_the_references_logits_and_continuations(dtype):
    """NVIDIA-Nemotron-Labs-Dory1.0-4B at source.json's revision, against tools/dory_reference.py's
    probe of the same weights on a GPU: the logits of a sentence, and of the last 64 of 1100 random
    ids, past the 1024-token window, on the probe's 256 columns, within the rule of the reference's
    own run at `dtype` against float64; each position's argmax the float64 one's wherever its
    top-two margin exceeds `MARGIN`; and, in fp32, each cache's greedy continuation of the
    sentence the float64 reference's."""
    source = json.loads((RELEASED / 'source.json').read_text())
    probe = np.load(RELEASED / 'probe.npz')
    loaded = Pretrained.load(source['repo'], revision=source['revision'], dtype=dtype)
    suffix = '' if dtype == 'float32' else '_bf16'
    for name, kept in (('sentence', slice(None)), ('window', slice(-64, None))):
        logits = np.asarray(loaded.model.apply(loaded.variables, probe[name][None]), np.float32)[0, kept]
        clear = probe[f'{name}_margin'] > MARGIN
        np.testing.assert_array_equal(logits.argmax(-1)[clear], probe[f'{name}_argmax'][clear])
        assert_as_exact_as_the_reference(logits[:, probe['columns']], probe[f'{name}_logits{suffix}'],
                                         probe[f'{name}_logits_f64'], f'Dory 4B {dtype} {name}')
    if dtype != 'float32':
        return
    prompt = probe['sentence'].tolist()
    for mode in ('per_loop', 'last_loop'):
        model = with_cache(loaded.model, mode).with_cache_capacity(64)
        task = TextGeneration(model, loaded.variables, sampling=Sampling(temperature=0.0))
        generated = task([prompt], 16, key=0).host()
        continuation = generated.tokens[0, generated.prompt_width:]
        np.testing.assert_array_equal(continuation, probe[f'generated_{mode}'])
