"""Dory, NVIDIA's looped nGPT, against vLLM's (vllm-project/vllm#60149 at 55076950):
tools/dory_reference.py transcribes its forward and writes tests/fixtures/hf/dory-tiny, the CPU
tests' reference, and tests/fixtures/hf/dory-1.0-4b holds the released config and what vLLM's
own engine computes of the release on a GPU (tools/dory_vllm_reference.py)."""

import dataclasses
import itertools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from decision_support import decision_rows
from interop_support import assert_reloads, assert_same_stored_tensors
from reference_error import (
    as_given,
    assert_as_exact,
    assert_as_exact_as_the_reference,
    assert_computes_the_oracle,
    assert_rounds_where_the_reference_does,
    chain_roundings,
    float64_forward,
    indexed_recording,
    recording,
    widened,
)
from tool_support import load as load_tool

from dew.checkpoints import Checkpoints
from dew.data.dataset import Loader, replayed
from dew.decision import Choice, DecisionObjective, Example, JointLayout, JointSchemaHead
from dew.inference import TextGeneration
from dew.inference.serving import Server
from dew.interop import Pretrained, PretrainedDecoder, hf_decoders
from dew.interop.sources import load_shards
from dew.lora import LoRA
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import Mixture
from dew.nn.inputs import ModelInputs
from dew.nn.kernels import generation
from dew.nn.kv_cache import KVCache
from dew.nn.mixers.gated_delta_net import GatedDeltaNetMixer
from dew.nn.mixers.mlp import MLPMixer
from dew.nn.ngpt import stepped
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


def assert_parity(logits, name: str, label: str) -> None:
    """Logits parity against the reference's array `name`, by tests/reference_error.py's rule against
    the reference's float64 array beside it."""
    assert_as_exact_as_the_reference(logits, REFERENCE[name], REFERENCE[f'{name}_f64'], label)


@pytest.mark.parametrize('loops', [1, 2, 4])
def test_dorys_float64_forward_computes_the_transcriptions_function(loops):
    """Dew's forward in float64, every nGPT gain, norm and step as wide as its inputs, sits within
    float64 rounding of the transcription's float64 logits at each loop count: Dew's own float64
    run, the truth a float64 comparison of Dory reads, computes vLLM's function."""
    loaded = load()
    model, ids = loaded.model.with_passes(loops), REFERENCE['input_ids']
    truth, _ = float64_forward(model, loaded.variables, ids)
    with jax.enable_x64(new_val=True):
        wide = model.clone(dtype=jnp.float64)
        roundings = chain_roundings(jax.make_jaxpr(wide.apply)(widened(loaded.variables), ids))
    assert_computes_the_oracle(np.asarray(truth), REFERENCE[f'logits_{loops}_f64'], f'{loops} loops',
                               roundings=roundings)


def test_the_release_tests_symmetries_keep_dorys_function_and_redraw_its_sums():
    """The orders the release is held to vLLM under (tools/dory_vllm_reference.py's `symmetry`: the
    residual stream's units, each feed-forward's hidden units, each attention's heads and value
    dimensions) on the tiny Dory: each taken back gives the weights' own bits; under each, Dew's float64
    logits sit within float64 rounding of the identity's, so each is the same function; and the fp32
    logits move, so each redraws the rounding."""
    loaded, ids = load(), REFERENCE['input_ids']
    config = json.loads((TINY / 'config.json').read_text())
    recorder = load_tool('dory_vllm_reference')
    exact, _ = float64_forward(loaded.model, loaded.variables, ids)
    plain = np.asarray(loaded.model.apply(loaded.variables, ids))
    with jax.enable_x64(new_val=True):
        wide = loaded.model.clone(dtype=jnp.float64)
        roundings = chain_roundings(jax.make_jaxpr(wide.apply)(widened(loaded.variables), ids))
    for index in range(3):
        drawn = recorder.symmetry(config, 1, index)
        ordered = symmetric(loaded.variables, config, drawn)
        jax.tree.map(np.testing.assert_array_equal,
                     symmetric(ordered, config, drawn, inverse=True), loaded.variables)
        found, _ = float64_forward(loaded.model, ordered, ids)
        assert_computes_the_oracle(np.asarray(found), np.asarray(exact), f'order {index} in float64',
                                   roundings=roundings)
        assert not np.array_equal(np.asarray(loaded.model.apply(ordered, ids)), plain), index


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
    A per-pass cache is exact, the full forward's: the uncached logits of the whole continuation
    hold to the same rule at the positions each step decoded, and the cached steps to the rule of
    those uncached logits. A last-loop cache is the model's own approximation, held to the
    reference's decoding under it."""
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
        assert_as_exact_as_the_reference(steps, whole, truth, "Dory's per-pass cache against its forward")


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


@pytest.mark.parametrize('change', [
    {'n_recurrent_loops': 1}, {'n_input_layers': 1, 'n_recurrent_layers': 1, 'n_output_layers': 6}],
    ids=['one-pass', 'mlp-block'])
def test_a_shared_cache_no_pass_rewrites_serves_prompts_in_pieces(tmp_path, change):
    """A last-loop cache that no later pass rewrites, under one pass or over a block of MLPs alone,
    which keep no cache, takes its prompts in pieces of two and reuses a cached prefix: each prompt
    served continues as the model alone continues it."""
    shared = changed(tmp_path / 'changed', {**change, 'recurrent_kv_cache_mode': 'last_loop'})
    loaded = Pretrained.load(shared, dtype='float32', attention_impl='xla')
    task = TextGeneration(loaded.model, loaded.variables, sampling=Sampling(temperature=0.0))
    server = Server.from_task(task, slots=2, capacity=32, kv_cache=KVCache(page_size=4, pages=40), chunk=2,
                              prefix_cache=True)
    prompts = [*ragged_prompts(), ragged_prompts()[1]]
    tickets = [server.submit(prompt, 6, key=index) for index, prompt in enumerate(prompts)]
    server.run()
    for ticket, prompt in zip(tickets, prompts, strict=True):
        served, alone = ticket.result().host(), task([prompt], 6, key=0).host()
        np.testing.assert_array_equal(served.tokens[0, served.prompt_width:],
                                      alone.tokens[0, alone.prompt_width:])
    assert server.prefix_hits > 0


@pytest.mark.parametrize('options', [
    {'chunk': 2}, {'kv_cache': KVCache(page_size=4, pages=40), 'prefix_cache': True}],
    ids=['chunked', 'prefix'])
def test_a_shared_loop_cache_takes_each_prompt_whole(options):
    """A last-loop cache holds the last pass's keys of every token before a call, so a prompt's
    second piece, or a prompt after a reused prefix, would read those where the whole prompt's
    passes read their own: a server refuses chunked prefill and prefix reuse over it, which the
    per-pass cache serves (`test_a_server_draws_each_prompt_as_the_reference_continues_it`)."""
    with pytest.raises(ValueError):
        Server.from_task(greedy('last_loop'), slots=2, capacity=32, **options)


DORY_GRADIENT_SEEDS = {'trained': 32, 'truncated': 64}
"""Declared seeds for check 1, sized by `seeds_for` on armada: the live torch reference's gradient of
every checkpoint tensor, through every pass or through the last pass alone."""


@pytest.mark.parametrize(('name', 'loops', 'backprop_steps'), [('trained', 2, None), ('truncated', 3, 1)])
def test_the_gradient_of_each_tensor_is_the_references(name, loops, backprop_steps):
    """The gradient of a loss over the logits in every tensor the checkpoint holds, its projections,
    nGPT gains and steps, the embeddings and the logit gain, as the reference's autograd takes it,
    within the rule: through every pass, and with the loop cut before its last pass, where the
    input layers still train through what that pass reinjects (Huginn's truncation). Each draw has
    fresh token ids and cotangents, with their batch rows permuted together, an exact symmetry of
    the summed loss. The same torch reference that wrote the fixture runs live in fp32 and float64;
    every gradient keeps its checkpoint name."""
    loaded = load()
    model = loaded.model.with_passes(loops, backprop_steps=backprop_steps)
    tensors = {key: np.array(value, copy=True) for key, value in loaded.export().items()}
    cut = None if backprop_steps is None else loops - backprop_steps

    def shapes(dtype):
        return {tensor: jax.ShapeDtypeStruct(value.shape, dtype) for tensor, value in tensors.items()}

    def exported(gradients):
        written = loaded.export({'params': gradients})
        return {tensor: np.asarray(written[tensor]) for tensor in tensors}

    def dew(ids, cotangent):
        def loss(params):
            return jnp.sum(model.apply({**loaded.variables, 'params': params}, ids) * cotangent)

        gradients = jax.grad(loss)(loaded.variables['params'])
        return jax.pure_callback(exported, shapes(cotangent.dtype), gradients)

    def torch_gradients(ids, cotangent):
        dtype = TOOL.torch.float64 if cotangent.dtype == np.float64 else TOOL.torch.float32
        reference = TOOL.DoryReference(loaded.config, tensors, dtype, loops=loops)
        return reference.gradients(np.array(ids), np.array(cotangent), tuple(tensors), cut=cut)

    def reference(ids, cotangent):
        return jax.pure_callback(torch_gradients, shapes(cotangent.dtype), ids, cotangent)

    def draw(seed):
        rng = np.random.default_rng(5 + seed)
        ids = rng.integers(12, loaded.config['vocab_size'], REFERENCE['input_ids'].shape, dtype=np.int32)
        cotangent = rng.normal(size=REFERENCE['cotangent'].shape).astype(np.float32)
        order = rng.permutation(len(ids))
        return ids[order], cotangent[order]

    dew, reference = jax.jit(dew), jax.jit(reference)
    label = f'Dory {name} gradients'
    assert_as_exact(dew, reference, draw, label, seeds=DORY_GRADIENT_SEEDS[name])


def test_a_step_computes_in_the_widest_of_its_operands():
    """nGPT's residual step runs in at least fp32 of its residual, its branch and its rate: a float32
    residual stepped toward a float64 branch by a float64 rate, or by a float64 rate alone, takes its
    rate and its difference in float64 and rounds once to the residual's float32."""
    with jax.enable_x64(new_val=True):
        residual = jnp.ones((2, 4), jnp.float32)
        for branch, alpha in ((jnp.ones((2, 4), jnp.float64), jnp.ones(4, jnp.float64)),
                              (jnp.ones((2, 4), jnp.float32), jnp.ones(4, jnp.float64))):
            traced = jax.make_jaxpr(lambda residual, branch, alpha: stepped(residual, branch, alpha, 0.5))(
                residual, branch, alpha)
            stepping = {equation.outvars[0].aval.dtype for equation in traced.jaxpr.eqns
                        if equation.primitive.name in ("abs", "sub")}
            assert stepping == {np.dtype(np.float64)}
            assert traced.out_avals[0].dtype == np.float32


def test_a_float64_model_computes_in_float64():
    """A float64 Dory computes its logits and gradients in float64 end to end, its nGPT attention
    included: each within float64's rounding of the torch reference's own float64 run, an
    independent implementation of the same tensors (`assert_computes_the_oracle`), where one fp32
    rounding point on the path would sit many orders further."""
    loaded = load()
    tensors = {key: np.array(value, copy=True) for key, value in loaded.export().items()}
    with jax.enable_x64(new_val=True):
        model = loaded.model.clone(dtype=jnp.float64)
        variables = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64), loaded.variables)
        ids, cotangent = REFERENCE['input_ids'], REFERENCE['cotangent'].astype(np.float64)

        def loss(params):
            return jnp.sum(model.apply({**variables, 'params': params}, ids) * cotangent)

        logits = np.asarray(model.apply(variables, ids))
        written = loaded.export({'params': jax.grad(loss)(variables['params'])})
        forward = chain_roundings(jax.make_jaxpr(lambda variables: model.apply(variables, ids))(variables))
        backward = chain_roundings(jax.make_jaxpr(jax.grad(loss))(variables['params']))
    reference = TOOL.DoryReference(loaded.config, tensors, TOOL.torch.float64)
    expected = reference.forward(ids)
    gradients = reference.gradients(np.array(ids), np.array(cotangent), tuple(tensors))
    assert logits.dtype == np.float64
    assert_computes_the_oracle(logits, expected, 'float64 logits', roundings=forward)
    for tensor in tensors:
        assert_computes_the_oracle(np.asarray(written[tensor]), gradients[tensor], f'float64 {tensor}',
                                   roundings=backward)


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


def peft_merged(checkpoint: Path, adapter: Path) -> dict[str, np.ndarray]:
    """The checkpoint's tensors with the adapter at `adapter` merged in by PEFT itself: its LoRA
    layers injected by the adapter's own config into a torch module of the checkpoint's linear
    projections under their checkpoint names, the saved factors loaded as PEFT loads them, and each
    layer's merge."""
    import torch
    from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
    from peft.tuners.lora import LoraLayer
    from safetensors.numpy import load_file

    tensors = load_file(str(checkpoint / 'model.safetensors'))
    base = torch.nn.Module()
    for name, value in tensors.items():
        *path, leaf, kind = name.split('.')
        if kind != 'weight' or value.ndim != 2 or leaf == 'embeddings':
            continue
        parent = base
        for part in path:
            if not hasattr(parent, part):
                parent.add_module(part, torch.nn.Module())
            parent = getattr(parent, part)
        linear = torch.nn.Linear(value.shape[1], value.shape[0], bias=False)
        linear.weight.data = torch.as_tensor(value)
        parent.add_module(leaf, linear)
    model = inject_adapter_in_model(LoraConfig.from_pretrained(str(adapter)), base)
    saved = load_file(str(adapter / 'adapter_model.safetensors'))
    set_peft_model_state_dict(model, {name.removeprefix('base_model.model.'): torch.as_tensor(value)
                                      for name, value in saved.items()})
    merged = dict(tensors)
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer) and module.lora_A:
            module.merge()
            merged[f'{name}.weight'] = module.base_layer.weight.detach().numpy()
    targets = {name.removeprefix('base_model.model.').split('.lora_')[0] + '.weight' for name in saved}
    assert {name for name in merged if merged[name] is not tensors[name]} == targets
    return merged


MERGED_SEEDS = 8
"""The merged-adapter site's declared draws (`seeds_for`)."""


def test_a_lora_on_every_projection_trains_resumes_saves_and_merges(tmp_path):
    """A rank-2 adapter on all eight projections of every layer, the attention's output gate
    (`g_proj`) apart from the feed-forward's `gate_proj`: one step moves every B factor, so the
    loss reaches each projection; a run resumed after its first step ends where two uninterrupted
    steps end; the saved factors load back to the same logits; PEFT, the saved adapter's consumer,
    merges it into every projection the checkpoint holds by its name there, and the transcription
    over that merge computes the adapted model's logits within the rule of its own run against
    float64; and the merged checkpoint reloads to logits on par with the adapted model's from
    float64."""
    from safetensors.numpy import load_file

    loaded = load()
    adapted = loaded.adapt(LoRA(rank=2, modules=PROJECTIONS), key=0)
    original = load_file(str(TINY / 'model.safetensors'))
    # Every projection the checkpoint holds, by its name there, is one PEFT's merge moves.
    projections = {name for name in original
                   if name.removesuffix('.weight').rsplit('.', 1)[-1] in PROJECTIONS}

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
    again = LoRA.load(loaded.model, loaded.variables, tmp_path / 'adapter', layouts=loaded.layouts)
    ids = REFERENCE['input_ids']
    logits = np.asarray(adapted.model.apply(resumed, ids))
    np.testing.assert_array_equal(np.asarray(again.model.apply(again.variables, ids)), logits)
    import torch

    config, peft = json.loads((TINY / 'config.json').read_text()), peft_merged(TINY, tmp_path / 'adapter')
    assert {name for name in peft if not np.array_equal(peft[name], original[name])} == projections
    assert_as_exact_as_the_reference(logits, TOOL.DoryReference(config, peft, torch.float32).forward(ids),
                                     TOOL.DoryReference(config, peft, torch.float64).forward(ids),
                                     'the adapter as PEFT merges it')
    adapted.save(tmp_path / 'merged', variables=resumed)
    merged = Pretrained.load(tmp_path / 'merged', dtype='float32', attention_impl='xla')
    # Merging folds each adapter into its projection's weight, a rounding of its own: the merged
    # model's logits are on par with the adapted model's over drawn ids, both from the adapted
    # model's float64 forward (check 1).
    vocabulary = adapted.model.vocab_size

    def draw(seed):
        return (np.random.default_rng([0xD0E, seed]).integers(0, vocabulary, ids.shape, dtype=np.int32),)

    def merged_logits(drawn):
        return merged.model.apply(merged.variables, drawn)

    def adapted_logits(drawn):
        return adapted.model.apply(resumed, drawn)

    assert_as_exact(merged_logits, adapted_logits, draw, 'merged adapter logits', seeds=MERGED_SEEDS,
                    truth_fn=as_given(lambda drawn: float64_forward(adapted.model, resumed, drawn)[0]))


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
    loaded = load(max_seq_len=1024)
    ids = REFERENCE['input_ids']
    table = loaded.model.apply(loaded.variables, method='output_table')
    hidden = loaded.model.apply(loaded.variables, ids, method='hidden_states')
    assert_as_exact_as_the_reference(np.asarray(hidden @ table.rows().T), REFERENCE['logits_2'],
                                     REFERENCE['logits_2_f64'], "Dory's vocabulary rows")
    head = JointSchemaHead(32, width=16, routing_layers=1, layers=1, heads=2, feedforward=32)
    objective = DecisionObjective(loaded, layout=JointLayout(max_len=1024), head=head)
    data = decision_rows(EXAMPLES, 8, EXAMPLES, objective=objective.data())
    trainer = Trainer(objective, optax.adam(1e-3), key=0, checkpoints=Checkpoints(str(tmp_path)))
    initial = trainer.initial_state()
    # The step donates the state it is handed; the host keeps what it started from.
    started = jax.device_get(initial.variables)
    state = trainer.fit(data, steps=1, log_every=1, state=initial)
    Checkpoints(str(tmp_path)).wait()
    for part in ('head', 'backbone'):
        before, after = started['params'][part], state.variables['params'][part]
        moved = jax.tree.map(lambda old, new: not np.array_equal(old, new), before, after)
        assert any(jax.tree.leaves(moved)), part
    asked = {'team': INTENT}
    live = objective.pipeline(state)('charged twice', asked)
    reloaded = objective.saved_task.from_run(str(tmp_path))('charged twice', asked)
    for answer, again in zip(live.values(), reloaded.values(), strict=True):
        np.testing.assert_array_equal(answer.probabilities, again.probabilities)


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
    {'rope_parameters': {kind: {'rope_type': 'default', 'rope_theta': 10000}
                         for kind in ('sliding_attention', 'full_attention', 'mlp')}},
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


def test_what_ngpts_arithmetic_does_not_reach_is_refused():
    """nGPT's arithmetic lives in the attention and the dense gated feed-forward, in a block's
    feed-forward or in the mixer's slot, where each of Dory's MLP layers computes it (the reference
    tests above). Routed experts, a gated delta net, a prediction depth and an MLP mixer's own
    experts, which would compute without it, are refused."""
    model = load().model
    ids = jnp.asarray(REFERENCE['input_ids'])
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


# Each parameter a symmetry reorders, by its name in an attention or a feed-forward layer: the map
# (tools/dory_vllm_reference.py's `maps`) and the axis it reorders.
SYMMETRIC = {'attention': {'q_proj': ('query', 1), 'k_proj': ('key', 1), 'v_proj': ('value', 1),
                           'gate_proj': ('output', 1), 's_gate': ('output', 0), 'sqk': ('query', 0),
                           'o_proj': ('output', 0)},
             'mlp': {'gate_proj': ('mlp', 1), 'up_proj': ('mlp', 1), 'suv_gate': ('mlp', 0),
                     'suv_up': ('mlp', 0), 'down_proj': ('mlp', 0)}}


def symmetric(variables, config: dict, drawn: dict[str, np.ndarray], *, inverse: bool = False):
    """`variables` under the symmetries `drawn` (tools/dory_vllm_reference.py's `symmetry`), or taken
    back from them where `inverse`: each attention's and feed-forward's parameters along the axes it
    reorders (`SYMMETRIC`), then the residual stream's units (tests/residual_orders.py), each leaf
    where it is, a device array on its device."""
    from residual_orders import permuted

    found = load_tool('dory_vllm_reference').maps(config, drawn)
    if inverse:
        found = {part: {layer: np.argsort(order) for layer, order in by.items()}
                 for part, by in found.items()}
    kinds = config['layer_types']

    def reorder(path, leaf):
        names = [getattr(entry, 'key', None) for entry in path]
        if not names[0].startswith('layers_'):
            return leaf
        layer = int(names[0].removeprefix('layers_'))
        parameter = names[-2] if names[-1] == 'kernel' else names[-1]
        named = SYMMETRIC['mlp' if kinds[layer] == 'mlp' else 'attention'].get(parameter)
        if named is None:
            return leaf
        part, axis = named
        order = found[part][layer]
        if isinstance(leaf, jax.Array):
            return jnp.take(leaf, jnp.asarray(order), axis=axis)
        return np.take(np.asarray(leaf), order, axis=axis)

    params = jax.tree_util.tree_map_with_path(reorder, variables['params'])
    residual = np.argsort(drawn['residual']) if inverse else drawn['residual']
    return permuted({**variables, 'params': params}, residual)


def decoding(model: CausalTransformer):
    """`model`'s empty cache of one row; its call over a cache, a prompt or a token; its prefill of a
    prompt scored at the prompt's last position alone, the state gathered before the vocabulary
    (`states_and_logits_at`); and its log-probabilities from logits, each compiled once for every set of
    weights of its shapes and binding no weights past its call."""
    opened = jax.jit(lambda variables: model.apply(variables, 1, method=CausalTransformer.init_cache,
                                                   mutable=['cache'])[1]['cache'])
    step = jax.jit(lambda variables, cache, ids: model.apply({**variables, 'cache': cache}, ids, decode=True,
                                                             mutable=['cache']))
    last = jax.jit(lambda variables, cache, ids: model.apply(
        {**variables, 'cache': cache}, ids, jnp.asarray([ids.shape[1] - 1]), decode=True, mutable=['cache'],
        method=CausalTransformer.states_and_logits_at))
    scored = jax.jit(lambda logits: jax.nn.log_softmax(logits.astype(jnp.promote_types(logits.dtype,
                                                                                          jnp.float32))))
    return opened, step, last, scored


def forced(programs, variables, prompt, tokens, *, prompt_rows: bool) -> jax.Array:
    """Dew's log-probabilities at each position of `prompt` followed by `tokens`, the prompt
    prefilled and each token but the last decoded a step at a time through the cache (`programs`,
    `decoding`'s), as a run that drew `tokens` computes them: the prompt's rows (its last alone, the
    vocabulary over that state alone, where `prompt_rows` is False), then a row a token."""
    opened, step, last, scored = programs
    ids = jnp.asarray(np.asarray(prompt, np.int32)[None])
    if prompt_rows:
        logits, written = step(variables, opened(variables), ids)
        rows = [logits[0]]
    else:
        (_, logits), written = last(variables, opened(variables), ids)
        rows = [logits]
    for token in tokens[:-1]:
        logits, written = step(variables, written['cache'], jnp.asarray([[token]], jnp.int32))
        rows.append(logits[0])
    return scored(jnp.concatenate(rows))


def assert_decodes_as_recorded(stored, site: str, model: CausalTransformer, weights, config: dict,
                               parts: dict, label: str) -> None:
    """Check 1 of `model` over `weights` (host arrays) decoding `parts` (each prompt, the tokens drawn
    after it and whether its own rows are scored) against the reference recorded at `site` as each row's
    distance from float64 under each of the release recorder's orders (`record_distances`): Dew's rows
    under a seed's order (`symmetric`), each order's weights placed alone, against Dew's own float64
    decoding of the same tokens, held to the recorded truth's sketch at every seed. The orders are exact
    symmetries of the function, so every seed's float64 truth is the identity's."""
    recorder = load_tool('dory_vllm_reference')
    programs = decoding(model)

    def rows(programs, variables):
        return {'log_probabilities': jnp.concatenate([
            forced(programs, variables, prompt, tokens, prompt_rows=prompt_rows)
            for prompt, tokens, prompt_rows in parts.values()])}

    def dew(index, order):
        ordered = jax.device_put(symmetric(weights, config, order))
        found = jax.tree.map(np.asarray, rows(programs, ordered))
        jax.tree.map(lambda leaf: leaf.delete(), ordered)
        return found

    # The float64 arm on the default device, a GPU's float64 units where one runs the test, its kernels
    # chosen as a CPU's, the portable math.
    with jax.enable_x64(new_val=True), generation.assumed('cpu'):
        wide = model.clone(dtype=jnp.float64)
        held = jax.device_put(widened(weights))
        wide_programs = decoding(wide)

    def truth(order):
        with jax.enable_x64(new_val=True), generation.assumed('cpu'):
            return rows(wide_programs, held)

    truth.dew_programs = True
    seeds = int(stored[f'{site}.seeds'])
    drawn, reference, recorded_truth = indexed_recording(
        stored, site, seeds, inputs_of=lambda seed: recorder.symmetry(config, *divmod(seed, recorder.ORDERS)),
        truth_fn=truth)
    assert_as_exact(as_given(dew), reference, drawn, label, seeds=seeds, truth_fn=recorded_truth)


def test_the_release_qualification_holds_dew_to_a_reference_kept_as_row_distances():
    """The release test's path on the tiny Dory, both caches: the transcription in fp32 under each of the
    release recorder's orders, kept as each row's distance from its float64 run alone, as the release's
    recording keeps vLLM's (`record_distances`), and Dew decoding the same tokens held to it by check 1,
    a sentence's rows and a prompt past the window scored at its last position alone."""
    import torch

    recorder, recording_of = load_tool('dory_vllm_reference'), load_tool('recorded_references')
    loaded = load()
    config = json.loads((TINY / 'config.json').read_text())
    ids = REFERENCE['input_ids'][0]
    for mode in ('per_loop', 'last_loop'):
        tensors = {key: np.array(value, copy=True) for key, value in loaded.export().items()}
        reference = TOOL.DoryReference(config, tensors, torch.float64, mode=mode)
        # The tiny Dory's window is 5: the second prompt runs past it.
        prompts = {'sentence': (ids[:4], True), 'window': (ids, False)}
        parts = {part: (prompt, reference.decode(prompt, 5)[0][len(prompt):], prompt_rows)
                 for part, (prompt, prompt_rows) in prompts.items()}

        def transcribed(tensors, dtype, parts=parts, mode=mode):
            ran = TOOL.DoryReference(config, tensors, dtype, mode=mode)
            logits = np.concatenate([ran.forced(prompt, tokens, prompt_rows)
                                     for prompt, tokens, prompt_rows in parts.values()])
            return torch.as_tensor(logits).log_softmax(-1).numpy()

        truth = transcribed(tensors, torch.float64)
        host = jax.device_get(loaded.variables)

        def order(seed):
            return recorder.symmetry(config, *divmod(seed, recorder.ORDERS))

        def distances(seed, transcribed=transcribed, truth=truth, host=host):
            tensors = loaded.export(symmetric(host, config, order(seed)))
            found = transcribed({key: np.array(value, copy=True) for key, value in tensors.items()},
                                torch.float32)
            return np.sqrt(np.mean(np.square(found - truth), axis=1))

        with jax.enable_x64(new_val=True):
            stored = recording_of.record_distances(
                mode, order, {'log_probabilities': np.stack([distances(seed) for seed in range(80)])},
                lambda seed, truth=truth: {'log_probabilities': truth}, {'log_probabilities': 'float32'})
        model = with_cache(loaded.model, mode).with_cache_capacity(32)
        assert_decodes_as_recorded(stored, mode, model, host, config, parts, f'tiny Dory, {mode}')


RELEASE_RUNS = load_tool('dory_vllm_reference').RUNS
"""The release recorder's runs: (loop count, cache mode, dtype)."""


@pytest.mark.network
@pytest.mark.gpu(reason='the 4B release, held to vLLM running it on a GPU')
@pytest.mark.parametrize('run', RELEASE_RUNS, ids=['-'.join(map(str, run)) for run in RELEASE_RUNS])
def test_the_release_computes_vllms_log_probabilities_through_each_cache_and_loop_count(run):
    """NVIDIA-Nemotron-Labs-Dory1.0-4B at source.json's revision against vLLM's own engine running it
    (tools/dory_vllm_reference.py; its recording tests/fixtures/hf/dory-1.0-4b/recording.npz, its
    provenance inside), in bf16 at one, two and four loops and in fp32 at two, through each cache: the
    log-probabilities of a sentence's prompt and of the 256 tokens vLLM drew greedily after it, each
    decoded a step at a time, and of 64 tokens drawn after 1100 random ids, past the 1024-token window,
    every entry of every row, by check 1 (`assert_decodes_as_recorded`).

    A seed is an order of the exact symmetries both engines hold the release under, on the same tokens
    (tools/dory_vllm_reference.py's `symmetry`: the residual stream's units, each feed-forward's hidden
    units, each attention's heads and value dimensions), which redraw 90 to 99.5% of the release's summed
    terms. The rest no order moves on either side, so neither error there is calibrated: the queries' and
    keys' products over a head's dimensions (RoPE's frequency pairs), the softmax's and the values' sums
    over the keys (the key order), the queries' and keys' unit norms, and the log-softmax over the
    vocabulary. The float64 truth is vLLM's own model code run in float64 on the same tokens, recorded as
    its sketch, which Dew's own float64 decoding is held to at every seed. On a last-loop run's rows
    past the window, vLLM's engine parts from its own model code
    (https://github.com/vllm-project/vllm/pull/60149#issuecomment-6270937659): there the reference is
    that model code at the engine's precision under the same orders.

    Each arm's precision: the engine computes a bf16 run in bf16 over fp32 accumulation, and an fp32 run
    with its linears at "highest" (VLLM_FLOAT32_MATMUL_PRECISION's default) and its Triton attention's
    products in TF32; the model-code arm takes the same. Dew leaves each product to JAX's default, which
    on this GPU multiplies a float32 run's linears and attention alike in TF32."""
    recorder = load_tool('dory_vllm_reference')
    loops, mode, dtype = run
    site = recorder.site(*run)
    stored = recording(RELEASED / 'recording.npz')
    source = json.loads((RELEASED / 'source.json').read_text())
    config = json.loads((RELEASED / 'config.json').read_text())
    parts = {part: (stored[f'{site}.prompt.{part}'], stored[f'{site}.continued.{part}'], prompt_rows)
             for part, _, prompt_rows in recorder.PARTS}
    loaded = Pretrained.load(source['repo'], revision=source['revision'], dtype=dtype)
    # The weights live on the host, each order's copy alone on the device: the 4B's weights twice over,
    # with its compiled step and cache, do not fit a 40 GB GPU.
    weights = jax.device_get(loaded.variables)
    for leaf in jax.tree.leaves(loaded.variables):
        if isinstance(leaf, jax.Array):
            leaf.delete()
    model = with_cache(loaded.model.with_passes(loops), mode).with_cache_capacity(recorder.CONTEXT)
    assert_decodes_as_recorded(stored, site, model, weights, config, parts, f'Dory 4B {site}')
