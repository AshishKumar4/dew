"""Phi's parallel biased decoder and affine training head against transformers 5.16.1."""

import contextlib
import json
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from reference_error import as_given, assert_as_exact, assert_as_exact_as_the_reference

from dew.interop import Pretrained

ROOT = Path(__file__).parent / 'fixtures' / 'hf'
DIRECTORY = ROOT / 'phi-tiny'



def test_phi_affine_head_trains_through_the_public_objective():
    from dew.objectives.lm import LMObjective

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    ids = jnp.asarray(np.load(DIRECTORY / 'input_ids.npy'))
    objective = LMObjective(loaded.model, seq_len=ids.shape[1] - 1, variables=loaded.variables,
                            head_chunks=3, head_tile=(3, 17), ema_decay=None)

    def loss(params):
        scores = objective.token_scores({'params': params}, ids)
        return scores.losses.mean(), scores

    (value, scores), gradient = jax.value_and_grad(loss, has_aux=True)(loaded.variables['params'])
    assert_as_exact_as_the_reference(scores.losses, np.load(DIRECTORY / 'losses.npy'),
                                     np.load(DIRECTORY / 'losses_f64.npy'), 'Phi affine-head likelihoods')
    logits = loaded.model.apply(loaded.variables, ids[:, :-1])
    np.testing.assert_array_equal(scores.correct, (logits.argmax(-1) == ids[:, 1:]).astype(jnp.float32))
    assert np.linalg.norm(gradient['head_bias']) > 0
    changed = jax.tree.map(lambda parameter, grad: parameter - 1e-3 * grad,
                           loaded.variables['params'], gradient)
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(gradient))
    assert float(loss(changed)[0]) < float(value)


YARN = {"rope_type": "yarn", "rope_theta": 10000.0, "partial_rotary_factor": 0.5, "factor": 4.0,
        "original_max_position_embeddings": 12, "beta_fast": 32.0, "beta_slow": 1.0}
"""A YaRN entry over Phi's half rotary, which transformers ramps over `head_dim * partial_rotary_factor`
(`_compute_yarn_parameters`): its rows of 32 tokens reach well past the 12 the ramp stretches from."""

YARN_SEEDS = 64
"""The YaRN source's declared draws (check 1, `assert_as_exact`), sized by `seeds_for` on armada."""


def yarn_source(tmp_path) -> Path:
    """phi-tiny's checkpoint under `YARN`."""
    directory = shutil.copytree(DIRECTORY, tmp_path / 'phi-yarn')
    config = json.loads((DIRECTORY / 'config.json').read_text())
    (directory / 'config.json').write_text(json.dumps({**config, 'rope_parameters': YARN}))
    return directory


def test_a_yarn_phi_trains_as_transformers_computes_it_and_writes_it_back(tmp_path):
    """Phi rotates half its head, the 'default' partial rotary, and transformers' YaRN ramps that half
    alone, and PhiModel's embedding leaves its pad row untrained by its lookups (padding_idx). Dew's
    logits and every parameter's gradient of the next-token loss over drawn rows of 32 tokens, the pad
    id among them, are on par with transformers' float32 run by check 1, from its float64 run; the export
    writes the ramp and the fraction in one record, which transformers reads to the logits it
    computes for the source, and which loads back to the same model."""
    import torch
    from transformers import AutoModelForCausalLM

    from dew.interop import PretrainedDecoder
    from dew.nn.rope import YarnScaling
    from tools.torch_precision import float64

    source = yarn_source(tmp_path)
    loaded = Pretrained.load(source, dtype='float32', attention_impl='reference')
    assert isinstance(loaded.model.rope_scaling, YarnScaling) and loaded.model.partial_rotary_factor == 0.5

    def dew(ids):
        def loss(params):
            logits = loaded.model.apply({**loaded.variables, 'params': params}, ids)
            chosen = jnp.take_along_axis(jax.nn.log_softmax(logits[:, :-1]), ids[:, 1:, None], -1)
            return -jnp.mean(chosen), logits
        (_, logits), gradient = jax.value_and_grad(loss, has_aux=True)(loaded.variables['params'])
        return logits, gradient

    stepped = jax.jit(dew)

    @as_given
    def ours(ids):
        logits, gradient = stepped(jnp.asarray(ids, jnp.int32))
        return {'logits': logits, 'gradients': {
            layout.name: np.asarray(layout.export({'params': gradient})).reshape(layout.shape)
            for layout in loaded.weight_layouts}}

    @as_given
    def reference(ids, wide=False):
        with float64() if wide else contextlib.nullcontext():
            model = AutoModelForCausalLM.from_pretrained(source, dtype=torch.float32)
            model = (model.to(torch.float64) if wide else model).eval()
            model.set_attn_implementation('eager')
            tokens = torch.from_numpy(np.asarray(ids, np.int64))
            output = model(tokens, labels=tokens)
            output.loss.backward()
        return {'logits': output.logits.detach().numpy(),
                'gradients': {name: parameter.grad.numpy() for name, parameter in model.named_parameters()}}

    def draw(seed):
        return (np.random.default_rng([0xA2, seed]).integers(0, 64, (4, 32)).astype(np.int32),)

    assert_as_exact(ours, reference, draw, 'phi under a partial YaRN', seeds=YARN_SEEDS,
                    truth_fn=lambda ids: reference(ids, wide=True))

    export = tmp_path / 'export'
    PretrainedDecoder.from_model(loaded.model, loaded.variables).save(export)
    written = json.loads((export / 'config.json').read_text())['rope_parameters']
    assert {key: written[key] for key in YARN} == YARN
    assert Pretrained.load(export, dtype='float32', attention_impl='reference').model == loaded.model

    def transformers_logits(directory):
        model = AutoModelForCausalLM.from_pretrained(directory, dtype=torch.float32).eval()
        with torch.no_grad():
            return model(torch.from_numpy(draw(0)[0].astype(np.int64))).logits.numpy()

    np.testing.assert_array_equal(transformers_logits(export), transformers_logits(source))
