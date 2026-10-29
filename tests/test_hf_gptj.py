"""GPT-J's interleaved partial rotary stored in the shared rotate-half layout."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from interop_support import fixture_ids, next_token_losses, transformers_arms, transformers_model
from reference_error import assert_as_exact

from dew.interop import Pretrained

ROOT = Path(__file__).parent / 'fixtures' / 'hf'
DIRECTORY = ROOT / 'gptj-tiny'

LIKELIHOOD_SEEDS = 8
"""The affine head's declared draws of ids against transformers' cross entropy over its logits (check 1),
sized by `seeds_for` paired with Dew's token scores on the CPU."""


def test_gptj_affine_head_trains_through_the_public_objective():
    """The public objective's per-token losses through GPT-J's biased head are on par with the cross
    entropy of transformers' logits, from its float64 twin's, over drawn ids (check 1); the head's bias
    takes a gradient, and a step along it lowers the loss."""
    from dew.objectives.lm import LMObjective

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    draw = fixture_ids(DIRECTORY)
    ids = jnp.asarray(draw(0)[0])
    objective = LMObjective(loaded.model, seq_len=ids.shape[1] - 1, variables=loaded.variables,
                            head_tile=(3, 17), ema_decay=None)

    def loss(params, ids):
        scores = objective.token_scores({'params': params}, ids)
        return scores.losses.mean(), scores.losses

    reference, truth = transformers_arms(transformers_model(DIRECTORY), next_token_losses)
    assert_as_exact(lambda ids: loss(loaded.variables['params'], jnp.asarray(ids))[1], reference, draw,
                    'GPT-J affine-head likelihoods', seeds=LIKELIHOOD_SEEDS, truth_fn=truth)

    (value, _), gradient = jax.value_and_grad(loss, has_aux=True)(loaded.variables['params'], ids)
    assert np.linalg.norm(gradient['head_bias']) > 0
    changed = jax.tree.map(lambda parameter, grad: parameter - 1e-3 * grad,
                           loaded.variables['params'], gradient)
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(gradient))
    assert float(loss(changed, ids)[0]) < float(value)
