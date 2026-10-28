"""A backbone with a head over its final states (`dew.nn.wrapping.HeadedModel`): a class read as its
label's tokens' logits at an answer state (`HeadedModel.label_logits`, a verbalizer) is the backbone's
full head there, on par from float64 (check 1), for labels of several token variants each; a head over
a backbone that gives no final states is refused."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from reference_error import assert_as_exact

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.wrapping import HeadedModel
from dew.objectives.base import joined

LABELS = ((ord("y"), ord("Y")), (ord("n"), ord("N")))
"""Two classes, each spelled by two tokens of a byte vocabulary."""
LABEL_SEEDS = 8
"""Check 1's declared seeds for a verbalizer's label logits."""


def backbone() -> CausalTransformer:
    return CausalTransformer(vocab_size=256, emb_features=32, num_layers=2, num_heads=2, mlp_features=48,
                             max_seq_len=32, attention_impl="xla", final_logit_softcap=3.0)


def test_a_verbalizers_label_logits_are_the_backbones_full_head_at_its_labels_tokens():
    """Each row's last state read as its labels' tokens' logits is the backbone's softcapped head
    there, the labels' columns taken: `[B, 1, 2, 2]` for two classes of two tokens each."""
    model = HeadedModel(backbone(), nn.Dense(2))
    tokens = jnp.zeros((3, 12), jnp.int32)
    initial = model.backbone.init(jax.random.key(0), tokens)

    def draw(seed):
        rng = np.random.default_rng([0x5EB, seed])
        moved = jax.tree.map(lambda leaf: leaf * (1 + 0.05 * rng.standard_normal(leaf.shape)), initial)
        return moved, jnp.asarray(rng.integers(0, 256, (3, 12)), jnp.int32)

    def read(variables, rows):
        states = model.backbone.apply(variables, rows, method="hidden_states")
        return model.label_logits(joined({"backbone": variables}), states[:, -1:], LABELS)

    def full(variables, rows):
        logits = model.backbone.apply(variables, rows)[:, -1:]
        return jnp.take(logits, jnp.asarray(LABELS), axis=-1)

    assert jax.eval_shape(read, *draw(0)).shape == (3, 1, 2, 2)
    assert_as_exact(read, full, draw, "verbalizer", seeds=LABEL_SEEDS)


def test_a_head_over_a_backbone_without_final_states_is_refused():
    with pytest.raises(TypeError):
        HeadedModel(nn.Dense(4), nn.Dense(2))
