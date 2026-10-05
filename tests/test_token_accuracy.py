"""Token accuracy against TRL's own (`tools/token_accuracy_reference.py`):
`SFTTrainer.compute_loss`'s argmax-against-labels count, run as published
on the logits of a fixed tiny decoder and on logits with planted ties."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew import Mean
from dew.artifacts import TokenScores
from dew.data.chat import Role
from dew.nn.backbones import CausalTransformer
from dew.objectives.base import Step
from dew.objectives.lm import LMObjective
from dew.objectives.lm.chunked import chunked_cross_entropy

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "token_accuracy"
MODEL = {"vocab_size": 32, "emb_features": 16, "num_layers": 2, "num_heads": 2, "mlp_features": 32,
         "max_seq_len": 16, "attention_impl": "reference"}


@pytest.mark.parametrize("chunks", [1, 4])
def test_a_tie_goes_to_the_lowest_column_as_trls_argmax_takes_it(chunks):
    """Every row's top logit sits at two columns at least a quarter of the
    vocabulary apart, so in four tiles the two are in different tiles; the
    chunked head's prediction is the lower, as torch's argmax in TRL's
    count picks it, and the count over the labels TRL keeps is TRL's. The
    logits enter as states through an identity head, which reproduces them
    exactly."""
    with np.load(FIXTURES / "ties.npz") as data:
        reference = {key: data[key] for key in data}
    labels = reference["labels"][:, 1:]
    kept = labels != -100
    states = jnp.asarray(reference["logits"][:, :-1])
    _, predicted, _ = chunked_cross_entropy(states, jnp.eye(states.shape[-1], dtype=jnp.float32),
                                            jnp.asarray(np.where(kept, labels, 0), jnp.int32), chunks)
    assert int(np.sum((np.asarray(predicted) == labels) & kept)) == int(reference["correct"])
    assert int(kept.sum()) == int(reference["total"]) and 0 < int(reference["correct"]) < int(kept.sum())


def test_assistant_token_accuracy_is_trls_count_over_uneven_batches():
    """A two-layer decoder's assistant-only accuracy over two batches of
    three and five rows, against TRL's count on the same logits with every
    non-assistant target labelled -100. Each batch's right and counted
    targets, through `LMObjective.evaluate` and through the loss's own
    `token_accuracy`, are TRL's exactly: the model's logits are the
    fixture's to well under half the smallest top-two gap of any counted
    position, so no rounding reorders a prediction. `Mean` over the
    batches' `TokenScores` is the pass's weighted count, right over counted
    targets in float64. TRL's `log` instead reports the mean of its
    batches' accuracies, which weighs a small batch's targets more; the two
    agree only on batches of equal size."""
    reference = dict(np.load(FIXTURES / "decoder.npz"))
    model = CausalTransformer(**MODEL)
    params = model.init(jax.random.key(0), jnp.ones((1, 12), jnp.int32))
    objective = LMObjective(model, seq_len=11, ema_decay=None, loss_role=Role.ASSISTANT, token_accuracy=True)
    variables = objective.init(jax.random.key(0))
    assert jax.tree.structure(variables["params"]) == jax.tree.structure(params["params"])
    variables = {**variables, "params": params["params"]}
    metric = Mean(lambda scores, batch: (np.sum(scores.correct * scores.weights), np.sum(scores.weights)),
                  reads=TokenScores, name="accuracy", better="higher")
    step = Step(step=jnp.asarray(0), key=jax.random.key(1), ema=None)
    partials, right, counted = [], 0, 0
    for index in range(2):
        tokens, roles = reference[f"{index}/tokens"], reference[f"{index}/roles"]
        logits = np.asarray(model.apply(params, jnp.asarray(tokens)))
        assert np.abs(logits - reference[f"{index}/logits"]).max() < float(reference[f"{index}/gap"]) / 4
        batch = {"text": jnp.asarray(tokens), "text_roles": jnp.asarray(roles)}
        scores = objective.evaluate(variables, batch, step)
        assert int(np.sum(scores.correct * scores.weights)) == int(reference[f"{index}/correct"])
        assert int(np.sum(scores.weights)) == int(reference[f"{index}/total"])
        _, aux = objective.loss(variables, batch, step)
        np.testing.assert_allclose(float(aux.metrics["token_accuracy"]),
                                   int(reference[f"{index}/correct"]) / int(reference[f"{index}/total"]),
                                   rtol=1e-6)
        partials.append(metric(scores, batch))
        right += int(reference[f"{index}/correct"])
        counted += int(reference[f"{index}/total"])
    assert metric.finalize(metric.merge(*partials)) == right / counted
    assert right / counted != pytest.approx(float(reference["logged"]), abs=1e-4)
