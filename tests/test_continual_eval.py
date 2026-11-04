"""Continual learning measured (dew.eval.continual): an accuracy matrix's average accuracy, forgetting and
backward transfer as their papers define them, a run's matrix assembled stage by stage, and a language
model's choice among labels by the log-likelihood its objective gives them, in context or not."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact

from dew.eval.continual import AccuracyMatrix, continual_run, label_accuracy, label_scores
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.continuum import ContinuumLevel
from dew.objectives.lm import LMObjective
from dew.objectives.lm.adaptation import ChunkAdaptation

VOCAB = 16

LABEL_SEEDS = 8
"""Check 1's declared seeds for the label scores' site."""


def test_an_accuracy_matrix_reads_average_forgetting_and_backward_transfer():
    """Three tasks: the average is the last stage's mean, a task's forgetting its best earlier accuracy
    less its last (Chaudhry et al. 2018), and the backward transfer the mean of each earlier task's last
    accuracy less its own stage's (Lopez-Paz and Ranzato 2017)."""
    matrix = AccuracyMatrix(np.array([[0.9, 0.1, 0.0], [0.6, 0.8, 0.2], [0.5, 0.7, 0.95]]))
    assert matrix.average() == pytest.approx((0.5 + 0.7 + 0.95) / 3, abs=0)
    np.testing.assert_array_equal(matrix.forgetting(), np.array([0.9 - 0.5, 0.8 - 0.7]))
    assert matrix.average_forgetting() == np.mean([0.9 - 0.5, 0.8 - 0.7])
    assert matrix.backward_transfer() == np.mean([0.5 - 0.9, 0.7 - 0.8])
    assert AccuracyMatrix(np.array([[0.4]])).record() == {
        "continual/average": 0.4, "continual/forgetting": 0.0, "continual/backward_transfer": 0.0}


def test_an_accuracy_matrix_is_square_and_its_accuracies_shares():
    """An accuracy matrix holds one row a stage and one column a task, and accuracies in [0, 1]."""
    with pytest.raises(ValueError):
        AccuracyMatrix(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        AccuracyMatrix(np.array([[1.5]]))


def test_a_continual_run_scores_every_task_after_each():
    """A run trains its tasks in order and scores every task after each, its matrix the scores by stage
    and task."""
    trained = []

    def fit(stage, task):
        trained.append((stage, task))

    def accuracy(stage, scored):
        return (10 * stage + scored) / 100

    matrix = continual_run(["first", "second", "third"], fit, accuracy)
    assert trained == [(0, "first"), (1, "second"), (2, "third")]
    np.testing.assert_array_equal(matrix.values, [[0.0, 0.01, 0.02], [0.1, 0.11, 0.12], [0.2, 0.21, 0.22]])


def test_label_accuracy_is_the_share_of_prompts_whose_best_label_is_their_answer():
    """A prompt's choice is its highest-scoring label, and the accuracy the share answered."""
    scores = np.array([[-1.0, -2.0], [-3.0, -0.5], [-0.1, -0.2]])
    assert label_accuracy(scores, [0, 1, 1]) == 2 / 3
    with pytest.raises(ValueError):
        label_accuracy(scores, [0])


def test_label_scores_refuse_a_prompt_with_no_token_to_score_a_label_from():
    """A label's first token is scored from the token before it, which an empty prompt does not have:
    the scores refuse it, where a score read from position -1, the row's last, left the first token of
    every label unscored and a one-token label scored 0 (row 3941)."""
    with pytest.raises(ValueError):
        label_scores(lambda rows: jnp.zeros(rows.shape, jnp.float32), [[]], [[1], [2, 3]])


@pytest.mark.parametrize("adapted", [False, True], ids=["plain", "in-context"])
def test_label_scores_are_the_labels_summed_log_likelihood(adapted):
    """Each label's score after each prompt is the sum of its tokens' log-likelihoods as the objective
    scores the row (check 1 against the float64 log-softmax of the model's logits): the plain model's, and
    a model whose continuum level steps in context, which chooses through its steps."""
    model = CausalTransformer(vocab_size=VOCAB, emb_features=8, num_layers=1, num_heads=2, head_dim=4,
                              mlp_features=16, max_seq_len=32, attention_impl="xla",
                              continuum=ContinuumLevel(hidden=4) if adapted else None)
    adaptation = ChunkAdaptation(window=2, rate=0.1) if adapted else None
    objective = LMObjective(model, 7, adaptation=adaptation)
    prompts = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    labels = [[10], [11, 12], [13, 14, 15]]

    def drawn(seed):
        rng = np.random.default_rng(seed)
        shapes = jax.eval_shape(lambda: objective.init(jax.random.key(0)))
        return (jax.tree.map(lambda leaf: (0.5 * rng.standard_normal(leaf.shape)).astype(np.float32),
                             shapes),)

    def scored(variables):
        return label_scores(lambda rows: objective.token_scores(variables, rows).losses, prompts, labels)

    def literal(variables):
        # Each pair's row alone at its own length, its label's losses summed by hand.
        return jnp.stack([jnp.stack([
            -jnp.sum(objective.token_scores(variables, jnp.asarray([[*prompt, *label]], jnp.int32)).losses[
                0, len(prompt) - 1:]) for label in labels]) for prompt in prompts])

    assert_as_exact(jax.jit(scored), jax.jit(literal), drawn,
                    f"label scores, {'in context' if adapted else 'plain'}", seeds=LABEL_SEEDS)
