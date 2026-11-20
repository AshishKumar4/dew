"""Generation by a language model whose continuum level steps in context (row 9066): its decode steps the
level as its training forward steps it (`dew.objectives.lm.adaptation.InContextDecoder`), a row's next
token read through the level its training forward gives that slot, and a saved run's task steps it alike.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from reference_error import assert_as_exact, dew_path, widened
from test_chunk_adaptation import LENGTH, LEVEL, MODEL, VOCAB, decoder, drawn_variables

from dew.inference.tasks import TextGeneration
from dew.objectives.lm import LMObjective
from dew.objectives.lm.adaptation import ChunkAdaptation, InContextDecoder, adapted_scores
from dew.sampling import Sampling
from dew.sampling.strategies import Beam, Speculative

ROWS = 2
PROMPT = 5
WINDOWS = (3, (2, 4))
"""The adaptations' windows: three, which two of the decode's windows close past its prompt, and a
chain's two levels of two and four."""

DECODE_SEEDS = 8
"""Check 1's declared seeds for the decode against the training forward (`seeds_for`, paired)."""

WIDE = decoder(LEVEL, dtype=jnp.float64)
"""The test decoder in float64: the truth's."""


def chained(window):
    """The decoder whose levels `window` names: one level of a window, or a chain of two."""
    return MODEL if isinstance(window, int) else decoder((LEVEL, LEVEL))


def drawn(seed: int, window):
    """A seed's variables (`drawn_variables`) and token rows, the second row's first two slots padding, as
    a padded prompt's are."""
    rng = np.random.default_rng([9066, seed])
    tokens = rng.integers(0, VOCAB, (ROWS, LENGTH)).astype(np.int32)
    mask = np.ones((ROWS, LENGTH), bool)
    mask[1, :2] = False
    return drawn_variables(seed, chained(window), scale=0.5), tokens, mask


def trained(model, adaptation: ChunkAdaptation):
    """`model`'s training forward's logits over a draw's rows (`adapted_scores`), each slot's target the
    next token, the last slot's weighing nothing."""
    def run(variables, tokens, mask):
        fields = {"attention_mask": mask}
        hidden = model.apply(variables, tokens, attention_mask=mask, method="hidden_states")
        targets = jnp.concatenate([tokens[:, 1:], tokens[:, :1]], axis=1)
        weights = jnp.ones(tokens.shape, jnp.float32).at[:, -1].set(0)
        return adapted_scores(model, variables, hidden, targets, weights, fields, adaptation, predict=False,
                              logits=True).logits
    return run


def decoded(model, adaptation: ChunkAdaptation):
    """The adapting decoder's logits over a draw's rows: the prompt's slots prefilled, each later token
    decoded alone, its cache carried between the calls (`InContextDecoder`)."""
    decoding = InContextDecoder(model, adaptation)

    def run(variables, tokens, mask):
        cache = decoding.apply(variables, tokens.shape[0], method="init_cache", mutable=["cache"])[1]["cache"]
        found, updated = decoding.apply({**variables, "cache": cache}, tokens[:, :PROMPT], decode=True,
                                        attention_mask=mask[:, :PROMPT], mutable=["cache"])
        steps = [found]
        for at in range(PROMPT, tokens.shape[1]):
            step, updated = decoding.apply({**variables, "cache": updated["cache"]}, tokens[:, at:at + 1],
                                           decode=True, mutable=["cache"])
            steps.append(step)
        return jnp.concatenate(steps, axis=1)
    return run


@pytest.mark.parametrize("window", WINDOWS, ids=["one level", "a chain"])
def test_a_decode_reads_the_logits_its_training_forward_gives(window):
    """The adapting decoder's logits at every slot of a padded prompt and the tokens decoded after it,
    two of its windows closing past the prompt, on par with its training forward's over the same tokens
    (check 1, the forward Dew's own path: `dew_path`), the truth the forward in float64: the decode steps
    each row's level where its training steps it and reads each slot through the version its training
    reads it at (row 9066)."""
    adaptation = ChunkAdaptation(window=window, rate=0.1)
    model = chained(window)
    wide = WIDE if isinstance(window, int) else decoder((LEVEL, LEVEL), dtype=jnp.float64)

    def truth(variables, tokens, mask):
        return trained(wide, adaptation)(widened(variables), tokens, mask)

    assert_as_exact(jax.jit(decoded(model, adaptation)), dew_path(jax.jit(trained(model, adaptation))),
                    lambda seed: drawn(seed, window), f"a decode stepping a level of window {window}",
                    seeds=DECODE_SEEDS, truth_fn=truth)


def test_an_adapting_policy_draws_what_its_saved_task_draws():
    """An adapting objective's policy (`LMObjective.policy`) and the task its settings rebuild
    (`TextGeneration.from_settings` of `task_settings`, the adaptation recorded) draw the same tokens: a
    saved run's task steps the level as the live one does (row 9066)."""
    objective = LMObjective(MODEL, LENGTH, adaptation=ChunkAdaptation(window=3, rate=0.1))
    variables = nn.unbox(objective.init(jax.random.key(0)))
    settings = objective.task_settings()
    assert settings is not None and settings.adaptation is not None
    saved = TextGeneration.from_settings(MODEL, variables, settings, None)
    assert isinstance(saved.model, InContextDecoder)
    prompt = np.random.default_rng(3).integers(0, VOCAB, (ROWS, PROMPT)).tolist()
    greedy = Sampling(temperature=0)
    live = objective.policy(variables, sampling=greedy)(prompt, 7, key=0)
    again = TextGeneration.from_settings(MODEL, variables, settings, None)(prompt, 7, key=0, sampling=greedy)
    np.testing.assert_array_equal(np.asarray(live.tokens), np.asarray(again.tokens))


def test_what_an_adapting_decoder_cannot_draw_is_refused():
    """A strategy that reads tokens it then rejects or reorders (a speculative block's, a beam's) draws
    past what an adapting decoder's cache rewinds, and a damped or stacked adaptation decodes none: each
    refused, naming why (ValueError)."""
    model = InContextDecoder(MODEL, ChunkAdaptation(window=3))
    variables = drawn_variables(0)
    for strategy in (Beam(width=2), Speculative()):
        with pytest.raises(ValueError, match="a token a row at a time"):
            TextGeneration(model, variables, strategy=strategy)
    for adaptation in (ChunkAdaptation(damped=True), ChunkAdaptation(layers=(0,))):
        with pytest.raises(ValueError):
            InContextDecoder(MODEL, adaptation)
