"""Check 1 of a server's draws against a full forward of its decoder, shared by the serving tests."""

import dataclasses
from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from reference_error import assert_as_exact, seeds_for, widened, without_ynn

from dew.inference.serving import Server
from dew.inference.tasks import TextGeneration
from dew.nn.inputs import ModelInputs
from dew.nn.visibility import with_served_blocks

SERVED_BUDGET = 3
"""The tokens each served request draws, where a site names no other."""


@dataclasses.dataclass(frozen=True)
class Served:
    """One served request: its prompt's first `length` tokens of the draw's row, with `flipped` places
    changed, carrying the groups `groups` names (a field's group of each token, -1 for none)."""

    length: int
    groups: Mapping[str, tuple[int, ...]] = dataclasses.field(default_factory=dict)
    flipped: tuple[int, ...] = ()


def served_inputs(row: np.ndarray, request: Served, vocab: int) -> ModelInputs:
    """`request`'s prompt of the draw's `row`, carrying its groups, a flipped token the next id of `vocab`."""
    prompt = row[:request.length].copy()
    prompt[list(request.flipped)] = (prompt[list(request.flipped)] + 1) % vocab
    return ModelInputs(jnp.asarray(prompt[None], jnp.int32),
                       {name: jnp.asarray([held], jnp.int32) for name, held in request.groups.items()})


def assert_served_as_exact(task: TextGeneration, requests: Sequence[Served], label: str, *, serving: dict,
                           hits: Sequence[int] | None = None, budget: int = SERVED_BUDGET) -> None:
    """Check 1 of a server's draws: greedy requests through a server over `task` with `serving`, all queued
    at once, or with `hits` one after another (`prefix_hits` after each, its shared prefix tokens), draw
    what the lone task draws, and score each drawn token's raw log probability on par with a full forward
    of the decoder over the prompt, its groups (the prefix run's where it carries none) and the tokens
    drawn before it, from the float64 twin's. `task` draws greedily and stops at its `budget` alone."""
    decoder, variables, vocab = task.model, task.variables, task.model.vocab_size
    wide = decoder.clone(dtype=jnp.float64, attention_impl="reference")
    width = max(request.length for request in requests)

    def draw(seed):
        row = np.random.default_rng([0x1536, seed]).integers(1, vocab, width).astype(np.int32)
        drawn = tuple(np.asarray(task(served_inputs(row, request, vocab), budget, key=index).host()
                                 .tokens)[0, -budget:] for index, request in enumerate(requests))
        return row, drawn

    def served(row, drawn):
        server = Server.from_task(task, **serving)
        tickets, shared = [], []
        for index, request in enumerate(requests):
            tickets.append(server.submit(served_inputs(row, request, vocab), budget, key=index))
            if hits is not None:
                server.run()
                shared.append(server.prefix_hits)
        server.run()
        assert hits is None or shared == list(hits), shared
        generations = [ticket.result().host() for ticket in tickets]
        for generation, tokens in zip(generations, drawn, strict=True):
            np.testing.assert_array_equal(generation.tokens[0, -budget:], tokens)
        return tuple(np.asarray(generation.raw_log_probs[0], np.float32) for generation in generations)

    def explicit(model, wide_weights: bool = False):
        forward = without_ynn(lambda weights, tokens, groups: jax.nn.log_softmax(
            model.apply(weights, tokens, **groups)))

        def scored(row, drawn):
            weights = widened(variables) if wide_weights else variables
            scores = []
            for request, tokens in zip(requests, drawn, strict=True):
                inputs = served_inputs(np.asarray(row), request, vocab)
                prompt, tokens = np.asarray(inputs.tokens), np.asarray(tokens)
                carried = {name: np.asarray(value) for name, value in inputs.token_fields.items()}
                held = with_served_blocks(carried, np.ones(prompt.shape, bool), task.prefix)
                tail = np.full((1, budget - 1), -1, np.int32)
                logits = forward(weights, jnp.asarray(np.concatenate([prompt, tokens[None, :-1]], 1)),
                                 {name: jnp.asarray(np.concatenate([value, tail], 1))
                                  for name, value in held.items()})
                read = logits[0, prompt.shape[1] - 1:]
                scores.append(np.asarray(jnp.take_along_axis(read, jnp.asarray(tokens)[:, None], -1)[:, 0]))
            return tuple(scores)
        return scored

    # Paired: the server and the full forward run one model's arithmetic over the same weights and tokens,
    # their errors moving together across draws, which a hybrid stack's recurrence spreads too widely for
    # the reference compared with itself to resolve (`seeds_for`).
    seeds = seeds_for(explicit(decoder), draw, label, truth_fn=explicit(wide, wide_weights=True),
                      dew_fn=served)
    assert_as_exact(served, explicit(decoder), draw, label, seeds=seeds,
                    truth_fn=explicit(wide, wide_weights=True))
