"""Check 1 of a server's draws against a full forward of its decoder, shared by the serving tests."""

import dataclasses
import itertools
from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from reference_error import as_given, assert_as_exact, seeds_for, widened

from dew.batch import Variables
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
                           hits: Sequence[int] | None = None, rounds: Sequence[int] | None = None,
                           budget: int = SERVED_BUDGET, alone: bool = True,
                           truth: tuple[nn.Module, Variables] | None = None) -> None:
    """Check 1 of a server's draws: greedy requests through a server over `task` with `serving`, queued in
    `rounds` (how many go in together before the server runs them; all at once where None, one a round
    where `hits` is given), with `hits` the shared prefix tokens after each round (`prefix_hits`), draw
    what the lone task draws, and score each drawn token's raw log probability on par with a full forward
    of the decoder over the prompt, its groups (the prefix run's where it carries none) and the tokens
    drawn before it, from the float64 twin's. `task` draws greedily and stops at its `budget` alone.

    Without `alone`, the tokens scored are the server's own draws, each draw one run of it: a model whose
    products round at boundaries (a quantized memory's moving weights, ledger 2032) draws, from one
    program of its arithmetic and another, tokens that need not agree, a lone task batched with itself
    among them. `truth` is the decoder and float variables whose float64 twin is the truth where the
    task's own do not widen to one (a served quantization's stored codes); the task's by default."""
    if rounds is None:
        rounds = (len(requests),) if hits is None else (1,) * len(requests)
    decoder, variables, vocab = task.model, task.variables, task.model.vocab_size
    held_model, held_variables = (decoder, variables) if truth is None else truth
    wide = held_model.clone(dtype=jnp.float64, attention_impl="reference")
    width = max(request.length for request in requests)

    # One server for every draw, as a deployment serves request after request: its programs compile
    # once, and a draw's prompts share no page with another's, their tokens drawn apart.
    server = Server.from_task(task, **serving)

    def serve(row) -> list:
        """The server's generations of `row`'s requests, queued in `rounds`, its prefix hits checked."""
        tickets, shared, before = [], [], server.prefix_hits
        for end in itertools.accumulate(rounds):
            tickets.extend(server.submit(served_inputs(row, requests[index], vocab), budget, key=index)
                           for index in range(len(tickets), end))
            server.run()
            shared.append(server.prefix_hits - before)
        assert hits is None or shared == list(hits), shared
        return [ticket.result().host() for ticket in tickets]

    def likelihoods(generations) -> tuple:
        return tuple(np.asarray(generation.raw_log_probs[0], np.float32) for generation in generations)

    def draw(seed):
        row = np.random.default_rng([0x1536, seed]).integers(1, vocab, width).astype(np.int32)
        if not alone:
            # The server's draws and their likelihoods as it scored them, from one run of it.
            generations = serve(row)
            drawn = tuple(np.asarray(generation.tokens)[0, -budget:] for generation in generations)
            return row, drawn, likelihoods(generations)
        drawn = tuple(np.asarray(task(served_inputs(row, request, vocab), budget, key=index).host()
                                 .tokens)[0, -budget:] for index, request in enumerate(requests))
        return row, drawn

    @as_given
    def served(row, drawn, scored=None):
        if scored is not None:
            return scored
        generations = serve(row)
        for generation, tokens in zip(generations, drawn, strict=True):
            np.testing.assert_array_equal(generation.tokens[0, -budget:], tokens)
        return likelihoods(generations)

    def explicit(model, wide_weights: bool = False):
        """The full forward's scores of each request's draws, a host loop over the requests whose
        forward compiles as Dew's users compile it, as the server's programs do."""
        forward = jax.jit(lambda weights, tokens, groups: jax.nn.log_softmax(
            model.apply(weights, tokens, **groups)))

        @as_given
        def scored(row, drawn, _served=None):
            weights = widened(held_variables) if wide_weights else variables
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
    reference, truth = explicit(decoder), explicit(wide, wide_weights=True)
    seeds = seeds_for(reference, draw, label, truth_fn=truth, dew_fn=served)
    assert_as_exact(served, reference, draw, label, seeds=seeds, truth_fn=truth)
