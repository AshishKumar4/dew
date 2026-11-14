"""Check 1 of a server's draws against a full forward of its decoder, shared by the serving tests."""

import dataclasses
import functools
import itertools
from collections.abc import Callable, Mapping, Sequence

import cache_roundings
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from reference_error import Followed, Rounded, as_given, assert_as_exact, widened

from dew.batch import Variables
from dew.inference.serving import Server
from dew.inference.tasks import TextGeneration
from dew.nn.inputs import ModelInputs
from dew.nn.kv_cache import KVCache
from dew.nn.visibility import with_served_blocks
from dew.sampling import Sampling

SERVED_BUDGET = 3
"""The tokens each served request draws, where a site names no other."""


@dataclasses.dataclass(frozen=True)
class Served:
    """One served request: its prompt's first `length` tokens of the draw's row, with `flipped` places
    changed, carrying the groups `groups` names (a field's group of each token, -1 for none) and the
    conditioning `conditioning` draws from the row (a cross-attention's context, one row of each
    entry), none where None."""

    length: int
    groups: Mapping[str, tuple[int, ...]] = dataclasses.field(default_factory=dict)
    flipped: tuple[int, ...] = ()
    conditioning: Callable[[np.ndarray], Mapping[str, np.ndarray]] | None = None

    def conditioned(self, row: np.ndarray) -> dict[str, np.ndarray]:
        """The request's conditioning in the draw of `row`, empty where it has none."""
        return {} if self.conditioning is None else dict(self.conditioning(row))


def assert_same_draws(served, alone) -> None:
    """A served request draws what it draws alone: its tokens, length and termination, to the bit. Its
    likelihoods are two programs' of one arithmetic, the server's step's sums and the lone task's, which
    check 1 holds over the server's arithmetic (`assert_served_as_exact`'s sites), not here."""
    served, alone = served.host(), alone.host()
    for field in ("tokens", "lengths", "terminated"):
        np.testing.assert_array_equal(getattr(served, field), getattr(alone, field), err_msg=field)


def assert_same_generation(actual, expected) -> None:
    """Two generations of one program over the same weights and inputs, its draws and both likelihoods to
    the bit: a task and the function it runs, a key given as a seed or as a key, a task and its placed
    copy, a server and one reloaded to the same storage, a cache held on the host and one resident."""
    actual, expected = actual.host(), expected.host()
    for field in ("tokens", "lengths", "terminated", "behavior_log_probs", "raw_log_probs"):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field), err_msg=field)


def served_inputs(row: np.ndarray, request: Served, vocab: int) -> ModelInputs:
    """`request`'s prompt of the draw's `row`, carrying its groups and its conditioning, a flipped token the
    next id of `vocab`."""
    prompt = row[:request.length].copy()
    prompt[list(request.flipped)] = (prompt[list(request.flipped)] + 1) % vocab
    return ModelInputs(jnp.asarray(prompt[None], jnp.int32),
                       {name: jnp.asarray([held], jnp.int32) for name, held in request.groups.items()},
                       {name: jnp.asarray(value) for name, value in request.conditioned(row).items()})


def to_budgets(generations: list, budget: int, label: str) -> list:
    """`generations`, refused unless each drew its whole `budget`: likelihoods past a request's last drawn
    token are padding, which no full forward scores, so the check serves a task that stops at its budget
    alone (a stop id or a length policy ends a request short; one ending on its budget's last token drew
    it whole)."""
    if any(np.any(np.asarray(generation.lengths) != budget) for generation in generations):
        raise ValueError(f"{label}: a request drew fewer tokens than its budget of {budget}; serve a task "
                         "that stops at its budget alone")
    return generations


def full_forward(model: nn.Module):
    """`model`'s log probabilities `[B, S, V]` over its weights, tokens, their fields and a request's
    conditioning (none where it is empty), compiled as Dew's users compile a forward."""
    def logits(weights, tokens, groups, conditioning):
        read = {"conditioning": conditioning} if conditioning else {}
        logits = model.apply(weights, tokens, **groups, **read)
        assert isinstance(logits, jax.Array)
        return jax.nn.log_softmax(logits)

    return jax.jit(logits)


def drawn_scores(forward, weights: Variables, prompt: np.ndarray, tokens: np.ndarray,
                 held: dict[str, np.ndarray], *, per_step: bool, policy: Sampling) -> list[np.ndarray]:
    """Each drawn token's log probability under `forward` (`[B, S, V]` log probabilities over `weights`,
    tokens and their `held` fields) after `prompt` `[1, P]` and the tokens drawn before it: with
    `per_step`, from a forward of its own over the prompt and those tokens, at the width the step that
    drew it reads (ledger 1543: a step's reference at the shapes the step computes); else from one
    forward over the whole row, as a float64 truth, whose products round alike at every shape. Where
    `policy` samples, each token's log probability under its law too (`behavior`), after the raw one."""
    def at(step: int) -> jax.Array:
        tail = np.full((1, step), -1, np.int32)
        logits = forward(weights, jnp.asarray(np.concatenate([prompt, tokens[None, :step]], 1)),
                         {name: jnp.asarray(np.concatenate([value, tail], 1))
                          for name, value in held.items()})
        return logits[0, prompt.shape[1] - 1:]

    steps = len(tokens)
    read = jnp.stack([at(step)[step] for step in range(steps)]) if per_step else at(steps - 1)
    laws = [read] if policy.temperature == 0 else [read, behavior(read, policy)]
    return [np.asarray(jnp.take_along_axis(law, jnp.asarray(tokens)[:, None], -1)[:, 0]) for law in laws]


def behavior(log_probs: jax.Array, policy: Sampling) -> jax.Array:
    """`log_probs` `[steps, V]` under `policy`'s law of a temperature and a top-k, in Transformers' order as
    `Sampling` runs them: the scores over the temperature, all but the `top_k` highest removed (every score
    tied at the cutoff kept), normalized over what is kept."""
    scaled = log_probs / policy.temperature
    if policy.top_k is not None:
        cutoff = jnp.sort(scaled, axis=-1)[:, -min(policy.top_k, scaled.shape[-1])][:, None]
        scaled = jnp.where(scaled < cutoff, -jnp.inf, scaled)
    return jax.nn.log_softmax(scaled, axis=-1)


def full_forward_scores(task: TextGeneration, requests: Sequence[Served], model: nn.Module,
                        variables: Variables, *, wide: bool):
    """The full forward's scores of each request's draws (`scored(row, drawn)`), a host loop over the
    requests whose forward of `model` over `variables` compiles as Dew's users compile it, as the server's
    programs do. The reference arm scores each drawn token from a forward of its own over the prompt and
    the tokens drawn before it, at the width the server's step reads (ledger 1543, the gate's ruling: a
    step's reference at the shapes the step computes); the float64 truth (`wide`: the float64 twin over
    the variables widened), whose products round alike at every shape, scores them all from one forward
    over the whole row. Every request's raw scores, then, where `task` samples, every request's under its
    law, as the served arm orders them."""
    forward, vocab = full_forward(model), task.model.vocab_size

    @as_given
    def scored(row, drawn, _served=None):
        weights = widened(variables) if wide else variables
        scores = []
        for request, tokens in zip(requests, drawn, strict=True):
            inputs = served_inputs(np.asarray(row), request, vocab)
            prompt, tokens = np.asarray(inputs.tokens), np.asarray(tokens)
            carried = {name: np.asarray(value) for name, value in inputs.token_fields.items()}
            held = with_served_blocks(carried, np.ones(prompt.shape, bool), task.prefix)
            given = request.conditioned(np.asarray(row))
            reading = functools.partial(forward, conditioning=widened(given) if wide else given)
            scores.append(drawn_scores(reading, weights, prompt, tokens, held, per_step=not wide,
                                       policy=task.sampling))
        return tuple(itertools.chain.from_iterable(zip(*scores, strict=True)))
    return scored


def served_row(seed: int, vocab: int, width: int) -> np.ndarray:
    """The row of `width` tokens a served check draws at `seed`, which each request's prompt is cut from."""
    return np.random.default_rng([0x1536, seed]).integers(1, vocab, width).astype(np.int32)


def served_rounds(server: Server, requests: Sequence[Served], row: np.ndarray, rounds: Sequence[int],
                  budget: int, hits: Sequence[int] | None, label: str) -> list:
    """`server`'s generations of `row`'s `requests`, each drawing `budget` tokens under its own key, queued
    in `rounds` (how many go in together before the server runs them), the shared prefix tokens after each
    round `hits` where given (`prefix_hits`); a request that ended before its budget is refused
    (`to_budgets`)."""
    vocab = server.model.vocab_size
    tickets, shared, before = [], [], server.prefix_hits
    for end in itertools.accumulate(rounds):
        tickets.extend(server.submit(served_inputs(row, requests[index], vocab), budget, key=index)
                       for index in range(len(tickets), end))
        server.run()
        shared.append(server.prefix_hits - before)
    assert hits is None or shared == list(hits), shared
    return to_budgets([ticket.result().host() for ticket in tickets], budget, label)


def assert_served_to_the_bit(task: TextGeneration, requests: Sequence[Served], label: str, *, serving: dict,
                             sized: dict, hits: Sequence[int], draws: int,
                             rounds: Sequence[int] | None = None, budget: int = SERVED_BUDGET) -> None:
    """A server over `task` with `serving` serves `requests` what a server with `sized` serves them, over
    `draws` draws of a served check's rows (`served_row`), every field to the bit, with its own shared prefix
    tokens after each round (`hits`): the two differ in what they keep of a prefix and where (how many
    states, on the device or the host), not in a step's program, so a state restored, read back from the
    host or computed again is the same bits, and check 1 of the server with `sized`
    (`assert_served_as_exact`, the site that names those options) holds this one's likelihoods too."""
    if rounds is None:
        rounds = (1,) * len(requests)
    mine, theirs = (Server.from_task(task, **options) for options in (serving, sized))
    vocab, width = task.model.vocab_size, max(request.length for request in requests)
    for seed in range(draws):
        row = served_row(seed, vocab, width)
        ours = served_rounds(mine, requests, row, rounds, budget, hits, label)
        for served, held in zip(ours, served_rounds(theirs, requests, row, rounds, budget, None, label),
                                strict=True):
            assert_same_generation(served, held)


def samples(policy: Sampling, label: str) -> bool:
    """Whether `policy` samples, refused where it does under a law the check does not score (`behavior`)."""
    # The law's own controls: which ids end or pad a row is no part of a token's likelihood.
    others = dataclasses.replace(policy, temperature=1.0, top_k=None, eos_token_ids=None, pad_token_id=None)
    if policy.temperature != 0 and others != Sampling():
        raise ValueError(f"{label}: the check scores a sampled draw under a temperature and a top-k alone, "
                         f"not {policy}")
    return policy.temperature != 0


def assert_served_as_exact(task: TextGeneration, requests: Sequence[Served], label: str, *, serving: dict,
                           seeds: tuple[int, int], hits: Sequence[int] | None = None,
                           rounds: Sequence[int] | None = None, budget: int = SERVED_BUDGET,
                           alone: bool = True, truth: tuple[nn.Module, Variables] | None = None) -> None:
    """Check 1 of a server's draws: requests through a server over `task` with `serving`, queued in
    `rounds` (how many go in together before the server runs them; all at once where None, one a round
    where `hits` is given), with `hits` the shared prefix tokens after each round (`prefix_hits`), draw
    what the lone task draws, and score each drawn token's raw log probability on par with a full forward
    of the decoder over the prompt, its groups (the prefix run's where it carries none) and the tokens
    drawn before it, from the float64 twin's. `task` stops at its `budget` alone and draws greedily, or
    samples at a temperature and a top-k, each request under its own key: its behavior log probabilities
    are then held the same way, to the same forward's under that law (`behavior`); a law of other controls
    is refused, which the check does not score.

    Without `alone`, the tokens scored are the server's own draws, each draw one run of it, and the lone
    task's own draws are held to the same full forward after it, so served and alone are each on par with
    it: a model whose products round at boundaries (a quantized memory's moving weights, ledger 2032)
    draws, from one program of its arithmetic and another, tokens that need not agree, a lone task
    batched with itself among them. `truth` is the decoder and float variables whose float64 twin is the
    truth where the task's own do not widen to one (a served quantization's stored codes); the task's by
    default. `seeds` are the served site's and the lone one's declared seeds, sized paired by `seeds_for`
    on armada, never here: sizing a site serves its draws again, run after run."""
    if rounds is None:
        rounds = (len(requests),) if hits is None else (1,) * len(requests)
    sampled = samples(task.sampling, label)
    decoder, variables, vocab = task.model, task.variables, task.model.vocab_size
    held_model, held_variables = (decoder, variables) if truth is None else truth
    wide = held_model.clone(dtype=jnp.float64, attention_impl="reference")
    width = max(request.length for request in requests)

    # One server for every draw, as a deployment serves request after request: its programs compile
    # once, and a draw's prompts share no page with another's, their tokens drawn apart.
    server = Server.from_task(task, **serving)
    services: dict[bytes, list] = {}

    def serve(row) -> list:
        """The server's generations of `row`'s requests (`served_rounds`): each row served once, as a
        deployment serves a prompt, and asked again (the gate's check that a draw and an arm are functions
        of their inputs) the generations of that one service, as a second would read the pages and states
        the first left, which no other draw's prompt shares."""
        if (key := np.asarray(row).tobytes()) not in services:
            services[key] = served_rounds(server, requests, row, rounds, budget, hits, label)
        return services[key]

    def drawn_row(seed) -> np.ndarray:
        return served_row(seed, vocab, width)

    def likelihoods(generations) -> tuple:
        """Each request's raw log probabilities, then, where the task samples, each one's under its law."""
        fields = ("raw_log_probs", "behavior_log_probs") if sampled else ("raw_log_probs",)
        return tuple(np.asarray(getattr(generation, field)[0], np.float32)
                     for field in fields for generation in generations)

    def draw(seed):
        row = drawn_row(seed)
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

    # Paired: the server and the full forward run one model's arithmetic over the same weights and tokens,
    # their errors moving together across draws, which a hybrid stack's recurrence spreads too widely for
    # the reference compared with itself to resolve: a caller's seeds are sized paired (`seeds_for`).
    reference = full_forward_scores(task, requests, decoder, variables, wide=False)
    truth = full_forward_scores(task, requests, wide, held_variables, wide=True)
    assert_as_exact(served, reference, draw, label, seeds=seeds[0], truth_fn=truth)
    if alone:
        return

    def drawn_alone(seed):
        """The lone task's draws of the seed's requests and their likelihoods as it scored them."""
        row = drawn_row(seed)
        generations = [task(served_inputs(row, request, vocab), budget, key=index).host()
                       for index, request in enumerate(requests)]
        return row, tuple(np.asarray(generation.tokens)[0, -budget:] for generation in generations), \
            likelihoods(generations)

    # The lone task held to the same full forward: served and alone each on par with it, so with each other.
    lone = as_given(lambda row, drawn, scored: scored)
    assert_as_exact(lone, reference, drawn_alone, f"{label} alone", seeds=seeds[1], truth_fn=truth)


def assert_served_through_its_cache_as_exact(task: TextGeneration, lengths: Sequence[int], label: str, *,
                                             serving: dict, budget: int, seeds: int) -> None:
    """Check 1 of a server over a quantized cache (`serving`'s `kv_cache`, `KVCache.quantized`; ledger 5681):
    requests of the prompt `lengths` of a drawn row (`served_row`), served one a round, score their own draws'
    raw log probabilities on par with the decoder's own cached decode of each prompt and its draws through
    a cache of the same format. Each arm gives the values its cache rounded and the codes and scales it
    stored (`cache_roundings`), and is held to the float64 forward over the request's tokens that reads the
    codes and scales its own cache held (`Followed`): a float32 rounding a value within its noise of a
    code's boundary decides it either way, so no truth that rounds for itself holds every arm
    (docs/design/testing.md, a quantized computation's truth). A request starts its row's slots afresh in
    a round of its own, so the roundings a round reports are its own (`Written.started`). `seeds` is the
    site's declared count, sized paired by `seeds_for` on armada."""
    quantized = serving["kv_cache"].quantized
    decoder, variables, vocab = task.model, task.variables, task.model.vocab_size
    width = max(lengths)

    def named(prefix: str, held: dict) -> dict:
        """A request's roundings by name as the site names them: its values under `rounded/<request>`, its
        decisions under `<request>`."""
        return {f"{prefix}/{name}": value for name, value in held.items()}

    # One server for every draw, its programs traced once under the capture, each request in its own round.
    with cache_roundings.traced():
        server = Server.from_task(task, **serving)
    services: dict[bytes, tuple[Rounded, tuple[np.ndarray, ...]]] = {}

    def serve(row: np.ndarray) -> tuple[Rounded, tuple[np.ndarray, ...]]:
        """The server's scores of `row`'s requests' draws with its cache's roundings, and the draws: each row
        served once, as a deployment serves a prompt (`assert_served_as_exact`'s `serve`)."""
        if (key := row.tobytes()) not in services:
            scores, drawn, rounded, decisions = [], [], {}, {}
            for index, length in enumerate(lengths):
                with cache_roundings.traced(), cache_roundings.collected() as written:
                    ticket = server.submit(served_inputs(row, Served(length), vocab), budget, key=index)
                    server.run()
                    generation, = to_budgets([ticket.result().host()], budget, label)
                drawn.append(np.asarray(generation.tokens)[0, -budget:])
                scores.append(np.asarray(generation.raw_log_probs[0], np.float32))
                values, decided = written.held(written.started(), length + budget - 1)
                rounded.update(named(f"rounded/{index}", values))
                decisions.update(named(str(index), decided))
            services[key] = Rounded({"scores": tuple(scores), **rounded}, decisions), tuple(drawn)
        return services[key]

    def draw(seed):
        row = served_row(seed, vocab, width)
        return row, serve(row)[1]

    @as_given
    def served(row, drawn):
        del drawn
        return serve(row)[0]

    cached = decoder.with_kv_cache(KVCache(quantized=quantized))
    opened = jax.jit(lambda weights: cached.apply(weights, 1, method="init_cache",
                                                  mutable=["cache"])[1]["cache"])

    @jax.jit
    def stepped(weights, cache, tokens):
        with cache_roundings.traced():
            logits, held = cached.apply({**weights, "cache": cache}, tokens, mutable=["cache"], decode=True)
        return jax.nn.log_softmax(logits[0, -1]), held["cache"]

    @as_given
    def reference(row, drawn):
        """The decoder's own cached decode of each prompt and the tokens the server drew after it, a call a
        token, and its cache's roundings."""
        scores, rounded, decisions = [], {}, {}
        for index, (length, tokens) in enumerate(zip(lengths, drawn, strict=True)):
            with cache_roundings.collected() as written:
                prompt = served_inputs(row, Served(length), vocab).tokens
                read, cache = stepped(variables, opened(variables), prompt)
                laws = [read]
                for token in tokens[:-1]:
                    read, cache = stepped(variables, cache, jnp.asarray([[token]], jnp.int32))
                    laws.append(read)
            scores.append(np.asarray(jnp.stack(laws))[np.arange(budget), tokens])
            values, decided = written.held(0, length + budget - 1)
            rounded.update(named(f"rounded/{index}", values))
            decisions.update(named(str(index), decided))
        return Rounded({"scores": tuple(scores), **rounded}, decisions)

    wide = decoder.clone(dtype=jnp.float64, attention_impl="reference").with_kv_cache(
        KVCache(quantized=quantized))

    @jax.jit
    def forward(weights, tokens, decided):
        with cache_roundings.traced(decided):
            cache = wide.apply(weights, 1, method="init_cache", mutable=["cache"])[1]["cache"]
            logits, _ = wide.apply({**weights, "cache": cache}, tokens, mutable=["cache"], decode=True)
        return jax.nn.log_softmax(logits[0])

    @as_given
    def truth(decisions, row, drawn):
        """Each request's tokens through the float64 twin in one call, its cache reading the codes and scales
        an arm's cache held (`decisions`), and the values its own roundings would have rounded."""
        weights, scores, rounded = widened(variables), [], {}
        for index, (length, tokens) in enumerate(zip(lengths, drawn, strict=True)):
            whole = np.concatenate([row[:length], tokens[:-1]])[None]
            own = {name.split("/", 1)[1]: held for name, held in decisions.items()
                   if name.split("/", 1)[0] == str(index)}
            with cache_roundings.collected() as written:
                laws = forward(weights, jnp.asarray(whole, jnp.int32), own)
            scores.append(np.asarray(laws[length - 1:])[np.arange(budget), tokens])
            rounded.update(named(f"rounded/{index}", written.held(0, whole.shape[1])[0]))
        return {"scores": tuple(scores), **rounded}

    followed = Followed(truth, cache_roundings.rounds(quantized))
    assert_as_exact(served, reference, draw, label, seeds=seeds, followed=followed)
