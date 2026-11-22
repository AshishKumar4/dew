"""A looped decoder's passes (`Loop`): the prelude, the block's first pass, its later passes under one
scan, and the coda.

The block's later passes are one body under flax's scan over the passes, so a step's program holds the
first pass and one later pass whatever the loop's count: Dory's at four loops traces two passes, not four.
The scan stacks what each later pass writes (`WRITTEN`) on a leading pass axis under the block's own layer
names, and the stack holds them as each pass stores them (`PassStack`). Under `shared_cache` every later
pass starts from the cache as the step entered it and only the last pass's writes stay
(`SharedPassView`): the passes before it carry the cache through the scan, each reading the step's
entering cache in place of the one the pass before left, and the carry is dropped; the last runs after
the scan, as the first does. What a pass sows, a router's or attention's statistics or captured
intermediates, extends the step's sows pass by pass, as the unrolled passes' did, a router's and
attention's at the block's own layer names (`BY_CALL`), each pass a call of the same weights. The
passes before the last `backprop_steps` pass no gradient back: the cut is static, the passes before it
scanned, the pass it cuts traced on its own and those after it scanned, so no pass before it keeps
residuals or runs a backward. Where every pass exits (`Loop.exiting`: an exit gate, or `exits`) each
pass's exit state leaves as the last pass's does, the coda run on its output under its own cache
entries, so the later passes are traced one by one.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core.scope import DenyList

from dew.batch import sows_appended

from ..precision import at_least_fp32, softplus
from ..protocols import EXITS
from .decoder_stack import BY_CALL, WRITTEN, Loop, PassView, SharedPassView
from .layer_names import group_layers


def _preamble(model, loop: Loop, x, injected, *, stop: bool, normed=None, theta=None):
    """A later pass's input from the state the pass before left: normed under `step_norm` (by `normed`,
    the norm the exit gate already read, where it read this state with no coda between), cut from the
    gradient where `stop` says, the block's input added back under `reinject`, or the block's input moved
    toward the state by θ under `implicit` (`a + θ⊙(h - a)`, `a` the block's input, θ the pass's own,
    `theta`, which `_theta` gives an implicit loop alone)."""
    if loop.step_norm:
        x = model.norm(x) if normed is None else normed
    x = jax.lax.stop_gradient(x) if stop else x
    if loop.reinject:
        x = model.norm(x + injected)
    if theta is not None:
        x = injected + theta.astype(x.dtype) * (x - injected)
    return x


def _theta(model, loop: Loop, steps: range, *, rows: bool = False):
    """The θ the later passes of `steps` read, under `implicit`: `model.loop_theta`, one every pass reads, or
    with `theta_passes` its row t - 1 for pass t, a pass's own: one pass's row, or with `rows` the passes'
    rows `[len(steps), d]` (None where they read one θ); None where the loop is not implicit."""
    if not loop.implicit or (rows and loop.theta_passes is None):
        return None
    if loop.theta_passes is None:
        return model.loop_theta
    held = model.loop_theta[steps.start - 1:steps.stop - 1]
    return held if rows else held[0]


def exit_log_distribution(logits: Sequence[jax.Array]) -> jax.Array:
    """The log of each token's probability of exiting at each pass `[passes, B, S]`, from every pass's
    exit-gate logit `[B, S, 1]`: a pass's its hazard `sigmoid(logit)` times the chance no pass before it
    exited, the last pass's that chance after them all, its own logit read for nothing (Ouro's exit pdf,
    OuroForCausalLM.forward). Summed from the logits' log-sigmoids in at least fp32, so a hazard that
    rounds to one leaves the passes after it a finite log probability, whose gradient stays finite,
    where the probability's own log would be log 0; a loop of one pass exits there surely."""
    hazards = jnp.stack(logits)[:-1, ..., 0].astype(at_least_fp32(logits[0].dtype))
    kept = jnp.cumsum(-softplus(hazards), axis=0)
    before = jnp.concatenate([jnp.zeros_like(logits[0][None, ..., 0], kept.dtype), kept])
    return before + jnp.concatenate([-softplus(-hazards), jnp.zeros_like(before[:1])])


EXIT_ENTROPY = 0.1
"""The exit objective's entropy weight β (`exit_objective`) a run that draws a fresh exit gate trains it under
where it states none: Ouro's for its first pretraining stage, which trained its gates from their
initialization, before it lowered β to 0.05 for the stages after (arXiv 2510.25741 v4, section 4.3.1 and its
training table)."""


def exit_objective(pass_losses: Sequence[jax.Array], logits: Sequence[jax.Array], entropy_weight: float,
                   dtype: jnp.dtype) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Ouro's exit objective (arXiv 2510.25741) per token, in `dtype`, from each pass's loss `[B, S]` and
    every pass's exit-gate logit `[B, S, 1]`: the loss expected under the exit distribution
    (`exit_log_distribution`) less `entropy_weight` times that distribution's entropy, the expectation
    and the entropy. Read from the log probabilities, so a hazard that rounds to one leaves every gradient
    finite. Every objective that trains a looped decoder's gate weighs its passes so."""
    log_chances = exit_log_distribution(logits).astype(dtype)
    chances = jnp.exp(log_chances)
    expected = jnp.sum(chances * jnp.stack(pass_losses).astype(dtype), axis=0)
    entropy = -jnp.sum(chances * log_chances, axis=0)
    return expected - jnp.asarray(entropy_weight, dtype) * entropy, expected, entropy


def exit_choice(logits: Sequence[jax.Array], threshold: float) -> jax.Array:
    """The pass each token exits at under `threshold` (`Loop.exit_threshold`), from the exit-gate logits
    of every pass before the last: the first whose cumulative exit probability
    (`exit_log_distribution`) reaches it, the last pass (`len(logits)`) where none before it does
    (OuroForCausalLM.forward's `exit_threshold`)."""
    # The last pass's logit is read for its shape alone.
    chances = jnp.exp(exit_log_distribution([*logits, logits[-1]]))
    reached = jnp.cumsum(chances[:-1], axis=0) >= threshold
    return jnp.where(jnp.any(reached, axis=0), jnp.argmax(reached, axis=0), len(logits))


def _stepped(loop: Loop, out, x, injected):
    """A later pass's state from the block's output `out` over its input `x`: under `implicit` the block's
    step taken from the block's input (`out + (a - x)`), the output itself otherwise."""
    return out + (injected - x) if loop.implicit else out


@dataclasses.dataclass
class PassStack:
    """What later passes write, stacked on a leading pass axis as a scan over `views`' passes reads it,
    and each pass's back where the unrolled passes put it. A collection a stack writes (`WRITTEN`) holds
    a pass's layer entries under the names its view (`PassView`) stores them by, but for what a pass
    observes of the block's weights (`BY_CALL`); under `shared`, and for those, every pass's entries are
    the block's own, its sows extending them in pass order, and under `shared` the cache is the
    scan's carry, which comes back dropped. What a pass sows elsewhere (`appended`: captured
    intermediates, a name of no layer) extends the tuples it sows into at each leaf, pass by pass, as
    the unrolled passes' sows did. Flax calls `inside` before `outside` within one trace, which holding
    the tree relies on."""

    views: Sequence[PassView]
    shared: bool = False
    appended: frozenset[str] = frozenset()
    held: dict = dataclasses.field(default_factory=dict)

    def _per_pass(self, collection: str) -> bool:
        """Whether `collection` keeps each pass's layer entries apart, under the pass's own names: none of
        what a pass observes of the block's weights (`BY_CALL`), which every pass sows at their names."""
        return collection in WRITTEN and collection not in BY_CALL and not self.shared

    def inside(self, variables: Mapping[str, Mapping]) -> dict:
        self.held = {collection: dict(tree) for collection, tree in variables.items()}
        out = {}
        for collection, tree in self.held.items():
            if self.shared and collection == 'cache':
                # The scan's carry, which each pass replaces with the step's entering cache.
                out[collection] = tree
            elif self._per_pass(collection):
                passes = [view.inside({collection: tree})[collection] for view in self.views]
                named = [{name: value for name, value in entries.items() if group_layers(name) is not None}
                         for entries in passes]
                out[collection] = {name: jax.tree.map(lambda *leaves: jnp.stack(leaves),
                                                      *(entries[name] for entries in named))
                                   for name in named[0]}
            else:
                # Each pass sows afresh; `outside` appends its sows to the step's.
                out[collection] = {}
        return out

    def outside(self, variables: Mapping[str, Mapping]) -> dict:
        out = {collection: dict(tree) for collection, tree in self.held.items()}
        for collection, tree in variables.items():
            if self.shared and collection == 'cache':
                continue
            for index, view in enumerate(self.views):
                entries = {name: jax.tree.map(lambda leaf, index=index: leaf[index], value)
                           for name, value in tree.items()}
                if self._per_pass(collection):
                    out.setdefault(collection, {}).update(
                        {view._moved(name, view.offset - view.first): value
                         for name, value in entries.items() if group_layers(name) is not None})
                    entries = {name: value for name, value in entries.items() if group_layers(name) is None}
                out[collection] = sows_appended(out.get(collection, {}), entries)
        return out


def pass_offset(loop: Loop, layers: int, step: int) -> int:
    """Where later pass `step` (from 1) of a `loop` over a stack of `layers` keeps its block's layer entries
    apart (`PassView.offset`): after the stack's, each pass's after the last's."""
    first, end = loop.layers or (0, layers)
    return layers + (step - 1) * (end - first)


def exit_offset(loop: Loop, layers: int, step: int) -> int:
    """Where pass `step`'s exit (from 0) keeps the entries of its coda, the layers after the block
    (`run_loop`'s exit states): after the later passes' own, which a shared cache's passes do not keep
    apart."""
    first, end = loop.layers or (0, layers)
    later = 0 if loop.shared_cache else (loop.steps - 1) * (end - first)
    return layers + later + step * (layers - end)


def pass_entries(loop: Loop, layers: int, runs: Sequence[tuple[int, int]] | None = None
                 ) -> tuple[tuple[int, int, int], ...]:
    """Each entry a `loop`'s passes keep beyond the stack's own, as `(stored first layer, first layer,
    count)` over the stack's `runs` of `(first, count)` layers (one layer each where None, a store banked
    by run's runs where given): each later pass's block's under a cache of its own (`shared_cache` off),
    and where every pass exits, each exit's coda's (`pass_offset`, `exit_offset`); a run's name moves with
    its first layer (`PassView`)."""
    runs = tuple((layer, 1) for layer in range(layers)) if runs is None else tuple(runs)
    first, end = loop.layers or (0, layers)
    block = [(start, count) for start, count in runs if first <= start < end]
    coda = [(start, count) for start, count in runs if start >= end]
    entries = [] if loop.shared_cache else [
        (pass_offset(loop, layers, step) + start - first, start, count)
        for step in range(1, loop.steps) for start, count in block]
    if loop.exiting and end < layers:
        entries += [(exit_offset(loop, layers, step) + start - end, start, count)
                    for step in range(loop.steps - 1) for start, count in coda]
    return tuple(entries)


def _passed(module, x, bank, *, block: tuple[int, int] | None, train: bool, decode: bool, **inputs):
    """One pass of a loop's block over `x`, reading the bank the pass before left, and the bank it leaves:
    under `Loop.messages` the block a message block at a time (`CrossPass.blocks`), each block's residual
    summary (`residual_summary`) banked on axis 2 (`[B, S, blocks, D]`) and every layer reading `bank`
    (None in a first pass); else the block at once and no bank."""
    messages = module.loop.messages
    if messages is None:
        return module.stack(x, span=block, train=train, decode=decode, **inputs), None
    first, end = block or (0, module.num_layers)
    summaries = []
    for span in messages.blocks(first, end):
        x = module.stack(x, span=span, train=train, decode=decode, prior_bank=bank, **inputs)
        summaries.append(module.residual_summary(x))
    return x, jnp.stack(summaries, axis=2)


def _exit_state(model, x, step: int, *, train: bool, decode: bool, **inputs):
    """A loop's pass `step`'s exit state (from 0): its output, or the coda run on it in the pass's own
    cache (`exit_offset`)."""
    loop, layers = model.loop, model.num_layers
    _, end = loop.layers or (0, layers)
    if end == layers:
        return x
    view = PassView(exit_offset(loop, layers, step), end, layers - end)

    def pass_coda(module, x):
        return module.stack(x, span=(end, layers), train=train, decode=decode, **inputs)

    coda = nn.map_variables(pass_coda, mapped_collections=WRITTEN, trans_in_fn=view.inside,
                            trans_out_fn=view.outside, mutable=True)
    return coda(model, x)


def _entered(model, x, bank, injected, *, stop: bool, normed=None, theta=None):
    """A loop's later pass's input and the bank it reads: under `messages` the block's input and the bank
    the pass before left, which the cut stops (`_passed`); else the pass before's state (`_preamble`)."""
    loop = model.loop
    if loop.messages is not None:
        return injected, jax.lax.stop_gradient(bank) if stop else bank
    return _preamble(model, loop, x, injected, stop=stop, normed=normed, theta=theta), None


def run_loop(model, x, *, train: bool, decode: bool, **inputs):
    """Run `model`'s layer stack over `x` once, or as its `loop` says (the module doc): the layers before
    its block once, the block `steps` times, each later pass reading the last one's output, normed under
    `step_norm`, and the layers after it once; the passes before the last `backprop_steps` pass no gradient
    back, through their exits neither. `remat` checkpoints each block pass outside decode. Where every pass
    exits (`Loop.exiting`) each before the last sows its head's states (and gate logit) into `exits`; with
    `exit_threshold` the output is each token's exiting pass's (`exit_choice`) unless the caller collects
    the exits, whose last pass is the model's output. Under `messages` every pass enters the block at its
    input and each later pass reads the bank of block summaries the pass before left (`_passed`)."""
    loop = model.loop
    if loop is None:
        return model.stack(x, train=train, decode=decode, **inputs)
    layers = model.num_layers
    first, end = loop.layers or (0, layers)
    block = None if loop.layers is None else (first, end)

    def pass_block(module, x, bank=None, **inputs):
        return _passed(module, x, bank, block=block, train=train, decode=decode, **inputs)

    run_block = nn.remat(pass_block) if loop.remat and not decode else pass_block
    if first:
        x = model.stack(x, span=(0, first), train=train, decode=decode, **inputs)
    injected = x
    shared = SharedPassView(block or (0, layers)) if loop.shared_cache else None
    first_pass = run_block if shared is None else nn.map_variables(
        run_block, mapped_collections=WRITTEN, trans_in_fn=shared.first, trans_out_fn=shared.outside,
        mutable=True)
    x, bank = first_pass(model, model.norm(x + injected) if loop.reinject else x, **inputs)
    cut = None if loop.backprop_steps is None else loop.steps - loop.backprop_steps
    exited: list[tuple[jax.Array, jax.Array]] = []

    def entering(module, x, bank, *, stop: bool, normed=None, theta=None):
        return _entered(module, x, bank, injected, stop=stop, normed=normed, theta=theta)

    def unrolled(x, bank, step: int, view):
        """A later pass traced alone over `view`'s collections, the one before it exiting first if all do."""
        normed = None
        if loop.exiting:
            # A pass the cut holds back exits without passing its own gradient back.
            leaving = _exit_state(model, jax.lax.stop_gradient(x) if cut is not None and step <= cut else x,
                                  step - 1, train=train, decode=decode, **inputs)
            output, logit = _sown_exit(model, leaving)
            if logit is not None:  # a gated exit, which the threshold's choice weighs (`_exited`)
                exited.append((leaving, logit))
            normed = output if end == layers and model.normed_output else None
        x, bank = entering(model, x, bank, stop=step == cut, normed=normed,
                           theta=_theta(model, loop, range(step, step + 1)))
        run = nn.map_variables(run_block, mapped_collections=WRITTEN, trans_in_fn=view.inside,
                               trans_out_fn=view.outside, mutable=True)
        out, bank = run(model, x, bank, **inputs)
        return _stepped(loop, out, x, injected), bank

    def own_view(step: int):
        return shared or PassView(pass_offset(loop, layers, step), first, end - first)

    def later_passes(x, bank, steps: range):
        """The later passes `steps` in order, none of them cut: one traced on its own, more under one scan,
        or each on its own where every pass exits."""
        if len(steps) == 1 or loop.exiting:
            # One pass is its own program: a scan of one would only stack and unstack its collections.
            for step in steps:
                x, bank = unrolled(x, bank, step, own_view(step))
            return x, bank
        views = [PassView(pass_offset(loop, layers, step), first, end - first) for step in steps]
        # The collections a pass sows into beside those a stack writes, which an unrolled pass appended
        # to: those the first pass made, and captured intermediates, which a capture of a module only the
        # later passes call first makes inside the scan (the final norm `_preamble` reads by).
        appended = frozenset(collection for collection in (*model.variables, 'intermediates')
                             if model.is_mutable_collection(collection)) - {*WRITTEN, 'params', 'constants'}
        view = PassStack(views, shared=shared is not None, appended=appended)
        mapped = (*WRITTEN, *sorted(appended))
        # Each shared pass reads the step's entering cache, else, where the step enters none, the cache
        # the first pass just opened, whose later writes the scan's dropped carry takes.
        entering_cache = None if shared is None else (shared.entering if shared.entering is not None
                                                     else model.variables.get('cache') or None)

        def later(module, carry, theta):
            # The block's input, its per-layer inputs and the entering cache close over the scan.
            x, bank = entering(module, *carry, stop=False,
                               theta=_theta(module, loop, steps) if theta is None else theta)
            if entering_cache is None:
                out, bank = run_block(module, x, bank, **inputs)
                return (_stepped(loop, out, x, injected), bank), None
            reading = nn.map_variables(run_block, mapped_collections='cache',
                                       trans_in_fn=lambda variables: {**variables, 'cache': entering_cache},
                                       mutable=True)
            out, bank = reading(module, x, bank, **inputs)
            return (_stepped(loop, out, x, injected), bank), None

        stacked = tuple(collection for collection in mapped if shared is None or collection != 'cache')
        scan = nn.scan(later, variable_axes={stacked: 0},
                       variable_carry='cache' if shared is not None else False,
                       variable_broadcast=DenyList(mapped), split_rngs={'params': False, 'dropout': True},
                       length=len(steps), metadata_params={nn.PARTITION_NAME: None})
        run = nn.map_variables(scan, mapped_collections=mapped, trans_in_fn=view.inside,
                               trans_out_fn=view.outside, mutable=True)
        # Each pass's own θ row, if it holds one.
        (x, bank), _ = run(model, (x, bank), _theta(model, loop, steps, rows=True))
        return x, bank

    # Under a shared cache the last pass keeps its writes and runs as the first does, after the scan.
    scanned = range(1, loop.steps - 1 if shared is not None else loop.steps)
    # The cut is static: the passes before it, the one it cuts, alone, and those after it.
    segments = ((scanned,) if cut is None or cut not in scanned else
                (range(scanned.start, cut), range(cut, cut + 1), range(cut + 1, scanned.stop)))
    for steps in segments:
        if steps:
            x, bank = later_passes(x, bank, steps)
    if shared is not None and loop.steps > 1:
        x, bank = unrolled(x, bank, loop.steps - 1, shared)
    if loop.reinject:
        x = model.norm(x)
    if end < layers:
        x = model.stack(x, span=(end, layers), train=train, decode=decode, **inputs)
    if loop.exit_threshold is None or not exited or model.is_mutable_collection(EXITS):
        return x
    return _exited(exited, x, loop.exit_threshold)


def _sown_exit(model, leaving):
    """A pass's exit state `leaving` as the last pass's leaves the stack (`exit_states`), and its exit-gate
    logit under a gate (None without one), sown into `exits` beside the states the head reads of it where
    the caller opens the collection."""
    output = model.exit_states(leaving)
    logit = model.early_exit_gate(output) if model.loop.exit_gate else None
    if not model.is_initializing():
        if logit is not None:
            model.sow(EXITS, 'logits', logit)
        model.sow(EXITS, 'states', model.head_states(output))
    return output, logit


def _exited(exited, last, threshold: float):
    """Each token's exiting pass's state (`exit_choice`), from the earlier passes' `(state, gate logit)`
    and the last pass's state, which a token none of them exits takes."""
    choice = exit_choice([logit for _, logit in exited], threshold)
    leaving = jnp.stack([state for state, _ in exited] + [last])
    return jnp.take_along_axis(leaving, choice[None, ..., None], axis=0)[0]
