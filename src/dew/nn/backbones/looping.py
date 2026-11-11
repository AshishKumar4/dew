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
intermediates, extends the step's sows pass by pass, as the unrolled passes' did. The passes before
the last `backprop_steps` pass no gradient back: the cut is static, the passes before it scanned, the
pass it cuts traced on its own and those after it scanned, so no pass before it keeps residuals or runs
a backward. Where every pass exits (`Loop.exiting`: an exit gate, or `exits`) each pass's exit state
leaves as the last pass's does, the coda run on its output under its own cache entries, so the later
passes are traced one by one.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core.scope import DenyList

from ..precision import at_least_fp32
from ..protocols import EXITS
from .decoder_stack import WRITTEN, Loop, PassView, SharedPassView
from .layer_names import group_layers


def _preamble(model, loop: Loop, x, injected, *, stop: bool, normed=None):
    """A later pass's input from the state the pass before left: normed under `step_norm` (by `normed`,
    the norm the exit gate already read, where it read this state with no coda between), cut from the
    gradient where `stop` says, the block's input added back under `reinject`, or the block's input moved
    toward the state by θ under `implicit` (`a + θ⊙(h - a)`, `a` the block's input, θ
    `model.loop_theta`)."""
    if loop.step_norm:
        x = model.norm(x) if normed is None else normed
    x = jax.lax.stop_gradient(x) if stop else x
    if loop.reinject:
        x = model.norm(x + injected)
    if loop.implicit:
        x = injected + model.loop_theta.astype(x.dtype) * (x - injected)
    return x


def exit_log_distribution(logits: Sequence[jax.Array]) -> jax.Array:
    """The log of each token's probability of exiting at each pass `[passes, B, S]`, from every pass's
    exit-gate logit `[B, S, 1]`: a pass's its hazard `sigmoid(logit)` times the chance no pass before it
    exited, the last pass's that chance after them all, its own logit read for nothing (Ouro's exit pdf,
    OuroForCausalLM.forward). Summed from the logits' log-sigmoids in at least fp32, so a hazard that
    rounds to one leaves the passes after it a finite log probability, whose gradient stays finite,
    where the probability's own log would be log 0; a loop of one pass exits there surely."""
    hazards = jnp.stack(logits)[:-1, ..., 0].astype(at_least_fp32(logits[0].dtype))
    kept = jnp.cumsum(jax.nn.log_sigmoid(-hazards), axis=0)
    before = jnp.concatenate([jnp.zeros_like(logits[0][None, ..., 0], kept.dtype), kept])
    return before + jnp.concatenate([jax.nn.log_sigmoid(hazards), jnp.zeros_like(before[:1])])


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
    a pass's layer entries under the names its view (`PassView`) stores them by; under `shared` every
    pass's entries are the block's own, its sows extending them in pass order, and the cache is the
    scan's carry, which comes back dropped. What a pass sows elsewhere (`appended`: captured
    intermediates, a name of no layer) extends the tuples it sows into at each leaf, pass by pass, as
    the unrolled passes' sows did. Flax calls `inside` before `outside` within one trace, which holding
    the tree relies on."""

    views: Sequence[PassView]
    shared: bool = False
    appended: frozenset[str] = frozenset()
    held: dict = dataclasses.field(default_factory=dict)

    def _per_pass(self, collection: str) -> bool:
        """Whether `collection` keeps each pass's layer entries apart, under the pass's own names."""
        return collection in WRITTEN and not self.shared

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
                out[collection] = _merged(out.get(collection, {}), entries)
        return out


def _merged(held: Mapping, sown: Mapping) -> dict:
    """A collection's `held` entries with a pass's `sown` ones appended (`_appended`)."""
    return {**held, **{name: _appended(held.get(name), value) for name, value in sown.items()}}


def _appended(held, sown):
    """`held` with `sown` appended at each leaf: a sown tuple extends the held one, a mapping merges
    key by key, and any other value replaces the held one, as a sow's reduction does."""
    if isinstance(sown, Mapping):
        merged: dict[str, object] = dict(held or {})
        for name, value in sown.items():
            merged[name] = _appended(merged.get(name), value)
        return merged
    if isinstance(sown, tuple):
        return (*(held or ()), *sown)
    return sown


def run_loop(model, x, *, train: bool, decode: bool, **inputs):
    """Run `model`'s layer stack over `x` once, or as its `loop` says (the module doc): the layers before
    its block once, the block `steps` times, each later pass reading the last one's output, normed under
    `step_norm`, and the layers after it once; the passes before the last `backprop_steps` pass no gradient
    back, through their exits neither. `remat` checkpoints each block pass outside decode. Where every pass
    exits (`Loop.exiting`) each before the last sows its head's states (and gate logit) into `exits`; with
    `exit_threshold` the output is each token's exiting pass's (`exit_choice`) unless the caller collects
    the exits, whose last pass is the model's output."""
    loop = model.loop
    if loop is None:
        return model.stack(x, train=train, decode=decode, **inputs)
    layers = model.num_layers
    first, end = loop.layers or (0, layers)
    block = None if loop.layers is None else (first, end)

    def pass_block(module, x, **inputs):
        return module.stack(x, span=block, train=train, decode=decode, **inputs)

    def pass_coda(module, x):
        return module.stack(x, span=(end, layers), train=train, decode=decode, **inputs)

    run_block = nn.remat(pass_block) if loop.remat and not decode else pass_block
    if first:
        x = model.stack(x, span=(0, first), train=train, decode=decode, **inputs)
    injected = x
    shared = SharedPassView(block or (0, layers)) if loop.shared_cache else None
    first_pass = run_block if shared is None else nn.map_variables(
        run_block, mapped_collections=WRITTEN, trans_in_fn=shared.first, trans_out_fn=shared.outside,
        mutable=True)
    x = first_pass(model, model.norm(x + injected) if loop.reinject else x, **inputs)
    cut = None if loop.backprop_steps is None else loop.steps - loop.backprop_steps
    exited: list[tuple[jax.Array, jax.Array]] = []

    def exit_state(x, step: int):
        """Pass `step`'s exit state (from 0): its output, or the coda run on it in the pass's own cache."""
        if end == layers:
            return x
        # After the block's later passes' own entries, which a shared cache's passes do not keep apart.
        later = 0 if shared is not None else (loop.steps - 1) * (end - first)
        view = PassView(layers + later + step * (layers - end), end, layers - end)
        coda = nn.map_variables(pass_coda, mapped_collections=WRITTEN, trans_in_fn=view.inside,
                                trans_out_fn=view.outside, mutable=True)
        return coda(model, x)

    def unrolled(x, step: int, view):
        """A later pass traced alone over `view`'s collections, the one before it exiting first if all do."""
        normed = None
        if loop.exiting:
            # A pass the cut holds back exits without passing its own gradient back.
            leaving = exit_state(jax.lax.stop_gradient(x) if cut is not None and step <= cut else x, step - 1)
            output, logit = _sown_exit(model, leaving)
            if logit is not None:  # a gated exit, which the threshold's choice weighs (`_exited`)
                exited.append((leaving, logit))
            normed = output if end == layers and model.normed_output else None
        x = _preamble(model, loop, x, injected, stop=step == cut, normed=normed)
        run = nn.map_variables(run_block, mapped_collections=WRITTEN, trans_in_fn=view.inside,
                               trans_out_fn=view.outside, mutable=True)
        return _stepped(loop, run(model, x, **inputs), x, injected)

    def own_view(step: int):
        return shared or PassView(layers + (step - 1) * (end - first), first, end - first)

    def later_passes(x, steps: range):
        """The later passes `steps` in order, none of them cut: one traced on its own, more under one scan,
        or each on its own where every pass exits."""
        if len(steps) == 1 or loop.exiting:
            # One pass is its own program: a scan of one would only stack and unstack its collections.
            for step in steps:
                x = unrolled(x, step, own_view(step))
            return x
        views = [PassView(layers + (step - 1) * (end - first), first, end - first) for step in steps]
        # The collections a pass sows into beside those a stack writes, which an unrolled pass appended
        # to: those the first pass made, and captured intermediates, which a capture of a module only the
        # later passes call first makes inside the scan (the final norm `_preamble` reads by).
        appended = frozenset(collection for collection in (*model.variables, 'intermediates')
                             if model.is_mutable_collection(collection)) - {*WRITTEN, 'params', 'constants'}
        view = PassStack(views, shared=shared is not None, appended=appended)
        mapped = (*WRITTEN, *sorted(appended))
        # Each shared pass reads the step's entering cache, else, where the step allocates it, the
        # cache the first pass just allocated, whose later writes the scan's dropped carry takes.
        entering = None if shared is None else (shared.entering if shared.entering is not None
                                               else model.variables.get('cache') or None)

        def later(module, x):
            # The block's input, its per-layer inputs and the entering cache close over the scan.
            x = _preamble(module, loop, x, injected, stop=False)
            if entering is None:
                return _stepped(loop, run_block(module, x, **inputs), x, injected), None
            reading = nn.map_variables(run_block, mapped_collections='cache',
                                       trans_in_fn=lambda variables: {**variables, 'cache': entering},
                                       mutable=True)
            return _stepped(loop, reading(module, x, **inputs), x, injected), None

        stacked = tuple(collection for collection in mapped if shared is None or collection != 'cache')
        scan = nn.scan(later, variable_axes={stacked: 0},
                       variable_carry='cache' if shared is not None else False,
                       variable_broadcast=DenyList(mapped), split_rngs={'params': False, 'dropout': True},
                       length=len(steps), metadata_params={nn.PARTITION_NAME: None})
        run = nn.map_variables(scan, mapped_collections=mapped, trans_in_fn=view.inside,
                               trans_out_fn=view.outside, mutable=True)
        x, _ = run(model, x)
        return x

    # Under a shared cache the last pass keeps its writes and runs as the first does, after the scan.
    scanned = range(1, loop.steps - 1 if shared is not None else loop.steps)
    # The cut is static: the passes before it, the one it cuts, alone, and those after it.
    segments = ((scanned,) if cut is None or cut not in scanned else
                (range(scanned.start, cut), range(cut, cut + 1), range(cut + 1, scanned.stop)))
    for steps in segments:
        if steps:
            x = later_passes(x, steps)
    if shared is not None and loop.steps > 1:
        x = unrolled(x, loop.steps - 1, shared)
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
