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
the last `backprop_steps` pass no gradient back, a cut each scanned pass's input takes where its index
says so.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core.scope import DenyList

from .decoder_stack import WRITTEN, Loop, PassView, SharedPassView


def _preamble(model, loop: Loop, x, stop, injected):
    """A later pass's input from the state the pass before left: normed under `step_norm`, cut from the
    gradient where `stop` says, the block's input added back under `reinject`; and the exit gate's logit
    for the normed state (None without the gate)."""
    normed = model.norm(x)
    logit = model.early_exit_gate(normed) if loop.exit_gate else None
    x = normed if loop.step_norm else x
    x = jnp.where(stop, jax.lax.stop_gradient(x), x) if isinstance(stop, jax.Array) else (
        jax.lax.stop_gradient(x) if stop else x)
    if loop.reinject:
        x = model.norm(x + injected)
    return x, logit


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
                named = [{name: value for name, value in entries.items() if _is_layer(name)}
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
                         for name, value in entries.items() if _is_layer(name)})
                    entries = {name: value for name, value in entries.items() if not _is_layer(name)}
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


def _is_layer(name: str) -> bool:
    return name.startswith('layers_')


def run_loop(model, x, *, train: bool, decode: bool, **inputs):
    """Run `model`'s layer stack over `x` once, or as its `loop` says (the module doc): the layers before
    its block once, the block `steps` times, each pass after the first reading the last one's output,
    normed under `step_norm`, and the layers after it once; the passes before the last
    `backprop_steps` pass no gradient back. `remat` checkpoints each block pass outside decode."""
    loop = model.loop
    if loop is None:
        return model.stack(x, train=train, decode=decode, **inputs)
    first, end = loop.layers or (0, model.num_layers)
    block = None if loop.layers is None else (first, end)

    def pass_block(module, x, **inputs):
        return module.stack(x, span=block, train=train, decode=decode, **inputs)

    run_block = nn.remat(pass_block) if loop.remat and not decode else pass_block
    if first:
        x = model.stack(x, span=(0, first), train=train, decode=decode, **inputs)
    injected = x
    shared = SharedPassView() if loop.shared_cache else None
    first_pass = run_block if shared is None else nn.map_variables(
        run_block, mapped_collections=WRITTEN, trans_in_fn=shared.first, trans_out_fn=shared.outside,
        mutable=True)
    x = first_pass(model, model.norm(x + injected) if loop.reinject else x, **inputs)
    cut = None if loop.backprop_steps is None else loop.steps - loop.backprop_steps

    def unrolled(x, step: int, view):
        """One later pass traced on its own, over the collections as `view` shows them."""
        x, logit = _preamble(model, loop, x, step == cut, injected)
        if logit is not None and not model.is_initializing():
            model.sow('exits', 'logits', logit)
        run = nn.map_variables(run_block, mapped_collections=WRITTEN, trans_in_fn=view.inside,
                               trans_out_fn=view.outside, mutable=True)
        return run(model, x, **inputs)

    # Under a shared cache the last pass keeps its writes and runs as the first does, after the scan.
    scanned = range(1, loop.steps - 1 if shared is not None else loop.steps)
    if len(scanned) == 1:
        # One pass is its own program: a scan of one would only stack and unstack its collections.
        (step,) = scanned
        x = unrolled(x, step, shared or PassView(model.num_layers, first, end - first))
    elif len(scanned) > 1:
        views = [PassView(model.num_layers + (step - 1) * (end - first), first, end - first)
                 for step in scanned]
        # The collections a pass sows into beside those a stack writes: captured intermediates, among
        # others, which an unrolled pass appended to.
        appended = frozenset(collection for collection in model.variables
                             if model.is_mutable_collection(collection)) - {*WRITTEN, 'params', 'constants'}
        view = PassStack(views, shared=shared is not None, appended=appended)
        mapped = (*WRITTEN, *sorted(appended))
        # Each shared pass reads the step's entering cache, else, where the step allocates it, the
        # cache the first pass just allocated, whose later writes the scan's dropped carry takes.
        entering = None if shared is None else (shared.entering if shared.entering is not None
                                               else model.variables.get('cache') or None)

        def later(module, x, stop):
            # The block's input, its per-layer inputs and the entering cache close over the scan.
            x, logit = _preamble(module, loop, x, stop, injected)
            if entering is None:
                return run_block(module, x, **inputs), logit
            reading = nn.map_variables(run_block, mapped_collections='cache',
                                       trans_in_fn=lambda variables: {**variables, 'cache': entering},
                                       mutable=True)
            return reading(module, x, **inputs), logit

        stacked = tuple(collection for collection in mapped if shared is None or collection != 'cache')
        scan = nn.scan(later, variable_axes={stacked: 0},
                       variable_carry='cache' if shared is not None else False,
                       variable_broadcast=DenyList(mapped), split_rngs={'params': False, 'dropout': True},
                       length=len(scanned))
        run = nn.map_variables(scan, mapped_collections=mapped, trans_in_fn=view.inside,
                               trans_out_fn=view.outside, mutable=True)
        stops = jnp.asarray([step == cut for step in scanned])
        x, logits = run(model, x, stops)
        if logits is not None and not model.is_initializing():
            for logit in logits:
                model.sow('exits', 'logits', logit)
    if shared is not None and loop.steps > 1:
        x = unrolled(x, loop.steps - 1, shared)
    if loop.reinject:
        x = model.norm(x)
    if end < model.num_layers:
        x = model.stack(x, span=(end, model.num_layers), train=train, decode=decode, **inputs)
    return x

