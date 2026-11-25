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
entries: inside the scan, before the pass it precedes (`ExitingPass`), its states the scan's outputs and
sown after it in pass order (`_sown_exits`); a decode's exits write their codas' caches pass by pass, so
its later passes are traced one by one.

A count of passes given per call (`run_loop`'s `passes`, a count drawn per microbatch) runs under one scan
of every pass the loop's `steps` allow, each pass a branch of `lax.cond` (`lax.switch` under a cut): run
where its step is below the count, skipped otherwise, and before the cut the count's last passes leave,
run passing no gradient back; one program for every count, each skipped pass costing nothing.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Collection, Mapping, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import lift, meta
from flax.core.scope import DenyList
from flax.linen.transforms import lift_direct_transform
from jax.sharding import NamedSharding

from dew.batch import sows_appended, unboxed
from dew.mesh import FSDP_AXIS, mesh_axes
from dew.nn.dense import captures_any_method

from ..precision import at_least_fp32, softplus
from ..protocols import EXIT_PASSES, EXITS, fixed_remat
from ..sharding import explicit_over, gathered_over, placed_layout, typed_on
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
    x = _stopped(x, stop)
    if loop.reinject:
        x = model.norm(x + injected)
    if theta is not None:
        x = injected + theta.astype(x.dtype) * (x - injected)
    return x


def _stopped(value, stop):
    """`value` cut from the gradient where `stop` says: a static bool, or a traced one (a count of passes
    given per call, whose cut moves with it, `_Passes.counted_step`), which the gradient passes through
    where false."""
    if isinstance(stop, bool):
        return jax.tree.map(jax.lax.stop_gradient, value) if stop else value
    return jax.tree.map(lambda leaf: jnp.where(stop, jax.lax.stop_gradient(leaf), leaf), value)


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


NEVER_EXITS = -1e4
"""The exit-gate logit a pass a call's count left unrun exits by (`never_exiting`): its chance of exiting,
`sigmoid` of it, rounds to zero in every float a gate computes in, and so does its share of every later
pass's log survival, so its loss and entropy terms weigh nothing and stay finite."""


def never_exiting(logits: Sequence[jax.Array], ran: Sequence[jax.Array | bool]) -> tuple[jax.Array, ...]:
    """Every pass's exit-gate `logits` (`Looped.exit_passes`), each earlier pass's at `NEVER_EXITS` where it
    did not run (`ran`, a count of passes given per call, `run_loop`'s `passes`): its share the last
    pass's, the pass the count stops at. A gate's last logit, the stack's own output's, is that pass's,
    which always runs."""
    if not logits or all(each is True for each in ran):
        return tuple(logits)
    if len(logits) - len(ran) not in (0, 1):
        raise ValueError(f"{len(logits)} exit-gate logits for {len(ran)} passes' exits")
    return (*(jnp.where(valid, logit, jnp.asarray(NEVER_EXITS, logit.dtype))
              for logit, valid in zip(logits[:len(ran)], ran, strict=True)), *logits[len(ran):])


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
    (`exit_log_distribution`) reaches it (`reached`), the last pass (`len(logits)`) where none before it does
    (OuroForCausalLM.forward's `exit_threshold`). Each pass's reach is read from its own and the earlier
    passes' logits alone, as a loop stopping its passes reads it (`_Passes.stopping`)."""
    choice = jnp.full(logits[0].shape[:-1], len(logits))
    for index in reversed(range(len(logits))):
        choice = jnp.where(reached(logits[:index + 1], threshold), index, choice)
    return choice


def reached(logits: Sequence[jax.Array], threshold: float) -> jax.Array:
    """Whether each token's cumulative exit probability through the last pass of `logits` (each earlier
    pass's and its own exit-gate logit) reaches `threshold`: its chance of exiting at one of them
    (`exit_log_distribution`, the pass after them read for nothing), summed."""
    chances = jnp.exp(exit_log_distribution([*logits, logits[-1]]))
    return jnp.sum(chances[:-1], axis=0) >= threshold


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

    views: Sequence[PassView | ExitingPass]
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
                        {view.stored(name): value
                         for name, value in entries.items() if group_layers(name) is not None})
                    entries = {name: value for name, value in entries.items() if group_layers(name) is None}
                out[collection] = sows_appended(out.get(collection, {}), entries)
        return out


@dataclasses.dataclass
class ExitingPass:
    """A later pass of a loop whose every pass exits (`Loop.exiting`), as one step of the scan over the
    passes sees what the stack writes: the exit of the pass before it, the coda run under that exit's own
    entries (`exit_offset`, `coda`), and its block under its own (`block`), each shown under its layers'
    own names (`PassView`)."""

    block: PassView
    coda: PassView

    def inside(self, variables: Mapping[str, Mapping]) -> dict:
        block, coda = self.block.inside(variables), self.coda.inside(variables)
        return {collection: {**coda[collection], **block[collection]} for collection in block}

    def stored(self, name: str) -> str:
        """The name the step's entry `name` is stored by: its block's or its coda's, by its layer."""
        run = group_layers(name)
        own = self.coda if run is not None and run.start >= self.coda.first else self.block
        return own.stored(name)


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
    if loop.exiting and loop.exit_at == "coda" and end < layers:
        # A gate on the block's output runs a cached call's coda once, on the states it selects
        # (`Loop.exit_at`): no exit's coda keeps a cache.
        entries += [(exit_offset(loop, layers, step) + start - end, start, count)
                    for step in range(loop.steps - 1) for start, count in coda]
    return tuple(entries)


def _passed(module, x, bank, *, block: tuple[int, int] | None, train: bool, decode: bool, store=None,
            **inputs):
    """One pass of a loop's block over `x`, reading the bank the pass before left, and the bank it leaves
    with the store it leaves: under `Loop.messages` the block a message block at a time
    (`CrossPass.blocks`), each block's residual summary (`residual_summary`) banked on axis 2
    (`[B, S, blocks, D]`) and every layer reading `bank` (None in a first pass); else the block at once and
    no bank. The pass's layers read their providers' publications from a copy of `store` (the prelude's,
    which every pass enters with) and leave their own in it, one store across its message blocks, so a
    layer reads its provider's own pass's publication or the prelude's, never another pass's (row 12929)."""
    messages = module.loop.messages
    store = None if store is None else dict(store)
    if messages is None:
        return module.stack(x, span=block, train=train, decode=decode, kv_store=store, **inputs), None, store
    first, end = block or (0, module.num_layers)
    summaries = []
    for span in messages.blocks(first, end):
        x = module.stack(x, span=span, train=train, decode=decode, prior_bank=bank, kv_store=store, **inputs)
        summaries.append(module.residual_summary(x))
    return x, jnp.stack(summaries, axis=2), store


def _exit_state(model, x, step: int, *, train: bool, decode: bool, store=None, **inputs):
    """A loop's pass `step`'s exit state (from 0): its output, or the coda run on it in the pass's own
    cache (`exit_offset`), its layers reading the pass's publications (`store`, a copy)."""
    loop, layers = model.loop, model.num_layers
    _, end = loop.layers or (0, layers)
    if end == layers:
        return x
    view = PassView(exit_offset(loop, layers, step), end, layers - end)

    def pass_coda(module, x):
        return module.stack(x, span=(end, layers), train=train, decode=decode,
                            kv_store=None if store is None else dict(store), **inputs)

    coda = nn.map_variables(pass_coda, mapped_collections=WRITTEN, trans_in_fn=view.inside,
                            trans_out_fn=view.outside, mutable=True)
    return coda(model, x)


def _coda_reads_block(model) -> bool:
    """Whether the layers after a loop's block may read what a layer of the block publishes (`kv_store`:
    keys and values a sharing layer reads, a publishing mixer's): a provider in the block and a coda after
    it. Its coda then reads the last pass's publications, as the unlooped stack's reads its provider's
    (`_Passes.fixed` traces that pass unrolled)."""
    first, end = model.loop.layers or (0, model.num_layers)
    return end < model.num_layers and any(spec.provider is not None for spec in model.specs[first:end])


def _entered(model, x, bank, injected, *, stop: bool, normed=None, theta=None):
    """A loop's later pass's input and the bank it reads: under `messages` the block's input and the bank
    the pass before left, which the cut stops (`_passed`); else the pass before's state (`_preamble`)."""
    loop = model.loop
    if loop.messages is not None:
        return injected, _stopped(bank, stop)
    return _preamble(model, loop, x, injected, stop=stop, normed=normed, theta=theta), None


def run_loop(model, x, *, train: bool, decode: bool, passes=None, **inputs):
    """Run `model`'s layer stack over `x` once, or as its `loop` says (the module doc): the layers before
    its block once, the block `steps` times, each later pass reading the last one's output, normed under
    `step_norm`, and the layers after it once; the passes before the last `backprop_steps` pass no gradient
    back, through their exits neither. `remat` checkpoints each block pass outside decode. Where every pass
    exits (`Loop.exiting`) each before the last sows its head's states (and gate logit) into `exits`; with
    `exit_threshold` the output is each token's exiting pass's (`exit_choice`) unless the caller collects
    the exits, whose last pass is the model's output. Under `messages` every pass enters the block at its
    input and each later pass reads the bank of block summaries the pass before left (`_passed`).

    `passes`, a traced int32 scalar from 1 to `steps` the same on every device (a count drawn per
    microbatch, `dew.training.phase_passes.SampledPasses`), runs that many passes of the one program the
    loop's `steps` compile (`_Passes.counted`); None runs every pass."""
    loop = model.loop
    if loop is None and passes is not None:
        raise ValueError("a count of passes is given a model that loops none of its layers")
    if loop is None:
        return model.stack(x, train=train, decode=decode, **inputs)
    if passes is not None and (decode or loop.shared_cache or loop.exit_threshold is not None):
        raise ValueError("a loop runs a count of passes given per call in training alone: not in decode, "
                         "under a shared cache or an exit threshold, which choose their passes themselves")
    if passes is not None and captures_any_method(model):
        raise ValueError("a count of passes given per call runs every pass under one scan, which stacks no "
                         "method's non-array return: capture the modules' calls (capture_intermediates=True)")
    layers, (first, end) = model.num_layers, loop.layers or (0, model.num_layers)
    # The prelude's publications (`kv_store`), which every pass enters with and the coda reads beside the
    # last pass's own (`_passed`): a provider's lifetime its part's and every part after it.
    prelude = {} if model.provides else None
    if first:
        x = model.stack(x, span=(0, first), train=train, decode=decode, kv_store=prelude, **inputs)
    if passes is not None and _coda_reads_block(model):
        raise ValueError("a count of passes given per call runs its passes under one scan, which leaves no "
                         "publication of its last pass for the layers after the block, which read one")

    def every_pass(module, x, inputs, passes, prelude):
        # The block's passes, first and later, what each exits, and the publications the last leaves.
        run = _Passes(module, train=train, decode=decode, inputs=inputs, prelude=prelude)
        run.injected = x
        first_pass = run.run_block if run.shared is None else nn.map_variables(
            run.run_block, mapped_collections=WRITTEN, trans_in_fn=run.shared.first,
            trans_out_fn=run.shared.outside, mutable=True)
        x, bank, run.left = first_pass(module, module.norm(x + run.injected) if loop.reinject else x,
                                       **inputs)
        x, _ = run.counted(x, bank, passes) if passes is not None else run.fixed(x, bank)
        return x, run.exited, run.left

    layout = placed_layout()
    # The layers the passes run: the block's, and under `exiting`, where every pass runs the coda, the coda's.
    read = (first, layers if loop.exiting else end)
    # A decode (a server's step) and a store banked by run over any layer the passes run (each run's bank read
    # by the scan the gathered passes do not run) read the block where it lies.
    if (loop.gathered and not decode and not model.is_initializing() and layout is not None
            and jax.sharding.get_abstract_mesh().shape.get(FSDP_AXIS, 1) > 1
            and not model.banked_collections(read)):
        gathered = functools.partial(_gathered_params, layout=layout, mesh=jax.sharding.get_abstract_mesh(),
                                     scope=('params', *model.scope.path), span=read)
        x, exited, left = _gathered_passes(model, every_pass, x, inputs, passes, prelude, gathered)
    else:
        x, exited, left = every_pass(model, x, inputs, passes, prelude)
    if loop.reinject:
        x = model.norm(x)
    chosen = loop.exit_threshold is not None and not model.is_mutable_collection(EXITS)
    if chosen and loop.exit_at == "block":
        # A gate on the block's output: each row's exiting pass's block output, the coda run once on them.
        x = _handed_on(model, exited, x, loop.exit_threshold)
    if end < layers:
        x = model.stack(x, span=(end, layers), train=train, decode=decode,
                        kv_store=None if left is None else dict(left), **inputs)
    if not chosen or loop.exit_at == "block":
        return x
    return _handed_on(model, exited, x, loop.exit_threshold)


def _gathered_passes(model, every_pass: Callable, x, inputs: dict, passes, prelude, gathered: Callable):
    """`every_pass(model, x, inputs)` traced with every Auto axis Explicit (`sharding.explicit_over`),
    every parameter the passes read gathered whole once before the first over the FSDP axis and every axis
    the stream's tokens are split over (a hybrid run's data axis, a sequence axis), a matrix reduced over
    them (`gathered`, `_gathered_params`), so each pass's gradient of it stays unreduced, the passes' sum
    of it reduced once after the loop (`Loop.gathered`): a product's gradient sums over every axis its
    tokens split. `inputs`' arrays, a count of `passes` given per call and the `prelude`'s publications
    held whole on every device, and what the passes write (their sown and cached collections) placed back
    on the mesh the step traces (`sharding.typed_on`). The passes' operations are typed on the mesh: a
    projection takes no launch mapped per shard there (`projections.rows_split`), the partitioner's product
    running it."""
    from dew.nn.kernels import generation

    whole = jax.sharding.PartitionSpec()
    mesh = jax.sharding.get_abstract_mesh()

    def held(module, x):
        # What the passes read from outside the region placed on its mesh: the inputs, a count of passes and
        # the prelude's publications.
        inside = jax.tree.map(lambda value: jax.sharding.reshard(value, whole)
                              if isinstance(value, jax.Array) else value, (inputs, passes, prelude))
        return every_pass(module, x, *inside)

    def run_inside(x):
        token_axes = {name for entry in jax.typeof(x).sharding.spec.partitions[:x.ndim - 1]
                      for name in mesh_axes(entry)}
        mapped = nn.map_variables(
            held, mapped_collections=True, init=False, mutable=True,
            trans_in_fn=functools.partial(gathered, axes=token_axes | {FSDP_AXIS}),
            trans_out_fn=lambda variables: {
                name: tree if name == 'params' else typed_on(tree, mesh)
                for name, tree in variables.items()})
        return mapped(model, x)
    # The block's layers run one by one in each pass: a scan over a run of them reads its layers' stacked
    # parameters as the scan's inputs, which Flax's lifted scan places by their spec's dimensions alone
    # (`flax.core.axes_scan.build_shaped_array`), refusing a reduced one.
    with generation.chosen(layer_scan="plain"):
        return explicit_over(run_inside, x)


def _gathered_params(variables: Mapping[str, Mapping], *, layout, mesh, scope: tuple[str, ...],
                     span: tuple[int, int], axes: Collection[str]) -> dict:
    """`variables`' `params` (a decoder's, under its variables `scope`, `('params', ...)`) with each leaf a
    gathered loop's passes read gathered whole over `axes` from where `layout` places it on the step's
    `mesh` (`sharding.gathered_over`, `Layout.leaf_spec`): a layer's within the block (`span`, `[first,
    end)`), met by any module name running it, its own (`layers_3`) or a run's (`layers_1_2`,
    `layer_names.group_layers`; the coda's too where every pass exits through it, `Loop.exiting`), and the
    decoder's own, no layer's (its step and reinjection norms, and the final norm and head an exiting pass
    reads); every other (the prelude's layers, the coda's where only the last pass runs it) as it is, which
    no pass reads. Each
    leaves its box (`flax.core.meta`'s) unconstrained (`dew.batch.unboxed`): its logical axes would place it
    by a constraint, which an Explicit axis takes none of, and a gathered one's back on the FSDP axis."""
    first, end = span

    def leaf(where, value):
        keys = tuple(jax.tree_util.DictKey(key) for key in scope) + tuple(where)
        runs = group_layers(where[0].key) if isinstance(where[0], jax.tree_util.DictKey) else None
        raw = unboxed(value)
        held = runs is None or (first <= runs.start and runs.stop <= end)
        return gathered_over(raw, layout.leaf_spec(mesh, keys, raw), axes) if held else raw

    params = jax.tree_util.tree_map_with_path(leaf, variables.get('params', {}),
                                              is_leaf=lambda value: isinstance(value, meta.AxisMetadata))
    return {**variables, 'params': params}


@dataclasses.dataclass
class _Passes:
    """A loop's later passes over a model's stack (`run_loop`): traced one by one (`unrolled`), under one
    scan (`scanned`), or under one scan of every pass the loop's `steps` allow, each pass run where a count
    given per call reaches it (`counted`). What each exits (`Loop.exiting`) leaves as the scan's outputs,
    sown after it in pass order (`_sown_exits`), and kept for an exit threshold's choice (`exited`)."""

    model: nn.Module
    train: bool
    decode: bool
    inputs: dict
    prelude: dict | None = None
    """The prelude's publications every pass enters with (`_passed`), None where no layer publishes."""
    injected: jax.Array | None = None
    exited: list = dataclasses.field(default_factory=list)
    left: dict | None = None
    """The publications the last pass traced unrolled left, its exit's and the coda's to read."""
    span: tuple[int, int] = dataclasses.field(init=False)
    shared: SharedPassView | None = dataclasses.field(init=False)
    """Under `shared_cache`, the view every pass's cache is read through: the first's, the scan's and the
    last's alike (`SharedPassView`)."""
    run_block: Callable = dataclasses.field(init=False)
    """One pass of the block (`_passed`), checkpointed under `remat` outside decode."""

    exit_outputs: bool = dataclasses.field(init=False)
    """Whether a scanned pass's exit leaves its normed output: where the exits are sown or chosen by a
    threshold, or the gate's parameters made (`_sown_exits`)."""
    exit_states: bool = dataclasses.field(init=False)
    """Whether a scanned pass's exit leaves its exit state: where a threshold chooses among them
    (`_handed_on`), the only reader of the state itself."""
    gates_block: bool = dataclasses.field(init=False)
    """Whether the gate reads each pass's block output, normed, not its coda's (`Loop.exit_at='block'`)."""
    stops: bool = dataclasses.field(init=False)
    """Whether the passes stop where every real token has exited (`stopping`)."""
    compacts: bool = dataclasses.field(init=False)
    """Whether a stopping pass runs over the rows still looping alone (`compacted`)."""

    def __post_init__(self):
        model, loop = self.model, self.loop
        self.span = loop.layers or (0, model.num_layers)
        self.shared = SharedPassView(self.span) if loop.shared_cache else None
        # A pass recomputed whole whatever the rung (`fixed_remat`).
        self.run_block = (nn.remat(fixed_remat(self.pass_block)) if loop.remat and not self.decode
                          else self.pass_block)
        self.exit_states = (loop.exiting and loop.exit_threshold is not None
                            and not model.is_mutable_collection(EXITS))
        self.gates_block = loop.exit_gate and loop.exit_at == "block"
        # Inference under a threshold stops its passes where every real token has exited: scoring, and a
        # cached call whose passes share a cache and whose gate reads the block, its rows' caches their
        # exiting passes' (`stopping`); over per-pass caches every pass runs (`Loop` doc).
        self.stops = (self.exit_states and not self.train and loop.steps > 1
                      and (not self.decode or (loop.shared_cache and self.gates_block)))
        # Scoring exiting by row runs a pass over its rows still looping alone, where nothing it writes beside
        # its state (a collection) holds a row of each (`compacted`).
        self.compacts = (self.stops and not self.decode and loop.exit_by == "row"
                         and model.scope is not None and model.scope.mutable is False)
        # A threshold reading a gate on the block's output runs no pass's coda (`Loop.exit_at`).
        self.exit_outputs = loop.exiting and (model.is_mutable_collection(EXITS) or model.is_initializing()
                                              or (self.exit_states and not self.gates_block))

    @property
    def loop(self) -> Loop:
        return self.model.loop

    @property
    def valid(self) -> jax.Array | None:
        """The call's real tokens (`AttentionMetadata.valid`), None where every token is."""
        metadata = self.inputs.get('attention_metadata')
        return None if metadata is None else metadata.valid

    def exit_of(self, module, x, step: int | None, *, stop: bool):
        """Pass `step`'s exit (from 0; None inside a scan, whose view its coda writes through) from its block
        output `x`, stopped where `stop`: the state a threshold selects of it (the coda's, or under
        `exit_at='block'` the block's own), its normed exit state where a head or the gate reads it (the coda
        run on `x` in the pass's own cache, `_exit_state`), and the normed block output the gate reads under
        `exit_at='block'` (None otherwise)."""
        loop = self.loop
        x = _stopped(x, stop)
        gated = module.exit_states(x) if self.gates_block else None
        leaving = output = None
        if self.exit_outputs or loop.exit_at == "coda":
            leaving = (self.coda(module, x) if step is None else
                       _exit_state(module, x, step, train=self.train, decode=self.decode, store=self.left,
                                   **self.inputs))
            output = module.exit_states(leaving)
        return (x if self.gates_block else leaving), output, gated

    def sown(self, module, output, gated):
        """An unrolled exit's gate logit (`gate_logit`, of `gated` where the gate reads the block, else of
        `output`), sown into `exits` with the states a head reads of `output`, as a scan's are
        (`_sown_exits`); None without a gate."""
        logit = (gate_logit(module, output if gated is None else gated, self.valid)
                 if self.loop.exit_gate else None)
        if not module.is_initializing():
            if logit is not None:
                module.sow(EXITS, 'logits', logit)
            if output is not None:
                module.sow(EXITS, 'states', module.head_states(output))
        return logit

    def pass_block(self, module, x, bank=None, **inputs):
        block = None if self.loop.layers is None else self.span
        return _passed(module, x, bank, block=block, train=self.train, decode=self.decode,
                       store=self.prelude, **inputs)

    def entering(self, module, x, bank, *, stop: bool, normed=None, theta=None):
        return _entered(module, x, bank, self.injected, stop=stop, normed=normed, theta=theta)

    def coda(self, module, x):
        """A pass's exit state from its output `x`: the coda run on it, under the step's view, its layers
        reading the prelude's publications (a scan's pass, whose own a coda reading them unrolls:
        `_coda_reads_block`)."""
        _, end = self.span
        layers = self.model.num_layers
        if end == layers:
            return x
        return module.stack(x, span=(end, layers), train=self.train, decode=self.decode,
                            kv_store=None if self.prelude is None else dict(self.prelude), **self.inputs)

    def fixed(self, x, bank):
        """Every later pass, the cut static: the passes before it scanned, the one it cuts, alone, and
        those after it scanned; under a shared cache the last after the scan, as the first runs. A loop
        that may stop at its threshold (`stops`) runs `stopping` instead."""
        if self.stops:
            return self.stopping(x, bank)
        loop, shared = self.loop, self.shared
        # A coda reading the block's publications reads the last pass's, which it leaves traced unrolled, and
        # an exiting pass's coda its own pass's: each pass unrolled (`_coda_reads_block`).
        reads = _coda_reads_block(self.model)
        last_apart = shared is not None or reads
        cut = None if loop.backprop_steps is None else loop.steps - loop.backprop_steps
        scanned = range(1, loop.steps - 1 if last_apart else loop.steps)
        segments = ((scanned,) if cut is None or cut not in scanned else
                    (range(scanned.start, cut), range(cut, cut + 1), range(cut + 1, scanned.stop)))
        for steps in segments:
            if not steps:
                continue
            # A pass at or before the cut exits without passing its own gradient back.
            exit_stop, enter_stop = cut is not None and steps.stop - 1 <= cut, steps.start == cut
            if (len(steps) == 1 or (loop.exiting and (self.decode or reads))
                    or captures_any_method(self.model)):
                # One pass is its own program; a decode's exits write their codas' caches pass by pass; a
                # capture of any method keeps what no scan stacks (a method's non-array return).
                for step in steps:
                    x, bank = self.unrolled(x, bank, step, self.own_view(step), exit_stop=exit_stop,
                                            enter_stop=step == cut)
            else:
                x, bank = self.scanned(x, bank, steps, exit_stop=exit_stop, enter_stop=enter_stop)
        if last_apart and loop.steps > 1:
            # The last pass apart, cut as its segment of the scan would be.
            last = loop.steps - 1
            x, bank = self.unrolled(x, bank, last, self.own_view(last),
                                    exit_stop=cut is not None and last <= cut, enter_stop=last == cut)
        return x, bank

    def stopping(self, x, bank):
        """Every later pass of an inference under a threshold (`stops`), each run only where some real token
        of the call has not exited by the passes before it (`reached`, read as `exit_choice` reads it): the
        loop computes no pass its output does not read, its output the passes' run whole (row 12924). Each
        pass traced alone, its block and its exit each skipped where every real token has exited
        (`_sown_switch`); a skipped exit's logit never exits (`NEVER_EXITS`). Under `exit_by='row'` in
        scoring each pass runs over the rows still looping alone (`compacted`, row 12927); a cached call's
        rows keep their exiting passes' cache writes (`_kept_exits`, row 12928)."""
        model, loop = self.model, self.loop
        _, end = self.span
        real = jnp.ones(x.shape[:2], bool) if self.valid is None else self.valid.astype(bool)
        caches = [self.block_cache(model)] if self.decode else []
        selected, output, gated = self.exit_of(model, x, 0, stop=False)
        logits = [self.sown(model, output, gated)]
        self.exited.append((selected, logits[0]))
        exited = reached(logits, loop.exit_threshold)
        normed_output = end == model.num_layers and model.normed_output
        for step in range(1, loop.steps):
            done = jnp.all(exited | ~real)
            pass_block = functools.partial(self.stopped_pass, step=step,
                                           normed=output if normed_output else None, rows=exited[:, 0])
            (x, bank), self.left = _sown_switch((~done).astype(jnp.int32), (self.held_pass, pass_block),
                                                model, (x, bank))
            if self.decode:
                caches.append(self.block_cache(model))
            if step == loop.steps - 1:
                break
            held = (jnp.zeros_like(selected), None if output is None else jnp.zeros_like(output),
                    jnp.full(logits[0].shape, NEVER_EXITS, logits[0].dtype))
            selected, output, logit = _sown_switch(
                (~done).astype(jnp.int32), (lambda module, x, held=held: held,
                                            functools.partial(self.stopped_exit, step=step)), model, x)
            self.exited.append((selected, logit))
            logits.append(logit)
            exited = exited | reached(logits, loop.exit_threshold)
        if self.decode:
            self._kept_exits(model, caches, logits)
        return x, bank

    def held_pass(self, module, carry):
        """A pass `stopping` skips: its state and bank as they stand, the publications the last pass left."""
        return carry, self.left

    def stopped_pass(self, module, carry, *, step: int, normed, rows):
        """Pass `step` of `stopping` over its own collections (`own_view`), from the state and bank `carry`,
        the step norm reading the exit's `normed` output: over the rows still looping alone where it compacts
        them (`compacts`, `rows` the rows already exited), else over all."""
        def ran(module, carry, normed):
            x, bank = carry
            entered, bank = self.entering(module, x, bank, stop=False, normed=normed,
                                          theta=_theta(module, self.loop, range(step, step + 1)))
            view = self.own_view(step)
            run = nn.map_variables(self.run_block, mapped_collections=WRITTEN, trans_in_fn=view.inside,
                                   trans_out_fn=view.outside, mutable=True)
            out, bank, left = run(module, entered, bank, **self.inputs)
            return (_stepped(self.loop, out, entered, self.injected), bank), left
        if not self.compacts:
            return ran(module, carry, normed)
        return compacted(ran, module, carry, normed, rows, self)

    def stopped_exit(self, module, x, *, step: int):
        """Pass `step`'s exit in `stopping` (`exit_of`): its selected state, normed output and gate logit."""
        selected, output, gated = self.exit_of(module, x, step, stop=False)
        return selected, output, self.sown(module, output, gated)

    def block_cache(self, module) -> dict:
        """The block's entries of the cache as the pass that just ran left them (its layers', a run's among
        them: `group_layers`)."""
        first, end = self.span
        return {name: value for name, value in module.variables.get('cache', {}).items()
                if (run := group_layers(name)) is not None and first <= run.start and run.stop <= end}

    def _kept_exits(self, module, caches: list[dict], logits) -> None:
        """Each row's cache entries of the block as its exiting pass left them (`exit_choice`, one pass a
        row under `exit_by='row'`), of `caches`, each pass's (`block_cache`): a later call's tokens read each
        earlier row's exiting pass's keys, values and states. An entry holding no row axis (a pool of pages)
        keeps no row's pass, and is refused."""
        choice = exit_choice(logits, self.loop.exit_threshold)[:, 0]
        batch = choice.shape[0]

        def kept(*passes):
            last = passes[-1]
            if last.ndim == 0 or all(each is last for each in passes):
                return last
            if last.shape[0] != batch:
                raise ValueError("a cached call skipping its passes keeps each row's exiting pass's cache, "
                                 f"and an entry of shape {last.shape} holds no row of its {batch}: a pool of "
                                 "pages; serve a looped decoder's early exit over a dense cache")
            stacked = jnp.stack(passes)
            index = choice.reshape(1, batch, *(1,) * (last.ndim - 1))
            return jnp.take_along_axis(stacked, index, axis=0)[0]
        for name in caches[-1]:
            module.put_variable('cache', name, jax.tree.map(kept, *(cache[name] for cache in caches)))

    def own_view(self, step: int):
        first, end = self.span
        offset = pass_offset(self.loop, self.model.num_layers, step)
        return self.shared or PassView(offset, first, end - first)

    def unrolled(self, x, bank, step: int, view, *, exit_stop: bool, enter_stop: bool):
        """A later pass traced alone over `view`'s collections, the one before it exiting first if all do."""
        model, loop = self.model, self.loop
        _, end = self.span
        normed = None
        if loop.exiting:
            selected, output, gated = self.exit_of(model, x, step - 1, stop=exit_stop)
            logit = self.sown(model, output, gated)
            if logit is not None:  # a gated exit, which the threshold's choice weighs (`_exited`)
                self.exited.append((selected, logit))
            normed = output if end == model.num_layers and model.normed_output else None
        x, bank = self.entering(model, x, bank, stop=enter_stop, normed=normed,
                                theta=_theta(model, loop, range(step, step + 1)))
        run = nn.map_variables(self.run_block, mapped_collections=WRITTEN, trans_in_fn=view.inside,
                               trans_out_fn=view.outside, mutable=True)
        out, bank, self.left = run(model, x, bank, **self.inputs)
        return _stepped(loop, out, x, self.injected), bank

    def stepped(self, module, carry, theta, *, exit_stop, enter_stop, block=None):
        """One later pass inside the scan over its step's collections: the exit of the pass before it
        where every pass exits (its exit state and its normed output, the scan's outputs where read:
        `exit_states`, `exit_outputs`), then its block (`block`, `run_block` where None)."""
        x, bank = carry
        loop = self.loop
        _, end = self.span
        selected = output = gated = normed = None
        if loop.exiting:
            # A scan's pass's coda reads the prelude's publications (`coda`), a coda reading the block's
            # having unrolled the passes (`fixed`).
            selected, output, gated = self.exit_of(module, x, None, stop=exit_stop)
            normed = output if end == module.num_layers and module.normed_output else None
        x, bank = self.entering(module, x, bank, stop=enter_stop, normed=normed, theta=theta)
        out, bank, _ = (block or self.run_block)(module, x, bank, **self.inputs)
        exits = (selected if self.exit_states else None, output if self.exit_outputs else None, gated)
        return (_stepped(loop, out, x, self.injected), bank), exits

    def scanned(self, x, bank, steps: range, *, exit_stop: bool, enter_stop: bool, passes=None):
        """The later passes `steps` under one scan over their collections (`PassStack`): each pass's what
        it writes stacked under its block's own layer names, and where every pass exits, the exit before
        it under its coda's (`ExitingPass`). With `passes`, a pass runs where its step is below it
        (`_counted_step`)."""
        model, loop, shared = self.model, self.loop, self.shared
        layers = model.num_layers
        first, end = self.span
        views = [PassView(pass_offset(loop, layers, step), first, end - first) for step in steps]
        if loop.exiting:
            views = [ExitingPass(view, PassView(exit_offset(loop, layers, step - 1), end, layers - end))
                     for view, step in zip(views, steps, strict=True)]
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

        def later(module, carry, step_inputs):
            # The block's input, its per-layer inputs and the entering cache close over the scan.
            theta, step = step_inputs
            theta = _theta(module, loop, steps) if theta is None else theta
            if passes is not None:
                return self.counted_step(module, carry, theta, step, passes)
            if entering_cache is None:
                return self.stepped(module, carry, theta, exit_stop=exit_stop, enter_stop=enter_stop)
            reading = nn.map_variables(
                functools.partial(self.stepped, exit_stop=exit_stop, enter_stop=enter_stop),
                mapped_collections='cache', mutable=True,
                trans_in_fn=lambda variables: {**variables, 'cache': entering_cache})
            return reading(module, carry, theta)

        stacked = tuple(collection for collection in mapped if shared is None or collection != 'cache')
        scan = nn.scan(later, variable_axes={stacked: 0},
                       variable_carry='cache' if shared is not None else False,
                       variable_broadcast=DenyList(mapped), split_rngs={'params': False, 'dropout': True},
                       length=len(steps), metadata_params={nn.PARTITION_NAME: None})
        run = nn.map_variables(scan, mapped_collections=mapped, trans_in_fn=view.inside,
                               trans_out_fn=view.outside, mutable=True)
        # Each pass's own θ row, if it holds one, and its step.
        (x, bank), (leavings, outputs, gated) = run(model, (x, bank), (
            _theta(model, loop, steps, rows=True), jnp.arange(steps.start, steps.stop, dtype=jnp.int32)))
        if self.exit_outputs or gated is not None:
            ran = None if passes is None else jnp.arange(steps.start, steps.stop) < passes
            self.exited.extend(_sown_exits(model, leavings, outputs, gated, ran, self.valid))
        return x, bank

    def counted(self, x, bank, passes):
        """Every later pass the loop's `steps` allow under one scan, each run where its step is below
        `passes` (`counted_step`): one program for every count."""
        return self.scanned(x, bank, range(1, self.loop.steps), exit_stop=False, enter_stop=False,
                            passes=passes)

    def counted_step(self, module, carry, theta, step, passes):
        """A later pass of a count given per call (`counted`): skipped at or past `passes`, its state and
        bank carried as they are, its exit's state zero; before the cut the last `backprop_steps` passes
        leave (`passes - backprop_steps`), run passing no gradient back, the state at the cut stopped, as
        the static cut's passes are (`fixed`); else run. Each branch of `lax.cond`/`lax.switch` runs alone
        and its transpose alone, so a skipped pass or one before the cut costs nothing backward, and the
        block's program is traced once a branch."""
        # The pass checkpointed whole, its branch among them, not its block within the branch: the scan
        # saves each step's entering carry alone, as a loop of every pass saves its block's. The pass the
        # cut stops exits and enters stopped, where a static cut stops them (`fixed`).
        backprop = self.loop.backprop_steps

        def branched(module, carry, step, passes):
            at_cut = False if backprop is None else step == passes - backprop

            def run(module, carry):
                return self.stepped(module, carry, theta, exit_stop=at_cut, enter_stop=at_cut,
                                    block=self.pass_block)

            def skip(module, carry):
                x, _ = carry
                zero = jnp.zeros_like(x)
                output = module.exit_states(zero) if self.exit_outputs else None
                gated = module.exit_states(zero) if self.gates_block else None
                return carry, (zero if self.exit_states else None, output, gated)

            if backprop is None:
                return _sown_switch((step < passes).astype(jnp.int32), (skip, run), module, carry)

            def held_back(module, carry):
                # A pass before the cut passes no gradient back through its block, its state stopped where
                # it leaves; its exit, read from the state stopped where it enters the coda, still trains
                # the coda, the norm and the gate, as the static cut's exits do (`fixed`).
                state, exits = self.stepped(module, carry, theta, exit_stop=True, enter_stop=False,
                                            block=self.pass_block)
                return jax.lax.stop_gradient(state), exits

            index = jnp.where(step >= passes, 0, jnp.where(step < passes - backprop, 1, 2))
            return _sown_switch(index, (skip, held_back, run), module, carry)

        stepping = nn.remat(fixed_remat(branched)) if self.loop.remat else branched
        return stepping(module, carry, step, passes)


def compacted(ran: Callable, module, carry, normed, rows: jax.Array, passes: _Passes):
    """`ran`'s pass (`_Passes.stopped_pass`) over the rows of `carry` (its state and bank) still looping
    (`rows` those already exited), gathered into the smallest of the batch's halvings that holds them (`[B,
    B // 2, ..., 1]`, one program a size under `lax.switch`), with every array of the call's inputs, the
    prelude's publications, the block's input and the step norm's `normed` whose leading axes are the
    call's rows and tokens, and scattered back: a pass over fewer rows computes fewer, a row that exited
    none (row 12927). A row in the bucket beside them runs the pass too, its state no output reads."""
    x, _ = carry
    batch, length = x.shape[:2]
    sizes = [batch]
    while sizes[-1] > 1:
        sizes.append(sizes[-1] // 2)
    looping = jnp.sum(~rows)
    # The rows still looping first; the smallest size that holds them.
    order = jnp.argsort(rows, stable=True)
    index = jnp.sum(jnp.asarray(sizes) >= looping) - 1

    def rowed(leaf) -> bool:
        # A row's array: its leading axes the call's rows and tokens (a state, its bank, an input, a mask, a
        # publication), not a table of the tokens' (a position's rotation), whatever its sizes.
        return isinstance(leaf, jax.Array) and leaf.shape[:2] == (batch, length)

    def taken(tree, picked):
        return jax.tree.map(lambda leaf: leaf[picked] if rowed(leaf) else leaf, tree)

    def put(whole, part, picked):
        return jax.tree.map(lambda full, each: full.at[picked].set(each) if rowed(full) else each,
                            whole, part)

    def bucket(size: int):
        def run(module, carry):
            picked = order[:size]
            held = passes.inputs, passes.prelude, passes.injected
            passes.inputs, passes.prelude, passes.injected = taken(held, picked)
            try:
                state, left = ran(module, taken(carry, picked), taken(normed, picked))
            finally:
                passes.inputs, passes.prelude, passes.injected = held
            kept = left if passes.left is None else put(passes.left, left, picked)
            return put(carry, state, picked), kept
        return run
    return _sown_switch(index, tuple(bucket(size) for size in sizes), module, carry)


def _sown_switch(index, branches, module, *operands):
    """`nn.switch` over Flax module functions whose last branch, a pass, sows what the others do not (its
    attention's statistics, its routers' loads, captures: what a pass writes beside its state): each other
    branch's collections take the pass's shape, its own entries kept and what it does not sow zero, so
    the branches agree, as `lax.switch` asks (`_shaped_like`). A skipped pass's sows are zero; a reader
    weighs them by the count it gave (`run_loop`'s `passes`)."""
    count = len(branches)

    def transform(*parts):
        functions, (scope, index, *operands) = parts[:count], parts[count:]

        def inner(scope_fn, repack_fn, variable_groups, rng_groups):
            def branch(function, *operands):
                scope = scope_fn(variable_groups, rng_groups)
                return function(scope, *operands), repack_fn(scope)

            pure = [functools.partial(branch, function) for function in functions]
            _, sown = jax.eval_shape(pure[-1], *operands)

            def agreeing(function):
                def run(*operands):
                    out, variables = function(*operands)
                    return out, _shaped_like(variables, sown)
                return run

            return jax.lax.switch(index, [agreeing(function) for function in pure], *operands)

        return lift.pack(inner, (True,), (True,), (True,), name='switch')(scope)

    return lift_direct_transform(transform, tuple(branches), module, index, *operands)


def _shaped_like(value, shape):
    """`value` (a branch's collections) in `shape`'s tree (the pass's, `jax.eval_shape`'s): each entry
    `value` holds at a place of `shape` kept, each it lacks zeros of its shape; each container rebuilt as
    `shape`'s own type (a `FrozenDict`, a `dict`, a tuple of sows), so the branches' trees agree."""
    rebuilt: Callable[..., object] = type(shape)
    if isinstance(shape, Mapping):
        held = value if isinstance(value, Mapping) else {}
        return rebuilt({key: _shaped_like(held.get(key), each) for key, each in shape.items()})
    if isinstance(shape, (tuple, list)):
        held = value if isinstance(value, (tuple, list)) else ()
        return rebuilt([_shaped_like(held[index] if index < len(held) else None, each)
                        for index, each in enumerate(shape)])
    if shape is None:
        return None
    if value is not None:
        return value
    # Any other node (a boxed variable's partitioning, `nn.LogicallyPartitioned`) by its leaves.
    return jax.tree.map(_zero_like, shape)


def _zero_like(shape: jax.ShapeDtypeStruct) -> jax.Array:
    """A zero typed as the pass's own entry `shape` is, which a zero of its shape alone would differ from:
    varying over the manual axes it varies over (a batch-split forward's statistic under `shard_map`, its
    rows' maxima), placed as it is on a mesh whose types carry the split."""
    placed = shape.sharding if isinstance(shape.sharding, NamedSharding) else None
    zero = jnp.zeros(shape.shape, shape.dtype, out_sharding=placed)
    manual = shape.manual_axis_type
    if manual is not None and manual.varying:
        # Both at once on a mesh of manual and explicit axes (a `shard_map` over one of its axes).
        return jax.lax.pcast(zero, tuple(sorted(manual.varying)), to="varying")
    return zero


def _handed_on(model, exited, last, threshold: float) -> jax.Array:
    """The state a loop under an exit threshold hands on (`_exited`), each token's exit pass, 1-based,
    written where a caller opens `EXIT_PASSES` (a write nothing reads otherwise): the choice its state is
    taken by, the last pass where no earlier one exits it, so one pass of a loop exits every token there."""
    chosen = jnp.ones(last.shape[:2], jnp.int32)
    if exited:
        last, chosen = _exited(exited, last, threshold)
    model.sow(EXIT_PASSES, 'chosen', chosen)
    return last


def _sown_exits(model, leavings, outputs, gated, ran, valid) -> list[tuple[jax.Array, jax.Array]]:
    """The scanned passes' exits (`_Passes.scanned`), their states a threshold selects `leavings` (None where
    no threshold reads them), their normed outputs `outputs` (None where no head reads them) and the states
    their gate reads `gated` (None where it reads the outputs: `Loop.exit_at`), stacked on a leading pass
    axis, sown into `exits` in pass order as `_Passes.exit_of` sows each: their gate logits (where the loop
    gates its exits, `gate_logit`) and the states the head reads; with `ran` (a count given per call),
    whether each pass ran, sown beside them (`ran`), a skipped pass's exit computed from a zero state. Their
    `(state, gate logit)` pairs for a threshold's choice (`_exited`)."""
    read = outputs if gated is None else gated
    logits = gate_logit(model, read, valid) if model.loop.exit_gate else None
    if not model.is_initializing():
        for index in range(read.shape[0]):
            if logits is not None:
                model.sow(EXITS, 'logits', logits[index])
            if outputs is not None:
                model.sow(EXITS, 'states', model.head_states(outputs[index]))
            if ran is not None:
                model.sow(EXITS, 'ran', ran[index])
    return [] if logits is None or leavings is None else list(zip(leavings, logits, strict=True))


def gate_logit(model, output: jax.Array, valid: jax.Array | None) -> jax.Array:
    """The exit gate's logit `[..., B, S, 1]` of a pass's normed exit states `output` `[..., B, S, D]`:
    each token's (`Loop.exit_by='token'`), or one a row, the gate of the row's mean over its real tokens
    (`valid`, every token where None) broadcast over the row (`'row'`): the hazard the exit objective
    trains and a threshold reads, alike."""
    if model.loop.exit_by == "token":
        return model.early_exit_gate(output)
    # Over the tokens' axis, a scan's passes stacked before it alike.
    wide = at_least_fp32(output.dtype)
    weights = (jnp.ones(output.shape[-3:-1], wide) if valid is None else valid.astype(wide))[..., None]
    pooled = jnp.sum(output.astype(wide) * weights, axis=-2, keepdims=True) / jnp.maximum(
        jnp.sum(weights, axis=-2, keepdims=True), 1)
    return jnp.broadcast_to(model.early_exit_gate(pooled.astype(output.dtype)), (*output.shape[:-1], 1))


def _exited(exited, last, threshold: float) -> tuple[jax.Array, jax.Array]:
    """Each token's exiting pass's state (`exit_choice`), from the earlier passes' `(state, gate logit)`
    and the last pass's state, which a token none of them exits takes, and that pass, 1-based `[B, L]`."""
    choice = exit_choice([logit for _, logit in exited], threshold)
    leaving = jnp.stack([state for state, _ in exited] + [last])
    return (jnp.take_along_axis(leaving, choice[None, ..., None], axis=0)[0],
            choice.astype(jnp.int32) + 1)
