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
    if first:
        x = model.stack(x, span=(0, first), train=train, decode=decode, **inputs)

    def every_pass(module, x, inputs, passes):
        # The block's passes, first and later, and what each exits.
        run = _Passes(module, train=train, decode=decode, inputs=inputs)
        run.injected = x
        first_pass = run.run_block if run.shared is None else nn.map_variables(
            run.run_block, mapped_collections=WRITTEN, trans_in_fn=run.shared.first,
            trans_out_fn=run.shared.outside, mutable=True)
        x, bank = first_pass(module, module.norm(x + run.injected) if loop.reinject else x, **inputs)
        x, _ = run.counted(x, bank, passes) if passes is not None else run.fixed(x, bank)
        return x, run.exited

    layout = placed_layout()
    # A decode (a server's step) and a store banked by run (each run's bank read by the scan the gathered
    # passes do not run) read the block where it lies.
    if (loop.gathered and not decode and not model.is_initializing() and layout is not None
            and jax.sharding.get_abstract_mesh().shape.get(FSDP_AXIS, 1) > 1
            and not model.banked_collections((first, end))):
        # Under `exiting` every pass runs the coda too, whose layers are gathered with the block's.
        x, exited = _gathered_passes(model, every_pass, x, inputs, passes, functools.partial(
            _gathered_params, layout=layout, mesh=jax.sharding.get_abstract_mesh(),
            scope=('params', *model.scope.path), span=(first, layers if loop.exiting else end)))
    else:
        x, exited = every_pass(model, x, inputs, passes)
    if loop.reinject:
        x = model.norm(x)
    if end < layers:
        x = model.stack(x, span=(end, layers), train=train, decode=decode, **inputs)
    if loop.exit_threshold is None or model.is_mutable_collection(EXITS):
        return x
    return _handed_on(model, exited, x, loop.exit_threshold)


def _gathered_passes(model, every_pass: Callable, x, inputs: dict, passes, gathered: Callable):
    """`every_pass(model, x, inputs)` traced with every Auto axis Explicit (`sharding.explicit_over`), every
    parameter the passes read gathered whole once before the first over the FSDP axis and every axis the
    stream's tokens are split over (a hybrid run's data axis, a sequence axis), a matrix reduced over them
    (`gathered`, `_gathered_params`), so each pass's gradient of it stays unreduced, the passes' sum of it
    reduced once after the loop (`Loop.gathered`): a product's gradient sums over every axis its tokens
    split. `inputs`' arrays and a count of `passes` given per call held whole on every device, and what the
    passes write (their sown and cached collections) placed back on the mesh the step traces
    (`sharding.typed_on`). The passes' operations are typed on the mesh: a
    projection takes no launch mapped per shard there (`projections.rows_split`), the partitioner's product
    running it."""
    from dew.nn.kernels import generation

    whole = jax.sharding.PartitionSpec()
    mesh = jax.sharding.get_abstract_mesh()

    def held(module, x):
        # What the passes read from outside the region placed on its mesh: the inputs and a count of passes.
        inside = jax.tree.map(lambda value: jax.sharding.reshard(value, whole)
                              if isinstance(value, jax.Array) else value, (inputs, passes))
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
    injected: jax.Array | None = None
    exited: list = dataclasses.field(default_factory=list)
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

    def __post_init__(self):
        model, loop = self.model, self.loop
        self.span = loop.layers or (0, model.num_layers)
        self.shared = SharedPassView(self.span) if loop.shared_cache else None
        # A pass recomputed whole whatever the rung (`fixed_remat`).
        self.run_block = (nn.remat(fixed_remat(self.pass_block)) if loop.remat and not self.decode
                          else self.pass_block)
        self.exit_states = (loop.exiting and loop.exit_threshold is not None
                            and not model.is_mutable_collection(EXITS))
        self.exit_outputs = loop.exiting and (self.exit_states or model.is_mutable_collection(EXITS)
                                              or model.is_initializing())

    @property
    def loop(self) -> Loop:
        return self.model.loop

    def pass_block(self, module, x, bank=None, **inputs):
        block = None if self.loop.layers is None else self.span
        return _passed(module, x, bank, block=block, train=self.train, decode=self.decode, **inputs)

    def entering(self, module, x, bank, *, stop: bool, normed=None, theta=None):
        return _entered(module, x, bank, self.injected, stop=stop, normed=normed, theta=theta)

    def coda(self, module, x):
        """A pass's exit state from its output `x`: the coda run on it, under the step's view."""
        _, end = self.span
        layers = self.model.num_layers
        if end == layers:
            return x
        return module.stack(x, span=(end, layers), train=self.train, decode=self.decode, **self.inputs)

    def fixed(self, x, bank):
        """Every later pass, the cut static: the passes before it scanned, the one it cuts, alone, and
        those after it scanned; under a shared cache the last after the scan, as the first runs."""
        loop, shared = self.loop, self.shared
        cut = None if loop.backprop_steps is None else loop.steps - loop.backprop_steps
        scanned = range(1, loop.steps - 1 if shared is not None else loop.steps)
        segments = ((scanned,) if cut is None or cut not in scanned else
                    (range(scanned.start, cut), range(cut, cut + 1), range(cut + 1, scanned.stop)))
        for steps in segments:
            if not steps:
                continue
            # A pass at or before the cut exits without passing its own gradient back.
            exit_stop, enter_stop = cut is not None and steps.stop - 1 <= cut, steps.start == cut
            if len(steps) == 1 or (loop.exiting and self.decode) or captures_any_method(self.model):
                # One pass is its own program; a decode's exits write their codas' caches pass by pass; a
                # capture of any method keeps what no scan stacks (a method's non-array return).
                for step in steps:
                    x, bank = self.unrolled(x, bank, step, self.own_view(step), exit_stop=exit_stop,
                                            enter_stop=step == cut)
            else:
                x, bank = self.scanned(x, bank, steps, exit_stop=exit_stop, enter_stop=enter_stop)
        if shared is not None and loop.steps > 1:
            x, bank = self.unrolled(x, bank, loop.steps - 1, shared, exit_stop=False, enter_stop=False)
        return x, bank

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
            leaving = _exit_state(model, jax.lax.stop_gradient(x) if exit_stop else x, step - 1,
                                  train=self.train, decode=self.decode, **self.inputs)
            output, logit = _sown_exit(model, leaving)
            if logit is not None:  # a gated exit, which the threshold's choice weighs (`_exited`)
                self.exited.append((leaving, logit))
            normed = output if end == model.num_layers and model.normed_output else None
        x, bank = self.entering(model, x, bank, stop=enter_stop, normed=normed,
                                theta=_theta(model, loop, range(step, step + 1)))
        run = nn.map_variables(self.run_block, mapped_collections=WRITTEN, trans_in_fn=view.inside,
                               trans_out_fn=view.outside, mutable=True)
        out, bank = run(model, x, bank, **self.inputs)
        return _stepped(loop, out, x, self.injected), bank

    def stepped(self, module, carry, theta, *, exit_stop, enter_stop, block=None):
        """One later pass inside the scan over its step's collections: the exit of the pass before it
        where every pass exits (its exit state and its normed output, the scan's outputs where read:
        `exit_states`, `exit_outputs`), then its block (`block`, `run_block` where None)."""
        x, bank = carry
        loop = self.loop
        _, end = self.span
        leaving = output = normed = None
        if loop.exiting:
            leaving = self.coda(module, _stopped(x, exit_stop))
            output = module.exit_states(leaving)
            normed = output if end == module.num_layers and module.normed_output else None
        x, bank = self.entering(module, x, bank, stop=enter_stop, normed=normed, theta=theta)
        out, bank = (block or self.run_block)(module, x, bank, **self.inputs)
        exits = (leaving if self.exit_states else None, output if self.exit_outputs else None)
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
        (x, bank), (leavings, outputs) = run(model, (x, bank), (
            _theta(model, loop, steps, rows=True), jnp.arange(steps.start, steps.stop, dtype=jnp.int32)))
        if self.exit_outputs:
            ran = None if passes is None else jnp.arange(steps.start, steps.stop) < passes
            self.exited.extend(_sown_exits(model, leavings, outputs, ran))
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
                return carry, (zero if self.exit_states else None, output)

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


def _sown_exits(model, leavings, outputs, ran) -> list[tuple[jax.Array, jax.Array]]:
    """The scanned passes' exits (`_Passes.scanned`), their exit states `leavings` (None where no threshold
    reads them) and normed outputs `outputs` stacked on a leading pass axis, sown into `exits` in pass order
    as `_sown_exit` sows each: their gate logits (where the loop gates its exits) and the states the head
    reads; with `ran` (a count given per call), whether each pass ran, sown beside them (`ran`), a
    skipped pass's exit computed from a zero state. Their `(state, gate logit)` pairs for a threshold's
    choice (`_exited`)."""
    logits = model.early_exit_gate(outputs) if model.loop.exit_gate else None
    if not model.is_initializing():
        for index in range(outputs.shape[0]):
            if logits is not None:
                model.sow(EXITS, 'logits', logits[index])
            model.sow(EXITS, 'states', model.head_states(outputs[index]))
            if ran is not None:
                model.sow(EXITS, 'ran', ran[index])
    return [] if logits is None or leavings is None else list(zip(leavings, logits, strict=True))


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


def _exited(exited, last, threshold: float) -> tuple[jax.Array, jax.Array]:
    """Each token's exiting pass's state (`exit_choice`), from the earlier passes' `(state, gate logit)`
    and the last pass's state, which a token none of them exits takes, and that pass, 1-based `[B, L]`."""
    choice = exit_choice([logit for _, logit in exited], threshold)
    leaving = jnp.stack([state for state, _ in exited] + [last])
    return (jnp.take_along_axis(leaving, choice[None, ..., None], axis=0)[0],
            choice.astype(jnp.int32) + 1)
