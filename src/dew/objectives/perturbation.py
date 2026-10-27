"""Training without backpropagation: node perturbation, as Dust trains a transformer.

Dust (Q Labs, qlabs.sh/research/dust) trains with forward passes alone. Each linear layer's output y_t is
jittered, y_t + sigma a_t with a_t ~ N(0, I), independently at every unit of the batch (a token, an
example), so one forward is a population of jitters; the change it makes in each unit's own loss is that
jitter's reward, and the reward-weighted noise averaged over draws estimates the loss's derivative at the
layer's output,

    g_t = 1 / (2 K sigma) sum_draws (l_t(y + sigma a) - l_t(y - sigma a)) a_t,

here from antithetic pairs, which cancel the reward's baseline. A layer's weight gradient is then the
product of that estimate with the layer's input from the clean forward, summed over units, as
backpropagation forms it, and an embedding's is the estimate scattered into the rows its ids read. Every
other parameter (a norm's scale, a mixing scalar, an expert's weights) is estimated by two-sided
weight-space evolution strategies, rewarded by the whole objective, and so is each use of a site's
parameter that no layer call makes: a vocabulary head the loss reads as a matrix and contracts itself
(`AffineHead.output_table`), a tied embedding's, whose matrix is jittered in that read alone and whose
estimate is added to the lookup's, so a vocabulary row the batch only predicts is estimated too.

`NodePerturbation`, mixed into any objective over a Flax model,

    class ForwardLM(NodePerturbation, LMObjective): ...
    ForwardLM(model, seq_len, draws=256, sigma=0.1)

does this for the objective's own loss: its value is the objective's, its statistics tree the objective's
(a `Ratio`, a routed decoder's `RoutedStatistics`), and only its derivative in the trained parameters is
estimated. The estimate is taken where the trainer asks for that derivative, under the cotangent the
trainer's reduction gives the statistics, pooled over the update's microbatches where it pools them, so
each statistic is weighted as the reduction weighs it and every denominator stays out. Each unit of the
first statistic is credited by its own share (`Objective.scored_units`: a supervised model's examples, a
language model's tokens with their configured terms) where a site's output's leading axes are the units',
and every other statistic's change, a router's balance term, is credited to every unit. Units that read
other units' outputs, a causal model's tokens, are credited by their own share alone, as Dust's direct
draws are: a lower-variance estimate, biased by what a jitter changes in later tokens.

Every pass reads the step's own draws (dropout, routing noise), and the model's other collections are
left as the clean loss leaves them. Each jitter is drawn in float32 and cast to its site's output dtype,
and the cast draw is both the jitter and what its reward multiplies. A site whose output's leading axis is
the batch's rows draws each example's jitter by its place among the update's examples (`Step.sample`),
and every other draw, a weight's or an unbatched site's, by the update's key, so a split or accumulated
update estimates what its batch whole does. A model that chooses its remat (`RematChoosing`) is estimated
without it, which a forward-only pass never needs; one that maps or scans its layers (`nn.scan`,
`jax.vmap` in its call) hides their calls' values from a capture outside them, and is refused.

Dust's own tuning, a draw budget per layer group, the attention internals scored through the attention
output's estimate, the head jittered on cached logits a vocabulary slab at a time, are the paper's
efficiency for its GPT and are not here: every site is jittered in every draw.
"""

from __future__ import annotations

import copy
import dataclasses
import string

import jax
import jax.numpy as jnp
from flax import linen as nn

from dew._distributions import counter_key
from dew.batch import Batch, Variables, leaves_by_path
from dew.data.dataset import rows_of
from dew.nn.precision import at_least_fp32, float32_einsum
from dew.nn.protocols import OutputTable, RematChoosing, head_table
from dew.objectives.base import Aux, Objective, Step, _has_aux, stream

NOISE = jnp.float32
"""The dtype every jitter is drawn in, whatever its site's: a float64 twin of a run draws what the run
draws, and a bfloat16 site jitters by the draw cast to bfloat16."""

Path = tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Site:
    """One `Dense`, `DenseGeneral` or `Embed` call of the model in the clean forward: its module's path,
    how it maps (`kind`: "dense", "embed" or "attend"), its input, its output's shape and dtype, and a
    dense call's contracted input axes, normalized and sorted, as its kernel's leading axes take them."""

    path: Path
    kind: str
    inputs: jax.Array
    shape: tuple[int, ...]
    dtype: jnp.dtype
    contracted: tuple[int, ...] = ()


@dataclasses.dataclass(frozen=True)
class TableUse:
    """One read of a model's vocabulary head as a matrix (`AffineHead.output_table`), which a loss
    contracts itself rather than calling a layer: the unbound module that read it, so the read can be
    taken again over other parameters, and the matrix's shape and dtype."""

    module: nn.Module
    shape: tuple[int, ...]
    dtype: jnp.dtype


def _reads_table(context, output) -> bool:
    """Whether a call reads a vocabulary head as a matrix (`AffineHead.output_table`)."""
    return context.method_name == "output_table" and isinstance(output, OutputTable)


def _table(context, output) -> TableUse | None:
    """The head read a call is, or None (`_reads_table`)."""
    if not _reads_table(context, output):
        return None
    unbound, _ = context.module.unbind()
    return TableUse(unbound, tuple(output.matrix.shape), output.matrix.dtype)


def _kind(context) -> str | None:
    """How a call maps its input, or None for a call no site reads: a `Dense`, and a `DenseGeneral` that
    batches no axes (`batch_dims`), contract input axes with their kernel."""
    module, method = context.module, context.method_name
    if method == "__call__" and isinstance(module, nn.Dense | nn.DenseGeneral):
        return None if isinstance(module, nn.DenseGeneral) and module.batch_dims else "dense"
    if isinstance(module, nn.Embed) and method in ("__call__", "attend"):
        return "embed" if method == "__call__" else "attend"
    return None


def _site(context, args, kwargs, output) -> Site | None:
    """The site a call is, or None (`_kind`)."""
    kind = _kind(context)
    if kind is None:
        return None
    module = context.module
    inputs = args[0] if args else next(iter(kwargs.values()))
    contracted: tuple[int, ...] = ()
    if kind == "dense":
        rank = jnp.ndim(inputs)
        axes = (-1,) if isinstance(module, nn.Dense) else (
            tuple(module.axis) if isinstance(module.axis, tuple | list) else (module.axis,))
        # Flax sorts the axes it contracts (`_normalize_axes`), and its kernel takes them in that order.
        contracted = tuple(sorted(axis % rank for axis in axes))
    return Site(tuple(module.path), kind, inputs, tuple(jnp.shape(output)), jnp.result_type(output),
                contracted)


class NodePerturbation[LossT, EffectsT](Objective[LossT, EffectsT]):
    """Trains the objective it is mixed into without backpropagation (the module's docstring), by `draws`
    antithetic pairs of output jitters of scale `sigma` a step, and `weight_draws` pairs of weight jitters
    of scale `weight_sigma` for the parameters no site holds. Everything else, the model, its data and its
    saved task, is the objective's own, whose constructor takes the other arguments."""

    def __init__(self, *args, draws: int = 64, sigma: float = 0.2, weight_draws: int = 8,
                 weight_sigma: float = 0.03, **kwargs):
        if draws < 1 or weight_draws < 1:
            raise ValueError("node perturbation draws at least one pair of each kind of jitter")
        self.draws, self.sigma = draws, sigma
        self.weight_draws, self.weight_sigma = weight_draws, weight_sigma
        super().__init__(*args, **kwargs)

    def loss(self, variables: Variables, batch: Batch, step: Step):
        if not step.training:
            return super().loss(variables, batch, step)
        held = {name: value for name, value in variables.items() if name != "params"}

        @jax.custom_vjp
        def estimated(params, rest):
            others, held_batch, held_step = rest
            return self._scored({**others, "params": params}, held_batch, held_step)

        def forward(params, rest):
            return estimated(params, rest), (params, rest)

        def backward(saved, cotangent):
            params, rest = saved
            others, held_batch, held_step = rest
            stats, _ = cotangent
            # None: the variables held, the batch and the step take no cotangent.
            return self.estimate(params, others, held_batch, held_step, stats), None

        estimated.defvjp(forward, backward)
        return estimated(variables["params"], (held, batch, step))

    def _scored(self, variables: Variables, batch: Batch, step: Step) -> tuple[LossT, Aux[EffectsT]]:
        """The objective's own statistics and `Aux` for `batch`, the `Aux` empty where it returns none
        (`_has_aux`): statistics that are themselves a tuple stay whole."""
        scored = super().loss(variables, batch, step)
        return scored if _has_aux(scored) else (scored, Aux[EffectsT](metrics={}))

    def estimate(self, params: Variables, held: Variables, batch: Batch, step: Step, cotangent) -> Variables:
        """The estimated derivative in every trained parameter of the statistics' inner product with
        `cotangent`, in the parameters' tree: what the trainer's reduction asks for."""
        estimating = self._forward_only()
        variables = {**held, "params": params}
        sites: list[Site] = []
        tables: list[TableUse] = []

        def captured(next_fun, args, kwargs, context):
            output = next_fun(*args, **kwargs)
            site, table = _site(context, args, kwargs, output), _table(context, output)
            if site is not None:
                sites.append(site)
            if table is not None:
                tables.append(table)
            return output

        flat = leaves_by_path(params)
        try:
            with nn.intercept_methods(captured):
                estimating._statistics_and_shares(variables, batch, step)
            gradients = self._sited(estimating, variables, batch, step, sites, cotangent, flat)
        except jax.errors.UnexpectedTracerError as escaped:
            raise ValueError("node perturbation reads each Dense and Embed call's input and output, which a "
                             "model that maps, scans or rematerializes its layers outside its remat policy "
                             "(nn.scan, jax.vmap or nn.remat in its call) hides from it; build the model "
                             "with its layers unrolled") from escaped
        gradients = self._remaining(estimating, variables, batch, step, cotangent, flat, gradients, tables)
        return jax.tree_util.tree_map_with_path(
            lambda path, leaf: gradients[tuple(entry.key for entry in path)].astype(leaf.dtype), params)

    def _forward_only(self) -> NodePerturbation:
        """This objective with every module that chooses its remat (`RematChoosing`) built without it: a
        forward-only pass recomputes nothing, and a capture reads the values a remat would hide."""
        entries = self.program_key()
        plain = [entry.module.with_remat(None) if isinstance(entry.module, RematChoosing) else entry.module
                 for entry in entries]
        estimating = copy.copy(self)
        if any(module is not entry.module for module, entry in zip(plain, entries, strict=True)):
            estimating.substitute(plain)
        return estimating

    def _statistics_and_shares(self, variables: Variables, batch: Batch,
                               step: Step) -> tuple[LossT, jax.Array | None]:
        """The objective's statistics and its units' shares of their first leaf (`scored_units`), or no
        shares where it states none."""
        units = self.scored_units(variables, batch, step)
        if units is not None:
            return units
        stats, _ = self._scored(variables, batch, step)
        return stats, None

    def _update_key(self, step: Step, name: str) -> jax.Array:
        """The update's key for the draws `name` names, the same for every microbatch the update pools."""
        return jax.random.fold_in(counter_key(step.key if step.example_key is None else step.example_key),
                                  stream(name))

    def _noise(self, step: Step, index: int, site: Site, draw: jax.Array, rows: int) -> jax.Array:
        """Site `index`'s jitter at `draw`, drawn in `NOISE` and cast to the site's output dtype: each
        example's own where the output's leading axis is the batch's rows, else the update's."""
        if site.shape[:1] == (rows,):
            # Each draw of the scan is its own; a recorder or witness of the step keeps the loss's own.
            def normal(key, shape, dtype):
                return jax.random.normal(jax.random.fold_in(key, draw), shape, dtype)

            quiet = dataclasses.replace(step, recorder=None, witness=None)
            noise = quiet.sample(f"node-perturbation/site/{index}", normal, site.shape[1:], dtype=NOISE,
                                 rows=rows)
        else:
            key = jax.random.fold_in(jax.random.fold_in(self._update_key(step, "node-perturbation/sites"),
                                                        index), draw)
            noise = jax.random.normal(key, site.shape, NOISE)
        return noise.astype(site.dtype)

    def _rewards(self, plus, minus, cotangent) -> tuple[jax.Array, jax.Array | None, jax.Array]:
        """A pair's change in the statistics' inner product with `cotangent` (`whole`), each unit's
        change in its share of the first statistic (`units`, None without shares), and the other
        statistics' change (`rest`), each over twice the jitter's scale."""
        (above, shared_above), (below, shared_below) = plus, minus
        weights = jax.tree.leaves(cotangent)
        changes = [jnp.sum(jnp.asarray(weight, at_least_fp32(jnp.result_type(high))) * (high - low))
                   for weight, high, low in zip(weights, jax.tree.leaves(above), jax.tree.leaves(below),
                                                strict=True)
                   if jnp.issubdtype(jnp.result_type(high), jnp.inexact)]
        whole = sum(changes[1:], changes[0])
        if shared_above is None or shared_below is None:
            return whole, None, whole
        first = jnp.asarray(weights[0], shared_above.dtype)
        units = first * (shared_above - shared_below)
        return whole, units, whole - changes[0]

    def _sited(self, estimating, variables, batch, step, sites: list[Site], cotangent,
               flat: dict[Path, jax.Array]) -> dict[Path, jax.Array]:
        """Each site's parameters' estimated gradient, summed over the calls of each module, by path."""
        rows = rows_of(batch)

        def jittered(sign: float, draw: jax.Array):
            index = iter(range(len(sites)))

            def jitter(next_fun, args, kwargs, context):
                output = next_fun(*args, **kwargs)
                if _kind(context) is None:
                    return output
                at = next(index)
                noise = self._noise(step, at, sites[at], draw, rows)
                return output + jnp.asarray(sign * self.sigma, noise.dtype) * noise

            with nn.intercept_methods(jitter):
                return estimating._statistics_and_shares(variables, batch, step)

        def credited(whole, units, rest, site: Site, dtype) -> jax.Array:
            # A unit's own change where the output's leading axes are the units', else the whole change.
            if units is None or site.shape[:units.ndim] != units.shape:
                return jnp.asarray(whole, dtype)
            change = (units + rest).astype(dtype)
            return change.reshape(change.shape + (1,) * (len(site.shape) - change.ndim))

        def draw(errors, index):
            whole, units, rest = self._rewards(jittered(1.0, index), jittered(-1.0, index), cotangent)
            return [error + credited(whole, units, rest, site, error.dtype) / (2 * self.sigma)
                    * self._noise(step, at, site, index, rows).astype(error.dtype)
                    for at, (error, site) in enumerate(zip(errors, sites, strict=True))], None

        errors, _ = jax.lax.scan(draw, [jnp.zeros(site.shape, at_least_fp32(site.dtype)) for site in sites],
                                 jnp.arange(self.draws))
        gradients: dict[Path, jax.Array] = {}
        for site, error in zip(sites, errors, strict=True):
            for name, value in self._reassembled(flat, site, error / self.draws).items():
                where = (*site.path, name)
                gradients[where] = value + gradients[where] if where in gradients else value
        return gradients

    @staticmethod
    def _reassembled(flat: dict[Path, jax.Array], site: Site, error: jax.Array) -> dict[str, jax.Array]:
        """A site's trained parameters' gradient from its output's estimated error, as backpropagation forms
        it: none for a module whose weights are not trained (a frozen base under an adapter)."""
        if site.kind == "dense":
            if (*site.path, "kernel") not in flat:
                return {}
            kernel = flat[(*site.path, "kernel")]
            # The input's free axes, in order, lead the output and the features follow; the kernel is the
            # contracted axes, sorted, then the features (`lax.dot_general`).
            letters = string.ascii_letters
            rank = site.inputs.ndim
            read = letters[:rank]
            free = "".join(read[axis] for axis in range(rank) if axis not in site.contracted)
            features = letters[rank:rank + error.ndim - len(free)]
            gradient = {"kernel": float32_einsum(
                f"{read},{free}{features}->{''.join(read[axis] for axis in site.contracted)}{features}",
                site.inputs.astype(error.dtype), error).astype(kernel.dtype)}
            if (*site.path, "bias") in flat:
                bias = flat[(*site.path, "bias")]
                gradient["bias"] = jnp.sum(error.reshape(-1, *bias.shape), 0).astype(bias.dtype)
            return gradient
        if (*site.path, "embedding") not in flat:
            return {}
        table = flat[(*site.path, "embedding")]
        if site.kind == "attend":
            return {"embedding": float32_einsum("...v,...d->vd", error,
                                                site.inputs.astype(error.dtype)).astype(table.dtype)}
        rows = jnp.zeros(table.shape, error.dtype).at[site.inputs.reshape(-1)].add(
            error.reshape(-1, table.shape[-1]))
        return {"embedding": rows.astype(table.dtype)}

    def _remaining(self, estimating, variables, batch, step, cotangent, flat: dict[Path, jax.Array],
                   gradients: dict[Path, jax.Array], tables: list[TableUse]) -> dict[Path, jax.Array]:
        """Every trained parameter's gradient: the sites', and two-sided weight-space evolution
        strategies', rewarded by the statistics' whole change under the update's draws, for each parameter
        no site holds, all its uses at once, and for each use of a site's parameter that no site calls,
        a vocabulary head the loss reads as a matrix (`TableUse`): its matrix jittered in that read alone,
        and the estimate taken back to the parameters the read takes it from, added to the sites'
        estimate of their other uses, a tied embedding's lookup among them."""
        others = [path for path in flat if path not in gradients]
        if not others and not (tables and gradients):
            return gradients
        key = self._update_key(step, "node-perturbation/weights")
        shapes = [flat[path].shape for path in others] + [table.shape for table in tables]
        dtypes = [flat[path].dtype for path in others] + [table.dtype for table in tables]

        def noise(index: int, draw: jax.Array) -> jax.Array:
            drawn = jax.random.normal(jax.random.fold_in(jax.random.fold_in(key, index), draw), shapes[index],
                                      NOISE)
            return drawn.astype(dtypes[index])

        def moved(sign: float, draw: jax.Array):
            point = {path: flat[path] + jnp.asarray(sign * self.weight_sigma, flat[path].dtype)
                     * noise(index, draw) for index, path in enumerate(others)}
            params = jax.tree_util.tree_map_with_path(
                lambda path, leaf: point.get(tuple(entry.key for entry in path), leaf), variables["params"])
            read = iter(range(len(others), len(shapes)))

            def jitter(next_fun, args, kwargs, context):
                output = next_fun(*args, **kwargs)
                if not _reads_table(context, output):
                    return output
                drawn = noise(next(read), draw)
                return output.replace(matrix=output.matrix + jnp.asarray(sign * self.weight_sigma,
                                                                          drawn.dtype) * drawn)

            with nn.intercept_methods(jitter):
                return estimating._statistics_and_shares({**variables, "params": params}, batch, step)

        def draw(sums, index):
            whole, _, _ = self._rewards(moved(1.0, index), moved(-1.0, index), cotangent)
            return [total + jnp.asarray(whole, total.dtype) / (2 * self.weight_sigma)
                    * noise(at, index).astype(total.dtype) for at, total in enumerate(sums)], None

        sums, _ = jax.lax.scan(draw, [jnp.zeros(shape, at_least_fp32(dtype)) for shape, dtype in
                                      zip(shapes, dtypes, strict=True)], jnp.arange(self.weight_draws))
        estimates = [total / self.weight_draws for total in sums]
        sited = dict(gradients)
        for table, estimate in zip(tables, estimates[len(others):], strict=True):
            # The read is the parameters' re-layout (a tied embedding as it is, a kernel through its
            # adapter), so its pullback takes the matrix's estimate to theirs without the model's backward.
            def matrix(trained, table=table) -> jax.Array:
                read = head_table(table.module, {**variables, "params": trained})
                assert read is not None, "the head read its matrix in the clean pass"
                return read.matrix

            _, back = jax.vjp(matrix, variables["params"])
            (taken,) = back(estimate.astype(table.dtype))
            # A parameter no site holds has its every use in its own estimate below already.
            for path, value in leaves_by_path(taken).items():
                if path in gradients:
                    sited[path] = sited[path] + value.astype(sited[path].dtype)
        return {**sited, **dict(zip(others, estimates[:len(others)], strict=True))}
