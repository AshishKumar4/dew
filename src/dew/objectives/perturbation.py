"""Training without backpropagation: node perturbation, as Dust trains a transformer.

Dust (Q Labs, qlabs.sh/research/dust) trains with forward passes alone. Each linear layer's output y_t is
jittered, y_t + sigma a_t with a_t ~ N(0, I), independently at every unit of the batch (a token, an
example), so one forward is a population of jitters; the change it makes in each unit's own loss is that
jitter's reward, and the reward-weighted noise averaged over draws estimates the loss's derivative at the
layer's output,

    g_t = 1 / (2 K sigma) sum_draws (l_t(y + sigma a) - l_t(y - sigma a)) a_t,

here from antithetic pairs, which cancel the reward's baseline. A layer's weight gradient is then the
outer product of that estimate with the layer's input from the clean forward, summed over units,
G = sum_t x_t g_t^T, as backpropagation forms it, and an embedding's is the estimate scattered into the
rows its ids read. Every other parameter (a norm's scale, a mixing scalar, a convolution's kernel) is
estimated by two-sided weight-space evolution strategies, rewarded by the whole loss.

`NodePerturbation`, mixed into any objective over a Flax model (`class Forward(NodePerturbation,
Supervised)`), does this: it reads every `nn.Dense` and `nn.Embed` call (`flax.linen.intercept_methods`), a subclass of either among them, and credits each unit
of an output by the objective's own loss of that unit (`Objective.unit_losses`, a language model's tokens,
a supervised model's examples) where the output's leading axes are the units', else by the whole loss.
Crediting a token's jitter by its own token's loss alone, as Dust's direct draws do, leaves out what the
jitter changes in later tokens' losses through attention: a lower-variance estimate, biased by that. No
derivative of the model is taken: the estimate is the loss's stated gradient (`Objective.with_gradients`),
which the trainer applies as it applies `jax.grad`'s. A model that maps or scans its layers (`nn.scan`,
`jax.vmap` inside its call) hides their calls' values from a capture outside them, and is refused.

Dust's own tuning, a draw budget per layer group, the attention internals scored through the attention
output's estimate, the head jittered on cached logits a vocabulary slab at a time, are the paper's
efficiency for its GPT and are not here: every site is jittered in every draw.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from flax import linen as nn, traverse_util

from dew.batch import Batch, Variables
from dew.nn.precision import float32_einsum
from dew.objectives.base import Objective, Ratio, Step


@dataclasses.dataclass(frozen=True)
class Site:
    """One `Dense` or `Embed` call of the model: its module's path, how it maps (`kind`: "dense", "embed"
    or "attend"), its input and the shape of its output in the clean forward."""

    path: tuple[str, ...]
    kind: str
    inputs: jax.Array
    shape: tuple[int, ...]


def _flat(tree) -> dict[tuple[str, ...], jax.Array]:
    """Every leaf of a tree of mappings, by the path of keys to it."""
    return {tuple(str(getattr(key, "key", key)) for key in path): leaf
            for path, leaf in jax.tree_util.tree_leaves_with_path(tree)}


def _kind(context) -> str | None:
    """How a call maps its input, or None for a call no site reads."""
    module, method = context.module, context.method_name
    if isinstance(module, nn.Dense) and method == "__call__":
        return "dense"
    if isinstance(module, nn.Embed) and method in ("__call__", "attend"):
        return "embed" if method == "__call__" else "attend"
    return None


class NodePerturbation(Objective[Ratio]):
    """Trains the objective it is mixed into without backpropagation (the module's docstring):

        class Forward(NodePerturbation, Supervised):
            draws = 256

    is `Supervised` with its loss's gradient estimated by `draws` antithetic pairs of output jitters of
    scale `sigma` a step, and `weight_draws` pairs of weight jitters of scale `weight_sigma` for the
    parameters no `Dense` or `Embed` call holds. Everything else, the model, its data and its saved task, is
    the objective's own, which comes after this class in the bases."""

    draws: int = 64
    sigma: float = 0.2
    weight_draws: int = 8
    weight_sigma: float = 0.03

    def loss(self, variables: Variables, batch: Batch, step: Step):
        stats = super().loss(variables, batch, step)
        stats, aux = stats if isinstance(stats, tuple) else (stats, None)
        if not step.training:
            return stats if aux is None else (stats, aux)
        if self.draws < 1 or self.weight_draws < 1:
            raise ValueError("node perturbation draws at least one pair of each kind of jitter")
        rule = self.estimate(variables, batch, step)
        ruled = self.with_gradients(stats, jax.tree.unflatten(jax.tree.structure(stats), [rule, None]),
                                    variables["params"])
        return ruled if aux is None else (ruled, aux)

    def units(self, variables: Variables, batch: Batch, step: Step) -> jax.Array:
        """The losses each jitter is credited by: the objective's units' (`unit_losses`), or its total."""
        units = self.unit_losses(variables, batch, step)
        if units is not None:
            return units
        stats = super().loss(variables, batch, step)
        stats = stats[0] if isinstance(stats, tuple) else stats
        if not isinstance(stats, Ratio):
            raise TypeError("node perturbation credits a jitter by a loss: an objective whose statistics are "
                            "no Ratio states its units (`Objective.unit_losses`)")
        return stats.total

    def estimate(self, variables: Variables, batch: Batch, step: Step) -> Variables:
        """The estimated derivative of the units' summed loss in every trained parameter, in their tree."""
        sites: list[Site] = []

        def captured(next_fun, args, kwargs, context):
            output = next_fun(*args, **kwargs)
            kind = _kind(context)
            if kind is not None:
                sites.append(Site(tuple(context.module.path), kind, args[0], tuple(jnp.shape(output))))
            return output

        try:
            with nn.intercept_methods(captured):
                clean = self.units(variables, batch, step)
            gradients = self._sited(variables, batch, step, sites, clean)
        except jax.errors.UnexpectedTracerError as escaped:
            raise ValueError("node perturbation reads each Dense and Embed call's input and output, which a "
                             "model that maps or scans its layers (nn.scan, jax.vmap in its call) hides from "
                             "it; build the model with its layers unrolled") from escaped
        return self._remaining(variables, batch, step, gradients)

    def _jittered(self, variables, batch, step, sites: list[Site], key: jax.Array, sign: float) -> jax.Array:
        """The units' losses with every site's output jittered by `sign` sigma times its draw from `key`."""
        index = iter(range(len(sites)))

        def jittered(next_fun, args, kwargs, context):
            output = next_fun(*args, **kwargs)
            if _kind(context) is None:
                return output
            noise = jax.random.normal(jax.random.fold_in(key, next(index)), jnp.shape(output), output.dtype)
            return output + sign * self.sigma * noise

        with nn.intercept_methods(jittered):
            return self.units(variables, batch, step)

    def _sited(self, variables, batch, step, sites: list[Site], clean: jax.Array
               ) -> dict[tuple[str, ...], jax.Array]:
        """Each site's parameters' estimated gradient, summed over the calls of each module, flat by path."""
        def credited(change: jax.Array, shape: tuple[int, ...]) -> jax.Array:
            # A unit's own change where the output's leading axes are the units', else the whole loss's.
            if shape[:change.ndim] != change.shape:
                change = jnp.sum(change)
            return change.reshape(change.shape + (1,) * (len(shape) - change.ndim))

        def draw(errors, key):
            plus = self._jittered(variables, batch, step, sites, key, 1.0)
            minus = self._jittered(variables, batch, step, sites, key, -1.0)
            change = (plus - minus) / (2 * self.sigma)
            return [error + credited(change, site.shape) * jax.random.normal(
                jax.random.fold_in(key, index), site.shape, error.dtype)
                for index, (error, site) in enumerate(zip(errors, sites, strict=True))], None

        dtype = jnp.result_type(clean, jnp.float32)
        keys = jax.random.split(jax.random.fold_in(step.key, 0x5D57), self.draws)
        errors, _ = jax.lax.scan(draw, [jnp.zeros(site.shape, dtype) for site in sites], keys)
        gradients: dict[tuple[str, ...], jax.Array] = {}
        for site, error in zip(sites, errors, strict=True):
            for name, value in self._reassembled(variables, site, error / self.draws).items():
                where = ("params", *site.path, name)
                gradients[where] = value + gradients[where] if where in gradients else value
        return gradients

    @staticmethod
    def _reassembled(variables, site: Site, error: jax.Array) -> dict[str, jax.Array]:
        """A site's parameters' gradient from its output's estimated error, as backpropagation forms it."""
        held = _flat(variables["params"])
        if site.kind == "dense":
            kernel = held[(*site.path, "kernel")]
            gradient = {"kernel": float32_einsum("...i,...o->io", site.inputs.astype(error.dtype),
                                                 error).astype(kernel.dtype)}
            if (*site.path, "bias") in held:
                gradient["bias"] = jnp.sum(error.reshape(-1, error.shape[-1]), 0).astype(kernel.dtype)
            return gradient
        table = held[(*site.path, "embedding")]
        if site.kind == "attend":
            return {"embedding": float32_einsum("...v,...d->vd", error,
                                                site.inputs.astype(error.dtype)).astype(table.dtype)}
        rows = jnp.zeros(table.shape, error.dtype).at[site.inputs.reshape(-1)].add(
            error.reshape(-1, table.shape[-1]))
        return {"embedding": rows.astype(table.dtype)}

    def _remaining(self, variables, batch, step, gradients: dict[tuple[str, ...], jax.Array]) -> Variables:
        """Every trained parameter's gradient: the sites', and two-sided weight-space evolution strategies'
        for the parameters no site holds, rewarded by the units' summed loss."""
        held = _flat({"params": variables["params"]})
        others = [path for path in held if path not in gradients]
        if others:
            def total(point: dict) -> jax.Array:
                moved = traverse_util.unflatten_dict({**held, **point})
                return jnp.sum(self.units({**variables, **moved}, batch, step))

            def draw(sums, key):
                noise = {path: jax.random.normal(jax.random.fold_in(key, index), held[path].shape,
                                                 jnp.result_type(held[path], jnp.float32))
                         for index, path in enumerate(others)}

                def at(sign: float) -> dict:
                    return {path: held[path] + sign * self.weight_sigma * noise[path] for path in others}
                change = (total(at(1.0)) - total(at(-1.0))) / (2 * self.weight_sigma)
                return {path: sums[path] + change * noise[path] for path in others}, None

            keys = jax.random.split(jax.random.fold_in(step.key, 0xE5), self.weight_draws)
            zeros = {path: jnp.zeros(held[path].shape, jnp.result_type(held[path], jnp.float32))
                     for path in others}
            sums, _ = jax.lax.scan(draw, zeros, keys)
            gradients = {**gradients, **{path: (sums[path] / self.weight_draws).astype(held[path].dtype)
                                         for path in others}}
        return traverse_util.unflatten_dict(gradients)["params"]

