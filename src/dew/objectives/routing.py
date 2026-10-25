"""How a run trains its routers: the one policy every objective composes.

A mixture's routers choose each token's experts (`dew.nn.moe.Router`). A run
trains them by up to three terms, each optional, which `ExpertBalance` holds:
the aux-loss-free balancing bias (`balance_rate`, arXiv 2408.15664), DeepSeek
V2's expert-level balance loss (`aux_loss_alpha` and `seq_aux`), and ST-MoE's
router z-loss (`router_z_loss`, arXiv 2202.08906).

An objective composes the policy with its own loss. Its model call opens the
routers' collection (`dew.nn.moe.ROUTER_COLLECTION`), `compose` adds the
balance and z-loss to its statistics (`RoutedStatistics`) and counts the slots
each router selected for the bias, and the commit applies those counts once
per update (`apply`). The layer's math stays in `dew.nn.moe`: the balance
loss's moments and the bias update's arithmetic. The counting is here, once.
"""

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import struct
from flax.traverse_util import unflatten_dict

from dew.nn.inputs import COORDINATES_FIELD, Coordinates, token_layout
from dew.nn.moe import (
    EXPERT_BIAS,
    EXPERT_BIAS_COLLECTION,
    ROUTER_INDICES,
    ROUTER_LOG_Z,
    ROUTER_SCORES,
    RouterMoments,
    global_router_loss,
    load_balance_update,
    sequence_router_moments,
)
from dew.nn.precision import at_least_fp32
from dew.objectives.base import Batch, Objective, Ratio, Variables, leaves_by_path, merge, sown_nodes

__all__ = ["ExpertBalance", "RoutedStatistics", "apply_routed", "reduce_routed", "router_counts",
           "router_moments", "router_scores", "router_z_terms"]


@struct.dataclass
class RoutedStatistics:
    """A prediction's statistics next to the routers' own terms.

    Each router term normalises by its own count, not by the counted tokens,
    so the terms travel as separate statistics until `ExpertBalance.reduce`
    adds them up.
    """
    prediction: Ratio
    sequence: tuple[Ratio, ...]
    global_routers: tuple[RouterMoments, ...]
    router_z: tuple[Ratio, ...] = ()


@dataclass(frozen=True)
class ExpertBalance:
    """How a run trains its routers, each term optional.

    `balance_rate` moves each router's balancing bias against its load by this
    much at every committed update, which is DeepSeek's aux-loss-free
    balancing. The model keeps the bias when its mixture sets `bias=True`.
    None leaves the bias where it is.

    `aux_loss_alpha` scales DeepSeek V2's expert-level balance loss
    (`dew.nn.moe.global_router_loss`), summed over the routers. `seq_aux`
    chooses its per-sequence form (`dew.nn.moe.sequence_router_moments`): a
    packed row's sequences are its documents, each balanced alone, and their
    losses are averaged, as DeepSeek defines it. Megatron-LM and MaxText
    balance a packed row as one sequence, so their packed loss differs. None
    adds no balance loss.

    `router_z_loss` is the routers' own z-loss (ST-MoE): this coefficient times
    the squared log partition of every router's gate logits, averaged over the
    positions each router saw in the step and summed over the routers.
    lm-engine's MoE adds 0.1 of it to its switch loss before `router_aux_loss_coef`
    scales the sum, which matches `router_z_loss = 0.1 * aux_loss_alpha` here. Zero
    adds nothing.
    """

    balance_rate: float | None = None
    aux_loss_alpha: float | None = None
    seq_aux: bool = True
    router_z_loss: float = 0.0

    def __post_init__(self) -> None:
        if self.balance_rate is not None and not (0 < self.balance_rate < float("inf")):
            raise ValueError(
                f"balance_rate moves each bias by this much per update, so it is positive and finite, "
                f"got {self.balance_rate}; None leaves the bias where it is")
        if self.aux_loss_alpha is not None and not (0 < self.aux_loss_alpha < float("inf")):
            raise ValueError(
                f"aux_loss_alpha scales the balance loss, so it is positive and finite, got "
                f"{self.aux_loss_alpha}; None adds no balance loss")
        if not (0 <= self.router_z_loss < float("inf")):
            raise ValueError(
                f"router_z_loss weights the routers' squared log partitions, so it is finite and "
                f"nonnegative, got {self.router_z_loss}; 0 adds nothing")

    def terms(self, prediction: Ratio, routing: Variables, batch: Batch, queries: jax.Array,
              fields: Mapping[str, jax.Array | Coordinates]
              ) -> tuple[Ratio | RoutedStatistics, dict[str, jax.Array]]:
        """The statistics the balance loss and the z-loss add to `prediction`, and the metrics they report.

        `routing` is what the routers sowed, `queries` marks the router
        positions that count, and `fields` are the token fields of the batch's
        rows, which name each row's documents and the packed width of a
        coordinates column.
        """
        documents = token_layout(fields)[1]
        statistics: Ratio | RoutedStatistics = prediction
        reported: dict[str, jax.Array] = {}
        if self.aux_loss_alpha is not None:
            statistics, reported["aux_loss"] = self._balance_loss(
                self.aux_loss_alpha, prediction, routing, batch, queries, fields)
        if self.router_z_loss:
            router_z = router_z_terms(routing, self.router_z_loss, batch, queries, documents)
            if not router_z:
                raise ValueError(
                    "router_z_loss needs routers that sow their gate's log partition "
                    "(dew.nn.moe.Router); this model's routers sow none")
            statistics = (RoutedStatistics(prediction, (), (), router_z) if isinstance(statistics, Ratio)
                          else dataclasses.replace(statistics, router_z=router_z))
            reported["router_z_loss"] = jnp.sum(jnp.stack([term.mean()[0] for term in router_z]))
        return statistics, reported

    def compose(self, params: Variables, prediction: Ratio, routing: Variables, batch: Batch,
                queries: jax.Array, fields: Mapping[str, jax.Array | Coordinates]
                ) -> tuple[Ratio | RoutedStatistics, Variables | None, dict[str, jax.Array]]:
        """The statistics an objective's `prediction` takes from this policy, the router counts its commit
        applies (None where `balance_rate` is unset), and the metrics both report.

        `params` holds the biases the counts are read against (`effects`); `routing`, `queries` and `fields`
        are the forward's, as `terms` and `effects` take them.
        """
        statistics, balanced = self.terms(prediction, routing, batch, queries, fields)
        effects, load = self.effects(params, routing, queries, fields)
        return statistics, effects, {**balanced, **load}

    def _balance_loss(self, alpha: float, prediction: Ratio, routing: Variables, batch: Batch,
                      queries: jax.Array, fields: Mapping[str, jax.Array | Coordinates]
                      ) -> tuple[RoutedStatistics, jax.Array]:
        """The balance loss's statistics beside the prediction's, and the loss they add to it.

        The reported number is what the terms add to the prediction's loss.
        """
        if not routing:
            raise ValueError("aux_loss_alpha requires a model with a mixture")
        documents = token_layout(fields)[1]
        laid = fields.get(COORDINATES_FIELD)
        max_segments = laid.max_segments if isinstance(laid, Coordinates) else None
        routers = router_scores(routing)
        sequence = []
        if self.seq_aux:
            for scores, indices in routers:
                valid = _router_queries(scores.shape[:2], queries, documents)
                moments = sequence_router_moments(
                    scores, indices, valid, None if documents is None else documents[:, :scores.shape[1]],
                    max_segments=max_segments)
                losses = global_router_loss(moments, alpha)
                supported = (moments.positions > 0).astype(losses.dtype)
                sequence.append(Ratio(Objective.row_mean(losses, batch).total,
                                      Objective.row_mean(supported, batch).total))
        global_routers = (() if self.seq_aux else
                          tuple(router_moments(s, i, batch, queries, documents) for s, i in routers))
        statistics = RoutedStatistics(prediction, tuple(sequence), global_routers)
        combined, _ = self.reduce(statistics)
        prediction_loss, _ = prediction.mean()
        return statistics, combined - prediction_loss

    def effects(self, params: Variables, routing: Variables, queries: jax.Array,
                fields: Mapping[str, jax.Array | Coordinates]
                ) -> tuple[Variables | None, dict[str, jax.Array]]:
        """The bias counts of the routers the forward ran, and the spread of their load.

        The counts are the effect the commit applies once per update (`apply`).
        The report is each router's busiest and idlest share of the batch. Both
        are None where `balance_rate` is unset.
        """
        if self.balance_rate is None:
            return None, {}
        if EXPERT_BIAS_COLLECTION not in params:
            raise ValueError("balance_rate requires a mixture with bias=True")
        counts = router_counts(params[EXPERT_BIAS_COLLECTION], routing, queries, token_layout(fields)[1])
        shares = [count / jnp.maximum(jnp.sum(count), 1) for count in jax.tree.leaves(counts)]
        if not shares:
            raise ValueError("balance_rate requires routers that keep a balancing bias, and these select "
                             "by their hash table alone")
        return counts, {"moe/max_load": jnp.mean(jnp.stack([x.max() for x in shares])),
                        "moe/min_load": jnp.mean(jnp.stack([x.min() for x in shares]))}

    def reduce(self, stats: RoutedStatistics) -> tuple[jax.Array, jax.Array]:
        """The objective value of routed statistics, and whether they have any mass.

        The value is the prediction's plus every router term's.
        """
        value, active = stats.prediction.mean()
        for term in stats.sequence:
            auxiliary, supported = term.mean()
            value, active = value + auxiliary, active | supported
        for term in stats.global_routers:
            if self.aux_loss_alpha is None:
                raise ValueError("global router statistics require aux_loss_alpha")
            value = value + global_router_loss(term, self.aux_loss_alpha)
            active = active | (term.positions > 0)
        for term in stats.router_z:
            auxiliary, supported = term.mean()
            value, active = value + auxiliary, active | supported
        return value, active

    def apply(self, variables: Variables, effects: Variables) -> Variables:
        """`variables` with each counted router's bias moved by its counts (`dew.nn.moe.load_balance_update`).

        Only the `moe` collection is returned, which is the one a balancing step rewrites.
        """
        if self.balance_rate is None:
            raise ValueError("router count effects require balance_rate")
        moe = variables[EXPERT_BIAS_COLLECTION]
        biases = leaves_by_path(moe)
        balanced = {path: _updated_bias(biases[path], count, self.balance_rate)
                    for path, count in leaves_by_path(effects).items()}
        return {EXPERT_BIAS_COLLECTION: merge(moe, unflatten_dict(balanced))}


def reduce_routed(balance: ExpertBalance | None,
                  stats: Ratio | RoutedStatistics) -> tuple[jax.Array, jax.Array]:
    """The objective value of a loss's statistics, and whether they have any mass.

    A `Ratio` reduces by its own mean, and routed statistics by `balance` (`ExpertBalance.reduce`), which
    a routed loss has to name.
    """
    if isinstance(stats, Ratio):
        return stats.mean()
    if balance is None:
        raise ValueError("routed statistics require a balance (ExpertBalance)")
    return balance.reduce(stats)


def apply_routed(balance: ExpertBalance | None, variables: Variables, effects: Variables) -> Variables:
    """`variables` with the router counts `balance` took applied (`ExpertBalance.apply`)."""
    if balance is None:
        raise ValueError("router count effects require a balance (ExpertBalance)")
    return balance.apply(variables, effects)


def _router_queries(shape: tuple[int, ...], queries: jax.Array | None,
                    documents: jax.Array | None) -> jax.Array:
    """The real queries of a router, including a prediction depth's shifted inputs."""
    if queries is None:
        return jnp.ones(shape, bool)
    width = shape[1]
    valid = queries[:, :width].astype(bool)
    for offset in range(1, queries.shape[1] - width + 1):
        valid = valid & queries[:, offset:offset + width].astype(bool)
        if documents is not None:
            valid = valid & (documents[:, :width] == documents[:, offset:offset + width])
    return valid


def _selected_counts(indices: jax.Array, valid: jax.Array, experts: int) -> jax.Array:
    """Each row's integer expert counts over its real queries."""
    weights = jnp.broadcast_to(valid[..., None], indices.shape).astype(jnp.int32)
    return jax.vmap(lambda ids, weight: jnp.bincount(ids.ravel(), weights=weight.ravel(),
                                                   length=experts))(indices, weights)


def router_counts(moe: Variables, routing: Variables, queries: jax.Array | None = None,
                  documents: jax.Array | None = None) -> Variables:
    """Count the slots each router the forward ran selected, at the path of the balancing bias it keeps in
    `moe`.

    The routers that ran are the ones `routing` holds sown `indices` for, so
    a router the forward skipped (a prediction depth left out, a draft stage)
    counts nothing; one that selects by a hash table keeps no bias and counts
    nothing either.
    """
    biases = leaves_by_path(moe)
    counts = {}
    for path, node in sown_nodes(routing, ROUTER_INDICES):
        if (*path, EXPERT_BIAS) in biases:
            (indices,) = node[ROUTER_INDICES]
            valid = _router_queries(indices.shape[:2], queries, documents)
            counts[(*path, EXPERT_BIAS)] = jnp.sum(
                _selected_counts(indices, valid, biases[(*path, EXPERT_BIAS)].shape[0]), axis=0)
    return unflatten_dict(counts)


def _updated_bias(bias: jax.Array, counts: jax.Array, rate: float) -> jax.Array:
    dtype = jnp.result_type(bias.dtype, rate, jnp.float32)
    correction = load_balance_update(counts, jnp.asarray(rate, dtype))
    return (bias.astype(dtype) + correction).astype(bias.dtype)


def router_z_terms(routing: Variables, weight: float, batch: Batch, queries: jax.Array | None = None,
                   documents: jax.Array | None = None) -> tuple[Ratio, ...]:
    """Return ST-MoE's router z-loss, one `Ratio` per router.

    This is eq. 5 of ST-MoE (arXiv 2202.08906): `weight` times the squared log
    partition of the gate logits, summed over the positions the router saw and
    divided by their count. The count adds across micro-batches, so a step's term
    is the mean over its routed positions, and the routers' terms add up, as
    lm-engine's per-layer `(logsumexp(logits) ** 2).mean()` does. The positions
    are `batch`'s rows' (`Objective.row_mean`).
    """
    terms = []
    # lm-engine: moe/module.py at 45b6b57b, before its 0.1 and router_aux_loss_coef.
    for _, node in sown_nodes(routing, ROUTER_LOG_Z):
        (log_z,) = node[ROUTER_LOG_Z]
        valid = _router_queries(log_z.shape, queries, documents)
        dtype = at_least_fp32(log_z.dtype)
        total = Objective.row_mean(jnp.where(valid, jnp.square(log_z.astype(dtype)), 0), batch).total
        mass = Objective.row_mean(valid.astype(dtype), batch).total
        terms.append(Ratio(weight * total, mass))
    return tuple(terms)


def router_moments(scores: jax.Array, indices: jax.Array, batch: Batch,
                   queries: jax.Array | None = None, documents: jax.Array | None = None) -> RouterMoments:
    """Return one router's routed-position statistics over `batch`'s rows (`Objective.row_mean`).

    `scores` is `[rows, positions, experts]` and `indices` the experts each
    position chose, `[rows, positions, top_k]`.
    """
    experts = scores.shape[-1]
    dtype = at_least_fp32(scores.dtype)
    valid = _router_queries(scores.shape[:2], queries, documents)
    # These counts enter a floating loss, unlike the exact integer bias effects.
    summed = Objective.row_mean(jnp.where(valid[..., None], scores.astype(dtype), 0), batch, axis=(0, 1))
    chosen = _selected_counts(indices, valid, experts)
    counts = Objective.row_mean(chosen, batch, axis=0).total.astype(dtype)
    positions = Objective.row_mean(valid.astype(dtype), batch).total
    return RouterMoments(summed.total, counts, positions, indices.shape[-1])


def router_scores(routing: Variables) -> list[tuple[jax.Array, jax.Array]]:
    """Collect every router's sown (scores, indices), in the tree's key order."""
    found: list[tuple[jax.Array, jax.Array]] = []
    for _, node in sown_nodes(routing, ROUTER_SCORES, ROUTER_INDICES):
        (scores,), (indices,) = node[ROUTER_SCORES], node[ROUTER_INDICES]
        found.append((scores, indices))
    return found
