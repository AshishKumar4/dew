"""The decoding components the ranks of a pool may hold apart, each of which the pool refuses before any
rank enters a collective (`multiprocess_worker.mode_decoding_components`): a criterion's payload, a
transform's identity, a transform's payload, a function's globals and a mapping's order (row 12171)."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils

from dew.sampling import decoding

RANKED_SHIFT = [4.0]
"""What `shifted_by_a_global` adds, a module global each rank may set apart: a function's globals are what it
computes, which a pool's agreement states (row 12171)."""


def shifted_by_a_global(state, scores):
    """Scores with `RANKED_SHIFT` added to their last token, the function the same name on every rank."""
    del state
    return scores.at[:, -1].add(RANKED_SHIFT[0])


def weighted_in_order(state, scores, *, weights):
    """Scores with each of `weights`' values added to its token in the order the mapping holds them, a
    later one's write over an earlier's at one token."""
    del state
    for token, weight in weights.items():
        scores = scores.at[:, token % scores.shape[-1]].set(weight)
    return scores


def divergences(chain: tuple, raise_last, rank: int) -> dict[str, tuple[tuple, tuple]]:
    """Each way rank 1's policy differs from the others' with every shape alike, by name: its logits chain
    and its stopping criteria."""
    return {
        "criterion payload": (
            chain,
            (decoding.EndOfSequence(jnp.asarray([3 if rank == 1 else 5], jnp.int32)),),
        ),
        "transform identity": (
            (
                *chain[:1],
                jax.tree_util.Partial(
                    (lambda state, scores: scores.at[:, 0].add(4.0)) if rank == 1 else raise_last
                ),
                *chain[2:],
            ),
            (),
        ),
        "transform payload": (
            (
                decoding.RepetitionPenalty(1.3),
                decoding.SuppressTokens(jnp.asarray([2 if rank == 1 else 6], jnp.int32)),
            ),
            (),
        ),
        # One function by one name on every rank, its global set apart on rank 1: its name and cells
        # alike, what it computes not (row 12171).
        "transform globals": ((*chain[:1], jax.tree_util.Partial(shifted_by_a_global), *chain[2:]), ()),
        # One mapping's two orders, whose later write at one token wins: the same pairs, another policy.
        "transform mapping order": (
            (*chain[:1], jax.tree_util.Partial(weighted_in_order, weights=(
                {1: 2.0, 14: 3.0} if rank == 1 else {14: 3.0, 1: 2.0})), *chain[2:]),
            (),
        ),
    }


def refused_divergences(task, request, chain: tuple, raise_last) -> list[str]:
    """The names of the `divergences` a pool's request under `task` refused, each refused on every rank,
    which all return from it to the rendezvous after: a peer's divergence accepted raises."""
    rank, processes = jax.process_index(), jax.process_count()
    RANKED_SHIFT[0] = 5.0 if rank == 1 else 4.0
    refused = []
    for name, (logits, stopping) in divergences(chain, raise_last, rank).items():
        try:
            task(request, 4, key=7,
                 sampling=dataclasses.replace(task.sampling, logits=logits, stopping=stopping))
        except (ValueError, RuntimeError, AssertionError) as failure:
            refused.append(name)
            del failure
        else:
            raise AssertionError(f"a peer's different {name} was accepted")
        # A rank stranded in a collective never reaches this rendezvous.
        arrivals = multihost_utils.process_allgather(np.asarray(rank, np.int32))
        if arrivals.tolist() != list(range(processes)):
            raise AssertionError("a rank did not return from the rejected request")
    return refused
