"""A recurrent mixer over documents that continue another (`dew.batch.Forks`: a request's questions over one
state, a prompt's completions, a tree's candidates) against each document's own row: its chain of ancestors
and itself (row 1506)."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from reference_error import assert_as_exact

from dew.batch import Forks, coordinates
from dew.nn.inputs import AttentionMetadata, MixerCall

LAYOUTS = {
    # Two requests, their documents numbered out of their order in the row: id 2 a shared part of 5 that
    # ids 3 and 1 continue, then id 5 a shared part of one token, shorter than a conv's history, that id 4
    # continues; two padding slots.
    "permuted": ([(2, 5), (3, 3), (1, 2), (5, 1), (4, 3)], [2, 0, 2, 5, 0], 2),
    # One token a document and no padding: as many documents as the row has slots, three shared parts of
    # one token each continued by one of one token, in turn.
    "one token each": ([(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)], [0, 1, 0, 3, 0, 5], 0),
    # A chain three deep, a tree's: id 1 of 3, id 2 of one token continuing it, ids 3 and 4 continuing id 2,
    # id 5 continuing id 3: a conv's history reaches through a short parent into its grandparent.
    "tree": ([(1, 3), (2, 1), (3, 2), (4, 1), (5, 2)], [0, 1, 2, 2, 3], 1),
}
"""Each layout's documents in the row's order (id, length), each id's parent id (0 for none), padding."""

FORKED_SEEDS = 32
"""Check 1's declared seeds for documents continuing another's memory and history against each one's row: at
12, the one-token layout's parameter gradients, sums over a few tokens, were left unresolved over 60."""

FORKED_ROWS = 8
"""The rows a draw lays its layout out in, each its own draw: a parameter's gradient sums over every row's
tokens, where over the one-token layout's six alone Mamba-2's `A_log` was left unresolved over 160 seeds,
its draws' log ratios spread 0.71 (check 1 asks for larger inputs there, not more seeds)."""


def assert_forks_as_own_rows(module, variables, width: int, layout: str, label: str) -> None:
    """`module` over the row `LAYOUTS[layout]` lays out gives each document's tokens, and the gradients of
    a drawn cotangent on them with respect to the row and the parameters, what its own row of its chain of
    ancestors and itself gives (check 1); padding comes back zero."""
    placed, enters, padding = LAYOUTS[layout]
    documents = np.concatenate([*(np.full(length, number, np.int32) for number, length in placed),
                                np.zeros(padding, np.int32)])
    valid = documents > 0
    rows = np.tile(documents, (FORKED_ROWS, 1))
    tokens = coordinates(valid=jnp.asarray(rows > 0), segments=jnp.asarray(rows), max_segments=len(enters))
    parents = np.asarray([enters] * FORKED_ROWS, np.int32)
    forks = Forks(jnp.asarray(parents), depth=Forks.depth_of(parents))
    call = MixerCall(AttentionMetadata(coordinates=dataclasses.replace(tokens, forks=forks)))
    begins = dict(zip((number for number, _ in placed), np.cumsum([0, *(length for _, length in placed)]),
                      strict=False))
    lengths = dict(placed)

    def draw(seed):
        rng = np.random.default_rng(seed)
        hidden = rng.normal(size=(FORKED_ROWS, len(documents), width))
        return (jnp.asarray(hidden, jnp.float32), jnp.asarray(rng.normal(size=hidden.shape), jnp.float32))

    def forked(params, hidden):
        return module.apply({**variables, "params": params}, hidden, call)

    # Each document's own row, its chain of ancestors before it, every one a row of one call padded at its
    # end: a token's output and hidden gradient its own row's, and a parameter's gradient one reduction over
    # every document's tokens, as the forked arm's is. Each document's own call's put the one-token layout's
    # parameter gradients a sum of six reductions against the forked arm's one, its order, not the
    # documents', in the ratio (Kimi delta attention's `o_norm` weight 1.6 times over 32 rows of it).
    def chain(number):
        above = []
        while enters[number - 1]:
            number = enters[number - 1]
            above.append(number)
        return above[::-1]

    shared_of = {number: sum(lengths[above] for above in chain(number)) for number, _ in placed}
    longest = max(shared_of[number] + length for number, length in placed)
    owned = np.arange(longest)[None, :] < np.asarray([[shared_of[n] + length] for n, length in placed])
    owned = np.repeat(owned, FORKED_ROWS, axis=0)
    own_rows = MixerCall(AttentionMetadata(coordinates=coordinates(
        valid=jnp.asarray(owned), segments=jnp.asarray(owned.astype(np.int32)))))

    def apart(params, hidden):
        rows = []
        for number, length in placed:
            pieces = [hidden[:, begins[above]:begins[above] + lengths[above]] for above in chain(number)]
            row = jnp.concatenate([*pieces, hidden[:, begins[number]:begins[number] + length]], axis=1)
            rows.append(jnp.pad(row, ((0, 0), (0, longest - row.shape[1]), (0, 0))))
        found = module.apply({**variables, "params": params}, jnp.concatenate(rows, axis=0), own_rows)
        out = jnp.zeros_like(hidden)
        for index, (number, length) in enumerate(placed):
            row = found[index * FORKED_ROWS:(index + 1) * FORKED_ROWS]
            out = out.at[:, begins[number]:begins[number] + length].set(
                row[:, shared_of[number]:shared_of[number] + length])
        return out

    def outputs(run):
        return jax.jit(lambda hidden, _: run(variables["params"], hidden))

    def gradients(run):
        def grads(hidden, cotangent):
            params, rows = jax.grad(lambda params, rows: jnp.sum(run(params, rows) * cotangent),
                                    argnums=(0, 1))(variables["params"], hidden)
            return {"params": params, "hidden": rows}
        return jax.jit(grads)

    assert_as_exact(outputs(forked), outputs(apart), draw, f"{label}, {layout}", seeds=FORKED_SEEDS)
    assert_as_exact(gradients(forked), gradients(apart), draw, f"{label}'s gradients, {layout}",
                    seeds=FORKED_SEEDS)
    assert not np.any(np.asarray(outputs(forked)(*draw(0)))[:, ~valid])
