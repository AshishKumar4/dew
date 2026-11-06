"""A recurrent mixer over documents that enter another (`dew.batch.Entered`, the questions of a row over one
state) against each document's own row: the part it enters and itself (row 1506)."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from reference_error import assert_as_exact

from dew.batch import Entered, coordinates
from dew.nn.inputs import AttentionMetadata, MixerCall

LAYOUTS = {
    # Two requests, their documents numbered out of their order in the row: id 2 a shared part of 5 that
    # ids 3 and 1 enter, then id 5 a shared part of one token, shorter than a conv's history, that id 4
    # enters; two padding slots.
    "permuted": ([(2, 5), (3, 3), (1, 2), (5, 1), (4, 3)], [2, 0, 2, 5, 0], 2),
    # One token a document and no padding: as many documents as the row has slots, three shared parts of
    # one token each entered by one of one token, in turn.
    "one token each": ([(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)], [0, 1, 0, 3, 0, 5], 0),
}
"""Each layout's documents in the row's order (id, length), each id's entered id (0 for none), padding."""

ENTERED_SEEDS = 32
"""Check 1's declared seeds for documents entering another's memory and history against each one's row: at
12, the one-token layout's parameter gradients, sums over a few tokens, were left unresolved over 60."""


def assert_entered_as_own_rows(module, variables, width: int, layout: str, label: str) -> None:
    """`module` over the row `LAYOUTS[layout]` lays out gives each document's tokens, and the gradients of
    a drawn cotangent on them with respect to the row and the parameters, what its own row of the part it
    enters and itself gives (check 1); padding comes back zero."""
    placed, enters, padding = LAYOUTS[layout]
    documents = np.concatenate([*(np.full(length, number, np.int32) for number, length in placed),
                                np.zeros(padding, np.int32)])
    valid = documents > 0
    tokens = coordinates(valid=jnp.asarray(valid[None]), segments=jnp.asarray(documents[None]))
    entered = Entered(jnp.asarray(documents[None]), jnp.asarray([enters], jnp.int32))
    call = MixerCall(AttentionMetadata(coordinates=dataclasses.replace(tokens, entered=entered)))
    begins = dict(zip((number for number, _ in placed), np.cumsum([0, *(length for _, length in placed)]),
                      strict=False))
    lengths = dict(placed)

    def draw(seed):
        rng = np.random.default_rng(seed)
        hidden = rng.normal(size=(1, len(documents), width))
        return (jnp.asarray(hidden, jnp.float32), jnp.asarray(rng.normal(size=hidden.shape), jnp.float32))

    def entering(params, hidden):
        return module.apply({**variables, "params": params}, hidden, call)

    def apart(params, hidden):
        out = jnp.zeros_like(hidden)
        for number, length in placed:
            own = hidden[:, begins[number]:begins[number] + length]
            root = enters[number - 1]
            shared = hidden[:, begins[root]:begins[root] + lengths[root]] if root else hidden[:, :0]
            row = module.apply({**variables, "params": params}, jnp.concatenate([shared, own], axis=1))
            out = out.at[:, begins[number]:begins[number] + length].set(row[:, shared.shape[1]:])
        return out

    def outputs(run):
        return jax.jit(lambda hidden, _: run(variables["params"], hidden))

    def gradients(run):
        def grads(hidden, cotangent):
            params, rows = jax.grad(lambda params, rows: jnp.sum(run(params, rows) * cotangent),
                                    argnums=(0, 1))(variables["params"], hidden)
            return {"params": params, "hidden": rows}
        return jax.jit(grads)

    assert_as_exact(outputs(entering), outputs(apart), draw, f"{label}, {layout}", seeds=ENTERED_SEEDS)
    assert_as_exact(gradients(entering), gradients(apart), draw, f"{label}'s gradients, {layout}",
                    seeds=ENTERED_SEEDS)
    assert not np.any(np.asarray(outputs(entering)(*draw(0)))[:, ~valid])
