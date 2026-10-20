"""The delta rule's port held where it rounds (bible §7's source-aware gate,
tests/reference_error.py's `assert_graph_as_exact`): each value its kernels hold (`port_graph`)
against Dew's XLA reference of the same math on common parents (tests/delta_rule_support.py's
`port_reference`), on the GPU where the kernels run."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from delta_rule_support import port_reference
from reference_error import assert_graph_as_exact
from reference_trees import evaluated, sum_family, tree_sum

from dew.nn import linear
from dew.nn.kernels import fla_gdn, fla_kda, generation
from dew.nn.kernels.graph import Graph, Node, Ordered, Pointwise, Sum
from dew.nn.linear import l2norm


def port_graph(per_key: bool, shape: tuple[int, ...], scale: float) -> Graph:
    """The delta rule's port as the values it rounds and holds on one sequence of whole chunks, each
    value laid out a chunk at a time `[B, N, C, H, ...]`: the prefix (`fla_gdn.chunk_local_cumsum`),
    the inverse (and Kimi's scores) of the intra-chunk solve, `w` and `u`, Kimi's gated keys, the
    memory each chunk enters (laid out by chunk `[B, N, H, K, V]`, which the kernels hold at its slot),
    its corrected values and the last, and the output, each node the port's own kernel on its given
    parents and the witness the port's forward (`_rule_fwd`) exposing what it holds."""
    batch, count, chunk = shape[:3]
    port = fla_kda if per_key else fla_gdn
    spans = fla_gdn.sequences(batch, count * chunk, chunk=chunk)
    kernels = {"chunk": chunk, "spans": spans, "interpret": not generation.triton_runs()}
    # The memory each chunk enters, held at its slot (`fla_gdn.Sequences`), laid out by chunk.
    entering = np.asarray(spans.chunks)[:, 2]
    stored = fla_gdn.slots(spans, batch, count * chunk, chunk)

    def by_chunk(states):
        return states[entering].reshape(batch, count, *states.shape[1:])

    def by_slot(states):
        states = jnp.asarray(states).reshape(batch * count, *states.shape[2:])
        return jnp.zeros((stored, *states.shape[1:]), states.dtype).at[entering].set(states)

    def flat(x):
        x = jnp.asarray(x)
        return x.reshape(batch, count * chunk, *x.shape[3:])

    def chunked(x):
        return x.reshape(batch, count, chunk, *x.shape[2:])

    def node(name, parents, kind, fn):
        def run(**given):
            return fn(*(given[parent] for parent in parents))
        return Node(name, parents, kind, jax.jit(run))

    def h0(heads, keys, values):
        return jnp.zeros((batch, heads, keys, values), jnp.float32)

    def recurrence(keys, w, u, prefix):
        states, corrected, final = fla_gdn.chunk_fwd_h(flat(keys), flat(w), flat(u), flat(prefix),
                                                       h0(u.shape[3], keys.shape[-1], u.shape[-1]), **kernels)
        return {"states": by_chunk(states), "corrected": chunked(corrected), "final": final}

    key_width = Ordered({"k": -1, "w": -1, **({"prefix": -1} if per_key else {})})
    if per_key:
        key_width = Ordered({"kg": -1, "w": -1, "prefix": -1})
    remembered = "kg" if per_key else "k"
    nodes = [node("prefix", ("g",), Ordered(), lambda g: chunked(fla_gdn.chunk_local_cumsum(
        flat(g), scale=fla_gdn.RCP_LN2, **kernels)))]
    if per_key:
        def solved(q, k, prefix, beta):
            return port.intra_solve(flat(q), flat(k), flat(prefix), flat(beta), scale=scale, **kernels)

        keyed = {"q": -1, "k": -1, "prefix": -1}
        nodes += [node("scores", ("q", "k", "prefix", "beta"), Sum(keyed),
                       lambda *given: chunked(solved(*given)[0])),
                  node("inverse", ("q", "k", "prefix", "beta"), Sum(keyed),
                       lambda *given: chunked(solved(*given)[1]))]
    else:
        nodes.append(node("inverse", ("k", "prefix", "beta"), Sum({"k": -1}),
                          lambda k, prefix, beta: chunked(port.kkt_solve(flat(k), flat(prefix), flat(beta),
                                                                         **kernels))))

    def w_u(k, v, beta, inverse, prefix):
        return port.recompute_w_u(flat(k), flat(v), flat(beta), flat(inverse), flat(prefix), **kernels)

    parents = ("k", "v", "beta", "inverse", "prefix")
    # A chunk's positions, permuted together in the inverse's columns and the operands' rows, reorder
    # each dense product's sum over them and nothing else.
    nodes += [node("w", parents, Sum({"inverse": -1, "k": 2, "beta": 2, "prefix": 2}),
                   lambda *given: chunked(w_u(*given)[0])),
              node("u", parents, Sum({"inverse": -1, "v": 2, "beta": 2}),
                   lambda *given: chunked(w_u(*given)[1]))]
    if per_key:
        nodes.append(node("kg", parents, Pointwise(), lambda *given: chunked(w_u(*given)[2])))
    for name, follows in (("states", {-2: remembered}), ("corrected", {}), ("final", {-2: remembered})):
        nodes.append(node(name, (remembered, "w", "u", "prefix"),
                          Ordered(key_width.axes, follows),
                          lambda *given, name=name: recurrence(*given)[name]))
    if per_key:
        nodes.append(node("output", ("q", "prefix", "scores", "states", "corrected"),
                          Sum({"q": -1, "prefix": -1, "states": -2}),
                          lambda q, prefix, scores, states, corrected: chunked(port.chunk_fwd_o(
                              flat(q), flat(corrected), flat(prefix), flat(scores), by_slot(states),
                              scale=scale, **kernels))))
    else:
        nodes.append(node("output", ("q", "k", "prefix", "states", "corrected"),
                          Sum({"q": -1, "k": -1, "states": -2}),
                          lambda q, k, prefix, states, corrected: chunked(port.chunk_fwd_o(
                              flat(q), flat(k), flat(corrected), by_slot(states), flat(prefix),
                              scale=scale, **kernels))))

    @jax.jit
    def witness(q, k, v, g, beta):
        heads, keys, values = v.shape[3], k.shape[-1], v.shape[-1]
        (out, final), saved = port._rule_fwd(flat(q), flat(k), flat(v), flat(g), flat(beta),
                                             h0(heads, keys, values), None, scale, chunk,
                                             kernels["interpret"])
        held = {"output": chunked(out), "final": final, "prefix": chunked(saved[3])}
        if per_key:
            w, kg, _, states, corrected = saved[8]
            return {**held, "scores": chunked(saved[5]), "inverse": chunked(saved[6]), "w": chunked(w),
                    "kg": chunked(kg), "states": by_chunk(states), "corrected": chunked(corrected)}
        w, states, corrected = saved[7]
        return {**held, "inverse": chunked(saved[5]), "w": chunked(w), "states": by_chunk(states),
                "corrected": chunked(corrected)}

    return Graph(("q", "k", "v", "g", "beta"), tuple(nodes), ("output", "final"),
                 lambda **given: witness(**given))


def reference_graph(dew: Graph, per_key: bool, scale: float) -> Graph:
    """`dew`'s graph with each node the XLA reference's arithmetic (`port_reference`) on the same
    parents, each Sum and Ordered node's members that arithmetic with every sum under each tree of the
    canonical Sum family (`reference_trees.sum_family`, the kernels' 64-wide blocks of keys and of a
    chunk's positions): the prefix's each of its elements, a sum of its chunk's earlier terms, the
    others every sum they make (`reference_trees.evaluated`); and the witness the nodes composed."""
    arithmetic = port_reference(per_key, scale)
    family = sum_family(linear.CHUNK_SIZE)

    def under(fn, parents, tree):
        """`fn` of the named `parents` with every sum it makes under `tree`."""
        ordered = evaluated(lambda *values: fn(**dict(zip(parents, values, strict=True))), tree)
        return lambda **given: ordered(*(given[parent] for parent in parents))

    def summed(tree):
        """The prefix with each element its own sum of its chunk's terms up to it under `tree`, the
        later terms masked to zero in place."""
        def prefix(g):
            g = jnp.asarray(g)
            g = g.astype(jnp.promote_types(g.dtype, jnp.float32))
            chunk = g.shape[2]
            earlier = jnp.arange(chunk)[:, None] >= jnp.arange(chunk)[None, :]
            terms = jnp.where(earlier.reshape(1, 1, chunk, chunk, *(1,) * (g.ndim - 3)), g[:, :, None], 0.0)
            return tree_sum(terms, 3, tree) * fla_gdn.RCP_LN2
        return prefix

    nodes = tuple(Node(node.name, node.parents, node.kind,
                       lambda fn=arithmetic[node.name], **given: fn(**given),
                       members={} if isinstance(node.kind, Pointwise) else
                       {tree: summed(tree) if node.name == "prefix" else
                        under(arithmetic[node.name], node.parents, tree) for tree in family})
                  for node in dew.nodes)

    def fused(**given):
        values = dict(given)
        for node in nodes:
            values[node.name] = node.fn(**{parent: values[parent] for parent in node.parents})
        return {node.name: values[node.name] for node in nodes}

    return Graph(dew.inputs, nodes, dew.outputs, fused)


@pytest.mark.gpu(reason="a kernel's graph is held where it runs, its witness the compiled kernel")
@pytest.mark.parametrize("per_key", [False, True], ids=["gated delta net", "Kimi delta attention"])
def test_the_port_rounds_where_its_reference_does(per_key):
    """A sequence on the port held where it rounds (bible §7's source-aware gate,
    `assert_graph_as_exact`), each value its kernels hold (`port_graph`) against the XLA reference's
    arithmetic on common parents (`port_reference`), the reference's and the port's own, each element
    against float64 from the same parents widened: the decay cumulated within each chunk, its
    elements' sums under the Sum family; the intra-chunk solve's sums over the key width under its
    orders; `w` and `u`, each a chunk's positions' sum under their orders; the memory across chunks,
    ordered, under the key width's orders; the output under them too. The sums the port's kernels keep
    inside, the solve's substitution, the memory's write over a chunk and the output's causal sum,
    keep their order on both sides. End to end, over orders of the key and value dimensions. A packed
    row's documents run as sequences of their own, each as this one (the isolation tests above)."""
    batch, length, heads, width, chunk = 1, 128, 2, 64, linear.CHUNK_SIZE
    count = length // chunk
    keys = jax.random.split(jax.random.key(37), 5)
    q = 0.3 * jax.random.normal(keys[0], (batch, length, heads, width))
    k = l2norm(jax.random.normal(keys[1], (batch, length, heads, width)))
    v = jax.random.normal(keys[2], (batch, length, heads, width))
    g = -0.1 * jax.nn.softplus(jax.random.normal(keys[3], (batch, length, heads, *(width,) * per_key)))
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (batch, length, heads)))
    inputs = {name: np.asarray(x).reshape(batch, count, chunk, *x.shape[2:])
              for name, x in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta))}
    scale = width ** -0.5
    dew = port_graph(per_key, (batch, count, chunk), scale)
    reference = reference_graph(dew, per_key, scale)

    def orbit(order):
        rng = np.random.default_rng(order)
        units = [np.arange(width) if order == 0 else rng.permutation(width) for _ in range(2)]
        moved = dict(inputs)
        for name in ("q", "k", *("g",) * per_key):
            moved[name] = np.take(moved[name], units[0], axis=-1)
        moved["v"] = np.take(moved["v"], units[1], axis=-1)
        back = [np.argsort(each) for each in units]
        return moved, lambda outputs: {"output": np.take(outputs["output"], back[1], axis=-1),
                                       "final": np.take(np.take(outputs["final"], back[0], axis=-2), back[1],
                                                        axis=-1)}

    assert_graph_as_exact(dew, reference, reference, inputs, orbit, "Kimi" if per_key else "GDN")
