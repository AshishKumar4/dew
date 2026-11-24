"""The recurrence template's chunked op (`dew.nn.kernels.recurrence_chunk.chunk`) on its portable math:
every hook combination it takes, a sequence over several chunks and a ragged last one, packed documents
too, held to the same recurrence run a token at a time (the decode step's reference,
`recurrence.step_reference`) under the accuracy gate (tests/reference_error.py), both from float64."""

import collections
import contextlib
import dataclasses
import functools
import itertools
import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tiny_models
from arm_arithmetic import on_float64_device
from arm_forms import ROUNDED
from contract_support import (
    ADDS,
    CANCELLING_LEVELS,
    FORKED,
    GIVEN,
    HELD,
    cancelling_entries,
    cancelling_orbit,
    normal_float32,
)
from delta_rule_support import stacked_documents
from flax import linen as nn
from jax.sharding import Mesh, PartitionSpec as P
from port_support import row_names_port
from reference_error import (
    FEWEST_SEEDS,
    assert_as_exact,
    assert_fp32_reduction_bound,
    assert_rounds_where_the_reference_does,
    chain_roundings,
    dot_precisions,
    equations,
    widened,
    with_gradients,
)

from dew.nn.kernels import (
    chunked_launch,
    generation,
    recurrence,
    recurrence_algebra as algebra,
    recurrence_chunk,
)
from dew.nn.kernels.recurrence_chunk import Chunk, chunk
from dew.nn.kernels.templates import bundles
from dew.nn.precision import l2norm

ROWS, LENGTH, KEY_HEADS, VALUE_HEADS, KEY_WIDTH, VALUE_WIDTH = 2, 70, 1, 2, 16, 8
CHUNK = 16
"""Tokens a chunk: 70 tokens are four whole chunks and a ragged six."""

NORMED_WIDTH = 32
"""The values' width under the epilogue's norm: a row of 8 can have its RMS come near zero, where the
norm magnifies every rounding before it, and the reference compared with itself then does not
resolve (Mamba2's SSD with its norm over packed documents, at 512 seeds)."""

HOOKS = [recurrence.Hooks(decay=decay, delta=delta, write_gate=not delta, skip=skip,
                          scale=KEY_WIDTH ** -0.5 if delta else None, write_bound=1.0 if delta else None)
         for decay, delta, skip in itertools.product(('none', 'scalar', 'vector'), (True, False),
                                                     (False, True))]
"""The decay kinds, the delta rule or a write gate alone (R = 0, Mamba2's SSD and the gated linear
attentions), with and without the skip; the delta rule's declaring its draws' sigmoid write gates of unit
keys (`write_bound`)."""

FORMED = [
    recurrence.Hooks(decay='scalar', gates=recurrence.Gates(), l2norm=True),
    recurrence.Hooks(decay='scalar', gates=recurrence.Gates(write='step', limit=(0.0, 0.5)), delta=False,
                     skip=True),
    recurrence.Hooks(decay='vector', l2norm=True, output=recurrence.OutputNorm(1e-6), write_bound=1.0),
    recurrence.Hooks(decay='scalar', scale=KEY_WIDTH ** -0.5,
                     output=recurrence.OutputNorm(1e-5, 'sigmoid'), write_bound=1.0),
    recurrence.Hooks(decay='scalar', rated=True, delta=False, skip=True),
    recurrence.Hooks(decay='scalar', rated=True, delta=False, skip=True, output=recurrence.OutputNorm(1e-5)),
    recurrence.Hooks(decay='scalar', rated=True, l2norm=True),
    recurrence.Hooks(decay='vector', delta=False, write_gate=True, skip=True,
                     output=recurrence.OutputNorm(1e-6, 'sigmoid')),
    recurrence.Hooks(decay='scalar', delta=False),
    recurrence.Hooks(decay='vector', delta=False, reads='before', own_weight=True),
    recurrence.Hooks(decay='scalar', reads='before', scale=KEY_WIDTH ** -0.5, write_bound=1.0),
    recurrence.Hooks(decay='vector', delta=False, write_gate=True, own_weight=True),
    recurrence.Hooks(decay='scalar', delta=False, write_gate=True, reads='apart'),
    recurrence.Hooks(decay='vector', delta=False, reads='apart'),
    recurrence.Hooks(decay='scalar', delta=True, reads='apart'),
]
"""The prologue's and epilogue's hooks: the gated delta rule's gates and l2 norm, Mamba2's step gates
clipped with the skip, the gated RMS norm after the l2 norm and after a plain scale, Mamba2's SSD as
`chunk_ssd` hands it over (a rated decay, the step its write gate), with its gated norm, a rated decay
under the delta rule and the l2 norm, the norm after the skip, a decay without any write gate, and the
readout of the memory entering its token (`Hooks.reads`): RWKV6's, a decay a key dimension with its own
write weighed (`Hooks.own_weight`), and the delta rule's; a write gate's own write weighed in a readout
after it, its own weight added to the query's read of the write, as the decode step adds it; and the
readout apart from its token's own write ('apart', row 7402): Mamba-3's, a decay a head and a write gate,
which the template forms, a decay a key dimension's, and the delta rule's, its corrected write apart from the
read (the state after its decay), which the portable math alone does (row 7608: its corrected scan refused an
apart read, a crash)."""

PACKED = (False, True)


def formed(forms: str, fn):
    """`fn` under the run's `forms` (`generation.numerics`): an fp32 call on the template in three parts
    under the accurate forms, as the portable math holds it, whose bar is the fp32 recurrence's own distance;
    in two under the fast forms (`mainloop.FAST_PARTS`), held to recorded upstream's instead, at the default
    product precision they round at (`rounds_products`; the suite's lanes configure 'highest')."""
    @functools.wraps(fn)
    def run(*args, **kwargs):
        with (generation.run_precision(dataclasses.replace(generation.numerics(), forms=forms)),
              jax.default_matmul_precision('default') if forms == 'fast' else contextlib.nullcontext()):
            return fn(*args, **kwargs)

    return run



ISSUING = tuple(next(name for name in generation.TARGETS
                    if (found := generation.architecture(name)) and found.issuers[:1] == (issuer,))
                for issuer in bundles.COMPOSITIONS if issuer != 'tpu')
"""A GPU generation leading each issuer the bundle registry composes (`bundles.COMPOSITIONS`): tcgen05's,
wgmma's and mma.sync's (sm100, sm90, sm80)."""

ROUTES = ((None, *(name for name in ISSUING if name != generation.INTERPRETED_GENERATION))
          if jax.default_backend() != 'gpu' else (None,))
"""The routes an interpreted check runs (`on_route`), each a GPU's own bundle at its own facts, one an
issuer of the registry (`ISSUING`): the interpreter's (`generation.interpreted_target`, wgmma's at the
widest tiles that hold) and every other's, an A100's mma.sync at its derived tiles, whose arithmetic the
interpreter runs as the A100 does (launch by launch to 1e-4, M-JS-29d), and a B200's tcgen05: wgmma's
alone interpreted, an A100's wrong Kimi delta attention gradients went unseen off a GPU (row 5194). On a
GPU, the device's own alone."""


def on_route(route: str | None):
    """Within the block, an interpreted check's launches take `route`'s bundle and facts (`ROUTES`)."""
    return generation.assumed(route, interpreted=True) if route else contextlib.nullcontext()


def routed(check, routes: tuple = ROUTES) -> Callable:
    """The decorator making `check` the test it decorates, run on each of `routes` (every one of `ROUTES`
    where none are given, `on_route`): its name and documentation the test's, its one parameter the
    route."""
    def run(route):
        with on_route(route):
            check()

    def declared(test):
        run.__name__, run.__qualname__, run.__doc__ = test.__name__, test.__qualname__, test.__doc__
        return pytest.mark.parametrize('route', [route for route in ROUTES if route in routes])(run)

    return declared


def chunk_on(kernel: str, hooks, *args, **kwargs):
    """`chunk` on `kernel`, which `generation.chosen` names for the delta rule's selector where it runs it,
    and for the rated scan's where that runs it: the plain-JAX mainloop ('xla') and the recurrence template
    ('mosaic_gpu') both, the port of mamba_ssm's kernels ('pallas') the scan's alone, the rated scan's
    reference under the fast forms (test_recurrence_template_stability's `reference_arm`)."""
    named = {**({} if kernel == 'pallas' else {'gated_delta_rule': kernel}),
             **({'ssd': kernel} if kernel in ('xla', 'mosaic_gpu', 'pallas') else {})}
    with generation.chosen(**named):
        return chunk(hooks, *args, **kwargs)
"""A row of one sequence, and of packed documents."""

SEEDS = dict(zip(((hooks, packed) for hooks in HOOKS + FORMED for packed in PACKED), (
    8, 8, 8, 8, 8, 8, 8, 8, 16, 16, 64, 64, 16, 16, 16, 16, 8, 8, 64, 64, 8, 8, 8, 8, 8, 8,
    64, 64, 128, 128, 16, 8, 16, 16, 64, 64, 32, 32, 32, 32, 8, 8, 16, 16, 16, 16, 16, 16, 32, 16, 8, 8,
    16, 16),
    strict=True))
"""Each site's declared seeds (hooks in `HOOKS` then `FORMED` order, one sequence then packed), sized
independently by `seeds_for` on armada (2026-10-21; the readouts apart 2026-11-17), on the draw with a
memory entering and a document starting on a chunk's first token."""


PACKED_DOCUMENTS = ((0, 23, 40), (9, 48))
"""`draw`'s packed documents, the turned and transition sites' too: each row's documents' first tokens, a
row's first continuing its given memory."""


def marked(documents, length: int) -> np.ndarray:
    """Each row's `documents` (its documents' first tokens) as the `[rows, length]` marks of those tokens."""
    starts = np.zeros((len(documents), length), bool)
    for row, firsts in enumerate(documents):
        starts[row, list(firsts)] = True
    return starts


def assert_packed_on_par(dew: Callable, reference: Callable, draw: Callable, label: str, *, packed: bool,
                        documents, length: int, seeds: int, **options) -> None:
    """Check 1 of a sequence rule `dew` against `reference` (its form a token at a time, each taking packed
    documents' `starts`), output and every operand's gradient (`with_gradients`) from float64 at `seeds`
    draws, over one sequence or, `packed`, the rows' `documents` (`marked`)."""
    starts = marked(documents, length) if packed else None
    assert_as_exact(jax.jit(with_gradients(functools.partial(dew, starts=starts))),
                    jax.jit(with_gradients(functools.partial(reference, starts=starts))), draw,
                    f'{label} packed={packed}', seeds=seeds, **options)


def hosted(x, dtype=jnp.float32) -> jax.Array:
    """A float64 NumPy draw rounded once to fp32, then to `dtype`, as JAX takes it. A recorded site's draws
    (`fla_sites`) are NumPy's, their arithmetic float64's, the same bits on any host and jaxlib: JAX's
    draws and their transcendentals on a host CPU were not (JS-DIGEST-a100: an A100 node drew two
    gpu-geometry sites' inputs apart from the recording's, where a cpu-geometry one's matched)."""
    return jnp.asarray(np.asarray(x, np.float64).astype(np.float32)).astype(dtype)


def sigmoid(x):
    """The logistic function of a float64 draw (`hosted`)."""
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x):
    """`log(1 + e^x)` of a float64 draw (`hosted`)."""
    return np.logaddexp(0.0, x)


def unit(x):
    """A float64 draw's vectors over its last axis scaled to unit length (`hosted`)."""
    return x / np.sqrt(np.sum(x * x, axis=-1, keepdims=True))


def cotangents(operands: Chunk, seed: int) -> tuple[jax.Array, jax.Array]:
    """Cotangents of `operands`' output `[B, S, Hv, Dv]`, in their values' dtype, and of their final memory
    `[B, Hv, Dk, Dv]`, fp32: normals of the seed's own stream (`hosted`)."""
    normal = np.random.default_rng([seed, 101]).standard_normal
    batch, length, heads, width = operands.value.shape
    return (hosted(normal((batch, length, heads, width)), operands.value.dtype),
            hosted(normal((batch, heads, operands.key.shape[-1], width))))


def draw(hooks: recurrence.Hooks, packed: bool, seed: int, *, length: int = LENGTH,
         value_heads: int = VALUE_HEADS, value_width: int | None = None) -> tuple[Chunk]:
    """A sequence's operands for `hooks` from `seed` (`hosted`), its keys' and values' dimensions each
    permuted by an order the seed draws (an exact symmetry of every head's recurrence): fresh queries (of unit
    norm where the hooks declare them), keys and values (`NORMED_WIDTH` wide under the norm), a log decay a
    head or a key dimension in (-1, 0), a gate in (0, 1), a skip, a memory entering, `length` tokens over
    `value_heads` value heads, the values `value_width` wide where given, and with `packed` documents
    starting at tokens 0, 23 and 40 of the first row and 9 and 48 of the second, 48 a
    chunk's first token."""
    rng = np.random.default_rng(seed)
    normal = np.random.default_rng([seed, 1]).standard_normal
    width = value_width or (VALUE_WIDTH if hooks.output is None else NORMED_WIDTH)
    key_order, value_order = rng.permutation(KEY_WIDTH), rng.permutation(width)
    turned = {}
    if hooks.turns:
        # A turned state's symmetry: its turned pairs permuted among themselves, each pair's keys adjacent
        # and its turn moved with it, the unturned keys among themselves; a turn of up to 3 radians a token.
        pairs = rng.permutation(hooks.turns)
        key_order = np.concatenate([np.stack([2 * pairs, 2 * pairs + 1], axis=-1).reshape(-1),
                                    2 * hooks.turns + rng.permutation(KEY_WIDTH - 2 * hooks.turns)])
        turn = rng.uniform(-3.0, 3.0, (ROWS, length, value_heads, hooks.turns))
        turned = {'turn': hosted(turn[..., pairs])}
    query = normal((ROWS, length, KEY_HEADS, KEY_WIDTH))
    # Unit queries where the hooks declare them (`recurrence.Hooks.unit_queries`, Kimi Linear's l2-normed).
    query = (unit(query) if hooks.unit_queries else query)[..., key_order]
    key = unit(normal((ROWS, length, KEY_HEADS, KEY_WIDTH)))[..., key_order]
    value = normal((ROWS, length, value_heads, width))[..., value_order]
    vector = hooks.decay == 'vector'
    log_decay = -softplus(normal((ROWS, length, value_heads) + ((KEY_WIDTH,) if vector else ())) - 1.0)
    if vector:
        log_decay = log_decay[..., key_order]
    written = normal((ROWS, length, value_heads))
    beta = sigmoid(written)
    skip = normal((value_heads,))
    state = (0.3 * normal((ROWS, value_heads, KEY_WIDTH, width)))[:, :, key_order][..., value_order]
    gates = {}
    if hooks.gates is not None:
        # Pre-activations: the gate and beta raw, `log_a` and `bias` a head's.
        gates = {'gate': hosted(normal((ROWS, length, value_heads))),
                 'log_a': hosted(0.5 * normal((value_heads,))), 'bias': hosted(normal((value_heads,)))}
        beta = written
    if hooks.rated:
        # The step through softplus is the write gate; the rate is `A = -exp(A_log)`, a head's.
        beta = softplus(written - 1.0)
        gates = {'rate': hosted(-np.exp(0.5 * normal((value_heads,))))}
    epilogue = {}
    if hooks.output is not None:
        epilogue = {'z': hosted(normal((ROWS, length, value_heads, width))[..., value_order]),
                    'weight': hosted((1.0 + 0.1 * normal((width,)))[value_order])}
    starts = jnp.asarray(marked(PACKED_DOCUMENTS, length)) if packed else None
    own = {'own': hosted(0.5 * normal((value_heads, KEY_WIDTH))[..., key_order])} if hooks.own_weight else {}
    reads_beta = hooks.gated_write if hooks.gates is None else hooks.gates.write == 'sigmoid'
    plain = hooks.decay != 'none' and hooks.gates is None and not hooks.rated
    return (Chunk(hosted(query), hosted(key), hosted(value), log_decay=hosted(log_decay) if plain else None,
                  beta=hosted(beta) if reads_beta else None, skip=hosted(skip) if hooks.skip else None,
                  state=hosted(state), starts=starts, **gates, **epilogue, **own, **turned),)


def token_by_token(hooks: recurrence.Hooks, operands: Chunk):
    """The sequence a token at a time through the decode step's reference, a packed document's first
    token starting from a zero memory: `(output, final memory)`, the output in the operands' dtype. A
    token's updates (`Hooks.substeps`) its keys', values' and write gates' in turn, the step's own axis. Host
    operands are taken as JAX arrays: a float64 truth's first call runs on the gate's widened host arrays,
    which the scan's traced token can't index."""
    operands = jax.tree.map(jnp.asarray, operands)
    group = operands.value.shape[2] // operands.query.shape[2]
    query, key = (jnp.repeat(x, group, axis=2) for x in (operands.query, operands.key))
    work = jnp.promote_types(operands.value.dtype, jnp.float32)
    state = operands.state.astype(work)
    rows, length = operands.query.shape[:2]
    starts = jnp.zeros((rows, length), bool) if operands.starts is None else jnp.asarray(operands.starts)

    def by_token(x):
        # A token's updates (`Hooks.substeps`) on an axis of their own, the step's.
        if x is None or hooks.substeps == 1:
            return x
        return x.reshape(rows, length, hooks.substeps, *x.shape[2:])

    key, value, beta = by_token(key), by_token(operands.value), by_token(operands.beta)

    def one_token(state, t):
        state = jnp.where(starts[:, t, None, None, None], 0.0, state)
        def at(x):
            return None if x is None else x[:, t]

        token = recurrence.Step(
            state, query=query[:, t], key=key[:, t], value=value[:, t],
            log_decay=at(operands.log_decay), gate=at(operands.gate), log_a=operands.log_a,
            bias=operands.bias, beta=at(beta), skip=operands.skip, z=at(operands.z),
            weight=operands.weight, rate=operands.rate, own=operands.own, turn=at(operands.turn),
            erase_key=at(operands.erase_key), erase_read=at(operands.erase_read),
            value_decay=at(operands.value_decay))
        out, _, state = recurrence.step_reference(hooks, token)
        return state, out

    state, out = jax.lax.scan(one_token, state, jnp.arange(length))
    return jnp.moveaxis(out, 0, 1), state


def release(hooks: recurrence.Hooks) -> str | None:
    """The upstream library a form's family has a release of, whose own code is its oracle (testing.md): fla's
    for the delta rule and for its simple GLA (lightning attention's, `recurrence_chunk.simple_gla`),
    mamba_ssm's for the rated scan; None for a write gate without the delta rule, whose oracle is the
    recurrence a token at a time (`token_by_token`)."""
    if hooks.delta or recurrence_chunk.simple_gla(hooks):
        return "fla"
    return "mamba_ssm" if hooks.rated else None


def oracle(hooks: recurrence.Hooks, dtype) -> str | None:
    """A cell's independent oracle at `dtype` (testing.md): its form's release (`release`) where the run's
    declaration admits that release's products for the dtype (`recurrence_chunk.upstream_precision`: fp32's
    TF32 at no more than the default) and its route takes the release's numbers (a simple GLA's under the
    fast forms alone, `recurrence_chunk.chunk_rounding`), else None, the recurrence a token at a time: no run
    of the release computes the call's numbers there."""
    found = release(hooks)
    taken = hooks.delta or hooks.rated or recurrence_chunk.chunk_rounding(delta=False, dtype=dtype,
                                                                          released=True)
    return found if found is not None and taken and recurrence_chunk.upstream_precision(dtype) else None


def against_the_portable_math(label: str, dew_fn, portable_fn, draw, truth_fn, seeds: int) -> str | None:
    """`dew_fn`'s distance from float64 against Dew's own chunk math's (`portable_fn`, the portable math),
    each leaf's pooled RMS ratio over the first `seeds` draws (`draw(seed)`, `truth_fn` the float64 truth on
    them): reported beside a cell's verdict, never judged (testing.md: Dew's own chunk math is no cell's
    reference). Computed once a site (`map_judge.sharded`'s one unit): by the map task that holds it, the
    judge reading it back, which draws nothing; None in every other map task."""
    from map_judge import sharded

    def distance(start: int, stop: int) -> str:
        del start, stop
        arms = {"dew": jax.jit(dew_fn), "portable": jax.jit(portable_fn)}
        errors: dict[str, dict[str, list[float]]] = {}
        for seed in range(seeds):
            drawn = draw(seed)
            with jax.enable_x64(new_val=True):
                exact = truth_fn(*widened(drawn))
            for arm, fn in arms.items():
                for (path, leaf), true in zip(jax.tree_util.tree_flatten_with_path(fn(*drawn))[0],
                                             jax.tree.leaves(exact), strict=True):
                    errors.setdefault(jax.tree_util.keystr(path), {}).setdefault(arm, []).append(
                        float(np.mean((np.asarray(leaf, np.float64) - np.asarray(true, np.float64)) ** 2)))
        return ", ".join(f"{name} {np.sqrt(sum(arms['dew']) / max(sum(arms['portable']), 1e-300)):.3g}"
                         for name, arms in errors.items())

    found = sharded(f"{label}, against the portable math", distance, 1, upto=1)
    return None if found is None else found[0]


@pytest.mark.parametrize('packed', PACKED, ids=['one sequence', 'packed'])
@pytest.mark.parametrize('hooks', HOOKS + FORMED, ids=str)
def test_the_chunked_op_is_as_exact_as_the_recurrence_token_by_token(hooks, packed):
    """`chunk`'s portable math in `CHUNK`-token chunks against the same recurrence a token at a time,
    output and final memory, over draws of fresh operands under orders of the keys' and values'
    dimensions, both from float64."""
    dew = jax.jit(functools.partial(chunk_on, 'xla', hooks, chunk_size=CHUNK))
    reference = jax.jit(functools.partial(token_by_token, hooks))
    assert_as_exact(dew, reference, functools.partial(draw, hooks, packed), f'{hooks} packed={packed}',
                    seeds=SEEDS[hooks, packed])


DELTA_PRODUCTS = [recurrence.Hooks(decay='scalar', l2norm=True, substeps=2),
                  recurrence.Hooks(decay='vector', l2norm=True, substeps=3)]
"""DeltaProduct's forms (`Hooks.substeps`): Gated DeltaProduct's, a decay a head over two updates a token
(fla's `chunk_gated_delta_product`), and a decay a key dimension over three."""


def substepped_draw(hooks: recurrence.Hooks, seed: int):
    """`draw`'s operands of `LENGTH` tokens of `hooks.substeps` updates each, and cotangents: its keys,
    values and write gates fresh at every update, its queries and decays a token's."""
    (laid,) = draw(hooks, packed=False, seed=seed, length=LENGTH * hooks.substeps)
    every = hooks.substeps
    operands = dataclasses.replace(laid, query=laid.query[:, ::every], log_decay=laid.log_decay[:, ::every])
    out, final = cotangents(laid, seed)
    return operands, (out[:, ::every], final)


@pytest.mark.parametrize('packed', PACKED, ids=['one sequence', 'packed'])
@pytest.mark.parametrize('hooks', DELTA_PRODUCTS, ids=['two updates a head', 'three updates a key dimension'])
def test_a_delta_products_chunks_are_as_exact_as_its_tokens(hooks, packed):
    """DeltaProduct (`Hooks.substeps`) on the chunked op's portable math, its updates the op's tokens in
    turn, against the decode step's reference a token at a time, its updates on their own axis
    (`token_by_token`): the output, the final memory and every operand's gradient from float64 (check 1, row
    9030)."""
    # Packed documents at `draw`'s tokens, marked outside the differentiated operands.
    starts = jnp.asarray(marked(((0, 23, 40), (9, 48)), LENGTH)) if packed else None

    def differentiated(fn):
        return jax.jit(with_gradients(lambda operands: fn(dataclasses.replace(operands, starts=starts))))

    assert_as_exact(differentiated(functools.partial(chunk_on, 'xla', hooks, chunk_size=CHUNK)),
                    differentiated(functools.partial(token_by_token, hooks)),
                    functools.partial(substepped_draw, hooks), f'{hooks} packed={packed}', seeds=8)


def substepped_entered_draw(hooks: recurrence.Hooks, seed: int):
    """`entered_draw`'s forked operands, entry writes among them, of `LENGTH` tokens of `hooks.substeps`
    updates each, and cotangents: its keys, values and write gates fresh at every update, its queries and
    decays a token's."""
    operands, (out, final) = entered_draw(hooks, seed, entries=True, length=LENGTH * hooks.substeps)
    every = hooks.substeps
    return (dataclasses.replace(operands, query=operands.query[:, ::every],
                                log_decay=operands.log_decay[:, ::every]), (out[:, ::every], final))


@pytest.mark.parametrize('by_firsts', [False, True], ids=['starts', 'firsts'])
def test_a_forked_delta_products_documents_are_as_exact_as_each_alone(by_firsts):
    """Gated DeltaProduct (`DELTA_PRODUCTS[0]`) over a forked call on the chunked op's portable math, each
    document entering its table slot's memory with its entry write, the documents given by their starts or
    their first slots alone (`Chunk.firsts`, a token's, which its updates scale): output, last memory and
    every gradient against each document alone a token at a time (`document_by_document`) from float64
    (check 1, row 9030)."""
    hooks = DELTA_PRODUCTS[0]

    def ours(operands):
        return chunk_on('xla', hooks, entering_documents(operands, by_firsts=by_firsts), chunk_size=CHUNK)

    assert_as_exact(jax.jit(with_gradients(ours)),
                    jax.jit(with_gradients(functools.partial(document_by_document, hooks))),
                    functools.partial(substepped_entered_draw, hooks),
                    f'{hooks} forked by_firsts={by_firsts}', seeds=8)


def stepped(hooks: recurrence.Hooks, state, query, key, value, log_decay, beta):
    """A decode token of `hooks` through `recurrence.step`: `(output, state')`."""
    token = recurrence.Step(state, query=query, key=key, value=value, log_decay=log_decay, beta=beta)
    out, _, updated = recurrence.step(hooks, token)
    return out, updated


def delta_product_token(state, query, key, value, log_decay, beta):
    """A DeltaProduct decode token as its paper writes it, a value head per key head: the memory `[R, H, K,
    V]` decayed once, `exp(g) S`, then each of its updates `S + beta_j k_j (v_j - S^T k_j)^T` in turn, read
    after the last, `S^T q`: `(output [R, H, V], memory)` (Siems et al. 2025, eq. 3; fla's
    `naive_gated_delta_product`)."""
    state = jnp.exp(log_decay)[..., None, None] * state
    for update in range(key.shape[1]):
        k, v, b = key[:, update], value[:, update], beta[:, update]
        corrected = v - jnp.einsum('rhkv,rhk->rhv', state, k)
        state = state + b[..., None, None] * k[..., :, None] * corrected[..., None, :]
    return jnp.einsum('rhkv,rhk->rhv', state, query), state


def test_a_delta_product_decode_token_is_as_exact_as_its_recurrence():
    """A decode token of three updates (`Hooks.substeps`) through the public step (`recurrence.step`):
    output, memory and every gradient against the token as DeltaProduct's paper writes it
    (`delta_product_token`: decayed once, read after its last update) from float64 (check 1, row 9030)."""
    hooks = recurrence.Hooks(decay='scalar', substeps=3)

    def draw(seed):
        normal = np.random.default_rng([9030, seed]).standard_normal
        rows, heads, keys, values = 2, 2, KEY_WIDTH, VALUE_WIDTH
        operands = (hosted(0.3 * normal((rows, heads, keys, values))), hosted(normal((rows, heads, keys))),
                    hosted(unit(normal((rows, 3, heads, keys)))), hosted(normal((rows, 3, heads, values))),
                    hosted(-softplus(normal((rows, heads)))), hosted(sigmoid(normal((rows, 3, heads)))))
        return *operands, (hosted(normal((rows, heads, values))), hosted(normal((rows, heads, keys, values))))

    assert_as_exact(jax.jit(with_gradients(functools.partial(stepped, hooks))),
                    jax.jit(with_gradients(delta_product_token)), draw, 'a DeltaProduct decode token',
                    seeds=8)


def test_a_delta_products_decode_memory_is_wide_between_its_updates():
    """A decode token of two updates (`Hooks.substeps`) on a bf16 memory from zero, one key and value, no
    decay and a unit query: the first writes the fp32 midpoint `m` of two adjacent bf16 values (a gate of
    one), the second its negative under a gate of one half, `m + (-m - m) / 2`: output and memory exactly
    zero, the memory held in fp32 between the updates (`recurrence._substeps`), where rounded to bf16 after
    the first its residual is half that rounding's."""
    midpoint = (1.0 + 2.0 ** -8) * 2.0 ** -3
    token = recurrence.Step(jnp.zeros((1, 1, 1, 1), jnp.bfloat16), query=jnp.ones((1, 1, 1), jnp.float32),
                            key=jnp.ones((1, 2, 1, 1), jnp.float32),
                            value=jnp.asarray([midpoint, -midpoint], jnp.float32).reshape(1, 2, 1, 1),
                            log_decay=jnp.zeros((1, 1), jnp.float32),
                            beta=jnp.asarray([1.0, 0.5], jnp.float32).reshape(1, 2, 1))
    out, _, updated = recurrence.step(recurrence.Hooks(decay='scalar', substeps=2, scale=1.0), token)
    assert (out.astype(jnp.float32).ravel().tolist(), updated.astype(jnp.float32).ravel().tolist()) == (
        [0.0], [0.0])


@pytest.mark.parametrize(('substeps', 'form'), [(2, {'delta': False}), (2, {'reads': 'before'}),
                                               (2, {'erase': 'untied'}), (2, {'rated': True}), (0, {})],
                         ids=['no delta rule', 'read before', 'an erase of its own', 'rated', 'none'])
def test_a_delta_product_outside_its_form_is_refused(substeps, form):
    """Substeps (`Hooks.substeps`) outside DeltaProduct's form, a token's delta-rule updates read after its
    last: a ValueError, beside an erase of its own whose laid tokens are a layout of their own (row 9030)."""
    with pytest.raises(ValueError):
        recurrence.Hooks(decay='scalar', substeps=substeps, **form)


@pytest.mark.parametrize('stream', ['key', 'value', 'beta'])
def test_a_delta_product_token_of_another_count_of_updates_is_refused(stream):
    """A decode token of three updates (`Hooks.substeps`) given two of one stream, its keys', values' or
    write gates': a ValueError, where the step took the updates it was given and left the third out (row
    9030)."""
    hooks = recurrence.Hooks(decay='scalar', substeps=3)
    shapes = {'key': (1, 3, 1, 4), 'value': (1, 3, 1, 2), 'beta': (1, 3, 1)}
    given = {name: jnp.ones((shape[0], 2 if name == stream else 3, *shape[2:]))
             for name, shape in shapes.items()}
    token = recurrence.Step(jnp.zeros((1, 1, 4, 2)), query=jnp.ones((1, 1, 4)),
                            log_decay=jnp.zeros((1, 1)), **given)
    with pytest.raises(ValueError):
        recurrence.step(hooks, token)


TURNED = recurrence.Hooks(decay='scalar', delta=False, write_gate=True, skip=True, turns=KEY_WIDTH // 2 - 2)
"""Mamba-3's form: a decay a head's, a write gate, the skip, and the state's first key pairs turned each token
(`Hooks.turns`), two pairs unturned. Its sites' oracle is the recurrence a token at a time, the spec's own
where the upstream has no form at the cell's dtype (docs/design/testing.md's recurrence cells): mamba_ssm's
Mamba-3 (`mamba3_siso_combined`, state-spaces/mamba e9594ce1c) casts its queries, keys, values, trapezoid
and angles to bf16 (mamba_ssm/ops/triton/mamba3/mamba3_siso_combined.py:384-392) and turns a token by
`tanh(angle) pi dt` (angle_dt.py:94, bounded), so an fp32 call's arbitrary turns have none there; its bf16
layer sites are held to it recorded (row 7923)."""


def steep_draw(seed: int):
    """`draw`'s operands for `TURNED` and cotangents, each row's even documents (`PACKED_DOCUMENTS`) decaying
    steeply, each token's log decay in (-120, -60), its others mildly, in (-1, 0): a chunk's earlier
    document's prefix past its later one's by more than fp32's exponents span."""
    (operands,) = draw(TURNED, packed=False, seed=seed)
    tokens = np.arange(LENGTH)
    steep = np.stack([np.sum(tokens[:, None] >= np.asarray(begins)[None, :], axis=1) % 2 == 1
                      for begins in PACKED_DOCUMENTS])
    rng = np.random.default_rng([seed, 8288])
    mild = rng.uniform(-1.0, 0.0, operands.log_decay.shape)
    log_decay = np.where(steep[..., None], rng.uniform(-120.0, -60.0, operands.log_decay.shape), mild)
    operands = dataclasses.replace(operands, log_decay=hosted(log_decay))
    return operands, cotangents(operands, seed)


VANISHED = (5, 9, 30, 33)
"""The tokens whose decays vanish in `vanishing`'s draws: two in one chunk of each row's first two
documents (`PACKED_DOCUMENTS`, `CHUNK`)."""


def vanishing(operands):
    """`operands` with the decays at `VANISHED` vanishing, -2^127 each (a zero decay; two summed are past
    fp32's range)."""
    log_decay = np.asarray(operands.log_decay, np.float64).copy()
    log_decay[:, list(VANISHED)] = -2.0 ** 127
    return dataclasses.replace(operands, log_decay=hosted(log_decay))


def vanishing_draw(seed: int):
    """`draw`'s operands for `TURNED` and cotangents, the decays at `VANISHED` vanishing (`vanishing`)."""
    (operands,) = draw(TURNED, packed=False, seed=seed)
    operands = vanishing(operands)
    return operands, cotangents(operands, seed)


@pytest.mark.parametrize(('packed', 'regime'), [(False, None), (True, None), (True, 'steep'),
                                                (False, 'vanishing')],
                         ids=['one sequence', 'packed', 'packed-steep', 'vanishing'])
def test_a_turned_state_through_the_chunked_op_is_as_exact_as_the_recurrence_token_by_token(packed, regime):
    """A state that turns its key pairs (`Hooks.turns`, Mamba-3's rotating basis) through `chunk`'s portable
    math in `CHUNK`-token chunks against the same recurrence a token at a time: output, final memory and
    every operand's gradient, the turns' among them, from float64 (check 1). A turn's gradient is the
    crossing form's (`templates.recurrence.turned.turn_gradient`), its pairs' scores' cotangents across each
    token (row 7419: a turn read as an angle summed over a document cancelled across its tokens to 120 times
    the recurrence's error; rows 7971, 7978). Under
    steep decays (`steep_draw`) a chunk's earlier document's writers lie past fp32's exponents from its last
    one's end, which the turns' gradient weighed by `exp` of their difference before it took them out, NaN
    (base review fbb1c2f71 F2); under vanishing decays (`vanishing_draw`, two of -2^127 in a chunk) its
    prefix summed them to -inf, NaN (audit KPM-02). Its oracle the recurrence: `TURNED`'s note."""
    def turned(operands, starts=None):
        return chunk_on('xla', TURNED, dataclasses.replace(operands, starts=starts), chunk_size=CHUNK)

    def tokens(operands, starts=None):
        return token_by_token(TURNED, dataclasses.replace(operands, starts=starts))

    def drawn(seed):
        (operands,) = draw(TURNED, packed=False, seed=seed)
        return operands, cotangents(operands, seed)

    drawing = {None: drawn, 'steep': steep_draw, 'vanishing': vanishing_draw}[regime]
    assert_packed_on_par(turned, tokens, drawing, f'{TURNED}', packed=packed, documents=PACKED_DOCUMENTS,
                         length=LENGTH, seeds=32)


OJA = recurrence.Hooks(decay='vector', erase='values', scale=KEY_WIDTH ** -0.5)
"""Oja's rule with a gate a value dimension (`Hooks.erase` 'values', fla's gated_oja_rule), row 9028."""


def oja_token(state, query, key, value, log_decay, beta):
    """A decode token of Oja's rule as its paper writes it, a value head per key head: the memory `S` `[R, H,
    K, V]` decayed along its values and its values erased from it, then written, `S D (I - beta v v^T) + beta
    k v^T`, `D = exp(log_decay)` `[R, H, V]`, read after, `q^T S'`."""
    decayed = state * jnp.exp(log_decay)[..., None, :]
    erased = jnp.einsum('rhkv,rhv->rhk', decayed, value)
    beta_ = beta[..., None, None]
    written = beta_ * key[..., :, None] * value[..., None, :]
    updated = decayed - beta_ * erased[..., None] * value[..., None, :] + written
    return jnp.einsum('rhk,rhkv->rhv', query, updated), updated


def test_an_oja_decode_token_is_as_exact_as_its_formula():
    """A decode token of Oja's rule (`OJA`, scale one) through the public step (`recurrence.step`): output,
    memory and every gradient against the token as its paper writes it (`oja_token`) from float64 (check 1,
    row 9028)."""
    hooks = dataclasses.replace(OJA, scale=1.0)
    rows, heads, keys, values = 8, 4, KEY_WIDTH, VALUE_WIDTH

    def draw(seed):
        rng = np.random.default_rng([9028, seed])
        normal = rng.standard_normal
        operands = (0.3 * normal((rows, heads, keys, values)), normal((rows, heads, keys)),
                    unit(normal((rows, heads, keys))), unit(normal((rows, heads, values))),
                    -rng.uniform(0, 1, (rows, heads, values)), rng.uniform(0, 1, (rows, heads)))
        seeded = (normal((rows, heads, values)), normal((rows, heads, keys, values)))
        return *(hosted(x) for x in operands), tuple(hosted(x) for x in seeded)

    assert_as_exact(jax.jit(with_gradients(functools.partial(stepped, hooks))),
                    jax.jit(with_gradients(oja_token)), draw, "an oja decode token", seeds=8)


def oja_edge(state, key, value, beta):
    """A decode token of Oja's rule over one key, a query of one, no decay and scale one, in fp32 (`OJA`'s
    through `recurrence.step`): `(output, memory)` of a memory `state` `[Dv]`, a key `key` `[1]`, values
    `value` `[Dv]` and a write gate `beta`."""
    def held(x, *shape):
        return jnp.reshape(jnp.asarray(x, jnp.float32), (1, 1, *shape))

    token = recurrence.Step(held(state, 1, len(state)), query=held([1.0], 1), key=held(key, 1),
                            value=held(value, len(value)), log_decay=jnp.zeros((1, 1), jnp.float32),
                            beta=held(beta))
    out, _, updated = recurrence.step(dataclasses.replace(OJA, decay='scalar', scale=1.0), token)
    return out.ravel(), updated.ravel()


@pytest.mark.parametrize(('state', 'key', 'value', 'beta', 'expected'), [
    ([-2.0 ** 127], [2.0 ** 127], [1.0], 0.25, [-2.0 ** 126]),
    ([2.0 ** 120, 0.0], [0.0], [2.0 ** -60, 1.0], 2.0 ** -100, [2.0 ** 120, -2.0 ** -40]),
    ([2.0 ** 127] * 4, [0.0], [0.5] * 4, 0.25, [2.0 ** 127 - 2.0 ** 125] * 4)],
    ids=['a correction past the largest', 'a gated value past the least', 'an ungated read past the largest'])
def test_an_oja_decode_tokens_gate_scales_its_formed_correction(state, key, value, beta, expected):
    """A decode token of Oja's rule (`oja_edge`), its memory and output exact in fp32: the gated correction
    `beta k - beta v^T S^T` one exact sum, which neither its ungated parts past fp32's largest (2^128) nor
    `beta v` past its least (2^-160) loses."""
    out, updated = oja_edge(state, key, value, beta)
    assert (out.tolist(), updated.tolist()) == (expected, expected)


def test_an_oja_decode_tokens_gate_gradient_sums_its_terms_exactly():
    """The gate's derivative of a decode token of Oja's rule (`oja_edge`) at a memory `[2^31, 2, 0, 0]`,
    values of 1/2 and a key of 2^30: its write's `k` 2^30 and its erase's read `v^T S^T` 2^30 + 1, summed
    exactly, give twice its second output's -1, where each rounded apart cancelled to zero."""
    def second(beta):
        return 2.0 * oja_edge([2.0 ** 31, 2.0, 0.0, 0.0], [2.0 ** 30], [0.5] * 4, beta)[0][1]

    assert float(jax.grad(second)(jnp.float32(0.25))) == -1.0


RWKV7_FLOOR = -math.exp(-0.5)
"""RWKV7's least log decay, `w = exp(-e^{-1/2} sigmoid(x))`'s."""

ADDITIVE = recurrence.Hooks(decay='vector', erase='additive', scale=KEY_WIDTH ** -0.5)
"""RWKV7's additive diagonal plus low rank, a token erasing along its own vector the state it enters beside
its write (`recurrence_chunk._erases_laid`, `linear_scan.Affine.erased_before`)."""

UNTIED = recurrence.Hooks(decay='vector', erase='untied', scale=KEY_WIDTH ** -0.5)
"""GDN-2's and Comba's erase vectors of their own under a decay a key dimension's, which laid tokens take."""

TRANSITIONS = [
    pytest.param(recurrence.Hooks(decay='scalar', erase='untied', scale=KEY_WIDTH ** -0.5), id='untied'),
    pytest.param(UNTIED, id='untied per key'),
    pytest.param(ADDITIVE, id='additive'),
    pytest.param(recurrence.Hooks(decay='scalar', delta=False, write_gate=True, value_decay=True),
                 id='two-sided')]
"""The delta rule's transitions beside its tied erase and a decay along the keys (`Hooks.erase`,
`Hooks.value_decay`, row 9029): GDN-2's and Comba's erase vectors of their own, a head's and a key
dimension's decay; RWKV7's additive diagonal plus low rank, its tokens erasing the state they enter (row
9670); two-sided GLA's decay along the values (Oja's rule's erase along the values, row 9028:
test_gated_oja)."""


def transition_draw(hooks: recurrence.Hooks, seed: int):
    """`draw`'s operands for `hooks` and cotangents, with its transition's own (`transitioned`)."""
    (operands,) = draw(hooks, packed=False, seed=seed)
    operands = transitioned(hooks, operands, seed)
    return operands, cotangents(operands, seed)


def transitioned(hooks: recurrence.Hooks, operands, seed: int):
    """`operands` with `hooks`' transition's own: erase vectors of unit norm (`erase_key`, `erase_read`;
    RWKV7's, additive, `a` scaled so `a b^T` stays inside the decay, a key's `exp(g) / 2` at most), or a log
    decay along the values in (-1, 0) (`value_decay`). An additive erase's log decays RWKV7's, in
    `[-exp(-1/2), 0]`, every seventh at its least."""
    rng = np.random.default_rng([9029, seed])
    if hooks.erase == 'additive':
        steps = rng.uniform(RWKV7_FLOOR, 0.0, operands.log_decay.shape)
        steps[:, ::7] = RWKV7_FLOOR
        operands = dataclasses.replace(operands, log_decay=hosted(steps))
    keys = operands.key.shape
    if hooks.erase in ('untied', 'additive'):
        a, b = (unit(rng.standard_normal(keys)) for _ in range(2))
        if hooks.erase == 'additive':
            a = 0.5 * np.exp(np.asarray(operands.log_decay, np.float64))[:, :, :keys[2]] * a
        operands = dataclasses.replace(operands, erase_key=hosted(a), erase_read=hosted(b))
    if hooks.value_decay:
        across = rng.uniform(-1.0, 0.0, operands.value.shape)
        operands = dataclasses.replace(operands, value_decay=hosted(across))
    return operands


@pytest.mark.parametrize('packed', PACKED, ids=['one sequence', 'packed'])
@pytest.mark.parametrize('hooks', TRANSITIONS)
def test_a_transitions_chunks_are_as_exact_as_its_tokens(hooks, packed):
    """Each transition (`TRANSITIONS`) through the chunked op, which the portable math takes (its forms the
    token update's derived, `scan.derive_scan`), against the decode step's reference a token at a time
    (`token_by_token`): output, final memory and every operand's gradient from float64 (check 1, row
    9029)."""
    def ours(operands, starts=None):
        return chunk(hooks, dataclasses.replace(operands, starts=starts), chunk_size=CHUNK)

    def tokens(operands, starts=None):
        return token_by_token(hooks, dataclasses.replace(operands, starts=starts))

    assert_packed_on_par(ours, tokens, functools.partial(transition_draw, hooks), f'{hooks}', packed=packed,
                         documents=PACKED_DOCUMENTS, length=LENGTH, seeds=8)


def transition_token(kind: str, state, query, key, value, log_decay, beta, a, b, e):
    """A decode token of each transition as its papers write it, a value head per key head, its memory `S`
    `[R, H, K, V]` read after its update, `q^T S'`: 'untied' (GDN-2's, Comba's) `(I - beta a b^T) D S + beta
    k v^T`; 'additive' (RWKV7's) `(D - a b^T) S + beta k v^T`; 'two-sided' (GLA's value gates) `D S E + beta
    k v^T`. `D` and `E` the exponents' `exp`, a key's and a value's each (`log_decay` `[R, H, K]`, `e` `[R,
    H, V]`); Oja's rule's is `oja_token`."""
    beta_ = beta[..., None, None]
    written = beta_ * key[..., :, None] * value[..., None, :]
    decayed = jnp.exp(log_decay)[..., None] * state
    if kind == 'untied':
        erased = jnp.einsum('rhk,rhkv->rhv', b, decayed)
        updated = decayed - beta_ * a[..., :, None] * erased[..., None, :]
    elif kind == 'additive':
        updated = decayed - a[..., :, None] * jnp.einsum('rhk,rhkv->rhv', b, state)[..., None, :]
    else:
        updated = decayed * jnp.exp(e)[..., None, :]
    updated = updated + written
    return jnp.einsum('rhk,rhkv->rhv', query, updated), updated


TRANSITION_TOKENS = {'untied': recurrence.Hooks(decay='vector', erase='untied', scale=1.0),
                     'additive': recurrence.Hooks(decay='vector', erase='additive', scale=1.0),
                     'two-sided': recurrence.Hooks(decay='vector', delta=False, write_gate=True,
                                                   value_decay=True, scale=1.0)}
"""Each transition's decode form (`transition_token`'s kinds), its decay a key dimension's."""


@pytest.mark.parametrize('kind', list(TRANSITION_TOKENS))
def test_a_transitions_decode_token_is_as_exact_as_its_formula(kind):
    """A decode token of each transition (`TRANSITION_TOKENS`) through the public step (`recurrence.step`):
    output, memory and every gradient against the token as its paper writes it (`transition_token`) from
    float64 (check 1, row 9029)."""
    hooks = TRANSITION_TOKENS[kind]
    rows, heads, keys, values = 2, 2, KEY_WIDTH, VALUE_WIDTH

    def draw(seed):
        rng = np.random.default_rng([9029, seed])
        normal = rng.standard_normal
        operands = (0.3 * normal((rows, heads, keys, values)), normal((rows, heads, keys)),
                    unit(normal((rows, heads, keys))), normal((rows, heads, values)),
                    -rng.uniform(0, 1, (rows, heads, keys)),
                    rng.uniform(0, 1, (rows, heads)), unit(normal((rows, heads, keys))),
                    unit(normal((rows, heads, keys))), -rng.uniform(0, 1, (rows, heads, values)))
        if kind == 'additive':
            # RWKV7's `a b^T` inside its decay: `a` a key's `exp(g) / 2` at most.
            operands = (*operands[:6], 0.5 * np.exp(operands[4]) * operands[6], *operands[7:])
        seeded = (normal((rows, heads, values)), normal((rows, heads, keys, values)))
        return *(hosted(x) for x in operands), tuple(hosted(x) for x in seeded)

    def stepped(state, query, key, value, log_decay, beta, a, b, e):
        token = recurrence.Step(state, query=query, key=key, value=value, log_decay=log_decay, beta=beta,
                                erase_key=a if kind in ('untied', 'additive') else None,
                                erase_read=b if kind in ('untied', 'additive') else None,
                                value_decay=e if kind == 'two-sided' else None)
        out, _, updated = recurrence.step(hooks, token)
        return out, updated

    assert_as_exact(jax.jit(with_gradients(stepped)),
                    jax.jit(with_gradients(functools.partial(transition_token, kind))), draw,
                    f'a {kind} decode token', seeds=8)


def test_an_additive_erases_decay_gradient_is_its_decays_alone():
    """RWKV7's additive erase (`Hooks.erase` 'additive', `S' = D S - a (b^T S)`) through the chunked op, one
    token of no decay (`g = 0`) and no write over a memory `(1, 2^-27)`, erase vectors `b = (1, 0)` and `a =
    (0, 1/2)`: its final memory's second key's gradient by the decay exactly 2^-27, the decay's alone, the
    erase reading the memory before its decay independent of it (`linear_scan.Affine.erased_before`, row
    9670). Laid as the delta rule's correction of the decayed memory (`b exp(-g)` read through `D S`), its
    decay entered twice, the second key's `2^-27 - 1/2` rounded to `-1/2` before its decay, and the two
    cancelled to zero (row 9029's head review)."""
    hooks = recurrence.Hooks(decay='scalar', erase='additive', scale=1.0)

    def held(x, *shape):
        return jnp.reshape(jnp.asarray(x, jnp.float32), shape)

    def second(log_decay):
        operands = Chunk(held([1.0, 0.0], 1, 1, 1, 2), held([0.0, 0.0], 1, 1, 1, 2), held([1.0], 1, 1, 1, 1),
                         log_decay=log_decay, beta=held([1.0], 1, 1, 1),
                         state=held([1.0, 2.0 ** -27], 1, 1, 2, 1), erase_key=held([0.0, 0.5], 1, 1, 1, 2),
                         erase_read=held([1.0, 0.0], 1, 1, 1, 2))
        return chunk(hooks, operands)[1][0, 0, 1, 0]

    assert float(jax.grad(second)(jnp.zeros((1, 1, 1), jnp.float32))[0, 0, 0]) == 2.0 ** -27


def test_an_additive_erase_keeps_its_rows_on_an_explicit_axis():
    """RWKV7's additive erase (its rank-two token, `recurrence_chunk._erases_laid`) from no memory entering,
    its rows split over an Explicit mesh axis: it traces and runs, its output and final memory finite and
    placed as its rows are, the zero memory it starts from placed so (row 9029's head review: seeded whole,
    a token loop's carry changed its sharding at its first token, which `lax.scan` refuses)."""
    (operands,) = draw(ADDITIVE, packed=False, seed=0)
    operands = dataclasses.replace(transitioned(ADDITIVE, operands, 0), state=None)
    mesh = jax.make_mesh((2,), ('data',), (jax.sharding.AxisType.Explicit,), devices=jax.devices()[:2])
    with jax.set_mesh(mesh):
        rows = jax.sharding.NamedSharding(mesh, P('data'))
        split = jax.tree.map(lambda x: jax.device_put(x, rows), operands)
        found = jax.jit(functools.partial(chunk, ADDITIVE, chunk_size=CHUNK))(split)
    for got in found:
        assert np.isfinite(np.asarray(got)).all()
        assert jax.typeof(got).sharding.spec[0] == jax.typeof(split.value).sharding.spec[0]


@pytest.mark.parametrize('hooks', [HOOKS[0], HOOKS[4], ADDITIVE],
                         ids=['no decay', 'the delta rule', 'an additive erase'])
def test_a_forked_call_on_an_explicit_row_axis_is_the_call_whole(hooks):
    """A forked call (`Chunk.loads`: documents taking their memories up from its table, entry writes) on the
    portable math, its operands' rows split over an Explicit mesh axis before it is differentiated: its
    output, last memory, every operand's gradient and their tangent along its values on par from float64
    with the same call whole (check 1), each placed as its rows are (row 9690: its walk's carries, seeded
    within the rows' map whole, met the rows' token maps split in a gather JAX refuses; a no-decay call's
    gates, formed whole, its map's)."""
    mesh = jax.make_mesh((2,), ('data',), (jax.sharding.AxisType.Explicit,), devices=jax.devices()[:2])
    rows = jax.sharding.NamedSharding(mesh, P('data'))
    run = functools.partial(chunk_on, 'xla', hooks, chunk_size=CHUNK)

    def drawn(seed):
        operands, seeded = entered_draw(hooks, seed, entries=True)
        tangent = jax.random.normal(jax.random.key(seed + 31), operands.value.shape, operands.value.dtype)
        return transitioned(hooks, operands, seed), seeded, tangent

    def arm(place):
        # The draw's operands, cotangents and tangent placed as `place` places them before the call is
        # differentiated, its documents laid within it (as `assert_forked_on_par`'s) and placed alike.
        def ran(operands):
            return run(place(entering_documents(operands)))

        def both(operands, cotangents, tangent):
            operands, cotangents, tangent = place((operands, cotangents, tangent))

            def along(value):
                return ran(dataclasses.replace(operands, value=value))

            pulled = with_gradients(ran)(operands, cotangents)
            return pulled, jax.jvp(along, (operands.value,), (tangent,))[1]

        return both

    split = arm(lambda tree: jax.tree.map(lambda x: jax.device_put(x, rows), tree))
    with jax.set_mesh(mesh):
        assert_as_exact(jax.jit(split), jax.jit(arm(lambda tree: tree)), drawn,
                        f'forked {hooks} on an explicit row axis', seeds=FEWEST_SEEDS)
        ((out, last), grads), tangents = jax.jit(split)(*drawn(0))
    for placed in (out, last, grads.value, *tangents):
        assert jax.typeof(placed).sharding.spec[0] == 'data'


@pytest.mark.parametrize('documents', [3, 0], ids=['three slots a row', 'none'])
def test_a_forked_calls_token_maps_keep_its_rows_on_an_explicit_axis(documents):
    """A forked call's token maps (`chunked_launch.Forked.tokens`: each token's load, store and document)
    over ten tokens, its rows split over an Explicit mesh axis: a row of three documents and one of two, its
    third slot unused (its start the call's length, its load and store nonzero, dropped), or a table of no
    documents, each map a row's own exactly (a document's first token its load and number, its last its
    store, every other token -1 or zero), int32 and placed as its rows are (row 9690: scattered into whole
    buffers by a row index, JAX refused indices and updates split on an Explicit axis)."""
    firsts = np.asarray([[0, 3, 7], [0, 5, 10]], np.int32)[:, :documents]
    loads = np.asarray([[1, 0, 2], [0, 3, 4]], np.int32)[:, :documents]
    stores = np.asarray([[1, 2, 3], [3, 1, 5]], np.int32)[:, :documents]
    expected = (np.full((2, 10), -1, np.int32), np.zeros((2, 10), np.int32), np.zeros((2, 10), np.int32))
    if documents:
        for row, (begins, ends) in enumerate((([0, 3, 7], [2, 6, 9]), ([0, 5], [4, 9]))):
            live = len(begins)
            expected[0][row, begins], expected[1][row, ends] = loads[row, :live], stores[row, :live]
            expected[2][row, begins] = range(live)
    mesh = jax.make_mesh((2,), ('data',), (jax.sharding.AxisType.Explicit,), devices=jax.devices()[:2])
    with jax.set_mesh(mesh):
        rows = jax.sharding.NamedSharding(mesh, P('data'))
        placed = (jax.device_put(x, rows) for x in (firsts, loads, stores))
        forked = chunked_launch.Forked(*placed, entry=None, held=3)
        found = jax.jit(lambda forked: forked.tokens(10))(forked)
    for got, want in zip(found, expected, strict=True):
        np.testing.assert_array_equal(np.asarray(got), want, strict=True)
        assert jax.typeof(got).sharding.spec[0] == 'data'


@pytest.mark.parametrize('hooks', [UNTIED, ADDITIVE], ids=['after its decay', 'before its decay'])
def test_a_forked_erase_runs_in_a_shard_map_that_checks_varying_axes(hooks):
    """A forked call of an erase of its own (`Chunk.loads`, entry writes and given memories), its rows, tables
    and documents split over a checked `shard_map`'s axis, its tokens correcting the state after its decay
    (an untied erase's laid tokens) or before it (an additive one's): its outputs, final memory and their
    tangent along its erase vectors finite, every carry typed as the operands that reach it vary (row 9029's
    head review)."""
    operands, _ = entered_draw(hooks, 0, entries=True)
    operands = entering_documents(transitioned(hooks, operands, 0))
    rows = P('data')
    mapped = jax.shard_map(functools.partial(chunk, hooks, chunk_size=CHUNK),
                           mesh=Mesh(np.asarray(jax.devices()[:2]), ('data',)), in_specs=rows, out_specs=rows,
                           check_vma=True)

    def run(erase_key):
        return mapped(dataclasses.replace(operands, erase_key=erase_key))

    found = jax.jit(lambda x: jax.jvp(run, (x,), (jnp.ones_like(x),)))(jnp.asarray(operands.erase_key))
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(found))


@pytest.mark.parametrize('by_firsts', [False, True], ids=['by starts', 'by first slots'])
@pytest.mark.parametrize('hooks', [UNTIED, ADDITIVE], ids=['after its decay', 'before its decay'])
def test_a_forked_erases_documents_are_as_exact_as_each_alone(hooks, by_firsts):
    """A forked call of an erase of its own (`Chunk.loads`), each document entering its table slot's memory
    with an entry write of its own, its documents given by their starts or their first slots alone, through
    tokens correcting the state after its decay (an untied erase's laid tokens) or before it (an additive
    one's): its output, last memory and every gradient against each document a sequence of its own a token at
    a time from float64 (check 1), the stored finals each a document's last token's."""
    def drawn(seed):
        operands, seeded = entered_draw(hooks, seed, entries=True)
        return transitioned(hooks, operands, seed), seeded

    assert_forked_on_par('xla', hooks, drawn, f'forked {hooks}', seeds=16, by_firsts=by_firsts)


@pytest.mark.parametrize('form', [{'erase': 'untied', 'delta': False}, {'erase': 'values', 'delta': False},
                                  {'value_decay': True}, {'erase': 'untied', 'value_decay': True},
                                  {'erase': 'values', 'value_decay': True},
                                  {'erase': 'untied', 'reads': 'before'},
                                  {'erase': 'values', 'reads': 'before'}, {'erase': 'erasing'}],
                         ids=['erase without the delta rule', 'values erased without it',
                              'values decayed under it', 'values decayed beside an erase',
                              'values decayed beside their erase', 'read before', 'values erased read before',
                              'unknown'])
def test_a_transition_outside_its_form_is_refused(form):
    """A transition outside its form (`Hooks.erase`, `Hooks.value_decay`): an erase of its own or along
    the values without the delta rule, a decay along the values under it or beside an erase, a readout before
    its token, or an unknown erase, a ValueError (rows 9029, 9028)."""
    with pytest.raises(ValueError):
        recurrence.Hooks(decay='scalar', **form)


def test_a_turned_readout_before_its_tokens_turn_is_refused():
    """A turned state's readout reads it after its token's turn ('after' or 'apart', `Hooks.turns`): the
    chunked op's frames read each token's query in its own turn's frame, so a readout of the state before
    the turn ('before') is refused, where it read the memory a turn late (cos 1 for 1 under a unit turn;
    base review fbb1c2f71 F1)."""
    with pytest.raises(ValueError):
        recurrence.Hooks(decay='scalar', delta=False, write_gate=True, reads='before', turns=1)


def vanishing_entered_draw(seed: int):
    """`entered_draw`'s turned operands, entry writes included, the decays at `VANISHED` vanishing."""
    operands, seeded = entered_draw(TURNED, seed, entries=True)
    return vanishing(operands), seeded


def turned_entered_draw(seed: int):
    """`entered_draw`'s turned operands with entry writes, its cotangents, and a direction of its turns,
    standard normal."""
    operands, seeded = entered_draw(TURNED, seed, entries=True)
    key = jax.random.fold_in(jax.random.key(seed), 8573)
    return operands, jax.random.normal(key, operands.turn.shape, operands.turn.dtype), seeded


def turn_curvature(fn):
    """`fn(operands)`'s outputs against their cotangents, its turns' gradient's derivative along a direction
    of the turns (`jax.jvp` of `jax.grad`): the turns' block of its Hessian, which a forked call's load
    rotations (`Forked.turning`) reach."""
    def run(operands, direction, seeded):
        def gradient(turn):
            def loss(turn):
                out, final = fn(dataclasses.replace(operands, turn=turn))
                return jnp.sum(out * seeded[0]) + jnp.sum(final * seeded[1])

            return jax.grad(loss)(turn)

        return jax.jvp(gradient, (operands.turn,), (direction,))[1]

    return run


def test_a_turned_forked_calls_turn_curvature_is_as_exact_as_each_document_alone():
    """A turned forked call's turns' block of its Hessian along a drawn direction (`turn_curvature`), its
    documents entering their table slots with entry writes (`turned_entered_draw`): on par from float64 with
    each document alone through the same op (check 1, the recurrence a token at a time in float64 the
    truth): a forked call's second order, which its custom VJP refused (row 8573), and which through the
    portable pullback lost each document's load rotation's part (its head review)."""
    def ours(operands):
        return chunk_on('xla', TURNED, entering_documents(operands), chunk_size=CHUNK)

    def own(hooks, piece):
        return chunk_on('xla', hooks, piece, chunk_size=CHUNK)

    def alone(operands):
        return document_by_document(TURNED, operands, own)

    # The gate widens the truth's inputs (`reference_error._truth_and_terms`).
    exact = jax.jit(turn_curvature(lambda operands: document_by_document(TURNED, operands)))
    assert_as_exact(jax.jit(turn_curvature(ours)), jax.jit(turn_curvature(alone)), turned_entered_draw,
                    f"xla {TURNED} a forked call's turn curvature", seeds=32, truth_fn=exact)


def test_a_turned_calls_mixed_derivative_at_zero_queries_is_as_exact_as_the_recurrence():
    """A turned call's turns' gradient differentiated along its queries (`jvp` of `grad`), a third of its
    queries zero, packed: on par from float64 with the recurrence a token at a time's (check 1), where the
    turns' gradient taking out the tokens whose readers were zero by their values lost a zero query's
    derivative (row 8519's head review: [1, 0] read [0, 0])."""
    def drawn(seed):
        (operands,) = draw(TURNED, packed=True, seed=seed)
        rng = np.random.default_rng([8519, seed])
        query = operands.query.at[:, ::3].set(0.0)
        direction, cotangent = (hosted(rng.standard_normal(x.shape)) for x in (query, operands.value))
        return dataclasses.replace(operands, query=query), direction, cotangent

    def mixed(run):
        def second(operands, direction, cotangent):
            def turns_gradient(query):
                given = dataclasses.replace(operands, query=query)

                def loss(turn):
                    return jnp.sum(run(dataclasses.replace(given, turn=turn))[0] * cotangent)

                return jax.grad(loss)(given.turn)

            return jax.jvp(turns_gradient, (operands.query,), (direction,))[1]

        return jax.jit(second)

    exact = mixed(functools.partial(token_by_token, TURNED))
    assert_as_exact(mixed(lambda given: chunk_on('xla', TURNED, given, chunk_size=CHUNK)), exact, drawn,
                    "a turned call's mixed derivative at zero queries", seeds=32, truth_fn=exact)


def assert_finite_where_float64_is(operands, seeded, hooks=TURNED) -> None:
    """`hooks`' chunked op over `operands` on the portable math, its output and every gradient under the
    cotangents `seeded`: finite wherever the recurrence a token at a time's in float64 is a normal float32,
    the leaf named where one is not (a mark's gradient, float0, skipped)."""
    ours = jax.jit(with_gradients(lambda given: chunk_on('xla', hooks, given, chunk_size=CHUNK)))(operands,
                                                                                                   seeded)
    with jax.enable_x64(new_val=True), on_float64_device():
        exact = jax.jit(with_gradients(lambda given: token_by_token(hooks, given)))(*widened((operands,
                                                                                              seeded)))
    leaves = jax.tree_util.tree_leaves_with_path(ours)
    for (path, made), truth in zip(leaves, jax.tree.leaves(exact), strict=True):
        if not jnp.issubdtype(made.dtype, jnp.floating):
            continue
        held = normal_float32(np.asarray(truth))
        wrong = held & ~np.isfinite(np.asarray(made, np.float64))
        assert not wrong.any(), (jax.tree_util.keystr(path), int(wrong.sum()))


def test_a_turned_documents_discarded_memory_keeps_every_gradient_finite():
    """A one-token turned call whose token begins a document (its memory entering, every entry 2^127,
    discarded), a query of its first key, zero keys, values, turns and log decays, an output cotangent of 2
    on each of its first two value columns: its output and every gradient finite wherever float64's is a
    normal float32, the turns' zero, where the turns' gradient contracted the discarded memory with the
    output's cotangent before its zero readers took it out, 2^128 then 0 times inf (row 8519, base review
    d6c1b2c1a F2)."""
    (operands,) = draw(TURNED, packed=False, seed=0, length=1)
    zero = {name: jnp.zeros_like(getattr(operands, name)) for name in ('key', 'value', 'log_decay', 'turn')}
    first = jnp.zeros_like(operands.query).at[..., 0].set(1.0)
    operands = dataclasses.replace(operands, query=first,
                                   beta=jnp.ones_like(operands.beta),
                                   state=jnp.full_like(operands.state, 2.0 ** 127),
                                   starts=jnp.ones(operands.value.shape[:2], bool), **zero)
    d_out = jnp.zeros_like(operands.value).at[..., :2].set(2.0)
    seeded = (d_out, jnp.zeros_like(operands.state))
    assert_finite_where_float64_is(operands, seeded)


@pytest.mark.parametrize("values_first", [False, True], ids=["keys first", "values first"])
def test_an_entry_write_of_no_terms_is_zero(values_first):
    """A document's entry write (`chunked_launch.entry_write`) of rank zero, keys `[2, 0, 3]` and values `[2,
    0, 4]` (or `[2, 4, 0]` laid values first) under a weight `[2, 1, 1]`: zeros `[2, 3, 4]` (`[2, 4, 3]`),
    the memory it enters unchanged, and its weight's gradient zero, where its aligned sum of no terms raised
    (row 8967)."""
    from dew.nn.kernels.chunked_launch import entry_write

    key = jnp.zeros((2, 0, 3), jnp.float32)
    value = jnp.zeros((2, 4, 0) if values_first else (2, 0, 4), jnp.float32)
    weight = jnp.ones((2, 1, 1), jnp.float32)

    def written(weight):
        return entry_write(key, value, weight, values_first=values_first)

    shape = (2, 4, 3) if values_first else (2, 3, 4)
    np.testing.assert_array_equal(np.asarray(written(weight)), np.zeros(shape))
    gradient = jax.grad(lambda weight: jnp.sum(written(weight)))(weight)
    np.testing.assert_array_equal(np.asarray(gradient), np.zeros((2, 1, 1)))


@pytest.mark.parametrize('hooks', [*HOOKS, TURNED], ids=str)
def test_a_document_start_takes_a_memory_past_the_largest_out_before_reading_it(hooks):
    """A one-token call whose token begins a document, its memory entering 2^127 in every entry (discarded),
    unit queries over every key, zero keys, values, turns and log decays, a unit output cotangent: its output
    and every gradient finite wherever float64's is a normal float32, on the portable math, where the
    memory was read, its sum over the keys past float32's largest, before the start took it out, zero times
    inf (row 8560)."""
    (operands,) = draw(hooks, packed=False, seed=0, length=1)
    names = [name for name in ('key', 'value', 'log_decay', 'turn') if getattr(operands, name) is not None]
    operands = dataclasses.replace(operands, query=jnp.ones_like(operands.query),
                                   state=jnp.full_like(operands.state, 2.0 ** 127),
                                   starts=jnp.ones(operands.value.shape[:2], bool),
                                   **{name: jnp.zeros_like(getattr(operands, name)) for name in names})
    seeded = (jnp.ones_like(operands.value), jnp.zeros_like(operands.state))
    assert_finite_where_float64_is(operands, seeded, hooks)


@pytest.mark.parametrize('where', ['entering', 'written'])
def test_a_live_spans_decay_that_underflows_reads_as_the_recurrence(where):
    """A three-token call of one document under log decays of -80 (each live, a span of two `exp(-160)`,
    zero in float32) of a scalar write gate's recurrence, unit queries, a NaN in the memory entering or in
    the first token's key: its output NaN where the recurrence a token at a time reads NaN, on the portable
    math, where the readout and the pairs (`scan._paired`, `scan.spread`) took a decay rounded to zero for a
    document's start and read zero (row 8560's head review, train 1079's review F1)."""
    hooks = recurrence.Hooks(decay='scalar', delta=False, write_gate=True)
    (operands,) = draw(hooks, packed=False, seed=0, length=3)
    key, state = jnp.zeros_like(operands.key), jnp.zeros_like(operands.state)
    if where == 'entering':
        state = jnp.full_like(state, jnp.nan)
    else:
        key = key.at[:, 0].set(jnp.nan)
    operands = dataclasses.replace(
        operands, query=jnp.ones_like(operands.query), log_decay=jnp.full_like(operands.log_decay, -80.0),
        state=state, key=key, starts=jnp.zeros(operands.value.shape[:2], bool))
    out, _ = chunk_on('xla', hooks, operands, chunk_size=CHUNK)
    reference, _ = token_by_token(hooks, operands)
    np.testing.assert_array_equal(np.isnan(np.asarray(out)), np.isnan(np.asarray(reference)))


def test_a_turned_calls_cotangent_near_the_largest_across_documents_keeps_every_gradient_finite():
    """Two one-token turned documents, zero queries, keys, turns and log decays, unit write gates, values of
    two and output cotangents of 2^127: its output and every gradient finite wherever float64's is a normal
    float32, the turns' zero, where the turns' gradient's own pairs contracted the later document's cotangent
    with the earlier one's value, 2^128, before their zero cross-document weight, NaN (row 8566)."""
    (operands,) = draw(TURNED, packed=False, seed=0, length=2)
    zero = {name: jnp.zeros_like(getattr(operands, name)) for name in ('query', 'key', 'log_decay', 'turn')}
    operands = dataclasses.replace(operands, value=jnp.full_like(operands.value, 2.0),
                                   beta=jnp.ones_like(operands.beta), state=jnp.zeros_like(operands.state),
                                   starts=jnp.ones(operands.value.shape[:2], bool), **zero)
    seeded = (jnp.full_like(operands.value, 2.0 ** 127), jnp.zeros_like(operands.state))
    assert_finite_where_float64_is(operands, seeded)


def cancelling_entries_draw(seed: int):
    """`entered_draw`'s turned forked call, each document's entry write of three terms: two of one key
    `(2^80, 0, ...)` under values of `2^47 u` and `-2^47 u` (`u` drawn in [1, 1.9]), their products 2^127
    times `u` and cancelling, and a drawn third, more than float32's exponent range below them."""
    operands, seeded = entered_draw(TURNED, seed, entries=True)
    rng = np.random.default_rng([8501, seed])
    key, value = operands.entry_key, operands.entry_value
    big = jnp.zeros_like(key[:, :, :1]).at[..., 0].set(2.0 ** 80)
    u = jnp.asarray(2.0 ** 47 * rng.uniform(1.0, 1.9, value[:, :, :1].shape), value.dtype)
    return dataclasses.replace(operands, entry_key=jnp.concatenate([big, big, key[:, :, :1]], axis=2),
                               entry_value=jnp.concatenate([u, -u, value[:, :, :1]], axis=2)), seeded


ENTRY_CASES = ["a tie", "a band below", "three levels beside the largest", "the levels' own ties"]
"""Entry writes whose terms cancel across exponent bands beside a surviving last one: a tie, a band below,
three levels beside the largest of drawn factors, and those levels as `contract_support.CANCELLING_LEVELS`
gives them, each key and value scaled by a drawn power of two, so its products keep their exact ties (row
8840)."""


@pytest.mark.parametrize("case", ENTRY_CASES)
@pytest.mark.parametrize("values_first", [False, True], ids=["keys first", "values first"])
def test_an_entry_write_of_cancelling_terms_is_its_exact_sum(values_first, case):
    """A document's entry write (`chunked_launch.entry_write`) of cancelling terms across exponent bands
    beside a surviving last term: its write, its tangent along the weight and the weight's gradient on par
    from float64 with the surviving last term alone (check 1, row 8715), in both layouts, where each band
    rounded its own sum (base review 644ef5535 F1), its bands' totals aligned at once dropped 2^-100 (base
    review f5e6e678d F1), and three levels cancelling beside the largest summed to inf (row 8757)."""
    from dew.nn.kernels.chunked_launch import entry_write

    def draw(seed):
        case_id = ENTRY_CASES.index(case)
        rng = np.random.default_rng([8715, seed, int(values_first), case_id])
        rows = 16
        if case == "the levels' own ties":
            keys, values = (np.asarray(part) for part in zip(*CANCELLING_LEVELS, strict=True))
            key = (keys[:, None] * 2.0 ** -rng.integers(0, 4, (rows, 1, 4))).astype(np.float32)
            terms = (values[:, None] * 2.0 ** -rng.integers(1, 4, (rows, 1, 4))).astype(np.float32)
        elif case == "three levels beside the largest":
            k0 = (2.0 ** 26 * rng.uniform(0.5, 0.9, (rows, 1, 4))).astype(np.float32)
            k1 = (2.0 * rng.uniform(0.5, 0.9, (rows, 1, 4))).astype(np.float32)
            k2 = rng.uniform(0.5, 0.9, (rows, 1, 4)).astype(np.float32)
            k3 = rng.uniform(0.5, 0.9, (rows, 1, 4)).astype(np.float32)
            key = np.concatenate([k0, k1, k2, k1, k2, k0, k3], axis=1)
            v0 = (2.0 ** 127 * rng.uniform(0.5, 0.9, (rows, 1, 4))).astype(np.float32)
            v1 = (2.0 ** 127 * rng.uniform(0.5, 0.9, (rows, 1, 4))).astype(np.float32)
            v2 = (2.0 ** 103 * rng.uniform(0.5, 0.9, (rows, 1, 4))).astype(np.float32)
            last = (0.5 * np.finfo(np.float32).max * rng.uniform(0.5, 0.9, (rows, 1, 4))).astype(np.float32)
            terms = np.concatenate([v0, v1, -v2, -v1, v2, -v0, last], axis=1)
        else:
            scale = 2.0 if case == "a tie" else 2.0 ** -100
            k0 = rng.uniform(0.5, 1.5, (rows, 1, 4)).astype(np.float32)
            k1 = rng.uniform(0.5, 1.5, (rows, 1, 4)).astype(np.float32)
            k2 = rng.uniform(0.5, 1.5, (rows, 1, 4)).astype(np.float32)
            key = np.concatenate([k0, k0, k1, k1, k1, k2], axis=1)
            u0 = (2.0 ** 127 * rng.uniform(0.5, 0.9, (rows, 1, 4))).astype(np.float32)
            u1 = (2.0 ** 25 * rng.uniform(0.5, 0.9, (rows, 1, 4))).astype(np.float32)
            last = (scale * rng.uniform(0.5, 1.5, (rows, 1, 4))
                    * rng.choice([-1.0, 1.0], (rows, 1, 4))).astype(np.float32)
            terms = np.concatenate([u0, -u0, u1, -0.5 * u1, -0.5 * u1, last], axis=1)
        value = np.swapaxes(terms, -1, -2) if values_first else terms
        weight = rng.uniform(0.5, 1.0, (rows, 1, 1)).astype(np.float32)
        dweight = rng.uniform(0.5, 1.0, (rows, 1, 1)).astype(np.float32)
        cotangent = (rng.uniform(-0.05, 0.05, (rows, 4, 4)) if case in ENTRY_CASES[2:]
                     else rng.standard_normal((rows, 4, 4))).astype(np.float32)
        return (jnp.asarray(key), jnp.asarray(value), jnp.asarray(weight), jnp.asarray(dweight),
                jnp.asarray(cotangent))

    def evaluated(written):
        def run(key, value, weight, dweight, cotangent):
            out, tangent = jax.jvp(lambda w: written(key, value, w), (weight,), (dweight,))
            _, pull = jax.vjp(lambda w: written(key, value, w), weight)
            return out, tangent, pull(cotangent)[0]

        return jax.jit(run)

    def formula(key, value, weight):
        if values_first:
            return weight * jnp.einsum("...vr,...rk->...vk", value, key)
        return weight * jnp.einsum("...rk,...rv->...kv", key, value)

    def surviving(key, value, weight):
        return formula(key[:, -1:], value[:, :, -1:] if values_first else value[:, -1:], weight)

    ours = evaluated(lambda k, v, w: entry_write(k, v, w, values_first=values_first))
    exact_surviving = evaluated(surviving)
    assert_as_exact(ours, exact_surviving, draw,
                    f"entry_write cancelling bands, values_first={values_first}, case={case}",
                    seeds=FEWEST_SEEDS, truth_fn=exact_surviving)


def test_a_turned_forked_calls_cancelling_entry_write_is_as_exact_as_each_document_alone():
    """A turned forked call whose documents' entry writes take two terms of 2^127 that cancel beside a third
    more than float32's exponent range below them (`cancelling_entries_draw`): its output, the row's last
    memory and every gradient on par with each document alone (`assert_forked_on_par`, check 1), where the
    turns' gradient formed the write's terms one at a time and summed them, inf - inf (base review 2354cbaf7
    F1), the weight's gradient summed `aligned_sum`'s terms plainly (row 8501), and the write aligned to its
    largest term dropped the third before the two cancelled (base review 224fcf53a F1)."""
    assert_forked_on_par('xla', TURNED, cancelling_entries_draw,
                         f"xla {TURNED} a forked call's cancelling entry writes", seeds=160)


def levels_entered_draw(hooks, seed: int):
    """`entered_draw`'s forked call under `hooks` whose documents' entry writes are `CANCELLING_LEVELS`'
    (`contract_support.cancelling_entries`, its cotangents too), nothing else written, decayed or skipped,
    and each token's memory's first row read by a unit query: its output, last memory and entry writes'
    gradients each a sum float32 holds exactly or rounds once, as float64's is."""
    operands, _ = entered_draw(hooks, seed, entries=True)
    entries = cancelling_entries(seed, KEY_HEADS, KEY_WIDTH, VALUE_HEADS, VALUE_WIDTH)
    query = np.zeros(operands.query.shape)
    query[..., 0] = 1.0
    # Every other operand zero: no key, value, log decay, write gate or step (a rated decay's `exp(A 0)`),
    # skip or memory given.
    zeroed = {name: jnp.zeros_like(getattr(operands, name))
              for name in ('key', 'value', 'log_decay', 'beta', 'skip', 'state')
              if getattr(operands, name) is not None}
    drawn = dataclasses.replace(operands, query=hosted(query), entry_key=hosted(entries.key),
                                entry_value=hosted(entries.value), entry_weight=hosted(entries.weight),
                                **zeroed)
    final = (ROWS, VALUE_HEADS, KEY_WIDTH, VALUE_WIDTH)
    return drawn, (hosted(entries.signed(operands.value.shape)), hosted(entries.signed(final)))


@pytest.mark.parametrize('hooks', [pytest.param(HOOKS[4], id='delta-rule'),
                                   pytest.param(FORMED[4], id='ssd')])
def test_a_forked_calls_entry_writes_of_cancelling_levels_are_exact(hooks):
    """A forked call on the portable math's walk, of the gated delta rule and of Mamba2's SSD (the rated
    scan's XLA handoff, its TPU scan's in test_ssd_kernel), whose documents' entry writes are three levels
    cancelling beside float32's largest (`levels_entered_draw`): its output, the row's last memory and its
    entry keys', values' and weights' gradients held by check 1 against each document a sequence of its own
    the recurrence a token at a time in float64, rounded once (exact, so the walk must be: a write summed in
    some order dropped a level at its tie as its negative survived, an ulp here and inf at float32's largest,
    row 8757; rows 8840, 9480, 9513)."""

    def leaves(walk):
        def run(operands, seeded):
            def entered(key, value, weight):
                return walk(dataclasses.replace(operands, entry_key=key, entry_value=value,
                                                entry_weight=weight))

            out, pull = jax.vjp(entered, operands.entry_key, operands.entry_value, operands.entry_weight)
            return dict(zip(('output', 'final', 'entry keys', 'entry values', 'entry weights'),
                            (*out, *pull(seeded)), strict=True))

        return run

    def ours(operands):
        out, final = chunk_on('xla', hooks, entering_documents(operands), chunk_size=CHUNK)
        return out.astype(jnp.float32), final.astype(jnp.float32)

    # The gate widens a truth's inputs; the reference is that truth rounded once.
    truth = jax.jit(leaves(functools.partial(document_by_document, hooks)))
    assert_as_exact(jax.jit(leaves(ours)), ROUNDED,
                    functools.partial(levels_entered_draw, hooks),
                    f"xla {hooks} a forked call's entry writes of cancelling levels",
                    seeds=cancelling_orbit(KEY_HEADS, VALUE_HEADS, VALUE_WIDTH), truth_fn=truth)


@pytest.mark.parametrize(('by_firsts', 'vanished'), [(False, False), (True, False), (False, True)],
                         ids=['starts', 'firsts', 'vanishing'])
def test_a_turned_states_forked_documents_are_as_exact_as_each_alone(by_firsts, vanished):
    """A state that turns its key pairs (`Hooks.turns`) over a forked call through the portable math, each
    document taking up its table slot's memory with its entry write added, its stored finals in the walk's
    table (Mamba-3's prefix and its questions, a true-frame memory each: each document read in its own last
    token's frame, the memory it takes up turned in as its walk loads it, `templates.recurrence.turned`),
    against each document a sequence of its own the recurrence a token at a time (`assert_forked_on_par`,
    check 1); the call giving its documents by their starts or by their first slots alone (`Chunk.firsts`),
    which the frames read as one document (base review fbb1c2f71 F4), and its decays vanishing in a chunk
    (`vanishing_entered_draw`, audit KPM-02). Its oracle the recurrence: `TURNED`'s note."""
    drawing = (vanishing_entered_draw if vanished
               else functools.partial(entered_draw, TURNED, entries=True))
    assert_forked_on_par('xla', TURNED, drawing, f'xla {TURNED} a forked call\'s documents', seeds=32,
                         by_firsts=by_firsts)


TURNED_LONG = 512
"""A long turned row's tokens: its documents' angles each a few hundred tokens' turns."""

LONG_DOCUMENTS = ((0, 150, 330), (0, 210, 400))
"""Each long row's documents' first tokens: three documents of 110 to 210 tokens a row."""


def short_memory_draw(seed: int, forked: bool):
    """`draw`'s (`entered_draw`'s where `forked`, its entry writes too) operands of `TURNED_LONG` tokens for
    `TURNED`, each token's log decay in (-3, -2), a memory keeping a few tokens, and cotangents."""
    if forked:
        operands, seeded = entered_draw(TURNED, seed, entries=True, length=TURNED_LONG)
    else:
        (operands,) = draw(TURNED, packed=False, seed=seed, length=TURNED_LONG)
        seeded = cotangents(operands, seed)
    rng = np.random.default_rng([seed, 7419])
    log_decay = hosted(rng.uniform(-3.0, -2.0, operands.log_decay.shape))
    return dataclasses.replace(operands, log_decay=log_decay), seeded


@pytest.mark.parametrize('layout', ['one sequence', 'packed', 'forked'])
def test_a_turned_states_long_documents_under_a_short_memory_are_as_exact_as_the_recurrence(layout):
    """A state that turns its key pairs (`Hooks.turns`) over long documents (`TURNED_LONG` tokens,
    `LONG_DOCUMENTS`) under a decay keeping a few tokens (`short_memory_draw`), through the portable math,
    against the recurrence a token at a time (each document alone from its table slot's memory where
    forked): output, final memory and every operand's gradient from float64 (check 1). Every document reads
    in its own last token's frame (`templates.recurrence.turned`): read in frames from its root's first
    token and turned back, a recent write took two roundings of a whole turn the recurrence's takes none of,
    the final memory's worst row 9.6 times the recurrence's (main's per-chunk frames 8.9; row 7419). Its
    oracle the recurrence: `TURNED`'s note."""
    if layout == 'forked':
        assert_forked_on_par('xla', TURNED, functools.partial(short_memory_draw, forked=True),
                             f'xla {TURNED} long forked documents under a short memory', seeds=32,
                             documents=LONG_DOCUMENTS)
        return

    def turned(operands, starts=None):
        return chunk_on('xla', TURNED, dataclasses.replace(operands, starts=starts), chunk_size=CHUNK)

    def tokens(operands, starts=None):
        return token_by_token(TURNED, dataclasses.replace(operands, starts=starts))

    assert_packed_on_par(turned, tokens, functools.partial(short_memory_draw, forked=False),
                         f'{TURNED} long documents under a short memory', packed=layout == 'packed',
                         documents=LONG_DOCUMENTS, length=TURNED_LONG, seeds=32)


def _forked_rows(operands, rows: tuple, length: int):
    """`operands`' first row's first `length` tokens as a forked call of `rows`: each a row's documents' first
    tokens, table slots and stored slots (`Chunk.loads`, `Chunk.stores`), over `GIVEN` given memories and
    `HELD` stored slots; every row the same tokens."""
    def taken(x):
        return None if x is None else jnp.broadcast_to(x[:1, :length], (len(rows), length, *x.shape[2:]))
    starts = jnp.zeros((len(rows), length), bool)
    for row, (begins, _, _) in enumerate(rows):
        starts = starts.at[row, jnp.asarray(begins)].set(True)
    named = {name: taken(getattr(operands, name)) for name in ('query', 'key', 'value', 'log_decay', 'beta',
                                                               'turn')}
    return dataclasses.replace(
        operands, **{name: value for name, value in named.items() if value is not None},
        state=jnp.broadcast_to(operands.state[:1], (len(rows), *operands.state.shape[1:])), starts=starts,
        loads=jnp.asarray([loads for _, loads, _ in rows], jnp.int32),
        stores=jnp.asarray([stores for _, _, stores in rows], jnp.int32), held=HELD)


@pytest.mark.parametrize('case', ['two trees', 'a later sibling'])
def test_a_turned_forked_documents_arithmetic_reads_only_its_ancestors(case):
    """A turned forked call's documents read no document but their ancestors (test_mixers' no token reaches
    another document, for forks): `two trees` a row of a document and its child, then an independent root
    and its child; `a later sibling` a document and its child, then a second child of the same document. The
    later tree's or sibling's tokens drawn again, the program's shapes and table the same: the shared
    documents' outputs bit for bit, one program's (row 7977), where a row's last token's frame moved an
    earlier tree's rotations by a rounding (row 7419)."""
    (operands,) = draw(TURNED, packed=False, seed=0)
    (other,) = draw(TURNED, packed=False, seed=1)
    operands, other = (dataclasses.replace(x, state=jnp.zeros((ROWS, GIVEN, *x.state.shape[1:])))
                       for x in (operands, other))
    row, length = {'two trees': (((0, 23, 40, 55), (0, 2, 0, 3), (1, 0, 2, 0)), 70),
                   'a later sibling': (((0, 23, 40), (0, 2, 2), (1, 0, 0)), 64)}[case]
    rows, shared = (row, row), 40
    drawn = ('query', 'key', 'value', 'log_decay', 'beta', 'turn')
    redrawn = dataclasses.replace(operands, **{
        name: getattr(operands, name).at[:, shared:].set(getattr(other, name)[:, shared:])
        for name in drawn if getattr(operands, name) is not None})

    def run(forked):
        return chunk_on('xla', TURNED, forked, chunk_size=CHUNK)[0]

    together, changed = (np.asarray(run(_forked_rows(x, rows, length))) for x in (operands, redrawn))
    # Each later document its own change, one's hiding none of another's; the shared ones' bits, signed zeros.
    for first, last in itertools.pairwise((*(start for start in row[0] if start >= shared), length)):
        assert not np.array_equal(together[:, first:last], changed[:, first:last]), (first, last)
    assert np.array_equal(together[:, :shared].view(np.uint8), changed[:, :shared].view(np.uint8))


DOCUMENTS = ((0, 23, 40), (0, 9, 48))
"""Each row's packed documents' first tokens under `draw`'s `packed`."""

def entered_draw(hooks, seed: int, entries: bool = False, length: int = LENGTH):
    """`draw`'s packed operands of `length` tokens over a table of `GIVEN` memories a row, with `entries` an
    entry write a document (`Chunk.entry_key`: two terms a document, a weight a value head), and cotangents
    of the output and the row's last memory."""
    (operands,) = draw(hooks, packed=True, seed=seed, length=length)
    keys = jax.random.split(jax.random.fold_in(jax.random.key(seed), 9), 6)
    given = 0.3 * jax.random.normal(keys[0], (ROWS, GIVEN, *operands.state.shape[1:]))
    out = jax.random.normal(keys[1], operands.value.shape)
    final = jax.random.normal(keys[2], operands.state.shape)
    # The documents' bounds and slots are the call's structure, which the arms close over: no gradient.
    drawn = dataclasses.replace(operands, state=given, starts=None)
    if entries:
        documents = len(DOCUMENTS[0])
        drawn = dataclasses.replace(
            drawn,
            entry_key=0.3 * jax.random.normal(keys[3], (ROWS, documents, 2, *operands.key.shape[2:])),
            entry_value=0.3 * jax.random.normal(keys[4], (ROWS, documents, 2, *operands.value.shape[2:])),
            entry_weight=jax.random.uniform(keys[5], (ROWS, documents, operands.value.shape[2])))
    return drawn, (out, final)


def entry_write(operands, row: int, document: int):
    """The entry write `operands` give a row's document (`Chunk.entry_key`), `[Hv, Dk, Dv]`: its weight
    times its terms' summed outer products, each key head's keys read by its value heads."""
    group = operands.entry_value.shape[3] // operands.entry_key.shape[3]
    key = jnp.repeat(operands.entry_key[row, document], group, axis=1)
    written = jnp.einsum('rhk,rhv->hkv', key, operands.entry_value[row, document])
    return operands.entry_weight[row, document][:, None, None] * written


def entering_documents(operands, documents=DOCUMENTS, by_firsts: bool = False, added: bool = False):
    """`operands` with `documents` (each row's first tokens, `DOCUMENTS` where not given) a forked call's,
    each entering its `FORKED` table slot, given by their starts or, `by_firsts`, by their first slots alone
    (`Chunk.firsts`), and where `added`, adding its `ADDS` slot to it (`Chunk.adds`)."""
    loads, stores = (jnp.asarray([row[part] for row in FORKED], jnp.int32) for part in (0, 1))
    adds = jnp.asarray(ADDS, jnp.int32) if added else None
    if by_firsts:
        return dataclasses.replace(operands, starts=None, firsts=jnp.asarray(documents, jnp.int32),
                                   loads=loads, stores=stores, adds=adds, held=HELD)
    starts = jnp.zeros(operands.query.shape[:2], bool)
    for row, begins in enumerate(documents):
        starts = starts.at[row, jnp.asarray(begins)].set(True)
    return dataclasses.replace(operands, starts=starts, loads=loads, stores=stores, adds=adds, held=HELD)


def _neutral(hooks: recurrence.Hooks, name: str) -> float:
    """A padded token's value of field `name` under `hooks` (`stacked_documents`'): zero, or where the hooks
    form the decay and the write gate from pre-activations (`recurrence.gated`), one far below zero."""
    formed = name == 'gate' or (name == 'beta' and hooks.gates is not None)
    return -1e30 if formed else 0.0


def document_by_document(hooks, operands, run=None, documents=DOCUMENTS, added: bool = False):
    """Each document of `operands` a sequence of its own through `run(hooks, piece)` (`token_by_token`
    where None), `documents` its rows' first tokens (`DOCUMENTS` where not given), in the row's order from
    its `FORKED` table slot's memory (a given one, or the final an earlier document stored, with its `ADDS`
    slot's added where `added`, its entry write where `operands` give one), its outputs back in
    place, and each row's last memory. The rows' k-th documents run as one call, a row each over the longest
    document's tokens, each padded after its own with tokens that leave its memory as it is
    (`stacked_documents`, `_neutral`):
    a call a document, the template's reference arm certified each document's launches at its own length,
    the six documents' 18 sets of the GPU tier's three forms, and with the float64 truth's six traces took
    240 s of an H100 job's harness (MAINGPU-be0a83791-PART2of10, tests/arm_arithmetic.py:147)."""
    run = token_by_token if run is None else run
    named = [name for name in ('query', 'key', 'value', 'log_decay', 'gate', 'beta', 'z', 'turn', 'erase_key',
                               'erase_read', 'value_decay') if getattr(operands, name) is not None]
    bounds = [tuple(zip(begins, (*begins[1:], operands.query.shape[1]), strict=True)) for begins in documents]
    longest = max(end - begin for row in bounds for begin, end in row)
    # A token's updates (`Hooks.substeps`) its keys', values' and write gates' in turn.
    every = {name: hooks.substeps if name in ('key', 'value', 'beta') else 1 for name in named}
    tables = [list(operands.state[row]) + [None] * HELD for row in range(len(documents))]
    pieces = [[] for _ in documents]
    finals = []

    for document in range(len(documents[0])):
        states = []
        for row, (loads, _) in enumerate(FORKED):
            load = loads[document]
            state = tables[row][load - 1] if load else jnp.zeros_like(operands.state[row, 0])
            if added and ADDS[row][document]:
                state = state + tables[row][ADDS[row][document] - 1]
            if operands.entry_key is not None:
                state = state + entry_write(operands, row, document)
            states.append(state)
        spans = [(row, *bounds[row][document]) for row in range(len(documents))]
        piece = dataclasses.replace(
            operands, **{name: stacked_documents(getattr(operands, name),
                                                 [(row, begin * every[name], end * every[name])
                                                  for row, begin, end in spans],
                                                 longest * every[name], _neutral(hooks, name))
                         for name in named},
            state=jnp.stack(states), starts=None, loads=None, stores=None, held=0, entry_key=None,
            entry_value=None, entry_weight=None)
        out, finals = run(hooks, piece)
        for row, (_, stores) in enumerate(FORKED):
            begin, end = bounds[row][document]
            pieces[row].append(out[row, :end - begin])
            if stores[document]:
                tables[row][GIVEN + stores[document] - 1] = finals[row]
    return jnp.stack([jnp.concatenate(row) for row in pieces]), finals


@pytest.mark.check1_shards(seeds=80, per_task=8)
@pytest.mark.parametrize('hooks', [HOOKS[4], HOOKS[9], HOOKS[6], FORMED[0], FORMED[9]], ids=str)
def test_documents_entering_memories_of_their_own_are_as_exact_as_each_alone(hooks):
    """80 draws, 8 a task: at 40 a skip's and a gated decay's bias's gradients, two entries a head, were
    inconclusive (intervals 1.58 and 1.56 wide, pooled 1.09 and 0.95; jsf-1506b); the gate holds a draw's
    float64 terms of the walk and its table, 6.4 GB at 8 draws on armada (11-10), so 16 a task passed a
    task's 12 GB.
    The portable math's walk over a forked call, each document laid from a chunk's edge and its table slot's
    memory taken up there, the stored finals kept in the walk's table (`xla_mainloop`, `_walk`), against
    each document a sequence of its own the recurrence a token at a time (`assert_forked_on_par`): a slot
    entered twice sums both documents' gradients. RWKV6's readout of the memory entering its token
    (`Hooks.reads`) among them: a window refused it, its document's memory entering mid-chunk (base review
    80b5b68ad F1)."""
    assert_forked_on_par('xla', hooks, functools.partial(entered_draw, hooks),
                         f'xla {hooks} a forked call\'s documents', seeds=80)


@pytest.mark.check1_shards(seeds=80, per_task=8)
@pytest.mark.parametrize('hooks', [HOOKS[4], FORMED[0]], ids=str)
def test_documents_adding_a_second_slot_are_as_exact_as_each_alone(hooks):
    """The portable math's walk over a forked call whose documents each add a second table slot to the memory
    they enter (`Chunk.adds`, `ADDS`: a given memory, and a final stored two documents before), against each
    document a sequence of its own from the two memories summed (`assert_forked_on_par`): an added slot's
    gradient its document's too. The template's and the TPU scan's tables take none yet (`ADDS_HANDOFF`)."""
    assert_forked_on_par('xla', hooks, functools.partial(entered_draw, hooks),
                         f'xla {hooks} a forked call\'s documents adding a slot', seeds=80, added=True)


@pytest.mark.parametrize('hooks', [HOOKS[4], HOOKS[6], HOOKS[8]], ids=str)
def test_the_portable_maths_products_state_ieee_float32_as_a_gpu_runs_them(hooks):
    """The portable math's training step over fp32 operands, traced as an A100 runs it under the run's default
    numerics: every fp32 product it states IEEE float32 (`recurrence_algebra.STATE_PRECISION`), the derived
    scan's own among them (`scan._portable`), never JAX's configured precision, whose default an A100's
    compiler ran as TF32 (M-JS-96: 1823 times the fp32 recurrence's distance from float64). The delta rule
    with a decay a head and a key dimension, and a write gate without it."""
    (operands,) = draw(hooks, packed=False, seed=0)
    step = with_gradients(functools.partial(chunk_on, 'xla', hooks, chunk_size=CHUNK))
    with generation.assumed('sm80'):
        precisions = dot_precisions(step, operands, cotangents(operands, 0))
    highest = (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST)
    narrowed = {(str(stated), shapes) for stated, dtype, shapes in precisions
                if dtype == jnp.float32 and stated != highest}
    assert precisions and not narrowed, narrowed


def test_a_decoder_of_equal_gated_delta_net_layers_traces_the_portable_walk_once():
    """A Qwen3.5 toy decoder (tests/architecture_cases.py's, three equal gated delta net layers) traced
    unrolled (`layer_scan='plain'`) for its training step on the portable math: each layer's walk the one
    trace (`_walk_once`: the forms its facts derive once, the walk one jitted function), where every layer
    derived its forms and traced its walk again, 0.65 s a layer (a 4B decoder's plan spent 70 s of its trace
    there, row 5556)."""
    model = tiny_models.build_model("causal_transformer+qwen35", dtype="float32")
    tokens = jnp.arange(2 * 16, dtype=jnp.int32).reshape(2, 16) % 16
    params = nn.unbox(model.init(jax.random.key(0), tokens))["params"]

    def loss(params):
        return jnp.sum(model.apply({"params": params}, tokens).astype(jnp.float32))

    with generation.chosen(gated_delta_rule='xla', layer_scan='plain'):
        graph = jax.make_jaxpr(jax.grad(loss))(params)
    # Each of the walk's traces (the step's forward, and the one its linearization takes) the three layers'.
    layers = collections.Counter(id(equation.params["jaxpr"]) for equation in equations(graph)
                                 if equation.params.get("name") == "recurrence_walk")
    assert layers and set(layers.values()) == {3}, layers


def test_fla_refuses_a_forked_call_named_or_handed_to_it():
    """fla's varlen port runs a call's documents as sequences of their own in parallel, so no forked call
    (`Chunk.loads`), whose documents enter earlier documents' finals, is one it takes (inherently). A forked
    call naming it (`generation.chosen`), and its table of memories `[B, S', Hv, Dk, Dv]` handed to its rule
    directly, are refused alike: the selector's typed refusal, its reason and row, never a shape's error."""
    from dew.nn.kernels.selection import KernelRefused

    hooks = HOOKS[4]
    operands = entering_documents(entered_draw(hooks, 0)[0])
    with pytest.raises(KernelRefused) as named:
        chunk_on('fla', hooks, operands, chunk_size=CHUNK)
    assert recurrence_chunk.PORT_FORKED in named.value.refusals
    g = jnp.zeros(operands.value.shape[:3])
    with pytest.raises(KernelRefused) as handed:
        recurrence_chunk.fla_rule(operands.query, operands.key, operands.value, g, g, operands.state)
    assert handed.value.refusals == [recurrence_chunk.PORT_FORKED]


def assert_forked_on_par(kernel: str, hooks, draw, site: str, *, seeds: int, alone: bool = False,
                         documents=DOCUMENTS, by_firsts: bool = False, added: bool = False) -> None:
    """A forked call on `kernel` (`entering_documents` of `draw`'s operands; `Chunk.loads`: a request's
    questions entering its shared part's final memory, a chain's each its predecessor's): its output, the
    row's last memory and every operand's gradient, the given memories' included, against each document a
    sequence of its own from its table slot's memory in the row's order (`document_by_document`), through
    `kernel`'s call where `alone`, else the recurrence a token at a time, which in float64 is their truth
    (check 1); `documents` its rows' first tokens (`DOCUMENTS` where not given), given by their first slots
    alone where `by_firsts` (`entering_documents`); where `added`, each document adding its `ADDS` slot
    (`Chunk.adds`)."""
    def ours(operands):
        out, final = chunk_on(kernel, hooks, entering_documents(operands, documents, by_firsts, added),
                              chunk_size=CHUNK)
        return out.astype(jnp.float32), final.astype(jnp.float32)

    def own(hooks, piece):
        return chunk_on(kernel, hooks, piece, chunk_size=CHUNK)

    def theirs(operands):
        out, final = document_by_document(hooks, operands, own if alone else None, documents, added)
        return out.astype(jnp.float32), final.astype(jnp.float32)

    wide = jax.jit(with_gradients(lambda operands: document_by_document(hooks, operands, documents=documents,
                                                                        added=added)))

    # The gate widens the truth's inputs (`reference_error._truth_and_terms`).
    assert_as_exact(jax.jit(with_gradients(ours)), jax.jit(with_gradients(theirs)), draw, site, seeds=seeds,
                    truth_fn=wide)


def _measured_routes(choice, generation_name: str, run, *shapes, mesh=None) -> list:
    """The routes `choice`'s selector reports for `run` over operands of `shapes` traced
    (`jax.eval_shape`) on an assumed `generation_name`, under the abstract `mesh` where given: whether a
    receipt-backed cell (`Op.declared`) overrode its kernels' order, its route's refusal 'measured'."""
    with (generation.assumed(generation_name), generation.routes() as found,
          contextlib.nullcontext() if mesh is None else jax.sharding.use_abstract_mesh(mesh)):
        jax.eval_shape(run, *shapes)
    return [(route.kernel, route.refusal is not None and route.refusal.kind == 'measured')
            for route in found if route.choice == choice]


@pytest.mark.parametrize(('rows', 'tokens', 'product', 'packed', 'mesh', 'ported'), [
    (1, 4096, 'forward', False, None, True), (1, 4096, 'training', False, None, True),
    (2, 4096, 'forward', False, None, False), (1, 8192, 'forward', False, None, False),
    (1, 4096, 'forward', True, None, False), (2, 4096, 'forward', False, ('data', 2), True),
    (1, 4096, 'forward', False, ('tensor', 2), False)],
    ids=['timed forward', 'timed training', 'two rows', 'longer', 'packed', 'two rows split', 'heads split'])
def test_a_b200s_port_cell_overrides_only_the_extent_its_receipt_timed(rows, tokens, product, packed, mesh,
                                                                     ported):
    """A bf16 gated delta net of Qwen3.5-27B's heads (16 key heads, 48 value heads, 128 wide) on a B200,
    whose forward and training a receipt timed on fla's port at one unpacked row of 4096 tokens
    (`recurrence_chunk.PORT_CELLS`, `TIMED`): the delta rule's selector runs the port by that receipt where
    each launch's shard computes that (two rows split over two devices among them), and for more rows,
    other tokens, packed documents or heads split apart no receipt overrides its kernels' order, where
    every length, batch and split at those heads took the port (row 8991)."""
    # A training step's mesh, its axes automatic: the rules place each operand (`sharding.grouped_heads`).
    mesh = None if mesh is None else jax.sharding.AbstractMesh((mesh[1],), (mesh[0],),
                                                               axis_types=(jax.sharding.AxisType.Auto,))
    gates = (jax.ShapeDtypeStruct((rows, tokens, 48), jnp.float32),
             jax.ShapeDtypeStruct((rows, tokens, 48), jnp.bfloat16))
    shapes = [jax.ShapeDtypeStruct((rows, tokens, heads, 128), jnp.bfloat16) for heads in (16, 16, 48)]
    starts = [jax.ShapeDtypeStruct((rows, tokens), jnp.bool_)] if packed else []

    def forward(query, key, value, g, beta, *starts):
        given = starts[0] if starts else None
        return recurrence_chunk.chunk_gated_delta_rule(query, key, value, g, beta, starts=given)[0]

    run = forward if product == 'forward' else jax.grad(lambda *o: forward(*o).astype(jnp.float32).sum(),
                                                        argnums=(0, 1, 2))
    found = _measured_routes(recurrence_chunk.GATED_DELTA.choice, 'sm100', run, *shapes, *gates, *starts,
                             mesh=mesh)
    assert all(measured == ported for _, measured in found) and found, found
    assert not ported or {kernel for kernel, _ in found} == {'fla'}, found


@pytest.mark.parametrize(('rows', 'tokens', 'packed', 'heads', 'width', 'chunk_size', 'keys', 'output',
                          'declared'), [
    (1, 4096, False, 96, 80, 256, jnp.bfloat16, jnp.float32, True),
    (1, 4096, False, 64, 64, 128, jnp.bfloat16, jnp.float32, True),
    (1, 4096, False, 96, 80, 256, jnp.bfloat16, jnp.bfloat16, False),
    (1, 4096, False, 64, 64, 128, jnp.bfloat16, jnp.bfloat16, False),
    (1, 4096, False, 96, 80, 256, jnp.float32, jnp.float32, False),
    (2, 4096, False, 96, 80, 256, jnp.bfloat16, jnp.float32, False),
    (1, 2048, False, 96, 80, 256, jnp.bfloat16, jnp.float32, False),
    (1, 4096, True, 96, 80, 256, jnp.bfloat16, jnp.float32, False),
    (1, 4096, False, 96, 96, 256, jnp.bfloat16, jnp.float32, False),
    (1, 4096, False, 32, 64, 128, jnp.bfloat16, jnp.float32, False),
    (1, 4096, False, 64, 64, 256, jnp.bfloat16, jnp.float32, False)],
    ids=['4B', '30B', '4B of a bf16 output', '30B of a bf16 output', '4B of float32 keys', '4B of two rows',
         '4B shorter', '4B packed', 'heads padded to 4B\'s', 'another head count', 'another chunk'])
def test_an_a100s_ssd_port_cells_override_only_the_geometries_their_receipt_timed(rows, tokens, packed, heads,
                                                                                width, chunk_size, keys,
                                                                                output, declared):
    """An SSD's bf16 training on an A100, whose receipt timed mamba_ssm's port faster than the template on
    Nemotron-H 4B's (96 heads of 80 and chunks of 256) and 30B's (64 heads of 64 and chunks of 128), each
    eight groups of 128-wide states, at one unpacked row of 4096 tokens (`recurrence_chunk.SCAN_CELLS`): the
    rated scan's selector runs the port by that receipt on those calls alone, and on another row count,
    length, chunk or head count, packed documents, heads only padded to the port's width as 4B's are (96
    wide), float32 keys, whose call the scan launches in float32, or an output in the operands' bf16, which
    the receipt timed none of (`ScanCell.wide`, row 11812), no receipt overrides its kernels' order, where
    every mma.sync training geometry took the port (row 8991)."""
    def loss(x, B, C, dt, A, D, *starts):
        operands = Chunk(C, B, x, beta=dt, rate=A, skip=D, starts=starts[0] if starts else None)
        out, _ = chunk(FORMED[4], operands, chunk_size=chunk_size, dtype=output)
        return out.astype(jnp.float32).sum()

    shapes = (jax.ShapeDtypeStruct((rows, tokens, heads, width), jnp.bfloat16),
              *[jax.ShapeDtypeStruct((rows, tokens, 8, 128), keys)] * 2,
              jax.ShapeDtypeStruct((rows, tokens, heads), jnp.float32),
              *[jax.ShapeDtypeStruct((heads,), jnp.float32)] * 2,
              *([jax.ShapeDtypeStruct((rows, tokens), jnp.bool_)] if packed else []))
    found = _measured_routes(recurrence_chunk.SCAN.choice, 'sm80', jax.grad(loss, argnums=(0, 1, 2, 3)),
                             *shapes)
    assert any(measured for _, measured in found) == declared, found
    assert not declared or ('pallas', True) in found, found


def test_the_ssd_port_refuses_a_forked_call_named_to_it():
    """mamba_ssm's state passing takes a memory at a sequence's first chunk alone, so no forked call
    (`Chunk.loads`), whose documents take their memories up mid-chunk, is one its port takes (inherently): a
    forked call naming it is refused, the selector's typed refusal and its reason, never a shape's error."""
    from dew.nn.kernels.selection import KernelRefused

    hooks = FORMED[4]
    operands = entering_documents(entered_draw(hooks, 0, entries=True)[0])
    with pytest.raises(KernelRefused) as named:
        chunk_on('pallas', hooks, operands, chunk_size=CHUNK)
    assert recurrence_chunk.SSD_PORT_FORKED in named.value.refusals


def narrowed_draw(hooks, seed: int):
    """`entered_draw`'s operands with their queries, keys, values and write gate in bf16, a 16-bit call as the
    recurrence template's fast backward trains it (`mainloop_backward.fast_backward_form`)."""
    operands, seeded = entered_draw(hooks, seed)
    return dataclasses.replace(operands, **{name: getattr(operands, name).astype(jnp.bfloat16)
                                            for name in ('query', 'key', 'value', 'beta')
                                            if getattr(operands, name) is not None}), seeded


FORKED_TEMPLATE = [HOOKS[4], HOOKS[8], FORMED[4], TURNED]
"""The forked call's forms on the template: the gated delta rule, Kimi delta attention, Mamba2's SSD and a
turned state (Mamba-3's, each document's frame on from its source's, `templates.recurrence.turned`)."""


def the_templates_forked_documents_are_as_exact_as_each_alone(hooks, added: bool = False):
    """The recurrence template over a forked call in 16 bits: its walks take each document's memory up from
    the call's table and keep the finals its documents store (`table.Taking`, `table.Returning`; ledger
    5376), where `added` each document adding its `ADDS` slot (`Chunk.adds`), as its call over each document
    alone (`assert_forked_on_par`), the same launches' arithmetic."""
    assert_forked_on_par('mosaic_gpu', hooks, functools.partial(narrowed_draw, hooks),
                         f'mosaic_gpu {hooks} a forked call\'s documents{" adding a slot" if added else ""}, '
                         'against each alone', seeds=8, alone=True, added=added)


@pytest.mark.check1_shards(seeds=8, per_task=1)
@pytest.mark.parametrize('hooks', FORKED_TEMPLATE, ids=str)
def test_the_templates_forked_documents_are_as_exact_as_each_alone_interpreted(hooks):
    """`the_templates_forked_documents_are_as_exact_as_each_alone` interpreted off a GPU. A draw a task: four
    took 287 to 381 s a task (jsf-table-phase at 693aa1c1c)."""
    the_templates_forked_documents_are_as_exact_as_each_alone(hooks)


@pytest.mark.check1_shards(seeds=8, per_task=1)
@pytest.mark.parametrize('hooks', [hooks for hooks in FORKED_TEMPLATE if not hooks.turns], ids=str)
def test_the_templates_documents_adding_a_slot_are_as_exact_as_each_alone_interpreted(hooks):
    """The template's walks over a forked call whose documents each add a second table slot (`Chunk.adds`,
    `ADDS`): the forward's taking sums both slots (`table.Taking`), the reverse walk returns the memory's
    gradient to both (`table.Returning`, the given slots' sum; Kimi delta attention's taken memories),
    interpreted off a GPU (`the_templates_forked_documents_are_as_exact_as_each_alone`); a turned call's
    documents add none (refused, row 8284)."""
    the_templates_forked_documents_are_as_exact_as_each_alone(hooks, added=True)


@pytest.mark.parametrize('hooks', FORKED_TEMPLATE, ids=str)
def test_a_forked_call_of_entry_writes_of_no_terms_lowers_on_the_template(hooks):
    """A forked call whose documents' entry writes take no terms (`Chunk.entry_key` `[B, D, 0, H, K]`),
    through the public chunked op on the template, its gradient lowered for an H100: each chunk's write
    zero (`chunked_launch.entry_write`, row 8967), where its documents' terms flattened by an inferred
    extent of no elements refused the call (`ForkedChunks.entries`, head review of 512810acf F3)."""
    operands = entering_documents(narrowed_draw(hooks, 0)[0])
    rows, documents, value_heads = operands.value.shape[0], len(DOCUMENTS[0]), operands.value.shape[2]
    operands = dataclasses.replace(
        operands, entry_key=jnp.zeros((rows, documents, 0, *operands.key.shape[2:])),
        entry_value=jnp.zeros((rows, documents, 0, *operands.value.shape[2:])),
        entry_weight=jnp.ones((rows, documents, value_heads)))

    def loss(given):
        out, final = chunk_on('mosaic_gpu', hooks, given)
        return jnp.sum(out.astype(jnp.float32)) + jnp.sum(final)

    with generation.assumed('sm90'):
        jax.jit(jax.grad(loss, allow_int=True)).trace(operands).lower(lowering_platforms=('cuda',))


def test_a_turned_forked_call_adding_a_slot_is_refused():
    """A turned call's documents adding a second table slot (`Chunk.adds` under `Hooks.turns`) are refused:
    its turns' gradient composes each document's frames from the one slot it loads (`turned._forked_frames`),
    so an added slot's turns would take no gradient (row 8284)."""
    operands, _ = entered_draw(TURNED, 0, entries=True)
    with pytest.raises(ValueError):
        chunk_on('xla', TURNED, entering_documents(operands, added=True), chunk_size=CHUNK)


UNDELTA_HEADS = 8
"""The value heads of the no-delta gradient check's draws: a skip's gradient a value of each head, two of them
left a site unresolved at 512 draws, eight at 64 (`seeds_for`, paired, on armada at 40ba1d89c6f)."""

# The forms without the delta rule the template's fast backward takes, a decay a head or none, each site's
# seeds sized by `seeds_for` (paired, at `UNDELTA_HEADS`): a write gate's with and without a skip, Mamba2's
# step gates over one sequence, and lightning attention's. Over packed documents the step gates', and the
# rated scan's (its step's per-head rate), need past 512 draws interpreted, a head's gradient a sum of every
# token's at fp32's rounding in either arm: the stability suite holds the rated scan's gradient natively
# under the accurate forms (tests/test_recurrence_template_stability.py, `forms_of`). A form under an epilogue
# reading the output, whose gradient is fp32, the template refuses (`mainloop_backward.fast_backward_form`).
@pytest.mark.parametrize(('hooks', 'packed', 'seeds'), [
    pytest.param(HOOKS[2], False, 8, id='write-one',
                 marks=pytest.mark.check1_shards(seeds=8, per_task=4)),
    pytest.param(HOOKS[2], True, 8, id='write-packed',
                 marks=pytest.mark.check1_shards(seeds=8, per_task=4)),
    pytest.param(HOOKS[6], False, 8, id='decayed-write-one',
                 marks=pytest.mark.check1_shards(seeds=8, per_task=4)),
    pytest.param(HOOKS[6], True, 8, id='decayed-write-packed',
                 marks=pytest.mark.check1_shards(seeds=8, per_task=4)),
    pytest.param(HOOKS[3], False, 64, id='write-skip-one',
                 marks=pytest.mark.check1_shards(seeds=64, per_task=4)),
    pytest.param(HOOKS[3], True, 64, id='write-skip-packed',
                 marks=pytest.mark.check1_shards(seeds=64, per_task=4)),
    pytest.param(HOOKS[7], False, 64, id='decayed-write-skip-one',
                 marks=pytest.mark.check1_shards(seeds=64, per_task=4)),
    pytest.param(HOOKS[7], True, 64, id='decayed-write-skip-packed',
                 marks=pytest.mark.check1_shards(seeds=64, per_task=4)),
    pytest.param(FORMED[1], False, 128, id='step-gates-one',
                 marks=pytest.mark.check1_shards(seeds=128, per_task=4)),
    pytest.param(FORMED[8], False, 8, id='lightning-one',
                 marks=pytest.mark.check1_shards(seeds=8, per_task=4)),
    pytest.param(FORMED[8], True, 8, id='lightning-packed',
                 marks=pytest.mark.check1_shards(seeds=8, per_task=4)),
])
def test_the_templates_gradient_without_the_delta_rule_is_as_exact_as_the_portable_maths_interpreted(
        hooks, packed, seeds):
    """A 16-bit call without the delta rule through the recurrence template interpreted under the accurate
    forms (the rated scan's numbers mamba_ssm's under the fast ones, `recurrence_chunk.fast_rounding`), its
    fast backward's intermediates held in fp32's three parts as its forward holds them
    (`chunk_forward.prepared`'s `held`): its output, final memory and every operand's gradient on par from
    float64 with the portable math's (check 1; the bar `recurrence_chunk.chunk_rounding` states for a form
    with no port), the truth the recurrence a token at a time in float64."""
    starts = draw(hooks, packed, 0, value_heads=UNDELTA_HEADS)[0].starts

    def drawn(seed):
        (operands,) = draw(hooks, packed, seed, value_heads=UNDELTA_HEADS)
        # The documents' starts are the call's structure, which every arm closes over: no gradient.
        operands = dataclasses.replace(operands, starts=None, **{
            name: getattr(operands, name).astype(jnp.bfloat16) for name in ('query', 'key', 'value')})
        return operands, cotangents(operands, seed)

    def restarted(fn):
        return jax.jit(with_gradients(lambda operands: fn(dataclasses.replace(operands, starts=starts))))

    def on(kernel):
        return restarted(functools.partial(chunk_on, kernel, hooks, chunk_size=CHUNK))

    with generation.run_precision(dataclasses.replace(generation.numerics(), forms='accurate')):
        assert_as_exact(on('mosaic_gpu'), on('xla'), drawn, f'{hooks} packed={packed} on the template',
                        seeds=seeds, truth_fn=restarted(functools.partial(token_by_token, hooks)))


def edge_draw(hooks, seed: int):
    """`draw`'s operands over a chunk's 64 tokens entering a memory of about 2^96, the largest `chunk` reads
    at its size through any decay, which each row's first token decays by exp2(-124.8), a normal fp32 whose
    prefix leaves fp32's normals at the next token: every later token reads the memory near its values."""
    (operands,) = draw(hooks, packed=False, seed=seed, length=64)
    g = operands.log_decay.at[:, 0].set(-124.8 * np.log(2.0))
    memory = jax.random.normal(jax.random.key(seed + 7), operands.state.shape) * 2.0 ** 96
    return (dataclasses.replace(operands, log_decay=g, state=memory),)


@pytest.mark.parametrize('hooks', [hooks for hooks in HOOKS if hooks.decay == 'scalar'], ids=str)
def test_a_memory_read_through_a_decay_past_fp32s_normals_is_as_exact_as_the_recurrence(hooks):
    """The portable math's output and final memory where a memory at `chunk`'s bound is read through a
    chunk's decay prefix past fp32's normals (`edge_draw`), on par with the recurrence a token at a time from
    float64, which decays the memory in place: a decay under 2^-126 reads it as zero, a term under fp32's
    resolution of the output within the bound (row 2344)."""
    dew = jax.jit(functools.partial(chunk_on, 'xla', hooks, chunk_size=CHUNK))
    reference = jax.jit(functools.partial(token_by_token, hooks))
    assert_as_exact(dew, reference, functools.partial(edge_draw, hooks), f'{hooks} past the normals',
                    seeds=FEWEST_SEEDS)


ROUNDED_ONCE = [recurrence.Hooks(decay='none', delta=False, write_gate=True, skip=True),
                recurrence.Hooks(decay='scalar', scale=0.1, skip=True),
                recurrence.Hooks(decay='scalar', scale=0.1, skip=True, output=recurrence.OutputNorm(1e-6))]
"""The rounding sites' hooks: a write gate without a decay, which only the portable math takes, and the delta
rule with a decay, with and without the norm, which the port takes too."""


@pytest.mark.parametrize('hooks, ported', [
    pytest.param(hooks, ported, id=f'{hooks}-{"port" if ported else "portable"}')
    for hooks in ROUNDED_ONCE for ported in ((False, True) if hooks.delta else (False,))])
def test_the_output_rounds_once_after_the_epilogue(hooks, ported):
    """The chunked op rounds its output once, after the skip and the norm, as the decode step's reference
    does, on the portable math and on the port (the delta rule with a decay, its kernels interpreted off
    a GPU): a bf16 token whose readout `0.1` (a unit query and key, the write gate or the scale 0.1)
    the skip `-0.10009765625` nearly cancels leaves the residual, within fp32's rounding over its chain of
    the float64 value (the oracle rule for a fixed input), and the norm reads it, where a readout rounded
    to bf16 first cancels to zero."""
    unit = jnp.zeros((1, 1, 1, 8), jnp.bfloat16).at[..., 0].set(1.0)
    beta = jnp.full((1, 1, 1), 1.0 if hooks.delta else 0.1, jnp.float32)
    operands = Chunk(unit, unit, jnp.ones((1, 1, 1, 8), jnp.bfloat16),
                     log_decay=jnp.zeros((1, 1, 1)) if hooks.decay != 'none' else None, beta=beta,
                     skip=jnp.full((1,), -0.10009765625, jnp.float32),
                     z=jnp.ones((1, 1, 1, 8)) if hooks.output else None,
                     weight=jnp.ones((8,)) if hooks.output else None)
    def op(operands):
        return chunk_on('fla' if ported else 'xla', hooks, operands, dtype=jnp.float32)[0]

    def reference(operands, beta, dtype, hooks=hooks):
        """The decode step's reference over the token under `hooks`, in `dtype`."""
        return recurrence.step_reference(hooks, recurrence.Step(
            jnp.zeros((1, 1, 8, 8), dtype), query=operands.query[:, 0], key=operands.key[:, 0],
            value=operands.value[:, 0],
            log_decay=None if operands.log_decay is None else operands.log_decay[:, 0], beta=beta[:, 0],
            skip=operands.skip, z=None if operands.z is None else operands.z[:, 0],
            weight=operands.weight), dtype=dtype)[0]

    found = op(operands)
    with jax.enable_x64(new_val=True):
        wide = widened((operands, beta))
        truth = np.asarray(reference(*wide, jnp.float64))
        residual = np.asarray(reference(*wide, jnp.float64, hooks=dataclasses.replace(hooks, output=None)))
    # The oracle rule for a fixed input: within fp32's rounding over the op's chain of roundings of its two
    # terms' magnitude (the readout and the skip, each about 0.1), times the norm's gain over the residual
    # where it reads it; a readout rounded to bf16 first misses by the whole residual.
    gain = float(np.max(np.abs(truth)) / np.max(np.abs(residual)))
    terms = 2 * gain * float(np.max(np.abs(np.asarray(operands.skip, np.float64))))
    assert_fp32_reduction_bound(found[:, 0], truth, np.full(truth.shape, terms),
                                chain_roundings(jax.make_jaxpr(op)(operands)))
    assert np.all(np.asarray(found) != 0.0)


def test_the_output_takes_the_dtype_asked_for():
    """An explicit `dtype` holds with and without an epilogue, and bf16 operands under the l2 norm, which
    widens them, come back in bf16."""
    ones = jnp.ones((1, 1, 1, 8), jnp.bfloat16)
    plain = recurrence.Hooks(decay='scalar', scale=0.25)
    wide = Chunk(*(x.astype(jnp.float32) for x in (ones, ones, ones)), log_decay=jnp.zeros((1, 1, 1)),
                 beta=jnp.ones((1, 1, 1)))
    assert chunk_on('xla', plain, wide, dtype=jnp.bfloat16)[0].dtype == jnp.bfloat16
    normed = recurrence.Hooks(decay='scalar', l2norm=True)
    assert chunk_on('xla', normed, Chunk(ones, ones, ones, log_decay=jnp.zeros((1, 1, 1)),
                                         beta=jnp.ones((1, 1, 1))))[0].dtype == jnp.bfloat16


RATED = recurrence.Hooks(decay='scalar', rated=True, delta=False, skip=True)
WHOLE = recurrence.Hooks(decay='scalar', delta=False, write_gate=True, skip=True)
"""A rated decay (`rate` times the write gate) and the same log decay given whole."""

RATED_SEEDS = 8
"""The rated-decay site's declared seeds, sized paired by `seeds_for` on armada."""


def rated_draw(seed: int) -> tuple:
    """A decode token over a memory and a chunked sequence, each with its write gate in [0, 0.5), a rate a
    head in [-2, -0.1) and a skip: `(token, sequence)`, each a dict of the step's or chunk's operands."""
    rng = np.random.default_rng(seed)
    # The token's heads a warpgroup's keys, as the launch steps them (`templates.recurrence.step.fits`).
    rows, heads, width, stepped = 2, 2, 16, 128
    rate = -jnp.asarray(rng.uniform(0.1, 2.0, (heads,)), jnp.float32)
    skip = jnp.asarray(rng.standard_normal((heads,)), jnp.float32)
    token = {name: jnp.asarray(rng.standard_normal((rows, heads, stepped)), jnp.float32)
             for name in ('query', 'key', 'value')}
    token |= {'state': jnp.asarray(rng.standard_normal((rows, heads, stepped, stepped)), jnp.float32),
              'beta': jnp.asarray(rng.uniform(0.0, 0.5, (rows, heads)), jnp.float32), 'rate': rate,
              'skip': skip}
    sequence = {name: jnp.asarray(rng.standard_normal((rows, LENGTH, heads, width)), jnp.float32)
                for name in ('query', 'key', 'value')}
    steps = jnp.asarray(rng.uniform(0.0, 0.5, (rows, LENGTH, heads)), jnp.float32)
    sequence |= {'beta': steps, 'rate': rate, 'skip': skip}
    return token, sequence


def decayed(hooks, token: dict, sequence: dict, launch=recurrence.step) -> tuple:
    """`hooks`' decode token on the `launch` and on its reference, and its chunked sequence on the portable
    math, `rated_draw`'s operands read as `hooks` reads them: a rated decay's rate, or that rate times the
    write gate given whole as the log decay."""
    def operands(given: dict, leading) -> dict:
        given = dict(given)
        rate = given.pop('rate')
        return given | ({'rate': rate} if hooks.rated else {'log_decay': rate[leading] * given['beta']})

    stepped = [step(hooks, recurrence.Step(**operands(token, (None, slice(None)))))[::2]
               for step in (launch, recurrence.step_reference)]
    chunked = chunk_on('xla', hooks, Chunk(**operands(sequence, (None, None, slice(None)))), chunk_size=CHUNK)
    return (*stepped, chunked)


def test_a_rated_decay_is_its_log_decay():
    """A rated decay (`rate` times the write gate) steps a decode token, on the launch and on its reference,
    and a chunked sequence on the portable math as the same log decay given whole: on par with the whole
    decay's from float64 over the site's seeds (check 1), whose truth is the reference's arithmetic in
    float64: the launch takes no float64 operands (interpreted under x64, its key sums' loop failed to
    lower, i32 against i64, once the interpreter joined two halves as a GPU does)."""
    assert_as_exact(functools.partial(decayed, RATED), functools.partial(decayed, WHOLE), rated_draw,
                    'a rated decay against its log decay', seeds=RATED_SEEDS,
                    truth_fn=functools.partial(decayed, WHOLE, launch=recurrence.step_reference))


def test_a_value_read_by_the_write_and_the_skip_rounds_its_gradient_once():
    """A 16-bit value both the recurrence writes and the skip adds (`D v`), their value gradients nearly
    cancelling (`D` near -1 against a write gate near 1): its gradient sums both in the working precision and
    rounds once, as the recurrence a token at a time does (`recurrence_chunk._read_once`). Rounded to bf16
    apart and added in it, the sum kept neither's low bits: a write gate's value gradient's worst row 11.6
    times the token recurrence's (row 5447)."""
    hooks = recurrence.Hooks(decay='none', delta=False, write_gate=True, skip=True)
    operands = Chunk(query=jnp.full((1, 1, 1, 1), 1.0078125, jnp.bfloat16),
                     key=jnp.ones((1, 1, 1, 1), jnp.bfloat16), value=jnp.ones((1, 1, 1, 8), jnp.bfloat16),
                     beta=jnp.full((1, 1, 1), 0.99609375, jnp.bfloat16),
                     skip=jnp.array([-0.9999], jnp.float32), state=jnp.zeros((1, 1, 1, 8), jnp.float32))

    def gradient(run, operands):
        def loss(value):
            out, _ = run(hooks, dataclasses.replace(operands, value=value))
            return jnp.sum(out.astype(jnp.promote_types(out.dtype, jnp.float32)))

        return jax.grad(loss)(operands.value)

    dew = jax.jit(lambda x: gradient(lambda hooks, x: chunk_on('xla', hooks, x), x))(operands)
    reference = jax.jit(lambda x: gradient(token_by_token, x))
    with jax.enable_x64(new_val=True):
        truth = reference(widened((operands,))[0])
    assert_rounds_where_the_reference_does(dew, reference(operands), truth, 'a write and a skip of one value')


def test_the_operands_are_checked_as_the_decode_steps():
    """A hook's missing operand is named, and a decay of the other kind's rank refused, rather than read
    as another computation."""
    ones = jnp.ones((1, 4, 1, 8))
    with pytest.raises(ValueError):
        chunk(recurrence.Hooks(decay='scalar'), Chunk(ones, ones, ones, log_decay=jnp.zeros((1, 4, 1))))
    with pytest.raises(ValueError):
        chunk(recurrence.Hooks(decay='vector'), Chunk(ones, ones, ones, log_decay=jnp.zeros((1, 4, 1)),
                                                      beta=jnp.ones((1, 4, 1))))


def test_aligned_unit_keys_erase_the_decay_gradient():
    """In one dimension k=beta=1 writes S'=v, independently of the decay. The WY inverse/readout
    paths cancelled only after rounding, leaving a spurious 4.74e-8 gradient in this exact zero."""
    unit = jnp.ones((1, 4, 1, 1), jnp.float32)
    value = jnp.array([0.314159, -0.271828, 0.577216, 1.414214], jnp.float32).reshape(unit.shape)
    decay = jnp.array([-0.17, -0.23, -0.31, -0.41], jnp.float32).reshape(unit.shape)
    beta = jnp.ones((1, 4, 1), jnp.float32)
    gradient = jax.jit(jax.grad(lambda g: jnp.sum(recurrence_chunk.xla_mainloop(
        recurrence_chunk.Tiles(4), unit, unit, value, g[..., 0], beta, scale=1.0, delta=True,
        dtype=jnp.float32)[0])))(decay)
    np.testing.assert_array_equal(np.asarray(gradient), np.zeros_like(np.asarray(gradient)))


@pytest.mark.parametrize("length, chunk_size", [(3, 1), (3, 4), (9, 4), (10, 4), (12, 4), (9, 16)])
@pytest.mark.parametrize("packed", PACKED, ids=["one sequence", "packed"])
@pytest.mark.parametrize("delta", [False, True], ids=["write", "delta"])
@pytest.mark.parametrize("channels", [1, 2], ids=["scalar decay", "vector decay"])
def test_a_ragged_last_chunk_keeps_its_tokens_and_document_resets(
        length, chunk_size, packed, delta, channels):
    """The plain-JAX mainloop over rows whose last chunk is ragged, a chunk of one token or of an empty
    prefix, documents starting before, on or after a chunk's first token: every output and the final
    memory the recurrence's exactly. Dyadic operands make the writes and outputs exact, so a misplaced reset
    or token has no tolerance."""
    key = np.zeros((2, length, 1, 2), np.float32)
    key[..., 0] = 1.0
    value = np.broadcast_to(np.arange(1, length + 1, dtype=np.float32)[None, :, None, None] / 4,
                             (2, length, 1, 1)).copy()
    g = np.zeros((2, length, 1, channels), np.float32)
    beta = np.full((2, length, 1), 0.5, np.float32)
    initial = np.full((2, 1, 2, 1), 2.0, np.float32)
    begins = np.zeros((2, length), bool)
    if packed:
        for row, positions in enumerate(((0, chunk_size - 1, chunk_size + 1, length - 1),
                                          (chunk_size - 2, 2 * chunk_size - 1))):
            begins[row, [i for i in positions if 0 <= i < length]] = True
    starts = jnp.asarray(begins) if packed else None
    expected_output = np.zeros_like(value)
    expected_final = initial.copy()
    for row in range(2):
        for t in range(length):
            if begins[row, t]:
                expected_final[row] = 0.0
            predicted = expected_final[row, 0, 0, 0] if delta else 0.0
            expected_final[row, 0, 0, 0] += 0.5 * (value[row, t, 0, 0] - predicted)
            expected_output[row, t, 0, 0] = expected_final[row, 0, 0, 0]
    def run(key, query, value, g, beta, initial):
        decay = g if channels > 1 else g[..., 0]
        return recurrence_chunk.xla_mainloop(recurrence_chunk.Tiles(chunk_size), query, key, value, decay,
                                             beta, initial, starts, scale=1.0, delta=delta, dtype=jnp.float32)

    output, final = jax.jit(run)(*(jnp.asarray(x) for x in (key, key, value, g, beta, initial)))
    np.testing.assert_array_equal(np.asarray(output), expected_output)
    np.testing.assert_array_equal(np.asarray(final), expected_final)


@pytest.mark.parametrize('staged', ['jit', 'remat', 'scan'])
def test_a_kernel_named_for_the_rule_takes_forward_mode_however_a_caller_stages_it(staged):
    """`jax.jvp` of the delta rule on the port (named, `generation.chosen`; interpreted off a GPU) staged by a
    jit, a remat or a scan, as layers stage it (`nn.remat`, `nn.scan`): its tangent the portable math's
    (`_per_product`'s `generation.linear_tangent`), on par with the portable rule's from float64. Its
    selector once routed by whether it saw a JVP tracer, which a remat or a scan hides from it, and there the
    port's custom VJP raised (row 2395)."""
    length, heads, width = 64, 2, 64

    def draw(seed):
        """The rule's operands, the queries and keys of unit norm, and a direction for each."""
        keys = jax.random.split(jax.random.key(41 + seed), 7)
        query, key = (l2norm(jax.random.normal(k, (1, length, 1, width))) for k in keys[:2])
        value = jax.random.normal(keys[2], (1, length, heads, width))
        g = -jnp.exp(jax.random.normal(keys[3], (1, length, heads)) - 3.0)
        beta = jax.nn.sigmoid(jax.random.normal(keys[4], (1, length, heads)))
        state = 0.1 * jax.random.normal(keys[5], (1, heads, width, width))
        operands = (query, key, value, g, beta, state)
        return (*operands, *(jax.random.normal(jax.random.fold_in(keys[6], index), x.shape)
                             for index, x in enumerate(operands)))

    def stage(fn):
        if staged == 'jit':
            return jax.jit(fn)
        if staged == 'remat':
            return jax.checkpoint(fn)
        return lambda *operands: jax.lax.scan(lambda carry, _: (carry, fn(*operands)), None, length=1)[1]

    def tangent(kernel):
        def tangents(*given):
            # The port at the default product precision its fp32 products are declared at (`port_support`).
            with row_names_port('gated_delta_rule', 'fla', port=kernel == 'fla'):
                found = jax.jvp(stage(recurrence_chunk.chunk_gated_delta_rule), given[:6], given[6:])[1]
            return dict(zip(('output', 'memory'), jax.tree.leaves(found), strict=True))
        return tangents

    assert_as_exact(tangent('fla'), tangent('xla'), draw, f"the rule's tangent, {staged}", seeds=FEWEST_SEEDS)


@pytest.mark.parametrize('shards', [pytest.param(3, marks=pytest.mark.mesh(devices=3)),
                                    pytest.param(4, marks=pytest.mark.mesh(devices=4))])
@pytest.mark.parametrize('dense', [True, False], ids=['dense', 'decay'])
def test_each_shard_enters_with_the_spans_before_it_applied_in_turn(dense, shards):
    """The memory each shard of a sequence enters its span with across a mesh axis (`recurrence_algebra
    .entering`, a Kogge-Stone prefix of the shards' spans) is the spans before it applied in turn to the
    initial memory (`Span.applied`), on par from float64 over the fewest draws: a dense transfer (the delta
    rule's) and a head's decay (Mamba-2's), over a power of two of shards and a count that is not."""
    batch, heads, width, columns = 2, 3, 4, 5

    def drawn(seed):
        keys = jax.random.split(jax.random.key(seed), 3)
        transfer = (jax.random.normal(keys[0], (shards, batch, heads, width, width)) / width if dense
                    else -jax.random.uniform(keys[0], (shards, batch, heads)))
        return (transfer, jax.random.normal(keys[1], (shards, batch, heads, width, columns)),
                jax.random.normal(keys[2], (batch, heads, width, columns)))

    mesh = Mesh(np.asarray(jax.devices()[:shards]), ('shards',))

    def entered(transfer, write, initial):
        def shard(transfer, write, initial):
            return algebra.entering(algebra.Span(transfer[0], write[0]), 'shards', initial)[None]

        return jax.shard_map(shard, mesh=mesh, in_specs=(P('shards'), P('shards'), P()),
                             out_specs=P('shards'))(transfer, write, initial)

    def in_turn(transfer, write, initial):
        memories = [initial]
        for index in range(shards - 1):
            memories.append(algebra.Span(transfer[index], write[index]).applied(memories[-1]))
        return jnp.stack(memories)

    assert_as_exact(jax.jit(entered), jax.jit(in_turn), drawn, f'{shards} shards entering, dense {dense}',
                    seeds=FEWEST_SEEDS)


PORTABLE_SEEDS = 8
"""The portable rule's differentiation sites' declared draws (check 1), sized by `seeds_for` (paired, on
armada): the remat rung's gradients, the public seam's tangents, two forwards and a TPU's scan under
forward mode."""


def portable_draw(seed: int, rows: int, packed: bool) -> tuple:
    """A small delta-rule call's operands `(query, key, value, g, beta)` (a decay a head), its document
    starts where `packed`, and a direction the shape of each operand: the portable rule's
    differentiation tests' draw."""
    keys = jax.random.split(jax.random.key(seed), 7)
    length, heads, width = 37, 2, 8
    query, key, value = (jax.random.normal(k, (rows, length, heads, width)) for k in keys[:3])
    g = -jax.nn.softplus(jax.random.normal(keys[3], (rows, length, heads, 1)))
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (rows, length, heads)))
    starts = jax.random.bernoulli(keys[5], 0.1, (rows, length)) if packed else None
    operands = (query, key, value, g, beta)
    return operands, starts, tuple(jax.random.normal(keys[6], x.shape) for x in operands)


@pytest.mark.parametrize("packed", PACKED, ids=["one sequence", "packed"])
def test_the_portable_rule_takes_forward_mode_at_its_public_seam(packed):
    """The plain-JAX mainloop (`recurrence_chunk.xla_mainloop`, a CPU's route) under a plain `jax.jvp` and
    `jax.linearize`, with no forward-mode context: its value and tangents on par with the token recurrence's
    own forward mode over the draws (check 1, `recurrent_delta_rule` in float64 the truth), and its gradient
    finite (row 1834: a custom VJP refused a plain `jax.jvp`)."""
    operands, starts, directions = portable_draw(11, 1, packed)

    def rule(starts):
        return lambda *operands: portable_rule(*operands, starts=starts)

    def recurrent(starts):
        return lambda *operands: recurrence_chunk.recurrent_delta_rule(*operands, starts=starts)

    def draw(seed):
        operands, starts, directions = portable_draw(seed, 1, packed)
        return (*operands, *directions, starts)

    def forward(fn):
        def run(*drawn):
            primal, tangent, starts = drawn[:5], drawn[5:10], drawn[10]
            value, along = jax.jvp(fn(starts), primal, tangent)
            return value, along, jax.linearize(fn(starts), *primal)[1](*tangent)
        return run

    assert_as_exact(jax.jit(forward(rule)), jax.jit(forward(recurrent)), draw, "public seam's forward mode",
                    seeds=PORTABLE_SEEDS)
    found = jax.jit(lambda *a: jax.jvp(rule(starts), a, directions))(*operands)
    cotangent = jax.tree.map(jnp.ones_like, found[0])
    def scored(*operands):
        return sum(jnp.vdot(x, c) for x, c in zip(rule(starts)(*operands), cotangent, strict=True))

    gradient = jax.grad(scored, argnums=range(5))(*operands)
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in gradient)


def portable_rule(query, key, value, g, beta, *, starts):
    """`portable_draw`'s operands through the plain-JAX mainloop (a CPU's route), its decay a head."""
    return recurrence_chunk.xla_mainloop(recurrence_chunk.Tiles(CHUNK), query, key, value, g[..., 0], beta,
                                         None, starts, scale=key.shape[-1] ** -0.5, delta=True,
                                         dtype=jnp.float32)


@pytest.mark.parametrize("packed", PACKED, ids=["one sequence", "packed"])
def test_the_portable_rule_takes_nested_differentiation(packed):
    """The plain-JAX mainloop (`portable_rule`) under forward over reverse (`jax.jvp` of `jax.grad`), reverse
    over forward
    (`jax.grad` of `jax.jvp`) and two forwards (`jax.jvp` of `jax.jvp`), which raised before (rows 1861: its
    saved bool resets took a bool tangent, not float0's; 1862: its pullback gave six cotangents of seven
    operands): each runs and is finite (the public seam's forward mode is the chunked math's by check 1,
    `test_the_portable_rule_takes_forward_mode_at_its_public_seam`)."""
    (query, key, value, g, beta), starts, (direction, *_) = portable_draw(13, 1, packed)

    def rule(query):
        return portable_rule(query, key, value, g, beta, starts=starts)

    def loss(fn, query):
        out, final = fn(query)
        return jnp.sum(out ** 2) + jnp.sum(final ** 2)

    def tangent(fn, query):
        return jax.jvp(functools.partial(loss, fn), (query,), (direction,))[1]

    def nested(fn):
        forward_over_reverse = jax.jvp(jax.grad(functools.partial(loss, fn)), (query,), (direction,))[1]
        reverse_over_forward = jax.grad(functools.partial(tangent, fn))(query)
        twice = jax.jvp(functools.partial(tangent, fn), (query,), (direction,))[1]
        return forward_over_reverse, reverse_over_forward, twice

    found = jax.jit(lambda: nested(rule))()
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in found)


def test_a_rated_scan_traced_for_a_jvp_on_a_tpu_gives_its_tangent():
    """A rated scan the TPU's Pallas scan tiles takes in reverse mode alone, its custom VJP giving no tangent
    (row 1869, audit K2): Mamba-2's SSD scan (`mamba2.SSD` without its skip, whose own custom VJP refuses a
    jvp, row 1933) traced for `jax.jvp` (`forward_mode_attention`) on an assumed v6e gives its tangent, on par
    with the scan's on this CPU over the draws (check 1), truth its float64 run."""
    import dataclasses

    from dew.nn.attention import forward_mode_attention
    from dew.nn.mixers.mamba2 import SSD

    hooks = dataclasses.replace(SSD, skip=False)

    def draw(seed):
        keys = jax.random.split(jax.random.key(seed), 6)
        x, direction = (jax.random.normal(key, (1, 64, 4, 16)) for key in keys[:2])
        B, C = (jax.random.normal(key, (1, 64, 2, 16)) for key in keys[2:4])
        dt = jax.nn.softplus(jax.random.normal(keys[4], (1, 64, 4)))
        A = -jnp.exp(jax.random.normal(keys[5], (4,)))
        return x, direction, B, C, dt, A

    def tangent(target=None):
        def run(x, direction, B, C, dt, A):
            def scan(x):
                return chunk(hooks, Chunk(C, B, x, beta=dt, rate=A), chunk_size=CHUNK, dtype=x.dtype)[0]
            with generation.assumed(target), forward_mode_attention():
                return jax.jvp(scan, (x,), (direction,))[1]
        return run

    assert_as_exact(jax.jit(tangent("v6e")), jax.jit(tangent()), draw, "a TPU's scan under jvp",
                    seeds=PORTABLE_SEEDS, truth_fn=tangent())


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("replicated", [(), (2, 5)], ids=["every operand split", "values and memory whole"])
@pytest.mark.parametrize("entering", [False, True], ids=["zero memory", "a memory entering"])
def test_the_derived_rule_runs_in_a_shard_map_that_checks_varying_axes(entering, replicated):
    """The portable math's forms derived from the token update (`recurrence_chunk.xla_mainloop`, a CPU's
    route, named `generation.chosen(gated_delta_rule='xla')`) inside a `shard_map` that checks varying
    axes over operands that vary by row, or with the values and the entering memory whole beside split keys
    and gates, from a zero memory and from one that enters: its carries vary as its operands do, and its
    outputs and gradients are finite."""
    keys = jax.random.split(jax.random.key(2911), 7)
    q = l2norm(jax.random.normal(keys[0], (2, 32, 1, 16)))
    k = l2norm(jax.random.normal(keys[1], (2, 32, 1, 16)))
    v = jax.random.normal(keys[2], (2, 32, 2, 16))
    g = -jnp.exp(jax.random.normal(keys[3], (2, 32, 2)) - 2.0)
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (2, 32, 2)))
    state = jax.random.normal(keys[5], (2, 2, 16, 16)) if entering else None
    do = jax.random.normal(keys[6], v.shape)
    rows = P("data")

    def rule(q, k, v, g, beta, *given):
        return recurrence_chunk.chunk_gated_delta_rule(q, k, v, g, beta, *given, chunk_size=16)[0]

    operands = (q, k, v, g, beta, *(() if state is None else (state,)))
    # A whole operand holds one device's rows, the same on every device.
    operands = tuple(x[:1] if index in replicated else x for index, x in enumerate(operands))
    specs = tuple(P() if index in replicated else rows for index in range(len(operands)))
    mapped = jax.shard_map(rule, mesh=Mesh(np.asarray(jax.devices()[:2]), ("data",)), in_specs=specs,
                           out_specs=rows, check_vma=True)
    with generation.chosen(gated_delta_rule='xla'):
        got = jax.jit(jax.value_and_grad(lambda *given: jnp.sum(mapped(*given) * do),
                                         argnums=range(len(operands))))(*operands)
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(got))


def test_a_call_inside_a_map_holding_an_axis_manual_selects_its_kernel():
    """The chunked op traced inside a map that holds one mesh axis manual and leaves another automatic (a
    DiLoCo replica's step, `dew.training.diloco`): its selectors bind the template's call on each shard of the
    axes its own map holds (`sharding.shard_shapes`), and the trace completes. A sharding over the whole mesh
    there named the manual axis, a manual subgroup JAX refuses, and the selection raised before any kernel ran
    (train 942's review of the shared bind)."""
    from jax.sharding import AbstractMesh, AxisType

    mesh = AbstractMesh((2, 2), ("outer", "data"), axis_types=(AxisType.Auto, AxisType.Auto))
    hooks = recurrence.Hooks(decay="scalar", delta=True)
    given = jax.ShapeDtypeStruct((4, 128, 2, 64), jnp.float32)
    heads = jax.ShapeDtypeStruct((4, 128, 2), jnp.float32)

    def inner(query, key, value, g, beta):
        return chunk(hooks, Chunk(query, key, value, log_decay=g, beta=beta))[0]

    mapped = jax.shard_map(inner, mesh=mesh, in_specs=P("outer"), out_specs=P("outer"), axis_names={"outer"})
    with jax.sharding.use_abstract_mesh(mesh):
        out = jax.eval_shape(mapped, given, given, given, heads, heads)
    assert out.shape == given.shape


RWKV6_SITES = tuple(f"rwkv6/{dtype}/{layout}" for dtype in ("bf16", "fp32")
                    for layout in ("one sequence", "packed"))
"""RWKV6's recorded sites (`fla_sites.RWKV6`): its recurrence in bf16 and fp32, one sequence and packed."""


@pytest.mark.parametrize("site", RWKV6_SITES)
@pytest.mark.check1_shards(seeds=64, per_task=8)
def test_rwkv6_through_the_chunked_op_is_as_exact_as_fla(site):
    """RWKV6's recurrence (a decay a key dimension, its readout of the memory entering its token, its own
    write weighed a key: `recurrence.Hooks.reads`, `Hooks.own_weight`) through the chunked op, its output,
    final memory and every operand's gradient, the own weight's included, on par from float64 with
    recorded fla-core's `chunk_rwkv6` at the site's declared seeds (check 1; the truth the recurrence a
    token at a time in float64, which the recorder held to fla's `naive_recurrent_rwkv6`). The portable
    math takes it on every route (`recurrence_chunk.READOUT_HANDOFF`)."""
    from fla_sites import OWN_LEAVES, RWKV6, the_template_is_as_exact_as_fla

    the_template_is_as_exact_as_fla(lambda operands: chunk(RWKV6, operands), site, OWN_LEAVES,
                                    label=f"RWKV6 through the chunked op against fla, {site}")


def _unit_turn_grad(q, k, v, do, state, *, turn=None):
    """Run `turned.turn_gradient` with unit decay/gate and zero final cotangent for P0 witnesses."""
    from dew.nn.kernels.templates.recurrence import turned

    length = q.shape[1]
    g = jnp.zeros((1, length, 1), jnp.float32)
    b = jnp.ones((1, length, 1), jnp.float32)
    t = jnp.zeros((1, length, 1, 1), jnp.float32) if turn is None else turn
    return turned.turn_gradient(q, k, v, do, g, b, state, jnp.zeros_like(state), t, None,
                                scale=1.0, group=1, pairs=1)


def test_a_turned_gradients_overflowed_term_reaches_no_later_tokens_sum():
    """Two tokens, no turn, decay or write, a memory entering of `(0, 2^126)` read by the first token's query
    `(1, 0)` under its output's cotangent 4 alone: the first token's turn gradient overflows (`2^128`), and
    the second's, which no term of the first reaches, is exactly zero, as the scan of its suffix keeps it
    (`recurrence_algebra.suffix_sums`), where a triangle's dot multiplied the first token's
    infinity by zero, NaN (rows 7969 and 7975's head review)."""
    q = jnp.zeros((1, 2, 1, 2), jnp.float32).at[0, 0, 0, 0].set(1.0)
    k = jnp.zeros((1, 2, 1, 2), jnp.float32)
    v = jnp.zeros((1, 2, 1, 1), jnp.float32)
    do = jnp.zeros((1, 2, 1, 1), jnp.float32).at[0, 0, 0, 0].set(4.0)
    state = jnp.zeros((1, 1, 2, 1), jnp.float32).at[0, 0, 1, 0].set(np.float32(2.0 ** 126))
    gradient = np.asarray(_unit_turn_grad(q, k, v, do, state))
    assert np.isinf(gradient[0, 0, 0, 0]) and gradient[0, 1, 0, 0] == 0.0


def test_turned_gradient_p0_regressions_f1_f2_f3():
    """Regressions for turned-frame base-review P0s (d80c85a37 F1/F2/F3, 1cf389a60 F1, e488dc198 F1):
    carry zero-reset tangent (-1), deferred scale restoration across both own-chunk and cross-chunk cross
    products, and float64 turn angle precision under x64."""
    from dew.nn.kernels.templates.recurrence import turned

    # d80c85a37 F1: carry reset at zero peak preserves tangent -1.0 (not -0.5)
    big = np.float32(2.0 ** 126)
    z129 = jnp.zeros((1, 129, 1, 2), jnp.float32)
    k129 = z129.at[0, 0, 0, 0].set(-big)
    v129 = jnp.zeros((1, 129, 1, 1), jnp.float32).at[0, 0, 0, 0].set(1.0)
    do129 = jnp.zeros((1, 129, 1, 1), jnp.float32)
    g129 = jnp.zeros((1, 129, 1), jnp.float32)
    b129 = jnp.ones((1, 129, 1), jnp.float32)
    t129 = jnp.zeros((1, 129, 1, 1), jnp.float32)
    s129 = jnp.array([[[[big], [np.float32(0.0)]]]], jnp.float32)
    ds129 = jnp.array([[[[np.float32(1.0)], [np.float32(0.0)]]]], jnp.float32)
    df129 = jnp.array([[[[np.float32(0.0)], [np.float32(1.0)]]]], jnp.float32)
    _, jvp129 = jax.jvp(
        lambda st: turned.turn_gradient(z129, k129, v129, do129, g129, b129, st, df129, t129, None,
                                        scale=1.0, group=1, pairs=1),
        (s129,), (ds129,))
    np.testing.assert_array_equal(np.asarray(jvp129[0, 128, 0, 0]), np.float32(-1.0))

    # d80c85a37 F2 & e488dc198 F1: own-chunk _halves defers inv_q restore until after _crossed
    z2 = jnp.zeros((1, 1, 2, 1), jnp.float32)
    v2 = jnp.array([[[[1.0]], [[0.0]]]], jnp.float32)
    for k_vals, do_val, want in (
        ([0.0, np.float32(2.0 ** -126)], np.float32(2.0 ** 127), np.float32(2.0 ** 127)),
        ([np.float32(2.0), np.float32(2.0 ** -126)], np.float32(2.0 ** 126), np.float32(2.0 ** 126)),
    ):
        q = jnp.array([[[[0.0, 0.0]], [[np.float32(2.0 ** 126), 0.0]]]], jnp.float32)
        k = jnp.array([[[k_vals], [[0.0, 0.0]]]], jnp.float32)
        do = jnp.array([[[[0.0]], [[do_val]]]], jnp.float32)
        grad = _unit_turn_grad(q, k, v2, do, z2)
        assert np.isfinite(np.asarray(grad)).all()
        np.testing.assert_array_equal(np.asarray(grad[0, 1, 0, 0]), want)

    # 1cf389a60 F1: cross-chunk _scaled_einsum defers post-scale until after _crossed
    q4 = jnp.array([[[[np.float32(2.0 ** 126), np.float32(2.0 ** -100)]]]], jnp.float32)
    st4 = jnp.array([[[[np.float32(2.0)], [np.float32(2.0 ** -126)]]]], jnp.float32)
    grad4 = _unit_turn_grad(q4, jnp.zeros((1, 1, 1, 2), jnp.float32), jnp.zeros((1, 1, 1, 1), jnp.float32),
                            jnp.array([[[[np.float32(2.0 ** 126)]]]], jnp.float32), st4)
    assert np.isfinite(np.asarray(grad4)).all()
    np.testing.assert_array_equal(np.asarray(grad4[0, 0, 0, 0]), np.float32(2.0 ** 126))

    # d80c85a37 F3: float64 turn pi/2 - 1e-9 preserves positive gradient sign under x64
    with jax.enable_x64(new_val=True):
        q3 = jnp.array([[[[0.0, 0.0]], [[1.0, 0.0]]]], jnp.float32)
        k3 = jnp.array([[[[0.0, 1.0]], [[0.0, 0.0]]]], jnp.float32)
        do3 = jnp.array([[[[0.0]], [[1.0]]]], jnp.float32)
        t3 = jnp.array([[[[0.0]], [[np.pi / 2.0 - 1e-9]]]], jnp.float64)
        grad3 = _unit_turn_grad(q3, k3, v2, do3, z2, turn=t3)
        assert float(grad3[0, 1, 0, 0]) > 0.0
