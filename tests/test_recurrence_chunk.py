"""The recurrence template's chunked op (`dew.nn.kernels.recurrence_chunk.chunk`) on its portable math:
every hook combination it takes, a sequence over several chunks and a ragged last one, packed documents
too, held to the same recurrence run a token at a time (the decode step's reference,
`recurrence.step_reference`) under the accuracy gate (tests/reference_error.py), both from float64."""

import contextlib
import dataclasses
import functools
import itertools
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec as P
from port_support import row_names_port
from reference_error import (
    FEWEST_SEEDS,
    assert_as_exact,
    assert_fp32_reduction_bound,
    assert_rounds_where_the_reference_does,
    chain_roundings,
    widened,
    with_gradients,
)

from dew.nn.kernels import generation, recurrence, recurrence_algebra as algebra, recurrence_chunk
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
]
"""The prologue's and epilogue's hooks: the gated delta rule's gates and l2 norm, Mamba2's step gates
clipped with the skip, the gated RMS norm after the l2 norm and after a plain scale, Mamba2's SSD as
`chunk_ssd` hands it over (a rated decay, the step its write gate), with its gated norm, a rated decay
under the delta rule and the l2 norm, the norm after the skip, and a decay without any write gate."""

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


def routed(check) -> Callable:
    """The decorator making `check` the test it decorates, run on each route (`ROUTES`, `on_route`): its
    name and documentation the test's, its one parameter the route."""
    def run(route):
        with on_route(route):
            check()

    def declared(test):
        run.__name__, run.__qualname__, run.__doc__ = test.__name__, test.__qualname__, test.__doc__
        return pytest.mark.parametrize('route', ROUTES)(run)

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
    8, 8, 8, 8, 8, 8, 8, 8, 16, 16, 64, 64, 16, 16, 16, 16, 8, 8, 64, 64, 8, 8, 8, 8,
    64, 64, 128, 128, 16, 8, 16, 16, 64, 64, 32, 32, 32, 32, 8, 8, 16, 16), strict=True))
"""Each site's declared seeds (hooks in `HOOKS` then `FORMED` order, one sequence then packed), sized
independently by `seeds_for` on armada (2026-10-21), on the draw with a memory entering and a
document starting on a chunk's first token."""


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
         value_heads: int = VALUE_HEADS) -> tuple[Chunk]:
    """A sequence's operands for `hooks` from `seed` (`hosted`), its keys' and values' dimensions each
    permuted by an order the seed draws (an exact symmetry of every head's recurrence): fresh queries, keys
    and values (`NORMED_WIDTH` wide under the norm), a log decay a head or a key dimension in (-1, 0),
    a gate in (0, 1), a skip, a memory entering, `length` tokens over `value_heads` value heads, and with
    `packed` documents starting at tokens 0, 23 and 40 of the first row and 9 and 48 of the second, 48 a
    chunk's first token."""
    rng = np.random.default_rng(seed)
    normal = np.random.default_rng([seed, 1]).standard_normal
    width = VALUE_WIDTH if hooks.output is None else NORMED_WIDTH
    key_order, value_order = rng.permutation(KEY_WIDTH), rng.permutation(width)
    query = normal((ROWS, length, KEY_HEADS, KEY_WIDTH))[..., key_order]
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
    starts = jnp.asarray(marked(((0, 23, 40), (9, 48)), length)) if packed else None
    reads_beta = hooks.gated_write if hooks.gates is None else hooks.gates.write == 'sigmoid'
    plain = hooks.decay != 'none' and hooks.gates is None and not hooks.rated
    return (Chunk(hosted(query), hosted(key), hosted(value), log_decay=hosted(log_decay) if plain else None,
                  beta=hosted(beta) if reads_beta else None, skip=hosted(skip) if hooks.skip else None,
                  state=hosted(state), starts=starts, **gates, **epilogue),)


def token_by_token(hooks: recurrence.Hooks, operands: Chunk):
    """The sequence a token at a time through the decode step's reference, a packed document's first
    token starting from a zero memory: `(output, final memory)`, the output in the operands' dtype. Host
    operands are taken as JAX arrays: a float64 truth's first call runs on the gate's widened host arrays,
    which the scan's traced token can't index."""
    operands = jax.tree.map(jnp.asarray, operands)
    group = operands.value.shape[2] // operands.query.shape[2]
    query, key = (jnp.repeat(x, group, axis=2) for x in (operands.query, operands.key))
    work = jnp.promote_types(operands.value.dtype, jnp.float32)
    state = operands.state.astype(work)
    rows, length = operands.value.shape[:2]
    starts = jnp.zeros((rows, length), bool) if operands.starts is None else jnp.asarray(operands.starts)

    def one_token(state, t):
        state = jnp.where(starts[:, t, None, None, None], 0.0, state)
        def at(x):
            return None if x is None else x[:, t]

        token = recurrence.Step(
            state, query=query[:, t], key=key[:, t], value=operands.value[:, t],
            log_decay=at(operands.log_decay), gate=at(operands.gate), log_a=operands.log_a,
            bias=operands.bias, beta=at(operands.beta), skip=operands.skip, z=at(operands.z),
            weight=operands.weight, rate=operands.rate)
        out, _, state = recurrence.step_reference(hooks, token)
        return state, out

    state, out = jax.lax.scan(one_token, state, jnp.arange(length))
    return jnp.moveaxis(out, 0, 1), state


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


DOCUMENTS = ((0, 23, 40), (0, 9, 48))
"""Each row's packed documents' first tokens under `draw`'s `packed`."""

GIVEN = 1
"""The memories a row's table holds before its documents store theirs (`Chunk.loads`: a row's start, a
cache)."""

FORKED = (((1, 2, 2), (1, 0, 0)), ((1, 2, 3), (1, 2, 0)))
"""Each row's documents' table slots and stored slots (`Chunk.loads`, `Chunk.stores`): the first row's start
continued by its first document, whose final both later documents enter (a slot's gradient summing two
documents'); the second row a chain from its start, each document entering its predecessor's final. Every
memory is entered, so no gradient's row is float64 zero, whose terms the gate would trace each draw."""

HELD = 2
"""The stored slots a row's table holds (`Chunk.held`)."""


def entered_draw(hooks, seed: int, entries: bool = False):
    """`draw`'s packed operands over a table of `GIVEN` memories a row, with `entries` an entry write a
    document (`Chunk.entry_key`: two terms a document, a weight a value head), and cotangents of the output
    and the row's last memory."""
    (operands,) = draw(hooks, packed=True, seed=seed)
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


def entering_documents(operands):
    """`operands` with `draw`'s packed documents a forked call's, each entering its `FORKED` table slot."""
    starts = jnp.zeros(operands.value.shape[:2], bool)
    for row, begins in enumerate(DOCUMENTS):
        starts = starts.at[row, jnp.asarray(begins)].set(True)
    loads, stores = (jnp.asarray([row[part] for row in FORKED], jnp.int32) for part in (0, 1))
    return dataclasses.replace(operands, starts=starts, loads=loads, stores=stores, held=HELD)


def document_by_document(hooks, operands, run=None):
    """Each document of `operands` a sequence of its own through `run(hooks, piece)` (`token_by_token`
    where None), in the row's order from its `FORKED` table slot's memory (a given one, or the final an
    earlier document stored, with its entry write added where `operands` give one), its outputs back in
    place, and each row's last memory."""
    run = token_by_token if run is None else run
    tokens = ('query', 'key', 'value', 'log_decay', 'gate', 'beta', 'z')
    outs, lasts = [], []
    for row, (begins, (loads, stores)) in enumerate(zip(DOCUMENTS, FORKED, strict=True)):
        ends = (*begins[1:], operands.value.shape[1])
        table = list(operands.state[row]) + [None] * HELD
        pieces, final = [], None
        for document, (begin, end, load, store) in enumerate(zip(begins, ends, loads, stores, strict=True)):
            state = table[load - 1] if load else jnp.zeros_like(operands.state[row, 0])
            if operands.entry_key is not None:
                state = state + entry_write(operands, row, document)
            fields = {name: getattr(operands, name)[row:row + 1, begin:end] for name in tokens
                      if getattr(operands, name) is not None}
            piece = dataclasses.replace(operands, **fields, state=state[None], starts=None, loads=None,
                                        stores=None, held=0, entry_key=None, entry_value=None,
                                        entry_weight=None)
            out, final = run(hooks, piece)
            pieces.append(out[0])
            if store:
                table[GIVEN + store - 1] = final[0]
        outs.append(jnp.concatenate(pieces))
        lasts.append(final[0])
    return jnp.stack(outs), jnp.stack(lasts)


@pytest.mark.check1_shards(seeds=80, per_task=8)
@pytest.mark.parametrize('hooks', [HOOKS[4], HOOKS[9], HOOKS[6], FORMED[0]], ids=str)
def test_documents_entering_memories_of_their_own_are_as_exact_as_each_alone(hooks):
    """80 draws, 8 a task: at 40 a skip's and a gated decay's bias's gradients, two entries a head, were
    inconclusive (intervals 1.58 and 1.56 wide, pooled 1.09 and 0.95; jsf-1506b); the gate holds a draw's
    float64 terms of the walk and its table, 6.4 GB at 8 draws on armada (11-10), so 16 a task passed a
    task's 12 GB.
    The portable math's walk over a forked call, each document laid from a chunk's edge and its table slot's
    memory taken up there, the stored finals kept in the walk's table (`xla_mainloop`, `_walk`), against
    each document a sequence of its own the recurrence a token at a time (`assert_forked_on_par`): a slot
    entered twice sums both documents' gradients."""
    assert_forked_on_par('xla', hooks, functools.partial(entered_draw, hooks),
                         f'xla {hooks} a forked call\'s documents', seeds=80)


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


def assert_forked_on_par(kernel: str, hooks, draw, site: str, *, seeds: int, alone: bool = False) -> None:
    """A forked call on `kernel` (`entering_documents` of `draw`'s operands; `Chunk.loads`: a request's
    questions entering its shared part's final memory, a chain's each its predecessor's): its output, the
    row's last memory and every operand's gradient, the given memories' included, against each document a
    sequence of its own from its table slot's memory in the row's order (`document_by_document`), through
    `kernel`'s call where `alone`, else the recurrence a token at a time, which in float64 is their truth
    (check 1)."""
    def ours(operands):
        out, final = chunk_on(kernel, hooks, entering_documents(operands), chunk_size=CHUNK)
        return out.astype(jnp.float32), final.astype(jnp.float32)

    def own(hooks, piece):
        return chunk_on(kernel, hooks, piece, chunk_size=CHUNK)

    def theirs(operands):
        out, final = document_by_document(hooks, operands, own if alone else None)
        return out.astype(jnp.float32), final.astype(jnp.float32)

    wide = jax.jit(with_gradients(lambda operands: document_by_document(hooks, operands)))

    def widened(leaf):
        floating = jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.floating)
        return jnp.asarray(leaf, jnp.float64) if floating else leaf

    def truth(operands, seeded):
        with jax.enable_x64(new_val=True):
            return wide(jax.tree.map(widened, operands), jax.tree.map(widened, seeded))

    assert_as_exact(jax.jit(with_gradients(ours)), jax.jit(with_gradients(theirs)), draw, site, seeds=seeds,
                    truth_fn=truth)


@pytest.mark.check1_shards(seeds=80, per_task=8)
@pytest.mark.parametrize('hooks', [FORMED[4], FORMED[5]], ids=str)
def test_the_ssd_ports_documents_entering_memories_of_their_own_are_as_exact_as_each_alone(hooks):
    """The port of mamba_ssm's kernels over a forked call, each document laid from a chunk's edge and the
    state passing taking up its table slot's memory there with its entry write added (`Chunk.entry_key`),
    keeping the finals its documents store (`_ported_documents`), as the port's call over each document
    alone from its slot's memory and entry write (`assert_forked_on_par`). The port's own numbers
    (mamba_ssm's) are tests/test_mamba_ssd.py's."""
    assert_forked_on_par('pallas', hooks, functools.partial(entered_draw, hooks, entries=True),
                         f'pallas {hooks} a forked call\'s documents, against each alone', seeds=80,
                         alone=True)


def narrowed_draw(hooks, seed: int):
    """`entered_draw`'s operands with their queries, keys, values and write gate in bf16, a 16-bit call as the
    recurrence template's fast backward trains it (`mainloop_backward.fast_backward_form`)."""
    operands, seeded = entered_draw(hooks, seed)
    return dataclasses.replace(operands, **{name: getattr(operands, name).astype(jnp.bfloat16)
                                            for name in ('query', 'key', 'value', 'beta')
                                            if getattr(operands, name) is not None}), seeded


@pytest.mark.check1_shards(seeds=8, per_task=1)
@pytest.mark.parametrize('hooks', [HOOKS[4], HOOKS[8], FORMED[4]], ids=str)
def test_the_templates_forked_documents_are_as_exact_as_each_alone_interpreted(hooks):
    """The recurrence template interpreted over a forked call in 16 bits, the gated delta rule, Kimi delta
    attention and Mamba2's SSD: its walks take each document's memory up from the call's table and keep
    the finals its documents store (`table.Taking`, `table.Returning`; ledger 5376), as its call over
    each document alone (`assert_forked_on_par`), the same launches' arithmetic. A draw a task: four
    took 287 to 381 s a task (jsf-table-phase at 693aa1c1c)."""
    assert_forked_on_par('mosaic_gpu', hooks, functools.partial(narrowed_draw, hooks),
                         f'mosaic_gpu {hooks} a forked call\'s documents, against each alone',
                         seeds=8, alone=True)


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


def test_a_call_runs_its_products_kernels_apart():
    """`_per_product`: a call's value is its forward kernel's where no derivative is taken; its gradient the
    training kernel's forward and pullback; and its tangent the portable math's, `run('xla', ...)`."""
    def run(kernel, x):
        return {'forward': 2.0, 'training': 3.0, 'xla': 5.0}[kernel] * x

    def kernel_of(product):
        return product

    x = jnp.arange(3.0)
    np.testing.assert_array_equal(recurrence_chunk._per_product(kernel_of, run, x), 2.0 * x)
    value, grad = jax.value_and_grad(lambda x: recurrence_chunk._per_product(kernel_of, run, x).sum())(x)
    assert value == 3.0 * x.sum()
    np.testing.assert_array_equal(grad, jnp.full(3, 3.0))
    np.testing.assert_array_equal(jax.jvp(functools.partial(recurrence_chunk._per_product, kernel_of, run),
                                          (x,), (jnp.ones(3),))[1], jnp.full(3, 5.0))


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
