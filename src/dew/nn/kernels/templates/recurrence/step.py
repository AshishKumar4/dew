"""The recurrence template's decode step on Mosaic GPU: one token over a row's state, a program a row and a
block of its value heads' columns, each thread one column of the state (kernel-templates.md §5).

A decode token reads its whole state once and writes it once; nothing in it is a product worth an MMA. A
thread holds a value column's `Dk` entries in registers, so every sum over the key dimension is its own
(`recurrence.update`'s `k^T S` and `q^T S'`, no exchange), and the key dimension's vectors (the queries and
keys normed and scaled, the decay) are staged once in shared memory, each read there as a scalar by every
thread. A program's warpgroup takes `scan.THREADS` columns: a head's block of them, or where a head is
narrower (Mamba2's 64-wide), as many heads as fill it (`_Block`), each head's scalars laid over its own
columns. The arithmetic is `recurrence.step_reference`'s, fp32 throughout, the held state rounded as it does
(`recurrence._held`): one hook path, the portable math's, which a call's gradient takes too
(`recurrence.step`'s `linear_tangent`).
"""

import dataclasses

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels import recurrence
from dew.nn.kernels.templates import mosaic, scan
from dew.nn.precision import L2NORM_EPS, rms_weighted


@dataclasses.dataclass(frozen=True)
class _Block:
    """A program's block of a row's state: `heads` value heads side by side, `columns` columns of each, a
    thread an entry (`heads * columns` is a warpgroup's threads)."""

    heads: int
    columns: int

    @property
    def layout(self):
        return plgpu.Layout.WG_STRIDED((self.heads, self.columns), vec_size=1)


def _block(value_width: int) -> _Block:
    """A program's block at `value_width` columns a head: a warpgroup's columns of one head, or of as many
    narrower heads as fill it."""
    columns = min(value_width, scan.THREADS)
    return _Block(scan.THREADS // columns, columns)


def fits(shape: tuple[int, ...], whole: bool) -> bool:
    """Whether a state of `shape` `[rows, Hv, Dk, Dv]` steps here: its key dimension the key vectors' (a
    whole number of a warpgroup's), its value heads a whole number of programs' blocks (`_block`: a power of
    two narrower than a warpgroup, as many heads as fill one dividing the heads, or a whole number of
    warpgroups); a head a warpgroup wide where a hook reads it `whole` (a gated norm over a head)."""
    _, value_heads, key_width, value_width = shape
    block = _block(value_width)
    tiles = (value_width % block.columns == 0 and value_heads % block.heads == 0
             and scan.THREADS % block.columns == 0)
    return key_width % scan.THREADS == 0 and tiles and (not whole or value_width == scan.THREADS)


def serves(hooks: recurrence.Hooks, operands: recurrence.Step) -> bool:
    """Whether a token of `hooks` and `operands` steps here (`fits`); a one-tap conv's is taken before any
    launch (`recurrence.step`)."""
    if hooks.conv is not None and (operands.held is None or operands.held.shape[1] == 0):
        return False
    return fits(operands.state.shape, hooks.output is not None)


def step(hooks: recurrence.Hooks, operands: recurrence.Step, *, dtype, family: mosaic.Family,
         interpret: bool = False):
    """`recurrence.step`'s launch on Mosaic GPU: `(output [rows, Hv, Dv], the moved history or None,
    state')`, the state as it is held, an idle row's (`Step.active`) as it came and its output zeros."""
    given = recurrence.read(hooks, operands)
    state = given['state']
    # A row's entry a head, `[rows, Hv]`, as the token's step takes its write gate and scalar decay.
    for name in ('beta', 'gate') + (('log_decay',) if hooks.decay == 'scalar' else ()):
        if name in given:
            given[name] = given[name].reshape(state.shape[0], -1)
    rows, value_heads, key_width, value_width = state.shape
    key_heads = (recurrence._pieces(hooks, value_heads, key_width, value_width, given['token'].shape[-1])
                 if hooks.conv is not None else given['query'].shape[1])
    if hooks.conv is not None:
        # Each tap's weights a row `[K, C]`, a program's channels of it read contiguously.
        given['taps'] = given['taps'].T
    if 'weight' in given:
        # The norm's weight a row, as a program's block reads its columns.
        given['weight'] = given['weight'][None]
    operand = given['token' if hooks.conv is not None else 'value'].dtype
    out_dtype = jnp.dtype(operand if dtype is None else dtype)
    active = (jnp.ones((rows,), jnp.int32) if operands.active is None
              else jnp.asarray(operands.active).astype(jnp.int32))
    names = tuple(given)
    block = _block(value_width)

    def entry(active_ref, *refs):
        inputs = dict(zip(names, refs[:len(names)], strict=True))
        state_out, *moved, out_ref, vectors, exchanged = refs[len(names):]
        _program(hooks, inputs, active_ref, state_out, moved[0] if moved else None, out_ref,
                 (vectors, exchanged), block=block, group=value_heads // key_heads,
                 key_heads=key_heads, interpret=interpret)

    history = ((jax.ShapeDtypeStruct(given['held'].shape, given['held'].dtype),) if hooks.conv is not None
               else ())
    out_type = (jax.ShapeDtypeStruct(state.shape, state.dtype), *history,
                jax.ShapeDtypeStruct((rows, value_heads, value_width), out_dtype))
    launched = mosaic.kernel(entry, out_type=out_type,
                             grid=(rows, value_heads // block.heads, value_width // block.columns),
                             grid_names=("rows", "heads", "columns"), warpgroups=1, interpret=interpret,
                             family=family.name, name="recurrence_step",
                             scratch=(plgpu.SMEM((block.heads, 3, key_width), jnp.float32), scan.scratch()))
    state_out, *moved, out = launched(active, *given.values())
    return out, moved[0] if moved else None, state_out


@dataclasses.dataclass(frozen=True)
class _Token:
    """A program's token: its row, its block's value heads (`heads`, `block.heads` of them from `first`),
    their key heads and its value columns."""

    row: jax.Array
    first: jax.Array
    heads: tuple
    keyed: tuple
    columns: pl.Slice
    block: _Block

    @property
    def head(self):
        """Its first value head (the one a conv or a gated norm reads, a block of one head)."""
        return self.heads[0]

    def tile(self, ref, *index):
        """`ref`'s block at the token's row, heads and `index` then its columns, fp32 `[heads, columns]`."""
        at = ref.at[(self.row, pl.ds(self.first, self.block.heads), *index, self.columns)]
        return plgpu.load(at, layout=self.block.layout, optimized=False).astype(jnp.float32)

    def stored(self, ref, value, *index):
        """`value` `[heads, columns]` stored at `ref`'s block, as `tile` reads it."""
        plgpu.store(ref.at[(self.row, pl.ds(self.first, self.block.heads), *index, self.columns)],
                    value.astype(ref.dtype), optimized=False)

    def per_head(self, scalars):
        """`scalars`, one a head of the block, laid over each head's columns `[heads, columns]` (a block
        of one head's scalar as it is)."""
        if self.block.heads == 1:
            return scalars[0]
        shape = (self.block.heads, self.block.columns)
        rows = plgpu.layout_cast(lax.broadcasted_iota(jnp.int32, shape, 0), self.block.layout)
        laid = scalars[-1]
        for index in range(self.block.heads - 2, -1, -1):
            laid = jnp.where(rows == index, scalars[index], laid)
        return laid


def _program(hooks, inputs, active_ref, state_out, moved, out_ref, scratch, *, block: _Block, group: int,
             key_heads: int, interpret: bool):
    """A program's token (`step`): its queries, keys and values (the conv's, `_Conv`, where the prologue
    takes one), each head's key vectors staged, its block of the state stepped, its output and state stored,
    and the conv's history moved; an idle row's state and history copied and its output zeros."""
    vectors, exchanged = scratch
    first = lax.axis_index("heads") * block.heads
    heads = tuple(first + index for index in range(block.heads))
    token = _Token(lax.axis_index("rows"), first, heads, tuple(head // group for head in heads),
                   pl.ds(lax.axis_index("columns") * block.columns, block.columns), block)
    state_ref = inputs['state']
    key_width = state_ref.shape[2]
    conv = (_Conv(inputs, moved, token, hooks.conv, key_heads, key_width, state_ref.shape[3], group)
            if hooks.conv is not None else None)

    @pl.when(active_ref[token.row] != 0)
    def _():
        if conv is None:
            keyed = [(_keyed(inputs['query'], token, p), _keyed(inputs['key'], token, p))
                     for p in range(block.heads)]
            value = token.tile(inputs['value'])
        else:
            windows = conv.windows()
            convolved = conv.convolved(windows)
            conv.write([window[1:] for window in windows])
            keyed = [convolved[1 + 2 * p:3 + 2 * p] for p in range(block.heads)]
            # The block's heads' values, a thread a channel as the block lays them (`_Block.layout`).
            value = convolved[0].reshape(block.heads, block.columns)
        writes = [_staged(hooks, inputs, token, p, *keyed[p], vectors, exchanged, interpret=interpret)
                  for p in range(block.heads)]
        mosaic.synced(interpret)
        out = _stepped(hooks, state_ref, state_out, value, vectors, token.per_head(writes), token)
        out = _epilogue(hooks, inputs, token, out, value, exchanged, interpret=interpret)
        token.stored(out_ref, out)

    @pl.when(active_ref[token.row] == 0)
    def _():
        def copied(k, carry):
            token.stored(state_out, token.tile(state_ref, k), k)
            return carry

        lax.fori_loop(0, key_width, copied, None)
        token.stored(out_ref, jnp.zeros((block.heads, block.columns), jnp.float32))
        if conv is not None:
            conv.write(conv.held())


def _keyed(ref, token: _Token, index: int):
    """The token's queries or keys for its block's `index`-th head, its key head's `[Dk]`, fp32."""
    at = ref.at[token.row, token.keyed[index]]
    return plgpu.load(at, layout=scan.blocked(ref.shape[-1]), optimized=False).astype(jnp.float32)


def _key_entry(vectors, token: _Token, which: int, k: int):
    """Key dimension `k`'s entry of each head's staged vector `which` (0 queries, 1 keys, 2 decay), laid over
    its head's columns."""
    return token.per_head([vectors[p, which, k] for p in range(token.block.heads)])


@dataclasses.dataclass(frozen=True)
class _Conv:
    """A program's view of the prologue's conv (`recurrence.Conv`): the row's token `[C]`, history
    `[rows, K - 1, C]`, taps `[K, C]` and bias `[C]`, in the conv's order of queries, keys and values
    (`recurrence.conv_pieces`); its streams the block's heads' values, contiguous, then each block head's key
    head's queries and keys; the moved history it writes, the queries' and keys' by their key head's first
    value head."""

    inputs: dict
    moved: object
    token: _Token
    conv: recurrence.Conv
    key_heads: int
    key_width: int
    value_width: int
    group: int

    @property
    def pieces(self):
        """Each stream's first channel and width: the block's values, then each head's queries and keys."""
        block = self.token.block
        whole = recurrence.conv_pieces(self.conv, self.key_heads, self.value_heads, self.key_width,
                                       self.value_width)
        # The token's columns of its heads: a block of narrower heads takes each whole.
        values = (whole['v'][0] + self.token.first * self.value_width + self.token.columns.start,
                  block.heads * block.columns)
        return (values, *((whole[name][0] + keyed * self.key_width, self.key_width)
                          for keyed in self.token.keyed for name in 'qk'))

    @property
    def value_heads(self) -> int:
        return self.group * self.key_heads

    def _load(self, ref, first, width):
        return plgpu.load(ref.at[pl.ds(first, width)], layout=scan.blocked(width), optimized=False)

    def windows(self):
        """Each stream's conv window, `K` vectors fp32 oldest first: the held history rounded to the token's
        dtype, as the layer's conv reads it, then the token."""
        token, held = self.inputs['token'], self.inputs['held']
        count = held.shape[1] + 1
        return [[*(self._load(held.at[self.token.row, j], first, width).astype(token.dtype)
                   .astype(jnp.float32) for j in range(count - 1)),
                 self._load(token.at[self.token.row], first, width).astype(jnp.float32)]
                for first, width in self.pieces]

    def convolved(self, windows):
        """Each stream's taps over its window, oldest first (`short_conv.preactivation`'s order), then its
        bias, then the activation: the block's values, then each head's queries and keys."""
        taps = self.inputs['taps']
        streams = []
        for window, (first, width) in zip(windows, self.pieces, strict=True):
            weights = [self._load(taps.at[j], first, width).astype(jnp.float32) for j in range(len(window))]
            pre = window[0] * weights[0]
            for inputs, weight in zip(window[1:], weights[1:], strict=True):
                pre = pre + inputs * weight
            if self.conv.bias:
                pre = pre + self._load(self.inputs['conv_bias'], first, width).astype(jnp.float32)
            # SiLU as the reference's, the input times its sigmoid.
            streams.append(pre if self.conv.activation is None else pre * _sigmoid(pre))
        return tuple(streams)

    def held(self):
        """Each stream's history as it was, for an idle row."""
        held = self.inputs['held']
        return [[self._load(held.at[self.token.row, j], first, width) for j in range(held.shape[1])]
                for first, width in self.pieces]

    def write(self, rows):
        """Each stream's moved history: the block's values', and each head's queries' and keys' where the
        head is its key head's first."""
        def store(piece, history):
            for j, row in enumerate(history):
                plgpu.store(self.moved.at[self.token.row, j, pl.ds(*piece)], row.astype(self.moved.dtype),
                            optimized=False)

        pieces = self.pieces
        store(pieces[0], rows[0])
        for p, head in enumerate(self.token.heads):
            @pl.when(head % self.group == 0)
            def _(p=p):
                store(pieces[1 + 2 * p], rows[1 + 2 * p])
                store(pieces[2 + 2 * p], rows[2 + 2 * p])


def _staged(hooks, inputs, token: _Token, index: int, query, key, vectors, exchanged, *, interpret: bool):
    """The block's `index`-th head's key vectors in `vectors[index]`, fp32: its queries normed and scaled,
    its keys normed (rows 0 and 1) and its decay a key dimension (row 2), as `recurrence.step_reference`'s
    prologue and gates form them; its write gate returned, a scalar."""
    key_width = vectors.shape[2]
    layout = scan.blocked(key_width)
    if hooks.l2norm:
        # `l2norm`'s scale, its root and quotient each rounded once (rsqrt's approximation 2 ulp off).
        def scale(x):
            return 1.0 / jnp.sqrt(scan.summed(x * x, exchanged, interpret=interpret) + L2NORM_EPS)

        query, key = query * scale(query), key * scale(key)
        query = query * (key_width ** -0.5 if hooks.scale is None else hooks.scale)
    elif hooks.scale is not None:
        query = query * hooks.scale
    decay, write = _gates(hooks, inputs, token.row, token.heads[index], layout)
    for row, value in enumerate((query, key, jnp.broadcast_to(decay, (key_width,)))):
        plgpu.store(vectors.at[index, row], value, optimized=False)
    return write


def _gates(hooks, inputs, row, head, layout):
    """Value head `head`'s decay (a scalar, or a key dimension's vector in `layout`) and write gate (a
    scalar) at `row`, as `recurrence.step_reference`'s `_gates` form them."""
    def scalar(name):
        """A row's entry a head (`[rows, Hv]`), fp32."""
        return inputs[name][row, head].astype(jnp.float32)

    def headed(name):
        """A head's entry (`[Hv]`), fp32."""
        return inputs[name][head].astype(jnp.float32)

    one = jnp.float32(1.0)
    if hooks.gates is not None:
        rate = _softplus(scalar('gate') + headed('bias'))
        if hooks.gates.limit is not None:
            rate = jnp.clip(rate, *hooks.gates.limit)
        decay = jnp.exp(-jnp.exp(headed('log_a')) * rate)
        return decay, rate if hooks.gates.write == 'step' else _sigmoid(scalar('beta'))
    if hooks.rated:
        write = scalar('beta')
        return jnp.exp(headed('rate') * write), write
    write = scalar('beta') if hooks.gated_write else one
    if hooks.decay == 'none':
        return one, write
    if hooks.decay == 'vector':
        decay = plgpu.load(inputs['log_decay'].at[row, head], layout=layout, optimized=False)
        return jnp.exp(decay.astype(jnp.float32)), write
    return jnp.exp(scalar('log_decay')), write


def _sigmoid(x):
    """The sigmoid, one quotient (`short_conv._sigmoid`'s): Mosaic GPU lowers no logistic."""
    return 1.0 / (1.0 + jnp.exp(-x))


def _softplus(x):
    """`jax.nn.softplus`, `max(x, 0) + log1p(exp(-|x|))`, its log1p as `log(u) y / (u - 1)` at `u = 1 + y`
    (Goldberg's, within an ulp; `y` where `u` rounds to one): Mosaic GPU lowers no log1p."""
    y = jnp.exp(-jnp.abs(x))
    u = 1.0 + y
    return jnp.maximum(x, 0.0) + jnp.where(u == 1.0, y, jnp.log(u) * (y / (u - 1.0)))


def _stepped(hooks, state_ref, state_out, value, vectors, write, token: _Token):
    """`recurrence.update` over the block's column of the state, a key a turn of a rolled loop, read from
    `state_ref` (its delta rule's correction a pass of its own, which reads it again) and written to
    `state_out` as it is held (`recurrence._held`): the output `q^T S'`. Each key dimension's vectors are read
    from `vectors` as each head's scalars (`_key_entry`) and each sum over the keys a pairwise tree
    (`_tree_added`). Every key's entry held at once, a 128-key column spilled 1260 to 1276 bytes a thread
    (sm80, sm90; the fused step's compile, jsf-r06j)."""
    key_width = state_ref.shape[2]

    def entry(which, k):
        return _key_entry(vectors, token, which, k)

    levels = key_width.bit_length()
    empty = tuple(jnp.zeros_like(value) for _ in range(levels))
    corrected = value
    if hooks.delta:
        def correction(k, partials):
            return _tree_added(partials, k, (entry(1, k) * entry(2, k)) * token.tile(state_ref, k))

        corrected = value - lax.fori_loop(0, key_width, correction, empty)[-1]

    def updated(k, partials):
        entry_k = token.tile(state_ref, k) * entry(2, k) + (entry(1, k) * write) * corrected
        # Held as `recurrence._held` holds it: bf16 rounded stochastically, salted by its key and value, the
        # key's entry laid over the block as the entry is (a scalar's bit cast lowers to no vector).
        keyed = jnp.zeros_like(entry_k) + entry(1, k)
        kept = (recurrence.salted_bf16(entry_k, keyed, value) if state_out.dtype == jnp.bfloat16
                else entry_k)
        token.stored(state_out, kept, k)
        return _tree_added(partials, k, entry(0, k) * entry_k)

    return lax.fori_loop(0, key_width, updated, empty)[-1]


def _tree_added(partials: tuple, index, term) -> tuple:
    """The `index`-th of a power of two's terms added into the pairwise tree of the ones before (`partials`,
    a complete subtree's sum a level, `index`'s set bits the levels held): the term carried up the held
    levels, each the earlier sum on the left, and left at the first free one; the last term leaves the total
    at the top. XLA's reduction sums pairwise: a thread's running sum of 128 keys' terms put the fused step's
    output at 1.55 to 1.61 times the XLA form's distance from float64 (jsf-r06h). Each level's choice is the
    loop index's, laid over the vectors (`mosaic.laid`)."""
    carry, active, found = term, jnp.ones((), bool), []
    for level, partial in enumerate(partials):
        held = ((index >> level) & 1) == 1
        added = mosaic.laid(active & held, carry)
        left = jnp.where(added, jnp.zeros_like(partial), partial)
        found.append(jnp.where(mosaic.laid(active & ~held, carry), carry, left))
        carry = jnp.where(added, partial + carry, carry)
        active = active & held
    return tuple(found)


def _epilogue(hooks, inputs, token: _Token, out, value, exchanged, *, interpret: bool):
    """The token's output past the update: its skip and its gated norm over the head's columns (a block of
    one head a warpgroup wide, `serves`), as `recurrence.step_reference`'s epilogue (`recurrence.skipped`,
    `recurrence.normed`)."""
    if hooks.skip:
        out = out + token.per_head([inputs['skip'][head].astype(jnp.float32) for head in token.heads]) * value
    if hooks.output is None:
        return out
    gate = token.tile(inputs['z'])
    weight = plgpu.load(inputs['weight'].at[pl.ds(0, 1), token.columns], layout=token.block.layout,
                        optimized=False).astype(jnp.float32)
    # As `precision.rms_statistics` takes it: the sum times the count's reciprocal, over a head's columns.
    total = scan.summed((out * out).reshape(-1), exchanged, interpret=interpret)
    mean = total * (1.0 / token.block.columns)
    return rms_weighted(out, lax.rsqrt(mean + hooks.output.epsilon), weight=weight, gate=gate,
                        activation=hooks.output.activation)
