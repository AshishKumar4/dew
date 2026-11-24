"""A diagonal linear recurrence, `h_t = exp(g_t) h_{t-1} + x_t` a channel, each token's output its state:
HGRN's (fla's `chunk_hgrn`) and the carry of every recurrence whose state is a vector of channels each
decaying alone (RG-LRU's, MinGRU's), routed by its selector (`DIAGONAL_SCAN`) to the recurrence template's
walk (`templates.recurrence.diagonal`) or the chunked op's portable math (`recurrence_chunk.chunk`, row 1453).
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

from dew.nn.precision import at_least_fp32
from dew.nn.sharding import grouped_heads, manual_map, varying

from . import generation, recurrence
from .generation import INHERENT, Choice
from .recurrence_chunk import _VARYING, CHUNK_SIZE, Chunk, _per_product, chunk
from .selection import Kernel, Op, Requirement, select
from .templates import scan as block_scan

DIAGONAL = recurrence.Hooks(decay='none', delta=False, value_decay=True, scale=1.0)
"""A diagonal recurrence's form (`diagonal_scan`): a key width of one, its decay along the values, no delta
rule."""

DIAGONAL_WIDTH = 64
"""The channels `diagonal_scan` lays a head: any grouping is the same recurrence, a channel its own state."""


@dataclasses.dataclass(frozen=True)
class DiagonalCase:
    """A diagonal scan (`diagonal_scan`) as its selector reads it (`DIAGONAL_SCAN`): whether a shard_map that
    checks varying axes holds its operands, its writes' dtype, its channels, and its device's instruction
    family (None off a GPU)."""

    varying: bool
    dtype: np.dtype
    channels: int
    family: str | None = None


DIAGONAL_SUPPORTS: tuple[Requirement[DiagonalCase], ...] = (
    Requirement('varying', lambda case: not case.varying, _VARYING, row='ledger 822'),
    Requirement('gpu', lambda case: case.family is not None,
                "no GPU of an instruction family the template's bundles cover: off one the template "
                "interprets, as the CPU suite runs it where a call names it", row=INHERENT, kind='measured'),
    Requirement('dtype', lambda case: case.dtype in (jnp.bfloat16, jnp.float16, jnp.float32),
                lambda case: f"{case.dtype} writes; the kernels take 16-bit and fp32 ones", row=INHERENT),
    Requirement('channels', lambda case: case.channels % block_scan.THREADS == 0,
                lambda case: f"{case.channels} channels, no whole number of a warpgroup's "
                             f"{block_scan.THREADS}", row='ledger 1453'),
)
"""What the template's diagonal scan (`templates.recurrence.diagonal`) needs of a call: no caller's
`shard_map` that checks varying axes, a GPU of a family its bundles cover, 16-bit or fp32 writes, and whole
warpgroups of channels."""

DIAGONAL_SCAN: Op[DiagonalCase] = Op(Choice.DIAGONAL_SCAN, (Kernel('mosaic_gpu', DIAGONAL_SUPPORTS),
                                                           Kernel('xla', (), generic=True)))
"""The diagonal scan's selector: the recurrence template's walk (`templates.recurrence.diagonal`) where it
takes the call, else the portable math (the chunked op's, `DIAGONAL`)."""


def diagonal_kernel(x, *operands) -> str:
    """The kernel the diagonal scan's selector (`DIAGONAL_SCAN`) gives a scan over writes `x` `[B, S, D]` and
    its other launch operands (`operands`: the gates, the state, the documents) on the target, reported as
    its route: any of them a checked `shard_map` varies refuses the kernels (`sharding.varying`; the writes
    alone read, replicated writes beside row-varying gates took them, ReviewRecurrenceBatch)."""
    from .templates import bundles

    on_gpu = generation.backend() == 'gpu'
    bundled = on_gpu and bundles.of(generation.device_generation(), x.dtype) is not None
    case = DiagonalCase(varying=bool(varying(x, *operands)), dtype=jnp.dtype(x.dtype),
                        channels=x.shape[-1],
                        family=generation.family(generation.device_generation()) if bundled else None)
    return select(DIAGONAL_SCAN, case).kernel


def diagonal_scan(x, g, state=None, *, chunk_size: int = CHUNK_SIZE, starts=None):
    """A diagonal linear recurrence, `h_t = exp(g_t) h_{t-1} + x_t` a channel, each token's output its state:
    HGRN's (fla's `chunk_hgrn`, fla/ops/hgrn, `x` the layer's write `(1 - a_t) i_t`, `g` its log forget gate)
    and the carry of every recurrence whose state is a vector of channels each decaying alone (RG-LRU's,
    MinGRU's). `x` and `g` `[B, S, D]` (`g` at most zero), `state` `[B, D]` the state entering, zero where
    None, `starts` `[B, S]` packed documents' first tokens, each from a zero state; `(output [B, S, D] in
    `x`'s dtype, final state [B, D] in at least fp32)`, on the kernel its selector gives each product
    (`DIAGONAL_SCAN`, `_per_product`): the recurrence template's walk (`templates.recurrence.diagonal`),
    interpreted off a GPU where a call names it, per shard of rows on a mesh (`manual_map`); or the portable
    math (`_diagonal_portable`). A call of no tokens walks none: its final state is the one entering."""
    if x.shape[1] == 0:
        wide = at_least_fp32(x.dtype)
        return x, jnp.zeros((x.shape[0], x.shape[-1]), wide) if state is None else state.astype(wide)

    def template(x, g, state, *given):
        from .templates.recurrence import diagonal as launches, mainloop

        family, interpreted = mainloop.issued(jnp.float32)
        return launches.diagonal_scan(x, g, state, given[0] if given else None, family=family,
                                      interpret=interpreted)

    def run(kernel, x, g, state):
        if kernel != 'mosaic_gpu':
            return _diagonal_portable(x, g, state, starts, chunk_size)
        rows, _ = grouped_heads((*x.shape[:2], 1, x.shape[-1]), 1)
        three, two = P(rows, None, None), P(rows, None)
        held = state if state is not None else jnp.zeros((x.shape[0], x.shape[-1]), jnp.float32)
        operands = (x, g, held, *([] if starts is None else [starts]))
        return manual_map(template, (three, three, two, two)[:len(operands)], (three, two))(*operands)

    given = [operand for operand in (g, state, starts) if operand is not None]
    return _per_product(DIAGONAL_SCAN.bound(lambda product: diagonal_kernel(x, *given)), run, x, g, state,
                        parameters=())


def _diagonal_portable(x, g, state, starts, chunk_size: int):
    """`diagonal_scan` on the portable math: the chunked op's recurrence of a key width of one with its decay
    along the values (`DIAGONAL`: queries and keys one, the values `x`, `Hooks.value_decay` `g`), its
    channels `DIAGONAL_WIDTH` a head (row 1453)."""
    rows, length, channels = x.shape
    width = DIAGONAL_WIDTH if channels % DIAGONAL_WIDTH == 0 else channels
    heads = channels // width
    one = jnp.ones((rows, length, heads, 1), x.dtype)
    entered = None if state is None else state.reshape(rows, heads, 1, width)
    operands = Chunk(one, one, x.reshape(rows, length, heads, width), state=entered, starts=starts,
                     value_decay=g.reshape(rows, length, heads, width))
    out, final = chunk(DIAGONAL, operands, chunk_size=chunk_size)
    return out.reshape(rows, length, channels), final.reshape(rows, channels)
