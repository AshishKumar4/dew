"""The recurrence template's chunked op: a sequence's tokens through the linear recurrence of
`dew.nn.kernels.recurrence` a chunk at a time, for training and prefill (`chunk`).

Per head, over a state `S` `[Dk, Dv]`, a token updates and reads

    S' = (I - delta beta k k^T) Gamma S + iota k v^T,    o = S'^T q (+ D v)

as the decode step does (`recurrence.step`), its form the same `recurrence.Hooks`. Within a chunk of
`C` tokens the dependencies resolve in matrix form, the forms derived from the token update
(`dew.algebra.scan`: the decays between positions, the delta rule's corrections one strictly lower inverse,
fla's WY form), and only the memory crosses chunks. Without the delta rule (`delta=False`, R = 0, Mamba2's
SSD and the gated linear attentions) the inverse is the identity and nothing is corrected.

The recurrence template's plain-JAX mainloop (`xla_mainloop`, those forms in XLA) computes every hook: Dew's
float32 reference and the CPU path. Kernels take the rest, by what each takes and in order, no GPU named: the
delta rule with one decay a head or one per key dimension runs on the recurrence template
(`template.mosaic_rule`) where its launches take the call and the cell is not a declared port cell
(`PORT_CELLS`), else on the port of fla 0.5.2's kernels (`fla_rule`, `fla_gdn` and `fla_kda`) on any GPU Dew's
Triton kernels run on (`GATED_DELTA`); without the delta rule, a rated decay (Mamba2's SSD) runs on the port
of mamba_ssm's kernels (`dew.nn.kernels.mamba_ssd`) on such a GPU, or on a TPU's Pallas scan
(`dew.nn.kernels.ssd`) (`SCAN`). `chunk` picks one per call and applies the hooks around it.
"""

import contextlib
import contextvars
import dataclasses
import functools
import math
from collections.abc import Callable
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

from dew.algebra.scan import RESET_DECAY
from dew.batch import document_firsts
from dew.nn.attention_masks import repeat_kv_heads
from dew.nn.derivatives import KernelGlue, Protected, contracted
from dew.nn.inputs import cu_seqlens
from dew.nn.precision import at_least_fp32, l2norm
from dew.nn.sharding import grouped_heads, manual_map, rows_like, varying_like

from . import fla_gdn, fla_kda, generation, mamba_ssd, recurrence, recurrence_algebra as algebra
from .generation import INHERENT, triton_runs
from .selection import Kernel, KernelRefused, Op, Requirement, select, selected
from .ssd import _chunk_scan, ssd_kernel_platform
from .templates import base, scan as block_scan

CHUNK_SIZE = 64
"""Tokens per chunk of the chunked form, the reference's default
(`torch_chunk_gated_delta_rule(..., chunk_size=64)`). The chunked and the
recurrent form agree at any size; the size only trades memory for
parallelism, and 64 is what the reference's kernels assume."""

def held_chunk(chunk_size: int, key_width: int) -> int:
    """The delta rule's chunk at `key_width`: `chunk_size`, at most twice the keys' width. A measured
    sizing policy, not a conditioning bound: over a long sequence (1024 tokens, the decay near one),
    64-token chunks of 16-wide keys put the final memory 1.7 to 1.8 times the token-by-token
    recurrence's error from float64 under every delta family, where chunks of 32 and 16 held it, and so
    did 64-token chunks of 32-, 64- and 128-wide keys (tests/test_recurrence_stability.py, its long
    sequence)."""
    return min(chunk_size, 2 * key_width)



def chunk_gated_delta_rule(query, key, value, g, beta, state=None, chunk_size: int = CHUNK_SIZE, *,
                           starts=None, max_segments: int | None = None, floor: float | None = None,
                           write_bound: float | None = None, firsts=None, entered=None):
    """`torch_chunk_gated_delta_rule` (modeling_qwen3_next.py:374-453) with one
    log decay per head, `g` `[B, S, H]`, or `chunk_kimi_delta_attention`
    (modeling_glm5_next.py:482-578) with one per key dimension, `[B, S, H,
    Dk]`: the delta rule's row of the chunked op (`chunk`), its queries
    scaled by `Dk^-1/2`. Its queries and keys may sit on fewer heads than the
    values, each serving `Hv // Hk` of them, as fla takes them.

    Its kernels (`GATED_DELTA`: the recurrence template, or fla 0.5.2's
    chunked rule on Dew's port of its kernels, `fla_rule`, on fla's numbers)
    take it where they take these operands; elsewhere the plain-JAX mainloop
    (`xla_mainloop`), in `chunk_size` chunks, which the kernels' 64 need
    not match: the rule's value does not depend on them. Each takes the
    operands as they come, which a layer rounds for the route it asks of
    them first (`kernel_routed`: the projections' dtype for a kernel, as
    fla's layer hands them over, the wide one for the math).

    `starts` `[B, S]` marks packed documents' first tokens, each starting
    from a zero memory (`state` continues a row's first document), at most
    `max_segments` a row (`Coordinates.max_segments`): the port runs each
    document as a sequence of its own (`fla_rule`), its kernels' grids sized
    by that bound, and the portable math resets its recurrence at each start,
    which needs none. `floor` is the least log decay a token the layer declares (`recurrence.Hooks.floor`:
    Kimi delta attention's lower bound), which the recurrence template's factored form of a decay a key
    dimension reads; `write_bound` the largest write gate it declares of keys it l2-normalized
    (`recurrence.Hooks.write_bound`), which the template's 16-bit solve reads (`mainloop._merged`).
    """
    hooks = recurrence.Hooks(decay='vector' if g.ndim == 4 else 'scalar', scale=key.shape[-1] ** -0.5,
                             floor=floor, write_bound=write_bound)
    return chunk(hooks, Chunk(query, key, value, log_decay=g, beta=beta, state=state, starts=starts,
                              firsts=firsts, entered=entered),
                 chunk_size=chunk_size, max_segments=max_segments)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Chunk(recurrence.Gated):
    """A sequence's operands for the chunked op (`chunk`), each `[B, S, ...]`; a hook's are None where
    it is off, as a decode token's are (`recurrence.Step`).

    `query` and `key` `[B, S, Hk, Dk]`, `value` `[B, S, Hv, Dv]`, each key head serving `Hv // Hk`
    value heads. `log_decay` is `[B, S, Hv]` (scalar) or `[B, S, Hv, Dk]` (vector), or with the gates
    `gate` `[B, S, Hv]`, `log_a` and `bias` `[Hv]`. `beta` `[B, S, Hv]` is the write gate (a raw beta
    with the gates), `skip` `[Hv]` the skip's `D`, `z` `[B, S, Hv, Dv]` and `weight` `[Dv]` the
    epilogue's, `rate` `[Hv]` a rated decay's, at most zero (`recurrence.Hooks.rated`). `state`
    `[B, Hv, Dk, Dv]` is the memory entering, zero where None, and `starts` `[B, S]` marks packed
    documents' first tokens, each from a zero memory, `firsts` their rows' documents' first slots as the
    batch built them (`dew.batch.Coordinates.firsts`; the kernels sort them from `starts` where None).

    Where `entered` `[B, D]` int is given (`dew.batch.Entered`'s), each packed document enters a memory of
    its own: `state` is `[B, D, Hv, Dk, Dv]`, a memory a slot, and a row's `d`-th document enters its
    `entered[d]`-th slot (one-based; zero, a zero memory); the final memory is each document's own, `[B, D,
    Hv, Dk, Dv]` (`DocumentRows`)."""
    query: jax.Array
    key: jax.Array
    value: jax.Array
    state: jax.Array | None = None
    starts: jax.Array | None = None
    firsts: jax.Array | None = None
    entered: jax.Array | None = None


def chunk(hooks: recurrence.Hooks, operands: Chunk, *, chunk_size: int = CHUNK_SIZE,
          max_segments: int | None = None, dtype=None):
    """`(output [B, S, Hv, Dv], final memory [B, Hv, Dk, Dv])` of a sequence through the recurrence of
    `hooks` (`recurrence.Hooks`, the decode step's form), the output in `dtype` (the queries', keys'
    and values' common dtype where None) and the memory in at least fp32.

    The prologue (the l2 norm, the queries' scale, the gates' activations) and the epilogue (the
    skip, the gated RMS norm) are the decode step's arithmetic (`recurrence.step_reference`) on every
    token, in at least fp32. The recurrence between them runs, for the delta rule with a decay, on the
    kernel the delta rule's selector gives each product (`GATED_DELTA`, `port_kernel`: the recurrence
    template, or the port of fla's kernels, `fla_rule`); without the delta rule, a rated decay (Mamba2's
    SSD) runs on the kernel the scan's selector gives each product (`SCAN`: the recurrence template, the port
    of mamba_ssm's kernels, or a TPU's Pallas scan where its tiling covers the geometry); else on the
    portable math (`xla_mainloop`, in `chunk_size` chunks, the delta rule's no longer than `held_chunk`,
    which a kernel's need not match), which `generation.chosen` names for either selector ('xla').
    Packed documents (`starts`, at most `max_segments` a row) each compute as they would alone. A conv
    runs before the op: it takes the conv's output. The operands the hooks read are checked as the
    decode step's (`recurrence.read`), and a rated decay is taken on its declared domain, a contraction.
    A memory entering is read through its chunk's decays at their size while it lies within 2^96 of the
    values: past it, a decay that leaves fp32's normals (a chunk's prefix under 2^-126) reads it as zero, as
    fla's chunked form does (row 2344: on par at 2^96 through a token's decay of exp2(-124.8), a write gate's
    output 150 times the recurrence's distance at 2^110). A call names a kernel through `generation.chosen`
    alone, the recurrence template's 'mosaic_gpu' among them (row 959: the selectors decide every call)."""
    if hooks.conv is not None:
        raise ValueError("the chunked op takes the conv's output: its queries, keys and values")
    given = recurrence.read(hooks, operands, optional=('state',))
    query, key, value = given['query'], given['key'], given['value']
    if value.shape[2] % key.shape[2]:
        raise ValueError(f"{value.shape[2]} value heads over {key.shape[2]} key heads: each key head serves "
                         "a whole number of value heads")
    ranks = {'scalar': 3, 'vector': 4}
    if 'log_decay' in given and given['log_decay'].ndim != ranks[hooks.decay]:
        raise ValueError(f"a {hooks.decay} decay is {ranks[hooks.decay]}-dimensional, [B, S, Hv"
                         f"{', Dk' * (hooks.decay == 'vector')}], not {given['log_decay'].shape}")
    dtype = jnp.result_type(query, key, value) if dtype is None else jnp.dtype(dtype)
    work = at_least_fp32(jnp.result_type(query, key, value))
    if hooks.l2norm:
        # Normalized in fp32 and rounded once to their own dtype, as fla's in-kernel norm and the layers'
        # conv hand them to a kernel: kept fp32, a bf16 call reached the template as an fp32 one, its final
        # memory 30 to 41 times fla's distance from float64 in two parts (row 184's GDN cell, acb1301632c).
        query, key = (l2norm(x.astype(work)).astype(x.dtype) for x in (query, key))
    scale = (key.shape[-1] ** -0.5 if hooks.l2norm else 1.0) if hooks.scale is None else hooks.scale
    g, beta = recurrence.log_gates(hooks, given, work)
    g = jnp.zeros(value.shape[:3], work) if g is None else g
    beta = jnp.ones(value.shape[:3], work) if beta is None else beta
    state, starts, firsts = operands.state, operands.starts, operands.firsts
    rows = None
    if operands.entered is not None:
        # Each document a row of its own from its slot's memory, every kernel's call unpacked.
        rows = DocumentRows.of(starts, operands.entered, firsts)
        query, key, value, g, beta = rows.laid(query, key, value, g, beta)
        state, starts, firsts, max_segments = rows.entering(state), None, None, None
    epilogue = hooks.skip or hooks.output is not None
    if hooks.rated and not hooks.delta and not hooks.l2norm and hooks.scale is None:
        # Each of its routes adds the skip, the port in its kernels.
        out, final = _rated_scan(hooks, query, key, value, g, beta, given, state, starts, firsts, chunk_size,
                                 work, dtype, max_segments=max_segments)
    else:
        # The delta rule's selector decides every form (`GATED_DELTA`): the forms its kernels do not run hand
        # over to the portable math there, reported as the call's route.
        def product_kernel(product: str) -> str:
            return port_kernel(query, key, value, g, beta, state, starts, max_segments=max_segments,
                               scale=scale, wide=epilogue, floor=hooks.floor, bound=hooks.inverse_bound,
                               product=product,
                               delta=hooks.delta, decayed=hooks.decay != 'none')

        def run(kernel, query, key, value, g, beta, state, starts, firsts):
            if kernel == 'mosaic_gpu':
                return template_rule(query, key, value, g, beta, state, starts, firsts,
                                     max_segments=max_segments, scale=scale, delta=hooks.delta, wide=epilogue,
                                     floor=hooks.floor, bound=hooks.inverse_bound,
                                     rounded=chunk_rounding(hooks.delta, jnp.result_type(query, key, value)))
            if kernel == 'fla':
                # Unrounded where an epilogue follows, as on the portable math.
                return fla_rule(query, key, value, g, beta, state, starts=starts, max_segments=max_segments,
                                scale=scale, wide=epilogue, firsts=firsts)
            size = held_chunk(chunk_size, key.shape[-1]) if hooks.delta else chunk_size
            return xla_mainloop(Tiles(size), query, key, value, g, beta, state, starts, scale=scale,
                                delta=hooks.delta, dtype=work, bound=hooks.inverse_bound)

        out, final = _per_product(product_kernel, run, query, key, value, g, beta, state, starts, firsts)
        out = recurrence.skipped(hooks, out.astype(work), value, given, work)
    if rows is not None:
        out, final = rows.unlaid(out), rows.leaving(final)
    # The epilogue in the working precision, and the output rounded once.
    return recurrence.normed(hooks, out, given, work).astype(dtype), final


@dataclasses.dataclass(frozen=True)
class DocumentRows:
    """A packed call's documents each a row of its own, `[B * D, S]`, the row's tokens from a document's
    first, past its end written nothing (its decay one, its write gate zero), so its final memory is the
    document's: each document enters its own memory (`Chunk.entered`) on any kernel, as a row of its own
    enters its call's `state`. A call holds `D` rows a row of `S` tokens, its work `D` times a packed row's
    where its documents are short."""

    begins: jax.Array
    """Each row's documents' first slots `[B, D]` (`dew.batch.document_firsts`), `S` past its last."""
    ends: jax.Array
    """Each document's end `[B, D]`, the next one's first slot or the row's length."""
    entered: jax.Array
    """The slot each document enters `[B, D]`, one-based, zero for a zero memory."""

    @classmethod
    def of(cls, starts, entered, firsts=None) -> "DocumentRows":
        """`starts` `[B, S]`' documents (or their first slots, `firsts`), each entering its `entered` slot."""
        documents = entered.shape[1]
        begins = document_firsts(starts, documents) if firsts is None else firsts[:, :documents]
        length = starts.shape[1]
        ends = jnp.concatenate([begins[:, 1:], jnp.full_like(begins[:, :1], length)], axis=1)
        return cls(begins, jnp.minimum(ends, length), jnp.asarray(entered, jnp.int32))

    def _index(self, length: int):
        """Each document row's token's slot in its packed row `[B, D, S]` and whether it is the document's."""
        offsets = jnp.arange(length, dtype=jnp.int32)
        slots = self.begins[..., None] + offsets
        return jnp.minimum(slots, length - 1), slots < self.ends[..., None]

    def laid(self, query, key, value, g, beta):
        """The operands `[B, S, ...]` as the documents' rows `[B * D, S, ...]`: past a document's end its
        decay one (`g` zero) and its write gate zero, its queries, keys and values zero."""
        slots, own = self._index(beta.shape[1])

        def taken(x, fill):
            picked = jax.vmap(lambda row, at: row[at])(x, slots)
            kept = own.reshape(*own.shape, *(1,) * (picked.ndim - own.ndim))
            picked = jnp.where(kept, picked, jnp.asarray(fill, x.dtype))
            return picked.reshape(-1, *picked.shape[2:])

        return (taken(query, 0), taken(key, 0), taken(value, 0), taken(g, 0), taken(beta, 0))

    def entering(self, state):
        """Each document row's memory `[B * D, Hv, Dk, Dv]`: its slot's of `state` `[B, D, Hv, Dk, Dv]`, zero
        for slot zero."""
        if state is None:
            raise ValueError("documents entering memories of their own (`Chunk.entered`) read the slots' "
                             "memories, `state` `[B, D, Hv, Dk, Dv]`")
        picked = jax.vmap(lambda row, slots: row[jnp.maximum(slots - 1, 0)])(state, self.entered)
        kept = (self.entered > 0).reshape(*self.entered.shape, 1, 1, 1)
        return jnp.where(kept, picked, 0.0).reshape(-1, *state.shape[2:])

    def unlaid(self, out):
        """The documents' rows' output `[B * D, S, ...]` back as the packed rows' `[B, S, ...]`: each token's
        its document's row's at its offset from the document's first."""
        rows, documents = self.begins.shape
        length = out.shape[1]
        out = out.reshape(rows, documents, *out.shape[1:])
        offsets = jnp.arange(length, dtype=jnp.int32)
        document = jnp.sum(offsets[None, :, None] >= self.begins[:, None, :], axis=-1) - 1
        document = jnp.clip(document, 0, documents - 1)
        at = offsets[None, :] - jnp.take_along_axis(self.begins, document, axis=1)
        return jax.vmap(lambda row, d, t: row[d, t])(out, document, jnp.clip(at, 0, length - 1))

    def leaving(self, final):
        """The documents' rows' final memories `[B * D, Hv, Dk, Dv]` as each document's, `[B, D, Hv, Dk,
        Dv]`."""
        return final.reshape(*self.begins.shape, *final.shape[1:])


@dataclasses.dataclass(frozen=True)
class PortCase:
    """A delta rule's call as the port's selector reads it (`GATED_DELTA`): whether a caller's
    `shard_map` that checks varying axes holds operands that vary (`varying`), the operands' common
    `dtype`, the keys' width (`key_width`), the recurrence template's implementations bound to the call
    (`call`, `templates.base.Call`; None where the caller asks of the port alone), the
    instruction family of the device it runs on (`family`, `generation.family`; None off a GPU), and the
    product it is selected for (`product`: 'forward' where no gradient is taken, 'training' where one is,
    `_per_product`), and the recurrence's form: whether it is the delta rule (`delta`) and decays
    (`decayed`), the largest entry of its chunk inverse the form proves (`bound`,
    `recurrence.Hooks.inverse_bound`; None where it proves none), and whether its decay is a key dimension's
    (`per_key`)."""

    varying: bool
    dtype: np.dtype
    key_width: int
    call: base.Call | None = None
    family: str | None = None
    product: str = 'forward'
    delta: bool = True
    decayed: bool = True
    bound: float | None = None
    per_key: bool = False

    @property
    def cell(self) -> tuple[str | None, str]:
        """The call's cell, its device's instruction family and its product (`selection.Celled`,
        `PORT_CELLS`), a decay a key dimension's (`per_key`, Kimi delta attention's) named in it."""
        return self.family, f"{self.product}, a decay a key dimension" if self.per_key else self.product

    @property
    def widest(self) -> int:
        """The widest keys the port takes in the call's dtype."""
        return fla_gdn.WIDEST_KEYS if self.dtype.itemsize == 2 else fla_gdn.WIDEST


_VARYING = ("a shard_map that checks varying axes holds the operands, and a Pallas call states no varying "
            "type for its results")

PORT_SUPPORTS: tuple[Requirement[PortCase], ...] = (
    Requirement('gated', lambda case: case.delta and case.decayed,
                "a recurrence other than the gated delta rule, which the rule's kernels run (fla's "
                "chunk_gated_delta_rule and chunk_kda, the template's delta-rule mainloop): no delta rule (a "
                "gated linear attention, a rated scan with its keys normed or scaled) or no decay (the plain "
                "delta rule), which the portable math runs", row=INHERENT, kind='handoff'),
    Requirement('varying', lambda case: not case.varying, _VARYING, row='ledger 822'),
    Requirement('float64', lambda case: case.dtype != jnp.float64,
                "float64 operands, whose rule is the reference math itself", row=INHERENT),
    Requirement('key_width', lambda case: case.key_width <= case.widest,
                lambda case: f"keys {case.key_width} wide; the port takes at most {case.widest} in "
                             f"{case.dtype.name}", row=INHERENT),
    Requirement('triton', lambda case: triton_runs(),
                "no GPU Dew's Triton kernels run on (sm80 or later): off one the port's kernels interpret, "
                "as the CPU suite runs them where a call names the port", row=INHERENT, kind='measured'),
)
"""What the port (`fla_rule`) needs of a rule's call: the gated delta rule, a decay with the delta rule (the
other forms hand over to the portable math, the call's route reporting it); a GPU Dew's Triton kernels run on,
any of them (no target's row names it: a GPU no row named took the portable math, 140 to 155 ms a training
step where the port took 3.1, template review 2026-10-26 P1-1), and every shape, dtype, chunk size and mesh,
except inside a caller's `shard_map` that checks varying axes over operands that vary, where a Pallas call,
which states no varying type for its results, cannot run; float64 operands, whose rule is the reference math
itself; and keys wider than the port takes in their dtype. Packed documents take the port at any bound: it
stores the memories by where they pass, as many as the tokens' chunks, a row's and one, whatever the bound
(`chunked_launch.Sequences`)."""

PORT_CELLS: dict[tuple[str | None, str], tuple[str, str]] = {
    ('wgmma', 'forward, a decay a key dimension'): (
        'fla', "on an H100 80GB HBM3, 16-bit, the port's forward 1.406 ms against the template's 1.624 on "
               "Kimi Linear's delta attention, the template's intra pairs 0.65 ms against the port's intra "
               "rows' 0.65 and its keys' weighted launch 0.25 against the port's 0.17 "
               "(window-h100-c70aaa8-b8109ee)"),
}
"""The cells, each an instruction family and a product (`PortCase.cell`), where a receipt shows Dew's port
of fla's kernels (`fla_rule`; "the port" below) faster than the recurrence template, each its old route's
kernel and that receipt; every time below sets the template against that port, not against upstream fla's
own kernels or FlashQLA's (tools/benchmark_recurrence.py's `fla` and `flashqla` arms): the delta rule's
selector runs the port on them where it takes the call (`Op.declared`, owner, bible da64ae5); the list
shrinks as the template wins its cells, and no cell is picked or ranked at run time. A gated delta net's
cells are all the template's. An A100's forward of Kimi delta attention is the template's, 5.06 ms against
the port's 8.81 (JS-kdacells-a100-a928ea2b4), and its training, 11.12 ms against 15.92
(batch-window-a100-b2ad84c61). Warpgroup MMA's training of Kimi delta attention is the template's, its
forward and gradient 3.926 ms a call against the port's 4.726 on an H100 80GB HBM3 and 3.916 against 4.694 on
a GH200 480GB (FS-RECROUTES-h100 and -gh200 at 26d3fdf5df: cold, exclusive, two rounds alternating the arms'
order). A gated delta net's training is the template's, its
keys launch a key block a program and its query/key launch's memory sum a product: its forward and gradient
on an H100 2.46 ms against the port's 3.12
on Qwen3.5-27B's gated delta net and 1.92 against 2.44 on Qwen3-Next's (JS-train80-h100-db2c4aca0), and on an
A100 SXM4 40GB 3.46 against 4.21 and 2.50 against 3.24 (JS-2075cellsd-a100-863706e94).
An A100's forward is the template's, its walks' 16-bit stores coalesced (`mosaic.coalesced`): 0.931 ms
against the port's 0.981 on Qwen3.5-27B's gated delta net and 0.692 against 0.759 on Qwen3-Next's, each in a
process of its own, three rounds (JS-2075cellsd-a100-863706e94; the port's Qwen3-Next forward took 0.650 in
two runs that followed a training run in one process, JS-2075cells-a100-491c05798).
An H100's forward is the template's, its solve launch two programs a multiprocessor on warpgroup MMA's layout
without a producer (`mainloop._solving`): on an H100 80GB HBM3 0.364 ms against the port's 0.514 on
Qwen3.5-27B's gated delta net and 0.279 against 0.327 on Qwen3-Next's (JS-solvefed-h100-92eacee68; on an
H100 PCIe, before its states and output launches kept a producer, 0.520 against 0.691,
JS-fed3-h100-9d3a38de3)."""

def _trains(call: base.Call | None) -> bool:
    """Whether the recurrence template's gradient runs the call (`Mosaic.trains`)."""
    from .templates.recurrence import template

    if call is None or not call.eligible('mosaic_gpu'):
        return False
    implementation, bound = call.bound['mosaic_gpu']
    assert isinstance(implementation, template.Mosaic)
    return implementation.trains(bound, call.target)


class Templated(Protocol):
    """A case the recurrence template's requirements read (`TEMPLATE_SUPPORTS`): the template's
    implementations bound to the call (`call`), the instruction family of the device it runs on (`family`,
    None off a GPU) and the product it is selected for (`product`)."""

    @property
    def call(self) -> base.Call | None: ...

    @property
    def family(self) -> str | None: ...

    @property
    def product(self) -> str: ...


TEMPLATE_SUPPORTS: tuple[Requirement[Templated], ...] = (
    Requirement('launch', lambda case: case.call is not None and case.call.eligible('mosaic_gpu'),
                "the recurrence template's launches take no call of these dtypes or widths, or no config "
                "fits this device, or no Mosaic bundle claims its instruction family", row='ledger 1294'),
    Requirement('device', lambda case: case.family is not None,
                "no GPU runs the call: off one, the mainloop runs interpreted where a call names it",
                row=INHERENT, kind='measured'),
    Requirement('backward', lambda case: case.product != 'training' or _trains(case.call),
                "a gradient the template's fast backward does not take (`mainloop_backward."
                "fast_backward_form`): fp32 operands under the accurate forms, an fp16 call without the "
                "delta rule, a decay a key dimension without it; or launches past the device's registers or "
                "shared memory at the call's tiles (`template.backward_holds`)", row='ledger 1294'),
)
"""What the recurrence template's Mosaic GPU mainloop (`templates.recurrence.template.Mosaic`) needs of a
call under either selector (`GATED_DELTA`, `SCAN`): its own launches' call (`Template.eligible`: dtypes,
widths, a decay a key dimension's floor, a bundle for the device's family, a config that fits), a GPU, and for
a training product its fast backward (`Mosaic.trains`; the rule's gradient takes no other, row 1926)."""

MOSAIC_SUPPORTS: tuple[Requirement[PortCase], ...] = (
    *PORT_SUPPORTS[1:2],
    *TEMPLATE_SUPPORTS)
"""What the recurrence template needs of a delta rule's call: the port's placement (a Pallas call's results),
and `TEMPLATE_SUPPORTS`, in
every form its mainloop implements, the delta rule with a decay or without one and the forms without the
delta rule alike (row 1963: the port's 'gated' form was the template's too). A declared port cell runs the
port first where the port takes the call (`PORT_CELLS`, `GATED_DELTA`'s declared cells)."""

MOSAIC = Kernel('mosaic_gpu', MOSAIC_SUPPORTS)


GATED_DELTA: Op[PortCase] = Op('gated_delta_rule',
                               (MOSAIC, Kernel('fla', PORT_SUPPORTS), Kernel('xla', (), generic=True)),
                               declared=PORT_CELLS)
"""The delta rule's selector: on a declared port cell (`PORT_CELLS`) the port where it takes the call;
otherwise the first of its kernels that takes the call, by what each takes (no table of GPUs): the
recurrence template at its derived tiles, else the port on any GPU Dew's Triton kernels run on, else the
template's plain-JAX mainloop (`xla_mainloop`), the forms derived from the token update."""


RESET = "first"
"""The derived rule's reset operand (`scan.Scan.reset`): a packed document's first tokens."""


@dataclasses.dataclass(frozen=True)
class Tiles:
    """The plain-JAX mainloop's config (`xla_mainloop`): its walk's chunk of `chunk` tokens."""

    chunk: int


def xla_mainloop(config: Tiles, query, key, value, g, beta, state=None, starts=None, *, scale: float,
                 delta: bool, dtype=None, bound: float | None = None):
    """The recurrence template's plain-JAX mainloop: the recurrence over `config.chunk`-token chunks by the
    forms derived from its token update (`dew.algebra.scan`, the compact
    WY form and its prefix derived, not written; each chunk's work that reads no memory at once, the memory's
    carry alone in sequence): each value head's update `h = D h - beta k (k^T D h) + beta
    k v^T` with the delta rule (`recurrence.update`'s), `h = D h + beta k v^T` without it, `D = exp(g)` a
    head's or each key's own, read `scale q^T h`; `(output, final memory)`, computed in at least fp32
    (`work`), the output in `dtype` (the queries', keys' and values' common dtype where None, as the port
    returns it; a caller's epilogue passes its working dtype, unrounded) and the memory in `work`, the memory
    entering `state` (zero where None). A length off the chunk grid is padded with tokens that leave the
    memory as it was (zero keys and write gates, a unit decay). `starts` `[B, S]`, where given, marks packed
    documents' first tokens, each entering a zero memory (the forms' reset, `scan.Scan.reset`). `bound`, the
    layer's write gate's largest where its keys are of unit norm (`recurrence.Hooks.inverse_bound`), is the
    declaration's range facts (`scan.Scan.ranges`)."""
    from dew.algebra import SATURATION, scan

    dtype = jnp.result_type(query, key, value) if dtype is None else jnp.dtype(dtype)
    work = at_least_fp32(jnp.result_type(query, key, value))
    group = value.shape[2] // key.shape[2]
    if group > 1:
        query, key = (jnp.repeat(x, group, axis=2) for x in (query, key))
    rows, length, heads, columns = value.shape
    width = key.shape[-1]
    per_key = g.ndim == 4

    def update(h, q, k, v, g, beta):
        del q
        decayed = jnp.exp(g).reshape(-1, 1) * h
        written = v - k @ decayed if delta else v
        return decayed + beta * k[:, None] * written[None, :]

    def readout(h, q, k, v, g, beta):
        del k, v, g, beta
        return scale * (q @ h)

    axes = {"q": ("time", "key"), "k": ("time", "key"), "v": ("time", "value"),
            "g": ("time", "key") if per_key else ("time",), "beta": ("time",)}
    reset = None if starts is None else RESET
    declared = axes if reset is None else {**axes, reset: ("time",)}
    # A log decay at most 0 (its decay at most 1); with a bound, keys of unit norm and gates at most it, so
    # two tokens' corrections overlap by at most it (Cauchy-Schwarz).
    ranges = {"g": (-math.inf, 0.0), **({} if bound is None else {"k": 1.0, "beta": bound, "overlap": bound})}
    forms = scan.derive_scan(scan.Scan(update, readout, declared, walked="time",
                                       state=((width, columns), work), reset=reset, ranges=ranges),
                             saturation=SATURATION)
    chunk_size = config.chunk
    chunks = -(-length // chunk_size)
    padded = chunks * chunk_size - length

    def laid(x):
        # [B, S, H, ...] padded and laid out as [B, H, chunks, C, ...].
        x = jnp.pad(x, [(0, 0), (0, padded)] + [(0, 0)] * (x.ndim - 2))
        x = jnp.moveaxis(x, 2, 1)
        return x.reshape(rows, heads, chunks, chunk_size, *x.shape[3:])

    operands = {name: laid(x.astype(work))
                for name, x in zip(axes, (query, key, value, g, beta), strict=True)}
    if starts is not None:
        # Each head's tokens reset where their row's document starts; a padded token starts none.
        operands[RESET] = laid(jnp.broadcast_to(starts[:, :, None], (rows, length, heads)))
    # The memory entering, and each carry below, typed as every operand varies (a checked `shard_map`'s
    # manual axes; an explicit axis's rows, as the portable math's): a step's outputs vary wherever its inputs
    # do, keys, gates or resets replicated or not.
    varying = (query, key, value, g, beta, starts)
    entering = varying_like(rows_like(jnp.zeros((rows, heads, width, columns), work), value) if state is None
                            else state.astype(work), *varying)

    def walking(operands, h):
        # Each chunk's values that read no memory at once; the memory alone in sequence; then each chunk's
        # outputs from the memory entering it, at once. Each chunk's entering memory kept, the adjoint's one
        # checkpoint a chunk.
        prepared = jax.vmap(forms.prepare)(operands)

        def one(carried, chunk):
            return forms.step(carried, chunk), carried

        # The walk's decay in the forms' own carry dtype (`Derived.init`'s, the state's float32 at least).
        empty = forms.init({})[0]
        decayed = varying_like(jnp.zeros((width,), empty.dtype), *jax.tree.leaves(operands))
        carried, entering = jax.lax.scan(one, (decayed, h), prepared)
        return (jax.vmap(forms.mapped)(prepared, entering), carried[1]), entering[1]

    def replayed(operands, h):
        # The walk token by token (`Derived.replay`), the derivative's arithmetic at every order.
        def one(h, chunk):
            outputs, h = forms.replay(chunk, h)
            return h, outputs

        h, outputs = jax.lax.scan(one, h, operands)
        return outputs, h

    def adjoint(entering, primals, cotangent, live):
        # Each chunk's adjoint (`Derived.backward`) from the memory entering it, the chunks in reverse.
        operands, _ = primals
        d_outputs, d_final = cotangent

        def one(d_h, chunk):
            chunk, h, d_out = chunk
            return forms.backward(chunk, (None, h), d_h, d_out)

        d_h, d_operands = jax.lax.scan(one, d_final, (operands, entering, d_outputs), reverse=True)
        # A reset's booleans take no cotangent (never `live`), held in their place.
        whole = jax.tree.leaves(({**operands, **d_operands}, d_h))
        return tuple(whole[index] for index in live)

    # The forms' walk for the value, its derivative the token update's replayed per chunk (algebra.md §6.9).
    walk = generation.linear_tangent(walking, adjoint, replayed, kind=Protected(), live_cotangents=True)

    def walked(operands, h):
        outputs, final = walk(operands, h)
        return outputs.reshape(-1, columns), final

    out, final = jax.vmap(jax.vmap(walked))(operands, entering)
    return jnp.moveaxis(out[:, :, :length], 1, 2).astype(dtype), final


def port_kernel(query, key, value, g, beta, state=None, starts=None, *, max_segments: int | None = None,
                scale: float | None = None, wide: bool = False, floor: float | None = None,
                bound: float | None = None, product: str = 'forward', report: bool = True, delta: bool = True,
                decayed: bool = True) -> str:
    """The kernel the delta rule's selector (`GATED_DELTA`) gives a call over these operands for `product`
    ('forward', 'training'), of the form `delta` and `decayed` state and the chunk inverse's `bound` its form
    proves (`recurrence.Hooks.inverse_bound`; `PortCase`), the template's
    implementations bound to it, on the target (`generation`'s, an assumed one's within `assumed`):
    reported as its route, or not where not `report` (`selected`). A kernel a call names
    (`generation.chosen`) insists on it (`selection.KernelRefused`: the port named over varying operands),
    but for a form the rule's kernels do not run, which the portable math takes."""
    from .templates.recurrence import template

    call = base.call((template.MOSAIC,), query, key, value, g, beta, state, starts,
                     max_segments=max_segments, scale=scale, delta=delta, wide=wide, floor=floor, bound=bound,
                     rounded=chunk_rounding(delta, jnp.result_type(query, key, value)),
                     recomputed=recomputed_memory())
    family = generation.family(generation.device_generation()) if generation.backend() == 'gpu' else None
    case = PortCase(bound=bound, varying=bool(jax.typeof(value).mat.varying),
                    dtype=jnp.dtype(jnp.result_type(query, key, value)), key_width=key.shape[-1], call=call,
                    family=family, product=product, delta=delta, decayed=decayed, per_key=g.ndim == 4)
    return select(GATED_DELTA, case).kernel if report else selected(GATED_DELTA, case).name


def _per_product(kernel_of: Callable[[str], str], run, *operands):
    """`run(kernel, *operands)` on the kernel a selector gives each product (`kernel_of('forward')`,
    `kernel_of('training')`), differentiated at every order through the portable math (`run('xla', ...)`,
    `generation.linear_tangent`): the forward's kernel where no derivative is taken, the training kernel's
    forward and its own pullback in reverse mode, and the portable tangent, in the kernels' dtypes, in
    forward mode however a jit, a scan or a remat staged the call, where a kernel's custom VJP gives none
    (row 2395). The training kernel's forward runs as it is (not `nested`), so the residuals its port names
    (`dew.nn.mixer_base.DELTA_RESIDUALS`) are a remat policy's to keep. The cells' products differ
    (`PORT_CELLS`, `SCAN_CELLS`, or a template that trains no such call), and a trace cannot tell them apart
    before its gradient is asked. A training kernel that refused the call refuses where a gradient is
    traced, not a forward alone. Where both products are the portable math's (a CPU's, or a device no
    kernel takes), the call is the portable math itself, whose own derivatives hold every order: traced
    through the products' split it traced the portable math three times over, half of a gated delta net
    layer's training step's trace on a CPU (row 3142)."""
    forward = kernel_of('forward')
    try:
        trained = kernel_of('training')
    except KernelRefused as refusal:
        training = functools.partial(_refusing, refusal)
    else:
        if forward == trained == 'xla':
            return run(forward, *operands)
        training = functools.partial(run, trained)
    primal = functools.partial(run, forward)

    def portable(*operands):
        shapes = jax.eval_shape(primal, *operands)
        return jax.tree.map(lambda x, shape: x.astype(shape.dtype), run('xla', *operands), shapes)

    def pulled(pullback, operands, cotangent):
        del operands
        return pullback(cotangent)

    return contracted("test_the_chunked_op_holds_every_order")(
        generation.linear_tangent(functools.partial(jax.vjp, training), pulled, portable, primal=primal,
                                  nested=False, kind=KernelGlue()))(*operands)


def _refusing(refusal: KernelRefused, *operands):
    """A product whose kernel refused the call (`_per_product`'s training): raises `refusal` where it is
    traced."""
    del operands
    raise refusal


def entering_memory(rule, query, key, value, g, beta, starts, *, axis: str, state=None):
    """The memory this shard of a sequence the mesh axis `axis` splits enters its slice with, `[B, Hv, Dk,
    Dv]` in at least fp32, inside a `shard_map` that holds `axis` manual: fla's context-parallel
    summary-then-states (fla/ops/cp/chunk_delta_h.py, fla-core 0.5.2). A delta rule's slice is affine in
    the memory it enters, `S = F S_0 + Z` (each token's transition on the left, `recurrence.update`), so
    each shard's summary is `rule`'s final memory over its slice from `[0 | I]` with values `[v | 0]`, in
    at least fp32 (rounding the identity block's products to the operands' dtype would round the
    transfer): `Z` the write its last document leaves from zero, `F` its transfer, zero where that document
    starts inside the slice. The summaries are each shard's span (`recurrence_algebra.Span`), whose exclusive
    prefix over `axis` (`recurrence_algebra.entering`) from the row's `state` (zero where None) on the first
    shard is the memory each enters, `E_r = F_{r-1} E_{r-1} + Z_{r-1}`; a slice that starts a document (its
    first token's `starts`) enters from zero. `rule(query, key, value, g, beta, state, *, starts)` returns
    `(output, final memory)`."""
    batch, _, heads, columns = value.shape
    width = key.shape[-1]
    wide = at_least_fp32(value.dtype)
    shard = jax.lax.axis_index(axis)
    begins = (jnp.zeros((batch,), bool) if starts is None else jnp.asarray(starts[:, 0], bool)) & (shard > 0)
    zero = rows_like(jnp.zeros((batch, heads, width, columns), wide), value)
    identity = jnp.where(begins[:, None, None, None], 0.0,
                         jnp.broadcast_to(jnp.eye(width, dtype=wide), (batch, heads, width, width)))
    widened = jnp.concatenate([value.astype(wide), jnp.zeros((*value.shape[:-1], width), wide)], axis=-1)
    _, summary = rule(query.astype(wide), key.astype(wide), widened, g, beta.astype(wide),
                      jnp.concatenate([zero, identity], axis=-1), starts=starts)
    span = algebra.Span(summary[..., columns:], summary[..., :columns])
    memory = algebra.entering(span, axis, None if state is None else state.astype(wide))
    return jnp.where(begins[:, None, None, None], 0.0, memory)


def kernel_routed(query, key, value, g, beta, *, floor: float | None = None,
                  write_bound: float | None = None) -> bool:
    """Whether the delta rule's selector gives a call over these operands a kernel, its forward's
    (`port_kernel`, unreported: the call reports its own), under the layer's declared `floor` and
    `write_bound` (`chunk_gated_delta_rule`'s): what a layer rounds its operands for, the projections' dtype
    for a kernel, as fla's layer hands them over, the wide one for the portable math."""
    bound = recurrence.Hooks(write_bound=write_bound).inverse_bound
    return port_kernel(query, key, value, g, beta, floor=floor, bound=bound, report=False) != 'xla'


def fla_rule(query, key, value, g, beta, state=None, *, starts=None, max_segments: int | None = None,
             scale: float | None = None, wide: bool = False, firsts=None):
    """The port's chunked rule for any call the portable math takes:
    `[B, S, Hk, Dk]` queries and keys, `[B, S, Hv, Dv]` values, the log decay
    `g` `[B, S, Hv]` per head (`fla_gdn.chunk_gated_delta_rule`) or `[B, S,
    Hv, Dk]` per key dimension (Kimi's delta attention, `fla_kda.chunk_kda`),
    `beta` `[B, S, Hv]`, the memory `[B, Hv, Dk, Dv]` or None, and packed
    documents' first tokens `starts` `[B, S]`, at most `max_segments` a row
    (the row length where None), the queries scaled by `scale` in the
    kernels (`Dk^-1/2` where None), the output in the operands' common dtype,
    or fp32 where `wide`, for an epilogue to read it unrounded. Each document
    runs as a sequence of its own (fla's varlen `cu_seqlens`), its chunks
    from its first token and its memory from zero, as it would alone; the
    row's first takes `state`, unless its first token begins a document of
    its own. The kernels' grids are sized by that bound, a chunk more for
    each document and the row's padding; the memories they keep are not
    (`chunked_launch.Sequences`).

    A known defect (ledger row 1328, its measured ratios and reproduction):
    fla's arithmetic, which the port is, holds the recurrence less exactly
    than the portable math, check 1 against the token-by-token reference,
    under saturated write gates (5.8 to 5.9 times the reference's error from
    float64, every family), aligned keys under a decay a key dimension
    (5.4), documents resetting on chunk edges (1.6, every family) and every
    token at the decay's bound (a row 8e-7 from float64 over its size, a
    decay a head and a key dimension). Its error grows with the chunk's
    length, its prefix difference `exp2(gc_s - gc_t)` uncompensated.

    The operands go to their common dtype, beta with them, as fla's layer
    holds them. On a mesh the rule runs per shard (`manual_map`): rows where
    the rules split them, and value heads with the key heads they read where
    the head split divides the key heads, the time axis whole; a split that
    does not divide them keeps the heads whole, so no key's gradient is
    summed across shards after rounding. Each shard's call is shaped to the
    kernels' blocks (`_blocked_rule`) and cut back. Off a GPU its kernels
    interpret, as the CPU suite runs them.
    """
    dtype = jnp.result_type(query, key, value)
    query, key, value, beta = (x.astype(dtype) for x in (query, key, value, beta))
    g = g.astype(jnp.float32)
    B, _, Hv, Dv = value.shape
    state = (jnp.zeros((B, Hv, key.shape[-1], Dv), jnp.float32) if state is None
             else state.astype(jnp.float32))
    documents = None if starts is None else query.shape[1] if max_segments is None else max_segments
    starts = jnp.zeros(query.shape[:2], bool) if starts is None else jnp.asarray(starts, bool)
    if documents is not None:
        # The kernels start each row from its memory; a first token that begins a document drops it.
        state = jnp.where(starts[:, 0, None, None, None], 0.0, state)
    four, three, memory, decays, resets = _shard_specs(value.shape, key.shape[2], g.ndim)
    if firsts is None:
        return manual_map(functools.partial(_blocked_rule, documents=documents, scale=scale, wide=wide),
                          (four, four, four, decays, three, memory, resets),
                          (four, memory))(query, key, value, g, beta, state, starts)
    return manual_map(functools.partial(_blocked_rule, documents=documents, scale=scale, wide=wide),
                      (four, four, four, decays, three, memory, resets, resets),
                      (four, memory))(query, key, value, g, beta, state, starts, firsts)


def _shard_specs(shape: tuple[int, ...], key_heads: int, decay_ndim: int) -> tuple:
    """A delta rule's per-shard placement (`manual_map`) over values of `shape` reading `key_heads`: the
    `[B, T, H, D]` operands', the write gate's `[B, T, H]`, the memory's `[B, H, Dk, Dv]`, the decay's (a
    head's or a key dimension's), and the documents' starts' `[B, T]`: rows where the rules split them, and
    value heads with the key heads they read where the head split divides the key heads (`grouped_heads`),
    the time axis whole."""
    rows, heads = grouped_heads(shape, key_heads)
    P = jax.sharding.PartitionSpec
    four = P(rows, None, heads, None)
    return four, P(rows, None, heads), P(rows, heads, None, None), P(*four[:decay_ndim]), P(rows, None)


def template_rule(query, key, value, g, beta, state=None, starts=None, firsts=None, skip=None, **options):
    """The recurrence template's op (`templates.recurrence.template.MOSAIC`, its `mosaic_rule`) on each
    shard (`manual_map`), placed as `fla_rule`: its tiles derived from each shard's shapes, its forward and
    its gradient on each device's own rows. Called on the whole batch, every device of a 2xH100 FSDP run ran
    its launches on the global batch, layer 0's solve over 4 rows for its 2, beside 40 GB of gathers a
    microbatch (row 1897). `options` are the op's (`max_segments`, `scale`, `delta`, `wide`, `floor`,
    `bound`, `rounded`); whether the block's remat rung recomputes the delta rule's memory is resolved as
    the selectors' binds resolve it (`recomputed_memory`)."""
    from .templates.recurrence import template

    four, three, memory, decays, resets = _shard_specs(value.shape, key.shape[2], g.ndim)
    optional = [(state, memory), (starts, resets), (firsts, resets),
                (skip, jax.sharding.PartitionSpec(four[2]))]
    operands = (query, key, value, g, beta, *(x for x, _ in optional if x is not None))
    specs = (four, four, four, decays, three, *(spec for x, spec in optional if x is not None))

    def local(query, key, value, g, beta, *given):
        given = iter(given)
        state, starts, firsts, skip = (None if x is None else next(given) for x, _ in optional)
        return template.MOSAIC(query, key, value, g, beta, state, starts, firsts=firsts, skip=skip,
                               recomputed=recomputed_memory(), **options)

    return manual_map(local, specs, (four, memory))(*operands)


RECOMPUTED: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar("recomputed", default=())
"""The residuals a decoder block's remat rung has the layers that compute them recompute, the block
keeping every other (`dew.nn.backbones.decoder_block.RematPolicy.recompute`): a delta rule traced
inside recomputes its memory in its backward where this names 'delta_memory' (`_blocked_rule`),
and keeps it otherwise. A block sets it as it traces its layers (`recomputing`)."""


def recomputed_memory() -> bool:
    """Whether the layers traced here recompute the delta rule's memory in their backward (`RECOMPUTED`): the
    template's call takes it as an argument (`template.mosaic_rule`'s `recomputed`), so the selectors' binds
    (`port_kernel`, `scan_kernel`) and the run's (`template_rule`) read it alike, and a launch a compile
    refused under the rung is the one the route judges (row 4133)."""
    return "delta_memory" in RECOMPUTED.get()


@contextlib.contextmanager
def recomputing(names: tuple[str, ...]):
    """Trace the layers inside with `names` recomputed by the layers that compute them (`RECOMPUTED`)."""
    token = RECOMPUTED.set(names)
    try:
        yield
    finally:
        RECOMPUTED.reset(token)


def _blocked_rule(query, key, value, g, beta, state, starts, firsts=None, *, documents: int | None,
                  scale: float | None = None, wide: bool = False):
    """The port's rule on one shard, shaped to its kernels' blocks: values
    wider than the port's widest split into more value heads of their key
    head, each key head's value heads as many as it serves (dqkwg's block
    holds them, loaded a head at a time, so no count need be a power of
    two); keys and values padded with zero columns to 64, 128 or 256,
    widths every block of the kernels' configs divides, a decay per key
    dimension with them. None of them changes a real output or gradient:
    every padded product is an exact zero. The outputs are cut back. Where
    `documents` bounds a row's packed documents, each begun at `starts`, each
    is a sequence of its own (fla's varlen `cu_seqlens`). The decay's shape picks
    the port: per head fla's gated delta rule, per key dimension its Kimi
    delta attention. It keeps its memory for the backward
    but where the block's remat rung recomputes it (`RECOMPUTED`)."""
    B, S, _, Dk = key.shape
    Hv, Dv = value.shape[2:]
    scale = Dk ** -0.5 if scale is None else scale
    split = -(-Dv // fla_gdn.WIDEST)
    width = -(-Dv // split)
    if split > 1:
        value = _padded(value, 3, split * width).reshape(B, S, Hv * split, width)
        g, beta = (jnp.repeat(x, split, axis=2) for x in (g, beta))
        state = jnp.moveaxis(_padded(state, 3, split * width).reshape(B, Hv, Dk, split, width), 3, 2)
        state = state.reshape(B, Hv * split, Dk, width)
    blocks = (max(64, 1 << (Dk - 1).bit_length()), max(64, 1 << (value.shape[3] - 1).bit_length()))
    query, key = (_padded(x, 3, blocks[0]) for x in (query, key))
    per_key = g.ndim == 4
    g = _padded(g, 3, blocks[0]) if per_key else g
    value = _padded(value, 3, blocks[1])
    state = _padded(_padded(state, 2, blocks[0]), 3, blocks[1])
    # A row's padding after its documents runs as one more sequence of its own.
    spans = None if documents is None else cu_seqlens(starts, B * (documents + 1), firsts)
    port = fla_kda.chunk_kda if per_key else fla_gdn.chunk_gated_delta_rule
    out, final = port(query, key, value, g, beta, state, spans, scale, CHUNK_SIZE,
                      not generation.triton_runs(), not recomputed_memory(), wide)
    out = out[..., :width].reshape(B, S, Hv, split * width)[..., :Dv]
    final = final[:, :, :Dk, :width]
    final = jnp.moveaxis(final.reshape(B, Hv, split, Dk, width), 2, 3).reshape(B, Hv, Dk, split * width)
    return out, final[..., :Dv]


def _padded(x, axis: int, size: int):
    """`x` padded with zeros on the end of `axis` to `size`."""
    if x.shape[axis] == size:
        return x
    return jnp.pad(x, [(0, size - x.shape[axis]) if i == axis else (0, 0) for i in range(x.ndim)])


def recurrent_delta_rule(query, key, value, g, beta, state=None, *, starts=None):
    """One token at a time, the decode path, in at least fp32 as a lax.scan over
    the time axis so the state rides the scan the way it rides the decode cache,
    returned as the chunked rule returns them (`xla_mainloop`).

    `g` is the log decay per key dimension, `[B, S, H, Dk]`, which is
    `recurrent_kimi_delta_attention` (modeling_glm5_next.py:428-478). The
    gated delta rule decays per head, one decay over every key dimension.
    A token `starts` `[B, S]` marks begins a packed document from a zero memory. Each token is the decode
    step's update (`recurrence.update`), the scan only its orchestration.
    """
    dtype = jnp.result_type(query, key, value)
    work = at_least_fp32(dtype)
    query, key, value, g, beta = (
        x.astype(work) for x in (query, key, value, g, beta))
    query = query * (key.shape[-1] ** -0.5)

    def one_token(s, step):
        s = jnp.where(step['start'][:, None, None, None], 0.0, s)
        out, s = recurrence.update(s, step['q'], step['k'], step['v'], jnp.exp(step['g']),
                                   step['beta'][..., None], delta=True)
        return s, out
    # The scan stacks along the first axis, so the operands go time-major.
    if state is None:
        state = jnp.zeros((query.shape[0], query.shape[-2], key.shape[-1],
                           value.shape[-1]), work)
    starts = jnp.zeros(query.shape[:2], bool) if starts is None else jnp.asarray(starts, bool)
    state, out = jax.lax.scan(
        one_token, state.astype(work),
        {name: jnp.moveaxis(x, 1, 0) for name, x in
         (('q', query), ('k', key), ('v', value), ('g', g), ('beta', beta), ('start', starts))})
    return jnp.moveaxis(out, 0, 1).astype(dtype), state


def _rated_scan(hooks: recurrence.Hooks, query, key, value, g, beta, given: dict, state, starts, firsts,
                chunk_size: int, work, dtype, *, max_segments: int | None = None):
    """`chunk` without the delta rule over a rated decay (Mamba2's SSD: `x` the values, `B` the keys,
    `C` the queries, `dt` the write gate and `A` the rate) on the kernel the scan's selector gives each
    product (`SCAN`, `scan_kernel`): the recurrence template; the port of mamba_ssm's kernels, which adds
    the skip itself and takes the rated decay's declared domain, a contraction (`recurrence.Hooks`), as its
    kernels clamp each pairwise decay to at most one; a TPU's Pallas scan where its tiling covers the
    geometry; or the portable math, the skip added after. `g` is the log decay the rate and the write gate
    `beta` form (`recurrence.log_gates`), which the port's kernels form themselves. The port and the TPU's
    scan take head and state widths padded with zero columns to powers of two (`_scan_widths`), which change
    no real output or gradient, the outputs cut back: Nemotron-H 4B's 80-wide heads took the portable math,
    6.89 ms a forward against mamba_ssm's 0.89 (SE-ssd-mamba-h100-0428aad7b). Returns the output in `work`,
    rounded to no narrower than `dtype`, the caller's (unrounded where it asks it wide, Mamba-2's mixer for
    the earlier shards' state it adds), and the memory `[B, Hv, Dk, Dv]`."""
    rate = given['rate'].astype(work)
    skip = given['skip'].astype(work) if hooks.skip else jnp.zeros(rate.shape, work)
    # Under the fast forms the skip in the template's own output and walk (`mainloop._output`, `_reverse`),
    # as mamba_ssm's kernels add it, the output in the operands' dtype where the caller asks it so and no
    # epilogue reads it after (`hooks.output`'s norm), else wide: added after, its fp32 output's skip and cast
    # took 0.045 ms past an H100's launches, and its gradient's 0.094 (ssd-rounded-h100-fdab2424c,
    # window-h100-c70aaa8-b8109ee). Where the template holds no 16-bit values ungated
    # (`mainloop.Call.gated_keys`: the accurate forms, or an fp32 call's two parts under the fast forms),
    # the skip is added after. Decided once: the selector judges the call the run makes (its skip's reverse
    # tile).
    fast = fast_rounding()
    operand = jnp.dtype(jnp.result_type(query, key, value))
    fused = fast and hooks.skip and operand.itemsize == 2
    narrow = fused and hooks.output is None and jnp.dtype(dtype) == operand

    def product_kernel(product: str) -> str:
        return scan_kernel(query, key, value, g, beta, state, starts, chunk_size=chunk_size,
                           max_segments=max_segments, product=product, skip=skip if fused else None,
                           wide=not narrow)

    def run(kernel, query, key, value, g, beta, rate, skip, state, starts, firsts):
        if kernel == 'mosaic_gpu':
            out, final = template_rule(query, key, value, g, beta, state, starts, firsts,
                                       skip=skip if fused else None, max_segments=max_segments, scale=1.0,
                                       delta=False, wide=not narrow, rounded=fast)
            out = out.astype(work)
            return (out if fused else recurrence.skipped(hooks, out, value, {'skip': skip}, work)), final
        operand = jnp.result_type(query, key, value)
        x, B, C = (jnp.asarray(t, operand) for t in (value, key, query))
        batch, _, heads, head_dim = x.shape
        state_size = B.shape[-1]
        if kernel in ('pallas', 'tpu'):
            # The SSD's memory is `[P, N]`, values by keys: the template's `[N, P]` transposed.
            carried = (rows_like(jnp.zeros((batch, heads, head_dim, state_size), work), x) if state is None
                       else jnp.swapaxes(state, -1, -2).astype(work))
            columns, rows = _scan_widths(head_dim), _scan_widths(state_size)
            wide_x, wide_B, wide_C = _padded(x, 3, columns), _padded(B, 3, rows), _padded(C, 3, rows)
            wide_state = _padded(_padded(carried, 2, columns), 3, rows)
            if kernel == 'pallas':
                out, final = _ssd_ported(wide_x, beta, rate, wide_B, wide_C, skip, wide_state, chunk_size,
                                         starts)
                out, final = out.astype(work), final.astype(work)
            else:
                out, final = _ssd_tpu(wide_x, beta, rate, wide_B, wide_C, skip, wide_state, chunk_size,
                                      starts, scan_case(x, B, chunk_size, padded=True).tpu, work)
            return out[..., :head_dim], jnp.swapaxes(final[..., :head_dim, :state_size], -1, -2)
        out, final = xla_mainloop(Tiles(chunk_size), query, key, value, g, beta, state, starts, scale=1.0,
                                  delta=False, dtype=work)
        return recurrence.skipped(hooks, out, value, {'skip': skip}, work), final

    return _per_product(product_kernel, run, query, key, value, g, beta, rate, skip, state, starts, firsts)


def _scan_widths(width: int) -> int:
    """A scan kernel's width for a head or state `width` (`_rated_scan`): the power of two at or past it, at
    least 16, the smallest block Triton's dot takes (`SSD_SUPPORTS`)."""
    return max(16, 1 << (width - 1).bit_length())


def ssd_operands(x, dt, A, B, C, chunk_size: int, starts=None):
    """A layer's SSD operands as the chunk scan takes them (`dew.nn.kernels.ssd._chunk_scan`):
    `(x dt, B, C, A dt)`, the groups of `B` and `C` expanded over the heads, the sequence padded to whole
    chunks of `chunk_size`, `x` scaled by its step and the log decay `A dt` a step, `RESET_DECAY` at each
    packed document's first token (`starts`)."""
    heads, length = x.shape[2], x.shape[1]
    # The `H // G` heads of one group read the same B and C, the reference's `repeat_interleave`.
    B, C = repeat_kv_heads(B, heads), repeat_kv_heads(C, heads)
    pad = (chunk_size - length % chunk_size) % chunk_size
    x, B, C = (jnp.pad(t, ((0, 0), (0, pad), (0, 0), (0, 0))) for t in (x, B, C))
    dt = jnp.pad(dt, ((0, 0), (0, pad), (0, 0)))
    a = A[None, None, :] * dt
    if starts is not None:
        a = jnp.where(jnp.pad(starts, ((0, 0), (0, pad)))[..., None], RESET_DECAY, a)
    return x * dt[..., None], B, C, a


def _ssd_tpu(x, dt, A, B, C, D, carried, chunk_size: int, starts, platform, work):
    """The rated scan on a TPU's Pallas kernel (`dew.nn.kernels.ssd`), on its operands in `work`
    (`ssd_operands`), per shard of rows and heads (`manual_map`). The skip `D x` is added after. Returns
    the output and the memory `[B, H, P, N]`."""
    length = x.shape[1]
    x, B, C = (jnp.asarray(t, work) for t in (x, B, C))
    skip = D[None, None, :, None] * x
    scaled, B, C, a = ssd_operands(x, dt, A, B, C, chunk_size, starts)

    def kernel(x, B, C, a, carried):
        return _chunk_scan(x, B, C, a, carried, chunk_size, platform)

    # The groups are expanded, so any split of the heads divides them.
    rows, split = grouped_heads(scaled.shape, scaled.shape[2])
    four, held = P(rows, None, split, None), P(rows, split, None, None)
    scanned, final = manual_map(kernel, (four, four, four, P(rows, None, split), held),
                                (four, held))(scaled, B, C, a, carried)
    return scanned[:, :length] + skip, final


@dataclasses.dataclass(frozen=True)
class ScanCase:
    """A rated decay's scan as the scan selector reads it (`SCAN`): whether a caller's `shard_map` that checks
    varying axes holds operands that vary (`varying`), the operands' `dtype`, the port's head width, state
    width and chunk (`widths`, padded, `_scan_widths`), the platform the TPU's Pallas scan builds for, None
    where it does not take the scan (`ssd_kernel_platform`), and as the delta rule's case reads them
    (`PortCase`) the recurrence template's implementations bound to the call (`call`), the device's
    instruction family (`family`) and the product (`product`)."""

    varying: bool
    dtype: np.dtype
    widths: tuple[int, int, int]
    tpu: str | None
    call: base.Call | None = None
    family: str | None = None
    product: str = 'forward'

    @property
    def cell(self) -> tuple[str | None, str]:
        """The call's cell, its device's instruction family and its product (`SCAN_CELLS`)."""
        return self.family, self.product


SSD_SUPPORTS: tuple[Requirement[ScanCase], ...] = (
    Requirement('varying', lambda case: not case.varying, _VARYING, row='ledger 822'),
    Requirement('float64', lambda case: case.dtype != jnp.float64,
                "float64 operands, whose scan is the reference math itself", row=INHERENT),
    Requirement('widths', lambda case: all(width >= 16 and width & (width - 1) == 0 for width in case.widths),
                lambda case: "head width {}, state {} and chunk {}; the port takes powers of two of at least "
                             "16".format(*case.widths), row=INHERENT),
    Requirement('triton', lambda case: triton_runs(),
                "no GPU Dew's Triton kernels run on (sm80 or later): off one the port's kernels interpret, "
                "as the CPU suite runs them where a call names the port", row=INHERENT, kind='measured'),
)
"""What mamba_ssm's kernels (`mamba_ssd.ssd_scan`) need of a scan: no caller's `shard_map` that checks
varying axes over operands that vary,
where a Pallas call, which states no varying type for its results, cannot run; no float64 operands,
whose scan is the reference math itself; head widths, state widths and chunks that are powers of
two of at least 16, the smallest block Triton's dot takes (a call's head and state widths padded to them,
`_rated_scan`); and a GPU Dew's Triton kernels run on, any of them."""

TPU_SCAN_SUPPORTS: tuple[Requirement[ScanCase], ...] = (
    Requirement('geometry', lambda case: case.tpu is not None,
                lambda case: "the TPU's Pallas scan takes a float32 scan on a TPU over power-of-two widths "
                             "and a chunk its program holds, not head width {}, state {} and chunk {}"
                             .format(*case.widths), row=INHERENT),
)
"""What the TPU's Pallas scan (`dew.nn.kernels.ssd`) needs: its tiling over the geometry
(`ssd_kernel_runs`)."""


def scan_case(x, B, chunk_size: int, *, padded: bool = False, call: base.Call | None = None,
              product: str = 'forward') -> ScanCase:
    """The `ScanCase` of a scan over values `x` `[B, S, H, P]` and keys `B` `[B, S, G, N]` in
    `chunk_size` chunks, their widths as the port and the TPU's scan take them where `padded`
    (`_scan_widths`, `_rated_scan`); on a TPU, the platform its Pallas scan builds for
    (`ssd_kernel_platform`); the template's `call` and the `product`."""
    widths = (x.shape[-1], B.shape[-1], chunk_size)
    if padded:
        widths = (_scan_widths(widths[0]), _scan_widths(widths[1]), chunk_size)
    family = generation.family(generation.device_generation()) if generation.backend() == 'gpu' else None
    return ScanCase(varying=bool(jax.typeof(x).mat.varying),
                    dtype=jnp.dtype(x.dtype), widths=widths,
                    tpu=(ssd_kernel_platform(chunk_size, widths[0], widths[1], dtype=at_least_fp32(x.dtype))
                         if generation.backend() == 'tpu' else None),
                    call=call, family=family, product=product)


def fast_rounding() -> bool:
    """Whether a call on the template takes the run's fast forms' numbers (`rounded`): the SSD's mamba_ssm's
    16-bit ones, an fp32 call's products TF32 (JAX's default precision for an fp32 product, and fla's), under
    the run's fast forms (`generation.numerics`), resolved at its route, so its launches and their caches key
    on it, never on the run's numerics read inside them."""
    return generation.numerics().forms == 'fast'


def chunk_rounding(delta: bool, dtype) -> bool:
    """Whether a call of `dtype` on the delta rule's route (`chunk`, `port_kernel`) takes the run's fast
    forms' numbers (`fast_rounding`): the delta rule's alone, whose bar is fla's (TF32 products in an fp32
    call). A form without the delta rule has no port, its bar the portable math at full fp32: an fp32 call's
    two bf16 parts put its training step 30 to 69 times the portable math's distance from float64 on every
    leaf, interpreted and on a GH200 alike (row 184's write gate cell, jsf-cellcpu, M-JS-9), so it takes
    three; a 16-bit call's rounded numbers are mamba_ssm's, the SSD's, which its scan's route alone asks
    (`_rated_scan`). Rounded there, a write gate without the delta rule passed its values between launches
    in bf16, its final memory 2.1e4 times the portable math's distance from float64 under aligned keys
    (interpreted at 2b595ff93, 9593 to 46978 times on an H100, gpuflow-a-abfff38d9)."""
    del dtype  # The run's declaration rounds a call by its dtype (`fast_rounding`, row 4673).
    return fast_rounding() and delta


def scan_kernel(query, key, value, g, beta, state=None, starts=None, *, chunk_size: int,
                max_segments: int | None = None, product: str = 'forward', skip=None,
                wide: bool = True) -> str:
    """The kernel the scan's selector (`SCAN`) gives a rated scan over these operands (`_rated_scan`'s:
    `value` the SSD's `x`, `key` its `B`, `query` its `C`) for `product` ('forward', 'training'), the
    template's implementations bound to it as the run calls it (its fused `skip`, None where added after,
    and its output `wide` or in the operands' dtype), on the target (`generation`'s, an assumed one's within
    `assumed`), reported as its route."""
    from .templates.recurrence import template

    call = base.call((template.MOSAIC,), query, key, value, g, beta, state, starts,
                     max_segments=max_segments, scale=1.0, delta=False, wide=wide, rounded=fast_rounding(),
                     skip=skip, recomputed=recomputed_memory())
    case = scan_case(value, key, chunk_size, padded=True, call=call, product=product)
    return select(SCAN, case).kernel


SCAN_CELLS: dict[tuple[str | None, str], tuple[str, str]] = {
    ('wgmma', 'training'): (
        'pallas', "on an H100 80GB HBM3, the port's forward and gradient 1.592 ms against the template's "
                  "1.877 on Nemotron-H 4B's SSD and 0.660 against 0.820 on 30B's, its query/key launch "
                  "0.385 and 0.160 of them (window-h100-c70aaa8-b8109ee)"),
    ('mma.sync', 'training'): (
        'pallas', "on an A100 SXM4 40GB, the port's forward and gradient 3.954 ms against the template's "
                  "4.217 on Nemotron-H 4B's SSD and 1.584 against 2.200 on 30B's (ssd-train-a100-fdab2424c)"),
}
"""The cells, each an instruction family and a product (`ScanCase.cell`), where a receipt timing the same
product (`chunk(SSD, ...)` with its skip, output width, chunk and geometry, row 1964) shows Dew's port of
mamba_ssm's kernels (`mamba_ssd.ssd_scan`; "the port" below) faster than the recurrence template, each its
old route's kernel and that receipt; every time below sets the template against that port, not against
mamba_ssm's own kernels (tools/benchmark_recurrence.py's `mamba_ssm` arm): the
scan's selector runs the port on them where it takes the call (`Op.declared`); the list shrinks as the
template wins its cells. An A100's forward is the template's: 1.418 ms against the port's 1.506 on Nemotron-H
4B's and 0.510 against 0.541 on 30B's (ssd-rounded-a100-fdab2424c), and warpgroup MMA's: on an H100 80GB HBM3
0.211 ms against 0.241 on 30B's and 0.522 against 0.617 on 4B's, on a GH200 480GB 0.207 against 0.225 and
0.509 against 0.595 (FS-RECROUTES-h100 and -gh200 at 26d3fdf5df: cold, exclusive, two rounds alternating the
arms' order). The template trains the rated scan under
the fast forms (`mainloop_backward.fast_backward_form`); under the accurate ones its training is the port's by
the template's refusal, not a cell."""

SCAN: Op[ScanCase] = Op('ssd', (Kernel('mosaic_gpu', (*SSD_SUPPORTS[:3], *TEMPLATE_SUPPORTS)),
                               Kernel('pallas', SSD_SUPPORTS), Kernel('tpu', TPU_SCAN_SUPPORTS),
                               Kernel('xla', (), generic=True)), declared=SCAN_CELLS)
"""The rated scan's selector: on a declared port cell (`SCAN_CELLS`) the port where it takes the call;
otherwise the first of its kernels that takes the call, by what each takes (no table of GPUs): the
recurrence template at its derived tiles (`TEMPLATE_SUPPORTS`, under the port's trace and placement), the
port of mamba_ssm's kernels on any GPU Dew's Triton kernels run on, the TPU's Pallas scan where its tiling
covers the geometry, else the portable math (a GPU no row named took it, 103 ms a training step of
Nemotron-H 30B's where the port took 2.58, template review 2026-10-26 P1-1)."""


def _ssd_ported(x, dt, A, B, C, D, state, chunk_size: int, starts):
    """The scan on mamba_ssm's kernels, each token's document counted from
    `starts` (mamba_ssm's `seq_idx`). On a mesh it runs per shard
    (`manual_map`): rows where the rules split them, and heads with their
    groups where the head split divides the groups, the time axis whole;
    off a GPU its kernels interpret, as the CPU suite runs them."""
    interpret = not triton_runs()
    seq = None if starts is None else jnp.cumsum(starts, axis=1, dtype=jnp.int32)

    def scan(x, dt, A, B, C, D, state, *seq):
        return mamba_ssd.ssd_scan(x, dt, A, B, C, D, state, *(seq or (None,)), chunk_size, interpret)

    operands = (x, dt, A, B, C, D, state, *([] if seq is None else [seq]))
    rows, heads = grouped_heads(x.shape, B.shape[2])
    four, held = P(rows, None, heads, None), P(rows, heads, None, None)
    specs = (four, P(rows, None, heads), P(heads), four, four, P(heads), held, P(rows, None))
    return manual_map(scan, specs[:len(operands)], (four, held))(*operands)


@dataclasses.dataclass(frozen=True)
class SelectiveCase:
    """A selective scan (Mamba's, `dew.nn.mixers.mamba.selective_scan`) as its selector reads it
    (`SELECTIVE`): whether a shard_map that checks varying axes holds its operands, its operands' dtype and
    its states' rows' narrower one (B's or C's, which stream each in its own: the narrower's rows are the
    shorter), its tokens, channels and states, whether the run's forms are the fast ones (`fast_rounding`)
    and its device's instruction family (None off a GPU); the forward and the training step route alike."""

    varying: bool
    dtype: np.dtype
    state_dtype: np.dtype
    length: int
    channels: int
    states: int
    fast: bool
    family: str | None = None


def _selective():
    """The template's selective scan (`templates.recurrence.selective`), its least chunk (`CHUNK`) read where
    a requirement asks."""
    from .templates.recurrence import selective

    return selective


COPY_BYTES = 16
"""The bytes a bulk copy's rows come in a whole number of (TMA's, cp.async's widest)."""

SELECTIVE_STATES = 16
"""The most states a channel of the template's selective scan carries, each three times in a thread's
registers (its rates, cotangents and the rates' gradients): its backward at Mamba's 16, 125 registers a
thread on an H100 (sel-ptxas2-sm90 at e3b43f7d6)."""

SELECTIVE_SUPPORTS: tuple[Requirement[SelectiveCase], ...] = (
    Requirement('varying', lambda case: not case.varying, _VARYING, row='ledger 822'),
    Requirement('gpu', lambda case: case.family is not None,
                "no GPU of an instruction family the template's bundles cover: off one the template "
                "interprets, as the CPU suite runs it where a call names it", row=INHERENT, kind='measured'),
    Requirement('dtype', lambda case: case.dtype in (jnp.bfloat16, jnp.float16, jnp.float32),
                lambda case: f"{case.dtype} operands; the template takes 16-bit and fp32 ones", row=INHERENT),
    Requirement('channels', lambda case: case.channels % block_scan.THREADS == 0,
                lambda case: f"{case.channels} channels, no whole number of the template's blocks of "
                             f"{block_scan.THREADS}", row='ledger 1300'),
    Requirement('states', lambda case: case.states <= SELECTIVE_STATES,
                lambda case: f"{case.states} states, past the {SELECTIVE_STATES} a thread's registers hold "
                             "three times", row='ledger 1300'),
    Requirement('state rows', lambda case: case.states * case.state_dtype.itemsize % COPY_BYTES == 0,
                lambda case: f"{case.states} states of {case.state_dtype}, a B or C row of no whole number "
                             f"of a bulk copy's {COPY_BYTES} bytes, which its streams copy unpadded",
                row='ledger 1300'),
    Requirement('tokens', lambda case: case.length >= _selective().CHUNK,
                lambda case: f"{case.length} tokens, fewer than a chunk of {_selective().CHUNK}: a decode "
                             "step's scan, the portable math's own steps", row=INHERENT, kind='handoff'),
    Requirement('forms', lambda case: case.fast,
                "the run's accurate forms, whose sums the portable math compensates (row 1698); the template "
                "sums at mamba_ssm's rounding, the fast forms' bar", row='ledger 2968', kind='measured'),
    Requirement('qualified', lambda case: case.family == 'wgmma',
                "the template's selective scan held to recorded mamba_ssm's distance from float64 on an H100 "
                "alone (selective-recorded2-h100-497ea24e7), not on this GPU", row='ledger 2968',
                kind='measured'),
)
"""What the template's selective scan (`templates.recurrence.selective`) needs of a call: no caller's
`shard_map` that checks varying axes, a GPU of a family its bundles cover, 16-bit or fp32 operands, whole
blocks of channels, at most `SELECTIVE_STATES` states, a chunk's tokens at least (a decode step hands off
to the portable math), the run's fast forms, and its
qualification against recorded mamba_ssm on the GPU's family: on an H100 the template is as exact as
mamba_ssm in every recorded case (tests/test_gpu_recurrence.py)."""

SELECTIVE: Op[SelectiveCase] = Op('selective_scan', (Kernel('mosaic_gpu', SELECTIVE_SUPPORTS),
                                                     Kernel('xla', (), generic=True)))
"""The selective scan's selector: the recurrence template's scan (`templates.recurrence.selective`) where
it takes the call, else the portable math. On an A100 its training step, B's and C's gradients added
atomically, took 11.3 ms at 8x2048x5120 and 9.22 at 4x2048x8192 (W3-a100-3-selective-ab-a100-8ca2795113, at
44bec4713f, whose backward base's is again)."""


def selective_kernel(x, A, B, C) -> str:
    """The kernel the selective scan's selector (`SELECTIVE`) gives a scan over `x` `[B, S, D]` with rates
    `A` `[D, N]` and states' rows `B`, `C` `[B, S, N]`, on the target, reported as its route."""
    from .templates import bundles

    # A GPU's family where a bundle claims it (consumer Blackwell's, mma.sync fed by TMA, has one; tcgen05
    # has none in a route yet).
    on_gpu = generation.backend() == 'gpu'
    bundled = on_gpu and bundles.of(generation.device_generation(), x.dtype) is not None
    family = generation.family(generation.device_generation()) if bundled else None
    case = SelectiveCase(varying=bool(jax.typeof(x).mat.varying), dtype=jnp.dtype(x.dtype),
                         state_dtype=min(jnp.dtype(jnp.result_type(B)), jnp.dtype(jnp.result_type(C)),
                                         key=lambda dtype: dtype.itemsize),
                         length=x.shape[1], channels=x.shape[-1], states=A.shape[-1], fast=fast_rounding(),
                         family=family)
    return select(SELECTIVE, case).kernel


def selective_scan(x, dt, A, B, C, D, state, starts, *, portable: Callable):
    """Mamba's selective scan (`dew.nn.mixers.mamba.selective_scan`'s operands, `state` `[B, D, N]` or None)
    on the kernel its selector gives each product (`SELECTIVE`, `_per_product`): the recurrence template's
    scan, interpreted off a GPU (where a call names it), per shard of rows on a mesh (`manual_map`: B and C
    serve every channel, so the channels stay whole); or `portable(x, dt, A, B, C, D, state, starts)`,
    the portable math."""
    from .templates.recurrence import selective

    def template(x, dt, A, B, C, D, state, *starts):
        return selective.SELECTIVE(x, dt, A, B, C, D, state, *(starts or (None,)),
                                   interpret=generation.backend() != 'gpu')

    def run(kernel, x, dt, A, B, C, D, state):
        if kernel != 'mosaic_gpu':
            return portable(x, dt, A, B, C, D, state, starts)
        rows, _ = grouped_heads((*x.shape[:2], 1, x.shape[-1]), 1)
        three = P(rows, None, None)
        held = state if state is not None else jnp.zeros((x.shape[0], x.shape[-1], A.shape[-1]), x.dtype)
        operands = (x, dt, A, B, C, D, held, *([] if starts is None else [starts]))
        specs = (three, three, P(None, None), three, three, P(None), three, P(rows, None))
        return manual_map(template, specs[:len(operands)], (three, three))(*operands)

    return _per_product(lambda product: selective_kernel(x, A, B, C), run, x, dt, A, B, C, D,
                        state)
