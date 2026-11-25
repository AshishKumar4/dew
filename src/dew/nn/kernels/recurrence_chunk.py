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

import dataclasses
import functools
import math
from collections.abc import Callable
from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

from dew import _cache
from dew.algebra import derive, scan
from dew.algebra.derive import FORMS_HELD
from dew.algebra.scan import RESET_DECAY, Reads
from dew.batch import document_firsts
from dew.mesh import mesh_axes
from dew.nn.attention_masks import repeat_kv_heads
from dew.nn.derivatives import KernelGlue, Protected, contracted, with_gradients_formed_at
from dew.nn.inputs import cu_seqlens
from dew.nn.kernels import recurrence_coupled
from dew.nn.nested import nested
from dew.nn.precision import (
    accurate_exp,
    asks_default_precision,
    at_least_fp32,
    exact_products,
    l2norm,
    rounds_products,
)
from dew.nn.residuals import recomputed_memory
from dew.nn.sharding import grouped_heads, manual_map, rows_like, rows_spec, shard_shapes, varying_like

from . import fla_gdn, fla_kda, generation, mamba_ssd, recurrence, recurrence_algebra as algebra
from .chunked_launch import Forked, zero_padded
from .generation import INHERENT, Choice, triton_runs
from .recurrence_forked import forked_adjoint, forked_replay, forked_steps, most_segments, taken_up
from .selection import Kernel, KernelRefused, Op, Refusal, Requirement, select, selected
from .ssd import _chunk_scan, ssd_kernel_platform
from .templates import base, report as launch_report, scan as block_scan

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
                           starts=None, max_segments: int | None = None,
                           write_bound: float | None = None, decay_bound: float | None = None,
                           unit_queries: bool = False, firsts=None, loads=None, stores=None, held: int = 0):
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
    which needs none. `write_bound` is the largest write gate the layer declares of keys it l2-normalized
    (`recurrence.Hooks.write_bound`), which the template's 16-bit solve reads (`mainloop._merged`), and
    `decay_bound` the least log decay a token the layer proves (`recurrence.Hooks.decay_bound`), which a
    decay a key dimension's template prefix reads, and `unit_queries` whether its queries are l2-normalized
    (`recurrence.Hooks.unit_queries`). `loads`, `stores` and `held` are a forked call's (`Chunk.loads`).
    """
    hooks = recurrence.Hooks(decay='vector' if g.ndim == 4 else 'scalar', scale=key.shape[-1] ** -0.5,
                             write_bound=write_bound, decay_bound=decay_bound, unit_queries=unit_queries)
    return chunk(hooks, Chunk(query, key, value, log_decay=g, beta=beta, state=state, starts=starts,
                              firsts=firsts, loads=loads, stores=stores, held=held),
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
    Under `Hooks.substeps` (DeltaProduct's) `key`, `value` and `beta` hold a token's `m` updates in turn,
    `[B, S m, ...]`, and every other operand its `S` tokens.
    Under Oja's rule (`recurrence.Hooks.erase` 'values') `log_decay` is `[B, S, Hv, Dv]`, a decay along the
    values.
    `erase_key` and `erase_read` `[B, S, Hk, Dk]` are an erase of its own's vectors (`recurrence.Hooks.erase`
    'untied' or 'additive'), `value_decay` `[B, S, Hv, Dv]` the log decay along the values
    (`Hooks.value_decay`); under Oja's rule (`Hooks.erase` 'values') `log_decay` is `[B, S, Hv, Dv]`, a
    decay along the values.

    A forked call (`dew.batch.Forks`, `dew.nn.linear.forked_scan`) gives `loads` and `stores` `[B, D]` int,
    each of a row's documents in the order of their first slots (`firsts`, else `starts`'s,
    `dew.batch.document_firsts`), and `held`, the most of a row's documents another enters: the walk keeps
    a table of memories, `state`'s `[B, S', Hv, Dk, Dv]` given ones first (a row's start, a cache), then
    `held` it stores, and a document enters table slot `loads[d]` (one-based; zero, a zero memory) and, where
    `stores[d]` is past zero, leaves its final memory in stored slot `stores[d]` (the table's `S' +
    stores[d]`-th), which only a later document of the row enters. A document storing holds a token. One
    walk over the row in order computes every document from the memory it enters, its parent's final among
    them: the forks' depth costs nothing (kernel-templates.md section 3). The final memory is the row's
    last, `[B, Hv, Dk, Dv]`. `entry_key` `[B, D, R, Hk, Dk]`, `entry_value` `[B, D, R, Hv, Dv]` and
    `entry_weight` `[B, D, Hv]`, where given, add `entry_weight sum_r entry_key_r entry_value_r^T` to the
    memory a document enters (Mamba-3's trapezoid: a document's first token writes its parent's last token
    again at its own weight, `dew.nn.mixers.mamba3.entered_mamba3`). `adds` `[B, D]`, where given, a second
    one-based table slot (zero, none) each document adds to the memory it enters, stored before it as a
    load's is: a looped pass's document enters its fork parent's final and its own final of the pass before
    (row 7662)."""
    query: jax.Array
    key: jax.Array
    value: jax.Array
    state: jax.Array | None = None
    starts: jax.Array | None = None
    firsts: jax.Array | None = None
    loads: jax.Array | None = None
    stores: jax.Array | None = None
    entry_key: jax.Array | None = None
    entry_value: jax.Array | None = None
    entry_weight: jax.Array | None = None
    adds: jax.Array | None = None
    held: int = dataclasses.field(default=0, metadata={'static': True})


def _erases_laid(hooks: recurrence.Hooks, query, key, value, g, beta, given: dict, starts, firsts):
    """A call erasing with vectors of its own (`recurrence.Hooks.erase` 'read', 'untied' or 'additive'): an
    additive one (RWKV7's) as it is, one token a token of the algebra's erase along its own vector reading the
    state the token enters (`_delta_forms`' `corrected_by` 'erased', `D h + a (c^T h)^T + beta k v^T`, reader
    `c = -b`); a read one so too, its keys' write corrected along its read (`corrected_by` 'after', `D h +
    beta k (v + c^T D h)^T`); an untied one (GDN-2's, Comba's) as two such a token: the decay and erase, keys
    `a`, values zero, gate `beta`, reader `c = -b`, then the write, keys `k`, values `v`, gate `beta`, no
    decay or correction, its query on the second, a packed document from its first. The erase vectors each key
    head's over its value heads, but a read erase's, each key head's. `(query, key, value, g, beta, erase,
    starts, firsts)`, `erase` the additive erase's `(a, c)`, the read's and the untied's `(None, c)`, the
    untied's laid `[B, 2S, ...]` (`_laid`)."""
    heads = value.shape[2]
    if hooks.erase == 'read':
        # Its reads and keys each key head's, as given (the template shares them over the key head's value
        # heads; the portable math repeats them, `_on_portable`), the keys in their dtype: a 16-bit call's a
        # 16-bit call on the template (`port_kernel`'s `reads`).
        return query, key, value, g, beta, (None, -given['erase_read'].astype(g.dtype)), starts, firsts
    b, a = (jnp.repeat(given[name], heads // given[name].shape[2], axis=2).astype(g.dtype)
            for name in ('erase_read', 'erase_key'))
    if hooks.erase == 'additive':
        return query, key, value, g, beta, (a, -b), starts, firsts
    group = heads // key.shape[2]
    keys = jnp.repeat(key, group, axis=2).astype(g.dtype)
    queries = jnp.repeat(query, group, axis=2)

    def laid(first, second):
        return jnp.stack([first, second], axis=2).reshape(first.shape[0], -1, *first.shape[2:])

    return (laid(jnp.zeros_like(queries), queries), laid(a, keys), laid(jnp.zeros_like(value), value),
            laid(g, jnp.zeros_like(g)), laid(beta, beta), (None, laid(-b, jnp.zeros_like(b))),
            None if starts is None else laid(starts, jnp.zeros_like(starts)),
            None if firsts is None else firsts * 2)


def _read_checked(hooks: recurrence.Hooks, operands: Chunk) -> dict:
    """`operands` as `hooks` read them (`recurrence.read`, the decode step's checks), a call of the conv's
    output, each key head serving a whole number of value heads, its decay of its kind's rank."""
    if hooks.conv is not None:
        raise ValueError("the chunked op takes the conv's output: its queries, keys and values")
    given = recurrence.read(hooks, operands, optional=('state',))
    key, value = given['key'], given['value']
    if value.shape[2] % key.shape[2]:
        raise ValueError(f"{value.shape[2]} value heads over {key.shape[2]} key heads: each key head serves "
                         "a whole number of value heads")
    for name, like in (('erase_key', key), ('erase_read', key), ('value_decay', value)):
        if name in given and given[name].shape != like.shape:
            raise ValueError(f"{name} is {like.shape}, as its side's operand, not {given[name].shape}")
    ranks = {'scalar': 3, 'vector': 4}
    if 'log_decay' in given and given['log_decay'].ndim != ranks[hooks.decay]:
        raise ValueError(f"a {hooks.decay} decay is {ranks[hooks.decay]}-dimensional, [B, S, Hv"
                         f"{', Dk' * (hooks.decay == 'vector')}], not {given['log_decay'].shape}")
    tokens = given['query'].shape[1]
    stepped = {name: given[name].shape[1] for name in ('key', 'value', 'beta') if name in given}
    held = {name: given[name].shape[1] for name in ('log_decay', 'z') if name in given}
    if operands.starts is not None:
        held['starts'] = operands.starts.shape[1]
    if any(length != tokens * hooks.substeps for length in stepped.values()) or any(
            length != tokens for length in held.values()):
        raise ValueError(f"{tokens} tokens of {hooks.substeps} updates each take keys, values and write "
                         f"gates of {tokens * hooks.substeps} and a decay of {tokens}, not {stepped | held}")
    if operands.loads is None:
        # A coupled momentum's state its pair `[B, Hv, 2, Dk, Dv]`, the memory then the momentum.
        rank, laid = (5, "[B, Hv, 2, Dk, Dv]") if hooks.coupled is not None else (4, "[B, Hv, Dk, Dv]")
        if operands.state is not None and operands.state.ndim != rank:
            raise ValueError(f"an unforked call's `state` is {rank}-dimensional {laid}, not "
                             f"{operands.state.shape} (a table [B, S', Hv, Dk, Dv] takes `loads`)")
        fields = {'stores': operands.stores, 'entry_key': operands.entry_key,
                  'entry_value': operands.entry_value, 'entry_weight': operands.entry_weight,
                  'adds': operands.adds}
        forked_only = [name for name, value in fields.items() if value is not None]
        if operands.held:
            forked_only.append('held')
        if forked_only:
            raise ValueError(f"{', '.join(forked_only)} belong to a forked call (`Chunk.loads`)")
    return given


def _prologue_of(hooks: recurrence.Hooks, operands: Chunk, dtype) -> tuple:
    """`operands` as the chunked op reads them (`_read_checked`) through the prologue's arithmetic, the
    decode step's: `(given, query, key, value, g, beta, scale, dtype, work)`, the queries and keys normed
    where `hooks` norm them, the log decay and the write gate in `work` (zero and one where `hooks` form
    none), the output's `dtype` (the operands' where None) and `work` at least fp32."""
    given = _read_checked(hooks, operands)
    query, key, value = given['query'], given['key'], given['value']
    dtype = jnp.result_type(query, key, value) if dtype is None else jnp.dtype(dtype)
    work = at_least_fp32(jnp.result_type(query, key, value))
    if hooks.l2norm:
        # Normalized in fp32 and rounded once to their own dtype, as fla's in-kernel norm and the layers'
        # conv hand them to a kernel: kept fp32, a bf16 call reached the template as an fp32 one, its final
        # memory 30 to 41 times fla's distance from float64 in two parts (row 184's GDN cell, acb1301632c).
        query, key = (l2norm(x.astype(work)).astype(x.dtype) for x in (query, key))
    scale = (key.shape[-1] ** -0.5 if hooks.l2norm else 1.0) if hooks.scale is None else hooks.scale
    g, beta = recurrence.log_gates(hooks, given, work)
    # No decay a token's (the queries' tokens: a token of updates, `Hooks.substeps`, decays once), and every
    # update's own write gate of one.
    g = jnp.zeros((*query.shape[:2], value.shape[2]), work) if g is None else g
    beta = jnp.ones(value.shape[:3], work) if beta is None else beta
    return given, query, key, value, g, beta, scale, dtype, work


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
    output 150 times the recurrence's distance at 2^110). A decay a value takes alone, a chunk's span past
    fp32's least normal, is kept as normal factors (`dew.algebra.scan._Split`, row 9660), while a
    product's own range is fla's: a term whose factors meet a decay past the least normal one at a time
    reads zero as fla's does. A call names a kernel through `generation.chosen`
    alone, the recurrence template's 'mosaic_gpu' among them (row 959: the selectors decide every call). A
    coupled momentum (`recurrence.Hooks.coupled`) runs on the portable math alone (`recurrence_coupled`)."""
    given, query, key, value, g, beta, scale, dtype, work = _prologue_of(hooks, operands, dtype)
    state = operands.state
    if hooks.coupled is not None:
        return recurrence_coupled.chunked(hooks, given, query, key, value, g, beta, operands,
                                          chunk_size=chunk_size, scale=scale, work=work, dtype=dtype)
    (query, key, value, g, beta, erase, starts, firsts, max_segments), stride = _laid(
        hooks, (query, key, value, g, beta), given, operands.starts, operands.firsts, max_segments)
    erase_left, correction = (None, None) if erase is None else erase
    forked = None
    if operands.loads is not None:
        forked = Forked.of(operands, starts, firsts, work)
        firsts = forked.firsts
        max_segments = firsts.shape[1]
    readout = hooks.reads == 'after' and not hooks.own_weight
    # A readout apart from its token's own write (Mamba-3's) the template forms under a decay a head or none
    # without the delta rule (`mainloop.Call.apart`); any other the portable math takes, as the decayed
    # query's read of the state entering the token, `(g q)^T S`, exactly the state after the update less
    # the token's write.
    apart = hooks.apart
    if hooks.rated and not hooks.delta and not hooks.l2norm and hooks.scale is None and readout:
        # Each of its routes adds the skip, the port in its kernels.
        out, final = _rated_scan(hooks, query, key, value, g, beta, given, state, starts, firsts, chunk_size,
                                 work, dtype, max_segments=max_segments, forked=forked)
    else:
        # The delta rule's selector decides every form (`GATED_DELTA`): the forms its kernels do not run hand
        # over to the portable math there, reported as the call's route. A kernel's output unrounded, as the
        # portable math's, where an epilogue reads it or the caller's dtype is not the operands' (Mamba-3's
        # fp32 rank terms add to it, `chunk_mamba3`), so the output is rounded once. The template's
        # numbers are read once, as the selector reads them: a remat's pullback runs the launch later, under
        # whatever precision its caller set then (a saved fp32 gradient under 'highest' took the accurate
        # forms' three parts, whose solve no sm90 thread holds, past the fast ones the selector gave it).
        epilogue = hooks.output is not None or dtype != jnp.result_type(query, key, value)
        wide_out = hooks.skip or epilogue
        # fla's simple GLA rounds as its release places it, values gated (`mainloop.Call.values_gated`).
        values_gated = simple_gla(hooks)
        transition, reads = _transition(hooks, correction)
        unit_pairs = _unit_pairs(hooks)
        # The least log decay a token the layer proves, which a decay a key dimension's prefix reads
        # (`mainloop.Call.prefix_limbs`); a head's takes its compensated pair whatever it is.
        decay_bound = hooks.decay_bound if hooks.decay == 'vector' else None
        rounded = chunk_rounding(hooks.delta, jnp.result_type(query, key, value), released=values_gated)
        skip = given['skip'] if hooks.skip else None
        # Without the delta rule the template takes the skip, its output wide where an epilogue or the
        # caller reads it so (`chunk_forward.forward`).
        fused_wide = wide_out if hooks.delta else epilogue

        def product_kernel(product: str) -> str:
            return port_kernel(query, key, value, g, beta, state, starts, max_segments=max_segments,
                               scale=scale, wide=fused_wide, bound=hooks.inverse_bound,
                               product=product, forked=forked, delta=hooks.delta,
                               decayed=hooks.decay != 'none', skip=None if hooks.delta else skip,
                               rounded=rounded, firsts=firsts, readout=readout, apart=apart,
                               values_gated=values_gated, unit_pairs=unit_pairs,
                               decay_bound=decay_bound, turns=hooks.turns, turn=given.get('turn'),
                               transition=transition, reads=reads)

        def run(kernel, query, key, value, g, beta, skip, state, starts, firsts, turn, correction, across,
                erase_left, *walked):
            # Every kernel walks the packed rows, each document from its own memory, its values read once
            # with the skip's (`_read_once`). Without the delta rule the template takes the skip, its
            # values' gradient summed with the recurrence's before it rounds once (`chunk_forward.forward`):
            # added after, the template's rounded to the values' dtype apart from the skip's, Mamba2's step
            # gates' values' gradient's worst row of 40 draws was 2.9e-2 from float64 against the portable
            # math's 3.5e-3 (interpreted).
            carried = None if forked is None else forked.carrying(*walked)
            wide, own = _read_once(value, work)
            fused = kernel == 'mosaic_gpu' and not hooks.delta
            if kernel == 'mosaic_gpu':
                out, final = _on_template(hooks, query, key, own, g, beta, state, skip if fused else None,
                                          turn, starts=starts, firsts=firsts, forked=carried,
                                          max_segments=max_segments, scale=scale, wide=fused_wide,
                                          rounded=rounded, values_gated=values_gated, unit_pairs=unit_pairs,
                                          decay_bound=decay_bound, reads=_transition(hooks, correction)[1])
            elif kernel == 'fla':
                out, final = fla_rule(query, key, own, g, beta, state, starts=starts,
                                      max_segments=max_segments, scale=scale, wide=wide_out, firsts=firsts)
            else:
                out, final = _on_portable(hooks, query, key, wide, g, beta, state, turn, starts=starts,
                                          firsts=firsts, forked=carried, scale=scale, dtype=work,
                                          own=given['own'] if hooks.own_weight else None,
                                          chunk_size=chunk_size, correction=correction, across=across,
                                          erase_left=erase_left)
            out = out.astype(work)
            return (out if fused else recurrence.skipped(hooks, out, wide, {'skip': skip}, work)), final

        # The skip's `D` (operand 5) is the call's parameter.
        out, final = _per_product(GATED_DELTA.bound(product_kernel), run, query, key, value, g, beta, skip,
                                  state, starts, firsts, given.get('turn'), correction,
                                  given['value_decay'].astype(work) if hooks.value_decay else None,
                                  erase_left, *(() if forked is None else forked.walked), parameters=(5,))
    # A token's output its last laid token's (`_laid`); the epilogue in work, the output rounded once.
    out = out[:, stride - 1::stride]
    return recurrence.normed(hooks, out, given, work).astype(dtype), final


def _transition(hooks: recurrence.Hooks, correction) -> tuple[bool, jax.Array | None]:
    """Whether a call's transition is its own (`PortCase.transition`), and the reads its keys erase along
    where the template takes them (`Hooks.erase` 'read' under a decay a head, `chunk_forward.forward`'s
    `reads`, the laid correction's reader negated): a read erase under a decay a key dimension is one."""
    reading = hooks.erase == 'read' and hooks.decay == 'scalar' and correction is not None
    reads = -correction if reading else None
    own = hooks.erase not in ('tied', 'read') or hooks.value_decay or (hooks.erase == 'read' and not reading)
    return bool(own), reads


def _laid(hooks: recurrence.Hooks, tokens: tuple, given: dict, starts, firsts, max_segments: int | None):
    """A call's `tokens` `(query, key, value, g, beta)` laid as the op's tokens: a token's updates in turn
    (`_substeps_laid`, `Hooks.substeps`), an untied erase's two tokens or an additive erase's one beside its
    erase vectors (`_erases_laid`), or as given, each a form of its own (`recurrence.Hooks`' validation).
    `((query, key, value, g, beta, erase, starts, firsts, max_segments), stride)`: the laid tokens a call's
    token takes, its output its last's, `erase` `_erases_laid`'s."""
    query, key, value, g, beta = tokens
    if hooks.substeps > 1:
        query, g, starts, firsts, max_segments = _substeps_laid(hooks.substeps, query, g, starts, firsts,
                                                                max_segments)
        return (query, key, value, g, beta, None, starts, firsts, max_segments), hooks.substeps
    if hooks.erase in ('read', 'untied', 'additive'):
        laid = _erases_laid(hooks, query, key, value, g, beta, given, starts, firsts)
        return (*laid, max_segments), 2 if hooks.erase == 'untied' else 1
    return (query, key, value, g, beta, None, starts, firsts, max_segments), 1


def _substeps_laid(substeps: int, query, g, starts, firsts, max_segments: int | None):
    """A call of `substeps` delta-rule updates a token (`Hooks.substeps`, DeltaProduct's) as the op's tokens,
    its keys', values' and write gates' own in turn: each token's query on its last update (zeros on the
    others, whose outputs the op drops) and its decay on its first (zeros, no decay, on the others), a packed
    document from its first token's first update, fla's `chunk_gated_delta_product`'s layout; a row of
    packed documents at most a document a token where `max_segments` bounds none, its updates holding none
    of their own."""
    if starts is not None and max_segments is None:
        max_segments = starts.shape[1]
    def spread(x, at: int):
        # Padded beside each token on an axis of its own, so the laid operand keeps the token's placement.
        before = at % substeps
        widths = ((0, 0), (0, 0), (before, substeps - 1 - before)) + ((0, 0),) * (x.ndim - 2)
        laid = jnp.pad(x[:, :, None], widths)
        return laid.reshape(x.shape[0], x.shape[1] * substeps, *x.shape[2:])

    return (spread(query, -1), None if g is None else spread(g, 0),
            None if starts is None else spread(starts, 0), None if firsts is None else firsts * substeps,
            max_segments)


@dataclasses.dataclass(frozen=True)
class PortCase:
    """A delta rule's call as the port's selector reads it (`GATED_DELTA`): whether a caller's
    `shard_map` that checks varying axes holds operands that vary (`varying`), the operands' common
    `dtype`, the keys' width (`key_width`), its heads' shape: key heads, value heads and the values' width
    (`heads`), the recurrence template's implementations bound to the call (`call`, `templates.base.Call`;
    None where the caller asks of the port alone), the instruction family of the device it runs on
    (`family`, `generation.family`; None off a GPU), and the product it is selected for (`product`:
    'forward' where no gradient is taken, 'training' where one is, `_per_product`), and the recurrence's
    form: whether it is the delta rule (`delta`) and decays (`decayed`), the largest entry of its chunk
    inverse the form proves (`bound`, `recurrence.Hooks.inverse_bound`; None where it proves none), whether
    its decay is a key dimension's
    (`per_key`), whether its documents enter earlier documents' finals (`forked`, `Chunk.loads`), and
    whether its readout reads the state after its token's update alone (`readout`: `Hooks.reads` 'after'
    and no `Hooks.own_weight`) or after it less its group's writes (`apart`, the group's tokens: 'apart',
    which the template forms and no port's kernels do; zero where it does not), the key pairs its state
    turns (`turns`, `Hooks.turns`), and whether its transition is other than the delta rule's own beside a
    decay along the keys (`transition`: `Hooks.erase` past 'tied', or `Hooks.value_decay`), and whether its
    keys erase along reads of their own (`reads`: `Hooks.erase` 'read' under a decay a head, the template's
    alone, `chunk_forward.forward`'s `reads`); its heads' shape
    and its extent on a launch's shard (`heads`: key heads, value heads and the values' width; `extent`),
    which a declared cell's receipt timed (`TIMED`)."""

    varying: bool
    dtype: np.dtype
    key_width: int
    heads: tuple[int, int, int]
    extent: "Extent"
    call: base.Call | None = None
    family: str | None = None
    product: str = 'forward'
    delta: bool = True
    decayed: bool = True
    bound: float | None = None
    per_key: bool = False
    forked: bool = False
    readout: bool = True
    apart: int = 0
    turns: int = 0
    transition: bool = False
    reads: bool = False

    @property
    def cell(self) -> tuple[str | None, "PortCell | None"]:
        """The call's cell (`selection.Celled`, `PORT_CELLS`): its device's instruction family and what a
        launch's shard computes (`PortCell`): a receipt routes the shapes it timed, the owner's rule (PLAN's
        Priorities: the fastest measured route remains until the template catches up)."""
        if self.transition or self.reads:
            # A transition of its own, or an erase along reads of its own, no receipt's workload
            # (`TRANSITION_HANDOFF`, `READS_HANDOFF`).
            return self.family, None
        keys, values, width = self.heads
        return self.family, PortCell(self.dtype.name, self.product, self.per_key, (keys, self.key_width),
                                     (values, width), self.extent)

    @property
    def widest(self) -> int:
        """The widest keys the port takes in the call's dtype."""
        return fla_gdn.WIDEST_KEYS if self.dtype.itemsize == 2 else fla_gdn.WIDEST


_VARYING = ("a shard_map that checks varying axes holds the operands, and a Pallas call states no varying "
            "type for its results")


def upstream_precision(dtype) -> bool:
    """Whether the run's declaration (`dew.nn.precision.effective_precision`) admits the upstream's products
    (fla's and mamba_ssm's Triton dots, on a GPU) for a call in `dtype`: a 16-bit call's, exact with fp32
    sums, where the declaration takes those (`exact_products`); an fp32 call's, TF32, at no more than the
    default. Where it does not, no run of the upstream computes at the declared precision: the port hands the
    call over (`_port_precision`), and a test's oracle is the recurrence a token at a time (testing.md)."""
    dtype = jnp.dtype(dtype)
    if dtype.itemsize == 2:
        return exact_products(None, dtype)
    return asks_default_precision(None, configured=True)


def _port_precision(case) -> bool:
    """Whether the run's declaration admits a port's products for a call of `case.dtype`
    (`upstream_precision`). Off a GPU (not `triton_runs`) its kernels interpret, their products exact under
    any declaration (`fla_gdn._dot`'s `exact`: an fp32 product summed in float64 and rounded once)."""
    return not triton_runs() or upstream_precision(case.dtype)


_PORT_PRECISION = ("a product precision the run declares that the port's products do not compute: above "
                   "the default for fp32 operands, whose products it takes in TF32, or a dot algorithm that "
                   "narrows 16-bit ones")


READOUT_HANDOFF = ("a readout of the state entering its token, weighing its token's own write "
                   "(`recurrence.Hooks.reads`, `Hooks.own_weight`: RWKV6's), or apart from its group's "
                   "writes (Mamba-3 MIMO's) where the template's mainloop forms none (`Hooks.apart`: under "
                   "a decay a key dimension, the delta rule or an own weight; the port's kernels form none "
                   "apart), which the portable math's forms derive (`scan.Scan.reads`)")
"""Why a kernel of the chunked op hands a call of another readout over to the portable math."""

TURN_HANDOFF = ("a state that turns its key pairs (`recurrence.Hooks.turns`: Mamba-3's rotating basis), "
                "which the template's and the portable math's turned wrapper takes "
                "(`templates.recurrence.turned`) and the port's kernels do not")
"""Why a port's kernel hands a turned call over to the portable math."""



TRANSITION_HANDOFF = ("an erase of its own or a decay along the values (`recurrence.Hooks.erase`: GDN-2's, "
                      "Comba's, RWKV7's or Oja's; `Hooks.value_decay`: two-sided GLA's), whose chunk forms "
                      "the portable math derives from its token update (`scan.derive_scan`) and neither "
                      "the port's kernels nor the template's mainloop form")
"""Why a kernel of the chunked op hands a call of another transition over to the portable math."""


PORT_FORKED = Refusal('forked', "a forked call, whose documents enter earlier documents' finals of the same "
                      "call (`Chunk.loads`): the port runs a call's documents as sequences of their own in "
                      "parallel (fla's varlen), which no document's memory can wait on", row=INHERENT)
"""Why fla's port takes no forked call, inherently: its documents run in parallel."""

SSD_PORT_FORKED = Refusal('forked', "a forked call (`Chunk.loads`), whose documents take their memories up "
                          "mid-chunk, where mamba_ssm's state passing takes a memory at a sequence's first "
                          "chunk alone: the template and the portable math walk it", row=INHERENT)
"""Why mamba_ssm's port takes no forked call, inherently: it takes memories at chunks' edges alone."""

READS_HANDOFF = ("an erase along reads of its own (`recurrence.Hooks.erase` 'read': the preconditioned delta "
                 "rules'), whose system and weights the template forms from them and fla's kernels do not")

TRANSITION: Requirement[PortCase] = Requirement('transition', lambda case: not case.transition,
                                                TRANSITION_HANDOFF, row=INHERENT, kind='handoff')
"""A call of a transition of its own handed to the portable math before any kernel's needs are asked: a
kernel a call names refuses no such call (`selection`'s order)."""

VARYING: Requirement[PortCase] = Requirement('varying', lambda case: not case.varying, _VARYING,
                                             row='ledger 822')

PORT_SUPPORTS: tuple[Requirement[PortCase], ...] = (
    TRANSITION,
    Requirement('reads', lambda case: not case.reads, READS_HANDOFF, row=INHERENT, kind='handoff'),
    Requirement('gated', lambda case: case.delta and case.decayed,
                "a recurrence other than the gated delta rule, which the rule's kernels run (fla's "
                "chunk_gated_delta_rule and chunk_kda, the template's delta-rule mainloop): no delta rule (a "
                "gated linear attention, a rated scan with its keys normed or scaled) or no decay (the plain "
                "delta rule), which the portable math runs", row=INHERENT, kind='handoff'),
    VARYING,
    Requirement('float64', lambda case: case.dtype != jnp.float64,
                "float64 operands, whose rule is the reference math itself", row=INHERENT),
    Requirement('key_width', lambda case: case.key_width <= case.widest,
                lambda case: f"keys {case.key_width} wide; the port takes at most {case.widest} in "
                             f"{case.dtype.name}", row=INHERENT),
    Requirement('triton', lambda case: triton_runs(),
                "no GPU Dew's Triton kernels run on (sm80 or later): off one the port's kernels interpret, "
                "as the CPU suite runs them where a call names the port", row=INHERENT, kind='measured'),
    Requirement('precision', _port_precision, _PORT_PRECISION, row=INHERENT, kind='handoff'),
    Requirement('forked', lambda case: not case.forked, PORT_FORKED.detail, row=INHERENT),
    Requirement('readout', lambda case: case.readout, READOUT_HANDOFF, row=INHERENT, kind='handoff'),
    Requirement('turns', lambda case: not case.turns, TURN_HANDOFF, row=INHERENT, kind='handoff'),
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

FLA_PORT = Kernel[PortCase]('fla', PORT_SUPPORTS)
"""Dew's port of fla's kernels (`fla_rule`)."""

class Extent(NamedTuple):
    """A launch's rows and tokens on its shard and whether its rows pack documents (`Chunk.starts`)."""

    rows: int
    tokens: int
    packed: bool


TIMED = Extent(1, 4096, packed=False)
"""The extent every declared cell's receipt timed (`PORT_CELLS`, `SCAN_CELLS`): one unpacked row of 4096
tokens, tools/benchmark_recurrence.py's `Shape` at its default batch and length, as M-JS-106's B200 rows
(690a7e87d85) and ssd-train-a100-fdab2424c ran it. A call of another is no declared cell's, and its kernels
take it in their order (row 8991)."""


class PortCell(NamedTuple):
    """A delta rule's cell a receipt declares (`PortCase.cell`): its operands' common dtype, its product,
    whether its decay is a key dimension's, its key heads and width and its value heads and width on a
    launch's shard, and its extent."""

    dtype: str
    product: str
    per_key: bool
    keys: tuple[int, int]
    values: tuple[int, int]
    extent: Extent


B200 = "on a B200, by device time"
"""Where the tcgen05 cells' receipts measured (M-JS-106 at 690a7e87d85: js106 one round, js106c five)."""

PORT_CELLS: dict[tuple[str | None, PortCell], tuple[Kernel[PortCase], str]] = {
    ('tcgen05', PortCell('bfloat16', 'forward', per_key=False, keys=(16, 128), values=(48, 128),
                         extent=TIMED)): (
        FLA_PORT, f"{B200}, the port's forward 0.394 ms against the template's 0.725 on Qwen3.5-27B's gated "
                  "delta net (js106c)"),
    ('tcgen05', PortCell('bfloat16', 'forward', per_key=False, keys=(16, 128), values=(32, 128),
                         extent=TIMED)): (
        FLA_PORT, f"{B200}, the port's forward 0.276 ms against the template's 0.399 on Qwen3-Next's gated "
                  "delta net (js106)"),
    ('tcgen05', PortCell('bfloat16', 'forward', per_key=True, keys=(32, 128), values=(32, 128),
                         extent=TIMED)): (
        FLA_PORT, f"{B200}, the port's forward 1.118 ms against the template's 2.510 on Kimi Linear's delta "
                  "attention (js106c)"),
    ('tcgen05', PortCell('bfloat16', 'training', per_key=False, keys=(16, 128), values=(48, 128),
                         extent=TIMED)): (
        FLA_PORT, f"{B200}, the port's forward and gradient 1.747 ms against the template's 1.889 on "
                  "Qwen3.5-27B's gated delta net (js106c)"),
    ('tcgen05', PortCell('bfloat16', 'training', per_key=True, keys=(32, 128), values=(32, 128),
                         extent=TIMED)): (
        FLA_PORT, f"{B200}, the port's forward and gradient 4.317 ms against the template's 6.642 on Kimi "
                  "Linear's delta attention (js106c)"),
}
"""The cells, each an instruction family and what a launch's shard computes (`PortCase.cell`: a receipt
routes the shapes it timed), where a receipt shows Dew's port of fla's kernels (`fla_rule`; "the port"
below) faster than the recurrence template, each its old route's kernel and that receipt; every time below
sets the template against that port, not against upstream fla's own kernels or FlashQLA's
(tools/benchmark_recurrence.py's `fla` and `flashqla` arms): the delta rule's selector runs the port on them
where it takes the call (`Op.declared`, owner, bible da64ae5); the list shrinks as the template wins its
cells, and no cell is picked or ranked at run time. A Blackwell's forward at each timed shape is the port's;
its training the port's at Qwen3.5-27B's gated delta net and Kimi Linear's delta attention, the template's at
Qwen3-Next's gated delta net, 1.196 ms against the port's 1.422 (js106). Every other cell is the
template's. Warpgroup MMA's forward of Kimi delta attention, the last, is the template's, 1.154 ms against
the port's 1.571 on a GH200 480GB and 1.647 against 2.412 on an H100 PCIe (M-JS-43b-gh200 and
-h100pcie at e52d05f558f: exclusive, two rounds alternating the arms; the cell's receipt had the port's
1.406 against the template's 1.624 on an H100 80GB HBM3 at c70aaa8, window-h100-c70aaa8-b8109ee). A gated
delta net's cells are all the template's. An A100's forward of Kimi delta attention is the template's, 5.06
ms against
the port's 8.81 (JS-kdacells-a100-a928ea2b4), and its training, 11.12 ms against 15.92
(batch-window-a100-b2ad84c61). Warpgroup MMA's training of Kimi delta attention is the template's, its
forward and gradient 3.926 ms a call against the port's 4.726 on an H100 80GB HBM3 and 3.916 against 4.694 on
a GH200 480GB (FS-RECROUTES-h100 and -gh200 at 26d3fdf5df: cold, exclusive, two rounds alternating the arms'
order). A gated delta net's training is the template's, its
keys launch a key block a program and its query/key launch's memory sum a product: its forward and gradient
on an H100 2.46 ms against the port's 3.12
on Qwen3.5-27B's gated delta net and 1.92 against 2.44 on Qwen3-Next's (JS-train80-h100-db2c4aca0), and on an
A100 SXM4 40GB 3.46 against 4.21 and 2.50 against 3.24 (JS-2075cellsd-a100-863706e94).
An A100's forward is the template's, its walks' 16-bit stores coalesced (`swizzling.coalesced`): 0.931 ms
against the port's 0.981 on Qwen3.5-27B's gated delta net and 0.692 against 0.759 on Qwen3-Next's, each in a
process of its own, three rounds (JS-2075cellsd-a100-863706e94; the port's Qwen3-Next forward took 0.650 in
two runs that followed a training run in one process, JS-2075cells-a100-491c05798).
An H100's forward is the template's, its solve launch two programs a multiprocessor on warpgroup MMA's layout
without a producer (`mainloop.widest_layout`): on an H100 80GB HBM3 0.364 ms against the port's 0.514 on
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
    None off a GPU), the product it is selected for (`product`), the operands' `dtype`, and whether the
    readout reads the state after its token's update alone (`readout`) or after it less its own write
    (`apart`)."""

    @property
    def call(self) -> base.Call | None: ...

    @property
    def family(self) -> str | None: ...

    @property
    def product(self) -> str: ...

    @property
    def dtype(self) -> np.dtype: ...

    @property
    def readout(self) -> bool: ...

    @property
    def apart(self) -> int: ...

    @property
    def turns(self) -> int: ...


TEMPLATE_SUPPORTS: tuple[Requirement[Templated], ...] = (
    # The measured device first: off a GPU the automatic route refuses here, before the launch's eligibility
    # fits a target's configs (18 fits a decision on the CPU, 0.86 of its 0.87 s, row 6740); a call naming
    # the template passes it (`selection`'s forcing) and reaches its eligibility interpreted.
    Requirement('device', lambda case: case.family is not None,
                "no GPU of an instruction family the template's bundles cover (a CPU, or a GPU of none, a "
                "Turing's): off a GPU the mainloop runs interpreted where a call names it",
                row=INHERENT, kind='measured'),
    Requirement('launch', lambda case: case.call is not None and case.call.eligible('mosaic_gpu'),
                "the recurrence template's launches take no call of these dtypes or widths, or no config "
                "fits this device or holds in its compiles (each spilling, the start's misses name them), "
                "or no Mosaic bundle claims its instruction family", row='ledger 1294'),
    Requirement('backward', lambda case: case.product != 'training' or _trains(case.call),
                "a gradient the template's fast backward does not take (`mainloop_backward."
                "fast_backward_form`): an fp32 decay a key dimension in three bf16 parts, an fp16 call "
                "without the delta rule, a decay a key dimension without it, or a bf16 unrounded call with "
                "a wide fp32 output; or launches past the device's registers or shared memory at the "
                "call's tiles (`template.backward_holds`)", row='ledger 1294'),
    Requirement('precision', lambda case: exact_products(None, case.dtype),
                "a dot algorithm the run declares that narrows the operands' products (TF32's, bf16 parts of "
                "an fp32 product), which the template's whole products (its accurate parts, or a 16-bit "
                "operand's own) do not compute; it rounds an fp32 product at the default alone, under the "
                "run's fast forms (`fast_rounding`)", row=INHERENT, kind='handoff'),
    Requirement('readout', lambda case: case.readout or case.apart > 0, READOUT_HANDOFF, row='ledger 6128',
                kind='handoff'),
)
"""What the recurrence template's Mosaic GPU mainloop (`templates.recurrence.template.Mosaic`) needs of a
call under either selector (`GATED_DELTA`, `SCAN`): its own launches' call (`Template.eligible`: dtypes,
widths, a bundle for the device's family, a config that fits), a GPU, and for
a training product its fast backward (`Mosaic.trains`; the rule's gradient takes no other, row 1926)."""

MOSAIC_SUPPORTS: tuple[Requirement[PortCase], ...] = (TRANSITION, VARYING, *TEMPLATE_SUPPORTS)
"""What the recurrence template needs of a delta rule's call: the port's placement (a Pallas call's results),
and `TEMPLATE_SUPPORTS`, a forked call's (`Chunk.loads`) as another's, its walk keeping the call's table, in
every form its mainloop implements, the delta rule with a decay or without one and the forms without the
delta rule alike (row 1963: the port's 'gated' form was the template's too). A declared port cell runs the
port first where the port takes the call (`PORT_CELLS`, `GATED_DELTA`'s declared cells)."""

MOSAIC = Kernel('mosaic_gpu', MOSAIC_SUPPORTS)


GATED_DELTA: Op[PortCase] = Op(Choice.GATED_DELTA_RULE,
                               (MOSAIC, FLA_PORT, Kernel('xla', (), generic=True)),
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


def _grouped(query, key, value, dtype):
    """A derived walk's output dtype (the operands' where `dtype` is None), its working dtype (at least
    fp32), and `query` and `key` repeated over the value heads each key head serves."""
    dtype = jnp.result_type(query, key, value) if dtype is None else jnp.dtype(dtype)
    work = at_least_fp32(jnp.result_type(query, key, value))
    group = value.shape[2] // key.shape[2]
    if group > 1:
        query, key = (jnp.repeat(x, group, axis=2) for x in (query, key))
    return dtype, work, query, key


def _chunked(named: dict, starts, chunk_size: int, work) -> dict:
    """A derived walk's operands (`_walk`): each of `named`'s `[B, S, H, ...]` in `work`, padded with tokens
    that leave the memory as it was and laid out `[B, H, chunks, C, ...]`; where `starts` `[B, S]` is
    given, each head's tokens' resets (`RESET`) where their row's document starts, a padded token starting
    none."""
    rows, length, heads = next(iter(named.values())).shape[:3]
    chunks = -(-length // chunk_size)
    padded = chunks * chunk_size - length

    def laid(x):
        x = jnp.pad(x, [(0, 0), (0, padded)] + [(0, 0)] * (x.ndim - 2))
        x = jnp.moveaxis(x, 2, 1)
        return x.reshape(rows, heads, chunks, chunk_size, *x.shape[3:])

    operands = {name: laid(x.astype(work)) for name, x in named.items()}
    if starts is not None:
        operands[RESET] = laid(jnp.broadcast_to(starts[:, :, None], (rows, length, heads)))
    return operands


def xla_mainloop(config: Tiles, query, key, value, g, beta, state=None, starts=None, firsts=None, *,
                 scale: float, delta: bool, dtype=None, bound: float | None = None,
                 forked: Forked | None = None, reads: Reads = 'after', own=None, group: int = 1,
                 correction=None, erase_left=None, across=None, values_read: bool = False, coupled=None):
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
    documents' first tokens, each entering a zero memory (the forms' reset, `scan.Scan.reset`). A forked call
    (`forked`, `Chunk.loads`) walks its documents in place, each taking its memory up at its first token from
    its table, `state` `[B, S', Hv, Dk, Dv]`'s given ones then those documents store at their last tokens
    (`_walk`), the final memory the row's last. `bound`,
    the layer's write gate's largest where its keys are of unit norm (`recurrence.Hooks.inverse_bound`), is
    the declaration's range facts (`scan.Scan.ranges`). The readout reads the memory entering its token where
    `reads` is 'before' (`recurrence.Hooks.reads`), or after its decay apart from every write of its `group`
    where 'apart' (`scan.Scan.group`: Mamba-3's ranks), and `own` `[Hv, Dk]`, where given, weighs the token's
    own write in it (`Hooks.own_weight`), each head's given every chunk. A turned call's turns are its
    wrapper's (`templates.recurrence.turned`), which launches this unturned. `correction` `[B, S, Hv, Dk]`,
    where given, reads each token's correction of the decayed memory in place of its keys, `h = D h + beta k
    (v + c^T D h)^T` (an untied erase's laid tokens, `_erases_laid`), or with `erase_left` `[B, S, Hv, Dk]`
    each token's erase along it of the memory it enters beside its write, `h = D h + a (c^T h)^T + beta k v^T`
    (an additive erase's); `across` `[B, S, Hv, Dv]`, where
    given, decays the memory along its values too, `h = D h E + beta k v^T`, `E = exp(across)` (two-sided
    GLA's, `recurrence.Hooks.value_decay`); `values_read` reads the memory along its values (`h scale
    q`: Oja's rule's transposed memory, `_portable`); and `coupled` `(p [B, S, Hv, Dk], log_mu [B, S, Hv])`,
    where given, is a coupled momentum's erase keys and log decay (`recurrence.Hooks.coupled`), the memory a
    pair `[B, Hv, 2, Dk, Dv]` (`_delta_forms`' `coupled`)."""
    dtype, work, query, key = _grouped(query, key, value, dtype)
    rows, length, heads, columns = value.shape
    tokens = None
    if forked is not None:
        # Each document's first token a reset of the chunk's own forms, its memory taken up there.
        tokens = forked.tokens(length)
        starts = tokens[0] >= 0
    width = key.shape[-1]
    facts = (("delta", delta), ("scale", scale), ("per_key", g.ndim == 4), ("width", width),
             ("columns", columns), ("work", jnp.dtype(work)), ("reset", starts is not None), ("bound", bound),
             ("reads", reads), ("own", own is not None), ("group", group),
             ("corrected_by", None if correction is None else 'after' if erase_left is None else 'erased'),
             ("across", across is not None),
             ("values_read", values_read), ("coupled", coupled is not None))
    held = 0 if forked is None else forked.held
    # Each token's read's width: the memory's values', or its keys' read along its values (`values_read`).
    read = width if values_read else columns
    given: dict[str, jax.Array] = dict(zip(("q", "k", "v", "g", "beta"), (query, key, value, g, beta),
                                           strict=True))
    if correction is not None:
        given["c"] = correction
    if erase_left is not None:
        given["a"] = erase_left
    if across is not None:
        given["e"] = across
    if coupled is not None:
        given["p"], given["mu"] = coupled
    operands = _chunked(given, starts, config.chunk, work)
    if own is not None:
        # Each head's weight every chunk, as the walk maps a row's and head's chunks.
        chunks = operands["q"].shape[2]
        operands["own"] = jnp.broadcast_to(own.astype(work)[None, :, None], (rows, heads, chunks, width))
    # The memory entering, and each carry below, typed as every operand varies (a checked `shard_map`'s
    # manual axes; an explicit axis's rows, as the portable math's): a step's outputs vary wherever its inputs
    # do, keys, gates or resets replicated or not.
    varying = (query, key, value, g, beta, starts, correction, erase_left, across, *(coupled or ()),
               *(() if forked is None else jax.tree.leaves((tokens, forked.entry, forked.adds))))
    if tokens is not None and forked is not None and state is not None:
        # The table's given memories `[B, Hv, S', Dk, Dv]`, each token's load, store and document
        # `[B, Hv, chunks, C]` (a padded token none), each document's added slot where the call gives them,
        # the entry write's operands each value head's, where the call gives one, and a turned call's load
        # rotation each value head's `[B, Hv, D, A]` (`Forked.turning`).
        adds = () if forked.adds is None else (
            jnp.broadcast_to(forked.adds[:, None], (rows, heads, forked.adds.shape[1])),)
        walked = (varying_like(jnp.moveaxis(state.astype(work), 1, 2), *varying),
                  *(_headed_tokens(x, fill, heads, config.chunk) for x, fill in zip(tokens, (-1, 0, 0),
                                                                                  strict=True)),
                  *adds, *(() if forked.entry is None else _headed_entry(forked.entry, heads, *varying)),
                  *(() if forked.turning is None else
                    (varying_like(jnp.moveaxis(part, 2, 1).astype(work), *varying)
                     for part in forked.turning)))
        walk = _walk_once(facts, width=width, recomputed=recomputed_memory(), held=held,
                          documents=forked.loads.shape[1], added=forked.adds is not None,
                          turned=forked.turning is not None, context=derive.context())
        outputs, final = _rows_local(walk, value)(operands, walked)
        out = jnp.moveaxis(outputs.reshape(*outputs.shape[:2], -1, read), 1, 2)
        return out[:, :length].astype(dtype), final
    # A coupled momentum's pair `[2, Dk, Dv]` a head's, the memory then the momentum.
    paired = (2,) if coupled is not None else ()
    entering = varying_like(rows_like(jnp.zeros((rows, heads, *paired, width, columns), work), value)
                            if state is None else state.astype(work), *varying)

    outputs, final = _walk_once(facts, width=width, recomputed=recomputed_memory(), held=0,
                                context=derive.context())(operands, entering)
    out = outputs.reshape(*outputs.shape[:2], -1, read)
    return jnp.moveaxis(out[:, :, :length], 1, 2).astype(dtype), final


@functools.lru_cache(maxsize=FORMS_HELD)
def _delta_forms(*, delta: bool, scale: float, per_key: bool, width: int, columns: int, work: jnp.dtype,
                 reset: bool, bound: float | None, reads: Reads = 'after', own: bool = False, group: int = 1,
                 corrected_by: str | None = None, across: bool = False, values_read: bool = False,
                 coupled: bool = False):
    """The delta rule's (or a plain linear recurrence's, without `delta`) chunk forms (`scan.derive_scan`) for
    these facts, every one its declaration reads: the update `h = D h - beta k (k^T D h) + beta k v^T` (`D h
    + beta k v^T` without the delta rule; `D h + beta k (v + c^T D h)^T` with a correction reader `c` of its
    own, `corrected_by` 'after': an untied erase's laid tokens, `_erases_laid`; `D h + a (c^T h)^T + beta k
    v^T`, 'erased', an erase along its own vector `a` reading the state the token enters beside the write:
    an additive erase's tokens; `D h E + beta k v^T` with a decay
    `E = exp(e)` along the values, `across`: two-sided GLA's) of a `[width, columns]` state in
    `work`, `D = exp(g)` a head's or each key's (`per_key`), read `scale q^T h` (`scale h q` along its values,
    `values_read`: Oja's rule's transposed state; of the state entering the token where `reads` is 'before',
    after its decay apart from every write of its `group` where 'apart', `scan.Scan.reads`, `Scan.group`;
    with the token's own write weighed a key, `scale ((q own) . (beta k)) v`, where
    `own`), packed documents' resets where `reset`, and the range facts `bound` proves. Derived once for
    equal facts, the latest `FORMS_HELD` kept: built from a call's fresh
    closures, every layer's call derived its own forms, their decomposition and carry splits (`derive_scan`'s
    caches a declaration's) again, 0.25 s a split (a 4B decoder's plan, row 5556). A `coupled` momentum's
    are its pair's (`recurrence_coupled.declaration`)."""
    from dew.algebra import SATURATION

    if coupled:
        return scan.derive_scan(recurrence_coupled.declaration(scale, width, columns, work, reset, bound),
                                saturation=SATURATION)

    def update(h, q, k, v, g, beta, own=None, c=None, e=None, a=None):
        del q, own
        # The language's exp, which the algebra reads as the decay `exp(z)` and evaluates by `scan._exp`
        # (accurate on every backend): Dew's own on a TPU is a protected span it reads no operation of.
        decayed = jnp.exp(g).reshape(-1, 1) * h
        if e is not None:
            decayed = decayed * jnp.exp(e)[None, :]
        if corrected_by == 'erased':
            # Its erase along its own vector, reading the state the token enters, beside its write.
            assert a is not None and c is not None, "an erase of its own takes its vector and its read"
            return decayed + a[:, None] * (c @ h)[None, :] + beta * k[:, None] * v[None, :]
        written = v + c @ decayed if c is not None else v - k @ decayed if delta else v
        return decayed + beta * k[:, None] * written[None, :]

    def readout(h, q, k, v, g, beta, own=None, c=None, e=None, a=None):
        del g, c, e, a
        out = scale * (h @ q if values_read else q @ h)
        # The token's own write weighed a key (`recurrence.Hooks.own_weight`, no delta rule's correction).
        return out if own is None else out + scale * jnp.sum(q * own * beta * k) * v

    axes = {"q": ("time", "value" if values_read else "key"), "k": ("time", "key"), "v": ("time", "value"),
            "g": ("time", "key") if per_key else ("time",), "beta": ("time",),
            **({"own": ("key",)} if own else {}), **({"c": ("time", "key")} if corrected_by else {}),
            **({"e": ("time", "value")} if across else {}),
            **({"a": ("time", "key")} if corrected_by == 'erased' else {})}
    declared = {**axes, RESET: ("time",)} if reset else axes
    # A log decay at most 0 (its decay at most 1); with a bound, keys of unit norm and gates at most it, so
    # two tokens' corrections overlap by at most it (Cauchy-Schwarz).
    ranges = {"g": (-math.inf, 0.0), **({"e": (-math.inf, 0.0)} if across else {}),
              **({} if bound is None else {"k": 1.0, "beta": bound, "overlap": bound})}
    declaration = scan.Scan(update, readout, declared, walked="time", state=((width, columns), work),
                            reset=RESET if reset else None, ranges=ranges, reads=reads, group=group)
    return scan.derive_scan(declaration, saturation=SATURATION)


@functools.lru_cache(maxsize=FORMS_HELD)
def _walk_once(facts: tuple, *, width: int, recomputed: bool, held: int, context: tuple, documents: int = 0,
               added: bool = False, turned: bool = False):
    """The walk (`_walk`) by the forms `_delta_forms(**dict(facts))` derives as one
    nested program (`dew.nn.nested`), traced once for equal shapes within a trace (`jax.jit`'s cache, keyed
    on the function and its operands' types) and compiled by the enclosing program, the latest `FORMS_HELD`
    kept. Keyed on every fact the walk reads: its declaration's, `width`, `recomputed`, `held`, `documents`
    `added` and `turned` (`_walk`'s), and `context`, what the forms' derivation reads (`derive.context`: the
    target, numerics, precision and checks a split is priced and traced under), which the walk reads no
    other way: JAX's trace cache keys on none of Dew's, and the split one context derives is no other's.
    Traced at every call, each layer's walk took 0.65 s of a trace again (jsf-gdn-layers: 8 equal layers'
    training step 5.2 s, 1 layer's 0.63 s). Run unrecorded (`dew._cache.unrecorded`): the walk is portable
    math, launching nothing and dispatching no op of its own, its routes its caller's, so it traces once under
    every receipt its caller records into, where each receipt's identity keyed it a trace of its own (row
    6740: a decision step's walk traced twice, 0.84 s against 0.30)."""
    del context
    walked = _walk(_delta_forms(**dict(facts)), width, recomputed, held, documents, added, turned)

    def recurrence_walk(operands, h):
        return walked(operands, h)

    staged = nested(recurrence_walk)

    def run(operands, h):
        with _cache.unrecorded():
            return staged(operands, h)

    return run


def _rows_local(walk, like):
    """`walk` (`_walk_once`'s, its operands and memories each a row's first) on each device's own rows where
    an Explicit axis splits the call's (`like`'s, `dew.nn.sharding.rows_spec`), else `walk` itself: a forked
    walk's scans seed their table, outputs and memories within its rows' map, whole, where the token maps
    they gather by are split over the rows, a gather JAX refuses to place (row 9690). Every operand is placed
    as the rows are first, the rest of it whole (a no-decay call's gates the prologue forms whole, a head
    split over another axis gathered), and the map checks its varying axes, the walk's carries typed as its
    operands vary (`varying_like`): a call run eagerly within an unchecked map over another axis JAX refuses,
    the two maps' check modes apart."""
    placed = rows_spec(like, 1)
    if placed is None or placed.spec[0] is None:
        return walk
    mapped = jax.shard_map(walk, mesh=placed.mesh, in_specs=P(placed.spec[0]), out_specs=P(placed.spec[0]),
                           axis_names=frozenset(mesh_axes(placed.spec[0])))

    def local(operands, walked):
        return mapped(*jax.tree.map(lambda x: rows_like(x, like), (operands, walked)))

    return local


def _headed_tokens(x, fill: int, heads: int, chunk: int):
    """A forked call's tokens' ints `[B, S]` (`Forked.tokens`) as a walk takes them, each value head's on its
    chunks `[B, Hv, chunks, C]`, a token padding the last chunk `fill`."""
    rows, length = x.shape
    chunks = -(-length // chunk)
    x = jnp.pad(x, ((0, 0), (0, chunks * chunk - length)), constant_values=fill)
    return jnp.broadcast_to(x.reshape(rows, 1, chunks, chunk), (rows, heads, chunks, chunk))


def _headed_entry(entry, heads: int, *varying) -> tuple:
    """A forked call's entry write's operands (`Forked.entry`: `key [B, D, R, Hk, Dk]`, `value [B, D, R, Hv,
    Dv]`, `weight [B, D, Hv]`) each value head's, as a walk forms one document's (`_opened`): `[B, Hv, D, R,
    Dk]`, `[B, Hv, D, R, Dv]`, `[B, Hv, D]`, the key heads each serving its value heads."""
    key, value, weight = entry
    key = jnp.repeat(key, heads // key.shape[3], axis=3)
    return tuple(varying_like(x, *varying) for x in (jnp.moveaxis(key, 3, 1), jnp.moveaxis(value, 3, 1),
                                                     jnp.moveaxis(weight, 2, 1)))


def chunk_gated_oja_rule(query, key, value, gv, beta, state=None, starts=None, *, scale: float | None = None,
                         chunk_size: int = CHUNK_SIZE, dtype=None):
    """Oja's rule with a gate a value dimension (fla's `chunk_gated_oja_rule`, fla/ops/gated_oja_rule, its
    `beta` a head's): each value head's memory `h` `[Dk, Dv]` decayed along its values, its values erased
    from it and written, `h = h D (I - beta v v^T) + beta k v^T` with `D = diag(exp(gv))`, each token read
    `scale q^T h` (`scale` `Dk^-1/2` where None). `query` and `key` `[B, S, Hk, Dk]`, `value` and `gv` `[B, S,
    Hv, Dv]`, `beta` `[B, S, Hv]`; `state` `[B, Hv, Dk, Dv]` the memory entering (zero where None), `starts`
    `[B, S]` each packed document's first token, which enters a zero memory. Returns `(output [B, S, Hv, Dv]
    in dtype (the operands' where None), final memory [B, Hv, Dk, Dv] in at least fp32)`.

    The chunked op's own (`chunk` under `recurrence.Hooks.erase` 'values', row 9028): on the transposed
    memory `h^T` the delta rule of keys `v`, values `beta k` and a decay a key dimension `gv`, read along its
    values, the chunk forms the delta rule's token update derives (`_delta_forms`' `values_read`), walked as
    its portable math walks them, their derivative the token update replayed a chunk at a time."""
    width = key.shape[-1]
    hooks = recurrence.Hooks(decay='vector', erase='values', scale=width ** -0.5 if scale is None else scale)
    return chunk(hooks, Chunk(query, key, value, log_decay=gv, beta=beta, state=state, starts=starts),
                 chunk_size=chunk_size, dtype=dtype)


def recurrent_gated_oja_rule(query, key, value, gv, beta, state=None, *, starts=None,
                             scale: float | None = None):
    """`chunk_gated_oja_rule`'s rule a token at a time as fla's reference states it
    (tests/ops/test_oja.py `recurrent_oja_ref`): the memory decayed along its values, `k' = k - h v` erased
    from it, `h += k' (beta v)^T`, read `scale q^T h`, in at least fp32 (float64 where the operands are);
    operands and results as `chunk_gated_oja_rule`'s."""
    dtype = jnp.result_type(query, key, value)
    work = at_least_fp32(dtype)
    group = value.shape[2] // key.shape[2]
    query, key = (jnp.repeat(x, group, axis=2) for x in (query, key))
    query, key, value, gv, beta = (x.astype(work) for x in (query, key, value, gv, beta))
    query = query * (key.shape[-1] ** -0.5 if scale is None else scale)

    def one_token(h, step):
        h = jnp.where(step['start'][:, None, None, None], 0.0, h) * accurate_exp(step['g'])[:, :, None, :]
        erased = step['k'] - jnp.einsum('bhkv,bhv->bhk', h, step['v'])
        h = h + erased[..., :, None] * (step['beta'][..., None] * step['v'])[..., None, :]
        return h, jnp.einsum('bhk,bhkv->bhv', step['q'], h)

    if state is None:
        state = jnp.zeros((*value.shape[:1], value.shape[2], key.shape[-1], value.shape[-1]), work)
    starts = jnp.zeros(query.shape[:2], bool) if starts is None else jnp.asarray(starts, bool)
    state, out = jax.lax.scan(
        one_token, state.astype(work),
        {name: jnp.moveaxis(x, 1, 0) for name, x in
         (('q', query), ('k', key), ('v', value), ('g', gv), ('beta', beta), ('start', starts))})
    return jnp.moveaxis(out, 0, 1).astype(dtype), state


def _walk(forms, width: int, recomputed: bool, held: int = 0, documents: int = 0, added: bool = False,
          turned: bool = False):
    """`xla_mainloop`'s walk over every row's and head's chunks (`[rows, heads, chunks, C, ...]` operands and
    the memory entering), by the token update's derived `forms` (`dew.algebra.scan`): each chunk's values
    that read no memory at once, the memory alone in sequence, then each chunk's outputs from the memory
    entering it at once; its derivative the token update's replayed per chunk (algebra.md §6.9), each chunk's
    adjoint from the memory entering it. That memory is the backward's one checkpoint a chunk, kept but where
    the block's remat rung recomputes the delta rule's memory (`recomputed`, `recomputed_memory`, read as the
    call is traced: a scan differentiates its body's custom rules after the block's trace has left it), where
    the adjoint forms it again, as the port and the template drop theirs.

    A forked call's walk (`h` `(given [S', Dk, Dv], loads, stores and documents [chunks, C], entry?)`,
    `Forked.tokens`) walks its documents in place and holds a table of `held` stored memories beside the
    carry (`forked_steps`): a document's first token enters its table slot's memory (`table_read`, with its
    entry write added) wherever it falls in its chunk, and its last leaves its memory in its stored slot
    (`table_stored`), which a later document takes up, of the same chunk or another. Backwards, each chunk's
    token walk with its hooks is differentiated (`forked_adjoint`): a stored slot's cotangent gathers each
    document that took it up and joins the walk's at the token that stored it; a given memory's is its own,
    the walk's last memory's cotangent the row's. Its sizes are the tokens' and the table's, none the
    documents' bound's but the operands' own."""
    def entered(operands, h, segments):
        memory, forked = taken_up(h, added, turned)
        # The walk's decay in the forms' own carry dtype (`Derived.init`'s, the state's float32 at least).
        empty = forms.init({})[0]
        decayed = varying_like(jnp.zeros((width,), empty.dtype), *jax.tree.leaves(operands))
        if forked is not None:
            outputs, entering, last, table = forked_steps(forms, operands, memory, forked, held, documents,
                                                           decayed, segments)
            return outputs, last, (entering, table)
        prepared = jax.vmap(forms.prepare)(operands)

        def one(carried, chunk):
            leaving, kept = forms.step(carried, chunk)
            return leaving, (carried, kept)

        # Each chunk's state solved once, in sequence, what it kept read again by its outputs at once.
        carried, (entering, kept) = jax.lax.scan(one, (decayed, memory), prepared)
        return jax.vmap(forms.mapped)(prepared, entering, kept), carried[1], entering[1]

    def walking_one(operands, h, segments=None):
        outputs, last, kept = entered(operands, h, segments)
        return (outputs, last), None if recomputed else kept

    def replayed_one(operands, h):
        # The walk token by token (`Derived.replay`), the derivative's arithmetic at every order.
        memory, forked = taken_up(h, added, turned)
        if forked is None:
            def one(h, chunk):
                outputs, h = forms.replay(chunk, h)
                return h, outputs

            h, outputs = jax.lax.scan(one, memory, operands)
            return outputs, h
        return forked_replay(forms.replay, operands, memory, forked, held)

    def adjoint_one(operands, h, kept, d_outputs, d_final, segments=None):
        memory, forked = taken_up(h, added, turned)
        if kept is None:
            kept = entered(operands, h, segments)[2]
        if forked is None:
            def one(d_h, chunk):
                chunk, h, d_out = chunk
                return forms.backward(chunk, (None, h), d_h, d_out)

            d_h, d_operands = jax.lax.scan(one, d_final, (operands, kept, d_outputs), reverse=True)
            return d_operands, d_h
        entering, table = kept
        return forked_adjoint(forms.backward, operands, memory, forked, entering, table, d_outputs, d_final)

    def each(function):
        # Over every row and head at once: the derivative's rule outside the batch, which a second order
        # transposes as one map.
        return jax.vmap(jax.vmap(function))

    def walking(operands, h):
        # Every row's most segments outside the rows' map, the walk's skip a conditional, not a select.
        segments = most_segments(h, added, turned)
        return each(functools.partial(walking_one, segments=segments))(operands, h)

    def adjoint(entering, primals, cotangent, live):
        operands, h = primals
        segments = most_segments(h, added, turned)
        d_operands, d_h = each(functools.partial(adjoint_one, segments=segments))(operands, h, entering,
                                                                                    *cotangent)
        # A reset's booleans take no cotangent (never `live`), held in their place.
        whole = jax.tree.leaves(({**operands, **d_operands}, d_h))
        return tuple(whole[index] for index in live)

    return contracted("test_the_chunked_delta_rule_holds_every_order")(
        generation.linear_tangent(walking, adjoint, each(replayed_one), kind=Protected(),
                                  live_cotangents=True))


def port_kernel(query, key, value, g, beta, state=None, starts=None, *, max_segments: int | None = None,
                scale: float | None = None, wide: bool = False,
                bound: float | None = None, product: str = 'forward', report: bool = True, delta: bool = True,
                decayed: bool = True, forked: Forked | None = None, skip=None, rounded: bool | None = None,
                firsts=None, readout: bool = True, apart: int = 0, values_gated: bool = False,
                unit_pairs: bool = False,
                decay_bound: float | None = None, turns: int = 0, turn=None, transition: bool = False,
                reads=None) -> str:
    """The kernel the delta rule's selector (`GATED_DELTA`) gives a call over these operands for `product`
    ('forward', 'training'), of the form `delta` and `decayed` state and the chunk inverse's `bound` its form
    proves (`recurrence.Hooks.inverse_bound`; `PortCase`), the template's
    implementations bound to it, its `skip` among them where it takes one (`chunk`: a form without the delta
    rule's), on the target (`generation`'s, an assumed one's within `assumed`), at the call's numbers
    `rounded` (`chunk_rounding`'s where None, read here; a call passes its own, read once at it, so a
    selection made later, a remat's pullback's under its caller's precision, reads the call's), its write
    gated on the values where `values_gated` (fla's simple GLA, `mainloop.Call.values_gated`): reported as its
    route, or not where not `report` (`selected`). A kernel a call names
    (`generation.chosen`) insists on it (`selection.KernelRefused`: the port named over varying operands),
    but for a form the rule's kernels do not run, which the portable math takes; `readout` whether the
    call's readout reads the state after its token's update alone (`PortCase.readout`) or after it less its
    own write (`apart`, which its call to the template carries), `turns` the key pairs its state turns
    (`PortCase.turns`, `recurrence.Hooks.turns`) and `turn` its turns, through which a turned call's launch
    binds its operands (`_turned_launch`); `reads` the reads its keys erase along where given
    (`PortCase.reads`), which the template's call binds."""
    dtype = jnp.dtype(jnp.result_type(query, key, value))
    rounded = chunk_rounding(delta, dtype) if rounded is None else rounded
    call, shards = None, (None, key, value)
    if not transition:
        # A transition of its own binds no template call: no kernel but the portable math forms it. A turned
        # call's as its launch binds them (`_turned_launch`).
        launched = (query, key, forked)
        if turns:
            launched = _turned_launch(query, key, value, state, turn, starts, forked)
        call, shards = template_bind(
            launched[0], launched[1], value, g, beta, state, starts, firsts, skip, launched[2],
            max_segments=max_segments, scale=scale, delta=delta, wide=wide, bound=bound, rounded=rounded,
            apart=apart, values_gated=values_gated, unit_pairs=unit_pairs, decay_bound=decay_bound,
            reads=reads)
    _, shard_key, shard_value, *_ = shards
    if turns and call is not None:
        # The port's case reads the call's own key heads a shard, as an unturned call's are laid (a turned
        # launch's are a value head's each).
        (shard_key,) = shard_shapes((key,), (_shard_specs(value.shape, key.shape[2], g.ndim)[0],))
    family = generation.family(generation.device_generation()) if generation.backend() == 'gpu' else None
    case = PortCase(bound=bound, varying=bool(jax.typeof(value).mat.varying), dtype=dtype,
                    key_width=key.shape[-1], heads=(shard_key.shape[-2], *shard_value.shape[-2:]),
                    extent=Extent(shard_value.shape[0], shard_value.shape[1],
                                  starts is not None or firsts is not None),
                    call=call, family=family, product=product, delta=delta, decayed=decayed,
                    per_key=g.ndim == 4, forked=forked is not None, readout=readout, apart=apart, turns=turns,
                    transition=transition, reads=reads is not None)
    return select(GATED_DELTA, case).kernel if report else selected(GATED_DELTA, case).name


def _turned_launch(query, key, value, state, turn, starts, forked: Forked | None) -> tuple:
    """A turned call's operands as its launch binds them, `(query, key, forked)` as shapes: its queries and
    keys a value head's each and in its documents' frames, a forked call's entry keys with them and its load
    rotation its documents' (`Forked.entry`, `Forked.turning`), formed by the functions its launch's
    operands are (`_value_headed`, `turned.bound`) under `jax.eval_shape`, so the call a selector certifies
    (`template_bind`) is the launch that runs. Restated by hand beside them, the two drifted: the selector
    certified a launch the step never ran (rows 7976, 12402, 12478)."""
    from .templates.recurrence import turned

    def bound(query, key, value, state, turn, starts, forked):
        entry = None if forked is None else forked.entry
        query, key, entry = _value_headed(query, key, value, entry)
        documents = None if forked is None else (forked.firsts, forked.loads, forked.stores)
        query, key, _, entry, turning = turned.bound(query, key, value, state, entry, turn, starts, documents)
        if forked is not None:
            forked = dataclasses.replace(forked, entry=entry, turning=turning)
        return query, key, forked

    return jax.eval_shape(bound, query, key, value, state, turn, starts, forked)


def _portable(hooks: recurrence.Hooks, tiles: Tiles, query, key, value, g, beta, state, starts, firsts, *,
              scale: float, dtype, forked, own, correction=None, across=None, erase_left=None):
    """The portable math's unturned chunked op (`xla_mainloop`) under `hooks`' readout, an erase's
    `correction` (along its own vector `erase_left` where given) and a decay along the values `across` its
    own: a readout apart from
    its group's writes (`Hooks.reads` 'apart', the group a token's `Hooks.ranks`) the scan's own
    (`scan.Scan.group`: each rank reads the state after its token's decay less every rank's write of its
    token, no write read and taken back off), its walk's chunk the largest multiple of the group within the
    call's, so no chunk cuts a group (a rank-three call's 64 tokens, 63). A turned call's turns are
    `_turned_or_plain`'s."""
    group = hooks.group
    if tiles.chunk % group:
        tiles = dataclasses.replace(tiles, chunk=max(group, tiles.chunk // group * group))
    if hooks.erase == 'values':
        # Oja's rule: on the transposed memory the delta rule of keys `v`, values `k` and a decay along its
        # keys, read along its values (`_delta_forms`' `values_read`), each key head's over its value heads.
        _, _, query, key = _grouped(query, key, value, dtype)
        if forked is not None and forked.entry is not None:
            # A document's entry write `w sum_r k_r v_r^T` on the transposed memory, `w sum_r v_r k_r^T`.
            entry_key, entry_value, weight = forked.entry
            entry_key = jnp.repeat(entry_key, value.shape[2] // entry_key.shape[3], axis=3)
            forked = dataclasses.replace(forked, entry=(entry_value, entry_key, weight))
        out, final = xla_mainloop(tiles, query, value, key, g, beta,
                                  None if state is None else jnp.swapaxes(state, -1, -2), starts, firsts,
                                  scale=scale, delta=True, dtype=dtype, forked=forked, values_read=True)
        return out, jnp.swapaxes(final, -1, -2)
    return xla_mainloop(tiles, query, key, value, g, beta, state, starts, firsts, scale=scale,
                        delta=hooks.delta, dtype=dtype, bound=hooks.inverse_bound, forked=forked,
                        reads=hooks.reads, own=own, group=group, correction=correction, erase_left=erase_left,
                        across=across)


def _read_once(value, work) -> tuple[jax.Array, jax.Array]:
    """`value` read once in the working precision `work` and in its own dtype from that one read: where a
    call reads its values twice, its recurrence's kernel and the skip's `D v` (`recurrence.skipped`), their
    gradients sum in `work` and round to the values' dtype once. Each widening a 16-bit value apart, the two
    rounded apart and added in it: a write gate's value gradient's worst row was 11.6 times the token
    recurrence's on an A100 and a CPU alike, equal to it read once, its forward bitwise the same (row 5447,
    M-MC-6-base-a100)."""
    wide = value.astype(work)
    return wide, wide.astype(value.dtype)


def _portable_training(kernel_of: Callable[[str], str]) -> bool:
    """Whether a call's training product is the portable math, asked uncertified at the call (a template's
    eligibility certifies in reverse mode alone, `Template.holds`, and certification only narrows the
    portable math's): not where a kernel the call names refuses it, which refuses where a gradient selects
    it again (`_per_product`'s `training`)."""
    try:
        return kernel_of('training') == 'xla'
    except KernelRefused:
        return False


def _per_product(kernel_of: Callable[[str], str], run, *operands, parameters: tuple[int, ...]):
    """`run(kernel, *operands)` on the kernel a selector gives each product (`kernel_of('forward')`,
    `kernel_of('training')`), differentiated at every order through the portable math (`run('xla', ...)`,
    `generation.linear_tangent`): the forward's kernel where no derivative is taken, the training kernel's
    forward and its own pullback in reverse mode, and the portable tangent, in the kernels' dtypes, in
    forward mode however a jit, a scan or a remat staged the call, where a kernel's custom VJP gives none
    (row 2395). The training kernel's forward runs as it is (not `nested`), so the residuals its port names
    (`dew.nn.residuals.DELTA_RESIDUALS`) are a remat policy's to keep. The cells' products differ
    (`PORT_CELLS`, `SCAN_CELLS`, or a template that trains no such call), and a trace cannot tell them apart
    before its gradient is asked. A training kernel that refused the call refuses where a gradient is
    traced, not a forward alone. Where both products are the portable math's (a CPU's, or a device no
    kernel takes), the call is the portable math itself, whose own derivatives hold every order: traced
    through the products' split it traced the portable math three times over, half of a gated delta net
    layer's training step's trace on a CPU (row 3142). The operands at `parameters` are the call's
    parameters (a skip's `D`, a rate, A), whose gradients reverse mode forms with the others' cotangents
    (`derivatives.with_gradients_formed_at`)."""
    forward = kernel_of('forward')
    if forward == 'xla' and _portable_training(kernel_of):
        return run(forward, *operands)

    def training(*operands):
        # Selected and certified as reverse mode runs it, its forward and pullback (`launch_report.pulled`),
        # which only a gradient calls, so a trace no gradient reaches (a prefill, a generation) certifies no
        # training launch (train 934's F4); selected outside it, eligibility certified the forward alone, took
        # the A100's gated delta net for its training, and raised Exhausted inside the step's trace (train
        # 929's F1). The call's numbers and named kernel are its own (`kernel_of`'s, read at the call): a
        # remat's pullback runs here under its caller's precision and choices. A kernel the call names that
        # refuses it refuses here, where a gradient is traced, not in a forward alone.
        with launch_report.pulled():
            return run(kernel_of('training'), *operands)

    primal = functools.partial(run, forward)

    def portable(*operands):
        shapes = jax.eval_shape(primal, *operands)
        return jax.tree.map(lambda x, shape: x.astype(shape.dtype), run('xla', *operands), shapes)

    def pulled(pullback, operands, cotangent):
        del operands
        return with_gradients_formed_at(tuple(pullback(cotangent)), parameters)

    return contracted("test_the_chunked_op_holds_every_order")(
        generation.linear_tangent(functools.partial(jax.vjp, training), pulled, portable, primal=primal,
                                  nested=False, kind=KernelGlue()))(*operands)


def entering_memory(rule, query, key, value, g, beta, starts, *, axis: str, chunk_size: int, state=None):
    """The memory this shard of a sequence the mesh axis `axis` splits enters its slice with, `[B, Hv, Dk,
    Dv]` in at least fp32, inside a `shard_map` that holds `axis` manual: fla's context-parallel
    summary-then-states (fla/ops/cp/chunk_delta_h.py, fla-core 0.5.2). A delta rule's slice is affine in
    the memory it enters, `S = F S_0 + Z` (each token's transition on the left, `recurrence.update`), so
    each shard's summary is its slice's affine map: `Z` the write its last document leaves from zero,
    `rule`'s final memory over its slice from zero in the layer's own operands (at the pass's own distance
    from float64), and `F` its transfer (`transferred`), zero where that document starts inside the slice.
    The summaries are each shard's span (`recurrence_algebra.Span`), whose exclusive prefix over `axis`
    (`recurrence_algebra.entering`) from the row's `state` (zero where None) on the first shard is the memory
    each enters, `E_r = F_{r-1} E_{r-1} + Z_{r-1}`; a slice that starts a document (its first token's
    `starts`) enters from zero. `rule(query, key, value, g, beta, state, *, starts)` returns `(output, final
    memory)`; `chunk_size` its chunk, the transfer's too. Taken as `rule`'s final memory from `[0 | I]`
    with values `[v | 0]` in fp32 operands, the summary cost 6.2 to 6.7 times its pass (M-JS-142) and the
    27B's 9.99 ms on a GH200 at 4096 tokens against 2.15 for the layer's bf16 operands, whose transfer,
    though, rounded its identity block's products to bf16 at 3.5 times the pass's distance from float64
    (M-JS-143c)."""
    batch, _, heads, columns = value.shape
    width = key.shape[-1]
    wide = at_least_fp32(value.dtype)
    shard = jax.lax.axis_index(axis)
    begins = (jnp.zeros((batch,), bool) if starts is None else jnp.asarray(starts[:, 0], bool)) & (shard > 0)
    zero = rows_like(jnp.zeros((batch, heads, width, columns), wide), value)
    _, write = rule(query, key, value, g, beta, zero, starts=starts)
    span = algebra.Span(transferred(query, key, value, g, beta, starts, chunk_size=chunk_size),
                        write.astype(wide))
    memory = algebra.entering(span, axis, None if state is None else state.astype(wide))
    return jnp.where(begins[:, None, None, None], 0.0, memory)


def transferred(query, key, value, g, beta, starts=None, *, chunk_size: int = CHUNK_SIZE) -> jax.Array:
    """The delta rule's transfer over a slice of its tokens, `[B, Hv, Dk, Dk]` in at least fp32: the linear
    map `F` the slice's tokens apply to the memory entering it (`S_out = F S_in + Z`), zero where a document
    starts in it (`starts`), each chunk's transition chained onto the chunks' before it in fp32 from the
    operands as they are given (`scan.derive_scan`'s walk carrying its transfer: `prepare(...,
    transition=True)`, `step`), as fla-core 0.5.2's context parallelism chains its transfer from bf16
    `k^T w` (ops/cp/chunk_delta_h.py:304-315), no product's result rounded on the way, in the delta rule's
    chunks at the keys' width (`held_chunk`, the portable math's). The forms' memory is one value column wide
    and zero: the transfer reads no value. A slice of no tokens transfers the memory as it is (the
    identity)."""
    _, work, _, key = _grouped(query, key, value, None)
    rows, length, heads, width = key.shape
    if length == 0:
        identity = jnp.broadcast_to(jnp.eye(width, dtype=work), (rows, heads, width, width))
        return varying_like(rows_like(identity, value), key, g, beta, starts)
    chunk_size = held_chunk(chunk_size, width)
    forms = _delta_forms(delta=True, scale=1.0, per_key=g.ndim == 4, width=width, columns=1,
                         work=jnp.dtype(work), reset=starts is not None, bound=None)
    unread = jnp.zeros((*value.shape[:3], 1), work)
    operands = _chunked({"q": key, "k": key, "v": unread, "g": g, "beta": beta}, starts, chunk_size, work)
    varying = (key, g, beta, starts)

    def walked(chunks: dict) -> jax.Array:
        """One row's and head's transfer over its chunks `[chunks, C, ...]`."""
        start = jax.tree.map(lambda x: varying_like(x, *varying),
                             forms.init(jax.tree.map(lambda x: x[0], chunks)))
        prepared = jax.vmap(functools.partial(forms.prepare, transition=True))(chunks)
        last, _ = jax.lax.scan(lambda walk, chunk: (forms.step(walk, chunk)[0], None), start, prepared)
        return last.transfer

    return jax.vmap(jax.vmap(walked))(operands)


def _unit_pairs(hooks: recurrence.Hooks) -> bool:
    """Whether `hooks` bound every own pair's operand at most one in size, which the template's own pairs may
    fold their decays against (`mainloop.Call.unit_pairs`): keys of unit norm (the op's `l2norm`, or as
    `write_bound` declares: `Hooks.inverse_bound`'s contract), queries so (`l2norm`, or as `unit_queries`
    declares), and under the delta rule each system row's write gate at most one (`write_bound`). Keys alone
    bounded, a query of 2^100 times a key of 2^-100 decayed by 2^-60 lost its term, the decayed key 2^-160
    rounding to zero (#3703 review)."""
    keys = hooks.l2norm or hooks.write_bound is not None
    gates = not hooks.delta or (hooks.write_bound is not None and hooks.write_bound <= 1)
    return keys and (hooks.l2norm or hooks.unit_queries) and gates


def kernel_routed(query, key, value, g, beta, *, write_bound: float | None = None,
                  decay_bound: float | None = None, unit_queries: bool = False) -> bool:
    """Whether the delta rule's selector gives a call over these operands a kernel, its forward's
    (`port_kernel`, unreported: the call reports its own), under the layer's declared `write_bound`,
    `decay_bound` and `unit_queries` (`chunk_gated_delta_rule`'s): what a layer rounds its operands for, the
    projections' dtype for a kernel, as fla's layer hands them over, the wide one for the portable math."""
    hooks = recurrence.Hooks(write_bound=write_bound, decay_bound=decay_bound, unit_queries=unit_queries)
    return port_kernel(query, key, value, g, beta, bound=hooks.inverse_bound,
                       decay_bound=hooks.decay_bound if g.ndim == 4 else None, report=False) != 'xla'


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
    if state is not None and state.ndim == 5:
        # A forked call's table (`Chunk.loads`) handed here, refused as the selector refuses the call.
        raise KernelRefused(GATED_DELTA.choice, 'fla', PORT_FORKED, iter(()))
    dtype = jnp.result_type(query, key, value)
    query, key, value, beta = (x.astype(dtype) for x in (query, key, value, beta))
    g = g.astype(jnp.float32)
    B, _, Hv, Dv = value.shape
    state = (jnp.zeros((B, Hv, key.shape[-1], Dv), jnp.float32) if state is None
             else state.astype(jnp.float32))
    segments = None if starts is None else query.shape[1] if max_segments is None else max_segments
    starts = jnp.zeros(query.shape[:2], bool) if starts is None else jnp.asarray(starts, bool)
    four, three, memory, decays, resets = _shard_specs(value.shape, key.shape[2], g.ndim)
    if segments is not None:
        # The kernels start each row from its memory; a first token that begins a document drops it.
        state = jnp.where(starts[:, 0, None, None, None], 0.0, state)
    if firsts is None:
        return manual_map(functools.partial(_blocked_rule, segments=segments, scale=scale, wide=wide),
                          (four, four, four, decays, three, memory, resets),
                          (four, memory))(query, key, value, g, beta, state, starts)
    return manual_map(functools.partial(_blocked_rule, segments=segments, scale=scale, wide=wide),
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


def template_rule(query, key, value, g, beta, state=None, starts=None, firsts=None, skip=None,
                  forked: Forked | None = None, reads=None, **options):
    """The recurrence template's op (`templates.recurrence.template.MOSAIC`, its `mosaic_rule`) on each
    shard (`manual_map`), placed as `fla_rule`: its tiles derived from each shard's shapes, its forward and
    its gradient on each device's own rows. Called on the whole batch, every device of a 2xH100 FSDP run ran
    its launches on the global batch, layer 0's solve over 4 rows for its 2, beside 40 GB of gathers a
    microbatch (row 1897). `options` are the op's (`max_segments`, `scale`, `delta`, `wide`, `bound`,
    `rounded`); whether the block's remat rung recomputes the delta rule's memory is resolved as
    the selectors' binds resolve it (`recomputed_memory`). A `forked` call's documents (`Chunk.loads`), its
    table's given memories `state` `[B, S', Hv, Dk, Dv]`, are each shard's rows'; `reads` the keys'."""
    from .templates.recurrence import template

    operands, specs, out_specs, bound = _template_launch(query, key, value, g, beta, state, starts, firsts,
                                                         skip, forked, reads, **options)

    def local(*given):
        arguments, keywords = bound(*given)
        return template.MOSAIC(*arguments, **keywords)

    return manual_map(local, specs, out_specs)(*operands)


def template_bind(query, key, value, g, beta, state=None, starts=None, firsts=None, skip=None,
                  forked: Forked | None = None, reads=None, **options) -> tuple[base.Call, tuple]:
    """The template's call a selector reads (`port_kernel`, `scan_kernel`), bound as `template_rule` binds
    its launch on each shard (`_template_launch`): each shard's shapes, its documents' firsts and its
    forked call's carried tables, so the launch its eligibility certifies (`Template.holds`) is the one the
    call runs; and the operands as each shard holds them (`shard_shapes`, the queries, keys and values
    first), as the ports lay theirs too (`_shard_specs`, `_ssd_ported`), which a declared cell keys on. Bound
    to the whole batch, without its firsts and with the whole fork, every packed or sharded training step
    certified a launch it never ran beside the one it did (train 942's F2)."""
    from .templates.recurrence import template

    operands, specs, _, bound = _template_launch(query, key, value, g, beta, state, starts, firsts, skip,
                                                 forked, reads, **options)
    shards = shard_shapes(operands, specs)
    arguments, keywords = bound(*shards)
    return base.call((template.MOSAIC,), *arguments, **keywords), shards


def _launched_template(context, query, key, value, g, beta, state, skip, entry, turning, **options):
    """`template_rule`'s `(output, final memory)` with its operands first (`turned.call`'s launch), its call's
    `context` `(starts, firsts, forked)` an operand too, a forked call's entry write `entry` and load rotation
    `turning` its documents' (`Forked.entry`, `Forked.turning`)."""
    starts, firsts, forked = context
    if forked is not None:
        forked = dataclasses.replace(forked, entry=entry, turning=turning)
    return template_rule(query, key, value, g, beta, state, starts, firsts, skip, forked=forked, **options)


def _on_template(hooks: recurrence.Hooks, query, key, value, g, beta, state, skip, turn, *, starts, firsts,
                 forked, scale: float, **options):
    """The chunked op's `(output, final memory)` on the template (`template_rule`, a turned call's through
    `_turned_or_plain`): its launches' readout apart (`Hooks.apart`) and its chunks' readout group whole in
    each (`Hooks.group`) one hooks' own, so the launch's mask and the turned frame cannot disagree."""
    launch = functools.partial(_launched_template, scale=scale, apart=hooks.apart, delta=hooks.delta,
                               bound=hooks.inverse_bound, **options)
    return _turned_or_plain(launch, query, key, value, g, beta, state, skip, turn, starts=starts,
                            firsts=firsts, forked=forked, scale=scale, group=hooks.group)


def _on_portable(hooks: recurrence.Hooks, query, key, value, g, beta, state, turn, *, starts, firsts, forked,
                 scale: float, dtype, own, chunk_size: int, correction=None, across=None, erase_left=None):
    """The chunked op's `(output, final memory)` on the portable math (`_portable`, its chunk `held_chunk`'s
    under the delta rule; a turned call's through `_turned_or_plain`; an erase of its own's (`correction`,
    `erase_left`, `chunk`'s), a decay along the values' (`across`) and Oja's rule's their own, which turn
    none)."""
    # Oja's rule's keys on its transposed memory are its values (`_portable`).
    width = value.shape[-1] if hooks.erase == 'values' else key.shape[-1]
    size = held_chunk(chunk_size, width) if hooks.delta else chunk_size
    if correction is not None and correction.shape[2] != value.shape[2]:
        # A read erase's reads each key head's (`_erases_laid`), its correction each value head's.
        correction = jnp.repeat(correction, value.shape[2] // correction.shape[2], axis=2)
    if correction is not None or across is not None or hooks.erase == 'values':
        return _portable(hooks, Tiles(size), query, key, value, g, beta, state, starts, firsts, scale=scale,
                         dtype=dtype, forked=forked, own=own, correction=correction, across=across,
                         erase_left=erase_left)
    launch = functools.partial(_portable_launch, hooks, Tiles(size), scale=scale, dtype=dtype, own=own)
    return _turned_or_plain(launch, query, key, value, g, beta, state, None, turn, starts=starts,
                            firsts=firsts, forked=forked, scale=scale, group=hooks.group)


def _portable_launch(hooks: recurrence.Hooks, tiles: Tiles, context, query, key, value, g, beta, state, skip,
                     entry, turning, *, scale: float, dtype, own):
    """`_portable`'s `(output, final memory)` with its operands first (`turned.call`'s launch), its call's
    `context` `(starts, firsts, forked)` an operand too, a forked call's entry write `entry` and load rotation
    `turning` its documents' (`Forked.entry`, `Forked.turning`), the skip the caller's."""
    del skip
    starts, firsts, forked = context
    if forked is not None:
        forked = dataclasses.replace(forked, entry=entry, turning=turning)
    return _portable(hooks, tiles, query, key, value, g, beta, state, starts, firsts, scale=scale,
                     dtype=dtype, forked=forked, own=own)


def _turned_or_plain(launch, query, key, value, g, beta, state, skip, turn, *, starts, firsts, forked,
                     scale: float, group: int):
    """`launch(context, query, key, value, g, beta, state, skip, entry, turning)`'s `(output, final
    memory)`, its call's arrays `context` `(starts, firsts, forked)` passed in, never closed over (a turned
    call's rule a `generation.linear_tangent`, whose maps take no traced value but their operands): a
    kernel's unturned chunked op, a forked call's entry write its documents' (`Forked.entry`); a turned
    call's (`turn` given, `Hooks.turns`) every kernel's alike, its keys and queries turned into its
    documents' frames, a value head's each (its turns'), a forked one's documents' entry keys too, and its
    turns' gradient in the crossing form (`templates.recurrence.turned`): the kernel's own speed and its
    turns' gradient exact."""
    from .templates.recurrence import turned

    entry = None if forked is None else forked.entry
    if turn is None:
        return launch((starts, firsts, forked), query, key, value, g, beta, state, skip, entry, None)
    if forked is not None and forked.adds is not None:
        raise ValueError("a turned call's documents add no second slot (`Chunk.adds`, row 8284): its turns' "
                         "gradient composes each document's frames from the one slot it loads "
                         "(`turned._forked_frames`)")
    query, key, entry = _value_headed(query, key, value, entry)
    documents = None if forked is None else (forked.firsts, forked.loads, forked.stores)
    held = 0 if forked is None else forked.held
    # The call's arrays its launch reads, the entry write and rotation its own operands (`turned.call`'s).
    unrotated = None if forked is None else dataclasses.replace(forked, entry=None, turning=None)
    context = (starts, firsts, unrotated)
    return turned.call(launch, query, key, value, g, beta, state, skip, turn, starts, scale=scale,
                       group=group, documents=documents, held=held, entry=entry, context=context)


def _value_headed(query, key, value, entry) -> tuple:
    """A turned call's queries, keys and entry keys a value head's each (its turns', `_turned_or_plain`)."""
    heads = value.shape[2]
    query, key = (jnp.repeat(t, heads // t.shape[2], axis=2) for t in (query, key))
    if entry is not None:
        entry = (jnp.repeat(entry[0], heads // entry[0].shape[3], axis=3), *entry[1:])
    return query, key, entry


def _template_launch(query, key, value, g, beta, state, starts, firsts, skip, forked: Forked | None, reads,
                     **options) -> tuple:
    """The template's launch on each shard: `(operands, specs, out_specs, bound)`, its operands in order with
    their placements (`_shard_specs`), its outputs', and `bound(*local)` the op's arguments and keywords
    from each shard's operands (`template_rule`'s launch and `template_bind`'s alike)."""
    four, three, memory, decays, resets = _shard_specs(value.shape, key.shape[2], g.ndim)
    P = jax.sharding.PartitionSpec
    walked = ()
    if forked is not None:
        # The table's memories a row's and head's, and the documents' operands as `Forked` places them.
        memory = P(four[0], None, four[2], None, None)
        walked = forked.specs(four[0], four[2])
    optional = [(state, memory), (starts, resets), (firsts, resets), (skip, P(four[2])), (reads, four)]
    operands = (query, key, value, g, beta, *(x for x, _ in optional if x is not None),
                *(() if forked is None else forked.walked))
    specs = (four, four, four, decays, three, *(spec for x, spec in optional if x is not None),
             *walked)

    def bound(query, key, value, g, beta, *given):
        given = iter(given)
        state, starts, firsts, skip, reads = (None if x is None else next(given) for x, _ in optional)
        return (query, key, value, g, beta, state, starts), dict(
            firsts=firsts, skip=skip, forked=None if forked is None else forked.carrying(*given),
            recomputed=recomputed_memory(), reads=reads, **options)

    return operands, specs, (four, P(four[0], four[2], None, None)), bound


def _blocked_rule(query, key, value, g, beta, state, starts, firsts=None, *, segments: int | None,
                  scale: float | None = None, wide: bool = False):
    """The port's rule on one shard, shaped to its kernels' blocks: values
    wider than the port's widest split into more value heads of their key
    head, each key head's value heads as many as it serves (dqkwg's block
    holds them, loaded a head at a time, so no count need be a power of
    two); keys and values padded with zero columns to 64, 128 or 256,
    widths every block of the kernels' configs divides, a decay per key
    dimension with them. None of them changes a real output or gradient:
    every padded product is an exact zero. The outputs are cut back. Where
    `segments` bounds a row's packed documents, each begun at `starts`, each
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
        value = zero_padded(value, 3, split * width).reshape(B, S, Hv * split, width)
        g, beta = (jnp.repeat(x, split, axis=2) for x in (g, beta))
        state = jnp.moveaxis(zero_padded(state, 3, split * width).reshape(B, Hv, Dk, split, width),
                             3, 2).reshape(B, Hv * split, Dk, width)
    blocks = (max(64, 1 << (Dk - 1).bit_length()), max(64, 1 << (value.shape[3] - 1).bit_length()))
    query, key = (zero_padded(x, 3, blocks[0]) for x in (query, key))
    per_key = g.ndim == 4
    g = zero_padded(g, 3, blocks[0]) if per_key else g
    value = zero_padded(value, 3, blocks[1])
    state = zero_padded(zero_padded(state, 2, blocks[0]), 3, blocks[1])
    # A row's padding after its documents runs as one more sequence of its own.
    spans = None if segments is None else cu_seqlens(starts, B * (segments + 1), firsts)
    port = fla_kda.chunk_kda if per_key else fla_gdn.chunk_gated_delta_rule
    out, final = port(query, key, value, g, beta, state, spans, scale, CHUNK_SIZE,
                      not generation.triton_runs(), not recomputed_memory(), wide)
    out = out[..., :width].reshape(B, S, Hv, split * width)[..., :Dv]
    final = final[:, :, :Dk, :width]
    final = jnp.moveaxis(final.reshape(B, Hv, split, Dk, width), 2, 3).reshape(B, Hv, Dk, split * width)
    return out, final[..., :Dv]


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
        out, s = recurrence.update(s, step['q'], step['k'], step['v'], accurate_exp(step['g']),
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
                chunk_size: int, work, dtype, *, max_segments: int | None = None,
                forked: Forked | None = None):
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
    the earlier shards' state it adds), and the memory `[B, Hv, Dk, Dv]`; a forked call (`forked`,
    `Chunk.loads`) walks its documents in place on the portable math and the TPU's scan, each taking its
    memory up at its first token (`_tpu_documents`), their table's given memories `state` `[B, S', Hv, Dk,
    Dv]`; mamba_ssm's port takes none (`SSD_SUPPORTS`)."""
    rate = given['rate'].astype(work)
    skip = given['skip'].astype(work) if hooks.skip else jnp.zeros(rate.shape, work)
    # The skip the template's own (`chunk_forward.forward`) and its gradient's (`mainloop_backward._reverse`),
    # as mamba_ssm's kernels add it: under the fast forms in its output launch, else added to its output
    # after its launches where the template holds no 16-bit values ungated (`mainloop.Call.gated_keys`: the
    # accurate forms, or an fp32 call's two parts under the fast forms); the output in the operands' dtype
    # where the caller asks it so and no epilogue reads it after (`hooks.output`'s norm), else wide: added
    # after the template, its fp32 output's skip and cast took 0.045 ms past an H100's launches, and its
    # gradient's 0.094 (ssd-rounded-h100-fdab2424c, window-h100-c70aaa8-b8109ee). Decided once: the selector
    # judges the call the run makes (its skip's reverse tile).
    operand = jnp.dtype(jnp.result_type(query, key, value))
    fast = fast_rounding(operand)
    narrow = hooks.output is None and jnp.dtype(dtype) == operand
    taken = skip if hooks.skip else None

    def product_kernel(product: str) -> str:
        return scan_kernel(query, key, value, g, beta, state, starts, chunk_size=chunk_size,
                           max_segments=max_segments, product=product, skip=taken, wide=not narrow,
                           forked=forked, rounded=fast, firsts=firsts)

    def run(kernel, query, key, value, g, beta, rate, skip, state, starts, firsts, *walked):
        carried = None if forked is None else forked.carrying(*walked)
        return launched(kernel, query, key, value, g, beta, rate, skip, state, starts, firsts, carried)

    def launched(kernel, query, key, value, g, beta, rate, skip, state, starts, firsts, forked):
        if kernel == 'mosaic_gpu':
            out, final = template_rule(query, key, value, g, beta, state, starts, firsts,
                                       skip=skip if hooks.skip else None, forked=forked,
                                       max_segments=max_segments, scale=1.0, delta=False, wide=not narrow,
                                       rounded=fast)
            return out.astype(work), final
        operand = jnp.result_type(query, key, value)
        x, B, C = (jnp.asarray(t, operand) for t in (value, key, query))
        batch, _, heads, head_dim = x.shape
        state_size = B.shape[-1]
        if kernel in ('pallas', 'tpu'):
            # The SSD's memory is `[P, N]`, values by keys: the template's `[N, P]` transposed.
            columns, rows = _scan_widths(head_dim), _scan_widths(state_size)
            documents = None
            if forked is not None and state is not None:
                # The TPU's scan walks each document in place, taking its memory up at its first token (the
                # port takes no forked call, `SSD_SUPPORTS`).
                documents = _tpu_documents(state, forked, x.shape[1], chunk_size, work, (columns, rows))
                carried = rows_like(jnp.zeros((batch, heads, head_dim, state_size), work), x)
                starts = None
            else:
                carried = (rows_like(jnp.zeros((batch, heads, head_dim, state_size), work), x)
                           if state is None else jnp.swapaxes(state, -1, -2).astype(work))
            wide_x = zero_padded(x, 3, columns)
            wide_B, wide_C = zero_padded(B, 3, rows), zero_padded(C, 3, rows)
            wide_state = zero_padded(zero_padded(carried, 2, columns), 3, rows)
            if kernel == 'pallas':
                out, final = _ssd_ported(wide_x, beta, rate, wide_B, wide_C, skip, wide_state, chunk_size,
                                         starts)
            else:
                out, final = _ssd_tpu(wide_x, beta, rate, wide_B, wide_C, skip, wide_state, chunk_size,
                                      starts, scan_case(x, B, chunk_size, padded=True).tpu, work, documents,
                                      0 if forked is None else forked.held)
            out, final = out.astype(work)[..., :head_dim], final.astype(work)
            return out, jnp.swapaxes(final[..., :head_dim, :state_size], -1, -2)
        # The values widened once for the mainloop and the skip, their gradients summed before they round.
        wide = value.astype(work)
        out, final = xla_mainloop(Tiles(chunk_size), query, key, wide, g, beta, state, starts, firsts,
                                  scale=1.0, delta=False, dtype=work, forked=forked)
        return recurrence.skipped(hooks, out, wide, {'skip': skip}, work), final

    # The rate and the skip's `D` (operands 5 and 6) are the call's parameters.
    return _per_product(SCAN.bound(product_kernel), run, query, key, value, g, beta, rate, skip, state,
                        starts, firsts, *(() if forked is None else forked.walked), parameters=(5, 6))


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


def _ssd_tpu(x, dt, A, B, C, D, carried, chunk_size: int, starts, platform, work, documents=None,
             held: int = 0):
    """The rated scan on a TPU's Pallas kernel (`dew.nn.kernels.ssd`), on its operands in `work`
    (`ssd_operands`), per shard of rows and heads (`manual_map`). The skip `D x` is added after. Returns
    the output and the memory `[B, H, P, N]`; a forked call's documents walked in place (`documents`,
    `_tpu_documents`, `held` stored)."""
    length = x.shape[1]
    x, B, C = (jnp.asarray(t, work) for t in (x, B, C))
    skip = D[None, None, :, None] * x
    scaled, B, C, a = ssd_operands(x, dt, A, B, C, chunk_size, starts)

    def kernel(x, B, C, a, carried, documents):
        return _chunk_scan(x, B, C, a, carried, documents, chunk_size, platform, held)

    specs, out_specs = ssd_specs(scaled.shape, documents)
    scanned, final = manual_map(kernel, specs, out_specs)(scaled, B, C, a, carried, documents)
    return scanned[:, :length] + skip, final


def ssd_specs(shape: tuple[int, ...], documents=None) -> tuple[tuple, tuple]:
    """The TPU scan's per-shard placements (`manual_map`) over chunk-scan operands of `shape` `[B, T, H, P]`
    (`ssd_operands`'): its operands' `(x, B, C, a, memory, documents)` and its outputs' `(scanned, memory)`,
    each shard its own rows and heads; `_ssd_tpu`'s launch and the TPU scan's contract's."""
    # The groups are expanded, so any split of the heads divides them.
    rows, split = grouped_heads(shape, shape[2])
    four, state = P(rows, None, split, None), P(rows, split, None, None)
    specs = (four, four, four, P(rows, None, split), state, _tpu_specs(documents, rows, split))
    return specs, (four, state)


@dataclasses.dataclass(frozen=True)
class ScanCase:
    """A rated decay's scan as the scan selector reads it (`SCAN`): whether a caller's `shard_map` that checks
    varying axes holds operands that vary (`varying`), the operands' `dtype`, the port's head width, state
    width and chunk (`widths`, padded, `_scan_widths`), the platform the TPU's Pallas scan builds for, None
    where it does not take the scan (`ssd_kernel_platform`), and as the delta rule's case reads them
    (`PortCase`) the recurrence template's implementations bound to the call (`call`), the device's
    instruction family (`family`), the product (`product`) and whether it is forked (`forked`); its value
    heads and key groups on a launch's shard (`shard_heads`), its head and state widths as the call gives
    them (`logical`, unpadded) and its extent on that shard (`extent`); None where a caller reads no cell
    (`scan_case`'s `shards`); and whether its output is wide, at least fp32 (`wide`), or in the operands'
    dtype."""

    varying: bool
    dtype: np.dtype
    widths: tuple[int, int, int]
    tpu: str | None
    logical: tuple[int, int]
    wide: bool = True
    shard_heads: tuple[int, int] | None = None
    extent: Extent | None = None
    call: base.Call | None = None
    family: str | None = None
    product: str = 'forward'
    forked: bool = False
    readout: bool = True
    apart: int = 0
    turns: int = 0

    @property
    def cell(self) -> tuple[str | None, "ScanCell"]:
        """The call's cell (`SCAN_CELLS`): its device's instruction family and what a launch's shard
        computes (`ScanCell`, row 8991)."""
        (heads, groups), (width, state) = self.shard_heads or (None, None), self.logical
        return self.family, ScanCell(self.dtype.name, self.product, (heads, width), (groups, state),
                                     self.widths[2], self.extent, self.wide)


class ScanCell(NamedTuple):
    """A rated scan's cell a receipt declares (`ScanCase.cell`): its values' dtype, its product, its value
    heads and their width and its key groups and the state's width on a launch's shard, unpadded, its chunk,
    its extent: None where its case read no shard (`scan_case`'s `shards`), a cell none declares; and
    whether its output is wide (`ScanCase.wide`): an output in the operands' dtype writes and reads half the
    bytes, which a receipt timing a wide one did not measure (row 11729)."""

    dtype: str
    product: str
    values: tuple[int | None, int]
    keys: tuple[int | None, int]
    chunk: int
    extent: Extent | None
    wide: bool


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
    Requirement('precision', _port_precision, _PORT_PRECISION, row=INHERENT, kind='handoff'),
    Requirement('forked', lambda case: not case.forked, SSD_PORT_FORKED.detail, row=INHERENT),
)
"""What mamba_ssm's kernels (`mamba_ssd.ssd_scan`) need of a scan: no caller's `shard_map` that checks
varying axes over operands that vary,
where a Pallas call, which states no varying type for its results, cannot run; no float64 operands,
whose scan is the reference math itself; head widths, state widths and chunks that are powers of
two of at least 16, the smallest block Triton's dot takes (a call's head and state widths padded to them,
`_rated_scan`); a GPU Dew's Triton kernels run on, any of them; and no forked call."""

TPU_SCAN_SUPPORTS: tuple[Requirement[ScanCase], ...] = (
    Requirement('geometry', lambda case: case.tpu is not None,
                lambda case: "the TPU's Pallas scan takes a float32 scan on a TPU over power-of-two widths "
                             "and a chunk its program holds, not head width {}, state {} and chunk {}"
                             .format(*case.widths), row=INHERENT),
    # The scan widens its operands first (`_ssd_tpu`'s `work`): its products are fp32's.
    Requirement('precision', lambda case: exact_products(None, at_least_fp32(case.dtype)),
                "a dot algorithm the run declares that narrows the scan's fp32 products, which the TPU's "
                "scan takes whole (`ssd._mm`, HIGHEST)", row=INHERENT, kind='handoff'),
)
"""What the TPU's Pallas scan (`dew.nn.kernels.ssd`) needs: its tiling over the geometry
(`ssd_kernel_runs`), and a declaration its whole fp32 products compute (`exact_products`)."""


def scan_case(x, B, chunk_size: int, *, padded: bool = False, call: base.Call | None = None,
              product: str = 'forward', forked: bool = False, packed: bool = False, shards=None,
              dtype: jnp.dtype | None = None, wide: bool = True) -> ScanCase:
    """The `ScanCase` of a scan over values `x` `[B, S, H, P]` and keys `B` `[B, S, G, N]` in
    `chunk_size` chunks, their widths as the port and the TPU's scan take them where `padded`
    (`_scan_widths`, `_rated_scan`); on a TPU, the platform its Pallas scan builds for
    (`ssd_kernel_platform`); the template's `call`, the `product`, and whether its rows pack documents
    (`packed`); and where `shards` gives `x` and `B` as each shard holds them (`template_bind`'s, as the port
    lays its launches too, `_ssd_ported`), its shard's heads, groups and extent, a declared cell's key; in
    `dtype`, the operands' common one the scan launches in (`_rated_scan`'s), `x`'s where None; its output
    `wide` or in that dtype."""
    dtype = jnp.dtype(x.dtype if dtype is None else dtype)
    widths = (x.shape[-1], B.shape[-1], chunk_size)
    if padded:
        widths = (_scan_widths(widths[0]), _scan_widths(widths[1]), chunk_size)
    family = generation.family(generation.device_generation()) if generation.backend() == 'gpu' else None
    held = {} if shards is None else {'shard_heads': (shards[0].shape[2], shards[1].shape[2]),
                                      'extent': Extent(shards[0].shape[0], shards[0].shape[1], packed)}
    return ScanCase(varying=bool(jax.typeof(x).mat.varying), dtype=dtype, widths=widths,
                    tpu=(ssd_kernel_platform(chunk_size, widths[0], widths[1], dtype=at_least_fp32(dtype))
                         if generation.backend() == 'tpu' else None),
                    logical=(x.shape[-1], B.shape[-1]), call=call, family=family, product=product,
                    forked=forked, wide=wide, **held)


def fast_rounding(dtype) -> bool:
    """Whether a call of `dtype` on the template takes the run's fast forms' numbers (`rounded`), resolved at
    its route, so its launches and their caches key on it, never on the run's numerics read inside them: a
    16-bit call's under the run's fast forms (`generation.numerics`), the 16-bit roundings of its
    intermediates mamba_ssm's and fla's, its products whole at any precision; an fp32 call's products TF32
    (JAX's default precision for an fp32 product, and fla's) where the run's declaration lets a route round
    them (`dew.nn.precision.rounds_products`: its fast forms at no more than the default precision; a run at
    'highest' keeps an fp32 product whole, the template's accurate parts)."""
    if jnp.dtype(dtype).itemsize == 2:
        return generation.numerics().forms == 'fast'
    return rounds_products()


def simple_gla(hooks: recurrence.Hooks) -> bool:
    """Whether a form is fla's simple GLA: a decay a head, no delta rule, write gate, gates, skip or output
    epilogue (lightning attention's, `dew.nn.mixers.lightning.LIGHTNING`; fla 0.5.2's `chunk_simple_gla`,
    which its `chunk_lightning_attn` calls at a fixed decay, ops/lightning_attn/chunk.py:73-84)."""
    return hooks.decay == 'scalar' and not (hooks.delta or hooks.write_gate or hooks.rated or hooks.gates
                                            or hooks.skip or hooks.output)


def chunk_rounding(delta: bool, dtype, *, released: bool = False) -> bool:
    """Whether a call of `dtype` on the delta rule's route (`chunk`, `port_kernel`) takes the run's fast
    forms' numbers (`fast_rounding`): a form whose upstream release is its bar (docs/design/testing.md) at
    that release's rounding, the delta rule's, fla's (TF32 products in an fp32 call), and fla's simple GLA
    without it (`released`, `simple_gla`): its states and scores bf16 at each product, its sums fp32 (fla
    0.5.2 ops/common/chunk_h.py:84-151, chunk_o.py:88-139), where held whole in fp32's three parts lightning
    attention ran 4.9 times fla's forward and 2.6 times its training on a GH200 (row 6823, M-JS-129); under
    the accurate forms it keeps them whole, its bar the recurrence a token at a time. A form no release
    holds has no port, its bar the portable math at full fp32: an fp32 call's
    two bf16 parts put its training step 30 to 69 times the portable math's distance from float64 on every
    leaf, interpreted and on a GH200 alike (row 184's write gate cell, jsf-cellcpu, M-JS-9), and its forward
    57 to 77 times (test_recurrence_mosaic's hooks7, hooks12 and hooks26 and the step-gated packed cell), so
    it takes three; a 16-bit call's rounded numbers are mamba_ssm's, the SSD's, which its scan's route alone
    asks (`_rated_scan`). Rounded there, a write gate without the delta rule passed its values between
    launches in bf16, its final memory 2.1e4 times the portable math's distance from float64 under aligned
    keys (interpreted at 2b595ff93, 9593 to 46978 times on an H100, gpuflow-a-abfff38d9); passed in the fast
    forms' two bf16 parts, 26 times (the stability suite's scalar write), and Mamba-3's bf16 layer's output
    and final state 46 and 29 times (jsf-theme4-head at 269dc1a2a1a, audit A5): no fast form of these is
    as exact as the portable math, so the fast forms leave them whole."""
    return fast_rounding(dtype) and (delta or released)


def scan_kernel(query, key, value, g, beta, state=None, starts=None, *, chunk_size: int,
                max_segments: int | None = None, product: str = 'forward', skip=None,
                wide: bool = True, forked: Forked | None = None, rounded: bool | None = None,
                firsts=None) -> str:
    """The kernel the scan's selector (`SCAN`) gives a rated scan over these operands (`_rated_scan`'s:
    `value` the SSD's `x`, `key` its `B`, `query` its `C`) for `product` ('forward', 'training'), the
    template's implementations bound to it as the run calls it (its fused `skip`, None where added after,
    and its output `wide` or in the operands' dtype), on the target (`generation`'s, an assumed one's within
    `assumed`), at the call's numbers `rounded` (`fast_rounding`'s where None; `port_kernel`'s rule),
    reported as its route."""
    # The scan launches its operands in their common dtype (`_rated_scan`'s `launched`), which its cell and
    # its rounding read.
    dtype = jnp.dtype(jnp.result_type(query, key, value))
    call, (_, shard_key, shard_value, *_) = template_bind(
        query, key, value, g, beta, state, starts, firsts, skip, forked, max_segments=max_segments, scale=1.0,
        delta=False, wide=wide, rounded=fast_rounding(dtype) if rounded is None else rounded)
    case = scan_case(value, key, chunk_size, padded=True, call=call, product=product,
                     forked=forked is not None, packed=starts is not None or firsts is not None,
                     shards=(shard_value, shard_key), dtype=dtype, wide=wide)
    return select(SCAN, case).kernel


SSD_PORT = Kernel[ScanCase]('pallas', SSD_SUPPORTS)
"""Dew's port of mamba_ssm's SSD kernels (`mamba_ssd.ssd_scan`)."""

SCAN_CELLS: dict[tuple[str | None, ScanCell], tuple[Kernel[ScanCase], str]] = {
    ('mma.sync', ScanCell('bfloat16', 'training', (96, 80), (8, 128), 256, TIMED, wide=True)): (
        SSD_PORT, "on an A100 SXM4 40GB, the port's forward and gradient 3.954 ms against the template's "
                  "4.217 on Nemotron-H 4B's SSD, its 80-wide heads padded to the port's 128 "
                  "(ssd-train-a100-fdab2424c)"),
    ('mma.sync', ScanCell('bfloat16', 'training', (64, 64), (8, 128), 128, TIMED, wide=True)): (
        SSD_PORT, "on an A100 SXM4 40GB, the port's forward and gradient 1.584 ms against the template's "
                  "2.200 on Nemotron-H 30B's SSD (ssd-train-a100-fdab2424c)"),
}
"""The cells, each an instruction family, a product and a geometry (`ScanCase.cell`), where a receipt timing
the same product (`chunk(SSD, ...)` with its skip, output width, chunk and geometry, row 1964) shows Dew's
port of mamba_ssm's kernels (`mamba_ssd.ssd_scan`; "the port" below) faster than the recurrence template, each
its old route's kernel and that receipt; every time below sets the template against that port, not against
mamba_ssm's own kernels (tools/benchmark_recurrence.py's `mamba_ssm` arm): the
scan's selector runs the port on them where it takes the call (`Op.declared`); the list shrinks as the
template wins its cells. Warpgroup MMA's training is the template's: on a GH200 480GB, by device time, its
forward and gradient 1.475 ms against the port's 2.065 on Nemotron-H 4B's SSD (its 80-wide heads padded to
the port's 128) and 0.865 against 0.836 on 30B's (LF-REC4-gh200 at 59b96a9b4c2), where an H100's had the
port 1.592 against 1.877 and 0.660 against 0.820 (window-h100-c70aaa8-b8109ee). An A100's forward is the
template's: 1.418 ms against the port's 1.506 on Nemotron-H
4B's and 0.510 against 0.541 on 30B's (ssd-rounded-a100-fdab2424c), and warpgroup MMA's: on an H100 80GB HBM3
0.211 ms against 0.241 on 30B's and 0.522 against 0.617 on 4B's, on a GH200 480GB 0.207 against 0.225 and
0.509 against 0.595 (FS-RECROUTES-h100 and -gh200 at 26d3fdf5df: cold, exclusive, two rounds alternating the
arms' order). The template trains the rated scan's bf16
calls under either forms, its intermediates in fp32's three parts under the accurate ones, and an fp32 call
under the fast forms (`mainloop_backward.fast_backward_form`); an fp32 call under the accurate ones it does
not, by its refusal, not a cell: at the default precision the port then trains it, its products one TF32 as
mamba_ssm's, and at a precision above it (the accurate numerics', `dew.nn.precision.ACCURATE`) the port
declines too (`_port_precision`) and the portable math trains it at that precision. An fp16 call's
gradient the template refuses alike (no recording holds mamba_ssm's fp16 backward against it), so the port,
the next kernel that takes it, trains it on a GPU: a refusal's route, not a cell (row 5739)."""

SCAN: Op[ScanCase] = Op(Choice.SSD, (Kernel('mosaic_gpu', (*SSD_SUPPORTS[:3], *TEMPLATE_SUPPORTS)),
                               SSD_PORT, Kernel('tpu', TPU_SCAN_SUPPORTS),
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
    groups where the head split divides the groups, the time axis whole; off a
    GPU its kernels interpret, as the CPU suite runs them."""
    interpret = not triton_runs()
    seq = None if starts is None else jnp.cumsum(starts, axis=1, dtype=jnp.int32)
    rows, heads = grouped_heads(x.shape, B.shape[2])
    four, carried = P(rows, None, heads, None), P(rows, heads, None, None)

    def scan(x, dt, A, B, C, D, state, *seq):
        return mamba_ssd.ssd_scan(x, dt, A, B, C, D, state, *(seq or (None,)), chunk_size, interpret)

    operands = (x, dt, A, B, C, D, state, *([] if seq is None else [seq]))
    specs = (four, P(rows, None, heads), P(heads), four, four, P(heads), carried, P(rows, None))
    return manual_map(scan, specs[:len(operands)], (four, carried))(*operands)


def _tpu_documents(state, forked: Forked, length: int, chunk_size: int, work, widths: tuple[int, int]):
    """A forked call's documents as the TPU's Pallas scan walks them in place (`ssd._ssd_scan`'s `forked`):
    `(given [B, S', H, P, N] float32, tokens, entry)`, the table's given memories (`state` `[B, S', H, N, P]`,
    the SSD's `[P, N]` transposed), each token's load, store, document (`Forked.tokens`) and added slot
    (`Chunk.adds`, at a document's first token), a padded token taking up none, and the entry write's
    operands each value head's (its keys `[B, D, R, H, N]`, values `[B, D, R, H, P]` and weights `[B, D,
    H]`; None where the call gives none); the memories and the entry write padded with zero columns to the
    scan's `widths` (`_scan_widths`), heads' then states'."""
    padding = -length % chunk_size
    loads, stores, begun = forked.tokens(length)
    # The slot each document's first token adds (`Chunk.adds`), zero at every other token and for none.
    adds = jnp.zeros_like(loads) if forked.adds is None else jnp.where(
        loads >= 0, jnp.take_along_axis(jnp.asarray(forked.adds, jnp.int32), begun, axis=1), 0)
    tokens = tuple(jnp.pad(t, ((0, 0), (0, padding)), constant_values=fill)
                   for t, fill in zip((loads, stores, begun, adds), (-1, 0, 0, 0), strict=True))
    given = zero_padded(zero_padded(jnp.swapaxes(state, -1, -2).astype(work), 3, widths[0]), 4, widths[1])
    entry = None
    if forked.entry is not None and forked.entry[0].shape[2] > 0:
        key, value, weight = forked.entry
        heads = value.shape[3]
        key = jnp.repeat(key, heads // key.shape[3], axis=3)
        entry = (zero_padded(key.astype(work), 4, widths[1]), zero_padded(value.astype(work), 4, widths[0]),
                 weight.astype(work))
    return given.astype(jnp.float32), tokens, entry


def _tpu_specs(documents, rows, heads):
    """`_tpu_documents`' placements on a shard map (`manual_map`): the memories `[B, S', H, P, N]` and the
    entry write's operands (`Forked.entry_specs`) by rows and heads, the tokens `[B, T]` by rows, None for
    absent ones."""
    if documents is None:
        return None
    _, tokens, entry = documents
    return (P(rows, None, heads, None, None), (P(rows, None),) * len(tokens),
            None if entry is None else Forked.entry_specs(rows, heads))


@dataclasses.dataclass(frozen=True)
class SelectiveCase:
    """A selective scan (Mamba's, `dew.nn.mixers.mamba.selective_scan`) as its selector reads it
    (`SELECTIVE`): whether a shard_map that checks varying axes holds its operands, its operands' dtype and
    its states' rows' narrower one (B's or C's, which stream each in its own: the narrower's rows are the
    shorter), its tokens, channels and states, whether the run's forms are the fast ones (its template forms
    its products whole in fp32 whatever the run's precision)
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
    # The template forms its state and products in fp32 whatever its operands' storage.
    Requirement('precision', lambda case: exact_products(None, at_least_fp32(case.dtype)),
                "a dot algorithm the run declares that narrows the scan's products, which the template takes "
                "whole in fp32", row=INHERENT, kind='handoff'),
    Requirement('forms', lambda case: case.fast,
                "the run's accurate forms, whose sums the portable math compensates (row 1698); the template "
                "sums at mamba_ssm's rounding, the fast forms' bar", row='ledger 2968', kind='measured'),
)
"""What the template's selective scan (`templates.recurrence.selective`) needs of a call: no caller's
`shard_map` that checks varying axes, a GPU of a family its bundles cover, 16-bit or fp32 operands, whole
blocks of channels, at most `SELECTIVE_STATES` states, a chunk's tokens at least (a decode step hands off
to the portable math), and the run's fast forms. Off Hopper too: on an A100 it beat the portable scan in
every cell, forward 4.0 to 9.1 times, training 11.2 to 13.8 (Mamba 130M and 2.8B at 2k and 8k tokens,
W84-a100, M-JS-23)."""

SELECTIVE: Op[SelectiveCase] = Op(Choice.SELECTIVE_SCAN, (Kernel('mosaic_gpu', SELECTIVE_SUPPORTS),
                                                     Kernel('xla', (), generic=True)))
"""The selective scan's selector: the recurrence template's scan (`templates.recurrence.selective`) where
it takes the call, else the portable math. On an A100 its training step, B's and C's gradients added
atomically, took 11.3 ms at 8x2048x5120 and 9.22 at 4x2048x8192 (W3-a100-3-selective-ab-a100-8ca2795113, at
44bec4713f, whose backward base's is again)."""


def selective_kernel(x, A, B, C) -> str:
    """The kernel the selective scan's selector (`SELECTIVE`) gives a scan over `x` `[B, S, D]` with rates
    `A` `[D, N]` and states' rows `B`, `C` `[B, S, N]` on the target, reported as its route."""
    from .templates import bundles

    # A GPU's family where a bundle claims it (consumer Blackwell's, mma.sync fed by TMA, has one; tcgen05
    # has none in a route yet).
    on_gpu = generation.backend() == 'gpu'
    bundled = on_gpu and bundles.of(generation.device_generation(), x.dtype) is not None
    family = generation.family(generation.device_generation()) if bundled else None
    case = SelectiveCase(varying=bool(jax.typeof(x).mat.varying), dtype=jnp.dtype(x.dtype),
                         state_dtype=min(jnp.dtype(jnp.result_type(B)), jnp.dtype(jnp.result_type(C)),
                                         key=lambda dtype: dtype.itemsize),
                         length=x.shape[1], channels=x.shape[-1], states=A.shape[-1],
                         fast=generation.numerics().forms == 'fast',
                         family=family)
    return select(SELECTIVE, case).kernel


def selective_scan(x, dt, A, B, C, D, state, starts, *, portable: Callable, loads=None, stores=None,
                   held: int = 0):
    """Mamba's selective scan (`dew.nn.mixers.mamba.selective_scan`'s operands, `state` `[B, D, N]` or None)
    on the kernel its selector gives each product (`SELECTIVE`, `_per_product`): the recurrence template's
    scan, interpreted off a GPU (where a call names it), per shard of rows on a mesh (`manual_map`: B and C
    serve every channel, so the channels stay whole); or `portable(x, dt, A, B, C, D, state, starts)`,
    the portable math. A forked call (`loads`, `stores`, `held`, as `Chunk.loads` takes them, `state` `[B,
    S', D, N]` its table's given states) walks each document in place, taking its state up at its first
    token, on either: the template's walk over its table (`selective.Documents`) or the portable math's
    (`portable`'s `forked`); its final state is the row's last.
    """
    forked = None
    if loads is not None:
        loads = jnp.asarray(loads, jnp.int32)
        forked = Forked(document_firsts(starts, loads.shape[1]), loads, jnp.asarray(stores, jnp.int32), None,
                        held)
    from .templates.recurrence import selective

    def template(x, dt, A, B, C, D, state, *given):
        if forked is not None:
            # A forked call's slots a row's (`Forked.walked`), its starts read into their firsts.
            return selective.SELECTIVE(x, dt, A, B, C, D, state, forked=forked.carrying(*given),
                                       interpret=generation.backend() != 'gpu')
        return selective.SELECTIVE(x, dt, A, B, C, D, state, *(given or (None,)),
                                   interpret=generation.backend() != 'gpu')

    def run(kernel, x, dt, A, B, C, D, state, *walked):
        if kernel != 'mosaic_gpu':
            laid = {} if forked is None else {'forked': forked.carrying(*walked)}
            return portable(x, dt, A, B, C, D, state, starts, **laid)
        rows, _ = grouped_heads((*x.shape[:2], 1, x.shape[-1]), 1)
        three = P(rows, None, None)
        if forked is not None:
            operands = (x, dt, A, B, C, D, state, *walked)
            specs = (three, three, P(None, None), three, three, P(None), P(rows, None, None, None),
                     *forked.specs(rows, None))
            return manual_map(template, specs, (three, three))(*operands)
        held = state if state is not None else jnp.zeros((x.shape[0], x.shape[-1], A.shape[-1]), x.dtype)
        operands = (x, dt, A, B, C, D, held, *([] if starts is None else [starts]))
        specs = (three, three, P(None, None), three, three, P(None), three, P(rows, None))
        return manual_map(template, specs[:len(operands)], (three, three))(*operands)

    # A and D (operands 2 and 5) are the call's parameters.
    return _per_product(SELECTIVE.bound(lambda product: selective_kernel(x, A, B, C)), run, x, dt, A, B, C, D,
                        state, *(() if forked is None else forked.walked), parameters=(2, 5))
