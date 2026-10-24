"""The attention template's Mosaic GPU mainloop (design §6, one mainloop per overlap structure): the order of
an instruction family's bundle's steps (`mosaic.Family`: its roles, operand streams, products and layouts)
around the template's algebra (`attention_algebra`), FlashAttention-3's structure on the wgmma family
(`mosaic_sm90.WGMMA`). It holds no instruction, layout or stream barrier of its own: only the overlap
structure, the query block's load and the consumers' schedule.

One program a query head and the family's consumers' query blocks (two warpgroups on wgmma, each a block of
`Tiles.block_q` rows), beside the family's loads streaming the key and value tiles (`Family.stream`,
`Family.walk`), every slot released by every consumer. Two consumers take turns on the tensor cores (a
schedule barrier between them), so one's softmax runs under the other's products. A step issues S = Q·Kᵀ,
releases the key slot, and hands S to the algebra as a tile (`algebra.Keys`) whose values are the value slot
and whose product accumulates the narrowed weights from registers into the running sums in place; while it
runs the consumer waits for its next key tile. A block reads the key tiles its spans reach (a full or causal
call's from its places, a spans call's from its block map, `attention.block_map`) and masks only those it
does not see whole (`algebra.seen`); a consumer keeps releasing the slots its partner alone reads.

The forward over every call form of 16-bit operands under every hook (`Hooks`: the score hook and its
parameters on each tile, the softmax's initial state in the walk's start, the output hook on its output), the
base-2 fold and natural logits, under the default's guards over whole launches (`algebra.walked_launches`: a
program walks its keys once, so the call walks again in the natural form where some row's logits pass the
fold's reach, and its overflowed rows again shifting under headroom). Its legality rule is `refused`."""

import dataclasses
import functools
import math
from collections.abc import Callable, Sequence
from typing import NamedTuple

import jax
import jax.extend.core as jax_core
import jax.numpy as jnp
import tokamax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels import generation
from dew.nn.kernels.templates import (
    attention_algebra as algebra,
    attention_portable,
    cost,
    mosaic,
    mosaic_sm90,
)
from dew.nn.kernels.templates.attention import (
    ACCURATE,
    DKV_GRADS,
    DQ_GRADS,
    NATURAL,
    TRACED_TILE,
    Gradients,
    Hooks,
    KeySpans,
    Kind,
    PaddedCall,
    Param,
    Split,
    block_map,
    gradients,
    hook_program,
    kept_dtype,
    padded_call,
    power_of_two,
    spanned,
    whole,
)
from dew.nn.kernels.templates.base import Template

BUNDLES: dict[str, mosaic.Family] = {mosaic_sm90.WGMMA.name: mosaic_sm90.WGMMA}
"""The instruction families this mainloop runs on (`generation.family`), each by its bundle: wgmma's. A
family without one (mma.sync's, until its thin mainloop) takes another mainloop (`MosaicForward.eligible`,
`MosaicBackward.eligible`)."""

ALIGNMENT = 1024
"""The byte boundary Mosaic GPU starts each shared-memory allocation at (JAX's Pallas Mosaic GPU
`SMEM_ALIGNMENT`, for the TMA swizzles), so a program stages each allocation rounded up to it (`_staged`)."""


@dataclasses.dataclass(frozen=True, order=True)
class Tiles:
    """A forward launch's config: each consumer's query block (`block_q`), the key tile (`block_k`), the key
    and value tiles in flight (`stages`), the consumers a program (`consumers`: on wgmma FlashAttention-3's
    two warpgroups take turns on the tensor cores, one's softmax under the other's products) and the
    instruction family whose bundle runs it (`family`, `BUNDLES`)."""

    block_q: int
    block_k: int
    stages: int
    consumers: int
    family: str = mosaic_sm90.WGMMA.name

    def __post_init__(self):
        if self.stages < 2:
            raise ValueError(f"{self}: two stages or more")


@dataclasses.dataclass(frozen=True, order=True)
class BackwardTiles:
    """A backward's config, one for its launches: the query block a dQ consumer takes and the key block a
    dK/dV consumer takes (`block_q`, `block_k`, so both launches' logits are one tile's), the tiles in
    flight (`stages`), the consumers a program (`consumers`), whether the fast form runs as one launch
    (`fused`, `_launch_backward`) and the instruction family whose bundle runs it (`family`)."""

    block_q: int
    block_k: int
    stages: int
    consumers: int
    fused: bool = False
    family: str = mosaic_sm90.WGMMA.name

    def __post_init__(self):
        if self.stages < 2:
            raise ValueError(f"{self}: two stages or more")


def _extents(family: mosaic.Family, dtype, extent: int) -> tuple[int, ...]:
    """A tile axis's extents on `family`, largest first: whole multiples of the rows a consumer's products
    cover (`Family.role_rows`) and of its swizzled rows (`Family.column_unit`; a tile's keys are its weights'
    product's depth, its queries the dK/dV launch's), at most a load's block (`Family.block_extent`) and the
    call's `extent` in whole multiples (a longer tile holds padding alone)."""
    unit = math.lcm(family.role_rows, family.column_unit(dtype))
    top = min(family.block_extent // unit, -(-max(extent, 1) // unit)) * unit
    return tuple(range(top, 0, -unit))


def _width(width: int, family: mosaic.Family, dtype) -> int:
    """A head width as a launch stages it: a power of two (`padded_call`) in whole swizzled rows
    (`Family.column_unit`)."""
    return whole(power_of_two(width), family.column_unit(dtype))


def _stages(steps: int, staged) -> tuple[int, ...]:
    """The stage counts a launch walking `steps` tiles takes, fewest first: two, then each more up to its
    steps (a slot past them loads past the operand, `_Program`) while its program's shared memory at that
    count (`staged(stages)`) fits a block of the device's (`generation.gpu_target`, the facts
    `generation.assumed` gives). Where the cost model ties, the fewest win: two stages ran fastest of two to
    four on an H100 (the forward at llama-d128 causal bf16, 2 x 4096)."""
    budget = generation.shared_budget(generation.gpu_target())
    counts = [2]
    while counts[-1] < steps and staged(counts[-1] + 1) <= budget:
        counts.append(counts[-1] + 1)
    return tuple(counts)


def _staged(allocations) -> int:
    """The shared memory `allocations` take, `(shape, dtype)` each, every one from an `ALIGNMENT` boundary."""
    return sum(whole(math.prod(shape) * jnp.dtype(dtype).itemsize, ALIGNMENT) for shape, dtype in allocations)


def _registers(family: mosaic.Family, carried, *phases) -> int:
    """The registers a consumer thread of `family` holds at a step's peak: its share (`mosaic.THREADS`) of the
    tiles its walk carries (`carried`: accumulators, kept rows) and of the step's largest phase (`phases`: the
    tiles alive at once between two of its products), each tile `(rows, columns, dtype)`, and what it holds
    beside them (`Family.held_registers`: addresses, indices, the carried rows' statistics). A hook
    parameter's tile is not counted. On an H100 at D 128, a forward consumer of 64 queries over 128 keys
    holds 200 of its 232 and runs fastest, one of 128 queries 264 and spills (2.9 to 4.6 ms against 0.49)."""
    def words(tiles) -> int:
        return sum(rows * columns * jnp.dtype(dtype).itemsize for rows, columns, dtype in tiles) // 4

    return (words(carried) + max(map(words, phases), default=0)) // mosaic.THREADS + family.held_registers


def _streams(blocks: dict, stages: int):
    """`Family.streams`' allocations as a ring family stages them: `stages` slots of each block."""
    return [((stages, *shape), dtype) for shape, dtype in blocks.values()]


def _forward_blocks(tiles: Tiles, width: int, value_width: int, dtype) -> dict:
    """The forward's streams' blocks: a key tile and its value tile."""
    return {"keys": ((tiles.block_k, width), dtype), "values": ((tiles.block_k, value_width), dtype)}


def _forward_staging(tiles: Tiles, width: int, value_width: int, dtype, residuals: bool) -> list:
    """The forward program's shared memory beside its streams, `(shape, dtype)` each in `_launch`'s order:
    its consumers' query stages and their rows' status; each one's output stage where the value is not as
    wide as the query (the output stores through the query's stage where it is); in a training forward its
    anchor keys' and split sums' stages (`_Residuals`)."""
    consumers = tiles.consumers
    staging = [((consumers, tiles.block_q, width), dtype), ((consumers, tiles.block_q), jnp.float32)]
    if value_width != width:
        staging.append(((consumers, tiles.block_q, value_width), dtype))
    if residuals:
        staging += [((consumers, tiles.block_q), jnp.int32),
                    ((consumers, tiles.block_q, value_width), jnp.float32)]
    return staging


def _walked(causal: bool, consumers: int, block_q: int, block_k: int, programs: int,
            keys: int) -> tuple[int, int]:
    """The key tiles a forward's or dQ launch's `programs` query programs walk: every consumer's summed, and
    the tiles each program streams (its consumers' widest reach) summed: every tile holding the call's keys,
    to each block's diagonal where `causal` (`_Program.walked`)."""
    held = -(-keys // block_k)
    consumed = streamed = 0
    for program in range(programs):
        reaches = [min(-(-(first + block_q) // block_k), held) if causal else held
                   for first in range(program * consumers * block_q, (program + 1) * consumers * block_q,
                                      block_q)]
        consumed, streamed = consumed + sum(reaches), streamed + max(reaches)
    return consumed, streamed


def _readers(causal: bool, consumers: int, block_q: int, block_k: int, programs: int, queries: int,
             keys: int) -> tuple[int, int]:
    """`_walked` for a dK/dV launch's `programs` key programs: the query tiles reading each consumer's keys
    (every tile holding the call's queries, from its diagonal where `causal`; none for keys past the call's,
    `_Program.readers`), summed over its consumers, and the tiles each program streams."""
    held = -(-queries // block_q)
    consumed = streamed = 0
    for program in range(programs):
        reaches = [0 if first >= keys else held - (min(first // block_q, held) if causal else 0)
                   for first in range(program * consumers * block_k, (program + 1) * consumers * block_k,
                                      block_k)]
        consumed, streamed = consumed + sum(reaches), streamed + max(reaches)
    return consumed, streamed


def _read_bytes(ba: tokamax.BoundArguments) -> int:
    """The bytes of every array a launch of `ba` reads, at its dtype (operands, spans, hook parameters,
    initial state, residuals)."""
    return sum(leaf.size * jnp.dtype(leaf.dtype).itemsize for leaf in jax.tree.leaves(dict(ba.arguments))
               if hasattr(leaf, "dtype"))


@dataclasses.dataclass(frozen=True)
class _Mainloop[Config: (Tiles, BackwardTiles)](Template[Config]):
    """What this mainloop's launches share as Template implementations (`MosaicForward`, `MosaicBackward`):
    the call's own arrays as arguments, padded by a launch to its config's blocks; the score hook (and the
    forward's output hook) as fields keyed by what they compute (`hook_program`); the configs of the
    target's family (`fits`), a consumer's registers its roles' share (`register_budget`), where its bundle
    is one of `BUNDLES` and the call is one it takes (`refused`)."""

    name = "mosaic_gpu"
    score_mod: Callable | None = None
    form: str = "spans"
    keys: int = 0
    score_kinds: tuple[Kind, ...] = ()
    score_dtypes: tuple[str, ...] = ()
    scale: float = 1.0
    natural: bool = NATURAL

    def spans(self, start, end, query_ids, key_ids) -> KeySpans:
        """The call's spans from its arrays and the op's static form and key count."""
        return KeySpans(start, end, query_ids, key_ids, self.keys, self.form)

    def fits(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> tuple[Config, ...]:
        family = generation.family(target.generation)
        return tuple(config for config in super().fits(ba, target) if config.family == family)

    def staged(self, ba: tokamax.BoundArguments, tiles: Config, steps: int) -> list[Config]:
        """`tiles` at each stage count a walk of `steps` tiles and the device's block take (`_stages`)."""
        target = generation.gpu_target()
        return [dataclasses.replace(tiles, stages=stages) for stages in _stages(
            steps, lambda stages: self.staged_bytes(ba, dataclasses.replace(tiles, stages=stages), target))]

    def register_budget(self, config: Config) -> int:
        """A consumer thread's share of the registers (`Family.registers`): the family's roles share the
        file (`mosaic.roles`)."""
        return BUNDLES[config.family].registers(config.consumers)

    def eligible(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> bool:
        # Interpreted only in a CPU process (`grouped_matmul._Mosaic.eligible`): the interpreter's callbacks
        # are ordered effects, which a GPU process's lowering cannot carry.
        interpretable = not ba.arguments["interpret"] or jax.default_backend() == "cpu"
        family = BUNDLES.get(generation.family(target.generation) or "")
        return (interpretable and family is not None and self.refusal(ba, family) is None
                and bool(self.fits(ba, target)))

    def refusal(self, ba: tokamax.BoundArguments, family: mosaic.Family) -> str | None:
        """Why the launch takes no call of `ba` on `family`, or None (`refused`)."""
        raise NotImplementedError

    def operands(self, specs: list[jax.ShapeDtypeStruct], seed: int) -> list[jax.Array]:
        return spanned(specs, super().operands(specs, seed))


@dataclasses.dataclass(frozen=True)
class MosaicForward(_Mainloop[Tiles]):
    """The template's forward on this mainloop over the call's own arrays (`forward` gives them): queries
    `[B, Q, H, D]` over keys `[B, K, Hk, D]` and values `[B, K, Hk, Dv]`, each query's keys `start` to `end`
    `[B, Q]` and, with documents, its id `[B, Q]` against the keys' `[B, K]`; the score and output hooks'
    parameters (`score_params`, `output_params`, laid out by `score_kinds` and `output_kinds`) and the
    softmax's initial state (`Initial`, or None). Returns the output `[B, Q, H, Dv]` and, with `residuals`,
    the backward's residuals (`forward`'s). Its configs come from the family's facts and the call's extents
    (`space`), fitted to the device's block and a consumer's registers (`fits`), the cost model's fastest
    the heuristic (`costed`)."""

    HOOKS = ("score_mod", "output_mod")
    config_cls = Tiles
    output_mod: Callable | None = None
    output_kinds: tuple[Kind, ...] = ()
    output_dtypes: tuple[str, ...] = ()
    output_width: int = TRACED_TILE

    def _fwd(self, query, key, value, start, end, query_ids, key_ids, score_params, output_params, initial, *,
             residuals: bool, interpret: bool, config: Tiles, return_residuals: bool = False):
        hooks = Hooks(self.score_mod, tuple(map(Param, score_params, self.score_kinds)), initial,
                      self.output_mod, tuple(map(Param, output_params, self.output_kinds)))
        run = _trained_forward if residuals else _plain_forward
        return run(query, key, value, self.spans(start, end, query_ids, key_ids), hooks, scale=self.scale,
                   natural=self.natural, tiles=config, interpret=interpret), None

    def space(self, ba: tokamax.BoundArguments) -> Sequence[Tiles]:
        """Every tile the forward takes on each family of `BUNDLES` at the call's extents: each count of
        consumers the family has, most first; each consumer's query block and the key tile in whole
        multiples of the family's unit (`_extents`), largest first, so where the cost model ties the fewest
        programs re-read the keys; at each stage count its walk and the device's block take (`_stages`). A
        consumer thread's registers bound them where they fit (`fits`, `register_budget`)."""
        query, key = ba.arguments["query"], ba.arguments["key"]
        configs = []
        for family in BUNDLES.values():
            for consumers in range(family.consumers, 0, -1):
                for block_q in _extents(family, query.dtype, -(-query.shape[1] // consumers)):
                    for block_k in _extents(family, query.dtype, key.shape[1]):
                        tiles = Tiles(block_q, block_k, 2, consumers, family.name)
                        keys = _padded(query.shape[1], key.shape[1], tiles)[1]
                        configs += self.staged(ba, tiles, keys // block_k)
        return tuple(configs)

    def staged_bytes(self, ba: tokamax.BoundArguments, config: Tiles, target: generation.GpuTarget) -> int:
        """The program's shared memory (`_forward_staging`) and its streams' slots, each allocation from an
        `ALIGNMENT` boundary, at the call's widths as the launch stages them (`_width`)."""
        family = BUNDLES[config.family]
        query, key, value = (ba.arguments[name] for name in ("query", "key", "value"))
        width, value_width = (_width(array.shape[-1], family, query.dtype) for array in (query, value))
        steps = _padded(query.shape[1], key.shape[1], config)[1] // config.block_k
        blocks = _forward_blocks(config, width, value_width, query.dtype)
        residuals = bool(ba.arguments.get("residuals"))
        staging = _forward_staging(config, width, value_width, query.dtype, residuals)
        return _staged([*staging, *_streams(blocks, min(config.stages, steps))])

    def registers(self, ba: tokamax.BoundArguments, config: Tiles) -> int:
        family = BUNDLES[config.family]
        query, value = ba.arguments["query"], ba.arguments["value"]
        return _forward_registers(family, config, _width(value.shape[-1], family, query.dtype), query.dtype,
                                  bool(ba.arguments.get("residuals")))

    def costed(self, ba: tokamax.BoundArguments, config: Tiles, target: generation.GpuTarget) -> cost.Cost:
        """Its programs (a query head's consumers' query blocks), each streaming the key tiles its consumers
        walk (`_walked`: every tile of a full call or a spans call's, a bound, to the diagonal of a causal
        one), each consumer step's two products (the logits, then the weights with the values; a training
        forward's one-hot product too) and exponentials; its shared memory's traffic: each streamed tile
        written once and read by each consumer's products, the query stage written once, the output (and
        the residuals) staged and stored; its arrays read once and its output and residuals written once.
        With one consumer its softmax runs between its products, with two under its partner's
        (`serial_exponentials`)."""
        family = BUNDLES[config.family]
        query, key, value = (ba.arguments[name] for name in ("query", "key", "value"))
        batch, queries, heads, _ = query.shape
        residuals = bool(ba.arguments.get("residuals"))
        size = jnp.dtype(query.dtype).itemsize
        width, value_width = (_width(array.shape[-1], family, query.dtype) for array in (query, value))
        consumers, block_q, block_k = config.consumers, config.block_q, config.block_k
        padded_q, padded_k = _padded(queries, key.shape[1], config)
        tiles = padded_q // (consumers * block_q)
        consumed, streamed = _walked(self.form == "causal", consumers, block_q, block_k, tiles, key.shape[1])
        products = width + value_width + (value_width if residuals else 0)
        exponentials = consumed / tiles * block_q * block_k
        # A query row's output, and a training forward's residuals: its split sums and five rows, fp32.
        row = value_width * size + ((value_width + 6) * 4 if residuals else 0)
        slot = block_k * (width + value_width) * size
        staged = (streamed * slot + consumed * (block_q * width * size + slot)) / tiles
        staged += consumers * block_q * (width * size + 2 * row)
        return cost.Cost(
            programs=batch * heads * tiles, device_bytes=_read_bytes(ba) + batch * queries * heads * row,
            program_flops=consumed / tiles * 2.0 * block_q * block_k * products, program_staged=staged,
            program_exponentials=exponentials if consumers > 1 else 0.0,
            serial_exponentials=0.0 if consumers > 1 else exponentials,
            mode="16bit", steps=streamed / tiles, stages=min(config.stages, padded_k // block_k),
            resident=cost.resident(target, self.staged_bytes(ba, config, target),
                                   mosaic.THREADS * (consumers + 1), self.register_budget(config)),
            # Each program streams its key and value tiles from the L2.
            l2_bytes=batch * heads * streamed * slot)

    def refusal(self, ba: tokamax.BoundArguments, family: mosaic.Family) -> str | None:
        query, value = ba.arguments["query"], ba.arguments["value"]
        hooks = Hooks(self.score_mod, tuple(map(Param, ba.arguments["score_params"], self.score_kinds)),
                      ba.arguments["initial"], self.output_mod,
                      tuple(map(Param, ba.arguments["output_params"], self.output_kinds)))
        return refused(query.dtype, query.shape[3], value.shape[3], hooks, backward=False, family=family)

    def traced(self, name: str, value: Callable | str | None) -> str | None:
        if name == "output_mod":
            return hook_program(name, value, self.output_kinds, self.output_dtypes, width=self.output_width)
        return hook_program(name, value, self.score_kinds, self.score_dtypes)


def _forward_registers(family: mosaic.Family, tiles: Tiles, value_width: int, dtype, residuals: bool) -> int:
    """`_registers` of a forward consumer at `tiles`: its weighted values' sums `[bq, Dv]` fp32 carried, and
    a step's logits fp32 beside its weights in the operands' dtype; a training forward's anchor value row
    carried too (`_trained`, in the operands' dtype), and its one-hot weights beside the product picking
    the row, fp32."""
    block_q, block_k = tiles.block_q, tiles.block_k
    carried = [(block_q, value_width, jnp.float32)]
    phases = [[(block_q, block_k, jnp.float32), (block_q, block_k, dtype)]]
    if residuals:
        carried.append((block_q, value_width, dtype))
        phases.append([(block_q, block_k, dtype), (block_q, value_width, jnp.float32)])
    return _registers(family, carried, *phases)


def _padded(queries: int, keys: int, config: Tiles | BackwardTiles) -> tuple[int, int]:
    """A call's queries and keys as a launch at `config` pads them (`padded_call`, `_block`)."""
    block = _block(config)
    return whole(max(queries, 1), block), whole(max(keys, 1), block)


def _block(config: Tiles | BackwardTiles) -> int:
    """The block a launch at `config` pads a call's queries and keys to: whole programs of query blocks and
    key tiles (a forward's), or of query blocks and key blocks (a backward's dQ and dK/dV programs)."""
    keys = config.consumers * config.block_k if isinstance(config, BackwardTiles) else config.block_k
    return math.lcm(config.consumers * config.block_q, keys)


def forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
            *, scale: float | None = None, natural: bool = NATURAL, residuals: bool = False,
            interpret: bool = False):
    """The template's forward on this mainloop (`MosaicForward`): `query` `[B, Q, H, D]` over `key`
    `[B, K, Hk, D]` and `value` `[B, K, Hk, Dv]`, and the keys `spans` leaves each query under `hooks`, as
    `dew.nn.kernels.templates.attention.reference` computes it; with `residuals` the training forward's,
    its output with the backward's residuals as every mainloop's training forward gives them
    (`attention_portable.attend(residuals=True)`): the maxima and denominators `[B, H, Q]`, each query's
    anchor key `[B, H, Q]` int32 and the split sums (`Split`). `natural` keeps the logits natural
    (`algebra.Forward.base2`); `interpret` runs it on the CPU (Mosaic GPU's interpret mode), as a GPU of the
    family `generation.assumed` names."""
    hooks = Hooks() if hooks is None else hooks
    op = MosaicForward(score_mod=hooks.score_mod, output_mod=hooks.output_mod,
                       score_kinds=tuple(param.kind for param in hooks.score_params),
                       output_kinds=tuple(param.kind for param in hooks.output_params),
                       score_dtypes=tuple(kept_dtype(param.value.dtype) for param in hooks.score_params),
                       output_dtypes=tuple(kept_dtype(param.value.dtype) for param in hooks.output_params),
                       output_width=power_of_two(value.shape[3]), form=spans.form, keys=spans.keys,
                       scale=query.shape[3] ** -0.5 if scale is None else scale, natural=natural)
    return op(query, key, value, spans.start, spans.end, spans.query_ids, spans.key_ids,
              tuple(param.value for param in hooks.score_params),
              tuple(param.value for param in hooks.output_params), hooks.initial, residuals=residuals,
              interpret=interpret)


def _plain_forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks, *,
                   scale: float, natural: bool, tiles: Tiles, interpret: bool) -> jax.Array:
    """`MosaicForward`'s launch without residuals, at `tiles`, under the default's guards over whole launches
    (`algebra.walked_launches`)."""
    family = BUNDLES[tiles.family]
    call, spec, (q, k, v), (spans_of, hooked) = _prepared(query, key, value, spans, hooks, scale=scale,
                                                          natural=natural, tiles=tiles, family=family)

    def launch(form: algebra.Forward, live: jax.Array | None):
        # The guarded launch reads its live rows and its keys' values' largest magnitudes (`algebra.room`).
        guard = None if live is None else _Guard(live.astype(jnp.int32),
                                                 jnp.swapaxes(algebra.magnitudes(v), 1, 2))
        out, status = _launch(q, k, v, spans_of, hooked, guard, spec=form, hooks=hooks, tiles=tiles,
                              family=family, training=False, interpret=interpret)
        return out, status[0], status[1] > 0

    def kept(live: jax.Array, guarded: jax.Array, plain: jax.Array) -> jax.Array:
        return jnp.where(jnp.swapaxes(live, 1, 2)[..., None], guarded, plain)

    out = algebra.walked_launches(spec, launch, kept)
    return out[:, :call.queries, :, :value.shape[3]]


def _trained_forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks, *,
                     scale: float, natural: bool, tiles: Tiles, interpret: bool
                     ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, Split]:
    """`MosaicForward`'s training launch at `tiles`: the algebra's training step taken in place
    (`algebra.split_taken`, `split_rows`). The old anchor's value row, which the split sums take back where
    the anchor moves, is the walk's own: each row's anchor's row kept in registers, picked from a value tile
    by its one-hot product where it moved.

    A program walks its keys once, so the default's guards compose over whole launches
    (`algebra.walked_launches`): the fold's rewalk is this kernel's natural launch, and rows whose weighted
    values overflowed take the plain-JAX mainloop's guarded training walk (its normalized split sums, shift
    -1, `algebra.split_residuals`), which this kernel's single pass does not take."""
    family = BUNDLES[tiles.family]
    call, spec, (q, k, v), (spans_of, hooked) = _prepared(query, key, value, spans, hooks, scale=scale,
                                                          natural=natural, tiles=tiles, family=family)
    queries_padded, value_width = call.query.shape[1], value.shape[3]

    def launch(form: algebra.Forward, live: jax.Array | None):
        if live is None:
            out, status, rows, anchors, others = _launch(q, k, v, spans_of, hooked, None, spec=form,
                                                         hooks=hooks, tiles=tiles, family=family,
                                                         training=True, interpret=interpret)
            split = Split(jnp.swapaxes(others, 1, 2), rows[2], rows[3], rows[4])
            return (out, rows[0], rows[1], anchors, split), status[0], status[1] > 0
        # The overflowed rows' guarded training walk, the plain-JAX mainloop's, padded as this call's; its
        # maxima and overflows `walked_launches` does not read.
        guarded = attention_portable.attend(query, key, value, spans, hooks, block_q=tiles.block_q,
                                            block_k=tiles.block_k, pad=queries_padded, scale=scale,
                                            residuals=True, natural=form.natural)
        padded = _padded_query_rows(guarded, queries_padded, v.shape[-1])
        return padded, padded[1], jnp.zeros_like(live)

    def kept(live: jax.Array, guarded, plain):
        def chosen(new: jax.Array, old: jax.Array) -> jax.Array:
            # Each leaf's query rows: the output's `[B, Qp, H, ...]`, the residuals' `[B, H, Qp, ...]`.
            rows = jnp.swapaxes(live, 1, 2) if new.ndim == 4 and new.shape[1] != live.shape[1] else live
            return jnp.where(rows.reshape(rows.shape + (1,) * (new.ndim - rows.ndim)), new, old)
        return jax.tree.map(chosen, guarded, plain)

    out, maximum, total, anchors, split = algebra.walked_launches(spec, launch, kept)
    queries = call.queries
    split = Split(split.others[:, :, :queries, :value_width], split.own[:, :, :queries],
                  split.others_total[:, :, :queries], split.shift[:, :, :queries])
    return (out[:, :queries, :, :value_width], maximum[:, :, :queries], total[:, :, :queries],
            anchors[:, :, :queries], split)


def _padded_query_rows(outputs: tuple, queries: int, width: int) -> tuple:
    """A training forward's outputs (`_trained_forward`'s) with their query rows padded to `queries` and their
    value widths to `width`, as this mainloop's launch gives them."""
    out, maximum, total, anchors, split = outputs

    def padded(array: jax.Array, rows_axis: int) -> jax.Array:
        pads = [(0, 0)] * array.ndim
        pads[rows_axis] = (0, queries - array.shape[rows_axis])
        if array.ndim == 4:
            pads[-1] = (0, width - array.shape[-1])
        return jnp.pad(array, pads)

    return (padded(out, 1), padded(maximum, 2), padded(total, 2), padded(anchors, 2),
            Split(*(padded(part, 2) for part in split)))


def _prepared(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks, *,
              scale: float, natural: bool, tiles: Tiles, family: mosaic.Family):
    """A call as this mainloop takes it at `tiles`: padded to the program's blocks (`padded_call`, `_block`)
    and its heads to whole swizzled rows (`_columns`), with its `algebra.Forward` (every form guarding its
    sums where its values can overflow them, the fold its logits' reach, `algebra.walked_launches`) and the
    arrays its programs read beside the operands (`_Spans`, `_Hooked`). Its legality is `refused`'s, which
    the op's eligibility reads."""
    call = padded_call(query, key, value, spans, hooks, _block(tiles))
    spec = algebra.Forward(spans.form, spans.keys, ids=spans.query_ids is not None, scale=scale,
                           score_mod=hooks.score_mod, natural=natural, headroom=algebra.headroom(value.dtype),
                           initial=hooks.initial is not None, queries=call.queries)
    unit = family.column_unit(query.dtype)
    operands = tuple(_columns(array, unit) for array in (call.query, call.key, call.value))
    read = (_spans_of(spec, call, tiles.block_q, tiles.block_k), _hooked_of(hooks, call, unit))
    return call, spec, operands, read


def _columns(array: jax.Array, unit: int) -> jax.Array:
    """`array` with its last axis padded to whole rows of `unit` entries, as the family's swizzled tiles hold
    them (`Family.column_unit`)."""
    return jnp.pad(array, ((0, 0),) * (array.ndim - 1) + ((0, -array.shape[-1] % unit),))


def refused(dtype, width: int, value_width: int, hooks: Hooks | None, *, backward: bool,
            family: mosaic.Family) -> str | None:
    """Why this mainloop takes no call of `dtype` operands, head widths `width` and `value_width` and `hooks`
    (its forward, or with `backward` its backward) on `family`'s bundle, or None: its legality rule, which
    the ops' eligibility (`MosaicForward.eligible`, `MosaicBackward.eligible`), the route and the stability
    suite read."""
    hooks = Hooks() if hooks is None else hooks
    sixteen = (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16))
    if backward and jnp.dtype(dtype) != jnp.dtype(jnp.bfloat16):
        return f"the Mosaic backward takes bf16 operands, not {jnp.dtype(dtype)}"
    if jnp.dtype(dtype) not in sixteen:
        return f"the Mosaic mainloop takes 16-bit operands, not {jnp.dtype(dtype)}"
    if backward and width != value_width:
        return "the Mosaic backward stages dK and dV through one buffer: equal head widths"
    widest = max(_width(width, family, dtype), _width(value_width, family, dtype))
    if widest > family.block_extent:
        return (f"the Mosaic mainloop loads a head {widest} wide in one block, past the "
                f"{family.block_extent} a load takes (`Family.block_extent`)")
    # The hooks as a launch runs them: on its family's least tile, its values in whole swizzled rows.
    unit = math.lcm(family.role_rows, family.column_unit(dtype))
    return unlowered(hooks, (unit, unit), _width(value_width, family, dtype))


def unlowered(hooks: Hooks, tile: tuple[int, int], value_width: int) -> str | None:
    """Why Mosaic GPU cannot lower `hooks` on this mainloop's tiles (`tile`, its outputs `value_width` wide),
    or None: a hook that forms an array of a unit axis beside a longer one (`q_pos[:, :1]`, a reshape), whose
    every operation jax 0.11.2's Mosaic GPU layout inference finds no layout for. A legality rule, which
    the route's chooser reads to send such a call to another route, until the fork's layout inference
    takes unit axes (ledger row 1428)."""
    rows, columns = tile

    def entries(kind: Kind, dtype, width: int) -> jax.ShapeDtypeStruct:
        # A parameter as `_param` gives it: a head's scalar, else the tile's entries.
        return jax.ShapeDtypeStruct(() if kind == Kind.HEAD else (rows, width), kept_dtype(dtype))

    traced = []
    if hooks.score_mod is not None:
        positions = jax.ShapeDtypeStruct(tile, jnp.int32)
        traced.append(jax.make_jaxpr(hooks.score_mod)(
            jax.ShapeDtypeStruct(tile, jnp.float32), positions, positions,
            *(entries(param.kind, param.value.dtype, columns) for param in hooks.score_params)).jaxpr)
    if hooks.output_mod is not None:
        traced.append(jax.make_jaxpr(hooks.output_mod)(
            jax.ShapeDtypeStruct((rows, value_width), jnp.float32),
            *(entries(param.kind, param.value.dtype, value_width) for param in hooks.output_params)).jaxpr)
    if any(_unit_axis(jaxpr) for jaxpr in traced):
        return ("the Mosaic mainloop takes no hook that forms a unit axis beside a longer one (Mosaic GPU's "
                "layout inference finds no layout for it)")
    return None


def _unit_axis(jaxpr: jax_core.Jaxpr) -> bool:
    """Whether any operation of `jaxpr`, or of a jaxpr inside it, gives a two-dimensional array of a unit
    axis beside a longer one."""
    for eqn in jaxpr.eqns:
        shapes = [getattr(var.aval, "shape", ()) for var in eqn.outvars]
        if any(len(shape) == 2 and 1 in shape and max(shape) > 1 for shape in shapes):
            return True
        if any(_unit_axis(inner) for inner in jax_core.jaxprs_in_params(eqn.params)):
            return True
    return False


class _Hooked(NamedTuple):
    """A call's hooks as its programs read them: the score hook's parameters and the output hook's, as their
    kinds lay them out (`Hooks.score_params`', `[H]`, `[B, H, Qp]`, `[B, H, Kp]`, `[B, H, Qp, Kp]` or
    `[B, Qp, H, Dv]`), and the softmax's initial state, its maximum and denominator `[B, H, Qp]` and sums
    `[B, H, Qp, Dv]` (None where it starts from none)."""

    score: tuple[jax.Array, ...]
    output: tuple[jax.Array, ...]
    initial: tuple[jax.Array, jax.Array, jax.Array] | None


def _hooked_of(hooks: Hooks, call: PaddedCall, unit: int) -> _Hooked:
    """`hooks` at the padded call's shapes (`padded_call`), the value widths of the output hook's parameters
    and the initial state's sums in whole rows of `unit` entries (`_columns`), as the launch's values are."""
    initial = None
    if hooks.initial is not None:
        state = hooks.initial
        widened = ((0, 0),) * 3 + ((0, call.value.shape[-1] - state.acc.shape[-1]),)
        acc = jnp.pad(call.rows(state.acc), widened)
        initial = call.rows(state.m, -jnp.inf), call.rows(state.total), _columns(acc, unit)
    output = tuple(_columns(value, unit) if param.kind == Kind.OUTPUT else value
                   for value, param in zip(call.output_params, hooks.output_params, strict=True))
    return _Hooked(call.score_params, output, initial)


class _Guard(NamedTuple):
    """A guarded launch's arrays (`algebra.walked_launches`): the rows that shift, `live` `[B, H, Qp]` (int32,
    nonzero where live), and each key's values' largest magnitude, `largest` `[B, Hk, Kp]`, which a step's
    `algebra.room` reads (its values lie in a slot's shared memory)."""

    live: jax.Array
    largest: jax.Array


def _param(family: mosaic.Family, ref, kind: Kind, row, head, first, like, k_first=None):
    """A hook parameter's values for a tile `like` of queries from `first` (and keys from `k_first`, a score
    tile's), as `kind` lays them out: the head's scalar, else the tile's entries (a score tile's `[bq, bk]`,
    an output tile's `[bq, Dv]`) in the accumulator's layout, its queries' and keys' broadcast. A score hook
    reads them and its coordinates as the tile's entries, where the reference and the other mainloops give a
    query's and a key's as `[bq, 1]` and `[1, bk]`: Mosaic GPU's layouts hold no singleton axis an elementwise
    operation takes (jax 0.11.2's layout inference finds none for a `[64, 1]` or `[1, 64]` array's convert or
    compare, a lowering probe for an H100), so a hook that reshapes or indexes an axis fails to lower here,
    never silently; a pointwise hook is the same function of either."""
    rows = pl.ds(first, like.shape[0])
    match kind:
        case Kind.HEAD:
            return ref[head]
        case Kind.QUERY:
            return algebra.expand(family.rows_of(ref.at[row, head, rows]), like)
        case Kind.KEY:
            return algebra.across(family.columns_of(ref.at[row, head, pl.ds(k_first, like.shape[1])]), like)
        case Kind.TILE:
            return family.tile_of(ref.at[row, head, rows, pl.ds(k_first, like.shape[1])])
        case Kind.OUTPUT:
            return family.tile_of(ref.at[row, rows, head])


class _Spans(NamedTuple):
    """A spans call's arrays as its programs read them: each query's interval `start`, `end` `[B, Qp]`, and
    its block map at the launch's tiles (`attention.block_map`): the key tiles each query block reads, `lo`,
    `hi`, and sees whole, `full_lo`, `full_hi` (`[B, Qp // block_q]`), and the query tiles reading each key
    block, `q_lo`, `q_hi` (`[B, Kp // block_k]`); with documents, the queries' and keys' ids (`[B, Qp]`,
    `[B, Kp]`) and the blocks' uniform ones (`query_doc`, `key_doc`)."""

    start: jax.Array
    end: jax.Array
    lo: jax.Array
    hi: jax.Array
    full_lo: jax.Array
    full_hi: jax.Array
    q_lo: jax.Array
    q_hi: jax.Array
    query_ids: jax.Array | None = None
    key_ids: jax.Array | None = None
    query_doc: jax.Array | None = None
    key_doc: jax.Array | None = None


def _spans_of(spec: algebra.Forward, call: PaddedCall, block_q: int, block_k: int) -> _Spans | None:
    """A spans call's `_Spans` at `block_q` by `block_k` tiles over its padded keys (None for a full or
    causal call, whose programs take their tiles from their places)."""
    if spec.form != "spans":
        return None
    blocks = block_map(KeySpans(call.start, call.end, call.query_ids, call.key_ids, call.key.shape[1]),
                       block_q, block_k)
    return _Spans(call.start, call.end, blocks.lo, blocks.hi, blocks.full_lo, blocks.full_hi, blocks.q_lo,
                  blocks.q_hi, call.query_ids, call.key_ids, blocks.query_doc, blocks.key_doc)


@dataclasses.dataclass(frozen=True)
class _Program[T: (Tiles, BackwardTiles)]:
    """A program's static geometry: its family's bundle, its tiles, the call's key tiles (`steps`), the slots
    a stream holds, its key count, whether it is causal, and its query count (`algebra.Forward.queries`; a
    tile holding padding rows is never whole)."""

    family: mosaic.Family
    tiles: T
    steps: int
    stages: int
    keys: int
    causal: bool
    queries: int | None = None

    def padded(self, first) -> jax.Array | bool:
        """Whether the query tile from `first` holds padding rows past the call's queries."""
        if self.queries is None or self.queries % self.tiles.block_q == 0:
            return False
        return first + self.tiles.block_q > self.queries

    def query_tile(self, count: int):
        """This program's query tile of the grid's `count`: a causal call's longest first (its last query
        tiles see the most keys), so the grid's tail holds its shortest programs, as FlashAttention-3's
        scheduler orders them; a full call's in order."""
        index = lax.axis_index("q_tiles")
        return count - 1 - index if self.causal else index

    def __post_init__(self):
        unit = self.family.rows
        if self.tiles.block_q % unit or self.tiles.block_k % unit:
            raise ValueError(f"{self.tiles}: query and key tiles of whole {unit} rows ({self.family.name})")

    def walked(self, first):
        """The key tiles a query block from `first` reads: those holding the call's keys (none wholly past
        them, `algebra.seen`), up to its diagonal where causal."""
        held = min(-(-self.keys // self.tiles.block_k), self.steps)
        if not self.causal:
            return jnp.int32(held)
        return jnp.minimum(pl.cdiv(first + self.tiles.block_q, self.tiles.block_k), held)

    def whole(self, first, walked):
        """The tiles every query of the block from `first` sees whole: those wholly within the call's keys
        (the padded call's last tiles may all be past them), and below its diagonal."""
        whole = jnp.int32(self.keys // self.tiles.block_k)
        if self.causal:
            whole = jnp.minimum(lax.div(first, self.tiles.block_k), whole)
        return jnp.minimum(jnp.maximum(whole, 0), walked)

    def reach(self, first, spans: _Spans | None, row):
        """The key tiles a query block from `first` reads, `[lo, hi)`, and whether it sees tile `index` whole
        (`whole(index)`, a tile that takes no mask): a full or causal call's from its places (`walked`,
        `whole`), a spans call's from its block map (`spans`, batch row `row`), whole where the ids of
        both blocks agree too."""
        if spans is None:
            walked = self.walked(first)
            # A block holding padding rows is whole nowhere: its padding sees no key (`algebra.seen`).
            count = jnp.where(self.padded(first), 0, self.whole(first, walked))
            return jnp.int32(0), walked, lambda index: index < count
        block = lax.div(first, self.tiles.block_q)
        full_lo, full_hi = spans.full_lo[row, block], spans.full_hi[row, block]

        def whole(index):
            inside = (index >= full_lo) & (index < full_hi)
            if spans.query_doc is None or spans.key_doc is None:
                return inside
            doc = spans.query_doc[row, block]
            return inside & (doc != 0) & (doc == spans.key_doc[row, index])

        return spans.lo[row, block], spans.hi[row, block], whole

    def readers(self, k_first, q_steps: int, spans: _Spans | None, row):
        """The query tiles reading a key block from `k_first`, `[lo, hi)` of the call's `q_steps`, and
        whether tile `index` sees it whole (`whole(index)`): a full or causal call's from its places, from
        its diagonal and none for a block wholly past the keys; a spans call's from its block map, whole
        where the ids of both blocks agree too (`reach`)."""
        block_q, block_k = self.tiles.block_q, self.tiles.block_k
        if spans is None:
            # The first query tile that sees the block: every one, or those from its diagonal (a causal call
            # of more keys than queries leaves its last keys unseen).
            lo = jnp.minimum(lax.div(k_first, block_q), q_steps) if self.causal else jnp.int32(0)
            lo = jnp.where(k_first >= self.keys, q_steps, lo)
            # The tiles from which every query sees the block whole: past its diagonal, and none if it holds
            # padding.
            whole = jnp.minimum(pl.cdiv(k_first + block_k - 1, block_q), q_steps) if self.causal else lo
            whole = jnp.where(k_first + block_k > self.keys, q_steps, jnp.maximum(whole, lo))
            # A query tile holding padding rows sees no block whole (`reach`).
            return lo, jnp.int32(q_steps), lambda index: (index >= whole) & jnp.logical_not(
                self.padded(index * block_q))
        block = lax.div(k_first, block_k)

        def seen_whole(index):
            inside = (block >= spans.full_lo[row, index]) & (block < spans.full_hi[row, index])
            if spans.query_doc is None or spans.key_doc is None:
                return inside
            doc = spans.key_doc[row, block]
            return inside & (doc != 0) & (doc == spans.query_doc[row, index])

        return spans.q_lo[row, block], spans.q_hi[row, block], seen_whole


def _streamed(reaches: list) -> tuple[jax.Array, jax.Array]:
    """The tiles a program streams: those any of its consumers' `reaches` (`[lo, hi)` each) reads."""
    return (functools.reduce(jnp.minimum, [lo for lo, _ in reaches]),
            functools.reduce(jnp.maximum, [hi for _, hi in reaches]))


def _coords(family: mosaic.Family, spans: _Spans | None, row, first, k_first, shape: tuple[int, int],
            q_pos=None, columns=None) -> algebra.Coords:
    """A tile's coordinates (`algebra.Coords`): its queries' and keys' places from `first` and `k_first`, the
    tile's entries in the accumulator's layout (`q_pos` the query places and `columns` the tile's key
    offsets, where the caller holds them; a hook reads them so, `_param`), and a spans call's queries'
    intervals and, with documents, both blocks' ids."""
    q_pos = first + family.iota(shape, 0) if q_pos is None else q_pos
    k_pos = k_first + (family.iota(shape, 1) if columns is None else columns)
    if spans is None:
        return algebra.Coords(q_pos, k_pos)
    rows = (row, pl.ds(first, shape[0]))
    bounds = family.rows_of(spans.start.at[rows]), family.rows_of(spans.end.at[rows])
    ids = None
    if spans.query_ids is not None and spans.key_ids is not None:
        ids = (family.rows_of(spans.query_ids.at[rows]),
               family.columns_of(spans.key_ids.at[row, pl.ds(k_first, shape[1])]))
    return algebra.Coords(q_pos, k_pos, bounds, ids)


class _Residuals(NamedTuple):
    """A training forward consumer's residual views and the shared stages it stores them through: its rows'
    maxima, denominators, anchor's own weight, other keys' weights' sum and their shift (`[bq]` fp32 views,
    through the consumer's row stage), its anchor keys (`[bq]` int32, through `keys_smem`) and its split
    sums (`[bq, Dv]` fp32, through `others_smem`)."""

    rows: tuple
    keys: object
    others: object
    keys_smem: object
    others_smem: object


def _consumer(program: _Program, spec: algebra.Forward, consumer, q_view, out_view, status_views, q_smem,
              q_loaded, out_smem, row_smem, streams, turn, streamed: tuple, first, spans: _Spans | None,
              hooked: _Hooked, guard: _Guard | None, place: tuple, *, hooks: Hooks,
              residuals: _Residuals | None, interpret: bool):
    """A consumer's query block (batch row, head and key head `place`): its walk of the key tiles it reads
    through the family's streams (`algebra.plain_step` a step) from the hooks' initial state, the score hook
    on each tile, passing those of the program's `streamed` range its partner alone reads; then its output
    under the output hook stored through `out_smem` (the query's stage where the widths agree), and its rows'
    maxima and whether their weighted values overflowed through `row_smem` to `status_views`
    (`algebra.walked_launches`). The plain walk's running sums are the family's accumulator, issued into in
    place; the guarded walk (`guard`, its rows `live` shifting) carries them, each tile's product fresh
    (`algebra.stepped`). With `residuals` the walk is the training step's (`algebra.training_step`'s
    arithmetic): the accumulator holds the split sums, every key's weighted values but the anchor's, taken
    in place (`algebra.split_taken`), the anchor's value row kept in registers, and its residuals stored."""
    row, head, kv_head = place
    family, tiles = program.family, program.tiles
    shape = (tiles.block_q, tiles.block_k)
    lo, hi, whole = program.reach(first, spans, row)
    family.loaded(((q_view, q_smem, q_loaded),))
    q = family.held(q_smem)
    q_pos, columns = first + family.iota(shape, 0), family.iota(shape, 1)
    # The output's sums are as wide as its stage, the values' width whatever the query's.
    value_width = out_smem.shape[-1]
    # The weighted values' running sums live in the family's accumulator (`Family.walk`), from the initial
    # state's where there is one; the walk carries the maxima and denominators, the state's `acc` an empty
    # place the algebra passes through (`algebra.Keys.accumulate`), a constant no loop may carry.
    state_rows = (row, head, pl.ds(first, tiles.block_q))
    live = None if guard is None else family.rows_of(guard.live.at[state_rows]) > 0
    if hooked.initial is not None:
        m0, total0, acc0 = hooked.initial
        start = algebra.resumed(spec, family.rows_of(m0.at[state_rows]),
                                family.rows_of(total0.at[state_rows]), family.tile_of(acc0.at[state_rows]),
                                live=live)
    else:
        # `algebra.started`'s empty state, in the family's layouts.
        start = algebra.Partial(family.zeros((tiles.block_q, value_width)),
                                family.full((tiles.block_q,), -jnp.inf), family.zeros((tiles.block_q,)))
        start = start if live is None else algebra.headroomed(start, live)
    pl.when(consumer == 1)(turn)
    # The slots before this block's first tile its partner reads, the turns it would have taken.
    family.passed(streams, streamed[0], lo, first=streamed[0], between=lambda: (turn(), turn()))

    def scored(index, slots):
        # A key tile's logits under the score hook, and their products where the logits are a factor of
        # them. A tile every query sees whole takes no mask: one walk, the choice a tile's (a walk of each,
        # these rings' waits and arrivals in two loops, faulted on an H100 under jax 0.11.2's lowering).
        s = family.product(q, slots["keys"].transpose((1, 0)), shape, turn)
        slots.release("keys")
        k_first = index * tiles.block_k
        coords = _coords(family, spans, row, first, k_first, shape, q_pos, columns)
        params = tuple(_param(family, ref, param.kind, row, head, first, s, k_first)
                       for ref, param in zip(hooked.score, hooks.score_params, strict=True))
        logits, products = algebra.logits_of(spec, s, coords, params, masked=jnp.logical_not(whole(index)))
        return k_first, coords, logits, products

    def value_of(slots):
        # The value slot, once the partner's products are issued: its turn on the tensor cores is over when
        # this one's come.
        turn(wait=False)
        slot = slots["values"]
        turn(arrive=False)
        return slot

    def dot(lhs, rhs):
        return family.registers_product(lhs, rhs, interpret)

    def step(index, slots, carry, accumulators):
        k_first, coords, logits, products = scored(index, slots)
        load = functools.partial(value_of, slots)
        if guard is not None:
            magnitudes = guard.largest

            def largest():
                return family.columns_of(magnitudes.at[row, kv_head, pl.ds(k_first, tiles.block_k)])

            keys = algebra.Keys(coords, logits, load, dot, products=products, largest=largest)
            return algebra.plain_step(spec, carry, keys), accumulators
        held = list(accumulators)

        def accumulate(scale, p, value):
            held[0] = family.rescaled_products(held[0], scale, ((p, None),), value, interpret)

        state = algebra.Partial(jnp.zeros((), jnp.float32), *carry)
        state = algebra.plain_step(spec, state, algebra.Keys(coords, logits, load, dot, accumulate, products))
        return (state.m, state.total), tuple(held)

    trained = None
    if residuals is not None:
        state, trained = _trained(spec, family, start, scored, value_of, (lo, hi), streams, streamed[0],
                                  kept_dtype=q_smem.dtype, interpret=interpret)
    elif guard is not None:
        state, _ = family.walk(streams, ((step, lo, hi),), start, (), interpret, first=streamed[0])
    else:
        rows, (acc,) = family.walk(streams, ((step, lo, hi),), (start.m, start.total),
                                   ((tiles.block_q, value_width),), interpret, first=streamed[0],
                                   initial=(start.acc,))
        state = algebra.Partial(acc, *rows)
    # The slots past this block's last tile (a causal block whose diagonal ends first) its partner reads.
    family.passed(streams, hi, streamed[1], first=streamed[0], between=lambda: (turn(), turn()))
    pl.when(consumer == 0)(turn)
    overflowed = jnp.where(algebra.overflowed(state.acc), 1.0, 0.0)
    family.stored((family.as_row(state.m), family.as_row(overflowed)), status_views, smem=row_smem)
    out_params = tuple(_param(family, ref, param.kind, row, head, first, state.acc)
                       for ref, param in zip(hooked.output, hooks.output_params, strict=True))
    out, maximum, denominator = algebra.finish(spec, state, initial=hooked.initial is not None,
                                               output_mod=hooks.output_mod, output_params=out_params)
    family.stored((out,), (out_view,), smem=out_smem)
    if residuals is not None and trained is not None:
        # The split sums as the walk left them, shift 0 (`algebra.split_residuals`' plain walk).
        others, own, others_total, key = trained
        family.stored((others,), (residuals.others,), smem=residuals.others_smem)
        rows = (maximum, denominator, own, others_total, family.zeros((tiles.block_q,)))
        family.stored(tuple(family.as_row(part) for part in rows), residuals.rows, smem=row_smem)
        family.stored((family.as_row(key),), (residuals.keys,), smem=residuals.keys_smem)


def _trained(spec: algebra.Forward, family: mosaic.Family, start: algebra.Partial, scored, value_of, reach,
             streams, first, *, kept_dtype, interpret: bool) -> tuple[algebra.Partial, tuple]:
    """A training forward consumer's walk of its key tiles `reach` (`[lo, hi)`, the streams' steps from
    `first`) from `start`: `algebra.training_step`'s arithmetic with the split sums in the family's
    accumulator, taken in place before the tile's other keys' product joins them (`algebra.split_taken`),
    and the old anchor's value row the one kept in registers, picked from a value tile by its one-hot
    product where the anchor moved. `scored(index, slots)` gives a tile's place, coordinates, logits and
    products, and `value_of(slots)` its value slot. Returns the state with the split sums joined
    (`algebra.split_output`, the weighted values the output normalizes) and the residuals the walk leaves:
    the split sums, the anchor's own weight, the other keys' weights' sum and the anchor keys."""
    rows, width = start.acc.shape

    def step(index, slots, carry, accumulators):
        m, total, peak, key, own, others_total, kept = carry
        _, coords, logits, products = scored(index, slots)
        new, alpha = algebra.peak(spec, m, logits, products)
        p = algebra.weights(spec, logits, new)
        value = value_of(slots)
        anchor = algebra.argmax(algebra.Argmax(peak, key), logits, coords.k_pos)
        moved = anchor.peak > peak
        at_anchor = coords.k_pos == algebra.expand(anchor.key, coords.k_pos)
        others, own_tile = algebra.anchor_split(algebra.narrow(p, value.dtype), at_anchor)
        (acc,) = accumulators
        acc = family.rescaled_products(acc, lambda taken: algebra.split_taken(taken, own, alpha, moved, kept),
                                       ((others, None),), value, interpret)
        own, others_total = algebra.split_rows(own, others_total, alpha, moved, others, own_tile)
        # The anchor's value row where it moved to this tile: its one-hot product with the tile's values.
        picked = family.registers_product(at_anchor.astype(value.dtype), value, interpret)
        kept = jnp.where(algebra.expand(moved, kept), picked.astype(kept.dtype), kept)
        total = algebra.denominator_after(total, alpha, p)
        return (new, total, anchor.peak, anchor.key, own, others_total, kept), (acc,)

    nothing = family.zeros((rows,))
    no_key = family.full((rows,), 0, jnp.int32)
    carry = (start.m, start.total, family.full((rows,), -jnp.inf), no_key, nothing, nothing,
             family.zeros((rows, width)).astype(kept_dtype))
    (m, total, _, key, own, others_total, kept), (acc,) = family.walk(
        streams, ((step, *reach),), carry, ((rows, width),), interpret, first=first, initial=(start.acc,))
    return algebra.Partial(algebra.split_output(acc, own, kept), m, total), (acc, own, others_total, key)


def _launch(query, key, value, spans: _Spans | None, hooked: _Hooked, guard: _Guard | None, *,
            spec: algebra.Forward, hooks: Hooks, tiles: Tiles, family: mosaic.Family, training: bool,
            interpret: bool):
    """The forward's launch in `spec`'s form (`guard`'s rows shifting where given): its output `[B, Qp, H,
    Dv]` and its rows' status `[2, B, H, Qp]` fp32, their maxima in the walk's units and 1 where their
    weighted values overflowed (`algebra.walked_launches`); in `training` its residuals too (`_Residuals`):
    the rows `[5, B, H, Qp]` fp32 (maxima, denominators, the anchor's own weight, the other keys' weights'
    sum and their shift), the anchor keys `[B, H, Qp]` int32 and the split sums `[B, Qp, H, Dv]` fp32."""
    batch, queries, heads, width = query.shape
    keys_count, kv_heads = key.shape[1], key.shape[2]
    group, consumers = heads // kv_heads, tiles.consumers
    block_q, block_k = tiles.block_q, tiles.block_k
    pair = consumers * block_q
    steps = keys_count // block_k
    # No more stages than key tiles: a slot past them would load past the keys.
    program = _Program(family, tiles, steps, min(tiles.stages, steps), spec.keys, spec.form == "causal",
                       spec.queries)
    # The output stores through the query's stage where the two are as wide (its rows are read before the
    # store); else it takes a stage of its own beside the query's and the rings (`_forward_staging`).
    reuse = value.shape[-1] == width
    staging = _forward_staging(tiles, width, value.shape[-1], query.dtype, training)

    def body(q_ref, k_ref, v_ref, *refs, streams):
        read, refs = refs[:len(leaves)], refs[len(leaves):]
        (out_ref, status_ref), refs = refs[:2], refs[2:]
        trained: tuple = ()
        if training:
            trained, refs = refs[:3], refs[3:]
        (q_smem, row_smem, *stages), (q_loaded, schedule) = refs[:len(staging)], refs[len(staging):]
        out_smem, stages = (q_smem, stages) if reuse else (stages[0], stages[1:])
        spans_of, hooked_of, guard_of = treedef.unflatten(read)
        row, head = lax.axis_index("batch"), lax.axis_index("heads")
        views = _views_of({"keys": k_ref, "values": v_ref}, row, lax.div(head, group))
        streams = family.opened(streams, views)
        pair_first = program.query_tile(queries // pair) * pair
        streamed = _streamed([program.reach(pair_first + index * block_q, spans_of, row)[:2]
                              for index in range(consumers)])

        def turn(*, arrive: bool = True, wait: bool = True):
            # The consumers' turns on the tensor cores, where a program has two (a schedule barrier).
            if consumers > 1 and arrive:
                plgpu.barrier_arrive(schedule)
            if consumers > 1 and wait:
                plgpu.barrier_wait(schedule)

        def consumer(index):
            first = pair_first + index * block_q
            rows = (row, pl.ds(first, block_q), head)
            status = tuple(status_ref.at[place, row, head, pl.ds(first, block_q)] for place in range(2))
            written = None
            if training:
                rows_ref, keys_ref, others_ref = trained
                keys_smem, others_smem = stages
                written = _Residuals(
                    tuple(rows_ref.at[place, row, head, pl.ds(first, block_q)] for place in range(5)),
                    keys_ref.at[row, head, pl.ds(first, block_q)], others_ref.at[rows], keys_smem.at[index],
                    others_smem.at[index])
            _consumer(program, spec, index, q_ref.at[rows], out_ref.at[rows], status, q_smem.at[index],
                      q_loaded.at[index], out_smem.at[index], row_smem.at[index], streams, turn, streamed,
                      first, spans_of, hooked_of, guard_of, (row, head, lax.div(head, group)), hooks=hooks,
                      residuals=written, interpret=interpret)

        def producer():
            family.stream(streams, views, *streamed)

        family.roles(consumers, consumer, producer)

    blocks = _forward_blocks(tiles, width, value.shape[-1], query.dtype)
    # The arrays the programs read beside the operands.
    leaves, treedef = jax.tree.flatten((spans, hooked, guard))

    def entry(*refs):
        family.scoped(
            functools.partial(body, *refs), *(plgpu.SMEM(shape, dtype) for shape, dtype in staging),
            plgpu.Barrier(num_barriers=consumers), plgpu.Barrier(num_arrivals=consumers),
            streams=family.streams(blocks, program.stages, consumers))

    out_type = (jax.ShapeDtypeStruct((batch, queries, heads, value.shape[-1]), query.dtype),
                jax.ShapeDtypeStruct((2, batch, heads, queries), jnp.float32))
    if training:
        out_type += (jax.ShapeDtypeStruct((5, batch, heads, queries), jnp.float32),
                     jax.ShapeDtypeStruct((batch, heads, queries), jnp.int32),
                     jax.ShapeDtypeStruct((batch, queries, heads, value.shape[-1]), jnp.float32))
    return family.kernel(entry, out_type=out_type, grid=(heads, queries // pair, batch),
                         grid_names=("heads", "q_tiles", "batch"), consumers=consumers,
                         interpret=interpret)(query, key, value, *leaves)


def _parts(accurate: bool, dtype) -> int:
    """The parts a backward tile's score cotangents take to their products in the form `accurate` names
    (`algebra.cotangent_parts`: one in the fast form's bf16, three in the accurate form's)."""
    tile = jax.ShapeDtypeStruct((1, 1), jnp.float32)
    return len(jax.eval_shape(lambda c: algebra.cotangent_parts(c, dtype, accurate=accurate), tile))


def _dq_blocks(config: BackwardTiles, width: int, value_width: int, dtype) -> dict:
    """The dQ launch's streams' blocks: a key tile and its value tile."""
    return {"keys": ((config.block_k, width), dtype), "values": ((config.block_k, value_width), dtype)}


def _dq_staging(config: BackwardTiles, width: int, value_width: int, dtype) -> list:
    """The dQ program's shared memory beside its streams, `(shape, dtype)` each in `_dq_launch`'s order: its
    consumers' query and output-cotangent stages, their dQ output stage (fp32) and their rows' stage."""
    consumers, block_q = config.consumers, config.block_q
    return [((consumers, block_q, width), dtype), ((consumers, block_q, value_width), dtype),
            ((consumers, block_q, width), jnp.float32), ((consumers, block_q), jnp.float32)]


def _dkv_blocks(config: BackwardTiles, width: int, value_width: int, dtype) -> dict:
    """The dK/dV launch's streams' blocks: a query tile and its output cotangents."""
    return {"queries": ((config.block_q, width), dtype), "douts": ((config.block_q, value_width), dtype)}


def _dkv_staging(config: BackwardTiles, width: int, value_width: int, dtype, *, parts: int,
                 column_scratch: int) -> list:
    """The dK/dV program's shared memory beside its streams, `(shape, dtype)` each in `_dkv_launch`'s order:
    its consumers' resident keys and values; a tile's weights and score cotangents' `parts` staged for the
    transposed products (`algebra.value_operands`, `algebra.cotangent_parts`); dK's and dV's output stage,
    which a fused launch's dQ tiles take too (fp32); and each consumer's scratch for the key gradients' sums
    across its warps (`Family.column_sums`)."""
    consumers, block_q, block_k = config.consumers, config.block_q, config.block_k
    return [((consumers, block_k, width), dtype), ((consumers, block_k, value_width), dtype),
            ((consumers, 1 + parts, block_q, block_k), dtype),
            ((consumers, max(block_q, block_k), width), jnp.float32),
            ((consumers, column_scratch), jnp.float32)]


def _backward_registers(family: mosaic.Family, config: BackwardTiles, width: int, value_width: int, dtype, *,
                        accurate: bool, dq: bool) -> int:
    """`_registers` of a backward launch's consumer at `config`: a dQ consumer's dQ sums `[bq, D]` fp32
    carried (and the accurate form's keyed sums), beside a step's logits and output cotangents' products
    fp32, then its score cotangents fp32 and their parts in the operands' dtype; a dK/dV consumer's dK and dV
    sums `[bk, D]`, `[bk, Dv]` fp32 carried, beside a step's two products, and a fused launch's dQ tile
    `[bq, D]` fp32."""
    block_q, block_k = config.block_q, config.block_k
    products = [(block_q, block_k, jnp.float32)] * 2
    if dq:
        carried = [(block_q, width, jnp.float32)] * (2 if accurate else 1)
        cotangents = [(block_q, block_k, jnp.float32), *[(block_q, block_k, dtype)] * _parts(accurate, dtype)]
        return _registers(family, carried, products, cotangents)
    carried = [(block_k, width, jnp.float32), (block_k, value_width, jnp.float32)]
    return _registers(family, carried, products, [(block_q, width, jnp.float32)] if config.fused else [])


@dataclasses.dataclass(frozen=True)
class MosaicBackward(_Mainloop[BackwardTiles]):
    """The template's backward on this mainloop (`gradients`) over the call's own arrays (`backward` gives
    them): its queries, keys, values and output cotangent, its spans' arrays, its score hook's parameters, its
    initial state (`Initial`, or None) and a training forward's residuals (`forward`'s). Returns its
    gradients (`Gradients`). Its configs come from the family's facts and the call's extents (`space`),
    fitted to the device's block and a consumer's registers (`fits`), the cost model's fastest the heuristic
    (`cost`, both launches')."""

    HOOKS = ("score_mod",)
    config_cls = BackwardTiles
    accurate: bool = ACCURATE

    def _fwd(self, query, key, value, dout, start, end, query_ids, key_ids, score_params, initial, residuals,
             *, interpret: bool, config: BackwardTiles, return_residuals: bool = False):
        family = BUNDLES[config.family]
        spans = self.spans(start, end, query_ids, key_ids)
        hooks = Hooks(self.score_mod, tuple(map(Param, score_params, self.score_kinds)), initial)
        spec = algebra.Forward(spans.form, spans.keys, ids=query_ids is not None, scale=self.scale,
                               score_mod=self.score_mod, natural=self.natural, initial=initial is not None,
                               queries=start.shape[1])
        launch = functools.partial(_launch_backward, spec=spec, accurate=self.accurate, tiles=config,
                                   family=family, interpret=interpret)
        return gradients(launch, query, key, value, spans, hooks, dout, residuals,
                         block=_block(config), scale=self.scale), None

    def space(self, ba: tokamax.BoundArguments) -> Sequence[BackwardTiles]:
        """Every config the backward takes on each family of `BUNDLES` at the call's extents: each count of
        consumers the family has, most first; each query and key block in whole multiples of its unit
        (`_extents`), largest first; the two launches, then the fast form's one where the call has no hook
        (`fused`); at each stage count the walks and the device's block take (`_stages`). A consumer
        thread's registers bound them where they fit (`fits`, `register_budget`)."""
        query, key = ba.arguments["query"], ba.arguments["key"]
        hooked = self.score_mod is not None or ba.arguments["initial"] is not None
        queries, keys, dtype = query.shape[1], key.shape[1], query.dtype
        configs = []
        for family in BUNDLES.values():
            for consumers in range(family.consumers, 0, -1):
                for block_q in _extents(family, dtype, -(-queries // consumers)):
                    for block_k in _extents(family, dtype, -(-keys // consumers)):
                        for fused in (False,) if self.accurate or hooked else (False, True):
                            tiles = BackwardTiles(block_q, block_k, 2, consumers, fused, family.name)
                            padded_q, padded_k = _padded(queries, keys, tiles)
                            configs += self.staged(ba, tiles, max(padded_q // block_q, padded_k // block_k))
        return tuple(configs)

    def staged_bytes(self, ba: tokamax.BoundArguments, config: BackwardTiles,
                     target: generation.GpuTarget) -> int:
        """The larger of its launches' programs' shared memory (`_dq_staging`, `_dkv_staging`) and their
        streams' slots, each allocation from an `ALIGNMENT` boundary; a fused backward's dK/dV program's
        alone."""
        family = BUNDLES[config.family]
        query, key, value = (ba.arguments[name] for name in ("query", "key", "value"))
        dtype = query.dtype
        width, value_width = (_width(array.shape[-1], family, dtype) for array in (query, value))
        padded_q, padded_k = _padded(query.shape[1], key.shape[1], config)
        dkv = _staged([*_dkv_staging(config, width, value_width, dtype, parts=_parts(self.accurate, dtype),
                                     column_scratch=family.column_scratch),
                       *_streams(_dkv_blocks(config, width, value_width, dtype),
                                 min(config.stages, padded_q // config.block_q))])
        if config.fused:
            return dkv
        dq = _staged([*_dq_staging(config, width, value_width, dtype),
                      *_streams(_dq_blocks(config, width, value_width, dtype),
                                min(config.stages, padded_k // config.block_k))])
        return max(dq, dkv)

    def registers(self, ba: tokamax.BoundArguments, config: BackwardTiles) -> int:
        family = BUNDLES[config.family]
        query, value = ba.arguments["query"], ba.arguments["value"]
        width, value_width = (_width(array.shape[-1], family, query.dtype) for array in (query, value))
        return max(_backward_registers(family, config, width, value_width, query.dtype,
                                       accurate=self.accurate, dq=dq) for dq in (False, True))

    def cost(self, ba: tokamax.BoundArguments, config: BackwardTiles, target: generation.GpuTarget) -> float:
        """Its launches' seconds (`_backward_costs`)."""
        return sum(cost.seconds(costed, target) for costed in _backward_costs(self, ba, config, target))

    def refusal(self, ba: tokamax.BoundArguments, family: mosaic.Family) -> str | None:
        query, value = ba.arguments["query"], ba.arguments["value"]
        hooks = Hooks(self.score_mod, tuple(map(Param, ba.arguments["score_params"], self.score_kinds)),
                      ba.arguments["initial"])
        return refused(query.dtype, query.shape[3], value.shape[3], hooks, backward=True, family=family)

    def traced(self, name: str, value: Callable | str | None) -> str | None:
        return hook_program(name, value, self.score_kinds, self.score_dtypes, differentiated=True)


def _backward_costs(op: MosaicBackward, ba: tokamax.BoundArguments, config: BackwardTiles,
                    target: generation.GpuTarget) -> list[cost.Cost]:
    """Each part's work at `config` (`cost.Cost`): the dQ launch's programs (a query head's consumers' query
    blocks) walking the key tiles their blocks reach (`_walked`), each consumer step's logits, output
    cotangents', dQ and the accurate form's keyed sums' products; the dK/dV launch's (a key head's
    consumers' key blocks) walking the query tiles reading them (`_readers`), each step's logits, output
    cotangents', dV and dK products, its parts staged between them (`serial_staged`), and a fused launch's
    dQ tile, added into dQ in fp32 through its output stage and the L2 (read and written) in place of the
    dQ launch. Their shared memory's traffic: each streamed tile written once and read by each consumer's
    products, each resident or staged block written once, each output staged and stored. Every array read
    once a launch and every gradient written once in fp32; and the passes between and around the launches
    (`gradients`, `dq_finish`; a fused launch's dQ zeroed and summed), over the device's memory. With one
    consumer the softmax runs between the products, with two under the partner's
    (`serial_exponentials`)."""
    family = BUNDLES[config.family]
    query, key, value = (ba.arguments[name] for name in ("query", "key", "value"))
    batch, queries, heads, _ = query.shape
    keys, dtype = key.shape[1], query.dtype
    size = jnp.dtype(dtype).itemsize
    width, value_width = (_width(array.shape[-1], family, dtype) for array in (query, value))
    consumers, block_q, block_k = config.consumers, config.block_q, config.block_k
    padded_q, padded_k = _padded(queries, keys, config)
    parts = _parts(op.accurate, dtype)
    causal = op.form == "causal"
    resident = cost.resident(target, op.staged_bytes(ba, config, target), mosaic.THREADS * (consumers + 1),
                             op.register_budget(config))
    read = _read_bytes(ba)
    query_rows, key_rows = batch * padded_q * heads, batch * padded_k * heads

    def launch(programs: int, consumed: float, streamed: float, products: float, staged: float, *,
               written: float, serial: float = 0.0, l2: float = 0.0, stages: int) -> cost.Cost:
        exponentials = consumed / programs * block_q * block_k
        return cost.Cost(
            programs=batch * heads * programs, device_bytes=read + written,
            program_flops=consumed / programs * 2.0 * block_q * block_k * products,
            program_staged=staged / programs,
            program_exponentials=exponentials if consumers > 1 else 0.0,
            serial_exponentials=0.0 if consumers > 1 else exponentials,
            serial_staged=consumed / programs * serial, mode="16bit", steps=streamed / programs,
            stages=stages, resident=resident, l2_bytes=l2)

    key_programs = padded_k // (consumers * block_k)
    consumed, streamed = _readers(causal, consumers, block_q, block_k, key_programs, padded_q // block_q,
                                  keys)
    slot = block_q * (width + value_width) * size
    staged_parts = (1 + parts) * block_q * block_k * size
    dkv_staged = (streamed * slot + consumed * (slot + block_k * (width + value_width) * size + staged_parts)
                  + key_programs * consumers * block_k * (width + value_width) * (size + 2 * 4))
    dkv_products = width + 2 * value_width + width * parts + (width if config.fused else 0)
    # A fused step's dQ tile staged and added by TMA, which its next store waits for.
    dkv_serial = staged_parts + (2 * block_q * width * 4 if config.fused else 0)
    written = key_rows * (width + value_width) * 4
    l2 = batch * heads * streamed * slot
    # Around the launches: D from the split sums and the anchor's value row, gathered from the values for
    # each query head; dK and dV summed over a key head's query heads.
    passes = (query_rows * (value_width * (4 + 2 * size) + 4 * 4)
              + key_rows * (width + value_width) * 4 + batch * keys * heads * (width + value_width) * size)
    if config.fused:
        written += query_rows * width * 4
        l2 += 2 * batch * heads * consumed * block_q * width * 4
        # dQ zeroed, then read out of its one copy.
        passes += 3 * query_rows * width * 4
        return [launch(key_programs, consumed, streamed, dkv_products, dkv_staged, written=written,
                       serial=dkv_serial, l2=l2, stages=min(config.stages, padded_q // block_q)),
                cost.Cost(programs=1, device_bytes=passes)]
    costs = [launch(key_programs, consumed, streamed, dkv_products, dkv_staged, written=written,
                    serial=dkv_serial, l2=l2, stages=min(config.stages, padded_q // block_q))]
    query_programs = padded_q // (consumers * block_q)
    consumed, streamed = _walked(causal, consumers, block_q, block_k, query_programs, keys)
    sums = 2 if op.accurate else 1
    slot = block_k * (width + value_width) * size
    # A step reads its queries and cotangents, its key and value slots, and its key slot again for dQ.
    dq_step = block_q * (width + value_width) * size + slot + block_k * width * size
    stages_once = (width + value_width) * size + 2 * 4 * width * sums
    dq_staged = streamed * slot + consumed * dq_step + query_programs * consumers * block_q * stages_once
    dq_products = width + value_width + width * parts * sums
    rows = 4 + 2 * len(op.score_kinds)
    costs.append(launch(query_programs, consumed, streamed, dq_products, dq_staged,
                        written=query_rows * (width * sums + rows) * 4, l2=batch * heads * streamed * slot,
                        stages=min(config.stages, padded_k // block_k)))
    # `dq_finish`: dQ's sums (and the keyed sums) read with their rows, dQ written, fp32.
    passes += query_rows * (width * (sums + 1) + rows) * 4
    return [*costs, cost.Cost(programs=1, device_bytes=passes)]


def _rows(family: mosaic.Family, ref, row, head, first, count: int):
    """A query block's per-row statistic `[count]` of a `[B, H, Q]` array, in a row's layout."""
    return family.rows_of(ref.at[row, head, pl.ds(first, count)])


def _views_of(refs: dict, row, head) -> dict:
    """The streams' views by name (`Family.opened`, `Family.stream`, `mosaic.viewed`): each `[B, S, H, D]`
    operand's `[S, D]` at batch row `row` and head `head`, step `j` its `j`-th block of rows."""
    return {name: (ref.at[row, :, head], lambda j: (j, 0)) for name, ref in refs.items()}


@dataclasses.dataclass(frozen=True)
class _Backward:
    """A backward launch's static choices: its program's geometry, its forward's (`spec`: the call's form, the
    logits' scale and units, its score hook and initial state), its form, its consumers, whether it is
    interpreted, the score hook's parameter kinds (`kinds`), and whether the dK/dV program adds each tile's
    dQ too (`fused`: the hookless fast form's one launch, dQ by atomic adds, D the forward's anchored one
    unrefined)."""

    program: _Program[BackwardTiles]
    spec: algebra.Forward
    accurate: bool
    consumers: int
    interpret: bool
    fused: bool
    kinds: tuple[Kind, ...]

    @property
    def how(self) -> algebra.Recompute:
        """How the backward recomputes the forward's weights."""
        return algebra.recompute(self.spec)

    @property
    def scale(self) -> float:
        """The logits' scale."""
        return self.spec.scale

    @property
    def summed(self) -> tuple[bool, ...]:
        """Which score parameters the dQ program sums per query (`DQ_GRADS`); the others' gradients the dK/dV
        program writes."""
        return tuple(kind in DQ_GRADS for kind in self.kinds)


def _score_params(family: mosaic.Family, refs, kinds: tuple[Kind, ...], row, head, first, like,
                  k_first=None) -> tuple:
    """A tile's score hook parameters (`_param`), one per kind of `kinds`, read from their refs."""
    return tuple(_param(family, ref, kind, row, head, first, like, k_first)
                 for ref, kind in zip(refs, kinds, strict=True))


def _scores(walk: _Backward, coords: algebra.Coords, masked: bool, params: tuple, s, dp, peak, total, anchor,
            delta, chosen):
    """A backward tile's weights unnormalized and over `total`, its score cotangents (`algebra.grad_scores`)
    and the score hook's slopes (`algebra.tile_logits`), from its products `s` and `dp`, its coordinates, its
    hook parameters `params`, its mask (`algebra.seen` where `masked`) and its queries' statistics."""
    seen = algebra.seen(walk.spec, coords, s) if masked else None
    how = walk.how
    logits, slopes = algebra.tile_logits(walk.spec.score_mod, s, seen, coords.q_pos, coords.k_pos, params,
                                         scale=walk.spec.scale if how.factor is None else 1.0, slopes=True)
    exponentials, weights = algebra.normalized(logits, peak, total, how)
    at_anchor = coords.k_pos == algebra.expand(chosen, coords.k_pos)
    cotangents = algebra.grad_scores(exponentials, dp, anchor, delta, seen, at_anchor, algebra.inverse(total))
    return exponentials, weights, cotangents, slopes, seen


def _dq_step(walk: _Backward, operands: tuple, stats: tuple, coords_of, params_of, whole, index, slots, rows,
             accumulators):
    """A dQ consumer's key tile `index`: its scores and score cotangents from the key and value slots
    (`slots`) and its queries and output cotangents (`operands`) under its rows' statistics (`stats`), the
    tile's coordinates (`coords_of(k_first)`), its hook parameters (`params_of(like, k_first=k_first)`) and
    whether its block sees it whole (`whole(index)`). The row sums are carried (`rows`); dQ's and the keyed
    sums' products are issued into `accumulators` in place (`algebra.dq_issued`)."""
    q, do = operands
    peak, prior, anchor, delta, chosen = stats
    family, tiles, empty = walk.program.family, walk.program.tiles, jnp.zeros((), jnp.float32)
    shape = (tiles.block_q, tiles.block_k)
    k_slot = slots["keys"]
    k_first = index * tiles.block_k
    s = family.product(q, k_slot.transpose((1, 0)), shape)
    dp = family.product(do, slots["values"].transpose((1, 0)), shape)
    slots.release("values")
    params = params_of(s, k_first=k_first)
    residual, mass, grads, weighed, keyed_mass, keyed_shift = rows
    # dQ's and the keyed sums' places stay empty: their products go to the accumulators.
    sums = algebra.DqSums(empty, residual, mass, grads, weighed, empty, keyed_mass, keyed_shift)

    def issued(*, masked: bool):
        # A tile every query sees whole (`masked` False) takes no mask; the row sums and the parts are taken
        # where the mask is read (`algebra.pulled`).
        scored = _scores(walk, coords_of(k_first), masked, params, s, dp, peak, prior, anchor, delta, chosen)
        exponentials, _, cotangents, slopes, seen = scored
        terms, parts, keyed = algebra.dq_issued(sums, exponentials, cotangents, slopes, seen, k_slot.dtype,
                                                prior, scale=walk.scale, kinds=walk.kinds,
                                                summed=walk.summed, accurate=walk.accurate)
        row_sums = (terms.residual, terms.mass, terms.grads, terms.weighed, terms.keyed_mass,
                    terms.keyed_shift)
        return row_sums, parts, keyed

    # One walk, its tiles' masks chosen a tile at a time: a walk of two loops into one warpgroup accumulator,
    # the first empty, faults on an H100 under jax 0.11.2's Mosaic GPU lowering.
    rows, parts, keyed = lax.cond(whole(index), functools.partial(issued, masked=False),
                                  functools.partial(issued, masked=True))
    dq_acc, *keyed_accs = accumulators
    dq_acc = family.register_products(dq_acc, parts, k_slot, walk.interpret)
    if keyed is None:
        return rows, (dq_acc,)
    factor, keyed_parts = keyed
    if factor is None:
        keyed_acc = family.register_products(keyed_accs[0], keyed_parts, k_slot, walk.interpret)
    else:
        # The keyed sums rescaled by their row factor, then the tile's products added in place.
        def rescale(value):
            return value * algebra.expand(factor, value)

        keyed_acc = family.rescaled_products(keyed_accs[0], rescale, keyed_parts, k_slot, walk.interpret)
    return rows, (dq_acc, keyed_acc)


def _dq_launch(query, key, value, dout, peak, prior, anchor, delta, key_index, spans: _Spans | None,
               score_params: tuple, *, walk: _Backward):
    """The dQ program's sums over its key tiles (`algebra.dq_issued`), unfinished: dQ's and, in the accurate
    form, the keys' product with the weights (`DqSums.keyed`), `[B, Qp, H, D]` fp32 each; and the rows
    `[4 + 2n, B, H, Qp]` fp32 (n the dQ parameters, `walk.summed`): the cotangents' and weights' row sums, the
    keyed sums' mass and exponent, then each dQ parameter's gradient and its weights' pullback, which
    `dq_finish` takes past the launch. Each query block of a pair is a consumer's, the key and value tiles
    streamed through a ring each. Returns dQ, the keyed sums (None in the fast form) and the rows."""
    batch, queries, heads, width = query.shape
    kv_heads, value_width = key.shape[2], value.shape[-1]
    program, consumers, kinds = walk.program, walk.consumers, walk.kinds
    family = program.family
    block_q, block_k = program.tiles.block_q, program.tiles.block_k
    pair = consumers * block_q
    summed = walk.summed
    rows_count = 4 + 2 * sum(summed)
    leaves, treedef = jax.tree.flatten((spans, score_params))

    def body(*refs, streams):
        (q_ref, k_ref, v_ref, do_ref, peak_ref, prior_ref, anchor_ref, delta_ref, index_ref), refs = (
            refs[:9], refs[9:])
        (spans_of, score_refs), refs = treedef.unflatten(refs[:len(leaves)]), refs[len(leaves):]
        dq_ref, refs = refs[0], refs[1:]
        keyed_ref, refs = (refs[0], refs[1:]) if walk.accurate else (None, refs)
        rows_ref, (q_smem, do_smem, out_smem, row_smem, q_loaded, do_loaded) = refs[0], refs[1:]
        row, head = lax.axis_index("batch"), lax.axis_index("heads")
        views = _views_of({"keys": k_ref, "values": v_ref}, row, lax.div(head, heads // kv_heads))
        streams = family.opened(streams, views)
        pair_first = program.query_tile(queries // pair) * pair
        streamed = _streamed([program.reach(pair_first + index * block_q, spans_of, row)[:2]
                              for index in range(consumers)])

        def consumer(index):
            first = pair_first + index * block_q
            rows = (row, pl.ds(first, block_q), head)
            family.loaded(((q_ref.at[rows], q_smem.at[index], q_loaded.at[index]),
                           (do_ref.at[rows], do_smem.at[index], do_loaded.at[index])))
            stats = tuple(_rows(family, ref, row, head, first, block_q)
                          for ref in (peak_ref, prior_ref, anchor_ref, delta_ref, index_ref))
            lo, hi, whole = program.reach(first, spans_of, row)
            nothing = family.zeros((block_q,))
            operands = family.held(q_smem.at[index]), family.held(do_smem.at[index])
            shape = (block_q, block_k)
            q_pos, columns = first + family.iota(shape, 0), family.iota(shape, 1)
            coords_of = functools.partial(_coords, family, spans_of, row, first, shape=shape, q_pos=q_pos,
                                          columns=columns)
            params_of = functools.partial(_score_params, family, score_refs, kinds, row, head, first)
            step = functools.partial(_dq_step, walk, operands, stats, coords_of, params_of, whole)
            start = (nothing, nothing, (nothing,) * len(kinds), (nothing,) * len(kinds), nothing,
                     nothing + algebra.NO_SHIFT)
            # The tiles before this block's first its partner reads.
            family.passed(streams, streamed[0], lo, first=streamed[0])
            (residual, mass, grads, weighed, keyed_mass, keyed_shift), accumulated = family.walk(
                streams, ((step, lo, hi),), start, ((block_q, width),) * (2 if walk.accurate else 1),
                walk.interpret, first=streamed[0])
            # The tiles past this block's last (a causal block whose diagonal ends first) its partner reads.
            family.passed(streams, hi, streamed[1], first=streamed[0])
            dq, *keyed = accumulated
            written = [(dq, dq_ref.at[rows])]
            if keyed_ref is not None:
                written.append((keyed[0], keyed_ref.at[rows]))
            family.stored(*zip(*written, strict=True), smem=out_smem.at[index])
            # Each row in a row's layout whatever its own (a call without a hook's keyed mass is a constant).
            summed_grads = tuple(part for part, on in zip(grads, summed, strict=True) if on)
            summed_weighed = tuple(part for part, on in zip(weighed, summed, strict=True) if on)
            parts = (residual, mass, keyed_mass, keyed_shift, *summed_grads, *summed_weighed)
            family.stored(tuple(family.as_row(part) for part in parts),
                          tuple(rows_ref.at[place, row, head, pl.ds(first, block_q)]
                                for place in range(rows_count)),
                          smem=row_smem.at[index])

        def producer():
            family.stream(streams, views, *streamed)

        family.roles(consumers, consumer, producer)

    blocks = _dq_blocks(program.tiles, width, value_width, query.dtype)

    def entry(*refs):
        family.scoped(
            functools.partial(body, *refs),
            *(plgpu.SMEM(shape, dtype) for shape, dtype in _dq_staging(program.tiles, width, value_width,
                                                                       query.dtype)),
            plgpu.Barrier(num_barriers=consumers), plgpu.Barrier(num_barriers=consumers),
            streams=family.streams(blocks, program.stages, consumers))

    sums = jax.ShapeDtypeStruct((batch, queries, heads, width), jnp.float32)
    stats_shape = jax.ShapeDtypeStruct((rows_count, batch, heads, queries), jnp.float32)
    out = family.kernel(
        entry, out_type=(sums, sums, stats_shape) if walk.accurate else (sums, stats_shape),
        grid=(heads, queries // pair, batch), grid_names=("heads", "q_tiles", "batch"), consumers=consumers,
        interpret=walk.interpret)(query, key, value, dout, peak, prior, anchor, delta, key_index, *leaves)
    dq, keyed, rows = out if walk.accurate else (out[0], None, out[1])
    return dq, keyed, rows


def _dkv_step(walk: _Backward, resident: tuple, rows_of, buffers, coords_of, params_of, whole, add_dq,
              store_tiles, sums_scratch, index, slots, carry, accumulators):
    """A dK/dV consumer's query tile `index`: its scores and score cotangents from the query and output
    cotangent slots (`slots`) and its resident keys and values (`resident`) under the tile's rows' statistics
    (`rows_of(first query)`), its coordinates (`coords_of(first query)`), its hook parameters
    (`params_of(first query, like)`) and whether it sees the keys whole (`whole(index)`). dV's and dK's
    products are issued into `accumulators` in place, each part staged in `buffers` as its transposed
    product's left operand; the key gradients' column sums (`carry`) take the pulled cotangents' key
    sums (`Family.column_sums` through the consumer's `sums_scratch`), and the tile gradients go to
    `store_tiles(first query, pieces)`."""
    k_smem, v_smem = resident
    family, tiles, kinds = walk.program.family, walk.program.tiles, walk.kinds
    shape = (tiles.block_q, tiles.block_k)
    dk_acc, dv_acc = accumulators
    q_slot, do_slot = slots["queries"], slots["douts"]
    q_first = index * tiles.block_q
    peak, denominator, centre, rest, chosen = rows_of(q_first)
    s = family.product(q_slot, k_smem.transpose((1, 0)), shape)
    dp = family.product(do_slot, v_smem.transpose((1, 0)), shape)
    params = params_of(q_first, s)

    def scores(*, masked: bool):
        # A tile every query sees whole (`masked` False) takes no mask; the pulled cotangents and the hook's
        # pieces are taken where the mask is read (`algebra.pulled`).
        exponentials, weights, cotangents, slopes, seen = _scores(
            walk, coords_of(q_first), masked, params, s, dp, peak, denominator, centre, rest, chosen)
        column_sums = functools.partial(family.column_sums, scratch=sums_scratch, interpret=walk.interpret)
        logits, pieces = algebra.pulled(cotangents, slopes, seen, kinds, column_sums)
        return exponentials, weights, logits, pieces

    # The tiles a query block sees whole unmasked, the rest masked, chosen a tile at a time in one walk
    # (`_dq_step`).
    exponentials, weights, logits, pieces = lax.cond(whole(index), functools.partial(scores, masked=False),
                                                     functools.partial(scores, masked=True))
    narrowed = algebra.value_weights(exponentials, weights, do_slot.dtype)
    parts = algebra.cotangent_parts(logits * walk.scale, q_slot.dtype, accurate=walk.accurate)
    # Each staged part's transpose the left operand (`algebra.cotangent_product`'s `aᵀ @ b`).
    dv_acc = family.staged_products(dv_acc, ((narrowed, None),), do_slot, buffers.at[pl.ds(0, 1)],
                                    walk.interpret, transposed=True)
    dk_acc = family.staged_products(dk_acc, parts, q_slot, buffers.at[pl.ds(1, len(parts))], walk.interpret,
                                    transposed=True)
    if add_dq is not None:
        # The tile's dQ, its staged cotangents times the resident keys (`algebra.cotangent_product`'s
        # `a @ b`), added into dQ: the query tile's other key blocks add theirs.
        add_dq(q_first, family.product(buffers.at[1], k_smem, (tiles.block_q, k_smem.shape[-1])))
    store_tiles(q_first, [piece for kind, piece in zip(kinds, pieces, strict=True) if kind == Kind.TILE])
    key_pieces = [piece for kind, piece in zip(kinds, pieces, strict=True) if kind == Kind.KEY]
    # The staged parts' reads done (the walk's settling) before the next tile's overwrite them.
    return tuple(total + piece for total, piece in zip(carry, key_pieces, strict=True)), (dk_acc, dv_acc)


def _dkv_launch(query, key, value, dout, peak, total, anchor, delta, key_index, spans: _Spans | None,
                score_params: tuple, *, walk: _Backward) -> tuple[jax.Array, ...]:
    """The dK/dV program under the dQ program's softmax: each consumer's `block_k` keys of a query head,
    resident, walked by the query tiles that see them, streamed with their output cotangents through a ring
    each; a tile's weights and score cotangents (the algebra's, rows the queries) staged in shared memory as
    the transposed products' left operands (`algebra.value_operands`, `cotangent_product`). Returns the
    query head's dK and dV `[B, Kp, H, D]` fp32, then each key and tile gradient of the score hook in kind
    order (`DKV_GRADS`: keys `[B, H, Kp]`, tiles `[B, H, Qp, Kp]`); `fused`, dQ `[B, Qp, H, D]` fp32 before
    them, each tile's staged cotangents times the resident keys added into it (`Family.added`). A key or
    tile gradient is written by the one program that holds its keys, straight from the registers; the tiles
    its walk does not read are zeros."""
    batch, queries, heads, width = query.shape
    keys_padded, kv_heads, value_width = key.shape[1], key.shape[2], value.shape[-1]
    program, consumers, kinds = walk.program, walk.consumers, walk.kinds
    family = program.family
    block_q, block_k = program.tiles.block_q, program.tiles.block_k
    pair = consumers * block_k
    q_steps = queries // block_q
    stages = min(program.tiles.stages, q_steps)
    graded = [kind for kind in kinds if kind in DKV_GRADS]
    leaves, treedef = jax.tree.flatten((spans, score_params))

    def body(*refs, streams):
        (q_ref, k_ref, v_ref, do_ref, peak_ref, total_ref, anchor_ref, delta_ref, index_ref), refs = (
            refs[:9], refs[9:])
        (spans_of, score_refs), refs = treedef.unflatten(refs[:len(leaves)]), refs[len(leaves):]
        dq_ref, refs = (refs[0], refs[1:]) if walk.fused else (None, refs)
        dk_ref, dv_ref, *refs = refs
        grad_refs, refs = refs[:len(graded)], refs[len(graded):]
        k_smem, v_smem, staged, out_smem, sums_smem, k_loaded, v_loaded = refs
        key_refs = [ref for ref, kind in zip(grad_refs, graded, strict=True) if kind == Kind.KEY]
        tile_refs = [ref for ref, kind in zip(grad_refs, graded, strict=True) if kind == Kind.TILE]
        row, head = lax.axis_index("batch"), lax.axis_index("heads")

        def add_dq(dq_ref, index, q_first, dq):
            # A tile's dQ through the consumer's output stage, free until its dK and dV are stored;
            # interpreted, into a copy of the consumer's own (`Family.added`).
            view = dq_ref.at[index if walk.interpret else 0, row, pl.ds(q_first, block_q), head]
            family.added(dq, view, walk.interpret, smem=out_smem.at[index, pl.ds(0, block_q)])

        views = _views_of({"queries": q_ref, "douts": do_ref}, row, head)
        streams = family.opened(streams, views)
        pair_first = lax.axis_index("k_tiles") * pair
        streamed = _streamed([program.readers(pair_first + index * block_k, q_steps, spans_of, row)[:2]
                              for index in range(consumers)])

        def rows_of(q_first):
            return tuple(_rows(family, ref, row, head, q_first, block_q)
                         for ref in (peak_ref, total_ref, anchor_ref, delta_ref, index_ref))

        def consumer(index):
            k_first = pair_first + index * block_k
            keys_view = (row, pl.ds(k_first, block_k), lax.div(head, heads // kv_heads))
            family.loaded(((k_ref.at[keys_view], k_smem.at[index], k_loaded.at[index]),
                           (v_ref.at[keys_view], v_smem.at[index], v_loaded.at[index])))
            lo, hi, whole = program.readers(k_first, q_steps, spans_of, row)

            def store_tiles(first, pieces):
                # Each tile's gradient entries straight to their place in the `[B, H, Qp, Kp]` outputs.
                for ref, piece in zip(tile_refs, pieces, strict=True):
                    ref[row, head, pl.ds(first, block_q), pl.ds(k_first, block_k)] = piece

            # The tiles before this block's first reader its partner reads.
            family.passed(streams, streamed[0], lo, first=streamed[0])
            adds = None if dq_ref is None else functools.partial(add_dq, dq_ref, index)
            coords_of = functools.partial(_coords, family, spans_of, row, k_first=k_first,
                                          shape=(block_q, block_k))
            params_of = functools.partial(_score_params, family, score_refs, kinds, row, head,
                                          k_first=k_first)
            step = functools.partial(_dkv_step, walk, (k_smem.at[index], v_smem.at[index]), rows_of,
                                     staged.at[index], coords_of, params_of, whole, adds, store_tiles,
                                     sums_smem.at[index])
            key_sums, (dk, dv) = family.walk(streams, ((step, lo, hi),),
                                             tuple(family.column_zeros(block_k) for _ in key_refs),
                                             ((block_k, width), (block_k, value_width)), walk.interpret,
                                             first=streamed[0])
            # The tiles past this block's last reader its partner reads.
            family.passed(streams, hi, streamed[1], first=streamed[0])
            out = (row, pl.ds(k_first, block_k), head)
            family.stored((dk, dv), (dk_ref.at[out], dv_ref.at[out]),
                          smem=out_smem.at[index, pl.ds(0, block_k)])
            if key_refs:
                # Each key gradient, the walk's column sums, straight to the `[B, H, Kp]` outputs.
                for ref, sums in zip(key_refs, key_sums, strict=True):
                    ref[row, head, pl.ds(k_first, block_k)] = sums
            if tile_refs:
                # The tiles the walk does not read: no query of this block sees them, so they are zeros.
                zeros = family.zeros((block_q, block_k))

                @pl.loop(0, lo)
                def _(tile):
                    store_tiles(tile * block_q, [zeros] * len(tile_refs))

                @pl.loop(hi, q_steps)
                def _(tile):
                    store_tiles(tile * block_q, [zeros] * len(tile_refs))

        def producer():
            family.stream(streams, views, *streamed)

        family.roles(consumers, consumer, producer)

    blocks = _dkv_blocks(program.tiles, width, value_width, query.dtype)
    staging = _dkv_staging(program.tiles, width, value_width, query.dtype,
                           parts=_parts(walk.accurate, query.dtype), column_scratch=family.column_scratch)

    def entry(*refs):
        family.scoped(
            functools.partial(body, *refs), *(plgpu.SMEM(shape, dtype) for shape, dtype in staging),
            plgpu.Barrier(num_barriers=consumers), plgpu.Barrier(num_barriers=consumers),
            streams=family.streams(blocks, stages, consumers))

    gradient = jax.ShapeDtypeStruct((batch, keys_padded, heads, width), jnp.float32)
    grad_shapes = {Kind.KEY: jax.ShapeDtypeStruct((batch, heads, keys_padded), jnp.float32),
                   Kind.TILE: jax.ShapeDtypeStruct((batch, heads, queries, keys_padded), jnp.float32)}
    launch = family.kernel(
        entry, out_type=(gradient, gradient, *(grad_shapes[kind] for kind in graded)),
        grid=(heads, keys_padded // pair, batch), grid_names=("heads", "k_tiles", "batch"),
        consumers=consumers, interpret=walk.interpret)
    operands = (query, key, value, dout, peak, total, anchor, delta, key_index, *leaves)
    if not walk.fused:
        return launch(*operands)
    # dQ added into zeros in place: one copy, or each consumer's own interpreted (`Family.added`).
    copies = consumers if walk.interpret else 1
    dq = jax.new_ref(jnp.zeros((copies, batch, queries, heads, width), jnp.float32))
    dk, dv = launch(*operands, dq)
    return jax.freeze(dq).sum(axis=0), dk, dv


def backward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None,
             dout: jax.Array, *, residuals: tuple[jax.Array, jax.Array, jax.Array, Split],
             scale: float | None = None, natural: bool = NATURAL, accurate: bool = ACCURATE,
             interpret: bool = False) -> Gradients:
    """The template's backward on this mainloop (`MosaicBackward`, `gradients`) from a training forward's
    `residuals` (its maxima, denominators, anchor keys and split sums): D about each query's anchor from the
    split sums (`algebra.split_delta`), a dQ launch and a dK/dV launch, and `dq_finish`'s refinement between
    them, in the form `accurate` names; or the fast form's one launch (`BackwardTiles.fused`,
    `_launch_backward`), a schedule the cost model ranks beside the two launches for a hookless call. Every
    call form of bf16 operands, under every score hook and the initial state. The output hook is no part of
    it: a differentiated call applies `output_mod` past the kernel, where its own derivative runs, and passes
    `dout` before it. `interpret` runs it on the CPU, as a GPU of the family `generation.assumed` names."""
    op, args = _backward_call(query, key, value, spans, hooks, dout, residuals, scale=scale, natural=natural,
                              accurate=accurate)
    return op(*args, interpret=interpret)


def backward_runs(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None,
                  dout: jax.Array, *, residuals: tuple[jax.Array, jax.Array, jax.Array, Split],
                  scale: float | None = None, natural: bool = NATURAL, accurate: bool = ACCURATE) -> bool:
    """Whether `backward` runs the call compiled on this device (`MosaicBackward.eligible`): its family has
    a bundle here, the call is one this mainloop takes (`refused`) and a config fits the device."""
    op, args = _backward_call(query, key, value, spans, hooks, dout, residuals, scale=scale, natural=natural,
                              accurate=accurate)
    return op.eligible(op.bind(*args, interpret=False), generation.gpu_target())


def _backward_call(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None,
                   dout: jax.Array, residuals: tuple, *, scale: float | None, natural: bool,
                   accurate: bool) -> tuple[MosaicBackward, tuple]:
    """`backward`'s op and its arguments."""
    hooks = Hooks() if hooks is None else hooks
    op = MosaicBackward(score_mod=hooks.score_mod,
                        score_kinds=tuple(param.kind for param in hooks.score_params),
                        score_dtypes=tuple(kept_dtype(param.value.dtype) for param in hooks.score_params),
                        form=spans.form, keys=spans.keys,
                        scale=query.shape[3] ** -0.5 if scale is None else scale, natural=natural,
                        accurate=accurate)
    return op, (query, key, value, dout, spans.start, spans.end, spans.query_ids, spans.key_ids,
                tuple(param.value for param in hooks.score_params), hooks.initial, residuals)


def _launch_backward(query, key, value, dout, peak0, total0, initial_term, maximum, key_index, anchor_value,
                     shift, others, own, others_total, forward_total, start, end, query_ids, key_ids,
                     score_params, *, score_mod, score_kinds, score_dtypes, scale: float,
                     spec: algebra.Forward, accurate: bool, tiles: BackwardTiles, family: mosaic.Family,
                     interpret: bool):
    """`gradients`' launch on this mainloop: D from the forward's split sums, each query's anchor key as the
    first pass chose it. Two launches: the dQ program's sums, their finish (`algebra.dq_finish`) and the dK/dV
    program under its refined D, whose cotangents sum to 0 where the first pass's do not (a small weight on a
    large value, bf16: dK's worst row 2.5e1 at the first pass's D, where check 1 allows 1.55e-2), the accurate
    form's always, and a score hook's and an initial state's, whose gradients need the refined D
    (`algebra.dq_step`). The fast form's one launch (`tiles.fused`, a hookless call's): the dK/dV program
    adding each tile's dQ, under the first pass's D and the forward's denominator. Neither is the faster
    everywhere: on an H100 at llama-d128 bf16 2 x 4096, 32 heads, the one launch takes 2.53 ms causal and 4.87
    full, two launches 2.67 and 4.35 (the one launch's dQ reductions, each consumer's 64 keys reducing its own
    tile into global memory, twice FlashAttention-3's traffic, cost most where every key block reads every
    query), and the one launch holds 384 MB of temporaries against 258 (its dQ in fp32), so the cost model
    ranks them (`MosaicBackward.cost`). Returns what `gradients` takes: dQ, the softmax's rows, each query's
    anchor key, the dQ parameters' gradients (`DQ_GRADS`; a head's as per-query sums `[1, B, H, 1]`, which
    `gradients` sums over the batch), then dK, dV and the dK/dV parameters' (`DKV_GRADS`)."""
    del score_mod, score_dtypes, scale
    width, value_width = query.shape[-1], value.shape[-1]
    _, state_total = algebra.off_anchor_start(peak0, total0, maximum, value_width, shift)
    cotangent = jnp.swapaxes(dout, 1, 2).astype(jnp.float32)
    anchor, delta, chosen = algebra.split_delta(others, own, others_total, forward_total, state_total,
                                                jnp.swapaxes(anchor_value, 1, 2), cotangent, key_index, shift)
    # The family's swizzled tiles hold whole rows (`Family.column_unit`).
    query, key, value, dout = (_columns(array, family.column_unit(array.dtype))
                               for array in (query, key, value, dout))
    steps = key.shape[1] // tiles.block_k
    program = _Program(family, tiles, steps, min(tiles.stages, steps), spec.keys, spec.form == "causal",
                       spec.queries)
    walk = _Backward(program, spec, accurate, tiles.consumers, interpret, tiles.fused, score_kinds)
    padded = PaddedCall(query, key, value, start, end, query_ids, key_ids, (), (), query.shape[1], spec.keys)
    spans = _spans_of(spec, padded, tiles.block_q, tiles.block_k)
    if walk.fused:
        dq, dk, dv = _dkv_launch(query, key, value, dout, maximum, forward_total, anchor, delta, chosen,
                                 spans, score_params, walk=walk)
        mass = algebra.split_mass(own, others_total, forward_total, shift)
        return (dq[..., :width], delta, anchor, maximum, forward_total, mass, chosen, dk[..., :width],
                dv[..., :value_width])
    dq_sum, keyed, rows = _dq_launch(query, key, value, dout, maximum, forward_total, anchor, delta, chosen,
                                     spans, score_params, walk=walk)
    count = sum(walk.summed)
    dq_swapped = jnp.swapaxes(dq_sum, 1, 2)
    # The fast form's keyed sums are never read (`dq_finish`): a placeholder there.
    keyed = jnp.zeros_like(dq_swapped) if keyed is None else jnp.swapaxes(keyed, 1, 2)
    sums = algebra.DqSums(dq_swapped, rows[0], rows[1], tuple(rows[4:4 + count]), tuple(rows[4 + count:]),
                          keyed, rows[2], rows[3])
    dq, refined, denominator, mass, dq_grads = algebra.dq_finish(sums, forward_total, state_total, peak0,
                                                                  (total0, initial_term), maximum, anchor,
                                                                  delta, accurate=accurate)
    dk, dv, *key_tile = _dkv_launch(query, key, value, dout, maximum, denominator, anchor, refined, chosen,
                                    spans, score_params, walk=walk)
    summed_kinds = [kind for kind in score_kinds if kind in DQ_GRADS]
    head_query = tuple(grad.sum(axis=-1)[None, :, :, None] if kind == Kind.HEAD else grad
                       for grad, kind in zip(dq_grads, summed_kinds, strict=True))
    return (jnp.swapaxes(dq, 1, 2)[..., :width], refined, anchor, maximum, denominator, mass, chosen,
            *head_query, dk[..., :width], dv[..., :value_width], *key_tile)
