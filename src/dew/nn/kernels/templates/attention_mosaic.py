"""The attention template's Mosaic GPU mainloop as Template implementations (`MosaicForward`,
`MosaicBackward`; design §6 and §7): the forward, its training form and the backward over every call form,
their configs generated from the instruction family's facts and the call's extents, never a table, fitted to
the device's block and a consumer's registers, and ranked by the cost model from the allocations, registers
and work their launches take. A GPU nobody measured of a family with a bundle here (`BUNDLES`) takes its own
facts' pick. The launches are `attention_mosaic_forward`'s and `attention_mosaic_backward`'s; what they share,
`attention_mosaic_program`'s."""

import dataclasses
import functools
import math
from collections.abc import Callable, Sequence
from typing import Literal, overload

import jax
import jax.numpy as jnp
import numpy as np
import tokamax
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels import generation
from dew.nn.kernels.templates import attention_algebra as algebra, cost, mosaic
from dew.nn.kernels.templates.attention import (
    ACCURATE,
    DQ_GRADS,
    NATURAL,
    TRACED_TILE,
    Gradients,
    Hooks,
    KeySpans,
    Kind,
    Param,
    Split,
    gradients,
    kept_dtype,
    power_of_two,
)
from dew.nn.kernels.templates.attention_mosaic_backward import (
    _dkv_allocations,
    _dq_allocations,
    _launch_backward,
)
from dew.nn.kernels.templates.attention_mosaic_forward import (
    _forward_allocations,
    _plain_forward,
    _trained_forward,
)
from dew.nn.kernels.templates.attention_mosaic_program import (
    BUNDLES,
    BackwardTiles,
    Tiles,
    _block,
    _padded,
    _parts,
    _width,
    refused,
)
from dew.nn.kernels.templates.base import Template


def _extents(family: mosaic.Family, dtype, extent: int) -> tuple[int, ...]:
    """A tile axis's extents on `family`, largest first: whole multiples of the rows a consumer's products
    cover (`Family.role_rows`) and of its swizzled rows (`Family.column_unit`; a tile's keys are its weights'
    product's depth, its queries the dK/dV launch's), at most what one copy moves (`Family.copy_extent`,
    where the family's copies bound it; else registers bound them, `fits`) and the call's `extent` in whole
    multiples (a longer tile holds padding alone)."""
    unit = math.lcm(family.role_rows, family.column_unit(dtype))
    top = -(-max(extent, 1) // unit)
    if family.copy_extent is not None:
        top = min(top, family.copy_extent // unit)
    return tuple(range(top * unit, 0, -unit))


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
    """The shared memory a program's `allocations` take, as `Family.scoped` takes them (`plgpu.SMEM` arrays
    and `plgpu.Barrier`s, its streams' allocations among them, nested in sequences and mappings): every
    array rounded up to the toolkit's alignment (`mosaic.allocated`), then every barrier
    (`mosaic.BARRIER_BYTES`), as JAX's Pallas Mosaic GPU lowering lays a kernel's arrays out and its barriers
    after them."""
    arrays, barriers = [], 0
    for allocation in jax.tree.leaves(allocations):
        if isinstance(allocation, plgpu.Barrier):
            barriers += int(np.prod(allocation.num_barriers))
        else:
            arrays.append(math.prod(allocation.shape) * jnp.dtype(allocation.dtype).itemsize)
    return mosaic.allocated(*arrays) + mosaic.BARRIER_BYTES * barriers


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


def _readers(causal: bool, consumers: int, block_q: int, block_k: int, programs: int, q_steps: int,
             keys: int) -> tuple[int, int]:
    """`_walked` for a dK/dV launch's `programs` key programs over `q_steps` query tiles: the tiles reading
    each consumer's keys (every one, from its diagonal where `causal`; none for keys past the call's,
    `_Program.readers`), summed over its consumers, and the tiles each program streams."""
    consumed = streamed = 0
    for program in range(programs):
        reaches = [0 if first >= keys else q_steps - (min(first // block_q, q_steps) if causal else 0)
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
    forward's output hook) as fields; the configs of the target's family (`fits`), a consumer's registers
    its roles' share (`register_budget`), where its bundle is one of `BUNDLES` and the call is one it
    takes (`refused`)."""

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
        """The program's shared memory (`_forward_allocations`, `_staged`), at the call's widths as the launch
        stages them (`_width`)."""
        family = BUNDLES[config.family]
        query, key, value = (ba.arguments[name] for name in ("query", "key", "value"))
        width, value_width = (_width(array.shape[-1], family, query.dtype) for array in (query, value))
        steps = _padded(query.shape[1], key.shape[1], config)[1] // config.block_k
        return _staged(_forward_allocations(family, config, width, value_width, query.dtype,
                                            bool(ba.arguments.get("residuals")), min(config.stages, steps)))

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


@overload
def forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
            *, scale: float | None = None, natural: bool = NATURAL, residuals: Literal[False] = False,
            interpret: bool = False) -> jax.Array:
    ...


@overload
def forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
            *, scale: float | None = None, natural: bool = NATURAL, residuals: Literal[True],
            interpret: bool = False) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, Split]:
    ...


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
    op, args = _forward_call(query, key, value, spans, hooks, scale=scale, natural=natural)
    return op(*args, residuals=residuals, interpret=interpret)


def forward_takes(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans,
                  hooks: Hooks | None = None, *, scale: float | None = None, natural: bool = NATURAL,
                  residuals: bool = False) -> bool:
    """Whether `forward` runs the call compiled on this device (`MosaicForward.eligible`): its family has a
    bundle here, the call is one this mainloop takes (`refused`) and a config fits the device."""
    op, args = _forward_call(query, key, value, spans, hooks, scale=scale, natural=natural)
    return op.eligible(op.bind(*args, residuals=residuals, interpret=False), generation.gpu_target())


def _forward_call(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None, *,
                  scale: float | None, natural: bool) -> tuple[MosaicForward, tuple]:
    """`forward`'s op and its arguments."""
    hooks = Hooks() if hooks is None else hooks
    op = MosaicForward(score_mod=hooks.score_mod, output_mod=hooks.output_mod,
                       score_kinds=tuple(param.kind for param in hooks.score_params),
                       output_kinds=tuple(param.kind for param in hooks.output_params),
                       score_dtypes=tuple(kept_dtype(param.value.dtype) for param in hooks.score_params),
                       output_dtypes=tuple(kept_dtype(param.value.dtype) for param in hooks.output_params),
                       output_width=power_of_two(value.shape[3]), form=spans.form, keys=spans.keys,
                       scale=query.shape[3] ** -0.5 if scale is None else scale, natural=natural)
    return op, (query, key, value, spans.start, spans.end, spans.query_ids, spans.key_ids,
                tuple(param.value for param in hooks.score_params),
                tuple(param.value for param in hooks.output_params), hooks.initial)


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

    config_cls = BackwardTiles
    accurate: bool = ACCURATE

    def _fwd(self, query, key, value, dout, start, end, query_ids, key_ids, score_params, initial, residuals,
             *, interpret: bool, config: BackwardTiles, return_residuals: bool = False):
        family = BUNDLES[config.family]
        spans = self.spans(start, end, query_ids, key_ids)
        hooks = Hooks(self.score_mod, tuple(map(Param, score_params, self.score_kinds)), initial)
        if reason := refused(query.dtype, query.shape[3], value.shape[3], hooks, backward=True,
                             family=family):
            raise NotImplementedError(reason)
        # The call's query count where the launch pads past it (`algebra.Forward.queries`).
        queries = start.shape[1]
        padded = _padded(queries, key.shape[1], config)[0] > queries
        spec = algebra.Forward(spans.form, spans.keys, ids=query_ids is not None, scale=self.scale,
                               score_mod=self.score_mod, natural=self.natural, initial=initial is not None,
                               queries=queries if padded else None)
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
        """The larger of its launches' programs' shared memory (`_dq_allocations`, `_dkv_allocations`,
        `_staged`); a fused backward's dK/dV program's alone."""
        family = BUNDLES[config.family]
        query, key, value = (ba.arguments[name] for name in ("query", "key", "value"))
        dtype = query.dtype
        width, value_width = (_width(array.shape[-1], family, dtype) for array in (query, value))
        padded_q, padded_k = _padded(query.shape[1], key.shape[1], config)
        dkv = _staged(_dkv_allocations(family, config, width, value_width, dtype,
                                       parts=_parts(self.accurate, dtype),
                                       stages=min(config.stages, padded_q // config.block_q)))
        if config.fused:
            return dkv
        dq = _staged(_dq_allocations(family, config, width, value_width, dtype,
                                     min(config.stages, padded_k // config.block_k)))
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


@functools.cache
def _keyed_parts(dtype) -> int:
    """The parts the accurate form's keyed weights take to their product with the keys
    (`algebra.weights_parts`: one in bf16)."""
    tile = jax.ShapeDtypeStruct((1, 1), jnp.float32)
    return len(jax.eval_shape(lambda weights: algebra.weights_parts(weights, dtype), tile))


def _backward_costs(op: MosaicBackward, ba: tokamax.BoundArguments, config: BackwardTiles,
                    target: generation.GpuTarget) -> list[cost.Cost]:
    """Each part's work at `config` (`cost.Cost`), each product's FLOPs and shared-memory reads as the launch
    issues it (a product reads its shared-memory operands, a register operand none):
    - the dQ launch's programs (a query head's consumers' query blocks) walking the key tiles their blocks
      reach (`_walked`), a step's logits `Q·Kᵀ` and output cotangents' `dO·Vᵀ`, then dQ's parts' products
      with the key slot (`algebra.cotangent_parts`) and the accurate form's keyed weights' parts' too
      (`algebra.weights_parts`), the parts in registers;
    - the dK/dV launch's (a key head's consumers' key blocks) walking the query tiles reading them
      (`_readers`), a step's logits and output cotangents' products against its resident keys and values,
      then dV's from its staged weights and dK's from each staged part, against the query and cotangent
      slots, the staged parts written before their products read them (`serial_staged`); a fused launch's
      dQ tile, the staged part times the resident keys, added into dQ in fp32 through its output stage and
      the L2 (read and written) in place of the dQ launch;
    - each streamed tile written once a program, each resident or staged block once, each output staged
      and stored; every array read once a launch and every gradient written once in fp32, the dQ launch's
      rows only for the score kinds it sums (`DQ_GRADS`);
    - the passes between and around the launches (`gradients`, `dq_finish`; a fused launch's dQ zeroed and
      summed), over the device's memory.
    With one consumer the softmax runs between the products, with two under the partner's
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
    keyed = _keyed_parts(dtype) if op.accurate else 0
    fused = 1 if config.fused else 0
    causal = op.form == "causal"
    resident = cost.resident(target, op.staged_bytes(ba, config, target), mosaic.THREADS * (consumers + 1),
                             op.register_budget(config))
    read = _read_bytes(ba)
    query_rows, key_rows = batch * padded_q * heads, batch * padded_k * heads
    tile = block_q * block_k

    def launch(programs: int, consumed: float, streamed: float, products: float, staged: float, *,
               written: float, serial: float = 0.0, l2: float = 0.0, stages: int) -> cost.Cost:
        exponentials = consumed / programs * tile
        return cost.Cost(
            programs=batch * heads * programs, device_bytes=read + written,
            program_flops=consumed / programs * 2.0 * tile * products, program_staged=staged / programs,
            program_exponentials=exponentials if consumers > 1 else 0.0,
            serial_exponentials=0.0 if consumers > 1 else exponentials,
            serial_staged=consumed / programs * serial, mode="16bit", steps=streamed / programs,
            stages=stages, resident=resident, l2_bytes=l2)

    key_programs = padded_k // (consumers * block_k)
    consumed, streamed = _readers(causal, consumers, block_q, block_k, key_programs, padded_q // block_q,
                                  keys)
    slot = block_q * (width + value_width) * size
    # A step's reads: the logits' and cotangents' slots and resident blocks, dV's staged weights and
    # cotangent slot, each dK part and its query slot, a fused step's staged part and resident keys.
    dkv_reads = ((block_q + block_k) * (width + value_width) + tile + block_q * value_width
                 + parts * (tile + block_q * width) + fused * (tile + block_k * width)) * size
    once = key_programs * consumers * block_k * (width + value_width) * (size + 2 * 4)
    dkv_staged = streamed * slot + consumed * dkv_reads + once
    dkv_products = width + 2 * value_width + width * parts + width * fused
    # A step's staged parts, written before their products read them, and a fused step's dQ tile, staged
    # and added by TMA, which its next store waits for.
    dkv_serial = (1 + parts) * tile * size + fused * 2 * block_q * width * 4
    written = key_rows * (width + value_width) * 4
    l2 = batch * heads * streamed * slot
    # Around the launches: D from the split sums and the anchor's value row, gathered from the values for
    # each query head; dK and dV summed over a key head's query heads.
    passes = (query_rows * (value_width * (4 + 2 * size) + 4 * 4)
              + key_rows * (width + value_width) * 4 + batch * keys * heads * (width + value_width) * size)
    stages = min(config.stages, padded_q // block_q)
    if config.fused:
        written += query_rows * width * 4
        l2 += 2 * batch * heads * consumed * block_q * width * 4
        # dQ zeroed, then read out of its one copy.
        passes += 3 * query_rows * width * 4
        return [launch(key_programs, consumed, streamed, dkv_products, dkv_staged, written=written,
                       serial=dkv_serial, l2=l2, stages=stages),
                cost.Cost(programs=1, device_bytes=passes)]
    costs = [launch(key_programs, consumed, streamed, dkv_products, dkv_staged, written=written,
                    serial=dkv_serial, l2=l2, stages=stages)]
    query_programs = padded_q // (consumers * block_q)
    consumed, streamed = _walked(causal, consumers, block_q, block_k, query_programs, keys)
    sums = 2 if op.accurate else 1
    slot = block_k * (width + value_width) * size
    # A step's reads: the logits' and cotangents' stages and slots, the key slot again for each dQ part and
    # keyed part.
    dq_reads = ((block_q + block_k) * (width + value_width) + (parts + keyed) * block_k * width) * size
    once = query_programs * consumers * block_q * ((width + value_width) * size + 2 * 4 * width * sums)
    dq_staged = streamed * slot + consumed * dq_reads + once
    dq_products = width + value_width + width * (parts + keyed)
    rows = 4 + 2 * sum(kind in DQ_GRADS for kind in op.score_kinds)
    costs.append(launch(query_programs, consumed, streamed, dq_products, dq_staged,
                        written=query_rows * (width * sums + rows) * 4, l2=batch * heads * streamed * slot,
                        stages=min(config.stages, padded_k // block_k)))
    # `dq_finish`: dQ's sums (and the keyed sums) read with their rows, dQ written, fp32.
    passes += query_rows * (width * (sums + 1) + rows) * 4
    return [*costs, cost.Cost(programs=1, device_bytes=passes)]


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


def backward_takes(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None,
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
