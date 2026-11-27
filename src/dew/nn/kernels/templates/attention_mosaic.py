"""The attention template's Mosaic GPU mainloop as Template implementations (`MosaicForward`,
`MosaicBackward`; design §6 and §7): the forward, its training form and the backward over every call form,
each launch's one config derived from the instruction family's and the device's facts and the call's extents
(`derived`), legal where its programs stage within the device's block and a consumer's registers hold its live
set (`legal`): no table, no sweep, no space ranked. A GPU nobody measured of a family with a bundle here
(`bundle`) takes its own facts' config. The launches are `attention_mosaic_forward`'s and
`attention_mosaic_backward`'s; what they share, `attention_mosaic_program`'s."""

import dataclasses
import functools
import math
from collections.abc import Callable, Hashable, Sequence, Set
from typing import ClassVar, Literal, overload

import jax
import jax.numpy as jnp
import tokamax

from dew._devices import deterministic_ops_requested
from dew.nn.kernels import generation, tuning
from dew.nn.kernels.selection import Refusal
from dew.nn.kernels.templates import (
    assembled,
    attention_algebra as algebra,
    band,
    cost,
    emission,
    mosaic,
    placed,
    product,
    report,
    resources,
    swizzling,
)
from dew.nn.kernels.templates.attention import (
    ACCURATE,
    DKV_GRADS,
    DQ_GRADS,
    NATURAL,
    TILED,
    AttentionLaunch,
    Gradients,
    Hooks,
    Kept,
    KeySpans,
    Kind,
    Split,
    gradients,
    head_width,
    kept_dtype,
    key_prologue_taken,
    kinds_of,
    logit_scale,
    prologued,
    prologues_taken,
    walked_initial,
)
from dew.nn.kernels.templates.attention_mosaic_backward import _launch_backward
from dew.nn.kernels.templates.attention_mosaic_coop import _coop_allocations
from dew.nn.kernels.templates.attention_mosaic_derivation import (
    WARPGROUP_WARPS,
    _choice_key,
    _column_blocks_of,
    _derived,
    _fewer_hooked,
    _least_keys,
    _planned_stages,
    _priced,
    _query_walk,
    _role_rows,
)
from dew.nn.kernels.templates.attention_mosaic_dkv import _dkv_allocations, dq_halves
from dew.nn.kernels.templates.attention_mosaic_dq import DQ_KERNEL, _dq_allocations
from dew.nn.kernels.templates.attention_mosaic_forward import (
    NATURAL_NAME,
    SplitOutput,
    _codes_of,
    _forward_allocations,
    _plain_walk,
    _training_walk,
    certified_walk,
    column_slices,
    folded_form,
    forward_block_plan,
    forward_products,
    logits_ahead,
    rewalks_in_place,
)
from dew.nn.kernels.templates.attention_mosaic_fused import (
    dq_summed,
    fused_reduces,
    fused_staging,
    staged_parts,
    streamed_statistics,
)
from dew.nn.kernels.templates.attention_mosaic_program import (
    BACKWARD_CONSUMERS,
    POINTWISE_ROWS,
    ROUNDED,
    BackwardTiles,
    Tiles,
    _block,
    _padded,
    _width,
    backward_bundle,
    backward_family,
    backward_layouts,
    bundle,
    bundle_of,
    extents,
    fewer_rows,
    forward_reduces,
    fuses,
    held_walked,
    holders,
    hook_reduces,
    launch_scope,
    layouts,
    loads_parts,
    most_consumers,
    placed_draws,
    placed_scratch,
    pointwise_tiles,
    reducing,
    refused,
    scaled_drop,
    score_places,
    split_queries,
    streams_operands,
    value_columns,
    varying_slopes,
    walked_extents,
    walked_layouts,
    walked_unit,
    whole_tiles,
)
from dew.nn.kernels.templates.attention_mosaic_registers import BackwardRegisters, ForwardRegisters
from dew.nn.kernels.templates.attention_mosaic_scores import step_live_tiles
from dew.nn.kernels.templates.base import Template
from dew.nn.kernels.templates.hooks import NO_HOOK, TRACED_TILE, Hook, Side
from dew.nn.precision import asks_default_precision


@dataclasses.dataclass(frozen=True)
class _Mainloop[Config: (Tiles, BackwardTiles)](AttentionLaunch, Template[Config]):
    """What this mainloop's launches share as Template implementations (`MosaicForward`, `MosaicBackward`):
    the call's own arrays as arguments, padded by a launch to its config's blocks; the score hook (and the
    forward's output hook) as fields keyed by what they compute (`hook_program`); the configs of the
    target's family (`fits`), a consumer's registers its roles' share (`register_budget`), where its bundle
    has a bundle here (`bundle`) and the call is one it takes (`refused`)."""

    name = "mosaic_gpu"
    GENERATES = True
    score_mod: Callable | None = None
    form: str = "spans"
    keys: int = 0
    score_kinds: tuple[Kind, ...] = ()
    score_dtypes: tuple[str, ...] = ()
    score_reduces: bool = False
    """Whether its score hook may reduce across warps (`hook_reduces`), its launches' warpgroups reserving
    their scratch (`reducing`), as its builder reads it off the hook: a keyed launch's hook is its
    fingerprint, which keeps no trace of it."""
    scale: float = 1.0
    natural: bool = NATURAL
    subtracted: bool = False
    """Whether an inference walk's fold takes its logits as a difference of raw products scaled after it is
    taken (`algebra.Forward.subtracted`: no fold's reach, no natural rewalk), a decode step's
    (`attention_decode`), whose walk the memory bounds."""
    per_span: int = 1
    """The queries each of the spans' intervals holds (`KeySpans.per_span`)."""

    @property
    def score_types(self) -> tuple:
        """The score hook's parameters as its kernels load them: each its `Kind` and dtype."""
        return tuple(zip(self.score_kinds, self.score_dtypes, strict=True))

    def spans(self, start, end, query_ids, key_ids) -> KeySpans:
        """The call's spans from its arrays and the op's static form, key count and queries an interval."""
        return KeySpans(start, end, query_ids, key_ids, self.keys, self.form, self.per_span)

    UNBUNDLED: ClassVar[str] = "the Mosaic mainloop has no bundle for {name} yet"
    """Why a launch runs on no device of a family it has no bundle of (`declined`)."""

    @staticmethod
    def _bundle(generation_name: str | None, dtype) -> mosaic.Family | None:
        """The bundle the launch runs on (`bundle`)."""
        return bundle(generation_name, dtype)

    @staticmethod
    def _layouts(generation_name: str | None, dtype, rows: int | None = None) -> tuple[mosaic.Family, ...]:
        """The bundles the launch takes (`layouts`)."""
        return layouts(generation_name, dtype, rows)

    def _named(self, generation_name: str | None, dtype, rows: int | None = None) -> dict[str, mosaic.Family]:
        """Every bundle a config of the launch may name, by name (`legal`): those it takes (`_layouts`)."""
        return {family.name: family for family in self._layouts(generation_name, dtype, rows)}

    def padded(self, ba: tokamax.BoundArguments, config: Config) -> float | None:
        """The share of the score grid, each head's queries by its keys, that the launch's blocks pad
        past the call's (`_padded`, `tuning.KernelOp.padded`): its query and key blocks' tails."""
        queries, keys = ba.arguments["query"].shape[1], _keys(ba)
        padded_queries, padded_keys = _padded(queries, keys, config)
        return 1 - queries * keys / (padded_queries * padded_keys)

    def registers(self, ba: tokamax.BoundArguments, config: Config, *,
                  target: generation.GpuTarget | None = None) -> int | None:
        """Live registers per consumer thread at `config` on `target` (overridden by subclasses)."""
        del ba, config, target
        return None

    def register_budget(self, config: Config, target: generation.GpuTarget | None = None) -> int:
        """A consumer thread's share of the registers (`Family.registers`): the family's roles share the
        file (`handoffs.Schedule`, row 11987)."""
        # A consumer's budget is its family's, whatever the operands' dtype (`mosaic.Family.registers`).
        family = bundle_of(config, jnp.bfloat16, target=target)
        return family.registers(config.consumers)
    def eligible(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> bool:
        return self.declined(ba, target) is None

    def declined(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> Refusal | None:
        """Why the launch runs no call of `ba` on `target`, or None: its family has no bundle here, the call
        is one it does not take (`refusal`, each reason with its ledger row), it is interpreted in a GPU
        process (`grouped_matmul._Mosaic.eligible`: the interpreter's callbacks are ordered effects, which a
        GPU process's lowering cannot carry), or no config fits the device (`derived`), certification
        refusing each it derives (`tuning`'s `_certified`: an A100's accurate backward 128 wide under a
        temperature, every config spilling at 255 registers, which the Pallas-Triton backward takes)."""
        name = generation.family(target.generation)
        query = ba.arguments["query"]
        family = self._bundle(target.generation, query.dtype)
        if family is None:
            return Refusal('family', self.UNBUNDLED.format(name=name or target.generation))
        if (reason := self.refusal(ba, family)) is not None:
            return reason
        if ba.arguments["interpret"] and jax.default_backend() != "cpu":
            return Refusal('interpret', "an interpreted Mosaic launch runs in a CPU process alone")
        if not self._has_config(ba, target):
            return Refusal('resources',
                           f"no config of the Mosaic mainloop fits {target.generation}'s block and registers")
        return None

    def _has_config(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> bool:
        """Whether any config fits `ba` on `target` (`derived(ba, target) is not None`); `MosaicForward`
        overrides to check feasibility without tracing kernels through `price_config`."""
        return self.derived(ba, target) is not None

    def uncertified(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> Refusal | None:
        """Why the launch the route selected runs no call of `ba` on `target` as the GPU assembles its
        configs (`tuning`'s `_certified`: each that spills or serializes its warpgroup MMAs refused, the next
        derived in its place), or None: a call the route sends here alone is certified, once it is selected
        (`attention_route._mainloop`), so a call another mainloop takes compiles none of its kernels. A
        process compiling for no GPU certifies none, and its derived config holds."""
        if not self._has_config(ba, target):
            return self.declined(ba, target)
        if not self.holds(ba):
            # Certification refused every config the derivation offers (`tuning`'s exhausted verdict, None).
            return Refusal('resources', f"every config of the Mosaic mainloop spills or serializes its "
                           f"products as {target.generation}'s assembler builds them")
        return None

    def legal(self, ba: tokamax.BoundArguments, config: Config, target: generation.GpuTarget) -> bool:
        """Whether a launch of `ba` runs at `config` on `target`: the target's family's bundle (`bundle`),
        tiles its program takes (`whole_tiles`), a call the launch takes (`refusal`), and the block and
        registers `Template.legal` holds it to. A pinned or stored config is held to it as a derived one."""
        query = ba.arguments["query"]
        family = self._named(target.generation, query.dtype, query.shape[1]).get(config.family)
        # A forward's and a dQ launch's layout streams through a producer (`Family.stream`); a layout whose
        # consumers feed their own rings (`split_queries`) runs the dK/dV launches alone, the keys-first one
        # and the fused one, each consumer its own keys over the stream they walk whole. The forward on it
        # has no use the producer's layout does not serve, and deadlocked (row 2599): it is no config.
        walks_whole = not isinstance(config, Tiles) and config.queries_first
        # One consumer feeding its own rings: no turns between consumers, whose dropped turns hung the
        # two-consumer fed forward; a decode step's 16-row tile on mma.sync's fed layout (an H100's,
        # `few_rows`), and every forward on a device whose bundle has no producer's layout (consumer
        # Blackwell's, an unseen NVIDIA GPU's: `bundles.of`, row 4727).
        alone = family is not None and isinstance(config, Tiles) and config.consumers == 1
        return (family is not None and (walks_whole or alone or not split_queries(family))
                and whole_tiles(family, config, parts=self.parts if isinstance(self, MosaicForward) else 1)
                and _band_held(self.score_kinds, config)
                and self.refusal(ba, family) is None and super().legal(ba, config, target))

    def _family(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> mosaic.Family | None:
        """The target's family's bundle where the launch takes the call there, else None."""
        query = ba.arguments["query"]
        family = self._bundle(target.generation, query.dtype)
        return None if family is None or self.refusal(ba, family) is not None else family

    WALKED: ClassVar[str] = "block_k"
    """The config's field naming the tile its consumers walk (a forward's keys, a backward's queries)."""

    def neighbours(self, ba: tokamax.BoundArguments, config: Config) -> Sequence[Config]:
        """The configs next to the derived `config` an opt-in tuning run times beside it (`variants`): the
        walked tile one unit size either side (`extents`), a stage fewer and more, a consumer fewer and
        more, each alone; `variants` keeps the legal ones."""
        query = ba.arguments["query"]
        family = bundle_of(config, query.dtype)
        walked = getattr(config, self.WALKED)
        reach = max(walked * 2, query.shape[1], _keys(ba))
        walks = (functools.partial(walked_extents, parts=self.parts)
                 if isinstance(self, MosaicForward) else extents)
        sizes = [size for size in walks(family, query.dtype, reach) if size in (walked // 2, walked * 2)]
        changes = [*({self.WALKED: size} for size in sizes),
                   {"stages": config.stages - 1}, {"stages": config.stages + 1},
                   {"consumers": config.consumers - 1}, {"consumers": config.consumers + 1}]
        return tuple(dataclasses.replace(config, **change) for change in changes
                     if all(value > 0 for value in change.values())
                     and change.get("stages", 2) >= 2
                     and change.get("consumers", 1) <= most_consumers(family, config))

    def remaining(self, ba: tokamax.BoundArguments) -> Config | None:
        # Its derived config, none where it derives none; a call it refuses refused (`refusal`).
        target = self.launch_target()
        family = self._bundle(target.generation, ba.arguments["query"].dtype)
        if family is not None and (reason := self.refusal(ba, family)):
            raise NotImplementedError(reason)
        return self.derived(ba, target)

    def refusal(self, ba: tokamax.BoundArguments, family: mosaic.Family) -> Refusal | None:
        """Why the launch takes no call of `ba` on `family`, or None (`refused`)."""
        raise NotImplementedError

    def tmem_columns(self, ba: tokamax.BoundArguments, config: Config, target: generation.GpuTarget) -> int:
        """The tensor-memory columns `config` allocates on `target` (0 where its family allocates none)."""
        del ba, config, target
        return 0

    def joined_roles(self, ba: tokamax.BoundArguments, config: Config,
                     target: generation.GpuTarget) -> tuple:
        """The raised warpgroup roles joined beside this launch's consumers (`MosaicForward` overrides)."""
        del ba, config, target
        return ()

    def _resident_resources(self, ba: tokamax.BoundArguments, config: Config,
                            target: generation.GpuTarget) -> tuple[int | None, int | None]:
        """Staged shared bytes and per-consumer registers for `config` on `target`."""
        staged = self.staged_bytes(ba, config, target) if target.shared_limit else None
        return staged, self.registers(ba, config, target=target)

    def resident(self, ba: tokamax.BoundArguments, config: Config, target: generation.GpuTarget) -> int:
        """The programs of its launch at `config` a multiprocessor holds at once: its family's residency at
        its registers, shared memory and rounded tensor memory (`resources.register_model`); one on a core
        whose family budgets no registers (a TPU's, `Family.budgets_registers`), and one at the least, a
        legal config's (row 9105)."""
        family = bundle_of(config, ba.arguments["query"].dtype, target=target)
        joined = self.joined_roles(ba, config, target)
        if not family.budgets_registers:
            return family.programs(config.consumers, joined=joined)
        staged, regs = self._resident_resources(ba, config, target)
        tmem = self.tmem_columns(ba, config, target)
        return max(1, resources.register_model(family, config.consumers, regs,
                                               shared=staged, target=target, tmem=tmem, joined=joined).most)

def _band_held(kinds: tuple[Kind, ...], config: Tiles | BackwardTiles) -> bool:
    """Whether every tile `config`'s launches read a band side of (`Kind.BAND`) lies within one of its
    origins' groups and wholly within its band or wholly past it: a program's query rows (its consumers')
    dividing `band.BAND_GROUP`, a tile's origin one scalar (`attention_mosaic_program._param`), and each key
    tile dividing `band.BAND_ALIGN` (`band.band_column`: a 192-key tile from key 192 straddles an origin of
    256, and read as past the band dropped keys 256 to 383's biases, head review), the forward's, the dQ
    launch's and the dK/dV launch's own (`BackwardTiles.dkv`)."""
    if Kind.BAND not in kinds:
        return True
    tiles = (config, config.dkv) if isinstance(config, BackwardTiles) else (config,)
    return all(band.BAND_GROUP % (tile.consumers * tile.block_q) == 0 and band.BAND_GROUP % tile.block_q == 0
               and band.BAND_ALIGN % tile.block_k == 0 for tile in tiles)


@dataclasses.dataclass(frozen=True)
class MosaicForward(_Mainloop[Tiles]):
    """The template's forward on this mainloop over the call's own arrays (`forward` gives them): queries
    `[B, Q, H, D]` over keys `[B, K, Hk, D]` and values `[B, K, Hk, Dv]`, each query's keys `start` to `end`
    `[B, Q]` and, with documents, its id `[B, Q]` against the keys' `[B, K]`; the score and output hooks'
    parameters (`score_params`, `output_params`, laid out by `score_kinds` and `output_kinds`) and the
    softmax's initial state (`Initial`, or None). Returns the output `[B, Q, H, Dv]` and, with `residuals`,
    the backward's residuals (`forward`'s). Its configs come from the family's facts and the call's extents
    (`derived`), legal within the device's block and a consumer's registers (`legal`)."""

    HOOKS = ("score_mod", "output_mod", "query_mod")
    config_cls = Tiles
    revision: int = 2
    output_mod: Callable | None = None
    output_kinds: tuple[Kind, ...] = ()
    output_dtypes: tuple[str, ...] = ()
    output_tiles: tuple[str, ...] | None = None
    """The dtypes of its output hook's tiles past its input where the hook is pointwise
    (`attention_mosaic_program.pointwise_tiles`), its epilogue finishing a chunk of columns at a time; None
    where it is not, or there is none. Read off the hook as its builder takes it (`_forward_call`): a keyed
    launch's hook is its fingerprint, which keeps no trace of it."""
    output_width: int = TRACED_TILE
    query_mod: Callable | None = None
    """The query prologue (`Hooks.query`), which each consumer takes on its query tile where it loads it,
    once a tile (`attention_mosaic_forward._query_of`); None where the call has none or its caller took it
    (`_forward_call`)."""
    query_kinds: tuple[Kind, ...] = ()
    query_dtypes: tuple[str, ...] = ()
    query_width: int = TRACED_TILE
    query_dtype: str = "float32"
    out_dtype: str | None = None
    """The inference walk's output dtype, the query's where None: fp32 for a split launch's slices, which a
    merge takes (`attention_decode`)."""
    masked: bool = False
    """Whether a spans call's rewalk (the fold's natural walk, the headroom guard) runs as a launch every time
    over its rows alone, no branch (`algebra.walked_launches`' `masked`: a decode step's, row 1978)."""
    splits: int = 1
    """The slices each query tile's keys are cut into, each a program of its own (`_Program.splits`): a
    split-KV inference launch's, its outputs each slice's (`attention_decode` merges them)."""
    parts: int = 1
    """The bf16 parts each of an fp32 call's operands enters as (`product.bf16_part_count`, `product.formed`),
    stacked along the batch axis, part `p`'s rows from `p·B`: every product its parts' (`product.terms`),
    each 16-bit (attention.md 4b). One for a 16-bit call."""

    rounds: float | None = None
    """A quantized model's weights' bound (`Hooks.rounds`), which the anchored training walk rounds them to
    at each row's maximum: a call that keeps its split residuals' alone. None where its weights stay whole."""
    scaled: bool = False
    """Whether its kernels walk scaled (`algebra.SCALED_WALK`): the op a call some row of which can be scaled
    launches in its common op's place (`_fwd`, `algebra.may_scale`), its own config derived from its own
    registers and shared memory (`scales_in_kernel`), which may be smaller than the common op's."""
    bounded: bool = False
    """Whether its caller has bounded the call's values below where any row can be scaled (a cache's call
    over the largest magnitudes its cache states, `attention_decode.attend`'s `bounds`), so its kernels walk
    unscaled, a `cached` call's too, reading no value to choose them."""
    in_key: bool = False
    """Whether the call's values are its keys' leading columns (an absorbed latent attention's: DeepSeek's
    key the latent 512 and the rotary 64, its value the latent 512; `dew.nn.attention.KeyColumns`): its
    programs stream the keys alone and take each step's value tile as a view of the key tile's columns, the
    slot held until the values' product is issued, the keys read once: its inference walk's (a training
    walk's anchored launches stream the value array the call gives, the keys' columns sliced)."""

    def _decides_and_registers(self, ba: tokamax.BoundArguments, config: Tiles,
                               target: generation.GpuTarget | None = None, *, peaks: bool = False,
                               columns: int | None = None, decides: bool | None = None,
                               priced: bool = True, masked: bool = False,
                               ) -> tuple[bool, ForwardRegisters, int | None]:
        """Resolve `decides`, `ForwardRegisters`, and (when `priced`) register count against `target`."""
        arguments = ba.arguments
        return self._decided_model(
            arguments["query"], arguments["value"], arguments["score_params"],
            arguments.get("residuals"), config, arguments.get("table"),
            arguments.get("sources"), arguments.get("scales"),
            arguments.get("initial"), arguments.get("drop"), arguments.get("start"),
            peaks=peaks, columns=columns, decides=decides, priced=priced, masked=masked, target=target)

    def deciding(self, ba: tokamax.BoundArguments, config: Tiles,
                 target: generation.GpuTarget | None = None) -> bool:
        """`decided` of the call of `ba` at `config` on `target`."""
        return self._decides_and_registers(ba, config, target, priced=False)[0]

    def _resident_resources(self, ba: tokamax.BoundArguments, config: Tiles,
                            target: generation.GpuTarget) -> tuple[int | None, int | None]:
        decides, _, regs = self._decides_and_registers(ba, config, target)
        staged = self.staged_bytes(ba, config, target, decides=decides) if target.shared_limit else None
        return staged, regs

    def _decided_model(self, query, value, score_params, residuals: Kept | None, config: Tiles,
                       table=None, sources=None, scales=None, initial=None, drop=None, start=None, *,
                       peaks: bool = False, columns: int | None = None, decides: bool | None = None,
                       priced: bool = True, masked: bool = False,
                       target: generation.GpuTarget | None = None,
                       ) -> tuple[bool, ForwardRegisters, int | None]:
        """Resolve whether `config` decides its scaled walk, its `ForwardRegisters`, and (when `priced`) its
        register count in one pass (rows 8843, 11987): builds the register facts once and reuses the scaled
        price so planning, residency and emission agree without double-pricing prologues."""
        resolved = self.launch_target() if target is None else target
        family = bundle_of(config, query.dtype, target=resolved)
        base = self.registered(query, value, score_params, residuals, table, sources, scales, initial, drop,
                               start)
        if decides is not None:
            model = dataclasses.replace(base, scaled=True) if decides else base
            count = model.at(family, config, peaks=peaks, columns=columns, masked=masked) if priced else None
            return decides, model, count
        computed = query.dtype if scales is not None else value.dtype
        # Only memory-bound walks (`streams`) and paged/quantized cache walks decide in-kernel; compute-bound
        # prefill/training forwards use `algebra.may_scale`'s single native-dtype max pass (~4-8 us) so the
        # hot forward inner loop pays no per-tile `keys_read` SMEM read, cross-warp barrier, `_bounded`
        # reduction, or scaled-walk register bloat (row 5272).
        if (not config.natural_form and residuals is not Kept.SPLIT and not self.scaled
                and self.rounds is None and algebra.headroom(computed)
                and family.budgets_registers and not family.places_on_role
                and (streams(query, resolved) or scales is not None or table is not None)):
            scaled_model = dataclasses.replace(base, scaled=True)
            scaled_price = scaled_model.at(family, config, masked=masked)
            if scaled_price is None or scaled_price <= self.register_budget(config, resolved):
                count = (None if not priced else scaled_price if not peaks and columns is None
                         else scaled_model.at(family, config, peaks=peaks, columns=columns, masked=masked))
                return True, scaled_model, count
        count = base.at(family, config, peaks=peaks, columns=columns, masked=masked) if priced else None
        return False, base, count

    def decided(self, query, value, score_params, residuals: Kept | None, config: Tiles, table=None,
                sources=None, scales=None, initial=None, drop=None, start=None, *,
                target: generation.GpuTarget | None = None) -> bool:
        """Whether its launch at `config` walks unscaled and decides a scaled second walk of a program whose
        rows pass fp32's largest (`after_walk.Decided`, `algebra.may_scale`, rows 5272, 5274): memory-bound
        inference launches (`streams`) and paged/quantized cache walks whose values can pass fp32's largest
        (`algebra.headroom`), on a register family whose consumers walk its rings, where the config holds the
        scaled walk's registers (`register_budget`). Compute-bound prefill/training forwards take `_fwd`'s
        1-kernel `algebra.may_scale` guard so the inner loop pays no per-tile magnitude reduction."""
        return self._decided_model(query, value, score_params, residuals, config, table, sources, scales,
                                   initial, drop, start, priced=False, target=target)[0]

    def _hooks(self, score_params, output_params, initial, drop, query_params=()) -> Hooks:
        return Hooks(score=Hook(self.score_mod, tuple(map(Side, score_params, self.score_kinds))),
                     initial=initial,
                     output=Hook(self.output_mod, tuple(map(Side, output_params, self.output_kinds))),
                     drop=drop, rounds=self.rounds,
                     query=Hook(self.query_mod, tuple(map(Side, query_params, self.query_kinds))))

    def _walk_setup(self, query, key, value, start, end, query_ids, key_ids, score_params, output_params,
                    initial, drop, table, sources, scales, query_params, *, residuals: Kept, interpret: bool,
                    config: Tiles, pricing: bool):
        """Shared walk setup for `_main_launch` and `_fwd` so pricing and execution pass identical statics
        to `tuning.lowered_once`."""
        hooks = self._hooks(score_params, output_params, initial, drop, query_params)
        spans = self.spans(start, end, query_ids, key_ids)
        target = self.launch_target()
        decides, registered, _ = self._decided_model(query, value, score_params, residuals, config, table,
                                                     sources, scales, initial, drop, start, priced=False,
                                                     target=target)
        family = bundle_of(config, query.dtype, target=target)
        products = None
        if family.places_on_role and residuals is not Kept.SPLIT and self.parts == 1:
            ba = self.bind(query, key, value, start, end, query_ids, key_ids, score_params, output_params,
                           initial, drop, table, sources, scales, query_params, residuals=residuals,
                           interpret=interpret)
            full_v = _width(value.shape[-1], family, query.dtype)
            widths = [width for _, width in column_slices(full_v, config.columns)]
            by_width = {w: self._products(ba, dataclasses.replace(config, columns=w), target)
                        for w in dict.fromkeys(widths)}
            products = tuple(by_width[w] for w in widths)
        form = {"scale": self.scale, "natural": self.natural, "tiles": config, "interpret": interpret,
                "counts": True if pricing else not self.scaled, "registered": registered,
                "tmem_columns": target.tmem_columns}
        scaled = (self.scaled or (self.cached(table, sources, scales) and not self.bounded)) and not decides
        return hooks, spans, decides, family, products, form, scaled

    def _walk_dispatch(self, query, key, value, spans, hooks, *, residuals: Kept, family, products, form,
                       scaled: bool, decides: bool, table, sources, scales):
        """Single residual-dispatch owner returning `(spec, launch, finished, in_place)` for both
        `_main_launch` and `_fwd` (row 10677)."""
        match residuals:
            case Kept.SPLIT:
                if scales is not None:
                    raise NotImplementedError("a quantized cache's keys are read by inference alone")
                spec, launch, finished = _training_walk(query, key, value, spans, hooks,
                                                        scaled=scaled, **form)
                return spec, launch, finished, False
            case Kept.NONE | Kept.OUTPUT:
                return _plain_walk(query, key, value, spans, hooks, rows=residuals is Kept.OUTPUT,
                                   out_dtype=self.out_dtype, masked=self.masked, splits=self.splits,
                                   table=table, sources=sources, subtracted=self.subtracted, scales=scales,
                                   parts=self.parts, scaled=scaled, decides=decides,
                                   in_key=self.reads_key_slot(family), products=products, **form)

    def _main_launch(self, query, key, value, start, end, query_ids, key_ids, score_params, output_params,
                     initial, drop=None, table=None, sources=None, scales=None, query_params=(), *,
                     residuals: Kept, interpret: bool, config: Tiles, pricing: bool = False):
        """The main forward launch at `config` without its guarded rewalk (`_walk_dispatch`): traced by
        `price_config` through `tuning.lowered_once` so the winning count's trace is held and reused."""
        hooks, spans, decides, family, products, form, scaled = self._walk_setup(
            query, key, value, start, end, query_ids, key_ids, score_params, output_params, initial, drop,
            table, sources, scales, query_params, residuals=residuals, interpret=interpret, config=config,
            pricing=pricing)
        spec, launch, _, _ = self._walk_dispatch(query, key, value, spans, hooks, residuals=residuals,
                                                 family=family, products=products, form=form, scaled=scaled,
                                                 decides=decides, table=table, sources=sources, scales=scales)
        return launch(spec, None)

    def _fwd(self, query, key, value, start, end, query_ids, key_ids, score_params, output_params, initial,
             drop=None, table=None, sources=None, scales=None, query_params=(), *, residuals: Kept,
             interpret: bool, config: Tiles, return_residuals: bool = False):
        hooks, spans, decides, family, products, form, scaled = self._walk_setup(
            query, key, value, start, end, query_ids, key_ids, score_params, output_params, initial, drop,
            table, sources, scales, query_params, residuals=residuals, interpret=interpret, config=config,
            pricing=False)
        if reason := refused(query.dtype, query.shape[3], value.shape[3], hooks, backward=False,
                             family=family):
            raise NotImplementedError(reason)

        def launched():
            spec, launch, finished, in_place = self._walk_dispatch(
                query, key, value, spans, hooks, residuals=residuals, family=family, products=products,
                form=form, scaled=scaled, decides=decides, table=table, sources=sources, scales=scales)
            return finished(algebra.walked_launches(spec, launch, in_place=in_place))

        if scaled or decides or self.bounded or hooks.rounds is not None or not algebra.headroom(value.dtype):
            return launched(), None
        # A call some row of which can be scaled (`algebra.may_scale`) launches the scaled op's kernels (its
        # own config, `scaled`), every other call, every ordinary input's, this op's unscaled ones, bit for
        # bit the scaled walk's there.
        may = algebra.may_scale(product.joined(value, self.parts), keys_axis=1,
                                initial=None if initial is None else initial.acc,
                                amplified=1.0 if drop is None else drop.reach)
        arguments = (query, key, value, start, end, query_ids, key_ids, score_params, output_params, initial,
                     drop, table, sources, scales, query_params)

        def rare():
            # A pin of this op's where the scaled op takes it (`pinnable`, which checks `proposes` for scaled
            # ops, row 11992), else its own derived config: its registers and shared memory are not the
            # common op's.
            scaled_op = dataclasses.replace(self, scaled=True)
            scaled_ba = scaled_op.bind(*arguments, residuals=residuals, interpret=interpret)
            with tuning.pinned_where_taken(scaled_op, scaled_ba) as taken:
                return taken(*arguments, residuals=residuals, interpret=interpret)
        return jax.lax.cond(may, rare, launched), None

    def pinnable(self, ba: tokamax.BoundArguments, config: Hashable) -> bool:
        """A scaled forward's pin must also fit its register model (`proposes`) so `pinned_where_taken` sets
        aside an enclosing common-op pin whose tile exceeds scaled registers (row 11992)."""
        if not super().pinnable(ba, config):
            return False
        assert self._typed(config)
        return not self.scaled or self.proposes(ba, config, self.launch_target())

    def completed(self, ba: tokamax.BoundArguments, config: Hashable) -> Hashable:
        """A pin's or an explicit config's natural rewalk derived once where it names none (`_relaunched`,
        rows 2621, 2647); the config as it is where it names any of its rewalk's tiles (its key tile, its
        consumers, its value columns: row 2675) or none derives (a config of another type, which `pinnable`
        refuses)."""
        if (not isinstance(config, Tiles) or config.natural_form
                or (config.relaunch_k, config.relaunch_consumers, config.relaunch_columns) != (None,) * 3):
            return config
        return self._relaunched(ba, config, generation.gpu_target()) or config

    def legal(self, ba: tokamax.BoundArguments, config: Tiles, target: generation.GpuTarget) -> bool:
        """Its launch's (`_launch_legal`), and a fold's natural rewalk at the tiles the config names (the
        derivation's, a tuning neighbour's, a pin's completed one, `completed`), their key tile dividing the
        config's and their consumers at most its, holding too (row 2584); a fold's config that names none
        rewalks at its own tiles, which must hold (rows 2621, 2647). A rewalk on another family
        (`Tiles.relaunch_family`: the register family's, where the config's takes no rewalk) is that family's
        derivation's own tiles (`_relaunched`), legal by its launch alone."""
        if not self._launch_legal(ba, config, target):
            return False
        if config.natural_form or self._unrewalked(ba):
            return True
        decides = self.deciding(ba, config, target=target)
        relaunch = config.relaunch
        if config.relaunch_family is not None:
            return self._launch_legal(ba, relaunch, target, decides=decides)
        return (config.block_k % relaunch.block_k == 0 and relaunch.consumers <= config.consumers
                and self._launch_legal(ba, relaunch, target, decides=decides))

    def _launch_legal(self, ba: tokamax.BoundArguments, config: Tiles, target: generation.GpuTarget, *,
                      decides: bool | None = None) -> bool:
        """`_Mainloop.legal`'s for one launch at `config`."""
        query = ba.arguments["query"]
        if self._named(target.generation, query.dtype, query.shape[1]).get(config.family) is None:
            return False
        family = bundle_of(config, query.dtype, target=target)
        # A family that places its products on a role issues its logits ahead (`_ahead`: tcgen05's rows,
        # whose issuer warp is a step ahead of its consumer), which past its tensor memory it does not: its
        # products' scratch then none it takes (`Issuer.product_scratch`). Checked before `super().legal`
        # because `staged_bytes` calls `_products` -> `product_scratch`, which requires `ahead=True`.
        if family.places_on_role and not self._ahead(ba, config, target):
            return False
        # A launch that reads its values from the key's slot (`reads_key_slot`) reads that slot within the
        # step that waits on it, which a walk issuing its logits a step ahead (`_ahead`: tcgen05's chunks)
        # already read the step before (`rings.Slots`' `following`): its program raises as it traces (a B200's
        # in-key forward priced tcgen05.chunks at 64 x 64 on two consumers as a rewalk's candidate and raised,
        # where its derivation offered it none to price before row 6216).
        if self.reads_key_slot(family) and self._ahead(ba, config, target):
            return False
        if not super().legal(ba, config, target):
            return False
        if decides and self.staged_bytes(ba, config, target, decides=True) > self.program_shared(config,
                                                                                                 target):
            return False
        codes = _codes_of(ba.arguments["key"], ba.arguments.get("scales"))
        if codes is not None and not mosaic.converts(codes):
            return False
        # A tile of fewer rows than a role's lays its warps along its keys, a whole 16-key product's columns
        # each (Mosaic's layouts take whole such tiles a warp: a 96-key tile's 24 refused, dd-q5p4; 64-key
        # tiles lower at widths 64 to 256 on sm80 and sm90, dd-k16b).
        if config.block_q < family.role_rows and config.block_k % (WARPGROUP_WARPS * 16):
            return False
        # Its products' accumulators within the multiprocessor's tensor memory, where its issuer keeps them
        # there (`Issuer.product_columns`: tcgen05's chunks, each consumer's sums across its walk beside its
        # logits).
        if family.product_columns(self._products(ba, config, target)) > (target.tmem_columns or 0):
            return False
        # Its products' right operands, the keys' tile in S = Q·Kᵀ (the heads' width deep, a key tile's
        # columns) and the values' in P·V (a key tile's rows of the launch's value columns, `Tiles.columns`),
        # ones its family's products take (`Issuer.takes_right`: tcgen05's 64-row products pad a narrow one
        # only at their own depth). Its natural rewalk's are judged as its launch's (`legal`).
        width = _width(query.shape[-1], family, query.dtype)
        val_columns = value_columns(_width(ba.arguments["value"].shape[-1], family, query.dtype),
                                    config.columns)
        if not (family.takes_right(width, config.block_k)
                and family.takes_right(config.block_k, val_columns)):
            return False
        # wgmma's P·V depth (`block_k`) must be a multiple of V's swizzle tile (`swizzle_for(val_columns)`):
        # 64 for >=64-column V blocks, 32 for 32-column V blocks (row 6508).
        staged = jnp.bfloat16 if self.parts > 1 else query.dtype
        if (self.parts > 1 and not family.operands_in_registers
                and walked_unit(family, query.dtype, self.parts) < family.walked_rows
                and config.block_k % swizzling.swizzle_columns(
                    staged, swizzling.swizzle_for(staged, val_columns))):
            return False
        # A paged call's key tiles (the launch's, its natural rewalk's) whole in its pages or its pages whole
        # in them (`_pages_hold`).
        tiles = (config.block_k, config.relaunch_k or config.block_k)
        page = _page(ba)
        return page is None or all(_pages_hold(page, tile, family) for tile in tiles)

    def derived(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> Tiles | None:
        """Its tiles from the facts (`_derived`): each consumer a unit of queries, the key tile the largest
        its registers and the block hold, two stages; the family's consumers taking turns, a step's softmax
        running under the other's products, the training forward's too (its anchored walk's step is the
        plain one's, TWOPASS4-h100: 1.05 ms at 64 x 64 on two consumers against 1.41 at 64 x 128 on one,
        causal llama-d128 2 x 4096 on an H100 PCIe). Its family the first of the device's bundles, ranked
        by the share of their rows its queries fill (`layouts`: a decode step's few rows mma.sync's before
        wgmma's, row 4042), that takes the call and derives a config: a config legal with its natural
        rewalk's tiles (`_relaunched`), so a tile whose rewalk no tile holds gives way to a smaller (a 48-row
        training walk's scaled op on an A100 at width 32, legal at 48 x 64 and its rewalk not, derived none
        where 32 x 64 holds both, et-why48). A config is legal as it launches, its rewalk's tiles named
        (`_launchable`), so a compile's refusal of it (`report.refused`, keyed by the config the launch
        traced) steps its derivation: read at its own tiles alone, the refusal reached nothing and a B200's
        spilling 4B forward compiled twice, the same program (PF03). A family that places its products on a
        role first (`layouts`: tcgen05's rows, FlashAttention 4's), at the one tile its consumer and issuer
        are written for (its 128 rows a thread each, 64 keys a step: `_derived` gives it one consumer)."""
        return self._tiles(ba, target)

    def _has_config(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> bool:
        """Feasibility check without pricing candidates or deepening stages: `declined` and `uncertified`
        check whether any legal launch exists without tracing kernels through `price_config`."""
        return self._tiles(ba, target, priced=False) is not None

    def _tiles(self, ba: tokamax.BoundArguments, target: generation.GpuTarget, *,
               skip: mosaic.Family | None = None, refused: Set[Tiles] = frozenset(),
               priced: bool = True) -> Tiles | None:
        """`derived` over the device's families but `skip` (a family whose natural rewalk runs on another,
        `_relaunched`), past the configs as they launch in `refused` (a rewalk's tiles a compile refused).
        Prefer whole configs and column splits that relieve SMEM/TMEM/residency (`_columns_help`) before
        falling back to deferred same-residency splits so legal pins remain viable when whole fails
        (rows 5535, 12210). With `priced=False`, returns the first legal config without tracing
        `price_config` or deepening."""
        query = ba.arguments["query"]
        families = [family for family in layouts(target.generation, query.dtype, query.shape[1])
                    if family is not skip and self.refusal(ba, family) is None]
        # Each offer's launch completed once (`_launchable`: its legality and rewalk's derivation), for its
        # fit, its price and the winner alike.
        launchable = functools.cache(lambda config: self._launchable(ba, config, target, priced=priced))
        choose = (functools.partial(self._choose_consumers, ba, target=target, launch=launchable) if priced
                  else None)
        for prefer_helping in (True, False):
            for family in families:
                def fits(config: Tiles, *, helping=prefer_helping) -> bool:
                    return ((not helping or self._columns_help(ba, config, target))
                            and launchable(config) not in (None, *refused))

                config = self._searched(ba, family, fits, choose=choose)
                if config is not None:
                    launched = launchable(config)
                    if launched is None or not priced:
                        return launched
                    return self._deepened(ba, launched, target, refused=refused)
        return None

    def _deepened(self, ba: tokamax.BoundArguments, config: Tiles, target: generation.GpuTarget, *,
                  refused: Set[Tiles] = frozenset(), want: int | None = None) -> Tiles:
        """Deepen a forward ring at its residency via `resources.plan`: compute-bound TMA and cp.async
        forwards up to `max(consumers, 2) + 1`, and streaming decode walks via `streaming` up to `want`
        (rows 5272, 5274, 4839, 10948, 10962)."""
        query = ba.arguments["query"]
        family = bundle_of(config, query.dtype, target=target)
        if not family.budgets_registers or (want is None and streams(query, target)):
            return config
        cap = min(max(config.consumers, 2) + 1, -(-_keys(ba) // config.block_k)) if want is None else want
        if cap <= config.stages:
            return config
        base_resident = self.resident(ba, config, target)
        products = self._products(ba, config, target)
        decides, _, regs = self._decides_and_registers(ba, config, target)
        planned = _planned_stages(
            family, lambda s: self._scope(ba, dataclasses.replace(config, stages=s), target,
                                          products=products, decides=decides),
            target, consumers=config.consumers, registers=regs,
            floor=config.stages, most=cap, resident=base_resident)
        for stages in range(planned, config.stages, -1):
            candidate = self._launchable(ba, dataclasses.replace(config, stages=stages), target)
            if (candidate is not None and candidate not in refused
                    and self.resident(ba, candidate, target) == base_resident):
                return candidate
        return config
    def price_config(self, ba: tokamax.BoundArguments, config: Tiles, target: generation.GpuTarget, *,
                     issued: bool = False) -> float | None:
        """Price `config`'s main launch on `target` (`cost.launch_seconds` over `emission.work_of`), traced
        through `tuning.lowered_once` under `report.launching`; with `issued`, its programs' warps hiding each
        other's issue intervals too (`emission.warps_of`, `Multiprocessor.issue_interval`: a decode walk's,
        whose one consumer a multiprocessor's residency decides, M-NA-21 and FS-5274-LAT1), which a
        prefill's derivation does not yet read (its register model's misses at three consumers,
        FS-LANE-FWD-TIP). As its hookless call (its hooks, their
        parameters and their kinds none), so a launch keyed by its hooks' fingerprints (`KernelOp.keyed`,
        which traces none) prices, and chooses, as its live call does. A heuristic proxy of the live call's
        price, not that price less a common cost: a score hook's work scales with each candidate's entries
        alike, but its absence can change the surrogate's own schedule (an output hook's rewalk in place, a
        transcendental hook's approximate math). None with `issued` where `target` states no issue
        interval: its walk's latency unpriced, its price its bytes' alone, level over every slice count (a
        G4's decode planned 2 slices, 16 of its 188 multiprocessors, 281 of a step's 288 us,
        M-NA-35-decodec-g4)."""
        multiprocessor = target.architecture.multiprocessor
        if issued and (multiprocessor is None or multiprocessor.issue_interval is None):
            return None
        hooks = ("score_params", "output_params", "query_params")
        names = ("query", "key", "value", "start", "end", "query_ids", "key_ids", "score_params",
                 "output_params", "initial", "drop", "table", "sources", "scales", "query_params")
        operands = tuple(() if name in hooks else tuning.shapes(ba.arguments[name]) for name in names)
        residuals = ba.arguments["residuals"]
        identity = report._identity(self, ba, target)
        hookless = dataclasses.replace(
            self, **dict.fromkeys(self.HOOKS), score_kinds=(), score_dtypes=(), score_reduces=False,
            output_kinds=(), output_dtypes=(), query_kinds=(), query_dtypes=())

        def priced(*args):
            return hookless._main_launch(*args, residuals=residuals, interpret=False, config=config,
                                         pricing=True)

        priced.__name__ = priced.__qualname__ = (
            f"priced_{type(self).__name__}_l{report._digest(identity)}_c{report._digest(repr(config))}")
        with generation.assumed(target), report.launching(self, ba, config):
            traced = jax.make_jaxpr(priced)(*operands)
        # Its launches as their emission counted them (`emission.launched_kernels`, the one discovery): None
        # where one is uncounted or under control flow, its price unknown.
        if not (found := emission.launched_kernels(traced.jaxpr)):
            return None
        resident = self.resident(ba, config, target)
        return sum(cost.launch_seconds(launched.work, emission.launch_programs(kernel), target,
                                       resident=resident, warps=launched.warps if issued else 0.0)
                   for kernel, launched in found)

    def _choose_consumers(self, ba: tokamax.BoundArguments, candidates: list[Tiles], *,
                          target: generation.GpuTarget, launch: Callable[[Tiles], Tiles | None] | None = None
                          ) -> Tiles:
        """Pick the fastest of `candidates`, each a consumer count at each walked tile that fits it
        (`_derived`), priced by `price_config` as it launches (`launch`, `_launchable` where None), storing
        the choice in the tuning store beside verdicts (`tuning.stored_choice` / `tuning.store_choice`) so
        warm starts trace nothing to price (rows 6710, 6216)."""
        if len(candidates) <= 1 or not target.multiprocessors or not cost.rates(target, device=False).tensor:
            return candidates[0]
        launch = launch or (lambda config: self._launchable(ba, config, target))
        launched = [launch(config) or config for config in candidates]
        facts = _priced(target)
        key = _choice_key(self, ba, tuple(launched), facts)
        stored = None if key is None else tuning.stored_choice(target.generation, key)
        if isinstance(stored, (tuple, list)):
            key_tuple = tuple(stored)
            matched = next((c for c in candidates if (c.block_q, c.consumers, c.block_k) == key_tuple), None)
            if matched is not None:
                return matched
        prices: dict[Tiles, float] = {}
        for config, full in zip(candidates, launched, strict=True):
            price = self.price_config(ba, full, facts)
            if price is None:
                return candidates[0]
            prices[config] = price
        best = min(candidates, key=prices.__getitem__)
        if key is not None:
            tuning.store_choice(target.generation, key, (best.block_q, best.consumers, best.block_k))
        return best

    def _columns_help(self, ba: tokamax.BoundArguments, config: Tiles,
                      target: generation.GpuTarget) -> bool:
        """Whether splitting value columns (`config.columns is not None`) relieves shared memory, tensor
        memory, or residency on a `places_on_role` family (`tcgen05.rows`): its consumer registers hold no
        value columns (sums live in tensor memory), so when `columns=None` already fits SMEM and TMEM at the
        same residency, splitting columns is deferred behind configs that help (rows 5535, 12210)."""
        if config.columns is None:
            return True
        family = bundle_of(config, ba.arguments["query"].dtype, target=target)
        if not family.places_on_role:
            return True
        whole = dataclasses.replace(config, columns=None)
        if not self._ahead(ba, whole, target):
            return True
        if family.product_columns(self._products(ba, whole, target)) > (target.tmem_columns or 0):
            return True
        if self.staged_bytes(ba, whole, target) > self.program_shared(whole, target):
            return True
        return self.resident(ba, config, target) > self.resident(ba, whole, target)

    def offered(self, ba: tokamax.BoundArguments, family: mosaic.Family, target: generation.GpuTarget
                ) -> tuple[Tiles, ...]:
        """Every config of `family` its derivation searches for the call of `ba` (`_searched`) that is legal,
        as it launches (`_launchable`), preferred configs (`_columns_help`) before deferred same-residency
        splits so all legal pins remain visible (row 12210); none where the family takes no such call
        (`refusal`)."""
        if self.refusal(ba, family) is not None:
            return ()
        preferred: dict[Tiles, None] = {}
        deferred: dict[Tiles, None] = {}

        def fits(config: Tiles) -> bool:
            if (launched := self._launchable(ba, config, target)) is not None:
                bucket = preferred if self._columns_help(ba, config, target) else deferred
                bucket[launched] = None
            return False

        self._searched(ba, family, fits)
        return tuple({**preferred, **deferred})

    def _searched(self, ba: tokamax.BoundArguments, family: mosaic.Family, fits: Callable[[Tiles], bool], *,
                  choose: Callable[[list[Tiles]], Tiles] | None = None) -> Tiles | None:
        """The derivation's search of `family`'s configs for the call of `ba` (`_derived`, over the value
        columns whole and then in blocks, `_column_blocks_of`): the first `fits` takes, or None."""
        query = ba.arguments["query"]
        for columns in _column_blocks_of(family, query.dtype, ba.arguments["value"].shape[3]):
            def tiles(rows: int, walked: int, consumers: int, columns=columns) -> Tiles:
                return Tiles(rows, walked, 2, consumers, family=family.name, columns=columns)

            config = _derived(family, query.dtype, query.shape[1], _keys(ba), tiles, fits, overlapped=True,
                              walks=functools.partial(walked_extents, parts=self.parts),
                              least_walked=functools.partial(_least_keys, family), choose=choose)
            if config is not None:
                return config
        return None

    def _launchable(self, ba: tokamax.BoundArguments, config: Tiles, target: generation.GpuTarget, *,
                    priced: bool = True) -> Tiles | None:
        """`config` as it launches, its natural rewalk's tiles named (`_relaunched`), where that is legal
        (`legal`: a compile's refusal of the launch among its rules, keyed by this config); None where it is
        not or no tile holds the rewalk."""
        if not (self._launch_legal(ba, config, target) and self.proposes(ba, config, target)):
            return None
        launched = self._relaunched(ba, config, target, priced=priced)
        return launched if launched is not None and self.takes(ba, launched, target) else None

    def streaming(self, ba: tokamax.BoundArguments, config: Tiles, target: generation.GpuTarget) -> Tiles:
        """`config` for a call whose walk the device's memory bounds (`streams`), its key tile the least its
        ring still streams at: a program pays its ring's first `stages` tiles before its first product, and
        a launch of more programs than the device runs at once pays it a program, so the least tile pays
        least, where its resident programs' rings in flight (`_staged_keys`) hold the bytes a
        multiprocessor's share of the device's stream keeps in flight (`GpuTarget.memory_rate` over its
        multiprocessors, for the part's measured `Measured.fixed`):
        a decode step of 128 rows over 4096 keys took 0.70 ms at 64-key tiles against 0.73 at 128 on an
        H100, and of 32 over 32768 1.35 against 1.38, its 128-wide K and V 32 KB a stage, two of them past
        the 45 KB a program keeps in flight (ET-STAGES2-h100).
        Its natural rewalk's tiles derived again (`_relaunched`); where no tile dividing the call's keys keeps
        the stream in flight, the largest that divides them, else `config` itself, as where the part's
        stream is unmeasured."""
        in_flight = self._in_flight(ba, config, target)
        if in_flight is None:
            return config
        # Its key tiles dividing the call's keys alone: a tile past them pads the whole cache, its keys and
        # values copied a step (`padded_call`; a 96-key tile over 32768, 383 us a step on an A100,
        # M-DD-5 W16, as long as its launch). A multiprocessor's bytes in flight are its resident programs'
        # together (`resident`): a larger tile holding fewer of them keeps no more in flight (an A100's
        # 16-row d64 decode at 256-key tiles staged 146 KB, one program a multiprocessor, 1.17x the hand
        # kernel's five resident programs of 20 KB, M-DD-7 W34).
        dividing = self._dividing(ba, config, target)
        picked = config
        for candidate in dividing:
            held = self.resident(ba, candidate, target)
            if held * self._staged_keys(ba, candidate, target) >= in_flight:
                picked = candidate
                break
        else:
            # Where none keeps the stream in flight, the largest dividing tile before a derived one that does
            # not divide the keys: a GH200's bounded decode at 192-key tiles over 32768 padded its cache,
            # 84.5 of its 134 us a step at width 128 (M-ET-70), where its 128-key walk alone is 49.5.
            if self.keys % config.block_k and dividing:
                picked = dividing[-1]
        return self._streamed_ring(ba, picked, target)

    def streamings(self, ba: tokamax.BoundArguments, config: Tiles, target: generation.GpuTarget
                   ) -> tuple[Tiles, ...]:
        """Each key tile a memory-bound walk of `config` may stream at (`streaming`'s offers: the launchable
        tiles dividing the call's keys, least first, else `config`), its ring deepened as `streaming`
        deepens it: a priced choice's candidates (`attention_decode._priced_slices`)."""
        return tuple(self._streamed_ring(ba, tile, target)
                     for tile in (self._dividing(ba, config, target) or [config]))

    def _dividing(self, ba: tokamax.BoundArguments, config: Tiles, target: generation.GpuTarget
                  ) -> list[Tiles]:
        """`config`'s launchable key tiles dividing the call's keys (`_launchable`), least first."""
        family = bundle_of(config, ba.arguments["query"].dtype, target=target)
        return [launched for launched in (
            self._launchable(ba, dataclasses.replace(config, block_k=extent), target)
            for extent in reversed(walked_extents(family, ba.arguments["query"].dtype, config.block_k,
                                                  parts=self.parts))
            if self.keys % extent == 0)
                if launched is not None]

    def _streamed_ring(self, ba: tokamax.BoundArguments, tile: Tiles, target: generation.GpuTarget) -> Tiles:
        """`tile`'s ring as a streaming walk deepens it: up to three stages, half a slice's key tiles."""
        slice_tiles = math.ceil(math.ceil(_keys(ba) / self.splits) / tile.block_k)
        return self._deepened(ba, tile, target, want=min(3, slice_tiles // 2))

    def streamed(self, ba: tokamax.BoundArguments, config: Tiles, target: generation.GpuTarget) -> int:
        """The programs a multiprocessor runs at once to keep its share of the device's stream in flight at
        `config` (`streaming`'s bytes in flight over a program's staged keys and values), at most those it
        holds (`resident`); those it holds where the stream is unmeasured. A memory-bound launch of more
        programs a multiprocessor streams no faster: a split decode's slices past them only lengthen its
        merge (an H100's B1 x 32768 decode at width 64 cut into 82 slices, five programs a multiprocessor
        where one keeps its stream, took 90.7 us against the hand kernel's 28.7, M-DD-4 W56)."""
        resident = self.resident(ba, config, target)
        in_flight = self._in_flight(ba, config, target)
        if in_flight is None:
            return resident
        return max(1, min(resident, math.ceil(in_flight / self._staged_keys(ba, config, target))))

    def _in_flight(self, ba: tokamax.BoundArguments, config: Tiles, target: generation.GpuTarget
                   ) -> float | None:
        """The bytes a multiprocessor keeps in flight to stream its share of the device's memory rate for a
        call its memory bounds (`streams`: `GpuTarget.memory_rate` over its multiprocessors, for the part's
        measured `Measured.fixed`); None where the call is not so bound or the part's stream is
        unmeasured."""
        rate, fixed = target.memory_rate, target.fixed_seconds
        if not (streams(ba.arguments["query"], target) and rate and fixed and target.multiprocessors):
            return None
        return rate / target.multiprocessors * fixed

    def _staged_keys(self, ba: tokamax.BoundArguments, config: Tiles,
                     target: generation.GpuTarget | None = None) -> int:
        """The bytes of keys and values a program's ring keeps in flight at `config`: its key tiles' rows of
        both, at the widths its family stages them (`_width`), in every slot but the one its walk reads
        while the others load (`stages - 1`): counted at all `stages`, an H100's decode step at 8 rows over
        32768 keys, width 64, was cut into slices of two programs a multiprocessor and streamed 69% of the
        device's memory rate where cuDNN's streamed 89% (232 against 181 us, M-ET-49b-h100)."""
        query, value = ba.arguments["query"], ba.arguments["value"]
        family = bundle_of(config, query.dtype, target=target)
        itemsize = jnp.dtype(query.dtype).itemsize
        per_key = sum(_width(array.shape[3], family, query.dtype) for array in (query, value)) * itemsize
        return max(1, config.stages - 1) * config.block_k * per_key

    def _relaunched(self, ba: tokamax.BoundArguments, config: Tiles, target: generation.GpuTarget, *,
                    priced: bool = True) -> Tiles | None:
        """`config` with its natural rewalk's tiles, a fold's (`Tiles.relaunch`): the largest key tile and the
        most consumers that hold its own registers, each dividing the config's (et-fwdhooks3: at the fold's
        tiles it spilled 2 KB at width 64); a call that keeps its logits natural rewalks nothing. None where
        no tile holds the rewalk."""
        config = dataclasses.replace(config, relaunch_k=None, relaunch_consumers=None, relaunch_columns=None,
                                     relaunch_family=None, relaunch_q=None)
        if self._unrewalked(ba):
            return config
        query, value = ba.arguments["query"], ba.arguments["value"]
        family = bundle_of(config, query.dtype, target=target)
        if not family.rewalks:
            return self._rewalked_apart(ba, config, family, target, priced=priced)
        decides = self.deciding(ba, config, target=target)
        # Its value columns whole first, then in blocks (`_column_blocks_of`): a launch a block (row 2526: at
        # width 256 no tile held the anchored rewalk's whole sums).
        col_blocks = _column_blocks_of(family, query.dtype, value.shape[-1])
        if config.columns is not None:
            col_blocks = [c for c in col_blocks if c is not None and c <= config.columns]
        relaunches = (dataclasses.replace(config, block_k=block, consumers=consumers, natural_form=True,
                                          columns=columns)
                      for columns in col_blocks
                      for consumers in range(config.consumers, 0, -1)
                      for block in walked_extents(family, query.dtype, config.block_k, parts=self.parts)
                      if config.block_k % block == 0)
        in_place = rewalks_in_place(family, query, value, masked=self.masked,
                                    has_output_mod=self.output_mod is not None,
                                    in_key=self.reads_key_slot(family), target=target)
        for tiles in relaunches:
            if not (self._launch_legal(ba, tiles, target, decides=decides)
                    and self.proposes(ba, tiles, target,
                                      held=lambda tiles=tiles: self.within(
                                          self.registers(ba, tiles, decides=decides, masked=in_place),
                                          tiles))):
                continue
            launched = dataclasses.replace(config, relaunch_k=tiles.block_k,
                                           relaunch_consumers=tiles.consumers,
                                           relaunch_columns=tiles.columns)
            if not any(NATURAL_NAME in kernel for kernel in report.refused_at(self, ba, launched, target)):
                return launched
        return None

    def _rewalked_apart(self, ba: tokamax.BoundArguments, config: Tiles, family: mosaic.Family,
                        target: generation.GpuTarget, *, priced: bool = True) -> Tiles | None:
        """`config` of a `family` whose program takes no masked rewalk (`Issuer.rewalks`: tcgen05's rows,
        whose consumer gives its issuer a plain walk's weights alone) with its rewalk on the device's register
        family, at the tiles that family's derivation gives the call (`_tiles`), past each a compile refused
        where the rewalk's kernel spilled (`report.refused_at`, `NATURAL_NAME`): the rewalk steps as that
        derivation steps under the refusal of its own config, the config's own kernel's refusal its
        derivation's (`legal`). Refused as one config, a B200's 4B rows forward stepped to a 16-column block
        of the rows where its chunks rewalk had spilled 112 bytes at 64 x 64 on two consumers
        (et-rows-spill). None where no tile holds the rewalk."""
        refused: set[Tiles] = set()
        while (fallback := self._tiles(ba, target, skip=family, refused=refused, priced=priced)) is not None:
            relaunch = fallback.relaunch if fallback.relaunch_k is not None else dataclasses.replace(
                fallback, natural_form=True)
            if not (self._launch_legal(ba, relaunch, target) and self.proposes(ba, relaunch, target)):
                return None
            launched = dataclasses.replace(config, relaunch_family=relaunch.family,
                                           relaunch_q=relaunch.block_q, relaunch_k=relaunch.block_k,
                                           relaunch_consumers=relaunch.consumers,
                                           relaunch_columns=relaunch.columns)
            if not any(NATURAL_NAME in kernel for kernel in report.refused_at(self, ba, launched, target)):
                return launched
            refused.add(fallback)
        return None

    def neighbours(self, ba: tokamax.BoundArguments, config: Tiles) -> Sequence[Tiles]:
        """`_Mainloop.neighbours`', each with its own natural rewalk's tiles (`_relaunched`), not the
        derived config's, which need not divide it (row 2584)."""
        target = generation.gpu_target()
        found = (self._relaunched(ba, neighbour, target) for neighbour in super().neighbours(ba, config))
        return tuple(neighbour for neighbour in found if neighbour is not None)

    def _unrewalked(self, ba: tokamax.BoundArguments) -> bool:
        """Whether the call walks no natural rewalk: its logits natural (`not spec.base2`), or a fold no
        logit's reach passes (`not spec.reaches`), at the scale its launch takes (`_folded`, row 11818)."""
        spec, _ = self._folded(ba.arguments["query"], ba.arguments.get("residuals"))
        return not spec.base2 or not spec.reaches

    def _folded(self, query, residuals) -> tuple[algebra.Forward, float]:
        """The call's folded form and folded power (`folded_form`, rows 11789, 11818), shared by `_unrewalked`
        and `registered` with the same `subtracted` condition as `_launch`."""
        subtracted = self.subtracted and residuals is not Kept.SPLIT
        return folded_form(self.form, self.keys, ids=False, dtype=query.dtype, width=query.shape[-1],
                           scale=self.scale, natural=self.natural, score_mod=self.score_mod,
                           subtracted=subtracted)

    def staged_bytes(self, ba: tokamax.BoundArguments, config: Tiles, target: generation.GpuTarget, *,
                     decides: bool | None = None) -> int:
        """The program's shared memory (`_forward_allocations`, `resources.shared_bytes`), at the call's
        widths as the launch stages them (`_width`)."""
        return self._staged_bytes(ba, config, target, products=self._products(ba, config, target),
                                  decides=decides)

    def _scope(self, ba: tokamax.BoundArguments, config: Tiles, target: generation.GpuTarget, *,
               products: tuple, decides: bool | None = None) -> resources.Scope:
        family = bundle_of(config, ba.arguments["query"].dtype, target=target)
        query, value = ba.arguments["query"], ba.arguments["value"]
        width, full_v = (_width(array.shape[-1], family, query.dtype) for array in (query, value))
        # A launch of a block of the value columns (`Tiles.columns`) stages that block.
        value_width = value_columns(full_v, config.columns)
        padded_q, padded_k = _padded(query.shape[1], _keys(ba), config)
        steps = padded_k // config.block_k
        training = ba.arguments.get("residuals") is Kept.SPLIT
        out_dtype = None if self.out_dtype is None else jnp.dtype(self.out_dtype)
        plan = forward_block_plan(full_v, config.columns, out_dtype,
                                  has_output_mod=self.output_mod is not None, training=training,
                                  pointwise=self.output_tiles is not None)
        # A paged call's key tile of several pages loaded as its launch loads it (`paged_reads`).
        page = _page(ba)
        deciding = self.deciding(ba, config, target=target) if decides is None else decides
        allocations = _forward_allocations(family, config, width, value_width, query.dtype, training,
                                           min(config.stages, steps), out_dtype=plan.out_dtype, page=page,
                                           codes=_codes_of(ba.arguments["key"], ba.arguments.get("scales")),
                                           parts=self.parts, params=self.score_types,
                                           in_key=self.reads_key_slot(family), decides=deciding)
        # A cache's scaled walk reads its keys' value magnitudes from its value slots (`algebra.SCALED_WALK`;
        # another scaled walk the call's, `attention_mosaic_forward._largest`), a reduction across warps
        # (`reducing`).
        arguments = ba.arguments
        cached = self.cached(arguments.get("table"), arguments.get("sources"), arguments.get("scales"))
        reduced = reducing(family, config.consumers, self.score_reduces or plan.output_hook,
                           magnitudes=(cached and self.scales_in_kernel(ba)) or deciding)
        fwd_spec, _ = self._folded(query, arguments.get("residuals"))
        fwd_spec = dataclasses.replace(
            fwd_spec, ids=arguments.get("query_ids") is not None,
            queries=query.shape[1] if padded_q > query.shape[1] or self.per_span > 1 else None,
            walked_keys=padded_k if padded_k > _keys(ba) else None)
        start = arguments["start"]
        span_runs = start.shape[-1] if start is not None and start.ndim == 3 else 1
        tpu_values = (None if family.budgets_registers else (
            {"keys": ((config.block_k, width), query.dtype),
             "values": ((config.block_k, value_width), query.dtype)},
            (((config.block_q, value_width), jnp.float32),
             *step_live_tiles(config.block_q, config.block_k, natural_form=config.natural_form,
                              scaled=self.scales_in_kernel(ba),
                              dropped=arguments.get("drop") is not None, dtype=query.dtype,
                              score_mod=self.score_mod, score_types=self.score_types, spec=fwd_spec,
                              runs=span_runs))))
        return launch_scope(family, allocations, reduced, reduces=forward_reduces(family, reduced),
                            products=products, over_rings=True, values=tpu_values)

    def _staged_bytes(self, ba: tokamax.BoundArguments, config: Tiles, target: generation.GpuTarget, *,
                      products: tuple, decides: bool | None = None) -> int:
        family = bundle_of(config, ba.arguments["query"].dtype, target=target)
        return resources.shared_bytes(family, self._scope(ba, config, target, products=products,
                                                          decides=decides))
    def _products(self, ba: tokamax.BoundArguments, config: Tiles,
                  target: generation.GpuTarget) -> tuple:
        """The launch's products at `config` as its program scopes them (`forward_products`): its logits a
        step ahead where it issues them so (`_ahead`), a scaled walk's small weights' accumulator beside its
        sums where its kernel walks scaled (`scales_in_kernel`), and its query in tensor memory where
        tensor memory holds it beside them without dropping rounded residency (`placed.QUERY`,
        `resources.register_model`, rows 9475, 9479)."""
        family = bundle_of(config, ba.arguments["query"].dtype, target=target)
        query, value = ba.arguments["query"], ba.arguments["value"]
        query_width = _width(query.shape[-1], family, query.dtype)
        value_width = value_columns(_width(value.shape[-1], family, query.dtype), config.columns)
        ahead = self._ahead(ba, config, target)
        scaled = self.scales_in_kernel(ba)
        training = ba.arguments.get("residuals") is Kept.SPLIT
        base = forward_products(family, config, value_width, query.dtype, parts=self.parts,
                                scaled=scaled, ahead=ahead, training=training,
                                tmem_columns=target.tmem_columns)
        if family.places_on_role and self.parts == 1:
            with_query = forward_products(family, config, value_width, query.dtype, parts=self.parts,
                                          scaled=scaled, ahead=ahead, training=training,
                                          query_width=query_width, tmem_columns=target.tmem_columns)
            if with_query != base:
                decides, _, regs = self._decides_and_registers(ba, config, target)
                base_staged = (self._staged_bytes(ba, config, target, products=base, decides=decides)
                               if target.shared_limit else None)
                with_staged = (self._staged_bytes(ba, config, target, products=with_query, decides=decides)
                               if target.shared_limit else None)
                tmem_limit = target.tmem_columns or 0
                base_tmem = family.product_columns(base)
                with_tmem = family.product_columns(with_query)
                base_joined = placed.correction_roles(family, config.consumers, base_tmem, tmem_limit)
                with_joined = placed.correction_roles(family, config.consumers, with_tmem, tmem_limit)
                base_most = resources.register_model(family, config.consumers, regs, shared=base_staged,
                                                     target=target, tmem=base_tmem,
                                                     joined=base_joined).most
                with_most = resources.register_model(family, config.consumers, regs, shared=with_staged,
                                                     target=target, tmem=with_tmem,
                                                     joined=with_joined).most
                if with_most >= base_most:
                    return with_query
        return base

    def joined_roles(self, ba: tokamax.BoundArguments, config: Tiles,
                     target: generation.GpuTarget) -> tuple:
        family = bundle_of(config, ba.arguments["query"].dtype, target=target)
        return placed.correction_roles(family, config.consumers, self.tmem_columns(ba, config, target),
                                       target.tmem_columns or 0)

    def tmem_columns(self, ba: tokamax.BoundArguments, config: Tiles, target: generation.GpuTarget) -> int:
        family = bundle_of(config, ba.arguments["query"].dtype, target=target)
        return family.product_columns(self._products(ba, config, target))

    def _ahead(self, ba: tokamax.BoundArguments, config: Tiles, target: generation.GpuTarget) -> bool:
        """Whether the launch at `config` issues its logits a step ahead of their read (`logits_ahead`)."""
        family = bundle_of(config, ba.arguments["query"].dtype, target=target)
        query, value = ba.arguments["query"], ba.arguments["value"]
        steps = _padded(query.shape[1], _keys(ba), config)[1] // config.block_k
        coded = _codes_of(ba.arguments["key"], ba.arguments.get("scales")) is not None
        value_width = value_columns(_width(value.shape[-1], family, query.dtype), config.columns)
        return logits_ahead(family, config, value_width,
                            query.dtype, stages=min(config.stages, steps), parts=self.parts, coded=coded,
                            scaled=self.scales_in_kernel(ba),
                            training=ba.arguments.get("residuals") is Kept.SPLIT,
                            tmem_columns=target.tmem_columns)

    def cached(self, table, sources, scales) -> bool:
        """Whether a call walks a cache's keys (masked, split, paged, by rows or over codes: a decode step's
        or a serving step's, which the memory bounds), whose kernel scales its rows' sums every tile where
        they could pass fp32's largest (`algebra.SCALED_WALK`): a pass over its values before it, to launch
        it unscaled, would cost the step's bandwidth. Every other call's common kernel holds no scale, a call
        some row of which can be scaled launching the scaled op's (`scaled`, `_fwd`)."""
        return (self.masked or self.splits > 1 or table is not None or sources is not None
                or scales is not None)

    def scales_in_kernel(self, ba: tokamax.BoundArguments) -> bool:
        """Whether the call's kernel walks scaled (the `scaled` op's, a `cached` call's; values that can pass
        fp32's largest, no rounded weights): it reads its keys' value magnitudes and holds its small weights'
        tier (`algebra.at_scale`)."""
        arguments = ba.arguments
        return self._scales(arguments.get("table"), arguments.get("sources"), arguments.get("scales"),
                            _computed(ba))

    def _scales(self, table, sources, scales, computed) -> bool:
        """`scales_in_kernel` of a call of `table`, `sources` and `scales`, its values read in `computed`."""
        return ((self.scaled or (self.cached(table, sources, scales) and not self.bounded))
                and self.rounds is None and algebra.headroom(computed))

    def registers(self, ba: tokamax.BoundArguments, config: Tiles, *, peaks: bool = False,
                  columns: int | None = None, decides: bool | None = None,
                  masked: bool = False, target: generation.GpuTarget | None = None) -> int | None:
        """A consumer's registers at `config`: its walk's (`forward_registers`) over the values' columns
        its launch walks (`columns`, a block of them; the config's own by default), or its peak walk's
        (`peaks`, `peaks_registers`), which a training call takes first; None on a family that budgets
        none (`Family.budgets_registers`: a TPU core's values spill to VMEM), which holds every config.
        Its launches state the same at their own tiles as they lower (`_decided_model`)."""
        return self._decides_and_registers(ba, config, target, peaks=peaks, columns=columns,
                                           decides=decides, masked=masked)[2]

    def registered(self, query, value, score_params, residuals, table=None, sources=None,
                   scales=None, initial=None, drop=None, start=None) -> ForwardRegisters:
        """The call's base register facts (`ForwardRegisters`): an `initial` state merges past an inference
        walk's keys (`attention_mosaic_forward._started`), a split training walk starting from it; a `drop`
        over fp16 values takes its kept weights in two parts (`algebra.kept_parts`); a spans call's `start`
        `[B, Q, R]` names the R key runs a query sees the union of, each further run's bounds held."""
        params = tuple(zip(self.score_kinds, (kept_dtype(param.dtype) for param in score_params),
                           strict=True))
        computed = query.dtype if scales is not None else value.dtype
        spec, taken = self._folded(query, residuals)
        return ForwardRegisters(query.dtype, query.shape[-1], value.shape[-1], residuals is Kept.SPLIT,
                                not spec.base2, params, self.parts,
                                self._scales(table, sources, scales, computed), codes=scales is not None,
                                positions=math.ceil(query.shape[1] / self.per_span)
                                if self.form == "spans" and self.per_span > 1 else 0, per_span=self.per_span,
                                runs=start.shape[-1] if start is not None and start.ndim == 3 else 1,
                                merged=initial is not None and residuals is not Kept.SPLIT,
                                started=initial is not None and residuals is Kept.SPLIT,
                                dropped=drop is not None,
                                weight_parts=2 if drop is not None and value.dtype == jnp.float16 else 1,
                                scaled_products=scaled_drop(computed, drop, initial),
                                prologue=(None if self.query_mod is None
                                          else (self.query_mod, self.query_kinds, self.query_dtypes)),
                                prescaled=spec.prescale != 1.0, rewalk_prescaled=taken != 1.0,
                                out_dtype=self.out_dtype,
                                has_output_mod=self.output_mod is not None, output=self.output_tiles)

    def refusal(self, ba: tokamax.BoundArguments, family: mosaic.Family) -> Refusal | None:
        query, value = ba.arguments["query"], ba.arguments["value"]
        hooks = self._hooks(ba.arguments["score_params"], ba.arguments["output_params"],
                            ba.arguments["initial"], ba.arguments["drop"])
        if self.rounds is not None and (self.parts > 1 or ba.arguments["residuals"] is not Kept.SPLIT
                                        or ba.arguments["table"] is not None):
            # A quantized model's weights round at the row's maximum, which the anchored walk alone knows.
            return ROUNDED
        if self.parts > 1 and (ba.arguments["table"] is not None or ba.arguments["sources"] is not None
                               or ba.arguments["residuals"] is Kept.SPLIT or hooks.drop is not None):
            # A training forward's natural rewalk of parts spills past its model (an H100's at 64 x 64, width
            # 64: 48 bytes on two consumers, 24 on one): the Pallas-Triton mainloop's.
            return Refusal('dtype', "the Mosaic mainloop's fp32 parts walk an inference call over dense "
                           "operands, no dropout")
        if self.parts > 1 and not asks_default_precision(None, configured=True):
            # Its parts' products sum in the tensor cores' accumulator, which an fp32 product above the
            # default precision asks exact (`product.promoted`): on an H100 its forward was 11.8 to 16.8
            # times the reference's error under HIGHEST, its accurate form alike (M-NA-48, row 13498), where
            # the Pallas-Triton mainloop's held check 1.
            return Refusal('dtype', "the Mosaic mainloop's fp32 parts sum in the tensor cores' accumulator, "
                           "an fp32 product's sums above the default precision unpromoted")
        if self.parts > 1 and not loads_parts(family):
            return Refusal('dtype', f"the Mosaic mainloop's fp32 parts load in pieces, which {family.name}'s "
                           "pipeline does not")
        if family.places_on_role and (reason := self._placed_refusal(ba, hooks, family)) is not None:
            return reason
        return refused(query.dtype, query.shape[3], value.shape[3], hooks, backward=False, family=family)

    def reads_key_slot(self, family: mosaic.Family) -> bool:
        """Whether its launches on `family` read the values from the key's slot (`in_key`, `reads_in_key`'s
        call): every family's but one placing its products on an issuing role, whose role declares the
        values' stream (tcgen05's rows), which streams the value array the call was handed."""
        return self.in_key and not family.places_on_role

    def _placed_refusal(self, ba: tokamax.BoundArguments, hooks: Hooks, family: mosaic.Family
                        ) -> Refusal | None:
        """Why a family that places its products on a role (`Issuer.places_on_role`, tcgen05's rows) takes no
        call of `ba`, or None: its consumer gives its issuer each step's weights and takes the sums' factor in
        their memory, and nothing else of a step; so a plain inference walk of 16-bit operands from the empty
        state (no hook, initial state, dropout, cache, scaled or split walk, mask relaunch or parts), its
        sums kept whole across the walk (its value width within its issuer's, `Issuer.keeps_sums`), and its
        output in the query's dtype."""
        arguments = ba.arguments
        query, value = arguments["query"], arguments["value"]
        given = (arguments.get("initial"), hooks.drop, arguments.get("table"), arguments.get("sources"),
                 arguments.get("scales"), hooks.score.fn, hooks.output.fn, self.query_mod)
        walked = (self.scaled or self.bounded or self.masked or self.subtracted or self.splits > 1
                  or self.rounds is not None or arguments.get("residuals") is Kept.SPLIT)
        if any(each is not None for each in given) or walked or self.parts > 1:
            return Refusal('hooks', f"{family.name}'s consumer gives its issuer a plain inference walk's "
                           "weights alone: no hook, initial state, dropout, cache, scaled, split or training "
                           "walk")
        if self.out_dtype not in (None, jnp.dtype(query.dtype).name) or not family.keeps_sums(value.shape[3]):
            return Refusal('dtype', f"{family.name} keeps the sums whole in tensor memory, its output the "
                           "query's dtype")
        return None


def _pages_hold(page: int, tile: int, family: mosaic.Family) -> bool:
    """Whether a paged call's key tiles of `tile` keys lie in its pages of `page`: whole tiles a page, each
    read where it lies (`attention_mosaic_forward._paged_views`), or whole pages a tile, any page, its rows
    gathered through the table where `family` gathers rows (`attention_mosaic_forward.paged_reads`)."""
    return page % tile == 0 or (tile % page == 0 and family.gathers_rows)


def streams(query, target: generation.GpuTarget) -> bool:
    """Whether a call's walk the device's memory bounds: the products its keys and values feed, a query row
    of `query`'s `[B, Q, H, D]` each (4 flops a key entry, its logit's and its weighted value's), fewer a byte
    of them than the device's 16-bit products a byte its memory streams, both the cost model's device rates
    (`cost.rates`: the tensor rate its instruction family, clock and multiprocessors derive, no catalog's
    peak, row 3799): a decode step's few rows a key head, never a prefill's. False where either rate is
    unknown."""
    rated = cost.rates(target, device=True)
    tensor, memory = rated.tensor, rated.device
    if not (tensor and memory):
        return False
    return 2 * query.shape[1] / jnp.dtype(query.dtype).itemsize < tensor / memory


def _keys(ba: tokamax.BoundArguments) -> int:
    """A call's keys, from its arrays: its key rows, or a paged call's table's slots, its pages a row times
    its pools' page (`forward`'s `table`)."""
    page = _page(ba)
    return ba.arguments["key"].shape[1] if page is None else ba.arguments["table"].shape[1] * page


def _computed(ba: tokamax.BoundArguments):
    """The dtype a call's values are read in: the query's where they are quantized codes (their `scales`
    given, `_codes_of`), their own otherwise."""
    query, value = ba.arguments["query"], ba.arguments["value"]
    return query.dtype if ba.arguments.get("scales") is not None else value.dtype


def _page(ba: tokamax.BoundArguments) -> int | None:
    """A paged call's keys a page, its pools' own (`forward`'s `table`); None for a call of rows of keys."""
    return None if ba.arguments.get("table") is None else ba.arguments["key"].shape[2]


@overload
def forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
            *, scale: float | None = None, natural: bool = NATURAL,
            residuals: Literal[Kept.NONE] = Kept.NONE, interpret: bool = False, out_dtype=None,
            masked: bool = False, splits: int = 1, table: jax.Array | None = None,
            sources: jax.Array | None = None, subtracted: bool = False,
            scales: tuple[jax.Array, jax.Array] | None = None, scaled: bool = False,
            bounded: bool = False, in_key: bool = False) -> jax.Array:
    ...


@overload
def forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
            *, scale: float | None = None, natural: bool = NATURAL,
            residuals: Literal[Kept.SPLIT, Kept.OUTPUT], interpret: bool = False,
            out_dtype=None, masked: bool = False, splits: int = 1, table: jax.Array | None = None,
            sources: jax.Array | None = None, subtracted: bool = False,
            scales: tuple[jax.Array, jax.Array] | None = None, scaled: bool = False, bounded: bool = False,
            in_key: bool = False) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, Split] | SplitOutput:
    ...


def forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
            *, scale: float | None = None, natural: bool = NATURAL, residuals: Kept = Kept.NONE,
            interpret: bool = False, out_dtype=None, masked: bool = False, splits: int = 1,
            table: jax.Array | None = None, sources: jax.Array | None = None, subtracted: bool = False,
            scales: tuple[jax.Array, jax.Array] | None = None, scaled: bool = False, bounded: bool = False,
            in_key: bool = False):
    """The template's forward on this mainloop (`MosaicForward`): `query` `[B, Q, H, D]` over `key`
    `[B, K, Hk, D]` and `value` `[B, K, Hk, Dv]`, and the keys `spans` leaves each query under `hooks`, as
    `dew.nn.kernels.templates.attention.reference` computes it; with `residuals` `Kept.SPLIT` the training
    forward's, its output with the backward's residuals as every mainloop's training forward gives them
    (`attention_portable.xla_mainloop(residuals=True)`): the maxima and denominators `[B, H, Q]`, each query's
    anchor key `[B, H, Q]` int32 and the split sums (`Split`); `Kept.OUTPUT` the same residuals of the
    inference walk, which keeps its output for the backward's D in place of an anchor (`output_split`,
    `takes_output_residuals`). `natural` keeps the logits natural
    (`algebra.Forward.base2`); `interpret` runs it on the CPU (Mosaic GPU's interpret mode), as a GPU of the
    family `generation.assumed` names; `out_dtype` the inference walk's output's (`MosaicForward.out_dtype`).
    A paged call's `key` and `value` are its pools `[Hk, P, page, D]` and `table` its page table `[B, pages
    a row]` int32: row `b`'s key `s` at offset `s % page` of page `table[b, s // page]`. With `sources`
    `[B]` int32 (a call of rows of keys, inference), batch row `b` reads key row `sources[b]` of `key` and
    `value` `[R, K, Hk, D]`: a serving step's work list over its cache's rows, no row copied. `subtracted`
    an inference walk's fold of raw products (`MosaicForward.subtracted`). With `splits`, each query tile's
    keys are cut into that many slices, every output the slices' own along the query axis
    (`MosaicForward.splits`): with `Kept.OUTPUT`, two arrays in place of the five, the slices' padded
    outputs and their stacked maxima and denominators (`attention_mosaic_forward._walked_outputs`), which its
    caller merges where they lie (`merged.merged_split`). With `scales`
    (an inference walk's), `key` and `value` are quantized codes, int8 or an FP8, and `scales` the keys' and
    the values' `[B, Hk, K]` fp32, one a key and key head in its row's key order: the call reads the keys and
    values they dequantize to, each converted to the query's dtype in its program (`_Codes`). `scaled`
    launches the scaled op whatever the call's values (`MosaicForward.scaled`: a mainloop whose own kernel
    holds no scale gives it a call some row of which can be scaled); `bounded` the unscaled one, a cache's
    call too, where its caller bounded its values (`MosaicForward.bounded`). `in_key` states
    `value` its keys' leading columns (`MosaicForward.in_key`): its programs read them from the key's slot,
    and the value array, a view of the key's, is no operand of the launch.
    """
    # The key prologue taken as XLA's program before the launch (`key_prologue_taken`); the query's where the
    # call's consumers cannot take it (`_forward_call`).
    in_key = reads_in_key(in_key, key, value, hooks, residuals, table=table, scales=scales)
    key, hooks = key_prologue_taken(key, Hooks() if hooks is None else hooks)
    if residuals is Kept.SPLIT:
        # The training walk reads an interval a query (`KeySpans.per_query`); the inference walk by position.
        spans = spans.per_query()
    op, args = _forward_call(query, key, value, spans, hooks, scale=scale, natural=natural,
                             out_dtype=out_dtype, masked=masked, splits=splits, table=table, sources=sources,
                             subtracted=subtracted, scales=scales, scaled=scaled, bounded=bounded,
                             residuals=residuals, in_key=in_key)
    return op(*args, residuals=residuals, interpret=interpret)


def forward_declined(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans,
                     hooks: Hooks | None = None, *, scale: float | None = None, natural: bool = NATURAL,
                     residuals: Kept = Kept.NONE, in_key: bool = False,
                     certified: bool = False) -> Refusal | None:
    """Why `forward` runs no call of these arrays compiled on this device (`MosaicForward.declined`), and
    where `certified` (the route's selected mainloop alone) as the GPU assembles its configs
    (`MosaicForward.uncertified`), or None."""
    spans = spans.per_query() if residuals is Kept.SPLIT else spans
    in_key = reads_in_key(in_key, key, value, hooks, residuals)
    key, hooks = key_prologue_taken(key, Hooks() if hooks is None else hooks)
    op, args = _forward_call(query, key, value, spans, hooks, scale=scale, natural=natural,
                             residuals=residuals, in_key=in_key)
    ba, target = op.bind(*args, residuals=residuals, interpret=False), generation.gpu_target()
    return op.uncertified(ba, target) if certified else op.declined(ba, target)


def reads_in_key(in_key: bool, key: jax.Array, value: jax.Array, hooks: Hooks | None,
                 residuals: Kept = Kept.NONE, *, table: jax.Array | None = None,
                 scales: tuple[jax.Array, jax.Array] | None = None) -> bool:
    """Whether a call whose values are its keys' leading columns (`in_key`) reads them from the key's slot:
    not under a key prologue, whose keys the slot holds, the values being the keys as given, nor where it
    keeps split residuals, whose anchored training launches stream the values as given
    (`_trained_forward`); nor where the slot holds no value tile its products read: a value wider than the
    key or ending within a swizzle group of its columns (a view no product reads), operands of 32 bits
    (their 16-bit parts load in pieces, `MosaicForward.parts`), a quantized cache's codes (`scales`) or a
    paged cache's gathered pages (`table`). Such a call reads them from the slice it was handed, as one
    without `in_key`: `in_key` states the operands, never another call. A plain forward keeping its rows
    (`Kept.OUTPUT`, a split decode step's slices) reads the key's slot as one keeping none."""
    group = swizzling.swizzle_columns(key.dtype, swizzling.SWIZZLE_BYTES)
    return (in_key and residuals is not Kept.SPLIT and (hooks is None or hooks.key.fn is None)
            and value.shape[-1] <= key.shape[-1] and value.shape[-1] % group == 0
            and jnp.dtype(key.dtype).itemsize == 2 and table is None and scales is None)


def certified(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
              *, scale: float | None = None, natural: bool = NATURAL, interpret: bool = False):
    """The forward's certify form at the tiles its call derives (`MosaicForward.derived`), its walk the plain
    one (no P·V a step late): those tiles, and `certified_walk`'s output and its state after each of their key
    tiles. Off in production; tests/attention_witness.py holds each state to one step's enclosure of the one
    before. A call no config holds raises the reason the forward declines it (`MosaicForward.declined`: an
    unqualified family's, or the device's block and registers). Its query and key as `forward` takes them,
    under their prologues where the launch does not take them (`key_prologue_taken`, `_forward_call`)."""
    key, hooks = key_prologue_taken(key, Hooks() if hooks is None else hooks)
    op, args = _forward_call(query, key, value, spans, hooks, scale=scale, natural=natural)
    ba = op.bind(*args, residuals=Kept.NONE, interpret=interpret)
    target = generation.gpu_target()
    config = op.derived(ba, target)
    if config is None:
        raise ValueError(f"the Mosaic forward declines the call: {op.declined(ba, target)}")
    family = bundle_of(config, query.dtype, target=target)
    if family.places_on_role:
        # Its walk a register family's (each key tile's state stored, `certified_walk`); one whose products a
        # role issues keeps its sums in that role's memory between the products it is given.
        raise ValueError(f"the Mosaic forward's certify form stores each key tile's state from registers; "
                         f"the call derives {family.name}, whose sums its issuer keeps in tensor memory")
    arguments = ba.arguments
    registered = op.registered(arguments["query"], arguments["value"], arguments["score_params"], Kept.NONE,
                               drop=arguments.get("drop"), start=arguments.get("start"))
    if hooks.query.fn is not None and op.query_mod is None:
        # A query prologue the launch does not take where it loads the query (`_forward_call`), taken here.
        query = prologued(query, hooks.query.fn, hooks.query.sides)
        hooks = dataclasses.replace(hooks, query=NO_HOOK)
    return config, certified_walk(query, key, value, spans, hooks, scale=op.scale, natural=op.natural,
                                  tiles=config, interpret=interpret, registered=registered,
                                  tmem_columns=target.tmem_columns)


def _forward_call(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None, *,
                  scale: float | None, natural: bool, out_dtype=None, masked: bool = False,
                  splits: int = 1, table: jax.Array | None = None, sources: jax.Array | None = None,
                  subtracted: bool = False, scales: tuple[jax.Array, jax.Array] | None = None,
                  scaled: bool = False, bounded: bool = False, residuals: Kept = Kept.NONE,
                  in_key: bool = False) -> tuple[MosaicForward, tuple]:
    """`forward`'s op and its arguments, a dropping call's draw among them, which its legality rule refuses
    (`refused`): the op it launches, `scaled` or `bounded` as the call names (`MosaicForward.scaled`,
    `MosaicForward.bounded`), so a plan of it derives that launch's config and reads that launch's
    refusals (`attention_decode.plan`). A query prologue its consumers take where they load the query
    (`MosaicForward.query_mod`) where the launch keeping `residuals` holds it at its own config
    (`_holds_prologue`), else XLA's program before the launch takes it. A hook's side operands are its
    function's (`hooks.Hook`): no score side operand reaches a launch without a score hook to read it (an
    unread key bias's stream on tcgen05's rows, whose issuer streams the keys and values alone, was never
    released and hung the launch)."""
    hooks = Hooks() if hooks is None else hooks
    walked_initial(hooks.initial)
    fp32 = all(jnp.dtype(x.dtype) == jnp.dtype(jnp.float32) for x in (query, key, value))
    parts = product.bf16_part_count(accurate=True) if fp32 else 1
    if hooks.query.fn is not None and (parts > 1 or head_width(query.shape[3]) != query.shape[3]):
        # A query prologue its consumers cannot take where they load the query: an fp32 call's query enters as
        # program before the launch takes it.
        query = prologued(query, hooks.query.fn, hooks.query.sides)
        hooks = dataclasses.replace(hooks, query=NO_HOOK)
    if parts > 1:
        # fp32 operands as their bf16 parts, stacked along the batch axis (`MosaicForward.parts`); the
        # output fp32, the call's. Its fold's logits a difference of raw products scaled after it
        # (`MosaicForward.subtracted`): no reach, so no natural rewalk, whose guarded walk of parts held no
        # registers at width 64 (8 bytes spilled on an H100, et-fp32regs7).
        query, key, value = (product.stacked(x, parts) for x in (query, key, value))
        out_dtype = jnp.float32 if out_dtype is None else out_dtype
        subtracted = True
    op = MosaicForward(score_mod=hooks.score.fn, output_mod=hooks.output.fn,
                       score_kinds=kinds_of(hooks.score.sides),
                       output_kinds=kinds_of(hooks.output.sides),
                       score_dtypes=tuple(kept_dtype(param.value.dtype) for param in hooks.score.sides),
                       output_dtypes=tuple(kept_dtype(param.value.dtype) for param in hooks.output.sides),
                       output_tiles=pointwise_tiles(hooks.output.fn, kinds_of(hooks.output.sides),
                                                    tuple(kept_dtype(param.value.dtype)
                                                          for param in hooks.output.sides),
                                                    (POINTWISE_ROWS, head_width(value.shape[3]))),
                       score_reduces=hook_reduces(hooks), output_width=head_width(value.shape[3]),
                       form=spans.form, keys=spans.keys,
                       scale=logit_scale(query.shape[3], scale), natural=natural,
                       out_dtype=None if out_dtype is None else jnp.dtype(out_dtype).name, masked=masked,
                       splits=splits, subtracted=subtracted, per_span=spans.per_span,
                       parts=parts, rounds=hooks.rounds, scaled=scaled, bounded=bounded, in_key=in_key,
                       query_mod=hooks.query.fn,
                       query_kinds=kinds_of(hooks.query.sides),
                       query_dtypes=tuple(jnp.dtype(param.value.dtype).name for param in hooks.query.sides),
                       query_width=query.shape[3], query_dtype=jnp.dtype(query.dtype).name)
    args = (query, key, value, spans.start, spans.end, spans.query_ids, spans.key_ids,
            tuple(param.value for param in hooks.score.sides),
            tuple(param.value for param in hooks.output.sides), hooks.initial, hooks.drop,
            table, sources, scales, tuple(param.value for param in hooks.query.sides))
    if op.query_mod is not None and not _holds_prologue(op, args, residuals):
        taken = dataclasses.replace(hooks, query=NO_HOOK)
        return _forward_call(prologued(query, hooks.query.fn, hooks.query.sides), key, value, spans, taken,
                             scale=scale, natural=natural, out_dtype=out_dtype, masked=masked, splits=splits,
                             table=table, sources=sources, subtracted=subtracted, scales=scales,
                             scaled=scaled, bounded=bounded, residuals=residuals, in_key=in_key)
    return op, args


def _holds_prologue(op: MosaicForward, args: tuple, residuals: Kept) -> bool:
    """Whether a forward launch keeping `residuals` takes its query prologue where its consumers load the
    query (`MosaicForward.query_mod`) at the config it derives without it: a neighbour joins a kernel that
    still holds its residency (kernel-templates.md's hooks), never one whose prologue's registers
    (`ForwardRegisters.prologue`) cost it a consumer or a tile its walk would run (Qwen3's QK-norm and RoPE
    at width 128 on a B200 priced 236 of a consumer's 240 registers, et-prologue-fits)."""
    target = generation.gpu_target()

    def holds(op: MosaicForward) -> bool:
        plain = dataclasses.replace(op, query_mod=None, query_kinds=(), query_dtypes=())
        held = op.fits(op.bind(*args, residuals=residuals, interpret=False), target)
        alone = plain.fits(plain.bind(*args[:-1], (), residuals=residuals, interpret=False), target)
        return bool(held) and held == alone

    # Its scaled op's too where the call may launch it (`MosaicForward._fwd`): the prologue is the call's,
    # either branch's.
    rare = op.scaled or op.bounded or op.rounds is not None or not algebra.headroom(args[2].dtype)
    return holds(op) and (rare or holds(dataclasses.replace(op, scaled=True)))


def producer_raised(family: mosaic.Family, target: generation.GpuTarget) -> bool:
    """Whether `family`'s program runs a producer warpgroup lowered beneath its consumers (`Family.roles`'
    role of weight 0) whose registers it gives them: each consumer's (`Family.registers`) past the program's
    even share of the multiprocessor's (`resources.entry_registers`), as a compile that honors the raises
    gives them (`resources.RAISES_HONORED`; FlashAttention 3's 24/240), on `target`'s multiprocessor whatever
    GPU the call is derived under (`generation.as_target`). The fused backward takes such a layout before a
    fed one (`MosaicBackward._one_launch`)."""
    warpgroups = family.roles(family.consumers).roles
    if not any(role.weight == 0 and not role.warp for role in warpgroups):
        return False
    held = sum(role.count for role in warpgroups if not role.warp)
    with generation.as_target(target):
        return family.registers(family.consumers) > resources.entry_registers(held)


def _columns_legal(ba: tokamax.BoundArguments, config: BackwardTiles, family: mosaic.Family, dtype) -> bool:
    """Whether a backward config's consumers and programs split the call's widths as `family` stages them
    (`MosaicBackward.legal`'s last part): a shared key block's consumers each their share of the columns, a
    width split among programs its shares."""
    if config.shared_keys:
        # Every consumer's share of the keys whole rows of a swizzle the exchange stages them at
        # (`swizzling.swizzle_for`), its share of the columns whole swizzled rows
        # (`attention_mosaic_coop`).
        share = _width(ba.arguments["query"].shape[-1], family, dtype) // config.consumers
        narrowest = family.narrowest_columns(dtype)
        return (config.fused and config.queries_first and config.consumers > 1 and config.columns == 1
                and (config.block_k // config.consumers) % narrowest == 0
                and share % family.column_unit(dtype) == 0)
    # A width split among programs adds its dQ once, in no fused launch, each share whole swizzled rows
    # and, where its family's products read their right operand's columns from a block staged at its
    # whole width (`Family.operands_in_registers`), whole swizzle tiles of that block: an A100's accurate
    # dQ under a score hook split 64 columns in two, its keys' 64-column tile sliced at 32, and failed to
    # lower ("slice size (32) divisible by the tile size (64)", FS-A100CHECK1d, acchook3).
    widths = [_width(ba.arguments[name].shape[-1], family, dtype) for name in ("query", "value")]
    def unit(width: int) -> int:
        if not family.operands_in_registers:
            return family.column_unit(dtype)
        tile = swizzling.swizzle_columns(dtype, swizzling.swizzle_for(dtype, width))
        return max(family.column_unit(dtype), tile)

    units = [unit(width) for width in widths]
    return config.columns == 1 or (not config.fused and all(
        (width / config.columns) % unit == 0 for width, unit in zip(widths, units, strict=True)))


@dataclasses.dataclass(frozen=True)
class MosaicBackward(_Mainloop[BackwardTiles]):
    """The template's backward on this mainloop (`gradients`) over the call's own arrays (`backward` gives
    them): its queries, keys, values and output cotangent, its spans' arrays, its score hook's parameters, its
    initial state (`Initial`, or None) and a training forward's residuals (`forward`'s). Returns its
    gradients (`Gradients`). Its config comes from the family's facts and the call's extents (`derived`: one
    launch for a fast-form call whose score hook takes no parameters, two in deterministic mode, for the
    accurate form and for a call whose hook does), legal within the device's block and a consumer's
    registers (`legal`)."""

    HOOKS = ("score_mod",)
    DIFFERENTIATED = True
    SPANS = 4
    """Its arrays' spans after the queries, keys, values and the output's cotangent (`_fwd`)."""
    config_cls = BackwardTiles
    WALKED = "block_q"
    accurate: bool = ACCURATE
    rounded: bool = True
    score_slopes: tuple[int, int] = (0, 0)
    """The tiles its score hook's pullback forms that vary in its dQ launch and its dK/dV launch
    (`varying_slopes`), as its builder reads them off the hook (`_backward_call`): a keyed launch's hook is
    its fingerprint, which keeps no trace of them."""
    score_reads: tuple[bool, bool] = (False, False)
    """Whether its score hook reads its queries' places and its keys' (`score_places`), as its builder reads
    them off the hook (`_backward_call`): a keyed launch's hook is its fingerprint, which keeps no trace of
    them, and its registers hold each it reads (`backward_registers`' `hooked`)."""

    def _fwd(self, query, key, value, dout, start, end, query_ids, key_ids, score_params, initial, residuals,
             drop=None, *, interpret: bool, config: BackwardTiles, return_residuals: bool = False):
        family = bundle_of(config, query.dtype)
        spans = self.spans(start, end, query_ids, key_ids)
        hooks = Hooks(score=Hook(self.score_mod, tuple(map(Side, score_params, self.score_kinds))),
                      initial=initial, drop=drop)
        if reason := refused(query.dtype, query.shape[3], value.shape[3], hooks, backward=True,
                             family=family):
            raise NotImplementedError(reason)
        queries = start.shape[1]
        padded_queries, padded_keys = _padded(queries, key.shape[1], config)
        spec = self._spec(queries, key.shape[1], ids=query_ids is not None,
                          initial=initial is not None, config=config)
        launch = functools.partial(_launch_backward, spec=spec, accurate=self.accurate, tiles=config,
                                   family=family, interpret=interpret,
                                   drop=placed_draws(drop, padded_queries, padded_keys), rounded=self.rounded,
                                   registered=self.registered(query, value, score_params, drop, config))
        return gradients(launch, query, key, value, spans, hooks, dout, residuals,
                         block=_block(config), scale=self.scale, rounded=self.rounded), None

    def _spec(self, queries: int, keys: int, *, ids: bool, initial: bool,
              config: BackwardTiles) -> algebra.Forward:
        """Visibility and geometry spec shared by `_fwd` emission and `_scopes` VMEM planning."""
        padded_queries, padded_keys = _padded(queries, keys, config)
        return algebra.Forward(self.form, self.keys, ids=ids, scale=self.scale,
                               score_mod=self.score_mod, natural=self.natural, initial=initial,
                               queries=queries if padded_queries > queries else None,
                               walked_keys=padded_keys if padded_keys > self.keys else None)

    UNBUNDLED: ClassVar[str] = ("the Mosaic backward runs on no bundle of {name}'s: its launches scope no "
                                "issuer's product scratch, which tcgen05's chunks keep (row 4839, its "
                                "tensor-memory form)")

    @staticmethod
    def _bundle(generation_name: str | None, dtype) -> mosaic.Family | None:
        """The bundle the backward's dQ launch runs on (`backward_bundle`)."""
        return backward_bundle(generation_name, dtype)

    @staticmethod
    def _layouts(generation_name: str | None, dtype, rows: int | None = None) -> tuple[mosaic.Family, ...]:
        """The bundles the backward's launches take (`backward_layouts`), whatever its rows."""
        del rows
        return backward_layouts(generation_name, dtype)

    def _named(self, generation_name: str | None, dtype, rows: int | None = None) -> dict[str, mosaic.Family]:
        # Those its launches take, and those whose walks are placed on a role (`walked_layouts`), which a
        # config names where its launch is pinned there (none a derivation offers before a receipt).
        del rows
        return {family.name: family for family in (*self._layouts(generation_name, dtype),
                                                   *walked_layouts(generation_name, dtype))}

    def _walked_legal(self, ba: tokamax.BoundArguments, config: BackwardTiles, family: mosaic.Family,
                      tiles: BackwardTiles, target: generation.GpuTarget) -> bool:
        """Whether a launch whose products `family` places on a role (`placed.Tcgen05Walked`, row 4839)
        takes the call at its `tiles` (the dK/dV launch's keys first, `placed.KEYS`, at `config.dkv`; the dQ
        launch's queries first, `placed.QUERIES`, at `config`): the fast form of a plain call (no score
        hook's parameters, whose gradients and operands its compute warpgroups read from no stream of their
        own; no dropout; no spans of several queries; the logits' queries no stage of a consumer's), its held
        block the issuer's rows and its consumers its compute warpgroups, one launch (`BackwardTiles.fused`,
        its dQ its own) keys first alone, within the multiprocessor's tensor memory: two turns of the scores
        and of their cotangents, a step's walked tile each, beside its kept sums' shares of their columns
        (`BackwardTiles.columns`)."""
        query = ba.arguments["query"]
        # Its operands 16-bit of one part (bf16: fp16's weights and cotangents two scaled parts, whose
        # products no placed walk issues), a fused walk's dQ products a 64- or 128-row tile of its queries
        # (tcgen05's M).
        if (jnp.dtype(query.dtype) != jnp.dtype(jnp.bfloat16) or self.accurate or self.score_kinds
                or ba.arguments["drop"] is not None or self.per_span > 1 or config.staged_logits
                or config.shared_keys or tiles.consumers != family.consumers
                or (config.fused and (not family.holds_keys or tiles.block_q not in (64, 128)))):
            return False
        held, walked = held_walked(family, tiles)
        if held != family.rows:
            return False
        # Its scratch as its launch allocates it (`placed_scratch`, the head widths as staged, `_width`).
        width, value_width = (_width(ba.arguments[name].shape[-1], family, query.dtype) // config.columns
                              for name in ("query", "value"))
        halves = dq_halves(width * config.columns) if config.fused else ()
        products = placed_scratch(family, walked, query.dtype, halves,
                                  (width, value_width) if family.holds_keys else (width,))["products"]
        return family.product_columns(products) <= (target.tmem_columns or 0)

    def legal(self, ba: tokamax.BoundArguments, config: BackwardTiles, target: generation.GpuTarget) -> bool:
        # The one launch adds each tile's dQ under the first pass's D: the fast form's, a call's of equal
        # head widths, its dK/dV program queries first on the dQ launch's layout or keys first on
        # one whose consumers share a key block (`split_queries`), each consumer's share of a slot's queries
        # a warpgroup MMA's rows (dQ's product's M). A score hook's and an initial state's only
        # queries first (at width 256 its consumers sharing a key block, `attention_mosaic_coop`), and a
        # hook of no parameters, whose gradients no launch writes (Gemma 2's softcap, FlashAttention 3's);
        # the state's (gpt-oss's sinks) the first pass's D and the keys' share of the weights give
        # (`gradient.split_mass`), its weights' denominator the forward's. A dropping call's only queries
        # first, each consumer its own keys and its own draw, under the dropping call's D (FlashAttention
        # 2's).
        hooked, dropped = bool(self.score_kinds), ba.arguments["drop"] is not None
        if config.fused and (self.score_mod is not None or ba.arguments["initial"] is not None or dropped
                             ) and not config.queries_first:
            return False
        if config.fused and dropped and config.shared_keys:
            return False
        dtype = ba.arguments["query"].dtype
        found = self._named(target.generation, dtype)
        dkv_family = found.get(config.dkv.family)
        # A family the device has no bundle of (a pin of an H100's wgmma on an A100) is not read here: the
        # mainloop's legality refuses it (`_Mainloop.legal`).
        if dkv_family is None or not whole_tiles(dkv_family, config.dkv):
            return False
        # A placement takes the launch it is written for (`placed.walked_launch`): keys first the dK/dV
        # launch's, queries first the dQ launch's; the one launch's keys first apart from its config's own.
        dq_named = found.get(config.family)
        walked_dkv = placed.walked_launch(dkv_family) == "dkv" if dkv_family.places_on_role else None
        if walked_dkv is False or (walked_dkv and not self._walked_legal(ba, config, dkv_family, config.dkv,
                                                                          target)):
            return False
        if dq_named is not None and dq_named.places_on_role and (
                config.fused or placed.walked_launch(dq_named) != "dq"):
            return False
        # A dK/dV launch whose producer streams a TILE parameter's keys-first blocks with its queries
        # (`streams_operands`) stages one key block's (`_dkv_allocations`): a launch of more holders takes
        # none, which no config of its family held either, a tile bias's two holders past a thread's
        # registers at the family's one tile (`Family.registers`, `resources.role_registers`; et-a2-legal),
        # until a capability holds them (row 5741).
        if (not config.fused and streams_operands(dkv_family) and Kind.TILE in self.score_kinds
                and holders(dkv_family, config.dkv.consumers) > 1):
            return False
        # The one launch's tiles a role's rows at least (`_role_rows`): no issuer specializes it below them
        # (`fewer_rows` the two launches' alone), its 32 x 32 tile on an sm120's fed mma.sync and an sm86's
        # mma.sync "cannot relayout" (et-sm120fused); a family placing its products on a role keys first
        # walks its own blocks (`BackwardTiles.dkv_blocks`). The dQ launch streams through a producer
        # (`runs_backward`), which a family whose consumers feed their own rings has not: the one launch
        # there, queries first.
        if not dkv_family.places_on_role:
            if (config.fused and min(config.block_q, config.block_k) < dkv_family.role_rows
                    and min(config.block_q, config.block_k) not in fewer_rows(dkv_family)):
                return False
            row_unit = backward_family(dkv_family, config.dkv).transposed_rows
            if config.block_q % row_unit or config.block_k % row_unit:
                return False
        dq_family = found.get(config.family)
        if not config.fused and dq_family is not None and split_queries(dq_family):
            return False
        if (not config.fused and dq_family is not None and dq_family.places_on_role
                and not self._walked_legal(ba, config, dq_family, config, target)):
            return False
        keys_first = config.dkv_apart
        if config.fused and (ba.arguments["query"].shape[-1] != ba.arguments["value"].shape[-1] or (
                keys_first and not dkv_family.places_on_role
                and not (split_queries(dkv_family)
                         and (config.block_q // config.dkv.consumers) % dkv_family.rows == 0))
                or (config.staged_logits and not keys_first)):
            return False
        dtype = ba.arguments["query"].dtype
        # The one launch stages its weights and score cotangents for its products (`swizzling.staged`): fp16's
        # as two parts at the walk's running scale (`attention_mosaic_fused._sixteen_step`), queries first,
        # each consumer its own keys or its share of a block they share (`attention_mosaic_coop`'s, every
        # share's scale joined): row 3886, its two launches 4.1 times the bf16 step on an H100.
        sixteen = jnp.dtype(dtype) == jnp.dtype(jnp.float16) and not config.queries_first
        if ((config.fused and (not fuses(self.accurate, hooked) or sixteen))
                or not super().legal(ba, config, target)):
            return False
        family = bundle_of(config, dtype, target=target)
        # Consumers past those its family's derivations take (`Family.shared_consumers`'s) the paired fused
        # launch's alone, queries first over keys each its own, its consumers' dQ summed (`dq_summed`): no
        # other walk of its family is written for them (a shared key block's, `_coop_launch`, loads by one).
        if (family is not None and config.consumers > family.consumers
                and not (config.queries_first and not config.shared_keys
                         and dq_summed(family, config.consumers))):
            return False
        # Deterministic mode takes no one launch: its programs add into dQ (and grouped dK and dV) across
        # programs, whose fixed order would wait on programs an ordinary launch need not hold resident
        # (row 4732: a ticket spins on a predecessor another stream's kernel keeps off the
        # multiprocessors). Its two launches' sums are each program's own, a key head's query heads'
        # summed in their order in XLA.
        if config.fused and deterministic_ops_requested():
            return False
        if (family is not None and min(config.block_q, config.block_k) < family.role_rows
                and not _fewer_hooked(self.score_mod, family)):
            # takes none fewer (`_fewer_hooked`): a hooked call's tiles of fewer fail to lower there.
            return False
        return _columns_legal(ba, config, family, dtype)

    def walked(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> BackwardTiles | None:
        """The config whose launches tcgen05's walked placements take for the call (`walked_layouts`,
        `placed.Tcgen05Walked`, row 4839), the derivation's search of them by what fits (`legal`: shared
        memory, tensor memory and registers as the model has them): the one launch keys first, its dQ its
        own (FlashAttention 4's), at the largest step of queries that holds; else the two launches, dQ queries
        first and dK/dV keys first, each at the largest walked tile that holds, dK's and dV's columns in the
        fewest shares that hold (at width 256 the one launch's resident keys and values and its query streams
        pass a B200's block, row 4839's arithmetic). None where the device has none or none holds. `derived`
        offers it first (M-ET-74's receipt, row 6918)."""
        query, key = ba.arguments["query"], ba.arguments["key"]
        found = {family.name: family for family in walked_layouts(target.generation, query.dtype)}
        keys, queries = found.get(placed.KEYS.name), found.get(placed.QUERIES.name)
        base = next((layout for layout in self._layouts(target.generation, query.dtype)
                     if not split_queries(layout)), None)
        if keys is None:
            return None
        steps = walked_extents(keys, query.dtype, query.shape[1])

        def keys_first(step: int, **tiles) -> BackwardTiles:
            return BackwardTiles(**tiles, dkv_family=keys.name, dkv_consumers=keys.consumers,
                                 dkv_blocks=(step, keys.rows))

        # The one launch's own tiles its padding's alone, no launch walking them (`_block`).
        fused = [keys_first(step, block_q=step, block_k=step, stages=2, consumers=1, fused=True,
                            family=base.name) for step in steps] if base is not None else []
        two = [keys_first(step, block_q=queries.rows, block_k=walked, stages=2, consumers=queries.consumers,
                          family=queries.name, columns=columns)
               for columns in (1, 2, 4) for walked in walked_extents(queries, query.dtype, key.shape[1])
               for step in steps] if queries is not None else []
        return next((config for config in (*fused, *two) if self.takes(ba, config, target)), None)

    def ladder(self, ba: tokamax.BoundArguments, config: Hashable) -> tuple[Hashable, ...]:
        # The query blocks its derivation walks below `config`'s for its consumer's rows (`_query_walk`) on
        # the layout its derivation walks (`_named`'s, its warps not yet arranged for the consumer's rows as
        # `bundle_of`'s are, mma.sync's at 32 rows two warps of 16, of which no fewer rows are), `config`'s
        # other fields kept, within its register cap and residency, largest first.
        if not self._typed(config):
            return ()
        target = generation.gpu_target()
        query = ba.arguments["query"]
        layout = self._named(target.generation, query.dtype, query.shape[1]).get(config.family)
        if layout is None:
            return ()
        walk = _query_walk(config.block_k if config.block_k in fewer_rows(layout) else None)
        regime = self._regime(ba, config, target)
        smaller = (dataclasses.replace(config, block_q=block)
                   for block in walk(layout, query.dtype, query.shape[1]) if block < config.block_q)
        return tuple(rung for rung in smaller
                     if self.takes(ba, rung, target) and self._regime(ba, rung, target) == regime)

    def _regime(self, ba: tokamax.BoundArguments, config: BackwardTiles, target: generation.GpuTarget
                ) -> tuple:
        """The assembler's conditions at `config`: each launch's register cap and whether its program moves
        registers, as its kernels' models state them (`resources.register_model` at the registers its
        emission states, `_launch_registers`), and the launch's residency."""
        models = [model for _, _, model in self._launch_models(ba, config, target)]
        return (*((model.cap(1), model.moves) for model in models), self.resident(ba, config, target))

    def derived(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> BackwardTiles | None:
        """Its tiles from the facts (`_derived`): each dK/dV consumer a unit of keys, the query tile both
        launches walk the largest the block holds, its registers the assembler's to judge (`ranks_alone`: a
        config that spills is refused where it compiles, and the derivation steps past it), two stages, one
        consumer, whose step
        issues its products back to back (TRAINC-h100: 6.88 against 3.72 ms causal, 9.68 against 7.56 full,
        at 64 x 64 with two consumers and one). The fast form of a hookless call (or one whose hook takes no
        parameters, a softcap, or under an initial state, sinks) takes one launch where one holds the whole
        width, its dK/dV program queries first adding each tile's dQ atomically, where the cost model prices
        it below the two launches (`_cheaper`: an A100's two launches measured faster at every cell,
        M-NA-13-a100, MHA d64 2706 against 2419 us, Llama d64 6688 against 4582, Qwen d128 12950 against
        12401, d256 20454 against 12909, its dQ adds throttled on the global queue, M-NA-9-a100; on an H100
        FUSEDFORM-h100: 1.05 to 1.4 times as fast at widths 64 and 128 over 1k to 8k tokens; ET-A05AB3-h100:
        5.20 against 6.19 ms the two keys-first launches at Qwen3-8B causal, 1.19 against 1.53 at width 64;
        check 1 on drawn inputs and finite over every stability family it takes, FUSEDSTAB-h100; 384 MB of
        temporaries against 258 at llama-d128 2 x 4096, its dQ fp32), a call over key spans, windows,
        documents or runs as a plain one (ET-CROSS-h100 and ET-SPAN4-h100, the attention launches a training
        step: Qwen3-8B's heads over four documents 0.67 to 0.72 times the two launches' from 1 x 1024 to
        8192, under a causal window of 4096 at 8192 0.65; width 64's 8 heads over 2 0.56 to 0.88, 16 over
        16 under the window 0.54; check 1 over documents and a window at 1 x 2048; of every call measured,
        row 3171's toy one alone the slower, 1 x 1024 over three short documents and a window of 256,
        1.38 times: 20 us a step); otherwise (deterministic mode, `deterministic_ops_requested`; the
        accurate form, a hook with parameters, a width no fused launch holds) two launches, the dQ launch
        then the keys-first dK/dV one, in a fixed order. On a device whose walks are placed on a role
        (tcgen05's, `walked_layouts`), the walked config first where it takes the call (`walked`: on an
        emulated B200 0.18, 0.15 and 0.42 of the mma.sync two launches' time at Qwen3-8B's 2 x 4096 and
        1 x 16384 and Qwen3-4B's width 256, M-ET-74, row 6918)."""
        if self._family(ba, target) is None:
            return None
        if (walked := self.walked(ba, target)) is not None:
            return self._deepened(ba, walked, target)
        candidates = self.candidates(ba, target)
        return self._cheaper(ba, candidates, target) if candidates else None

    def candidates(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> list[BackwardTiles]:
        """The legal one launch and two launches `derived` prices (`_one_launch`, `_two_launches`), each
        deepened (`_deepened`), in that order; none on a target of no family."""
        family = self._family(ba, target)
        if family is None:
            return []
        query = ba.arguments["query"]
        # The dK/dV launch on the layout whose consumers feed their rings where one is, its two consumers
        # splitting each slot's queries (`split_queries`); else on the dQ launch's.
        fed = [layout for layout in self._layouts(target.generation, query.dtype) if split_queries(layout)]
        # The dK/dV launch's consumers a backward's at most (`BACKWARD_CONSUMERS`), whatever its layout's.
        dkv = ({"dkv_family": fed[0].name, "dkv_consumers": min(fed[0].consumers, BACKWARD_CONSUMERS)}
               if fed else {})
        return [self._deepened(ba, config, target) for config in (
            self._one_launch(ba, target, family, fed, dkv), self._two_launches(ba, target, family, dkv))
            if config is not None]

    def _cheaper(self, ba: tokamax.BoundArguments, candidates: list[BackwardTiles],
                 target: generation.GpuTarget) -> BackwardTiles:
        """The one launch or the two `candidates` hold whose launches the cost model prices least
        (`price_config`), a candidate it cannot price whole never the cheaper beside one it can, as
        `dew.algebra.cost.cheaper` reads it: one whose work the target prices no rate of (`cost.units`'
        unpriced: a GPU stating no `global_reduction_bytes_per_clock` beside the fused launch's dQ adds), or
        one of whose launches' registers its model holds past their cap (`_launch_models`), whose spills no
        rate prices (an H100's two launches at 128 x 64 one consumer, model 331 at a budget of 248: its dQ
        took 248 registers and 696 bytes of stack, refused, where the fused one certified). Stored beside the
        verdicts so a warm start traces nothing to price (`_choice_key`, `tuning.stored_choice`); one whose
        launches the model cannot count (`price_config`'s None) priced whole by nothing; the first where one
        alone is legal or the target holds no rates."""
        if len(candidates) <= 1 or not target.multiprocessors or not cost.rates(target, device=False).tensor:
            return candidates[0]
        facts = _priced(target)
        key = _choice_key(self, ba, tuple(candidates), facts)
        stored = None if key is None else tuning.stored_choice(target.generation, key)
        if isinstance(stored, (tuple, list)):
            matched = next((config for config in candidates if config.fused == bool(stored[0])), None)
            if matched is not None:
                return matched
        # A price is (unpriced, seconds): a candidate priced whole before any that is not, then the faster;
        # at the generation's facts of the target's size (`_priced`).
        prices = [self.price_config(ba, config, facts) or (True, math.inf) for config in candidates]
        best = min(zip(prices, range(len(candidates)), candidates, strict=True))[2]
        if key is not None:
            tuning.store_choice(target.generation, key, (int(best.fused),))
        return best

    def price_config(self, ba: tokamax.BoundArguments, config: BackwardTiles, target: generation.GpuTarget
                     ) -> tuple[bool, float] | None:
        """Price `config`'s launches on `target`, the fused one's or the two's summed (`cost.launch_seconds`
        over `emission.launched_kernels`, each kernel at its own launch's residency, `_launch_models`), traced
        under `report.launching`: whether any of their work lies at a rate the target states none of, a
        multiprocessor's or the whole device's (`cost.units`' unpriced; its memory's bandwidth, which the
        first prices infinite), or a step of no latency (`cost.spanned`'s), or a launch's registers' model
        passes its cap (spills no rate prices), and the seconds; None where a launch's work is uncounted, a
        launch lies under a loop or a branch, or it traces none. Every fact read is `target`'s, whatever GPU
        is in effect where it is asked. As its hookless call (its hook fields none): a hook a choice prices
        takes no parameters, its work the same a score in either form, and a launch keyed by its hooks'
        fingerprints (`KernelOp.keyed`, which traces none) prices as its live call does; one whose hook takes
        parameters (which holds no fused launch, so no choice) None."""
        if ba.arguments["score_params"]:
            return None
        names = ("query", "key", "value", "dout", "start", "end", "query_ids", "key_ids", "score_params",
                 "initial", "residuals", "drop")
        operands = tuple(tuning.shapes(ba.arguments[name]) for name in names)
        identity = report._identity(self, ba, target)

        hookless = dataclasses.replace(self, **dict.fromkeys(self.HOOKS))

        def priced(*args):
            return hookless._fwd(*args, interpret=False, config=config)[0]

        priced.__name__ = priced.__qualname__ = (
            f"priced_{type(self).__name__}_l{report._digest(identity)}_c{report._digest(repr(config))}")
        with generation.assumed(target):
            with report.launching(self, ba, config):
                traced = jax.make_jaxpr(priced)(*operands)
            # Its launches as their emission counted them (`emission.launched_kernels`): None where one is
            # uncounted or under control flow.
            if not (found := emission.launched_kernels(traced.jaxpr)):
                return None
            budgeted = bundle_of(config, ba.arguments["query"].dtype, target=target).budgets_registers
            launches = self._launch_models(ba, config, target) if budgeted else ()
            unpriced = any(registers > model.cap(1) for _, registers, model in launches)
            # Each kernel at its launch's residency: the dQ launch's kernels `DQ_KERNEL`'s, the rest the
            # dK/dV launch's, its model the last; the launch's on a family that budgets no registers.
            residency = ({dq: max(1, launches[0 if dq else -1][2].most) for dq in (True, False)}
                         if launches else dict.fromkeys((True, False), self.resident(ba, config, target)))
            rates = [cost.rates(target, device=device) for device in (False, True)]
            latency = cost.step_latency(target, device=False)
            total = 0.0
            for kernel, launched in found:
                work = launched.work
                dq = assembled.kernel_of(kernel.params.get("name") or "")[0].startswith(DQ_KERNEL)
                total += cost.launch_seconds(work, emission.launch_programs(kernel), target,
                                             resident=residency[dq])
                unpriced = (unpriced or any(any(cost.units(work, at)[1]) for at in rates)
                            or bool(cost.spanned(work, latency)[1]))
        return unpriced, total

    def _launch_models(self, ba: tokamax.BoundArguments, config: BackwardTiles, target: generation.GpuTarget
                       ) -> tuple[tuple[BackwardTiles, int, resources.RegisterModel], ...]:
        """Each launch `config` runs, the dQ launch's first (none where `fused`), then the dK/dV launch's at
        its own tiles (`BackwardTiles.dkv`, the fused launch's too, whose layout may be its own, `dkv_apart`):
        its tiles, the registers a consumer holds (`_launch_registers`), and its register model at them and
        its own shared memory (`resources.register_model`, each term `target`'s whatever GPU is in effect)."""
        dtype = ba.arguments["query"].dtype
        dq, dkv = self._launch_registers(ba, config, target=target)
        # `_scopes` lists the dK/dV launch's program first, then the dQ launch's.
        staged = ([resources.shared_bytes(layout, scope)
                   for layout, _, scope in self._scopes(ba, config, target)]
                  if target.shared_limit else [None, None])
        launches = (((config.dkv, dkv, staged[0]),) if config.fused
                    else ((config, dq, staged[1]), (config.dkv, dkv, staged[0])))
        return tuple((tiles, registers, resources.register_model(
            bundle_of(tiles, dtype, target=target), tiles.consumers, registers, target=target, shared=shared))
            for tiles, registers, shared in launches)

    def _deepened(self, ba: tokamax.BoundArguments, config: BackwardTiles,
                  target: generation.GpuTarget) -> BackwardTiles:
        """Deepen a backward launch's ring up to `max(consumers, 2) + 1` stages at its residency via
        `resources.plan`, preserving streamed `peak`/`total`/`delta` statistics on fused launches
        (rows 4839, 5272, 10948, 10962)."""
        query = ba.arguments["query"]
        family = bundle_of(config, query.dtype, target=target)
        if not family.budgets_registers:
            return config
        q_block = config.dkv_blocks[0] if config.dkv_blocks is not None else config.block_q
        q_steps = -(-query.shape[1] // q_block)
        steps = q_steps if config.fused else min(q_steps, -(-ba.arguments["key"].shape[1] // config.block_k))
        cap = 3 if config.dkv_blocks is not None else max(config.consumers, 2) + 1
        want = min(cap, steps)
        if want <= config.stages:
            return config
        base_resident = self.resident(ba, config, target)
        # Only queries-first fused/coop launches stream per-query statistics rings, sized at padded `_width`
        # matching `_scopes` (row 11989).
        if config.queries_first:
            padded_width = _width(query.shape[3], family, query.dtype)
            reduced = reducing(family, config.consumers, self.score_reduces,
                               scaled=staged_parts(query.dtype) > 1)
            anchored = ba.arguments["drop"] is not None
            grouped = query.shape[2] != ba.arguments["key"].shape[2]

            def streamed_at(stages: int) -> int:
                at = dataclasses.replace(config, stages=stages)
                if config.shared_keys:
                    return streamed_statistics(family, at, padded_width, query.dtype, stages,
                                               reduced=reduced, allocations_of=_coop_allocations,
                                               anchored=anchored, target=target)
                return fused_staging(family, at, padded_width, query.dtype, stages, reduced=reduced,
                                     grouped=grouped, rounded=self.rounded, anchored=anchored,
                                     target=target).streamed

            base_stats = streamed_at(config.stages)
            while want > config.stages and streamed_at(want) < min(3, base_stats):
                want -= 1
        base_scopes = self._scopes(ba, config, target)
        dq_regs, dkv_regs = self._launch_registers(ba, config, target=target)
        launch_regs = (dkv_regs,) if config.fused else (dkv_regs, dq_regs)
        planned = min(
            _planned_stages(layout, lambda s, i=i: self._scopes(
                ba, dataclasses.replace(config, stages=s), target)[i][2],
                target, consumers=consumers, registers=regs, floor=config.stages, most=want,
                resident=base_resident)
            for i, ((layout, consumers, _), regs) in enumerate(zip(base_scopes, launch_regs, strict=True)))
        for stages in range(planned, config.stages, -1):
            candidate = dataclasses.replace(config, stages=stages)
            if self.takes(ba, candidate, target) and self.resident(ba, candidate, target) == base_resident:
                return candidate
        return config

    def _staged_helps(self, ba: tokamax.BoundArguments, config: BackwardTiles,
                      target: generation.GpuTarget) -> bool:
        """Whether `config.staged_logits` relieves shared memory (makes an over-budget tile fit or raises
        residency): when the un-staged config already fit shared memory at the same residency and was
        refused for register spills, staging logits in shared memory cannot reduce registers (row 11975)."""
        if not config.staged_logits:
            return True
        unstaged = dataclasses.replace(config, staged_logits=False)
        return (self.staged_bytes(ba, unstaged, target) > self.program_shared(unstaged, target)
                or self.resident(ba, config, target) > self.resident(ba, unstaged, target))

    def _one_launch(self, ba: tokamax.BoundArguments, target: generation.GpuTarget, family: mosaic.Family,
                    fed: list, dkv: dict) -> BackwardTiles | None:
        """`derived`'s one launch: on the fed layout with its consumers (FlashAttention 3's CTA), keys first
        on the dQ launch's layout, or a key block its consumers share; None where none holds."""
        query, key = ba.arguments["query"], ba.arguments["key"]

        def fits(config: BackwardTiles) -> bool:
            return self._staged_helps(ba, config, target) and self.takes(ba, config, target)
        derive = functools.partial(_derived, least_held=_role_rows)
        # The one launch, legal where the form, the hooks, the widths, the registers and the mode allow
        # (`legal`), the first that holds (design §7: legality steps down, nothing is ranked). Queries first
        # on the layout whose consumers feed their rings, each consumer its own keys over the query stream
        # it feeds (FlashAttention 3's CTA), at the layout's consumers, whom no producer's warpgroup shares
        # the registers with (ET-FFED3-h100, the fused launch: 2.40 against 2.71 ms causal and 4.32 against
        # 5.54 full at Qwen3-8B's widths; at width 64, 64 x 64 on two consumers 0.47 ms against 128 x 64 on
        # one, 0.59): another layout than the dQ launch's, whose own fed layout the next launches name, and
        # at two consumers or more, one's softmax under the other's products, which one consumer has not.
        # Before a fed layout, the dQ launch's own where its producer's registers raise its consumers'
        # (`producer_raised`): an H100's warpgroup MMA at 24/240 (`setmaxnreg`, honoured since post43),
        # M-NA-8-gh200 20261124T011529Z, the dK/dV launch 1.06 against the fed layout's 1.24 ms at width 64
        # and 1.86 against 2.18 at 128 (Llama 3.2 1B's, Qwen3-8B's), spill-free; ET-FFED3's choice of the
        # fed layout predates the raises.
        producer = [family] if fed and producer_raised(family, target) else []
        for layout in (*producer, *(layout for layout in fed if layout.name != family.name)):
            def fed_fused(rows: int, walked: int, consumers: int, layout=layout) -> BackwardTiles:
                return BackwardTiles(walked, rows, 2, consumers, fused=True, family=layout.name)

            config = derive(layout, query.dtype, key.shape[1], query.shape[1], fed_fused, fits,
                            overlapped=True)
            if config is not None and config.consumers > 1:
                return config
        # Queries first on the dQ launch's layout at the consumers a launch may name there where they walk
        # every streamed step paired and its adds are element-wise (`dq_summed`: an A100's collective cp.async
        # ring, `Family.shared_consumers`), each its own keys, a step's dQ summed between them before one add:
        # FlashAttention 2's eight warps over a 128-key block (M-NA-9-a100: one consumer's 64-key program ran
        # 6.25% of the warps FlashAttention 2's did, 2.05 to 2.12 times its dQ adds; row 5272).
        if dq_summed(family, family.shared_consumers) and family.shared_consumers > family.consumers:
            def paired(rows: int, walked: int, consumers: int) -> BackwardTiles:
                del consumers  # the launch names its family's paired consumers
                return BackwardTiles(walked, rows, 2, family.shared_consumers, fused=True, family=family.name)

            if config := derive(family, query.dtype, key.shape[1], query.shape[1], paired, fits,
                                overlapped=False):
                return config
        # Else keys first on the dQ launch's layout, its dK/dV on the fed one where there is one, the logits'
        # queries a stream of their own, then staged a step, then on the dQ launch's layout alone.
        for shared in [*({**dkv, "staged_logits": staged} for staged in (False, True) if dkv), {}]:
            def fused(rows: int, walked: int, consumers: int, shared=shared) -> BackwardTiles:
                return BackwardTiles(walked, rows, 2, consumers, fused=True, family=family.name, **shared)

            for rows in (None, *fewer_rows(family)):
                if config := derive(family, query.dtype, key.shape[1], query.shape[1], fused, fits,
                                    overlapped=False, rows=rows, walks=_query_walk(rows)):
                    return config
        # Where no consumer's registers hold its own keys' whole dK and dV sums, the consumers share
        # one key block, each its share of the columns (FlashAttention 3's width 256: ET-COOP1-h100,
        # 0.68 ms against the two launches' 1.30 at Qwen3-8B's heads 256 wide, causal; where the sums
        # fit, sharing cost 9% at width 128 and 46% at 64, ET-COOP2-h100).
        for layout in fed:
            def sharing(rows: int, walked: int, consumers: int, layout=layout) -> BackwardTiles:
                return BackwardTiles(walked, rows, 2, consumers, fused=True, family=layout.name,
                                     shared_keys=True)

            if config := derive(layout, query.dtype, key.shape[1], query.shape[1], sharing, fits,
                                  overlapped=True):
                return config
        return None

    def _two_launches(self, ba: tokamax.BoundArguments, target: generation.GpuTarget, family: mosaic.Family,
                      dkv: dict) -> BackwardTiles | None:
        """`derived`'s two launches, the dQ launch then the keys-first dK/dV one; None where none holds."""
        query, key = ba.arguments["query"], ba.arguments["key"]

        def fits(config: BackwardTiles) -> bool:
            return self._staged_helps(ba, config, target) and self.takes(ba, config, target)

        derive = functools.partial(_derived, least_held=_role_rows)

        # The fewest dK/dV programs a key block's widths split among whose share of the sums its consumer's
        # registers hold (`BackwardTiles.columns`: a width of 256's whole dK and dV sums pass them).
        # Its logits' queries a stream of their own, else where that passes the block staged a step
        # (`BackwardTiles.staged_logits`). Where no tile of a role's rows holds the block, fewer rows a
        # consumer on a family whose warps arrange themselves by them (`fewer_rows`, `bundle_of`), each
        # launch's tiles that many rows a side (an A100's heads 256 wide: every 64-row tile's two stages of
        # queries and output cotangents beside its resident keys and values past its 166,912 bytes, the
        # two launches at 32 x 32 150,560, et-bwd256sm80).
        # Fewer rows a consumer for a call with no score hook, or on a family reading its fragments from
        # registers under a hook reading no query's place (`_fewer_hooked`): a row-indexed hook's term sliced
        # at fewer rows fails to lower (FS-A100CHECK1d's row-indexed).
        # Whole columns at fewer rows before a column split at the role's: a split recomputes each step's
        # score products once a share, where fewer rows keep the work and halve the tile (an A100's
        # bf16 backward 128 wide at 32 x 32 whole took 0.82 to 0.88 of 64 x 64 in two shares, 1 x 8192
        # causal 36.1 against 41.7 ms; M-ET-29b, W91-a100).
        # Where its family's products complete before the next issues (mma.sync's, `products_overlap`), the
        # largest tile a multiprocessor holds two programs of first, one program's warps issuing under the
        # other's products (an A100's step at Qwen3-8B's heads, causal 2 x 8192: 32 x 32 at two programs
        # 56.65 ms against 64 x 32 at one 69.06, 2 x 2048 over 128 heads 17.43 against 20.33, M-ET-56).
        # The two launches' dQ launch streams through a producer: on the call's bundle, else on the first of
        # its layouts a producer fills (consumer Blackwell's, whose mma.sync bundle feeds its consumers'
        # rings first, `bundles.MmaProduced`).
        dq = next((layout for layout in (family, *self._layouts(target.generation, query.dtype))
                   if not split_queries(layout)), None)
        if dq is None:
            return None
        residing = [] if dq.products_overlap or (not dq.budgets_registers and dq.programs(1) < 2) else [
            lambda config: fits(config) and self.resident(ba, config, target) >= 2]
        for wanted in (*residing, fits):
            for columns in (1, 2, 4):
                fewer = fewer_rows(dq) if _fewer_hooked(self.score_mod, dq) else ()
                for rows in (None, *fewer):
                    walks = _query_walk(rows)
                    for staged in (False, True):
                        def tiles(rows: int, walked: int, consumers: int, columns=columns, staged=staged
                                  ) -> BackwardTiles:
                            return BackwardTiles(walked, rows, 2, consumers, family=dq.name,
                                                 columns=columns, staged_logits=staged, **dkv)

                        if config := derive(dq, query.dtype, key.shape[1], query.shape[1], tiles, wanted,
                                              overlapped=False, walks=walks, rows=rows):
                            return config
        return None

    def _scopes(self, ba: tokamax.BoundArguments, config: BackwardTiles,
                target: generation.GpuTarget) -> list[tuple[mosaic.Family, int, resources.Scope]]:
        family = bundle_of(config, ba.arguments["query"].dtype, target=target)
        query, key, value = (ba.arguments[name] for name in ("query", "key", "value"))
        dtype = query.dtype
        width, value_width = (_width(array.shape[-1], family, dtype) for array in (query, value))
        padded_q, padded_k = _padded(query.shape[1], key.shape[1], config)
        # Each launch on its own layout (`BackwardTiles.dkv`).
        dkv_family = bundle_of(config.dkv, dtype, target=target)
        q_stages = min(config.stages, padded_q // config.block_q)
        if config.queries_first:
            # The fused launch reduces across no warps but a score hook's and fp16's running scale's
            # (`reducing`), its statistics and D its operands and its sums its products' (an A100's at
            # 64 x 64 reserves no reduction scratch, its lowering's 114,688 bytes the model's without it,
            # et-fused80).
            # The fused launch resolves its staging plan once against `target` (`fused_staging`).
            reduced = reducing(family, config.consumers, self.score_reduces, scaled=staged_parts(dtype) > 1)
            if config.shared_keys:
                allocs = _coop_allocations(family, config, width, dtype, q_stages, reduced=reduced,
                                           target=target)
            else:
                allocs = fused_staging(family, config, width, dtype, q_stages, reduced=reduced,
                                       grouped=query.shape[2] != key.shape[2], rounded=self.rounded,
                                       anchored=ba.arguments["drop"] is not None, target=target).allocations
            launches = [(family, config, allocs, config.shared_keys or fused_reduces(family, reduced),
                         False, True)]
        else:
            dkv_stages = min(config.stages, padded_q // config.dkv.block_q)
            launches = [(dkv_family, config.dkv, _dkv_allocations(
                dkv_family, config.dkv, width, value_width, dtype, stages=dkv_stages, columns=config.columns,
                tiled=any(kind in TILED for kind in self.score_kinds), params=self.score_types), True, True,
                         config.fused)]
        if not config.fused:
            launches.append((family, config, _dq_allocations(family, config, width, value_width, dtype,
                                                             min(config.stages, padded_k // config.block_k),
                                                             params=self.score_types),
                             True, False, False))
        # The fused launch's warpgroups reserve fp16's running scale's reduction scratch too, as it launches
        # (`reducing`'s `scaled`: `_fused_launch`'s and `_coop_launch`'s).
        scaled = config.fused and staged_parts(dtype) > 1
        spec = self._spec(query.shape[1], key.shape[1], ids=ba.arguments.get("query_ids") is not None,
                          initial=ba.arguments.get("initial") is not None, config=config)
        start = ba.arguments["start"]
        return [(layout, tiles.consumers, launch_scope(
            layout, allocations, reducing(layout, tiles.consumers, self.score_reduces, scaled=scaled),
            reduces=reduces, values=(None if layout.budgets_registers else (
                {"queries": ((tiles.block_q, width), dtype), "keys": ((tiles.block_k, width), dtype),
                 "values": ((tiles.block_k, value_width), dtype)},
                (((tiles.block_k, width), jnp.float32),
                 ((tiles.block_k, value_width), jnp.float32),
                 ((tiles.block_q, width), jnp.float32),
                 *step_live_tiles(tiles.block_q, tiles.block_k, backward=True, accurate=self.accurate,
                                  dropped=ba.arguments["drop"] is not None, dtype=dtype,
                                  score_mod=self.score_mod, score_types=self.score_types,
                                  spec=spec, keys_first=keys_first, fused=is_fused,
                                  runs=(start.shape[-1] if start is not None and start.ndim == 3
                                        else 1)))))))
            for layout, tiles, allocations, reduces, keys_first, is_fused in launches]

    def staged_bytes(self, ba: tokamax.BoundArguments, config: BackwardTiles,
                     target: generation.GpuTarget) -> int:
        """The larger of its launches' programs' shared memory (`_dq_allocations`, `_dkv_allocations`,
        `resources.shared_bytes`); a fused backward's dK/dV program's alone."""
        return max(resources.shared_bytes(layout, scope)
                   for layout, _, scope in self._scopes(ba, config, target))
    def registers(self, ba: tokamax.BoundArguments, config: BackwardTiles, *,
                  target: generation.GpuTarget | None = None) -> int | None:
        # None on a family that budgets none (`Family.budgets_registers`), which holds every config.
        if not bundle_of(config, ba.arguments["query"].dtype, target=target).budgets_registers:
            return None
        dq, dkv = self._launch_registers(ba, config, target=target)
        return dkv if config.fused else max(dq, dkv)

    def _launch_registers(self, ba: tokamax.BoundArguments, config: BackwardTiles, *,
                          target: generation.GpuTarget | None = None) -> tuple[int, int]:
        """Each launch's registers a consumer holds (`backward_registers`): the dQ launch's at `config` (none
        runs where `fused`), the dK/dV launch's at its own tiles (`BackwardTiles.dkv`), its consumers
        splitting a slot's queries where it runs keys first on a layout whose consumers feed their rings
        (`split_queries`), not the fused launch queries first on that layout (`_fused_launch`)."""
        arguments = ba.arguments
        registered = self.registered(arguments["query"], arguments["value"], arguments["score_params"],
                                     arguments["drop"], config)
        return (registered.at(config, dq=True, target=target),
                registered.at(config.dkv, dq=False, target=target))

    def registered(self, query, value, score_params, drop, config: BackwardTiles) -> BackwardRegisters:
        """The call's facts its launches' registers read at `config` (`BackwardRegisters`)."""
        params = tuple(zip(self.score_kinds, (kept_dtype(param.dtype) for param in score_params),
                           strict=True))
        return BackwardRegisters(query.dtype, query.shape[-1], value.shape[-1], self.accurate, params,
                                 self.score_slopes, drop is not None, config.queries_first, self.score_reads)

    def ranks_alone(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> bool:
        # A backward takes its ranking's first legal config, the assembler judging its registers and its
        # products' issue (a kernel that spills or whose warpgroup MMAs ptxas serializes refused,
        # `assembled.Binary.refused`, and the next ranked in its place): the one its register model proposed
        # ran slower on every launch measured (M-FS-15: a GH200's 0.692, an A100's six at 0.62 to 0.94).
        del ba, target
        return True



    def refusal(self, ba: tokamax.BoundArguments, family: mosaic.Family) -> Refusal | None:
        query, value = ba.arguments["query"], ba.arguments["value"]
        hooks = Hooks(score=Hook(self.score_mod,
                                 tuple(map(Side, ba.arguments["score_params"], self.score_kinds))),
                      initial=ba.arguments["initial"], drop=ba.arguments["drop"])
        if split_queries(family) and not fuses(self.accurate, bool(self.score_kinds)):
            # A device whose backward bundle's consumers feed their own rings (a B200's, consumer
            # Blackwell's) runs the one launch alone, which this form does not take (`fuses`).
            form = "accurate form" if self.accurate else "hook"
            return Refusal('family', f"the backward's {form} takes two launches, whose dQ launch streams "
                           f"through a producer {family.name} has not")
        return refused(query.dtype, query.shape[3], value.shape[3], hooks, backward=True, family=family)


def backward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None,
             dout: jax.Array, *, residuals: tuple[jax.Array, jax.Array, jax.Array, Split],
             scale: float | None = None, natural: bool = NATURAL, accurate: bool = ACCURATE,
             interpret: bool = False, rounded: bool = True) -> Gradients:
    """The template's backward on this mainloop (`MosaicBackward`, `gradients`) from a training forward's
    `residuals` (its maxima, denominators, anchor keys and split sums): D about each query's anchor from the
    split sums (`gradient.split_delta`), a dQ launch and a dK/dV launch, and `dq_finish`'s refinement between
    them, in the form `accurate` names; or the fast form's one launch (`BackwardTiles.fused`,
    `_launch_backward`), the derived default of a call whose score hook takes no parameters
    (`MosaicBackward.derived`). Every
    call form of bf16 operands, under every score hook and the initial state. The output hook is no part of
    it: a differentiated call applies `output_mod` past the kernel, where its own derivative runs, and passes
    `dout` before it. `interpret` runs it on the CPU, as a GPU of the family `generation.assumed` names. Its
    query's and key's gradients are under their prologues (`attention_route._pulled` takes them back)."""
    # The prologues taken as XLA's program before the launch (`prologues_taken`).
    query, key, hooks = prologues_taken(query, key, hooks)
    op, args = _backward_call(query, key, value, spans, hooks, dout, residuals, scale=scale, natural=natural,
                              accurate=accurate, rounded=rounded)
    return op(*args, interpret=interpret)


def backward_declined(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans,
                      hooks: Hooks | None, dout: jax.Array, *,
                      residuals: tuple[jax.Array, jax.Array, jax.Array, Split], scale: float | None = None,
                      natural: bool = NATURAL, accurate: bool = ACCURATE, rounded: bool = True,
                      certified: bool = False) -> Refusal | None:
    """Why `backward` runs no call of these arrays compiled on this device (`MosaicBackward.declined`), and
    where `certified` (the route's selected mainloop alone) as the GPU assembles its configs
    (`MosaicBackward.uncertified`), or None."""
    op, args = _backward_call(query, key, value, spans, hooks, dout, residuals, scale=scale, natural=natural,
                              accurate=accurate, rounded=rounded)
    ba, target = op.bind(*args, interpret=False), generation.gpu_target()
    return op.uncertified(ba, target) if certified else op.declined(ba, target)


def _backward_call(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None,
                   dout: jax.Array, residuals: tuple, *, scale: float | None, natural: bool,
                   accurate: bool, rounded: bool = True) -> tuple[MosaicBackward, tuple]:
    """`backward`'s op and its arguments, a dropping call's draw among them (`_forward_call`), its spans an
    interval a query (`KeySpans.per_query`)."""
    hooks, spans = Hooks() if hooks is None else hooks, spans.per_query()
    walked_initial(hooks.initial)
    op = MosaicBackward(score_mod=hooks.score.fn,
                        score_kinds=kinds_of(hooks.score.sides),
                        score_dtypes=tuple(kept_dtype(param.value.dtype) for param in hooks.score.sides),
                        score_reduces=hook_reduces(hooks), form=spans.form, keys=spans.keys,
                        scale=logit_scale(query.shape[3], scale), natural=natural,
                        accurate=accurate, rounded=rounded, score_reads=score_places(hooks),
                        score_slopes=(varying_slopes(hooks, DQ_GRADS), varying_slopes(hooks, DKV_GRADS)))
    return op, (query, key, value, dout, spans.start, spans.end, spans.query_ids, spans.key_ids,
                tuple(param.value for param in hooks.score.sides), hooks.initial, residuals, hooks.drop)
