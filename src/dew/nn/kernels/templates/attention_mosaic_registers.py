"""Register-pressure models for the attention template's Mosaic GPU launches (`attention_mosaic`):
forward, peak, prologue, and backward consumer register counts (`forward_registers`, `peaks_registers`,
`prologue_registers`, `backward_registers`, `ForwardRegisters`, `BackwardRegisters`) priced against
`resources.live_registers`."""

import dataclasses
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax.core import ShapedArray

from dew.nn.kernels import generation
from dew.nn.kernels.templates import mosaic, resources, swizzling
from dew.nn.kernels.templates.attention import TILED, Kind
from dew.nn.kernels.templates.attention_mosaic_program import (
    STATISTICS,
    BackwardTiles,
    BlockPlan,
    Tiles,
    _products,
    _width,
    bundle_of,
    forward_block_plan,
    powered_in_kernel,
    query_swizzle,
    scopes_products,
    split_queries,
    value_columns,
)


def _fused_held(logits: tuple, dtype, *, dropped: bool) -> list:
    """The tiles a fused backward launch holds beside both products and its pullback (`backward_registers`):
    a dropping call's multipliers (`algebra.drawn`); fp16's parts formed at the running scale beside the tile
    and its scaled copy, the other left operand fp32 still live (`algebra.running_parts`: a 128-wide fused
    launch 281 registers live at its peak, the scale's exp2, and 2240 bytes spilled where one tile priced it
    at 246, et-c7512). A score hook's output and slopes are its pullback's, held once: a bf16 step on mma.sync
    issues its later products in loops of their own (`Issuer.looped_products`), so none holds another's
    fragments, an overlap a second count of the slopes had covered (row 5477); an fp16 step's parts are priced
    here, and a family reading its operands from shared memory holds its narrowed weights beside the pullback
    (`backward_registers`)."""
    return (([logits] if dropped else [])
            + ([logits] * 3 if jnp.dtype(dtype) == jnp.dtype(jnp.float16) else []))


def _hook_live(family: mosaic.Family, block_q: int, block_k: int, params: tuple, *, grads: str | None = None,
               keys_first: bool = False, held: bool = False) -> tuple[list, int]:
    """A step's score-hook parameters (`params`, each its `Kind` and dtype), live beside its logits: a
    TILE's `[bq, bk]` tile, a KEY's vector in the accumulator's column layout and a QUERY's in its row
    layout (`Family.column_registers`, `row_registers`); a HEAD's scalar no tile; each in the dtype its kernel
    loads it (`kept_dtype`). With `grads`, a backward launch's gradients of them too, fp32, where it carries
    them: a KEY's across its walk in the dK/dV launch (`grads="keys"`), a QUERY's in the dQ launch
    (`grads="queries"`, `DQ_GRADS`); a TILE's gradient is the step's score cotangents through the hook's
    slope, formed in their place and stored, no tile beside them (an A100's dK/dV under a tile bias at
    64 x 64, width 64: 255 registers unspilled and none spilled at a cap of 247, where its tile beside them
    priced 280). A `keys_first` step's
    tile lays its keys along its rows, so a KEY's vector is in the row layout and a QUERY's in the column. A
    walk that holds a QUERY's vector across its steps (`held`, `held_params`) prices it apart
    (`_held_registers`). Returns the step's tiles and the vectors' registers."""
    keys, queries = ((family.row_registers, family.column_registers) if keys_first
                     else (family.column_registers, family.row_registers))
    step, vectors = [], 0
    for kind, dtype in params:
        if kind in TILED:
            # A band's tile beside its query tile's origin, a scalar (`attention_mosaic_program._param`).
            step.append((block_q, block_k, dtype))
        elif kind is Kind.KEY:
            vectors += keys(block_k, dtype) + (keys(block_k, jnp.float32) if grads == "keys" else 0)
        elif kind is Kind.QUERY and not held:
            vectors += queries(block_q, dtype) + (queries(block_q, jnp.float32) if grads == "queries" else 0)
    return step, vectors


def _held_registers(family: mosaic.Family, block_q: int, params: tuple, *, residuals: bool = False,
                    natural: bool = False) -> int:
    """The registers a forward walk holds across its steps in the accumulator's row layout: its query tile's
    score-hook parameters (`held_params`, a QUERY's vector in the dtype its kernel loads it) and its per-query
    rows, a training walk's (`residuals`, `TRAINING_ROWS`) or an inference fold walk's (`INFERENCE_ROWS`)."""
    count = TRAINING_ROWS if residuals else 0 if natural or family.scalars_whole else INFERENCE_ROWS
    rows = count * family.row_registers(block_q, jnp.float32)
    return rows + sum(family.row_registers(block_q, dtype) for kind, dtype in params if kind is Kind.QUERY)


PREDICATE_REGISTERS = 7
"""The predicate registers a GPU thread holds (P0 to P6): a step's predicates past them ptxas saves to
general registers (P2R) until the selects that read them (`_anchor_registers`)."""


def _anchor_registers(family: mosaic.Family, block_q: int, block_k: int) -> int:
    """The registers a queries-first backward step's anchor select holds (`at_anchor`,
    `attention_backward_algebra.softmax_cotangents`) across the step, from the compares that form it to the
    selects that read it: its predicates packed a bit an entry; a row pass's of them, a role's rows, pending,
    all but those its predicate registers hold (`PREDICATE_REGISTERS`); and the places it compares, the keys'
    column vector and the anchor keys' row vector, int32. The SASS of dQ under ALiBi at 64 x 64, width 64, on
    an A100: 32 `key != anchor` compares a thread packed into one word, 25 saved by P2R, restored by ISETP and
    FSEL, 43 words at its need (200); the accurate form's at 64 x 64 and width 128 on an H100 1, 24 and 18,
    its need 255; dQ at 128 x 64, width 64, under sinks 2, 27 and 20. Priced beside every phase: beside its
    products alone, the accurate form's dQ under a temperature took 64 x 64 at width 64, where it spills 232
    bytes (ET-PINBWD), its hook's slopes live across the select."""
    entries = block_q * block_k // resources.THREADS
    pending = max(0, family.role_rows * block_k // resources.THREADS - PREDICATE_REGISTERS)
    places = family.column_registers(block_k, jnp.int32) + family.row_registers(block_q, jnp.int32)
    return -(-entries // 32) + pending + places


def _mask_places(family: mosaic.Family, block_q: int, block_k: int) -> list:
    """The int32 tiles a step's masks hold where its key tile's indices do not fold (`Family.folds_indices`):
    the queries' and the keys' places and their difference, each entry's formed apart (an A100's 64 x 96 at
    width 64 under a score hook: ptxas 255 registers and 16 to 56 bytes spilled where the two places priced
    240; at 64 x 192 the mask's two subtractions and compares an entry, `folds_indices`); none where they
    fold."""
    return [] if family.folds_indices(block_k) else [(block_q, block_k, jnp.int32)] * 3


def _runs_bounds(family: mosaic.Family, block_q: int, runs: int) -> int:
    """The registers a walk of a call of `runs` key runs a query (`ForwardRegisters.runs`) holds across its
    key loop past one run's: each further run's start and end rows (`Family.row_registers`), loaded once and
    kept, its compares formed and counted per tile (`attention_algebra.seen`: one select an entry, no
    logits tile more a run, FS-6766-MAIN2). An H100's rewalk at 64 x 64, three consumers (FS-6766-REWALK2,
    main 36cfd3203): 24 loads at six runs, two rows a thread by start and end, 18 of them spilled at its 128
    registers with the float state they displaced, 8 bytes at three runs; MoBA's three runs certified its
    forward four times (row 6766)."""
    return 2 * max(runs - 1, 0) * family.row_registers(block_q, jnp.int32)


def _converted(block_k: int, *widths: int) -> list[list[tuple]]:
    """A quantized call's phases converting a slot of codes beside its sums (`Family.converted`, chunked to
    an 8-bit swizzle width `swizzling.swizzle_columns(jnp.int8, swizzling.SWIZZLE_BYTES)`, rows 5449,
    10141): its codes read in the accumulator's layout, biased to int32 and taken to fp32, a tile of each
    live at once."""
    limit = swizzling.swizzle_columns(jnp.int8, swizzling.SWIZZLE_BYTES)
    chunked = dict.fromkeys(min(width, limit) for width in widths)
    return [[(block_k, width, jnp.int32), (block_k, width, jnp.float32)] for width in chunked]


def _epilogue_registers(family: mosaic.Family, carried: list, block_q: int, value_width: int, dtype, *,
                        residuals: bool, plan: BlockPlan | None = None, masked: bool = False,
                        output: tuple[str, ...] | None = None) -> int:
    """Wide (>128) inference epilogue's live registers (chunked, or whole-width finish + masked preserve),
    a pointwise output hook's tiles past its input at its chunk (`output`, their dtypes:
    `attention_mosaic_program.pointwise_tiles`) beside the chunk's finish."""
    if residuals or not family.budgets_registers or family.places_on_role or value_width <= 128:
        return 0
    block = plan or BlockPlan(split=False, out_dtype=None, output_hook=False)
    ed = dtype if block.out_dtype is None else block.out_dtype
    w = block.epilogue_chunk(family, value_width, dtype, training=residuals, masked=masked) or value_width
    r, live = family.row_registers(block_q, jnp.float32), resources.live_registers
    hooked = ([(block_q, w, jnp.dtype(kept)) for kept in output]
              if output is not None and block.output_hook else [])
    fin = live(family, carried, [(block_q, w, jnp.float32)] * 3 + [(block_q, w, ed)] + hooked) + 2 * r
    return max(fin, live(family, [], [(block_q, value_width, ed)] * 2) + r) if masked else fin


def _role_forward_registers(family: mosaic.Family, block_q: int, block_k: int, dtype, *,
                            natural: bool) -> int:
    """Registers for a consumer of role-issued products (`Issuer.places_on_role`, tcgen05's rows): row maxima
    and denominators carried, logits fp32 from tensor memory beside narrowed weights (sums kept in TMEM,
    rescaled by column blocks; mask places are row scalars beside constant columns; ptxas held 4B 128x64
    width 256 in 196 regs, et-rows-abi, and 8B width 128 in 176 at 128x64 and 248 at 128x128, et-sm100-128).
    Natural-form exponentials (`algebra.natural_exp`) additionally hold logits and exponent bits beside the
    reduced argument (row 5535)."""
    logits, weights = (block_q, block_k, jnp.float32), (block_q, block_k, dtype)
    phases = [[logits, weights]]
    if natural:
        phases.append([logits, logits, (block_q, block_k, jnp.int32)])
    return resources.live_registers(family, [(block_q, 1, jnp.float32)] * 2, *phases)


def forward_registers(family: mosaic.Family, tiles: Tiles, value_width: int, dtype, residuals: bool, *,
                      natural: bool = False, params: tuple = (), width: int | None = None, parts: int = 1,
                      scaled: bool = False, codes: bool = False, positions: int = 0, per_span: int = 1,
                      runs: int = 1,
                      merged: bool = False, started: bool = False, dropped: bool = False,
                      weight_parts: int = 1, scaled_products: bool = False,
                      block_plan: BlockPlan | None = None, masked: bool = False,
                      output: tuple[str, ...] | None = None) -> int:
    """The registers a forward consumer holds at `tiles` (`resources.live_registers`): its weighted values'
    sums `[bq, Dv]` fp32 carried, and a step's logits fp32 beside its weights in the operands' dtype. A
    training forward (`_trained`, at its anchor known) holds the other keys' weights beside them too, in the
    operands' dtype, and the key indices int32 its anchor test compares, a thread's one per column its
    fragment holds, which its fragment's two rows share (an MMA accumulator's layout: half the tile's words;
    TWOPASS5-h100: uncounted, 64 x 128 on two consumers derived and spilled, 6.6 ms against 64 x 64's 1.06);
    its peak walk holds less. The `natural` form's exponential (`algebra.natural_exp`) holds at its peak the
    scaled logits' rounded product and its residual (`two_product`), two fp32 tiles, beside the product's
    bit split, an int32 tile, the logits themselves read entry by entry into them (ET-PINSOFT: width 256
    under a softcap at 64 x 64, ptxas 255 unspilled where three fp32 tiles priced 260; the IR's breadth-first
    live set holds the third, ET-HOOKBREAK); a training walk's then its power's integral part built in bits,
    two int32 tiles beside its weights (ET-HOOKBREAK: a natural training walk 174 at width 64, its fold's
    110 and 64 int32). A score hook's key and tile parameters are live beside the logits it reads
    (`_hook_live`), a phase of their own: the hook adds them into its logits, dead past it (ET-PINFWD:
    key-bias and tile-bias at 64 x 64 on two consumers, width 64, ptxas 168 unspilled where they priced
    beside the weights 180 and 196); a query's vector the walk reads once before its steps, held beside every
    phase (`held_params`, `_held_registers`). Where its family reads operands from registers, its products'
    fragments beside their tiles (`_products`): the logits' product of `width`-deep queries and keys, and the
    values' product of the weights. A key tile whose indices do not fold holds its masks' places
    (`_mask_places`). An fp32 call's `parts` (`MosaicForward.parts`) issue its logits' products' terms into
    one accumulator and form its weights' parts one at a time from its fp32 weights, held whole across
    them, its remainder fp32 and the part in the operands' dtype beside them
    (`attention_mosaic_forward._parted_products`: priced as 16-bit weights, the fp32 forward at 64 x 64 on
    three consumers, width 64, counted 100 where ptxas needed 152, and spilled 16 bytes at the 128 three
    consumers leave, et-fp32model). A `scaled` walk's
    (`MosaicForward.scales_in_kernel`) holds beside its step its weights at the scale and its small weights
    raised, fp32, the small ones' parts, and a part's product's own accumulator `[bq, Dv]` fp32
    (`algebra.at_scale`, `attention_mosaic_forward._accumulated`): priced as one fp32 tile and the
    accumulator, the H100's scaled forward at width 128 spilled 232 bytes (et-scaled-b2); its natural form's
    exponential raises the small ones within the step, beside its tiles and the weights at the scale (priced
    apart from them, its rewalk needed 248 of the 228 priced and spilled 44 bytes, et-scaledneed; an fp32
    call's beside its logits alone, its parts formed after it: priced beside the parts' tiles, an H100's
    scaled fp32 forward at width 64 under a softcap priced 276 and needed 184, et-sneedfp32); an fp32
    call's parts walk its weights at the scale its own fp32 weights (priced apart, its decode step's 292
    refused the 64 x 64 tile ptxas holds in 240, et-fp32decneed, and one fp32 tile fewer 212, et-parity-w); a
    family whose products read registers its values' fragments beside the parts' products (unpriced, an
    A100's at width 128 spilled 128 bytes at 255).
    """
    block_q, block_k = tiles.block_q, tiles.block_k
    if family.places_on_role:
        return _role_forward_registers(family, block_q, block_k, dtype, natural=natural)
    hooked, vectors = _hook_live(family, block_q, block_k, params, held=True)
    logits, weights = (block_q, block_k, jnp.float32), (block_q, block_k, dtype)
    bits = (block_q, block_k, jnp.int32)
    drop_tiles = ([bits, logits] if weight_parts > 1 else [bits]) if dropped else []
    # An fp32 call's weights fp32, their remainder fp32 and the part taken (`_parted_products`); a dropping
    # call holds the keep bits' hash tile beside the step's weights (and over fp16, where `weight_parts > 1`,
    # the scaled fp32 tile for `algebra.kept_parts` beside its two fp16 parts, row 9526).
    step = ([logits] * (3 if parts > 1 else 1)
            + [weights] * ((2 if residuals else 1) + weight_parts - 1) + drop_tiles)
    if residuals:
        step.append((block_q // 2, block_k, jnp.int32))
        if family.operands_in_registers:
            # Its values' block's fragments whole beside the step, where the family's products read
            # registers: an A100's training walk at 64 x 128, width 64, loads the step's query, key and
            # value blocks up front, their fragments live at once beside its logits (the liveness pass at
            # its peak, census80b), ptxas 255 registers and 48 to 56 bytes spilled where the step priced 252.
            step.append((block_k, value_width * family.rhs_copies, dtype))
    masks = _mask_places(family, block_q, block_k)
    step += masks
    if runs == 2:
        # A conservative charge of a logits tile more for two runs' walk, sized to a measured spill: an H100's
        # walk at 64 x 128, three consumers, spilled 220 bytes at two traced runs where three and four spilled
        # none (MoBA's top_k 2, its own chunk and one selected; FS-6766-ATT1, probe dynruns4 at 0461980dd).
        # The cause is inferred, not read in its SASS: LLVM taking two runs' count back to their or
        # (`zext a + zext b != 0`), whose select becomes a select a run, each reading the unmasked logits.
        step.append(logits)
    # The step's tiles the exponential's raised small weights are formed beside: a 16-bit call's step (an
    # inference walk's logits and weights and its masks' places; a training walk's spilled 224 to 248 bytes
    # at 238 of 240 priced so, et-scaledneed5); an fp32 call's parts walk its logits alone, its weights'
    # parts formed after the exponential (`_parted_products`).
    core = (list(step) if residuals else [logits, weights, *masks]) if parts == 1 else [logits]
    scaled_terms = []
    if scaled:
        step, scaled_terms = _scaled_step(family, step, logits, weights, (block_q, value_width, jnp.float32),
                                          block_k=block_k, parts=parts, dtype=dtype, residuals=residuals)
    phases = [step + ([(block_q, value_width, jnp.float32)] if scaled_products and not scaled else [])]
    if family.operands_in_registers:
        # Its weights relaid as the values' product's left operand (`MmaIssuer._left`), a copy in each warp
        # along the columns (`Family.lhs_copies`), beside the step's logits, where `_products` holds a
        # register left operand no fragment: unpriced, three of an sm103's narrow bounded walks priced 225
        # and 241 spilled 12 and 444 bytes at 255; priced, 257 (narrow-replication-grid).
        phases.append([logits, (block_q, block_k * family.lhs_copies, dtype)])
    if natural:
        # The exponential's terms (`algebra.natural_exp`); a scaled walk's raise its small weights within its
        # step (`algebra.raised_weights`), beside the step's own tiles and its weights at the scale, before
        # their parts and the parts' accumulator.
        phases.append([logits] * 2 + [bits] + ([*core, *scaled_terms] if scaled else []))
        if residuals:
            phases.append([logits, weights, weights, bits, bits, *drop_tiles])
    kept = residuals and family.keeps_sums(value_width)
    if scopes_products(family):
        phases = _scoped_phases(phases, logits, weights, (block_q, value_width, jnp.float32), kept=kept,
                                rescaled=scaled, parted=parts > 1 and not scaled,
                                added=scaled or jnp.dtype(dtype) == jnp.dtype(jnp.float16), merged=merged)
    depth = value_width if width is None else width
    products = _products(
        family, dtype,
        resources.Product(block_q, depth, block_k, beside=[logits], transposed=True),
        resources.Product(block_q, block_k, value_width, beside=[weights], held=True),
        narrow=tiles.narrow, lhs_swizzle=query_swizzle(family))
    carried = [] if kept else [(block_q, value_width, jnp.float32)]
    scored = resources.live_registers(family, carried, [logits, *hooked]) + vectors
    if codes:
        phases += _converted(block_k, depth, value_width)
    epilogue = _epilogue_registers(family, carried, block_q, value_width, dtype, residuals=residuals,
                                   plan=block_plan, masked=masked, output=output)
    held = _held_registers(family, block_q, params, residuals=residuals, natural=natural) + (
        _started_rows(family, block_q) if started else 0) + _runs_bounds(family, block_q, runs)
    walked = max(max(resources.live_registers(family, carried, *phases, *products), scored) + held, epilogue)
    return max(walked, _reach_registers(family, block_q, positions, per_span)) if positions else walked


def _scaled_step(family: mosaic.Family, step: list, logits: tuple, weights: tuple, accumulator: tuple, *,
                 block_k: int, parts: int, dtype, residuals: bool) -> tuple[list, list]:
    """A scaled walk's `step` (`forward_registers`' `scaled`) with what it holds beside the unscaled one's,
    and its scaled terms, which its natural form's exponential holds too."""
    value_width = accumulator[1]
    # Its small weights raised from their exponents, fp32, and a part's product's own accumulator beside
    # the walk's while it issues (`operands.Issuer.summed`), and its weights at the scale fp32 beside
    # them; an fp32 call's parts walk forms the small ones' parts in turn as it forms theirs
    # (`_parted_products`; its weights at the scale held as the step's fp32 weights, an H100's fp32 decode
    # step at 64 x 64 priced 212 where ptxas needed 240, et-parity-w); a 16-bit call's holds the small
    # ones' three parts (`algebra.small_parts`), a family whose products read registers its values'
    # block's fragments whole beside them, which the weights' product and each part's read.
    # tcgen05's fp32 parts walk reads each part's product back beside them (priced below), its weights at
    # the scale held no apart (a B200's at 64 x 64 over 16 columns needed 224).
    scaled_terms = [logits] if parts > 1 and scopes_products(family) else [logits, logits]
    step += [*scaled_terms, accumulator]
    if parts == 1:
        step += [weights] * (3 if jnp.dtype(dtype).itemsize == 2 else 1)
        if family.operands_in_registers and not residuals:
            step.append((block_k, value_width * family.rhs_copies, dtype))
    if scopes_products(family) and parts > 1:
        # An fp32 call's small part's product read back from tensor memory in its layout, converted to the
        # walk's and taken back by its power (`algebra.restored`) beside its parts' step (tcgen05's
        # chunks, `Tcgen05ChunksIssuer.summed`): unpriced, a B200's scaled fp32 forward at width 128
        # priced 220 at 64 x 64 over 32 columns and spilled 176 bytes at 248, and 204 over 16 where it
        # needed 224 (et-sneedfp32c). A 16-bit call's step holds no such tile beside its own (its scaled
        # forward at width 64 needed 208 at 64 x 64 on one consumer, et-sneedbf16).
        step += [accumulator] * 3
    return step, scaled_terms


def _started_rows(family: mosaic.Family, block_q: int) -> int:
    """The registers a training walk started from an initial state (sinks, a resumed call's) holds beside its
    own: the state's peak and total rows, carried across the walk (priced none, an H100's 64-wide split
    training forward under sinks or resumed priced 126 and spilled 8 bytes at the 128 its entry share leaves,
    where its plain one held)."""
    return 2 * family.row_registers(block_q, jnp.float32)


def _scoped_phases(phases: list[list[tuple]], logits: tuple, weights: tuple, sums: tuple, *, kept: bool,
                   rescaled: bool, parted: bool, added: bool, merged: bool) -> list[list[tuple]]:
    """A forward's `phases` on a family that scopes its products in tensor memory (`scopes_products`): its
    logits read back from there in their layout and converted to the walk's, both live at once
    (`Tcgen05ChunksIssuer.taken`, `summed`), a second time in each phase holding them (a B200's 64 x 64
    forward on three consumers, width 64, priced 126 of 160 and spilled 20 bytes; 64 x 128 132 and 184,
    et-h1need). Its weighted values' sums, each read in their own layout (`mosaic_sm100._read`'s WGMMA
    layout): where it keeps them there across the walk (`kept`: a training walk's, `Issuer.keeps_sums`), in
    no register but where a phase reads them: once after the walk; a `rescaled` walk's (a scaled one's, its
    rows' change of scale) within each step, beside the step's narrowed weights
    (`Tcgen05ChunksIssuer.rescaled_products`);
    beside a product of their own accumulator added into them (`added`: a scaled walk's small weights', an
    fp16 call's, `Tcgen05ChunksIssuer.added_into`). Where it carries them in registers, the weighted values'
    product read back beside them, once (priced twice, a B200's 64 x 64 forward on two consumers over 128 of
    256 columns 204, where ptxas held it in 154 with no register moved), an unscaled fp32 call's (`parted`)
    parts each from a fresh accumulator beside its step (an sm103's 64 x 64 decode priced 188 spilled 60
    bytes at 248, so 252, bounded-removal). Where an initial state is `merged` past the walk
    (`attention_mosaic_forward._started`), its sums beside the merged sums (`algebra.merge`; priced as the
    walk's alone, a B200's 64 x 64 forward on three consumers, width 128, under sinks priced 156 and spilled
    16 bytes at 160). A register family's merge takes the walk's sums in place."""
    phases = [[*phase, logits] for phase in phases]
    if kept:
        if rescaled:
            phases.append([sums, weights])
        if added:
            phases.append([sums] * 2)
    elif parted:
        phases[0] += [sums] * 2
    phases.append([sums])
    return [*phases, [sums] * 2] if merged else phases


def _reach_registers(family: mosaic.Family, block_q: int, positions: int, per_span: int) -> int:
    """The registers a packed-spans walk's reach holds before its walk (`_Program._positions`, a call of
    `positions` intervals of `per_span` queries each): two 64-bit addresses of its bounds a position its block
    holds rows of, every one loaded at once (bounded-ptxlive2: 260 address words at 128 rows, 132 at 64; an
    sm103's 128 x 64 spilled 1192 bytes, its reach alone removed needing 248, its 64 rows 184)."""
    return family.scalar_registers() + 4 * min(positions, block_q // per_span + 1)


def peaks_registers(family: mosaic.Family, tiles: Tiles, dtype, *, params: tuple, width: int) -> int:
    """The registers a peak walk's consumer holds (`attention_mosaic_forward._launch`'s `peaks`, a training
    call's first walk): it reads the keys alone for each row's largest logit, so it carries a row's maximum
    and holds a step's logits fp32 and its score hook's parameters (`_hook_live`), its masks' places where
    the keys' indices do not fold, and its logits' product's fragments where its family reads operands from
    registers; no weighted values, weights or exponentials (FS-1971-LEAST-ATTN: the forward's walk priced it
    238 where ptxas's least spill-free cap was 112). Where its family's products run past their step
    (`Family.in_flight`: warpgroup MMA's), the next step's logits accumulate beside the step's it reduces;
    tcgen05's read back beside their conversion (one priced, an H100's and a B200's at 64 x 64 width 64 37
    registers where ptxas needed 64 to 72, et-parity-u)."""
    block_q, block_k = tiles.block_q, tiles.block_k
    hooked, vectors = _hook_live(family, block_q, block_k, params, held=True)
    logits = (block_q, block_k, jnp.float32)
    # A product read back from tensor memory, converted to the walk's layout and reduced, each live at once
    # (tcgen05's chunks: two priced, a B200's peak walk at 64 x 64, width 128, priced 69 where ptxas needed
    # 80, et-parity-w).
    held = 1 + family.in_flight + 2 * scopes_products(family)
    step = [logits] * held + _mask_places(family, block_q, block_k)
    products = _products(
        family, dtype,
        resources.Product(block_q, width, block_k, beside=[logits], transposed=True),
        narrow=tiles.narrow, lhs_swizzle=query_swizzle(family))
    carried = [(block_q, 1, jnp.float32)]
    scored = resources.live_registers(family, carried, [logits, *hooked]) + vectors
    return (max(resources.live_registers(family, carried, step, *products), scored)
            + _held_registers(family, block_q, params))


@dataclasses.dataclass(frozen=True)
class ForwardRegisters:
    """A forward call's facts its launches' consumer registers read (`forward_registers`, `peaks_registers`):
    operand dtype, query/value widths, training split sums (`residuals`), natural logits, score hook
    parameters, fp32 parts, in-kernel scaling, quantized codes, packed positions, initial-state merge/start,
    dropout/weight parts, query prologue, prescale flags, output dtype, output-hook presence and its pointwise
    tiles."""
    dtype: Any
    width: int
    value_width: int
    residuals: bool
    natural: bool
    params: tuple
    parts: int
    scaled: bool
    codes: bool = False
    positions: int = 0
    per_span: int = 1
    runs: int = 1
    """The key runs a query's visible keys are the union of (`MosaicForward.registered`'s `start`): a walk
    of more than one holds each further run's bounds across its key loop (`_runs_bounds`)."""
    merged: bool = False
    started: bool = False
    dropped: bool = False
    weight_parts: int = 1
    scaled_products: bool = False
    prologue: tuple | None = None
    """The query prologue its consumers take on their query tiles (`MosaicForward.query_mod`), with its
    parameters' kinds and dtypes, `(hook, kinds, dtypes)`; None where it has none."""
    prescaled: bool = False
    rewalk_prescaled: bool = False
    out_dtype: Any = None
    has_output_mod: bool = False
    output: tuple | None = None
    """The output hook its consumers apply as they finish, `(hook, kinds, dtypes)` (`MosaicForward`'s), whose
    program at each candidate's own output tile decides and prices its epilogue
    (`attention_mosaic_program.pointwise_tiles`, as the walk replays it there, `sliced_hook`); None where it
    has none."""

    def at(self, family: mosaic.Family, tiles: Tiles, *, peaks: bool = False,
           columns: int | None = None, masked: bool = False) -> int | None:
        """A consumer's registers at `tiles` on `family`: its walk's over the values' columns its launch
        walks (`columns`, a block of them; the tiles' own by default), or its peak walk's (`peaks`), and its
        query prologue's before either walks (`prologue_registers`); None on a family that budgets none
        (`Family.budgets_registers`)."""
        if not family.budgets_registers:
            return None
        width = _width(self.width, family, self.dtype)
        in_kernel_prescale = self.prescaled or (tiles.natural_form and self.rewalk_prescaled)
        if self.prologue is not None:
            before = prologue_registers(family, tiles.block_q, width, self.dtype, *self.prologue)
        elif (in_kernel_prescale and powered_in_kernel(family, self.dtype, parts=self.parts)
              and query_swizzle(family) is None):
            before = resources.live_registers(family, [], [(tiles.block_q, width, self.dtype)])
        else:
            before = 0
        if peaks:
            return max(before, peaks_registers(family, tiles, self.dtype, params=self.params, width=width))
        padded_v = _width(self.value_width, family, self.dtype)
        value_width = value_columns(padded_v, columns or tiles.columns)
        # The forward's one plan (`forward_block_plan`): a hook its launch keeps traced at this candidate's
        # own output tile alone, as its walk takes it (`sliced_hook`).
        plan = forward_block_plan(padded_v, columns or tiles.columns, self.out_dtype,
                                  has_output_mod=self.has_output_mod, training=self.residuals,
                                  output=self.output, rows=tiles.block_q)
        hooked = plan.tile_dtypes if plan.pointwise else None
        return max(before, forward_registers(
            family, tiles, value_width, self.dtype, self.residuals,
            natural=tiles.natural_form or self.natural, params=self.params, width=width, parts=self.parts,
            scaled=self.scaled, codes=self.codes,
            positions=self.positions, per_span=self.per_span, runs=self.runs, merged=self.merged,
            started=self.started,
            dropped=self.dropped, weight_parts=self.weight_parts,
            scaled_products=self.scaled_products, block_plan=plan, masked=masked, output=hooked))


def prologue_registers(family: mosaic.Family, block_q: int, width: int, dtype, hook: Callable,
                       kinds: tuple[Kind, ...], dtypes: tuple[str, ...]) -> int:
    """The registers a consumer's query prologue holds before its walk
    (`attention_mosaic_forward._prologued`): every tile its traced program makes at a query tile of
    `block_q` rows `width` wide in `dtype`, the tile it reads and its parameters' among them, priced live at
    once, its program's order being ptxas's (a light prologue's live tiles alone priced the 4B call's
    inference forward's natural rewalk within a B200 consumer's 240 registers, which spilled 48 bytes,
    et-a6-h; Qwen3's QK-norm and RoPE at 64 x 256 held more at every config, et-a6-d)."""
    params = [jax.ShapeDtypeStruct(() if kind is Kind.HEAD else (block_q, width), kept)
              for kind, kept in zip(kinds, dtypes, strict=True)]
    closed = jax.make_jaxpr(hook)(jax.ShapeDtypeStruct((block_q, width), dtype),
                                  jax.ShapeDtypeStruct((block_q, 1), jnp.int32), *params)
    values = [var.aval for var in (*closed.jaxpr.invars,
                                   *(var for equation in closed.jaxpr.eqns for var in equation.outvars))]
    tiles = [(aval.shape[0], aval.shape[1], aval.dtype) for aval in values
             if isinstance(aval, ShapedArray) and len(aval.shape) == 2 and aval.shape[0] == block_q]
    return resources.live_registers(family, [], tiles)


DQ_ROWS = 7
"""The per-query rows a dQ step holds across its walk (`backward_registers`)."""


TRAINING_ROWS = 5
"""The per-query rows a training forward's walk holds at its peak (`forward_registers`): the maximum, the
denominator and the anchor it carries, and its anchor reduction's shuffled and loaded rows (the corrected IR
live set, ET-HOOKBREAK at the held term: 3 carried rows and 2 at the peak's shuffle beside its tiles, 174 and
176 at 64 x 64 and width 64 where the model without them held 164 and 166)."""


INFERENCE_ROWS = 5
"""The per-query rows an inference fold walk holds at its peak (`forward_registers`), its denominator's
reduction (`attention_algebra.plain_step`'s `weighed`): its denominator and new maximum carried, the factor
its sums and denominator take, and the reduction's partial and shuffled rows. Priced none, an H100's two
consumers at 64 x 128, width 128, priced 160 and spilled 48 bytes at their 168 (the base sync's training
step's output walk, 2ee20e39c); ptxas's least spill-free cap ran 9 to 24 past the model at every width and
key tile (FS-FOLD-CAUSAL, FS-FOLD-FULL). Its natural form takes none: its exponential's terms price it at or
past ptxas's need without them (the same census: 160 where 152 held, 224 where 200); nor a family whose
scalars are a measured whole (`Family.scalars_whole`, mma.sync's), which holds them: an A100's inference walk
at width 128 held unnarrowed in 247 of its 255 (et-a100-infer3)."""


def _dq_registers(family: mosaic.Family, config: BackwardTiles, width: int, value_width: int, dtype,
                  products: list, held: list, vectors: int, *, accurate: bool, slopes: int) -> int:
    """`backward_registers`' dQ consumer's: its dQ sums (and the accurate form's keyed sums) carried, its
    per-query rows, a step's two products, then its score cotangents, one part of them at a time, its
    hook's parameters beside its pullback (`held`) and its products' fragments where they read registers."""
    block_q, block_k = config.block_q, config.block_k
    logits = (block_q, block_k, jnp.float32)
    # dQ's anchor select priced as its SASS holds it (`_anchor_registers`), within the family's held
    # scalars where they are a measured whole (`Family.scalars_whole`); its hook's parameters and
    # gradients beside its pullback, not its products: they stream in and out of its entries (dQ under
    # a tile bias at 64 x 64, width 128: S's and dP's accumulators the only float tiles at its peak).
    anchor = _anchor_registers(family, block_q, block_k) if not family.scalars_whole else 0
    products = products[:2]
    pullback = products[1:] + [(block_q, block_k, jnp.float32)] * slopes + held
    carried = [(block_q, width // config.columns, jnp.float32)] * (2 if accurate else 1)
    cotangents = [(block_q, block_k, jnp.float32), (block_q, block_k, dtype)]
    # Its per-query rows in the accumulator's row layout (`Family.row_registers`), live across the walk:
    # the maximum, the prior total, the anchor, D, the anchor key and the residual and mass sums
    # (DQROWPROOF64/128: seven rows, D64 at 128 queries past the budget).
    rows = DQ_ROWS * family.row_registers(block_q, jnp.float32)
    # Its products' fragments where they read registers (`_products`): the logits', the output
    # cotangents' beside them, and dQ's of the score cotangents in registers. The keys are read by the
    # logits and dQ, so stay live while the intervening product reads the values.
    fragments = _products(
        family, dtype,
        resources.Product(block_q, width, block_k, beside=[logits], transposed=True, source="keys"),
        resources.Product(block_q, value_width, block_k,
                          beside=[logits] * 2, transposed=True, source="values"),
        resources.Product(block_q, block_k, width // config.columns,
                          beside=cotangents[1:], held=True, source="keys"),
        narrow=config.narrow)
    # A score hook's slopes and parameters live until its pullback takes them (`gradient.pulled`).
    hooked = [logits] * slopes + held
    if accurate and family.operands_in_registers:
        # The accurate form's parts (`algebra.formed_parts`) issued against one right operand, the keys,
        # whose fragments its family loads once and holds across them, two of its depth's loads ahead,
        # beside the cotangents, the keyed sums' weights fp32 and the part (an A100's dQ at 64 x 64:
        # one LDSM's registers read by four HMMAs over three parts, 218 to 221 live at its peak), and a
        # hook's pullback beside them (an A100's at 64 x 64 and width 128 under alibi spilled 16 bytes).
        right = resources.operand_fragments(family, block_q, block_k, width // config.columns,
                                            resources.OperandRead(dtype, parts=0),
                                            resources.OperandRead(dtype, retained=True))
        fragments.append([cotangents[0], (block_q, block_k, jnp.float32), cotangents[1], *right, *hooked])
    if jnp.dtype(dtype) == jnp.dtype(jnp.float16):
        # fp16's parts scaled (`product.Scaled`): each part's product its own accumulator, restored and
        # added into dQ's (`operands.Issuer.summed`), beside the cotangents and both parts, formed first.
        fragments.append([cotangents[0], cotangents[1], cotangents[1],
                          (block_q, width // config.columns, jnp.float32)])
    if accurate and hooked:
        # The accurate form's cotangents fp32 formed beside the keyed sums' weights fp32
        # (`softmax_cotangents`) and the hook's pullback: an H100's dQ at 64 x 64, width 64, spilled 128
        # bytes under a softcap and 208 under alibi, a tile a slope past its model (acchook3).
        fragments.append([cotangents[0], logits, *hooked])
    phases = resources.live_registers(family, carried, products, pullback, cotangents, *fragments)
    return phases + anchor + rows + vectors


def backward_registers(family: mosaic.Family, config: BackwardTiles, width: int, value_width: int, dtype, *,
                        accurate: bool, dq: bool, split: bool = False, params: tuple = (),
                        slopes: int = 0, dropped: bool = False,
                        hooked: tuple[bool, bool] = (False, False)) -> int:
    """The registers a backward launch's consumer holds at `config` (`resources.live_registers`): a dQ
    consumer's dQ sums `[bq, D]` fp32 carried (and the accurate form's keyed sums) and its per-query rows,
    beside a step's logits and output cotangents' products fp32, then its score cotangents fp32 and one part
    of them in the operands' dtype at a time (`algebra.formed_parts`; ptxas held the accurate form's three at
    width 128 in 255 registers, unspilled, where all three would be 262); a dK/dV consumer's dK and dV sums
    `[bk, D]`, `[bk, Dv]` fp32 carried, beside a step's two products (a fused launch's queries first, its dQ
    tile `[bq, D]` fp32 too) and its per-query statistics, each of a consumer's share of a slot's queries
    where its consumers `split` them (`split_queries`), else of the slot's; every sum its share of the
    `columns`. A
    queries-first step's masked tile holds its predicates beside its products (`algebra.seen` of its places'
    int tiles, `_coords`, a byte an entry), from its logits' mask to its cotangents' (ptxas: the fused
    launch at 128 x 64 and width 64 spills 60 bytes, its model 252 without them; 64 x 64 at width 128 holds
    255 unspilled, 242); a keys-first step's places are vectors broadcast at each use. A score hook's
    parameters and their gradients the launch forms are live beside its products (`_hook_live`), the places
    it reads as values (`hooked`, its queries' and its keys', `score_places`) in a keys-first launch, and its
    pullback's `slopes` tiles fp32 that vary (`varying_slopes`: its output and its slopes along its
    differentiated inputs) beside its output cotangents' product once the logits' is spent, a phase of their
    own (dQ under a temperature at 128 x 64, width 64, spilled 4316 bytes, its SASS's spills their float
    chains, FMUL, FADD and FSEL, the products' and dQ's accumulators live beside them)."""
    block_q, block_k = config.block_q, config.block_k
    if family.places_on_role:
        # A compute warpgroup of a walk whose products a role issues (`placed.Tcgen05Walked`): its resident
        # side's row a thread (keys first a key, queries first a query), a part of a step's walked tile at
        # once (`attention_mosaic_dkv.PART`, `attention_mosaic_dq.PART`), its scores and their cotangents
        # read from tensor memory fp32 beside their statistics (keys first every thread the part's whole, a
        # column's layout; queries first its row's, carried with its rows' sums) and its weights and score
        # cotangents formed fp32 (its kept sums in tensor memory, not its registers).
        part, held = 16, (block_q if dq else block_k)
        carried = [(held, 1, jnp.float32)] * 8 if dq else [(held, part, jnp.float32)] * 2
        return resources.live_registers(family, carried, [(held, part, jnp.float32)] * 4)
    logits = (block_q, block_k, jnp.float32)
    held, vectors = _hook_live(family, block_q, block_k, params, grads="queries" if dq else "keys")
    products = [logits] * 2 + [(block_q, block_k, jnp.int8)] + held
    pullback = products[1:] + [logits] * slopes
    if dq:
        return _dq_registers(family, config, width, value_width, dtype, products, held, vectors,
                             accurate=accurate, slopes=slopes)
    if config.shared_keys:
        # Shared keys (`attention_mosaic_coop`): a consumer's share of the head's columns of dK and dV
        # carried, beside a step's two products over its share of the keys and its share of the dQ tile,
        # its statistics row vectors.
        share = config.consumers
        carried = [(block_k, width // share, jnp.float32), (block_k, value_width // share, jnp.float32)]
        products = [(block_q, block_k // share, jnp.float32)] * 2 + [(block_q, block_k // share, jnp.int8)]
        if jnp.dtype(dtype) == jnp.dtype(jnp.float16):
            # fp16's operands as two parts at the running scale (`attention_mosaic_coop._sixteen_step`): an
            # operand's scaled tile and both its parts beside the products.
            products += [(block_q, block_k // share, jnp.float32), *[(block_q, block_k // share, dtype)] * 2]
        statistics = len(STATISTICS) * family.row_registers(block_q, jnp.float32)
        # A score hook's slopes held beside both products to its pullback (as the fused launch's below).
        slope_tiles = [(block_q, block_k // share, jnp.float32)] * slopes
        return resources.live_registers(family, carried, products + slope_tiles,
                                        [(block_q, width // share, jnp.float32)]) + statistics
    carried = [(block_k, width // config.columns, jnp.float32), (block_k, value_width // config.columns,
                                                                 jnp.float32)]
    if config.fused and not split:
        # Queries first (`attention_mosaic_fused._fused_launch`), on either layout: its statistics row
        # vectors.
        statistics = len(STATISTICS) * family.row_registers(block_q, jnp.float32)
        dq_tile = [(block_q, width, jnp.float32)]
        # Its products' fragments where they read registers (`_products`): the logits' and the output
        # cotangents', then dV's, dK's and dQ's of the weights and score cotangents it stages.
        fragments = _products(
            family, dtype,
            resources.Product(block_q, width, block_k, beside=[logits], transposed=True),
            resources.Product(block_q, value_width, block_k, beside=[logits] * 2, transposed=True),
            resources.Product(block_k, block_q, value_width, beside=[logits]),
            resources.Product(block_k, block_q, width, beside=[logits]),
            resources.Product(block_q, block_k, width, beside=dq_tile),
            narrow=config.narrow)
        if not family.operands_in_registers:
            # A family reading its operands from shared memory holds the weights narrowed for dV's product
            # beside the pullback until they are staged: wgmma's at 64 x 64, width 128, under a softcap held
            # six packed words of them at its peak, where it spilled 104 bytes (softcap2, row 5477).
            fragments.append([*pullback, (block_q, block_k, dtype)])
        held = _fused_held(logits, dtype, dropped=dropped)
        return resources.live_registers(family, carried, products + held, pullback + held, dq_tile,
                                        *fragments) + statistics
    return _keys_first_registers(family, config, width, value_width, dtype, carried,
                                 block_q // config.consumers if split else block_q, accurate=accurate,
                                 params=params, slopes=slopes, hooked=hooked)


def _keys_first_registers(family: mosaic.Family, config: BackwardTiles, width: int, value_width: int, dtype,
                          carried: list, queries: int, *, accurate: bool, params: tuple, slopes: int,
                          hooked: tuple[bool, bool]) -> int:
    """`backward_registers`' keys-first dK/dV consumer at `config`, its `queries` a slot's or its share's:
    its `carried` sums beside a step's two products and its hook's live values and places, its statistics in
    its tiles' column layout, and its products' fragments."""
    block_k = config.block_k
    held, vectors = _hook_live(family, queries, block_k, params, grads="keys", keys_first=True)
    # The places a score hook reads (`score_places`), each a vector beside its products until the hook
    # reads it: its mask folds them into its compares, a hook takes them as values (a B200's row-indexed
    # dK/dV at 64 x 64, width 128, over two column shares: 24 bytes spilled at a model of 233, its query
    # places' 16 words among ptxas's live registers, FINAL14).
    vectors += (family.column_registers(queries, jnp.int32) if hooked[0] else 0)
    vectors += (family.row_registers(block_k, jnp.int32) if hooked[1] else 0)
    # Its two products' accumulators, and where its family's products run past their step
    # (`Family.in_flight`: warpgroup MMA's) the next step's beside them in a launch of its own (an H100's
    # keys-first dK/dV at 64 x 64 over two consumers' query shares priced 108 to 124 registers where ptxas
    # needed 136 at every hook, et-parity-w); a fused launch's step waits on its dQ adds between them.
    ahead = 0 if config.fused else family.in_flight
    step = [(queries, block_k, jnp.float32)] * (2 * (1 + ahead)) + held
    pullback = step[1:] + [(queries, block_k, jnp.float32)] * slopes
    # The keys-first step's statistics in its tiles' column layout (`Family.column_registers`), each read
    # where it is used, from global memory (`STATISTICS`), one's column vector live at a time (an A100's
    # keys-first launch at 64 x 64, width 128: 17 to 18 words at its peak, where all five's are 80); a fused
    # launch's dQ tile `[queries, D]` fp32 beside its score cotangents.
    statistics = family.column_registers(queries, jnp.float32)
    added = [[step[0], (queries, width, jnp.float32)]] if config.fused else []
    # Its products' fragments where they read registers (`_products`), keys first: the logits' and the
    # output cotangents' of the keys against the queries over the whole head, which every column share
    # recomputes (`attention_mosaic_dkv`'s products read the slots whole), then dV's and dK's of the weights
    # and the score cotangents in registers, its share of their columns.
    weights = (block_k, queries, dtype)
    columns = config.columns
    dq_product = (resources.Product(queries, block_k, width,
                                    beside=[step[0], (queries, width, jnp.float32)]),) if config.fused else ()
    fragments = _products(
        family, dtype,
        resources.Product(block_k, width, queries, beside=step[:1], transposed=True),
        resources.Product(block_k, value_width, queries, beside=step, transposed=True),
        resources.Product(block_k, queries, value_width // columns, beside=[weights], held=True),
        resources.Product(block_k, queries, width // columns, beside=[weights], held=True),
        *dq_product,
        narrow=config.narrow)
    if accurate and family.operands_in_registers:
        # The accurate form's dK parts issued against one right operand, the queries, held at its whole depth
        # across them, beside the cotangents fp32, dV's weights and the part (an A100's dK/dV at 64 x 64:
        # one LDSM's registers read by HMMAs of all three parts, 64 words of them at its peak; one part:
        # ptxas's least 240 fell to 192).
        right = (queries, width // columns * family.rhs_copies, dtype)
        # A score hook's parameters' key gradients beside them (an A100's at 64 x 64 under a key bias spilled
        # 32 bytes, acchook3).
        fragments.append([step[0], weights, weights, right, *held])
    if jnp.dtype(dtype) == jnp.dtype(jnp.float16):
        # fp16's dV and dK parts scaled (`product.Scaled`): each product its own accumulator, restored and
        # added into dV's or dK's (`operands.Issuer.summed`), beside the cotangents and both parts, formed
        # first.
        restore = (block_k, max(width, value_width) // columns, jnp.float32)
        fragments.append([step[0], weights, weights, restore])
    phases = resources.live_registers(family, carried, step, pullback, *added, *fragments)
    return phases + statistics + vectors


@dataclasses.dataclass(frozen=True)
class BackwardRegisters:
    """A backward call's facts its launches' consumer registers read (`backward_registers`), the one
    computation its derivation plans with (`MosaicBackward._launch_registers`) and each launch states its
    kernel's register model by as it lowers (`Family.kernel`'s `registers`): its operands' dtype, its query
    and value heads' widths, its form (`accurate`), its score hook's parameters' kinds and kept dtypes and
    the tiles its pullback varies in the dQ launch and the dK/dV launch (`slopes`), whether it drops
    entries, whether its config walks queries first (`BackwardTiles.queries_first`), and the places its
    score hook reads, its queries' and its keys' (`hooked`, `score_places`)."""
    dtype: Any
    width: int
    value_width: int
    accurate: bool
    params: tuple
    slopes: tuple[int, int]
    dropped: bool
    queries_first: bool
    hooked: tuple[bool, bool]

    def at(self, tiles: BackwardTiles, *, dq: bool,
           target: generation.GpuTarget | None = None) -> int:
        """A consumer's registers in the dQ launch (`dq`) or the dK/dV launch at `tiles`, on the bundle they
        run on (`bundle_of`, row 11987)."""
        family = bundle_of(tiles, self.dtype, target=target)
        width, value_width = (_width(each, family, self.dtype) for each in (self.width, self.value_width))
        return backward_registers(family, tiles, width, value_width, self.dtype, accurate=self.accurate,
                                  dq=dq, split=split_queries(family) and not self.queries_first,
                                  params=self.params, slopes=self.slopes[0 if dq else 1],
                                  dropped=self.dropped, hooked=self.hooked)

