"""How a `CausalTransformer` runs its layer stack: the plain loop, scanned runs on one stage, or a
pipeline over the stage axis (`stack`), the runs one stage scans (`one_stage_runs`) and the view a
pipeline stacks them under (`stage_view`). The decoder's `stack` reads them here, within its scope."""

from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING

from flax import linen as nn

from dew.mesh import LayoutRefused

from ..dense import QUANTIZATION, captures_any_method
from ..host_memory import on_host
from ..hyper_connections import Carried
from ..precision import at_least_fp32
from ..sharding import pipeline_stages, stage_sites
from .decoder_stack import (
    PipelineStage,
    StackView,
    StageSplit,
    microbatch_count,
    off_stages,
    run_pipeline,
    run_stack,
    split_stages,
    stored_by_stage,
)
from .layer_names import group_layers
from .layer_plan import run_period, scan_groups

if TYPE_CHECKING:
    from .causal_transformer import CausalTransformer


def scanned() -> bool:
    """Whether a decoder's stack runs its like and periodic runs under the scan, the default, or every layer
    unrolled (`generation.chosen(layer_scan="plain")`, `Choice.LAYER_SCAN`): Qwen3.5-4B's periodic stack
    lowers unrolled to 112k operations, scanned to its period's (row 4074). How it compiles, by its
    structure alone, never a model's field."""
    from dew.nn.kernels.generation import Choice, chosen_kernel

    return chosen_kernel(Choice.LAYER_SCAN, "scanned") == "scanned"


def _frozen_staged(decoder: CausalTransformer) -> tuple[str, ...]:
    """The collections a one-stage stage-local run whose layers are all frozen (`DecoderBank.frozen`)
    keeps on their stored layers to read in place (`LAYER_ROWS`), empty otherwise."""
    sites = [site for site in stage_sites() if site.frozen]
    if not sites:
        return ()
    view = stage_view(decoder, 1)
    path = decoder.path
    if not any(site.view == view and len(site.namespace) >= len(path)
               and (not path or site.namespace[-len(path):] == path)
               for site in sites):
        return ()
    return tuple(name for name in ('params', QUANTIZATION)
                 if name in decoder.variables and view.holds(decoder.variables[name]))


def stack(decoder: CausalTransformer, x, *, train: bool, decode: bool, per_layer_input,
          attention_metadata=None, span: tuple[int, int] | None = None, prior_bank=None, kv_store=None):
    """Run the layer stack over `x` as the plain loop, as scanned runs, or as a pipeline over the stages.

    `span` runs layers [first, end) alone, as a `Loop` over a block runs its
    prelude, its block and its coda; on one stage, with no banked store.
    `prior_bank` is a looped pass's bank of its pass before's block summaries
    (`dew.nn.cross_pass`), which every layer reads alike; a pipeline takes none.
    `kv_store` is the store a span's layers read their providers' keys, values
    and publications from and leave their own in, a loop's part's (prelude,
    pass, coda: `dew.nn.backbones.looping`), filled in place; a fresh one
    where None.

    `init` draws like runs under the scan and then unstacks them, so the
    tree is the plain loop's; under a stage mesh or with `decode`, init
    runs the plain loop. Scanned (`scanned`) or over a stage axis, the stack runs
    under a `StackView`, which stacks the leaves while the loops run. The
    loops carry one residual dtype, so the stream enters in the dtype it
    settles in (`residual_dtype`). A store built bank by bank
    (`dew.inference.banks`) already holds one array per run, and the view
    leaves it alone. A pipeline runs the layers its stages repeat
    (`split_stages`) and the layers before and after them on every stage,
    and raises `LayoutRefused` for `decode` and for a banked store. With
    `residual_in_fp32` the stream enters widened, so every sublayer's output
    adds to it in fp32 whichever loop runs.
    """
    stored = off_stages(decoder, type(decoder).stack, functools.partial(stage_view, decoder))
    if stored is not None:
        return stored(decoder, x, train=train, decode=decode, per_layer_input=per_layer_input,
                      attention_metadata=attention_metadata, span=span, prior_bank=prior_bank,
                      kv_store=kv_store)
    if decoder.residual_in_fp32 and isinstance(x, Carried):
        x = x._replace(streams=x.streams.astype(at_least_fp32(x.streams.dtype)))
    elif decoder.residual_in_fp32:
        x = x.astype(at_least_fp32(x.dtype))
    stages = pipeline_stages()
    first, end = (0, decoder.num_layers) if span is None else span
    store = kv_store if kv_store is not None else ({} if decoder.provides else None)

    def unrolled():
        # Each layer run directly, in the stream's own dtype: no scan carries it.
        return run_stack(
            decoder.layers, decoder.block, decoder.specs,
            tuple((index, 1) for index in range(first, end)), x,
            train=train, decode=decode, kv_store=store,
            per_layer_input=per_layer_input, attention_metadata=attention_metadata, prior_bank=prior_bank)

    # A seeded initialization (PTQ's abstract annotation pass) reads the
    # supplied layout; the scan's init=True pass would draw over it first.
    if (decoder.is_initializing() and stages == 1 and not decode
            and not decoder.has_variable('params', f'layers_{first}')):
        drawn = _drawn_in_runs(decoder, x, (first, end), train=train, decode=decode,
                               per_layer_input=per_layer_input,
                               attention_metadata=attention_metadata, prior_bank=prior_bank, store=store)
        if drawn is not None:
            return drawn
    if decoder.is_initializing() or (stages == 1 and not scanned()):
        return unrolled()
    staged = stored_by_stage(decoder.variables)
    banked = () if staged else decoder.banked_collections(span)
    held = decoder.variables.get('cache', {})
    if decode and 'params' in banked and not any(group_layers(name) for name in held):
        # A decode over a store banked by run that opens its cache opens it banked as `init_cache` does
        # (`decoder_cache._banked_runs`), each run's layers' under the run's name, as the run's scan
        # carries it; a cache handed in a layer at a time goes back so.
        banked = (*banked, 'cache')
    if stages == 1 and not staged and not banked and span is None:
        staged = _frozen_staged(decoder)
    if stages == 1 and staged:
        if span is not None:
            raise LayoutRefused(f"a loop over layers {span} runs part of the stack, and a stage-local "
                                f"state holds whole runs; {list(staged)} arrived stacked")
        # A stage-local state on one stage holds each run as the scan reads it (`stage_view`).
        view, split = (dataclasses.replace(stage_view(decoder, 1), staged=staged),
                       StageSplit(0, decoder.num_layers, 0))
    elif stages == 1:
        groups, periods = one_stage_runs(decoder, first, end, banked=bool(banked))
        if span is not None and banked and not set(groups).issubset(decoder.groups):
            raise LayoutRefused(f"a loop over layers {span} runs part of the stack, and a store banked "
                                f"by run holds whole runs; {list(banked)} arrived banked")
        if not any(count > 1 for _, count in groups) or (
                not banked and 'streaming' not in decoder.variables and captures_any_method(decoder)):
            # No run scans, or a caller's capture of any method may keep what no scan stacks (a method's
            # non-array return, wherever its filter selects it): each layer runs as the plain stack runs it,
            # as a loop's passes do under such a capture (`looping`).
            return unrolled()
        view, split = (StackView(groups, banked=banked, periods=periods),
                       StageSplit(0, decoder.num_layers, 0))
    else:
        if prior_bank is not None:
            raise LayoutRefused("a looped pass reads the bank its pass before left in every layer, which "
                                "a pipeline over the stage axis does not carry: run the loop on one stage")
        view, split = pipelined(decoder, x, stages, decode=decode, banked=banked, span=span)
    dtype = decoder.residual_dtype(
        x, train=train, decode=decode, per_layer_input=per_layer_input,
        attention_metadata=attention_metadata, first=first, kv_store=store)
    # A carried collapse stays fp32, as the reference keeps it.
    x = x._replace(streams=x.streams.astype(dtype)) if isinstance(x, Carried) else x.astype(dtype)
    run = nn.map_variables(stacked, mapped_collections=True, trans_in_fn=view.stack,
                           trans_out_fn=view.unstack, init=False, mutable=True)
    # The layers outside the run the stages repeat (none on one stage) run on every stage, as the
    # plain loop runs them, around the pipeline; a provider among them fills the store a later one
    # reads.

    def outside(x, first: int, last: int):
        return x if first == last else run_stack(
            decoder.layers, decoder.block, decoder.specs, tuple((index, 1) for index in range(first, last)),
            x,
            train=train, decode=False, kv_store=store, per_layer_input=per_layer_input,
            attention_metadata=attention_metadata)

    x = outside(x, 0, split.prefix)
    # One store across the layers around the run and the run's own: a loop's part's, passed in, or the
    # stack's own.
    x, left = run(decoder, view, x, train, decode, per_layer_input, attention_metadata, prior_bank, store)
    _taken_back(store, left)
    return outside(x, decoder.num_layers - split.suffix, decoder.num_layers)


def _drawn_in_runs(decoder: CausalTransformer, x, span: tuple[int, int], *, train: bool, decode: bool,
                   per_layer_input, attention_metadata, prior_bank, store):
    """An initialization's layers [first, end) drawn under one scan a run where they draw alike, a layer
    whose start is its own (DEX's heads and lambda_init, an adapter's own factors: `drawn_like`) apart and
    its run stacking it after; None where no run scans. Its providers write `store`, which the layers
    after the span read (a loop's next part)."""
    first, end = span
    drawn = tuple(decoder.drawn_like(index) for index in range(decoder.num_layers))
    groups = tuple((first + start, count)
                   for start, count in scan_groups(drawn[first:end], decoder.bank_layers))
    if not any(count > 1 for _, count in groups):
        return None
    view = StackView(groups)
    run = nn.map_variables(stacked, mapped_collections=True, trans_in_fn=view.stack,
                           trans_out_fn=view.unstack, init=True, mutable=True)
    x, left = run(decoder, view, x, train, decode, per_layer_input, attention_metadata, prior_bank, store)
    _taken_back(store, left)
    return x


@nn.compact
def stacked(decoder: CausalTransformer, view: StackView, x, train: bool, decode: bool, per_layer_input,
            attention_metadata=None, prior_bank=None, kv_store=None):
    """The stack inside `view`: scanned runs, or the pipeline over stages; compact, so the stages it builds
    are the decoder's own. With the publications its providers leave (`kv_store`, the store it read, which
    leaves the transform mapping its view as an output: an argument passes in rebuilt, its writes its own),
    None where none publishes."""
    if view.stages == 1:
        store = kv_store if kv_store is not None else ({} if decoder.provides else None)
        return run_stack(
            decoder.layers, decoder.block, decoder.specs, view.groups, x,
            train=train, decode=decode, kv_store=store,
            per_layer_input=per_layer_input, attention_metadata=attention_metadata,
            banked='params' in view.banked, periods=view.periods, prior_bank=prior_bank), store
    return run_pipeline(
        decoder,
        view,
        x,
        lambda module: PipelineStage(block=module.block,
                                     specs=module.specs[view.offset:view.offset + view.per_stage],
                                     groups=view.groups, train=train, offset=view.offset, name='stages'),
        layered=per_layer_input,
        rows=attention_metadata,
    ), None


def _taken_back(store: dict | None, left: dict | None) -> None:
    """The publications a mapped run left (`stacked`), written back into the caller's `store`."""
    if store is not None and left is not None:
        store.update(left)


def pipelined(decoder: CausalTransformer, x, stages: int, *, decode: bool, banked: tuple[str, ...],
              span: tuple[int, int] | None) -> tuple[StackView, StageSplit]:
    """The view a pipeline over `stages` runs the stack's repeated run through, and how the stack
    splits around it (`split_stages`); `LayoutRefused` where the call or the store cannot pipeline."""
    if span is not None:
        raise LayoutRefused(f"a loop over layers {span} runs part of the stack, which a pipeline "
                            "over the stage axis cuts into stages of its own; run it on one stage")
    if decode:
        raise LayoutRefused(
            "decoding appends one token at a time to the cache, which no "
            "pipeline over the stage axis runs; decode outside jax.set_mesh")
    if banked:
        raise LayoutRefused(
            f"a pipeline over the stage axis stacks every stage's copy of a "
            f"layer, which a store already banked by run cannot be reshaped "
            f"into; {list(banked)} arrived banked. Place the weights per layer "
            f"for a pipeline, or run the stack whole")
    split = split_stages(decoder.specs, stages)
    hc = decoder.hyper_connections
    if hc is not None and hc.single_pass and any(
            spec.streams is not None
            for spec in decoder.specs[split.prefix:decoder.num_layers - split.suffix]):
        # The streams of the layers the stages repeat; a range the layers around them hold stays out.
        raise ValueError(
            "Single-Pass mHC carries each sublayer's collapse beside the streams, "
            "which the pipeline's microbatch buffers do not hold; run the stack whole")
    batch_axis = 1 if decoder.altup is not None else 0
    view = dataclasses.replace(
        stage_view(decoder, stages), microbatches=microbatch_count(x.shape[batch_axis]),
        # What enters the loop is read on every iteration; what the
        # loop creates (the routers' sowing) comes out per iteration.
        broadcast=tuple(name for name, tree in decoder.variables.items() if tree),
        staged=stored_by_stage(decoder.variables), batch_axis=batch_axis)
    return view, split


def one_stage_runs(decoder: CausalTransformer, first: int, end: int, *, banked: bool = False
                   ) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...]]:
    """The runs layers [first, end) scan in on one stage and their periods (`run_period`): by period over
    resident weights; the runs a store's banks hold where a step reads a layer at a time (banked,
    streamed, on the host), and like runs where a layer leaves what a later one reads (`provides`)."""
    groups = decoder.groups if (first, end) == (0, decoder.num_layers) else tuple(
        (first + start, count)
        for start, count in scan_groups(decoder.scan_keys[first:end], decoder.bank_layers))
    if scanned() and not (banked or decoder.provides or 'streaming' in decoder.variables
                                    or on_host(decoder.variables.get('params', {}))):
        groups = tuple((first + start, count)
                       for start, count in scan_groups(decoder.scan_keys[first:end], periodic=True))
    return groups, tuple(run_period(decoder.scan_keys, start, count) for start, count in groups)


def stage_view(decoder: CausalTransformer, stages: int) -> StackView:
    """How a pipeline over `stages` stacks the run of layers they repeat (`split_stages`): each
    stage's runs as its scan takes them, from the store's rows (`scan_keys`); one stage's its scan's."""
    if stages == 1:
        groups, periods = one_stage_runs(decoder, 0, decoder.num_layers)
        return StackView(groups, periods=periods)
    split = split_stages(decoder.specs, stages)
    start, count = split.prefix, split.per_stage
    groups = (scan_groups(decoder.scan_keys[start:start + count], decoder.bank_layers)
              if scanned() else tuple((index, 1) for index in range(count)))
    return StackView(groups, stages=stages, offset=start)
