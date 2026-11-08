"""A quantized source's bindings at load time, beside its weights' decoding (`dew.interop.codecs`): the
stored input scales its projections compute under (`input_quantization`), and the MXFP4 expert banks a
load serves packed (`Pretrained.load`'s `packed_experts`: `packed_banks`, `with_banks`, `bank_axes`)."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping

import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict

from dew.interop import unread
from dew.interop.codecs import source_quantization
from dew.interop.streaming import WeightLayout
from dew.nn.kernels.gemm import NvFp4
from dew.objectives.base import Variables

_log = logging.getLogger(__name__)


def input_quantization(model: nn.Module, variables: Variables, layouts: tuple[WeightLayout, ...],
                        config: Mapping[str, object], grid: Mapping[str, unread.Stored]
                        ) -> tuple[nn.Module, Variables]:
    """Write each stored input scale into the `QUANTIZATION` variables of every Linear its source
    weight binds, and return the model computing under them.

    A source weight split along its outputs (Phi-3's fused `qkv_proj`, `gate_up_proj`) binds one
    Linear a member, each reading the fused projection's one input, so each takes its scale. An
    expert stack's members read their own tokens, and a split along the inputs reads parts of the
    input, which one stored scale does not quantize; those are refused before returning a model
    with the wrong activation forward. Every stored scale must bind a Linear.
    """
    from dew.quantization import INPUT_SCALES, QUANTIZATION, InputKind, InputQuantization, InputQuantizer

    codec = source_quantization(config)
    if codec is None or codec.input_scale_dtype is None:
        return model, variables
    sites: dict[tuple[str, ...], tuple[InputKind, unread.Stored]] = {}
    consumed: set[str] = set()
    for layout in layouts:
        part = layout.name.removesuffix('.weight') + codec.input_suffix
        if part not in grid:
            continue
        # A kernel leaf is [inputs, outputs]: a split on its last axis splits the outputs.
        if (any(path[-1] != 'kernel' for path in layout.paths) or layout.expert_index is not None
                or layout.concatenate not in (None, -1)):
            raise ValueError(f"{layout.name} needs its input QDQ per expert or input part, "
                             "which one stored input scale does not quantize")
        consumed.add(part)
        scale = unread.held(grid[part])
        if scale.dtype != np.float32 or scale.size != 1:
            raise ValueError(f"{part} must be one float32 stored global scale, "
                             f"got {scale.dtype} {scale.shape}")
        kind: InputKind = 'fp8' if codec.input_kind(layout.name) == 'fp8' else 'nvfp4'
        if unread.valued(scale):
            value = float(scale.reshape(()))
            quantizer = (InputQuantizer.fp8(value) if kind == 'fp8' else
                         InputQuantizer.nvfp4(value, codec.input_scale_dtype == 'float8_e4m3fn',
                                              format=codec.input_format))
            stored = np.asarray(quantizer.scale)
        else:
            # A load without its weights holds the scale's shape and dtype; its value is checked on load.
            stored = scale.reshape(())
        for path in layout.paths:
            site = path[1:-1]
            # A fused tensor and an equal member stored beside it bind one Linear twice: they must
            # quantize its input alike, or one calibration would be silently dropped.
            held_kind, held_scale = sites.setdefault(site, (kind, stored))
            agrees = held_kind == kind and unread.equal(held_scale, stored)
            if not agrees:
                raise ValueError(f"{layout.name}'s input quantizer and another stored for "
                                 f"{'/'.join(site)} disagree; one Linear takes one")
    unbound = {name for name in grid if name.endswith(codec.input_suffix)} - consumed
    if unbound:
        raise ValueError(f"input scales {sorted(unbound)} bind no Linear; an unbound scale would "
                         "drop its QDQ")
    scheme = InputQuantization(codec.input_scale_dtype == 'float8_e4m3fn', codec.input_format,
                               tuple(sorted(('/'.join(path), kind) for path, (kind, _) in sites.items())))
    held = unflatten_dict({(*path, INPUT_SCALES[kind]): scale for path, (kind, scale) in sites.items()})
    return scheme.apply(model), {**variables, QUANTIZATION: held}


def bank_axes(table: Mapping[tuple[str, ...], tuple[str | None, ...]], variables: Variables
               ) -> dict[tuple[str, ...], tuple[str | None, ...]]:
    """`table` (a model's drawn axes, `decoder_axes`) with each packed bank's leaves (`with_banks`) at its
    dense matrix's axes as the bank lays them out (`quantization.expert_leaf_axes`), by their fields' paths
    (`dew.nn.sharding.parameter_path`), as placement reads a leaf's."""
    from dew.quantization import QUANTIZATION, expert_leaf_axes

    banks = {path: bank for path, bank in flatten_dict(
        dict(variables.get(QUANTIZATION, {})), is_leaf=lambda _, leaf: isinstance(leaf, NvFp4)).items()
        if isinstance(bank, NvFp4)}
    held = dict(table)
    for path, bank in banks.items():
        dense = table.get(("params", *path))
        if dense is None:
            continue
        names = [member.name for member in dataclasses.fields(bank)]
        for name, axes in zip(names, expert_leaf_axes(bank, dense), strict=True):
            held[(QUANTIZATION, *path, name)] = axes
    return held


def packed_banks(tensors: Mapping[str, unread.Stored], config: Mapping[str, object]
                  ) -> tuple[dict[str, unread.Stored], dict[str, NvFp4]]:
    """`tensors` with each MXFP4 expert bank a load serves packed (`Pretrained.load`'s `packed_experts`)
    taken out of the source's decoding, and those banks by their source names (`dew.nn.kernels.gemm.NvFp4`,
    built on the host, exactly the MXFP4 values: `codecs.mxfp4_as_nvfp4`). In each one's place an unread
    leaf of its decoded shape and dtype, which the decoder's translation binds as it binds a load without
    weights (`unread.Unread`), and `with_banks` replaces. A bank whose depth is no whole NVFP4 stages
    (`gemm.NVFP4_STAGE`), whose experts' group powers span past one tensor scale's reach
    (`mxfp4_tensor_powers` None), or that holds no experts (a 2-D weight's blocks), is decoded as the source
    states it, and named in the load's log, once a load; another source format is refused."""
    from dew.interop import codecs
    from dew.nn.kernels.gemm import NVFP4_STAGE

    quantization = source_quantization(config)
    if quantization is None or quantization is not codecs.MXFP4:
        raise ValueError("packed_experts serves an MXFP4 source's expert banks (gpt-oss's quant_method "
                         f"mxfp4); this source's quantization is {config.get('quantization_config')!r}")
    tensors, banks = dict(tensors), {}
    for name in quantization.names(tensors):
        blocks, scales = (tensors[name + suffix] for suffix in codecs.MXFP4_SUFFIXES)
        powers = None
        if len(blocks.shape) != 4:
            reason = "it holds no experts"
        elif blocks.shape[2] * codecs.GROUP % NVFP4_STAGE:
            reason = f"its depth {blocks.shape[2] * codecs.GROUP} is no whole {NVFP4_STAGE}-deep stages"
        else:
            powers = codecs.mxfp4_tensor_powers(blocks, scales)
            reason = "its experts' group powers span past one tensor scale's reach"
        if powers is None:
            _log.warning("packed_experts: %s stays dense: %s", name, reason)
            continue
        experts, outputs, groups, _ = blocks.shape
        banks[name] = NvFp4.of(*codecs.mxfp4_as_nvfp4(blocks, scales, powers))
        for suffix in codecs.MXFP4_SUFFIXES:
            del tensors[name + suffix]
        tensors[name] = unread.Unread((experts, groups * codecs.GROUP, outputs), np.dtype(np.float32))
    return tensors, banks


def with_banks(variables: Variables, layouts: tuple[WeightLayout, ...], banks: Mapping[str, NvFp4]
                ) -> Variables:
    """`variables` with each packed bank (`packed_banks`) in `QUANTIZATION` under the path its source name's
    layout binds, the unread leaf there taken out of `params`: a bank its layout reads as it stands, one
    path, untransposed, uncut, whole (gpt-oss's experts), the expert layers read it so
    (`dew.quantization.expert_matrix`); any other binding is refused."""
    if not banks:
        return variables
    from dew.quantization import QUANTIZATION

    params = dict(flatten_dict(dict(variables["params"])))
    held = dict(flatten_dict(dict(variables.get(QUANTIZATION, {}))))
    bound = {layout.name: layout for layout in layouts}
    for name, bank in banks.items():
        layout = bound.get(name)
        if layout is None or len(layout.paths) != 1 or layout.paths[0][0] != "params" or (
                layout.transpose or layout.concatenate is not None or layout.expert_index is not None):
            raise ValueError(f"{name} binds {None if layout is None else layout.paths}, which a packed bank "
                             "does not take: it holds a whole [experts, in, out] matrix at one path")
        path = layout.paths[0][1:]
        del params[path]
        held[path] = bank
    return {**variables, "params": unflatten_dict(params), QUANTIZATION: unflatten_dict(held)}
