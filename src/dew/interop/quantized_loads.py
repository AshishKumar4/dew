"""A quantized source's bindings at load time, beside its weights' decoding (`dew.interop.codecs`): the
stored input scales its projections compute under (`input_quantization`), and the 4-bit weights a load
serves packed (`Pretrained.load`'s `packed_weights`: `packed_banks`, `with_banks`, `bank_axes`), an MXFP4
source's expert banks and a ModelOpt source's W4A16_NVFP4 Linears."""

from __future__ import annotations

import logging
from collections.abc import Mapping

import jax
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict

from dew.interop import unread
from dew.interop.codecs import source_quantization
from dew.interop.streaming import WeightLayout
from dew.nn.kernels.formats import NvFp4
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
    dense matrix's axes as the bank lays them out (`NvFp4.leaf_axes`, an expert bank's and a Linear's
    kernel's alike), by their fields' paths (`dew.nn.sharding.parameter_path`), as placement reads a
    leaf's."""
    from dew.quantization import QUANTIZATION

    banks = {path: bank for path, bank in flatten_dict(
        dict(variables.get(QUANTIZATION, {})), is_leaf=lambda _, leaf: isinstance(leaf, NvFp4)).items()
        if isinstance(bank, NvFp4)}
    held = dict(table)
    for path, bank in banks.items():
        dense = table.get(("params", *path))
        if dense is None:
            continue
        for name, axes in bank.leaf_axes(dense).items():
            held[(QUANTIZATION, *path, name)] = axes
    return held


def packed_banks(tensors: Mapping[str, unread.Stored], config: Mapping[str, object]
                  ) -> tuple[dict[str, unread.Stored], dict[str, NvFp4]]:
    """`tensors` with each 4-bit weight a load serves packed (`Pretrained.load`'s `packed_weights`) taken
    out of the source's decoding, and those weights by their source names as NVFP4 banks
    (`dew.nn.kernels.formats.NvFp4`, built on the host, exactly the values the source's codes state), each
    as its format reads it (`SourceQuantization.as_nvfp4`: an MXFP4 source's expert banks, ModelOpt's
    W4A16_NVFP4 Linears). In each one's place an unread leaf
    of its decoded shape and dtype, which the decoder's translation binds as it binds a load without weights
    (`unread.Unread`), and `with_banks` replaces. A weight its format reads as no NVFP4 (its reason: its
    depth no whole NVFP4 stages, `formats.unstaged`, among them) is decoded as the source states it, and
    named in the load's log with why, once a load; a source of no quantized weights is refused."""
    quantization = source_quantization(config)
    if quantization is None:
        raise ValueError("packed_weights serves a quantized source's 4-bit weights as their codes; this "
                         "source states no quantization")
    tensors, banks = dict(tensors), {}
    for name in quantization.names(tensors):
        packed = quantization.as_nvfp4(tensors, name)
        if not isinstance(packed, str):
            parts, shape = packed
            # A load that reads no values holds the bank's shapes (`jax.eval_shape`), as it holds a weight's.
            packing = (NvFp4.of if all(isinstance(part, np.ndarray) for part in parts)
                       else lambda *parts: jax.eval_shape(NvFp4.of, *parts))
            packed = packing(*parts), shape
        if isinstance(packed, str):
            _log.warning("packed_weights: %s stays dense: %s", name, packed)
            continue
        banks[name], shape = packed
        for part in (name, *quantization.partners(name)):
            tensors.pop(part, None)
        tensors[name] = unread.Unread(shape, np.dtype(np.float32))
    return tensors, banks


def _kernel_of(layout: WeightLayout, bank: NvFp4) -> bool:
    """Whether `layout` binds a Linear's kernel `[in, out]` from the source's `[out, in]` weight `bank`
    packs, transposed whole (a layer whose kernel holds more axes, a DenseGeneral's heads, reads no packed
    matrix)."""
    return (layout.transpose == (1, 0) and tuple(layout.shape) == tuple(reversed(bank.shape))
            and layout.paths[0][-1] == "kernel")


def with_banks(variables: Variables, layouts: tuple[WeightLayout, ...], banks: Mapping[str, NvFp4]
                ) -> Variables:
    """`variables` with each packed bank (`packed_banks`) in `QUANTIZATION` under the path its source name's
    layout binds, the unread leaf there taken out of `params`: one path, uncut, whole, and a bank its
    layout reads as it stands (gpt-oss's experts, `[experts, in, out]`, which the expert layers read so,
    `dew.nn.dense.packed_matrix`) or a Linear's kernel its layout transposes from the source's
    `[out, in]` (a ModelOpt weight's, packed `[in, out]` already, which the layer reads so,
    `dew.nn.dense.packed_matrix`); any other binding is refused, naming the weight, which a load
    without `packed_weights` decodes."""
    if not banks:
        return variables
    from dew.quantization import QUANTIZATION

    params = dict(flatten_dict(dict(variables["params"])))
    held = dict(flatten_dict(dict(variables.get(QUANTIZATION, {}))))
    bound = {layout.name: layout for layout in layouts}
    for name, bank in banks.items():
        layout = bound.get(name)
        dense = len(bank.scale.shape) == 0
        if layout is None or len(layout.paths) != 1 or layout.paths[0][0] != "params" or (
                layout.concatenate is not None or layout.expert_index is not None or layout.padded
                or layout.chunks) or (
                not _kernel_of(layout, bank) if dense else layout.transpose is not None):
            raise ValueError(f"{name} binds {None if layout is None else layout.paths}, which a packed bank "
                             "does not take: it holds a whole [experts, in, out] matrix, or a Linear's "
                             "kernel [in, out], at one path; load it without packed_weights")
        path = layout.paths[0][1:]
        del params[path]
        held[path] = bank
    return {**variables, "params": unflatten_dict(params), QUANTIZATION: unflatten_dict(held)}
