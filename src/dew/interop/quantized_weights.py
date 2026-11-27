"""A quantized source's weights as a load holds them (`QuantizedWeight`), all an export writes each back
from: decoded as the load reads them (`decoded_weights`), or served packed, undecoded (`weight_records`)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import partial

import jax
import numpy as np

from dew.interop import hf_decoders as decoders, unread
from dew.interop.codecs import SourceQuantization, source_quantization
from dew.interop.safetensors_io import tensor_specs


@dataclass(frozen=True)
class QuantizedWeight:
    """A quantized source's weight as its load holds it, all an export writes it back from: its stored
    tensors as the source's headers state them (`stored`: the weight's and its parts', which the export
    writes it back as, `SourceQuantization.requantize`), the scales and zeros it encodes against as the
    source stored them (`grid`), and how the load holds it (`decoded`: the dtype it decoded the weight
    into, its param dtype as resolved, or None for a bank it serves packed, undecoded, which an export
    decodes exactly in float32, `quantization.packed_values`)."""

    stored: Mapping[str, jax.ShapeDtypeStruct]
    grid: Mapping[str, unread.Stored]
    decoded: str | None


def weight_records(codec: SourceQuantization, tensors: Mapping[str, unread.Stored], names: Iterable[str],
                   decoded: Mapping[str, unread.Stored] | None) -> dict[str, QuantizedWeight]:
    """Each of `names`' record as `codec` stores it in `tensors`: its parts the source holds (an optional
    part, compressed-tensors' act-order index, may be absent), its grid, and the dtype of its decode in
    `decoded` (None where there is none: served packed)."""
    specs = tensor_specs(tensors)
    return {name: QuantizedWeight(
        {part: specs[part] for part in (name, *codec.partners(name)) if part in specs},
        {part: tensors[part] for part in codec.grid(name) if part in tensors},
        None if decoded is None else str(np.dtype(decoded[name].dtype))) for name in names}


def decoded_weights(tensors: Mapping[str, unread.Stored], config: Mapping[str, object], param_dtype: str, *,
                    media: bool) -> tuple[Mapping[str, unread.Stored], dict[str, QuantizedWeight], str | None,
                                          decoders.Checked]:
    """The tensors with a quantized source's weights decoded, and what `save`
    needs to write them back: each quantized weight's record (`QuantizedWeight`:
    its stored tensors, its grid and the dtype it was decoded into) and the
    scales' dtype; and the spellings checked equal before a narrowing cast,
    which the readers then take as equal (`decoders.Checked`). `media` is
    whether the loader reads the source as a media wrapper (`media_wrapper`)."""
    quantization = source_quantization(config)
    if quantization is None:
        return tensors, {}, None, decoders.NOTHING_CHECKED
    names, scale_dtype = quantization.names(tensors), quantization.scale_dtype(tensors)
    aliases: tuple[tuple[str, str], ...] = ()
    if param_dtype != "float32":
        aliases = decoders.validate_source_aliases(
            quantization.tensor_names(tensors), partial(quantization.read, tensors, set(names)),
            config, media=media)
    # A new mapping: the source's own still holds the stored parts and grid each record states.
    decoded = quantization.dequantize(tensors, param_dtype=param_dtype)
    quantized = weight_records(quantization, tensors, names, decoded)
    _share_quantized_aliases(decoded, aliases, names)
    return decoded, quantized, scale_dtype, decoders.checked_groups(aliases)


def _share_quantized_aliases(tensors: dict[str, unread.Stored], aliases: tuple[tuple[str, str], ...],
                             quantized: tuple[str, ...]) -> None:
    """Share only aliases verified on original values before codec narrowing.

    A component may mix quantized and unquantized copies, or several MTP
    copies of a tied head. Each group of spellings the checked pairs join
    (`alias_groups`) reuses its quantized member's decoded storage: no
    second cast or weight copy.
    """
    for group in decoders.alias_groups(aliases):
        representative = next((name for name in quantized if name in group), None)
        if representative is not None:
            value = tensors[representative]
            for name in group:
                tensors[name] = value
