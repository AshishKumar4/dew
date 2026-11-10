"""How `Pretrained.load` reads a source, decided on its config alone before any weight downloads
(`source_route`): a registered decoder family's, a media wrapper's, DiffusionGemma's, an unregistered
decoder a registered family's convention computes (tier 2) or transformers' own forward (tier 3)."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from dew import _records
from dew.interop import decoder_parts, hf_decoders as decoders, sources, verify
from dew.interop.codecs import source_quantization

if TYPE_CHECKING:
    from dew.interop.torchax_fallback import TorchAdmission


@dataclasses.dataclass(frozen=True)
class DecoderRoute:
    """A registered decoder family's source, `record` its translated config."""

    record: decoder_parts.DecoderFields


@dataclasses.dataclass(frozen=True)
class WrapperRoute:
    """A media wrapper's source, `wrapper` its translated config."""

    wrapper: decoder_parts.WrapperFields


@dataclasses.dataclass(frozen=True)
class DiffusionGemmaRoute:
    """DiffusionGemma's source, read by `dew.interop.diffusion_gemma`."""


@dataclasses.dataclass(frozen=True)
class VerifiedRoute:
    """An unregistered decoder read through a registered family's convention, `verified` the mapping
    transformers' class checked (tier 2, `dew.interop.verify`)."""

    verified: verify.VerifiedMapping


@dataclasses.dataclass(frozen=True)
class TorchRoute:
    """An unregistered causal LM no registered convention computes, run as transformers' own forward
    by torchax (tier 3, `dew.interop.torchax_fallback`), `admitted` the decision that it runs."""

    admitted: TorchAdmission


type SourceRoute = DecoderRoute | WrapperRoute | DiffusionGemmaRoute | VerifiedRoute | TorchRoute
"""How `Pretrained.load` reads a source, decided on its config alone before any weight downloads
(`source_route`): each case carries what its reader reads."""


def source_route(config: Mapping[str, object], *,
                 verified: Callable[..., verify.VerifiedMapping | verify.Unmapped] = verify.verify_mapping,
                 admitted: Callable[[verify.Unmapped], TorchAdmission] | None = None,
                 qualified: Callable[[Mapping[str, object], verify.ProbeUnavailable], TorchAdmission]
                 | None = None,
                 remote: Path | None = None, pinned: sources.SourceReference | None = None) -> SourceRoute:
    """The route `Pretrained.load` reads `config`'s source by, or the refusal it raises for it, on the
    config alone: a storage format the codecs cannot decode, a registered family's or wrapper's own
    refusal of the config, a speculative drafter of any model_type (`decoder_parts.DrafterRefused`),
    or an unregistered model_type no registered family's convention computes and tier 3 does not run
    (`torchax_fallback.admitted`; tier 3 reads float weights only), or whose shrunken probe cannot run
    and tier 3 does not qualify on its full config (`torchax_fallback.qualified`).
    `verified` checks an unregistered decoder against transformers (`verify.verify_mapping`), and
    `admitted` decides whether tier 3 runs one it leaves unmapped (`torchax_fallback.admitted` when
    None), and `qualified` whether it runs one whose probe cannot run (`torchax_fallback.qualified` when
    None); a census over many configs passes a memo of each. `remote` is the repository whose own
    modeling code the load opted into running (`trust_remote_code`), and `pinned` the repository, Hub
    commit and folder it was fetched at, which a verification of that code is recorded under."""
    source_quantization(config)
    decoder_parts.refuse_drafter(config)
    if (route := native_route(config)) is not None:
        return route
    try:
        verdict = verified(config) if remote is None else verified(config, remote=remote, pinned=pinned)
    except verify.ProbeUnavailable as unavailable:
        # No shrunken model runs: tier 3 qualifies the source on its full config, no weights read.
        from dew.interop import torchax_fallback

        torchax_fallback.refuse_quantized(config)
        qualify = torchax_fallback.qualified if qualified is None else qualified
        return TorchRoute(qualify(config, unavailable))
    match verdict:
        case verify.VerifiedMapping() as mapping:
            return VerifiedRoute(mapping)
        case verify.Unmapped() as unmapped:
            from dew.interop import torchax_fallback

            torchax_fallback.refuse_quantized(config)
            return TorchRoute((admitted or torchax_fallback.admitted)(unmapped))


def native_route(config: Mapping[str, object]) -> DecoderRoute | WrapperRoute | DiffusionGemmaRoute | None:
    """The route of a source Dew translates itself, on its config alone: DiffusionGemma, a media wrapper
    or a registered decoder family; None for an unregistered model_type (tiers 2 and 3)."""
    family = config.get("model_type")
    if family == "diffusion_gemma":
        return DiffusionGemmaRoute()
    if media_wrapper(config):
        return WrapperRoute(decoders.translate_wrapper_config(config))
    if isinstance(family, str) and family not in decoders.families():
        return None
    return DecoderRoute(decoders.translate_config(config))


def lazy_route(config: Mapping[str, object]) -> DecoderRoute | WrapperRoute:
    """The route of a source read one tensor at a time (`native_backbone`): an unquantized registered
    decoder or its media wrapper, of a family whose preparation leaves its tensors as they are stored, or
    splits a packed one into views of it (`Packed.viewed`).
    Any other source is refused on its config, before any weight is read or any verification runs."""
    if source_quantization(config) is not None:
        raise ValueError("a lazy read takes unquantized safetensors; a whole-model codec is not bounded")
    decoder_parts.refuse_drafter(config)
    match native_route(config):
        case DecoderRoute() | WrapperRoute() as route:
            text = _records.text(route.wrapper["text_model_type"] if isinstance(route, WrapperRoute)
                                 else config["model_type"], "model_type")
            entry = decoders.family_entry(text)
            # A packed tensor split into views of itself stays unread until a row is placed; a preparation,
            # or parts interleaved in runs, copies it at translation.
            if (entry.prepare is not decoder_parts.DecoderFamily.prepare
                    or not all(packing.viewed for packing in entry.packed)):
                raise ValueError(f"a lazy read takes a family whose tensors translate as they are stored or "
                                 f"split into views of themselves; {text}'s preparation or interleaved "
                                 f"packing copies checkpoint weights")
            return route
        case _:
            raise ValueError(f"a lazy read takes a registered native decoder or its media wrapper, not "
                             f"{config.get('model_type')!r}")


def media_wrapper(config: Mapping[str, object]) -> bool:
    """Whether a source is a media wrapper: its decoder under text_config, or a bundle its decoder
    family reads whole (`DecoderFamily.wrapper`, `decoders.bundles`), its decoder at its root or nested.

    Where its model_type is a registered decoder family, the text half is
    the model unless the family reads its media bundle whole. DiffusionGemma
    nests its text config too.
    """
    family = config.get("model_type")
    return family != "diffusion_gemma" and (decoders.bundles(config) if family in decoders.families()
                                            else decoders.carries_text(config))
