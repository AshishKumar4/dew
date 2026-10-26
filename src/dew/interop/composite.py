"""Metadata describing a native backbone with a separately stored readout.

Source kinds describe files and constructor records. Snapshot resolution, tensor
loading and component assembly belong to `Pretrained.load`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from flax import linen as nn

from dew._records import JSON
from dew.interop.decoder_parts import Packed
from dew.interop.weights import TensorPath

if TYPE_CHECKING:
    from dew.batch import Variables
    from dew.interop.pretrained import Pretrained


@dataclass(frozen=True)
class NativeComponent:
    """A native component's config, weight folder and optional stored tensor-name prefix."""

    config: Mapping[str, object]
    config_folder: str = ""
    weights_folder: str = ""
    tensor_prefix: str = ""

    def own_tensors[T](self, shared: Mapping[str, T]) -> Mapping[str, T]:
        """This component's tensors of the shard set it shares with a readout: those under its tensor
        prefix, named without it."""
        if not self.tensor_prefix:
            return shared
        cut = len(self.tensor_prefix)
        return {name[cut:]: tensor for name, tensor in shared.items() if name.startswith(self.tensor_prefix)}


@dataclass(frozen=True)
class HeadComponent:
    """A readout's constructor and weight codec. `weight_file` None reads the backbone's shared
    shard set outside its tensor prefix. `reshape` gives flat stored maps their native head axes."""

    record: JSON
    path: TensorPath
    packed: tuple[Packed, ...] = ()
    weight_file: str | None = None
    reshape: Callable[[nn.Module, dict], dict] | None = None
    initialize: Callable[[nn.Module, Pretrained], Variables] | None = None

    def build(self, backbone: Pretrained | None, dtype: str, attention_impl: str, *,
              shapes: Mapping[str, tuple[int, ...]] | None) -> nn.Module:
        """Build the readout against the backbone's output contract."""
        raise NotImplementedError


@dataclass(frozen=True)
class CompositeSource:
    """A backbone and readout with the native model and bundle types that own their composition.
    `fields` are task metadata passed to the bundle; they contain no loader or placement policy."""

    backbone: NativeComponent | None
    head: HeadComponent
    bundle: Callable[..., Pretrained]
    tokenizer_folder: str
    fields: Mapping[str, object] = field(default_factory=dict)


class SourceKind:
    """A source's metadata codec, registered by import path in `registry.source_kinds`."""

    @staticmethod
    def describe(directory: Path) -> CompositeSource | None:
        raise NotImplementedError
