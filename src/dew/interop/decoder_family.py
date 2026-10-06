"""The `DecoderFamily` record each registered family fills in.

It holds a family's config translation, tensor paths and export vocabulary;
`decoder_families.ENTRIES` holds one per family.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from dew.interop.decoder_config import DecoderFields, WrapperFields
from dew.interop.decoder_export import _dense_decoder_weights
from dew.interop.decoder_paths import Packed, _dew_path, _hf_name
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.mixers import MixerBase
from dew.nn.text_encoders import check_tree


class WeightPreparer(Protocol):
    """Checkpoint storage transforms, with translated geometry where layout needs it."""

    def __call__(self, tensors: Mapping[str, np.ndarray],
                 config: Mapping[str, object] | None = None, /) -> Mapping[str, np.ndarray]: ...


@dataclass(frozen=True)
class DecoderFamily:
    """Holds one family's config, tensor paths and export vocabulary.

    A dew config carries no provenance tag, so `matches` reads the fields
    the backbone would be built from and names the family whose reference
    computes them; the same fields come from a built model at export and
    from a config dict at weight translation. Entries are ordered from the
    most specific layout to the plain dense decoder, and the first match
    wins.
    """

    model_types: tuple[str, ...]
    translate_config: Callable[[Mapping[str, object], set[str]], DecoderFields]
    matches: Callable[[CausalTransformer], bool]
    export_model_type: str
    architecture: str
    export_fields: Callable[[CausalTransformer], Mapping[str, object]]
    preserve_source_layout: bool = field(kw_only=True)
    """Bind source tensor names/config for export instead of deriving them from the model."""
    weight_path: Callable[[str, Mapping[str, object]], tuple[str, ...] | None] = _dew_path
    export_path: Callable[[str, Mapping[str, object]], str | None] = _hf_name
    export_weights: Callable[[CausalTransformer, Mapping[str, object], Mapping[str, object]],
                             Mapping[str, np.ndarray]] = _dense_decoder_weights
    """Whole-variable encoder; the families with one add their checks or
    storage to the shared writer (`_decoder_tensors`)."""
    sandwich_norms: bool = False
    prepare: WeightPreparer = field(default=lambda tensors, _config=None: dict(tensors))
    """Storage the path map cannot read as stored that no `packed` entry
    describes: GPT-2's causal buffers, GPT-NeoX's head-interleaved qkv,
    DeepSeek V4's grouped output projection. A quantized format is undone
    before this, by `Pretrained.load`, which records what it undid for the
    export."""
    packed: tuple[Packed, ...] = ()
    """Source tensors that hold several the path map reads, split on load and
    packed again on export: fused experts and GPT-2's Conv1D projections."""
    tied_head_names: tuple[str, str] = ('lm_head.weight', 'model.embed_tokens.weight')
    """The head and the embedding a tied checkpoint stores two copies of, in
    the source's own names. A wrapper nests both under its language model."""
    zero_padded: tuple[str, ...] = ()
    """Suffixes of the 1-D source tensors a checkpoint stores longer than their
    leaf, zeros past it: Kimi K3's KDA `A_log`. `prepare` checks and
    trims the tail; export writes the zeros back (`WeightLayout.padded`)."""
    constants: Callable[[Path, Mapping[str, object]], Mapping[str, object]] = lambda directory, record: {}
    """The `constants` entries a family derives from its source directory beside
    the tensors (`with_constants`), which no export writes back: V4.1's engram token map."""
    wrapper: Callable[[Mapping[str, object], set[str]], WrapperFields] | None = None
    """Reads a media bundle released under the family's own model_type, which
    keeps the decoder's tensors unprefixed and `wrapper_projector_names` beside them."""
    wrapper_projector_names: tuple[str, ...] = ()

    def prepare_weights(self, tensors: Mapping[str, np.ndarray],
                        config: Mapping[str, object] | None = None) -> Mapping[str, np.ndarray]:
        """The checkpoint's tensors as the path map reads them: `prepare`'s, with
        every `packed` tensor split into its parts."""
        prepared = self.prepare(tensors, config)
        if not self.packed:
            return prepared
        split: dict[str, np.ndarray] = {}
        for name, tensor in prepared.items():
            packing = self.packing(name)
            split.update({name: tensor} if packing is None else packing.split(name, tensor, config))
        return split

    def packing(self, name: str) -> Packed | None:
        """The `packed` entry the source tensor `name` is, if any."""
        return next((packing for packing in self.packed if name.endswith(packing.name)), None)


def _kind_mixers(fields: CausalTransformer) -> list[MixerBase]:
    """Return the mixer values of the model's named kinds."""
    return [kind.mixer for kind in (fields.kinds or {}).values() if kind.mixer is not None]


def _every_layer_windowed(fields: CausalTransformer) -> bool:
    windows = {name: kind.window for name, kind in (fields.kinds or {}).items()}
    return all(windows.get(layer) is not None
               for layer in fields.layer_types or ('full_attention',))


def _check_tree(variables: Mapping[str, object], model) -> None:
    """`check_tree` against a decoder, whose `init` reads one row of token ids."""
    check_tree(variables, model, np.zeros((1, 2), np.int32))
