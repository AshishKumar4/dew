"""What every vision family shares: the tower and projector bases, the
geometry a conditioner reads off a tower, the vision-config readers, and where
a kind's tensors sit in a media checkpoint (`TowerLayout`, `ProjectorLayout`),
which each family's class holds and `dew.registry`'s kinds name.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Collection, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Protocol

from flax import linen as nn

from dew import _records
from dew.batch import Variables
from dew.interop.config_records import NativeFields
from dew.interop.decoder_parts import read_rope_theta
from dew.interop.layouts import ConfigField, ConfigLayout, ConfigPart, Route, Routes, Table
from dew.interop.unread import Stored
from dew.interop.weights import placed
from dew.registry import Configured, Record, projectors, towers

if TYPE_CHECKING:
    from dew.interop.streaming import WeightLayout

type Reader = Callable[[Mapping[str, Stored], Callable[[str], tuple[str, ...] | None], str],
                       tuple[Variables, tuple[WeightLayout, ...]]]
"""A tower's tensors into its variables in a storage dtype, with the layouts that write them back:
`dew.interop.weights.placed`, or a patch-folding ViT's `translate_vit_weights`."""


type StoredShapes = Callable[[TowerBase], Mapping[str, tuple[int, ...]]]
"""The shapes a release stores some of a component's tensors in, by their names, from the component's
fields (a tower's, a projector's), where the tensor folds or unfolds its leaf's axes as well as
transposing them: a patch convolution `[out, in, time, h, w]` its Dense map `[in * time * h * w, out]`
holds, a position grid `[h, w, d]` its table's rows."""


def written(routes: Routes, variables: Mapping[str, Mapping[str, object]], kept: Collection[str] = (),
            shapes: Mapping[str, tuple[int, ...]] | None = None) -> tuple[WeightLayout, ...]:
    """The layouts that write a component's leaves (`variables`, its collections; abstract ones will do)
    under the names `routes` give them, oriented as a reader of those names orients them (a kernel's
    `[out, in, ...]` as `[..., in, out]`, `stored_shape`), but a `kept` name as its leaf is, and a name
    `shapes` folds stored in its shape; refused (`ExportRefused`) where a leaf has no name, which a model
    of that layout cannot write."""
    import jax
    import numpy as np
    from flax.traverse_util import flatten_dict

    from dew.interop.decoder_parts import refuse_export
    from dew.interop.streaming import WeightLayout
    from dew.interop.weights import orders, stored_shape

    layouts, missing = [], []
    for collection, tree in variables.items():
        for path, leaf in flatten_dict(dict(tree)).items():
            rooted = (collection, *path)
            name = routes.write(rooted, {})
            if name is None:
                missing.append("/".join(rooted))
                continue
            if not isinstance(leaf, jax.Array | jax.ShapeDtypeStruct | np.ndarray):
                raise TypeError(f"{'/'.join(rooted)} is a {type(leaf).__name__}, not a leaf")
            rank = len(leaf.shape)
            order = None if name in kept else orders(rooted, rank)[1]
            shape = (shapes or {}).get(name) or (tuple(leaf.shape) if name in kept
                                                 else stored_shape(path, tuple(leaf.shape)))
            layouts.append(WeightLayout(name, (rooted,), tuple(shape), order))
    if missing:
        refuse_export(", ".join(missing), "no tensor of the release's layout holds it")
    return tuple(layouts)


@dataclasses.dataclass(frozen=True)
class TowerLayout:
    """Where one tower kind's tensors sit in a media checkpoint and the leaves they fill, both ways:
    `prefix` before every name, and `routes` (`dew.interop.layouts`) between a name past it and its
    leaf, rooted at its collection (`params`, and Gemma 4's `constants` beside it). `read` binds them
    (`placed`, or, for a tower that runs its patch convolution as one Dense map,
    `translate_vit_weights`), and `shapes` states the stored shapes of what it folds, by the tower's
    fields, so a tower built in Dew writes them as a read one does (`tensors`). Loading
    (`tower_variables`), routing a checkpoint's names (`hf_decoders`) and writing a tower (`tensors`,
    `interop.pretrained`) read it alike."""

    prefix: str
    routes: Routes
    read: Reader = placed
    roots: tuple[str, ...] = ()
    """The leading names of a tower stored under several roots, an empty `prefix` beside them, whose
    names `routes` read whole: MiMo-V2.6's audio tower, its encoder's and its tokenizer folder's."""
    shapes: StoredShapes | None = None
    config: ConfigLayout | None = None
    """The tower's vision_config, both ways (`dew.interop.layouts.ConfigLayout`): read into its fields
    (its reader), written from its value (`vision_config`); None for a tower whose config is read apart."""
    model_type: str | None = None
    """The vision_config's own model_type, which a written config states."""

    def vision_config(self, tower: TowerBase) -> Mapping[str, object]:
        """The vision_config `tower` is written with, which `fields` reads back as it (`config`)."""
        assert self.config is not None, "the tower reads its config apart"
        return {**({} if self.model_type is None else {"model_type": self.model_type}),
                **self.config.write(tower)}

    def path(self, name: str) -> tuple[str, ...] | None:
        """The leaf of tensor `name`, rooted at its collection; None for one the tower does not compute
        with (a buffer, SigLIP's pooling head); an unknown name raises ValueError."""
        found = self.routes.read(name, {})
        if found is None:
            raise ValueError(f"unknown tensor name {name!r}")
        return found.path

    def variables(self, hf_tensors: Mapping[str, Stored], param_dtype: str
                  ) -> tuple[Variables, tuple[WeightLayout, ...]]:
        """The tower's variables from its tensors, in the requested storage, and the layouts that
        write each tensor it placed back from them; a tensor the tower does not compute with, as
        SigLIP's pooling head, has none."""
        return self.read(hf_tensors, self.path, param_dtype)

    def tensors(self, tower: TowerBase,
                variables: Mapping[str, Mapping[str, object]]) -> tuple[WeightLayout, ...]:
        """The layouts that write `tower`'s leaves (`variables`, its collections) under its names
        past `prefix` (`written`), each folded as a read one is (`shapes`)."""
        return written(self.routes, variables, shapes=None if self.shapes is None else self.shapes(tower))



@dataclasses.dataclass(frozen=True)
class ProjectorLayout:
    """Where one projector kind's tensors sit in a media checkpoint and the leaves they fill:
    `prefix` before every name (or several, whose tensors each keep the name past their own),
    `tensors` each name's leaf path, `strip` a further prefix a name may carry (a merger's), and
    `read` the projector's own reader where it checks its tensors (`placed` otherwise)."""

    prefix: str | tuple[str, ...]
    tensors: Mapping[str, tuple[str, ...]]
    strip: str = ""
    read: Callable[..., tuple[Variables, tuple[WeightLayout, ...]]] | None = None
    kept: frozenset[str] = frozenset()
    """Names whose tensor is its leaf as it is, no Linear's transpose: a matrix the reference multiplies
    as it stores it (Gemma 3's `mm_input_projection_weight`)."""

    def path(self, name: str) -> tuple[str, ...]:
        """The leaf of tensor `name`, shared by load and source export."""
        name = name.removeprefix(self.strip) if self.strip else name
        if name not in self.tensors:
            raise ValueError(f"unknown projector tensor {name!r}")
        return self.tensors[name]

    @property
    def routes(self) -> Routes:
        """`tensors` both ways, each leaf under `params` (`dew.interop.layouts`)."""
        return Table(*(Route(name, "/".join(("params", *path))) for name, path in self.tensors.items()))

    def variables(self, hf_tensors: Mapping[str, Stored], param_dtype: str
                  ) -> tuple[Variables, tuple[WeightLayout, ...]]:
        """The projector's parameters, in the requested storage, and the layouts that write its
        tensors back from them."""
        if self.read is not None:
            return self.read(hf_tensors, param_dtype=param_dtype)
        return placed(hf_tensors, self.path, param_dtype)

    def written(self, params: Mapping[str, object]) -> tuple[WeightLayout, ...]:
        """The layouts that write a projector's parameters (`params`) under its names past `prefix`
        (the first of several), its `kept` names as their leaves are."""
        return written(self.routes, {"params": params}, self.kept)

PIXEL_VALUES_KEY = "pixel_values"
"""The batch field carrying images as the checkpoint's processor emitted them."""


class ProjectorBase(nn.Module):
    """A native projector module whose constructor is its record schema. The registry names
    its kind and `checkpoint` locates the tensors it consumes."""

    checkpoint: ClassVar[ProjectorLayout | None] = None

    @property
    def input_width(self) -> int | None:
        """Declared feature width when this projector consumes raw media without an encoder."""
        return None


def head_width(hidden_size: int, num_heads: int) -> int:
    """Each head's width where `num_heads` split a tower's `hidden_size`
    evenly, and a refusal where they leave a remainder."""
    if hidden_size % num_heads:
        raise ValueError(f"hidden_size ({hidden_size}) must split over num_heads ({num_heads})")
    return hidden_size // num_heads


@dataclasses.dataclass(frozen=True)
class TowerGeometry:
    """The shapes a conditioner has to invent to create a tower's media leaves.

    A token-only init has no batch, so `VisionConditioner` and
    `AudioConditioner` build one smallest input the tower accepts. What that
    is differs by kind: a fixed-resolution tower states its `image_size`, a
    patch-and-pool tower states the patch and block that make one, and an
    audio tower states how many mel bins a frame carries. None means this
    tower has no such field, and the conditioner uses its own default.
    """

    image_size: int | None = None
    patch_size: int | None = None
    block_size: int | None = None
    channels: int | None = None
    mel_features: int | None = None
    frames_per_token: int | None = None
    """How many input frames an audio tower encodes into one token: its convolutions' time strides
    times any frame reduction after them."""
    position_size: int | None = None
    retains_patch_slots: bool = False
    pools_frames: bool = False
    """Whether an item's frames pool into one frame's tokens (MoonViT's `tpool_patch_merger`): its tokens
    are its first frame's merge blocks, wherever it holds more."""


class TowerBase:
    """One tower kind's value: its fields, how it builds its module, and where its tensors sit in a
    media checkpoint (`checkpoint`, None for a tower read another way, as the audio towers are)."""

    checkpoint: ClassVar[TowerLayout | None] = None

    def build(self) -> nn.Module:
        """The Flax module for this value."""
        raise NotImplementedError(
            f"{type(self).__name__} names a tower kind but builds no module")

    def geometry(self) -> TowerGeometry:
        """What an initialising input has to look like for this tower.

        Nothing by default: a tower states the fields it has, and a
        conditioner reads them here instead of asking the value at runtime
        whether it carries each one.
        """
        return TowerGeometry()


_BLOCK_NORMS = {"norm1": "input_layernorm", "norm2": "post_attention_layernorm"}
"""A ViT block's two norms as `clip_layer`'s decoder block names them."""


def tensor_routes(tensors: Mapping[str, tuple[str, ...]], collection: str = "params") -> tuple[Route, ...]:
    """Each of `tensors`, a name and its leaf's path, as a route onto that leaf under `collection`."""
    return tuple(Route(name, "/".join((collection, *path))) for name, path in tensors.items())


def block_routes(maps: Mapping[tuple[str, str], tuple[str, ...]], *, prefix: str = "blocks",
                 norm_bias: bool = True, block: str = "blocks",
                 norms: Mapping[str, str] = _BLOCK_NORMS) -> Table:
    """An indexed block's tensors, both ways: `{prefix}.N.<norm>.weight|bias`
    onto `{block}_N`'s norm `norms` names (its scale and, where `norm_bias`, its bias), and
    `{prefix}.N.<module>.<linear>.weight|bias` onto the map `maps` places each (module, linear) pair at,
    its kernel and bias, under `params`."""
    routes = []
    for module, ours in norms.items():
        routes.append(Route(f"{prefix}.{{index}}.{module}.weight", f"params/{block}_{{index}}/{ours}/scale"))
        if norm_bias:
            routes.append(Route(f"{prefix}.{{index}}.{module}.bias", f"params/{block}_{{index}}/{ours}/bias"))
    for (module, linear), ours in maps.items():
        leaf = "/".join(ours)
        stored, held = f"{prefix}.{{index}}.{module}.{linear}", f"params/{block}_{{index}}/{leaf}"
        routes += [Route(f"{stored}.weight", f"{held}/kernel"), Route(f"{stored}.bias", f"{held}/bias")]
    return Table(*routes)


def unplaced(pattern: str) -> Route:
    """The names `pattern` (a regular expression) matches, which the component does not compute with: no
    leaf (a buffer, a head no reader runs, a merger its projector reads)."""
    from dew.interop.layouts import Slot

    return Route("{name}", None, MappingProxyType({"name": Slot(pattern)}))


def tower_layout(kind: str) -> TowerLayout:
    """Where tower kind `kind`'s tensors sit: its class's `checkpoint` (`dew.registry.towers`)."""
    return _held(towers[kind].checkpoint if kind in towers else None, "tower", kind)


def projector_layout(kind: str) -> ProjectorLayout:
    """Where projector kind `kind`'s tensors sit: its class's `checkpoint`
    (`dew.registry.projectors`)."""
    return _held(projectors[kind].checkpoint if kind in projectors else None, "projector", kind)


def tower_variables(kind: str, hf_tensors: Mapping[str, Stored], param_dtype: str
                    ) -> tuple[Variables, tuple[WeightLayout, ...]]:
    """One vision tower's variables, in the requested storage (`tower_layout`), and the layouts
    that write its tensors back from them."""
    return tower_layout(kind).variables(hf_tensors, param_dtype)


def projector_variables(kind: str, hf_tensors: Mapping[str, Stored], param_dtype: str
                        ) -> tuple[Variables, tuple[WeightLayout, ...]]:
    """One projector kind's parameters, in the requested storage (`projector_layout`), and the
    layouts that write its tensors back from them."""
    return projector_layout(kind).variables(hf_tensors, param_dtype)


def _held[Layout](layout: Layout | None, what: str, kind: str) -> Layout:
    if layout is None:
        raise ValueError(f"{what} kind {kind!r} has no weight map here")
    return layout


def _vision_section(hf_config: Mapping[str, object]) -> Mapping[str, object]:
    """The vision half of a wrapper config, or a bare vision config."""
    return _records.record(hf_config.get("vision_config", hf_config), "vision_config")


def merger_projector(hf_config: Mapping[str, object], text_width: int, kind: str, *,
                     bias: bool = True) -> Record:
    """The `kind` projector record of a wrapper whose vision tower ends in its own merger
    (`out_hidden_size`, `spatial_merge_size`), as Qwen 3.5's and MiMo-V2's do: the merged
    features enter the text embeddings as they are, so a merger width beside the decoder width
    refuses."""
    vision = _vision_section(hf_config)
    merged = _records.integer(vision["out_hidden_size"], "out_hidden_size")
    if merged != int(text_width):
        raise ValueError(f"out_hidden_size ({merged}) is not the decoder width ({text_width}), "
                         "the merger output enters the text embeddings as it is")
    width = _records.integer(vision["hidden_size"], "hidden_size")
    merge = _records.integer(vision.get("spatial_merge_size", 2), "spatial_merge_size")
    return {"class": kind, "fields": {"hidden_size": width, "spatial_merge_size": merge,
                                      "width": width * merge ** 2, "out_width": merged, "bias": bias}}


def _side(value: object, field: str) -> int:
    """A square image or patch size as one side: an int, or a pair with equal sides."""
    if isinstance(value, (list, tuple)):
        if len(value) != 2 or value[0] != value[1]:
            raise ValueError(
                f"{field} {list(value)!r} is not square, this trunk tiles squares")
        value = value[0]
    return _records.integer(value, field)


def trunk_geometry(vision: Mapping[str, object]) -> dict[str, int]:
    """The width, MLP width, depth and head count a ViT config states, under
    the trunk fields of the families that share the reference's names."""
    return {
        "hidden_size": _records.integer(vision["hidden_size"], "hidden_size"),
        "intermediate_size": _records.integer(vision["intermediate_size"], "intermediate_size"),
        "num_layers": _records.integer(vision["num_hidden_layers"], "num_hidden_layers"),
        "num_heads": _records.integer(vision["num_attention_heads"], "num_attention_heads"),
    }


def activation_name(config: Mapping[str, object], runs: Collection[str], default: str,
                    fields: tuple[str, ...] = ("hidden_act",)) -> str:
    """The activation the first of `fields` that `config` states names, else
    the family's `default`; a name outside `runs`, the activations its trunk
    computes, refuses."""
    name = str(next((config[field] for field in fields if field in config), default))
    if name not in runs:
        raise ValueError(f"{fields[0]} {name!r} is not expressible: this trunk runs {', '.join(runs)}")
    return name




def rope_base(vision: Mapping[str, object], default: float, *, axial: bool = False) -> float:
    """A trunk's plain rotary base: `rope_parameters`' rope_theta, else the
    flat field, else the family's `default`. A scaled rope refuses, as any
    rope does where only plain rotary has a place (`read_rope_theta`), and so
    does the decoders' 'none' alias: the towers dispatch any type but
    'default' to Transformers' rope initializers, which name no 'none'
    (modeling_gemma4.py:716-719). An `axial` tower's rotary, its grid's rows
    and columns each over half the head, is named 'axial' too: Transformers
    5.17 writes that name where earlier releases wrote 'default', and reads
    either as the one its vision rotary runs (configuration_gemma4.py's
    `default_rope_type`, modeling_gemma4.py's Gemma4VisionRotaryEmbedding)."""
    entry = _records.record(vision.get("rope_parameters") or {}, "rope_parameters")
    kind = entry.get("rope_type", entry.get("type"))
    if kind == "none":
        raise ValueError("rope_parameters: a vision tower's rotary type is 'default', not 'none'")
    if axial and kind == "axial":
        entry = {**{name: value for name, value in entry.items() if name != "type"}, "rope_type": "default"}
    theta = read_rope_theta(entry, "rope_parameters")
    return _records.number(vision.get("rope_theta", default), "rope_theta") if theta is None else theta


def integer_field(key: str, field: str | None = None, default: int | None = None, *,
                  required: bool = False) -> ConfigField:
    """A config key holding an integer field (`field`, the key's name where None), its config class's
    `default` where the config omits it (none where `required`)."""
    return (ConfigField(key, field or key, parse=_records.integer) if required
            else ConfigField(key, field or key, default, parse=_records.integer))


def number_field(key: str, field: str | None = None, default: float | None = None) -> ConfigField:
    """A config key holding a float field, its config class's `default` where the config omits it."""
    return ConfigField(key, field or key, default, parse=_records.number)


def side_field(key: str, default: int) -> ConfigField:
    """A square image or patch size, an int or an equal pair (`_side`), written as its side."""
    return ConfigField(key, key, default, parse=_side)


def activation_field(runs: Collection[str], default: str, *keys: str) -> ConfigField:
    """The trunk's activation under the first of `keys` a config states (`activation_name`), refused
    outside `runs`, written under the first."""
    def parse(value: Configured, key: str) -> str:
        return activation_name({key: value}, runs, default, (key,))
    return ConfigField(keys[0], "hidden_act", default, parse=parse, aliases=keys[1:])


class Rotary(Protocol):
    """A tower that rotates by a plain base."""

    rope_theta: float


def rope_part(default: float, *, axial: bool = False) -> ConfigPart:
    """A trunk's plain rotary base (`rope_base`), written as its `rope_parameters`."""
    def read(vision: Mapping[str, object], used: set[str],
             record: NativeFields[Rotary]) -> NativeFields[Rotary]:
        used.update(("rope_parameters", "rope_theta"))
        record["rope_theta"] = rope_base(vision, default, axial=axial)
        return record

    def write(tower: Rotary) -> Mapping[str, object]:
        return {"rope_parameters": {"rope_type": "default", "rope_theta": tower.rope_theta}}
    return ConfigPart(read, write)


TRUNK_FIELDS = (integer_field("hidden_size", required=True),
                integer_field("intermediate_size", required=True),
                integer_field("num_hidden_layers", "num_layers", required=True),
                integer_field("num_attention_heads", "num_heads", required=True))
"""The width, MLP width, depth and head count a ViT config states (`trunk_geometry`), both ways."""


