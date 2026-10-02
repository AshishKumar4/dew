"""The layer plan of a `CausalTransformer`: each layer's kind, its spec, and
which consecutive layers scan together.

A `LayerKind` names what a layer type builds (its mixer, and the attention
fields it overrides); `ResolvedKind` is one filled in from the model's
fields; a `LayerSpec` is one layer's plan; `scan_groups` finds the runs of
consecutive layers whose specs agree, which `nn.scan` stacks into one group.
"""

import dataclasses
from collections.abc import Mapping, Sequence

from dew.registry import mixers

from ..attention_residuals import ResidualSite
from ..mixers import MixerBase
from ..rope import RopeScaling, YarnScaling


@dataclasses.dataclass(frozen=True)
class LayerKind:
    """What the layers of one kind in the pattern do differently.

    The pattern names each layer's kind, and this is what the kind means: a
    windowed kind is the "sliding attention" of the reference configs, and
    `rope_theta` and `head_dim` are the model's unless this kind states its
    own. Rotary positions rotate every dimension of a windowed kind; Gemma 4
    puts its partial rotary on the global layers and its sliding layers
    rotate whole.

    `mixer` is this kind's token mixer, a value from the `mixers` registry;
    None rides the model's mixer. A hybrid stack names its per-layer mixers
    here, keyed by the names already in the pattern.
    """

    window: int | None = None
    """Keys a layer of this kind attends, its own included; None attends all."""
    chunk: int | None = None
    """Chunked local attention: a layer of this kind reads only the keys at
    or before each query whose position shares the query's
    `position // chunk`, MaxText's `chunk_attn_window_size` and Llama 4's
    `attention_chunk_size`. None attends all; a kind sets a window or a
    chunk, not both."""
    num_kv_heads: int | None = None
    """This kind's key/value head count; None takes the model's. Gemma 4's
    global layers keep fewer than its sliding ones (num_global_key_value_heads)."""
    rope_theta: float | None = None  # set: this kind takes this base over the model's
    rope_scaling: RopeScaling | None = None
    """This kind's llama3 ramp or its record; None rides the model's."""
    yarn: YarnScaling | None = None
    """This kind's YaRN ramp or its record; None rides the model's. OLMo 3
    scales its full-attention layers alone (configuration_olmo3.py:110-113),
    so a YaRN ramp is a kind's as much as the model's."""
    head_dim: int | None = None
    mixer: MixerBase | None = None
    """This kind's mixer value or its record; None is the model's mixer."""

    def __post_init__(self):
        # A kind's mixer and ramp arrive as values from code and as records
        # from a config, like the model's own; anything else is neither.
        if isinstance(self.mixer, Mapping):
            object.__setattr__(self, "mixer", mixers.from_record(self.mixer))
        elif self.mixer is not None and not isinstance(self.mixer, MixerBase):
            raise ValueError(
                f"a kind's mixer is a mixer value, its record, or None, "
                f"not {self.mixer!r}")
        if isinstance(self.rope_scaling, Mapping):
            object.__setattr__(self, "rope_scaling", RopeScaling(**self.rope_scaling))
        if isinstance(self.yarn, Mapping):
            object.__setattr__(self, "yarn", YarnScaling(**self.yarn))


@dataclasses.dataclass(frozen=True)
class ResolvedKind:
    """One kind of layer with the model's defaults filled in.

    `LayerKind` is what a config states, so a field it leaves to the model is
    None there. This is what the model resolved it to, so `rope_theta` and
    `head_dim` are numbers; only the window stays optional, because attending
    the whole sequence is what a kind without one does. `mixer` passes
    through: it needs no resolution, only the model's default when unset.
    """

    window: int | None
    chunk: int | None
    num_kv_heads: int
    rope_theta: float
    rope_scaling: RopeScaling | None
    yarn: YarnScaling | None
    head_dim: int
    mixer: MixerBase | None


@dataclasses.dataclass(frozen=True)
class LayerSpec:
    """What one layer of the stack is, resolved: everything its block's
    parameters and computation depend on that the layers do not share.

    Two layers with equal specs have parameters of the same shapes and run
    the same program, so a scan can run them as iterations of one body and a
    pipeline can run them at the same position of different stages.
    Everything the whole model sets (norms, the attention dials, per-layer
    inputs, AltUp) is the same for every layer and so is not repeated here.
    """

    layer_type: str
    kind: ResolvedKind
    routed: bool
    """The feed-forward routes to the mixture's experts."""
    hash_routed: bool
    """The routed feed-forward selects its experts by the token table."""
    width: int
    """The dense feed-forward width, doubled on a sharing layer when the model asks."""
    sparsity: float
    """The gaussian top-k fraction on the feed-forward gate, 0 for none."""
    kv_shared: bool
    """The layer reads its keys and values from an earlier layer's."""
    provider: int | None
    """The layer's own index when a later layer reads what it leaves in the
    kv_store, its keys and values or a CSA2 layer's publications; such a
    layer runs unrolled, since what it stashes leaves the stack's loop."""
    residual_site: ResidualSite | None
    """The layer's place among Kimi K3's blocks of attention residuals, None
    without them. It differs at every block boundary, so a scanned run never
    crosses one."""
    engram: int | None = None
    """The layer's place among the engram layers, whose bucket ids it reads
    before its attention; None for a layer without a lookup."""
    prediction_slot: int | None = None
    """The layer's place among the DSpark drafter's target layers, whose
    input streams' mean it records; None for the rest."""


def scan_groups(specs: Sequence[LayerSpec],
                bank_layers: int | None = None) -> tuple[tuple[int, int], ...]:
    """The stack as runs of layers, `(first, count)` each, in order.

    Consecutive layers with equal specs form one run, which a scan runs as
    iterations of one body; a layer with no equal neighbour is a run of one,
    which stays unrolled. The grouping is read off the specs, never written
    by hand, so a model's pattern decides what scans.

    `bank_layers` caps how many layers one run holds, which is how many a
    parameter bank stacks: a longer run splits into consecutive runs of at
    most that many layers. A host-resident bank is built and read one bank
    at a time, so the cap is what bounds the memory either costs.
    """
    if bank_layers is not None and bank_layers < 1:
        raise ValueError(f"bank_layers counts the layers one run holds, got {bank_layers}")
    groups: list[tuple[int, int]] = []
    for index, spec in enumerate(specs):
        if groups and specs[groups[-1][0]] == spec and groups[-1][1] != bank_layers:
            first, count = groups[-1]
            groups[-1] = (first, count + 1)
        else:
            groups.append((index, 1))
    return tuple(groups)


def group_name(first: int, count: int) -> str:
    """The module name of a scanned run: `layers_3_7` runs layers 3 through 7."""
    return f'layers_{first}_{first + count - 1}'


def group_layers(name: str) -> range | None:
    """The layers a stack module name runs: `layers_3_7` as range(3, 8),
    `layers_3` as range(3, 4), anything else as None. The inverse of
    `group_name`, for a reader keyed by single layers that meets the run."""
    if not name.startswith("layers_"):
        return None
    parts = name[len("layers_"):].split("_")
    if not all(part.isdigit() for part in parts) or len(parts) > 2:
        return None
    first, last = int(parts[0]), int(parts[-1])
    return range(first, last + 1)
