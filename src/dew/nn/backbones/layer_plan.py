"""The layer plan of a `CausalTransformer`: each layer's kind, its spec, and which layers scan together.

A `LayerKind` names what a layer type builds: its mixer, and the attention
fields it overrides. A `ResolvedKind` is a kind with the model's fields
filled in, and a `LayerSpec` is one layer's plan. `scan_groups` finds the
runs of consecutive layers whose specs are equal, and `nn.scan` stacks each
run into one group.
"""

import dataclasses
from collections.abc import Mapping, Sequence

from dew.registry import mixers

from ..attention_residuals import ResidualSite
from ..mixers import MixerBase
from ..rope import LongRopeScaling, RopeScaling, YarnScaling, rope_scaling_from_record


@dataclasses.dataclass(frozen=True)
class LayerKind:
    """What the layers of one kind in the pattern do differently.

    The pattern names each layer's kind, and this record says what the kind
    means. A windowed kind is the "sliding attention" of the reference
    configs. `rope_theta` and `head_dim` are the model's unless the kind sets
    its own. Rotary positions rotate every dimension of a windowed kind
    unless it states its own `partial_rotary_factor`: Gemma 4 puts its
    partial rotary on the global layers, and its sliding layers rotate every
    dimension, where MiMo-V2-Flash rotates a third of every head.

    `mixer` is the kind's token mixer, a value from the `mixers` registry;
    None uses the model's mixer. A hybrid stack names its per-layer mixers
    here, keyed by the kind names already in the pattern.
    """

    window: int | None = None
    """The number of keys a causal layer of this kind attends to, its own
    included. None attends to all keys. A bidirectional layer reads its whole
    row, as DiffusionGemma's decoder does, unless `bidirectional_window`."""
    bidirectional_window: bool = False
    """Whether a bidirectional layer of this kind keeps `window` on both sides
    of a query, reading the keys within window - 1 positions of it, as
    ModernBERT's local layers do (modeling_modernbert.py, |q - k| <=
    local_attention // 2). A causal layer ignores it."""
    chunk: int | None = None
    """The chunk size of chunked local attention. A layer of this kind reads
    only the keys at or before each query whose position has the same
    `position // chunk` as the query. This is MaxText's
    `chunk_attn_window_size` and Llama 4's `attention_chunk_size`. None
    attends to all keys. A kind sets a window or a chunk, not both."""
    num_kv_heads: int | None = None
    """This kind's key/value head count; None takes the model's. Gemma 4's
    global layers have fewer than its sliding ones (num_global_key_value_heads)."""
    rope_theta: float | None = None  # set: this kind takes this base over the model's
    rope_scaling: RopeScaling | LongRopeScaling | None = None
    """This kind's llama3 ramp, or its record. None uses the model's."""
    # OLMo 3's per-kind YaRN: configuration_olmo3.py:110-113.
    yarn: YarnScaling | None = None
    """This kind's YaRN ramp, or its record. None uses the model's. OLMo 3
    scales only its full-attention layers, so a YaRN ramp can belong to a
    kind as well as to the model."""
    head_dim: int | None = None
    partial_rotary_factor: float | None = None
    """The fraction of this kind's head the rotary turns; None takes the
    model's on a kind that attends the whole sequence and every dimension on
    a windowed one."""
    sinks: bool | None = None
    """Whether this kind's attention adds a learned sink logit per head
    (`dew.nn.attention_sinks`); None takes the model's `attention_sinks`.
    MiMo-V2-Flash sinks its windowed layers alone."""
    mixer: MixerBase | None = None
    """This kind's mixer value, or its record. None uses the model's mixer."""

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
            object.__setattr__(self, "rope_scaling", rope_scaling_from_record(self.rope_scaling))
        if isinstance(self.yarn, Mapping):
            object.__setattr__(self, "yarn", YarnScaling(**self.yarn))


@dataclasses.dataclass(frozen=True)
class ResolvedKind:
    """One kind of layer with the model's defaults filled in.

    `LayerKind` is what a config states, so a field the config leaves to the
    model is None there. `ResolvedKind` is what the model resolved it to, so
    `num_kv_heads`, `rope_theta` and `head_dim` are numbers. The window, the
    chunk and the rotary ramps stay optional, since a kind may have none.
    `mixer` is passed through unchanged, and the model's default applies when
    it is unset.
    """

    window: int | None
    bidirectional_window: bool
    chunk: int | None
    num_kv_heads: int
    rope_theta: float
    rope_scaling: RopeScaling | LongRopeScaling | None
    yarn: YarnScaling | None
    head_dim: int
    partial_rotary_factor: float | None
    sinks: bool
    mixer: MixerBase | None


@dataclasses.dataclass(frozen=True)
class LayerSpec:
    """One layer's resolved plan.

    The spec holds everything that the block's parameters and computation
    depend on and that differs between layers. Two layers with equal specs
    have parameters of the same shapes and run the same program. A scan can
    therefore run them as iterations of one body, and a pipeline can run them
    at the same position of different stages. Settings of the whole model
    (norms, the attention settings, per-layer inputs, AltUp) are the same for
    every layer, so they are not repeated here.
    """

    layer_type: str
    kind: ResolvedKind
    routed: bool
    """Whether the feed-forward routes to the mixture's experts."""
    hash_routed: bool
    """Whether the routed feed-forward selects its experts by the token table."""
    width: int
    """The dense feed-forward width, doubled on a sharing layer when the model asks for it."""
    sparsity: float
    """The gaussian top-k fraction on the feed-forward gate, 0 for none."""
    kv_shared: bool
    """Whether the layer reads its keys and values from an earlier layer."""
    provider: int | None
    """The layer's own index when a later layer reads what this layer leaves
    in the kv_store: its keys and values, or a CSA2 layer's publications.
    Such a layer runs unrolled, because what it stores has to leave the
    stack's loop. None for the other layers."""
    residual_site: ResidualSite | None
    """The layer's place among Kimi K3's blocks of attention residuals, or None
    without them. It differs at every block boundary, so a scanned run never
    crosses one."""
    engram: int | None = None
    """The layer's place among the engram layers, whose bucket ids it reads
    before its attention; None for a layer without a lookup."""
    prediction_slot: int | None = None
    """The layer's place among the DSpark drafter's target layers, whose
    input streams' mean it records; None for the rest."""
    attention_norm: bool = True
    """Whether the block norms its attention input; ModernBERT's first block does not."""


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


__all__ = ["LayerKind", "LayerSpec", "ResolvedKind"]
