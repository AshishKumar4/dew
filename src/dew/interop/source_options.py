"""What a run builds of the source it starts from (`SourceOptions`), and `Pretrained.load`'s keywords for it:
below every reader, a run's config (`dew.config`, its `--source` options) and a source's load
(`dew.inference.loading`), so a load reads them importing no trainer (row 5999)."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Annotated, Literal, Required, TypedDict

from dew.distributed.placement import Layout, MeshSpec
from dew.nn.dex import DifferentialExtension
from dew.registry import record_argument


@dataclasses.dataclass(frozen=True)
class SourceOptions:
    """What a run builds of the source it starts from: `Pretrained.load`'s keywords for its parts,
    beside where it reads them (`TrainerConfig.source_placement`), every run kind's alike
    (`RunConfig.source_keywords`)."""

    text_only: bool = False
    """A multimodal wrapper's language model alone: no tower, projector or audio encoder built, nor
    their tensors read (`Pretrained.load`'s `text_only`)."""
    prediction_depths: int | None = None
    """The first that many multi-token prediction depths, 0 for none; None the checkpoint's own
    (`Pretrained.load`'s `prediction_depths`)."""
    attention: dict[str, Literal["causal", "block_bidirectional"]] | None = None
    """How each layer kind's attention reads, by kind (`full_attention`): causal, or
    block_bidirectional, each of a row's blocks read whole, which a run's rows then lie in
    (`Pretrained.load`'s `modeling_overrides`); None the checkpoint's own:
    `--source.attention full_attention block_bidirectional`."""
    dex: Annotated[DifferentialExtension, record_argument(DifferentialExtension)] | None = None
    """DEX's differential on chosen heads (`dew.nn.dex`, `Pretrained.load`'s `dex`), its record as
    JSON: `--source.dex '{"layers": [[0, [1, 3]], [5, [0, 2]]], "anneal_steps": 1000}'`, its
    `trains` the paper's recipe unless "all"; None for none."""
    scan_layers: bool = False
    """The decoder's runs of like layers, and of layers repeating one period, under flax's scan
    (`Pretrained.load`'s `scan_layers`): a step's program grows with the period, not the depth."""
    packed_experts: bool = False
    """An MXFP4 source's routed experts served as their codes, NVFP4 banks the expert layers multiply
    as they are (`Pretrained.load`'s `packed_experts`, row 4516): its experts' memory a quarter of bf16's."""
    param_dtype: str | None = None
    """The dtype the source's parameters are stored in, its checkpoint's own as 'auto'; None stores a
    source the run trains in its numerics' `dtypes.params` and one it keeps frozen as its checkpoint
    does (`RunConfig.source_keywords`). A frozen bfloat16 base beside a float32 head and adapter is
    `--source.param-dtype bfloat16`."""
    step: int | str | None = None
    """Where the run starts from a saved Dew run (`--pretrained <run directory>`), the checkpoint it reads:
    a step, `latest`, `best`, `best:<metric>`, or None for the one the run's task loads by default
    (`dew.interop.saved_sources.released_run`). The run records the step it read (`RunConfig.pinned`)."""
    ema: bool | None = None
    """Where the run starts from a saved Dew run, whether it reads that run's EMA weights; None its
    task's own choice (`SavedRun.variables`)."""

    def overrides(self) -> Mapping[str, object] | None:
        """The `modeling_overrides` these options state, None where they state none."""
        return None if self.attention is None else {"attention": dict(self.attention)}

    def keywords(self) -> SourceBuild:
        """`Pretrained.load`'s keywords for what a run builds of its source. With `weights=False` beside
        them a load is the run's base as the run builds it, its config and its variables' shapes, read
        from the source's config and safetensors headers alone, no weight downloaded (a release of
        PyTorch pickles alone, whose shapes no header states, is read whole): a check that a base
        loads under a run's options, on a disk too small for its weights."""
        return {"text_only": self.text_only, "prediction_depths": self.prediction_depths,
                "modeling_overrides": self.overrides(), "dex": self.dex, "scan_layers": self.scan_layers,
                "packed_experts": self.packed_experts, "step": self.step, "ema": self.ema}


class SourcePlacement(TypedDict, total=False):
    """Where a run reads the source it starts from: `Pretrained.load`'s keywords
    (`TrainerConfig.source_placement`)."""

    weights: bool
    on_host: bool
    mesh: MeshSpec
    layout: Layout


class SourceBuild(TypedDict, total=False):
    """`Pretrained.load`'s keywords for what a run builds of its source (`SourceOptions.keywords`)."""

    text_only: bool
    prediction_depths: int | None
    modeling_overrides: Mapping[str, object] | None
    dex: DifferentialExtension | None
    scan_layers: bool
    packed_experts: bool
    step: int | str | None
    ema: bool | None


class SourceOptionsKeywords(SourcePlacement, SourceBuild, total=False):
    """`Pretrained.load`'s keywords for a run's source: what it builds of it (`SourceOptions`), where
    it reads it (`SourcePlacement`) and the dtype it stores its parameters in
    (`RunConfig.source_keywords`)."""

    param_dtype: Required[str]
