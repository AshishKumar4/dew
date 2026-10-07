"""The language-model run, as one typed record.

`LMRunConfig` is what an LM recipe or script parses from its command line and
writes as `run.json` next to the checkpoints. Every run records the model,
the data, the optimizer and the trainer. A decoder run also records the
tokenizer its ids came from and the preview policy it generates with.
`TextGeneration.from_run` and `Pretrained.from_run` read those two fields back
from that file, so a run that does not record them loads as weights with no
way to turn text into ids.

The three objectives a decoder can be trained under are one field, because
they share every other field. `lm` is the next-token loss, `masked_diffusion`
is MDLM over a bidirectional model, and `block_diffusion` is DiffusionGemma's
own fine-tuning objective. `recipes/lm/train.py` adds one restriction: it
reads token files, and it builds the objective this record names.
"""

from __future__ import annotations

import dataclasses

from dew.config import DataSpec, ModelConfig, ObjectiveConfig, OptimConfig, RunConfig
from dew.data import TokenWindows
from dew.registry import objectives
from dew.sampling.text import Sampling


@dataclasses.dataclass(frozen=True)
class LMRunConfig(RunConfig):
    """A `RunConfig` plus the settings specific to language models."""

    objective: ObjectiveConfig = dataclasses.field(default_factory=lambda: ObjectiveConfig("lm"))
    """The objective and its arguments: lm, masked_diffusion (MDLM), or
    block_diffusion (the official DiffusionGemma fine-tuning objective)."""
    model: ModelConfig = dataclasses.field(
        default_factory=lambda: ModelConfig("causal_transformer", {"dtype": "bfloat16"}))
    data: DataSpec = dataclasses.field(default_factory=TokenWindows)
    optim: OptimConfig = dataclasses.field(
        default_factory=lambda: OptimConfig(
            learning_rate=6e-4,
            weight_decay=0.1, clip_grads=1.0))
    tokenizer: str = "byte"
    """The tokenizer the ids were written with: 'byte', or an HF tokenizer name."""
    sample_prompt: str = ""
    """The prompt that validation samples continue; an empty prompt continues a newline."""
    sample_tokens: int = 128
    """The number of tokens generated per validation sample; 0 logs no text."""
    sampling: Sampling = dataclasses.field(
        default_factory=lambda: Sampling(temperature=0.8, top_k=40))
    """The preview policy, recorded with the run for inference."""
    pretrained: str | None = None
    """The Hugging Face decoder to continue training from: a hub repo id,
    `repo@revision` (a branch, tag or commit), or a local directory in that
    layout. A run records a hub repo as `repo@commit`, with the commit it
    resolved to. The checkpoint sets the architecture, so of its fields
    --model.max-seq-len alone may be set."""

    def __post_init__(self) -> None:
        if self.objective.name not in (objectives.paths[name] for name in ("lm", "masked_diffusion",
                                                                             "block_diffusion")):
            raise ValueError(
                f"--objective {self.objective.name!r} is not lm, masked_diffusion or block_diffusion")
        if self.objective.name == objectives.paths["block_diffusion"]:
            if self.pretrained is None:
                raise ValueError("block_diffusion fine-tuning requires --pretrained")
            if self.trainer.quantization is not None:
                raise ValueError("block_diffusion has no quantized-training term")
