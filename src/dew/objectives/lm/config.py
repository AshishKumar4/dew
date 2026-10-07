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

from dew.config import DataSpec, ModelConfig, OptimConfig, RunConfig
from dew.data import TokenWindows
from dew.registry import objectives
from dew.sampling.text import Sampling

from .objective import IndexerTraining


@dataclasses.dataclass(frozen=True)
class LMRunConfig(RunConfig):
    """A `RunConfig` plus the settings specific to language models."""

    objective: str = "lm"
    """The loss convention: lm, masked_diffusion (MDLM), or block_diffusion
    (the official DiffusionGemma fine-tuning objective)."""
    model: ModelConfig = dataclasses.field(
        default_factory=lambda: ModelConfig("causal_transformer"))
    data: DataSpec = dataclasses.field(default_factory=TokenWindows)
    optim: OptimConfig = dataclasses.field(
        default_factory=lambda: OptimConfig(
            learning_rate=6e-4,
            weight_decay=0.1, clip_grads=1.0))
    tokenizer: str = "byte"
    """The tokenizer the ids were written with: 'byte', or an HF tokenizer name."""
    ema_decay: float | None = None
    """The decay of an EMA copy that validation and previews read. None keeps
    no copy, and 1.0 keeps a frozen copy."""
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
    resolved to. The checkpoint sets the architecture, so --model.config may
    then give max_seq_len alone."""
    balance_rate: float | None = None
    """How far a sparse run moves each router's balancing bias against its
    load every step (DeepSeek's aux-loss-free balancing). It needs a mixture
    with bias=True; unset leaves the bias unchanged."""
    aux_loss_alpha: float | None = None
    """The weight of the expert balance loss (`LMObjective.aux_loss_alpha`).
    With --no-seq-aux it is the Switch loss over the step's routed positions,
    which is lm-engine's `router_aux_loss_coef`. Unset adds no balance loss."""
    seq_aux: bool = True
    """Whether the balance loss is computed within each sequence (DeepSeek V2).
    False computes it over the whole step."""
    router_z_loss: float = 0.0
    """The routers' z-loss weight (`LMObjective.router_z_loss`). lm-engine
    uses 0.1 times its aux coefficient. Zero adds nothing."""
    mtp_weight: float | None = None
    """DeepSeek's lambda on the multi-token-prediction term. It needs a model
    with num_nextn_predict_layers above zero; unset leaves the term out."""
    indexer: IndexerTraining | None = None
    """DeepSeek-V3.2's lightning-indexer training phase. `indexer:indexer-training
    --indexer.phase warmup` freezes everything except the indexer, on a model
    whose mla mixer names the indexer's heads and no top-k. `sparse` trains
    the whole model on its top-k, with the KL term beside the cross entropy.
    Unset trains no indexer term."""
    token_accuracy: bool = True
    """Whether to report the argmax accuracy beside the loss. False skips that
    argmax over every logit."""
    block_prompt_tokens: int = 256
    """The length of the clean prompt prefix in a block-diffusion token row."""
    block_canvas_size: int | None = None
    """The training canvas width; None uses the checkpoint's canvas length."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.objective not in (objectives.paths[name] for name in ("lm", "masked_diffusion",
                                                                        "block_diffusion")):
            raise ValueError(
                f"--objective {self.objective!r} is not lm, masked_diffusion or block_diffusion")
        if self.objective == objectives.paths["block_diffusion"]:
            if self.pretrained is None:
                raise ValueError("block_diffusion fine-tuning requires --pretrained")
            if (self.balance_rate is not None or self.mtp_weight is not None
                    or self.trainer.quantization is not None):
                raise ValueError("block_diffusion has no balancing, MTP or quantized-training term")
