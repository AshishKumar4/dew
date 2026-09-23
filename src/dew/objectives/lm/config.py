"""The language-model run, as one typed record.

`LMRunConfig` is what an LM recipe or script parses from its command line and
writes as `run.json` next to the checkpoints. Beside the model, the data, the
optimizer and the trainer every run carries, a decoder run records the
tokenizer its ids came from and the preview policy it generates under, and
those are the fields `TextGeneration.from_run` and `dew.interop.export_run`
read back out of that file: a run that does not record them loads as weights
with no way to turn text into ids.

The three objectives a decoder is trained under are one field, because they
share every other one: `lm` is the next-token loss, `masked_diffusion` is
MDLM over a bidirectional model, and `block_diffusion` is DiffusionGemma's
own fine-tuning objective. `recipes/lm/train.py` adds the restriction that
it reads token files and builds the objective this record names.
"""

from __future__ import annotations

import dataclasses

from dew.config import DataSpec, ModelConfig, OptimConfig, RunConfig
from dew.data import TokenWindows
from dew.sampling.text import Sampling

from .objective import IndexerTraining


@dataclasses.dataclass(frozen=True)
class LMRunConfig(RunConfig):
    """A run, plus the language model's own knobs."""

    objective: str = "lm"
    """Loss convention: lm, masked_diffusion (MDLM), or block_diffusion
    (the official DiffusionGemma fine-tuning objective)."""
    model: ModelConfig = dataclasses.field(
        default_factory=lambda: ModelConfig("causal_transformer"))
    data: DataSpec = dataclasses.field(default_factory=TokenWindows)
    optim: OptimConfig = dataclasses.field(
        default_factory=lambda: OptimConfig(
            learning_rate=6e-4,
            weight_decay=0.1, clip_grads=1.0))
    tokenizer: str = "byte"
    """What the ids were written with: 'byte', or an HF tokenizer name."""
    ema_decay: float | None = 0.999
    """None disables EMA; 1.0 retains a frozen copy."""
    sample_prompt: str = ""
    """Prompt the validation samples continue; empty continues a newline."""
    sample_tokens: int = 128
    """Tokens generated per validation sample; 0 logs no text."""
    sampling: Sampling = dataclasses.field(
        default_factory=lambda: Sampling(temperature=0.8, top_k=40))
    """The preview policy, recorded with the run for inference."""
    pretrained: str | None = None
    """Hugging Face decoder to continue training: a hub repo id, `repo@revision`
    (a branch, tag or commit), or a local directory in that layout. A run
    records a hub repo as `repo@commit`, the commit it resolved to. The
    checkpoint decides the architecture, so --model.config may then carry
    max_seq_len alone."""
    balance_rate: float | None = None
    """How far a sparse run moves each router's balancing bias against its
    load every step (DeepSeek's aux-loss-free balancing). Needs a mixture
    with bias=True; unset leaves the bias where it is."""
    aux_loss_alpha: float | None = None
    """The expert balance loss's weight (`LMObjective.aux_loss_alpha`); with
    --no-seq-aux it is the Switch loss over the step's routed positions,
    lm-engine's `router_aux_loss_coef`. Unset adds no balance loss."""
    seq_aux: bool = True
    """Form the balance loss within each sequence (DeepSeek V2) rather than
    over the whole step."""
    router_z_loss: float = 0.0
    """The routers' z-loss weight (`LMObjective.router_z_loss`); lm-engine
    uses 0.1 times its aux coefficient. Zero adds nothing."""
    mtp_weight: float | None = None
    """DeepSeek's lambda on the multi-token-prediction term. Needs a model
    with num_nextn_predict_layers above zero; unset leaves the term out."""
    indexer: IndexerTraining | None = None
    """DeepSeek-V3.2's lightning-indexer phase: `indexer:indexer-training
    --indexer.phase warmup` freezes everything but the indexer of a model
    whose mla mixer names the indexer's heads and no top-k; `sparse`
    trains the whole model on its top-k with the KL beside the cross
    entropy. Unset trains no indexer term."""
    token_accuracy: bool = True
    """Report the argmax accuracy beside the loss; False skips the argmax
    over every logit it costs."""
    block_prompt_tokens: int = 256
    """Clean prompt prefix in a block-diffusion token row."""
    block_canvas_size: int | None = None
    """Training canvas width; None uses the checkpoint canvas length."""

    def __post_init__(self) -> None:
        if self.objective not in ("lm", "masked_diffusion", "block_diffusion"):
            raise ValueError(
                f"--objective {self.objective!r} is not lm, masked_diffusion or block_diffusion")
        if self.objective == "block_diffusion":
            if self.pretrained is None:
                raise ValueError("block_diffusion fine-tuning requires --pretrained")
            if (self.balance_rate is not None or self.mtp_weight is not None
                    or self.trainer.quantization is not None):
                raise ValueError("block_diffusion has no balancing, MTP or quantized-training term")
