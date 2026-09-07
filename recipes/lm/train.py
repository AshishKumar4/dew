"""Train autoregressive, masked-diffusion or block-diffusion models on token files.

A sibling of the diffusion and JEPA recipes: same trainer, same sharding, same
checkpoints, and a different objective. The data is not images but the
`train.bin` / `val.bin` / `meta.json` a tokenizer run wrote, so the recipe
takes the vocabulary from the data, not the command line.

    python tools/tokenize_text.py --input data/shakespeare.txt \
        --out data/shakespeare-byte --tokenizer byte
    python recipes/lm/train.py --data.path data/shakespeare-byte \
        --data.seq-len 256 --trainer.batch-size 32 --trainer.epochs 10 \
        --model.config '{"emb_features": 384, "num_layers": 6, "num_heads": 6}'

`data:packed-tokens` packs whole documents into the windows instead.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import tyro

from dew.config import ModelConfig, OptimConfig, RunConfig
from dew.data import ByteTokenizer, HFTokenizer, PackedTokens, TokenWindows
from dew.objectives.lm import LMObjective, Samples
from dew.sampling import Sampling
from dew.registry import datasets, metrics, models
from dew.training import TrainState, prepare_process, run_timestamp
from dew.training.quantization import Quantization, apply_quantization

if TYPE_CHECKING:
    # tyro reads the runtime annotation, a Union of the registered specs, and
    # a type checker cannot read a variable in a type expression. Statically
    # the field holds the two token datasets __post_init__ lets through.
    TokenSpec = TokenWindows | PackedTokens
else:
    TokenSpec = datasets.union


@dataclass(frozen=True)
class LmRunConfig(RunConfig):
    """A run, plus the language model's own knobs."""

    model: ModelConfig = field(
        default_factory=lambda: ModelConfig("causal_transformer"))
    data: TokenSpec = field(default_factory=TokenWindows)
    optim: OptimConfig = field(
        default_factory=lambda: OptimConfig(
            learning_rate=6e-4, learning_rate_peak=6e-4, learning_rate_end=6e-5,
            weight_decay=0.1, clip_grads=1.0))
    tokenizer: str = "byte"
    """What the token files were written with: 'byte', or an HF tokenizer name."""
    ema_decay: float | None = 0.999
    """None disables EMA; 1.0 retains a frozen copy."""
    sample_prompt: str = ""
    """Prompt the validation samples continue; empty continues a newline."""
    sample_tokens: int = 128
    """Tokens generated per validation sample; 0 logs no text."""
    pretrained: Optional[str] = None
    """Hugging Face decoder to continue training: a hub repo id or a local
    directory in that layout. The checkpoint decides the architecture, so
    --model.config may then carry max_seq_len alone."""
    balance_rate: Optional[float] = None
    """How far a sparse run moves each router's balancing bias against its
    load every step (DeepSeek's aux-loss-free balancing). Needs a mixture
    with bias=True; unset leaves the bias where it is."""
    mtp_weight: Optional[float] = None
    """DeepSeek's lambda on the multi-token-prediction term. Needs a model
    with num_nextn_predict_layers above zero; unset leaves the term out."""
    objective: str = "lm"
    """Loss convention: lm, masked_diffusion (MDLM), or block_diffusion
    (the official DiffusionGemma fine-tuning objective)."""
    block_prompt_tokens: int = 256
    """Clean prompt prefix in a block-diffusion token row."""
    block_canvas_size: int | None = None
    """Training canvas width; None uses the checkpoint canvas length."""
    quantization: Optional[Quantization] = None
    """Quantized-training spec, wrapped around the built model before the
    objective sees it; unset trains in the compute dtype."""

    def __post_init__(self):
        if not isinstance(self.data, (TokenWindows, PackedTokens)):
            raise ValueError(
                "the language model recipe trains on token files: "
                "data:token-windows or data:packed-tokens")
        if self.objective not in ("lm", "masked_diffusion", "block_diffusion"):
            raise ValueError(
                f"--objective {self.objective!r} is not lm, masked_diffusion or block_diffusion")
        if self.objective == "block_diffusion":
            if not isinstance(self.data, TokenWindows):
                raise ValueError("block_diffusion requires data:token-windows, not packed documents")
            if self.pretrained is None:
                raise ValueError("block_diffusion fine-tuning requires --pretrained")
            if self.balance_rate is not None or self.mtp_weight is not None or self.quantization is not None:
                raise ValueError("block_diffusion has no balancing, MTP or quantized-training term")


def token_directory(path: Optional[str]) -> Path:
    """The directory tools/tokenize_text.py wrote, which --data.path names."""
    if not path:
        raise ValueError("--data.path is the token directory tools/tokenize_text.py wrote")
    directory = Path(path)
    if not (directory / "meta.json").is_file():
        raise FileNotFoundError(
            f"{directory / 'meta.json'} is missing: --data.path is the token directory "
            "that tools/tokenize_text.py wrote, not a dataset name")
    return directory


def context_length(config: LmRunConfig, samples: Optional[Samples]) -> int:
    """How far the position table and the KV cache have to reach.

    Generation decodes into a cache sized once at build time, so a sampling
    budget longer than the training context is what decides the model's
    max_seq_len; the sequence length being trained on is the floor.
    """
    if config.objective == "block_diffusion":
        return config.data.seq_len + 1
    if samples is None:
        return config.data.seq_len
    return max(config.data.seq_len, len(samples.prompt) + samples.max_new_tokens)


def model_fields(config: LmRunConfig, vocab_size: int, max_seq_len: int) -> dict:
    """The fields the registry builds the model from."""
    # Data decides the vocabulary. Training and sampling decide the context.
    return {**config.model.fields(), "max_seq_len": max_seq_len, "vocab_size": vocab_size}


def load_pretrained(pretrained: str, model_config: ModelConfig, vocab_size: int,
                    max_seq_len: int, meta: dict):
    """The decoder a --pretrained run continues, its variables and the fields
    it was built from.

    The checkpoint decides every architecture field, so the only thing
    --model.config may still say is how far the KV cache reaches. The fields
    that come back are dew's, not the checkpoint's, so a pretrained run logs
    the same vocabulary a fresh one does, compute dtype and kernel included.
    The tokenizer of the token files has to be the one the checkpoint was
    trained with: continuing pretraining on ids from another vocabulary trains
    the embedding table against noise.
    """
    from dew.interop import load_pretrained as load_checkpoint

    overridden = sorted(set(model_config.config) - {"max_seq_len"})
    if overridden:
        raise ValueError(
            f"--model.config carries {overridden}, which the checkpoint at "
            f"{pretrained} decides. Only max_seq_len is still a choice.")

    loaded = load_checkpoint(
        pretrained, dtype=model_config.dtype, attention_impl=model_config.attention_impl,
        max_seq_len=model_config.config.get("max_seq_len", max_seq_len))
    model, variables, fields = loaded.model, loaded.variables, loaded.model_config
    expected = checkpoint_tokenizer(pretrained)
    if meta["tokenizer"] != expected:
        raise ValueError(
            f"the token files were written with {meta['tokenizer']}, and "
            f"{pretrained} expects {expected}. Retokenize with "
            f"--tokenizer {expected}.")
    # A decoder's embedding table is usually padded past the tokenizer's ids
    # (Qwen3 stores 151936 rows for 151669 tokens), so covering them is the
    # requirement, not matching the count.
    if model.vocab_size < vocab_size:
        raise ValueError(
            f"{pretrained} has room for {model.vocab_size} ids and the "
            f"token files use {vocab_size}")
    return model, variables, fields


def checkpoint_tokenizer(pretrained: str) -> str:
    """The tokenizer name a checkpoint expects its ids to come from.

    A hub repo is its own tokenizer's name. A directory written by
    save_pretrained_decoder records the name it was exported with, since the
    path it happens to sit at says nothing.
    """
    generation_config = Path(pretrained) / "generation_config.json"
    if generation_config.is_file():
        recorded = json.loads(generation_config.read_text()).get("tokenizer_name")
        if recorded:
            return recorded
    return pretrained


def build_tokenizer(name: str):
    """The tokenizer that decodes generated ids back into text."""
    return ByteTokenizer() if name == "byte" else HFTokenizer(name)


def build_samples(config: LmRunConfig) -> Optional[Samples]:
    """What the objective generates and decodes at every validation."""
    if config.sample_tokens <= 0:
        return None
    tokenizer = build_tokenizer(config.tokenizer)
    return Samples(
        prompt=tokenizer.encode(config.sample_prompt or "\n"),
        max_new_tokens=config.sample_tokens, sampling=Sampling(temperature=0.8, top_k=40),
        decode=tokenizer.decode)


def run_summary(config: LmRunConfig, fields: dict) -> dict:
    """Flat view of the run, for the tracker."""
    return {
        **fields,
        "architecture": config.model.architecture,
        "dataset": config.data.path,
        "sequence_length": config.data.seq_len,
        "tokenizer": config.tokenizer,
        "batch_size": config.trainer.batch_size,
        "learning_rate": config.optim.learning_rate,
    }


def build_masked_objective(config: LmRunConfig, model, fields):
    """The MDLM objective over a bidirectional model, its mask id from the run.

    A --pretrained diffusion checkpoint carries mask_token_id in the fields it
    was built from; a from-scratch run names it in --model.config beside
    causal=False. The validation text is the unmasked rows decoded with the
    run's tokenizer, or bare ids when --sample-tokens is 0."""
    from dew.diffusion.discrete import MDLM
    from dew.objectives.diffusion.masked import MaskedDiffusionObjective

    mask = fields.get("mask_token_id")
    if mask is None:
        raise ValueError(
            "masked_diffusion trains a model with a mask token id: continue a "
            "--pretrained diffusion checkpoint, which carries one, or name "
            "mask_token_id in --model.config beside causal=False")
    decode = None if config.sample_tokens <= 0 else build_tokenizer(config.tokenizer).decode
    return MaskedDiffusionObjective(
        model, MDLM(mask_id=int(mask))(), config.data.seq_len,
        ema_decay=config.ema_decay, decode=decode)


def build_block_objective(config: LmRunConfig, model, pretrained):
    """Split each complete token-window row into a clean prompt and response canvases."""
    from dew.nn.diffusion_gemma import DiffusionGemma
    from dew.objectives.diffusion.block import BlockDiffusionObjective

    if not isinstance(model, DiffusionGemma):
        raise ValueError("block_diffusion requires a DiffusionGemma checkpoint")
    width = model.canvas_length if config.block_canvas_size is None else config.block_canvas_size
    response = config.data.seq_len + 1 - config.block_prompt_tokens
    if width < 1 or response < width or response % width:
        raise ValueError("seq_len + 1 must equal block_prompt_tokens plus whole training canvases")
    return BlockDiffusionObjective(
        model, prompt_length=config.block_prompt_tokens, num_canvases=response // width,
        canvas_size=width, pretrained=pretrained, ema_decay=config.ema_decay)


def main(config: LmRunConfig) -> TrainState:
    prepare_process(config.trainer.wandb, config.trainer.multi_host,
                    config.trainer.xla_flags, config.trainer.compilation_cache_dir)

    tokens = token_directory(config.data.path)
    # The tokenizer and vocabulary the token files were written with.
    meta = json.loads((tokens / "meta.json").read_text())
    if config.tokenizer != meta['tokenizer']:
        # Decoding with a different tokenizer than the ids were written with
        # produces text that says nothing about the model.
        raise ValueError(
            f"--tokenizer {config.tokenizer} does not match the token files, which "
            f"were written with {meta['tokenizer']}")
    vocab_size = int(meta['vocab_size'])

    data = config.data.load(batch=config.trainer.batch_size)
    if not data.steps_per_epoch:
        raise ValueError(
            f"{data.records} training windows do not fill one batch of "
            f"{config.trainer.batch_size}, so an epoch is no steps at all: read "
            "more data or lower --trainer.batch-size")

    samples = None if config.objective == "block_diffusion" else build_samples(config)
    context = context_length(config, samples)

    pretrained = None
    if config.pretrained is None:
        fields = model_fields(config, vocab_size, context)
        model = models.build(config.model.architecture, **fields)
    else:
        model, pretrained, fields = load_pretrained(
            config.pretrained, config.model, vocab_size, context, meta)
    if config.quantization is not None:
        model = apply_quantization(model, config.quantization)
    name = config.trainer.name or (
        f"{config.objective}-{tokens.name}/seq-{config.data.seq_len}/"
        f"lr-{config.optim.learning_rate}/"
        f"date-{run_timestamp()}")
    summary = {"model": fields, "arguments": run_summary(config, fields),
               "dataset": {"path": config.data.path, "records": data.records,
                           "tokens": meta.get("train_tokens")}}
    if config.objective == "masked_diffusion":
        return config.train(build_masked_objective(config, model, fields), data,
                            name=name, summary=summary)
    if config.objective == "block_diffusion":
        return config.train(build_block_objective(config, model, pretrained), data,
                            name=name, summary=summary)
    objective = LMObjective(
        model,
        config.data.seq_len,
        ema_decay=config.ema_decay,
        samples=samples,
        pretrained=pretrained,
        balance_rate=config.balance_rate,
        mtp_weight=config.mtp_weight,
        qk_stats=config.optim.optimizer == "muonclip",
    )
    return config.train(objective, data, name=name, metrics=(metrics.perplexity(),), summary=summary)


if __name__ == '__main__':
    main(tyro.cli(tyro.conf.CascadeSubcommandArgs[LmRunConfig]))
