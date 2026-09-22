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
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

import tyro

from dew.config import ModelConfig
from dew.data import PackedTokens, TokenWindows, tokenizer_for
from dew.objectives.lm import LMObjective, LMRunConfig, Samples
from dew.registry import datasets, metrics, models
from dew.training import TrainState, prepare_process, run_timestamp

if TYPE_CHECKING:
    # tyro reads the runtime annotation, a Union of the registered specs, and
    # a type checker cannot read a variable in a type expression. Statically
    # the field holds the two token datasets __post_init__ lets through.
    TokenSpec = TokenWindows | PackedTokens
else:
    TokenSpec = datasets.union


@dataclass(frozen=True)
class LmRunConfig(LMRunConfig):
    """The shipped LM run, narrowed to the token files this recipe reads.

    Everything else a decoder run records is `dew.objectives.lm.LMRunConfig`,
    which a script that trains on some other layout of the same ids uses as
    it stands. What this adds is the one thing the recipe itself requires:
    `--data.path` is a directory `tools/tokenize_text.py` wrote, or with
    data:packed-tokens several with their weights (`--data.path a 0.7 b 0.3`).
    """

    data: TokenSpec = field(default_factory=TokenWindows)

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.data, (TokenWindows, PackedTokens)):
            raise ValueError(
                "the language model recipe trains on token files: "
                "data:token-windows or data:packed-tokens")
        if self.objective == "block_diffusion" and not isinstance(self.data, TokenWindows):
            raise ValueError("block_diffusion requires data:token-windows, not packed documents")


def read_corpora(data: TokenSpec) -> str | list[str] | None:
    """What the run reads: --data.path, or every corpus the phases name."""
    return data.corpora if isinstance(data, PackedTokens) and data.phases else data.path


def token_directories(path: str | Mapping[str, float] | list[str] | None) -> list[Path]:
    """The directories tools/tokenize_text.py wrote, which --data.path names:
    one, or each corpus of a weighted mixture or of the phases."""
    if not path:
        raise ValueError("--data.path is the token directory tools/tokenize_text.py wrote")
    directories = [Path(path)] if isinstance(path, str) else [Path(name) for name in sorted(path)]
    for directory in directories:
        if not (directory / "meta.json").is_file():
            raise FileNotFoundError(
                f"{directory / 'meta.json'} is missing: --data.path is the token directory "
                "that tools/tokenize_text.py wrote, not a dataset name")
    return directories


def token_meta(path: str | Mapping[str, float] | list[str] | None) -> dict:
    """The tokenizer and vocabulary the token files were written with.

    A mixture's corpora feed one embedding table, so they have to record
    the same tokenizer and vocabulary; `train_tokens` is their sum.
    """
    metas = [json.loads((directory / "meta.json").read_text())
             for directory in token_directories(path)]
    recorded = {(meta["tokenizer"], int(meta["vocab_size"])) for meta in metas}
    if len(recorded) > 1:
        raise ValueError(
            f"the corpora of --data.path feed one embedding table, and record "
            f"different (tokenizer, vocab_size): {sorted(recorded)}")
    counts = [meta.get("train_tokens") for meta in metas]
    return {**metas[0], "train_tokens": None if None in counts else sum(counts)}


def context_length(config: LmRunConfig, samples: Samples | None) -> int:
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
    """The decoder a --pretrained run continues, its variables, the fields
    it was built from and the reference it was read at.

    `pretrained` is a local directory, a Hub repo or `repo@revision`; the
    reference returned pins a Hub repo to the commit it resolved to, so the
    run.json that records it names those exact weights.

    The checkpoint decides every architecture field, so the only thing
    --model.config may still say is how far the KV cache reaches. The fields
    that come back are dew's, not the checkpoint's, so a pretrained run logs
    the same vocabulary a fresh one does, compute dtype and kernel included.
    The tokenizer of the token files has to be the one the checkpoint was
    trained with: continuing pretraining on ids from another vocabulary trains
    the embedding table against noise.
    """
    from dew.interop import load_pretrained as load_checkpoint, split_revision

    overridden = sorted(set(model_config.config) - {"max_seq_len"})
    if overridden:
        raise ValueError(
            f"--model.config carries {overridden}, which the checkpoint at "
            f"{pretrained} decides. Only max_seq_len is still a choice.")

    context = model_config.config.get("max_seq_len", max_seq_len)
    if context is not None and not isinstance(context, int):
        raise ValueError(
            f"--model.config max_seq_len is {context!r}; the context a checkpoint "
            f"is reloaded at is a number of tokens")
    name, revision = split_revision(pretrained)
    loaded = load_checkpoint(
        name, dtype=model_config.dtype, attention_impl=model_config.attention_impl,
        max_seq_len=context, revision=revision)
    model, variables, fields = loaded.model, loaded.variables, loaded.model_config
    expected = checkpoint_tokenizer(loaded.source, name)
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
    reference = name if loaded.revision is None else f"{name}@{loaded.revision}"
    return model, variables, fields, reference


def checkpoint_tokenizer(directory: Path, name: str) -> str:
    """The tokenizer name the checkpoint in `directory`, read as `name`,
    expects its ids to come from.

    A checkpoint written by save_pretrained_decoder records the name it was
    exported with, since the path or repo it happens to sit at says nothing;
    any other hub repo is its own tokenizer's name.
    """
    generation_config = directory / "generation_config.json"
    if generation_config.is_file():
        recorded = json.loads(generation_config.read_text()).get("tokenizer_name")
        if recorded:
            return recorded
    return name


def build_samples(config: LmRunConfig) -> Samples | None:
    """What the objective generates and decodes at every validation."""
    if config.sample_tokens <= 0:
        return None
    tokenizer = tokenizer_for(config.tokenizer)
    return Samples(
        prompt=tokenizer.encode(config.sample_prompt or "\n"),
        max_new_tokens=config.sample_tokens, sampling=config.sampling,
        decode=tokenizer.decode)


def run_summary(config: LmRunConfig, fields: Mapping[str, object]) -> dict:
    """Flat view of the run, for the tracker."""
    return {
        **fields,
        "architecture": config.model.architecture,
        "dataset": read_corpora(config.data),
        "sequence_length": config.data.seq_len,
        "tokenizer": config.tokenizer,
        "batch_size": config.trainer.batch_size,
        "learning_rate": config.optim.learning_rate,
    }


def build_masked_objective(config: LmRunConfig, model, fields, pretrained):
    """The MDLM objective over a bidirectional model, its mask id from the run.

    A --pretrained diffusion checkpoint carries mask_token_id in the fields it
    was built from and its weights in `pretrained`, so the run continues from
    them; a from-scratch run names the mask id in --model.config beside
    causal=False and draws its tree from the key. The validation text is the
    unmasked rows decoded with the run's tokenizer, or bare ids when
    --sample-tokens is 0.

    A token window is `--data.seq-len + 1` ids wide: the LM objective spends
    the extra id on the shift, and masked diffusion has no shift, so it
    denoises the whole row rather than dropping a token off every window."""
    from dew.diffusion.discrete import MDLM
    from dew.objectives.diffusion.masked import MaskedDiffusionObjective

    mask = fields.get("mask_token_id")
    if mask is None:
        raise ValueError(
            "masked_diffusion trains a model with a mask token id: continue a "
            "--pretrained diffusion checkpoint, which carries one, or name "
            "mask_token_id in --model.config beside causal=False")
    decode = None if config.sample_tokens <= 0 else tokenizer_for(config.tokenizer).decode
    return MaskedDiffusionObjective(
        model, MDLM(mask_id=int(mask))(), config.data.seq_len + 1,
        ema_decay=config.ema_decay, decode=decode, pretrained=pretrained)


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
                    config.trainer.xla_flags, config.trainer.compilation_cache_dir,
                    layout=config.trainer.layout)

    meta = token_meta(read_corpora(config.data))
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
        model, pretrained, fields, reference = load_pretrained(
            config.pretrained, config.model, vocab_size, context, meta)
        # run.json names the commit the weights were read at.
        config = replace(config, pretrained=reference)
    # run.json records the resolved model as
    # built, vocabulary and context included, so `dew.pipeline` rebuilds it.
    settings = config.model.precision_settings()
    resolved = {name: value for name, value in fields.items() if name not in settings}
    if config.objective == "block_diffusion":
        resolved["max_seq_len"] = model.max_seq_len
    config = replace(config, model=replace(config.model, config=resolved))
    name = config.trainer.name or (
        f"{config.objective}-{'+'.join(d.name for d in token_directories(read_corpora(config.data)))}/"
        f"seq-{config.data.seq_len}/"
        f"lr-{config.optim.learning_rate}/"
        f"date-{run_timestamp()}")
    summary = {"model": fields, "arguments": run_summary(config, fields),
               "dataset": {"path": read_corpora(config.data), "records": data.records,
                           "tokens": meta.get("train_tokens")}}
    validation = (metrics.perplexity(),)
    if config.objective == "masked_diffusion":
        return config.train(build_masked_objective(config, model, fields, pretrained), data,
                            name=name, metrics=validation, summary=summary)
    if config.objective == "block_diffusion":
        return config.train(build_block_objective(config, model, pretrained), data,
                            name=name, metrics=validation, summary=summary)
    objective = LMObjective(
        model,
        config.data.seq_len,
        ema_decay=config.ema_decay,
        samples=samples,
        pretrained=pretrained,
        balance_rate=config.balance_rate,
        aux_loss_alpha=config.aux_loss_alpha,
        seq_aux=config.seq_aux,
        router_z_loss=config.router_z_loss,
        mtp_weight=config.mtp_weight,
        indexer=config.indexer,
        qk_stats=config.optim.optimizer == "muonclip",
        token_accuracy=config.token_accuracy,
    )
    return config.train(objective, data, name=name, metrics=validation, summary=summary)


if __name__ == '__main__':
    main(tyro.cli(tyro.conf.CascadeSubcommandArgs[LmRunConfig]))
