"""Train autoregressive, masked-diffusion or block-diffusion models on token files.

A sibling of the diffusion and JEPA recipes: same trainer, same sharding, same
checkpoints, and a different objective. The data is not images but the
`train.bin` / `val.bin` / `meta.json` a tokenizer run wrote, so the recipe
takes the vocabulary from the data, not the command line.

    curl -o data/shakespeare.txt --create-dirs \\
        https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    dew tokenize --input data/shakespeare.txt \\
        --out data/shakespeare-byte --tokenizer byte
    python recipes/lm/train.py --data.path data/shakespeare-byte \\
        --data.seq-len 256 --trainer.batch-size 32 --trainer.epochs 10 \\
        --model.config '{"emb_features": 384, "num_layers": 6, "num_heads": 6}'

`data:packed-tokens` packs whole documents into the windows instead.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from dew.config import ModelConfig
from dew.data import ByteTokenizer, HFTokenizer, PackedTokens, TokenWindows
from dew.inference import RunProcessor
from dew.objectives.lm import LMObjective, LMRunConfig, Perplexity, Samples
from dew.registry import datasets, models, objectives
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
    `--data.path` is a directory `dew tokenize` wrote, or with
    data:packed-tokens several with their weights (`--data.path a 0.7 b 0.3`).
    """

    data: TokenSpec = field(default_factory=TokenWindows)

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.data, (TokenWindows, PackedTokens)):
            raise ValueError(
                "the language model recipe trains on token files: "
                "data:token-windows or data:packed-tokens")
        if self.objective == objectives.paths["block_diffusion"] and not isinstance(self.data, TokenWindows):
            raise ValueError("block_diffusion requires data:token-windows, not packed documents")


def read_corpora(data: TokenSpec) -> str | list[str] | None:
    """What the run reads: --data.path, or every corpus the phases name."""
    return data.corpora if isinstance(data, PackedTokens) and data.phases else data.path


def token_directories(path: str | Mapping[str, float] | list[str] | None) -> list[Path]:
    """The directories `dew tokenize` wrote, which --data.path names:
    one, or each corpus of a weighted mixture or of the phases."""
    if not path:
        raise ValueError("--data.path is the token directory `dew tokenize` wrote")
    directories = [Path(path)] if isinstance(path, str) else [Path(name) for name in sorted(path)]
    for directory in directories:
        if not (directory / "meta.json").is_file():
            raise FileNotFoundError(
                f"{directory / 'meta.json'} is missing: --data.path is the token directory "
                "that `dew tokenize` wrote, not a dataset name")
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
    if config.objective == objectives.paths["block_diffusion"]:
        return config.data.seq_len + 1
    if samples is None:
        return config.data.seq_len
    return max(config.data.seq_len, len(samples.prompt) + samples.max_new_tokens)


def model_fields(config: LmRunConfig, vocab_size: int, max_seq_len: int) -> dict:
    """The fields the registry builds the model from."""
    # Data decides the vocabulary. Training and sampling decide the context.
    return {**config.model.fields, "max_seq_len": max_seq_len, "vocab_size": vocab_size}


def pretrained_source(pretrained: str, model_config: ModelConfig, vocab_size: int,
                      max_seq_len: int, meta: dict):
    """The bundle a --pretrained run continues and the reference it was read at.

    `pretrained` is a local directory, a Hub repo or `repo@revision`; the
    reference returned pins a Hub repo to the commit it resolved to, so the
    run.json that records it names those exact weights.

    The checkpoint decides every architecture field, so the model flags may
    still say only how far the KV cache reaches, the compute dtype and the
    attention kernel. The fields
    that come back are dew's, not the checkpoint's, so a pretrained run logs
    the same vocabulary a fresh one does, compute dtype and kernel included.
    The tokenizer of the token files has to be the one the checkpoint was
    trained with: continuing pretraining on ids from another vocabulary trains
    the embedding table against noise.
    """
    from dew.interop import Pretrained, split_revision

    choices = ("max_seq_len", "dtype", "attention_impl")
    overridden = sorted(set(model_config.fields) - set(choices))
    if overridden:
        raise ValueError(
            f"--model sets {overridden}, which the checkpoint at {pretrained} decides. "
            f"Only {', '.join(choices)} are still choices.")

    context = model_config.fields.get("max_seq_len", max_seq_len)
    if context is not None and not isinstance(context, int):
        raise ValueError(
            f"--model.max_seq_len is {context!r}; the context a checkpoint "
            f"is reloaded at is a number of tokens")
    name, revision = split_revision(pretrained)
    loaded = Pretrained.load(
        name, dtype=model_config.fields.get("dtype"),
        attention_impl=str(model_config.fields.get("attention_impl", "auto")),
        max_seq_len=context, revision=revision)
    if not same_vocabulary(str(meta["tokenizer"]), loaded, name):
        raise ValueError(
            f"the token files were written with {meta['tokenizer']}, and {pretrained} "
            f"expects its own tokenizer's ids. Retokenize with --tokenizer {name}.")
    # A decoder's embedding table is usually padded past the tokenizer's ids
    # (Qwen3 stores 151936 rows for 151669 tokens), so covering them is the
    # requirement, not matching the count.
    if loaded.model.vocab_size < vocab_size:
        raise ValueError(
            f"{pretrained} has room for {loaded.model.vocab_size} ids and the "
            f"token files use {vocab_size}")
    reference = name if loaded.revision is None else f"{name}@{loaded.revision}"
    return loaded, reference


def same_vocabulary(written: str, loaded, name: str) -> bool:
    """Whether the token files' tokenizer `written` is the vocabulary of the
    checkpoint `loaded`, read as `name`.

    An export carries its tokenizer's own files, not the name it was made
    with, so the vocabularies are compared: the byte vocabulary, which has
    no files, by the `byte` an export records for it; an HF one by its
    token-to-id table, against the files in the checkpoint's directory, or
    by name where the checkpoint carries none.
    """
    if written == "byte" or loaded.tokenizer == "byte":
        return written == loaded.tokenizer
    if loaded.processor is None:
        return written == name
    carried = HFTokenizer(str(loaded.source), local_files_only=True).tokenizer
    return run_tokenizer(written).tokenizer.get_vocab() == carried.get_vocab()


def run_tokenizer(name: str) -> ByteTokenizer | HFTokenizer:
    """The tokenizer `--tokenizer` names: "byte" for Dew's UTF-8 vocabulary,
    any other name a Hugging Face tokenizer."""
    return ByteTokenizer() if name == "byte" else HFTokenizer(name)


def build_samples(config: LmRunConfig) -> Samples:
    """What the objective generates and decodes at every validation. The
    policy is kept with no preview budget too: the run records it, and its
    export and `dew.pipeline` decode with it."""
    if config.sample_tokens <= 0:
        # No preview, so no tokenizer to read: the policy alone.
        return Samples(prompt=[], max_new_tokens=0, sampling=config.sampling)
    tokenizer = run_tokenizer(config.tokenizer)
    return Samples(
        prompt=tokenizer.encode(config.sample_prompt or "\n"),
        max_new_tokens=config.sample_tokens, sampling=config.sampling,
        decode=tokenizer.decode)


def run_summary(config: LmRunConfig, fields: Mapping[str, object]) -> dict:
    """Flat view of the run, for the tracker."""
    return {
        **fields,
        "architecture": config.model.label,
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
    decode = None if config.sample_tokens <= 0 else run_tokenizer(config.tokenizer).decode
    return MaskedDiffusionObjective(
        model, MDLM(mask_id=int(mask))(), config.data.seq_len + 1,
        ema_decay=config.ema_decay, decode=decode, variables=pretrained,
        processor=RunProcessor(run_tokenizer(config.tokenizer)))


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
        canvas_size=width, variables=pretrained, ema_decay=config.ema_decay,
        processor=RunProcessor(run_tokenizer(config.tokenizer)))


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

    samples = None if config.objective == objectives.paths["block_diffusion"] else build_samples(config)
    context = context_length(config, samples)

    source = None
    if config.pretrained is None:
        fields = model_fields(config, vocab_size, context)
        model = models.build(config.model.name, **fields)
    else:
        source, reference = pretrained_source(
            config.pretrained, config.model, vocab_size, context, meta)
        model, fields = source.model, source.model_config
        # run.json names the commit the weights were read at.
        config = replace(config, pretrained=reference)
    # run.json records the resolved model as
    # built, vocabulary and context included, so `dew.pipeline` rebuilds it.
    resolved = dict(fields)
    if config.objective == objectives.paths["block_diffusion"]:
        resolved["max_seq_len"] = model.max_seq_len
    config = replace(config, model=replace(config.model, fields=resolved))
    name = config.trainer.name or (
        f"{objectives.label(str(config.objective))}-{'+'.join(d.name for d in token_directories(read_corpora(config.data)))}/"
        f"seq-{config.data.seq_len}/"
        f"lr-{config.optim.learning_rate}/"
        f"date-{run_timestamp()}")
    summary = {"model": fields, "arguments": run_summary(config, fields),
               "dataset": {"path": read_corpora(config.data), "records": data.records,
                           "tokens": meta.get("train_tokens")}}
    # Perplexity scores each validation pass; --trainer.eval-every None runs none.
    validation = () if config.trainer.eval_interval(data) is None else (Perplexity(),)
    pretrained = None if source is None else source.variables
    if config.lora is not None:
        # The adapter binds to the model and the tree training starts from,
        # the loaded weights or a fresh draw from the run's key, so the
        # objective trains its factors alone.
        if source is None:
            key = jax.random.key(config.trainer.key)
            base = model.init(key, jnp.zeros((1, config.data.seq_len), jnp.int32))
            adapter = config.lora.apply(model, base, key=jax.random.fold_in(key, 1))
            model, pretrained = adapter.model, adapter.variables
        else:
            source = source.adapt(config.lora, key=config.trainer.key)
            model, pretrained = source.model, source.variables
    if config.objective == objectives.paths["masked_diffusion"]:
        return config.train(build_masked_objective(config, model, fields, pretrained), data,
                            name=name, metrics=validation, summary=summary)
    if config.objective == objectives.paths["block_diffusion"]:
        return config.train(build_block_objective(config, model, pretrained), data,
                            name=name, metrics=validation, summary=summary)
    options = {
        "ema_decay": config.ema_decay,
        "samples": samples,
        # The run's tokenizer, which its checkpoints record for every loader.
        "processor": RunProcessor(run_tokenizer(config.tokenizer)),
        "balance_rate": config.balance_rate,
        "aux_loss_alpha": config.aux_loss_alpha,
        "seq_aux": config.seq_aux,
        "router_z_loss": config.router_z_loss,
        "mtp_weight": config.mtp_weight,
        "indexer": config.indexer,
        "qk_stats": config.optim.optimizer == "muonclip",
        "token_accuracy": config.token_accuracy,
    }
    objective = LMObjective(model if source is None else source, config.data.seq_len,
                            variables=pretrained, **options)
    return config.train(objective, data, name=name, metrics=validation, summary=summary)


if __name__ == '__main__':
    main(LmRunConfig.cli())
