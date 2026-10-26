"""The language-model run, as one typed record.

`LMRunConfig` is what an LM run parses from its command line and writes as
`run.json` next to the checkpoints. Every run records the model, the data,
the optimizer and the trainer. A decoder run also records the tokenizer its
ids came from and the preview policy it generates with.
`TextGeneration.from_run` and `Pretrained.from_run` read those two fields back
from that file, so a run that does not record them loads as weights with no
way to turn text into ids.

The three objectives a decoder can be trained under are one field, because
they share every other field. `lm` is the next-token loss, `masked_diffusion`
is MDLM over a bidirectional model, and `block_diffusion` is DiffusionGemma's
own fine-tuning objective. `run` trains it on the token files `dew tokenize`
wrote, which decide the vocabulary.
"""

from __future__ import annotations

import dataclasses
from typing import ClassVar, NamedTuple

import jax
import jax.numpy as jnp

from dew.config import ModelConfig, ObjectiveConfig, OptimConfig, Prepared, RunConfig
from dew.data.text import HFTokenizer, tokenizer_for
from dew.objectives.lm.objective import Perplexity, Samples
from dew.records import integer, text
from dew.registry import DtypeName, objectives, resolve_dtype
from dew.sampling.text import Sampling


class Tokens(NamedTuple):
    """The token windows an LM run trains on (`LMRunConfig.tokens`)."""
    tokenizer: str
    vocab_size: int
    seq_len: int
    packed: bool


@dataclasses.dataclass(frozen=True)
class LMRunConfig(RunConfig):
    """A `RunConfig` plus the settings specific to language models."""

    RESUMABLE: ClassVar[tuple[str, ...]] = (*RunConfig.RESUMABLE, "prompt", "max_new_tokens", "sampling")

    objective: ObjectiveConfig = dataclasses.field(default_factory=lambda: ObjectiveConfig("lm"))
    """The objective and its arguments: lm, masked_diffusion (MDLM),
    block_diffusion (the official DiffusionGemma fine-tuning objective), or
    any objective by its import path that `LMObjective`'s arguments build:
    the decoder, `seq_len`, its starting `variables`, `samples` and
    `processor`. A step whose optimizer clips by the attention logit maxima
    (`muonclip`) asks the objective for them (`Step.qk_stats`), whichever
    objective it is."""
    model: ModelConfig = dataclasses.field(
        default_factory=lambda: ModelConfig("causal_transformer", {"dtype": "bfloat16"}))
    optim: OptimConfig = dataclasses.field(
        default_factory=lambda: OptimConfig(
            learning_rate=6e-4,
            weight_decay=0.1, clip_grads=1.0))
    tokenizer: str = "byte"
    """The tokenizer the ids were written with: 'byte', or an HF tokenizer name."""
    prompt: str = ""
    """The prompt that validation samples continue; an empty prompt continues a newline."""
    max_new_tokens: int = 128
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
    pretrained_param_dtype: DtypeName = "float32"
    """The dtype `pretrained`'s parameters are stored and trained in."""

    def __post_init__(self) -> None:
        if self.pretrained is None and self.pretrained_param_dtype != "float32":
            raise ValueError("--pretrained-param-dtype stores a --pretrained checkpoint's parameters; "
                             "this run loads none")
        if self.objective.name == objectives.paths["block_diffusion"] and self.pretrained is None:
            raise ValueError("block_diffusion fine-tuning requires --pretrained")

    def prepare(self) -> Prepared:
        """The objective this run names over its token files.

        The files decide the vocabulary; training and sampling decide the
        context, since generation decodes into a cache sized once at build
        time. A `pretrained` checkpoint decides the architecture instead, and
        the run records it as the commit it was read at. A fresh model records
        its vocabulary and context; a fine-tune records only its model choices,
        since the source supplies its geometry on every load. The objective's
        inference record holds the built model for `dew.pipeline`. An adapter
        binds to the model and the tree training starts from, the loaded weights
        or a fresh draw from the run's key, so the objective trains its factors
        alone. Perplexity scores each validation pass.
        """
        tokens = self.tokens()
        if objectives.label(self.objective.name) == "block_diffusion" and tokens.packed:
            raise ValueError("block_diffusion trains on spans of the stream, not packed documents")
        written, vocab_size = tokens.tokenizer, tokens.vocab_size
        if self.tokenizer != written:
            # Decoding with a different tokenizer than the ids were written with
            # produces text that says nothing about the model.
            raise ValueError(f"--tokenizer {self.tokenizer} does not match the token files, which were "
                             f"written with {written}")
        kind = objectives.label(self.objective.name)
        samples = None if kind == "block_diffusion" else self.samples()
        context = (tokens.seq_len + 1 if samples is None else
                   max(tokens.seq_len, len(samples.prompt) + samples.max_new_tokens))
        run, source = self, None
        if self.pretrained is None:
            # A fresh masked-diffusion model may corrupt to the id past the
            # tokenizer's, as MDLM does, so the table covers its mask id too.
            mask = self.model.arguments.get("mask_token_id")
            table = max(vocab_size, mask + 1) if isinstance(mask, int) else vocab_size
            fields = {**self.model.fields, "max_seq_len": context, "vocab_size": table}
            model = self.model.build(max_seq_len=context, vocab_size=table)
        else:
            source, run = self.pretrained_source(written, vocab_size, context)
            model, fields = source.model, dict(source.model_config)
        if source is None:
            run = dataclasses.replace(run, model=dataclasses.replace(self.model, fields=fields))
        variables = None if source is None else source.variables
        if self.lora is not None:
            if source is None:
                key = jax.random.key(self.trainer.key)
                base = model.init(key, jnp.zeros((1, tokens.seq_len), jnp.int32))
                adapter = self.lora.apply(model, base, key=jax.random.fold_in(key, 1))
                model, variables = adapter.model, adapter.variables
            else:
                source = source.adapt(self.lora, key=self.trainer.key)
                model, variables = source.model, source.variables
        if kind == "masked_diffusion":
            objective = self.masked_objective(model, fields, variables)
        elif kind == "block_diffusion":
            objective = self.block_objective(model, variables)
        else:
            objective = self.objective.build(
                model=model if source is None else source, seq_len=tokens.seq_len, variables=variables,
                samples=samples, processor=self.processor())
        # Compiled once: the run's length and its cadences are read off this loader, which `train` trains on.
        loaded, held = self.compiled(objective.data(), self.chain(), self.validation)
        # The run's own default metric, asked for where it holds a validation split to score.
        scored = self.trainer.eval_interval(loaded) is not None and bool(held)
        validation = (Perplexity(),) if scored else ()
        return Prepared(run, lambda name: run.train(objective, loaded, name=name, metrics=validation,
                                                     validation=held))

    def tokens(self) -> Tokens:
        """The token windows this run trains on, as its chain states them (`Dataset.facts`): their
        tokenizer, vocabulary and length. Raises ValueError for a chain that states none, which an LM
        run cannot size its model for."""
        facts = self.chain().facts()
        missing = [name for name in ("tokenizer", "vocab_size", "seq_len") if facts.get(name) is None]
        if missing:
            raise ValueError(f"an LM run trains on token windows (dew.data.presets.lm, sft or "
                             f"token_windows); this chain states no {', '.join(missing)}")
        return Tokens(text(facts["tokenizer"], "tokenizer"), integer(facts["vocab_size"], "vocab_size"),
                      integer(facts["seq_len"], "seq_len"), bool(facts.get("packed", False)))

    def processor(self):
        """The run's tokenizer as its checkpoints record it for every loader."""
        from dew.inference import RunProcessor

        return RunProcessor(tokenizer_for(self.tokenizer))

    def samples(self) -> Samples:
        """What the objective generates and decodes at every validation. The
        policy is kept with no preview budget too: the run records it, and its
        export and `dew.pipeline` decode with it."""
        if self.max_new_tokens <= 0:
            return Samples(prompt=[], max_new_tokens=0, sampling=self.sampling)
        tokenizer = tokenizer_for(self.tokenizer)
        return Samples(prompt=tokenizer.encode(self.prompt or "\n"), max_new_tokens=self.max_new_tokens,
                       sampling=self.sampling, decode=tokenizer.decode)

    def pretrained_source(self, written: str, vocab_size: int, context: int):
        """The bundle `pretrained` names, and this run with it pinned to the commit it was read at.

        The checkpoint decides every architecture field, so the model flags
        may say only how far the KV cache reaches, the compute dtype and the
        attention kernel. The token files' tokenizer has to be the
        checkpoint's: continuing pretraining on ids from another vocabulary
        trains the embedding table against noise.
        """
        from dew.interop import Pretrained, split_revision

        assert self.pretrained is not None
        choices = ("max_seq_len", "dtype", "attention_impl")
        overridden = sorted(set(self.model.fields) - set(choices))
        if overridden:
            raise ValueError(f"--model sets {overridden}, which the checkpoint at {self.pretrained} decides. "
                             f"Only {', '.join(choices)} are still choices.")
        reach = self.model.fields.get("max_seq_len", context)
        if reach is not None and not isinstance(reach, int):
            raise ValueError(f"--model.max_seq_len is {reach!r}; the context a checkpoint is reloaded at "
                             "is a number of tokens")
        name, revision = split_revision(self.pretrained)
        dtype = resolve_dtype(self.model.fields.get("dtype"))
        loaded = Pretrained.load(name, dtype=jnp.bfloat16 if dtype is None else dtype,
                                 param_dtype=self.pretrained_param_dtype,
                                 attention_impl=str(self.model.fields.get("attention_impl", "auto")),
                                 max_seq_len=reach, revision=revision,
                                 # An adapter keeps its source as it is.
                                 **self.trainer.source_placement(frozen=self.lora is not None))
        if not same_vocabulary(written, loaded, name):
            raise ValueError(f"the token files were written with {written}, and {self.pretrained} "
                             f"expects its own tokenizer's ids. Retokenize with --tokenizer {name}.")
        # A decoder's embedding table is usually padded past the tokenizer's ids
        # (Qwen3 stores 151936 rows for 151669 tokens), so covering them is the
        # requirement, not matching the count.
        if loaded.model.vocab_size < vocab_size:
            raise ValueError(f"{self.pretrained} has room for {loaded.model.vocab_size} ids and the token "
                             f"files use {vocab_size}")
        reference = name if loaded.revision is None else f"{name}@{loaded.revision}"
        return loaded, dataclasses.replace(self, pretrained=reference)

    def masked_objective(self, model, fields, variables):
        """MDLM over a bidirectional model, its mask id from the model's fields.

        A `pretrained` diffusion checkpoint carries `mask_token_id`; a
        from-scratch run names it beside --model.no-causal. A token window is
        `seq_len + 1` ids wide, and masked diffusion has no shift, so it
        denoises the whole row. The validation text is the unmasked rows
        decoded with the run's tokenizer, or bare ids with no preview budget."""
        from dew.diffusion.discrete import MDLM

        mask = fields.get("mask_token_id")
        if mask is None:
            raise ValueError("masked_diffusion trains a model with a mask token id: continue a --pretrained "
                             "diffusion checkpoint, which carries one, or name --model.mask-token-id beside "
                             "--model.no-causal")
        decode = None if self.max_new_tokens <= 0 else tokenizer_for(self.tokenizer).decode
        return self.objective.build(model=model, process=MDLM(mask_id=int(mask))(),
                                    seq_len=self.tokens().seq_len + 1, decode=decode, variables=variables,
                                    processor=self.processor())

    def block_objective(self, model, variables):
        """Each complete token-window row split into the clean prompt the run
        names (`--objective.prompt-length`) and response canvases."""
        from dew.nn.protocols import BlockDenoiser

        if not isinstance(model, BlockDenoiser):
            raise ValueError("block_diffusion requires a model that denoises canvases, as "
                             "DiffusionGemma does")
        prompt = self.objective.fields.get("prompt_length")
        width = self.objective.fields.get("canvas_size") or model.canvas_length
        if not isinstance(prompt, int) or not isinstance(width, int):
            raise ValueError("block_diffusion splits each row at --objective.prompt-length")
        response = self.tokens().seq_len + 1 - prompt
        if width < 1 or response < width or response % width:
            raise ValueError("seq_len + 1 must equal prompt_length plus whole training canvases")
        return self.objective.build(model=model, num_canvases=response // width, variables=variables,
                                    processor=self.processor())


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
    return HFTokenizer(written).tokenizer.get_vocab() == carried.get_vocab()


__all__ = ["LMRunConfig"]
