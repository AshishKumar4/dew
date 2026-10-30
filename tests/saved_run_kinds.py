"""The run kinds the saved-run corpus holds a run of (tests/test_saved_runs.py), which its writer
(tools/saved_runs_reference.py) trains: every run kind, an LM run whose decoder its source's model nests,
and each variant a run serializes apart (row 2640): a decision row of its own segments and option lines,
a lettered one, a block-diffusion run over a nested decoder, a run on the examples clock, fresh runs in
bfloat16 and in float16 (whose loss a dynamic scale scales), and a run over a source stored as it states."""

import dataclasses
from pathlib import Path

import numpy as np
from test_lm_recipe import SEQ, write_token_files
from test_run_options import KINDS, decision, lm, read
from token_support import write_token_corpus

from dew.config import ObjectiveConfig, RunConfig, SourceOptions, TrainerConfig
from dew.data import Loading, presets
from dew.data.sources.tokens import TokenFiles
from dew.decision.config import Letters
from dew.decision.layout import Labels, built_in
from dew.nn.precision import numerics_of
from dew.registry import ModelConfig

MULTIMODAL = "tests/fixtures/hf/gemma3-tiny-mm"
"""A multimodal source whose decoder its model nests (`MultimodalTransformer.language_model`), by its
path in the repo, which a record keeps and a later checkout reads where it sits."""

QWEN = "tests/fixtures/hf/qwen38-dense-tiny"
"""A tiny causal source a fresh decision head and an LM run read, by its path in the repo."""

DIFFUSION_GEMMA = "tests/fixtures/hf/diffusion-gemma-sft"
"""A block-diffusion release whose text decoder its model nests (`DiffusionGemma.text`), by its path in
the repo."""


def windows(path):
    """An LM run's windows of a token store, eight a batch."""
    return read(TokenFiles(path).pipe(presets.lm, None, seq_len=SEQ, batch=None, pack=None).batch(8))


def over(source: str, root: Path, drawn: bool) -> RunConfig:
    """An LM run over `source`, its windows token files in the source's own tokenizer."""
    corpus = write_token_files(root / "source-tokens", 40 * SEQ, 8 * SEQ, tokenizer=source)
    return dataclasses.replace(lm(root, drawn), model=ModelConfig("causal_transformer", {"dtype": "float32"}),
                               pretrained=source, tokenizer=source, data=windows(corpus.train),
                               validation=windows(corpus.val))


def multimodal(root: Path, drawn: bool) -> RunConfig:
    """An LM run over a multimodal source, its decoder nested in the source's model, as a record holds
    it (row 2274: runs saved so stopped loading), over token files in the source's own tokenizer."""
    return over(MULTIMODAL, root, drawn)


def segmented(root: Path, drawn: bool) -> RunConfig:
    """A decision run whose rows are its own segments, option lines among them (`--segments`), the
    question layout's own row written as segments (`built_in`), as a record holds them (row 2627)."""
    return dataclasses.replace(decision(root, drawn), segments=built_in(None, "question_first"))


def lettered(root: Path, drawn: bool) -> RunConfig:
    """A decision run of a fresh letter readout (`--head letters`) over a tiny causal source, its rows
    lettered option lines read at an answer cue."""
    # The tiny tokenizer spells each letter as one token bare ("{}"), not after a space.
    return dataclasses.replace(decision(root, drawn), pretrained=QWEN,
                               head=Letters(labels=Labels(variants=("{}",)), count=4),
                               source=SourceOptions(text_only=True))


def blockwise(root: Path, drawn: bool) -> RunConfig:
    """A block-diffusion run over a release whose text decoder its model nests (`DiffusionGemma.text`)."""
    # Its windows as `test_lm_recipe.gemma_spans` writes them, the tokenizer by its path in the repo.
    corpus = write_token_corpus(root / "gemma-tokens", np.random.RandomState(160).randint(4, 32, 160),
                                tokenizer=DIFFUSION_GEMMA, vocab_size=32, eos_id=1)
    spans = TokenFiles(corpus.train).pipe(presets.lm, None, seq_len=11, batch=8, pack=None).loading(
        Loading(workers=0))
    model = ModelConfig("causal_transformer", {"dtype": "float32", "attention_impl": "xla"})
    return dataclasses.replace(lm(root, drawn), pretrained=DIFFUSION_GEMMA, tokenizer=DIFFUSION_GEMMA,
                               objective=ObjectiveConfig("block_diffusion", {"prompt_length": 4}),
                               model=model, data=spans, validation=None, max_new_tokens=0)


def clocked(root: Path, drawn: bool) -> RunConfig:
    """An LM run on the examples clock (`--trainer.examples`), its progress the examples it consumed:
    `lm`'s windows, unpacked, which its record reads back with no store to read (a packed chain checks
    its store's document ends as it is built)."""
    return dataclasses.replace(lm(root, drawn), trainer=TrainerConfig(examples=16))


def fresh(root: Path, drawn: bool, precision: str = "bf16-mixed") -> RunConfig:
    """An LM run whose fresh model states no dtype, computing in its numerics' (`precision`)."""
    run = lm(root, drawn)
    fields = {name: value for name, value in run.model.fields.items() if name != "dtype"}
    return dataclasses.replace(run, model=ModelConfig("causal_transformer", fields),
                               trainer=TrainerConfig(precision=numerics_of(precision)))


def scaled(root: Path, drawn: bool) -> RunConfig:
    """`fresh`'s run computing in float16, whose loss a dynamic scale scales (`RunConfig.built`)."""
    return fresh(root, drawn, "16-mixed")


def stored(root: Path, drawn: bool) -> RunConfig:
    """An LM run over a source whose parameters it stores in bfloat16 (`--source.param-dtype`)."""
    return dataclasses.replace(over(QWEN, root, drawn), source=SourceOptions(param_dtype="bfloat16"))


SAVED_KINDS = {**KINDS, "multimodal": multimodal, "decision-segments": segmented,
               "decision-lettered": lettered, "block-diffusion": blockwise, "lm-examples": clocked,
               "lm-bf16": fresh, "lm-float16": scaled, "lm-stored": stored}
"""Each kind the corpus saves a run of: every run kind (`test_run_options.KINDS`), an LM run whose decoder
its source's model nests, and each serialized variant beside them."""
