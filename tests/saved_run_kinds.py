"""The run kinds the saved-run corpus holds a run of (tests/test_saved_runs.py), which its writer
(tools/saved_runs_reference.py) trains: every run kind, an LM run whose decoder its source's model nests,
and each variant a run serializes apart (row 2640): a decision row of its own segments and option lines,
a lettered one, a block-diffusion run over a nested decoder, a run on the examples clock, fresh runs in
bfloat16 and in float16 (whose loss a dynamic scale scales), and a run over a source stored as it states."""

import collections
import dataclasses
import re
import tempfile
from pathlib import Path

import jax
import numpy as np
from test_lm_recipe import SEQ, write_token_files
from test_run_options import KINDS, decision, lm, read
from token_support import write_token_corpus

from dew.config import ObjectiveConfig, RunConfig, SourceOptions, TrainerConfig
from dew.data import Loading, presets
from dew.data.sources.tokens import TokenFiles
from dew.decision import Choice
from dew.decision.config import Letters
from dew.decision.layout import AnswerCue, Instructions, Labels, OptionLines, State, Text, built_in
from dew.decision.task import Decide
from dew.inference.images import TextToImage
from dew.inference.tasks import BlockGeneration, TextGeneration
from dew.nn.kernels import options
from dew.nn.precision import numerics_of
from dew.registry import ModelConfig
from dew.training.budget import Pause

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


def lettered_in_context(root: Path, drawn: bool) -> RunConfig:
    """A decision run of a letter readout whose question is read as the row's context (row 2008):
    the state and the answer cue in the row, the question and its lettered options in the context its
    cross-attention reads. A main whose option lines named their stream (0b4eb412b, rows 2008 to 2450)
    recorded them so, which a later main reads (row 2627)."""
    labels = Labels(variants=("{}",))
    segments = (Text("STATE:\n"), State(), Instructions({"choice": "{}"}, stream="context"),
                OptionLines(line="\n{label}. {option}", bare="\n{label}. {option}", join="",
                            noul_first="false"),
                AnswerCue(template="\nAnswer:", many="\nAnswer:"))
    return dataclasses.replace(lettered(root, drawn), segments=segments, context_layers=(0,),
                               head=Letters(labels=labels, count=4))


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
    """An LM run on the examples clock (`--trainer.examples`), its progress the examples it consumed,
    paused halfway (`--trainer.stop.fraction`), so a resume trains on to its bound: `lm`'s windows,
    unpacked, which its record reads back with no store to read (a packed chain checks its store's
    document ends as it is built)."""
    return dataclasses.replace(lm(root, drawn), trainer=TrainerConfig(examples=32, stop=Pause(0.5)))


def fresh(root: Path, drawn: bool, precision: str = "bf16-mixed") -> RunConfig:
    """An LM run whose fresh model states no dtype, computing in its numerics' (`precision`)."""
    run = lm(root, drawn)
    fields = {name: value for name, value in run.model.fields.items() if name != "dtype"}
    return dataclasses.replace(run, model=ModelConfig("causal_transformer", fields),
                               trainer=TrainerConfig(precision=numerics_of(precision)))


def scaled(root: Path, drawn: bool) -> RunConfig:
    """`fresh`'s run computing in float16, whose loss a dynamic scale scales (`RunConfig.built`)."""
    return fresh(root, drawn, "16-mixed")


def chosen_scale(root: Path, drawn: bool) -> RunConfig:
    """An LM run computing in float32 that states a dynamic loss scale, which no compute would choose
    (`--trainer.dynamic-scale`), as a run saved before its numerics records it."""
    return dataclasses.replace(lm(root, drawn), trainer=TrainerConfig(dynamic_scale=True))


def stored(root: Path, drawn: bool) -> RunConfig:
    """An LM run over a source whose parameters it stores in bfloat16 (`--source.param-dtype`)."""
    return dataclasses.replace(over(QWEN, root, drawn), source=SourceOptions(param_dtype="bfloat16"))


def computed(task) -> dict[str, int]:
    """How many matrix products a loaded task multiplies in each dtype, its operands' dtypes joined by a
    comma where they differ: the products of the programs one request of it lowers (`answered`), read
    from the StableHLO JAX writes as it lowers them (`jax_dump_ir_to`). A model whose outer module states
    bfloat16 while one projection of its decoder computes in float32 multiplies one product fewer in
    bfloat16. JAX's caches and Dew's outer programs (`dew.nn.kernels.options.program`, which an equal
    model's earlier task compiled) are cleared first, so every program lowers anew. The products are the
    suite's CPU's, as the corpus's writer counts them under the suite's environment, whose wide
    arithmetic multiplies a product summed in float32 in float64 (`dew.nn.precision.accumulated`); an
    accelerator lowers its own (bible §7 runs these files on the CPU)."""
    assert jax.default_backend() == "cpu", "a saved run's products are counted on the suite's CPU"
    jax.clear_caches()
    options._program.cache_clear()
    before = jax.config.values["jax_dump_ir_to"]
    with tempfile.TemporaryDirectory() as dump:
        jax.config.update("jax_dump_ir_to", dump)
        try:
            answered(task)
        finally:
            jax.config.update("jax_dump_ir_to", before)
        lowered = "".join(path.read_text() for path in Path(dump).glob("*.mlir"))
    products = collections.Counter(",".join(sorted({DTYPES[operand.rsplit("x", 1)[-1]] for operand in pair}))
                                   for pair in PRODUCT.findall(lowered))
    assert products, "the request lowered no matrix product"
    return dict(sorted(products.items()))


PRODUCT = re.compile(r"stablehlo\.dot_general [^\n]*: \(tensor<([^>]*)>, tensor<([^>]*)>\)")
"""A matrix product of a lowered program and its operands' tensor types, shape and element type."""

DTYPES = {"bf16": "bfloat16", "f16": "float16", "f32": "float32", "f64": "float64"}
"""The dtype StableHLO's element type names."""


def answered(task) -> None:
    """One request of a loaded task, as its kind's users make it, and its answer: a language model
    continues a prompt by one token through the tokenizer its run recorded, a decision answers a
    question about a state, and a text-to-image run draws an image in one step."""
    if isinstance(task, TextGeneration):
        assert len(task.decode(task("ab", max_new_tokens=1, key=0))) == 1
    elif isinstance(task, BlockGeneration):
        # A response of at least one token past the prompt, its decoder having stepped.
        generated = task([[4, 5, 6, 7]], 4, key=0).host()
        assert (np.asarray(generated.lengths) > 0).all() and (np.asarray(generated.decoder_steps) > 0).all()
    elif isinstance(task, Decide):
        question = {"team": Choice("which team", ["billing", "technical", "sales"])}
        [answers] = task([({"text": "the site is down"}, question)])
        assert answers["team"].choice in ("billing", "technical", "sales")
    elif isinstance(task, TextToImage):
        assert len(task(["a cat"], steps=1, key=0).pil()) == 1
    else:
        raise AssertionError(f"no request is written for a saved {type(task).__name__}")


SAVED_KINDS = {**KINDS, "multimodal": multimodal, "decision-segments": segmented,
               "decision-lettered": lettered, "decision-lettered-context": lettered_in_context,
               "block-diffusion": blockwise, "lm-examples": clocked,
               "lm-bf16": fresh, "lm-float16": scaled, "lm-scaled": chosen_scale, "lm-stored": stored}
"""Each kind the corpus saves a run of: every run kind (`test_run_options.KINDS`), an LM run whose decoder
its source's model nests, and each serialized variant beside them, a lettered question read in context
among them."""
