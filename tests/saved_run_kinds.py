"""The run kinds the saved-run corpus holds a run of (tests/test_saved_runs.py), which its writer
(tools/saved_runs_reference.py) trains: every run kind, an LM run whose decoder its source's model nests,
and each variant a run serializes apart (row 2640): a decision row of its own segments and option lines,
a lettered one, a block-diffusion run over a nested decoder, a run on the examples clock, fresh runs in
bfloat16 and in float16 (whose loss a dynamic scale scales), and a run over a source stored as it states."""

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import test_run_options as run_options
from test_lm_recipe import SEQ, write_token_files
from test_run_options import KINDS, decision, lm, read

from dew.config import RunConfig, SourceOptions, TrainerConfig
from dew.data import presets
from dew.data.sources.tokens import TokenFiles
from dew.decision import Choice
from dew.decision.layout import built_in
from dew.decision.task import Decide
from dew.inference.images import TextToImage
from dew.inference.tasks import BlockGeneration, TextGeneration
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
    """`test_run_options.lettered`'s fresh letter readout, its source by its path in the repo."""
    return dataclasses.replace(run_options.lettered(root, drawn), pretrained=QWEN)


def blockwise(root: Path, drawn: bool) -> RunConfig:
    """`test_run_options.blockwise`'s block-diffusion run over a release whose text decoder its model nests,
    its release and tokenizer, its token files' too, by their path in the repo."""
    return run_options.blockwise(root, drawn, release=DIFFUSION_GEMMA)


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


def computed(directory: Path | str, step: int | None = None) -> dict[str, list[str]]:
    """Each module's dtypes, by its variable scope's path, that one request of the task a saved run at
    `directory` loads, at `step` or else its default, multiplies in: its products' operands'
    (`saved_run_products`), read in a fresh interpreter, where every program the request runs lowers, as
    the corpus's writer reads them. A model whose outer module states bfloat16 while one projection of
    its decoder computes in float32 states float32 at that projection's scope. The products are the
    suite's CPU's, in the suite's environment, whose wide arithmetic multiplies a product summed in
    float32 in float64 (`dew.nn.precision.accumulated`); an accelerator lowers its own (bible §7 runs
    these files on the CPU)."""
    root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [str(root / "src"), *filter(None, [os.environ.get("PYTHONPATH")])])}
    # A record names the repo's sources by their path in it, read from the checkout's root.
    done = subprocess.run([sys.executable, str(Path(__file__).with_name("saved_run_products.py")),
                           str(directory), *([] if step is None else [str(step)])], cwd=root,
                          env=environment, capture_output=True, text=True, timeout=600, check=False)
    assert done.returncode == 0, done.stderr[-3000:]
    found = json.loads(done.stdout.splitlines()[-1])
    assert found, "the request lowered no matrix product"
    return found


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
               "decision-lettered": lettered, "block-diffusion": blockwise, "lm-examples": clocked,
               "lm-bf16": fresh, "lm-float16": scaled, "lm-scaled": chosen_scale, "lm-stored": stored}
"""Each kind the corpus saves a run of: every run kind (`test_run_options.KINDS`), an LM run whose decoder
its source's model nests, and each serialized variant beside them."""
