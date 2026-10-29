"""The run kinds the saved-run corpus holds a run of (tests/test_saved_runs.py), which its writer
(tools/saved_runs_reference.py) trains: every run kind and an LM run whose decoder its source's model
nests."""

import dataclasses
from pathlib import Path

from test_lm_recipe import SEQ, write_token_files
from test_run_options import KINDS, lm, read

from dew.config import RunConfig
from dew.data import presets
from dew.data.sources.tokens import TokenFiles
from dew.registry import ModelConfig

MULTIMODAL = "tests/fixtures/hf/gemma3-tiny-mm"
"""A multimodal source whose decoder its model nests (`MultimodalTransformer.language_model`), by its
path in the repo, which a record keeps and a later checkout reads where it sits."""


def multimodal(root: Path, drawn: bool) -> RunConfig:
    """An LM run over a multimodal source, its decoder nested in the source's model, as a record holds
    it (row 2274: runs saved so stopped loading), over token files in the source's own tokenizer."""
    corpus = write_token_files(root / "multimodal-tokens", 40 * SEQ, 8 * SEQ, tokenizer=MULTIMODAL)

    def windows(path):
        return read(TokenFiles(path).pipe(presets.lm, None, seq_len=SEQ, batch=None, pack=None).batch(8))
    return dataclasses.replace(lm(root, drawn), model=ModelConfig("causal_transformer", {"dtype": "float32"}),
                               pretrained=MULTIMODAL, tokenizer=MULTIMODAL, data=windows(corpus.train),
                               validation=windows(corpus.val))


SAVED_KINDS = {**KINDS, "multimodal": multimodal}
"""Each kind the corpus saves a run of: every run kind (`test_run_options.KINDS`) and an LM run whose
decoder its source's model nests."""
