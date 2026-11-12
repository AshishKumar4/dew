"""A decoder Dew exports, read by llama.cpp.

tools/llama_cpp_reference.py records llama.cpp v0.5.0 converting a tiny
Llama's export with its own convert_hf_to_gguf.py to an F32 GGUF, and the
vocabulary the GGUF carries; tools/llama_cpp_worker.py records libllama's
float32 logits over drawn ids beside transformers' float64 logits on the same
export. Here the export is written again and has to be the one llama.cpp
read, config, generation config and weights; its tokenizer has to be the
GGUF's vocabulary, piece for piece and special id for special id; and
llama.cpp's logits of the export are on par with Dew's own of the same weights
by check 1.
"""

import functools
import json
import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from reference_runs import assert_arms_as_recorded

EXPORTED = Path(__file__).resolve().parent / "fixtures" / "llama_cpp" / "reference.npz"
"""tools/llama_cpp_reference.py's record of the export llama.cpp converted and the vocabulary its GGUF
carries."""


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "llama_cpp" / "recording.npz"
"""tools/venv_reference.py's `llama-cpp` recording (tools/llama_cpp_worker.py), its outputs whole."""
LLAMA_SEEDS = 8
"""The site's declared seeds, sized by `seeds_for` paired as its recording confirmed them: libllama's
logits the arm judged, Dew's the reference (tools/venv_reference.py's `judged`)."""
OUTPUTS = ("logits",)


def draw(seed: int) -> dict[str, np.ndarray]:
    """One row of 24 ids over the decoder's vocabulary but its specials (`SPECIALS`), drawn for `seed`."""
    from tools.llama_cpp_reference import FIELDS, SPECIALS

    rng = np.random.default_rng([4735, seed])
    return {"ids": rng.integers(len(SPECIALS), FIELDS["vocab_size"], (1, 24)).astype(np.int32)}


@functools.cache
def exported():
    """The decoder llama.cpp reads, as Dew builds and exports it (`llama_cpp_reference.export`)."""
    from tools.llama_cpp_reference import export

    return export(Path(tempfile.mkdtemp(prefix="llama-cpp-export-")) / "export")


def dew(_, inputs) -> dict:
    """Dew's float32 logits over a draw's ids."""
    model, variables = exported()
    return {"logits": np.asarray(model.apply(variables, jnp.asarray(inputs["ids"])), np.float32)}


def test_llama_cpp_reads_the_decoder_dew_exports_at_dews_logits():
    """libllama's float32 logits of Dew's export on par with Dew's own float32 logits of the same weights
    over drawn ids, both measured from transformers' float64 run of the export (check 1): the export is
    the product under test, so the arm judged is what llama.cpp computes from it and the reference what
    Dew computes before exporting. A dropped or misnamed tensor, a wrong norm epsilon or a rope base
    llama.cpp reads otherwise moves libllama's logits far past Dew's distance from float64."""
    assert_arms_as_recorded((FIXTURE, draw, dew, OUTPUTS), "llama-cpp", LLAMA_SEEDS, "llama.cpp logits",
                            host=True, judged=True)


def test_llama_cpp_reads_the_export_dew_writes(tmp_path):
    """The export written again is the one llama.cpp read, config, generation config and weights; its
    tokenizer is the GGUF's vocabulary, piece for piece and special id for special id."""
    from transformers import AutoTokenizer

    from tools.llama_cpp_reference import digests, export

    export(tmp_path / "export")
    with np.load(EXPORTED) as stored:
        recorded = {name: stored[name] for name in stored.files}
    meta = json.loads(recorded.pop("meta").tobytes())
    assert digests(tmp_path / "export") == meta["digests"], (
        "Dew's export changed: rerun tools/llama_cpp_reference.py's export and consume, "
        "so llama.cpp converts and reads the new files")
    tokenizer = AutoTokenizer.from_pretrained(str(tmp_path / "export"), local_files_only=True)
    assert tokenizer.convert_ids_to_tokens(list(range(len(tokenizer)))) == recorded["gguf.pieces"].tolist()
    special = [tokenizer.unk_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id]
    assert special == recorded["gguf.special"].tolist()
