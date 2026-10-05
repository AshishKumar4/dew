"""A decoder Dew exports, read by llama.cpp.

tools/llama_cpp_reference.py records llama.cpp v0.5.0 converting a tiny
Llama's export with its own convert_hf_to_gguf.py to an F32 GGUF, and
libllama's float32 logits over fixed ids, beside transformers' float64
logits on the same export, and the vocabulary the GGUF carries. Here the
export is written again and has to be the one llama.cpp read, config,
generation config and weights; its tokenizer has to be the GGUF's
vocabulary, piece for piece and special id for special id; and Dew's
logits hold tests/reference_error.py's rule against llama.cpp's.
"""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from reference_error import assert_as_exact_as_the_reference

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "llama_cpp" / "reference.npz"


def test_llama_cpp_reads_the_decoder_dew_exports_at_dews_logits(tmp_path):
    from transformers import AutoTokenizer

    from tools.llama_cpp_reference import digests, export

    model, variables = export(tmp_path / "export")
    with np.load(FIXTURE) as stored:
        recorded = {name: stored[name] for name in stored.files}
    meta = json.loads(recorded.pop("meta").tobytes())
    assert digests(tmp_path / "export") == meta["digests"], (
        "Dew's export changed: rerun tools/llama_cpp_reference.py's export and consume, "
        "so llama.cpp converts and reads the new files")
    tokenizer = AutoTokenizer.from_pretrained(str(tmp_path / "export"), local_files_only=True)
    assert tokenizer.convert_ids_to_tokens(list(range(len(tokenizer)))) == recorded["gguf.pieces"].tolist()
    special = [tokenizer.unk_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id]
    assert special == recorded["gguf.special"].tolist()
    logits = model.apply(variables, jnp.asarray(recorded["ids"]))
    assert_as_exact_as_the_reference(np.asarray(logits), recorded["llama_cpp.fp32"],
                                     recorded["transformers.fp64"], "llama.cpp")
