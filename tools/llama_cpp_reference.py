#!/usr/bin/env python3
"""Write the fixture of llama.cpp reading a decoder Dew exports.

llama.cpp v0.5.0 (ggml-org/llama.cpp at 7fe450e19305b828c199d602c23a8337aaa1f03b,
its source archive checked against its SHA-256) converts the export with its
own convert_hf_to_gguf.py to an F32 GGUF, and libllama, built from the same
source for the CPU, computes the logits of every position of fixed ids with
an F32 KV cache, flash attention off, on one thread. transformers runs the
export in float64 (`diffusers_wan_reference.float64`) for the truth both
are measured from.

The decoder is a tiny Llama from the registry with weights on a grid of
2^-8 drawn from numpy's PCG64 integers (`weights`): fixed values every
machine reproduces bit for bit, where a draw through jax.random's
transcendentals rounds by the CPU's instruction set. It is exported by
`PretrainedDecoder.from_model(...).save` with the
committed tokenizer in tests/fixtures/llama_cpp/tokenizer: a Llama-style
byte-fallback BPE, which the converter reads without a pre-tokenizer hash
(gguf-py's LlamaHfVocab); a byte-level BPE such as the other committed
tokenizers is refused unless its hash is one of a published model's.

The converter runs in the Dew environment with sentencepiece added, which
it imports unconditionally. Its declared environment (transformers 4.57.6,
requirements/requirements-convert_hf_to_gguf.txt) cannot read the
tokenizer_config.json transformers 5 writes, which names the class
`TokenizersBackend`; the weights and config it converts do not depend on
which transformers parses the tokenizer.

    uv venv --python .venv/bin/python ~/.cache/dew/reference-venvs/llama-cpp-t5
    echo $PWD/.venv/lib/python3.12/site-packages \\
        > ~/.cache/dew/reference-venvs/llama-cpp-t5/lib/python3.12/site-packages/dew-env.pth
    uv pip install --python ~/.cache/dew/reference-venvs/llama-cpp-t5/bin/python --no-deps \\
        sentencepiece==0.2.2
    python tools/llama_cpp_reference.py tokenizer
    PYTHONPATH=src python tools/llama_cpp_reference.py export DIR
    PYTHONPATH=src python tools/llama_cpp_reference.py consume DIR \\
        ~/.cache/dew/reference-venvs/llama-cpp-t5/bin/python

`consume` fetches and builds llama.cpp under ~/.cache/dew/upstream (cmake,
ninja and a C++ compiler) and writes tests/fixtures/llama_cpp/reference.npz:
the ids, llama.cpp's float32 logits, transformers' float64 ones, the
SHA-256 of the export's config, generation config and weights (the weights
by contents, as tools/lora_export_reference.py's `digest` hashes them), and
the vocabulary the GGUF carries (its pieces by id, and its unknown,
beginning and end ids), read back with the pinned gguf-py.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "llama_cpp"
TOKENIZER = FIXTURE / "tokenizer"
COMMIT = "7fe450e19305b828c199d602c23a8337aaa1f03b"
ARCHIVE_SHA256 = "a6861d549427f814dc591c439e08206f67ffaba0248344d421589abf18199e67"
SOURCE = Path.home() / ".cache" / "dew" / "upstream" / "ggml-org" / "llama.cpp" / COMMIT
FIELDS = {"vocab_size": 384, "emb_features": 64, "num_layers": 2, "num_heads": 4, "num_kv_heads": 2,
          "mlp_features": 128, "max_seq_len": 64, "qk_norm": False, "tie_embeddings": False}
SEED = 61
IDS = (np.arange(3, 3 + 24 * 7, 7) % FIELDS["vocab_size"]).astype(np.int32)
DIGESTED = ("config.json", "generation_config.json", "model.safetensors")
"""The files Dew writes itself. The tokenizer files are transformers' own
serialization, which the tokenizers release decides, so what the GGUF
carries of them is recorded as a vocabulary instead (`gguf_vocabulary`)."""
SPECIAL_KEYS = ("tokenizer.ggml.unknown_token_id", "tokenizer.ggml.bos_token_id",
                "tokenizer.ggml.eos_token_id")

CORPUS = """Dew is a JAX research framework for diffusion models and language models.
It trains, samples and exports models, and reads published checkpoints back.
A tokenizer splits text into pieces, and a decoder predicts the next piece.
The converter writes every tensor into a single file that the runtime maps.
"""
SPECIALS = ("<unk>", "<s>", "</s>")

# The logits of every position of IDS, as float32 rows, from libllama's C API.
PROGRAM = r"""
#include "llama.h"
#include <cstdio>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 4) return 2;
    std::vector<llama_token> ids;
    FILE * input = std::fopen(argv[2], "rb");
    if (input == nullptr) return 2;
    for (llama_token id; std::fread(&id, sizeof(id), 1, input) == 1;) ids.push_back(id);
    std::fclose(input);
    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], model_params);
    if (model == nullptr) return 1;
    llama_context_params params = llama_context_default_params();
    params.n_ctx = params.n_batch = params.n_ubatch = ids.size();
    params.n_threads = params.n_threads_batch = 1;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    params.type_k = params.type_v = GGML_TYPE_F32;
    llama_context * context = llama_init_from_model(model, params);
    if (context == nullptr) return 1;
    llama_batch batch = llama_batch_init(ids.size(), 0, 1);
    for (size_t i = 0; i < ids.size(); ++i) {
        batch.token[i] = ids[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = true;
    }
    batch.n_tokens = ids.size();
    if (llama_decode(context, batch) != 0) return 1;
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    FILE * output = std::fopen(argv[3], "wb");
    for (size_t i = 0; i < ids.size(); ++i) {
        std::fwrite(llama_get_logits_ith(context, i), sizeof(float), vocab, output);
    }
    std::fclose(output);
    llama_batch_free(batch);
    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
"""


def write_tokenizer() -> None:
    """A Llama-style BPE: the byte tokens <0x00>..<0xFF> in the vocabulary
    for fallback, word starts marked by ▁ and no pre-tokenizer, its merges
    learned from CORPUS."""
    from tokenizers import AddedToken, Tokenizer, decoders, models, normalizers, trainers

    def shell(model):
        tokenizer = Tokenizer(model)
        tokenizer.normalizer = normalizers.Sequence([normalizers.Prepend("▁"), normalizers.Replace(" ", "▁")])
        tokenizer.decoder = decoders.Sequence([decoders.Replace("▁", " "), decoders.ByteFallback(),
                                               decoders.Fuse(), decoders.Strip(" ", 1, 0)])
        return tokenizer

    trained = shell(models.BPE(unk_token="<unk>", byte_fallback=True, fuse_unk=True))
    trained.train_from_iterator(CORPUS.splitlines(), trainers.BpeTrainer(
        vocab_size=FIELDS["vocab_size"] - 256, special_tokens=list(SPECIALS), show_progress=False))
    learned = json.loads(trained.to_str())["model"]
    pieces = [piece for piece, _ in sorted(learned["vocab"].items(), key=lambda item: item[1])
              if piece not in SPECIALS]
    vocab = {piece: index for index, piece in
             enumerate([*SPECIALS, *(f"<0x{byte:02X}>" for byte in range(256)), *pieces])}
    merges = [tuple(merge.split(" ")) if isinstance(merge, str) else tuple(merge)
              for merge in learned["merges"]]
    tokenizer = shell(models.BPE(vocab=vocab, merges=merges, unk_token="<unk>", byte_fallback=True,
                                 fuse_unk=True))
    tokenizer.add_special_tokens([AddedToken(token, special=True, normalized=False) for token in SPECIALS])
    TOKENIZER.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(TOKENIZER / "tokenizer.json"))
    config = {"tokenizer_class": "PreTrainedTokenizerFast", "bos_token": "<s>", "eos_token": "</s>",
              "unk_token": "<unk>", "add_bos_token": False, "add_eos_token": False,
              "clean_up_tokenization_spaces": False}
    (TOKENIZER / "tokenizer_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"{TOKENIZER}: {len(vocab)} tokens")


def export(directory: Path):
    """Write the decoder's export into `directory`; returns the model and
    the variables it wrote."""
    import jax
    import jax.numpy as jnp

    from dew.data.text import HFTokenizer
    from dew.interop.pretrained import PretrainedDecoder
    from dew.registry import models

    model = models.build("causal_transformer", **FIELDS, dtype="float32", attention_impl="xla")
    variables = weights(jax.eval_shape(model.init, jax.random.key(SEED), jnp.zeros((1, 8), jnp.int32)))
    PretrainedDecoder.from_model(model, variables, tokenizer=HFTokenizer(str(TOKENIZER))).save(str(directory))
    return model, variables


def weights(shapes):
    """Every leaf of `shapes` on a grid of 2^-8, exact in float32: a norm's
    scale 1 + k / 256 for k in [-16, 16], every other leaf k / 256 for k in
    [-24, 24] (a uniform spread of 0.055, about a Linear's initialization
    at width 64), drawn in order from PCG64 integers seeded with SEED."""
    import jax
    import jax.numpy as jnp

    rng = np.random.default_rng(SEED)

    def draw(path, shape):
        scale = jax.tree_util.keystr(path).endswith("['scale']")
        steps = rng.integers(-16, 17, shape.shape) if scale else rng.integers(-24, 25, shape.shape)
        return jnp.asarray((256 * scale + steps) / 256, shape.dtype)

    return jax.tree_util.tree_map_with_path(draw, shapes)


def digests(directory: Path) -> dict[str, str]:
    from tools.lora_export_reference import digest

    return {name: digest(directory / name) for name in DIGESTED}


def fetch() -> Path:
    """The pinned source, fetched once and checked against its SHA-256."""
    if (SOURCE / "convert_hf_to_gguf.py").is_file():
        return SOURCE
    SOURCE.mkdir(parents=True, exist_ok=True)
    archive = SOURCE / "source.tar.gz"
    url = f"https://github.com/ggml-org/llama.cpp/archive/{COMMIT}.tar.gz"
    with urllib.request.urlopen(url, timeout=300) as response:
        archive.write_bytes(response.read())
    found = hashlib.sha256(archive.read_bytes()).hexdigest()
    if found != ARCHIVE_SHA256:
        raise RuntimeError(f"{url} has SHA-256 {found}, not the pinned {ARCHIVE_SHA256}")
    with tarfile.open(archive) as source:
        for member in source.getmembers():
            member.name = member.name.split("/", 1)[1] if "/" in member.name else ""
        source.extractall(SOURCE, members=[m for m in source.getmembers() if m.name], filter="data")
    archive.unlink()
    return SOURCE


def build(source: Path) -> Path:
    """libllama for the CPU and the logits program over it; returns the program."""
    build = source / "build-cpu"
    program = build / "logits"
    if program.is_file():
        return program
    subprocess.run(["cmake", "-S", str(source), "-B", str(build), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
                    "-DBUILD_SHARED_LIBS=ON", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF",
                    "-DLLAMA_BUILD_SERVER=OFF", "-DLLAMA_BUILD_TOOLS=OFF", "-DLLAMA_CURL=OFF",
                    "-DGGML_OPENMP=OFF"], check=True)
    subprocess.run(["ninja", "-C", str(build), "llama"], check=True)
    (build / "logits.cpp").write_text(PROGRAM)
    subprocess.run(["g++", "-O2", "-std=c++17", str(build / "logits.cpp"), f"-I{source / 'include'}",
                    f"-I{source / 'ggml' / 'include'}", f"-L{build / 'bin'}", "-lllama", "-lggml",
                    "-lggml-base", f"-Wl,-rpath,{build / 'bin'}", "-o", str(program)], check=True)
    return program


def gguf_vocabulary(source: Path, gguf: Path) -> tuple[np.ndarray, np.ndarray]:
    """The GGUF's pieces by id, and its unknown, beginning and end ids."""
    sys.path.insert(0, str(source / "gguf-py"))
    from gguf import GGUFReader

    fields = GGUFReader(str(gguf)).fields
    pieces = np.asarray(fields["tokenizer.ggml.tokens"].contents())
    return pieces, np.asarray([fields[key].contents() for key in SPECIAL_KEYS], np.int64)


def consume(directory: Path, converter_python: str) -> None:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    from tools.diffusers_wan_reference import float64

    source = fetch()
    program = build(source)
    export_dir, gguf = directory / "export", directory / "model-f32.gguf"
    subprocess.run([converter_python, str(source / "convert_hf_to_gguf.py"), str(export_dir),
                    "--outtype", "f32", "--outfile", str(gguf)], check=True)
    (directory / "ids.bin").write_bytes(IDS.tobytes())
    subprocess.run([str(program), str(gguf), str(directory / "ids.bin"), str(directory / "logits.bin")],
                   check=True, capture_output=True)
    theirs = np.fromfile(directory / "logits.bin", np.float32).reshape(1, IDS.size, FIELDS["vocab_size"])
    with float64():
        model = AutoModelForCausalLM.from_pretrained(export_dir, dtype=torch.float64,
                                                     attn_implementation="eager").eval()
        with torch.no_grad():
            truth = model(torch.from_numpy(IDS[None]).long()).logits.numpy()
    converter = subprocess.run([converter_python, "-c", "import torch, transformers; "
                                "print(torch.__version__, transformers.__version__)"],
                               check=True, capture_output=True, text=True).stdout.split()
    meta = {"digests": digests(export_dir), "llama.cpp": COMMIT, "converter torch": converter[0],
            "converter transformers": converter[1], "truth transformers": transformers.__version__,
            "truth torch": torch.__version__}
    FIXTURE.mkdir(parents=True, exist_ok=True)
    pieces, special = gguf_vocabulary(source, gguf)
    arrays = {"ids": IDS[None], "llama_cpp.fp32": theirs, "transformers.fp64": truth, "gguf.pieces": pieces,
              "gguf.special": special, "meta": np.frombuffer(json.dumps(meta).encode(), np.uint8)}
    np.savez_compressed(FIXTURE / "reference.npz", **arrays)
    gap = float(np.sqrt(np.mean((theirs - truth) ** 2)))
    print(f"{FIXTURE / 'reference.npz'}: llama.cpp off float64 by {gap:.3g} (rms)")


def main() -> None:
    sys.path[:0] = [str(ROOT)]
    if sys.argv[1:] == ["tokenizer"]:
        write_tokenizer()
    elif sys.argv[1:2] == ["export"] and len(sys.argv) == 3:
        export(Path(sys.argv[2]) / "export")
        print(f"{sys.argv[2]}/export: {digests(Path(sys.argv[2]) / 'export')}")
    elif sys.argv[1:2] == ["consume"] and len(sys.argv) == 4:
        consume(Path(sys.argv[2]), sys.argv[3])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
