"""GGUF files read as the HF config and tensors transformers reads from them.

The offline cases write tiny GGUF files with gguf-py's own writer, in the
layout llama.cpp's converter produces: Llama's q_proj and k_proj rows in
llama.cpp's rotary order (convert_hf_to_gguf.py `LlamaModel.permute`,
restated in `permute` below), every other tensor as HF stores it
transposed into GGUF's reversed shape. The reference is transformers 5.16.1
reading the same file (`AutoModelForCausalLM.from_pretrained(..., gguf_file=)`);
both sides dequantize the same bytes with gguf-py, so tensors agree exactly.

The network case reads bartowski's SmolLM2-135M-Instruct GGUF files at a
pinned commit against transformers over the same file.
"""

import os
import sys

import numpy as np
import pytest

from dew.interop import gguf as dew_gguf, load_pretrained

gguf = pytest.importorskip("gguf")
torch = pytest.importorskip("torch")

HIDDEN, HEADS, KV_HEADS, FFN, LAYERS, VOCAB = 256, 4, 2, 256, 2, 256
"""Every matrix's row is 256 values wide, one K-quant super-block."""


def permute(weight: np.ndarray, heads: int) -> np.ndarray:
    """llama.cpp's converter reordering HF q_proj/k_proj rows (`LlamaModel.permute`)."""
    return (weight.reshape(heads, 2, weight.shape[0] // heads // 2, *weight.shape[1:])
            .swapaxes(1, 2).reshape(weight.shape))


def head_dim(architecture: str) -> int:
    """Qwen3Config's head_dim, which transformers' GGUF table never reads, is 128."""
    return 128 if architecture == "qwen3" else HIDDEN // HEADS


def hf_tensors(architecture: str, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """A random HF checkpoint of the tiny decoder, tied head, in [out, in] layout."""
    width = head_dim(architecture)
    shapes = {"model.embed_tokens.weight": (VOCAB, HIDDEN), "model.norm.weight": (HIDDEN,)}
    for layer in range(LAYERS):
        prefix = f"model.layers.{layer}."
        shapes |= {prefix + "self_attn.q_proj.weight": (HEADS * width, HIDDEN),
                   prefix + "self_attn.k_proj.weight": (KV_HEADS * width, HIDDEN),
                   prefix + "self_attn.v_proj.weight": (KV_HEADS * width, HIDDEN),
                   prefix + "self_attn.o_proj.weight": (HIDDEN, HEADS * width),
                   prefix + "mlp.gate_proj.weight": (FFN, HIDDEN),
                   prefix + "mlp.up_proj.weight": (FFN, HIDDEN),
                   prefix + "mlp.down_proj.weight": (HIDDEN, FFN),
                   prefix + "input_layernorm.weight": (HIDDEN,),
                   prefix + "post_attention_layernorm.weight": (HIDDEN,)}
        if architecture == "qwen2":
            shapes |= {prefix + f"self_attn.{name}.bias": (rows,) for name, rows in
                       (("q_proj", HEADS * width), ("k_proj", KV_HEADS * width),
                        ("v_proj", KV_HEADS * width))}
        if architecture == "qwen3":
            shapes |= {prefix + "self_attn.q_norm.weight": (width,),
                       prefix + "self_attn.k_norm.weight": (width,)}
    return {name: (rng.standard_normal(shape) * (0.05 if len(shape) == 2 else 1.0)
                   + (1.0 if name.endswith("norm.weight") else 0.0)).astype(np.float32)
            for name, shape in shapes.items()}


GGUF_NAMES = {"model.embed_tokens": "token_embd", "model.norm": "output_norm",
              "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k",
              "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output",
              "self_attn.q_norm": "attn_q_norm", "self_attn.k_norm": "attn_k_norm",
              "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down",
              "input_layernorm": "attn_norm", "post_attention_layernorm": "ffn_norm"}
"""llama.cpp's names, written out here as the file's author sees them."""


def gguf_name(hf_name: str) -> str:
    stem, _, suffix = hf_name.rpartition(".")
    if stem.startswith("model.layers."):
        layer, _, module = stem.removeprefix("model.layers.").partition(".")
        return f"blk.{layer}.{GGUF_NAMES[module]}.{suffix}"
    return f"{GGUF_NAMES[stem]}.{suffix}"


def q4_k(rows: int, columns: int, rng: np.random.Generator) -> np.ndarray:
    """Random Q4_K super-blocks: fp16 d and dmin, 12 packed scale bytes, 128 nibble bytes.

    gguf-py cannot quantize to Q4_K, so the bytes are drawn directly with
    small finite fp16 scales; any byte pattern is a valid Q4_K block.
    """
    blocks = rows * columns // 256
    scales = (rng.uniform(1e-3, 2e-3, (blocks, 2)).astype(np.float16)).view(np.uint8)
    rest = rng.integers(0, 256, (blocks, 140), dtype=np.uint8)
    return np.concatenate([scales, rest], axis=1).reshape(rows, columns // 256 * 144)


def byte_tokens() -> list[str]:
    """GPT-2's byte-level alphabet, one token per byte, with the last two bytes'
    slots given to the two merges' products, so any text without them encodes.
    Two, because a one-element GGUF array reads back as a scalar."""
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    return [bytes_to_unicode()[index] for index in range(VOCAB - 2)] + ["lo", "llo"]


def write_gguf(path, architecture: str, qtype, tensors: dict[str, np.ndarray],
               rng: np.random.Generator, *, extra: tuple[str, ...] = (), scaling: float | None = None) -> None:
    """Write `tensors` as llama.cpp would: metadata, a byte-level BPE vocabulary, and
    every matrix in `qtype` (norms and biases stay F32, as llama.cpp keeps them)."""
    writer = gguf.GGUFWriter(str(path), architecture)
    writer.add_block_count(LAYERS)
    writer.add_context_length(128)
    writer.add_embedding_length(HIDDEN)
    writer.add_feed_forward_length(FFN)
    writer.add_head_count(HEADS)
    writer.add_head_count_kv(KV_HEADS)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_rope_freq_base(10000.0)
    writer.add_vocab_size(VOCAB)
    if scaling is not None:
        writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
        writer.add_rope_scaling_factor(scaling)
    if architecture == "llama":
        writer.add_rope_dimension_count(HIDDEN // HEADS)
    if architecture == "qwen3":
        writer.add_key_length(head_dim(architecture))
        writer.add_value_length(head_dim(architecture))
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(byte_tokens())
    writer.add_token_types([1] * VOCAB)
    writer.add_token_merges(["l o", "l lo"])
    writer.add_bos_token_id(0)
    writer.add_eos_token_id(1)
    for name, value in tensors.items():
        if architecture == "llama" and ".q_proj." in name:
            value = permute(value, HEADS)
        elif architecture == "llama" and ".k_proj." in name:
            value = permute(value, KV_HEADS)
        if value.ndim == 1 or qtype == gguf.GGMLQuantizationType.F32:
            writer.add_tensor(gguf_name(name), value)
        elif qtype == gguf.GGMLQuantizationType.Q4_K:
            writer.add_tensor(gguf_name(name), q4_k(*value.shape, rng), raw_dtype=qtype)
        else:
            writer.add_tensor(gguf_name(name), gguf.quants.quantize(value, qtype), raw_dtype=qtype)
    for name in extra:
        writer.add_tensor(name, np.ones(HIDDEN // HEADS // 2, np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def transformers_tensors(directory, name: str) -> dict[str, np.ndarray]:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(directory), gguf_file=name, dtype=torch.float32)
    return {key: value.numpy() for key, value in model.state_dict().items()}


CASES = [("llama", "F32"), ("llama", "Q8_0"), ("llama", "Q4_K"), ("qwen2", "Q8_0"),
         ("qwen3", "Q8_0")]


@pytest.mark.parametrize("architecture, qtype", CASES, ids=[f"{a}-{q}" for a, q in CASES])
def test_tensors_are_the_ones_transformers_reads(tmp_path, architecture, qtype):
    """Names, un-permuted Q/K rows and dequantized values against transformers'
    own GGUF load of the same file. A swapped reshape axis, the query head
    count used for k_proj, or an un-permute applied to Qwen (llama.cpp
    reorders only Llama) each moves rows, and the equality fails."""
    rng = np.random.default_rng(0)
    source = hf_tensors(architecture, rng)
    write_gguf(tmp_path / "model.gguf", architecture, gguf.GGMLQuantizationType[qtype], source, rng)

    config, ours = dew_gguf.read(tmp_path / "model.gguf")
    theirs = transformers_tensors(tmp_path, "model.gguf")

    # transformers' state dict holds the tied head twice; the file holds it once.
    assert set(ours) == set(theirs) - {"lm_head.weight"}
    for name, value in ours.items():
        np.testing.assert_array_equal(np.asarray(value, np.float32), theirs[name], err_msg=name)
    if qtype == "F32":
        for name, value in source.items():
            np.testing.assert_array_equal(ours[name], value, err_msg=name)
    assert config["tie_word_embeddings"] is True


@pytest.mark.parametrize("architecture", ["llama", "qwen3"])
def test_the_config_is_the_one_transformers_builds(tmp_path, architecture):
    """The file's metadata over the config class's defaults: Qwen 3's head_dim
    comes from Qwen3Config, not from the file."""
    from transformers import AutoConfig

    rng = np.random.default_rng(0)
    write_gguf(tmp_path / "model.gguf", architecture, gguf.GGMLQuantizationType.Q8_0,
               hf_tensors(architecture, rng), rng)

    assert dew_gguf.read(tmp_path / "model.gguf")[0] == AutoConfig.from_pretrained(
        str(tmp_path), gguf_file="model.gguf").to_diff_dict()


def test_load_pretrained_reads_the_file_and_its_tokenizer(tmp_path):
    """The whole load: logits against transformers over the same GGUF, and the
    tokenizer the file carries, since the directory ships none."""
    from transformers import AutoModelForCausalLM

    rng = np.random.default_rng(0)
    write_gguf(tmp_path / "model.gguf", "llama", gguf.GGMLQuantizationType.Q8_0,
               hf_tensors("llama", rng), rng)

    pretrained = load_pretrained(tmp_path, dtype="float32", attention_impl="reference",
                                 gguf_file="model.gguf")
    reference = AutoModelForCausalLM.from_pretrained(str(tmp_path), gguf_file="model.gguf",
                                                     dtype=torch.float32)
    reference.set_attn_implementation("eager")
    ids = rng.integers(0, VOCAB, (2, 12))
    ours = np.asarray(pretrained.model.apply(pretrained.variables, ids), np.float32)
    with torch.no_grad():
        theirs = reference(input_ids=torch.from_numpy(ids)).logits.numpy()

    # Twice the measured 6.2e-6 (CPU), logits of magnitude 4: fp32 rounding.
    np.testing.assert_allclose(ours, theirs, atol=1.3e-5, rtol=0)
    assert pretrained.processor is not None, "the file's tokenizer did not become the processor"
    tokens = np.asarray(pretrained.processor("Hello, GGUF").tokens)
    # The file's merges reached the processor: "llo" is the second merge's product.
    assert VOCAB - 1 in tokens.tolist()[0]
    assert pretrained.processor.decode(tokens) == ["Hello, GGUF"]
    assert pretrained.revision is None


def test_the_files_tokenizer_loads_without_torch(tmp_path):
    """GGUF repos ship no tokenizer files, and transformers' own GGUF route
    to the tokenizer needs torch; the gguf extra alone loads one, in a process
    where torch cannot import, and it tokenizes as transformers' does."""
    import subprocess

    from transformers import AutoTokenizer

    rng = np.random.default_rng(0)
    write_gguf(tmp_path / "model.gguf", "llama", gguf.GGMLQuantizationType.Q8_0, hf_tensors("llama", rng), rng)
    expected = AutoTokenizer.from_pretrained(str(tmp_path), gguf_file="model.gguf")("Hello, GGUF").input_ids
    script = """
import sys
sys.modules["torch"] = None
import numpy as np
from dew.interop import load_pretrained
loaded = load_pretrained(sys.argv[1], dtype="float32", attention_impl="reference", gguf_file="model.gguf")
print(np.asarray(loaded.processor("Hello, GGUF").tokens).tolist()[0])
"""
    run = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True,
                         env={**os.environ, "JAX_PLATFORMS": "cpu"})
    assert run.returncode == 0, run.stderr[-1500:]
    assert run.stdout.split("\n")[-2] == str(expected)


def test_metadata_the_table_does_not_read_is_refused(tmp_path):
    """llama.cpp writes a linear or YaRN rope scaling under `rope.scaling.*`,
    which transformers' table drops; the model would rotate unscaled."""
    rng = np.random.default_rng(0)
    write_gguf(tmp_path / "model.gguf", "llama", gguf.GGMLQuantizationType.F32, hf_tensors("llama", rng), rng,
               scaling=4.0)

    with pytest.raises(ValueError, match=r"llama\.rope\.scaling\.type='linear'.*base_model"):
        dew_gguf.read(tmp_path / "model.gguf")


def test_a_tensor_with_no_hf_name_is_refused(tmp_path):
    """Llama 3's rope_freqs carries a frequency scaling the metadata does not
    state; dropping it would load a different rotary embedding."""
    rng = np.random.default_rng(0)
    write_gguf(tmp_path / "model.gguf", "llama", gguf.GGMLQuantizationType.F32,
               hf_tensors("llama", rng), rng, extra=("rope_freqs.weight",))

    with pytest.raises(ValueError, match=r"'rope_freqs.weight' has no counterpart.*base_model"):
        dew_gguf.read(tmp_path / "model.gguf")


def test_an_unread_architecture_is_refused(tmp_path):
    rng = np.random.default_rng(0)
    write_gguf(tmp_path / "model.gguf", "gemma2", gguf.GGMLQuantizationType.F32, {}, rng)

    with pytest.raises(ValueError, match=r"architecture 'gemma2' is not read by Dew.*base_model"):
        dew_gguf.read(tmp_path / "model.gguf")


def test_a_missing_file_names_the_files_there(tmp_path):
    rng = np.random.default_rng(0)
    write_gguf(tmp_path / "model-Q8_0.gguf", "llama", gguf.GGMLQuantizationType.F32, {}, rng)

    with pytest.raises(FileNotFoundError, match=r"\['model-Q8_0.gguf'\]"):
        load_pretrained(tmp_path, gguf_file="model-Q4_K_M.gguf")


# Pinned: the bounds below are this commit's files.
REPO, REVISION = "bartowski/SmolLM2-135M-Instruct-GGUF", "09816acd5d99df7be770d85ea30822623dab342c"
PROMPTS = ("The Cascade Range runs from northern California through Oregon and Washington, "
           "and its tallest volcano is",
           "The capital of France is")
# Twice the worst fp32 residual measured over PROMPTS against transformers
# over the same file: 6.63e-5 CPU and 6.20e-5 GPU (Q8_0), 1.55e-4 CPU and
# 8.77e-5 GPU (Q4_K_M), the GPU at the suite's highest matmul precision.
# Both sides dequantize the same bytes with gguf-py, so what remains is the
# fp32 forward's rounding, which tests/test_released_checkpoints.py puts at
# 1.47e-4 for the SmolLM2-135M safetensors.
LOGITS = {"SmolLM2-135M-Instruct-Q8_0.gguf": 1.4e-4, "SmolLM2-135M-Instruct-Q4_K_M.gguf": 3.2e-4}


def available(name: str) -> bool:
    if os.environ.get("DEW_NETWORK_TESTS") == "1":
        return True
    from huggingface_hub import try_to_load_from_cache
    return isinstance(try_to_load_from_cache(REPO, name, revision=REVISION), str)


@pytest.mark.network
@pytest.mark.parametrize("name", list(LOGITS))
def test_released_gguf_logits_match_transformers(name):
    """Dew over the released file against transformers over the same file, fp32."""
    from transformers import AutoModelForCausalLM

    if not available(name):
        pytest.skip(f"{REPO}/{name} at {REVISION[:8]} is neither cached nor DEW_NETWORK_TESTS=1")
    pretrained = load_pretrained(REPO, revision=REVISION, gguf_file=name, dtype="float32",
                                 attention_impl="reference", max_seq_len=64)
    reference = AutoModelForCausalLM.from_pretrained(REPO, revision=REVISION, gguf_file=name,
                                                     dtype=torch.float32)
    reference.set_attn_implementation("eager")

    assert pretrained.revision == REVISION
    assert pretrained.processor is not None, "the file's tokenizer did not become the processor"
    for text in PROMPTS:
        ids = np.asarray(pretrained.processor(text).tokens)
        assert pretrained.processor.decode(ids) == [text]
        ours = np.asarray(pretrained.model.apply(pretrained.variables, ids), np.float32)
        with torch.no_grad():
            theirs = reference(input_ids=torch.from_numpy(ids.astype(np.int64))).logits.numpy()
        assert np.array_equal(ours.argmax(-1), theirs.argmax(-1))
        difference = float(np.abs(ours - theirs).max())
        assert difference < LOGITS[name], f"max |logit difference| {difference:.3e}"
