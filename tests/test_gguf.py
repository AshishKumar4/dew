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

import gguf
import jax
import numpy as np
import pytest
import torch
from reference_error import assert_as_exact

from dew.interop import Pretrained, gguf as dew_gguf

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



def write_byte_vocabulary(writer) -> None:
    """The byte-level BPE vocabulary every non-Gemma file here carries (`byte_tokens`), its two
    merges and its special ids."""
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(byte_tokens())
    writer.add_token_types([1] * VOCAB)
    writer.add_token_merges(["l o", "l lo"])
    writer.add_bos_token_id(0)
    writer.add_eos_token_id(1)

def write_gguf(
    path,
    architecture: str,
    qtype,
    tensors: dict[str, np.ndarray],
    rng: np.random.Generator,
    *,
    extra: tuple[str, ...] = (),
    scaling: float | None = None,
) -> None:
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
    write_byte_vocabulary(writer)
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


def test_load_pretrained_reads_the_file_and_its_tokenizer(tmp_path):
    """The whole load: logits against transformers over the same GGUF, and the
    tokenizer the file carries, since the directory ships none. The directory
    also ships a diffusion pipeline's index, and the selected file is still
    what loads; a second payload selector beside it is refused."""
    from transformers import AutoModelForCausalLM

    rng = np.random.default_rng(0)
    write_gguf(tmp_path / "model.gguf", "llama", gguf.GGMLQuantizationType.Q8_0,
               hf_tensors("llama", rng), rng)
    (tmp_path / "model_index.json").write_text('{"_class_name": "FluxPipeline"}')

    with pytest.raises(ValueError):
        Pretrained.load(tmp_path, gguf_file="model.gguf", single_file="model.safetensors")
    pretrained = Pretrained.load(tmp_path, dtype="float32", attention_impl="reference",
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
    write_gguf(
        tmp_path / "model.gguf", "llama", gguf.GGMLQuantizationType.Q8_0, hf_tensors("llama", rng), rng
    )
    expected = AutoTokenizer.from_pretrained(str(tmp_path), gguf_file="model.gguf")("Hello, GGUF").input_ids
    script = """
import sys
sys.modules["torch"] = None
import numpy as np
from dew.interop import Pretrained
loaded = Pretrained.load(sys.argv[1], dtype="float32", attention_impl="reference", gguf_file="model.gguf")
print(np.asarray(loaded.processor("Hello, GGUF").tokens).tolist()[0])
"""
    run = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True,
                         env={**os.environ, "JAX_PLATFORMS": "cpu"})
    assert run.returncode == 0, run.stderr[-1500:]
    assert run.stdout.split("\n")[-2] == str(expected)


# Pinned: the bounds below are this commit's files.
REPO, REVISION = "bartowski/SmolLM2-135M-Instruct-GGUF", "09816acd5d99df7be770d85ea30822623dab342c"
_SELECTED = """
import json, sys
from dew.interop import Pretrained
from dew.lora import LoRA

loaded = Pretrained.load("dew/tiny-gguf", dtype="float32", attention_impl="reference", gguf_file="model.gguf")
adapted = loaded.adapt(LoRA(rank=2, modules=("q_proj",)), key=0)
adapted.adapter.save(adapted.variables, sys.argv[1])
print(json.dumps([loaded.revision, loaded.origin]))
"""


def test_an_adapter_on_a_gguf_file_chosen_out_of_a_hub_repo_names_no_base(tmp_path):
    """The load resolves the repo at a commit, but a reader of the repo's id alone,
    as PEFT reads an adapter's base, loads the repo's own weights, not the file a
    selector chose, so an adapter bound to the file names no base. The process
    runs offline over a Hub cache holding the file at one commit."""
    import json
    import subprocess

    from interop_support import hub_cache

    rng = np.random.default_rng(0)
    (tmp_path / "repo").mkdir()
    write_gguf(tmp_path / "repo" / "model.gguf", "llama", gguf.GGMLQuantizationType.Q8_0,
               hf_tensors("llama", rng), rng)
    commit = hub_cache(tmp_path / "hub", "dew/tiny-gguf", tmp_path / "repo")
    env = {**os.environ, "HF_HUB_CACHE": str(tmp_path / "hub"), "HF_HUB_OFFLINE": "1", "JAX_PLATFORMS": "cpu"}
    run = subprocess.run([sys.executable, "-c", _SELECTED, str(tmp_path / "adapter")], capture_output=True,
                         text=True, env=env)
    assert run.returncode == 0, run.stderr[-2000:]
    assert json.loads(run.stdout.split("\n")[-2]) == [commit, None]
    written = json.loads((tmp_path / "adapter" / "adapter_config.json").read_text())
    assert not {"base_model_name_or_path", "revision"} & set(written)


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


@pytest.mark.network
@pytest.mark.parametrize("name", list(LOGITS))
def test_released_gguf_logits_match_transformers(name):
    """Dew over the released file against transformers over the same file, fp32."""
    from transformers import AutoModelForCausalLM

    pretrained = Pretrained.load(REPO, revision=REVISION, gguf_file=name, dtype="float32",
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


# Gemma's and the MoE architectures' files: each written as llama.cpp's converter writes its source
# (conversion/gemma.py, conversion/qwen.py, conversion/llama.py), and loaded against that source.
_GEMMA2 = {"sliding_window": 4, "attn_logit_softcapping": 50.0, "final_logit_softcapping": 30.0,
           "query_pre_attn_scalar": 16}
_QWEN2_MOE = {"num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 48,
              "shared_expert_intermediate_size": 80, "norm_topk_prob": False}
SOURCES = {
    "gemma2": ("gemma2", "gemma2", _GEMMA2),
    # llama.cpp scales a 46-layer Gemma 2's queries by its width over its heads, not its head width.
    "gemma2-27b": ("gemma2", "gemma2", {**_GEMMA2, "num_hidden_layers": 46, "head_dim": 32}),
    "gemma3": ("gemma3", "gemma3_text", {"sliding_window": 4, "query_pre_attn_scalar": 16}),
    # A file without a sliding window attends every layer whole.
    "gemma3-global": ("gemma3", "gemma3_text", {"query_pre_attn_scalar": 16,
                                                "layer_types": ["full_attention"] * 2}),
    "qwen2moe": ("qwen2moe", "qwen2_moe", _QWEN2_MOE),
    # An older converter's file stores the shared expert gate's one row flat.
    "qwen2moe-flat": ("qwen2moe", "qwen2_moe", _QWEN2_MOE),
    "qwen3moe": ("qwen3moe", "qwen3_moe", {"num_experts": 4, "num_experts_per_tok": 2,
                                           "moe_intermediate_size": 48, "norm_topk_prob": True}),
    "mixtral": ("mixtral", "mixtral", {"num_local_experts": 4, "num_experts_per_tok": 2}),
}
"""Each case's file architecture and source: its HF model_type at the probe's tiny sizes
(`interop_support.tiny_config`, two layers) and the fields beside them, the config values its metadata
states or, where it states none, llama.cpp computes (Gemma's query scale divisor, the head width for
every size but the 27B models')."""

SOURCE_SEEDS = {"gemma2": 8, "gemma2-27b": 8, "gemma3": 8, "gemma3-global": 8, "qwen2moe": 8,
                "qwen2moe-flat": 8, "qwen3moe": 8, "mixtral": 8}
"""Each file's declared check-1 seeds for its logits against the source, sized by `seeds_for` on armada."""

LLAMA_CPP_NAMES = {
    "gemma": {"input_layernorm": "attn_norm", "post_attention_layernorm": "post_attention_norm",
              "pre_feedforward_layernorm": "ffn_norm", "post_feedforward_layernorm": "post_ffw_norm"},
    "moe": {"mlp.gate": "ffn_gate_inp", "mlp.shared_expert.gate_proj": "ffn_gate_shexp",
            "mlp.shared_expert.up_proj": "ffn_up_shexp", "mlp.shared_expert.down_proj": "ffn_down_shexp",
            "mlp.shared_expert_gate": "ffn_gate_inp_shexp", "block_sparse_moe.gate": "ffn_gate_inp"},
}
"""llama.cpp's names for the modules `GGUF_NAMES` does not hold, as the file's author sees them."""


def checkpoint_tensors(architecture: str, config, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """A random checkpoint of `config` under the source's own stored names, [out, in] layout: per
    expert, as the released MoE checkpoints store them."""
    hidden, heads, kv_heads, width = (config.hidden_size, config.num_attention_heads,
                                      config.num_key_value_heads, config.head_dim)
    shapes = {"model.embed_tokens.weight": (config.vocab_size, hidden), "model.norm.weight": (hidden,)}
    if not config.tie_word_embeddings:
        shapes["lm_head.weight"] = (config.vocab_size, hidden)
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}."
        shapes |= {prefix + "self_attn.q_proj.weight": (heads * width, hidden),
                   prefix + "self_attn.k_proj.weight": (kv_heads * width, hidden),
                   prefix + "self_attn.v_proj.weight": (kv_heads * width, hidden),
                   prefix + "self_attn.o_proj.weight": (hidden, heads * width),
                   prefix + "input_layernorm.weight": (hidden,),
                   prefix + "post_attention_layernorm.weight": (hidden,)}
        if architecture.startswith("gemma"):
            shapes |= {prefix + f"{norm}.weight": (hidden,)
                       for norm in ("pre_feedforward_layernorm", "post_feedforward_layernorm")}
        if architecture in ("gemma3", "qwen3moe"):
            shapes |= {prefix + f"self_attn.{norm}.weight": (width,) for norm in ("q_norm", "k_norm")}
        if architecture == "qwen2moe":
            shapes |= {prefix + f"self_attn.{name}.bias": (rows,) for name, rows in
                       (("q_proj", heads * width), ("k_proj", kv_heads * width),
                        ("v_proj", kv_heads * width))}
        if architecture.startswith("gemma"):
            shapes |= {prefix + f"mlp.{name}.weight": shape for name, shape in
                       (("gate_proj", (config.intermediate_size, hidden)),
                        ("up_proj", (config.intermediate_size, hidden)),
                        ("down_proj", (hidden, config.intermediate_size)))}
            continue
        if architecture == "mixtral":
            experts, router, ffn, names = config.num_local_experts, "block_sparse_moe.gate", \
                config.intermediate_size, ("w1", "w3", "w2")
            module = "block_sparse_moe.experts"
        else:
            experts, router, ffn, names = config.num_experts, "mlp.gate", config.moe_intermediate_size, \
                ("gate_proj", "up_proj", "down_proj")
            module = "mlp.experts"
        shapes |= {prefix + f"{router}.weight": (experts, hidden)}
        for expert in range(experts):
            gate, up, down = (prefix + f"{module}.{expert}.{name}.weight" for name in names)
            shapes |= {gate: (ffn, hidden), up: (ffn, hidden), down: (hidden, ffn)}
        if architecture == "qwen2moe":
            shared = config.shared_expert_intermediate_size
            shapes |= {prefix + "mlp.shared_expert.gate_proj.weight": (shared, hidden),
                       prefix + "mlp.shared_expert.up_proj.weight": (shared, hidden),
                       prefix + "mlp.shared_expert.down_proj.weight": (hidden, shared),
                       prefix + "mlp.shared_expert_gate.weight": (1, hidden)}
    def drawn(name, shape):
        if name.endswith("norm.weight") and architecture.startswith("gemma"):
            # Gemma's w, stored as 1 + w: multiples of 2**-10 under one half, so 1 + w and back are exact.
            return rng.integers(-400, 400, shape) / 1024
        if name.endswith("norm.weight"):
            return 1.0 + 0.3 * rng.standard_normal(shape)
        return rng.standard_normal(shape) * (0.1 if len(shape) == 2 else 1.0)
    return {name: drawn(name, shape).astype(np.float32) for name, shape in shapes.items()}


def llama_cpp_files(case: str, config, tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """`tensors` as llama.cpp's converter stores them, by their GGUF names: Gemma's norms plus one,
    Mixtral's Q/K rows reordered as Llama's, each MoE projection's experts stacked, and the shared
    expert gate as the Linear's [1, hidden] (an older converter's flat row where the case says so)."""
    architecture = SOURCES[case][0]
    stored: dict[str, np.ndarray] = {}
    stacked: dict[str, list[np.ndarray]] = {}
    projections = {"w1": "gate", "w3": "up", "w2": "down", "gate_proj": "gate", "up_proj": "up",
                   "down_proj": "down"}
    for name, value in tensors.items():
        stem, _, suffix = name.rpartition(".")
        layer, _, module = stem.removeprefix("model.layers.").partition(".")
        if architecture.startswith("gemma") and name.endswith("norm.weight"):
            value = value + 1
        if architecture == "mixtral" and module in ("self_attn.q_proj", "self_attn.k_proj"):
            value = permute(value, config.num_attention_heads if "q_proj" in module
                            else config.num_key_value_heads)
        if ".experts." in module:
            projection = projections[module.rpartition(".")[2]]
            stacked.setdefault(f"blk.{layer}.ffn_{projection}_exps.{suffix}", []).append(value)
            continue
        if module == "mlp.shared_expert_gate" and case.endswith("-flat"):
            value = value[0]
        if not stem.startswith("model.layers."):
            stored[f"{ {**GGUF_NAMES, 'lm_head': 'output'}[stem]}.{suffix}"] = value
            continue
        names = {**GGUF_NAMES, **LLAMA_CPP_NAMES["moe"],
                 **(LLAMA_CPP_NAMES["gemma"] if architecture.startswith("gemma") else {})}
        stored[f"blk.{layer}.{names[module]}.{suffix}"] = value
    return {**stored, **{name: np.stack(values) for name, values in stacked.items()}}


def write_source(directory, case: str, rng: np.random.Generator):
    """The source checkpoint at `directory` and its GGUF file beside it, `model.gguf`, both F32."""
    from interop_support import tiny_config
    from safetensors.numpy import save_file

    architecture, model_type, fields = SOURCES[case]
    config = tiny_config(model_type, **{"num_hidden_layers": 2, "pad_token_id": None,
                                        "tie_word_embeddings": architecture.startswith("gemma"), **fields})
    config.head_dim = config.head_dim or config.hidden_size // config.num_attention_heads
    config.save_pretrained(directory)
    tensors = checkpoint_tensors(architecture, config, rng)
    save_file(tensors, str(directory / "model.safetensors"))
    file_arch = "llama" if architecture == "mixtral" else architecture
    writer = gguf.GGUFWriter(str(directory / "model.gguf"), file_arch)
    writer.add_block_count(config.num_hidden_layers)
    writer.add_context_length(config.max_position_embeddings)
    writer.add_embedding_length(config.hidden_size)
    writer.add_feed_forward_length(config.intermediate_size)
    writer.add_head_count(config.num_attention_heads)
    writer.add_head_count_kv(config.num_key_value_heads)
    writer.add_layer_norm_rms_eps(config.rms_norm_eps)
    rope = config.rope_parameters
    writer.add_rope_freq_base(rope.get("full_attention", rope)["rope_theta"])
    writer.add_vocab_size(config.vocab_size)
    if architecture in ("gemma2", "gemma3", "qwen3moe"):
        writer.add_key_length(config.head_dim)
        writer.add_value_length(config.head_dim)
    if architecture.startswith("gemma") and "sliding_attention" in config.layer_types:
        writer.add_sliding_window(config.sliding_window)
    if architecture == "gemma2":
        writer.add_attn_logit_softcapping(config.attn_logit_softcapping)
        writer.add_final_logit_softcapping(config.final_logit_softcapping)
    if architecture == "mixtral":
        writer.add_rope_dimension_count(config.head_dim)
        writer.add_expert_count(config.num_local_experts)
        writer.add_expert_used_count(config.num_experts_per_tok)
    if architecture in ("qwen2moe", "qwen3moe"):
        writer.add_expert_count(config.num_experts)
        writer.add_expert_used_count(config.num_experts_per_tok)
        writer.add_expert_feed_forward_length(config.moe_intermediate_size)
    if architecture == "qwen2moe":
        writer.add_expert_shared_feed_forward_length(config.shared_expert_intermediate_size)
    if architecture.startswith("gemma"):
        # Gemma's vocabulary is a sentencepiece unigram: scored pieces beside the control tokens.
        writer.add_tokenizer_model("llama")
        writer.add_token_list(["<pad>", "<eos>", "<bos>", "<unk>"]
                              + [f"piece{index}" for index in range(VOCAB - 4)])
        writer.add_token_scores([0.0] * 4 + [-float(index) for index in range(VOCAB - 4)])
        writer.add_token_types([3, 3, 3, 2] + [1] * (VOCAB - 4))
        writer.add_unk_token_id(3)
        writer.add_bos_token_id(2)
        writer.add_eos_token_id(1)
        writer.add_pad_token_id(0)
    else:
        write_byte_vocabulary(writer)
    for name, value in llama_cpp_files(case, config, tensors).items():
        writer.add_tensor(name, value)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return tensors


@pytest.mark.parametrize("case", sorted(SOURCES))
def test_a_gemma_or_moe_file_loads_the_model_it_was_converted_from(tmp_path, case):
    """llama.cpp's conversion is undone: the file reads back as its source's tensors bit for bit (Gemma's
    norms one lower again, each expert's projections apart, Mixtral's Q/K rows in HF order), under the
    config its source states, and the loaded model's logits over drawn ids hold check 1 against
    transformers' run of the source."""
    from interop_support import clearing_selections, token_draw, torch_logits
    from transformers import AutoModelForCausalLM

    from tools.torch_precision import float64, float64_twin

    rng = np.random.default_rng(1607)
    source = write_source(tmp_path, case, rng)
    config, tensors = dew_gguf.read(tmp_path / "model.gguf")
    assert set(tensors) == set(source)
    for name, value in tensors.items():
        np.testing.assert_array_equal(np.asarray(value, np.float32), source[name], err_msg=name)
    _, model_type, fields = SOURCES[case]
    assert config["model_type"] == model_type
    for field, value in fields.items():
        assert config.get(field, value) == value, field

    loaded = Pretrained.load(tmp_path, dtype="float32", attention_impl="reference", gguf_file="model.gguf")
    def built(dtype):
        return AutoModelForCausalLM.from_pretrained(str(tmp_path), dtype=dtype, attn_implementation="eager",
                                                    experts_implementation="eager").eval()

    reference = built(torch.float32)
    wide = float64_twin(reference, lambda: built(torch.float64)).eval()
    forward = jax.jit(loaded.model.apply)

    draw = clearing_selections(loaded, token_draw(0x1607, VOCAB, (2, 12)), case)
    # The twin runs inside `float64`, which widens its code's casts to float32 (Gemma's norms).
    assert_as_exact(lambda ids: np.asarray(forward(loaded.variables, ids), np.float64),
                    torch_logits(reference), draw, f"{case} file logits", seeds=SOURCE_SEEDS[case],
                    truth_fn=torch_logits(wide, float64))
