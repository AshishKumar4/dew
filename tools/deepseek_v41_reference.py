"""DeepSeek-V4.1-Flash tiny fixture from the release's own inference code.

No transformers class models V4.1: the release ships its reference as
`inference/model.py` (with `engram.py`, `vision.py` and
`image_processor.py`) in the model repo. This tool downloads those files at
DEEPSEEK_V41_REVISION, installs `tools/deepseek_v41_kernels.py` as their
`kernel` module and runs `Transformer` in fp32 on CPU over dense weights.

Run it in its own environment, never the project's:

    uv venv ~/.cache/dew/reference-venvs/deepseek-v41 --python 3.12
    VIRTUAL_ENV=... uv pip install torch==2.10.0 numpy safetensors sympy \
        tokenizers transformers huggingface_hub tilelang==0.1.8
    PYTHONPATH=. ~/.cache/dew/reference-venvs/deepseek-v41/bin/python \
        tools/deepseek_v41_reference.py [--search N | --check-kernels]

`--check-kernels` compares the stand-ins with the release's tilelang kernels
on a CUDA GPU (`check_kernels`). `--search N` scores N seeds from `--seed`
(`search`); SEED is the best of the first 119.

It writes tests/fixtures/hf/deepseek-v41-tiny: the release's config.json
spelling at toy width, a model.safetensors under the release's tensor names,
the tokenizer the engram hash reads, source.json and reference.npz with

    input_ids                  [2, 16] prompt rows
    logits                     full-sequence logits
    loss, learning_rate        next-token cross entropy, one SGD step's rate
    updated_logits             logits after that step with torch's gradients
    decode_prompt              the prefill length of the cached run
    prompt_logits              that prefill's logits
    decode_logits              teacher-forced one-token steps after it
    generated                  greedy ids after the prefill
    draft_ids, draft_logits,   DSpark's forward_spec at every decode step
    draft_confidence
    qat_logits                 the full forward with the quantizers on

every output but the last with the quantizers off (`run` says why).
source.json records the smallest rounding and top-k margins each met.

The model's two fp32 departures from the bf16 release are stated where they
happen: the engram lookup keeps the model's dtype where the release casts
to its bf16 (model.py:320), and every quantizer passes its gradient
straight through (the paper's QAT, section 2.4.4).
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO = "deepseek-ai/DeepSeek-V4.1-Flash"
DEEPSEEK_V41_REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
CODE = ("inference/model.py", "inference/engram.py", "inference/vision.py",
        "inference/image_processor.py")
ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "hf" / "deepseek-v41-tiny"
TOKENIZER = ROOT / "tests" / "fixtures" / "tokenizers" / "tiny-tools"
SEED = 81

# DeepSeek-V4.1-Flash at toy width, one entry per ModelArgs field the release
# sets (inference/config.json). The released stack is two sliding layers, 18
# layers at ratio 2 whose Full-mode sources are layers 2, 8 and 14, and 20
# layers at ratio 1 whose one Full layer (20, the first decoder layer and the
# candidate source) is followed by Reindex layers 24, 28, 32 and 36 with
# Reuse between; three DSpark stages read layers 37-39. The toy stack keeps
# every transition: sliding 0-1, ratio 2 with Full 2 and 4 and Reuse 3 and 5
# (so a Reuse layer reads the latest of two sources), ratio 1 with Full 6,
# Reindex 8 and 10 and Reuse 7, 9 and 11, engram on 1 and 4 (the release's
# 1 and 14: a sliding layer and a Full one), DSpark over 9-11.
#
# Width: head_dim 32 is the narrowest the quantizers allow (FP8 over blocks
# of 32 on the window, FP4 over 16 on compressed KV and over 32 on the
# indexer); rope_head_dim 8 leaves YaRN four frequencies, so both branches
# of its ramp move one. The indexer keeps 4 of up to 16 entries per query
# and the candidate pool of 3 blocks of 2 is smaller than a 16-token context
# and no smaller than the 4 kept, the release's invariant (2048 * 8 >= 512).
TINY = dict(
    max_batch_size=2, max_seq_len=32, temperature=0.0, dtype="bf16", expert_dtype=None,
    vocab_size=384, dim=64, moe_inter_dim=32, n_layers=12, n_mtp_layers=3, n_heads=4,
    n_routed_experts=8, n_shared_experts=1, n_activated_experts=2, score_func="sqrtsoftplus",
    route_scale=1.5, swiglu_limit=2.0, q_lora_rank=16, head_dim=32, rope_head_dim=8,
    norm_eps=1e-20, o_groups=2, o_lora_rank=8, window_size=4,
    compress_ratios=(0, 0, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0),
    kv_source_layers=(2, 4, 6), index_source_layers=(2, 4, 6, 8, 10),
    compress_rope_theta=160000.0, original_seq_len=65536, rope_theta=10000.0,
    rope_factor=16, beta_fast=32, beta_slow=1,
    index_n_heads=8, index_head_dim=32, index_topk=4,
    candidate_source_layer=6, candidate_topk_blocks=3, candidate_block_size=2,
    hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
    engram_layer_ids=(1, 4), engram_max_ngram_size=4, engram_vocab_size=50,
    engram_n_heads=2, engram_head_dim=32, engram_pad_id=2,
    dspark_block_size=5, dspark_noise_token_id=383, dspark_target_layer_ids=(9, 10, 11),
    dspark_markov_rank=16, dspark_n_routed_experts=4, dspark_n_activated_experts=2,
)
LENGTH = 16
PROMPT = 8
LEARNING_RATE = 0.05


def reference_code() -> Path:
    """The pinned inference directory, downloaded once into the hub cache."""
    from huggingface_hub import hf_hub_download

    paths = [Path(hf_hub_download(REPO, name, revision=DEEPSEEK_V41_REVISION)) for name in CODE]
    return paths[0].parent


def import_reference():
    """`model` and `engram` from the pinned code, over the torch kernels."""
    sys.modules["kernel"] = importlib.import_module("tools.deepseek_v41_kernels")
    directory = str(reference_code())
    if directory not in sys.path:
        sys.path.insert(0, directory)
    model = importlib.import_module("model")
    engram = importlib.import_module("engram")
    model.ParallelEngramEmbedding.forward = _engram_lookup
    if not getattr(model.Indexer.forward, "publishes", False):
        model.Indexer.forward = _publishing_indexer(model)
    return model, engram


def _publishing_indexer(model):
    """Indexer.forward (model.py:527-580) with its keys published on every call.

    The release publishes an index-key owner's cache only when this call
    closed a compressed group (model.py:537-548), where the compressed KV is
    published on every call (:748). A one-token decode step that closes no
    ratio-2 group therefore scores against the last keys any layer published,
    the ratio-1 layers' of the step before, and selects different entries
    than the prefill over the same tokens does; the paper's Full mode reads
    its own keys (section 2.3.1). Publishing the owner's cache first restores
    that, and the tool's cached decode then reproduces its full forward.
    """
    forward = model.Indexer.forward

    def publishing(self, x, qr, latent, start_pos, offset):
        if self.owns_k:
            model.shared_attn.index_k = self.k_cache
        return forward(self, x, qr, latent, start_pos, offset)

    publishing.publishes = True
    return publishing


def _engram_lookup(self, indices):
    """ParallelEngramEmbedding.forward (model.py:312-325) with its last cast
    to the table's dtype: the release casts to bf16, its model dtype."""
    values = torch.nn.functional.embedding(indices, self.weight)
    scales = torch.nn.functional.embedding(indices, self.scale)
    values = values.float().unflatten(-1, (-1, self.block_size)) * scales.float().unsqueeze(-1)
    return values.flatten(-2).to(self.weight.dtype)


def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER)


def build(model_module, engram_module, args, seed: int):
    """The toy Transformer in fp32 with its weights drawn, and its dense state dict."""
    torch.manual_seed(seed)
    net = model_module.Transformer(args, tokenizer()).float()
    generator = torch.Generator().manual_seed(seed)
    for name, tensor in net.named_parameters():
        tensor.requires_grad_(requires_grad=False)
        tensor.copy_(draw(name, tensor.shape, generator))
        # The engram table's dequantization scales are storage, held at one
        # over the dense table the fixture ships; they take no step.
        tensor.requires_grad_(not name.endswith("engram.embed.scale"))
    net.head.forward = functools.partial(_full_head, net.head)
    hashes = net.engram_hash
    # NgramHashState.forward runs under inference_mode (engram.py:153); its
    # integer ids leave it as ordinary tensors so autograd can index with them.
    hashes.forward = lambda *inputs: type(hashes).forward(hashes, *inputs).clone()
    return net


def _full_head(head, x, full_logits=True):
    return type(head).forward(head, x, full_logits=True)


def draw(name: str, shape, generator) -> torch.Tensor:
    """One parameter's toy values: every weight scaled to its fan-in, norms
    near one, and the mHC, sink and routing tensors in their trained ranges."""
    def normal(std):
        return torch.randn(tuple(shape), generator=generator) * std

    leaf = name.rsplit(".", 1)[-1]
    if name.endswith("engram.embed.scale"):
        return torch.ones(tuple(shape))
    if "norm" in name.rsplit(".", 2)[-2] or leaf in ("q_weight", "k_weight"):
        return 1 + normal(0.1)
    if leaf == "attn_sink":
        return normal(0.5)
    if leaf in ("hc_attn_fn", "hc_ffn_fn"):
        return normal(0.05)
    if leaf in ("hc_attn_base", "hc_ffn_base"):
        return normal(0.5)
    if leaf in ("hc_attn_scale", "hc_ffn_scale"):
        return 0.5 + torch.rand(tuple(shape), generator=generator)
    if name.endswith("gate.bias"):
        return normal(0.1)
    if leaf in ("image_start", "image_end", "image_newline"):
        return normal(1.0)
    if name == "embed.weight" or name.endswith("markov_head.embed.weight") or ".embed.weight" in name:
        return normal(1.0)
    return normal(shape[-1] ** -0.5)


def model_args(model_module, engram_module):
    """TINY with the engram sizes the release derives: each table holds the
    prime buckets its layout draws, and the hash multipliers derive from the
    tokenizer's compressed vocabulary (engram.py:95-127, :136-146)."""
    args = model_module.ModelArgs(**TINY)
    layout = engram_module.EngramLayout.from_args(
        model_module.ModelArgs(**{**TINY, "engram_num_embeddings": (0,) * len(TINY["engram_layer_ids"])}))
    args.engram_num_embeddings = tuple(sum(sum(per) for per in layer) for layer in layout.primes)
    args.engram_compressed_vocab_size = engram_module.build_compressed_token_map(tokenizer())[1]
    return args


def reset(net):
    """Detach the caches a previous graph wrote; prefill rewrites what it reads."""
    for buffer in net.buffers():
        buffer.detach_()
    net.engram_hash.cache.zero_()
    for block in list(net.layers) + list(net.mtp):
        attention = block.attn
        compressor = getattr(attention, "compressor", None)
        if compressor is not None and hasattr(compressor, "score_state"):
            compressor.kv_state.zero_()
            compressor.score_state.fill_(float("-inf"))


def forward(net, ids, start_pos=0):
    """Transformer.forward (model.py:1241-1272) under autograd, full logits."""
    return type(net).forward.__wrapped__(net, ids, start_pos)


def draft(net, ids, main_hidden, start_pos):
    return type(net).forward_spec.__wrapped__(net, ids, main_hidden, start_pos)


class Margins:
    """The smallest distance from a rounding or a selection boundary that the
    reference met, so the fixture can be held to a JAX port at fp32 noise.

    Quantization, per site (the FP8 window keys, the FP4 compressed entries,
    the FP4 index queries and keys): a scaled value's distance to the nearest
    midpoint between two representable values, over the spacing there, and
    each scale's distance to its own rounding step. Selection: every top-k's
    gap between its k-th and (k+1)-th finite value, over the values' scale.
    """

    def __init__(self, prefix: str = ""):
        self.prefix = prefix
        self.values: dict[str, float] = {}

    def note(self, key, value):
        value = float(value)
        if np.isfinite(value):
            key = self.prefix + key
            self.values[key] = min(self.values.get(key, np.inf), value)


MARGINS: Margins | None = None
E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _grid_margin(scaled: torch.Tensor, grid: torch.Tensor) -> float:
    magnitude = scaled.detach().abs().flatten().float()
    midpoints = (grid[1:] + grid[:-1]) / 2
    spacing = grid[1:] - grid[:-1]
    distance = (magnitude[:, None] - midpoints[None]).abs() / spacing[None]
    return float(distance.min()) if distance.numel() else np.inf


def _e4m3_grid() -> torch.Tensor:
    codes = torch.arange(0, 127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    return torch.unique(codes[torch.isfinite(codes)])


def _log2_margin(quotient: torch.Tensor) -> float:
    exponent = torch.log2(quotient.double())
    return float((exponent - exponent.round()).abs().min())


def _power_of_two_floor(value: torch.Tensor) -> torch.Tensor:
    return torch.exp2(torch.floor(torch.log2(value.double()))).float()


def set_quantizers(model_module, *, enabled: bool):
    """Point model.py's imported quantizers at the kernels, wrapped to report
    their margins, or at the identity: the fixture's architecture outputs run
    without quantization, its `qat_logits` with it (see `run`)."""
    kernels = sys.modules["kernel"]
    if not enabled:
        model_module.act_quant = lambda x, *args, **kwargs: x
        model_module.fp4_act_quant = lambda x, *args, **kwargs: x
        return
    e4m3 = _e4m3_grid()

    def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
        if MARGINS is not None:
            blocks = x.detach().float().unflatten(-1, (-1, block_size))
            quotient = blocks.abs().amax(-1, keepdim=True).clamp_min(1e-4) * torch.tensor(
                1 / 448, dtype=torch.float32)
            MARGINS.note("window_scale", _log2_margin(quotient))
            scale = kernels._power_of_two_ceil(quotient)
            MARGINS.note("window", _grid_margin((blocks / scale).clamp(-448, 448), e4m3))
        return kernels.act_quant(x, block_size, scale_fmt, scale_dtype, inplace)

    def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
        if MARGINS is not None:
            blocks = x.detach().float().unflatten(-1, (-1, block_size))
            amax = blocks.abs().amax(-1, keepdim=True)
            if scale_dtype == torch.float8_e4m3fn:
                site = "entries"
                quotient = amax.clamp_min(6 * 2 ** -9) / 6
                MARGINS.note("entries_scale",
                             _grid_margin(quotient / _power_of_two_floor(quotient), e4m3[e4m3 >= 1]))
                scale = quotient.to(torch.float8_e4m3fn).float()
            else:
                site = "index"
                quotient = amax.clamp_min(6 * 2 ** -126) * torch.tensor(1 / 6, dtype=torch.float32)
                MARGINS.note("index_scale", _log2_margin(quotient))
                scale = kernels._power_of_two_ceil(quotient)
            MARGINS.note(site, _grid_margin((blocks / scale).clamp(-6, 6), E2M1_GRID))
        return kernels.fp4_act_quant(x, block_size, inplace, scale_dtype)

    model_module.act_quant, model_module.fp4_act_quant = act_quant, fp4_act_quant


def install_selection_probe():
    """Wrap torch.Tensor.topk to report every selection's margin."""
    topk = torch.Tensor.topk
    if getattr(topk, "probed", False):
        return

    def probe_topk(self, k, dim=-1, largest=True, sorted=True):
        if MARGINS is not None and k < self.size(dim):
            ordered = self.detach().float().sort(dim=dim, descending=True).values
            kth = ordered.narrow(dim, k - 1, 1)
            after = ordered.narrow(dim, k, 1)
            finite = torch.isfinite(kth) & torch.isfinite(after)
            if finite.any():
                scale = ordered.masked_fill(~torch.isfinite(ordered), 0).abs().amax(
                    dim, keepdim=True).clamp_min(1e-6)
                MARGINS.note("selection", ((kth - after) / scale)[finite].min())
        return topk(self, k, dim=dim, largest=largest, sorted=sorted)

    probe_topk.probed = True
    torch.Tensor.topk = probe_topk


def cross_entropy(logits, ids):
    return torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1))


def run(seed: int, measure: bool):
    """Every reference output for one seed, and the margins it met.

    The architecture outputs (the forward, the update, the cached decode,
    greedy generation and DSpark's drafts) run with the quantizers off: a
    fake-quantizer's rounding turns fp32 noise between two frameworks into
    whole steps wherever a value sits near a rounding boundary, and across
    every forward here some value always does. `qat_logits` runs the one
    full forward with them on, for a seed whose margins keep that forward
    clear of every boundary; the quantizers themselves are compared bit for
    bit elsewhere (tests/test_deepseek_v41.py).
    """
    global MARGINS
    model_module, engram_module = import_reference()
    install_selection_probe()
    args = model_args(model_module, engram_module)
    net = build(model_module, engram_module, args, seed)
    generator = torch.Generator().manual_seed(seed + 1)
    ids = torch.randint(3, args.vocab_size - 1, (2, LENGTH), generator=generator)
    margins = {}

    set_quantizers(model_module, enabled=True)
    MARGINS = Margins("qat_") if measure else None
    with torch.no_grad():
        reset(net)
        _, qat_logits, _ = forward(net, ids)
    if MARGINS is not None:
        margins.update(MARGINS.values)

    set_quantizers(model_module, enabled=False)
    MARGINS = Margins() if measure else None
    reset(net)
    _, logits, _ = forward(net, ids)
    loss = cross_entropy(logits, ids)
    loss.backward()
    parameters = {name: p for name, p in net.named_parameters() if p.requires_grad}
    gradients = {name: (torch.zeros_like(p) if p.grad is None else p.grad.detach().clone())
                 for name, p in parameters.items()}
    state = {name: p.detach().clone() for name, p in parameters.items()}

    with torch.no_grad():
        for name, p in parameters.items():
            p.sub_(LEARNING_RATE * gradients[name])
        reset(net)
        _, updated, _ = forward(net, ids)
        for name, p in parameters.items():
            p.copy_(state[name])

        decode_logits, draft_ids, draft_logits, draft_confidence = [], [], [], []
        reset(net)
        next_ids, prompt_logits, main_hidden = forward(net, ids[:, :PROMPT])
        draft(net, next_ids[:, -1], main_hidden, 0)
        for position in range(PROMPT, LENGTH):
            next_ids, step_logits, main_hidden = forward(net, ids[:, position:position + 1], position)
            decode_logits.append(step_logits[:, 0])
            drafted = draft(net, next_ids[:, -1], main_hidden, position)
            draft_ids.append(drafted[0])
            draft_logits.append(drafted[1])
            draft_confidence.append(drafted[2])

        reset(net)
        generated = []
        next_ids, _, _ = forward(net, ids[:, :PROMPT])
        token = next_ids[:, -1]
        for position in range(PROMPT, LENGTH):
            generated.append(token)
            next_ids, _, _ = forward(net, token[:, None], position)
            token = next_ids[:, -1]
    if MARGINS is not None:
        margins.update(MARGINS.values)
    MARGINS = None
    outputs = {
        "input_ids": ids.numpy().astype(np.int32),
        "logits": logits.detach().numpy(),
        "qat_logits": qat_logits.numpy(),
        "loss": np.float32(loss.item()),
        "learning_rate": np.float32(LEARNING_RATE),
        "updated_logits": updated.numpy(),
        "decode_prompt": np.int32(PROMPT),
        "prompt_logits": prompt_logits.numpy(),
        "decode_logits": torch.stack(decode_logits, 1).numpy(),
        "generated": torch.stack(generated, 1).numpy().astype(np.int32),
        "draft_ids": torch.stack(draft_ids, 1).numpy().astype(np.int32),
        "draft_logits": torch.stack(draft_logits, 1).numpy(),
        "draft_confidence": torch.stack(draft_confidence, 1).numpy(),
    }
    return net, state, outputs, margins, gradients


# The fp32 noise a margin has to clear, in its own units: a value that sits
# closer than this to a rounding or selection boundary can round or select
# differently in two fp32 implementations of the same arithmetic.
NOISE = {"qat_window": 2e-5, "qat_entries": 2e-5, "qat_index": 5e-6, "qat_selection": 2e-3,
         "selection": 5e-5}
SCALE_NOISE = 1e-4


def score(margins: dict[str, float]) -> float:
    """The smallest margin over the noise it has to clear (NOISE, and
    SCALE_NOISE for every `*_scale` margin); a seed scoring 1 or more keeps
    every quantizer and selection of the fixture clear of fp32 noise."""
    return min(value / (SCALE_NOISE if key.endswith("_scale") else NOISE[key])
               for key, value in margins.items())


def search(first: int, count: int):
    """Report every seed's margins and `score`, one JSON line each, then the
    seed that scores highest. SEED is that seed over `--seed 0 --search 119`."""
    best = None
    for seed in range(first, first + count):
        *_, margins, _ = run(seed, measure=True)
        line = {"seed": seed, "score": score(margins), **margins}
        print(json.dumps(line), flush=True)
        if best is None or line["score"] > best["score"]:
            best = line
    print(json.dumps({"best": best}), flush=True)


def official_kernels():
    """The release's tilelang kernels (inference/kernel.py at the pinned
    revision), under a name of their own beside the stand-in `kernel`."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO, "inference/kernel.py", revision=DEEPSEEK_V41_REVISION)
    spec = importlib.util.spec_from_file_location("deepseek_v41_tilelang_kernels", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.view(torch.int16) if x.dtype == torch.bfloat16 else x.view(torch.int32)


def _emulated_sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """sparse_attn as the kernel rounds it (kernel.py:363-387) for one block
    of at most 64 indices: the unnormalized probabilities enter the value
    product in bf16 (acc_s_cast), while their sum stays fp32."""
    batch = torch.arange(q.size(0), device=q.device)[:, None, None]
    valid = topk_idxs >= 0
    keys = kv[batch, topk_idxs.clamp_min(0).long()].float()
    logits = torch.einsum("bmhd,bmkd->bmhk", q.float(), keys) * softmax_scale
    logits = logits.masked_fill(~valid[:, :, None, :], float("-inf"))
    peak = logits.amax(-1, keepdim=True).clamp_min(-1e30)
    weights = torch.exp(logits - peak)
    total = weights.sum(-1, keepdim=True) + torch.exp(attn_sink.float()[None, None, :, None] - peak)
    return (torch.einsum("bmhk,bmkd->bmhd", weights.bfloat16().float(), keys) / total).to(q.dtype)


def check_kernels():
    """Compare tools/deepseek_v41_kernels.py with the tilelang kernels on CUDA.

    The kernels take bf16 activations only ('input X dtype expected
    bfloat16'), so the comparison runs on bf16 inputs over nine magnitude
    decades, from blocks under every amax floor to blocks whose E4M3 scale
    saturates, with an all-zero block and every E2M1 tie. The quantizers
    must match bit for bit. sparse_attn's stand-in keeps its probabilities
    in fp32, which defines the fp32 semantics the fixture runs; the kernel
    rounds them to bf16 before the value product, so the comparison
    reports the stand-in's difference and then the emulated kernel's, which
    has to stay within one bf16 ulp (accumulation order). hc_split_sinkhorn
    has to stay within one fp32 ulp of its unit-scale outputs.
    """
    official = official_kernels()
    kernels = sys.modules["kernel"] = importlib.import_module("tools.deepseek_v41_kernels")
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(0)
    failures = []

    def report(name, mismatches, total, **extra):
        print(json.dumps({"check": name, "mismatches": int(mismatches), "of": int(total), **extra}))
        if mismatches:
            failures.append(name)

    quantizers = {
        "act_quant fp8/32 ue8m0": lambda k, x: k.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu, True),
        "fp4_act_quant fp4/16 e4m3": lambda k, x: k.fp4_act_quant(x, 16, True, torch.float8_e4m3fn),
        "fp4_act_quant fp4/32 e8m0": lambda k, x: k.fp4_act_quant(x, 32, True),
    }
    magnitudes = (1e-6, 1e-4, 1e-2, 0.3, 3.0, 40.0, 3e2, 3e3, 3e4)
    blocks = []
    for magnitude in magnitudes:
        x = torch.randn(64, 512, device=device, generator=generator) * torch.rand(
            64, 1, device=device, generator=generator) * (3 * magnitude)
        blocks.append(x.bfloat16())
    x = torch.cat(blocks)
    x[3, :32] = 0
    saturating = int((x.float().unflatten(-1, (-1, 16)).abs().amax(-1) / kernels.FP4_MAX
                      > kernels.FP8_MAX).sum())
    for name, quantize in quantizers.items():
        theirs, ours = quantize(official, x.clone()), quantize(kernels, x.clone())
        report(name, (_bits(theirs) != _bits(ours)).sum(), x.numel(),
               **({"saturating_blocks": saturating} if "e4m3" in name else {}))

    grid = torch.tensor([0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6], device=device)
    ties = torch.cat([grid, -grid, grid + 1e-3, grid - 1e-3]).clamp(-6, 6).repeat(3)[:64]
    t = torch.zeros(2, 32, device=device)
    t.view(-1)[:64] = ties
    t[:, 0] = 6.0  # each block's amax, so its power-of-two scale is exactly one
    t = t.bfloat16()
    theirs, ours = (quantizers["fp4_act_quant fp4/32 e8m0"](k, t.clone()) for k in (official, kernels))
    report("fp4 E2M1 ties", (_bits(theirs) != _bits(ours)).sum(), t.numel())

    batch, queries, heads, width, keys, top = 2, 5, 16, 512, 40, 24
    q = torch.randn(batch, queries, heads, width, device=device, generator=generator).bfloat16()
    kv = torch.randn(batch, keys, width, device=device, generator=generator).bfloat16()
    sink = torch.randn(heads, device=device, generator=generator)
    idx = torch.randint(-1, keys, (batch, queries, top), device=device, generator=generator,
                        dtype=torch.int32)
    idx[0, 0] = -1  # a query with nothing to attend
    theirs = official.sparse_attn(q, kv, sink, idx, width ** -0.5).float()
    port = kernels.sparse_attn(q, kv, sink, idx, width ** -0.5).float()
    emulated = _emulated_sparse_attn(q, kv, sink, idx, width ** -0.5).float()
    ulp = torch.exp2(torch.floor(torch.log2(theirs.abs().clamp_min(2 ** -126))) - 7)
    residual = (emulated - theirs).abs()
    print(json.dumps({"check": "sparse_attn stand-in, fp32 probabilities",
                      "differing": int((port != theirs).sum()), "of": theirs.numel(),
                      "max_abs": float((port - theirs).abs().max()),
                      "max_output": float(theirs.abs().max())}))
    report("sparse_attn emulated bf16 probabilities, beyond one bf16 ulp",
           (residual > ulp).sum(), theirs.numel(), differing=int((residual > 0).sum()),
           max_abs=float(residual.max()))

    mixes = torch.randn(3, 7, 24, device=device, generator=generator)
    scale = torch.randn(3, device=device, generator=generator)
    base = torch.randn(24, device=device, generator=generator)
    theirs = official.hc_split_sinkhorn(mixes, scale, base, 4, 20, 1e-6)
    ours = kernels.hc_split_sinkhorn(mixes, scale, base, 4, 20, 1e-6)
    residual = max(float((a - b).abs().max()) for a, b in zip(theirs, ours, strict=True))
    report("hc_split_sinkhorn beyond one fp32 ulp", residual > 2 ** -23, 1, max_abs=residual)
    print(json.dumps({"device": torch.cuda.get_device_name(), "torch": torch.__version__}))
    if failures:
        raise SystemExit(f"the stand-ins depart from the kernels: {failures}")


# ModelArgs fields onto the release's config.json text_config spelling.
TEXT_CONFIG = {
    "vocab_size": "vocab_size", "dim": "hidden_size", "moe_inter_dim": "moe_intermediate_size",
    "n_layers": "num_hidden_layers", "n_heads": "num_attention_heads", "head_dim": "head_dim",
    "rope_head_dim": "qk_rope_head_dim", "q_lora_rank": "q_lora_rank", "o_lora_rank": "o_lora_rank",
    "o_groups": "o_groups", "swiglu_limit": "swiglu_limit", "norm_eps": "rms_norm_eps",
    "rope_theta": "rope_theta", "n_routed_experts": "n_routed_experts",
    "n_shared_experts": "n_shared_experts", "n_activated_experts": "num_experts_per_tok",
    "score_func": "scoring_func", "route_scale": "routed_scaling_factor",
    "window_size": "sliding_window", "compress_ratios": "compress_ratios",
    "compress_rope_theta": "compress_rope_theta", "kv_source_layers": "kv_source_layer_ids",
    "index_source_layers": "index_source_layer_ids", "index_n_heads": "index_n_heads",
    "index_head_dim": "index_head_dim", "index_topk": "index_topk",
    "candidate_source_layer": "candidate_source_layer_id",
    "candidate_topk_blocks": "candidate_topk_blocks", "candidate_block_size": "candidate_block_size",
    "hc_mult": "hc_mult", "hc_sinkhorn_iters": "hc_sinkhorn_iters", "hc_eps": "hc_eps",
    "engram_layer_ids": "engram_layer_ids", "engram_num_embeddings": "engram_num_embeddings",
    "engram_max_ngram_size": "engram_max_ngram_size", "engram_vocab_size": "engram_vocab_size",
    "engram_n_heads": "engram_n_heads", "engram_head_dim": "engram_head_dim",
    "engram_pad_id": "engram_pad_token_id",
    "engram_compressed_vocab_size": "engram_compressed_vocab_size",
    "n_mtp_layers": "num_nextn_predict_layers", "dspark_block_size": "dspark_block_size",
    "dspark_noise_token_id": "dspark_noise_token_id",
    "dspark_target_layer_ids": "dspark_target_layer_ids",
    "dspark_markov_rank": "dspark_markov_rank",
    "dspark_n_routed_experts": "dspark_n_routed_experts",
    "dspark_n_activated_experts": "dspark_num_experts_per_tok",
}


def config_json(args) -> dict:
    """The release's config.json (text half) for the toy arguments."""
    text = {"model_type": "deepseek_v41_text"}
    for field, key in TEXT_CONFIG.items():
        value = getattr(args, field)
        text[key] = list(value) if isinstance(value, tuple) else value
    text.update(
        num_key_value_heads=1, hidden_act="silu", attention_bias=False, attention_dropout=0.0,
        initializer_range=0.02, use_cache=True, tie_word_embeddings=False,
        max_position_embeddings=args.max_seq_len, topk_method="noaux_tc", norm_topk_prob=True,
        rope_scaling={"rope_type": "yarn", "factor": args.rope_factor, "beta_fast": args.beta_fast,
                      "beta_slow": args.beta_slow,
                      "original_max_position_embeddings": args.original_seq_len})
    return {
        "architectures": ["DeepseekV41ForCausalLM"], "model_type": "deepseek_v41",
        "dtype": "float32", "bos_token_id": 0, "eos_token_id": 1, "pad_token_id": 2,
        "image_token_id": 382, "text_config": text,
    }


def write(seed: int):
    from safetensors.torch import save_file

    net, state, outputs, margins, _ = run(seed, measure=True)
    FIXTURE.mkdir(parents=True, exist_ok=True)
    args = model_args(importlib.import_module("model"), importlib.import_module("engram"))
    assert args.engram_num_embeddings == tuple(
        layer.engram.embed.num_embeddings for layer in net.layers if layer.engram is not None)
    tensors = {name: tensor.contiguous() for name, tensor in state.items()
               if not name.endswith("engram.embed.scale")}
    save_file(tensors, FIXTURE / "model.safetensors", metadata={"format": "pt"})
    (FIXTURE / "config.json").write_text(json.dumps(config_json(args), indent=1) + "\n")
    (FIXTURE / "generation_config.json").write_text(json.dumps(
        {"bos_token_id": 0, "eos_token_id": 1, "pad_token_id": 2, "do_sample": False}, indent=1) + "\n")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(TOKENIZER / name, FIXTURE / name)
    np.savez(FIXTURE / "reference.npz", **outputs)
    print(json.dumps({"seed": seed, **margins}))
    import transformers

    (FIXTURE / "source.json").write_text(json.dumps({
        "released": {"repo": REPO, "revision": DEEPSEEK_V41_REVISION, "code": list(CODE)},
        "kernels": "tools/deepseek_v41_kernels.py",
        "torch": torch.__version__, "transformers": transformers.__version__,
        "seed": seed, "margins": margins, "tool": "tools/deepseek_v41_reference.py",
    }, indent=1) + "\n")
    print(f"{FIXTURE}: {len(tensors)} tensors, loss {outputs['loss']:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", type=int, default=0, help="score the margins of this many seeds")
    parser.add_argument("--seed", type=int, default=SEED, help="the fixture's seed, or the first searched")
    parser.add_argument("--check-kernels", action="store_true",
                        help="compare the torch stand-ins with the tilelang kernels on CUDA")
    options = parser.parse_args()
    torch.set_default_dtype(torch.float32)
    if options.check_kernels:
        check_kernels()
    elif options.search:
        search(options.seed, options.search)
    else:
        write(options.seed)


if __name__ == "__main__":
    main()
