#!/usr/bin/env python3
"""The released Dory as vLLM's own engine computes it, for tests/test_dory.py's release test.

No transformers class computes Dory; vLLM's model does (vllm-project/vllm#60149, open, at
55076950 of shan18/vllm). The pull request changes Python files alone, so vLLM's Python-only
build installs exactly its tree: the checkout over the precompiled wheel of its base commit
(`BASE`). On a GPU node, with Dew's environment in .venv:

    git clone https://github.com/shan18/vllm vllm-pr
    git -C vllm-pr checkout 55076950b6c155f384d6766d63563c3bc20a0e3f
    uv venv -p 3.12 vllm-env
    VIRTUAL_ENV=vllm-env VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_LOCATION=<WHEEL> \
        uv pip install -e vllm-pr ninja
    PYTHONPATH=tools vllm-env/bin/python tools/dory_vllm_reference.py record DIR
    PYTHONPATH=tools vllm-env/bin/python tools/dory_vllm_reference.py truth DIR
    PYTHONPATH=tools vllm-env/bin/python tools/dory_vllm_reference.py upstream DIR
    PYTHONPATH=tools vllm-env/bin/python tools/dory_vllm_reference.py orders DIR
    PYTHONPATH=tools vllm-env/bin/python tools/dory_vllm_reference.py model_orders DIR
    PYTHONPATH=tools vllm-env/bin/python tools/dory_vllm_reference.py sums DIR
    HF_TOKEN=... PYTHONPATH=tools vllm-env/bin/python tools/dory_vllm_reference.py publish DIR

`record` runs the engine once a `RUNS` entry, each in a process of its own: a loop count, a cache
mode and a dtype, set through `hf_overrides` as the model card sets them, with no prefix caching,
decoding greedily one request at a time with every step's log-probabilities over the whole
vocabulary (`logprobs=-1`): the sentence `PROBE_TEXT` for `STEPS` tokens, its prompt's rows too
(`prompt_logprobs=-1`), and `WINDOW` random ids, past the release's 1024-token sliding window,
for `WINDOW_STEPS`. `truth` runs tools/dory_reference.py's transcription of the model in float64
over the same tokens, prefilling the prompt and decoding each drawn token as the run's cache mode
keeps them, and writes the probe and its provenance into source.json. `upstream` runs vLLM's own
arithmetic on the same tokens, the pull request's model code in float64 as its own tests build it
(`widened_float`): the truth every row and entry is measured against, which the transcription,
the CPU tests' reference, is held to.

A row of the probe is one position's log-probabilities on `COLUMNS` columns, `RANDOM` drawn once
for every row, then the row's own, the union of vLLM's and float64's top `TOP` filled out by
float64's next best; and at `positions` the whole vocabulary.

One run is a coin flip at a few positions of every bf16 run, vLLM's too: a rounding there moves
the continuation's log-probabilities by whole units. So the release is held to vLLM over `ORDERS`
orders a batch (tests/reference_error.py's K-order rule), in `BATCHES` batches, each order a draw
of the exact symmetries both engines hold it under (`symmetry`: the residual stream's units, each
feed-forward's hidden units, each attention's heads and value dimensions), on the same prompts:
`orders` sets the engine's weights in place to each (`LLM.apply_model`), decoding the identity
run's tokens again through a logits processor that draws them (tools/dory_vllm_forcing.py) beside
the same raw log-probabilities. It keeps each
order's rows on their columns and each entry of every row's whole vocabulary's absolute distance
from float64 (`errors`); `model_orders` writes the model code's at the engine's precision on a
last-loop run's rows past the window (`SHARED`) in their place, and `sums` sums each batch's
entries. Order zero reproduces the probe's rows bit for bit. `publish` puts the probe and the
batches' sums in `DATASET`, at a revision source.json pins with each file's sha256 (`fetched`).
"""

from __future__ import annotations

import functools
import gc
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

FIXTURE = Path(__file__).resolve().parents[1] / 'tests' / 'fixtures' / 'hf' / 'dory-1.0-4b'
PULL = 'vllm-project/vllm#60149 at shan18/vllm 55076950b6c155f384d6766d63563c3bc20a0e3f'
BASE = '153a17166e5d07bc38184dedeea542d2e8885e55'
WHEEL = (f'https://wheels.vllm.ai/{BASE}/vllm-0.32.0rc1.dev307%2Bg153a17166-cp38-abi3-'
         'manylinux_2_28_x86_64.whl')
"""The pull request's base and its precompiled wheel, which its Python-only build installs under
the checkout's Python files."""
PROBE_TEXT = 'The capital of France is Paris, and the capital of Japan is'
STEPS = 256
WINDOW = 1100
WINDOW_STEPS = 64
CONTEXT = 2048
"""The engine's `max_model_len`, past both prompts and their continuations."""
MEMORY = 0.75
"""The share of the GPU the engine takes (`gpu_memory_utilization`): the rest holds the weights as
loaded, which `permute` reorders from."""
RUNS = (*((loops, mode, 'bfloat16') for loops in (1, 2, 4) for mode in ('per_loop', 'last_loop')),
        *((2, mode, 'float32') for mode in ('per_loop', 'last_loop')))
"""(loop count, cache mode, dtype): bf16, the release's, at one, two (its own) and four loops,
and fp32 at its own count, whose per-pass cache qualifies the float64 truth as vLLM's function."""
COLUMNS = 64
RANDOM = 32
TOP = 16
ORDERS_SEED = 11
ORDERS = 52
"""The orders a batch holds: tests/reference_error.py's ORDERS, which the release test checks."""
BATCHES = 4
"""The batches of `ORDERS` orders a run is recorded under, the first holding the identity, the rest
fresh: the element rule judges each entry on one and re-measures a flagged one on the others."""
DATASET = 'dewml/dory-1.0-4b-vllm-reference'
"""Where the probe and its orders live: source.json pins its revision and each file's sha256."""


def tag(key: str) -> str:
    """A run's name as a file name."""
    return key.replace('/', '-')


def name(loops: int, mode: str, dtype: str) -> str:
    return f'{loops}/{mode}/{dtype}'


def prompts(tokenizer_ids: list[int], vocab: int) -> dict[str, np.ndarray]:
    """The sentence's ids and `WINDOW` random ids."""
    return {'sentence': np.asarray(tokenizer_ids, np.int32),
            'window': np.random.default_rng(7).integers(1000, vocab, WINDOW).astype(np.int32)}


def dense(flat, vocab: int) -> np.ndarray:
    """vLLM's flat log-probabilities (`FlatLogprobs`), a `[positions, vocab]` float32 array."""
    rows = np.full((len(flat.start_indices), vocab), -np.inf, np.float32)
    ids, values = np.asarray(flat.token_ids), np.asarray(flat.logprobs, np.float32)
    for row, (start, end) in enumerate(zip(flat.start_indices, flat.end_indices, strict=True)):
        rows[row, ids[start:end]] = values[start:end]
    return rows


def engine(loops: int, mode: str, dtype: str, **options):
    """vLLM's engine over the release at `source.json`'s revision, its loop count and cache mode set
    as the model card sets them, without prefix caching."""
    from vllm import LLM

    source = json.loads((FIXTURE / 'source.json').read_text())
    return LLM(model=source['repo'], revision=source['revision'], dtype=dtype, max_model_len=CONTEXT,
               gpu_memory_utilization=MEMORY, max_logprobs=-1, enable_prefix_caching=False, seed=0,
               hf_overrides={'n_recurrent_loops': loops, 'recurrent_kv_cache_mode': mode}, **options)


def scored(llm, prompt: np.ndarray, steps: int, prompt_rows: bool, forced: np.ndarray | None = None):
    """`steps` tokens after `prompt`, greedy or the `forced` ones (`Forced`), and the log-probabilities
    over the whole vocabulary of each row: the prompt's but its first where `prompt_rows`, then a
    row a step, each row the position's before its token."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    params = SamplingParams(
        temperature=0.0, max_tokens=steps, ignore_eos=True, logprobs=-1, detokenize=False,
        prompt_logprobs=-1 if prompt_rows else None, flat_logprobs=True,
        extra_args=None if forced is None else {'forced': forced.tolist(), 'prompt': len(prompt)})
    (output,) = llm.generate([TokensPrompt(prompt_token_ids=prompt.tolist())], params, use_tqdm=False)
    drawn, vocab = output.outputs[0], llm.llm_engine.model_config.get_vocab_size()
    rows = dense(drawn.logprobs, vocab)
    if prompt_rows:
        # prompt_logprobs[i] scores token i, the logits at position i - 1; the first is None.
        rows = np.concatenate([dense(output.prompt_logprobs, vocab)[1:], rows])
    return np.asarray(drawn.token_ids, np.int32), rows


PARTS = (('sentence', STEPS, True), ('window', WINDOW_STEPS, False))
"""Each prompt, the tokens drawn after it, and whether its own rows are kept."""


def record_run(directory: Path, loops: int, mode: str, dtype: str) -> None:
    """One run's tokens and log-probabilities, as vLLM's engine computes them."""
    import torch
    import vllm

    llm = engine(loops, mode, dtype)
    ids = prompts(llm.get_tokenizer().encode(PROBE_TEXT), llm.llm_engine.model_config.get_vocab_size())
    arrays: dict[str, np.ndarray] = dict(ids)
    for part, steps, prompt_rows in PARTS:
        arrays[f'{part}_generated'], arrays[f'{part}_logprobs'] = scored(llm, ids[part], steps, prompt_rows)
    np.savez(directory / f'{tag(name(loops, mode, dtype))}.npz', **arrays)
    (directory / 'provenance.json').write_text(json.dumps({
        'reference': f"{PULL}, installed by vLLM's Python-only build over {WHEEL}",
        'vllm': vllm.__version__, 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name()}, indent=2) + '\n')


def record(directory: Path) -> None:
    """Every run, each in a process of its own: an engine holds its GPU until its process ends."""
    directory.mkdir(parents=True, exist_ok=True)
    for loops, mode, dtype in RUNS:
        subprocess.run([sys.executable, __file__, 'run', str(directory), str(loops), mode, dtype], check=True)
        print(f'recorded {name(loops, mode, dtype)}', flush=True)


def log_softmax(logits: np.ndarray) -> np.ndarray:
    top = logits.max(axis=-1, keepdims=True)
    return logits - top - np.log(np.exp(logits - top).sum(axis=-1, keepdims=True))


def own_columns(vllm_rows: np.ndarray, truth: np.ndarray, random: np.ndarray) -> np.ndarray:
    """Each row's own columns: the union of vLLM's and float64's top `TOP` outside `random`, filled
    out by float64's next best."""
    width = COLUMNS - RANDOM
    own = np.empty((len(truth), width), np.int32)
    taken = set(random.tolist())
    for row, (theirs, exact) in enumerate(zip(vllm_rows, truth, strict=True)):
        ranked = np.argsort(-exact)
        candidates = (*np.argsort(-theirs)[:TOP], *ranked[:TOP])
        chosen = list(dict.fromkeys(int(column) for column in candidates if column not in taken))
        chosen += [int(column) for column in ranked if column not in taken and column not in chosen][
            :width - len(chosen)]
        own[row] = chosen[:width]
    return own


def columns_of(probe, key: str) -> np.ndarray:
    """Each row's `COLUMNS` columns of run `key`: the shared random ones, then its own."""
    own = probe[f'{key}/columns']
    return np.concatenate([np.broadcast_to(probe['random'], (len(own), RANDOM)), own], axis=1)


def truth(directory: Path) -> None:
    """The float64 truth on every run's tokens, each row's columns, and the probe."""
    import torch
    from dory_reference import DoryReference

    config, tensors = release()
    random = np.random.default_rng(9).choice(config['vocab_size'], RANDOM, replace=False).astype(np.int32)
    probe: dict[str, np.ndarray] = {'random': random}
    reference = DoryReference(config, tensors, torch.float64, device='cuda')
    for loops, mode, dtype in RUNS:
        run = np.load(directory / f'{tag(name(loops, mode, dtype))}.npz')
        probe.update({key: run[key] for key in ('sentence', 'window')})
        reference.c.update(n_recurrent_loops=loops, recurrent_kv_cache_mode=mode)
        rows = log_softmax(np.concatenate([
            reference.forced(run['sentence'], run['sentence_generated']),
            reference.forced(run['window'], run['window_generated'], prompt_rows=False)]))
        theirs = np.concatenate([run['sentence_logprobs'], run['window_logprobs']])
        key = name(loops, mode, dtype)
        probe.update({f'{key}/sentence_generated': run['sentence_generated'],
                      f'{key}/window_generated': run['window_generated'],
                      f'{key}/columns': own_columns(theirs, rows, random)})
        columns, whole = columns_of(probe, key), positions(probe)
        probe.update({f'{key}/vllm': np.take_along_axis(theirs, columns, axis=1),
                      f'{key}/truth': np.take_along_axis(rows, columns, axis=1),
                      f'{key}/vllm_full': theirs[whole], f'{key}/truth_full': rows[whole]})
        print(f'{key}: {len(rows)} rows', flush=True)
    np.savez_compressed(FIXTURE / 'probe.npz', **probe)
    source = json.loads((FIXTURE / 'source.json').read_text())
    provenance = json.loads((directory / 'provenance.json').read_text())
    (FIXTURE / 'source.json').write_text(json.dumps({**source, 'probe': {
        **provenance, 'truth': 'tools/dory_reference.py in float64 on the same GPU',
        'recorder': 'tools/dory_vllm_reference.py'}}, indent=2) + '\n')
    print(f'{FIXTURE / "probe.npz"}: {len(probe)} arrays')


def release() -> tuple[dict, dict]:
    """The release's config and its tensors, as torch tensors on the host."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    source = json.loads((FIXTURE / 'source.json').read_text())
    folder = Path(snapshot_download(source['repo'], revision=source['revision'],
                                    allow_patterns=['config.json', '*.safetensors']))
    tensors = {key: value for path in sorted(folder.glob('*.safetensors'))
               for key, value in load_file(path).items()}
    return json.loads((folder / 'config.json').read_text()), tensors


def positions(probe) -> np.ndarray:
    """The rows every run keeps over the whole vocabulary: the sentence prompt's last, two decode
    steps after it (the first and the middle) and the last, past the window."""
    prompt = len(probe['sentence'])
    return np.asarray([prompt - 1, prompt, prompt + STEPS // 2, prompt + STEPS - 2 + WINDOW_STEPS])


@functools.cache
def reference_attention(tf32: bool = False):
    """The pull request's tests/models/test_dory.py ReferenceAttention: single-request causal GQA,
    sliding-window and decode caching, its positions on its queries' device. With `tf32`, a float32
    model's attention multiplies in TF32 as the engine's Triton attention does (its tl.dot's), its
    softmax in float32, while every other product stays at the engine's "highest"."""
    import torch

    def attend(q, keys, values, mask, scale):
        if not tf32:
            return torch.nn.functional.scaled_dot_product_attention(
                q, keys, values, attn_mask=mask, scale=scale, enable_gqa=True)
        # scaled_dot_product_attention's own arithmetic, its two products in TF32 alone.
        repeat = q.shape[0] // keys.shape[0]
        keys, values = keys.repeat_interleave(repeat, dim=0), values.repeat_interleave(repeat, dim=0)
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            scores = (q @ keys.transpose(-1, -2)) * scale
            weights = scores.masked_fill(~mask, float('-inf')).softmax(dim=-1)
            return weights @ values
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous

    class ReferenceAttention(torch.nn.Module):
        def __init__(self, num_heads, head_size, scale, num_kv_heads, **kwargs):
            super().__init__()
            self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
            self.head_size, self.scale = head_size, scale
            self.prefix = kwargs['prefix']
            self.window = kwargs['per_layer_sliding_window']
            self.keys = self.values = None

        def forward(self, q, k, v):
            q = q.view(-1, self.num_heads, self.head_size).transpose(0, 1)
            k = k.view(-1, self.num_kv_heads, self.head_size).transpose(0, 1)
            v = v.view(-1, self.num_kv_heads, self.head_size).transpose(0, 1)
            offset = self.start_pos
            self.keys = k if self.keys is None else torch.cat((self.keys[:, :offset], k), dim=1)
            self.values = v if self.values is None else torch.cat((self.values[:, :offset], v), dim=1)
            key_pos = torch.arange(self.keys.shape[1], device=q.device)
            query_pos = torch.arange(offset, offset + q.shape[1], device=q.device)[:, None]
            mask = key_pos <= query_pos
            if self.window is not None:
                mask &= key_pos > query_pos - self.window
            out = attend(q, self.keys, self.values, mask, self.scale)
            return out.transpose(0, 1).reshape(-1, self.num_heads * self.head_size)

    return ReferenceAttention


def widened_float():
    """vLLM's model code in float64 as Dew's float64 truth runs Dew: each of its casts to float32
    (`Tensor.float`, a float32 `torch.arange`) keeps a float64 operand float64, a narrower one
    rounding to float32 as it did."""
    import contextlib

    import torch

    @contextlib.contextmanager
    def widened():
        cast, arange = torch.Tensor.float, torch.arange

        def at_least_float32(tensor, *args, **kwargs):
            return tensor if tensor.dtype == torch.float64 else cast(tensor, *args, **kwargs)

        def wide_arange(*args, dtype=None, **kwargs):
            return arange(*args, dtype=torch.float64 if dtype == torch.float32 else dtype, **kwargs)

        torch.Tensor.float, torch.arange = at_least_float32, wide_arange
        try:
            yield
        finally:
            torch.Tensor.float, torch.arange = cast, arange

    return widened()


def model_code(config: dict, tensors: dict, loops: int, mode: str, dtype, tf32: bool = False,
               device: str = 'cuda'):
    """The pull request's DoryForCausalLM over the release in `dtype` (a torch dtype), built as its
    tests/models/test_dory.py builds it, with that test's reference attention in the engine's
    attention's place, under `dtype` as the engine's loader builds a model (its rotary tables take
    it), and a call that decodes: `rows(prompt, tokens, prompt_rows)` prefills `prompt` and decodes
    each of `tokens` but the last a step at a time, the log-probabilities of the prompt's rows (its
    last alone where `prompt_rows` is False) and a row a token, in at least float32. `tf32` takes
    the engine's TF32 attention products (`reference_attention`)."""
    from types import SimpleNamespace

    import torch
    from vllm.config import CompilationConfig, DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import parallel_state
    from vllm.model_executor.models import dory
    from vllm.transformers_utils.configs.dory import DoryConfig
    from vllm.utils.torch_utils import set_default_torch_dtype

    attention = reference_attention(tf32)
    parallel_state._TP = SimpleNamespace(world_size=1, rank_in_group=0)
    dory.Attention = attention
    vc = VllmConfig(device_config=DeviceConfig(device=device),
                    compilation_config=CompilationConfig(mode=0, custom_ops=['none']))
    hf = DoryConfig(**{**config, 'n_recurrent_loops': loops, 'recurrent_kv_cache_mode': mode})
    vc.model_config = SimpleNamespace(hf_config=hf, head_dtype=None, dtype=dtype)
    with set_current_vllm_config(vc), set_default_torch_dtype(dtype), torch.device(device):
        model = dory.DoryForCausalLM(vllm_config=vc).to(dtype)
        model.load_weights((key, value.to(dtype)) for key, value in tensors.items())
    # One process holds the whole vocabulary, as the test's logits check takes it.
    model.logits_processor._gather_logits = lambda logits: logits

    def start(module, args):
        # The runner's slot mapping for one request's contiguous tokens, as the test's.
        for cache in module.modules():
            if isinstance(cache, attention):
                cache.start_pos = int(args[1][0])

    model.register_forward_pre_hook(start)

    def rows(prompt: np.ndarray, tokens: np.ndarray, prompt_rows: bool) -> np.ndarray:
        for cache in model.modules():
            if isinstance(cache, attention):
                cache.keys = cache.values = None
        ids = torch.as_tensor(np.concatenate([prompt, tokens]), device=device)
        with set_current_vllm_config(vc), torch.inference_mode():
            hidden = model(ids[:len(prompt)], torch.arange(len(prompt), device=device))
            found = [model.compute_logits(hidden if prompt_rows else hidden[-1:])]
            for at in range(len(prompt), len(ids) - 1):
                hidden = model(ids[at:at + 1], torch.arange(at, at + 1, device=device))
                found.append(model.compute_logits(hidden))
            logits = torch.cat(found)
            # The engine's sampler takes log-probabilities in float32 (Sampler.compute_logprobs).
            wide = torch.promote_types(logits.dtype, torch.float32)
            return logits.log_softmax(dim=-1, dtype=wide).cpu().numpy()

    return model, rows


def upstream(directory: Path) -> None:
    """vLLM's own model code in float64 on every run's tokens, the truth every row and entry is
    measured against: `model_code` under `widened_float`, prefilling each prompt and decoding each
    drawn token as the run's cache mode keeps them. Each run's rows over the whole vocabulary
    (`directory/upstream-{run}.npy`), on its columns (`{key}/upstream`) and at `positions`
    (`{key}/upstream_full`)."""
    import torch

    config, tensors = release()
    probe = dict(np.load(FIXTURE / 'probe.npz'))
    with widened_float():
        for (loops, mode), runs in itertools.groupby(sorted(RUNS), key=lambda run: run[:2]):
            model, rows = model_code(config, tensors, loops, mode, torch.float64)
            for run in runs:
                key = name(*run)
                found = np.concatenate([rows(probe[part], probe[f'{key}/{part}_generated'], prompt_rows)
                                        for part, _, prompt_rows in PARTS])
                np.save(directory / f'upstream-{tag(key)}.npy', found)
                probe[f'{key}/upstream'] = np.take_along_axis(found, columns_of(probe, key), axis=1)
                probe[f'{key}/upstream_full'] = found[positions(probe)]
                print(f'{key}: upstream in float64', flush=True)
            # This loop count's model leaves the device before the next one's is built.
            del model, rows
            gc.collect()
            torch.cuda.empty_cache()
    np.savez_compressed(FIXTURE / 'probe.npz', **probe)


SHARED = tuple(run for run in RUNS if run[1] == 'last_loop')
"""The runs whose rows past the window take the model code in the run's dtype as the reference
(`model_orders`): there vLLM's engine parts from the model code's last-loop cache
(vllm-project/vllm#60149, issuecomment-6270937659), which the model code computes as specified."""


def model_orders(directory: Path) -> None:
    """For each `SHARED` run, the model code at the engine's precision (`model_code`: bf16
    throughout, or float32 with TF32 attention products) on the window's rows under every order of
    every batch (`symmetry`), in the engine's place in the run's files: each order's rows on their
    columns and each entry's error (`errors`)."""
    import torch

    config, tensors = release()
    probe = np.load(FIXTURE / 'probe.npz')
    for loops, mode, dtype in SHARED:
        key = name(loops, mode, dtype)
        window = slice(-len(probe[f'{key}/window_generated']), None)
        # The engine's precision: bf16 throughout, or float32 with its attention's products in TF32.
        model, rows = model_code(config, tensors, loops, mode, getattr(torch, dtype), tf32=dtype == 'float32')
        truth = np.load(directory / f'upstream-{tag(key)}.npy')[window]
        columns = columns_of(probe, key)[window]
        unpermuted = None
        for batch in range(BATCHES):
            kept = np.load(directory / f'columns-{tag(key)}-{batch}.npy', mmap_mode='r+')
            held = errors(directory, key, batch)
            for index in range(ORDERS):
                drawn = symmetry(config, batch, index)
                seen = permute(model, config, drawn)
                if unpermuted is None:
                    unpermuted = seen
                assert np.array_equal(seen, unpermuted[drawn['residual']]), (
                    f'{key}: the model holds no order {batch}/{index}')
                found = rows(probe['window'], probe[f'{key}/window_generated'], prompt_rows=False)
                kept[index, window] = np.take_along_axis(found, columns, axis=1)
                held[index, window] = np.abs(found - truth)
                print(f'{key} model code, order {batch}/{index}: '
                      f'{np.sqrt(np.mean(np.square(found - truth))):.4f}', flush=True)
            kept.flush()
            held.flush()
        del model, rows
        gc.collect()
        torch.cuda.empty_cache()


def symmetry(config: dict, batch: int, index: int) -> dict[str, np.ndarray]:
    """Order `index` of `batch`: a draw of the exact symmetries both engines hold the release under, each
    layer's its own. The residual stream's units (`residual` `[width]`); each MLP layer's hidden units
    (`mlp` `[MLP layers, intermediate]`); each attention layer's key-value heads (`kv` `[attention
    layers, kv heads]`, new slot i holding old head kv[i]), the query heads within each group (`within`
    `[attention layers, kv heads, group]`) and each key-value head's value dimensions (`value`
    `[attention layers, kv heads, head_dim]`). The first batch's first order is the identity. Seeded from
    (`ORDERS_SEED`, batch, index), so both engines and every stage draw it alike.

    Each reorders sums and nothing else: the residual every projection's and norm's over the width; the
    hidden units the down projection's and the hidden norm's; the heads and value dimensions the output
    projection's and the attention output's norm's. The queries' and keys' dimensions stay: their order
    is RoPE's frequency pairs, which neither engine's frequencies follow."""
    kinds = config['layer_types']
    attention = sum(kind != 'mlp' for kind in kinds)
    heads, kv, width = config['num_attention_heads'], config['num_key_value_heads'], config['head_dim']
    shapes = {'residual': (config['hidden_size'],),
              'mlp': (len(kinds) - attention, config['intermediate_size']),
              'kv': (attention, kv), 'within': (attention, kv, heads // kv), 'value': (attention, kv, width)}
    if batch == index == 0:
        return {part: np.broadcast_to(np.arange(shape[-1]), shape).copy() for part, shape in shapes.items()}
    rng = np.random.default_rng([ORDERS_SEED, batch, index])
    return {part: rng.permuted(np.broadcast_to(np.arange(shape[-1]), shape), axis=-1)
            for part, shape in shapes.items()}


def maps(config: dict, drawn: dict[str, np.ndarray]) -> dict[str, dict[int, np.ndarray]]:
    """`drawn`'s index maps by layer, each new position's old position along its axis: `query` (a
    query's own dimensions, heads moved), `output` (an attention output's, heads moved and dimensions
    as their values'), `key` and `value` over the key-value heads, `mlp` over a feed-forward's hidden
    units."""
    kinds = config['layer_types']
    heads, kv, width = config['num_attention_heads'], config['num_key_value_heads'], config['head_dim']
    group = heads // kv
    parts = ('query', 'output', 'key', 'value', 'mlp')
    found: dict[str, dict[int, np.ndarray]] = {part: {} for part in parts}
    attention = mlp = 0
    for layer, kind in enumerate(kinds):
        if kind == 'mlp':
            found['mlp'][layer] = drawn['mlp'][mlp]
            mlp += 1
            continue
        old_kv, within, value = drawn['kv'][attention], drawn['within'][attention], drawn['value'][attention]
        # New query slot j, in group j // group, holds the old query head of its key-value head's group.
        old_query = (old_kv[:, None] * group + within).reshape(-1)
        dims = np.arange(width)
        found['query'][layer] = (old_query[:, None] * width + dims).reshape(-1)
        found['output'][layer] = (old_query[:, None] * width + np.repeat(value, group, axis=0)).reshape(-1)
        found['key'][layer] = (old_kv[:, None] * width + dims).reshape(-1)
        found['value'][layer] = (old_kv[:, None] * width + value).reshape(-1)
        attention += 1
    return found


TRACED = ((1, 'per_loop', 'bfloat16'), (4, 'last_loop', 'bfloat16'))
"""The runs whose Dew per-order errors `dew_traces` keeps for the element rule's calibration: the run
whose one draw parted from vLLM's, and the one with the heaviest tails."""
TRACED_BATCHES = 8


def dew_traces(directory: Path) -> None:
    """Dew's per-order errors on each `TRACED` run, `TRACED_BATCHES` batches of `symmetry`'s orders,
    as the release test computes them (tests/test_dory.py), each `[ORDERS, rows, vocab]` float32 in
    `directory/dew-errors-{run}-{batch}.npy`, kept where they are recorded for the element rule's
    calibration: a split of Dew's own batches is a null the rule must pass, under Dew's environment."""
    import jax
    import test_dory

    from dew.interop import Pretrained

    source = json.loads((FIXTURE / 'source.json').read_text())
    probe = np.load(FIXTURE / 'probe.npz')
    config = json.loads((FIXTURE / 'config.json').read_text())
    for loops, mode, dtype in TRACED:
        key = name(loops, mode, dtype)
        truth = np.load(directory / f'upstream-{tag(key)}.npy')
        loaded = Pretrained.load(source['repo'], revision=source['revision'], dtype=dtype)
        model = test_dory.with_cache(loaded.model.with_passes(loops), mode).with_cache_capacity(CONTEXT)
        programs, placed = test_dory.decoding(model), jax.device_put(loaded.variables)
        for batch in range(TRACED_BATCHES):
            held = np.lib.format.open_memmap(directory / f'dew-errors-{tag(key)}-{batch}.npy', mode='w+',
                                             dtype=np.float32, shape=(ORDERS, *truth.shape))
            for index in range(ORDERS):
                ordered = test_dory.symmetric(placed, config, symmetry(config, batch, index))
                found = np.concatenate([test_dory.forced(programs, ordered, probe[part],
                                                         probe[f'{key}/{part}_generated'],
                                                         prompt_rows=part == 'sentence')
                                        for part in ('sentence', 'window')])
                held[index] = np.abs(found - truth)
                print(f'Dew {key} order {batch}/{index}', flush=True)
            held.flush()


TREES = ('pairwise', 'blocked-32', 'blocked-128')
"""The ensemble's members besides vLLM, each Dew's same math with every sum under a declared tree
(tests/reference_trees.py): vLLM's key order is fixed, so the trees are what vary its key sums."""

MEMBERS = ('vllm', *TREES)
"""The reference's ensemble: an element's reference error is the worst over them."""


def trees(directory: Path, runs=None, orders_of: int = ORDERS) -> None:
    """Each tree member's per-element errors on every run (`runs`, every one by default), summed over
    each batch's orders (`symmetry`, the same orbit as vLLM's), `absolutes-{run}-{member}-{batch}.npy`
    `[rows, vocab]` float32: Dew's model at the run's dtype, its attention the XLA math every sum of
    which the interpreter reaches, decoding the run's tokens as the release test does
    (tests/test_dory.py `decoding`, `forced`), every sum under the member's tree. Under Dew's
    environment, on its GPU; `orders_of` below ORDERS times a pilot."""
    import time

    import jax
    import test_dory

    from dew.interop import Pretrained

    source = json.loads((FIXTURE / 'source.json').read_text())
    probe = np.load(FIXTURE / 'probe.npz')
    config = json.loads((FIXTURE / 'config.json').read_text())
    for loops, mode, dtype in runs or RUNS:
        key = name(loops, mode, dtype)
        truth = np.load(directory / f'upstream-{tag(key)}.npy')
        loaded = Pretrained.load(source['repo'], revision=source['revision'], dtype=dtype,
                                 attention_impl='xla')
        model = test_dory.with_cache(loaded.model.with_passes(loops), mode).with_cache_capacity(CONTEXT)
        placed = jax.device_put(loaded.variables)
        for member in TREES:
            programs = test_dory.decoding(model, member)
            for batch in range(BATCHES):
                total, began = np.zeros(truth.shape), time.perf_counter()
                for index in range(orders_of):
                    ordered = test_dory.symmetric(placed, config, symmetry(config, batch, index))
                    found = np.concatenate([test_dory.forced(programs, ordered, probe[part],
                                                             probe[f'{key}/{part}_generated'],
                                                             prompt_rows=part == 'sentence')
                                            for part in ('sentence', 'window')])
                    total += np.abs(found - truth)
                    print(f'{key} {member} order {batch}/{index}: {time.perf_counter() - began:.1f} s',
                          flush=True)
                np.save(directory / f'absolutes-{tag(key)}-{member}-{batch}.npy', total.astype(np.float32))


def errors(directory: Path, key: str, batch: int, shape: tuple[int, ...] | None = None) -> np.ndarray:
    """Run `key`'s per-order errors in `batch`, `[ORDERS, rows, vocab]` float32 on disk: each entry's
    absolute distance from float64 (`upstream`'s rows) under each order, kept where they are recorded
    for the element rule's calibration. With `shape`, created for writing; else opened for reading
    and writing."""
    path = directory / f'errors-{tag(key)}-{batch}.npy'
    if shape is not None:
        return np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=shape)
    return np.load(path, mmap_mode='r+')


def permute(model, config: dict, drawn: dict[str, np.ndarray]) -> np.ndarray:
    """vLLM's `model` under the symmetries `drawn` (`symmetry`), each parameter set in place from the
    model's own weights as it loaded them, which the first call keeps beside them on their device
    (`MEMORY` leaves the engine room for them): each attention's
    fused query, key and value rows, output gate rows and gains, output columns; each feed-forward's
    fused gate and up rows, gains and down columns (`maps`); then every parameter's one axis of the
    residual width, as tests/residual_orders.py reorders Dew's. Returns the embeddings' first row, by
    which a caller sees the order taken."""
    import re

    import torch

    if not hasattr(model, 'dory_loaded'):
        model.dory_loaded = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    loaded, found = model.dory_loaded, maps(config, drawn)
    hidden, residual = config['hidden_size'], torch.as_tensor(drawn['residual'])
    heads, kv, width = config['num_attention_heads'], config['num_key_value_heads'], config['head_dim']
    queries, keys = heads * width, kv * width
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            value = loaded[name]
            layer = re.search(r'\.layers\.(\d+)\.', name)
            layer = None if layer is None else int(layer.group(1))
            rows = columns = None
            if name.endswith('qkv_proj.weight'):
                rows = np.concatenate([found['query'][layer], queries + found['key'][layer],
                                       queries + keys + found['value'][layer]])
            elif name.endswith(('g_proj.weight', 's_gate')):
                rows = found['output'][layer]
            elif name.endswith('sqk'):
                rows = found['query'][layer]
            elif name.endswith('o_proj.weight'):
                columns = found['output'][layer]
            elif name.endswith('gate_up_proj.weight'):
                rows = np.concatenate([found['mlp'][layer],
                                       config['intermediate_size'] + found['mlp'][layer]])
            elif name.endswith(('suv_gate', 'suv_up')):
                rows = found['mlp'][layer]
            elif name.endswith('down_proj.weight'):
                columns = found['mlp'][layer]
            if rows is not None:
                value = value.index_select(0, torch.as_tensor(rows, device=value.device))
            if columns is not None:
                value = value.index_select(1, torch.as_tensor(columns, device=value.device))
            axes = [axis for axis, size in enumerate(value.shape) if size == hidden]
            if axes:
                (axis,) = axes
                value = value.index_select(axis, residual.to(value.device))
            parameter.copy_(value)
    return model.backbone.embeddings.weight[0].float().cpu().numpy()


def orders_run(directory: Path, loops: int, mode: str, dtype: str) -> None:
    """One run under every order of every batch (`symmetry`), the engine's weights set in place to
    each and the identity run's tokens drawn again (`Forced`): for each batch, each order's rows on
    their columns (`columns-{run}-{batch}.npy`) and each entry's error (`errors`)."""
    import functools

    probe = np.load(FIXTURE / 'probe.npz')
    config = json.loads((FIXTURE / 'config.json').read_text())
    key = name(loops, mode, dtype)
    truth = np.load(directory / f'upstream-{tag(key)}.npy')
    llm = engine(loops, mode, dtype, logits_processors=['dory_vllm_forcing:Forced'])
    columns = columns_of(probe, key)

    def rows() -> np.ndarray:
        found = []
        for part, _, prompt_rows in PARTS:
            tokens = probe[f'{key}/{part}_generated']
            drawn, scores = scored(llm, probe[part], len(tokens), prompt_rows, forced=tokens)
            assert np.array_equal(drawn, tokens), f'{key} {part} drew otherwise'
            found.append(scores)
        return np.concatenate(found)

    unpermuted = None
    for batch in range(BATCHES):
        kept = np.lib.format.open_memmap(directory / f'columns-{tag(key)}-{batch}.npy', mode='w+',
                                         dtype=np.float32, shape=(ORDERS, *columns.shape))
        held = errors(directory, key, batch, (ORDERS, *truth.shape))
        for index in range(ORDERS):
            drawn = symmetry(config, batch, index)
            (seen,) = llm.apply_model(functools.partial(permute, config=config, drawn=drawn))
            if unpermuted is None:
                unpermuted = seen
            assert np.array_equal(seen, unpermuted[drawn['residual']]), (
                f'{key}: the engine holds no order {batch}/{index}')
            found = rows()
            kept[index] = np.take_along_axis(found, columns, axis=1)
            if batch == index == 0:
                assert np.array_equal(kept[index], probe[f'{key}/vllm']), f'{key}: order 0 is not the probe'
                assert np.array_equal(found[positions(probe)], probe[f'{key}/vllm_full']), (
                    f'{key}: order 0 is not the probe')
            held[index] = np.abs(found - truth)
            apart = np.sqrt(np.mean(np.square(found - truth)))
            print(f'{key} order {batch}/{index}: {apart:.4f}', flush=True)
        kept.flush()
        held.flush()


def sums(directory: Path) -> None:
    """vLLM's per-element errors on each run summed over each batch's orders,
    `absolutes-{run}-vllm-{batch}.npy` `[rows, vocab]` float32, which `publish` puts beside the probe
    with the tree members' (`trees`): the element rule reads each entry's mean over a batch
    (tests/reference_error.py)."""
    for run in RUNS:
        for batch in range(BATCHES):
            held = errors(directory, name(*run), batch)
            total = np.zeros(held.shape[1:], np.float64)
            for order in held:
                total += order
            np.save(directory / f'absolutes-{tag(name(*run))}-vllm-{batch}.npy', total.astype(np.float32))


def orders(directory: Path) -> None:
    """Every run under every order, each in a process of its own: `orders_run`'s files, which
    `publish` puts beside the probe."""
    import os

    for loops, mode, dtype in RUNS:
        # The weights' permutation reaches the engine's process as a pickled function.
        subprocess.run([sys.executable, __file__, 'order', str(directory), str(loops), mode, dtype],
                       check=True, env={**os.environ, 'VLLM_ALLOW_INSECURE_SERIALIZATION': '1'})


CARD = """---
license: apache-2.0
tags: [dew, dory, vllm, reference]
---
# Dory 1.0 4B as vLLM computes it

Reference statistics for the release test of [Dew](https://github.com/AshishKumar4/dew)
(`tests/test_dory.py`), recorded by `tools/dory_vllm_reference.py` from vLLM's own engine running
nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16: {pull}, installed by vLLM's Python-only build over
{wheel}.

Runs (loop count / cache mode / dtype): {runs}. Each decodes a sentence greedily for 256 tokens and
64 tokens after 1100 random ids, past the 1024-token sliding window.

- `probe.npz`: the prompts, each run's drawn tokens, each row's probe columns with vLLM's and the
  float64 truth's log-probabilities on them, four rows over the whole vocabulary, and vLLM's own
  model code run in float64 on the same tokens.
- The orders are tools/dory_vllm_reference.py's `symmetry`, drawn from their seeds: each a
  permutation of the residual stream's units and of each layer's hidden units, heads and value
  dimensions, the first batch's first the identity.
- `upstream-<run>.npy`: every row over the whole vocabulary from vLLM's own model code in float64 on
  the run's tokens, the truth.
- `columns-<run>-<batch>.npy`: under each order of a batch, vLLM's rows on the probe columns,
  `[52, rows, 64]` float32.
- `absolutes-<run>-<batch>.npy`: every entry's absolute distance from the float64 truth summed over
  a batch's 52 orders, `[rows, vocabulary]` float32.

On a last-loop run's rows past the window these files hold the pull request's model code at the
engine's precision (bf16, or fp32 with TF32 attention products) instead of the engine, the reference
there: the engine's last-loop cache parts from the model code past the window
(https://github.com/vllm-project/vllm/pull/60149#issuecomment-6270937659).

Dew's `tests/fixtures/hf/dory-1.0-4b/source.json` pins this repository's revision and each file's
sha256.
"""
"""The dataset card `publish` writes beside the files."""


def files(directory: Path) -> dict[str, Path]:
    """The probe's files by their names in `DATASET`."""
    names = [*(f'upstream-{tag(name(*run))}.npy' for run in RUNS),
             *(f'columns-{tag(name(*run))}-{batch}.npy' for run in RUNS for batch in range(BATCHES)),
             *(f'absolutes-{tag(name(*run))}-{member}-{batch}.npy' for run in RUNS for member in MEMBERS
               for batch in range(BATCHES))]
    return {'probe.npz': FIXTURE / 'probe.npz', **{file: directory / file for file in names}}


def publish(directory: Path) -> None:
    """The probe and its orders in one commit of `DATASET`, and source.json pinning that commit and
    each file's sha256 (`fetched`)."""
    from huggingface_hub import CommitOperationAdd, HfApi

    api, kept = HfApi(), files(directory)
    api.create_repo(DATASET, repo_type='dataset', exist_ok=True)
    card = CARD.format(pull=PULL, wheel=WHEEL, runs=', '.join(name(*run) for run in RUNS))
    commit = api.create_commit(
        DATASET, repo_type='dataset', commit_message='The released Dory 4B as vLLM#60149 computes it',
        operations=[CommitOperationAdd(path_in_repo='README.md', path_or_fileobj=card.encode()),
                    *(CommitOperationAdd(path_in_repo=file, path_or_fileobj=str(path))
                      for file, path in kept.items())])
    source = json.loads((FIXTURE / 'source.json').read_text())
    source['probe'].update(
        dataset=DATASET, revision=commit.oid,
        truth="the pull request's model code in float64 (upstream), tools/dory_reference.py held to it",
        files={file: digest(path) for file, path in kept.items()})
    (FIXTURE / 'source.json').write_text(json.dumps(source, indent=2) + '\n')
    print(f'{DATASET}@{commit.oid}: {len(kept)} files')


def fetched(file: str) -> Path:
    """`file` of the probe, from `DATASET` at the revision source.json pins, its sha256 checked."""
    from huggingface_hub import hf_hub_download

    probe = json.loads((FIXTURE / 'source.json').read_text())['probe']
    path = Path(hf_hub_download(probe['dataset'], file, repo_type='dataset', revision=probe['revision']))
    found = digest(path)
    if found != probe['files'][file]:
        raise ValueError(f"{file} at {probe['revision']} hashes to {found}, "
                         f"source.json pins {probe['files'][file]}")
    return path


def digest(path: Path) -> str:
    """`path`'s sha256, read a block at a time."""
    import hashlib

    hashed = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1 << 24), b''):
            hashed.update(block)
    return hashed.hexdigest()


if __name__ == '__main__':
    stage, where = sys.argv[1], Path(sys.argv[2])
    if stage in ('run', 'order'):
        (record_run if stage == 'run' else orders_run)(where, int(sys.argv[3]), sys.argv[4], sys.argv[5])
    else:
        {'record': record, 'truth': truth, 'upstream': upstream, 'orders': orders,
         'model_orders': model_orders, 'sums': sums, 'trees': trees, 'dew_traces': dew_traces,
         'publish': publish}[stage](where)
