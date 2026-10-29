#!/usr/bin/env python3
"""A tiny Dory and what NVIDIA's reference computes over it, for tests/test_dory.py.

No transformers class computes Dory; vLLM's does (vllm-project/vllm#60149,
vllm/model_executor/models/dory.py at 55076950), and it needs vLLM's engine
and a GPU. `DoryReference` is that file's forward transcribed to torch, line
for line (each step names its line), over the same tensors, at any dtype:
fp32 for the reference and float64 for the truth. Its decoding keeps the two
caches vLLM keeps: 'per_loop', one a recurrent layer and pass, and
'last_loop', one a recurrent layer that each pass overwrites, a pass reading
its own keys and values for the step's tokens and the last pass's for the
earlier ones (dory.py:146-149).

    PYTHONPATH=src:tools .venv/bin/python tools/dory_reference.py

writes tests/fixtures/hf/dory-tiny/: a random DoryForCausalLM of eight
layers, attentions and MLPs alternating as the release's do, over a width
of 32 with a window of 5, its middle four layers looped, a tokenizer of the
release's kind, fp32 and float64 logits at one, two and four loops and of
the variant configs, both caches' greedy continuations with each step's
logits, packed documents, prompts of two lengths, and the cotangent of a
loss over the logits, whose gradients the tests take from this reference
live.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

FIXTURE = Path(__file__).resolve().parents[1] / 'tests' / 'fixtures' / 'hf' / 'dory-tiny'
EPS = 1e-6

CONFIG = {
    'architectures': ['DoryForCausalLM'], 'model_type': 'dory', 'vocab_size': 64, 'hidden_size': 32,
    'num_hidden_layers': 8, 'intermediate_size': 48, 'num_attention_heads': 4, 'num_key_value_heads': 2,
    'head_dim': 16, 'max_position_embeddings': 64, 'tie_word_embeddings': False, 'hidden_act': 'silu',
    'initializer_range': 0.0177, 'sliding_window': 5, 'ngpt_alpha_init': 0.05,
    'attention_output_gate_logit_scale_init': None, 'mlp_bias': False, 'attention_bias': False,
    'n_input_layers': 2, 'n_recurrent_layers': 4, 'n_output_layers': 2, 'n_recurrent_loops': 2,
    'layer_types': ['sliding_attention', 'mlp', 'full_attention', 'mlp', 'sliding_attention', 'mlp',
                    'full_attention', 'mlp'],
    'rope_parameters': {
        'sliding_attention': {'rope_type': 'default', 'rope_theta': 10000, 'partial_rotary_factor': 1.0},
        'full_attention': {'rope_type': 'default', 'rope_theta': 1000000, 'partial_rotary_factor': 1.0}},
    'bos_token_id': 1, 'eos_token_id': [2, 11], 'pad_token_id': 11, 'dtype': 'float32',
}
"""The released config's fields at a tiny size (NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 at b366b87f)."""


def justnorm(x):
    """dory.py:37-41."""
    xf = x.double() if x.dtype == torch.float64 else x.float()
    return (xf * torch.rsqrt(xf.pow(2).sum(dim=-1, keepdim=True) + EPS)).to(x.dtype)


def wide(x):
    """The reference's `.float()`, which a float64 run keeps at float64."""
    return x if x.dtype == torch.float64 else x.float()


def slerp_alpha(a, b, alpha, alpha_scale):
    """dory.py:44-50."""
    ya, yb = wide(justnorm(a)), wide(justnorm(b))
    lr = (wide(alpha) * alpha_scale).abs()
    return justnorm(ya + lr * (yb - ya)).to(a.dtype)


def rotate(x, positions, theta, factor=1.0):
    """vLLM's neox-style rotary on CUDA: a rope `int(head_dim * factor)` wide over each head's
    leading dims, its two halves turned together, the rest passed through (get_rope,
    rotary_embedding/__init__.py:66-76). The angles are fp32, as `_compute_cos_sin_cache` builds
    them (base.py:94-103), float64 in a float64 run; the cos and sin round to the activations'
    dtype, as `_match_cos_sin_cache_dtype` casts the cache (base.py:105-131); and the kernel turns
    each pair in fp32 and rounds once (pos_encoding_kernels.cu:9-33, all at 55076950)."""
    width = int(x.shape[-1] * factor)
    exact = torch.float64 if x.dtype == torch.float64 else torch.float32
    inv = 1.0 / (theta ** (torch.arange(0, width, 2, dtype=exact, device=x.device) / width))
    angles = positions.to(exact)[:, None] * inv[None, :]
    half = width // 2
    cos, sin = (wide(table.to(x.dtype))[:, None, :] for table in (angles.cos(), angles.sin()))
    x1, x2, rest = wide(x[..., :half]), wide(x[..., half:width]), x[..., width:]
    return torch.cat([(x1 * cos - x2 * sin).to(x.dtype), (x2 * cos + x1 * sin).to(x.dtype), rest], dim=-1)


class DoryReference:
    """dory.py's forward over `tensors` (the checkpoint's names) at `dtype`."""

    def __init__(self, config: dict, tensors: dict[str, np.ndarray], dtype=torch.float32, loops=None,
                 mode=None, device='cpu'):
        self.c = dict(config)
        if loops is not None:
            self.c['n_recurrent_loops'] = loops
        if mode is not None:
            self.c['recurrent_kv_cache_mode'] = mode
        self.w = {name: torch.as_tensor(value).to(device, dtype) for name, value in tensors.items()}
        self.device = device
        self.dtype = dtype
        c = self.c
        self.std = c['initializer_range']
        self.d, self.heads_count, self.kv = c['head_dim'], c['num_attention_heads'], c['num_key_value_heads']
        g0 = c['attention_output_gate_logit_scale_init']
        g0 = c['hidden_size'] ** 0.5 if g0 is None else g0
        self.sqk_factor = self.d ** 0.5 / self.std
        self.gate_factor = g0 ** 0.5 / self.std
        self.suv_factor = c['hidden_size'] ** 0.5
        self.alpha_scale = c['ngpt_alpha_init'] / self.std

    def _w(self, name):
        return self.w[name]

    def heads(self, j, h, positions):
        """dory.py:135-149: attention layer `j`'s rotated, normed and scaled query heads, its
        keys and values as its cache stores them, and its gate's logits."""
        p = f'backbone.layers.{j}.mixer.'
        q = h @ self._w(p + 'q_proj.weight').T
        k = h @ self._w(p + 'k_proj.weight').T
        v = h @ self._w(p + 'v_proj.weight').T
        gate = h @ self._w(p + 'g_proj.weight').T
        kind = self.c['layer_types'][j]
        rope = self.c['rope_parameters'][kind]
        theta, factor = rope['rope_theta'], rope.get('partial_rotary_factor', 1.0)
        t = h.shape[0]
        q = rotate(q.view(t, self.heads_count, self.d), positions, theta, factor)
        k = rotate(k.view(t, self.kv, self.d), positions, theta, factor)
        sqk = (wide(self._w(p + 'sqk')) * self.sqk_factor).view(1, self.heads_count, self.d)
        q = (sqk.square() * wide(justnorm(q))).to(q.dtype)
        return q, justnorm(k), v.view(t, self.kv, self.d), gate

    def attention(self, j, h, positions, cache, loop):
        """dory.py:134-161 over the step's tokens, `cache` the (keys, values, positions) earlier
        tokens left this layer (None for none); returns the output and the step's own entries.

        The attention is vLLM's kernels' arithmetic (triton_unified_attention.py:543-599 and
        triton_attention_helpers.py:462-484 at 55076950): fp32 scores of the stored queries and
        keys, scaled in fp32, their exponentials from the row's maximum in fp32, those rounded to
        the values' dtype for the fp32 product with the values, the sum divided by the fp32 sum of
        the exponentials, and the result rounded to the values' dtype. The kernel walks the keys in
        tiles and shifts each tile's exponentials by the maximum so far; this takes the whole row's
        maximum, which moves where the exponentials round, not by how much."""
        p = f'backbone.layers.{j}.mixer.'
        q, k, v, gate = self.heads(j, h, positions)
        t = h.shape[0]
        own = (k, v, positions)
        if cache is not None:
            k, v, key_positions = (torch.cat([held, new]) for held, new in zip(cache, own, strict=True))
        else:
            key_positions = positions
        repeat = self.heads_count // self.kv
        keys, values = k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1)
        scores = (1.0 / math.sqrt(self.d)) * torch.einsum('qhd,khd->hqk', wide(q), wide(keys))
        distance = positions[:, None] - key_positions[None, :]
        visible = distance >= 0
        if self.c['layer_types'][j] == 'sliding_attention':
            visible &= distance < self.c['sliding_window']
        scores = scores.masked_fill(~visible[None], float('-inf'))
        weights = torch.exp(scores - scores.amax(dim=-1, keepdim=True))
        summed = torch.einsum('hqk,khd->qhd', wide(weights.to(values.dtype)), wide(values))
        out = (summed / weights.sum(dim=-1).transpose(0, 1)[..., None]).to(values.dtype)
        sg = (wide(self._w(p + 's_gate')) * self.gate_factor).square()
        out = (wide(out.reshape(t, -1)) * torch.sigmoid(wide(gate) * sg)).to(out.dtype)
        out = justnorm(out)
        return out @ self._w(p + 'o_proj.weight').T, own

    def mlp(self, j, h):
        """dory.py:188-197."""
        p = f'backbone.layers.{j}.mixer.'
        g, u = h @ self._w(p + 'gate_proj.weight').T, h @ self._w(p + 'up_proj.weight').T
        g = (wide(g) * (wide(self._w(p + 'suv_gate')) * self.suv_factor)).to(g.dtype)
        u = (wide(u) * (wide(self._w(p + 'suv_up')) * self.suv_factor)).to(u.dtype)
        m = (F.silu(wide(g)) * wide(u)).to(h.dtype)
        return justnorm(m) @ self._w(p + 'down_proj.weight').T

    def layer(self, j, h, positions, caches, loop):
        """dory.py:222-229, `caches` a layer's held entries by cache key, updated in place."""
        if self.c['layer_types'][j] == 'mlp':
            alpha = self._w(f'backbone.layers.{j}.mlp_alpha')
            return slerp_alpha(h, self.mlp(j, h), alpha, self.alpha_scale)
        n_in, n_rec = self.c['n_input_layers'], self.c['n_recurrent_layers']
        recurrent = n_in <= j < n_in + n_rec
        per_loop = self.c.get('recurrent_kv_cache_mode', 'per_loop') == 'per_loop'
        key = (j, loop if recurrent and per_loop else 0)
        out, own = self.attention(j, h, positions, caches.get(key) if caches is not None else None, loop)
        if caches is not None:
            pending = caches.setdefault('pending', {})
            pending[key] = own
        return slerp_alpha(h, out, self._w(f'backbone.layers.{j}.attn_alpha'), self.alpha_scale)

    def embedded(self, ids):
        rows = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        return self._w('backbone.embeddings.weight')[rows]

    def first_keys(self, ids):
        """The keys the first layer caches for each row of `ids` `[B, S]`: rotated and unit-normed,
        in the run's dtype, as float64 arrays."""
        with torch.no_grad():
            places = torch.arange(ids.shape[1], device=self.device)
            return np.stack([self.heads(0, self.embedded(row), places)[1].to(torch.float64).cpu().numpy()
                             for row in ids])

    def hidden(self, ids, positions, caches=None, cut=None):
        """dory.py:254-280 over the step's tokens. `cut` is the pass whose incoming state carries
        no gradient (Dew's `Loop.backprop_steps`, Huginn's truncation): the input layers still
        train through what every pass from it reinjects."""
        c = self.c
        h = self.embedded(ids)
        positions = positions.to(self.device)
        n_in, n_rec = c['n_input_layers'], c['n_recurrent_layers']
        for j in range(n_in):
            h = self.layer(j, h, positions, caches, 0)
        input_h = h
        state = input_h
        for loop in range(c['n_recurrent_loops']):
            if loop == cut:
                state = state.detach()
            h = justnorm(state + input_h)
            for j in range(n_in, n_in + n_rec):
                h = self.layer(j, h, positions, caches, loop)
            state = justnorm(h)
        h = state
        for j in range(n_in + n_rec, c['num_hidden_layers']):
            h = self.layer(j, h, positions, caches, 0)
        return h

    def logits(self, h):
        """dory.py:323-328."""
        logits = h @ self._w('lm_head.weight').T
        return wide(logits) * (wide(self._w('lm_head.logit_scale')) / self.std)

    def forward(self, ids):
        """The logits of each row of `ids` `[B, S]`, uncached."""
        with torch.no_grad():
            return np.stack([self.logits(self.hidden(row, torch.arange(len(row)))).cpu().numpy()
                             for row in ids])

    def gradients(self, ids, cotangent, names, cut=None):
        """The gradient of `sum(logits * cotangent)` over the rows of `ids` in each tensor `names`
        holds, `cut` as `hidden` reads it."""
        for name in names:
            self.w[name].requires_grad_(requires_grad=True)
        total = sum((self.logits(self.hidden(row, torch.arange(len(row)), cut=cut))
                     * torch.as_tensor(weights, device=self.device)).sum()
                    for row, weights in zip(ids, cotangent, strict=True))
        total.backward()
        return {name: self.w[name].grad.cpu().numpy() for name in names}

    def decode(self, prompt, steps: int):
        """Greedy decoding of one prompt, cached as `recurrent_kv_cache_mode` says: the tokens and
        each step's logits, the prompt's last first."""
        caches: dict = {}
        tokens, seen = list(prompt), []
        ids, positions = torch.as_tensor(np.asarray(prompt)), torch.arange(len(prompt))
        with torch.no_grad():
            for _ in range(steps):
                logits = self.logits(self.hidden(ids, positions, caches))[-1]
                for key, own in caches.pop('pending').items():
                    # A later write of the same key (the last loop's) replaces the earlier
                    # pass's entries for the step's tokens; the held earlier tokens stay.
                    held = caches.get(key)
                    caches[key] = own if held is None else tuple(
                        torch.cat([old, new]) for old, new in zip(held, own, strict=True))
                seen.append(logits.cpu().numpy())
                tokens.append(int(logits.argmax()))
                ids, positions = torch.as_tensor(tokens[-1:]), torch.as_tensor([len(tokens) - 1])
        return np.asarray(tokens), np.stack(seen)


def random_tensors(config: dict, seed: int) -> dict[str, np.ndarray]:
    """A Dory's tensors at the scales training leaves them: unit-norm embedding rows and head
    columns, projections at the initializer's scale, and gains near the values that make them 1."""
    rng = np.random.default_rng(seed)
    c, std = config, config['initializer_range']
    hidden, ffn, d = c['hidden_size'], c['intermediate_size'], c['head_dim']
    q, kv = c['num_attention_heads'] * d, c['num_key_value_heads'] * d
    tensors = {'backbone.embeddings.weight': rng.normal(size=(c['vocab_size'], hidden)),
               'lm_head.weight': rng.normal(size=(c['vocab_size'], hidden)),
               'lm_head.logit_scale': 30 * std * rng.uniform(0.5, 1.5, c['vocab_size'])}
    for name in ('backbone.embeddings.weight', 'lm_head.weight'):
        tensors[name] /= np.linalg.norm(tensors[name], axis=-1, keepdims=True)
    for j, kind in enumerate(c['layer_types']):
        p = f'backbone.layers.{j}.'
        if kind == 'mlp':
            tensors |= {p + 'mlp_alpha': std * rng.uniform(0.5, 2.0, hidden),
                        p + 'mixer.gate_proj.weight': rng.normal(0, std, (ffn, hidden)) * 5,
                        p + 'mixer.up_proj.weight': rng.normal(0, std, (ffn, hidden)) * 5,
                        p + 'mixer.down_proj.weight': rng.normal(0, std, (hidden, ffn)) * 5,
                        p + 'mixer.suv_gate': rng.uniform(0.5, 1.5, ffn) / math.sqrt(hidden),
                        p + 'mixer.suv_up': rng.uniform(0.5, 1.5, ffn) / math.sqrt(hidden)}
        else:
            tensors |= {p + 'attn_alpha': std * rng.uniform(0.5, 2.0, hidden),
                        p + 'mixer.q_proj.weight': rng.normal(0, std, (q, hidden)) * 5,
                        p + 'mixer.k_proj.weight': rng.normal(0, std, (kv, hidden)) * 5,
                        p + 'mixer.v_proj.weight': rng.normal(0, std, (kv, hidden)) * 5,
                        p + 'mixer.g_proj.weight': rng.normal(0, std, (q, hidden)) * 5,
                        p + 'mixer.o_proj.weight': rng.normal(0, std, (hidden, q)) * 5,
                        p + 'mixer.sqk': std * rng.uniform(0.5, 1.5, q) * math.sqrt(2),
                        p + 'mixer.s_gate': std * rng.uniform(0.5, 1.5, q) / hidden ** 0.25}
    return {name: value.astype(np.float32) for name, value in tensors.items()}


VARIANTS = {
    'partial_rotary': {'rope_parameters': {
        'sliding_attention': {'rope_type': 'default', 'rope_theta': 10000, 'partial_rotary_factor': 0.5},
        'full_attention': {'rope_type': 'default', 'rope_theta': 1000000, 'partial_rotary_factor': 0.25}}},
    'zero_alpha': {'ngpt_alpha_init': 0.0},
    'zero_gate': {'attention_output_gate_logit_scale_init': 0.0},
    'odd_partition': {'n_input_layers': 1, 'n_recurrent_layers': 4, 'n_output_layers': 3},
    'no_recurrent': {'n_input_layers': 4, 'n_recurrent_layers': 0, 'n_output_layers': 4},
}
"""Configs over the same tensors that DoryConfig accepts and the released one does not take: a
partial rotary per kind, an nGPT step or gate logit scale of zero, a loop whose block starts at an
MLP and ends at an attention, and a loop over no layer."""


PACKING = ((4, 7), (9, 2), (3, 3, 5))
"""The documents of each packed row, by length: each row 11 tokens, the window of 5 crossing a
document's start in every row."""

RAGGED = (3, 7)
"""The lengths of two prompts decoded side by side."""

SPECIALS = ('<unk>', '<s>', '</s>', '[INST]', '[/INST]', '[AVAILABLE_TOOLS]', '[/AVAILABLE_TOOLS]',
            '[TOOL_RESULTS]', '[/TOOL_RESULTS]', '[TOOL_CALLS]', '<|im_start|>', '<|im_end|>')
"""The release's first twelve tokens, by id (tokenizer.json's added_tokens at b366b87f)."""


def write_tokenizer() -> None:
    """A byte-level BPE as the release's (tokenizer_config.json at b366b87f): its special tokens
    at their ids, then lower-case letters, digits, the space and some punctuation as single-byte
    tokens with no merges, 56 ids in all; anything else reads as <unk>."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    alphabet = pre_tokenizers.ByteLevel.alphabet()
    pieces = [char for char in sorted(alphabet) if char in 'abcdefghijklmnopqrstuvwxyz0123456789']
    pieces += ['\u0120', '.', ',', '?', '!', "'", '-', ':']
    vocab = {token: index for index, token in enumerate((*SPECIALS, *pieces))}
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(list(SPECIALS))
    tokenizer.save(str(FIXTURE / 'tokenizer.json'))
    (FIXTURE / 'tokenizer_config.json').write_text(json.dumps({
        'add_bos_token': False, 'add_eos_token': False, 'bos_token': '<s>', 'eos_token': '<|im_end|>',
        'pad_token': '<|im_end|>', 'unk_token': '<unk>', 'clean_up_tokenization_spaces': False,
        'tokenizer_class': 'PreTrainedTokenizerFast'}, indent=2) + '\n')


def write_tiny() -> None:
    from safetensors.numpy import save_file

    FIXTURE.mkdir(parents=True, exist_ok=True)
    tensors = random_tensors(CONFIG, 1234)
    save_file(tensors, str(FIXTURE / 'model.safetensors'))
    (FIXTURE / 'config.json').write_text(json.dumps(CONFIG, indent=2) + '\n')
    (FIXTURE / 'generation_config.json').write_text(json.dumps(
        {'bos_token_id': 1, 'eos_token_id': [2, 11], 'pad_token_id': 11, 'do_sample': False},
        indent=2) + '\n')
    write_tokenizer()
    rng = np.random.default_rng(5)
    ids = rng.integers(12, CONFIG['vocab_size'], (2, 9))
    arrays = {'input_ids': ids}

    def both(name, compute, config=CONFIG, **fields):
        arrays[name] = compute(DoryReference(config, tensors, torch.float32, **fields))
        arrays[f'{name}_f64'] = compute(DoryReference(config, tensors, torch.float64, **fields))

    for loops in (1, 2, 4):
        both(f'logits_{loops}', lambda reference: reference.forward(ids), loops=loops)
    both('keys', lambda reference: reference.first_keys(ids))
    arrays['keys_bf16'] = DoryReference(CONFIG, tensors, torch.bfloat16).first_keys(ids)
    for mode in ('per_loop', 'last_loop'):
        tokens, steps = DoryReference(CONFIG, tensors, torch.float64, mode=mode).decode(ids[0, :6], 8)
        arrays[f'generated_{mode}'], arrays[f'step_logits_{mode}_f64'] = tokens, steps
        _, steps32 = DoryReference(CONFIG, tensors, torch.float32, mode=mode).decode(ids[0, :6], 8)
        arrays[f'step_logits_{mode}'] = steps32
    for name, change in VARIANTS.items():
        both(f'logits_{name}', lambda reference: reference.forward(ids), {**CONFIG, **change})

    # Packed rows: each document's own logits, laid out where the row holds it.
    width = sum(PACKING[0])
    documents = [[rng.integers(12, CONFIG['vocab_size'], length) for length in row] for row in PACKING]
    arrays['packed_ids'] = np.stack([np.concatenate(row) for row in documents])
    arrays['packed_segments'] = np.asarray(
        [[index + 1 for index, length in enumerate(row) for _ in range(length)] for row in PACKING], np.int32)
    arrays['packed_positions'] = np.asarray([[step for length in row for step in range(length)]
                                             for row in PACKING], np.int32)
    assert arrays['packed_ids'].shape == (len(PACKING), width)
    both('packed_logits', lambda reference: np.stack([np.concatenate(
        [reference.forward(document[None])[0] for document in row]) for row in documents]))

    prompts = [rng.integers(12, CONFIG['vocab_size'], length) for length in RAGGED]
    arrays['ragged_prompts'] = np.stack([np.pad(prompt, (0, max(RAGGED) - len(prompt)))
                                         for prompt in prompts])
    for mode in ('per_loop', 'last_loop'):
        reference = DoryReference(CONFIG, tensors, torch.float64, mode=mode)
        arrays[f'ragged_{mode}'] = np.stack([reference.decode(prompt, 6)[0][len(prompt):]
                                             for prompt in prompts])

    arrays['cotangent'] = rng.normal(size=(*ids.shape, CONFIG['vocab_size'])).astype(np.float32)
    np.savez(FIXTURE / 'reference.npz', **arrays)
    (FIXTURE / 'source.json').write_text(json.dumps({
        'reference': 'vllm-project/vllm#60149 vllm/model_executor/models/dory.py at '
                     '55076950b6c155f384d6766d63563c3bc20a0e3f, transcribed in tools/dory_reference.py',
        'released_config': 'nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 at '
                           'b366b87f762ea334ba3eb55912031baba1515810'},
        indent=2) + '\n')
    print(f'{FIXTURE}: {sorted(path.name for path in FIXTURE.iterdir())}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    write_tiny()
