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

The transcription is ours, so the GPU test holds the released checkpoint to
vLLM's own logits, which tools/dory_vllm_record.py records.

    PYTHONPATH=src:tools .venv-3.12/bin/python tools/dory_reference.py

writes tests/fixtures/hf/dory-tiny/: a random DoryForCausalLM of eight
layers, four (attention, mlp) pairs over a width of 32 with a window of 5,
its middle four layers looped, with fp32 and float64 logits at one, two and
four loops, and both caches' greedy continuations with each step's logits.
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


def rotate(x, positions, theta):
    """vLLM's neox-style rotary over whole heads: the two halves of each head turned together."""
    half = x.shape[-1] // 2
    inv = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float64) * 2 / x.shape[-1]))
    angles = positions.double()[:, None] * inv[None, :]
    cos, sin = angles.cos().to(x.dtype)[:, None, :], angles.sin().to(x.dtype)[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class DoryReference:
    """dory.py's forward over `tensors` (the checkpoint's names) at `dtype`."""

    def __init__(self, config: dict, tensors: dict[str, np.ndarray], dtype=torch.float32, loops=None,
                 mode=None):
        self.c = dict(config)
        if loops is not None:
            self.c['n_recurrent_loops'] = loops
        if mode is not None:
            self.c['recurrent_kv_cache_mode'] = mode
        self.w = {name: torch.from_numpy(np.asarray(value)).to(dtype) for name, value in tensors.items()}
        self.dtype = dtype
        c = self.c
        self.std = c['initializer_range']
        self.d, self.heads, self.kv = c['head_dim'], c['num_attention_heads'], c['num_key_value_heads']
        g0 = c['attention_output_gate_logit_scale_init'] or c['hidden_size'] ** 0.5
        self.sqk_factor = self.d ** 0.5 / self.std
        self.gate_factor = g0 ** 0.5 / self.std
        self.suv_factor = c['hidden_size'] ** 0.5
        self.alpha_scale = c['ngpt_alpha_init'] / self.std

    def _w(self, name):
        return self.w[name]

    def attention(self, j, h, positions, cache, loop):
        """dory.py:134-161 over the step's tokens, `cache` the (keys, values, positions) earlier
        tokens left this layer (None for none); returns the output and the step's own entries."""
        p = f'backbone.layers.{j}.mixer.'
        q = h @ self._w(p + 'q_proj.weight').T
        k = h @ self._w(p + 'k_proj.weight').T
        v = h @ self._w(p + 'v_proj.weight').T
        gate = h @ self._w(p + 'g_proj.weight').T
        kind = self.c['layer_types'][j]
        theta = self.c['rope_parameters'][kind]['rope_theta']
        t = h.shape[0]
        q = rotate(q.view(t, self.heads, self.d), positions, theta)
        k = rotate(k.view(t, self.kv, self.d), positions, theta)
        sqk = (wide(self._w(p + 'sqk')) * self.sqk_factor).view(1, self.heads, self.d)
        q = (sqk.square() * wide(justnorm(q))).to(q.dtype)
        k = justnorm(k)
        v = v.view(t, self.kv, self.d)
        own = (k, v, positions)
        if cache is not None:
            k, v, key_positions = (torch.cat([held, new]) for held, new in zip(cache, own, strict=True))
        else:
            key_positions = positions
        repeat = self.heads // self.kv
        keys, values = k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1)
        logits = torch.einsum('qhd,khd->hqk', q, keys) / math.sqrt(self.d)
        distance = positions[:, None] - key_positions[None, :]
        visible = distance >= 0
        if kind == 'sliding_attention':
            visible &= distance < self.c['sliding_window']
        logits = logits.masked_fill(~visible[None], float('-inf'))
        out = torch.einsum('hqk,khd->qhd', torch.softmax(wide(logits), dim=-1).to(values.dtype), values)
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
        key = (j, loop if recurrent and self.c.get('recurrent_kv_cache_mode', 'per_loop') == 'per_loop' else 0)
        out, own = self.attention(j, h, positions, caches.get(key) if caches is not None else None, loop)
        if caches is not None:
            pending = caches.setdefault('pending', {})
            pending[key] = own
        return slerp_alpha(h, out, self._w(f'backbone.layers.{j}.attn_alpha'), self.alpha_scale)

    def hidden(self, ids, positions, caches=None):
        """dory.py:254-280 over the step's tokens."""
        c = self.c
        h = self._w('backbone.embeddings.weight')[torch.as_tensor(ids, dtype=torch.long)]
        n_in, n_rec = c['n_input_layers'], c['n_recurrent_layers']
        for j in range(n_in):
            h = self.layer(j, h, positions, caches, 0)
        input_h = h
        state = input_h
        for loop in range(c['n_recurrent_loops']):
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
            return np.stack([self.logits(self.hidden(row, torch.arange(len(row)))).numpy() for row in ids])

    def decode(self, prompt, steps: int):
        """Greedy decoding of one prompt, cached as `recurrent_kv_cache_mode` says: the tokens and
        each step's logits, the prompt's last first."""
        caches: dict = {}
        tokens, seen = list(prompt), []
        ids, positions = torch.as_tensor(prompt), torch.arange(len(prompt))
        with torch.no_grad():
            for _ in range(steps):
                logits = self.logits(self.hidden(ids, positions, caches))[-1]
                for key, own in caches.pop('pending').items():
                    # A later write of the same key (the last loop's) replaces the earlier
                    # pass's entries for the step's tokens; the held earlier tokens stay.
                    held = caches.get(key)
                    caches[key] = own if held is None else tuple(
                        torch.cat([old, new]) for old, new in zip(held, own, strict=True))
                seen.append(logits.numpy())
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


def write_tiny() -> None:
    from safetensors.numpy import save_file

    FIXTURE.mkdir(parents=True, exist_ok=True)
    tensors = random_tensors(CONFIG, 1234)
    save_file(tensors, str(FIXTURE / 'model.safetensors'))
    (FIXTURE / 'config.json').write_text(json.dumps(CONFIG, indent=2) + '\n')
    (FIXTURE / 'generation_config.json').write_text(json.dumps(
        {'bos_token_id': 1, 'eos_token_id': [2, 11], 'pad_token_id': 11, 'do_sample': False},
        indent=2) + '\n')
    ids = np.random.default_rng(5).integers(12, CONFIG['vocab_size'], (2, 9))
    arrays = {'input_ids': ids}
    for loops in (1, 2, 4):
        arrays[f'logits_{loops}'] = DoryReference(CONFIG, tensors, torch.float32, loops).forward(ids)
        arrays[f'logits_{loops}_f64'] = DoryReference(CONFIG, tensors, torch.float64, loops).forward(ids)
    for mode in ('per_loop', 'last_loop'):
        tokens, steps = DoryReference(CONFIG, tensors, torch.float64, mode=mode).decode(ids[0, :6], 8)
        arrays[f'generated_{mode}'], arrays[f'step_logits_{mode}_f64'] = tokens, steps
        _, steps32 = DoryReference(CONFIG, tensors, torch.float32, mode=mode).decode(ids[0, :6], 8)
        arrays[f'step_logits_{mode}'] = steps32
    np.savez(FIXTURE / 'reference.npz', **arrays)
    (FIXTURE / 'source.json').write_text(json.dumps({
        'reference': 'vllm-project/vllm#60149 vllm/model_executor/models/dory.py at '
                     '55076950b6c155f384d6766d63563c3bc20a0e3f, transcribed in tools/dory_reference.py',
        'released_config': 'nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 at '
                           'b366b87f762ea334ba3eb55912031baba1515810'},
        indent=2) + '\n')
    print(f'{FIXTURE}: {sorted(path.name for path in FIXTURE.iterdir())}')


if __name__ == '__main__':
    write_tiny()
