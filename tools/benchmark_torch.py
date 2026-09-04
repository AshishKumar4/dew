#!/usr/bin/env python3
"""PyTorch twin of tools/benchmark_step.py for the parity study in
docs/research/benchmark-parity.md.

Each model here is a line-by-line port of the Dew module it is compared with
(src/dew/nn/backbones/{causal_transformer,dit,unet}.py and the objectives),
including every dtype cast Dew makes: fp32 master parameters cast to bf16 at
each use, RMSNorm/LayerNorm statistics in fp32, RoPE in fp32, softmax in fp32,
the fp32 (TF32) output head, the fp32 loss, Adam with Optax's constants, an
EMA copy of every parameter updated each step, and the finiteness check on the
loss. Diffusion cases use fixed NumPy image, CFG-mask, timestep, and noise
tensors. The paired JAX harness uses the same byte values. This avoids
comparing JAX threefry with PyTorch Philox. The measured step is otherwise the
whole of the Trainer's train step, not a bare forward pass.

Environment: a uv venv at /tmp/torchbench with torch CUDA 12 wheels
(see the report for the exact versions used). Compiler scratch and caches
default to $XDG_CACHE_HOME/dew/torchbench; set TMPDIR,
TORCHINDUCTOR_CACHE_DIR, or TRITON_CACHE_DIR to override them. Run from the
dew repo root:

    /tmp/torchbench/bin/python tools/benchmark_torch.py --model causal_transformer --mode compile
    /tmp/torchbench/bin/python tools/benchmark_torch.py --model simple_dit --mode eager --attention sdpa
    /tmp/torchbench/bin/python tools/benchmark_torch.py --model unet --mode max-autotune --json-out /tmp/out.json
    /tmp/torchbench/bin/python tools/benchmark_torch.py --model causal_transformer --size large --profile

Modes: eager, compile (torch.compile default), max-autotune
(torch.compile mode="max-autotune"), and max-autotune-no-graphs
(mode="max-autotune-no-cudagraphs"). Attention: reference reproduces Dew's
flax reference path (bf16 QK^T, fp32 softmax, fp32 PV product); sdpa uses
F.scaled_dot_product_attention with the backend forced by --sdpa-backend.

Timing: --warmup steps, then --steps steps timed by wall clock with one
synchronize at the end (the method of benchmark_step.py), plus CUDA events
around every step for median/p10/p90. One case per process, so
max_memory_allocated is this case's own peak. --profile runs
torch.profiler over --profile-steps steps after warmup and reports kernel
count, GPU busy fraction and time by kernel category. Under
`nsys profile --capture-range=cudaProfilerApi` the same window is captured.
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass

# Triton and gcc use the system temporary directory while torch.compile runs.
# This workstation's /tmp is a quota-limited tmpfs. Keep compiler scratch and
# caches on disk unless the caller has chosen other locations explicitly.
_cache_home = os.path.expanduser(os.environ.get('XDG_CACHE_HOME', '~/.cache'))
_scratch = os.path.join(_cache_home, 'dew', 'torchbench')
os.environ.setdefault('TMPDIR', os.path.join(_scratch, 'tmp'))
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', os.path.join(_scratch, 'inductor'))
os.environ.setdefault('TRITON_CACHE_DIR', os.path.join(_scratch, 'triton'))
for _path in (os.environ['TMPDIR'], os.environ['TORCHINDUCTOR_CACHE_DIR'],
              os.environ['TRITON_CACHE_DIR']):
    os.makedirs(_path, exist_ok=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BF16 = torch.bfloat16
F32 = torch.float32
TEXT_TOKENS = 77
TEXT_FEATURES = 768
SIGMA_DATA = 0.5
P_MEAN, P_STD = -0.4, 1.0
UNCONDITIONAL_PROB = 0.12
EMA_DECAY = 0.999
ADAM = dict(lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
PEAK_BF16_SPEC = 97.5e12


# ----------------------------------------------------------------------------
# Presets: Dew's small preset and the larger compute-dominated cases
# ----------------------------------------------------------------------------

@dataclass
class Case:
    model: str
    config: dict
    batch_size: int
    image_size: int = 64
    seq_len: int = 0


def preset(model: str, size: str) -> Case:
    if model == 'causal_transformer':
        if size == 'small':
            return Case(model, dict(vocab_size=50304, emb_features=768, num_layers=3, num_heads=12,
                                    mlp_features=3072, max_seq_len=512),
                        batch_size=16, seq_len=512)
        return Case(model, dict(vocab_size=50304, emb_features=768, num_layers=12, num_heads=12,
                                mlp_features=3072, max_seq_len=1024),
                    batch_size=8, seq_len=1024)
    if model == 'simple_dit':
        if size == 'small':
            return Case(model, dict(patch_size=4, emb_features=384, num_layers=6, num_heads=6, mlp_ratio=4),
                        batch_size=16)
        return Case(model, dict(patch_size=4, emb_features=768, num_layers=12, num_heads=12, mlp_ratio=4),
                    batch_size=32)
    if model == 'unet':
        if size != 'small':
            raise ValueError('the unet has no large preset')
        return Case(model, dict(emb_features=256, feature_depths=[64, 128, 256],
                                attention_heads=[None, 4, 4], num_res_blocks=2, num_middle_res_blocks=1),
                    batch_size=16)
    raise ValueError(model)


# ----------------------------------------------------------------------------
# Shared pieces: bf16 linear over fp32 master weights, norms, rotary, attention
# ----------------------------------------------------------------------------

class Linear(nn.Module):
    """flax nn.Dense(dtype=bf16): fp32 kernel, bf16 compute, bf16 output."""

    def __init__(self, fan_in, fan_out, bias=True, compute_dtype=BF16, zero_init=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(fan_out, fan_in))
        self.bias = nn.Parameter(torch.zeros(fan_out)) if bias else None
        self.compute_dtype = compute_dtype
        if zero_init:
            nn.init.zeros_(self.weight)
        else:
            # flax default: lecun_normal (truncated normal, fan_in scaling)
            nn.init.trunc_normal_(self.weight, std=math.sqrt(1.0 / fan_in) / 0.87962566, a=-2.0, b=2.0)

    def forward(self, x):
        dt = self.compute_dtype
        return F.linear(x.to(dt), self.weight.to(dt), None if self.bias is None else self.bias.to(dt))


class RMSNorm(nn.Module):
    """Dew RMSNorm: fp32 statistics, fp32 scale, output cast to bf16."""

    def __init__(self, dim, eps=1e-5, out_dtype=BF16):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.out_dtype = out_dtype

    def forward(self, x):
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.scale).to(self.out_dtype)


def layer_norm_noaffine(x, eps=1e-5):
    """flax nn.LayerNorm(use_scale=False, use_bias=False, dtype=bf16)."""
    return F.layer_norm(x.float(), (x.shape[-1],), eps=eps).to(BF16)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x, cos, sin):
    """Rotate-half rotary embedding in fp32, cast back to the input dtype.
    x: [B, H, S, D] or [B, S, H, D] with cos/sin broadcastable to it (fp32, D wide)."""
    xf = x.float()
    return (xf * cos + rotate_half(xf) * sin).to(x.dtype)


def rotary_tables(seq_len, head_dim, theta=10000.0, device='cuda'):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=F32, device=device) / head_dim))
    angles = torch.arange(seq_len, dtype=F32, device=device)[:, None] * inv_freq[None, :]
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1)
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1)
    return cos, sin


class AttentionCore:
    """The attention kernel choice, shared by every model.

    reference: flax nn.dot_product_attention with force_fp32_for_softmax=True
    and dtype=bf16 as Dew runs it: q scaled in bf16, QK^T in bf16, softmax in
    fp32, and the probabilities stay fp32 for the PV product (an fp32/TF32
    matmul whose output is fp32).
    sdpa: F.scaled_dot_product_attention with one forced backend; bf16 in and out.
    """

    def __init__(self, kind, backend, fp32_softmax=True):
        self.kind = kind
        self.fp32_softmax = fp32_softmax
        from torch.nn.attention import SDPBackend
        self.backend = {'flash': SDPBackend.FLASH_ATTENTION, 'cudnn': SDPBackend.CUDNN_ATTENTION,
                        'efficient': SDPBackend.EFFICIENT_ATTENTION, 'math': SDPBackend.MATH}[backend]

    def __call__(self, q, k, v, causal):
        """q, k, v: [B, S, H, D] in bf16. Returns [B, S, H*D]."""
        B, S, H, D = q.shape
        if self.kind == 'sdpa':
            from torch.nn.attention import sdpa_kernel
            with sdpa_kernel(self.backend):
                out = F.scaled_dot_product_attention(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=causal)
            return out.transpose(1, 2).reshape(B, S, H * D)
        qh = (q / math.sqrt(D)).transpose(1, 2)            # bf16, as flax divides in dtype
        kh = k.transpose(1, 2)
        vh = v.transpose(1, 2)
        scores = torch.matmul(qh, kh.transpose(-1, -2))     # bf16 [B, H, S, S]
        if causal:
            mask = torch.ones(S, S, dtype=torch.bool, device=q.device).tril()
            scores = torch.where(mask, scores, torch.finfo(BF16).min)
        if self.fp32_softmax:
            probs = torch.softmax(scores.float(), dim=-1)   # fp32 probabilities
            out = torch.matmul(probs, vh.float())           # fp32 (TF32) PV, fp32 out
        else:
            probs = torch.softmax(scores, dim=-1)           # bf16 softmax (UNet stages)
            out = torch.matmul(probs, vh)
        return out.transpose(1, 2).reshape(B, S, H * D)


# ----------------------------------------------------------------------------
# causal_transformer (src/dew/nn/backbones/causal_transformer.py)
# ----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d, heads, head_dim, attention: AttentionCore):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        self.q_proj = Linear(d, heads * head_dim, bias=False)
        self.k_proj = Linear(d, heads * head_dim, bias=False)
        self.v_proj = Linear(d, heads * head_dim, bias=False)
        self.o_proj = Linear(heads * head_dim, d, bias=False)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.attention = attention

    def forward(self, x, cos, sin):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.heads, self.head_dim)
        q, k = self.q_norm(q), self.k_norm(k)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        return self.o_proj(self.attention(q, k, v, causal=True))


class GatedMLP(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.gate_proj = Linear(d, hidden, bias=False)
        self.up_proj = Linear(d, hidden, bias=False)
        self.down_proj = Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderBlock(nn.Module):
    def __init__(self, d, heads, head_dim, hidden, attention):
        super().__init__()
        self.input_layernorm = RMSNorm(d)
        self.self_attn = CausalSelfAttention(d, heads, head_dim, attention)
        self.post_attention_layernorm = RMSNorm(d)
        self.mlp = GatedMLP(d, hidden)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


class CausalTransformer(nn.Module):
    """Tied embeddings, swiglu, qk RMSNorm, rotary positions, fp32 head."""

    def __init__(self, cfg, attention, head_dtype=F32):
        super().__init__()
        d, L, H = cfg['emb_features'], cfg['num_layers'], cfg['num_heads']
        head_dim = d // H
        hidden = cfg['mlp_features']
        self.embed_tokens = nn.Parameter(torch.empty(cfg['vocab_size'], d))
        nn.init.normal_(self.embed_tokens, std=math.sqrt(1.0 / d))  # flax Embed default: variance_scaling fan_in
        self.layers = nn.ModuleList(DecoderBlock(d, H, head_dim, hidden, attention) for _ in range(L))
        self.norm = RMSNorm(d)
        self.head_dtype = head_dtype
        cos, sin = rotary_tables(cfg['max_seq_len'], head_dim)
        self.register_buffer('cos', cos[None, :, None, :], persistent=False)  # [1, S, 1, D]
        self.register_buffer('sin', sin[None, :, None, :], persistent=False)

    def forward(self, tokens):
        S = tokens.shape[1]
        x = F.embedding(tokens, self.embed_tokens).to(BF16)
        cos, sin = self.cos[:, :S], self.sin[:, :S]
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        # fp32 head over the tied fp32 table (TF32 on this card, as XLA's default)
        return torch.matmul(x.to(self.head_dtype), self.embed_tokens.to(self.head_dtype).t()).float()


def lm_loss(model, batch, metrics=True):
    tokens = batch['text']
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    logits = model(inputs)
    V = logits.shape[-1]
    # The batch stores int32 ids, as Dew's token pipeline does. Eager
    # nll_loss has no int32 index kernel, so the target slice is cast here.
    losses = F.cross_entropy(logits.view(-1, V), targets.reshape(-1).long(), reduction='none')
    ce = losses.mean()
    aux = {'ce': ce}
    if metrics:  # LMObjective returns perplexity and token accuracy every step
        aux['perplexity'] = torch.exp(ce)
        aux['token_accuracy'] = (logits.argmax(-1) == targets).float().mean()
    return ce, aux


# ----------------------------------------------------------------------------
# simple_dit (src/dew/nn/backbones/dit.py, src/dew/nn/dit.py, src/dew/nn/vit.py)
# ----------------------------------------------------------------------------

def sincos_2d_pos_embed(dim, H, W):
    """flax get_2d_sincos_pos_embed as used by scan_ordered_pos_embed (raster)."""
    grid_h = np.arange(H, dtype=np.float32)
    grid_w = np.arange(W, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape(2, 1, H, W)

    def one_d(d, pos):
        omega = np.arange(d // 2, dtype=np.float64) / (d / 2.0)
        omega = 1.0 / 10000 ** omega
        out = np.einsum('m,d->md', pos.reshape(-1), omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)
    emb = np.concatenate([one_d(dim // 2, grid[0]), one_d(dim // 2, grid[1])], axis=1)
    return torch.tensor(emb, dtype=F32)


class FourierEmbedding(nn.Module):
    def __init__(self, features, scale=16):
        super().__init__()
        freqs = np.random.RandomState(42).normal(size=(features // 2,)).astype(np.float32) * scale
        self.register_buffer('freqs', torch.tensor(freqs), persistent=False)

    def forward(self, t):
        emb = t.float()[:, None] * (2 * math.pi * self.freqs)[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class TimeProjection(nn.Module):
    """Two fp32 DenseGeneral layers with gelu (tanh form, flax default)."""

    def __init__(self, fan_in, features):
        super().__init__()
        self.l1 = Linear(fan_in, features, compute_dtype=F32)
        self.l2 = Linear(features, features, compute_dtype=F32)

    def forward(self, x):
        x = F.gelu(self.l1(x), approximate='tanh')
        return F.gelu(self.l2(x), approximate='tanh')


class DiTAttention(nn.Module):
    """RoPEAttention: biased q/k/v/out projections, rotary on q and k."""

    def __init__(self, d, heads, attention):
        super().__init__()
        self.heads, self.dim_head = heads, d // heads
        self.to_q = Linear(d, d)
        self.to_k = Linear(d, d)
        self.to_v = Linear(d, d)
        self.to_out = Linear(d, d)
        self.attention = attention

    def forward(self, x, cos, sin):
        B, S, _ = x.shape
        q = self.to_q(x).view(B, S, self.heads, self.dim_head)
        k = self.to_k(x).view(B, S, self.heads, self.dim_head)
        v = self.to_v(x).view(B, S, self.heads, self.dim_head)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        return self.to_out(self.attention(q, k, v, causal=False))


class ModulatedBlock(nn.Module):
    def __init__(self, d, heads, mlp_ratio, attention):
        super().__init__()
        self.ada_proj = Linear(d, 6 * d, zero_init=True)
        self.attn = DiTAttention(d, heads, attention)
        self.mlp1 = Linear(d, mlp_ratio * d)
        self.mlp2 = Linear(mlp_ratio * d, d)

    def forward(self, x, cond, cos, sin):
        ada = self.ada_proj(F.silu(cond))[:, None, :]
        scale_mlp, shift_mlp, gate_mlp, scale_attn, shift_attn, gate_attn = ada.chunk(6, dim=-1)
        h = layer_norm_noaffine(x) * (1 + scale_attn) + shift_attn
        x = x + gate_attn * self.attn(h, cos, sin)
        h = layer_norm_noaffine(x) * (1 + scale_mlp) + shift_mlp
        h = self.mlp2(F.gelu(self.mlp1(h), approximate='tanh'))
        return x + gate_mlp * h


class SimpleDiT(nn.Module):
    def __init__(self, cfg, attention, image_size):
        super().__init__()
        d, L, H, p = cfg['emb_features'], cfg['num_layers'], cfg['num_heads'], cfg['patch_size']
        self.p, self.d = p, d
        self.patch_weight = nn.Parameter(torch.empty(d, 3, p, p))
        nn.init.trunc_normal_(self.patch_weight, std=math.sqrt(1.0 / (3 * p * p)) / 0.87962566, a=-2.0, b=2.0)
        self.patch_bias = nn.Parameter(torch.zeros(d))
        grid = image_size // p
        self.register_buffer('pos_embed', sincos_2d_pos_embed(d, grid, grid), persistent=False)
        self.fourier = FourierEmbedding(d)
        self.time_proj = TimeProjection(d, d * cfg['mlp_ratio'])
        self.time_out = Linear(d * cfg['mlp_ratio'], d)
        self.text_proj = Linear(TEXT_FEATURES, d)
        self.blocks = nn.ModuleList(ModulatedBlock(d, H, cfg['mlp_ratio'], attention) for _ in range(L))
        # PatchSequenceOutput's final_norm keeps its affine (modulated=False)
        self.final_norm_scale = nn.Parameter(torch.ones(d))
        self.final_norm_bias = nn.Parameter(torch.zeros(d))
        self.final_proj = Linear(d, p * p * 3, compute_dtype=F32, zero_init=True)
        cos, sin = rotary_tables(4096, d // H)
        self.register_buffer('cos', cos[None, :, None, :], persistent=False)
        self.register_buffer('sin', sin[None, :, None, :], persistent=False)

    def forward(self, x, temb, textcontext):
        B, H, W, C = x.shape
        p = self.p
        # NHWC storage viewed as NCHW is exactly channels_last, so the conv reads it in place
        tokens = F.conv2d(x.permute(0, 3, 1, 2).to(BF16), self.patch_weight.to(BF16),
                          self.patch_bias.to(BF16), stride=p)
        tokens = tokens.flatten(2).transpose(1, 2)                      # [B, S, d] bf16
        tokens = tokens + self.pos_embed.to(BF16)[None]
        cond = self.time_out(self.time_proj(self.fourier(temb)))
        cond = cond + self.text_proj(textcontext).mean(dim=1)
        S = tokens.shape[1]
        cos, sin = self.cos[:, :S], self.sin[:, :S]
        for block in self.blocks:
            tokens = block(tokens, cond, cos, sin)
        normed = F.layer_norm(tokens.float(), (self.d,), self.final_norm_scale, self.final_norm_bias, 1e-5).to(BF16)
        out = self.final_proj(normed)                                     # fp32 [B, S, p*p*3]
        h = w = H // p
        out = out.view(B, h, w, p, p, 3).permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, 3)
        return out


# ----------------------------------------------------------------------------
# unet (src/dew/nn/backbones/unet.py, src/dew/nn/blocks.py, src/dew/nn/attention.py)
# ----------------------------------------------------------------------------

class Conv(nn.Module):
    """flax nn.Conv(dtype=bf16): fp32 HWIO kernel, bf16 compute, SAME padding."""

    def __init__(self, cin, cout, k=3, stride=1):
        super().__init__()
        weight = torch.empty(cout, cin, k, k)
        nn.init.trunc_normal_(weight, std=math.sqrt(1.0 / (cin * k * k)) / 0.87962566, a=-2.0, b=2.0)
        self.weight = nn.Parameter(weight.to(memory_format=torch.channels_last))
        self.bias = nn.Parameter(torch.zeros(cout))
        self.stride, self.k = stride, k

    def forward(self, x):
        # SAME padding for k=3: pad 1 (stride 1) or asymmetric (0, 1) for stride 2 on even sizes
        if self.k == 3 and self.stride == 2:
            x = F.pad(x, (0, 1, 0, 1))
            pad = 0
        else:
            pad = self.k // 2
        return F.conv2d(x, self.weight.to(BF16), self.bias.to(BF16), stride=self.stride, padding=pad)


class GroupNorm(nn.Module):
    def __init__(self, channels, groups=8, eps=1e-4):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.groups, self.eps = groups, eps

    def forward(self, x):
        return F.group_norm(x.float(), self.groups, self.weight, self.bias, self.eps).to(BF16)


class ResidualBlock(nn.Module):
    def __init__(self, cin, cout, temb_features):
        super().__init__()
        self.norm1 = GroupNorm(cin)
        self.conv1 = Conv(cin, cout)
        self.temb_projection = Linear(temb_features, cout)
        self.norm2 = GroupNorm(cout)
        self.conv2 = Conv(cout, cout)
        self.residual_conv = Conv(cin, cout, k=1) if cin != cout else None

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb_projection(temb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        residual = x if self.residual_conv is None else self.residual_conv(x)
        return h + residual


class CrossAttentionBlock(nn.Module):
    """TransformerBlock(only_pure_attention=True): RMSNorm the input, then
    NormalAttention over the text context (queries from x, keys and values from
    the 768-wide context), bias-free projections, added back to the normed input."""

    def __init__(self, channels, heads, attention):
        super().__init__()
        self.norm = RMSNorm(channels, eps=1e-4)
        self.heads, self.dim_head = heads, channels // heads
        self.to_q = Linear(channels, channels, bias=False)
        self.to_k = Linear(TEXT_FEATURES, channels, bias=False)
        self.to_v = Linear(TEXT_FEATURES, channels, bias=False)
        self.to_out = Linear(channels, channels, bias=False)
        self.attention = attention

    def forward(self, x, context):
        B, C, H, W = x.shape
        seq = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        seq = self.norm(seq)
        q = self.to_q(seq).view(B, H * W, self.heads, self.dim_head)
        k = self.to_k(context).view(B, context.shape[1], self.heads, self.dim_head)
        v = self.to_v(context).view(B, context.shape[1], self.heads, self.dim_head)
        out = self.to_out(self.attention(q, k, v, causal=False))
        out = seq + out
        return out.view(B, H, W, C).permute(0, 3, 1, 2)


class Unet(nn.Module):
    def __init__(self, cfg, attention):
        super().__init__()
        depths = cfg['feature_depths']
        heads = cfg['attention_heads']
        emb = cfg['emb_features']
        self.fourier = FourierEmbedding(emb)
        self.time_proj = TimeProjection(emb, emb)
        self.stem = Conv(3, depths[0])
        self.down_blocks = nn.ModuleList()
        self.down_attn = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        skips = [depths[0]]
        ch = depths[0]
        for i, (dim_out, h) in enumerate(zip(depths, heads)):
            blocks = nn.ModuleList()
            for j in range(cfg['num_res_blocks']):
                blocks.append(ResidualBlock(ch, ch, emb))
                skips.append(ch)
            self.down_blocks.append(blocks)
            self.down_attn.append(CrossAttentionBlock(ch, h, attention) if h else nn.Identity())
            if i != len(depths) - 1:
                self.downsamples.append(Conv(ch, dim_out, stride=2))
                ch = dim_out
        mid = depths[-1]
        self.middle = nn.ModuleList()
        for j in range(cfg['num_middle_res_blocks']):
            self.middle.append(nn.ModuleList([
                ResidualBlock(ch if j == 0 else mid, mid, emb),
                CrossAttentionBlock(mid, heads[-1], attention) if heads[-1] else nn.Identity(),
                ResidualBlock(mid, mid, emb)]))
        ch = mid
        self.up_blocks = nn.ModuleList()
        self.up_attn = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, (dim_out, h) in enumerate(zip(reversed(depths), reversed(heads))):
            blocks = nn.ModuleList()
            for j in range(cfg['num_res_blocks']):
                blocks.append(ResidualBlock(ch + skips.pop(), dim_out, emb))
                ch = dim_out
            self.up_blocks.append(blocks)
            self.up_attn.append(CrossAttentionBlock(ch, h, attention) if h else nn.Identity())
            if i != len(depths) - 1:
                # Dew: Upsample(features=feature_depths[-i]); for i == 0 that is depths[0]
                up_features = depths[-i]
                self.upsamples.append(Conv(ch, up_features))
                ch = up_features
        self.pre_final = Conv(ch, depths[0])
        self.final_res = ResidualBlock(depths[0] + skips.pop(), depths[0], emb)
        self.out_norm = GroupNorm(depths[0], eps=1e-6)  # flax GroupNorm default epsilon
        self.out_conv = Conv(depths[0], 3)

    def forward(self, x, temb, textcontext):
        temb = self.time_proj(self.fourier(temb))
        x = x.permute(0, 3, 1, 2).to(BF16)  # NHWC storage -> channels_last NCHW view
        x = self.stem(x)
        downs = [x]
        for i, blocks in enumerate(self.down_blocks):
            for j, block in enumerate(blocks):
                x = block(x, temb)
                if j == len(blocks) - 1:
                    x = self.down_attn[i](x, textcontext) if not isinstance(self.down_attn[i], nn.Identity) else x
                downs.append(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)
        for res1, attn, res2 in self.middle:
            x = res1(x, temb)
            x = attn(x, textcontext) if not isinstance(attn, nn.Identity) else x
            x = res2(x, temb)
        for i, blocks in enumerate(self.up_blocks):
            for j, block in enumerate(blocks):
                x = torch.cat([x, downs.pop()], dim=1)
                x = block(x, temb)
                if j == len(blocks) - 1 and not isinstance(self.up_attn[i], nn.Identity):
                    x = self.up_attn[i](x, textcontext)
            if i < len(self.upsamples):
                x = F.interpolate(x, scale_factor=2, mode='nearest')
                x = self.upsamples[i](x)
        x = self.pre_final(x)
        x = torch.cat([x, downs.pop()], dim=1)
        x = self.final_res(x, temb)
        x = F.silu(self.out_norm(x))
        x = self.out_conv(x)
        return x.permute(0, 2, 3, 1).float()  # back to NHWC, fp32 for the loss


# ----------------------------------------------------------------------------
# The diffusion objective (src/dew/objectives/diffusion/objective.py, EDM preset)
# ----------------------------------------------------------------------------

def diffusion_loss(model, batch):
    data = (batch['image'] - 127.5) / 127.5                                   # fp32 NHWC
    B = data.shape[0]
    device = data.device
    # These tensors are generated once by NumPy and kept on device. The JAX
    # parity harness uses the same seed and arrays. This compares frameworks,
    # not threefry against Philox.
    uncond = batch['uncond']
    # StubTextEncoder: tokens are ones, embedded as 1/77 broadcast to 768 features;
    # the unconditional context is the embedding of zeros
    text = torch.full((B, TEXT_TOKENS, TEXT_FEATURES), 1.0 / TEXT_TOKENS, device=device)
    text = torch.where(uncond[:, None, None], torch.zeros((), device=device), text)
    t = batch['timestep']
    sigma = torch.exp(t * P_STD + P_MEAN)
    noise = batch['noise']
    s = sigma[:, None, None, None]
    x_t = data + s * noise
    c_in = 1.0 / (torch.sqrt(SIGMA_DATA ** 2 + s ** 2) + 1e-8)
    temb = torch.log(sigma + 1e-12) / 4
    pred = model(x_t * c_in, temb, text)
    c_out = s * SIGMA_DATA / (torch.sqrt(SIGMA_DATA ** 2 + s ** 2) + 1e-8)
    c_skip = SIGMA_DATA ** 2 / (SIGMA_DATA ** 2 + s ** 2 + 1e-8)
    x0 = c_out * pred + c_skip * x_t
    losses = 0.5 * (x0 - data) ** 2                                           # optax.l2_loss
    weights = 1.0 / SIGMA_DATA ** 2 + 1.0 / s ** 2
    return (losses * weights).mean(), {}


# ----------------------------------------------------------------------------
# Step, timing, profiling
# ----------------------------------------------------------------------------

def build(case: Case, args):
    attention = AttentionCore(args.attention, args.sdpa_backend)
    if case.model == 'causal_transformer':
        model = CausalTransformer(case.config, attention, head_dtype=getattr(torch, args.head_dtype))
    elif case.model == 'simple_dit':
        model = SimpleDiT(case.config, attention, case.image_size)
    else:
        # UNet attention stages run with force_fp32_for_softmax=False in the small preset
        model = Unet(case.config, AttentionCore(args.attention, args.sdpa_backend, fp32_softmax=False))
    return model.cuda()


def make_batch(case: Case, seed=0):
    rng = np.random.default_rng(seed)
    if case.seq_len:
        text = rng.integers(0, case.config['vocab_size'], size=(case.batch_size, case.seq_len + 1)).astype(np.int32)
        return {'text': torch.from_numpy(text)}
    shape = (case.batch_size, case.image_size, case.image_size, 3)
    image = rng.integers(0, 256, size=shape).astype(np.float32)
    # Fixed tensors for both frameworks. The exact arrays are a function of
    # the seed, shape and these calls, so the benchmark needs no fixture file.
    uncond = (rng.random(case.batch_size) < UNCONDITIONAL_PROB)
    timestep = rng.standard_normal(case.batch_size).astype(np.float32)
    noise = rng.standard_normal(shape).astype(np.float32)
    return {'image': torch.from_numpy(image), 'uncond': torch.from_numpy(uncond),
            'timestep': torch.from_numpy(timestep), 'noise': torch.from_numpy(noise)}


def categorize(name):
    n = name.lower()
    if any(k in n for k in ('fmha', 'flash', 'attention', 'fused_attn', 'sdpa', 'cudnn::fusion')):
        return 'attention'
    if any(k in n for k in ('conv', 'fprop', 'dgrad', 'wgrad', 'cudnn', 'nhwc', 'nchw', 'implicit')):
        return 'conv'
    if any(k in n for k in ('gemm', 'cutlass', 'nvjet', 'matmul', 'xmma_gemm', 'cublas')) or \
            ('triton' in n and ('_mm' in n or 'bmm' in n or 'addmm' in n)):
        return 'gemm'
    if any(k in n for k in ('adam', 'multi_tensor', 'foreach', 'lerp')):
        return 'optimizer'
    if any(k in n for k in ('cross_entropy', 'nll_loss', 'log_softmax', 'argmax', 'reduce_kernel')) and 'softmax_warp' not in n:
        return 'loss/reduce'
    if any(k in n for k in ('memcpy', 'memset', 'copy_', 'cat_', 'catarray', 'transpose', 'index_select',
                            'indexing', 'gather', 'scatter', 'embedding')):
        return 'copy/gather'
    return 'elementwise/norm'


def profile_step(step, n_steps):
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(n_steps):
            step()
        torch.cuda.synchronize()
    events = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    kernels = []
    for e in events:
        name = e.name
        if name.startswith('Memcpy') or name.startswith('Memset'):
            kind = 'memcpy/memset'
        else:
            kind = None
        kernels.append((name, e.time_range.start, e.time_range.end, kind))
    if not kernels:
        return None
    kernels.sort(key=lambda k: k[1])
    # GPU busy: union of kernel intervals (streams may overlap)
    busy, cur_s, cur_e = 0.0, kernels[0][1], kernels[0][2]
    for _, s, e, _ in kernels[1:]:
        if s > cur_e:
            busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy += cur_e - cur_s
    window = kernels[-1][2] - kernels[0][1]
    by_cat, by_name, count_cat, count_name = {}, {}, {}, {}
    for name, s, e, kind in kernels:
        cat = kind or categorize(name)
        by_cat[cat] = by_cat.get(cat, 0.0) + (e - s)
        count_cat[cat] = count_cat.get(cat, 0) + 1
        by_name[name] = by_name.get(name, 0.0) + (e - s)
        count_name[name] = count_name.get(name, 0) + 1
    top = sorted(by_name.items(), key=lambda kv: -kv[1])[:15]
    per_cat = {}
    for name, us in sorted(by_name.items(), key=lambda kv: -kv[1]):
        cat = categorize(name) if not (name.startswith('Memcpy') or name.startswith('Memset')) else 'memcpy/memset'
        if len(per_cat.setdefault(cat, [])) < 4:
            per_cat[cat].append((name[:80], round(us / n_steps / 1e3, 3), count_name[name] / n_steps))
    return dict(profiled_steps=n_steps, kernels_per_step=len(kernels) / n_steps,
                gpu_busy_fraction=busy / window, gpu_busy_ms_per_step=busy / n_steps / 1e3,
                window_ms_per_step=window / n_steps / 1e3,
                ms_by_category={k: v / n_steps / 1e3 for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1])},
                kernels_by_category={k: v / n_steps for k, v in count_cat.items()},
                top_kernels=[(name[:90], round(us / n_steps / 1e3, 3)) for name, us in top],
                top_by_category=per_cat)


def analytic_flops(case: Case):
    """One formula per model class, applied identically to both frameworks:
    6 x matmul params x tokens, plus 12 x layers x S^2 x width per sample for
    attention (full square, causal not discounted), plus the small heads.
    The UNet formula is the sum over its convolutions and denses of
    6 x MACs per forward (forward + 2 backward), taken from the module graph."""
    cfg, B = case.config, case.batch_size
    if case.model == 'causal_transformer':
        d, L, H, S, V = cfg['emb_features'], cfg['num_layers'], cfg['num_heads'], case.seq_len, cfg['vocab_size']
        Fh = cfg['mlp_features']
        per_layer = 4 * d * d + 3 * d * Fh
        T = B * S
        return dict(matmul=6 * (L * per_layer + d * V) * T, attention=12 * L * B * S * S * d,
                    head=6 * d * V * T, tokens=T)
    if case.model == 'simple_dit':
        d, L, p = cfg['emb_features'], cfg['num_layers'], cfg['patch_size']
        S = (case.image_size // p) ** 2
        T = B * S
        r = cfg['mlp_ratio']
        per_layer = 4 * d * d + 2 * d * r * d
        # patch embed and text projection take inputs without gradients: forward + weight
        # gradient only (4x); everything else is forward + two backward matmuls (6x)
        small = 4 * (p * p * 3 * d) * T + 6 * (d * p * p * 3) * T + 6 * L * (6 * d * d) * B \
            + 6 * B * (d * r * d + r * d * r * d + r * d * d) + 4 * B * TEXT_TOKENS * TEXT_FEATURES * d
        return dict(matmul=6 * L * per_layer * T + small, attention=12 * L * B * S * S * d, tokens=T)
    return None


def unet_flops(model, case):
    """Forward MACs of every conv and linear in the torch UNet, from hooks."""
    macs = []

    def conv_hook(mod, inputs, output):
        w = mod.weight
        macs.append(output.numel() * w.shape[1] * w.shape[2] * w.shape[3])

    def linear_hook(mod, inputs, output):
        macs.append(output.numel() * mod.weight.shape[1])

    def attn_hook(mod, inputs, output):
        x, ctx = inputs
        B, C, H, W = x.shape
        macs.append(2 * B * mod.heads * H * W * ctx.shape[1] * mod.dim_head)
    hooks = []
    for m in model.modules():
        if isinstance(m, Conv):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, Linear):
            hooks.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, CrossAttentionBlock):
            hooks.append(m.register_forward_hook(attn_hook))
    batch = {k: v.cuda() for k, v in make_batch(case).items()}
    with torch.no_grad():
        diffusion_loss(model, batch)
    for h in hooks:
        h.remove()
    return dict(matmul=6 * sum(macs), attention=0, tokens=case.batch_size * case.image_size ** 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, choices=['causal_transformer', 'simple_dit', 'unet'])
    ap.add_argument('--size', default='small', choices=['small', 'large'])
    ap.add_argument('--mode', default='compile',
                    choices=['eager', 'compile', 'max-autotune', 'max-autotune-no-graphs'])
    ap.add_argument('--attention', default='reference', choices=['reference', 'sdpa'])
    ap.add_argument('--sdpa-backend', default='flash', choices=['flash', 'cudnn', 'efficient', 'math'])
    ap.add_argument('--head-dtype', default='float32', choices=['float32', 'bfloat16'],
                    help='causal_transformer output head dtype (Dew: float32)')
    ap.add_argument('--no-ema', action='store_true', help='skip the EMA update Dew makes every step')
    ap.add_argument('--no-metrics', action='store_true', help='skip perplexity/token accuracy (LM aux)')
    ap.add_argument('--no-optimizer', action='store_true', help='forward+backward only')
    ap.add_argument('--no-tf32', action='store_true', help='full fp32 for fp32 matmuls (XLA default is TF32)')
    ap.add_argument('--h2d', action='store_true', help='copy the batch host->device every step from pinned memory')
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--steps', type=int, default=100)
    ap.add_argument('--profile', action='store_true')
    ap.add_argument('--profile-steps', type=int, default=5)
    ap.add_argument('--nsys', action='store_true', help='bracket --profile-steps steps with cudaProfilerStart/Stop')
    ap.add_argument('--batch-size', type=int, default=None)
    ap.add_argument('--vocab-size', type=int, default=None,
                    help='LM vocabulary ablation; the matched preset is 50304')
    ap.add_argument('--json-out', default=None)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = not args.no_tf32
    torch.backends.cudnn.allow_tf32 = not args.no_tf32
    torch.backends.cudnn.benchmark = True

    case = preset(args.model, args.size)
    if args.batch_size:
        case.batch_size = args.batch_size
    if args.vocab_size is not None:
        if case.model != 'causal_transformer':
            raise ValueError('--vocab-size applies only to causal_transformer')
        case.config['vocab_size'] = args.vocab_size
    model = build(case, args)
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    ema = [p.detach().clone() for p in params] if not args.no_ema else None
    optimizer = torch.optim.Adam(params, fused=True, **ADAM)

    host_batch = make_batch(case, args.seed)
    if args.h2d:
        host_batch = {k: v.pin_memory() for k, v in host_batch.items()}
        device_batch = None
    else:
        device_batch = {k: v.cuda() for k, v in host_batch.items()}

    def loss_fn(batch):
        if case.model == 'causal_transformer':
            return lm_loss(model, batch, metrics=not args.no_metrics)
        return diffusion_loss(model, batch)

    if args.mode == 'eager':
        compiled = loss_fn
    else:
        mode = {'compile': None, 'max-autotune': 'max-autotune',
                'max-autotune-no-graphs': 'max-autotune-no-cudagraphs'}[args.mode]
        compiled = torch.compile(loss_fn, mode=mode)

    def step():
        if args.h2d:
            batch = {k: v.to('cuda', non_blocking=True) for k, v in host_batch.items()}
        else:
            batch = device_batch
        optimizer.zero_grad(set_to_none=True)
        loss, aux = compiled(batch)
        loss.backward()
        if not args.no_optimizer:
            optimizer.step()
            if ema is not None:
                torch._foreach_lerp_(ema, [p.detach() for p in params], 1.0 - EMA_DECAY)
        finite = torch.isfinite(loss)
        return loss, aux, finite

    t0 = time.perf_counter()
    for _ in range(args.warmup):
        loss, aux, finite = step()
    torch.cuda.synchronize()
    warmup_s = time.perf_counter() - t0
    torch.cuda.reset_peak_memory_stats()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.steps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.steps)]
    t0 = time.perf_counter()
    for i in range(args.steps):
        starts[i].record()
        loss, aux, finite = step()
        ends[i].record()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    per_step = np.array([s.elapsed_time(e) for s, e in zip(starts, ends)])
    # Gaps between consecutive steps show whether the host kept the device fed
    gaps = np.array([ends[i].elapsed_time(starts[i + 1]) for i in range(args.steps - 1)])

    flops = unet_flops(model, case) if case.model == 'unet' else analytic_flops(case)
    total_flops = flops['matmul'] + flops['attention']
    step_s = wall / args.steps
    row = dict(
        framework='torch', torch=torch.__version__, cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(), device=torch.cuda.get_device_name(0),
        model=case.model, size=args.size, mode=args.mode, attention=args.attention,
        sdpa_backend=args.sdpa_backend if args.attention == 'sdpa' else None,
        head_dtype=args.head_dtype if case.model == 'causal_transformer' else None,
        ema=ema is not None, metrics=not args.no_metrics, optimizer=not args.no_optimizer,
        tf32=not args.no_tf32, h2d=args.h2d, batch_size=case.batch_size,
        seq_len=case.seq_len, image_size=case.image_size,
        vocab_size=case.config.get('vocab_size'), params=n_params,
        warmup=args.warmup, warmup_s=round(warmup_s, 1), steps=args.steps,
        ms_per_step=step_s * 1e3, samples_per_sec=case.batch_size / step_s,
        tokens_per_sec=flops['tokens'] / step_s,
        event_ms_median=float(np.median(per_step)), event_ms_p10=float(np.percentile(per_step, 10)),
        event_ms_p90=float(np.percentile(per_step, 90)),
        inter_step_gap_ms_median=float(np.median(gaps)) if len(gaps) else None,
        analytic_flops=total_flops, analytic_tflops=total_flops / step_s / 1e12,
        mfu_spec=total_flops / step_s / PEAK_BF16_SPEC,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30,
        loss=float(loss), finite=bool(finite),
    )
    if args.profile:
        row['profile'] = profile_step(step, args.profile_steps)
    if args.nsys:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(args.profile_steps):
            step()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    print(json.dumps(row, indent=1))
    if args.json_out:
        rows = []
        if os.path.exists(args.json_out):
            with open(args.json_out) as fh:
                rows = json.load(fh)
        rows.append(row)
        with open(args.json_out, 'w') as fh:
            json.dump(rows, fh, indent=1)
    return row


if __name__ == '__main__':
    main()
