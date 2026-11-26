"""Inkling 2's multi-token-prediction draft as vLLM runs it, ported to plain torch ops: the reference
tests/test_inkling2_draft.py holds Dew's block drafter (`dew.nn.block_draft`) to, which the suite cannot
run (vLLM's Inkling 2 runs on a Blackwell alone, its kernels Triton and FlashInfer's).

Each function cites the vLLM code it ports (vllm-project/vllm at VLLM, model_executor/models/inkling2/:
nvidia/dspark.py, nvidia/ops/dspark.py, speculator.py, configs.py), op for op at the dtype the weights
are given in: where a kernel computes in fp32 and rounds to bf16, the port computes in at least fp32
(`_wide`) and rounds to the weights' dtype, so in float64 every rounding is the identity and the port is
its own float64 twin. Two departures, both of arithmetic vLLM states it approximates: the Markov head's
latent goes through `hidden_out` and the head as the reference implementation's draft runs it, where vLLM
folds them into one table once (`Inkling2MarkovHead.fold`, "folded once"); and the block's attention is
the plain softmax over its visible keys, where vLLM's FlashAttention splits it (`block_decode_attention`).

The target's taps (`visit_taps`) are transformers' own Inkling 2 (huggingface/transformers at 3a96649,
models/inkling2_loop/modeling_inkling2_loop.py): each tapped visit's partial block, the `hidden_states` a
decoder layer's call is handed (vLLM's `stack[num_blocks]` at the visit, a53_loop.py
`Inkling2LoopModel.forward`), and the final norm's output.
"""

from __future__ import annotations

import torch

VLLM = 'abb8194'
"""The vLLM commit whose Inkling 2 draft this ports (its loop model at a53e0a0)."""


def _wide(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float64 if x.dtype == torch.float64 else torch.float32)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """FlashInfer's `rmsnorm` (dspark.py `_RMSNorm`): fp32 statistics and product, rounded once."""
    wide = _wide(x)
    return (wide * torch.rsqrt(wide.square().mean(-1, keepdim=True) + eps) * _wide(weight)).to(x.dtype)


def added_norm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float):
    """FlashInfer's `fused_add_rmsnorm` (`_RMSNorm.forward` with a residual): the stream `residual + x`
    rounded, and its norm from the unrounded sum."""
    wide = _wide(residual) + _wide(x)
    normed = wide * torch.rsqrt(wide.square().mean(-1, keepdim=True) + eps) * _wide(weight)
    return normed.to(x.dtype), wide.to(residual.dtype)


def mix_rmsnorm(targets: torch.Tensor, alpha: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """`mix_rmsnorm` (ops/dspark.py): the `alpha`-weighted sum of `targets` `[..., T, D]`'s taps, normed,
    in fp32, rounded once to the taps' dtype."""
    mixed = (_wide(targets) * _wide(alpha)[:, None]).sum(-2)
    normed = mixed * torch.rsqrt(mixed.square().mean(-1, keepdim=True) + eps) * _wide(weight)
    return normed.to(targets.dtype)


def rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    """`rope_` (ops/dspark.py) over `[..., H, D]` heads at `positions` `[...]`: NeoX halves, at vLLM's
    `get_rope` default table (its inverse frequencies `theta ** -(2i / D)`), in fp32, rounded once."""
    width = x.shape[-1]
    wide = _wide(x)
    inverse = 1.0 / (theta ** (torch.arange(0, width, 2, dtype=wide.dtype) / width))
    angles = positions.to(wide.dtype)[..., None] * inverse
    cos, sin = torch.cos(angles)[..., None, :], torch.sin(angles)[..., None, :]
    first, second = wide[..., :width // 2], wide[..., width // 2:]
    return torch.cat([first * cos - second * sin, second * cos + first * sin], dim=-1).to(x.dtype)


def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    return torch.nn.functional.linear(x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype))


class Draft:
    """The draft's weights (`mtp.safetensors`, by its names) and config (`mtp_config`), in `dtype`, beside
    the target's embedding (normed, `embed`), head rows `head` `[vocab, D]` and logit cap."""

    def __init__(self, weights: dict[str, torch.Tensor], config: dict, embed, head: torch.Tensor,
                 softcap: float, dtype: torch.dtype):
        self.w = {name: tensor.to(dtype) for name, tensor in weights.items()}
        self.config, self.embed, self.head, self.softcap = config, embed, head.to(dtype), softcap
        self.dtype = dtype
        # `refresh_derived_weights`: the taps' weights a row each, softmax in fp32, held in the weights'
        # dtype.
        self.alpha = torch.softmax(_wide(self.w['fuse_weight']), dim=-1).to(dtype)

    def _layer(self, index: int, name: str) -> torch.Tensor:
        return self.w[f'layers.{index}.{name}']

    def context(self, taps: torch.Tensor, index: int, places: torch.Tensor):
        """Layer `index`'s context keys and values `[B, M, kv, D]` from the taps `[B, M, T * D]` at `places`
        (`precompute_and_store_context_kv`, `store_context_kv`)."""
        c, width = self.config, self.config['hidden_size']
        rows, length = taps.shape[:2]
        mixed = mix_rmsnorm(taps.view(rows, length, -1, width), self.alpha[index],
                            self._layer(index, 'dflare_norm.weight'), c['rms_norm_eps'])
        heads = (rows, length, c['num_key_value_heads'], c['head_dim'])
        keys = rms_norm(linear(mixed, self._layer(index, 'self_attn.k_proj_ctx.weight')).view(heads),
                        self._layer(index, 'self_attn.k_norm_ctx.weight'), c['rms_norm_eps'])
        values = linear(mixed, self._layer(index, 'self_attn.v_proj_ctx.weight')).view(heads)
        return rope(keys, places, c['rope_theta']), values

    def attention(self, index: int, x: torch.Tensor, places: torch.Tensor, keys, values, visible):
        """`Inkling2DraftSelfAttention.forward`: the block `x` `[B, K, D]` at `places`, its queries and keys
        normed per head and rounded, rotated (`qk_norm_rope_`), reading the context's `visible` `[B, M]`
        keys and the whole block."""
        c = self.config
        rows, length = x.shape[:2]
        query = linear(x, self._layer(index, 'self_attn.q_proj.weight')).view(
            rows, length, c['num_attention_heads'], c['head_dim'])
        own = (rows, length, c['num_key_value_heads'], c['head_dim'])
        block_keys = linear(x, self._layer(index, 'self_attn.k_proj.weight')).view(own)
        block_values = linear(x, self._layer(index, 'self_attn.v_proj.weight')).view(own)
        eps, theta = c['rms_norm_eps'], c['rope_theta']
        query = rope(rms_norm(query, self._layer(index, 'self_attn.q_norm.weight'), eps), places, theta)
        block_keys = rope(rms_norm(block_keys, self._layer(index, 'self_attn.k_norm.weight'), eps), places,
                          theta)
        all_keys = torch.cat([keys, block_keys], dim=1)
        all_values = torch.cat([values, block_values], dim=1)
        group = c['num_attention_heads'] // c['num_key_value_heads']
        all_keys, all_values = (t.repeat_interleave(group, dim=2) for t in (all_keys, all_values))
        scores = torch.einsum('bqhd,bkhd->bhqk', _wide(query), _wide(all_keys)) * c['head_dim'] ** -0.5
        seen = torch.cat([visible, torch.ones(rows, length, dtype=torch.bool)], dim=1)
        scores = scores.masked_fill(~seen[:, None, None, :], float('-inf'))
        out = torch.einsum('bhqk,bkhd->bqhd', torch.softmax(scores, dim=-1), _wide(all_values)).to(x.dtype)
        return linear(out.reshape(rows, length, -1), self._layer(index, 'self_attn.o_proj.weight'))

    def mlp(self, index: int, x: torch.Tensor) -> torch.Tensor:
        """`Inkling2DraftMLP`: `silu_and_mul`'s fp32 product rounded once, then the down projection."""
        gate = _wide(linear(x, self._layer(index, 'mlp.gate_proj.weight')))
        up = _wide(linear(x, self._layer(index, 'mlp.up_proj.weight')))
        product = (gate * torch.sigmoid(gate) * up).to(x.dtype)
        return linear(product, self._layer(index, 'mlp.down_proj.weight'))

    def states(self, taps: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """The block's final states `[B, K, D]` after each row's whole context `taps` `[B, M, T * D]`
        (`Inkling2DSparkDecoder.forward` over `embed_block`): the anchor's normed embedding, then masks."""
        c = self.config
        rows, length = taps.shape[:2]
        block = c['block_size']
        context_places = torch.arange(length).expand(rows, length)
        places = (length + torch.arange(block)).expand(rows, block)
        x = self.w['mask_embed'].expand(rows, block, -1).clone()
        x[:, 0] = self.embed(anchor).to(self.dtype)
        residual = None
        for index in range(c['num_hidden_layers']):
            keys, values = self.context(taps, index, context_places)
            visible = torch.ones(rows, length, dtype=torch.bool)
            if index in c['local_layer_ids']:
                visible = context_places >= length - c['sliding_window_size']
            eps = c['rms_norm_eps']
            if residual is None:
                residual, normed = x, rms_norm(x, self._layer(index, 'input_layernorm.weight'), eps)
            else:
                normed, residual = added_norm(x, residual, self._layer(index, 'input_layernorm.weight'), eps)
            attended = self.attention(index, normed, places, keys, values, visible)
            normed, residual = added_norm(attended, residual,
                                          self._layer(index, 'post_attention_layernorm.weight'), eps)
            x = self.mlp(index, normed)
        return added_norm(x, residual, self.w['norm.weight'], c['rms_norm_eps'])[0]

    def markov_step(self, state, previous, context):
        """`Inkling2MarkovHead.step` through `markov_gate` (ops/dspark.py), each op rounded to the state's
        dtype as the reference's bf16 torch ops round; the correction `head(hidden_out(latent))`."""
        rank = self.config['markov_rank']
        weight = self.w['markov_head.rnn_proj.weight']
        inputs = torch.cat([state, self.w['markov_head.markov_w1.weight'][previous]], dim=-1)
        raw = context + _wide(inputs) @ _wide(weight[:, :2 * rank]).T

        def rounded(x):
            return _wide(x.to(state.dtype))

        gate, candidate, out = rounded(raw).chunk(3, dim=-1)
        gate, candidate = rounded(torch.sigmoid(gate)), rounded(torch.tanh(candidate))
        kept = rounded(gate * _wide(state))
        update = rounded(rounded(1.0 - gate) * candidate)
        state = (kept + update).to(state.dtype)
        latent = torch.tanh(out).to(state.dtype)
        hidden = linear(latent, self.w['markov_head.hidden_out.weight'],
                        self.w['markov_head.hidden_out.bias'])
        return state, linear(hidden, self.head)

    def draft(self, taps: torch.Tensor, anchor: torch.Tensor, forced: torch.Tensor) -> torch.Tensor:
        """Each drafted place's logits `[B, K - 1, vocab]` after the whole context `taps` from `anchor` `[B]`,
        each step fed `forced` `[B, K - 1]`'s token before it (the speculator's `_sample_sequential`,
        `markov_logits`): `softcap(bf16(base + bias))`, the base the head over the place's states."""
        c, width, rank = self.config, self.config['hidden_size'], self.config['markov_rank']
        rows = taps.shape[0]
        hidden = self.states(taps, anchor)[:, 1:]
        base = linear(hidden, self.head)
        carried = mix_rmsnorm(taps[:, -1].view(rows, -1, width), self.alpha[-1],
                              self.w['markov_head.dflare_norm.weight'], c['rms_norm_eps'])
        weight = self.w['markov_head.rnn_proj.weight']
        inputs = torch.cat([hidden, carried[:, None].expand_as(hidden)], dim=-1)
        context = _wide(inputs) @ _wide(weight[:, 2 * rank:]).T + _wide(self.w['markov_head.rnn_proj.bias'])
        state = torch.zeros(rows, rank, dtype=self.dtype)
        previous, scored = anchor, []
        for index in range(c['block_size'] - 1):
            state, bias = self.markov_step(state, previous, context[:, index])
            summed = _wide((base[:, index] + bias).to(self.dtype))
            scored.append((self.softcap * torch.tanh(summed / self.softcap)).to(self.dtype))
            previous = forced[:, index]
        return torch.stack(scored, dim=1)


def visit_taps(model, ids: torch.Tensor, visits: list[int]) -> torch.Tensor:
    """transformers' Inkling 2 (`Inkling2LoopForCausalLM` `model`) over `ids`: each of `visits`' partial
    block (the `hidden_states` each decoder layer's call is handed, its calls in order the visits), then the
    final norm's output, concatenated `[B, S, (len(visits) + 1) * D]`."""
    handed: list[torch.Tensor] = []

    def record(_module, args, kwargs):
        handed.append((args[0] if args else kwargs['hidden_states']).detach().clone())

    hooks = [layer.register_forward_pre_hook(record, with_kwargs=True) for layer in model.model.layers]
    try:
        with torch.no_grad():
            final = model.model(input_ids=ids, use_cache=False).last_hidden_state
    finally:
        for hook in hooks:
            hook.remove()
    return torch.cat([*(handed[visit] for visit in visits), final], dim=-1)
