"""A torch stand-in for DeepSeek-V4.1-Flash's `inference/kernel.py`.

The release's `inference/model.py` imports its tilelang kernels from a
module named `kernel`. The reference tool installs this module under that
name so the pinned model code runs on CPU, in fp32 and under autograd. Each
function reproduces its kernel's arithmetic operation for operation
(kernel.py at DEEPSEEK_V41_REVISION, the lines cited per function), except
that sparse_attn keeps its probabilities in fp32 where the kernel rounds
them to bf16; `tools/deepseek_v41_reference.py --check-kernels` compares
them against the tilelang kernels on a GPU.

The quantizers are fake-quantizers, as the kernels' `inplace=True` path is:
the value leaves in the input's dtype, rounded through the storage format.
Under autograd they pass the gradient straight through, an estimator the
reference chooses: the release is inference code and the paper names none.
The GEMMs over fp8 and fp4 weights have no stand-in: the reference runs
dense weights.
"""

import torch

FP8_MAX = 448.0
FP4_MAX = 6.0


def _power_of_two_ceil(value: torch.Tensor) -> torch.Tensor:
    """`fast_round_scale` (kernel.py:22-37): 2 ** ceil(log2(value)) read off the
    fp32 exponent and mantissa bits, exact for every normal fp32."""
    bits = value.to(torch.float32).view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return ((exponent + 127) << 23).view(torch.float32)


def _straight_through(x: torch.Tensor, rounded: torch.Tensor) -> torch.Tensor:
    """`rounded` bit for bit in any dtype, the identity's gradient backward.

    `x.detach() - x` is +0, and subtracting +0 leaves every value as it
    is, a negative zero included, which the kernels keep and adding +0
    would lose."""
    return rounded.to(x.dtype).detach() - (x.detach() - x)


def _fake_quant(x: torch.Tensor, rounded: torch.Tensor, inplace: bool):
    value = _straight_through(x, rounded)
    if inplace:
        x.copy_(value)
        return x
    raise NotImplementedError("only the fake-quantizing inplace path has a stand-in")


def fp8_quotient(amax: torch.Tensor) -> torch.Tensor:
    """act_quant's scale before its rounding (kernel.py:74-80): amax floored
    at 1e-4 times fp32(1/448)."""
    return amax.clamp_min(1e-4) * torch.tensor(1 / FP8_MAX, dtype=torch.float32)


def fp8_scale(amax: torch.Tensor, power_of_two: bool) -> torch.Tensor:
    """act_quant's scale for a block's amax: the quotient, rounded up to a
    power of two when the scale format asks for it."""
    quotient = fp8_quotient(amax)
    return _power_of_two_ceil(quotient) if power_of_two else quotient


def fp4_quotient(amax: torch.Tensor, e4m3: bool) -> torch.Tensor:
    """fp4_act_quant's scale before its rounding (kernel.py:159-166): amax
    over 6 with amax floored at 6 * 2**-9 and the quotient saturating at
    E4M3's 448 (compressed KV, whose kernel casts with cvt.rn.satfinite), or
    amax times fp32(1/6) with amax floored at 6 * 2**-126 (the indexer)."""
    if e4m3:
        return (amax.clamp_min(FP4_MAX * 2 ** -9) / FP4_MAX).clamp_max(FP8_MAX)
    return amax.clamp_min(FP4_MAX * 2 ** -126) * torch.tensor(1 / FP4_MAX, dtype=torch.float32)


def fp4_scale(amax: torch.Tensor, e4m3: bool) -> torch.Tensor:
    """fp4_act_quant's scale for a block's amax: the quotient cast to E4M3,
    or rounded up to a power of two."""
    quotient = fp4_quotient(amax, e4m3)
    return quotient.to(torch.float8_e4m3fn).float() if e4m3 else _power_of_two_ceil(quotient)


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    """FP8 E4M3 per `block_size` channels (kernel.py:40-124): the value
    x / fp8_scale clamped to +-448, cast to E4M3, times the scale."""
    blocks = x.float().unflatten(-1, (-1, block_size))
    scale = fp8_scale(blocks.abs().amax(-1, keepdim=True), scale_fmt is not None)
    quantized = (blocks / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * scale
    return _fake_quant(x, quantized.flatten(-2), inplace)


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    """FP4 E2M1 per `block_size` channels (kernel.py:127-204): the value
    x / fp4_scale clamped to +-6 and rounded to E2M1, times the scale."""
    blocks = x.float().unflatten(-1, (-1, block_size))
    scale = fp4_scale(blocks.abs().amax(-1, keepdim=True), scale_dtype == torch.float8_e4m3fn)
    quantized = e2m1((blocks / scale).clamp(-FP4_MAX, FP4_MAX)) * scale
    return _fake_quant(x, quantized.flatten(-2), inplace)


_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
# Mantissa bit of each grid value; a tie goes to the neighbour whose bit is 0.
_E2M1_ODD = torch.tensor([False, True, False, True, False, True, False, True])


def e2m1(values: torch.Tensor) -> torch.Tensor:
    """Round fp32 values within +-6 to the nearest E2M1 value, ties to even,
    which the kernel's cvt.rn to float4_e2m1fn does."""
    grid = _E2M1.to(values.device, values.dtype)
    magnitude = values.abs()
    upper = torch.bucketize(magnitude, grid).clamp(1, len(grid) - 1)
    lower = upper - 1
    below, above = grid[lower], grid[upper]
    distance_below, distance_above = magnitude - below, above - magnitude
    odd = _E2M1_ODD.to(values.device)
    take_above = (distance_above < distance_below) | (
        (distance_above == distance_below) & odd[lower])
    rounded = torch.where(take_above, above, below)
    return torch.copysign(rounded, values)


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """Attention of each query over the keys its row of `topk_idxs` names
    (kernel.py:310-403): -1 names nothing, the keys are the values, each
    head's sink joins the softmax denominator and nothing else, fp32 logits."""
    batch = torch.arange(q.size(0), device=q.device)[:, None, None]
    valid = topk_idxs >= 0
    keys = kv[batch, topk_idxs.clamp_min(0).long()]  # [b, m, k, d]
    logits = torch.einsum("bmhd,bmkd->bmhk", q.float(), keys.float()) * softmax_scale
    logits = logits.masked_fill(~valid[:, :, None, :], float("-inf"))
    sink = attn_sink.float()[None, None, :, None].expand(*logits.shape[:3], 1)
    probs = torch.softmax(torch.cat([logits, sink], dim=-1), dim=-1)[..., :-1]
    return torch.einsum("bmhk,bmkd->bmhd", probs, keys.float()).to(q.dtype)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """Split mHC's mixes into pre, post and the Sinkhorn-normalised comb
    (kernel.py:406-474)."""
    hc = hc_mult
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = (mixes[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).unflatten(-1, (hc, hc))
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def fp8_gemm(*args, **kwargs):
    raise NotImplementedError("the reference runs dense weights; fp8 GEMMs have no stand-in")


def fp4_gemm(*args, **kwargs):
    raise NotImplementedError("the reference runs dense weights; fp4 GEMMs have no stand-in")
