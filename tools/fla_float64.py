"""fla-core's kernels as a float64 truth runs them on the CPU: each call a released model makes into fla's
Triton kernels (fp32 and bf16 only), answered by fla's own published torch references at the same call
boundary, so the rest of the model runs as released.

A truth installs them with `installed(modeling)` inside `torch_precision.float64()`, which widens the float32
casts fla's references make (`.float()`, `.to(torch.float)`), as it widens the model's own. They answer a
whole-sequence forward, unpadded or padded, without a cache: a call that carries a cache or a residual is
refused, as no truth runs one.

- `chunk_kda` and `fused_recurrent_kda`: `naive_recurrent_kda` (fla/ops/kda/naive.py) on each sequence of
  `cu_seqlens`, after what the call's flags put in the kernel: the l2 norm of queries and keys
  (`l2norm`'s, its eps 1e-6: fla/modules/l2norm.py), the gate (`naive_kda_gate`, or
  `naive_kda_lowerbound_gate` under a lower bound: fla/ops/kda/gate.py) and the beta's sigmoid; the final
  state transposed where `transpose_state_layout` asks for it.
- `fused_kda_gate` (fla 0.4.0's separate gate): `kda_gate_ref`, its own reference, as released.
- `ShortConvolution`: the nn.Conv1d it is, depthwise and causal over each sequence of `cu_seqlens` alone,
  then its activation.
- `FusedRMSNormGated`: `rms_norm_ref` (fla/modules/layernorm_gated.py) without a gate, times the module's
  output gate (its kernel's `y * sigmoid(g)`, or `y * g * sigmoid(g)` for swish).
"""

import contextlib
import importlib

import torch
import torch.nn.functional as F

from torch_precision import replaced


def sequences(cu_seqlens: torch.Tensor | None, tokens: int) -> list[tuple[int, int]]:
    """Each sequence's [start, end) of a call's tokens: `cu_seqlens`'s, else the whole."""
    if cu_seqlens is None:
        return [(0, tokens)]
    bounds = cu_seqlens.tolist()
    return list(zip(bounds[:-1], bounds[1:], strict=True))


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """fla's `l2norm` (fla/modules/l2norm.py: its kernel's x * rsqrt(sum(x^2) + eps)), whose entry runs
    Triton."""
    return x * torch.rsqrt(torch.sum(x * x, dim=-1, keepdim=True) + eps)


def kda(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=False, cu_seqlens=None, A_log=None, dt_bias=None, use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=False, safe_gate=False, lower_bound=None, transpose_state_layout=False):
    """`chunk_kda` and `fused_recurrent_kda` at their call boundary, in fla's own references."""
    from fla.ops.kda.naive import naive_recurrent_kda

    if initial_state is not None:
        raise ValueError("a truth runs a whole sequence from no state: no cache")
    if use_qk_l2norm_in_kernel:
        q, k = l2norm(q), l2norm(k)
    if use_gate_in_kernel:
        from fla.ops.kda import gate

        g = (gate.naive_kda_gate(g, A_log, dt_bias, output_dtype=g.dtype) if lower_bound is None else
             gate.naive_kda_lowerbound_gate(g, A_log, dt_bias, lower_bound, output_dtype=g.dtype))
    if use_beta_sigmoid_in_kernel:
        beta = beta.sigmoid()
    outputs, states = [], []
    for start, end in sequences(cu_seqlens, q.shape[1]):
        out, state = naive_recurrent_kda(q[:, start:end], k[:, start:end], v[:, start:end], g[:, start:end],
                                         beta[:, start:end], scale=scale,
                                         output_final_state=output_final_state)
        outputs.append(out)
        states.append(state)
    if not output_final_state:
        return torch.cat(outputs, 1), None
    state = torch.cat(states, 0)
    return torch.cat(outputs, 1), state.transpose(-1, -2) if transpose_state_layout else state


def short_convolution(self, x, residual=None, mask=None, cache=None, output_final_state=False,
                      cu_seqlens=None, chunk_indices=None):
    """`ShortConvolution.forward`: its weight's causal depthwise convolution over each sequence alone, then
    its activation."""
    if residual is not None or mask is not None or cache is not None or output_final_state:
        raise ValueError("a truth runs a whole sequence with no cache, residual or mask")
    width = self.weight.shape[-1]
    outputs = []
    for start, end in sequences(cu_seqlens, x.shape[1]):
        piece = F.pad(x[:, start:end].transpose(1, 2), (width - 1, 0))
        outputs.append(F.conv1d(piece, self.weight, self.bias, groups=self.weight.shape[0]).transpose(1, 2))
    out = torch.cat(outputs, 1)
    if self.activation in ("silu", "swish"):
        out = F.silu(out)
    elif self.activation is not None:
        raise ValueError(f"ShortConvolution's activation {self.activation!r} has no reference here")
    return out, None


def gated_norm(self, x, g, residual=None, prenorm=False, residual_in_fp32=False):
    """`FusedRMSNormGated.forward`: the RMS norm (`rms_norm_ref` without a gate) times the output gate."""
    from fla.modules.layernorm_gated import rms_norm_ref

    if residual is not None or prenorm:
        raise ValueError("a truth's gated norm takes no residual")
    out = rms_norm_ref(x, self.weight, self.bias, eps=self.eps)
    if self.activation == "sigmoid":
        return out * torch.sigmoid(g)
    if self.activation in ("silu", "swish"):
        return out * F.silu(g)
    raise ValueError(f"FusedRMSNormGated's activation {self.activation!r} has no reference here")


@contextlib.contextmanager
def installed(modeling):
    """The references in place of fla's kernels where the released module `modeling` calls them: the names
    it imported (`chunk_kda`, `fused_recurrent_kda`, and fla 0.4.0's `fused_kda_gate`) and the two fla
    modules' forwards, each restored after the block."""
    from fla.modules import FusedRMSNormGated, ShortConvolution

    patches = [(modeling, "chunk_kda", kda), (modeling, "fused_recurrent_kda", kda),
               (ShortConvolution, "forward", short_convolution), (FusedRMSNormGated, "forward", gated_norm)]
    if hasattr(modeling, "fused_kda_gate"):
        patches.append((modeling, "fused_kda_gate", importlib.import_module("fla.ops.kda.gate").kda_gate_ref))
    with replaced(*patches):
        yield
