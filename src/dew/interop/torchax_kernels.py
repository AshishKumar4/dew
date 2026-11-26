"""Tier 3's kernel parts: a source's standard parts Dew computes by kernels of its own routed through them
inside the traced program (`kernel_parts`, row 13044), each identified by its code (`parts.identify`) and
attested to the causality walk (`sequence_causality.attested`), as `torchax_fallback.functional_parts`
installs them."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING

import jax.numpy as jnp

from dew.interop.sequence_causality import CAUSAL_ATTENTION, TOKENWISE, attested

if TYPE_CHECKING:
    from transformers import PreTrainedModel


def kernel_parts(model: PreTrainedModel) -> None:
    """Route each of `model`'s standard parts Dew computes by a kernel of its own through that kernel, as
    a native model's do, identified by its code (`parts.identify`, the known classes of Dew's registered
    families, under any renaming), never by a model's name: an RMSNorm (Llama's, Gemma's offset one, OLMo
    2's weight-first one: `_norm_forward`, Dew's fused norm), a gated MLP's SiLU product (`_mlp_forward`,
    Dew's fused SwiGLU), and a gated delta net's chunked rule (`_delta_net_forward`, Dew's chunked
    recurrence, in place of the `torch_chunk_gated_delta_rule` its code calls). Each keeps its module, so
    its parameters keep their names, its forward the instance's own; a part whose read differs from what
    the kernel computes (a weight of another dtype than its input's, a biased MLP, another activation)
    keeps transformers' forward."""
    from dew.interop.parts import Ambiguous, PartView

    config, known = model.config, {}
    for _, module in list(model.named_modules()):
        # An instance's own forward is none of its class's; each class matched once (its miss too), each
        # instance's fields read off it.
        if 'forward' in vars(module):
            continue
        if type(module) not in known:
            known[type(module)] = _kernel_part(module)
        matched = known[type(module)]
        if matched is None:
            continue
        make, part, renaming = matched
        try:
            fields = part.read(PartView(module, renaming, config))
        except Ambiguous:
            continue
        forward = make(module, part, renaming, fields)
        if forward is not None:
            vars(module)['forward'] = forward


def _kernel_part(module) -> tuple | None:
    """The forward maker, known part and renaming of the first of `kernel_parts`' roles `module`'s class
    computes by its code, or None."""
    from dew.interop.parts import Ambiguous, identify

    for role, make in (('norm', _norm_forward), ('mlp', _mlp_forward),
                       ('gated_delta_net', _delta_net_forward)):
        try:
            return (make, *identify(module, (role,)))
        except Ambiguous:
            continue
    return None


_NORM_FIELDS = frozenset({'norm_eps', 'scale_after_cast', 'scale_offset'})
"""An RMSNorm read's fields Dew's fused norm computes alone; a read of any other keeps its forward."""


def _norm_forward(module, part, renaming, fields) -> Callable | None:
    """An RMSNorm's forward through Dew's fused norm (`rms_normalized`, the generation's measured kernel),
    its epsilon, offset and rounding the part's (`fields`), in float32 statistics; None for a read it does
    not compute."""
    if set(fields) != _NORM_FIELDS:
        return None
    # The weight the known part holds (`KnownPart.parameters`), by the source's name for it, read at each
    # call: a traced call binds the module's parameters to its own values (`functional_call`).
    named = renaming.attributes.get('weight', 'weight')
    original = type(module).forward.__get__(module)

    def forward(hidden_states):
        import torch
        from torchax.interop import jax_view, torch_view

        from dew.nn.norms import rms_normalized

        weight = getattr(module, named)
        # A weight of the input's dtype alone: transformers' product of a narrow input by a wide weight
        # promotes its result, which Dew's norm, rounding to one dtype, does not; and at most float32, the
        # statistics' dtype the source casts to (`x.float()`), where Dew's widen with a wider input.
        if (hidden_states.dtype != weight.dtype or hidden_states.dtype == torch.float64
                or not _traced(hidden_states)):
            return original(hidden_states)
        x, scale = jax_view((hidden_states, weight))
        # An offset weight `1 + w` added in float32, as the source's `self.weight.float()` adds it.
        scale = scale.astype(jnp.float32) if fields['scale_offset'] else scale

        def normed(x, scale):
            return rms_normalized(x, scale, fields['norm_eps'], x.dtype, fields['scale_offset'],
                                  fields['scale_after_cast'], fp32_statistics=True)
        # Attested token by token, so the causality walk reads its launch (`sequence_causality.TOKENWISE`).
        return torch_view(attested(normed, TOKENWISE)(x, scale))
    return forward


_LLAMA_MLP = "transformers.models.llama.modeling_llama:LlamaMLP"
"""The gated MLP `_mlp_forward` routes: Llama's three projections by code, under any renaming."""


def _mlp_forward(module, part, renaming, fields) -> Callable | None:
    """A gated MLP's forward with its SiLU product through Dew's fused SwiGLU (`gated_product`), its gate,
    up and down projections the module's own; None for any other MLP part (a packed or renamed one's,
    one with multipliers: `_LLAMA_MLP`), a biased one or another activation."""
    if (part.location != _LLAMA_MLP or set(fields) != {'mlp', 'mlp_features', 'mlp_bias'}
            or fields['mlp'] != 'swiglu' or fields['mlp_bias']):
        return None
    gate, up, down = (getattr(module, renaming.attributes.get(name, name))
                      for name in ('gate_proj', 'up_proj', 'down_proj'))
    original = type(module).forward.__get__(module)

    def forward(x):
        from torchax.interop import jax_view, torch_view

        from dew.nn.activations import gated_product

        if not _traced(x):
            return original(x)
        gated, upped = jax_view((gate(x), up(x)))
        return down(torch_view(attested(gated_product('swiglu'), TOKENWISE)(gated, upped)))
    return forward


def _delta_net_forward(module, part, renaming, fields) -> Callable | None:
    """A gated delta net's forward whose chunked rule is Dew's (`_dew_chunk_rule`), its code otherwise: a
    copy of its forward whose `torch_chunk_gated_delta_rule` names Dew's, where its module's binding is
    transformers' chunked rule by its code (`_is_chunk_rule`); None where it is not."""
    import inspect

    # Past its wrappers (accelerate's offload hooks, which tier 3 sets none of), the module binding its rule.
    function = inspect.unwrap(type(module).forward)
    bound = function.__globals__.get('torch_chunk_gated_delta_rule')
    if bound is None or not _is_chunk_rule(bound):
        return None
    return _ruled(function, bound).__get__(module)


@functools.cache
def _ruled(function: Callable, bound: Callable) -> Callable:
    """`function` copied once, its `torch_chunk_gated_delta_rule` Dew's (`_dew_chunk_rule`)."""
    import types

    names = {**function.__globals__, 'torch_chunk_gated_delta_rule': _dew_chunk_rule(bound)}
    copied = types.FunctionType(function.__code__, names, function.__name__, function.__defaults__,
                                function.__closure__)
    copied.__kwdefaults__ = function.__kwdefaults__
    return copied


def _is_chunk_rule(bound) -> bool:
    """Whether `bound` (a module's `torch_chunk_gated_delta_rule`, under transformers' hub-kernel wrapper)
    computes Qwen3-Next's chunked delta rule by its code (`same_code`), as Dew's does."""
    import inspect
    import types

    from transformers.models.qwen3_next import modeling_qwen3_next

    from dew.interop.code_identity import Renaming, same_code

    known = inspect.unwrap(modeling_qwen3_next.torch_chunk_gated_delta_rule)
    candidate = inspect.unwrap(bound)
    return isinstance(candidate, types.FunctionType) and same_code(candidate, known, Renaming())


def _dew_chunk_rule(original: Callable) -> Callable:
    """transformers' chunked delta rule's signature over Dew's (`chunk_gated_delta_rule`): queries and keys
    l2-normalized in float32 where asked (FLA's `l2norm` at 1e-6), the rule over its kernel's operands (the
    source's dtype where its kernel takes them, else float32), its output in the queries' dtype; `original`
    where it runs outside a trace or packs documents (`cu_seqlens`). Its write gates are at most one
    (`write_bound`, as Dew's `GatedDeltaNet.declared` states its own) where its keys are l2-normalized: the
    delta nets `kernel_parts` routes are Qwen3.5's and Qwen3-Next's by their code (`parts.identify`), each
    gating its writes by `b.sigmoid()`; OLMo Hybrid's, doubling it, is none of them."""
    def rule(query, key, value, g, beta, chunk_size=64, initial_state=None, output_final_state=False,
             use_qk_l2norm_in_kernel=False, **kwargs):
        from torchax.interop import jax_view, torch_view

        from dew.nn.kernels.recurrence_chunk import chunk_gated_delta_rule, kernel_routed
        from dew.nn.precision import l2norm

        if kwargs.get('cu_seqlens') is not None or not _traced(query):
            return original(query, key, value, g=g, beta=beta, chunk_size=chunk_size,
                            initial_state=initial_state, output_final_state=output_final_state,
                            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **kwargs)
        given = jax_view((query, key, value))
        q, k, v, decay, write = (x.astype(jnp.float32) for x in (*given, *jax_view((g, beta))))
        if use_qk_l2norm_in_kernel:
            q, k = l2norm(q, 1e-6), l2norm(k, 1e-6)
        # The rule's operands rounded once to the dtype it takes, as a native layer's (`linear.py`'s gated
        # delta net): the projections' where the kernel takes them (`kernel_routed`, a bf16 source on a
        # GPU), float32 for the portable arithmetic (row 13280).
        narrow = given[0].dtype
        declared = {"write_bound": 1.0 if use_qk_l2norm_in_kernel else None,
                    "unit_queries": use_qk_l2norm_in_kernel}
        rounded_q, rounded_k, rounded_v = (x.astype(narrow) for x in (q, k, v))
        if narrow != jnp.float32 and q.shape[1] > 1 and kernel_routed(
                rounded_q, rounded_k, rounded_v, decay, write, **declared):
            q, k, v = rounded_q, rounded_k, rounded_v
        state = None if initial_state is None else jax_view(initial_state).astype(jnp.float32)
        # Attested causal over the queries' positions (`sequence_causality.CAUSAL_ATTENTION`): each token's
        # output reads its own and earlier tokens' queries, keys, values, decays and writes and the whole
        # entering memory, the final memory every token.
        out, final = attested(lambda q, k, v, decay, write, state: chunk_gated_delta_rule(
            q, k, v, decay, write, state, chunk_size, **declared),
            CAUSAL_ATTENTION, inputs=5, outputs=1)(q, k, v, decay, write, state)
        return (torch_view(out.astype(jax_view(query).dtype)),
                torch_view(final) if output_final_state else None)
    return rule


def _traced(tensor) -> bool:
    """Whether `tensor` is torchax's, inside a trace; an eager call (a probe's) keeps transformers'."""
    import torchax

    return isinstance(tensor, torchax.tensor.Tensor)
