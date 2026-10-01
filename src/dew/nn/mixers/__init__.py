"""Token mixers: what a decoder layer mixes across the sequence, by kind.

A mixer is the per-layer token interaction a `DecoderBlock` holds as
`self_attn`: any module with the `(x, decode=..., positions=...,
segment_ids=...) -> x` signature. Grouped-query causal attention is the
`attention` kind; MLA, the gated delta rule and the other mixers register
beside it, each as a frozen dataclass value carrying the reference's field
names.

The backbone names one value on its `mixer` field, None for attention, and
a kind builds its own `DecoderBlock` factory from a `MixerContext`: the
layer geometry the backbone owns (heads, head dims, the kind-resolved rotary
base, the window, the KV-sharing slot) plus the run's dtype and kernel
choices. Geometry is stated once, here, so a new kind reads what it needs
without the backbone growing a branch per kind. The backbone builds every
mixer through `mixer.build(ctx)`.
"""


# Kind modules register where they are defined and read their contracts from
# base, not this partly initialized hub. Imports remain alphabetical by kind.
from .. import deepseek_v4 as deepseek_v4, dsa_kpool as dsa_kpool, kda as kda, llama4 as llama4, mla as mla
from . import gated_delta_net as gated_delta_net, mamba2 as mamba2
from .attention import AttentionMixer as AttentionMixer
from .base import MixerBase as MixerBase, MixerContext as MixerContext, mixers as mixers
