"""Translate OLMo 3, a post-norm block with one rotary table per layer kind.

The released YaRN sits on the full-attention layers alone, which is why the
kinds carry their own rope rather than the model.
"""

from __future__ import annotations

from dew.interop.decoder_config import DecoderSpelling, decoder_layout
from dew.interop.decoder_parts import DecoderFamily, RopeSpelling
from dew.interop.layouts import ConfigLayout, constant

_SPELLING = RopeSpelling(theta=500000.0, per_kind=True, local_theta=500000.0)
"""Olmo3Config's rope: a record per layer type at its `default_theta`, which a
flat rope_theta never reaches on the sliding entry (configuration_olmo3.py:84, 101-126)."""


OLMO3_CONFIG = ConfigLayout(
    decoder_layout(DecoderSpelling(rope=_SPELLING, pattern=('sliding_attention',) * 3 + ('full_attention',),
                                   qk_norm=True, scale_after_cast=False)),
    constant('sandwich_norms', value=True), constant('pre_norms', value=False),
    constant('qk_norm_scope', value='projection'))
"""OLMo 3's config, both ways.

The block is post-norm: both norms sit on the sublayer outputs (modeling_olmo3.py:259-266, the sandwich
pair without the input pair), the q/k RMSNorms cover the whole projection before the head split (:162-163,
:178-179), and three sliding layers follow one full one (configuration_olmo3.py:96-98).

`Olmo3RotaryEmbedding` builds one frequency table per layer kind, calling `ROPE_INIT_FUNCTIONS[rope_type]`
for the kinds whose entry names a type of its own (modeling_olmo3.py:277-291). A ramp is therefore a kind's,
not the model's. A flat `rope_scaling` belongs to the full-attention layers alone, where the reference moves
it (configuration_olmo3.py:110-113): the released 7B checkpoints put a YaRN there and the sliding layers
rotate plainly. A flat `rope_theta` moves there too, so the sliding layers of a flat config rotate at the
class's own base whatever rope_theta says (500000 in both on the released 7B)."""


OLMO3 = DecoderFamily(
    ('olmo3',),
    OLMO3_CONFIG,
    lambda fields: not fields.pre_norms,
    'olmo3',
    'Olmo3ForCausalLM',
    preserve_source_layout=True,
    # Olmo3Config's norm epsilon, window and context.
    defaults={'rms_norm_eps': 1e-5, 'sliding_window': 4096, 'max_position_embeddings': 2048,
              'pad_token_id': 1},
    padding_idx=True,
)
