"""Translate SenseNova-Loop's language model (`sensenova_loop`'s text, sensenova/SenseNova-Loop-9B-Preview's
remote code, modeling_sensenova_loop.py at 9937e91c).

It is Qwen 3.5's hybrid decoder, its config Qwen 3.5's text_config, under two settings its wrapper states at
its own scope, which its language model reads (`hf_decoders._text_config`): `loop_settings` runs the layers
[start, end) `eval_count` times, each pass reading the last one's output as it is and keeping its own
cache entries (`execution_plan`, `_VisitCache`, :25-74, :160-205), and `mhc` carries the residual as two or
four streams over the layers [start, end), entered as copies of it and left by their mean (`enter`,
`leave`, :146-157). Its routers (`MHCRouter`, :86-135) are mHC's sites under settings of their own: a weight
and a bias with no scale, no floor on the collapse, the mix normalized in log space rows first, which is
the column-first Sinkhorn with no floor (`dew.nn.hyper_connections`), mixing the streams as it stands, and
the read scaled by `pre_scale`, or a fixed identity mix. Its tensors are Qwen 3.5's, the routers'
`mhc_attn` and `mhc_ffn` beside them (`decoder_parts.LAYER_ROUTES`).
"""

from collections.abc import Mapping

from dew import _records
from dew.interop.decoder_parts import DECODER_ROUTES, DecoderFamily, DecoderFields, refuse
from dew.interop.families.qwen import QWEN35_TEXT, qwen_mtp_depth, qwen_mtp_routes
from dew.interop.layouts import ConfigLayout, ConfigPart
from dew.nn.backbones.causal_transformer import CausalTransformer

SENSENOVA_LOOP_TEXT_TYPE = 'sensenova_loop_text'
"""The type SenseNova-Loop's language model reads as: Qwen 3.5's text_config with the wrapper's
`loop_settings` and `mhc` beside it (`hf_decoders._text_config`)."""


def _loop(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """The loop `loop_settings` states: its block [start, end) run `eval_count` times, each pass reading
    the last one's output as it is; a schedule of several counts the release's own code refuses too."""
    used.add('loop_settings')
    loop = _records.record(hf_config.get('loop_settings'), 'loop_settings')
    first, end, count = (_records.integer(loop.get(key), f'loop_settings.{key}')
                         for key in ('start', 'end', 'eval_count'))
    if loop.get('counts') != [count]:
        refuse(f"loop_settings.counts {loop.get('counts')!r}",
               "the release's own code runs a fixed count, its eval_count, alone")
    record['loop'] = {'steps': count, 'layers': [first, end], 'step_norm': False}
    return record


def _loop_written(model: CausalTransformer) -> Mapping[str, object]:
    loop = model.loop
    assert loop is not None and loop.layers is not None
    first, end = loop.layers
    return {'loop_settings': {'start': first, 'end': end, 'eval_count': loop.steps, 'counts': [loop.steps]}}


def _streams(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """The mHC streams `mhc` states, over its layers, under its routers' settings (the module doc)."""
    used.add('mhc')
    mhc = hf_config.get('mhc')
    if mhc is None:
        return record
    mhc = _records.record(mhc, 'mhc')
    first, end, streams, iterations = (_records.integer(mhc.get(key), f'mhc.{key}')
                                       for key in ('start', 'end', 'streams', 'iterations'))
    record['hyper_connections'] = {
        'hc_mult': streams, 'hc_eps': 0.0, 'hc_sinkhorn_iters': iterations, 'head': 'mean',
        'hc_scaled': False, 'hc_mix_transposed': True,
        'hc_pre_scale': _records.number(mhc.get('pre_scale', 1.0), 'mhc.pre_scale'),
        'hc_fixed_identity': _records.boolean(mhc.get('fixed_identity'), 'mhc.fixed_identity'),
        'hc_layers': [first, end]}
    return record


def _streams_written(model: CausalTransformer) -> Mapping[str, object]:
    streams = model.hyper_connections
    if streams is None:
        return {}
    assert streams.hc_layers is not None
    first, end = streams.hc_layers
    return {'mhc': {'start': first, 'end': end, 'streams': streams.hc_mult,
                    'iterations': streams.hc_sinkhorn_iters, 'fixed_identity': streams.hc_fixed_identity,
                    'pre_scale': streams.hc_pre_scale}}


SENSENOVA_LOOP_TEXT = DecoderFamily(
    (SENSENOVA_LOOP_TEXT_TYPE,),
    ConfigLayout(QWEN35_TEXT, ConfigPart(_loop, _loop_written), ConfigPart(_streams, _streams_written)),
    lambda fields: bool(fields.loop is not None and fields.loop.layers is not None and (
        fields.hyper_connections is None or fields.hyper_connections.hc_mix_transposed)
        and 'linear_attention' in (fields.layer_types or ())),
    SENSENOVA_LOOP_TEXT_TYPE,
    'SenseNovaLoopForCausalLM',
    routes=qwen_mtp_routes(DECODER_ROUTES), prediction_depth=qwen_mtp_depth,
    preserve_source_layout=True,
    padding_idx=True,
    defaults={'max_position_embeddings': 32768},
    prediction_counts=('mtp_num_hidden_layers',),
)
"""SenseNova-Loop's language model, both ways (the module doc)."""
