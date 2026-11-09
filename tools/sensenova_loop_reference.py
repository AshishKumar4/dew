#!/usr/bin/env python3
"""The SenseNova-Loop fixture tests/test_sensenova_loop.py checks against, from SenseNova's own remote code
(sensenova/SenseNova-Loop-9B-Preview at 9937e91c, modeling_sensenova_loop.py) on the suite's transformers:

    PYTHONPATH=src:tools:tests .venv-3.12/bin/python tools/sensenova_loop_reference.py [--recording]

tests/fixtures/hf/sensenova-loop-tiny/ is a random SenseNovaLoopForConditionalGeneration over
qwen35-tiny-mm's text config at eight layers, three gated delta nets and an attention layer twice, the
layers [2, 6) run twice (`loop_settings`) and two mHC streams over them (`mhc`, their read scaled by 0.75
where the release's is 1, so the scale is read), beside qwen35-tiny-mm's tower, which a text-only load
leaves out: its config, its weights and the input ids whose shape check 1's draw takes. `--recording`
records instead its logits over check 1's seeds (`recorded_references.record_logits`), as
recorded_logits.npz there, against its float64 twin: the release's routers compute in float32 whatever
the model's dtype (`MHCRouter._apply`, :104-114), so the twin's take their mixing in float64
(`_wide_routers`), the truth of the same formula.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from reference_source import hub_modules

REPO, REVISION = 'sensenova/SenseNova-Loop-9B-Preview', '9937e91c7301b7dcbb5bf17e3e4e95c499a11e99'
FIXTURES = Path(__file__).resolve().parents[1] / 'tests' / 'fixtures' / 'hf'
TINY = FIXTURES / 'sensenova-loop-tiny'
SHAPE = (4, 48)
"""The ids' shape check 1 draws in."""


def remote():
    """SenseNova-Loop's modeling module, imported from the pinned remote code."""
    (modeling,) = hub_modules('sensenova_loop_remote', REPO, REVISION, ('modeling_sensenova_loop.py',))
    return modeling


def tiny_config(modeling):
    """The tiny config: qwen35-tiny-mm's wrapper and tower, its text at eight layers, the loop and the
    streams over [2, 6)."""
    base = json.loads((FIXTURES / 'qwen35-tiny-mm' / 'config.json').read_text())
    layers = base['text_config']['layer_types'] * 2
    text = {**base['text_config'], 'num_hidden_layers': len(layers), 'layer_types': layers}
    tokens = {key: base[key] for key in ('image_token_id', 'video_token_id', 'vision_start_token_id',
                                         'vision_end_token_id')}
    return modeling.SenseNovaLoopConfig(
        text_config=text, vision_config=base['vision_config'], tie_word_embeddings=False, **tokens,
        loop_settings={'start': 2, 'end': 6, 'eval_count': 2, 'counts': [2]},
        mhc={'start': 2, 'end': 6, 'streams': 2, 'iterations': 20, 'fixed_identity': False,
             'pre_scale': 0.75})


def write_tiny(modeling) -> None:
    from dew.interop.verify import scatter_weights

    TINY.mkdir(parents=True, exist_ok=True)
    config = tiny_config(modeling)
    config._attn_implementation = 'eager'
    torch.manual_seed(0)
    model = modeling.SenseNovaLoopForConditionalGeneration(config).float().eval()
    scatter_weights(model, 1234)
    model.save_pretrained(TINY, safe_serialization=True)
    np.save(TINY / 'input_ids.npy', np.random.default_rng(0).integers(0, config.text_config.vocab_size,
                                                                     SHAPE).astype(np.int32))


def _wide_routers(modeling, model) -> None:
    """`model`'s routers computing their mixing in float64, the release's formula (`MHCRouter.forward`,
    `sinkhorn`, :77-135) without its float32 casts, their weights promoted."""

    def sinkhorn(logits, iterations):
        for _ in range(iterations):
            logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
            logits = logits - torch.logsumexp(logits, dim=-2, keepdim=True)
        return logits.exp()

    class WideRouter(modeling.MHCRouter):
        def _apply(self, fn, recurse=True):
            return torch.nn.Module._apply(self, fn, recurse)

        def forward(self, residual):
            flat = residual.flatten(-2)
            normalized = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)
            logits = torch.nn.functional.linear(normalized, self.weight, self.bias)
            streams = self.streams
            pre = logits[..., :streams].sigmoid()
            post = 2 * logits[..., streams:2 * streams].sigmoid()
            mix = sinkhorn(logits[..., 2 * streams:].unflatten(-1, (streams, streams)), self.iterations)
            branch = torch.matmul(pre.unsqueeze(-2), residual).squeeze(-2)
            return (branch * self.pre_scale if self.pre_scale != 1.0 else branch), post, mix

    for module in model.modules():
        if isinstance(module, modeling.MHCRouter):
            module.__class__ = WideRouter
            module.weight.data, module.bias.data = module.weight.data.double(), module.bias.data.double()


def write_recording(modeling) -> None:
    """sensenova-loop-tiny's committed weights in the remote code's class, fp32 on eager attention, and
    their float64 twin (`_wide_routers` its routers), recorded over check 1's seeds of ids."""
    import transformers
    from recorded_references import record_logits
    from torch_precision import float64_twin

    model = modeling.SenseNovaLoopForConditionalGeneration.from_pretrained(
        TINY, dtype=torch.float32, attn_implementation='eager').eval()
    truth = float64_twin(model, lambda: modeling.SenseNovaLoopForConditionalGeneration(model.config))
    _wide_routers(modeling, truth)
    vocabulary = model.config.text_config.vocab_size

    def draw(seed):
        return {'input_ids': np.random.default_rng(seed).integers(0, vocabulary, SHAPE).astype(np.int32)}

    command = 'PYTHONPATH=src:tools:tests .venv-3.12/bin/python tools/sensenova_loop_reference.py --recording'
    record_logits(TINY, model, truth, command, reference='SenseNovaLoopForConditionalGeneration (remote code)',
                  revision=f'{REPO}@{REVISION}', license='Apache-2.0', transformers=transformers.__version__,
                  draw=draw, planned=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recording', action='store_true',
                        help="record the tiny model's logits over the seeds")
    arguments = parser.parse_args()
    modeling = remote()
    if arguments.recording:
        write_recording(modeling)
    else:
        write_tiny(modeling)
