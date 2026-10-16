"""The revision-pinned published tiny Gemma 4's logits in float64, on the text and image batches its
wrapper parity measurement reads (`tools.wrapper_checkpoint_parity.parity_batches`).

Measured with Transformers 5.16.1 and Torch 2.14.0 on CPU. The model is built in float64 from its
config (`float64`), so its rotary tables, the vision tower's among them, are computed in float64 rather
than widened from float32, and every float32 pin in it (the RMSNorms, softmax, the vision pooler's and
standardizer's casts) is widened; the fp32 checkpoint weights load into it exactly.
"""

import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from tools.diffusers_wan_reference import float64
from tools.wrapper_checkpoint_parity import parity_batches

CHECKPOINT = 'trl-internal-testing/tiny-Gemma4ForConditionalGeneration'
REVISION = '0dc1746b7f9f623b748e735ed5a4302eb4baf346'


def reference_logits() -> list:
    """Each parity batch's (float64 logits, token ids), the text batch's first, on CPU from the pinned
    fp32 weights."""
    torch.set_num_threads(1)
    processor = AutoProcessor.from_pretrained(CHECKPOINT, revision=REVISION)
    weights = AutoModelForImageTextToText.from_pretrained(
        CHECKPOINT, dtype=torch.float32, attn_implementation='eager', revision=REVISION).state_dict()
    batches = parity_batches(processor, 'gemma4')
    with float64(), torch.no_grad():
        model = AutoModelForImageTextToText.from_config(
            AutoConfig.from_pretrained(CHECKPOINT, revision=REVISION), dtype=torch.float64,
            attn_implementation='eager').eval()
        model.load_state_dict(weights)
        return [(model(**batch, use_cache=False).logits.double().numpy(), batch['input_ids'].numpy())
                for batch in batches]
