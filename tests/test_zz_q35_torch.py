"""Scratch direct transformers diagnostic."""
import sys
from pathlib import Path
import numpy as np
import torch
import transformers
import test_multimodal_model as T


def test_torch():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'tools'))
    from multimodal_reference import numerics
    from tools.torch_precision import float64, float64_twin
    from transformers import AutoModelForImageTextToText
    source=T.FIXTURE.parent / T.FAMILIES['qwen35'][1]
    torch.set_num_threads(1)
    seed=10
    inputs=T.drawing_text('qwen35')(seed)
    encoded={k:torch.from_numpy(v) for k,v in T.text_rows(inputs['ids']).items()}
    model=AutoModelForImageTextToText.from_pretrained(source,dtype=torch.float32,attn_implementation='eager').eval()
    twin=float64_twin(model)
    a=numerics(model,encoded)['updated_logits'].astype(np.float64)
    with float64():
        b=numerics(twin,encoded)['updated_logits']
    e=(a-b)**2
    pos=e.sum(-1); k=np.unravel_index(pos.argmax(),pos.shape)
    norm=np.sqrt(np.mean(e[3,0]))
    raise AssertionError('PROBE\ntransformers %s torch %s seed %d mse %.12g peak %s fraction %.8g maxerr %.8g row3pos0rms %.8g' %(transformers.__version__,torch.__version__,seed,e.mean(),k,pos[k]/pos.sum(),np.abs(a-b).max(),norm))
