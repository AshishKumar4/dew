"""The workload shared by the two Qwen3.5 backbone-step runners.

The seed draws NumPy PCG64 token rows before either framework places them;
neither framework's initializer or RNG consumption can change the inputs.
The projection is one float32 linspace divided by sqrt(hidden width), as
the original torch reference defined it. Its bytes and the token bytes are
hashed into each record. Historical torch results used torch.randint and
are not retroactively a matched pair with this workload.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

MODEL = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
WHOLE_PEAK_SCOPE = ("process start through timed-window end; includes load/compile/warmup, "
                    "excludes diagnostics")


def config(smoke: bool) -> dict:
    """Both frameworks and the work writer build the same checkpoint or tiny geometry."""
    if smoke:
        return {
            "model_type": "qwen3_5_text", "vocab_size": 128, "hidden_size": 32,
            "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": 16, "linear_key_head_dim": 8,
            "linear_value_head_dim": 8, "linear_num_key_heads": 2, "linear_num_value_heads": 2,
            "layer_types": ["linear_attention", "full_attention"], "use_cache": False,
            "tie_word_embeddings": True,
            "rope_parameters": {"rope_type": "default", "rope_theta": 1e6, "partial_rotary_factor": 0.5,
                                "mrope_interleaved": True, "mrope_section": [2, 1, 1]},
        }
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(MODEL, "config.json", revision=REVISION))
    return json.loads(path.read_text())["text_config"]


def inputs(seed: int, vocab: int, hidden: int, batch: int, sequence: int):
    """Identical host values, independent of device RNG and model initialization."""
    tokens = np.random.Generator(np.random.PCG64(seed)).integers(
        0, vocab, (batch, sequence), dtype=np.int32)
    projection = np.linspace(-1, 1, hidden, dtype=np.float32) / np.float32(hidden ** 0.5)
    return tokens, projection


def workload(seed: int, vocab: int, hidden: int, batch: int, sequence: int, smoke: bool,
             warmup: int, steps: int) -> dict:
    """Only matched arithmetic and input fields; implementations are recorded separately."""
    tokens, projection = inputs(seed, vocab, hidden, batch, sequence)
    return {
        "model": MODEL, "revision": REVISION, "batch": batch, "sequence": sequence, "seed": seed,
        "vocab": vocab, "hidden": hidden, "warmup": warmup, "steps": steps,
        "tokens": "fixed NumPy PCG64 synthetic unpadded causal rows, resident on device",
        "tokens_sha256": hashlib.sha256(tokens.tobytes()).hexdigest(),
        "projection_sha256": hashlib.sha256(projection.tobytes()).hexdigest(),
        "loss": "squared fixed scalar hidden-state projection; no vocabulary projection",
        "parameters": "fp32", "compute": "fp32 CPU smoke" if smoke else "bf16 compute and residuals",
        "optimizer": "AdamW", "optimizer_state": "fp32", "lr": 2.5e-5,
        "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 1e-4,
        "decay_mask": "ndim >= 2 and leaf name != bias, as Qwen3.5-0.8B",
        "schedule": "warmup_cosine(init=0,peak=2.5e-5,warmup=10,decay_steps=220,end=0)",
        "gradient_clipping": None, "gradient_checkpointing": False, "gradient_accumulation": 1,
        "ema": False, "full_finetune": True,
        "timing": "wall window synchronized only at both endpoints; no profiling or compilation",
    }
