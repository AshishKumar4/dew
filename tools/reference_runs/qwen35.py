"""The workload shared by the two Qwen3.5 backbone-step runners.

The seed draws NumPy PCG64 token rows before either framework places them;
neither framework's initializer or RNG consumption can change the inputs.
The projection is one float32 linspace divided by sqrt(hidden width), as
the original torch reference defined it. Its bytes and the token bytes are
hashed into each record. Historical torch results used torch.randint and
are not retroactively a matched pair with this workload.
"""

import hashlib

import numpy as np

MODEL = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
FLOPS_PER_STEP = 81_029_926_748_160
TOKENS_PER_STEP = 8 * 3072
WHOLE_PEAK_SCOPE = ("process start through timed-window end; includes load/compile/warmup, "
                    "excludes diagnostics")


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
        "model_flops_per_step": None if smoke else FLOPS_PER_STEP,
    }
