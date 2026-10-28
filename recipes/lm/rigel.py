"""Rigel: a hybrid Mamba-2 + GQA base model with 128-expert top-2 MoE on every
layer, reconstructed from the Rigel blog post and lm-engine 45b6b57b.

    python recipes/lm/rigel.py --corpora web data/web code data/code \\
        --trainer.checkpoint-dir runs/rigel
    python recipes/lm/rigel.py --width 256 --corpora web data/web --steps 2000 \\
        --batch-size 32 --seq-len 1024

`--corpora` is name, directory pairs; the second line is a muP proxy on one
corpus, and any `LMRunConfig` flag follows the recipe's own.

The architecture reproduces the published parameter counts exactly at width
1024 (2,345,567,552 total, 260,998,464 active non-embedding): 40 layers in
M, M, M, A order, Mamba-2 at expand 2 with heads of 64 and a 128-wide state in
one group, 16/4 grouped-query attention with heads of 64 under exclusive self
attention and no positional encoding over a 4,096 window, 128 SwiGLU experts
of width d/8, top-2 softmax routing, tied 100,352-token embeddings.

The optimizer is lm-engine's muP AdamW on its power schedule as Rigel ran it:
5,000 warmup steps to 0.01, min(0.01, 4B * (step * tokens per step) ** -0.51)
with B the batch, then linear to zero over the last 29% of the steps. The
data follows Rigel's five phases at their published shares and boundaries,
rescaled with the run's length; a corpus the run does not name drops out of
every phase it appears in. Rigel's m_emb, m_residual, m_width, init range and
aux coefficient are unpublished: these are Granite 4.0-H's multipliers
(embedding 12, residual 0.22, logits 8 at width 1024, init range 0.1) and
lm-engine's example coefficient 0.001 with 0.1 of it on the router z-loss.
`--width` scales the model as a muP proxy: heads of 64 stay, head counts,
the Mamba-2 width and expert width follow d, and m_width is d / 128.
"""

import dataclasses
from collections.abc import Mapping

import tyro

from dew.config import ObjectiveConfig, TrainerConfig
from dew.config.lm import LMRunConfig
from dew.data import mix, phases, presets
from dew.data.pipeline import Pipeline
from dew.data.sources.tokens import TokenFiles
from dew.optim import OptimConfig, ParamGroup, Power, PowerTail
from dew.registry import ModelConfig

LAYERS = 40
VOCAB = 100_352
STEPS = 725_000
BATCH = 1_152
SEQ_LEN = 4_096

# Rigel's five phases: the step each ends before and its shares in percent
# (the blog's chart data; boundaries read off the W&B loss chart).
PHASES: tuple[tuple[int | None, Mapping[str, float]], ...] = (
    (215_000, {"web": 71, "code": 20, "math": 7, "multilingual": 2}),
    (315_000, {"web": 15, "code": 20, "math": 7, "multilingual": 2, "stem": 56}),
    (415_000, {"web": 57, "code": 18, "math": 19, "multilingual": 6}),
    (515_000, {"web": 2.2, "code": 35, "math": 35, "multilingual": 3.5, "nemotron_cc_v2": 20,
               "finepdf": 4.3}),
    (None, {"code": 19.1, "math": 6.9, "multilingual": 0.8, "stem": 11.5, "nemotron_cc_v2": 45.8,
            "finepdf": 11, "other": 4.9}),
)
DECAY_START = 515_000


def model_config(width: int = 1024) -> dict:
    """Rigel's `causal_transformer` fields at `width` (1024 is Rigel)."""
    if width % 128:
        raise ValueError(f"the width scales in heads of 64 and experts of width d/8; got {width}")
    heads = width // 64
    return {
        "emb_features": width, "num_layers": LAYERS, "num_heads": heads,
        "num_kv_heads": max(heads // 4, 1), "head_dim": 64,
        "qk_norm": False, "attention_bias": False, "tie_embeddings": True, "mlp": "swiglu",
        "layer_types": ["mamba", "mamba", "mamba", "attention"] * (LAYERS // 4),
        "kinds": {
            "mamba": {"mixer": {"class": "mamba2", "fields": {"num_heads": 2 * width // 64, "head_dim": 64,
                                "state_size": 128, "n_groups": 1, "conv_kernel": 4,
                                "chunk_size": 256, "use_conv_bias": True}}},
            "attention": {"window": 4096, "mixer": {"class": "attention", "fields": {"nope": True,
                                                     "exclusive_self_attention": True}}},
        },
        "mixture": {"experts": 128, "top_k": 2, "expert_features": width // 8},
        "embedding_multiplier": 12.0, "residual_multiplier": 0.22,
        "logits_scaling": width / 128, "initializer_range": 0.1, "depth_scaled_init": True,
    }


def optim_config(width: int, steps: int, batch: int, seq_len: int) -> OptimConfig:
    """lm-engine's muP AdamW on Rigel's power-then-linear schedule, its
    boundaries rescaled from 725,000 steps to `steps`."""
    scale = steps / STEPS
    return OptimConfig(
        optimizer="adamw", b1=0.9, b2=0.95, optimizer_opts={"eps": 1e-10},
        learning_rate=Power(peak_value=0.01, warmup_steps=max(round(5_000 * scale), 1),
                       a=4.0 * batch, b=-0.51, c=float(batch * seq_len),
                       tail=PowerTail(start=round(DECAY_START * scale))),
        weight_decay=0.1, clip_grads=1.0, param_groups=ParamGroup.mup(width / 128))


def schedule(corpora: Mapping[str, str], steps: int) -> tuple[tuple[dict[str, float], int | None], ...]:
    """Rigel's phases over the named corpus directories, each mixture of directories by share and the
    step it ends before, boundaries rescaled to `steps`, the last running to the end; a phase of one
    store holds it at 1.0. An empty phase extends the preceding mixture, or the first non-empty
    mixture when it leads the run."""
    scale = steps / STEPS
    mixtures: list[tuple[dict[str, float], int | None]] = []
    for until, shares in PHASES:
        end = None if until is None else round(until * scale)
        mixture = {corpora[name]: share for name, share in shares.items() if name in corpora}
        if not mixture:
            if mixtures:
                mixtures[-1] = (mixtures[-1][0], end)
            continue
        # One store reads all of its phase, whatever its share.
        mixture = dict.fromkeys(mixture, 1.0) if len(mixture) == 1 else mixture
        if mixtures and mixtures[-1][0] == mixture:
            mixtures[-1] = (mixture, end)
        else:
            mixtures.append((mixture, end))
    if not mixtures:
        raise ValueError(f"none of {sorted(corpora)} is a Rigel corpus: "
                         f"{sorted({name for _, shares in PHASES for name in shares})}")
    mixtures[-1] = (mixtures[-1][0], None)
    return tuple(mixtures)


def phased(corpora: Mapping[str, str], steps: int, seq_len: int, batch: int) -> Pipeline:
    """Rigel's phases (`schedule`) as one chain: each store packed by itself into rows of `seq_len + 1`
    tokens, a phase's stores mixed by their tokens, and the phases shuffled, repeated and batched
    together."""
    def packed(url: str) -> Pipeline:
        return TokenFiles(url).pipe(presets.lm, None, seq_len=seq_len, batch=None, pack="first-fit", bins=8)

    def phase(mixture: dict[str, float]) -> Pipeline:
        if len(mixture) == 1:
            return packed(next(iter(mixture)))
        return mix([packed(url) for url in mixture], list(mixture.values()), by="tokens")

    return phases(*((phase(mixture), until) for mixture, until in schedule(corpora, steps))
                  ).shuffle(0).repeat().batch(batch)


@dataclasses.dataclass(frozen=True)
class RigelArgs:
    """What sizes a Rigel run; everything else is `LMRunConfig`'s."""

    corpora: dict[str, str]
    """Rigel's corpus names to tokenized directories, as name, directory
    pairs (`--corpora web data/web code data/code`): web, code, math,
    multilingual, stem, nemotron_cc_v2, finepdf, other."""
    width: int = 1024
    steps: int = STEPS
    batch_size: int = BATCH
    seq_len: int = SEQ_LEN
    tokenizer: str = "ibm-granite/granite-4.0-h-micro"


def run_config(args: RigelArgs) -> LMRunConfig:
    """The LM run config for `args`."""
    return LMRunConfig(
        model=ModelConfig("causal_transformer", {**model_config(args.width), "dtype": "bfloat16"}),
        data=phased(args.corpora, args.steps, args.seq_len, args.batch_size),
        optim=optim_config(args.width, args.steps, args.batch_size, args.seq_len),
        trainer=TrainerConfig(steps=args.steps, eval_every=None, checkpoint_every=5_000),
        objective=ObjectiveConfig("lm", {"balance": {
            "aux_loss_alpha": 0.001, "seq_aux": False, "router_z_loss": 0.0001}}),
        tokenizer=args.tokenizer, max_new_tokens=0)


if __name__ == "__main__":
    args, rest = tyro.cli(RigelArgs, return_unknown_args=True)
    LMRunConfig.cli(rest, default=run_config(args)).run()
