"""A pinned text, code and chat pretraining mixture: three public Hub datasets at fixed commits, one tokenizer
at a fixed commit, mixed by tokens and packed into rows a trainer reads.

    python recipes/lm/mixture.py --show 2000

The sources are English Wikipedia's 2023-11-01 dump (prose, `text`), The Stack v1's filtered Python
functions (`content`) and UltraChat 200k's SFT conversations (`messages`), each read as the Hub stores it,
streamed from its own Parquet shards. Text and code are tokenized and ended by EOS; a conversation is rendered through the
tokenizer's chat template, and keeps only its tokens, as a pretraining row does not weigh speakers apart. The
tokenizer is SmolLM2's, whose template renders each message the same whatever follows it, as `chat.template`
needs (Qwen3's drops an earlier turn's reasoning once a later one follows, which it refuses). The
three are mixed at 70/20/10 by valid target tokens (`mix(by="tokens")`) and packed first-fit into rows of
`SEQ_LEN + 1` tokens. `--show N` reads the first N mixed documents and prints what a run would log of them
(`Dataset.stats`), then each source's tokens among them and its share beside the share asked; `build` returns
the batched pipeline for `trainer.fit`.
"""

import dataclasses

import tyro

from dew.data import Hub, chat, mix, text
from dew.data.pipeline import Pipeline

TOKENIZER = "HuggingFaceTB/SmolLM2-360M-Instruct"
TOKENIZER_REVISION = "a10cc1512eabd3dde888204e902eca88bddb4951"
SEQ_LEN = 2048
WEIGHTS = {"web": 0.7, "code": 0.2, "chat": 0.1}


def tokenizer():
    """The mixture's tokenizer at its pinned commit, which every step names by its files' hash."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER, revision=TOKENIZER_REVISION)


def sources(tok) -> dict[str, Pipeline]:
    """Each source at its pinned commit, read from its own shards as it streams, as token documents under
    "text"."""
    web = Hub("wikimedia/wikipedia", name="20231101.en", columns=("text",),
              revision="b04c8d1ceb2f5cd4588862100d08de323dccfbaa")
    code = Hub("bigcode/python-stack-v1-functions-filtered", columns=("content",),
               revision="8e72b18ab4a29b52fbd6a39c1441f67d09633ade")
    conversations = Hub("HuggingFaceH4/ultrachat_200k", split="train_sft", columns=("messages",),
                        revision="8049631c405ae6576f93f445c6b8166f76f5505a")
    return {"web": web.read("streamed").map(text.tokenize(tok, eos=True)).named("web"),
            "code": code.read("streamed").map(text.tokenize(tok, field="content", eos=True)).named("code"),
            "chat": conversations.read("streamed").map(chat.template(tok)).select_columns("text")
            .named("chat")}


def mixed(tok) -> Pipeline:
    """The three sources mixed at `WEIGHTS` by valid target tokens, unpacked."""
    parts = sources(tok)
    return mix([parts[name] for name in WEIGHTS], list(WEIGHTS.values()), by="tokens")


def build(batch: int, seed: int = 0) -> Pipeline:
    """Rows of `SEQ_LEN + 1` tokens, packed first-fit and shuffled, in batches of `batch`."""
    return mixed(tokenizer()).pack(seq_len=SEQ_LEN + 1).shuffle(seed=seed, window=10_000).batch(batch)


@dataclasses.dataclass(frozen=True)
class Args:
    show: int = 2000
    """Mixed documents to read and count."""


def main(args: Args) -> None:
    tok = tokenizer()
    counts = mixed(tok).stats(records=args.show)
    for metric, value in sorted(counts.items()):
        print(f"{metric}: {value:g}")
    # The mix read each source's first rows in order: the same rows read alone hold its tokens.
    parts = sources(tok)
    tokens = {name: parts[name].stats(records=int(counts[f"data/rows/{name}"]))["data/tokens"]
              for name in WEIGHTS}
    total = sum(tokens.values())
    for name, weight in WEIGHTS.items():
        print(f"{name}: {tokens[name]:.0f} tokens, {tokens[name] / total:.3f} of the mix (asked {weight})")


if __name__ == "__main__":
    main(tyro.cli(Args))
