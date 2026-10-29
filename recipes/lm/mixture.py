"""A pinned text, code and chat pretraining mixture: three public Hub datasets at fixed commits, one tokenizer
at a fixed commit, mixed by tokens and packed into rows a trainer reads.

    python recipes/lm/mixture.py --show 2000

The sources are FineWeb-Edu's 10B-token sample (web text, `text`), CodeParrot's clean Python validation split
(`content`) and UltraChat 200k's SFT conversations (`messages`), each read as the Hub stores it, streamed from
its own shards. Text and code are tokenized and ended by EOS; a conversation is rendered through the
tokenizer's chat template, and keeps only its tokens, as a pretraining row does not weigh speakers apart. The
three are mixed at 70/20/10 by valid target tokens (`mix(by="tokens")`) and packed first-fit into rows of
`SEQ_LEN + 1` tokens. `--show N` reads the first N mixed documents and prints what a run would log of them
(`Dataset.stats`): each source's rows and tokens, whose shares stand beside the shares asked; `build` returns
the batched pipeline for `trainer.fit`.
"""

import dataclasses

import tyro

from dew.data import Hub, chat, mix, text
from dew.data.pipeline import Pipeline

TOKENIZER, TOKENIZER_REVISION = "Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"
SEQ_LEN = 2048
WEIGHTS = {"web": 0.7, "code": 0.2, "chat": 0.1}


def tokenizer():
    """The mixture's tokenizer at its pinned commit, which every step names by its files' hash."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER, revision=TOKENIZER_REVISION)


def sources(tok) -> dict[str, Pipeline]:
    """Each source at its pinned commit, as token documents under "text"."""
    web = Hub("HuggingFaceFW/fineweb-edu", name="sample-10BT", columns=("text",),
              revision="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9")
    code = Hub("codeparrot/codeparrot-clean-valid", columns=("content",),
               revision="4db92d2ec0c1b4c41eeb439cfae16854511d9dcd")
    conversations = Hub("HuggingFaceH4/ultrachat_200k", split="train_sft", columns=("messages",),
                        revision="8049631c405ae6576f93f445c6b8166f76f5505a")
    return {"web": web.map(text.tokenize(tok, eos=True)).named("web"),
            "code": code.map(text.tokenize(tok, field="content", eos=True)).named("code"),
            "chat": conversations.map(chat.template(tok)).select_columns("text").named("chat")}


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
    for metric, value in sorted(mixed(tokenizer()).stats(records=args.show).items()):
        print(f"{metric}: {value:g}")


if __name__ == "__main__":
    main(tyro.cli(Args))
