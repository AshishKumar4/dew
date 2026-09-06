#!/usr/bin/env python3
"""Write TRL's assistant mask for the tiny chat fixture to tests/fixtures/rl/chat.npz.

The reference is TRL 1.12's SFT path with `assistant_only_loss=True`: the
tokenizer already carries `{% generation %}` markers, so
`get_training_chat_template` returns None and the mask is transformers'
`return_assistant_tokens_mask` over the conversation
(`trl/trainer/sft_trainer.py`, `tokenize_fn`, the conversational branch).
Dew's `render_conversation` must produce the same ids and the same assistant
span. Runs in an environment with TRL installed; Dew never imports TRL.

The conversation exercises two assistant turns, a system message and a tool
message. Its JSON is echoed in the fixture, and the SFT test reads the
conversation back from there.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
TOKENIZER = FIXTURES / "tokenizers" / "tiny-chat"

CONVERSATION = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is two plus two?"},
    {"role": "assistant", "content": "Two plus two is four."},
    {"role": "tool", "content": "4"},
    {"role": "user", "content": "And three plus three?"},
    {"role": "assistant", "content": "Three plus three is six."},
]


def main(argv: list[str] | None = None) -> None:
    import transformers
    import trl
    from transformers import AutoTokenizer
    from trl.chat_template_utils import (
        get_training_chat_template,
        is_chat_template_prefix_preserving,
        is_chat_template_stop_token_trained,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=FIXTURES / "rl" / "chat.npz")
    out = parser.parse_args(argv).out

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    training_template = get_training_chat_template(tokenizer)
    assert training_template is None, (
        "the fixture template already carries generation markers, so TRL must "
        f"use it as is, got a replacement: {training_template!r:.80}")
    prefix_preserving = is_chat_template_prefix_preserving(tokenizer)
    stop_trained = is_chat_template_stop_token_trained(tokenizer)
    rendered = tokenizer.apply_chat_template(
        CONVERSATION, tokenize=True, return_dict=True,
        return_assistant_tokens_mask=True)
    ids = np.asarray(rendered["input_ids"], dtype=np.int64)
    mask = np.asarray(rendered["assistant_masks"], dtype=np.int64)
    assert 1 in mask.tolist(), "the fixture conversation has no assistant tokens"

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        input_ids=ids,
        assistant_mask=mask,
        conversation=np.array(json.dumps(CONVERSATION)),
        template=np.array("trl_qwen3_training"),
        trl_version=np.array(trl.__version__),
        transformers_version=np.array(transformers.__version__),
        prefix_preserving=np.asarray(prefix_preserving),
        stop_token_trained=np.asarray(stop_trained),
    )
    print(f"{out}: {out.stat().st_size / 1e3:.1f} kB")
    print(f"  {len(ids)} tokens, {int(mask.sum())} assistant, "
          f"prefix_preserving={prefix_preserving} stop_trained={stop_trained}")


if __name__ == "__main__":
    main()
