"""Check 1's draws for the Qwen-Image source checks (tests/test_qwen_image_source.py), which
tools/diffusers_qwen_image_reference.py records the source over: a case's walk's inputs (`case_draw`) and
the pipeline's (`pipeline_draw`), each a function of its seed and the fixture's record (`qwen_image.json`),
which states every case's geometry and the pipeline's prompts and size."""

import numpy as np

LETTERS = np.asarray(list("abcdefghijklmnopqrstuvwxyz "))
"""The characters a drawn prompt is made of, each one token of the byte tokenizer."""


def case_draw(record: dict, name: str):
    """Seed `seed`'s inputs to case `name`'s walk: its packed latents, its prompt states zero past each
    row's length and its cotangent, standard normal."""
    case = record["cases"][name]
    config, (rows, columns), lengths = case["config"], case["grid"], case["lengths"]
    batch, tokens = len(lengths), max(lengths)
    mask = np.arange(tokens)[None] < np.asarray(lengths)[:, None]
    index = list(record["cases"]).index(name)

    def draw(seed: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng([0x2094, index, seed])
        return {"packed": rng.standard_normal((batch, rows * columns, config["in_channels"]), np.float32),
                "context": (rng.standard_normal((batch, tokens, config["context_in_dim"]), np.float32)
                            * mask[..., None]),
                "probe": rng.standard_normal((batch, rows * columns, config["out_channels"]), np.float32)}
    return draw


def pipeline_draw(record: dict):
    """Seed `seed`'s inputs to the pipeline's walk: a prompt a row, drawn letters and spaces as long as
    the record's prompts (one token each, so every draw's prompts take their tokens), as UTF-8 bytes, and
    the initial latents the rows walk from, standard normal."""
    pipeline = record["pipeline"]
    prompts, config = pipeline["prompts"], pipeline["config"]
    rows, columns = pipeline["height"] // 16, pipeline["width"] // 16

    def draw(seed: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng([0x2094, len(record["cases"]), seed])
        drawn = {f"prompt.{row}": np.frombuffer("".join(rng.choice(LETTERS, len(prompt))).encode(), np.uint8)
                 for row, prompt in enumerate(prompts)}
        return {**drawn, "x_T": rng.standard_normal((len(prompts), rows * columns, config["in_channels"]),
                                                    np.float32)}
    return draw


def prompts_of(inputs: dict) -> list[str]:
    """A pipeline draw's prompts (`pipeline_draw`), as text."""
    count = sum(key.startswith("prompt.") for key in inputs)
    return [bytes(np.asarray(inputs[f"prompt.{row}"])).decode() for row in range(count)]
