#!/usr/bin/env python3
"""Audit whether a chat template keeps agent histories append-only in token space.

`pack` merges call k + 1 into call k's training row only when call k + 1's
prompt ids start with call k's prompt ids plus its sampled ids
(`dew.objectives.rl.sessions.merges`). Whether that holds is decided by the
model family's chat template and by harness settings: how observations
return (a `tool` or a `user` message), whether reasoning is kept in
history, how tool-call arguments are re-serialized. Run this before
training a new family or harness setting (agentic RL memo, section 6).

Two audits, one JSON report on stdout:

- `template`: renders a two-turn tool episode with the tokenizer's own chat
  template under each harness setting, takes the sampled ids as what that
  template writes for the assistant turn through its end-of-turn token, and
  checks the strict rule and the prompt-only rule. A case where the
  prompt-only rule holds and the strict one fails is one a lenient merger
  would get wrong. Sampled ids are canonical encodings, so a real sampler
  can only add divergences.
- `sessions`: reads recorded sessions, one JSON object per line with
  `calls: [{prompt_ids, sampled_ids}, ...]` (a gateway trace mapped to
  Dew's `Call` fields), and reports calls per chain and where each split
  diverged.

  ../../.venv/bin/python tools/audit_template.py template --tokenizer Qwen/Qwen3-0.6B
  ../../.venv/bin/python tools/audit_template.py sessions traces.jsonl
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

TOOLS = [{"type": "function", "function": {
    "name": "bash", "description": "run a shell command",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
def divergence(left: Sequence[int], right: Sequence[int]) -> int | None:
    """The first index where two id sequences differ, or None when one prefixes the other."""
    return next((index for index, (a, b) in enumerate(zip(left, right, strict=False)) if a != b), None)


def template_case(tokenizer, end: int, observation_role: str, *, reasoning: bool,
                  arguments_as_string: bool = False, compact_json: bool = False) -> dict:
    """Render turn 1, derive the sampled turn from the template, render turn 2, check both rules.

    The sampled ids are what the template itself writes for the assistant
    turn, from the end of the generation prompt through the first end token,
    so a family's own tool-call and reasoning syntax is what the model is
    taken to have sampled. `compact_json` has the model sample the arguments
    as compact JSON that the server then parses; the history carries the
    parsed arguments for the template to re-serialize.
    """
    from dew.objectives.rl.sessions import Call, merges

    def render(messages, prompt=True):
        return list(tokenizer.apply_chat_template(messages, tools=TOOLS, add_generation_prompt=prompt,
                                                  tokenize=True, return_dict=False))

    def assistant(arguments):
        turn = {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_0", "type": "function", "function": {"name": "bash", "arguments": arguments}}]}
        if reasoning:
            turn["reasoning_content"] = "I should list the files first."
        return turn

    messages = [{"role": "system", "content": "You are a coding agent."},
                {"role": "user", "content": "Fix the failing test in repo/."}]
    parsed = '{"command": "ls"}' if arguments_as_string else {"command": "ls"}
    first = render(messages)
    written = render([*messages, assistant('{"command":"ls"}' if compact_json else parsed)], prompt=False)
    if written[:len(first)] != first:
        raise ValueError("the template renders an assistant turn that does not start with its own "
                         "generation prompt, so the sampled turn cannot be read off it")
    tail = written[len(first):]
    if end not in tail:
        raise ValueError("the template's assistant turn never writes the end token")
    sampled = tail[:tail.index(end) + 1]
    observation = ({"role": "tool", "tool_call_id": "call_0", "name": "bash", "content": "src/ tests/"}
                   if observation_role == "tool" else {"role": "user", "content": "Observation: src/ tests/"})
    second = render([*messages, assistant(parsed), observation])
    history = [*first, *sampled]
    strict = merges(history, Call(tuple(second), (), (), "stop", 0))
    at = divergence(second, history)
    return {
        "observation_role": observation_role, "reasoning_in_sampled_turn": reasoning,
        "arguments_as_json_string": arguments_as_string, "sampled_compact_json": compact_json,
        "strict_prefix_holds": strict, "prompt_only_prefix_holds": second[:len(first)] == first,
        "first_divergence_index": None if strict else at,
        "divergence": None if strict or at is None else {
            "next_prompt": tokenizer.decode(second[max(0, at - 5):at + 12]),
            "prompt_plus_sampled": tokenizer.decode(history[max(0, at - 5):at + 12])},
    }


def audit_template(name: str, end_token: str | None) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    end = tokenizer.convert_tokens_to_ids(end_token) if end_token else tokenizer.eos_token_id
    if end is None:
        raise ValueError("the tokenizer names no EOS; pass --end-token")
    cases = [template_case(tokenizer, end, role, reasoning=reasoning)
             for role in ("tool", "user") for reasoning in (False, True)]
    cases.append(template_case(tokenizer, end, "tool", reasoning=False, arguments_as_string=True))
    cases.append(template_case(tokenizer, end, "tool", reasoning=False, compact_json=True))
    return {"tokenizer": name, "end_token": tokenizer.convert_ids_to_tokens(end),
            "append_only": all(case["strict_prefix_holds"] for case in cases),
            "lenient_merge_would_corrupt": [case for case in cases
                                            if case["prompt_only_prefix_holds"] and not case["strict_prefix_holds"]],
            "cases": cases}


def audit_sessions(lines: Sequence[str]) -> dict:
    """Calls per strict chain over recorded sessions, and where every split diverged."""
    from dew.objectives.rl.sessions import Call, Session, Status, chains

    calls_total = chains_total = 0
    splits = []
    for number, line in enumerate(lines):
        if not line.strip():
            continue
        record = json.loads(line)
        calls = tuple(Call(tuple(call["prompt_ids"]), tuple(call["sampled_ids"]),
                           tuple(float(value) for value in call.get("behavior_log_probs",
                                                                    [0.0] * len(call["sampled_ids"]))),
                           "stop", 0) for call in record["calls"])
        if not calls:
            continue
        width = max(len(call.prompt_ids) + len(call.sampled_ids) for call in calls)
        built = chains(Session("audit", "audit", 0, 0, calls, Status.COMPLETED, 0.0), width)
        calls_total += len(calls)
        chains_total += len(built)
        history: list[int] = []
        for index, call in enumerate(calls):
            if index and tuple(call.prompt_ids[:len(history)]) != tuple(history):
                splits.append({"session": record.get("id", number), "call": index,
                               "first_divergence_index": divergence(call.prompt_ids, history),
                               "prompt_only_prefix_holds": tuple(call.prompt_ids[:len(calls[index - 1].prompt_ids)])
                               == calls[index - 1].prompt_ids})
            history = [*call.prompt_ids, *call.sampled_ids]
    return {"calls": calls_total, "chains": chains_total,
            "calls_per_chain": calls_total / chains_total if chains_total else 0.0, "splits": splits}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    template = commands.add_parser("template", help="render a tool episode with a chat template")
    template.add_argument("--tokenizer", required=True, help="Hugging Face tokenizer name or directory")
    template.add_argument("--end-token", help="the end-of-turn token the engine samples; default EOS")
    sessions = commands.add_parser("sessions", help="read recorded sessions as JSON lines")
    sessions.add_argument("path", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "template":
        report = audit_template(arguments.tokenizer, arguments.end_token)
    else:
        report = audit_sessions(arguments.path.read_text().splitlines())
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
