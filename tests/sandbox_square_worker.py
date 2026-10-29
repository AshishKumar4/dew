"""A JSON-lines square tool worker for SubprocessEnvironment tests.

Argument one selects a failure mode: ok, hang, allocate, garbage, crash or
fork; or wait, whose reset waits until the pipe named by argument three
carries a byte; or a chat harness, chat or compact, whose observations carry
messages and the request it parsed from each action (`chat_turn`), compact's
tool result compacting the context. The square tool vocabulary matches
tests/test_tool_episodes.py.
"""

import base64
import json
import os
import sys
import time
from pathlib import Path

PAD, START, CALL_THREE, EOS, NINE, ANSWER_NINE, WRONG, CALL_FOUR, SIXTEEN, ANSWER_SIXTEEN = range(10)
TASK = {"role": "user", "content": "square three or four, then give the square"}


def chat_turn(mode, value=None, given=None):
    """The messages and parsed request a chat harness (`mode`) observes: the task at reset, the tool
    call and its result (compact: a summary replacing the history) for `value`, and the answer `given`."""
    if value is None and given is None:
        return {"messages": [TASK]}
    if given is not None:
        return {"messages": [], "request": {"role": "assistant", "content": str(given)}}
    call = {"id": "call-0", "type": "function",
            "function": {"name": "square", "arguments": json.dumps({"value": value})}}
    request = {"role": "assistant", "tool_calls": [call]}
    if mode == "compact":
        summary = {"role": "user", "content": f"square({value}) = {value * value}; give the square"}
        return {"messages": [summary], "request": request, "compacted": True}
    return {"messages": [{"role": "tool", "tool_call_id": "call-0", "content": str(value * value)}],
            "request": request}


def reply(**observation):
    sys.stdout.write(json.dumps(observation) + "\n")
    sys.stdout.flush()


def main():
    mode = sys.argv[1]
    answer = None
    identity = None
    for line in sys.stdin:
        request = json.loads(line)
        if request["operation"] == "get_state":
            state = json.dumps({"answer": answer, "identity": identity}).encode()
            reply(state=base64.b64encode(state).decode("ascii"))
            continue
        if request["operation"] == "set_state":
            state = json.loads(base64.b64decode(request["state"]))
            answer, identity = state["answer"], state["identity"]
            reply(restored=True)
            continue
        if request["operation"] == "reset":
            identity = request["episode"]
            if mode == "hang":
                time.sleep(3600)
            if mode == "wait":
                with open(sys.argv[3], "rb") as gate:
                    gate.read(1)
            if mode == "cpu":
                while True:
                    pass
            if mode == "allocate":
                block = bytearray(4 * 1024 ** 3)
                block[-1] = 1
            if mode == "garbage":
                sys.stdout.write("not json\n")
                sys.stdout.flush()
                continue
            if mode == "crash":
                sys.stderr.write("worker boom\n")
                sys.exit(3)
            pids = [os.getpid()]
            if mode == "fork":
                child = os.fork()
                if child == 0:
                    time.sleep(3600)
                    os._exit(0)
                pids.append(child)
            reply(context=[START], status="running", detail=json.dumps(pids),
                  **(chat_turn(mode) if mode in ("chat", "compact") else {}))
            continue
        action = request["action"]
        command = action["tokens"][:-1]
        if mode == "pause_final" and answer is not None:
            Path(sys.argv[3]).write_text("pending final action")
            time.sleep(3600)
        if len(sys.argv) > 2:
            with open(sys.argv[2], "a") as audit:
                audit.write(json.dumps({"identity": identity, "tokens": action["tokens"]}) + "\n")
                audit.flush()
                os.fsync(audit.fileno())
        if answer is None:
            value = {CALL_THREE: 3, CALL_FOUR: 4}[command[0]]
            answer = value * value
            observation = {9: NINE, 16: SIXTEEN}[answer]
            if mode in ("chat", "compact"):
                # A compacted context restarts from the tool's result.
                context = ([START, observation] if mode == "compact"
                           else [*action["context"], *action["tokens"], observation])
                reply(context=context, status="running", detail=f"square({value}) = {answer}",
                      **chat_turn(mode, value))
                continue
            reply(context=[*action["context"], *action["tokens"], observation], status="running",
                  detail=f"square({value}) = {answer}")
            continue
        given = {ANSWER_NINE: 9, ANSWER_SIXTEEN: 16, WRONG: 8}[command[0]]
        reply(context=[], status="completed", detail=json.dumps({"answer": given, "expected": answer}),
              **(chat_turn(mode, given=given) if mode in ("chat", "compact") else {}))


if __name__ == "__main__":
    main()
