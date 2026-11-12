"""A process that serves the tiny decision task's requests from a scoring server, keeping its programs or
handed them (tests/test_decision_serving.py): `python kept_scoring_worker.py keep|take|verify <file>`.
Prints each request's response, the programs kept and refused or the traces and compiles the server
needed, as JSON; `verify` hands the programs with each one's own lowering held to the kept one's
(`dew.training.ahead.kept`'s `verify`)."""

import json
import sys
from dataclasses import replace
from pathlib import Path

from dew.decision import Decide
from dew.decision.task import Budget
from dew.inference import ScoringServer
from dew.telemetry.compiles import Compiles
from dew.training.ahead import keeping, kept

mode, path = sys.argv[1], sys.argv[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "laya"
REQUEST = json.loads((FIXTURES / "cases.json").read_text())["quickstart"]
BODIES = [REQUEST, {**REQUEST, "state": REQUEST["state"] * 4},
          {**REQUEST, "questions": dict(list(REQUEST["questions"].items())[:1])}]
# One row a pass, so a window count at each of the layout's lengths.
task = replace(Decide.from_pretrained(FIXTURES / "tiny", attention_impl="xla"),
               budget=Budget(options=16, rows=1))


def served(server: ScoringServer) -> list:
    futures = [server.submit(body) for body in BODIES]
    server.run()
    return [future.result() for future in futures]


found: dict = {}
if mode == "keep":
    with keeping() as programs:
        server = ScoringServer.from_task(*task.scoring())
    saved = programs.save(path)
    found = {"kept": len(saved.kept), "programs": len(server.programs),
             "refused": list(programs.refused.values())}
    responses = served(server)
elif mode == "verify":
    with kept(path, verify=True):
        server = ScoringServer.from_task(*task.scoring())
    found = {"programs": len(server.programs)}
    responses = served(server)
else:
    with Compiles("logits") as compiles:
        with kept(path):
            server = ScoringServer.from_task(*task.scoring())
        responses = served(server)
    found = {"programs": len(server.programs),
             "needed": sorted(program.kind for program in compiles.programs)}
print(json.dumps({"responses": json.loads(json.dumps(responses, default=str)), **found}))
