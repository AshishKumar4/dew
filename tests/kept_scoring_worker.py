"""A process that serves the tiny decision task's requests from a scoring server, keeping its programs or
handed them (tests/test_decision_serving.py): `python kept_scoring_worker.py keep|take <file>`. Prints
each request's response, the programs kept and refused or the forward's compiles the server needed, as
JSON."""

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
task = replace(Decide.from_pretrained(FIXTURES / "tiny"),
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
else:
    with Compiles("logits") as compiles:
        # Each program traced and lowered, and run as kept where it lowers to the kept one.
        with kept(path):
            server = ScoringServer.from_task(*task.scoring())
        responses = served(server)
    found = {"programs": len(server.programs),
             "compiled": [program.name for program in compiles.programs if program.kind == "compiled"]}
print(json.dumps({"responses": json.loads(json.dumps(responses, default=str)), **found}))
