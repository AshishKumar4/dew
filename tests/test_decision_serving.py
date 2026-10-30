"""examples/serve_decisions.py: Jev's endpoint over Starlette, on Laya's toy checkpoint."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import httpx
import jax
import numpy as np
import optax
import pytest
from decision_support import decision_rows
from starlette.testclient import TestClient
from tool_support import load

from dew.checkpoints import Checkpoints
from dew.decision import Choice, Decide, Example
from dew.decision.calibration import Temperatures
from dew.decision.objective import DecisionObjective
from dew.inference import ScoringServer
from dew.training import Trainer

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "laya"
REQUEST = json.loads((FIXTURES / "cases.json").read_text())["quickstart"]


@pytest.fixture(scope="module")
def example():
    return load("serve_decisions", "examples")


@pytest.fixture(scope="module")
def decide() -> Decide:
    return Decide.from_pretrained(FIXTURES / "tiny", attention_impl="xla")


def test_the_endpoint_answers_what_systemone_answers(example, decide):
    """`POST /v1/systemone` returns the task's own Jev response, field for field."""
    with TestClient(example.app(example.served(decide))) as client:
        response = client.post("/v1/systemone", json=REQUEST)
        assert response.status_code == 200
        assert response.json() == json.loads(json.dumps(decide.systemone(REQUEST)))
        assert client.get("/health").json() == {"model": decide.name}


@pytest.mark.parametrize("body", [
    b"{not json", b"[1, 2]",
    json.dumps({"questions": REQUEST["questions"]}).encode(),
    json.dumps({"state": "x", "questions": {"q": {"type": "rank", "instructions": "x"}}}).encode(),
])
def test_a_request_jev_would_refuse_gets_a_422_with_its_reason(example, decide, body):
    with TestClient(example.app(example.served(decide))) as client:
        response = client.post("/v1/systemone", content=body, headers={"content-type": "application/json"})
    assert response.status_code == 422 and response.json()["error"]


def test_an_api_key_is_required_once_set(example, decide):
    with TestClient(example.app(example.served(decide), api_key="secret")) as client:
        def status(key: str | None) -> int:
            headers = {} if key is None else {"authorization": f"Bearer {key}"}
            return client.post("/v1/systemone", json=REQUEST, headers=headers).status_code

        assert (status(None), status("wrong"), status("secret")) == (401, 401, 200)


def test_a_strict_endpoint_drops_the_images_jev_does_not_know(example, decide):
    body = {**REQUEST, "images": ["aGVsbG8="]}
    with TestClient(example.app(example.served(decide))) as client:
        assert client.post("/v1/systemone", json=body).status_code == 422
    with TestClient(example.app(example.served(decide, strict=True))) as client:
        strict = client.post("/v1/systemone", json=body)
    assert strict.status_code == 200 and strict.json() == json.loads(json.dumps(decide.systemone(REQUEST)))


class Counted:
    """`task` whose passes are counted, each by the rows it scores; its first pass waits until
    `expected` requests were admitted, so the rest wait in the queue meanwhile."""

    def __init__(self, task, expected: int):
        self.task, self.expected, self.calls = task, expected, []
        self.admitted, self.all_in = 0, threading.Event()

    def admit(self, request):
        scored = self.task.admit(request)
        self.admitted += 1
        if self.admitted == self.expected:
            self.all_in.set()
        return scored

    def planned(self):
        return self.task.planned()

    def forward(self, variables, rows):
        if not self.calls:
            self.all_in.wait()
        self.calls.append(len(rows))
        return self.task.forward(variables, rows)

    def compile(self, variables):
        return self.task.compile(variables)

    def answer(self, payload, outputs):
        return self.task.answer(payload, outputs)


def test_concurrent_requests_are_answered_together_each_with_its_own_response(example, decide):
    """Six requests posted at once, one of them a body Jev refuses: the refused one gets its own 422
    at admission, and the five's rows are answered in at most two passes, the first one's wait for
    them all holding the rest in the queue, each as it is alone in every field a batch cannot move
    (tests/test_decision_batches.py)."""
    from test_decision_batches import CASES, discrete

    bodies = [*CASES.values(), {"questions": REQUEST["questions"]}]
    task, variables = decide.scoring()
    counted = Counted(task, expected=len(bodies) - 1)
    together = threading.Barrier(len(bodies))
    with TestClient(example.app(ScoringServer.from_task(counted, variables))) as client:
        def post(body):
            together.wait()
            return client.post("/v1/systemone", json=body)

        with ThreadPoolExecutor(len(bodies)) as pool:
            responses = list(pool.map(post, bodies))
    assert counted.admitted == len(bodies) - 1 and len(counted.calls) <= 2 and max(counted.calls) >= 4
    for body, response in zip(bodies, responses, strict=True):
        if "state" not in body:
            assert response.status_code == 422 and response.json()["error"]
        else:
            assert response.status_code == 200
            assert discrete(response.json()) == discrete(json.loads(json.dumps(decide.systemone(body))))


def test_a_reload_under_concurrent_requests_pairs_each_answer_with_one_tasks_weights_and_calibration(
        example, decide):
    """A reload, while eight threads post requests one after another, swaps the task and its weights at
    once: every answer is exactly the lone answer of the task its response names, the old one's or
    the new one's, its weights with its own calibration, never one's weights with the other's
    calibration. One request a pass keeps each pass the lone answer's program. The reload comes once
    eight answers are back, so answers of both come."""
    from test_decision_batches import CASES

    old = replace(decide, name="v1")
    new = replace(decide, name="v2", calibration=replace(decide.calibration, temperatures=Temperatures(
        types={"choice": 3.0, "score": 3.0, "noul": 3.0})),
        variables=jax.tree.map(lambda leaf: leaf * 1.25 if np.issubdtype(leaf.dtype, np.floating) else leaf,
                               decide.variables))
    alone = {task.name: task for task in (old, new)}
    server = ScoringServer.from_task(*old.scoring(), rows=1)
    answered: list[tuple[dict, httpx.Response]] = []
    some = threading.Event()

    def poster(client, bodies):
        for body in bodies:
            answered.append((body, client.post("/v1/systemone", json=body)))
            if len(answered) >= 8:
                some.set()

    with TestClient(example.app(server)) as client, ThreadPoolExecutor(8) as pool:
        workers = [pool.submit(poster, client, list(CASES.values()) * 2) for _ in range(8)]
        assert some.wait(120), "eight answers did not come back"
        server.reload(*new.scoring())
        for worker in workers:
            worker.result()
    named = {response.json()["model"] for _, response in answered}
    assert named == {"v1", "v2"}
    for body, response in answered:
        assert response.status_code == 200
        assert response.json() == json.loads(json.dumps(alone[response.json()["model"]].systemone(body)))


def test_the_served_weights_are_restored_in_the_dtype_they_serve_in(example):
    """The endpoint restores a task's weights in the dtype it serves in unless told another, so a run
    trained in float32 takes the memory it is served in (ledger 1889): its leaves' dtypes are a
    bfloat16 load's, and a stated storage dtype holds."""
    import jax

    def dtypes(decide):
        return jax.tree.map(lambda leaf: leaf.dtype, decide.variables)

    tiny = str(FIXTURES / "tiny")
    served = example.loaded(example.Options(model=tiny, dtype="bfloat16"))
    narrow = Decide.from_pretrained(tiny, dtype="bfloat16", param_dtype="bfloat16")
    wide = Decide.from_pretrained(tiny, dtype="bfloat16", param_dtype="float32")
    assert dtypes(served) == dtypes(narrow) != dtypes(wide)
    stated = example.loaded(example.Options(model=tiny, dtype="bfloat16", param_dtype="float32"))
    assert dtypes(stated) == dtypes(wide)


def test_a_saved_run_records_its_widths_and_a_server_of_it_compiles_its_passes_up_front(decide, tmp_path):
    """A fit records the most questions a row and options a question its data holds, as its layout steps
    bound them (`Loader.widths`, `DecisionObjective.bind_widths`), with its checkpoint, over data that
    states none ahead; the task loaded from the run pads each pass to them (`Budget`), so a scoring server
    over it compiles its passes' programs up front (ledger 2074). A record written before runs recorded
    widths loads with none, and its server compiles none ahead."""
    asked = Example("the card was charged twice", {"q": Choice("Which team?", ["cards", "loans", "fraud"])},
                    {"q": "cards"})
    laid = decide.layout.rows(decide.tokenizer, decide.specials, asked.state, asked.questions)
    widths = (max(len(row.questions) for row in laid), max(len(question.options) for row in laid
                                                           for question in row.questions))
    objective = DecisionObjective(decide)
    assert objective.widths is None
    checkpoints = Checkpoints(str(tmp_path / "today"), keep=1)
    state = Trainer(objective, optax.sgd(0.0), key=0, checkpoints=checkpoints).fit(
        decision_rows([asked] * 8, 8, objective=objective.data()), steps=1, checkpoint_every=1, log_every=100)
    checkpoints.wait()
    record = objective.inference_record()
    assert isinstance(record, dict) and (record["questions"], record["options"]) == widths
    earlier = {name: value for name, value in record.items() if name not in ("questions", "options")}
    with Checkpoints(str(tmp_path / "earlier")) as written:
        written.save(1, state, artifact=earlier)
        written.wait()
    today, before = (Decide.from_run(str(tmp_path / name)) for name in ("today", "earlier"))
    # A pass's question slots are its windows' (`Decide.pass_shapes`): its options pad to the run's.
    assert today.budget.options == widths[1] and before.budget.options is None
    server = ScoringServer.from_task(*today.scoring())
    assert server.programs and server.programs == today.pass_shapes
    assert not ScoringServer.from_task(*before.scoring()).programs


def test_a_scoring_server_compiles_its_passes_up_front_and_none_while_it_serves(decide, caplog):
    """A scoring server (`ScoringServer.from_task`) over a task whose budget declares the questions and
    options a pass pads to (`Budget`), the run's widest, compiles every pass's program up front, a few
    shapes the task declares once (`Decide.pass_shapes`): a few row counts at each of the layout's
    lengths. Then requests of every length, question and option count, alone and together, compile no
    forward while it serves; a server that compiled none up front compiles them as they come, which
    the same log shows (ledger 2033)."""
    import logging

    from test_decision_batches import CASES

    from dew.decision.task import Budget

    bodies = [*CASES.values(), {**REQUEST, "state": REQUEST["state"] * 6}]
    single = {**REQUEST, "questions": dict(list(REQUEST["questions"].items())[:1])}

    def compiled_while_serving(server: ScoringServer) -> list[str]:
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            for batch in (bodies, [single]):
                futures = [server.submit(body) for body in batch]
                server.run()
                for future in futures:
                    future.result()
        # The forward's own programs (`Decide._forward`, jitted as jit(logits)); a host-side pad is not one.
        return [record.getMessage() for record in caplog.records
                if record.getMessage().startswith("Compiling jit(logits)")]

    with jax.log_compiles():
        # A width of its own, so the cold server compiles programs the declared one has not.
        cold = replace(decide, budget=Budget(options=24))
        assert compiled_while_serving(ScoringServer(*cold.scoring(), rows=64))
        declared = replace(decide, budget=Budget(options=16))
        server = ScoringServer.from_task(*declared.scoring())
        assert server.programs == declared.pass_shapes
        assert len(set(server.programs)) == len(server.programs) <= 16
        assert not compiled_while_serving(server)


def test_a_call_over_requests_scores_in_a_servers_passes_over_its_programs(decide, caplog):
    """`Decide`'s call over requests scores them in the passes a scoring server runs (`RowScoring`, one
    path for a call and a server, ledger 2514): over a task whose budget declares its widths, after its
    passes' programs compile up front (`compile_scoring`), requests of every length, question and option
    count compile no forward, and each is answered as its lone call answers it, in the fields a padded
    pass cannot move."""
    import logging

    from test_decision_batches import CASES

    from dew.decision import Question
    from dew.decision.task import Budget

    declared = replace(decide, budget=Budget(options=16))
    declared.compile_scoring()
    requests = [(body["state"], {name: Question.from_wire(wire) for name, wire in body["questions"].items()})
                for body in [*CASES.values(), REQUEST]]
    with jax.log_compiles(), caplog.at_level(logging.WARNING):
        caplog.clear()
        together = declared(requests)
        compiled = [record.getMessage() for record in caplog.records
                    if record.getMessage().startswith("Compiling jit(logits)")]
    assert not compiled, compiled
    def discrete(answers) -> dict:
        # The fields a padded pass cannot move (tests/test_decision_batches.py's `DISCRETE`).
        return {name: (answer.kind, int(np.argmax(answer.probabilities)), answer.abstained)
                for name, answer in answers.items()}

    for (state, questions), answers in zip(requests, together, strict=True):
        assert discrete(answers) == discrete(declared(state, questions))


def test_a_tasks_scoring_plan_is_built_by_its_first_call_and_held_by_the_rest(decide):
    """A task's calls score over its one scoring plan (`RowScoring`: its passes' shapes, each pass
    length's window bounds and its packs), which its first call builds and the rest read, where a plan
    rebuilt each call laid its probe rows out and bounded every ladder length again (ledger 2590)."""
    import dataclasses

    from dew.decision import Question

    bounded: list[int] = []

    class Counted(type(decide.layout)):
        """The task's own layout, counting the window bounds it is asked for."""

        def bounds(self, *args, **kwargs):
            bounded.append(self.max_len)
            return super().bounds(*args, **kwargs)

    stated = {field.name: getattr(decide.layout, field.name) for field in dataclasses.fields(decide.layout)
              if field.init}
    task = replace(decide, layout=Counted(**stated))
    questions = {name: Question.from_wire(wire) for name, wire in REQUEST["questions"].items()}
    first = task(REQUEST["state"], questions)
    assert bounded, "the first call bounds its passes' windows"
    bounded.clear()
    again = task(REQUEST["state"], questions)
    assert not bounded, bounded
    for name, answer in again.items():
        np.testing.assert_array_equal(answer.probabilities, first[name].probabilities)


def test_a_scoring_pass_looks_at_every_waiting_request(decide):
    """A pass takes rows from every waiting request while its plan takes them, not from a bounded few: a
    hundred short requests one pass's windows hold are answered in one step, where a pass of 64 requests
    at most left the rest to a second, its windows part empty (ledger 2514)."""
    from dew.decision import Question

    single = {**REQUEST, "questions": dict(list(REQUEST["questions"].items())[:1])}
    questions = {name: Question.from_wire(wire) for name, wire in single["questions"].items()}
    plan = decide.planned()
    assert all(plan.add(row) for _ in range(100) for row in decide._encoded(single["state"], questions))
    server = ScoringServer(*decide.scoring())
    futures = [server.submit(single) for _ in range(100)]
    server.step()
    assert all(future.done() for future in futures)


def test_a_reload_after_a_cancelled_request_plans_the_next_pass_with_the_new_task(decide):
    """A request cancelled while it waits leaves the queue as its pass is chosen, and with it any plan of its
    row, so the request after it, laid out by the task a reload put in place, is planned by that task: a
    pass the old task's windows would hold, but not the new one's, splits into passes the new one holds."""
    from dew.decision.task import Budget

    wide = replace(decide, budget=Budget(rows=8))
    server = ScoringServer(*wide.scoring(), rows=64)
    dropped = server.submit(REQUEST)
    assert dropped.cancel()
    narrow = replace(decide, budget=Budget(rows=1))
    task, variables = narrow.scoring()
    server.reload(task, variables)
    # A short state, so the old task's windows would hold several of its rows.
    many = {**REQUEST, "state": "Hi.", "questions": {f"{name}{copy}": question for copy in range(3)
                                                     for name, question in REQUEST["questions"].items()}}
    answered = server.submit(many)
    server.run()
    alone = narrow.systemone(many)
    assert answered.result() == alone
