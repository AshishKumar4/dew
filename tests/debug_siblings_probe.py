import sys
sys.path.insert(0, "tests")
import dataclasses
import tempfile
from pathlib import Path
import jax
import numpy as np
from run_kinds import shared, requested, trained_by_cli
from model_support import moved
from option_witness import answers, answered_alike, replaced
from dew.decision import Choice, Decide
from dew.decision.layout import DecisionInputs
from dew.decision.shared_state import entering

root = Path(tempfile.mkdtemp(prefix="dew-siblings-"))
run = shared(root, drawn=False)
directory = trained_by_cli(run, root, "shared-probe", {"siblings": "True"})
task = Decide.from_run(str(directory))
task = dataclasses.replace(task, variables=moved(task.variables, jax.random.key(2639), 0.1))
questions = {
    "team": Choice("which team", {"billing": "invoices and charges", "technical": "the site and the app", "sales": "plans and prices"}),
    "urgent": Choice("how urgent", {"now": "blocks work", "later": "can wait"}),
}
print("TASK", task.layout, "max_len", task.layout.max_len, "siblings", task.layout.siblings, flush=True)
for siblings in (False, True):
    own = replaced(task, "layout.siblings", siblings)
    rows = own.rows([("the site is down", questions)])[0]
    print("LAYOUT", siblings, "attention", own.layout.attention, "entering", entering(*own.layout.attention), flush=True)
    for row in rows:
        print("ROW", siblings, "length", len(row.tokens), "streams", row.streams, "positions", row.positions, "questions", row.questions, flush=True)
    result = requested(own)
    print("RESULT", siblings, repr(result), flush=True)
print("EXACT_ALIKE", answered_alike(answers(task), answers(replaced(task, "layout.siblings", False))), flush=True)
print("HEAD_LABEL_IDS", task.model.head.label_ids, flush=True)
states = []
for siblings in (False, True):
    own = replaced(task, "layout.siblings", siblings)
    rows = own.rows([("the site is down", questions)])[0]
    inputs = DecisionInputs.collate(rows, own.specials.pad, attention=own.layout.attention)
    state = np.asarray(own.model.states(own.variables, inputs))
    states.append(state)
    print("RAW_LOGITS", siblings, own.scored(rows), flush=True)
for question in rows[0].questions:
    at = question.readout
    print("CUE_STATE_MAX_DIFF", question.name, float(np.max(np.abs(states[1][0, at] - states[0][0, at]))), flush=True)
labels = dataclasses.replace(task.layout.labels, names=("token4", "token5", "token6", "token7"))
head = task.model.head.clone(label_ids=labels.ids(task.tokenizer, 4))
fixed = dataclasses.replace(task, model=dataclasses.replace(task.model, head=head),
                            layout=dataclasses.replace(task.layout, labels=labels))
print("DISTINCT_LABEL_IDS", fixed.model.head.label_ids, flush=True)
for siblings in (False, True):
    own = replaced(fixed, "layout.siblings", siblings)
    print("DISTINCT_RESULT", siblings, repr(requested(own)), flush=True)
print("DISTINCT_EXACT_ALIKE", answered_alike(answers(fixed), answers(replaced(fixed, "layout.siblings", False))), flush=True)
assert answered_alike(answers(task), answers(replaced(task, "layout.siblings", False)))
assert not answered_alike(answers(fixed), answers(replaced(fixed, "layout.siblings", False)))
