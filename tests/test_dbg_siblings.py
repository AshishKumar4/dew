"""Scratch regression: DEW_DBG_DISTINCT_LABELS=1 removes the witness's label collision."""

import dataclasses
import os

from option_witness import held, replaced, unread
from run_kinds import shared, trained_by_cli

from dew.decision import Decide


def test_shared_siblings_changes_answer(tmp_path):
    run = shared(tmp_path, drawn=False)
    if os.environ.get("DEW_DBG_DISTINCT_LABELS") == "1":
        labels = dataclasses.replace(
            run.head.labels, names=("token4", "token5", "token6", "token7"))
        run = dataclasses.replace(run, head=dataclasses.replace(run.head, labels=labels))
    directory = trained_by_cli(run, tmp_path, "shared-regression", {"siblings": "True"})
    task = Decide.from_run(str(directory))
    before = held(replaced(task, "layout.siblings", False))
    failure = unread(task, before, True)
    print("LABEL_IDS", task.model.head.label_ids, "UNREAD", failure, flush=True)
    assert failure is None
