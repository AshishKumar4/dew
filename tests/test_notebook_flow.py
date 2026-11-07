"""A training run built and run from a notebook through Dew's public API, as a user writes one: a run
configured in Python, trained, logged, checkpointed, scored and read back cell by cell in one kernel, its
record the one `dew train` reads (row 3727)."""

import json
import os
from pathlib import Path

import nbformat
from lane_environment import outside_suite
from nbclient import NotebookClient
from process_support import dew_train, logged_losses

import dew
from dew.config import RunConfig

CELLS = ['''
import json
from pathlib import Path

from dew.config import TrainerConfig
from dew.config.lm import LMRunConfig
from dew.data import Files, presets
from dew.registry import ModelConfig

rows = Path("play.jsonl")
rows.write_text("".join(json.dumps({"text": "to be or not to be, that is the question. " * 8}) + "\\n"
                        for _ in range(40)))
train, held = Files(str(rows), format="jsonl").holdout(records=4)
run = LMRunConfig(
    model=ModelConfig("causal_transformer", {"emb_features": 16, "num_layers": 1, "num_heads": 2,
                                             "max_seq_len": 48, "dtype": "float32"}),
    data=train.repeat().pipe(presets.lm, "byte", seq_len=32, batch=4, pack=None),
    validation=held.pipe(presets.lm, "byte", seq_len=32, batch=4, pack=None),
    trainer=TrainerConfig(name="notebook", checkpoint_dir=str(Path("checkpoints").resolve()), steps=4,
                          log_every=1, eval_every=2, checkpoint_every=2, multi_host=False,
                          compilation_cache_dir=None))
''', '''
state = run.run()
int(state.step)
''', '''
from dew.training import LocalTracker

run_dir = Path("checkpoints") / "notebook"
logged = {}
with LocalTracker(run_dir / "tracking") as journal:
    for step, scores in journal.scalars():
        logged.setdefault(str(step), set()).update(scores)
print(json.dumps({step: sorted(names) for step, names in logged.items()}))
''', '''
import dew

task = dew.pipeline(str(run_dir))
print(json.dumps({"continued": task("to be", 4, key=0).text[0]}))
''']
"""The notebook: a byte-level language run configured in Python with `LMRunConfig` over a JSON lines
file it writes, trained by `run()` (the process set up, the objective built, validation every two steps,
a journal, checkpoints under the run's name), its journal read back, and the run directory loaded as the
task it saved."""


def environment() -> dict[str, str]:
    """A user's machine (`outside_suite`: one CPU device, none of the suite's settings), with the Dew under
    test importable wherever this process found it."""
    path = os.pathsep.join(filter(None, [str(Path(dew.__file__).parents[1]), os.environ.get("PYTHONPATH")]))
    return {**outside_suite(os.environ), "PYTHONPATH": path}


def executed(directory: Path) -> list[list]:
    """The notebook run cell by cell in one kernel under `directory`, each cell's outputs."""
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(cell) for cell in CELLS])
    NotebookClient(notebook, kernel_name="python3", timeout=600,
                   resources={"metadata": {"path": str(directory)}}).execute(env=environment())
    return [cell.outputs for cell in notebook.cells]


def printed(outputs) -> dict:
    """A cell's JSON printed to its standard output."""
    return json.loads("".join(output.text for output in outputs
                              if output.output_type == "stream" and output.name == "stdout"))


def test_a_notebook_trains_scores_checkpoints_and_reloads_a_run_dew_train_continues(tmp_path):
    """The notebook's run trains its four steps, its journal logs every step's loss and scores the held-out
    split every two steps, it checkpoints under its name with the record `run()` writes, and the run directory
    loads as a text generator that continues a prompt. `dew train` reads that record and continues the
    same run two steps further, as it continues one it started: the notebook's run has the defaults and
    planning the CLI's has."""
    outputs = executed(tmp_path)
    assert not [output for cell in outputs for output in cell if output.output_type == "error"], outputs
    assert outputs[1][-1].data["text/plain"] == "4"
    logged = printed(outputs[2])
    assert all("train/loss" in logged.get(str(step), ()) for step in range(1, 5)), sorted(logged)
    assert any(name.startswith("val/") for names in logged.values() for name in names)
    assert isinstance(printed(outputs[3])["continued"], str)
    run_dir = tmp_path / "checkpoints" / "notebook"
    assert RunConfig.load(str(run_dir)).trainer.steps == 4

    dew_train(tmp_path, environment(), str(run_dir / "run.json"), "--set", "trainer.steps=6")
    assert len(logged_losses(run_dir)) == 6
