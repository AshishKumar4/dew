"""The training display in a notebook, executed in a real Jupyter kernel (nbclient) and saved.

The panel is one output that IPython's display handle updates in place while the run lasts, so a
saved notebook, and the site that renders it (site/scripts/notebook-outputs.mjs), keep its last
frame; the summary follows it, and a notebook gets no line per logging interval. Whatever else the
cell prints meanwhile, from any thread and in any pieces, stays in its outputs.
"""

import os
import re
from pathlib import Path

import nbformat
from nbclient import NotebookClient

import dew

TESTS = Path(__file__).resolve().parent
LINES = 300

# From the first logged step to the tenth, while the panel redraws, a thread prints LINES numbered
# lines, each in two flushed writes, a little apart; the tenth step's log waits for the last.
FIT = f"""\
import threading
import time

import jax
import optax
from affine_run import TimingRegression, batches

from dew.data.dataset import Loader
from dew.training import Trainer


def burst():
    for index in range({LINES}):
        print(f"line {{index}}", end="", flush=True)
        print(flush=True)
        time.sleep(0.004)


class Printing:
    printer = threading.Thread(target=burst)

    def log(self, scalars, step):
        if step == 1:
            self.printer.start()
        if step == 10:
            self.printer.join()

    def artifact(self, value, step):
        pass

    def close(self):
        pass


trainer = Trainer(TimingRegression(), optax.sgd(0.01), key=jax.random.key(0), tracker=Printing())
state = trainer.fit(Loader(lambda partition: batches(), None, None, 8), steps=40, log_every=1)
"""


def text(output) -> str:
    if output.output_type == "stream":
        return output.get("text", "")
    return output.get("data", {}).get("text/plain", "")


def test_a_notebook_keeps_the_panels_last_frame_the_summary_and_every_line_printed_beside_them(tmp_path):
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(FIT)])
    # The kernel imports the Dew under test, wherever this process found it.
    path = os.pathsep.join(filter(None, [str(Path(dew.__file__).parents[1]), os.environ.get("PYTHONPATH")]))
    NotebookClient(notebook, kernel_name="python3", resources={"metadata": {"path": str(TESTS)}}).execute(
        env={**os.environ, "PYTHONPATH": path})
    nbformat.write(notebook, tmp_path / "fit.ipynb")
    outputs = nbformat.read(tmp_path / "fit.ipynb", as_version=4).cells[0].outputs

    assert not [output for output in outputs if output.output_type == "error"], outputs
    widget = "application/vnd.jupyter.widget-view+json"
    assert not [output for output in outputs if widget in output.get("data", {})]
    assert not [output for output in outputs if "step 1/40" in text(output)], "a line per logging interval"
    panels = [output for output in outputs if output.output_type == "display_data" and "╭─" in text(output)]
    # The frame the run opened with, compiling at step 0, updated in place to its last.
    frame, summary = panels
    assert "dew" in text(frame) and "Affine" in text(frame) and "40/40" in text(frame)
    # Ended: the frame's bottom border carries the finished mark, not a spinner.
    assert "loss" in text(frame) and "✓" in text(frame)
    # The summary's step count among its words, its styling's escape codes aside.
    assert "Affine" in text(summary) and "40" in re.sub(r"\x1b\[[0-9;]*m", " ", text(summary)).split()
    # Styled, so the site renders each as the panel it is, not as one more line of a log.
    assert all("<span" in "".join(output.data["text/html"]) for output in panels)
    printed = "".join(output.text for output in outputs if output.output_type == "stream")
    assert printed.splitlines() == [f"line {index}" for index in range(LINES)]
