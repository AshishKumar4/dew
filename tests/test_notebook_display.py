"""The training display in a notebook, executed in a real Jupyter kernel (nbclient) and saved.

The panel is one output that IPython's display handle updates in place while the run lasts, so a
saved notebook, and the site that renders it (site/scripts/notebook-outputs.mjs), keep its last
frame; the summary follows it, and a notebook gets no line per logging interval. Whatever else the
cell prints meanwhile, from any thread and in any pieces, stays in its outputs.
"""

import os
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

import dew

TESTS = Path(__file__).resolve().parent

# A thread prints numbered lines, each in two flushed writes, for as long as the fit runs.
FIT = """\
import threading
import time

import jax
import optax
from affine_run import TimingRegression, batches

from dew.data.dataset import Loader
from dew.training import Trainer

done = threading.Event()

def count():
    index = 0
    while not done.is_set():
        print(f"line {index}", end="", flush=True)
        print(flush=True)
        index += 1
        time.sleep(0.002)
    print(f"lines {index}", flush=True)

counting = threading.Thread(target=count)
counting.start()
trainer = Trainer(TimingRegression(), optax.sgd(0.01), key=jax.random.key(0))
state = trainer.fit(Loader(lambda partition: batches(), None, None, 8), steps=40, log_every=1)
done.set()
counting.join()
"""


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    """The outputs of a notebook that fits the small regression, as saved."""
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(FIT)])
    # The kernel imports the Dew under test, wherever this process found it.
    path = os.pathsep.join(filter(None, [str(Path(dew.__file__).parents[1]), os.environ.get("PYTHONPATH")]))
    NotebookClient(notebook, kernel_name="python3", resources={"metadata": {"path": str(TESTS)}}).execute(
        env={**os.environ, "PYTHONPATH": path})
    saved = tmp_path_factory.mktemp("notebook") / "fit.ipynb"
    nbformat.write(notebook, saved)
    return nbformat.read(saved, as_version=4).cells[0].outputs


def text(output) -> str:
    if output.output_type == "stream":
        return output.get("text", "")
    return output.get("data", {}).get("text/plain", "")


def test_a_notebook_keeps_the_panels_last_frame_as_one_output_and_the_summary(outputs):
    assert not [output for output in outputs if output.output_type == "error"], outputs
    widget = "application/vnd.jupyter.widget-view+json"
    assert not [output for output in outputs if widget in output.get("data", {})]
    assert not [output for output in outputs if "step 1/40" in text(output)], "a line per logging interval"
    panels = [output for output in outputs if output.output_type == "display_data" and "╭─" in text(output)]
    # The frame the run opened with, compiling at step 0, updated in place to its last.
    frame, summary = panels
    assert "dew" in text(frame) and "Affine" in text(frame) and "40/40" in text(frame)
    assert "finished" in text(frame) and "loss" in text(frame)
    assert "Affine" in text(summary) and "40 steps" in text(summary)
    # Styled, so the site renders each as the panel it is, not as one more line of a log.
    assert all("<span" in "".join(output.data["text/html"]) for output in panels)


def test_every_line_the_cell_prints_beside_the_panel_stays_in_its_outputs(outputs):
    assert not [output for output in outputs if output.output_type == "error"], outputs
    printed = "".join(output.text for output in outputs if output.output_type == "stream")
    lines = printed.splitlines()
    total = int(lines[-1].removeprefix("lines "))
    assert total > 100
    assert lines[:-1] == [f"line {index}" for index in range(total)]
